from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import replace, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import pandas as pd
from scipy.stats import spearmanr, kendalltau

# -----------------------------
# Project imports
# -----------------------------
from data.data_sample_interface import DataSample
from data.dataset.unity_sim_dataset_loader import UnitySimDatasetLoader
from data.cache.cached_dataset_loader import CachedDatasetLoader

from data.extractor.rgb.dinov2_rgb_extractor import DINOv2RGBExtractor
from data.extractor.pointcloud.pointnet2_lidar_extractor import PointNet2LiDARExtractor
from data.extractor.joint.concat_joint_embedding_extractor import ConcatJointEmbeddingExtractor
from data.extractor.residual.unity_projection_residual_extractor import ProjectionResidualExtractor

from metric.product_mmd import ProductMMD
from metric.copula_decomposition import CopulaDecomposition
from metric.calibration_residual_transport import CalibrationResidualTransport

from task_network.data.task_unity_sim_dataloader import UnitySimDetectionDataset
from task_network.frustrum_pointnet.unity.unity_frustum_extractor import FrustumExtractor
from task_network.frustrum_pointnet.frustum_pointnet_full_points import FrustumPointNet

"""
python -m run.eval.run_dataset_sweep_task_and_metrics \
  --baseline-name baseline \
  --dataset baseline=/data_private/data/eval_multi_sensor:lidar \
  --dataset z10=/data_private/data/eval_multi_sensor:lidar_z10 \
  --dataset z15=/data_private/data/eval_multi_sensor:lidar_z15 \
  --change baseline="baseline lidar" \
  --change z10="LiDAR z rotation 10 deg" \
  --change z15="LiDAR z rotation 15 deg" \
  --pointcloud-sensor FOV-Grid_PointCloud \
  --ckpt E:\domain_gap_metric\results\train\baseline_point_only\best_frustum_pointnet.pt \
  --n-task-frames 100 \
  --n-metric-frames 100 \
  --residual-max-points 100 \
  --out-csv /data_private/output/results_multi_lidar.csv \
  --out-json /data_private/output/results_multi_lidar.json \
  --device cuda
"""


# ============================================================
# Small utilities
# ============================================================

@dataclass(frozen=True)
class DatasetEntry:
    name: str
    path: Path
    lidar_capture_id: str = "lidar"

def parse_dataset_arg(x: str) -> DatasetEntry:
    """
    Parse:
      NAME=PATH
      NAME=PATH:LIDAR_CAPTURE_ID

    Examples:
      baseline=/data/eval_multi
      z10=/data/eval_multi:lidar_z10
    """
    if "=" not in x:
        raise argparse.ArgumentTypeError(
            f"Dataset must be NAME=PATH or NAME=PATH:LIDAR_ID, got: {x!r}"
        )

    name, rest = x.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Dataset name must not be empty.")

    if ":" in rest:
        path_str, lidar_capture_id = rest.rsplit(":", 1)
        lidar_capture_id = lidar_capture_id.strip()
        if not lidar_capture_id:
            lidar_capture_id = "lidar"
    else:
        path_str = rest
        lidar_capture_id = "lidar"

    return DatasetEntry(
        name=name,
        path=Path(path_str),
        lidar_capture_id=lidar_capture_id,
    )


def resolve_device(device: Optional[str]) -> str:
    if device is None or device.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.lower() == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device


def sample_ids(ids: Sequence[str], n: int, seed: int) -> List[str]:
    ids = list(ids)
    if n <= 0 or n >= len(ids):
        return ids
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids[:n]


def sample_indices(n_total: int, n: int, seed: int) -> List[int]:
    idx = list(range(n_total))
    if n <= 0 or n >= n_total:
        return idx
    rng = random.Random(seed)
    rng.shuffle(idx)
    return idx[:n]


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def median(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    m = len(ys) // 2
    if len(ys) % 2 == 1:
        return float(ys[m])
    return float(0.5 * (ys[m - 1] + ys[m]))


# ============================================================
# Box geometry and approximate oriented 3D IoU
# ============================================================

def quaternion_rotation_error_deg(q_pred: torch.Tensor, q_gt: torch.Tensor) -> float:
    q_pred = q_pred.detach().float().flatten().cpu()
    q_gt = q_gt.detach().float().flatten().cpu()

    q_pred = q_pred / q_pred.norm().clamp_min(1e-12)
    q_gt = q_gt / q_gt.norm().clamp_min(1e-12)

    dot = torch.abs(torch.sum(q_pred * q_gt)).clamp(-1.0, 1.0)
    angle_rad = 2.0 * torch.acos(dot)
    return float(torch.rad2deg(angle_rad).item())

def quat_xyzw_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q.detach().float().flatten().cpu()
    qx, qy, qz, qw = q.tolist()
    n = max((qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5, 1e-12)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return torch.tensor([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=torch.float32)


def rotmat_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    R = R.detach().float().cpu()
    trace = R.trace()
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
    return q / q.norm().clamp_min(1e-12)


def transform_box_lidar_to_cam(
    center_lidar: torch.Tensor,
    quat_lidar: torch.Tensor,
    T_cam_lidar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    center_h = torch.cat([center_lidar.detach().float().cpu(), torch.ones(1)])
    T = T_cam_lidar.detach().float().cpu()
    center_cam = (T @ center_h)[:3]

    R_cam_lidar = T[:3, :3]
    R_box_lidar = quat_xyzw_to_rotmat(quat_lidar)
    R_box_cam = R_cam_lidar @ R_box_lidar
    quat_cam = rotmat_to_quat_xyzw(R_box_cam)
    return center_cam, quat_cam


def compute_box_corners(center: torch.Tensor, size: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    center = center.detach().float().flatten().cpu()
    size = size.detach().float().flatten().cpu().clamp_min(1e-6)
    hx, hy, hz = (size / 2.0).tolist()
    corners_local = torch.tensor([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ], dtype=torch.float32)
    R = quat_xyzw_to_rotmat(quat)
    return corners_local @ R.T + center[None, :]


def points_inside_obb(points: torch.Tensor, center: torch.Tensor, size: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    points = points.detach().float().cpu()
    center = center.detach().float().flatten().cpu()
    size = size.detach().float().flatten().cpu().clamp_min(1e-6)
    R = quat_xyzw_to_rotmat(quat)
    # Since R maps local -> world/camera, inverse is R.T.
    local = (points - center[None, :]) @ R
    half = size[None, :] / 2.0
    return (torch.abs(local) <= half).all(dim=1)


def approximate_oriented_3d_iou(
    center_a: torch.Tensor,
    size_a: torch.Tensor,
    quat_a: torch.Tensor,
    center_b: torch.Tensor,
    size_b: torch.Tensor,
    quat_b: torch.Tensor,
    *,
    num_samples: int = 4096,
    seed: int = 0,
) -> float:
    """
    Deterministic Monte-Carlo approximation of oriented 3D IoU.

    This is not an exact geometric OBB IoU implementation, but it is robust,
    dependency-free, and good enough to track trends in perturbation sweeps.
    Increase --iou-samples for smoother values.
    """
    if num_samples <= 0:
        return float("nan")

    corners = torch.cat([
        compute_box_corners(center_a, size_a, quat_a),
        compute_box_corners(center_b, size_b, quat_b),
    ], dim=0)
    lo = corners.min(dim=0).values
    hi = corners.max(dim=0).values
    extent = (hi - lo).clamp_min(1e-6)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    pts = lo[None, :] + torch.rand((num_samples, 3), generator=gen) * extent[None, :]

    in_a = points_inside_obb(pts, center_a, size_a, quat_a)
    in_b = points_inside_obb(pts, center_b, size_b, quat_b)
    inter = (in_a & in_b).sum().item()
    union = (in_a | in_b).sum().item()
    if union == 0:
        return 0.0
    return float(inter / union)


# ============================================================
# Task-network evaluation
# ============================================================

def load_task_model(ckpt_path: Path, device: str, num_object_points: int) -> FrustumPointNet:
    model = FrustumPointNet(
        input_channels=3,
        num_seg_classes=2,
        num_object_points=num_object_points,
        use_bn=False,
    ).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded task checkpoint: {ckpt_path}")
    print(f"[INFO] Checkpoint epoch: {ckpt.get('epoch')}")
    print(f"[INFO] Best eval center err: {ckpt.get('best_eval_center_err')}")
    return model


@torch.no_grad()
def evaluate_task_network(
    *,
    root_dir: Path,
    pointcloud_sensor: str,
    lidar_capture_id: str,
    model: FrustumPointNet,
    device: str,
    n_frames: int,
    seed: int,
    num_points: int,
    iou_samples: int,
    box_margin_px: float,
) -> Dict[str, object]:
    ds = UnitySimDetectionDataset(root_dir=root_dir, pointcloud_sensor=pointcloud_sensor, lidar_capture_id=lidar_capture_id)
    indices = sample_indices(len(ds), n_frames, seed)

    extractor = FrustumExtractor(
        box_margin_px=box_margin_px,
        min_depth=1e-4,
        num_sampled_points=num_points,
        sampling="random",
        pad_repeat=True,
    )

    center_errs_lidar: List[float] = []
    rot_errs_deg = []
    center_errs_cam: List[float] = []
    ious: List[float] = []
    confidences: List[float] = []
    skipped = 0

    per_object_rows = []

    for local_i, idx in enumerate(indices, start=1):
        if local_i % 25 == 0 or local_i == 1 or local_i == len(indices):
            print(f"[INFO] Task eval frame {local_i}/{len(indices)}")

        image, lidar_xyz, target = ds[idx]
        image_size = tuple(int(v) for v in target["image_size"].tolist())
        T_cam_lidar = target["T_cam_lidar"].float()
        T_lidar_cam = torch.linalg.inv(T_cam_lidar)

        boxes_2d = target["boxes_2d_xyxy"]
        has_3d = target["has_3d_box"]

        for obj_idx in range(boxes_2d.shape[0]):
            if not bool(has_3d[obj_idx]):
                continue
            box_2d = boxes_2d[obj_idx]
            if (box_2d[2] <= box_2d[0]) or (box_2d[3] <= box_2d[1]):
                skipped += 1
                continue

            gt_center_cam = target["boxes_3d_center"][obj_idx].float()
            gt_size = target["boxes_3d_size"][obj_idx].float().clamp_min(1e-6)
            gt_quat_cam = target["boxes_3d_quat"][obj_idx].float()

            try:
                result = extractor.extract(
                    lidar_xyz=lidar_xyz,
                    box_2d_xyxy=box_2d,
                    intrinsics=target["intrinsics"],
                    T_cam_lidar=T_cam_lidar,
                    image_size=image_size,
                    gt_box_center_cam=gt_center_cam,
                    gt_box_size=gt_size,
                    gt_box_quat_xyzw=gt_quat_cam,
                )
            except Exception as e:
                skipped += 1
                print(f"[WARN] Frustum extraction failed: {type(e).__name__}: {e}")
                continue

            if result.sampled_points_lidar is None or result.sampled_points_lidar.shape[0] == 0:
                skipped += 1
                continue

            points_batch = result.sampled_points_lidar.float().unsqueeze(0).to(device)
            pred = model(points_batch)

            pred_center_lidar = pred["center_lidar"][0].detach().cpu()
            pred_size = pred["size_lidar"][0].detach().cpu().clamp_min(1e-6)
            pred_quat_lidar = pred["quat_xyzw"][0].detach().cpu()
            pred_conf = float(pred["confidence"][0].detach().cpu().item())

            gt_center_h = torch.cat([gt_center_cam.cpu(), torch.ones(1)])
            gt_center_lidar = (T_lidar_cam.cpu() @ gt_center_h)[:3]

            pred_center_cam, pred_quat_cam = transform_box_lidar_to_cam(
                pred_center_lidar, pred_quat_lidar, T_cam_lidar
            )

            center_err_lidar = torch.linalg.norm(pred_center_lidar - gt_center_lidar).item()
            center_err_cam = torch.linalg.norm(pred_center_cam - gt_center_cam.cpu()).item()
            rot_err_deg = quaternion_rotation_error_deg(
                pred_quat_cam,
                gt_quat_cam.cpu(),
            )
            iou = approximate_oriented_3d_iou(
                pred_center_cam, pred_size, pred_quat_cam,
                gt_center_cam.cpu(), gt_size.cpu(), gt_quat_cam.cpu(),
                num_samples=iou_samples,
                seed=seed + idx * 1000 + obj_idx,
            )

            center_errs_lidar.append(float(center_err_lidar))
            center_errs_cam.append(float(center_err_cam))
            rot_errs_deg.append(float(rot_err_deg))
            ious.append(float(iou))
            confidences.append(float(pred_conf))

            per_object_rows.append({
                "frame_idx": int(idx),
                "object_idx": int(obj_idx),
                "frame_id": str(target["frame_id"]),
                "sequence_id": str(target["sequence_id"]),
                "label": int(target["labels"][obj_idx]),
                "label_name": str(target["label_names"][obj_idx]),
                "center_error_lidar": float(center_err_lidar),
                "center_error_cam": float(center_err_cam),
                "rotation_error_deg": float(rot_err_deg),
                "iou3d_approx": float(iou),
                "confidence": float(pred_conf),

                "num_frustum_points": int(result.frustum_points_lidar.shape[0]),
                "num_sampled_points": int(result.sampled_points_lidar.shape[0]),

                "gt_center_x": float(gt_center_cam[0]),
                "gt_center_y": float(gt_center_cam[1]),
                "gt_center_z": float(gt_center_cam[2]),

                "gt_size_x": float(gt_size[0]),
                "gt_size_y": float(gt_size[1]),
                "gt_size_z": float(gt_size[2]),
            })

    return {
        "summary":{
            "task_num_frames": float(len(indices)),
            "task_num_objects": float(len(center_errs_lidar)),
            "task_num_skipped": float(skipped),
            "center_error_lidar_mean": mean(center_errs_lidar),
            "center_error_lidar_median": median(center_errs_lidar),
            "center_error_cam_mean": mean(center_errs_cam),
            "center_error_cam_median": median(center_errs_cam),
            "rotation_error_deg_mean": mean(rot_errs_deg),
            "rotation_error_deg_median": median(rot_errs_deg),
            "iou3d_approx_mean": mean(ious),
            "iou3d_approx_median": median(ious),
            "iou3d_approx_at_025": mean([1.0 if x >= 0.25 else 0.0 for x in ious]),
            "iou3d_approx_at_050": mean([1.0 if x >= 0.50 else 0.0 for x in ious]),
            "confidence_mean": mean(confidences),
        },
        "per_object_rows": per_object_rows,
    }


# ============================================================
# E1/E2/E3 metric computation
# ============================================================

def populate_sample(
    sample: DataSample,
    rgb_extractor: DINOv2RGBExtractor,
    lidar_extractor: PointNet2LiDARExtractor,
    joint_extractor: ConcatJointEmbeddingExtractor,
    residual_extractor: Optional[ProjectionResidualExtractor],
) -> DataSample:
    rgb_emb = rgb_extractor.extract(sample)
    lidar_emb = lidar_extractor.extract(sample)
    updated = replace(sample, rgb_emb=rgb_emb, lidar_emb=lidar_emb)
    joint_emb = joint_extractor.extract(updated)
    updated = replace(updated, joint_emb=joint_emb)

    if residual_extractor is not None:
        try:
            residuals = residual_extractor.extract(updated)
            updated = replace(updated, residuals=residuals)
        except Exception as e:
            print(f"[WARN] Residual extraction failed for {sample.frame_id}: {type(e).__name__}: {e}")
    return updated

def sample_has_finite_fields(sample: DataSample, fields: Sequence[str]) -> bool:
    for field in fields:
        x = getattr(sample, field, None)
        if x is None:
            continue
        if torch.is_tensor(x) and not torch.isfinite(x).all():
            return False
    return True

def load_metric_samples(
    *,
    root_dir: Path,
    pointcloud_sensor: str,
    lidar_capture_id: str,
    setup_id: str,
    n_frames: int,
    seed: int,
    rgb_extractor: DINOv2RGBExtractor,
    lidar_extractor: PointNet2LiDARExtractor,
    joint_extractor: ConcatJointEmbeddingExtractor,
    residual_extractor: Optional[ProjectionResidualExtractor],
    use_embedding_cache: bool,
    cache_dir_name: str,
    refresh_embedding_cache: bool,
) -> List[DataSample]:
    raw_loader = UnitySimDatasetLoader(
        root_dir=root_dir,
        pointcloud_sensor=pointcloud_sensor,
        lidar_capture_id=lidar_capture_id,
        setup_id=setup_id,
        use_identity_extrinsic_if_missing=residual_extractor is not None,
    )

    extractors = [
        rgb_extractor,
        lidar_extractor,
        joint_extractor,
    ]
    if residual_extractor is not None:
        extractors.append(residual_extractor)

    if use_embedding_cache:
        loader = CachedDatasetLoader(
            raw_loader,
            cache_root=root_dir / cache_dir_name,
            extractors=extractors,
            enabled=True,
            write_cache=True,
            refresh_cache=refresh_embedding_cache,
            drop_raw_after_populate=True,
        )
        print(f"[INFO] Metric cache enabled for setup={setup_id}, lidar_capture_id={lidar_capture_id}")
        print(f"[INFO] Cache root: {root_dir / cache_dir_name}")
        print(f"[INFO] Cache id: {loader.cache_id}")
    else:
        loader = raw_loader
        print(f"[INFO] Metric cache disabled for setup={setup_id}, lidar_capture_id={lidar_capture_id}")

    frame_ids = sample_ids(loader.get_frame_ids(), n_frames, seed)
    samples: List[DataSample] = []

    for i, frame_id in enumerate(frame_ids, start=1):
        if i % 25 == 0 or i == 1 or i == len(frame_ids):
            action = "loading cached/encoding" if use_embedding_cache else "encoding"
            print(f"[INFO] Metric {action} {setup_id} frame {i}/{len(frame_ids)}: {frame_id}")

        sample = loader.load_sample(frame_id)

        if not use_embedding_cache:
            sample = populate_sample(
                sample,
                rgb_extractor,
                lidar_extractor,
                joint_extractor,
                residual_extractor,
            )
            sample = replace(sample, rgb=None, lidar_xyz=None, lidar_feat=None)

        required_finite_fields = ["rgb_emb", "lidar_emb", "joint_emb"]
        if residual_extractor is not None:
            required_finite_fields.append("residuals")

        if not sample_has_finite_fields(sample, required_finite_fields):
            print(f"[WARN] Skipping non-finite cached sample: setup={setup_id}, frame={frame_id}")
            continue

        samples.append(sample)

    return samples


def with_residuals(samples: Sequence[DataSample]) -> List[DataSample]:
    return [s for s in samples if s.residuals is not None]


def _total_residual_count(samples: Sequence[DataSample]) -> int:
    total = 0
    for s in samples:
        if s.residuals is None:
            continue
        r = s.residuals
        if r.ndim == 0:
            total += 1
        else:
            total += int(r.shape[0])
    return total


def cap_residuals_for_metric(
    samples: Sequence[DataSample],
    *,
    max_total_points: int,
    seed: int,
) -> List[DataSample]:
    """
    Globally cap residual rows before the expensive pairwise residual metric.

    Important:
    - --residual-max-points controls how many residuals are extracted/cached per frame.
    - --residual-metric-max-total-points controls how many residuals are used in
      CalibrationResidualTransport per domain.

    If max_total_points <= 0, no cap is applied.
    """
    if max_total_points <= 0:
        return list(samples)

    chunks: List[torch.Tensor] = []
    first_sample: Optional[DataSample] = None

    for s in samples:
        if s.residuals is None:
            continue

        r = s.residuals.detach().cpu().float()
        if r.ndim == 0:
            r = r.reshape(1, 1)
        elif r.ndim == 1:
            r = r[:, None]
        elif r.ndim != 2:
            raise ValueError(
                f"Residuals must have shape (M,) or (M,K), got {tuple(r.shape)}"
            )

        if r.shape[0] == 0:
            continue

        chunks.append(r)
        if first_sample is None:
            first_sample = s

    if not chunks or first_sample is None:
        return []

    all_residuals = torch.cat(chunks, dim=0)
    total = int(all_residuals.shape[0])

    if total <= max_total_points:
        return list(samples)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    idx = torch.randperm(total, generator=gen)[:max_total_points]
    capped_residuals = all_residuals[idx].contiguous()

    # One representative sample is enough because CalibrationResidualTransport
    # concatenates residual rows across samples internally.
    return [replace(first_sample, residuals=capped_residuals)]


def compute_e_metrics(
    a_samples: Sequence[DataSample],
    b_samples: Sequence[DataSample],
    *,
    residual_metric_max_total_points: int,
    seed: int,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    metrics = [
        ProductMMD(),
        CopulaDecomposition(marginal_metric="mmd"),
        CopulaDecomposition(marginal_metric="sinkhorn"),
    ]

    for metric in metrics:
        try:
            result = metric(a_samples, b_samples)
            out[metric.name] = float(result.score)
            # Useful sub-scores for E2; harmless if absent.
            for k, v in result.details.items():
                if isinstance(v, (int, float)):
                    out[f"{metric.name}.{k}"] = float(v)
        except Exception as e:
            print(f"[WARN] {metric.name} failed: {type(e).__name__}: {e}")
            out[metric.name] = float("nan")

    a_res_raw = with_residuals(a_samples)
    b_res_raw = with_residuals(b_samples)

    out["A_residual_frame_count"] = float(len(a_res_raw))
    out["B_residual_frame_count"] = float(len(b_res_raw))
    out["A_residual_point_count_raw"] = float(_total_residual_count(a_res_raw))
    out["B_residual_point_count_raw"] = float(_total_residual_count(b_res_raw))

    a_res = cap_residuals_for_metric(
        a_res_raw,
        max_total_points=residual_metric_max_total_points,
        seed=seed + 10_001,
    )
    b_res = cap_residuals_for_metric(
        b_res_raw,
        max_total_points=residual_metric_max_total_points,
        seed=seed + 20_001,
    )

    out["A_residual_point_count_metric"] = float(_total_residual_count(a_res))
    out["B_residual_point_count_metric"] = float(_total_residual_count(b_res))
    out["residual_metric_max_total_points"] = float(residual_metric_max_total_points)

    if residual_metric_max_total_points > 0:
        print(
            "[INFO] Residual metric cap: "
            f"A {int(out['A_residual_point_count_raw'])} -> {int(out['A_residual_point_count_metric'])}, "
            f"B {int(out['B_residual_point_count_raw'])} -> {int(out['B_residual_point_count_metric'])}"
        )
    else:
        print(
            "[INFO] Residual metric cap disabled: "
            f"A={int(out['A_residual_point_count_raw'])}, "
            f"B={int(out['B_residual_point_count_raw'])}"
        )

    for backend in ["sinkhorn", "mmd"]:
        name = f"Calibration-Residual-Transport-{backend}"
        if len(a_res) < 2 or len(b_res) < 2:
            out[name] = float("nan")
            continue
        try:
            metric = CalibrationResidualTransport(backend=backend)
            result = metric(a_res, b_res)
            out[name] = float(result.score)
        except Exception as e:
            print(f"[WARN] {name} failed: {type(e).__name__}: {e}")
            out[name] = float("nan")

    return out


# ============================================================
# Output helpers
# ============================================================

def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote CSV: {path}")


def save_json(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[INFO] Wrote JSON: {path}")


def print_compact_table(rows: List[Dict[str, object]]) -> None:
    cols = [
        "dataset",
        "change",
        "center_error_cam_mean",
        "iou3d_approx_mean",
        "PRODUCT-MMD",
        "CopulaDecomp-MMD+HSIC",
        "CopulaDecomp-SINKHORN+HSIC",
        "Calibration-Residual-Transport-sinkhorn",
    ]
    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)
    print(" | ".join(f"{c:>32s}" for c in cols))
    print("-" * 140)
    for row in rows:
        vals = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:32.6f}")
            else:
                vals.append(f"{str(v):>32s}")
        print(" | ".join(vals))

def compute_and_save_correlations(rows, out_csv: Path, baseline_name: str) -> None:
    import pandas as pd
    from scipy.stats import spearmanr, kendalltau

    df = pd.DataFrame(rows)

    performance_cols = [
        "center_error_lidar_mean",
        "center_error_lidar_median",
        "center_error_cam_mean",
        "center_error_cam_median",
        "iou3d_approx_mean",
        "iou3d_approx_median",
        "iou3d_approx_at_025",
        "iou3d_approx_at_050",
        "confidence_mean",
        "rotation_error_deg_mean",
        "rotation_error_deg_median",
    ]

    metric_cols = [
        "PRODUCT-MMD",
        "CopulaDecomp-MMD+HSIC",
        "CopulaDecomp-MMD+HSIC.d_r",
        "CopulaDecomp-MMD+HSIC.d_l",
        "CopulaDecomp-MMD+HSIC.d_dep",
        "CopulaDecomp-SINKHORN+HSIC",
        "CopulaDecomp-SINKHORN+HSIC.d_r",
        "CopulaDecomp-SINKHORN+HSIC.d_l",
        "CopulaDecomp-SINKHORN+HSIC.d_dep",
        "Calibration-Residual-Transport-sinkhorn",
        "Calibration-Residual-Transport-mmd",
    ]

    performance_cols = [c for c in performance_cols if c in df.columns]
    metric_cols = [c for c in metric_cols if c in df.columns]

    baseline_rows = df[df["dataset"] == baseline_name]

    if len(baseline_rows) == 0:
        print(f"[WARN] Baseline dataset '{baseline_name}' not found.")
        return

    baseline = baseline_rows.iloc[0]

    for col in performance_cols:
        if "iou" in col or "confidence" in col:
            df[f"delta_{col}"] = baseline[col] - df[col]
        else:
            df[f"delta_{col}"] = df[col] - baseline[col]

    df_eval = df[df["dataset"] != baseline_name].copy()

    rows_out = []

    for metric in metric_cols:
        for perf in performance_cols:
            delta_col = f"delta_{perf}"
            tmp = df_eval[[metric, delta_col]].dropna()

            if len(tmp) < 3:
                continue

            spear = spearmanr(tmp[metric], tmp[delta_col])
            kend = kendalltau(tmp[metric], tmp[delta_col])

            rows_out.append({
                "group": "all",
                "n": len(tmp),
                "metric": metric,
                "performance_delta": delta_col,
                "spearman_rho": spear.statistic,
                "spearman_p": spear.pvalue,
                "kendall_tau": kend.statistic,
                "kendall_p": kend.pvalue,
            })

    corr = pd.DataFrame(rows_out)

    if len(corr) > 0:
        corr["abs_spearman"] = corr["spearman_rho"].abs()
        corr = corr.sort_values("abs_spearman", ascending=False)

    out_path = out_csv.with_name(out_csv.stem + "_correlations.csv")
    corr.to_csv(out_path, index=False)
    print(f"[INFO] Wrote correlations: {out_path}")
# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FrustumPointNet task performance and E1/E2/E3 domain-gap metrics for multiple Unity datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset_arg,
        help="Dataset entry as NAME=PATH or NAME=PATH:LIDAR_CAPTURE_ID. Repeat this argument for all datasets.",
    )
    parser.add_argument(
        "--baseline-name",
        required=True,
        help="Name of the dataset entry used as A/baseline for domain-gap metrics.",
    )
    parser.add_argument(
        "--change",
        action="append",
        default=[],
        help="Optional label as NAME=TEXT, e.g. translation_5cm=LiDAR translated 5 cm.",
    )
    parser.add_argument("--pointcloud-sensor", required=True, help="Unity point cloud sensor name, e.g. FOV-Grid_PointCloud")
    parser.add_argument("--ckpt", required=True, type=Path, help="Path to trained FrustumPointNet checkpoint")
    parser.add_argument("--out-csv", type=Path, default=Path("results_task_and_metrics.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("results_task_and_metrics.json"))
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n-task-frames", type=int, default=500, help="Frames per dataset for task-network evaluation. <=0 means all.")
    parser.add_argument("--n-metric-frames", type=int, default=500, help="Frames per dataset for metric computation. <=0 means all.")

    parser.add_argument("--num-points", type=int, default=1024, help="Frustum points for task network")
    parser.add_argument("--num-object-points", type=int, default=512, help="Object points in FrustumPointNet")
    parser.add_argument("--box-margin-px", type=float, default=2.0)
    parser.add_argument("--iou-samples", type=int, default=4096, help="Monte-Carlo samples for approximate oriented 3D IoU")

    parser.add_argument("--lidar-sampling-mode", choices=["random", "fps", "none"], default="fps")
    parser.add_argument("--lidar-num-points", type=int, default=2048)
    parser.add_argument("--no-residuals", action="store_true", help="Disable E3 residual extraction")
    parser.add_argument("--residual-edge-threshold", type=float, default=0.2)
    parser.add_argument("--residual-max-points", type=int, default=5000)
    parser.add_argument(
        "--residual-metric-max-total-points",
        type=int,
        default=0,
        help=(
            "Global cap on residual rows per domain used inside "
            "CalibrationResidualTransport. <=0 disables the cap. "
            "For the old behavior with --n-metric-frames 100 and "
            "--residual-max-points 100, use 10000. "
            "Recommended when caching many residuals per frame: 5000-10000."
        ),
    )

    parser.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help=(
            "Disable loading/writing cached metric embeddings. By default the runner "
            "uses CachedDatasetLoader and stores caches under ROOT/.embedding_cache."
        ),
    )
    parser.add_argument(
        "--cache-dir-name",
        default=".embedding_cache",
        help="Cache directory name inside each dataset root.",
    )
    parser.add_argument(
        "--refresh-embedding-cache",
        action="store_true",
        help="Recompute metric embeddings/residuals and overwrite existing cache files.",
    )

    args = parser.parse_args()

    datasets: Dict[str, DatasetEntry] = {d.name: d for d in args.dataset}
    if args.baseline_name not in datasets:
        raise ValueError(f"baseline-name {args.baseline_name!r} not found in --dataset entries: {list(datasets)}")

    change_labels: Dict[str, str] = {name: name for name in datasets}
    for item in args.change:
        if "=" not in item:
            raise ValueError(f"--change must be NAME=TEXT, got {item!r}")
        name, text = item.split("=", 1)
        change_labels[name] = text

    device = resolve_device(args.device)
    print(f"[INFO] Device: {device}")

    # 1) Load task network once.
    task_model = load_task_model(args.ckpt, device, args.num_object_points)

    # 2) Initialize metric extractors once.
    print("[INFO] Initializing metric extractors...")
    rgb_extractor = DINOv2RGBExtractor(device=device)
    lidar_extractor = PointNet2LiDARExtractor(
        num_points=None if args.lidar_sampling_mode == "none" else args.lidar_num_points,
        sampling_mode=args.lidar_sampling_mode,
        device=device,
    )
    joint_extractor = ConcatJointEmbeddingExtractor(l2_normalize_each=False)
    residual_extractor = None
    if not args.no_residuals:
        residual_extractor = ProjectionResidualExtractor(
            edge_threshold=args.residual_edge_threshold,
            max_points=args.residual_max_points,
        )

    # 3) Encode baseline metric samples once.
    baseline_entry = datasets[args.baseline_name]
    baseline_path = baseline_entry.path
    print("\n" + "=" * 100)
    print(f"[INFO] Loading baseline metric samples: {args.baseline_name} -> {baseline_path}")
    print("=" * 100)
    baseline_metric_samples = load_metric_samples(
        root_dir=baseline_path,
        pointcloud_sensor=args.pointcloud_sensor,
        lidar_capture_id=baseline_entry.lidar_capture_id,
        setup_id=args.baseline_name,
        n_frames=args.n_metric_frames,
        seed=args.seed,
        rgb_extractor=rgb_extractor,
        lidar_extractor=lidar_extractor,
        joint_extractor=joint_extractor,
        residual_extractor=residual_extractor,
        use_embedding_cache=not args.no_embedding_cache,
        cache_dir_name=args.cache_dir_name,
        refresh_embedding_cache=args.refresh_embedding_cache,
    )

    rows: List[Dict[str, object]] = []

    all_object_rows = []

    for ds_i, (name, entry) in enumerate(datasets.items(), start=1):
        path = entry.path
        lidar_capture_id = entry.lidar_capture_id
        print("\n" + "#" * 100)
        print(f"[INFO] Dataset {ds_i}/{len(datasets)}: {name} -> {path}")
        print("#" * 100)

        # Task performance for this dataset.
        task_result = evaluate_task_network(
            root_dir=path,
            pointcloud_sensor=args.pointcloud_sensor,
            lidar_capture_id=lidar_capture_id,
            model=task_model,
            device=device,
            n_frames=args.n_task_frames,
            seed=args.seed,
            num_points=args.num_points,
            iou_samples=args.iou_samples,
            box_margin_px=args.box_margin_px,
        )

        task_scores = task_result["summary"]
        per_object_rows = task_result["per_object_rows"]

        for r in per_object_rows:
            rr = dict(r)
            rr["dataset"] = name
            rr["change"] = change_labels.get(name, name)
            all_object_rows.append(rr)

        # Metric samples for this dataset. Reuse baseline samples if same dataset.
        if name == args.baseline_name:
            metric_samples = baseline_metric_samples
        else:
            metric_samples = load_metric_samples(
                root_dir=path,
                pointcloud_sensor=args.pointcloud_sensor,
                lidar_capture_id=lidar_capture_id,
                setup_id=name,
                n_frames=args.n_metric_frames,
                seed=args.seed,
                rgb_extractor=rgb_extractor,
                lidar_extractor=lidar_extractor,
                joint_extractor=joint_extractor,
                residual_extractor=residual_extractor,
                use_embedding_cache=not args.no_embedding_cache,
                cache_dir_name=args.cache_dir_name,
                refresh_embedding_cache=args.refresh_embedding_cache,
            )

        metric_scores = compute_e_metrics(
            baseline_metric_samples,
            metric_samples,
            residual_metric_max_total_points=args.residual_metric_max_total_points,
            seed=args.seed,
        )

        row: Dict[str, object] = {
            "dataset": name,
            "path": str(path),
            "lidar_capture_id": lidar_capture_id,
            "change": change_labels.get(name, name),
            "is_baseline": bool(name == args.baseline_name),
        }
        row.update(task_scores)
        row.update(metric_scores)
        rows.append(row)

        print("[INFO] Row snapshot:")
        for k in [
            "center_error_cam_mean",
            "iou3d_approx_mean",
            "PRODUCT-MMD",
            "CopulaDecomp-MMD+HSIC",
            "CopulaDecomp-SINKHORN+HSIC",
            "Calibration-Residual-Transport-sinkhorn",
        ]:
            print(f"  {k:42s}: {row.get(k)}")

    print_compact_table(rows)
    save_csv(rows, args.out_csv)
    save_json(rows, args.out_json)


    save_csv(
        all_object_rows,
        args.out_csv.with_name(args.out_csv.stem + "_per_object.csv")
    )

    compute_and_save_correlations(rows, args.out_csv, args.baseline_name)


if __name__ == "__main__":
    main()
