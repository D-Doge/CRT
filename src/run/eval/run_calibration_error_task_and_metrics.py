from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from data.data_sample_interface import DataSample
from data.dataset.unity_sim_dataset_loader import UnitySimDatasetLoader
from data.cache.cached_dataset_loader import CachedDatasetLoader

from data.extractor.rgb.dinov2_rgb_extractor import DINOv2RGBExtractor
from data.extractor.pointcloud.pointnet2_lidar_extractor import PointNet2LiDARExtractor
from data.extractor.joint.concat_joint_embedding_extractor import ConcatJointEmbeddingExtractor
from data.extractor.residual.unity_projection_residual_extractor import ProjectionResidualExtractor

from task_network.data.task_unity_sim_dataloader import UnitySimDetectionDataset
from task_network.frustrum_pointnet.unity.unity_frustum_extractor import FrustumExtractor
from task_network.frustrum_pointnet.frustum_pointnet_full_points import FrustumPointNet

# Reuse the unchanged metric/task helper functions from your existing eval script.
from run.eval.run_dataset_sweep_task_and_metrics_multi_lidar import (
    resolve_device,
    sample_ids,
    sample_indices,
    mean,
    median,
    load_task_model,
    transform_box_lidar_to_cam,
    quaternion_rotation_error_deg,
    approximate_oriented_3d_iou,
    compute_e_metrics,
    print_compact_table,
    save_csv,
    save_json,
    compute_and_save_correlations,
)


# Must match the calibration-error precompute script.
# Names are the short error suffixes; the actual cache setup id can be either:
#   etx+1
# or, with --prefix-lidar-to-error-setup-id:
#   lidar_etx+1
ERROR_SPECS = [
    # name, tx_m, ty_m, tz_m, rx_deg, ry_deg, rz_deg
    ("etx_1", +0.01, 0, 0, 0, 0, 0),
    ("ety_1", 0, +0.01, 0, 0, 0, 0),
    ("etz_1", 0, 0, +0.01, 0, 0, 0),

    ("etx_2", +0.02, 0, 0, 0, 0, 0),
    ("ety_2", 0, +0.02, 0, 0, 0, 0),
    ("etz_2", 0, 0, +0.02, 0, 0, 0),

    ("etx_3", +0.03, 0, 0, 0, 0, 0),
    ("ety_3", 0, +0.03, 0, 0, 0, 0),
    ("etz_3", 0, 0, +0.03, 0, 0, 0),

    ("etx_4", +0.04, 0, 0, 0, 0, 0),
    ("ety_4", 0, +0.04, 0, 0, 0, 0),
    ("etz_4", 0, 0, +0.04, 0, 0, 0),

    ("etx_5", +0.05, 0, 0, 0, 0, 0),
    ("ety_5", 0, +0.05, 0, 0, 0, 0),
    ("etz_5", 0, 0, +0.05, 0, 0, 0),

]


@dataclass(frozen=True)
class ErrorEntry:
    name: str
    setup_id: str
    change: str
    tx_m: float
    ty_m: float
    tz_m: float
    rx_deg: float
    ry_deg: float
    rz_deg: float


def rot_x(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)


def rot_y(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


def rot_z(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


def make_error_transform(
    tx_m: float,
    ty_m: float,
    tz_m: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> torch.Tensor:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)

    E = torch.eye(4, dtype=torch.float32)
    E[:3, :3] = R
    E[:3, 3] = torch.tensor([tx_m, ty_m, tz_m], dtype=torch.float32)
    return E


def inject_calibration_error(T_cam_lidar: torch.Tensor, E_cam: torch.Tensor) -> torch.Tensor:
    """
    T_cam_lidar maps LiDAR -> camera.
    Error is left-multiplied in the camera frame:
        T_bad = E_cam @ T_true
    """
    return E_cam @ T_cam_lidar.detach().float()


class CalibrationErrorMetricLoader:
    """
    Wraps UnitySimDatasetLoader and injects the same calibration error used during
    error precomputation. This makes CachedDatasetLoader look for the correct
    setup_id cache and also allows recomputation if cache is missing.
    """

    def __init__(self, base_loader, *, setup_id: str, E_cam: torch.Tensor):
        self.base_loader = base_loader
        self.setup_id = setup_id
        self.E_cam = E_cam

    def get_frame_ids(self):
        return self.base_loader.get_frame_ids()

    def load_sample(self, frame_id: str) -> DataSample:
        sample = self.base_loader.load_sample(frame_id)
        if sample.T_cam_lidar is None:
            raise ValueError(f"Sample {frame_id} has no T_cam_lidar.")
        T_bad = inject_calibration_error(sample.T_cam_lidar, self.E_cam)

        meta = dict(sample.meta)
        meta["calibration_error_setup"] = self.setup_id
        meta["calibration_error_T_left_multiplied"] = True

        return replace(
            sample,
            setup_id=self.setup_id,
            T_cam_lidar=T_bad,
            meta=meta,
        )

    def __getattr__(self, name):
        return getattr(self.base_loader, name)


class CalibrationErrorDetectionDataset(torch.utils.data.Dataset):
    """
    Wraps UnitySimDetectionDataset and injects calibration error into target["T_cam_lidar"].
    This keeps the task-eval code identical except for the calibration matrix used.
    """

    def __init__(self, base_dataset: UnitySimDetectionDataset, E_cam: torch.Tensor):
        self.base_dataset = base_dataset
        self.E_cam = E_cam

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, lidar_xyz, target = self.base_dataset[idx]
        target = dict(target)

        if target.get("T_cam_lidar") is None:
            raise ValueError(f"Target idx={idx} has no T_cam_lidar.")

        target["T_cam_lidar"] = inject_calibration_error(target["T_cam_lidar"], self.E_cam)
        return image, lidar_xyz, target


def sample_has_finite_fields(sample: DataSample, fields: Sequence[str]) -> bool:
    for field in fields:
        x = getattr(sample, field, None)
        if x is None:
            continue
        if torch.is_tensor(x) and not torch.isfinite(x).all():
            return False
    return True


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


def load_metric_samples_error(
    *,
    root_dir: Path,
    pointcloud_sensor: str,
    lidar_capture_id: str,
    setup_id: str,
    E_cam: Optional[torch.Tensor],
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

    if E_cam is not None:
        raw_loader = CalibrationErrorMetricLoader(
            raw_loader,
            setup_id=setup_id,
            E_cam=E_cam,
        )

    extractors = [rgb_extractor, lidar_extractor, joint_extractor]
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
        loader._cache_id = "pc_FOV-Grid_PointCloud__rgb_rgb_camera__lidar_lidar__rgb_emb_dinov2_vits14__lidar_emb_pointnet2_ssg_fps_np2048__joint_em__49fd59e867ca"
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
            sample = populate_sample(sample, rgb_extractor, lidar_extractor, joint_extractor, residual_extractor)
            sample = replace(sample, rgb=None, lidar_xyz=None, lidar_feat=None)

        required_finite_fields = ["rgb_emb", "lidar_emb", "joint_emb"]
        if residual_extractor is not None:
            required_finite_fields.append("residuals")

        if not sample_has_finite_fields(sample, required_finite_fields):
            print(f"[WARN] Skipping non-finite sample: setup={setup_id}, frame={frame_id}")
            continue

        samples.append(sample)

    return samples


@torch.no_grad()
def evaluate_task_network_error(
    *,
    root_dir: Path,
    pointcloud_sensor: str,
    lidar_capture_id: str,
    E_cam: Optional[torch.Tensor],
    model: FrustumPointNet,
    device: str,
    n_frames: int,
    seed: int,
    num_points: int,
    iou_samples: int,
    box_margin_px: float,
) -> Dict[str, object]:
    base_ds = UnitySimDetectionDataset(
        root_dir=root_dir,
        pointcloud_sensor=pointcloud_sensor,
        lidar_capture_id=lidar_capture_id,
    )
    ds = CalibrationErrorDetectionDataset(base_ds, E_cam) if E_cam is not None else base_ds

    indices = sample_indices(len(ds), n_frames, seed)

    extractor = FrustumExtractor(
        box_margin_px=box_margin_px,
        min_depth=1e-4,
        num_sampled_points=num_points,
        sampling="random",
        pad_repeat=True,
    )

    center_errs_lidar: List[float] = []
    rot_errs_deg: List[float] = []
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
                pred_center_lidar,
                pred_quat_lidar,
                T_cam_lidar,
            )

            center_err_lidar = torch.linalg.norm(pred_center_lidar - gt_center_lidar).item()
            center_err_cam = torch.linalg.norm(pred_center_cam - gt_center_cam.cpu()).item()
            rot_err_deg = quaternion_rotation_error_deg(pred_quat_cam, gt_quat_cam.cpu())
            iou = approximate_oriented_3d_iou(
                pred_center_cam,
                pred_size,
                pred_quat_cam,
                gt_center_cam.cpu(),
                gt_size.cpu(),
                gt_quat_cam.cpu(),
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
        "summary": {
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


def build_error_entries(
    *,
    lidar_capture_id: str,
    prefix_lidar_to_error_setup_id: bool,
    selected_errors: Optional[Sequence[str]],
) -> List[ErrorEntry]:
    selected = set(selected_errors or [])
    out: List[ErrorEntry] = []

    for name, tx, ty, tz, rx, ry, rz in ERROR_SPECS:
        if selected and name not in selected:
            continue

        setup_id = f"{lidar_capture_id}_{name}" if prefix_lidar_to_error_setup_id else name

        parts = []
        if tx:
            parts.append(f"tx={tx*100:.0f}cm")
        if ty:
            parts.append(f"ty={ty*100:.0f}cm")
        if tz:
            parts.append(f"tz={tz*100:.0f}cm")
        if rx:
            parts.append(f"rx={rx:.1f}deg")
        if ry:
            parts.append(f"ry={ry:.1f}deg")
        if rz:
            parts.append(f"rz={rz:.1f}deg")

        out.append(ErrorEntry(
            name=name,
            setup_id=setup_id,
            change=", ".join(parts) if parts else "baseline",
            tx_m=tx,
            ty_m=ty,
            tz_m=tz,
            rx_deg=rx,
            ry_deg=ry,
            rz_deg=rz,
        ))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate task performance and E1/E2/E3 metrics for injected calibration-error "
            "setups using precomputed error embedding/residual caches."
        )
    )

    parser.add_argument("--root-dir", required=True, type=Path)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--pointcloud-sensor", required=True)
    parser.add_argument("--lidar-capture-id", default="lidar")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path, default=Path("results_calib_error_task_metrics.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("results_calib_error_task_metrics.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n-task-frames", type=int, default=500)
    parser.add_argument("--n-metric-frames", type=int, default=500)

    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-object-points", type=int, default=512)
    parser.add_argument("--box-margin-px", type=float, default=2.0)
    parser.add_argument("--iou-samples", type=int, default=4096)

    parser.add_argument("--lidar-sampling-mode", choices=["random", "fps", "none"], default="fps")
    parser.add_argument("--lidar-num-points", type=int, default=2048)
    parser.add_argument("--no-residuals", action="store_true")
    parser.add_argument("--residual-edge-threshold", type=float, default=0.2)
    parser.add_argument("--residual-max-points", type=int, default=100)
    parser.add_argument("--residual-metric-max-total-points", type=int, default=0)

    parser.add_argument("--no-embedding-cache", action="store_true")
    parser.add_argument(
        "--cache-dir-name",
        default=".embedding_cache_calib_errors",
        help="Must match the error precompute script cache dir.",
    )
    parser.add_argument("--refresh-embedding-cache", action="store_true")

    parser.add_argument(
        "--error",
        action="append",
        default=[],
        help="Optional error suffix to evaluate, e.g. etx+1. Repeat to select a subset. Default: all ERROR_SPECS.",
    )
    parser.add_argument(
        "--prefix-lidar-to-error-setup-id",
        action="store_true",
        help=(
            "Use setup_id like lidar_etx+1. Enable this if your precompute script used "
            "setup_id=f'{lidar_capture_id}_{name}'."
        ),
    )

    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[INFO] Device: {device}")

    entries = build_error_entries(
        lidar_capture_id=args.lidar_capture_id,
        prefix_lidar_to_error_setup_id=args.prefix_lidar_to_error_setup_id,
        selected_errors=args.error,
    )

    if not entries:
        raise ValueError("No error entries selected. Check --error names.")

    print("[INFO] Error entries:")
    for e in entries:
        print(f"  {e.setup_id:20s}  {e.change}")

    task_model = load_task_model(args.ckpt, device, args.num_object_points)

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

    print("\n" + "=" * 100)
    print(f"[INFO] Loading baseline metric samples: {args.baseline_name} -> {args.root_dir}")
    print("=" * 100)

    baseline_metric_samples = load_metric_samples_error(
        root_dir=args.root_dir,
        pointcloud_sensor=args.pointcloud_sensor,
        lidar_capture_id=args.lidar_capture_id,
        setup_id=args.baseline_name,
        E_cam=None,
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
    all_object_rows: List[Dict[str, object]] = []

    # Baseline row first.
    eval_entries: List[Tuple[str, str, Optional[torch.Tensor], str]] = [
        (args.baseline_name, "baseline", None, args.baseline_name)
    ]

    for e in entries:
        E_cam = make_error_transform(e.tx_m, e.ty_m, e.tz_m, e.rx_deg, e.ry_deg, e.rz_deg)
        eval_entries.append((e.setup_id, e.change, E_cam, e.name))

    for ds_i, (setup_id, change, E_cam, short_name) in enumerate(eval_entries, start=1):
        print("\n" + "#" * 100)
        print(f"[INFO] Setup {ds_i}/{len(eval_entries)}: {setup_id} | {change}")
        print("#" * 100)

        task_result = evaluate_task_network_error(
            root_dir=args.root_dir,
            pointcloud_sensor=args.pointcloud_sensor,
            lidar_capture_id=args.lidar_capture_id,
            E_cam=E_cam,
            model=task_model,
            device=device,
            n_frames=args.n_task_frames,
            seed=args.seed,
            num_points=args.num_points,
            iou_samples=args.iou_samples,
            box_margin_px=args.box_margin_px,
        )

        task_scores = task_result["summary"]

        for r in task_result["per_object_rows"]:
            rr = dict(r)
            rr["dataset"] = setup_id
            rr["change"] = change
            rr["error_name"] = short_name
            all_object_rows.append(rr)

        if setup_id == args.baseline_name:
            metric_samples = baseline_metric_samples
        else:
            metric_samples = load_metric_samples_error(
                root_dir=args.root_dir,
                pointcloud_sensor=args.pointcloud_sensor,
                lidar_capture_id=args.lidar_capture_id,
                setup_id=setup_id,
                E_cam=E_cam,
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
            "dataset": setup_id,
            "path": str(args.root_dir),
            "lidar_capture_id": args.lidar_capture_id,
            "change": change,
            "error_name": short_name,
            "is_baseline": bool(setup_id == args.baseline_name),
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
    save_csv(all_object_rows, args.out_csv.with_name(args.out_csv.stem + "_per_object.csv"))
    compute_and_save_correlations(rows, args.out_csv, args.baseline_name)


if __name__ == "__main__":
    main()
