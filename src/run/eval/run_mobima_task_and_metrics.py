from __future__ import annotations

"""
Evaluate a KITTI-trained FrustumPointNet and the project's domain-gap metrics
on the MOBIMA multi-LiDAR dataset.

Expected layout
---------------
ROOT/
  _classes.json
  dataset_meta.json
  calibration.json
  samples/
    001882/
      image_rgb.jpg
      mid360.pcd
      mid70.pcd
      velodyne.pcd
      metadata.json
      labels_3d.json
      pseudo_labels_2d.json

The script treats each RGB + LiDAR pair as a sensor setup. All selected setups
use the same RGB frames, so the real-world experiment isolates the LiDAR / fusion
configuration change as much as possible.

Example
-------
python -m run.eval.run_mobima_task_and_metrics \
  --root-dir /data_private/mobima \
  --baseline-sensor mid360 \
  --sensor mid360 --sensor mid70 --sensor velodyne \
  --ckpt /data_private/output/checkpoints/frustum_pointnet_kitti_run2/best_frustum_pointnet_kitti.pt \
  --n-task-frames 100 \
  --n-metric-frames 100 \
  --lidar-num-points 2048 \
  --lidar-sampling-mode fps \
  --residual-max-points 100 \
  --residual-metric-max-total-points 10000 \
  --out-csv /data_private/output/mobima_task_metrics.csv \
  --out-json /data_private/output/mobima_task_metrics.json \
  --device cuda

Notes
-----
* labels_3d.json vertices are assumed to be in calibration.json's
  reference_frame (currently MID360).
* Camera projection uses the OpenCV optical convention declared by
  calibration.json: x right, y down, z forward. There is NO Unity Y flip.
* Intrinsics are scaled from dataset_meta.json's camera_info resolution to the
  actual image_rgb.jpg resolution.
* pseudo_labels_2d.json is used only to obtain the 2D frustum proposal.
  labels_3d.json is used for the true 3D task error.
* Coverage-aware task evaluation keeps every GT object with a matched 2D proposal;
  if a LiDAR has zero points in that object's frustum, the object contributes IoU=0.
  Center error remains valid-prediction-only because no arbitrary failure distance is used.
* Calibration Residual Transport uses a MOBIMA/OpenCV projection residual
  extractor implemented below: Sobel edge map -> distance transform ->
  projected LiDAR distance-to-edge samples.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, sobel
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau, spearmanr

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
from data.data_sample_interface import DataSample
from data.extractor.rgb.dinov2_rgb_extractor import DINOv2RGBExtractor
from data.extractor.pointcloud.pointnet2_lidar_extractor import PointNet2LiDARExtractor
from data.extractor.joint.concat_joint_embedding_extractor import ConcatJointEmbeddingExtractor

from metric.product_mmd import ProductMMD
from metric.copula_decomposition import CopulaDecomposition
from metric.calibration_residual_transport import CalibrationResidualTransport

from task_network.frustrum_pointnet.frustum_pointnet_full_points import FrustumPointNet


SUPPORTED_SENSORS = ("mid360", "mid70", "velodyne")
DEFAULT_TASK_CLASSES = ("car", "van", "truck")
LABELCLOUD_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


# =============================================================================
# General helpers
# =============================================================================


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(device: Optional[str]) -> str:
    if device is None or device.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.lower() == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return device


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def median(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(float(x) for x in xs)
    m = len(ys) // 2
    if len(ys) % 2:
        return ys[m]
    return 0.5 * (ys[m - 1] + ys[m])


def stable_int_seed(*parts: object, base: int = 0) -> int:
    payload = "|".join(str(x) for x in (base, *parts)).encode("utf-8")
    digest = hashlib.sha1(payload).digest()[:8]
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF


def sample_ids(ids: Sequence[str], n: int, seed: int) -> List[str]:
    out = list(ids)
    if n <= 0 or n >= len(out):
        return out
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:n]


def hash_config(payload: Mapping[str, Any]) -> str:
    txt = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:16]


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Image + PCD loading
# =============================================================================


def load_rgb_image(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def image_size_from_tensor(rgb: torch.Tensor) -> Tuple[int, int]:
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected RGB CHW tensor, got {tuple(rgb.shape)}")
    return int(rgb.shape[1]), int(rgb.shape[2])


def _pcd_numpy_dtype(type_code: str, size: int) -> np.dtype:
    key = (type_code.upper(), int(size))
    mapping = {
        ("F", 4): np.dtype("<f4"),
        ("F", 8): np.dtype("<f8"),
        ("I", 1): np.dtype("<i1"),
        ("I", 2): np.dtype("<i2"),
        ("I", 4): np.dtype("<i4"),
        ("I", 8): np.dtype("<i8"),
        ("U", 1): np.dtype("<u1"),
        ("U", 2): np.dtype("<u2"),
        ("U", 4): np.dtype("<u4"),
        ("U", 8): np.dtype("<u8"),
    }
    if key not in mapping:
        raise ValueError(f"Unsupported PCD TYPE/SIZE combination: {key}")
    return mapping[key]


def _load_pcd_open3d(path: Path) -> torch.Tensor:
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"PCD {path} uses binary_compressed data. Install open3d or export "
            "PCDs as ASCII/binary."
        ) from exc
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError(f"Open3D could not read non-empty XYZ from {path}")
    return torch.from_numpy(xyz.copy()).float()


def load_pcd_xyz(path: Path) -> torch.Tensor:
    """Read x/y/z from ASCII or uncompressed binary PCD; Open3D fallback for compressed."""
    if not path.is_file():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    header_lines: List[str] = []
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PCD header: {path}")
            text = line.decode("ascii", errors="strict").strip()
            header_lines.append(text)
            if text.upper().startswith("DATA "):
                data_kind = text.split(None, 1)[1].strip().lower()
                data_offset = f.tell()
                break

    def tokens(name: str) -> Optional[List[str]]:
        prefix = name.upper() + " "
        for line in header_lines:
            if line.upper().startswith(prefix):
                return line.split()[1:]
        return None

    fields = tokens("FIELDS") or tokens("FIELD")
    if not fields:
        raise ValueError(f"PCD has no FIELDS: {path}")
    if not all(axis in fields for axis in ("x", "y", "z")):
        raise ValueError(f"PCD lacks x/y/z fields: {fields}")

    sizes = [int(x) for x in (tokens("SIZE") or ["4"] * len(fields))]
    types = list(tokens("TYPE") or ["F"] * len(fields))
    counts = [int(x) for x in (tokens("COUNT") or ["1"] * len(fields))]
    points_tokens = tokens("POINTS")
    width_tokens = tokens("WIDTH")
    height_tokens = tokens("HEIGHT")
    if points_tokens:
        n_points = int(points_tokens[0])
    elif width_tokens:
        n_points = int(width_tokens[0]) * int(height_tokens[0] if height_tokens else 1)
    else:
        n_points = -1

    if data_kind == "binary_compressed":
        return _load_pcd_open3d(path)

    if data_kind == "ascii":
        with path.open("rb") as f:
            f.seek(data_offset)
            txt = f.read().decode("utf-8", errors="strict").strip()
        if not txt:
            raise ValueError(f"Empty PCD: {path}")
        expanded_names: List[str] = []
        for name, count in zip(fields, counts):
            expanded_names.extend([name] if count == 1 else [f"{name}_{i}" for i in range(count)])
        vals = np.fromstring(txt, sep=" ", dtype=np.float64)
        cols = len(expanded_names)
        if vals.size % cols != 0:
            arr = np.loadtxt(path, comments="#", skiprows=len(header_lines), dtype=np.float64)
            arr = np.atleast_2d(arr)
        else:
            arr = vals.reshape(-1, cols)
        col = {name: i for i, name in enumerate(expanded_names)}
        xyz = arr[:, [col["x"], col["y"], col["z"]]].astype(np.float32, copy=False)
        if xyz.shape[0] == 0:
            raise ValueError(f"Empty PCD: {path}")
        return torch.from_numpy(np.ascontiguousarray(xyz)).float()

    if data_kind != "binary":
        raise ValueError(f"Unsupported PCD DATA kind {data_kind!r}: {path}")

    dtype_fields = []
    for name, size, type_code, count in zip(fields, sizes, types, counts):
        dt = _pcd_numpy_dtype(type_code, size)
        if count == 1:
            dtype_fields.append((name, dt))
        else:
            dtype_fields.append((name, dt, (count,)))
    dtype = np.dtype(dtype_fields)
    with path.open("rb") as f:
        f.seek(data_offset)
        raw = f.read()
    count = n_points if n_points >= 0 else len(raw) // dtype.itemsize
    arr = np.frombuffer(raw, dtype=dtype, count=count)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32, copy=False)
    if xyz.shape[0] == 0:
        raise ValueError(f"Empty PCD: {path}")
    return torch.from_numpy(np.ascontiguousarray(xyz)).float()


# =============================================================================
# Calibration / geometry
# =============================================================================


def as_transform(matrix: Any, name: str) -> torch.Tensor:
    T = torch.tensor(matrix, dtype=torch.float32)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {tuple(T.shape)}")
    return T


def transform_points(points_xyz: torch.Tensor, T_target_source: torch.Tensor) -> torch.Tensor:
    pts = points_xyz.detach().float()
    T = T_target_source.detach().float()
    ones = torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)
    return torch.cat([pts, ones], dim=1) @ T.T[:, :3]


def transform_point(point_xyz: torch.Tensor, T_target_source: torch.Tensor) -> torch.Tensor:
    p = point_xyz.detach().float().flatten()
    ph = torch.cat([p, torch.ones(1, dtype=p.dtype)])
    return (T_target_source.detach().float() @ ph)[:3]


def project_camera_points_opencv(
    cam_xyz: torch.Tensor,
    K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """OpenCV optical frame: x right, y down, z forward. No axis flip."""
    xyz = cam_xyz.detach().float()
    K = K.detach().float()
    z = xyz[:, 2]
    safe = torch.where(torch.abs(z) > 1e-8, z, torch.full_like(z, 1e-8))
    u = K[0, 0] * xyz[:, 0] / safe + K[0, 2]
    v = K[1, 1] * xyz[:, 1] / safe + K[1, 2]
    return torch.stack([u, v], dim=1), z


def project_lidar_points_opencv(
    lidar_xyz: torch.Tensor,
    K: torch.Tensor,
    T_cam_lidar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cam = transform_points(lidar_xyz, T_cam_lidar)
    uv, depth = project_camera_points_opencv(cam, K)
    return cam, uv, depth


def scaled_intrinsics(camera_info: Mapping[str, Any], image_size: Tuple[int, int]) -> torch.Tensor:
    K_raw = camera_info.get("K")
    if K_raw is None or len(K_raw) != 9:
        raise ValueError("camera_info.K must contain 9 values")
    K = torch.tensor(K_raw, dtype=torch.float32).reshape(3, 3)
    src_h = int(camera_info.get("height", image_size[0]))
    src_w = int(camera_info.get("width", image_size[1]))
    dst_h, dst_w = image_size
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid camera_info resolution: {src_w}x{src_h}")
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    K = K.clone()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def quat_xyzw_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q.detach().float().flatten()
    q = q / q.norm().clamp_min(1e-12)
    qx, qy, qz, qw = q.unbind()
    return torch.stack([
        torch.stack([1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)]),
        torch.stack([2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)]),
        torch.stack([2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)]),
    ], dim=0)


def rotmat_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    R = R.detach().float()
    U, _, Vh = torch.linalg.svd(R)
    R = U @ Vh
    if torch.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vh

    trace = float(torch.trace(R).item())
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = float(R[2, 1] - R[1, 2]) / s
        qy = float(R[0, 2] - R[2, 0]) / s
        qz = float(R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + float(R[0, 0] - R[1, 1] - R[2, 2]))) * 2.0
        qw = float(R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = float(R[0, 1] + R[1, 0]) / s
        qz = float(R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1e-12, 1.0 + float(R[1, 1] - R[0, 0] - R[2, 2]))) * 2.0
        qw = float(R[0, 2] - R[2, 0]) / s
        qx = float(R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = float(R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(1e-12, 1.0 + float(R[2, 2] - R[0, 0] - R[1, 1]))) * 2.0
        qw = float(R[1, 0] - R[0, 1]) / s
        qx = float(R[0, 2] + R[2, 0]) / s
        qy = float(R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
    return q / q.norm().clamp_min(1e-12)


def compute_box_corners(center: torch.Tensor, size: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    center = center.detach().float().flatten()
    size = size.detach().float().flatten().clamp_min(1e-6)
    hx, hy, hz = (size / 2.0).tolist()
    local = torch.tensor([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=torch.float32)
    R = quat_xyzw_to_rotmat(quat)
    return local @ R.T + center[None, :]


def points_inside_obb(
    points: torch.Tensor,
    center: torch.Tensor,
    size: torch.Tensor,
    quat: torch.Tensor,
) -> torch.Tensor:
    pts = points.detach().float()
    center = center.detach().float().flatten()
    size = size.detach().float().flatten().clamp_min(1e-6)
    R = quat_xyzw_to_rotmat(quat)
    local = (pts - center[None, :]) @ R
    return (torch.abs(local) <= (size[None, :] / 2.0 + 1e-6)).all(dim=1)


def approximate_oriented_3d_iou(
    center_a: torch.Tensor,
    size_a: torch.Tensor,
    quat_a: torch.Tensor,
    center_b: torch.Tensor,
    size_b: torch.Tensor,
    quat_b: torch.Tensor,
    *,
    num_samples: int,
    seed: int,
) -> float:
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
    inter = int((in_a & in_b).sum().item())
    union = int((in_a | in_b).sum().item())
    return float(inter / union) if union else 0.0


# =============================================================================
# MOBIMA context
# =============================================================================


class MobimaContext:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.samples_dir = self.root_dir / "samples"
        self.dataset_meta_path = self.root_dir / "dataset_meta.json"
        self.calibration_path = self.root_dir / "calibration.json"
        self.classes_path = self.root_dir / "_classes.json"

        for p in (self.samples_dir, self.dataset_meta_path, self.calibration_path):
            if not p.exists():
                raise FileNotFoundError(f"Required MOBIMA path not found: {p}")

        self.dataset_meta = load_json(self.dataset_meta_path)
        self.calibration = load_json(self.calibration_path)
        self.reference_frame = str(self.calibration.get("reference_frame", "mid360"))
        self.camera_info = self._camera_info()
        self.T_cam_ref = self._load_direct_transform(self.reference_frame, "camera")

        # Fail early if calibration declares a non-OpenCV optical camera convention.
        camera_cfg = self.calibration.get("frames", {}).get("camera", {})
        conv = camera_cfg.get("coordinate_convention", {})
        if conv:
            expected = {"x": "right", "y": "down", "z": "forward"}
            for k, v in expected.items():
                if str(conv.get(k, "")).lower() != v:
                    raise ValueError(
                        "This runner expects OpenCV optical camera coordinates "
                        f"{expected}, but calibration.json declares {conv}."
                    )

    def _camera_info(self) -> Mapping[str, Any]:
        camera_info = self.dataset_meta.get("camera_info")
        if not isinstance(camera_info, Mapping):
            raise ValueError("dataset_meta.json does not contain camera_info")
        return camera_info

    def _load_direct_transform(self, source: str, target: str) -> torch.Tensor:
        key = f"{source}_to_{target}"
        entry = self.calibration.get("transforms", {}).get(key)
        if not isinstance(entry, Mapping) or "matrix" not in entry:
            raise KeyError(f"calibration.json lacks transforms.{key}.matrix")
        return as_transform(entry["matrix"], key)

    def T_sensor_to_reference(self, sensor: str) -> torch.Tensor:
        if sensor == self.reference_frame:
            return torch.eye(4, dtype=torch.float32)
        return self._load_direct_transform(sensor, self.reference_frame)

    def T_camera_from_sensor(self, sensor: str) -> torch.Tensor:
        # p_cam = T_ref_to_cam @ T_sensor_to_ref @ p_sensor
        return self.T_cam_ref @ self.T_sensor_to_reference(sensor)

    def T_sensor_from_reference(self, sensor: str) -> torch.Tensor:
        return torch.linalg.inv(self.T_sensor_to_reference(sensor))

    def frame_dir(self, frame_id: str) -> Path:
        return self.samples_dir / frame_id

    def image_path(self, frame_id: str) -> Path:
        return self.frame_dir(frame_id) / "image_rgb.jpg"

    def pcd_path(self, frame_id: str, sensor: str) -> Path:
        return self.frame_dir(frame_id) / f"{sensor}.pcd"

    def metadata_path(self, frame_id: str) -> Path:
        return self.frame_dir(frame_id) / "metadata.json"

    def labels3d_path(self, frame_id: str) -> Path:
        return self.frame_dir(frame_id) / "labels_3d.json"

    def pseudo2d_path(self, frame_id: str) -> Path:
        return self.frame_dir(frame_id) / "pseudo_labels_2d.json"

    def discover_common_sensor_frames(self, sensors: Sequence[str]) -> List[str]:
        ids: List[str] = []
        for d in sorted(self.samples_dir.iterdir()):
            if not d.is_dir():
                continue
            if not (d / "image_rgb.jpg").is_file():
                continue
            if all((d / f"{s}.pcd").is_file() for s in sensors):
                ids.append(d.name)
        if not ids:
            raise RuntimeError(f"No common frames found for sensors={list(sensors)}")
        return ids

    def discover_task_frames(self, sensors: Sequence[str]) -> List[str]:
        ids = []
        for frame_id in self.discover_common_sensor_frames(sensors):
            d = self.frame_dir(frame_id)
            if (d / "labels_3d.json").is_file() and (d / "pseudo_labels_2d.json").is_file():
                ids.append(frame_id)
        if not ids:
            raise RuntimeError(
                "No frames containing both labels_3d.json and pseudo_labels_2d.json were found."
            )
        return ids

    def load_image_and_intrinsics(self, frame_id: str) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        rgb = load_rgb_image(self.image_path(frame_id))
        size = image_size_from_tensor(rgb)
        K = scaled_intrinsics(self.camera_info, size)
        return rgb, K, size

    def timestamp_s(self, frame_id: str, sensor: Optional[str] = None) -> Optional[float]:
        path = self.metadata_path(frame_id)
        if not path.is_file():
            return None
        meta = load_json(path)
        key_ns = "rgb_time_ns"
        if sensor is not None:
            lidar = meta.get("lidars", {}).get(sensor, {})
            if isinstance(lidar, Mapping) and lidar.get("time_ns") is not None:
                return float(lidar["time_ns"]) * 1e-9
        if meta.get(key_ns) is not None:
            return float(meta[key_ns]) * 1e-9
        return None

    def raw_sample(
        self,
        frame_id: str,
        sensor: str,
        *,
        rgb: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
        lidar_xyz: Optional[torch.Tensor] = None,
    ) -> DataSample:
        if rgb is None or K is None or image_size is None:
            rgb, K, image_size = self.load_image_and_intrinsics(frame_id)
        if lidar_xyz is None:
            lidar_xyz = load_pcd_xyz(self.pcd_path(frame_id, sensor))
        return DataSample(
            setup_id=sensor,
            frame_id=frame_id,
            timestamp_s=self.timestamp_s(frame_id, sensor),
            sequence_id="mobima",
            rgb=rgb,
            lidar_xyz=lidar_xyz,
            intrinsics=K,
            T_cam_lidar=self.T_camera_from_sensor(sensor),
            image_size=image_size,
            meta={
                "dataset_id": "mobima",
                "sensor": sensor,
                "reference_frame": self.reference_frame,
                "root_dir": str(self.root_dir),
                "image_path": str(self.image_path(frame_id)),
                "pcd_path": str(self.pcd_path(frame_id, sensor)),
            },
        )


# =============================================================================
# LabelCloud 3D boxes + pseudo 2D matching
# =============================================================================


@dataclass(frozen=True)
class GTBox:
    name: str
    vertices_ref: torch.Tensor     # (8,3)
    center_ref: torch.Tensor       # (3,)
    size_local: torch.Tensor       # (3,) [edge03, edge04, edge01]
    R_box_ref: torch.Tensor        # (3,3), local -> reference
    quat_ref: torch.Tensor         # (4,) XYZW


@dataclass(frozen=True)
class PseudoBox:
    name: str
    confidence: float
    bbox_xyxy: torch.Tensor


def labelcloud_object_to_gt_box(obj: Mapping[str, Any]) -> GTBox:
    vertices = torch.tensor(obj.get("vertices"), dtype=torch.float32)
    if vertices.shape != (8, 3):
        raise ValueError(f"LabelCloud box must have 8x3 vertices, got {tuple(vertices.shape)}")

    # labelCloud ordering (also used by the viewer): 0-1 and 0-3 are bottom-face
    # edges, 0-4 joins the two faces. We only need a physically equivalent OBB;
    # axis signs do not matter for the cuboid.
    e0 = vertices[3] - vertices[0]
    e1 = vertices[4] - vertices[0]
    e2_raw = vertices[1] - vertices[0]

    s0 = e0.norm().clamp_min(1e-6)
    s1 = e1.norm().clamp_min(1e-6)
    s2 = e2_raw.norm().clamp_min(1e-6)

    a0 = e0 / s0
    # Gram-Schmidt makes the stored rotation numerically orthonormal.
    e1_orth = e1 - torch.dot(e1, a0) * a0
    a1 = e1_orth / e1_orth.norm().clamp_min(1e-6)
    a2 = torch.linalg.cross(a0, a1)
    a2 = a2 / a2.norm().clamp_min(1e-6)

    R = torch.stack([a0, a1, a2], dim=1)
    center = vertices.mean(dim=0)
    size = torch.stack([s0, s1, s2]).float()
    quat = rotmat_to_quat_xyzw(R)

    return GTBox(
        name=str(obj.get("name", "")).lower(),
        vertices_ref=vertices,
        center_ref=center,
        size_local=size,
        R_box_ref=R,
        quat_ref=quat,
    )


def load_gt_boxes(path: Path, task_classes: Sequence[str]) -> List[GTBox]:
    payload = load_json(path)
    allowed = {x.lower() for x in task_classes}
    out: List[GTBox] = []
    for obj in payload.get("objects", []):
        name = str(obj.get("name", "")).lower()
        if name not in allowed:
            continue
        try:
            out.append(labelcloud_object_to_gt_box(obj))
        except Exception as exc:
            print(f"[WARN] Skipping malformed 3D box in {path}: {exc}")
    return out


def load_pseudo_boxes(
    path: Path,
    image_size: Tuple[int, int],
    min_conf: float,
) -> List[PseudoBox]:
    payload = load_json(path)
    actual_h, actual_w = image_size
    info = payload.get("image", {})
    pseudo_w = int(info.get("width", actual_w) or actual_w)
    pseudo_h = int(info.get("height", actual_h) or actual_h)
    sx = float(actual_w) / max(float(pseudo_w), 1.0)
    sy = float(actual_h) / max(float(pseudo_h), 1.0)

    out: List[PseudoBox] = []
    for obj in payload.get("objects", []):
        conf = float(obj.get("confidence", 1.0))
        if conf < min_conf:
            continue
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        box = torch.tensor([x1*sx, y1*sy, x2*sx, y2*sy], dtype=torch.float32)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        out.append(PseudoBox(
            name=str(obj.get("class", "")).lower(),
            confidence=conf,
            bbox_xyxy=box,
        ))
    return out


def project_gt_vertices_to_bbox(
    gt: GTBox,
    K: torch.Tensor,
    T_cam_ref: torch.Tensor,
    image_size: Tuple[int, int],
) -> Optional[torch.Tensor]:
    cam = transform_points(gt.vertices_ref, T_cam_ref)
    uv, depth = project_camera_points_opencv(cam, K)
    valid = depth > 1e-4
    if int(valid.sum().item()) < 2:
        return None
    uv = uv[valid]
    x1 = float(uv[:, 0].min().item())
    y1 = float(uv[:, 1].min().item())
    x2 = float(uv[:, 0].max().item())
    y2 = float(uv[:, 1].max().item())
    h, w = image_size
    # Clamp for a meaningful 2D overlap score if part of a box is outside image.
    x1 = max(0.0, min(float(w - 1), x1))
    x2 = max(0.0, min(float(w - 1), x2))
    y1 = max(0.0, min(float(h - 1), y1))
    y2 = max(0.0, min(float(h - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)


def bbox_iou_2d(a: torch.Tensor, b: torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a.tolist()]
    bx1, by1, bx2, by2 = [float(v) for v in b.tolist()]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def class_compatible(gt_name: str, pseudo_name: str) -> bool:
    gt = gt_name.lower()
    ps = pseudo_name.lower()
    if gt == ps:
        return True
    # YOLO has no dedicated "van" class. A van is most often predicted as car
    # and sometimes truck, so allow either only for GT van.
    if gt == "van" and ps in {"car", "truck"}:
        return True
    return False


def match_gt_to_pseudo(
    gt_boxes: Sequence[GTBox],
    pseudo_boxes: Sequence[PseudoBox],
    *,
    K: torch.Tensor,
    T_cam_ref: torch.Tensor,
    image_size: Tuple[int, int],
    min_iou: float,
    allow_class_mismatch: bool,
) -> List[Tuple[int, int, float, bool]]:
    """Return (gt_idx, pseudo_idx, projected_iou, class_match), one-to-one."""
    if not gt_boxes or not pseudo_boxes:
        return []

    proj = [project_gt_vertices_to_bbox(g, K, T_cam_ref, image_size) for g in gt_boxes]
    n, m = len(gt_boxes), len(pseudo_boxes)
    score = np.full((n, m), -1e6, dtype=np.float64)
    iou_mat = np.zeros((n, m), dtype=np.float64)
    compat = np.zeros((n, m), dtype=bool)

    for i, gt in enumerate(gt_boxes):
        for j, ps in enumerate(pseudo_boxes):
            is_compat = class_compatible(gt.name, ps.name)
            compat[i, j] = is_compat
            if not is_compat and not allow_class_mismatch:
                continue
            iou = bbox_iou_2d(proj[i], ps.bbox_xyxy) if proj[i] is not None else 0.0
            iou_mat[i, j] = iou
            # Class compatibility dominates; IoU breaks ties between same-class objects.
            score[i, j] = iou + (1.0 if is_compat else 0.0)

    row_ind, col_ind = linear_sum_assignment(-score)
    matches: List[Tuple[int, int, float, bool]] = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        if score[i, j] < -1e5:
            continue
        iou = float(iou_mat[i, j])
        is_compat = bool(compat[i, j])

        # If there is exactly one compatible pseudo box for this GT class, the
        # class identity is enough to pair it. This is useful when calibration or
        # a partly-outside-image box makes projected bbox IoU unreliable.
        n_compat = int(compat[i].sum())
        unique_class_fallback = is_compat and n_compat == 1
        if iou < min_iou and not unique_class_fallback:
            continue
        matches.append((i, j, iou, is_compat))
    return matches


# =============================================================================
# Frustum extraction for OpenCV optical camera geometry
# =============================================================================


def extract_frustum_points(
    lidar_xyz: torch.Tensor,
    box_xyxy: torch.Tensor,
    K: torch.Tensor,
    T_cam_lidar: torch.Tensor,
    image_size: Tuple[int, int],
    *,
    num_points: int,
    box_margin_px: float,
    min_depth: float,
    seed: int,
) -> Tuple[torch.Tensor, int]:
    _, uv, depth = project_lidar_points_opencv(lidar_xyz, K, T_cam_lidar)
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    x1 -= box_margin_px
    y1 -= box_margin_px
    x2 += box_margin_px
    y2 += box_margin_px
    h, w = image_size

    u, v = uv[:, 0], uv[:, 1]
    mask = depth > float(min_depth)
    mask &= torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(depth)
    mask &= (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    mask &= (u >= 0.0) & (u <= float(w - 1)) & (v >= 0.0) & (v <= float(h - 1))
    frustum = lidar_xyz[mask]
    n_raw = int(frustum.shape[0])
    if n_raw == 0:
        raise ValueError("Frustum contains zero LiDAR points")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if n_raw >= num_points:
        idx = torch.randperm(n_raw, generator=gen)[:num_points]
        sampled = frustum[idx]
    else:
        pad = torch.randint(0, n_raw, (num_points - n_raw,), generator=gen)
        sampled = torch.cat([frustum, frustum[pad]], dim=0)
    return sampled.float(), n_raw


# =============================================================================
# Task evaluation
# =============================================================================


def load_task_model(ckpt_path: Path, device: str, num_object_points: int) -> FrustumPointNet:
    model = FrustumPointNet(
        input_channels=3,
        num_seg_classes=2,
        num_object_points=num_object_points,
        use_bn=False,
    ).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded task checkpoint: {ckpt_path}")
    print(f"[INFO] Checkpoint epoch: {ckpt.get('epoch') if isinstance(ckpt, dict) else None}")
    print(f"[INFO] Best eval center err: {ckpt.get('best_eval_center_err') if isinstance(ckpt, dict) else None}")
    return model


def gt_box_in_sensor(gt: GTBox, ctx: MobimaContext, sensor: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T_sensor_ref = ctx.T_sensor_from_reference(sensor)
    center = transform_point(gt.center_ref, T_sensor_ref)
    R_sensor_ref = T_sensor_ref[:3, :3]
    R_box_sensor = R_sensor_ref @ gt.R_box_ref
    quat = rotmat_to_quat_xyzw(R_box_sensor)
    return center, gt.size_local.clone(), quat


def center_to_camera(center_sensor: torch.Tensor, T_cam_sensor: torch.Tensor) -> torch.Tensor:
    return transform_point(center_sensor, T_cam_sensor)


def quaternion_rotation_error_deg(q_pred: torch.Tensor, q_gt: torch.Tensor) -> float:
    a = q_pred.detach().float().flatten()
    b = q_gt.detach().float().flatten()
    a = a / a.norm().clamp_min(1e-12)
    b = b / b.norm().clamp_min(1e-12)
    dot = torch.abs(torch.dot(a, b)).clamp(-1.0, 1.0)
    return float(torch.rad2deg(2.0 * torch.acos(dot)).item())


@torch.no_grad()
def evaluate_task_network(
    *,
    ctx: MobimaContext,
    sensor: str,
    frame_ids: Sequence[str],
    model: FrustumPointNet,
    device: str,
    seed: int,
    num_points: int,
    iou_samples: int,
    box_margin_px: float,
    min_depth: float,
    task_classes: Sequence[str],
    min_pseudo_confidence: float,
    min_match_iou: float,
    allow_class_mismatch: bool,
) -> Dict[str, object]:
    """
    Evaluate FrustumPointNet while explicitly accounting for LiDAR coverage.

    There are two task views:

    1) Valid-prediction metrics
       Center error, IoU, confidence, etc. are computed only for objects whose
       2D proposal contains at least one LiDAR point and for which the network
       can therefore produce a prediction.

    2) Coverage-aware metrics
       The denominator is every GT object that was successfully matched to a
       pseudo-2D proposal. If the selected LiDAR contributes zero points to that
       object's frustum, the object is retained and assigned IoU = 0.

       We intentionally do NOT invent a center-error penalty for such failures,
       because there is no meaningful predicted center to compare against.

    This separation prevents narrow-FOV / low-coverage sensors from appearing
    artificially strong simply because uncovered objects disappear from the
    task averages.
    """
    center_err_sensor: List[float] = []
    center_err_cam: List[float] = []

    # Successful predictions only.
    ious_valid: List[float] = []
    confidences: List[float] = []
    rot_errs: List[float] = []

    # Every GT object that has a usable pseudo-2D proposal. Frustum failures
    # are inserted as IoU=0 so missing LiDAR coverage is a real task failure.
    ious_all_matched: List[float] = []

    per_object_rows: List[Dict[str, object]] = []

    skipped_no_boxes = 0
    skipped_no_match = 0
    skipped_frustum = 0

    num_gt_objects = 0
    num_matched_objects = 0
    num_valid_predictions = 0
    num_frames_loaded = 0

    T_cam_sensor = ctx.T_camera_from_sensor(sensor)

    for local_i, frame_id in enumerate(frame_ids, start=1):
        if local_i == 1 or local_i % 25 == 0 or local_i == len(frame_ids):
            print(f"[INFO] Task eval {sensor}: frame {local_i}/{len(frame_ids)} ({frame_id})")

        # Load camera/annotations first. These define the sensor-independent
        # evaluation objects and 2D proposals.
        try:
            rgb, K, image_size = ctx.load_image_and_intrinsics(frame_id)
            gt_boxes = load_gt_boxes(ctx.labels3d_path(frame_id), task_classes)
            pseudo_boxes = load_pseudo_boxes(
                ctx.pseudo2d_path(frame_id), image_size, min_pseudo_confidence
            )
        except Exception as exc:
            print(f"[WARN] Task frame {frame_id} annotation load failed: {type(exc).__name__}: {exc}")
            skipped_no_boxes += 1
            continue

        num_frames_loaded += 1
        num_gt_objects += len(gt_boxes)

        if not gt_boxes or not pseudo_boxes:
            skipped_no_boxes += 1
            if gt_boxes and not pseudo_boxes:
                skipped_no_match += len(gt_boxes)
            continue

        matches = match_gt_to_pseudo(
            gt_boxes,
            pseudo_boxes,
            K=K,
            T_cam_ref=ctx.T_cam_ref,
            image_size=image_size,
            min_iou=min_match_iou,
            allow_class_mismatch=allow_class_mismatch,
        )
        if not matches:
            skipped_no_match += len(gt_boxes)
            continue

        matched_gt = {i for i, _, _, _ in matches}
        skipped_no_match += max(0, len(gt_boxes) - len(matched_gt))
        num_matched_objects += len(matches)

        # If the LiDAR file itself cannot be loaded, every matched proposal in
        # this frame is an unavailable LiDAR prediction. Count them as coverage
        # failures instead of dropping them from the denominator.
        try:
            lidar_xyz = load_pcd_xyz(ctx.pcd_path(frame_id, sensor))
            lidar_load_error: Optional[Exception] = None
        except Exception as exc:
            lidar_xyz = torch.empty((0, 3), dtype=torch.float32)
            lidar_load_error = exc
            print(
                f"[WARN] {sensor}/{frame_id} LiDAR load failed; "
                f"all {len(matches)} matched objects count as task failures: "
                f"{type(exc).__name__}: {exc}"
            )

        for gt_idx, pseudo_idx, match_iou, class_match in matches:
            gt = gt_boxes[gt_idx]
            ps = pseudo_boxes[pseudo_idx]

            failure_reason: Optional[str] = None
            sampled: Optional[torch.Tensor] = None
            n_frustum = 0

            if lidar_load_error is not None:
                failure_reason = (
                    f"lidar_load_failed:{type(lidar_load_error).__name__}:"
                    f"{lidar_load_error}"
                )
            else:
                try:
                    sampled, n_frustum = extract_frustum_points(
                        lidar_xyz,
                        ps.bbox_xyxy,
                        K,
                        T_cam_sensor,
                        image_size,
                        num_points=num_points,
                        box_margin_px=box_margin_px,
                        min_depth=min_depth,
                        seed=stable_int_seed(sensor, frame_id, gt_idx, base=seed),
                    )
                except Exception as exc:
                    failure_reason = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[WARN] {sensor}/{frame_id} frustum failed for {gt.name}: "
                        f"{failure_reason}"
                    )

            if failure_reason is not None or sampled is None:
                skipped_frustum += 1
                ious_all_matched.append(0.0)

                # Keep failed objects in the audit CSV. Center error/confidence
                # are undefined because the task network produced no prediction.
                per_object_rows.append({
                    "sensor": sensor,
                    "frame_id": frame_id,
                    "gt_object_index": gt_idx,
                    "gt_class": gt.name,
                    "pseudo_class": ps.name,
                    "pseudo_confidence": ps.confidence,
                    "gt_pseudo_projected_iou": match_iou,
                    "class_match": class_match,
                    "task_status": "frustum_failure",
                    "frustum_success": False,
                    "failure_reason": failure_reason or "unknown_frustum_failure",
                    "center_error_sensor": float("nan"),
                    "center_error_cam": float("nan"),
                    "iou3d_approx": 0.0,
                    "confidence": float("nan"),
                    "rotation_error_deg_naive": float("nan"),
                    "num_frustum_points": 0,
                    "num_sampled_points": 0,
                    "gt_center_ref_x": float(gt.center_ref[0]),
                    "gt_center_ref_y": float(gt.center_ref[1]),
                    "gt_center_ref_z": float(gt.center_ref[2]),
                    "gt_size_0": float(gt.size_local[0]),
                    "gt_size_1": float(gt.size_local[1]),
                    "gt_size_2": float(gt.size_local[2]),
                })
                continue

            pred = model(sampled.unsqueeze(0).to(device))
            pred_center = pred["center_lidar"][0].detach().cpu()
            pred_size = pred["size_lidar"][0].detach().cpu().clamp_min(1e-6)
            pred_quat = pred["quat_xyzw"][0].detach().cpu()
            pred_conf = float(pred["confidence"][0].detach().cpu().item())

            gt_center, gt_size, gt_quat = gt_box_in_sensor(gt, ctx, sensor)
            pred_cam = center_to_camera(pred_center, T_cam_sensor)
            gt_cam = transform_point(gt.center_ref, ctx.T_cam_ref)

            err_sensor = float(torch.linalg.norm(pred_center - gt_center).item())
            err_cam = float(torch.linalg.norm(pred_cam - gt_cam).item())
            iou = approximate_oriented_3d_iou(
                pred_center, pred_size, pred_quat,
                gt_center, gt_size, gt_quat,
                num_samples=iou_samples,
                seed=stable_int_seed("iou", sensor, frame_id, gt_idx, base=seed),
            )
            rot_err = quaternion_rotation_error_deg(pred_quat, gt_quat)

            num_valid_predictions += 1
            center_err_sensor.append(err_sensor)
            center_err_cam.append(err_cam)
            ious_valid.append(iou)
            ious_all_matched.append(iou)
            confidences.append(pred_conf)
            rot_errs.append(rot_err)

            per_object_rows.append({
                "sensor": sensor,
                "frame_id": frame_id,
                "gt_object_index": gt_idx,
                "gt_class": gt.name,
                "pseudo_class": ps.name,
                "pseudo_confidence": ps.confidence,
                "gt_pseudo_projected_iou": match_iou,
                "class_match": class_match,
                "task_status": "predicted",
                "frustum_success": True,
                "failure_reason": "",
                "center_error_sensor": err_sensor,
                "center_error_cam": err_cam,
                "iou3d_approx": iou,
                "confidence": pred_conf,
                "rotation_error_deg_naive": rot_err,
                "num_frustum_points": n_frustum,
                "num_sampled_points": num_points,
                "gt_center_ref_x": float(gt.center_ref[0]),
                "gt_center_ref_y": float(gt.center_ref[1]),
                "gt_center_ref_z": float(gt.center_ref[2]),
                "gt_size_0": float(gt.size_local[0]),
                "gt_size_1": float(gt.size_local[1]),
                "gt_size_2": float(gt.size_local[2]),
            })

    pseudo_match_rate = (
        float(num_matched_objects) / float(num_gt_objects)
        if num_gt_objects > 0 else float("nan")
    )
    frustum_success_rate = (
        float(num_valid_predictions) / float(num_matched_objects)
        if num_matched_objects > 0 else float("nan")
    )
    frustum_failure_rate = (
        float(skipped_frustum) / float(num_matched_objects)
        if num_matched_objects > 0 else float("nan")
    )
    prediction_success_rate = (
        float(num_valid_predictions) / float(num_gt_objects)
        if num_gt_objects > 0 else float("nan")
    )

    summary: Dict[str, float] = {
        "task_num_frames": float(len(frame_ids)),
        "task_num_frames_loaded": float(num_frames_loaded),

        # Explicit denominators / failure accounting.
        "task_num_gt_objects": float(num_gt_objects),
        "task_num_matched_objects": float(num_matched_objects),
        "task_num_valid_predictions": float(num_valid_predictions),
        # Backward-compatible meaning: objects with a valid network prediction.
        "task_num_objects": float(num_valid_predictions),
        "task_num_skipped_no_boxes": float(skipped_no_boxes),
        "task_num_skipped_no_match": float(skipped_no_match),
        "task_num_skipped_frustum": float(skipped_frustum),
        "task_pseudo_match_rate": pseudo_match_rate,
        "task_frustum_success_rate": frustum_success_rate,
        "task_frustum_failure_rate": frustum_failure_rate,
        "task_prediction_success_rate": prediction_success_rate,

        # Center error is intentionally VALID-PREDICTION ONLY.
        "center_error_sensor_mean_valid": mean(center_err_sensor),
        "center_error_sensor_median_valid": median(center_err_sensor),
        "center_error_cam_mean_valid": mean(center_err_cam),
        "center_error_cam_median_valid": median(center_err_cam),

        # Backward-compatible aliases. These retain their previous conditional
        # meaning and do NOT include an arbitrary penalty for failed frustums.
        "center_error_sensor_mean": mean(center_err_sensor),
        "center_error_sensor_median": median(center_err_sensor),
        "center_error_cam_mean": mean(center_err_cam),
        "center_error_cam_median": median(center_err_cam),

        # IoU among successful predictions only (old behavior).
        "iou3d_approx_mean_valid": mean(ious_valid),
        "iou3d_approx_median_valid": median(ious_valid),
        "iou3d_approx_at_025_valid": mean([1.0 if x >= 0.25 else 0.0 for x in ious_valid]),
        "iou3d_approx_at_050_valid": mean([1.0 if x >= 0.50 else 0.0 for x in ious_valid]),

        # Backward-compatible aliases for the old valid-only IoU metrics.
        "iou3d_approx_mean": mean(ious_valid),
        "iou3d_approx_median": median(ious_valid),
        "iou3d_approx_at_025": mean([1.0 if x >= 0.25 else 0.0 for x in ious_valid]),
        "iou3d_approx_at_050": mean([1.0 if x >= 0.50 else 0.0 for x in ious_valid]),

        # Coverage-aware task metrics. Every GT object with a matched 2D proposal
        # contributes; a zero-point / unavailable LiDAR frustum contributes IoU=0.
        "iou3d_approx_mean_all_matched": mean(ious_all_matched),
        "iou3d_approx_median_all_matched": median(ious_all_matched),
        "iou3d_approx_at_025_all_matched": mean(
            [1.0 if x >= 0.25 else 0.0 for x in ious_all_matched]
        ),
        "iou3d_approx_at_050_all_matched": mean(
            [1.0 if x >= 0.50 else 0.0 for x in ious_all_matched]
        ),

        "confidence_mean_valid": mean(confidences),
        "confidence_mean": mean(confidences),

        # The LabelCloud cuboid is physically well-defined, but axis signs are
        # arbitrary. Keep this diagnostic, but do not use it as the main task metric.
        "rotation_error_deg_naive_mean_valid": mean(rot_errs),
        "rotation_error_deg_naive_median_valid": median(rot_errs),
        "rotation_error_deg_naive_mean": mean(rot_errs),
        "rotation_error_deg_naive_median": median(rot_errs),
    }
    return {"summary": summary, "per_object_rows": per_object_rows}


# =============================================================================
# MOBIMA CRT residual extraction
# =============================================================================


def rgb_to_gray_numpy(rgb: torch.Tensor) -> np.ndarray:
    x = rgb.detach().cpu().float().numpy()
    gray = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    return gray.astype(np.float32, copy=False)


def edge_distance_map(rgb: torch.Tensor, edge_threshold: float) -> np.ndarray:
    gray = rgb_to_gray_numpy(rgb)
    gx = sobel(gray, axis=1, mode="reflect")
    gy = sobel(gray, axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    max_mag = float(np.max(mag)) if mag.size else 0.0
    if max_mag <= 1e-12:
        raise ValueError("RGB image has no usable Sobel gradients")
    mag = mag / max_mag
    edges = mag >= float(edge_threshold)
    if not np.any(edges):
        raise ValueError(
            f"No RGB edges survived edge_threshold={edge_threshold}; lower the threshold."
        )
    # distance_transform_edt computes distance to the nearest zero. Invert edges
    # so edge pixels are zero and every other pixel gets distance-to-edge.
    return distance_transform_edt(~edges).astype(np.float32)


def projection_residuals_mobima(
    lidar_xyz: torch.Tensor,
    K: torch.Tensor,
    T_cam_lidar: torch.Tensor,
    image_size: Tuple[int, int],
    distance_map: np.ndarray,
    *,
    max_points: int,
    min_depth: float,
    seed: int,
) -> torch.Tensor:
    _, uv, depth = project_lidar_points_opencv(lidar_xyz, K, T_cam_lidar)
    h, w = image_size
    u, v = uv[:, 0], uv[:, 1]
    valid = depth > float(min_depth)
    valid &= torch.isfinite(u) & torch.isfinite(v)
    valid &= (u >= 0.0) & (u <= float(w - 1)) & (v >= 0.0) & (v <= float(h - 1))
    uv_valid = uv[valid]
    if uv_valid.shape[0] == 0:
        raise ValueError("No LiDAR points project into the RGB image")

    if max_points > 0 and uv_valid.shape[0] > max_points:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        idx = torch.randperm(uv_valid.shape[0], generator=gen)[:max_points]
        uv_valid = uv_valid[idx]

    px = torch.round(uv_valid[:, 0]).long().clamp(0, w - 1).cpu().numpy()
    py = torch.round(uv_valid[:, 1]).long().clamp(0, h - 1).cpu().numpy()
    residual = distance_map[py, px]
    return torch.from_numpy(np.asarray(residual, dtype=np.float32)).flatten()


# =============================================================================
# Metric caching + encoding
# =============================================================================


class MetricCache:
    def __init__(
        self,
        root: Path,
        *,
        ctx: MobimaContext,
        rgb_extractor: DINOv2RGBExtractor,
        lidar_extractor: PointNet2LiDARExtractor,
        residual_edge_threshold: float,
        residual_max_points: int,
        residual_min_depth: float,
        residuals_enabled: bool,
        refresh: bool,
        enabled: bool,
    ) -> None:
        self.root = root
        self.enabled = bool(enabled)
        self.refresh = bool(refresh)
        calibration_hash = sha1_file(ctx.calibration_path)
        meta_hash = sha1_file(ctx.dataset_meta_path)

        self.rgb_key = hash_config({
            "rgb_extractor": rgb_extractor.extractor_id,
        })
        self.sensor_key = hash_config({
            "lidar_extractor": lidar_extractor.extractor_id,
            "calibration_sha1": calibration_hash,
            "dataset_meta_sha1": meta_hash,
            "residuals_enabled": residuals_enabled,
            "residual_edge_threshold": residual_edge_threshold,
            "residual_max_points": residual_max_points,
            "residual_min_depth": residual_min_depth,
            "residual_impl": "mobima_opencv_sobel_distance_v1",
        })

    def rgb_path(self, frame_id: str) -> Path:
        return self.root / "rgb" / self.rgb_key / f"{frame_id}.pt"

    def sensor_path(self, sensor: str, frame_id: str) -> Path:
        return self.root / sensor / self.sensor_key / f"{frame_id}.pt"

    def load_rgb(self, frame_id: str) -> Optional[torch.Tensor]:
        if not self.enabled or self.refresh:
            return None
        p = self.rgb_path(frame_id)
        if not p.is_file():
            return None
        payload = torch.load(p, map_location="cpu", weights_only=False)
        return payload.get("rgb_emb") if isinstance(payload, dict) else None

    def save_rgb(self, frame_id: str, rgb_emb: torch.Tensor) -> None:
        if not self.enabled:
            return
        p = self.rgb_path(frame_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rgb_emb": rgb_emb.detach().cpu()}, p)

    def load_sensor(self, sensor: str, frame_id: str) -> Optional[Dict[str, torch.Tensor]]:
        if not self.enabled or self.refresh:
            return None
        p = self.sensor_path(sensor, frame_id)
        if not p.is_file():
            return None
        payload = torch.load(p, map_location="cpu", weights_only=False)
        return payload if isinstance(payload, dict) else None

    def save_sensor(self, sensor: str, frame_id: str, payload: Dict[str, torch.Tensor]) -> None:
        if not self.enabled:
            return
        p = self.sensor_path(sensor, frame_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        clean = {k: v.detach().cpu() for k, v in payload.items() if torch.is_tensor(v)}
        torch.save(clean, p)


def finite_tensor_or_none(x: Optional[torch.Tensor]) -> bool:
    return x is None or bool(torch.isfinite(x).all().item())


def encode_metric_samples_all_sensors(
    *,
    ctx: MobimaContext,
    sensors: Sequence[str],
    frame_ids: Sequence[str],
    rgb_extractor: DINOv2RGBExtractor,
    lidar_extractor: PointNet2LiDARExtractor,
    joint_extractor: ConcatJointEmbeddingExtractor,
    cache: MetricCache,
    residuals_enabled: bool,
    residual_edge_threshold: float,
    residual_max_points: int,
    residual_min_depth: float,
    seed: int,
) -> Dict[str, List[DataSample]]:
    out: Dict[str, List[DataSample]] = {s: [] for s in sensors}

    for frame_i, frame_id in enumerate(frame_ids, start=1):
        if frame_i == 1 or frame_i % 25 == 0 or frame_i == len(frame_ids):
            print(f"[INFO] Metric encoding common frame {frame_i}/{len(frame_ids)}: {frame_id}")

        rgb_emb = cache.load_rgb(frame_id)
        sensor_cached = {s: cache.load_sensor(s, frame_id) for s in sensors}

        # We need the actual image if RGB embedding or any residual is missing.
        need_image = rgb_emb is None or (
            residuals_enabled
            and any(c is None or c.get("residuals") is None for c in sensor_cached.values())
        )
        rgb: Optional[torch.Tensor] = None
        K: Optional[torch.Tensor] = None
        image_size: Optional[Tuple[int, int]] = None
        dist_map: Optional[np.ndarray] = None

        if need_image:
            rgb, K, image_size = ctx.load_image_and_intrinsics(frame_id)

        if rgb_emb is None:
            assert rgb is not None and K is not None and image_size is not None
            sample_rgb = DataSample(
                setup_id="mobima_rgb",
                frame_id=frame_id,
                rgb=rgb,
                intrinsics=K,
                image_size=image_size,
            )
            torch.manual_seed(stable_int_seed("rgb", frame_id, base=seed))
            rgb_emb = rgb_extractor.extract(sample_rgb)
            cache.save_rgb(frame_id, rgb_emb)

        for sensor in sensors:
            cached = sensor_cached[sensor]
            lidar_emb: Optional[torch.Tensor] = None
            residuals: Optional[torch.Tensor] = None
            if cached is not None:
                lidar_emb = cached.get("lidar_emb")
                residuals = cached.get("residuals")

            need_lidar = lidar_emb is None or (residuals_enabled and residuals is None)
            lidar_xyz: Optional[torch.Tensor] = None
            if need_lidar:
                lidar_xyz = load_pcd_xyz(ctx.pcd_path(frame_id, sensor))

            if lidar_emb is None:
                assert lidar_xyz is not None
                if rgb is None or K is None or image_size is None:
                    rgb, K, image_size = ctx.load_image_and_intrinsics(frame_id)
                raw = ctx.raw_sample(
                    frame_id,
                    sensor,
                    rgb=rgb,
                    K=K,
                    image_size=image_size,
                    lidar_xyz=lidar_xyz,
                )
                torch.manual_seed(stable_int_seed("lidar_emb", sensor, frame_id, base=seed))
                lidar_emb = lidar_extractor.extract(raw)

            if residuals_enabled and residuals is None:
                assert lidar_xyz is not None
                if rgb is None or K is None or image_size is None:
                    rgb, K, image_size = ctx.load_image_and_intrinsics(frame_id)
                if dist_map is None:
                    dist_map = edge_distance_map(rgb, residual_edge_threshold)
                residuals = projection_residuals_mobima(
                    lidar_xyz,
                    K,
                    ctx.T_camera_from_sensor(sensor),
                    image_size,
                    dist_map,
                    max_points=residual_max_points,
                    min_depth=residual_min_depth,
                    seed=stable_int_seed("residual", sensor, frame_id, base=seed),
                )

            payload = {"lidar_emb": lidar_emb}
            if residuals is not None:
                payload["residuals"] = residuals
            cache.save_sensor(sensor, frame_id, payload)

            base_sample = DataSample(
                setup_id=sensor,
                frame_id=frame_id,
                timestamp_s=ctx.timestamp_s(frame_id, sensor),
                sequence_id="mobima",
                rgb_emb=rgb_emb,
                lidar_emb=lidar_emb,
                residuals=residuals,
                meta={"dataset_id": "mobima", "sensor": sensor},
            )
            joint_emb = joint_extractor.extract(base_sample)
            sample = replace(base_sample, joint_emb=joint_emb)

            if not all(finite_tensor_or_none(x) for x in (rgb_emb, lidar_emb, joint_emb, residuals)):
                print(f"[WARN] Non-finite metric tensor; skipping {sensor}/{frame_id}")
                continue
            out[sensor].append(sample)

    return out


# =============================================================================
# Metric computation (same E1/E2/E3 family as current multi-LiDAR runner)
# =============================================================================


def with_residuals(samples: Sequence[DataSample]) -> List[DataSample]:
    return [s for s in samples if s.residuals is not None]


def total_residual_count(samples: Sequence[DataSample]) -> int:
    total = 0
    for s in samples:
        if s.residuals is None:
            continue
        total += int(s.residuals.numel()) if s.residuals.ndim <= 1 else int(s.residuals.shape[0])
    return total


def cap_residuals_for_metric(
    samples: Sequence[DataSample],
    *,
    max_total_points: int,
    seed: int,
) -> List[DataSample]:
    if max_total_points <= 0:
        return list(samples)
    chunks: List[torch.Tensor] = []
    representative: Optional[DataSample] = None
    for s in samples:
        if s.residuals is None:
            continue
        r = s.residuals.detach().cpu().float()
        if r.ndim == 0:
            r = r.reshape(1, 1)
        elif r.ndim == 1:
            r = r[:, None]
        elif r.ndim != 2:
            raise ValueError(f"Residuals must be (M,) or (M,K), got {tuple(r.shape)}")
        if r.shape[0] == 0:
            continue
        chunks.append(r)
        representative = representative or s
    if not chunks or representative is None:
        return []
    all_r = torch.cat(chunks, dim=0)
    if all_r.shape[0] <= max_total_points:
        return list(samples)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    idx = torch.randperm(all_r.shape[0], generator=gen)[:max_total_points]
    return [replace(representative, residuals=all_r[idx].contiguous())]


def compute_domain_gap_metrics(
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
            for k, v in result.details.items():
                if isinstance(v, (int, float)):
                    out[f"{metric.name}.{k}"] = float(v)
        except Exception as exc:
            print(f"[WARN] {metric.name} failed: {type(exc).__name__}: {exc}")
            out[metric.name] = float("nan")

    a_raw = with_residuals(a_samples)
    b_raw = with_residuals(b_samples)
    out["A_residual_frame_count"] = float(len(a_raw))
    out["B_residual_frame_count"] = float(len(b_raw))
    out["A_residual_point_count_raw"] = float(total_residual_count(a_raw))
    out["B_residual_point_count_raw"] = float(total_residual_count(b_raw))

    a_res = cap_residuals_for_metric(
        a_raw, max_total_points=residual_metric_max_total_points, seed=seed + 10_001
    )
    b_res = cap_residuals_for_metric(
        b_raw, max_total_points=residual_metric_max_total_points, seed=seed + 20_001
    )
    out["A_residual_point_count_metric"] = float(total_residual_count(a_res))
    out["B_residual_point_count_metric"] = float(total_residual_count(b_res))
    out["residual_metric_max_total_points"] = float(residual_metric_max_total_points)

    for backend in ("sinkhorn", "mmd"):
        name = f"Calibration-Residual-Transport-{backend}"
        if len(a_res) == 0 or len(b_res) == 0:
            out[name] = float("nan")
            continue
        try:
            result = CalibrationResidualTransport(backend=backend)(a_res, b_res)
            out[name] = float(result.score)
        except Exception as exc:
            print(f"[WARN] {name} failed: {type(exc).__name__}: {exc}")
            out[name] = float("nan")
    return out


# =============================================================================
# Outputs
# =============================================================================


def add_performance_deltas(rows: List[Dict[str, object]], baseline_sensor: str) -> None:
    """
    Add deltas with the convention: positive = worse than baseline.

    Error metrics:
        target - baseline
    Quality/success metrics:
        baseline - target
    """
    baseline_rows = [r for r in rows if r.get("sensor") == baseline_sensor]
    if not baseline_rows:
        raise ValueError(f"Baseline sensor {baseline_sensor!r} not in rows")
    base = baseline_rows[0]

    error_cols = [
        "center_error_sensor_mean_valid",
        "center_error_sensor_median_valid",
        "center_error_cam_mean_valid",
        "center_error_cam_median_valid",
        # Backward-compatible aliases.
        "center_error_sensor_mean",
        "center_error_sensor_median",
        "center_error_cam_mean",
        "center_error_cam_median",
        "rotation_error_deg_naive_mean_valid",
        "rotation_error_deg_naive_median_valid",
        "rotation_error_deg_naive_mean",
        "rotation_error_deg_naive_median",
    ]
    score_cols = [
        # Valid-only quality metrics.
        "iou3d_approx_mean_valid",
        "iou3d_approx_median_valid",
        "iou3d_approx_at_025_valid",
        "iou3d_approx_at_050_valid",
        # Backward-compatible aliases.
        "iou3d_approx_mean",
        "iou3d_approx_median",
        "iou3d_approx_at_025",
        "iou3d_approx_at_050",
        "confidence_mean_valid",
        "confidence_mean",
        # Coverage-aware quality/success metrics.
        "iou3d_approx_mean_all_matched",
        "iou3d_approx_median_all_matched",
        "iou3d_approx_at_025_all_matched",
        "iou3d_approx_at_050_all_matched",
        "task_frustum_success_rate",
        "task_prediction_success_rate",
    ]

    for row in rows:
        for col in error_cols:
            try:
                row[f"delta_{col}"] = float(row[col]) - float(base[col])
            except Exception:
                row[f"delta_{col}"] = float("nan")
        for col in score_cols:
            try:
                row[f"delta_{col}"] = float(base[col]) - float(row[col])
            except Exception:
                row[f"delta_{col}"] = float("nan")


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote CSV: {path}")


def save_json(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, allow_nan=True)
    print(f"[INFO] Wrote JSON: {path}")


def print_summary(rows: Sequence[Mapping[str, object]]) -> None:
    cols = [
        "sensor",
        "task_frustum_success_rate",
        "center_error_cam_mean_valid",
        "delta_center_error_cam_mean_valid",
        "iou3d_approx_mean_all_matched",
        "delta_iou3d_approx_mean_all_matched",
        "PRODUCT-MMD",
        "CopulaDecomp-MMD+HSIC",
        "CopulaDecomp-SINKHORN+HSIC",
        "Calibration-Residual-Transport-sinkhorn",
    ]
    width = max(200, 38 * len(cols))
    print("\n" + "=" * width)
    print("MOBIMA SUMMARY")
    print(
        "Task interpretation: center error = valid predictions only; "
        "IoU all_matched = coverage-aware with failed frustums assigned IoU=0."
    )
    print("=" * width)
    print(" | ".join(f"{c:>36s}" for c in cols))
    print("-" * width)
    for row in rows:
        vals = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:36.6f}")
            else:
                vals.append(f"{str(v):>36s}")
        print(" | ".join(vals))


def save_correlations(rows: List[Dict[str, object]], out_csv: Path, baseline_sensor: str) -> None:
    targets = [r for r in rows if r.get("sensor") != baseline_sensor]
    # With MID360/MID70/Velodyne there are only two target configurations.
    # A rank correlation over N=2 is not scientifically useful, so only write it
    # when at least 3 target configurations are supplied.
    if len(targets) < 3:
        print(
            f"[WARN] Only {len(targets)} non-baseline sensor setups: skipping "
            "Spearman/Kendall correlations. Use the per-sensor table as the real-world stress test."
        )
        return

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
    delta_cols = [
        # Conditional localization error: valid predictions only.
        "delta_center_error_cam_mean_valid",
        # Primary coverage-aware task degradation for real sensor comparison.
        "delta_iou3d_approx_mean_all_matched",
        # Explicit sensor-coverage degradation.
        "delta_task_frustum_success_rate",
    ]
    result_rows: List[Dict[str, object]] = []
    for metric in metric_cols:
        for delta in delta_cols:
            pairs = []
            for row in targets:
                try:
                    x, y = float(row[metric]), float(row[delta])
                    if math.isfinite(x) and math.isfinite(y):
                        pairs.append((x, y))
                except Exception:
                    pass
            if len(pairs) < 3:
                continue
            x = [p[0] for p in pairs]
            y = [p[1] for p in pairs]
            sp = spearmanr(x, y)
            kd = kendalltau(x, y)
            result_rows.append({
                "n": len(pairs),
                "metric": metric,
                "performance_delta": delta,
                "spearman_rho": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
                "kendall_tau": float(kd.statistic),
                "kendall_p": float(kd.pvalue),
            })
    if result_rows:
        save_csv(result_rows, out_csv.with_name(out_csv.stem + "_correlations.csv"))


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compare FrustumPointNet performance and ProductMMD/Copula/CRT "
            "domain-gap scores across MOBIMA LiDAR setups."
        )
    )
    ap.add_argument("--root-dir", required=True, type=Path, help="MOBIMA dataset root")
    ap.add_argument(
        "--sensor",
        action="append",
        choices=SUPPORTED_SENSORS,
        default=None,
        help="LiDAR setup to evaluate. Repeat for multiple sensors. Default: all three.",
    )
    ap.add_argument(
        "--baseline-sensor",
        default="mid360",
        choices=SUPPORTED_SENSORS,
        help="Source/baseline setup A for domain-gap scores and task deltas.",
    )
    ap.add_argument("--ckpt", required=True, type=Path, help="KITTI-trained FrustumPointNet checkpoint")
    ap.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n-task-frames", type=int, default=100, help="Labeled frames for task eval; <=0 = all")
    ap.add_argument("--n-metric-frames", type=int, default=100, help="Common unlabeled frames for metrics; <=0 = all")
    ap.add_argument("--num-points", type=int, default=1024, help="Frustum point count expected by task network")
    ap.add_argument("--num-object-points", type=int, default=512, help="FrustumPointNet object point count")
    ap.add_argument("--box-margin-px", type=float, default=2.0)
    ap.add_argument("--min-depth", type=float, default=1e-4)
    ap.add_argument("--iou-samples", type=int, default=4096)
    ap.add_argument(
        "--task-classes",
        nargs="+",
        default=list(DEFAULT_TASK_CLASSES),
        help="3D GT classes to evaluate. Default matches KITTI training: car van truck.",
    )
    ap.add_argument("--min-pseudo-confidence", type=float, default=0.35)
    ap.add_argument(
        "--min-match-iou",
        type=float,
        default=0.05,
        help=(
            "Min projected 3D GT / pseudo-2D IoU for ambiguous same-class matching. "
            "If a GT class has exactly one compatible pseudo box, class-only fallback is allowed."
        ),
    )
    ap.add_argument(
        "--allow-class-mismatch",
        action="store_true",
        help="Allow 3D/pseudo matching across different classes if projection IoU is best.",
    )

    ap.add_argument("--lidar-sampling-mode", choices=["random", "fps", "none"], default="fps")
    ap.add_argument("--lidar-num-points", type=int, default=2048)
    ap.add_argument("--no-residuals", action="store_true", help="Disable CRT residual extraction")
    ap.add_argument("--residual-edge-threshold", type=float, default=0.2)
    ap.add_argument("--residual-max-points", type=int, default=100)
    ap.add_argument("--residual-min-depth", type=float, default=0.05)
    ap.add_argument(
        "--residual-metric-max-total-points",
        type=int,
        default=10000,
        help="Global residual-row cap per setup used by CRT; <=0 disables cap.",
    )

    ap.add_argument("--no-embedding-cache", action="store_true")
    ap.add_argument("--cache-dir-name", default=".embedding_cache_mobima")
    ap.add_argument("--refresh-embedding-cache", action="store_true")

    ap.add_argument("--out-csv", type=Path, default=Path("mobima_task_and_metrics.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("mobima_task_and_metrics.json"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sensors = list(dict.fromkeys(args.sensor or list(SUPPORTED_SENSORS)))
    if args.baseline_sensor not in sensors:
        raise ValueError(
            f"baseline sensor {args.baseline_sensor!r} must also be present in --sensor {sensors}"
        )

    ctx = MobimaContext(args.root_dir)
    device = resolve_device(args.device)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Dataset root: {ctx.root_dir}")
    print(f"[INFO] Reference frame: {ctx.reference_frame}")
    print(f"[INFO] Sensors: {sensors}; baseline={args.baseline_sensor}")
    for sensor in sensors:
        print(f"[INFO] T_camera_from_{sensor}:\n{ctx.T_camera_from_sensor(sensor)}")

    # Use exactly the same scene/frame subset for all LiDAR setups.
    metric_candidates = ctx.discover_common_sensor_frames(sensors)
    metric_frame_ids = sample_ids(metric_candidates, args.n_metric_frames, args.seed)
    task_candidates = ctx.discover_task_frames(sensors)
    task_frame_ids = sample_ids(task_candidates, args.n_task_frames, args.seed)
    print(
        f"[INFO] Common metric frames: {len(metric_frame_ids)}/{len(metric_candidates)}; "
        f"labeled task frames: {len(task_frame_ids)}/{len(task_candidates)}"
    )

    task_model = load_task_model(args.ckpt, device, args.num_object_points)

    print("[INFO] Initializing metric extractors...")
    rgb_extractor = DINOv2RGBExtractor(device=device)
    lidar_extractor = PointNet2LiDARExtractor(
        num_points=None if args.lidar_sampling_mode == "none" else args.lidar_num_points,
        sampling_mode=args.lidar_sampling_mode,
        device=device,
    )
    joint_extractor = ConcatJointEmbeddingExtractor(l2_normalize_each=False)

    cache = MetricCache(
        args.root_dir / args.cache_dir_name,
        ctx=ctx,
        rgb_extractor=rgb_extractor,
        lidar_extractor=lidar_extractor,
        residual_edge_threshold=args.residual_edge_threshold,
        residual_max_points=args.residual_max_points,
        residual_min_depth=args.residual_min_depth,
        residuals_enabled=not args.no_residuals,
        refresh=args.refresh_embedding_cache,
        enabled=not args.no_embedding_cache,
    )
    if cache.enabled:
        print(f"[INFO] Metric cache: {cache.root}")
        print(f"[INFO] RGB cache key: {cache.rgb_key}; sensor cache key: {cache.sensor_key}")
    else:
        print("[INFO] Metric cache disabled")

    metric_samples = encode_metric_samples_all_sensors(
        ctx=ctx,
        sensors=sensors,
        frame_ids=metric_frame_ids,
        rgb_extractor=rgb_extractor,
        lidar_extractor=lidar_extractor,
        joint_extractor=joint_extractor,
        cache=cache,
        residuals_enabled=not args.no_residuals,
        residual_edge_threshold=args.residual_edge_threshold,
        residual_max_points=args.residual_max_points,
        residual_min_depth=args.residual_min_depth,
        seed=args.seed,
    )
    baseline_samples = metric_samples[args.baseline_sensor]
    if len(baseline_samples) < 2:
        raise RuntimeError("Fewer than 2 valid baseline metric samples were encoded")

    rows: List[Dict[str, object]] = []
    all_object_rows: List[Dict[str, object]] = []

    for i, sensor in enumerate(sensors, start=1):
        print("\n" + "#" * 100)
        print(f"[INFO] Sensor {i}/{len(sensors)}: {sensor}")
        print("#" * 100)
        task_result = evaluate_task_network(
            ctx=ctx,
            sensor=sensor,
            frame_ids=task_frame_ids,
            model=task_model,
            device=device,
            seed=args.seed,
            num_points=args.num_points,
            iou_samples=args.iou_samples,
            box_margin_px=args.box_margin_px,
            min_depth=args.min_depth,
            task_classes=args.task_classes,
            min_pseudo_confidence=args.min_pseudo_confidence,
            min_match_iou=args.min_match_iou,
            allow_class_mismatch=args.allow_class_mismatch,
        )
        task_summary = task_result["summary"]
        all_object_rows.extend(task_result["per_object_rows"])

        gap = compute_domain_gap_metrics(
            baseline_samples,
            metric_samples[sensor],
            residual_metric_max_total_points=args.residual_metric_max_total_points,
            seed=args.seed,
        )
        row: Dict[str, object] = {
            "sensor": sensor,
            "baseline_sensor": args.baseline_sensor,
            "is_baseline": sensor == args.baseline_sensor,
            "root_dir": str(args.root_dir),
            "n_common_metric_frames_requested": len(metric_frame_ids),
            "n_metric_samples_valid": len(metric_samples[sensor]),
        }
        row.update(task_summary)
        row.update(gap)
        rows.append(row)

    add_performance_deltas(rows, args.baseline_sensor)
    print_summary(rows)
    save_csv(rows, args.out_csv)
    save_json(rows, args.out_json)
    save_csv(all_object_rows, args.out_csv.with_name(args.out_csv.stem + "_per_object.csv"))
    save_correlations(rows, args.out_csv, args.baseline_sensor)

    print("\n[DONE] MOBIMA evaluation complete.")
    print(
        "[DONE] Primary coverage-aware MOBIMA comparison: metric score vs "
        "delta_iou3d_approx_mean_all_matched (positive = worse than baseline)."
    )
    print(
        "[DONE] center_error_cam_mean_valid remains available as a conditional "
        "localization diagnostic for objects with valid LiDAR frustums."
    )


if __name__ == "__main__":
    main()
