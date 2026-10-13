"""Image/point-cloud/cache helpers copied from the supplied MOBIMA v2 runner.
No detector, task evaluation, residual extraction or CRT code is included.
"""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
from PIL import Image
from data.data_sample_interface import DataSample
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

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

def as_transform(matrix: Any, name: str) -> torch.Tensor:
    T = torch.tensor(matrix, dtype=torch.float32)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {tuple(T.shape)}")
    return T

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
