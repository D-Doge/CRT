from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from data.data_sample_interface import DataSample
from data.dataset.datasetLoaderInterface import DatasetLoader


# ============================================================
# Small helpers
# ============================================================


def _load_rgb_image(path: Path) -> torch.Tensor:
    """
    Load RGB image as float32 tensor in CHW layout and range [0, 1].
    """
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()  # H, W, 3
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t



def _parse_pcd_ascii(path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Minimal ASCII .pcd parser.

    Returns
    -------
    lidar_xyz : (N, 3) float32 tensor
    lidar_feat : (N, F) float32 tensor or None
    """
    if not path.is_file():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PCD header: {path}")
            header_lines.append(line.strip())
            if line.strip().lower().startswith("data"):
                break

        def _get_token(prefix: str, default: Optional[str] = None) -> Optional[str]:
            prefix = prefix.lower()
            for h in header_lines:
                if h.lower().startswith(prefix):
                    parts = h.split(None, 1)
                    return parts[1] if len(parts) > 1 else ""
            return default

        data_type = (_get_token("data") or "").strip().lower()
        if data_type != "ascii":
            raise ValueError(
                f"Only ASCII PCD is currently supported, but got DATA {data_type!r} in {path}"
            )

        fields = ((_get_token("fields") or _get_token("field")) or "").split()
        if not fields:
            raise ValueError(f"PCD file has no FIELDS entry: {path}")

        data_txt = f.read().strip()
        if not data_txt:
            return torch.zeros((0, 3), dtype=torch.float32), None

        vals = np.fromstring(data_txt, sep=" ", dtype=np.float32)
        num_fields = len(fields)

        if vals.size % num_fields != 0:
            rows = [
                list(map(float, ln.split()))
                for ln in data_txt.splitlines()
                if ln.strip()
            ]
            arr = np.asarray(rows, dtype=np.float32)
        else:
            arr = vals.reshape(-1, num_fields)

    col_map = {name: i for i, name in enumerate(fields)}

    needed = ["x", "y", "z"]
    if any(k not in col_map for k in needed):
        raise ValueError(f"PCD file lacks x/y/z fields. Found fields={fields}")

    xyz = arr[:, [col_map["x"], col_map["y"], col_map["z"]]]
    xyz_t = torch.from_numpy(xyz).float()

    extra_cols = [i for name, i in col_map.items() if name not in {"x", "y", "z"}]
    if not extra_cols:
        return xyz_t, None

    feat = arr[:, extra_cols]
    feat_t = torch.from_numpy(feat).float()
    return xyz_t, feat_t



def _extract_step_name(frame_json_path: Path) -> str:
    name = frame_json_path.name
    suffix = ".frame_data.json"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected frame json name: {frame_json_path.name}")
    return name[: -len(suffix)]



def _find_capture(frame_data: Dict[str, Any], capture_id: str) -> Dict[str, Any]:
    captures = frame_data.get("captures", [])
    for c in captures:
        if c.get("id") == capture_id:
            return c
    raise KeyError(f"Could not find capture with id={capture_id!r} in frame_data.json")



def _find_annotation(
    capture: Dict[str, Any],
    *,
    ann_type: Optional[str] = None,
    ann_id: Optional[str] = None,
    transform_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    for ann in capture.get("annotations", []):
        if ann_type is not None and ann.get("@type") != ann_type:
            continue
        if ann_id is not None and ann.get("id") != ann_id:
            continue
        if transform_name is not None and ann.get("transform_name") != transform_name:
            continue
        return ann
    return None



def _find_pointcloud_annotation(capture: Dict[str, Any], pointcloud_sensor: str) -> Optional[Dict[str, Any]]:
    """
    Match by annotation id, @type, or filename suffix/content.
    This keeps the loader robust if the sensor name is passed as e.g.
    'FOV-Grid_PointCloud' or 'MID70_PointCloud'.
    """
    sensor_l = pointcloud_sensor.lower()
    for ann in capture.get("annotations", []):
        ann_id = str(ann.get("id", "")).lower()
        ann_type = str(ann.get("@type", "")).lower()
        filename = str(ann.get("filename", "")).lower()
        if (
            ann_id == sensor_l
            or ann_id.startswith(sensor_l + "_")
            or ann_type == sensor_l
            or ann_type.startswith(sensor_l + "_")
            or f".{sensor_l}.pcd" in filename
            or f".{sensor_l}_" in filename
            or filename.endswith(f"{sensor_l}.pcd")
        ):
            return ann
    return None



def _infer_image_path(sequence_dir: Path, capture: Dict[str, Any], fallback: Path) -> Path:
    filename = capture.get("filename")
    if filename:
        return sequence_dir / filename
    return fallback



def _extract_intrinsics_from_annotation(capture: Dict[str, Any]) -> Optional[torch.Tensor]:
    ann = _find_annotation(capture, ann_type="CameraIntrinsics")
    if ann is None:
        return None

    keys = [
        "k00", "k01", "k02",
        "k10", "k11", "k12",
        "k20", "k21", "k22",
    ]
    if not all(k in ann for k in keys):
        return None

    K = torch.tensor(
        [
            [float(ann["k00"]), float(ann["k01"]), float(ann["k02"])],
            [float(ann["k10"]), float(ann["k11"]), float(ann["k12"])],
            [float(ann["k20"]), float(ann["k21"]), float(ann["k22"])],
        ],
        dtype=torch.float32,
    )
    return K



def _extract_intrinsics_from_unity_projection(
    capture: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Optional[torch.Tensor]:
    """
    Fallback if no CameraIntrinsics annotation exists.
    Converts Unity Perception projection-style 3x3 matrix into an OpenCV-style K.
    """
    mat = capture.get("matrix")
    projection = capture.get("projection")

    if mat is None:
        return None
    if projection is not None and str(projection).lower() != "perspective":
        return None

    t = torch.tensor(mat, dtype=torch.float32)
    if t.numel() != 9:
        return None

    P = t.view(3, 3)
    h, w = image_size
    p00 = float(P[0, 0])
    p11 = float(P[1, 1])

    fx = p00 * (w / 2.0)
    fy = p11 * (h / 2.0)
    cx = w / 2.0
    cy = h / 2.0

    return torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )



def _extract_relative_transform(
    lidar_capture: Dict[str, Any],
    *,
    transform_name: str = "lidar_to_cam",
) -> Optional[torch.Tensor]:
    ann = _find_annotation(
        lidar_capture,
        ann_type="RelativeTransform",
        transform_name=transform_name,
    )
    if ann is None:
        return None

    keys = [
        "m00", "m01", "m02", "m03",
        "m10", "m11", "m12", "m13",
        "m20", "m21", "m22", "m23",
        "m30", "m31", "m32", "m33",
    ]
    if not all(k in ann for k in keys):
        return None

    T = torch.tensor(
        [
            [float(ann["m00"]), float(ann["m01"]), float(ann["m02"]), float(ann["m03"])],
            [float(ann["m10"]), float(ann["m11"]), float(ann["m12"]), float(ann["m13"])],
            [float(ann["m20"]), float(ann["m21"]), float(ann["m22"]), float(ann["m23"])],
            [float(ann["m30"]), float(ann["m31"]), float(ann["m32"]), float(ann["m33"])],
        ],
        dtype=torch.float32,
    )
    return T



def _extract_image_size(rgb: torch.Tensor) -> Tuple[int, int]:
    _, h, w = rgb.shape
    return h, w



def _make_identity_extrinsic_if_missing(use_identity: bool) -> Optional[torch.Tensor]:
    if not use_identity:
        return None
    return torch.eye(4, dtype=torch.float32)


# ============================================================
# Index entry
# ============================================================


@dataclass(frozen=True)
class _UnityIndexEntry:
    frame_id: str
    sequence_id: str
    sequence_dir: Path
    frame_json_path: Path
    image_path: Path
    pcd_path: Path
    timestamp_s: Optional[float]


# ============================================================
# DatasetLoader implementation
# ============================================================


class UnitySimDatasetLoader(DatasetLoader):
    """
    DatasetLoader for Unity simulated RGB + point cloud data.

    Updated behavior:
    - reads camera intrinsics from the CameraIntrinsics annotation when present
    - reads T_cam_lidar from the LiDAR capture's RelativeTransform annotation
    - supports separate ids for the RGB capture and the LiDAR capture
    - resolves the point cloud file through the LiDAR annotation filename when present
    """

    def __init__(
        self,
        root_dir: str | Path,
        pointcloud_sensor: str,
        *,
        dataset_id: str = "unity_sim",
        setup_id: Optional[str] = None,
        rgb_capture_id: str = "rgb_camera",
        lidar_capture_id: str = "lidar",
        image_extension_fallback: str = ".png",
        relative_transform_name: str = "lidar_to_cam",
        use_identity_extrinsic_if_missing: bool = False,
        validate_paths_on_init: bool = True,
    ) -> None:
        self._root_dir = Path(root_dir)
        self._pointcloud_sensor = pointcloud_sensor
        self._dataset_id_value = dataset_id
        self._setup_id = setup_id or pointcloud_sensor
        self._rgb_capture_id = rgb_capture_id
        self._lidar_capture_id = lidar_capture_id
        self._image_extension_fallback = image_extension_fallback
        self._relative_transform_name = relative_transform_name
        self._use_identity_extrinsic_if_missing = use_identity_extrinsic_if_missing

        if not self._root_dir.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self._root_dir}")

        self._index: List[_UnityIndexEntry] = self._build_index(
            validate_paths=validate_paths_on_init
        )
        self._frame_id_to_entry: Dict[str, _UnityIndexEntry] = {
            entry.frame_id: entry for entry in self._index
        }

    @property
    def dataset_id(self) -> str:
        return self._dataset_id_value

    @property
    def pointcloud_sensor(self) -> str:
        return self._pointcloud_sensor

    def __len__(self) -> int:
        return len(self._index)

    def get_frame_ids(self) -> List[str]:
        return [entry.frame_id for entry in self._index]

    def load_sample(self, frame_id: str) -> DataSample:
        entry = self._frame_id_to_entry.get(frame_id)
        if entry is None:
            raise KeyError(f"Unknown frame_id: {frame_id}")

        with open(entry.frame_json_path, "r", encoding="utf-8") as f:
            frame_data = json.load(f)

        rgb_capture = _find_capture(frame_data, capture_id=self._rgb_capture_id)
        lidar_capture = _find_capture(frame_data, capture_id=self._lidar_capture_id)

        rgb = _load_rgb_image(entry.image_path)
        lidar_xyz, lidar_feat = _parse_pcd_ascii(entry.pcd_path)
        image_size = _extract_image_size(rgb)

        intrinsics = _extract_intrinsics_from_annotation(rgb_capture)
        if intrinsics is None:
            intrinsics = _extract_intrinsics_from_unity_projection(rgb_capture, image_size)

        T_cam_lidar = _extract_relative_transform(
            lidar_capture,
            transform_name=self._relative_transform_name,
        )
        if T_cam_lidar is None:
            T_cam_lidar = _make_identity_extrinsic_if_missing(
                self._use_identity_extrinsic_if_missing
            )

        sample = DataSample(
            setup_id=self._setup_id,
            frame_id=entry.frame_id,
            timestamp_s=entry.timestamp_s,
            sequence_id=entry.sequence_id,
            rgb=rgb,
            lidar_xyz=lidar_xyz,
            lidar_feat=lidar_feat,
            intrinsics=intrinsics,
            T_cam_lidar=T_cam_lidar,
            image_size=image_size,
            meta={
                "dataset_id": self.dataset_id,
                "root_dir": str(self._root_dir),
                "sequence_dir": str(entry.sequence_dir),
                "frame_json_path": str(entry.frame_json_path),
                "image_path": str(entry.image_path),
                "pcd_path": str(entry.pcd_path),
                "pointcloud_sensor": self._pointcloud_sensor,
                "rgb_capture_id": self._rgb_capture_id,
                "lidar_capture_id": self._lidar_capture_id,
                "relative_transform_name": self._relative_transform_name,
                "unity_frame": frame_data.get("frame"),
                "unity_sequence": frame_data.get("sequence"),
                "unity_step": frame_data.get("step"),
                "rgb_capture_position": rgb_capture.get("position"),
                "rgb_capture_rotation": rgb_capture.get("rotation"),
                "lidar_capture_position": lidar_capture.get("position"),
                "lidar_capture_rotation": lidar_capture.get("rotation"),
                "intrinsics_source": (
                    "CameraIntrinsics" if _extract_intrinsics_from_annotation(rgb_capture) is not None else "projection_fallback"
                ),
                "extrinsic_source": (
                    "RelativeTransform"
                    if _extract_relative_transform(lidar_capture, transform_name=self._relative_transform_name) is not None
                    else ("identity_fallback" if self._use_identity_extrinsic_if_missing else None)
                ),
            },
        )
        return sample

    # --------------------------------------------------------
    # Internal indexing
    # --------------------------------------------------------

    def _build_index(self, *, validate_paths: bool) -> List[_UnityIndexEntry]:
        seq_dirs = sorted(p for p in self._root_dir.glob("sequence.*") if p.is_dir())
        if not seq_dirs:
            raise RuntimeError(f"No sequence.* folders found under {self._root_dir}")

        entries: List[_UnityIndexEntry] = []

        for seq_dir in seq_dirs:
            frame_json_paths = sorted(seq_dir.glob("step*.frame_data.json"))

            for frame_json_path in frame_json_paths:
                with open(frame_json_path, "r", encoding="utf-8") as f:
                    frame_data = json.load(f)

                rgb_capture = _find_capture(frame_data, capture_id=self._rgb_capture_id)
                lidar_capture = _find_capture(frame_data, capture_id=self._lidar_capture_id)
                step_name = _extract_step_name(frame_json_path)

                default_image_path = seq_dir / f"{step_name}.{self._rgb_capture_id}{self._image_extension_fallback}"
                image_path = _infer_image_path(seq_dir, rgb_capture, default_image_path)
                if not image_path.suffix:
                    image_path = image_path.with_suffix(self._image_extension_fallback)

                pcd_ann = _find_pointcloud_annotation(lidar_capture, self._pointcloud_sensor)
                if pcd_ann is not None and pcd_ann.get("filename"):
                    pcd_path = seq_dir / str(pcd_ann["filename"])
                else:
                    pcd_name = f"{step_name}.{self._lidar_capture_id}.{self._pointcloud_sensor}.pcd"
                    pcd_path = seq_dir / pcd_name

                sequence_id = seq_dir.name
                frame_id = f"{sequence_id}/{step_name}"

                timestamp_s = frame_data.get("timestamp")
                if timestamp_s is not None:
                    try:
                        timestamp_s = float(timestamp_s)
                    except Exception:
                        timestamp_s = None

                if validate_paths:
                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"Expected image file not found for frame_id={frame_id}: {image_path}"
                        )
                    if not pcd_path.is_file():
                        raise FileNotFoundError(
                            f"Expected point cloud file not found for frame_id={frame_id}: {pcd_path}"
                        )

                entries.append(
                    _UnityIndexEntry(
                        frame_id=frame_id,
                        sequence_id=sequence_id,
                        sequence_dir=seq_dir,
                        frame_json_path=frame_json_path,
                        image_path=image_path,
                        pcd_path=pcd_path,
                        timestamp_s=timestamp_s,
                    )
                )

        if not entries:
            raise RuntimeError(f"No frames found under {self._root_dir}")

        return entries
