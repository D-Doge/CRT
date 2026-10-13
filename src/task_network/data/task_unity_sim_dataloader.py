from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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

    return torch.tensor(
        [
            [float(ann["k00"]), float(ann["k01"]), float(ann["k02"])],
            [float(ann["k10"]), float(ann["k11"]), float(ann["k12"])],
            [float(ann["k20"]), float(ann["k21"]), float(ann["k22"])],
        ],
        dtype=torch.float32,
    )


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

    return torch.tensor(
        [
            [float(ann["m00"]), float(ann["m01"]), float(ann["m02"]), float(ann["m03"])],
            [float(ann["m10"]), float(ann["m11"]), float(ann["m12"]), float(ann["m13"])],
            [float(ann["m20"]), float(ann["m21"]), float(ann["m22"]), float(ann["m23"])],
            [float(ann["m30"]), float(ann["m31"]), float(ann["m32"]), float(ann["m33"])],
        ],
        dtype=torch.float32,
    )


def _make_identity_extrinsic_if_missing(use_identity: bool) -> Optional[torch.Tensor]:
    if not use_identity:
        return None
    return torch.eye(4, dtype=torch.float32)


def _extract_image_size(rgb: torch.Tensor) -> Tuple[int, int]:
    _, h, w = rgb.shape
    return int(h), int(w)


def _empty_2d_boxes() -> torch.Tensor:
    return torch.zeros((0, 4), dtype=torch.float32)


def _empty_3d_centers() -> torch.Tensor:
    return torch.zeros((0, 3), dtype=torch.float32)


def _empty_3d_sizes() -> torch.Tensor:
    return torch.zeros((0, 3), dtype=torch.float32)


def _empty_quats() -> torch.Tensor:
    return torch.zeros((0, 4), dtype=torch.float32)


def _extract_2d_boxes(capture: Dict[str, Any]) -> List[Dict[str, Any]]:
    ann = _find_annotation(
        capture,
        ann_type="type.unity.com/unity.solo.BoundingBox2DAnnotation",
    )
    if ann is None:
        return []
    return list(ann.get("values", []))


def _extract_3d_boxes(capture: Dict[str, Any]) -> List[Dict[str, Any]]:
    ann = _find_annotation(
        capture,
        ann_type="type.unity.com/unity.solo.BoundingBox3DAnnotation",
    )
    if ann is None:
        return []
    return list(ann.get("values", []))


def _merge_2d_3d_annotations(
    ann2d: List[Dict[str, Any]],
    ann3d: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge 2D and 3D annotations by instanceId.

    Keeps all visible instances that appear in either annotation set.
    """
    merged: Dict[int, Dict[str, Any]] = {}

    for a in ann2d:
        instance_id = int(a["instanceId"])
        merged.setdefault(instance_id, {})
        merged[instance_id]["ann2d"] = a

    for a in ann3d:
        instance_id = int(a["instanceId"])
        merged.setdefault(instance_id, {})
        merged[instance_id]["ann3d"] = a

    out = []
    for instance_id, payload in merged.items():
        ann_2d = payload.get("ann2d")
        ann_3d = payload.get("ann3d")

        label_id = None
        label_name = None
        if ann_2d is not None:
            label_id = int(ann_2d["labelId"])
            label_name = str(ann_2d["labelName"])
        if ann_3d is not None:
            label_id = int(ann_3d["labelId"]) if label_id is None else label_id
            label_name = str(ann_3d["labelName"]) if label_name is None else label_name

        entry = {
            "instance_id": instance_id,
            "label_id": -1 if label_id is None else int(label_id),
            "label_name": "" if label_name is None else str(label_name),
            "ann2d": ann_2d,
            "ann3d": ann_3d,
        }
        out.append(entry)

    out.sort(key=lambda x: x["instance_id"])
    return out


def _build_target_from_annotations(
    merged_ann: List[Dict[str, Any]],
    intrinsics: Optional[torch.Tensor],
    T_cam_lidar: Optional[torch.Tensor],
    image_size: Tuple[int, int],
    frame_id: str,
    sequence_id: str,
) -> Dict[str, Any]:
    boxes_2d = []
    labels = []
    label_names = []
    instance_ids = []

    centers_3d = []
    sizes_3d = []
    quats_3d = []

    has_3d_mask = []

    for obj in merged_ann:
        instance_ids.append(obj["instance_id"])
        labels.append(obj["label_id"])
        label_names.append(obj["label_name"])

        ann2d = obj["ann2d"]
        if ann2d is not None:
            ox, oy = ann2d["origin"]
            w, h = ann2d["dimension"]
            x1 = float(ox)
            y1 = float(oy)
            x2 = float(ox) + float(w)
            y2 = float(oy) + float(h)
            boxes_2d.append([x1, y1, x2, y2])
        else:
            boxes_2d.append([0.0, 0.0, 0.0, 0.0])

        ann3d = obj["ann3d"]
        if ann3d is not None:
            centers_3d.append([float(v) for v in ann3d["translation"]])
            sizes_3d.append([float(v) for v in ann3d["size"]])
            quats_3d.append([float(v) for v in ann3d["rotation"]])
            has_3d_mask.append(True)
        else:
            centers_3d.append([0.0, 0.0, 0.0])
            sizes_3d.append([0.0, 0.0, 0.0])
            quats_3d.append([0.0, 0.0, 0.0, 1.0])
            has_3d_mask.append(False)

    target = {
        "boxes_2d_xyxy": torch.tensor(boxes_2d, dtype=torch.float32) if boxes_2d else _empty_2d_boxes(),
        "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
        "label_names": label_names,
        "instance_ids": torch.tensor(instance_ids, dtype=torch.long) if instance_ids else torch.zeros((0,), dtype=torch.long),
        "boxes_3d_center": torch.tensor(centers_3d, dtype=torch.float32) if centers_3d else _empty_3d_centers(),
        "boxes_3d_size": torch.tensor(sizes_3d, dtype=torch.float32) if sizes_3d else _empty_3d_sizes(),
        "boxes_3d_quat": torch.tensor(quats_3d, dtype=torch.float32) if quats_3d else _empty_quats(),
        "has_3d_box": torch.tensor(has_3d_mask, dtype=torch.bool) if has_3d_mask else torch.zeros((0,), dtype=torch.bool),
        "intrinsics": intrinsics,
        "T_cam_lidar": T_cam_lidar,
        "image_size": torch.tensor(image_size, dtype=torch.long),
        "frame_id": frame_id,
        "sequence_id": sequence_id,
    }
    return target

def extract_camera_pose(capture: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
    """
    Extract camera position and rotation (world space) from a capture entry.
    Returns dict with 'position' (3,) and 'rotation' (4,) XYZW, or None.
    """
    pos = capture.get("position")
    rot = capture.get("rotation")
    if pos is None or rot is None:
        return None
    return {
        "position": torch.tensor(pos, dtype=torch.float32),   # (3,)
        "rotation": torch.tensor(rot, dtype=torch.float32),   # (4,) XYZW
    }

def _extract_unity_projection_matrix(capture: Dict[str, Any]) -> Optional[torch.Tensor]:
    """
    Extract Unity Perception projection-style 3x3 matrix from the RGB capture.

    This is NOT the OpenCV K matrix.
    It is the raw capture["matrix"] used by Unity/datasetinsights-style projection.
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

    return t.view(3, 3)
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
# PyTorch detection dataset
# ============================================================

class UnitySimDetectionDataset(Dataset):
    """
    PyTorch Dataset for Unity simulation data with RGB, point cloud,
    camera calibration, and 2D/3D box annotations.

    Returns from __getitem__:
        image:  torch.Tensor, shape (3, H, W), float32 in [0,1]
        point_cloud: torch.Tensor, shape (N, 3)
        target: dict

    Notes
    -----
    - 2D boxes are returned as XYXY pixel boxes.
    - 3D box translation/size/rotation are returned exactly as stored in the JSON.
    - This assumes the 3D annotation attached to the RGB capture is expressed in the camera frame. Rotation order XYZW
    """

    def __init__(
        self,
        root_dir: str | Path,
        pointcloud_sensor: str,
        *,
        rgb_capture_id: str = "rgb_camera",
        lidar_capture_id: str = "lidar",
        image_extension_fallback: str = ".png",
        relative_transform_name: str = "lidar_to_cam",
        use_identity_extrinsic_if_missing: bool = False,
        validate_paths_on_init: bool = True,
        image_transform=None,
        pointcloud_transform=None,
        return_lidar_feat: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.pointcloud_sensor = pointcloud_sensor
        self.rgb_capture_id = rgb_capture_id
        self.lidar_capture_id = lidar_capture_id
        self.image_extension_fallback = image_extension_fallback
        self.relative_transform_name = relative_transform_name
        self.use_identity_extrinsic_if_missing = use_identity_extrinsic_if_missing
        self.image_transform = image_transform
        self.pointcloud_transform = pointcloud_transform
        self.return_lidar_feat = bool(return_lidar_feat)

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        self.index = self._build_index(validate_paths=validate_paths_on_init)
        self.frame_ids = [entry.frame_id for entry in self.index]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        entry = self.index[idx]

        with open(entry.frame_json_path, "r", encoding="utf-8") as f:
            frame_data = json.load(f)

        rgb_capture = _find_capture(frame_data, capture_id=self.rgb_capture_id)
        lidar_capture = _find_capture(frame_data, capture_id=self.lidar_capture_id)

        

        image = _load_rgb_image(entry.image_path)
        lidar_xyz, lidar_feat = _parse_pcd_ascii(entry.pcd_path)
        image_size = _extract_image_size(image)

        intrinsics = _extract_intrinsics_from_annotation(rgb_capture)
        if intrinsics is None:
            intrinsics = _extract_intrinsics_from_unity_projection(rgb_capture, image_size)

        unity_projection_matrix = _extract_unity_projection_matrix(rgb_capture)

        T_cam_lidar = _extract_relative_transform(
            lidar_capture,
            transform_name=self.relative_transform_name,
        )
        if T_cam_lidar is None:
            T_cam_lidar = _make_identity_extrinsic_if_missing(
                self.use_identity_extrinsic_if_missing
            )

        ann2d = _extract_2d_boxes(rgb_capture)
        ann3d = _extract_3d_boxes(rgb_capture)
        merged_ann = _merge_2d_3d_annotations(ann2d, ann3d)

        target = _build_target_from_annotations(
            merged_ann=merged_ann,
            intrinsics=intrinsics,
            T_cam_lidar=T_cam_lidar,
            image_size=image_size,
            frame_id=entry.frame_id,
            sequence_id=entry.sequence_id,
        )

        target["timestamp_s"] = entry.timestamp_s
        target["meta"] = {
            "frame_json_path": str(entry.frame_json_path),
            "image_path": str(entry.image_path),
            "pcd_path": str(entry.pcd_path),
            "pointcloud_sensor": self.pointcloud_sensor,
            "rgb_capture_id": self.rgb_capture_id,
            "lidar_capture_id": self.lidar_capture_id,
        }

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.pointcloud_transform is not None:
            lidar_xyz = self.pointcloud_transform(lidar_xyz)

        if self.return_lidar_feat:
            target["lidar_feat"] = lidar_feat

        cam_pose = extract_camera_pose(rgb_capture)
        if cam_pose is not None:
            target["camera_position_world"] = cam_pose["position"]
            target["camera_rotation_world"] = cam_pose["rotation"]
        
        target["unity_projection_matrix"] = unity_projection_matrix

        return image, lidar_xyz, target

    def get_frame_id(self, idx: int) -> str:
        return self.frame_ids[idx]

    def _build_index(self, *, validate_paths: bool) -> List[_UnityIndexEntry]:
        seq_dirs = sorted(p for p in self.root_dir.glob("sequence.*") if p.is_dir())
        if not seq_dirs:
            raise RuntimeError(f"No sequence.* folders found under {self.root_dir}")

        entries: List[_UnityIndexEntry] = []

        for seq_dir in seq_dirs:
            frame_json_paths = sorted(seq_dir.glob("step*.frame_data.json"))

            for frame_json_path in frame_json_paths:
                with open(frame_json_path, "r", encoding="utf-8") as f:
                    frame_data = json.load(f)

                rgb_capture = _find_capture(frame_data, capture_id=self.rgb_capture_id)
                lidar_capture = _find_capture(frame_data, capture_id=self.lidar_capture_id)
                step_name = _extract_step_name(frame_json_path)

                default_image_path = seq_dir / f"{step_name}.{self.rgb_capture_id}{self.image_extension_fallback}"
                image_path = _infer_image_path(seq_dir, rgb_capture, default_image_path)
                if not image_path.suffix:
                    image_path = image_path.with_suffix(self.image_extension_fallback)

                pcd_ann = _find_pointcloud_annotation(lidar_capture, self.pointcloud_sensor)
                if pcd_ann is not None and pcd_ann.get("filename"):
                    pcd_path = seq_dir / str(pcd_ann["filename"])
                else:
                    pcd_name = f"{step_name}.{self.lidar_capture_id}.{self.pointcloud_sensor}.pcd"
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
            raise RuntimeError(f"No frames found under {self.root_dir}")

        return entries


# ============================================================
# Custom collate fn
# ============================================================

def unity_sim_detection_collate_fn(batch):
    """
    Keeps variable-size point clouds and variable number of boxes as lists.

    Returns:
        images: list[Tensor]
        point_clouds: list[Tensor]
        targets: list[dict]
    """
    images, point_clouds, targets = zip(*batch)
    return list(images), list(point_clouds), list(targets)