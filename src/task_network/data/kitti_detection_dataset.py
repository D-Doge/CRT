from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_VEHICLE_CLASSES: Tuple[str, ...] = ("Car", "Van", "Truck")


@dataclass(frozen=True)
class KittiObjectLabel:
    class_name: str
    truncated: float
    occluded: int
    alpha: float
    box_2d_xyxy: Tuple[float, float, float, float]
    dimensions_hwl: Tuple[float, float, float]
    location_bottom_center_rect: Tuple[float, float, float]
    rotation_y: float


def _load_rgb_image(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"KITTI image not found: {path}")
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def _read_image_size(path: Path) -> Tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"KITTI image not found: {path}")
    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def _load_velodyne_bin(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"KITTI Velodyne file not found: {path}")
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Invalid KITTI .bin length in {path}: {raw.size} floats")
    points = raw.reshape(-1, 4)
    xyz = torch.from_numpy(points[:, :3].copy()).float()
    reflectance = torch.from_numpy(points[:, 3:4].copy()).float()
    return xyz, reflectance


def _parse_calibration_file(path: Path) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"KITTI calibration file not found: {path}")

    values: Dict[str, torch.Tensor] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, payload = line.split(":", 1)
            nums = [float(v) for v in payload.split()]
            values[key] = torch.tensor(nums, dtype=torch.float32)

    def first(*keys: str) -> torch.Tensor:
        for key in keys:
            if key in values:
                return values[key]
        raise KeyError(f"None of calibration keys {keys} found in {path}")

    P2 = first("P2").reshape(3, 4)
    R0_rect = first("R0_rect", "R_rect", "R_rect_00").reshape(3, 3)
    Tr_velo_to_cam = first(
        "Tr_velo_to_cam",
        "Tr_velo_cam",
        "Tr_velo_to_cam_0",
    ).reshape(3, 4)

    T_cam_lidar = torch.eye(4, dtype=torch.float32)
    T_cam_lidar[:3, :4] = Tr_velo_to_cam

    R0_h = torch.eye(4, dtype=torch.float32)
    R0_h[:3, :3] = R0_rect

    T_rect_lidar = R0_h @ T_cam_lidar
    T_lidar_rect = torch.linalg.inv(T_rect_lidar)
    P_lidar_to_image = P2 @ T_rect_lidar

    return {
        "P2": P2,
        "R0_rect": R0_rect,
        "Tr_velo_to_cam": Tr_velo_to_cam,
        "T_rect_lidar": T_rect_lidar,
        "T_lidar_rect": T_lidar_rect,
        "P_lidar_to_image": P_lidar_to_image,
    }


def _parse_label_file(path: Path) -> List[KittiObjectLabel]:
    if not path.is_file():
        raise FileNotFoundError(f"KITTI label file not found: {path}")

    labels: List[KittiObjectLabel] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 15:
                raise ValueError(
                    f"Invalid KITTI label at {path}:{line_no}; expected >=15 fields, "
                    f"got {len(fields)}"
                )

            labels.append(
                KittiObjectLabel(
                    class_name=fields[0],
                    truncated=float(fields[1]),
                    occluded=int(float(fields[2])),
                    alpha=float(fields[3]),
                    box_2d_xyxy=tuple(float(v) for v in fields[4:8]),
                    dimensions_hwl=tuple(float(v) for v in fields[8:11]),
                    location_bottom_center_rect=tuple(float(v) for v in fields[11:14]),
                    rotation_y=float(fields[14]),
                )
            )
    return labels


def _rotation_y_camera(rotation_y: float) -> torch.Tensor:
    c = float(np.cos(rotation_y))
    s = float(np.sin(rotation_y))
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=torch.float32,
    )


def _rotmat_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    """Convert a proper rotation matrix to a normalized XYZW quaternion."""
    R = R.detach().float()
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape (3,3), got {tuple(R.shape)}")

    # Remove small numerical non-orthogonality introduced by file precision.
    U, _, Vh = torch.linalg.svd(R)
    R = U @ Vh
    if torch.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vh

    trace = float(torch.trace(R).item())
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = float((R[2, 1] - R[1, 2]).item()) / s
        qy = float((R[0, 2] - R[2, 0]).item()) / s
        qz = float((R[1, 0] - R[0, 1]).item()) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1.0 + float(R[0, 0] - R[1, 1] - R[2, 2])) ** 0.5 * 2.0
        qw = float((R[2, 1] - R[1, 2]).item()) / s
        qx = 0.25 * s
        qy = float((R[0, 1] + R[1, 0]).item()) / s
        qz = float((R[0, 2] + R[2, 0]).item()) / s
    elif R[1, 1] > R[2, 2]:
        s = (1.0 + float(R[1, 1] - R[0, 0] - R[2, 2])) ** 0.5 * 2.0
        qw = float((R[0, 2] - R[2, 0]).item()) / s
        qx = float((R[0, 1] + R[1, 0]).item()) / s
        qy = 0.25 * s
        qz = float((R[1, 2] + R[2, 1]).item()) / s
    else:
        s = (1.0 + float(R[2, 2] - R[0, 0] - R[1, 1])) ** 0.5 * 2.0
        qw = float((R[1, 0] - R[0, 1]).item()) / s
        qx = float((R[0, 2] + R[2, 0]).item()) / s
        qy = float((R[1, 2] + R[2, 1]).item()) / s
        qz = 0.25 * s

    q = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
    return q / torch.linalg.norm(q).clamp_min(1e-12)


def _empty(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype)


class KittiDetectionDataset(Dataset):
    """
    KITTI object-detection loader normalized to the project's training contract.

    Input layout (either is accepted):
        ROOT/training/{image_2,velodyne,calib,label_2}
    or:
        ROOT/{image_2,velodyne,calib,label_2}

    All 3D box targets are converted to the Velodyne/LiDAR frame:
        center: geometric box center in LiDAR XYZ
        size:   local [length, height, width]
        quat:   local-box -> LiDAR rotation, XYZW
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        frame_ids: Optional[Sequence[str]] = None,
        split_file: Optional[str | Path] = None,
        class_whitelist: Sequence[str] = DEFAULT_VEHICLE_CLASSES,
        return_lidar_reflectance: bool = False,
        validate_paths_on_init: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.data_dir = self._resolve_data_dir(self.root_dir)
        self.image_dir = self.data_dir / "image_2"
        self.velodyne_dir = self.data_dir / "velodyne"
        self.calib_dir = self.data_dir / "calib"
        self.label_dir = self.data_dir / "label_2"
        self.class_whitelist = tuple(class_whitelist)
        self.return_lidar_reflectance = bool(return_lidar_reflectance)

        if frame_ids is not None and split_file is not None:
            raise ValueError("Provide either frame_ids or split_file, not both")

        if split_file is not None:
            self.frame_ids = self.read_split_file(split_file)
        elif frame_ids is not None:
            self.frame_ids = [self.normalize_frame_id(x) for x in frame_ids]
        else:
            self.frame_ids = self.discover_frame_ids(self.root_dir)

        if not self.frame_ids:
            raise RuntimeError("No KITTI frame IDs were found")

        if validate_paths_on_init:
            self._validate_paths()

    @staticmethod
    def _resolve_data_dir(root_dir: Path) -> Path:
        training = root_dir / "training"
        if training.is_dir():
            return training
        return root_dir

    @staticmethod
    def normalize_frame_id(frame_id: str | int) -> str:
        value = str(frame_id).strip()
        if not value:
            raise ValueError("Empty KITTI frame ID")
        if value.isdigit():
            return f"{int(value):06d}"
        return value

    @classmethod
    def discover_frame_ids(cls, root_dir: str | Path) -> List[str]:
        data_dir = cls._resolve_data_dir(Path(root_dir))
        label_dir = data_dir / "label_2"
        if not label_dir.is_dir():
            raise FileNotFoundError(f"KITTI label directory not found: {label_dir}")
        return sorted(path.stem for path in label_dir.glob("*.txt"))

    @classmethod
    def read_split_file(cls, split_file: str | Path) -> List[str]:
        path = Path(split_file)
        if not path.is_file():
            raise FileNotFoundError(f"Split file not found: {path}")
        ids = [
            cls.normalize_frame_id(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Split file contains duplicate IDs: {path}")
        return ids

    def _paths(self, frame_id: str) -> Dict[str, Path]:
        return {
            "image": self.image_dir / f"{frame_id}.png",
            "velodyne": self.velodyne_dir / f"{frame_id}.bin",
            "calib": self.calib_dir / f"{frame_id}.txt",
            "label": self.label_dir / f"{frame_id}.txt",
        }

    def _validate_paths(self) -> None:
        for frame_id in self.frame_ids:
            for kind, path in self._paths(frame_id).items():
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing KITTI {kind} file for frame {frame_id}: {path}"
                    )

    def __len__(self) -> int:
        return len(self.frame_ids)

    def get_frame_id(self, idx: int) -> str:
        return self.frame_ids[idx]

    def _build_target(self, frame_id: str) -> Dict[str, object]:
        paths = self._paths(frame_id)
        calib = _parse_calibration_file(paths["calib"])
        labels = _parse_label_file(paths["label"])
        image_size = _read_image_size(paths["image"])

        boxes_2d: List[List[float]] = []
        centers_lidar: List[torch.Tensor] = []
        sizes_local_xyz: List[List[float]] = []
        quats_lidar: List[torch.Tensor] = []
        class_names: List[str] = []
        truncated: List[float] = []
        occluded: List[int] = []
        alpha: List[float] = []
        rotation_y: List[float] = []

        T_lidar_rect = calib["T_lidar_rect"]
        R_lidar_rect = T_lidar_rect[:3, :3]

        for label in labels:
            if label.class_name not in self.class_whitelist:
                continue

            x1, y1, x2, y2 = label.box_2d_xyxy
            h, w, l = label.dimensions_hwl
            if x2 <= x1 or y2 <= y1 or min(h, w, l) <= 0:
                continue

            # KITTI label location is the bottom-face center in rectified camera
            # coordinates. Convert it to the geometric box center.
            center_rect = torch.tensor(
                label.location_bottom_center_rect,
                dtype=torch.float32,
            )
            center_rect[1] -= float(h) / 2.0
            center_rect_h = torch.cat([center_rect, torch.ones(1)])
            center_lidar = (T_lidar_rect @ center_rect_h)[:3]

            R_box_rect = _rotation_y_camera(label.rotation_y)
            R_box_lidar = R_lidar_rect @ R_box_rect
            quat_lidar = _rotmat_to_quat_xyzw(R_box_lidar)

            boxes_2d.append([x1, y1, x2, y2])
            centers_lidar.append(center_lidar)
            # Box-local axes in the KITTI corner convention: x=length,
            # y=height, z=width.
            sizes_local_xyz.append([l, h, w])
            quats_lidar.append(quat_lidar)
            class_names.append(label.class_name)
            truncated.append(label.truncated)
            occluded.append(label.occluded)
            alpha.append(label.alpha)
            rotation_y.append(label.rotation_y)

        n = len(boxes_2d)
        target: Dict[str, object] = {
            "boxes_2d_xyxy": (
                torch.tensor(boxes_2d, dtype=torch.float32)
                if boxes_2d
                else _empty((0, 4), torch.float32)
            ),
            "labels": torch.ones((n,), dtype=torch.long),  # one merged vehicle class
            "label_names": class_names,
            "boxes_3d_center_lidar": (
                torch.stack(centers_lidar)
                if centers_lidar
                else _empty((0, 3), torch.float32)
            ),
            "boxes_3d_size_local_xyz": (
                torch.tensor(sizes_local_xyz, dtype=torch.float32)
                if sizes_local_xyz
                else _empty((0, 3), torch.float32)
            ),
            "boxes_3d_quat_lidar_xyzw": (
                torch.stack(quats_lidar)
                if quats_lidar
                else _empty((0, 4), torch.float32)
            ),
            "truncated": torch.tensor(truncated, dtype=torch.float32),
            "occluded": torch.tensor(occluded, dtype=torch.long),
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "rotation_y_camera": torch.tensor(rotation_y, dtype=torch.float32),
            "P2": calib["P2"],
            "R0_rect": calib["R0_rect"],
            "Tr_velo_to_cam": calib["Tr_velo_to_cam"],
            "T_rect_lidar": calib["T_rect_lidar"],
            "T_lidar_rect": calib["T_lidar_rect"],
            "P_lidar_to_image": calib["P_lidar_to_image"],
            "image_size": torch.tensor(image_size, dtype=torch.long),
            "frame_id": frame_id,
            "sequence_id": "kitti_object",
            "meta": {str(k): str(v) for k, v in paths.items()},
        }
        return target

    def get_target(self, idx: int) -> Dict[str, object]:
        return self._build_target(self.frame_ids[idx])

    def get_lidar_and_target(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        frame_id = self.frame_ids[idx]
        paths = self._paths(frame_id)
        lidar_xyz, reflectance = _load_velodyne_bin(paths["velodyne"])
        target = self._build_target(frame_id)
        if self.return_lidar_reflectance:
            target["lidar_reflectance"] = reflectance
        return lidar_xyz, target

    def __getitem__(self, idx: int):
        frame_id = self.frame_ids[idx]
        paths = self._paths(frame_id)
        image = _load_rgb_image(paths["image"])
        lidar_xyz, reflectance = _load_velodyne_bin(paths["velodyne"])
        target = self._build_target(frame_id)
        if self.return_lidar_reflectance:
            target["lidar_reflectance"] = reflectance
        return image, lidar_xyz, target


def kitti_detection_collate_fn(batch):
    images, point_clouds, targets = zip(*batch)
    return list(images), list(point_clouds), list(targets)
