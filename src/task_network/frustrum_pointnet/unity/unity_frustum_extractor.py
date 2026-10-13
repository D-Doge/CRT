from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def _validate_box_xyxy(box_2d_xyxy: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(box_2d_xyxy):
        raise TypeError(f"box_2d_xyxy must be torch.Tensor, got {type(box_2d_xyxy)}")

    box = box_2d_xyxy.detach().float().flatten()
    if box.numel() != 4:
        raise ValueError(f"box_2d_xyxy must have 4 values, got shape {tuple(box_2d_xyxy.shape)}")

    x1, y1, x2, y2 = box.tolist()
    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid XYXY box: {(x1, y1, x2, y2)}")

    return box


def _validate_intrinsics(K: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(K):
        raise TypeError(f"intrinsics must be torch.Tensor, got {type(K)}")
    K = K.detach().float()
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3,3), got {tuple(K.shape)}")
    return K


def _validate_transform(T_cam_lidar: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(T_cam_lidar):
        raise TypeError(f"T_cam_lidar must be torch.Tensor, got {type(T_cam_lidar)}")
    T = T_cam_lidar.detach().float()
    if T.shape != (4, 4):
        raise ValueError(f"T_cam_lidar must have shape (4,4), got {tuple(T.shape)}")
    return T


def _validate_lidar_xyz(lidar_xyz: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(lidar_xyz):
        raise TypeError(f"lidar_xyz must be torch.Tensor, got {type(lidar_xyz)}")
    x = lidar_xyz.detach().float()
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"lidar_xyz must have shape (N,3), got {tuple(x.shape)}")
    if x.shape[0] == 0:
        raise ValueError("lidar_xyz is empty")
    return x


def transform_lidar_to_camera(
    lidar_xyz: torch.Tensor,
    T_cam_lidar: torch.Tensor,
) -> torch.Tensor:
    """
    Transform LiDAR-frame points into the camera frame.

    Args:
        lidar_xyz: (N, 3)
        T_cam_lidar: (4, 4) mapping lidar -> camera

    Returns:
        cam_xyz: (N, 3)
    """
    pts = _validate_lidar_xyz(lidar_xyz)
    T = _validate_transform(T_cam_lidar)

    n = pts.shape[0]
    ones = torch.ones((n, 1), dtype=pts.dtype, device=pts.device)
    pts_h = torch.cat([pts, ones], dim=1)                # (N,4)

    cam_h = (T @ pts_h.T).T                              # (N,4)
    cam_xyz = cam_h[:, :3]
    return cam_xyz


def project_camera_points(
    cam_xyz: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project Unity camera-frame 3D points into image pixels using OpenCV-style K.

    Unity camera frame:
        +X right, +Y up, +Z forward

    OpenCV image coordinates:
        +u right, +v down

    Therefore we flip Y before applying K:
        x_cv =  x_unity
        y_cv = -y_unity
        z_cv =  z_unity
    """
    if not torch.is_tensor(cam_xyz):
        raise TypeError(f"cam_xyz must be torch.Tensor, got {type(cam_xyz)}")

    xyz = cam_xyz.detach().float()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"cam_xyz must have shape (N,3), got {tuple(xyz.shape)}")

    K = _validate_intrinsics(intrinsics)

    x = xyz[:, 0]
    y = -xyz[:, 1]
    z = xyz[:, 2]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z_safe = torch.clamp(z, min=1e-8)

    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy

    uv = torch.stack([u, v], dim=1)
    return uv, z


def points_in_box_mask(
    uv: torch.Tensor,
    z: torch.Tensor,
    box_2d_xyxy: torch.Tensor,
    *,
    image_size: Optional[Tuple[int, int]] = None,
    min_depth: float = 1e-6,
    box_margin_px: float = 0.0,
) -> torch.Tensor:
    """
    Compute mask of projected points inside a 2D box and in front of the camera.

    Args:
        uv: (N,2)
        z: (N,)
        box_2d_xyxy: (4,)
        image_size: optional (H,W), if provided also requires point projection inside image
        min_depth: minimum valid positive depth
        box_margin_px: expands box on all sides by this amount

    Returns:
        mask: (N,) bool
    """
    if not torch.is_tensor(uv):
        raise TypeError(f"uv must be torch.Tensor, got {type(uv)}")
    if not torch.is_tensor(z):
        raise TypeError(f"z must be torch.Tensor, got {type(z)}")

    uv = uv.detach().float()
    z = z.detach().float()

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must have shape (N,2), got {tuple(uv.shape)}")
    if z.ndim != 1 or z.shape[0] != uv.shape[0]:
        raise ValueError(f"z must have shape (N,), got {tuple(z.shape)}")

    box = _validate_box_xyxy(box_2d_xyxy)
    x1, y1, x2, y2 = box.tolist()

    x1 -= float(box_margin_px)
    y1 -= float(box_margin_px)
    x2 += float(box_margin_px)
    y2 += float(box_margin_px)

    u = uv[:, 0]
    v = uv[:, 1]

    valid = z > float(min_depth)
    valid = valid & torch.isfinite(u) & torch.isfinite(v)

    inside_box = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    valid = valid & inside_box

    if image_size is not None:
        h, w = image_size
        inside_img = (u >= 0.0) & (u <= float(w - 1)) & (v >= 0.0) & (v <= float(h - 1))
        valid = valid & inside_img

    return valid


def sample_points(
    points: torch.Tensor,
    num_points: int,
    *,
    sampling: str = "random",
    pad_repeat: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a fixed number of points from (N, C).

    Args:
        points: (N,C)
        num_points: number of output points
        sampling: 'random' or 'none'
        pad_repeat: if True and N < num_points, repeat points to pad

    Returns:
        sampled_points: (num_points,C) if sampling != 'none', else original
        indices: indices into input points
    """
    if not torch.is_tensor(points):
        raise TypeError(f"points must be torch.Tensor, got {type(points)}")
    x = points.detach()
    if x.ndim != 2:
        raise ValueError(f"points must have shape (N,C), got {tuple(x.shape)}")
    if x.shape[0] == 0:
        raise ValueError("Cannot sample from zero points")
    if sampling not in {"random", "none"}:
        raise ValueError(f"Unknown sampling mode: {sampling}")
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}")

    n = x.shape[0]
    device = x.device

    if sampling == "none":
        idx = torch.arange(n, device=device, dtype=torch.long)
        return x, idx

    if n >= num_points:
        idx = torch.randperm(n, device=device)[:num_points]
        return x[idx], idx

    if not pad_repeat:
        idx = torch.arange(n, device=device, dtype=torch.long)
        return x, idx

    pad_count = num_points - n
    pad_idx = torch.randint(0, n, (pad_count,), device=device)
    idx = torch.cat([torch.arange(n, device=device), pad_idx], dim=0)
    return x[idx], idx


def mask_points_in_3d_box_camera_frame(
    cam_xyz: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    box_quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    """
    Determine whether camera-frame points lie inside an oriented 3D box.

    Assumes:
    - box_center: (3,)
    - box_size:   (3,) full box lengths along local box axes
    - box_quat_xyzw: (4,) quaternion in x,y,z,w order

    Returns:
        inside_mask: (N,) bool
    """
    if not torch.is_tensor(cam_xyz):
        raise TypeError(f"cam_xyz must be torch.Tensor, got {type(cam_xyz)}")
    pts = cam_xyz.detach().float()
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"cam_xyz must have shape (N,3), got {tuple(pts.shape)}")

    center = box_center.detach().float().flatten()
    size = box_size.detach().float().flatten()
    quat = box_quat_xyzw.detach().float().flatten()

    if center.numel() != 3:
        raise ValueError("box_center must have 3 values")
    if size.numel() != 3:
        raise ValueError("box_size must have 3 values")
    if quat.numel() != 4:
        raise ValueError("box_quat_xyzw must have 4 values")

    qx, qy, qz, qw = quat.tolist()
    qnorm = max((qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5, 1e-12)
    qx /= qnorm
    qy /= qnorm
    qz /= qnorm
    qw /= qnorm

    # Rotation matrix from local box frame to camera frame
    R = torch.tensor(
        [
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
        ],
        dtype=pts.dtype,
        device=pts.device,
    )

    # Camera frame -> local box frame
    local = (pts - center[None, :]) @ R.T
    half = size[None, :] / 2.0

    inside = torch.abs(local[:, 0]) <= half[:, 0]
    inside = inside & (torch.abs(local[:, 1]) <= half[:, 1])
    inside = inside & (torch.abs(local[:, 2]) <= half[:, 2])
    return inside


@dataclass
class FrustumExtractionResult:
    """
    Output container for a single extracted frustum.
    """
    frustum_points_lidar: torch.Tensor      # (M,3) in LiDAR frame
    frustum_points_cam: torch.Tensor        # (M,3) in camera frame
    frustum_uv: torch.Tensor                # (M,2) projected pixels
    frustum_depth: torch.Tensor             # (M,)
    frustum_mask_full: torch.Tensor         # (N,) mask on original point cloud

    sampled_points_cam: Optional[torch.Tensor] = None
    sampled_points_lidar: Optional[torch.Tensor] = None
    sampled_indices_in_frustum: Optional[torch.Tensor] = None

    gt_point_mask_in_frustum: Optional[torch.Tensor] = None
    box_center_cam: Optional[torch.Tensor] = None
    box_size: Optional[torch.Tensor] = None
    box_quat_xyzw: Optional[torch.Tensor] = None


class FrustumExtractor:
    """
    Pure geometry frustum extractor for Frustum PointNet-style pipelines.

    Main behavior:
    - LiDAR -> camera transform
    - project into image
    - keep points inside 2D box

    Optional:
    - fixed-point sampling
    - per-point GT mask from 3D box
    """

    def __init__(
        self,
        *,
        box_margin_px: float = 0.0,
        min_depth: float = 1e-6,
        num_sampled_points: Optional[int] = None,
        sampling: str = "random",
        pad_repeat: bool = True,
    ) -> None:
        if box_margin_px < 0:
            raise ValueError(f"box_margin_px must be >= 0, got {box_margin_px}")
        if min_depth < 0:
            raise ValueError(f"min_depth must be >= 0, got {min_depth}")
        if num_sampled_points is not None and num_sampled_points <= 0:
            raise ValueError(f"num_sampled_points must be > 0, got {num_sampled_points}")
        if sampling not in {"random", "none"}:
            raise ValueError(f"Unknown sampling mode: {sampling}")

        self.box_margin_px = float(box_margin_px)
        self.min_depth = float(min_depth)
        self.num_sampled_points = num_sampled_points
        self.sampling = sampling
        self.pad_repeat = bool(pad_repeat)

    def extract(
        self,
        *,
        lidar_xyz: torch.Tensor,
        box_2d_xyxy: torch.Tensor,
        intrinsics: torch.Tensor,
        T_cam_lidar: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        gt_box_center_cam: Optional[torch.Tensor] = None,
        gt_box_size: Optional[torch.Tensor] = None,
        gt_box_quat_xyzw: Optional[torch.Tensor] = None,
    ) -> FrustumExtractionResult:
        """
        Extract one frustum for one 2D box.

        Returns all frustum points, and optionally sampled points and
        per-point GT labels from a 3D box.
        """
        lidar_xyz = _validate_lidar_xyz(lidar_xyz)
        K = _validate_intrinsics(intrinsics)
        T = _validate_transform(T_cam_lidar)
        box = _validate_box_xyxy(box_2d_xyxy)

        cam_xyz = transform_lidar_to_camera(lidar_xyz, T)
        uv, depth = project_camera_points(cam_xyz, K)

        full_mask = points_in_box_mask(
            uv=uv,
            z=depth,
            box_2d_xyxy=box,
            image_size=image_size,
            min_depth=self.min_depth,
            box_margin_px=self.box_margin_px,
        )

        frustum_points_lidar = lidar_xyz[full_mask]
        frustum_points_cam = cam_xyz[full_mask]
        frustum_uv = uv[full_mask]
        frustum_depth = depth[full_mask]

        if frustum_points_cam.shape[0] == 0:
            raise ValueError("Frustum contains zero points")

        sampled_points_cam = None
        sampled_points_lidar = None
        sampled_indices_in_frustum = None
        gt_point_mask_in_frustum = None

        if self.num_sampled_points is not None:
            sampled_points_cam, sampled_indices_in_frustum = sample_points(
                frustum_points_cam,
                num_points=self.num_sampled_points,
                sampling=self.sampling,
                pad_repeat=self.pad_repeat,
            )
            sampled_points_lidar = frustum_points_lidar[sampled_indices_in_frustum]

        if (
            gt_box_center_cam is not None
            and gt_box_size is not None
            and gt_box_quat_xyzw is not None
        ):
            ref_points = sampled_points_cam if sampled_points_cam is not None else frustum_points_cam
            gt_point_mask_in_frustum = mask_points_in_3d_box_camera_frame(
                cam_xyz=ref_points,
                box_center=gt_box_center_cam,
                box_size=gt_box_size,
                box_quat_xyzw=gt_box_quat_xyzw,
            )

        return FrustumExtractionResult(
            frustum_points_lidar=frustum_points_lidar,
            frustum_points_cam=frustum_points_cam,
            frustum_uv=frustum_uv,
            frustum_depth=frustum_depth,
            frustum_mask_full=full_mask,
            sampled_points_cam=sampled_points_cam,
            sampled_points_lidar=sampled_points_lidar,
            sampled_indices_in_frustum=sampled_indices_in_frustum,
            gt_point_mask_in_frustum=gt_point_mask_in_frustum,
            box_center_cam=gt_box_center_cam,
            box_size=gt_box_size,
            box_quat_xyzw=gt_box_quat_xyzw,
        )