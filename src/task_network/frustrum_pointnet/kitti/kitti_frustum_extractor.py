from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _as_float_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    return x.detach().float()


def _validate_lidar_xyz(lidar_xyz: torch.Tensor) -> torch.Tensor:
    xyz = _as_float_tensor(lidar_xyz, "lidar_xyz")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"lidar_xyz must have shape (N,3), got {tuple(xyz.shape)}")
    if xyz.shape[0] == 0:
        raise ValueError("lidar_xyz is empty")
    return xyz


def _validate_projection(P2: torch.Tensor) -> torch.Tensor:
    P = _as_float_tensor(P2, "P2")
    if P.shape != (3, 4):
        raise ValueError(f"P2 must have shape (3,4), got {tuple(P.shape)}")
    return P


def _validate_transform(T_rect_lidar: torch.Tensor) -> torch.Tensor:
    T = _as_float_tensor(T_rect_lidar, "T_rect_lidar")
    if T.shape != (4, 4):
        raise ValueError(
            f"T_rect_lidar must have shape (4,4), got {tuple(T.shape)}"
        )
    return T


def _validate_box_xyxy(box_2d_xyxy: torch.Tensor) -> torch.Tensor:
    box = _as_float_tensor(box_2d_xyxy, "box_2d_xyxy").flatten()
    if box.numel() != 4:
        raise ValueError(f"box_2d_xyxy must contain 4 values, got {box.numel()}")
    x1, y1, x2, y2 = box.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid XYXY box: {(x1, y1, x2, y2)}")
    return box


def transform_lidar_to_rect_camera(
    lidar_xyz: torch.Tensor,
    T_rect_lidar: torch.Tensor,
) -> torch.Tensor:
    """Transform KITTI Velodyne points to the rectified camera frame."""
    xyz = _validate_lidar_xyz(lidar_xyz)
    T = _validate_transform(T_rect_lidar).to(device=xyz.device, dtype=xyz.dtype)

    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)
    rect_h = xyz_h @ T.T
    return rect_h[:, :3]


def project_rect_camera_to_image(
    rect_xyz: torch.Tensor,
    P2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project rectified KITTI camera points into image_2.

    KITTI rectified camera coordinates already use x-right, y-down, z-forward,
    so unlike the Unity projector no Y-axis flip is applied.
    """
    xyz = _as_float_tensor(rect_xyz, "rect_xyz")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"rect_xyz must have shape (N,3), got {tuple(xyz.shape)}")

    P = _validate_projection(P2).to(device=xyz.device, dtype=xyz.dtype)
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)
    uvw = xyz_h @ P.T

    depth = xyz[:, 2]
    denom = uvw[:, 2]
    safe = torch.where(
        torch.abs(denom) > 1e-8,
        denom,
        torch.full_like(denom, 1e-8),
    )
    uv = torch.stack([uvw[:, 0] / safe, uvw[:, 1] / safe], dim=1)
    return uv, depth


def project_lidar_to_image(
    lidar_xyz: torch.Tensor,
    P2: torch.Tensor,
    T_rect_lidar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rectified-camera XYZ, image UV, and rectified-camera depth."""
    rect_xyz = transform_lidar_to_rect_camera(lidar_xyz, T_rect_lidar)
    uv, depth = project_rect_camera_to_image(rect_xyz, P2)
    return rect_xyz, uv, depth


def points_in_2d_box_mask(
    uv: torch.Tensor,
    depth: torch.Tensor,
    box_2d_xyxy: torch.Tensor,
    *,
    image_size: Optional[Tuple[int, int]] = None,
    min_depth: float = 1e-4,
    box_margin_px: float = 0.0,
) -> torch.Tensor:
    """Mask projected points that are in front of the camera and inside a 2D box."""
    uv = _as_float_tensor(uv, "uv")
    depth = _as_float_tensor(depth, "depth").flatten()
    box = _validate_box_xyxy(box_2d_xyxy)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must have shape (N,2), got {tuple(uv.shape)}")
    if depth.shape[0] != uv.shape[0]:
        raise ValueError("depth and uv must contain the same number of points")
    if min_depth < 0:
        raise ValueError(f"min_depth must be >= 0, got {min_depth}")
    if box_margin_px < 0:
        raise ValueError(f"box_margin_px must be >= 0, got {box_margin_px}")

    x1, y1, x2, y2 = box.tolist()
    x1 -= float(box_margin_px)
    y1 -= float(box_margin_px)
    x2 += float(box_margin_px)
    y2 += float(box_margin_px)

    u = uv[:, 0]
    v = uv[:, 1]
    mask = depth > float(min_depth)
    mask &= torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(depth)
    mask &= (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)

    if image_size is not None:
        h, w = (int(image_size[0]), int(image_size[1]))
        mask &= (u >= 0.0) & (u <= float(w - 1))
        mask &= (v >= 0.0) & (v <= float(h - 1))

    return mask


def quat_xyzw_to_rotmat(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Quaternion to a local-box -> enclosing-frame rotation matrix."""
    q = _as_float_tensor(quat_xyzw, "quat_xyzw").flatten()
    if q.numel() != 4:
        raise ValueError(f"quat_xyzw must contain 4 values, got {q.numel()}")

    q = q / torch.linalg.norm(q).clamp_min(1e-12)
    qx, qy, qz, qw = q.unbind()

    return torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ]
            ),
            torch.stack(
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ]
            ),
            torch.stack(
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ]
            ),
        ],
        dim=0,
    )


def points_in_oriented_box_lidar(
    lidar_xyz: torch.Tensor,
    box_center_lidar: torch.Tensor,
    box_size_local_xyz: torch.Tensor,
    box_quat_xyzw_lidar: torch.Tensor,
) -> torch.Tensor:
    """
    Determine whether LiDAR-frame points lie inside an oriented 3D box.

    `box_size_local_xyz` is ordered along the box's local x/y/z axes. For a
    converted KITTI box this is [length, height, width].
    """
    points = _validate_lidar_xyz(lidar_xyz)
    center = _as_float_tensor(box_center_lidar, "box_center_lidar").flatten()
    size = _as_float_tensor(box_size_local_xyz, "box_size_local_xyz").flatten()
    if center.numel() != 3:
        raise ValueError("box_center_lidar must contain 3 values")
    if size.numel() != 3:
        raise ValueError("box_size_local_xyz must contain 3 values")
    if torch.any(size <= 0):
        raise ValueError(f"Box dimensions must be positive, got {size.tolist()}")

    center = center.to(device=points.device, dtype=points.dtype)
    size = size.to(device=points.device, dtype=points.dtype)
    R_box_to_lidar = quat_xyzw_to_rotmat(box_quat_xyzw_lidar).to(
        device=points.device, dtype=points.dtype
    )

    # Column-vector convention: p_lidar = R * p_local + center.
    # For row-vector point storage the inverse is (p-center) @ R.
    local = (points - center[None, :]) @ R_box_to_lidar
    half = size[None, :] / 2.0
    return (torch.abs(local) <= (half + 1e-6)).all(dim=1)


def sample_points(
    points: torch.Tensor,
    num_points: int,
    *,
    pad_repeat: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample a fixed point count, repeating points when necessary."""
    x = _as_float_tensor(points, "points")
    if x.ndim != 2:
        raise ValueError(f"points must have shape (N,C), got {tuple(x.shape)}")
    if x.shape[0] == 0:
        raise ValueError("Cannot sample an empty point set")
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}")

    n = x.shape[0]
    if n >= num_points:
        idx = torch.randperm(n, generator=generator, device=x.device)[:num_points]
        return x[idx], idx

    if not pad_repeat:
        idx = torch.arange(n, device=x.device)
        return x, idx

    pad_count = num_points - n
    pad_idx = torch.randint(
        low=0,
        high=n,
        size=(pad_count,),
        generator=generator,
        device=x.device,
    )
    idx = torch.cat([torch.arange(n, device=x.device), pad_idx], dim=0)
    return x[idx], idx


@dataclass
class KittiFrustumExtractionResult:
    frustum_points_lidar: torch.Tensor
    frustum_points_rect: torch.Tensor
    frustum_uv: torch.Tensor
    frustum_depth: torch.Tensor
    frustum_mask_full: torch.Tensor

    sampled_points_lidar: Optional[torch.Tensor] = None
    sampled_points_rect: Optional[torch.Tensor] = None
    sampled_indices_in_frustum: Optional[torch.Tensor] = None
    gt_point_mask_in_frustum: Optional[torch.Tensor] = None


class KittiFrustumExtractor:
    """Pure KITTI geometry for 2D-frustum extraction and GT point masking."""

    def __init__(
        self,
        *,
        box_margin_px: float = 5.0,
        min_depth: float = 1e-4,
        num_sampled_points: Optional[int] = 1024,
        pad_repeat: bool = True,
    ) -> None:
        if box_margin_px < 0:
            raise ValueError("box_margin_px must be non-negative")
        if min_depth < 0:
            raise ValueError("min_depth must be non-negative")
        if num_sampled_points is not None and num_sampled_points <= 0:
            raise ValueError("num_sampled_points must be positive or None")

        self.box_margin_px = float(box_margin_px)
        self.min_depth = float(min_depth)
        self.num_sampled_points = num_sampled_points
        self.pad_repeat = bool(pad_repeat)

    def extract(
        self,
        *,
        lidar_xyz: torch.Tensor,
        box_2d_xyxy: torch.Tensor,
        P2: torch.Tensor,
        T_rect_lidar: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        gt_box_center_lidar: Optional[torch.Tensor] = None,
        gt_box_size_local_xyz: Optional[torch.Tensor] = None,
        gt_box_quat_xyzw_lidar: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> KittiFrustumExtractionResult:
        lidar_xyz = _validate_lidar_xyz(lidar_xyz)
        rect_xyz, uv, depth = project_lidar_to_image(
            lidar_xyz=lidar_xyz,
            P2=P2,
            T_rect_lidar=T_rect_lidar,
        )

        full_mask = points_in_2d_box_mask(
            uv=uv,
            depth=depth,
            box_2d_xyxy=box_2d_xyxy,
            image_size=image_size,
            min_depth=self.min_depth,
            box_margin_px=self.box_margin_px,
        )

        frustum_lidar = lidar_xyz[full_mask]
        frustum_rect = rect_xyz[full_mask]
        frustum_uv = uv[full_mask]
        frustum_depth = depth[full_mask]

        if frustum_lidar.shape[0] == 0:
            raise ValueError("Frustum contains zero LiDAR points")

        sampled_lidar = None
        sampled_rect = None
        sampled_indices = None
        if self.num_sampled_points is not None:
            sampled_lidar, sampled_indices = sample_points(
                frustum_lidar,
                self.num_sampled_points,
                pad_repeat=self.pad_repeat,
                generator=generator,
            )
            sampled_rect = frustum_rect[sampled_indices]

        gt_point_mask = None
        gt_args = (
            gt_box_center_lidar,
            gt_box_size_local_xyz,
            gt_box_quat_xyzw_lidar,
        )
        if all(x is not None for x in gt_args):
            ref_points = sampled_lidar if sampled_lidar is not None else frustum_lidar
            gt_point_mask = points_in_oriented_box_lidar(
                ref_points,
                gt_box_center_lidar,  # type: ignore[arg-type]
                gt_box_size_local_xyz,  # type: ignore[arg-type]
                gt_box_quat_xyzw_lidar,  # type: ignore[arg-type]
            )
        elif any(x is not None for x in gt_args):
            raise ValueError("All three GT box tensors must be provided together")

        return KittiFrustumExtractionResult(
            frustum_points_lidar=frustum_lidar,
            frustum_points_rect=frustum_rect,
            frustum_uv=frustum_uv,
            frustum_depth=frustum_depth,
            frustum_mask_full=full_mask,
            sampled_points_lidar=sampled_lidar,
            sampled_points_rect=sampled_rect,
            sampled_indices_in_frustum=sampled_indices,
            gt_point_mask_in_frustum=gt_point_mask,
        )
