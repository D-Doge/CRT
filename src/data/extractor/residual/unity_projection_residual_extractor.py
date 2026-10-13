from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from data.data_sample_interface import DataSample
from data.extractor.ExtractorInterface import DataSampleFieldExtractor


class ProjectionResidualExtractor(DataSampleFieldExtractor):
    """
    Computes projection-based calibration residuals for a single DataSample.

    Current implementation:
    - Project LiDAR points into the RGB image using intrinsics + T_cam_lidar
    - Compute an RGB edge map with Sobel gradients
    - For each valid projected LiDAR point, store distance to nearest image edge
    - Return residuals as a 1D tensor of shape (M,)

    Required DataSample fields:
    - rgb
    - lidar_xyz
    - intrinsics
    - T_cam_lidar
    - image_size OR rgb shape from which image size can be inferred

    Notes
    -----
    - T_cam_lidar is assumed to map LiDAR coordinates -> camera coordinates.
    - Residuals are returned in pixel units.
    - Only valid projected points are kept:
        * positive camera depth
        * finite projection
        * projected inside image bounds
    """

    def __init__(
        self,
        edge_threshold: float = 0.2,
        sobel_ksize: int = 3,
        max_points: int | None = None,
    ) -> None:
        if edge_threshold < 0:
            raise ValueError(f"edge_threshold must be >= 0, got {edge_threshold}")
        if sobel_ksize != 3:
            raise ValueError("Only sobel_ksize=3 is currently supported.")
        if max_points is not None and max_points <= 0:
            raise ValueError(f"max_points must be > 0, got {max_points}")

        self._edge_threshold = float(edge_threshold)
        self._sobel_ksize = int(sobel_ksize)
        self._max_points = max_points

    @property
    def output_field(self) -> str:
        return "residuals"

    @property
    def extractor_id(self) -> str:
        mp = "all" if self._max_points is None else str(self._max_points)
        return f"projection_residual_edge_v1_thr{self._edge_threshold:g}_mp{mp}"

    @torch.no_grad()
    def extract(self, sample: DataSample) -> torch.Tensor:
        if sample.rgb is None:
            raise ValueError("ProjectionResidualExtractor: sample.rgb is None")
        if sample.lidar_xyz is None:
            raise ValueError("ProjectionResidualExtractor: sample.lidar_xyz is None")
        if sample.intrinsics is None:
            raise ValueError("ProjectionResidualExtractor: sample.intrinsics is None")
        if sample.T_cam_lidar is None:
            raise ValueError("ProjectionResidualExtractor: sample.T_cam_lidar is None")

        rgb = self._prepare_rgb(sample.rgb)
        H, W = self._infer_image_size(sample, rgb)

        lidar_xyz = self._prepare_lidar(sample.lidar_xyz)
        K = self._prepare_intrinsics(sample.intrinsics)
        T = self._prepare_transform(sample.T_cam_lidar)

        if self._max_points is not None and lidar_xyz.shape[0] > self._max_points:
            idx = torch.linspace(
                0,
                lidar_xyz.shape[0] - 1,
                steps=self._max_points,
            ).long()

            lidar_xyz = lidar_xyz[idx]

        edge_map = self._compute_edge_map(rgb)              # (H, W) bool
        dist_map = self._distance_to_edge_map(edge_map)     # (H, W) float32

        uv, valid = self._project_points(lidar_xyz, K, T, H, W)

        if valid.sum().item() == 0:
            raise ValueError(
                "ProjectionResidualExtractor: no valid LiDAR points project into the image."
            )

        uv_valid = uv[valid]  # (M, 2)

        # nearest-neighbor pixel sampling of distance transform
        u_idx = uv_valid[:, 0].round().long().clamp(0, W - 1)
        v_idx = uv_valid[:, 1].round().long().clamp(0, H - 1)

        residuals = dist_map[v_idx, u_idx].detach().cpu().float()

        if residuals.numel() == 0:
            raise ValueError("ProjectionResidualExtractor: residuals are empty.")

        return residuals

    # ------------------------------------------------------------------
    # Input prep
    # ------------------------------------------------------------------

    def _prepare_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(rgb):
            raise TypeError(f"Expected sample.rgb to be torch.Tensor, got {type(rgb)}")

        x = rgb.detach().cpu()

        if x.ndim != 3:
            raise ValueError(f"Expected rgb with 3 dims, got shape {tuple(x.shape)}")

        # CHW
        if x.shape[0] == 3:
            pass
        # HWC
        elif x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        else:
            raise ValueError(
                f"Expected rgb shape (3,H,W) or (H,W,3), got {tuple(x.shape)}"
            )

        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            if x.max().item() > 1.5:
                x = x / 255.0

        return x

    def _infer_image_size(self, sample: DataSample, rgb_chw: torch.Tensor) -> Tuple[int, int]:
        if sample.image_size is not None:
            H, W = sample.image_size
            return int(H), int(W)

        _, H, W = rgb_chw.shape
        return int(H), int(W)

    def _prepare_lidar(self, lidar_xyz: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(lidar_xyz):
            raise TypeError(
                f"Expected sample.lidar_xyz to be torch.Tensor, got {type(lidar_xyz)}"
            )

        x = lidar_xyz.detach().cpu().float()

        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"Expected lidar_xyz shape (N,3), got {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError("Point cloud is empty.")

        return x

    def _prepare_intrinsics(self, intrinsics: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(intrinsics):
            raise TypeError(
                f"Expected sample.intrinsics to be torch.Tensor, got {type(intrinsics)}"
            )
        K = intrinsics.detach().cpu().float()

        if K.shape != (3, 3):
            raise ValueError(f"Expected intrinsics shape (3,3), got {tuple(K.shape)}")
        return K

    def _prepare_transform(self, T_cam_lidar: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(T_cam_lidar):
            raise TypeError(
                f"Expected sample.T_cam_lidar to be torch.Tensor, got {type(T_cam_lidar)}"
            )
        T = T_cam_lidar.detach().cpu().float()

        if T.shape != (4, 4):
            raise ValueError(f"Expected T_cam_lidar shape (4,4), got {tuple(T.shape)}")
        return T

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _compute_edge_map(self, rgb_chw: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            edge_map: bool tensor of shape (H, W)
        """
        gray = self._rgb_to_gray(rgb_chw)   # (1, 1, H, W)

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)  # (1,1,3,3)

        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0],
              [ 0.0,  0.0,  0.0],
              [ 1.0,  2.0,  1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)  # (1,1,3,3)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)

        mag = torch.sqrt(gx * gx + gy * gy).squeeze(0).squeeze(0)  # (H, W)

        # normalize for thresholding stability
        max_mag = mag.max()
        if max_mag.item() > 0:
            mag = mag / max_mag

        edge_map = mag >= self._edge_threshold

        # avoid pathological "no edges anywhere"
        if not edge_map.any():
            # fallback: mark strongest 1% pixels as edges, at least one pixel
            flat = mag.flatten()
            k = max(1, int(round(0.01 * flat.numel())))
            vals, idx = torch.topk(flat, k=k)
            edge_map = torch.zeros_like(flat, dtype=torch.bool)
            edge_map[idx] = True
            edge_map = edge_map.view_as(mag)

        return edge_map

    def _distance_to_edge_map(self, edge_map: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance-to-nearest-edge in pixel units.
        """
        edge_np = edge_map.detach().cpu().numpy().astype(bool)

        # distance_transform_edt computes distance to nearest zero.
        # We therefore pass ~edge_map so that edges become zeros.
        dist_np = distance_transform_edt(~edge_np).astype(np.float32)

        return torch.from_numpy(dist_np)

    def _project_points(
        self,
        lidar_xyz: torch.Tensor,
        K: torch.Tensor,
        T_cam_lidar: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project LiDAR points into the RGB image.

        T_cam_lidar maps LiDAR -> Unity camera frame:
            +X right, +Y up, +Z forward

        OpenCV-style K expects image coordinates:
            +u right, +v down

        Therefore y must be flipped before applying K.
        """
        N = lidar_xyz.shape[0]

        ones = torch.ones((N, 1), dtype=lidar_xyz.dtype)
        xyz1 = torch.cat([lidar_xyz, ones], dim=1)  # (N, 4)

        cam_xyz1 = (T_cam_lidar @ xyz1.T).T
        cam_xyz = cam_xyz1[:, :3]

        x = cam_xyz[:, 0]
        y = -cam_xyz[:, 1]   # Unity camera -> OpenCV image convention
        z = cam_xyz[:, 2]

        z_positive = z > 1e-8
        z_safe = torch.clamp(z, min=1e-8)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        u = fx * x / z_safe + cx
        v = fy * y / z_safe + cy

        finite = torch.isfinite(u) & torch.isfinite(v)
        inside = (u >= 0.0) & (u <= (W - 1)) & (v >= 0.0) & (v <= (H - 1))
        valid = z_positive & finite & inside

        uv = torch.stack([u, v], dim=1)
        return uv, valid

    def _rgb_to_gray(self, rgb_chw: torch.Tensor) -> torch.Tensor:
            """
            Input:  (3, H, W)
            Output: (1, 1, H, W)
            """
            r, g, b = rgb_chw[0], rgb_chw[1], rgb_chw[2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray.unsqueeze(0).unsqueeze(0)