from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    def __init__(self, channels, *, use_bn: bool = True):
        super().__init__()
        layers = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CenterRefinementNet(nn.Module):
    """
    PointNet-style center refinement module.

    Input:
        object_points_lidar: (B, N, 3)

    Output:
        center_offset_lidar: (B, 3)

    Meaning:
        refined_center_lidar = object_points_mean_lidar + center_offset_lidar
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 512,
        normalize_points: bool = True,
        use_bn: bool = True,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.normalize_points = normalize_points

        self.point_mlp = SharedMLP(
            [input_channels, 64, 128, 256, hidden_dim],
            use_bn=use_bn,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    @staticmethod
    def _normalize(
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        xyz = points[..., :3]

        centroid = xyz.mean(dim=1, keepdim=True)
        centered = xyz - centroid

        scale = torch.linalg.norm(centered, dim=-1).amax(dim=1, keepdim=True)
        scale = scale.clamp_min(1e-6).unsqueeze(-1)

        xyz_norm = centered / scale

        if points.shape[-1] > 3:
            points_norm = torch.cat([xyz_norm, points[..., 3:]], dim=-1)
        else:
            points_norm = xyz_norm

        return points_norm, {
            "object_mean_lidar": centroid.squeeze(1),
            "scale": scale.squeeze(1),
        }

    def forward(self, object_points_lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        if object_points_lidar.ndim != 3:
            raise ValueError(
                f"Expected object_points_lidar (B,N,C), got {tuple(object_points_lidar.shape)}"
            )
        if object_points_lidar.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected C={self.input_channels}, got C={object_points_lidar.shape[-1]}"
            )

        if self.normalize_points:
            points_in, stats = self._normalize(object_points_lidar)
        else:
            points_in = object_points_lidar
            stats = {
                "object_mean_lidar": object_points_lidar[..., :3].mean(dim=1),
                "scale": torch.ones(
                    object_points_lidar.shape[0],
                    1,
                    device=object_points_lidar.device,
                    dtype=object_points_lidar.dtype,
                ),
            }

        x = points_in.transpose(1, 2).contiguous()  # (B,C,N)
        feat = self.point_mlp(x)                    # (B,H,N)
        global_feat = torch.max(feat, dim=2).values # (B,H)

        offset_norm = self.fc(global_feat)           # (B,3)

        if self.normalize_points:
            center_offset_lidar = offset_norm * stats["scale"]
        else:
            center_offset_lidar = offset_norm

        refined_center_lidar = stats["object_mean_lidar"] + center_offset_lidar
        centered_points_lidar = object_points_lidar[..., :3] - refined_center_lidar[:, None, :]

        return {
            "center_offset_lidar": center_offset_lidar,
            "refined_center_lidar": refined_center_lidar,
            "centered_points_lidar": centered_points_lidar,
            **stats,
        }


@dataclass
class CenterRefinementLossConfig:
    loss_type: str = "smooth_l1"  # "smooth_l1" or "l1" or "mse"
    beta: float = 0.1


class CenterRefinementLoss(nn.Module):
    def __init__(self, cfg: Optional[CenterRefinementLossConfig] = None):
        super().__init__()
        self.cfg = cfg or CenterRefinementLossConfig()

    def forward(
        self,
        pred_center_lidar: torch.Tensor,
        gt_center_lidar: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.loss_type == "smooth_l1":
            return F.smooth_l1_loss(
                pred_center_lidar,
                gt_center_lidar,
                beta=self.cfg.beta,
            )
        if self.cfg.loss_type == "l1":
            return F.l1_loss(pred_center_lidar, gt_center_lidar)
        if self.cfg.loss_type == "mse":
            return F.mse_loss(pred_center_lidar, gt_center_lidar)

        raise ValueError(f"Unknown loss_type: {self.cfg.loss_type}")