from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

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


class FrustumBoxHead(nn.Module):
    """
    PointNet-style 3D box regression head.

    Input:
        centered_points_lidar: (B, N, 3)

    Output:
        center_residual_lidar: (B, 3)
        size_lidar:            (B, 3), positive
        quat_xyzw:             (B, 4), normalized
        confidence_logit:      (B,)
        confidence:            (B,)

    Intended center composition:
        final_center_lidar = refined_center_lidar + center_residual_lidar
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 512,
        use_bn: bool = False,
        min_size: float = 1e-3,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.min_size = float(min_size)

        self.point_mlp = SharedMLP(
            [input_channels, 64, 128, 256, hidden_dim],
            use_bn=use_bn,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        self.center_residual_head = nn.Linear(128, 3)
        self.size_head = nn.Linear(128, 3)
        self.quat_head = nn.Linear(128, 4)

        # Object / box quality logit.
        # Later DoC can use either this scalar confidence or the raw logit.
        self.confidence_head = nn.Linear(128, 1)

    def forward(self, centered_points_lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        if centered_points_lidar.ndim != 3:
            raise ValueError(
                f"Expected centered_points_lidar (B,N,C), got {tuple(centered_points_lidar.shape)}"
            )
        if centered_points_lidar.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected C={self.input_channels}, got C={centered_points_lidar.shape[-1]}"
            )

        x = centered_points_lidar.transpose(1, 2).contiguous()  # (B,C,N)

        feat = self.point_mlp(x)                    # (B,H,N)
        global_feat = torch.max(feat, dim=2).values # (B,H)

        h = self.fc(global_feat)                    # (B,128)

        center_residual_lidar = self.center_residual_head(h)

        # Positive box dimensions.
        size_lidar = F.softplus(self.size_head(h)) + self.min_size

        quat_raw = self.quat_head(h)
        quat_xyzw = F.normalize(quat_raw, p=2, dim=1)

        confidence_logit = self.confidence_head(h).squeeze(-1)
        confidence = torch.sigmoid(confidence_logit)

        return {
            "center_residual_lidar": center_residual_lidar,
            "size_lidar": size_lidar,
            "quat_xyzw": quat_xyzw,
            "confidence_logit": confidence_logit,
            "confidence": confidence,
            "box_feature": h,
        }


@dataclass
class BoxHeadLossConfig:
    center_weight: float = 1.0
    size_weight: float = 1.0
    quat_weight: float = 0.2
    confidence_weight: float = 0.1
    beta: float = 0.1


class FrustumBoxHeadLoss(nn.Module):
    def __init__(self, cfg: Optional[BoxHeadLossConfig] = None):
        super().__init__()
        self.cfg = cfg or BoxHeadLossConfig()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        *,
        gt_center_residual_lidar: torch.Tensor,
        gt_size_lidar: torch.Tensor,
        gt_quat_xyzw: torch.Tensor,
        gt_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        center_loss = F.smooth_l1_loss(
            pred["center_residual_lidar"],
            gt_center_residual_lidar,
            beta=self.cfg.beta,
        )

        size_loss = F.smooth_l1_loss(
            pred["size_lidar"],
            gt_size_lidar,
            beta=self.cfg.beta,
        )

        gt_quat = F.normalize(gt_quat_xyzw, p=2, dim=1)
        pred_quat = F.normalize(pred["quat_xyzw"], p=2, dim=1)

        # q and -q represent the same rotation, so use abs dot product.
        quat_dot = torch.abs(torch.sum(pred_quat * gt_quat, dim=1))
        quat_loss = (1.0 - quat_dot).mean()

        if gt_confidence is None:
            gt_confidence = torch.ones_like(pred["confidence_logit"])

        confidence_loss = F.binary_cross_entropy_with_logits(
            pred["confidence_logit"],
            gt_confidence.float(),
        )

        total = (
            self.cfg.center_weight * center_loss
            + self.cfg.size_weight * size_loss
            + self.cfg.quat_weight * quat_loss
            + self.cfg.confidence_weight * confidence_loss
        )

        return {
            "loss": total,
            "center_loss": center_loss,
            "size_loss": size_loss,
            "quat_loss": quat_loss,
            "confidence_loss": confidence_loss,
        }