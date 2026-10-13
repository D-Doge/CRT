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


class FrustumPointSegNet(nn.Module):
    """
    Simple PointNet-style frustum point segmentation network.

    Input:
        points_lidar: (B, N, C), usually C=3 for xyz in LiDAR frame

    Output:
        logits: (B, N, num_classes), usually num_classes=2
                class 0 = background, class 1 = object
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        normalize_points: bool = True,
        use_bn: bool = True,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.normalize_points = normalize_points

        self.local_mlp1 = SharedMLP(
            [input_channels, 64, 64, 64],
            use_bn=use_bn,
        )

        self.local_mlp2 = SharedMLP(
            [64, 128, 1024],
            use_bn=use_bn,
        )

        self.seg_head = SharedMLP(
            [64 + 1024, 512, 256, 128],
            use_bn=use_bn,
        )

        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)

    @staticmethod
    def _normalize_xyz(points: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Normalize xyz per frustum.

        Keeps the input coordinate system LiDAR-based, but removes absolute
        translation/scale so training is easier.

        points: (B, N, C)
        """
        xyz = points[..., :3]
        centroid = xyz.mean(dim=1, keepdim=True)
        xyz_centered = xyz - centroid

        scale = torch.linalg.norm(xyz_centered, dim=-1).amax(dim=1, keepdim=True)
        scale = scale.clamp_min(1e-6).unsqueeze(-1)

        xyz_norm = xyz_centered / scale

        if points.shape[-1] > 3:
            out = torch.cat([xyz_norm, points[..., 3:]], dim=-1)
        else:
            out = xyz_norm

        stats = {
            "centroid_lidar": centroid.detach(),
            "scale": scale.detach(),
        }
        return out, stats

    def forward(self, points_lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        if points_lidar.ndim != 3:
            raise ValueError(
                f"Expected points_lidar with shape (B,N,C), got {tuple(points_lidar.shape)}"
            )
        if points_lidar.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected C={self.input_channels}, got C={points_lidar.shape[-1]}"
            )

        if self.normalize_points:
            points_in, norm_stats = self._normalize_xyz(points_lidar)
        else:
            points_in = points_lidar
            norm_stats = {}

        # (B,N,C) -> (B,C,N)
        x = points_in.transpose(1, 2).contiguous()

        local_feat = self.local_mlp1(x)          # (B,64,N)
        global_feat = self.local_mlp2(local_feat)  # (B,1024,N)
        global_feat = torch.max(global_feat, dim=2, keepdim=True).values  # (B,1024,1)

        n = points_lidar.shape[1]
        global_repeated = global_feat.expand(-1, -1, n)  # (B,1024,N)

        fused = torch.cat([local_feat, global_repeated], dim=1)  # (B,1088,N)

        seg_feat = self.seg_head(fused)
        logits = self.classifier(seg_feat)       # (B,num_classes,N)
        logits = logits.transpose(1, 2).contiguous()  # (B,N,num_classes)

        return {
            "logits": logits,
            **norm_stats,
        }


@dataclass
class SegLossConfig:
    object_weight: float = 2.0
    background_weight: float = 1.0


class FrustumSegmentationLoss(nn.Module):
    def __init__(self, cfg: Optional[SegLossConfig] = None):
        super().__init__()
        self.cfg = cfg or SegLossConfig()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,N,2)
        labels: (B,N), values 0/1
        """
        if logits.ndim != 3:
            raise ValueError(f"Expected logits (B,N,C), got {tuple(logits.shape)}")
        if labels.ndim != 2:
            raise ValueError(f"Expected labels (B,N), got {tuple(labels.shape)}")

        weight = torch.tensor(
            [self.cfg.background_weight, self.cfg.object_weight],
            dtype=logits.dtype,
            device=logits.device,
        )

        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1).long(),
            weight=weight,
        )