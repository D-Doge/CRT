from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from task_network.frustrum_pointnet.frustum_pointnet_seg import (
    FrustumPointSegNet,
    FrustumSegmentationLoss,
)
from task_network.frustrum_pointnet.center_refinement import (
    CenterRefinementNet,
    CenterRefinementLoss,
)
from task_network.frustrum_pointnet.box_head import (
    FrustumBoxHead,
    FrustumBoxHeadLoss,
)


def masked_object_points(
    points_lidar: torch.Tensor,
    seg_logits: torch.Tensor,
    *,
    num_object_points: int = 512,
    use_gt_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Select object points from frustum.

    points_lidar: (B,N,3)
    seg_logits:   (B,N,2)
    use_gt_mask:  optional (B,N), bool/long

    returns:
        object_points_lidar: (B,num_object_points,3)
    """
    B, N, C = points_lidar.shape
    device = points_lidar.device

    object_points = []

    for b in range(B):
        if use_gt_mask is not None:
            mask = use_gt_mask[b].bool()
            idx = torch.where(mask)[0]
        else:
            prob_obj = torch.softmax(seg_logits[b], dim=-1)[:, 1]
            k = min(num_object_points, N)
            idx = torch.topk(prob_obj, k=k, largest=True).indices

        if idx.numel() == 0:
            # fallback: use highest predicted object probabilities
            prob_obj = torch.softmax(seg_logits[b], dim=-1)[:, 1]
            idx = torch.topk(prob_obj, k=min(num_object_points, N), largest=True).indices

        if idx.numel() >= num_object_points:
            selected = idx[:num_object_points]
        else:
            pad_count = num_object_points - idx.numel()
            pad_idx = idx[torch.randint(0, idx.numel(), (pad_count,), device=device)]
            selected = torch.cat([idx, pad_idx], dim=0)

        object_points.append(points_lidar[b, selected])

    return torch.stack(object_points, dim=0)


class FrustumPointNet(nn.Module):
    """
    Full Frustum PointNet-style model using GT/provided 2D boxes upstream.

    Input:
        points_lidar: (B,N,3)

    Output:
        final 3D boxes in LiDAR frame:
            center_lidar: (B,3)
            size_lidar:   (B,3)
            quat_xyzw:    (B,4)
            confidence:   (B,)

        plus intermediate segmentation/center outputs.
    """

    def __init__(
        self,
        *,
        input_channels: int = 3,
        num_seg_classes: int = 2,
        num_object_points: int = 512,
        use_bn: bool = False,
    ):
        super().__init__()

        self.num_object_points = int(num_object_points)

        self.seg_net = FrustumPointSegNet(
            input_channels=input_channels,
            num_classes=num_seg_classes,
            normalize_points=True,
            use_bn=use_bn,
        )

        self.center_net = CenterRefinementNet(
            input_channels=3,
            normalize_points=True,
            use_bn=use_bn,
        )

        self.box_head = FrustumBoxHead(
            input_channels=3,
            use_bn=use_bn,
        )

    def forward(
        self,
        points_lidar: torch.Tensor,
        *,
        gt_seg_mask: Optional[torch.Tensor] = None,
        use_gt_mask_for_box: bool = False,
    ) -> Dict[str, torch.Tensor]:
        seg_out = self.seg_net(points_lidar)
        seg_logits = seg_out["logits"]

        object_points_lidar = masked_object_points(
            points_lidar[..., :3],
            seg_logits,
            num_object_points=self.num_object_points,
            use_gt_mask=gt_seg_mask if use_gt_mask_for_box else None,
        )

        center_out = self.center_net(object_points_lidar)
        box_out = self.box_head(center_out["centered_points_lidar"])

        center_lidar = (
            center_out["refined_center_lidar"]
            + box_out["center_residual_lidar"]
        )

        return {
            # Final detection output
            "center_lidar": center_lidar,
            "size_lidar": box_out["size_lidar"],
            "quat_xyzw": box_out["quat_xyzw"],
            "confidence": box_out["confidence"],
            "confidence_logit": box_out["confidence_logit"],

            # Segmentation output
            "seg_logits": seg_logits,

            # Intermediate outputs
            "object_points_lidar": object_points_lidar,
            "refined_center_lidar": center_out["refined_center_lidar"],
            "center_residual_lidar": box_out["center_residual_lidar"],
            "centered_points_lidar": center_out["centered_points_lidar"],
            "box_feature": box_out["box_feature"],
        }


@dataclass
class FrustumPointNetLossConfig:
    seg_weight: float = 1.0
    center_weight: float = 1.0
    box_weight: float = 1.0


class FrustumPointNetLoss(nn.Module):
    def __init__(self, cfg: Optional[FrustumPointNetLossConfig] = None):
        super().__init__()
        self.cfg = cfg or FrustumPointNetLossConfig()

        self.seg_loss = FrustumSegmentationLoss()
        self.center_loss = CenterRefinementLoss()
        self.box_loss = FrustumBoxHeadLoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Required batch fields:
            seg_label:        (B,N)
            gt_center_lidar:  (B,3)
            gt_size_lidar:    (B,3)
            gt_quat_xyzw:     (B,4)
        """

        seg_loss = self.seg_loss(
            pred["seg_logits"],
            batch["seg_label"],
        )

        center_loss = self.center_loss(
            pred["refined_center_lidar"],
            batch["gt_center_lidar"],
        )

        gt_center_residual = (
            batch["gt_center_lidar"]
            - pred["refined_center_lidar"].detach()
        )

        box_losses = self.box_loss(
            {
                "center_residual_lidar": pred["center_residual_lidar"],
                "size_lidar": pred["size_lidar"],
                "quat_xyzw": pred["quat_xyzw"],
                "confidence_logit": pred["confidence_logit"],
            },
            gt_center_residual_lidar=gt_center_residual,
            gt_size_lidar=batch["gt_size_lidar"],
            gt_quat_xyzw=batch["gt_quat_xyzw"],
            gt_confidence=torch.ones_like(pred["confidence_logit"]),
        )

        total = (
            self.cfg.seg_weight * seg_loss
            + self.cfg.center_weight * center_loss
            + self.cfg.box_weight * box_losses["loss"]
        )

        return {
            "loss": total,
            "seg_loss": seg_loss,
            "center_loss": center_loss,
            "box_loss": box_losses["loss"],
            "box_center_loss": box_losses["center_loss"],
            "box_size_loss": box_losses["size_loss"],
            "box_quat_loss": box_losses["quat_loss"],
            "box_confidence_loss": box_losses["confidence_loss"],
        }