from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from data.data_sample_interface import DataSample
from data.extractor.ExtractorInterface import DataSampleFieldExtractor


class ConcatJointEmbeddingExtractor(DataSampleFieldExtractor):
    """
    Builds a joint embedding by concatenating cached rgb_emb and lidar_emb.

    This is a lightweight derived-field extractor:
    - input: sample.rgb_emb and sample.lidar_emb
    - output: joint_emb

    Optional behavior:
    - L2-normalize each modality embedding before concatenation
    """

    def __init__(self, l2_normalize_each: bool = False) -> None:
        self._l2_normalize_each = bool(l2_normalize_each)

    @property
    def output_field(self) -> str:
        return "joint_emb"

    @property
    def extractor_id(self) -> str:
        if self._l2_normalize_each:
            return "joint_concat_l2norm_v1"
        return "joint_concat_v1"

    @torch.no_grad()
    def extract(self, sample: DataSample) -> torch.Tensor:
        if sample.rgb_emb is None:
            raise ValueError(
                f"ConcatJointEmbeddingExtractor: sample.rgb_emb is None for "
                f"setup_id={sample.setup_id}, frame_id={sample.frame_id}"
            )
        if sample.lidar_emb is None:
            raise ValueError(
                f"ConcatJointEmbeddingExtractor: sample.lidar_emb is None for "
                f"setup_id={sample.setup_id}, frame_id={sample.frame_id}"
            )

        if not torch.is_tensor(sample.rgb_emb):
            raise TypeError(
                f"ConcatJointEmbeddingExtractor: rgb_emb must be torch.Tensor, "
                f"got {type(sample.rgb_emb)}"
            )
        if not torch.is_tensor(sample.lidar_emb):
            raise TypeError(
                f"ConcatJointEmbeddingExtractor: lidar_emb must be torch.Tensor, "
                f"got {type(sample.lidar_emb)}"
            )

        rgb = sample.rgb_emb.detach().cpu().flatten().float()
        lidar = sample.lidar_emb.detach().cpu().flatten().float()

        if rgb.numel() == 0:
            raise ValueError("ConcatJointEmbeddingExtractor: rgb_emb is empty")
        if lidar.numel() == 0:
            raise ValueError("ConcatJointEmbeddingExtractor: lidar_emb is empty")

        if self._l2_normalize_each:
            rgb = F.normalize(rgb, p=2, dim=0)
            lidar = F.normalize(lidar, p=2, dim=0)

        joint = torch.cat([rgb, lidar], dim=0)
        return joint