from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

from data.data_sample_interface import DataSample
from data.extractor.ExtractorInterface import DataSampleFieldExtractor


class PointNet2LiDARExtractor(DataSampleFieldExtractor):
    """
    Extracts a PointNet++ global LiDAR embedding from DataSample.lidar_xyz.

    Expected input:
    - sample.lidar_xyz is present
    - shape (N, 3)
    - dtype can be float32 / float64

    Output:
    - 1D torch.Tensor of shape (1024,) stored as lidar_emb upstream

    Notes
    -----
    This wraps the PointNet++ classification backbone and uses its global feature
    output as the embedding:
        logits, global_feat = model(x)
        embedding = global_feat.squeeze(-1)   # -> (B, 1024)

    Sampling
    --------
    The extractor supports different sampling strategies controlled via `sampling_mode`:

    - "random":
        Randomly subsamples the point cloud to `num_points` if N > num_points.
        Pads by repeating points if N < num_points.
        Fast, but may destroy structural differences (e.g., LiDAR sampling patterns).

    - "fps":
        Uses farthest point sampling (FPS) to select `num_points`.
        Better preserves geometric structure and spatial coverage than random sampling.
        Still enforces a fixed point count.

    - "none":
        Uses the full point cloud without subsampling or padding.
        Preserves true point density and sensor-specific structure.
        May result in variable input sizes and slower processing.

    The extractor:
    - optionally samples or pads points depending on `sampling_mode`
    - converts to shape [1, 3, N] (PointNet++ input format)
    - runs inference only (no gradients)

    Important
    ---------
    The choice of sampling strategy can influence the resulting embeddings and
    therefore the measured domain gap. In particular, random subsampling may
    reduce observable differences between sensors by homogenizing point distributions.
    """

    def __init__(
        self,
        num_points: Optional[int] = 2048,
        sampling_mode: str = "fps",   # NEW
        normal_channel: bool = False,
        num_class: int = 40,
        device: Optional[str] = None,
    ) -> None:

        if sampling_mode not in {"random", "fps", "none"}:
            raise ValueError(f"Invalid sampling_mode: {sampling_mode}")

        if sampling_mode != "none" and num_points is None:
            raise ValueError("num_points must be set when using sampling")

        self._sampling_mode = sampling_mode
        self._num_points = num_points

        base_dir = Path(__file__).resolve().parent
        self._model_repo_dir = base_dir / "Pointnet_Pointnet2_pytorch"
        self._checkpoint_path = (
            base_dir
            / "Pointnet_Pointnet2_pytorch/log/classification/"
            / "pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
        )
        
        self._normal_channel = bool(normal_channel)
        self._num_class = int(num_class)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not self._model_repo_dir.exists():
            raise FileNotFoundError(f"Model repo not found: {self._model_repo_dir}")

        if not self._checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self._checkpoint_path}")

        repo_parent = str(self._model_repo_dir.parent.resolve())
        if repo_parent not in sys.path:
            sys.path.insert(0, repo_parent)

        from Pointnet_Pointnet2_pytorch.models.pointnet2_cls_ssg import get_model

        self.model = get_model(
            num_class=self._num_class,
            normal_channel=self._normal_channel,
        ).to(self._device)

        checkpoint = torch.load(
            str(self._checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @property
    def output_field(self) -> str:
        return "lidar_emb"

    @property
    def extractor_id(self) -> str:
        if self._sampling_mode == "none":
            return "pointnet2_ssg_none"
        return f"pointnet2_ssg_{self._sampling_mode}_np{self._num_points}"

    @torch.no_grad()
    def extract(self, sample: DataSample) -> torch.Tensor:
        if sample.lidar_xyz is None:
            raise ValueError("lidar_xyz is None")

        x = self._prepare_lidar_tensor(sample.lidar_xyz)  # [N, 3]

        x = self._apply_sampling(x)

        x = x.transpose(0, 1).contiguous().unsqueeze(0).to(self._device)

        logits, global_feat = self.model(x)

        feat = global_feat.squeeze(0).squeeze(-1).detach().cpu().flatten()

        return feat

    # -------------------------
    # Sampling logic
    # -------------------------
    def _apply_sampling(self, x: torch.Tensor) -> torch.Tensor:
        if self._sampling_mode == "none":
            return x

        n = x.shape[0]
        num_points = self._num_points

        if n == num_points:
            return x

        if n > num_points:
            if self._sampling_mode == "random":
                idx = torch.randperm(n)[:num_points]
                return x[idx]

            elif self._sampling_mode == "fps":
                idx = self._farthest_point_sampling(x, num_points)
                return x[idx]

        # Padding (same for random/fps)
        pad_count = num_points - n
        pad_idx = torch.randint(low=0, high=n, size=(pad_count,))
        return torch.cat([x, x[pad_idx]], dim=0)

    def _farthest_point_sampling(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Simple FPS implementation (O(N*k))
        x: (N, 3)
        returns indices (k,)
        """
        N = x.shape[0]
        device = x.device

        centroids = torch.zeros(k, dtype=torch.long, device=device)
        distances = torch.full((N,), float("inf"), device=device)

        # deterministic initial point: farthest from centroid
        cloud_center = x.mean(dim=0, keepdim=True)
        init_dist = torch.sum((x - cloud_center) ** 2, dim=1)
        farthest = torch.argmax(init_dist)

        for i in range(k):
            centroids[i] = farthest
            centroid = x[farthest]  # (1,3)

            dist = torch.sum((x - centroid) ** 2, dim=1)
            distances = torch.minimum(distances, dist)

            farthest = torch.argmax(distances)

        return centroids

    # -------------------------
    # Input validation
    # -------------------------
    def _prepare_lidar_tensor(self, lidar_xyz: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(lidar_xyz):
            raise TypeError(f"Expected torch.Tensor, got {type(lidar_xyz)}")

        x = lidar_xyz.detach().cpu()

        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"Expected shape (N,3), got {tuple(x.shape)}")

        if x.shape[0] < 1:
            raise ValueError("Empty point cloud")

        return x.float()