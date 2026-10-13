from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from data.data_sample_interface import DataSample
from data.dataset.datasetLoaderInterface import DatasetLoader
from data.extractor.ExtractorInterface import DataSampleFieldExtractor


def _safe_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
    return x.strip("_")


def _tensor_or_none(x):
    if x is None:
        return None
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor or None, got {type(x)}")
    return x.detach().cpu()


class CachedDatasetLoader(DatasetLoader):
    """
    Wraps a raw DatasetLoader and adds cached derived fields.

    Raw loader:
        loads rgb, lidar_xyz, calibration, metadata

    CachedDatasetLoader:
        loads/saves rgb_emb, lidar_emb, joint_emb, residuals, etc.
    """

    def __init__(
        self,
        base_loader: DatasetLoader,
        *,
        cache_root: str | Path,
        extractors: Sequence[DataSampleFieldExtractor],
        enabled: bool = True,
        write_cache: bool = True,
        refresh_cache: bool = False,
        drop_raw_after_populate: bool = True,
        cache_id: Optional[str] = None,
    ) -> None:
        self.base_loader = base_loader
        self.cache_root = Path(cache_root)
        self.extractors = list(extractors)

        self.enabled = bool(enabled)
        self.write_cache = bool(write_cache)
        self.refresh_cache = bool(refresh_cache)
        self.drop_raw_after_populate = bool(drop_raw_after_populate)

        self._cache_id = cache_id or self._make_cache_id()

    @property
    def dataset_id(self) -> str:
        return f"{self.base_loader.dataset_id}_cached"

    @property
    def cache_id(self) -> str:
        return self._cache_id

    def __len__(self) -> int:
        return len(self.base_loader)

    def get_frame_ids(self) -> List[str]:
        return self.base_loader.get_frame_ids()

    def load_sample(self, frame_id: str) -> DataSample:
        sample = self.base_loader.load_sample(frame_id)

        if not self.enabled:
            return self._compute_fields(sample)

        cache_path = self._cache_path(sample)

        if cache_path.is_file() and not self.refresh_cache:
            cached_sample = self._load_cache(sample, cache_path)
            return self._drop_raw(cached_sample) if self.drop_raw_after_populate else cached_sample

        populated = self._compute_fields(sample)

        if self.write_cache:
            self._save_cache(populated, cache_path)

        return self._drop_raw(populated) if self.drop_raw_after_populate else populated

    def load_samples(self, frame_ids: Sequence[str]) -> List[DataSample]:
        return [self.load_sample(fid) for fid in frame_ids]

    # --------------------------------------------------------
    # Cache internals
    # --------------------------------------------------------

    def _make_cache_id(self) -> str:
        extractor_parts = [
            f"{ex.output_field}={ex.extractor_id}"
            for ex in self.extractors
        ]

        loader_config = {
            "base_dataset_id": self.base_loader.dataset_id,
            "setup_id": getattr(self.base_loader, "_setup_id", None),
            "pointcloud_sensor": getattr(self.base_loader, "_pointcloud_sensor", None),
            "rgb_capture_id": getattr(self.base_loader, "_rgb_capture_id", None),
            "lidar_capture_id": getattr(self.base_loader, "_lidar_capture_id", None),
            "relative_transform_name": getattr(self.base_loader, "_relative_transform_name", None),
        }

        payload = {
            "loader_config": loader_config,
            "extractors": extractor_parts,
        }

        raw = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

        readable_parts = [
            f"pc={loader_config.get('pointcloud_sensor')}",
            f"rgb={loader_config.get('rgb_capture_id')}",
            f"lidar={loader_config.get('lidar_capture_id')}",
            *extractor_parts,
        ]

        readable = "__".join(_safe_name(p) for p in readable_parts if p is not None)
        readable = readable[:120]

        return f"{readable}__{digest}"

    def _cache_path(self, sample: DataSample) -> Path:
        safe_setup = _safe_name(sample.setup_id)

        frame_parts = [
            _safe_name(part)
            for part in str(sample.frame_id).replace("\\", "/").split("/")
            if part
        ]

        return (
            self.cache_root
            / self._cache_id
            / safe_setup
            / Path(*frame_parts)
        ).with_suffix(".pt")

    def _compute_fields(self, sample: DataSample) -> DataSample:
        updated = sample

        for extractor in self.extractors:
            value = extractor.extract(updated)
            updated = replace(updated, **{extractor.output_field: value})

        return updated

    def _save_cache(self, sample: DataSample, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "cache_id": self._cache_id,
            "setup_id": sample.setup_id,
            "frame_id": sample.frame_id,
            "sequence_id": sample.sequence_id,
            "timestamp_s": sample.timestamp_s,

            "rgb_emb": _tensor_or_none(sample.rgb_emb),
            "lidar_emb": _tensor_or_none(sample.lidar_emb),
            "joint_emb": _tensor_or_none(sample.joint_emb),
            "residuals": _tensor_or_none(sample.residuals),
            "logits": _tensor_or_none(sample.logits),
            "conf": _tensor_or_none(sample.conf),
        }

        torch.save(payload, path)

    def _load_cache(self, sample: DataSample, path: Path) -> DataSample:
        payload = torch.load(path, map_location="cpu")

        return replace(
            sample,
            rgb_emb=payload.get("rgb_emb"),
            lidar_emb=payload.get("lidar_emb"),
            joint_emb=payload.get("joint_emb"),
            residuals=payload.get("residuals"),
            logits=payload.get("logits"),
            conf=payload.get("conf"),
        )

    def _drop_raw(self, sample: DataSample) -> DataSample:
        return replace(
            sample,
            rgb=None,
            lidar_xyz=None,
            lidar_feat=None,
        )