from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from data.data_sample_interface import DataSample


class DatasetLoader(ABC):
    """
    Generic interface for loading raw DataSample objects from a dataset.

    Responsibilities
    ----------------
    - Understand dataset-specific file structure and metadata
    - Load raw sensor data and calibration
    - Return DataSample objects

    Does NOT handle
    ----------------
    - embedding extraction
    - residual extraction
    - metric computation
    - caching of derived fields
    """

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """
        Unique dataset identifier.

        Examples:
        - "sim_unity_v1"
        - "kitti_raw"
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Number of accessible samples / frames.
        """
        pass

    @abstractmethod
    def get_frame_ids(self) -> List[str]:
        """
        Return all available frame IDs in deterministic order.
        """
        pass

    @abstractmethod
    def load_sample(self, frame_id: str) -> DataSample:
        """
        Load one sample by frame_id and return it as a DataSample.

        This should populate only raw / geometric fields that are naturally
        available from the dataset, for example:
        - setup_id
        - frame_id
        - timestamp_s
        - sequence_id
        - rgb
        - lidar_xyz
        - intrinsics
        - T_cam_lidar
        - image_size
        """
        pass

    def load_samples(self, frame_ids: Sequence[str]) -> List[DataSample]:
        """
        Convenience helper for loading multiple samples.
        """
        return [self.load_sample(fid) for fid in frame_ids]