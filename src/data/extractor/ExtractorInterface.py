from abc import ABC, abstractmethod
import torch
try:
    from data.data_sample_interface import DataSample
except Exception:  # pragma: no cover
    from ..data_sample_interface import DataSample


class DataSampleFieldExtractor(ABC):
    """
    Generic interface for extracting a single field of a DataSample.

    Responsibilities:
    - Defines which field it produces (e.g., 'rgb_emb', 'logits', ...)
    - Computes that field from a DataSample

    Does NOT handle:
    - caching
    - batching
    - dataset logic
    """

    @property
    @abstractmethod
    def output_field(self) -> str:
        """
        Name of the DataSample attribute this extractor produces.
        Example: 'rgb_emb', 'lidar_emb', 'logits', ...
        """
        pass

    @property
    @abstractmethod
    def extractor_id(self) -> str:
        """
        Unique identifier for caching/versioning.

        Should include:
        - model name
        - config/version if relevant

        Example:
        - 'dinov2_vits14'
        - 'pointnet2_ssg_v1'
        - 'detector_x_v3'
        """
        pass

    @abstractmethod
    def extract(self, sample: DataSample) -> torch.Tensor:
        """
        Compute the field value for a single DataSample.

        Args:
            sample: DataSample

        Returns:
            torch.Tensor corresponding to output_field
        """
        pass