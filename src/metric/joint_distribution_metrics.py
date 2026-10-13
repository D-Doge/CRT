"""Additional joint baselines compatible with the existing Metric interface."""
from dataclasses import dataclass
from typing import Sequence
import torch
from metric.metric_interface import Metric, MetricResult
from metric.distribution_core import frechet_distance, polynomial_mmd_subsets


def _features(samples, field):
    features = []
    for sample in samples:
        value = getattr(sample, field, None)
        if not torch.is_tensor(value):
            raise ValueError(f'{field} must contain a cached torch.Tensor.')
        features.append(value.detach().to(device='cpu', dtype=torch.float64).flatten().numpy())
    import numpy as np
    return np.stack(features)


@dataclass
class JointFrechetFloat64(Metric):
    name: str = 'JOINT-FRECHET-FLOAT64'
    required_fields: Sequence[str] = ('joint_emb',)
    requires_paired: bool = False

    def compute(self, A, B, pairing=None):
        score, details = frechet_distance(_features(A, 'joint_emb'), _features(B, 'joint_emb'))
        details.update(field='joint_emb', return_sqrt=False)
        return MetricResult(score=score, details=details)


@dataclass
class JointPolynomialMMD(Metric):
    subset_size: int = 1000
    subsets: int = 100
    seed: int = 0
    name: str = 'JOINT-POLYNOMIAL-MMD'
    required_fields: Sequence[str] = ('joint_emb',)
    requires_paired: bool = False

    def compute(self, A, B, pairing=None):
        score, details = polynomial_mmd_subsets(_features(A, 'joint_emb'),
            _features(B, 'joint_emb'), self.subset_size, self.subsets, self.seed)
        details.update(field='joint_emb', return_sqrt=False)
        return MetricResult(score=score, details=details)
