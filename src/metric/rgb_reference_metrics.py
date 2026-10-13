"""Reference RGB evaluation upstream of the cached-representation Metric API.

This helper performs image feature extraction and may download Inception
weights. It deliberately is NOT a Metric subclass (those must not do I/O or
extract features). Existing DINOv2 rgb_emb/joint_emb fields are untouched.
"""
from dataclasses import dataclass
from importlib.metadata import version
from typing import Optional
import torch
from torch.utils.data import Dataset
from metric.metric_interface import MetricResult


def rgb_uint8_chw(image, input_range='uint8'):
    """Explicit input contract; no resizing, cropping or ImageNet normalization.

    Float images require input_range='0_1' or '0_255'. Conversion to uint8
    rounds to the nearest integer and records this policy in result metadata.
    """
    from PIL import Image
    import numpy as np
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2, 0, 1).contiguous()
    if not torch.is_tensor(image):
        raise TypeError('RGB must be a PIL image or torch.Tensor.')
    x = image.detach().cpu()
    if x.ndim != 3:
        raise ValueError('Expected RGB CHW or HWC.')
    if x.shape[0] != 3:
        if x.shape[-1] != 3:
            raise ValueError('Expected three RGB channels.')
        x = x.permute(2, 0, 1)
    if input_range not in ('uint8', '0_1', '0_255'):
        raise ValueError('Unknown input_range.')
    if x.dtype == torch.uint8:
        return x.contiguous()
    if input_range == 'uint8':
        raise ValueError('Float RGB requires explicit input_range: 0_1 or 0_255.')
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('Expected finite floating-point RGB.')
    upper = 1.0 if input_range == '0_1' else 255.0
    if x.numel() == 0 or x.min() < 0 or x.max() > upper:
        raise ValueError(f'RGB outside [0,{upper}].')
    return (x.double() * (255 / upper)).round().to(torch.uint8).contiguous()


class _RGBDataset(Dataset):
    def __init__(self, samples, input_range):
        self.samples, self.input_range = samples, input_range

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rgb = self.samples[index].rgb
        if rgb is None:
            raise ValueError('Raw RGB is required; embedding-only caches are insufficient.')
        return rgb_uint8_chw(rgb, self.input_range)


@dataclass(frozen=True)
class RGBReferenceConfig:
    input_range: str = 'uint8'
    cuda: bool = False
    batch_size: int = 1  # supports different native image sizes
    kid_subset_size: int = 1000
    kid_subsets: int = 100
    seed: int = 0
    weights_path: Optional[str] = None


def compute_reference_rgb_metrics(A, B, config=None):
    """Return {'RGB-FID': MetricResult, 'RGB-KID': MetricResult}.

    Delegates Inception preprocessing, features and metric arithmetic to
    torch-fidelity. Pass ordered sequences of DataSample with original raw RGB.
    Reference preprocessing is internal to the dependency, not supplied here.
    """
    config = config or RGBReferenceConfig()
    if min(len(A), len(B)) < 2:
        raise ValueError('At least two RGB samples per domain are required.')
    if config.kid_subset_size < 2 or config.kid_subsets < 1 or config.batch_size < 1:
        raise ValueError('Invalid subset/batch configuration.')
    if config.input_range not in ('uint8', '0_1', '0_255'):
        raise ValueError('Unknown input_range.')
    if config.cuda and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable.')
    import torch_fidelity
    subset_size = min(config.kid_subset_size, len(A), len(B))
    kwargs = dict(input1=_RGBDataset(A, config.input_range),
        input2=_RGBDataset(B, config.input_range), cuda=config.cuda,
        batch_size=config.batch_size, fid=True, kid=True, isc=False, ppl=False,
        feature_extractor='inception-v3-compat', feature_layer_fid='2048',
        feature_layer_kid='2048', kid_subset_size=subset_size,
        kid_subsets=config.kid_subsets, kid_kernel='poly',
        kid_kernel_poly_degree=3, kid_kernel_poly_gamma=None,
        kid_kernel_poly_coef0=1.0, rng_seed=config.seed, verbose=False,
        save_cpu_ram=True)
    if config.weights_path is not None:
        kwargs['feature_extractor_weights_path'] = config.weights_path
    result = torch_fidelity.calculate_metrics(**kwargs)
    metadata = dict(N_A=len(A), N_B=len(B), D=2048,
        backend='torch-fidelity', backend_version=version('torch-fidelity'),
        feature_extractor='inception-v3-compat', input_range=config.input_range,
        float_to_uint8='round_to_nearest', seed=config.seed,
        weights_path=config.weights_path, cuda=config.cuda,
        batch_size=config.batch_size)
    fid = MetricResult(score=float(result['frechet_inception_distance']),
                       details=dict(metadata), n_A=len(A), n_B=len(B))
    kid_details = dict(metadata, std=float(result['kernel_inception_distance_std']),
        subset_size_requested=config.kid_subset_size, subset_size_used=subset_size,
        subsets=config.kid_subsets, estimator='unbiased_mmd_squared',
        clamped=False, return_sqrt=False)
    kid = MetricResult(score=float(result['kernel_inception_distance_mean']),
                       details=kid_details, n_A=len(A), n_B=len(B))
    return {'RGB-FID': fid, 'RGB-KID': kid}
