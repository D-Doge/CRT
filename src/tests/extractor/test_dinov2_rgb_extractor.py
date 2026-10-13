import torch

from data.data_sample_interface import DataSample
from data.extractor.rgb.dinov2_rgb_extractor import DINOv2RGBExtractor


def test_dinov2_rgb_extractor_runs_on_random_tensor():
    # Random RGB image in CHW format
    rgb = torch.rand(3, 256, 256)

    sample = DataSample(
        setup_id="test_setup",
        frame_id="frame_0001",
        rgb=rgb,
    )

    extractor = DINOv2RGBExtractor(device="cpu")
    emb = extractor.extract(sample)

    assert isinstance(emb, torch.Tensor)
    assert emb.ndim == 1
    assert emb.numel() > 0
    assert torch.isfinite(emb).all()