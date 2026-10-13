import pytest
import torch

from data.data_sample_interface import DataSample
from data.extractor.pointcloud.pointnet2_lidar_extractor import PointNet2LiDARExtractor


@pytest.mark.parametrize(
    "sampling_mode,num_input_points,num_points",
    [
        ("random", 4096, 2048),
        ("fps", 4096, 2048),
        ("none", 4096, None),
    ],
)
def test_pointnet2_lidar_extractor_runs_for_all_sampling_modes(
    sampling_mode: str,
    num_input_points: int,
    num_points: int | None,
):
    torch.manual_seed(0)

    lidar_xyz = torch.rand(num_input_points, 3)

    sample = DataSample(
        setup_id="test_setup",
        frame_id="frame_0001",
        lidar_xyz=lidar_xyz,
    )

    extractor = PointNet2LiDARExtractor(
        device="cpu",
        sampling_mode=sampling_mode,
        num_points=num_points,
    )

    emb = extractor.extract(sample)

    assert torch.is_tensor(emb)
    assert emb.ndim == 1
    assert emb.shape[0] == 1024
    assert emb.dtype == torch.float32


@pytest.mark.parametrize(
    "sampling_mode,num_input_points,num_points,expected_n",
    [
        ("random", 4096, 2048, 2048),
        ("fps", 4096, 2048, 2048),
        ("none", 4096, None, 4096),
    ],
)
def test_pointnet2_lidar_extractor_sampling_shapes(
    sampling_mode: str,
    num_input_points: int,
    num_points: int | None,
    expected_n: int,
):
    torch.manual_seed(0)

    lidar_xyz = torch.rand(num_input_points, 3)

    extractor = PointNet2LiDARExtractor(
        device="cpu",
        sampling_mode=sampling_mode,
        num_points=num_points,
    )

    x = extractor._prepare_lidar_tensor(lidar_xyz)
    x = extractor._apply_sampling(x)

    assert x.shape == (expected_n, 3)


def test_pointnet2_lidar_extractor_rejects_invalid_sampling_mode():
    with pytest.raises(ValueError, match="Invalid sampling_mode"):
        PointNet2LiDARExtractor(
            device="cpu",
            sampling_mode="invalid_mode",
            num_points=2048,
        )


@pytest.mark.parametrize("sampling_mode", ["random", "fps"])
def test_pointnet2_lidar_extractor_requires_num_points_for_sampling_modes(
    sampling_mode: str,
):
    with pytest.raises(ValueError, match="num_points must be set"):
        PointNet2LiDARExtractor(
            device="cpu",
            sampling_mode=sampling_mode,
            num_points=None,
        )