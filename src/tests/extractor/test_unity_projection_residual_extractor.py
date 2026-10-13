import torch

from data.data_sample_interface import DataSample
from data.extractor.residual.unity_projection_residual_extractor import ProjectionResidualExtractor


def test_projection_residual_extractor_runs_and_returns_1d_residuals():
    H, W = 64, 64

    # Simple synthetic image with a strong vertical edge in the middle
    rgb = torch.zeros(3, H, W, dtype=torch.float32)
    rgb[:, :, : W // 2] = 0.0
    rgb[:, :, W // 2 :] = 1.0

    # Points in front of camera
    # Chosen so that projections land roughly across the image width
    lidar_xyz = torch.tensor(
        [
            [-1.0, 0.0, 4.0],
            [-0.5, 0.0, 4.0],
            [ 0.0, 0.0, 4.0],
            [ 0.5, 0.0, 4.0],
            [ 1.0, 0.0, 4.0],
        ],
        dtype=torch.float32,
    )

    fx = fy = 20.0
    cx = W / 2.0
    cy = H / 2.0

    K = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    T_cam_lidar = torch.eye(4, dtype=torch.float32)

    sample = DataSample(
        setup_id="A",
        frame_id="0001",
        rgb=rgb,
        lidar_xyz=lidar_xyz,
        intrinsics=K,
        T_cam_lidar=T_cam_lidar,
        image_size=(H, W),
    )

    extractor = ProjectionResidualExtractor(edge_threshold=0.2)
    residuals = extractor.extract(sample)

    assert torch.is_tensor(residuals)
    assert residuals.ndim == 1
    assert residuals.numel() > 0
    assert residuals.dtype == torch.float32
    assert torch.all(residuals >= 0)


def test_projection_residual_extractor_points_near_edge_have_smaller_residual():
    H, W = 64, 64

    rgb = torch.zeros(3, H, W, dtype=torch.float32)
    rgb[:, :, : W // 2] = 0.0
    rgb[:, :, W // 2 :] = 1.0

    fx = fy = 20.0
    cx = W / 2.0
    cy = H / 2.0

    K = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    T_cam_lidar = torch.eye(4, dtype=torch.float32)

    # One point projects near the image center edge, one far left
    near_edge = torch.tensor([[0.0, 0.0, 4.0]], dtype=torch.float32)
    far_from_edge = torch.tensor([[-2.0, 0.0, 4.0]], dtype=torch.float32)

    sample_near = DataSample(
        setup_id="A",
        frame_id="near",
        rgb=rgb,
        lidar_xyz=near_edge,
        intrinsics=K,
        T_cam_lidar=T_cam_lidar,
        image_size=(H, W),
    )

    sample_far = DataSample(
        setup_id="A",
        frame_id="far",
        rgb=rgb,
        lidar_xyz=far_from_edge,
        intrinsics=K,
        T_cam_lidar=T_cam_lidar,
        image_size=(H, W),
    )

    extractor = ProjectionResidualExtractor(edge_threshold=0.2)

    r_near = extractor.extract(sample_near)
    r_far = extractor.extract(sample_far)

    assert r_near.numel() == 1
    assert r_far.numel() == 1
    assert r_near.item() <= r_far.item()