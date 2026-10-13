import pytest
import torch

from data.data_sample_interface import DataSample
from data.extractor.joint.concat_joint_embedding_extractor import (
    ConcatJointEmbeddingExtractor,
)


def test_concat_joint_embedding_extractor_concatenates_embeddings():
    sample = DataSample(
        setup_id="test_setup",
        frame_id="frame_0001",
        rgb_emb=torch.tensor([1.0, 2.0, 3.0]),
        lidar_emb=torch.tensor([4.0, 5.0]),
    )

    extractor = ConcatJointEmbeddingExtractor(l2_normalize_each=False)
    joint = extractor.extract(sample)

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    assert extractor.output_field == "joint_emb"
    assert extractor.extractor_id == "joint_concat_v1"
    assert torch.is_tensor(joint)
    assert joint.ndim == 1
    assert torch.equal(joint, expected)


def test_concat_joint_embedding_extractor_flattens_inputs():
    sample = DataSample(
        setup_id="test_setup",
        frame_id="frame_0002",
        rgb_emb=torch.tensor([[1.0, 2.0, 3.0]]),   # shape (1, 3)
        lidar_emb=torch.tensor([[4.0], [5.0]]),    # shape (2, 1)
    )

    extractor = ConcatJointEmbeddingExtractor(l2_normalize_each=False)
    joint = extractor.extract(sample)

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    assert joint.shape == (5,)
    assert torch.equal(joint, expected)


def test_concat_joint_embedding_extractor_l2_normalizes_each_part():
    sample = DataSample(
        setup_id="test_setup",
        frame_id="frame_0003",
        rgb_emb=torch.tensor([3.0, 4.0]),   # norm = 5
        lidar_emb=torch.tensor([0.0, 2.0]), # norm = 2
    )

    extractor = ConcatJointEmbeddingExtractor(l2_normalize_each=True)
    joint = extractor.extract(sample)

    expected = torch.tensor([0.6, 0.8, 0.0, 1.0])

    assert extractor.extractor_id == "joint_concat_l2norm_v1"
    assert torch.allclose(joint, expected, atol=1e-6)


def test_concat_joint_embedding_extractor_raises_if_rgb_missing():
    sample = DataSample(
        setup_id="test_setup",
        frame_id="frame_0004",
        lidar_emb=torch.tensor([1.0, 2.0]),
    )

    extractor = ConcatJointEmbeddingExtractor()

    with pytest.raises(ValueError, match="rgb_emb is None"):
        extractor.extract(sample)


def test_concat_joint_embedding_extractor_raises_if_lidar_missing():
    sample = DataSample(
        setup_id="test_setup",
        frame_id="frame_0005",
        rgb_emb=torch.tensor([1.0, 2.0]),
    )

    extractor = ConcatJointEmbeddingExtractor()

    with pytest.raises(ValueError, match="lidar_emb is None"):
        extractor.extract(sample)


def test_concat_joint_embedding_extractor_raises_on_empty_embedding():
    sample = DataSample(
        setup_id="test_setup",
        frame_id="frame_0006",
        rgb_emb=torch.tensor([]),
        lidar_emb=torch.tensor([1.0, 2.0]),
    )

    extractor = ConcatJointEmbeddingExtractor()

    with pytest.raises(ValueError, match="rgb_emb is empty"):
        extractor.extract(sample)