import pytest
import torch

from data.data_sample_interface import DataSample
from metric.calibration_residual_transport import CalibrationResidualTransport


def make_samples(residuals: torch.Tensor, setup_id: str = "A"):
    samples = []
    for i in range(residuals.shape[0]):
        samples.append(
            DataSample(
                setup_id=setup_id,
                frame_id=f"{setup_id}_{i}",
                residuals=residuals[i].clone(),
            )
        )
    return samples


def test_residual_transport_zero_for_identical_mmd():
    torch.manual_seed(0)
    R = torch.randn(30, 12)

    A = make_samples(R, setup_id="A")
    B = make_samples(R.clone(), setup_id="B")

    metric = CalibrationResidualTransport(backend="mmd")
    res = metric.compute(A, B)

    assert res.score < 1e-6


def test_residual_transport_detects_shift_mmd():
    torch.manual_seed(0)
    R_A = torch.randn(30, 12)
    R_B = R_A + 0.8

    A = make_samples(R_A, setup_id="A")
    B = make_samples(R_B, setup_id="B")

    metric = CalibrationResidualTransport(backend="mmd")
    res = metric.compute(A, B)

    assert res.score > 1e-4


def test_residual_transport_sinkhorn_runs_and_is_symmetric():
    torch.manual_seed(0)
    R_A = torch.randn(12, 6, 2)
    R_B = torch.randn(12, 6, 2) + 0.3

    A = make_samples(R_A, setup_id="A")
    B = make_samples(R_B, setup_id="B")

    metric = CalibrationResidualTransport(backend="sinkhorn", sinkhorn_epsilon=0.5)
    r1 = metric.compute(A, B).score
    r2 = metric.compute(B, A).score

    assert abs(r1 - r2) < 1e-5


def test_residual_transport_accepts_multichannel_residuals():
    torch.manual_seed(0)
    R_A = torch.randn(20, 8, 3)
    R_B = R_A.clone()
    R_B[:, :, 1] += 0.5

    A = make_samples(R_A, setup_id="A")
    B = make_samples(R_B, setup_id="B")

    metric = CalibrationResidualTransport(backend="mmd")
    res = metric.compute(A, B)

    assert res.details["K"] == 3
    assert res.score > 1e-4


def test_residual_transport_channel_mismatch_raises():
    torch.manual_seed(0)
    A = make_samples(torch.randn(10, 6, 2), setup_id="A")
    B = make_samples(torch.randn(10, 6, 3), setup_id="B")

    metric = CalibrationResidualTransport(backend="mmd")

    with pytest.raises(ValueError):
        metric.compute(A, B)
