from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any, Literal, List

import torch

# Robust imports depending on how you run modules (python -m, pytest, etc.)
try:
    from metric.metric_interface import Metric, MetricResult
    from data.data_sample_interface import DataSample
except Exception:  # pragma: no cover
    from .metric_interface import Metric, MetricResult
    from ..data.data_sample_interface import DataSample


BackendKind = Literal["sinkhorn", "mmd"]


def _stack_residuals(samples: Sequence[DataSample]) -> torch.Tensor:
    """
    Stack cached residual signals from a list of samples.

    Accepted per-sample shapes:
      - (M,)   -> treated as (M, 1)
      - (M, K) -> K residual channels

    Returns
    -------
    X : torch.Tensor
        Shape (N_total, K), where all residual rows from all samples are concatenated.
    """
    chunks: List[torch.Tensor] = []
    k_ref: Optional[int] = None

    for s in samples:
        x = s.residuals
        if x is None:
            raise ValueError("Missing field 'residuals' in DataSample.")
        if not torch.is_tensor(x):
            raise TypeError(f"Field 'residuals' must be a torch.Tensor, got {type(x)}.")

        x = x.detach()
        if x.ndim == 1:
            x = x[:, None]
        elif x.ndim != 2:
            raise ValueError(
                f"Residuals must have shape (M,) or (M,K), got {tuple(x.shape)}"
            )

        if x.shape[0] == 0:
            raise ValueError("Residual tensor must contain at least one residual value.")

        if k_ref is None:
            k_ref = int(x.shape[1])
        elif int(x.shape[1]) != k_ref:
            raise ValueError(
                f"Residual channel dims differ across samples: expected K={k_ref}, got K={x.shape[1]}"
            )

        chunks.append(x)

    return torch.cat(chunks, dim=0)


def _rbf_kernel_from_sq_dists(sq_dists: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    return torch.exp(-sq_dists / (2.0 * sigma2))


def _median_heuristic_sigma2(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    max_pairs: int = 20000,
    eps: float = 1e-12,
) -> torch.Tensor:
    Z = torch.cat([X, Y], dim=0)
    n = Z.shape[0]
    k = min(n, int((max_pairs ** 0.5) + 1))
    Zs = Z[:k]

    sq = torch.cdist(Zs, Zs, p=2.0) ** 2
    sq = sq[~torch.eye(sq.shape[0], dtype=torch.bool, device=sq.device)]
    if sq.numel() == 0:
        return torch.tensor(1.0, device=Z.device, dtype=Z.dtype)

    med = torch.median(sq)
    sigma2 = torch.clamp(med / 2.0, min=eps)
    return sigma2


def _mmd2_unbiased_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    sigma2: Optional[float] = None,
    max_pairs: int = 20000,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    n = X.shape[0]
    m = Y.shape[0]
    if n < 2 or m < 2:
        raise ValueError("Need at least 2 residual samples in each domain for unbiased MMD.")

    sq_xx = torch.cdist(X, X, p=2.0) ** 2
    sq_yy = torch.cdist(Y, Y, p=2.0) ** 2
    sq_xy = torch.cdist(X, Y, p=2.0) ** 2

    sigma2_t = (
        torch.tensor(float(sigma2), dtype=X.dtype, device=X.device)
        if sigma2 is not None
        else _median_heuristic_sigma2(X, Y, max_pairs=max_pairs)
    )

    K_xx = _rbf_kernel_from_sq_dists(sq_xx, sigma2_t)
    K_yy = _rbf_kernel_from_sq_dists(sq_yy, sigma2_t)
    K_xy = _rbf_kernel_from_sq_dists(sq_xy, sigma2_t)

    K_xx = K_xx - torch.diag(torch.diag(K_xx))
    K_yy = K_yy - torch.diag(torch.diag(K_yy))

    mmd2 = K_xx.sum() / (n * (n - 1)) + K_yy.sum() / (m * (m - 1)) - 2.0 * K_xy.mean()
    mmd2 = torch.clamp(mmd2, min=0.0)

    info = {
        "sigma2": float(sigma2_t.item()),
        "mmd2": float(mmd2.item()),
        "N_A_residuals": int(n),
        "N_B_residuals": int(m),
    }
    return mmd2, info


def _uniform_weights(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    return torch.full((n,), 1.0 / n, device=device, dtype=dtype)


def _cost_matrix_sqeuclidean(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(X, Y, p=2.0) ** 2


def _sinkhorn_ot_cost(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    *,
    epsilon: float,
    max_iters: int,
    tol: float,
    stable: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be > 0, got {max_iters}")
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")

    exponent = -C / float(epsilon)
    if stable:
        exponent = torch.clamp(exponent, min=-80.0, max=0.0)
    K = torch.exp(exponent)

    tiny = torch.finfo(C.dtype).tiny
    K = torch.clamp(K, min=tiny)

    u = torch.ones_like(a)
    v = torch.ones_like(b)
    err = None
    it_used = 0

    for it in range(max_iters):
        u_prev = u
        Kv = torch.clamp(K @ v, min=tiny)
        u = a / Kv
        KTu = torch.clamp(K.transpose(0, 1) @ u, min=tiny)
        v = b / KTu

        err = torch.max(torch.abs(u - u_prev)).item()
        it_used = it + 1
        if err < tol:
            break

    P = u[:, None] * K * v[None, :]
    cost = torch.sum(P * C)
    info = {
        "iterations": int(it_used),
        "converged": bool(err is not None and err < tol),
        "final_err": float(err if err is not None else 0.0),
        "transport_mass": float(P.sum().item()),
    }
    return cost, info


def _sinkhorn_divergence(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    epsilon: float,
    max_iters: int,
    tol: float,
    stable: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    n = X.shape[0]
    m = Y.shape[0]
    if n == 0 or m == 0:
        raise ValueError("Need non-empty residual samples in both domains.")

    a = _uniform_weights(n, device=X.device, dtype=X.dtype)
    b = _uniform_weights(m, device=Y.device, dtype=Y.dtype)

    C_xy = _cost_matrix_sqeuclidean(X, Y)
    C_xx = _cost_matrix_sqeuclidean(X, X)
    C_yy = _cost_matrix_sqeuclidean(Y, Y)

    ot_xy, info_xy = _sinkhorn_ot_cost(a, b, C_xy, epsilon=epsilon, max_iters=max_iters, tol=tol, stable=stable)
    ot_xx, info_xx = _sinkhorn_ot_cost(a, a, C_xx, epsilon=epsilon, max_iters=max_iters, tol=tol, stable=stable)
    ot_yy, info_yy = _sinkhorn_ot_cost(b, b, C_yy, epsilon=epsilon, max_iters=max_iters, tol=tol, stable=stable)

    sink = ot_xy - 0.5 * ot_xx - 0.5 * ot_yy
    sink = torch.clamp(sink, min=0.0)

    info = {
        "ot_xy": float(ot_xy.item()),
        "ot_xx": float(ot_xx.item()),
        "ot_yy": float(ot_yy.item()),
        "sinkhorn": float(sink.item()),
        "xy_info": info_xy,
        "xx_info": info_xx,
        "yy_info": info_yy,
    }
    return sink, info


@dataclass
class CalibrationResidualTransport(Metric):
    """
    Calibration-aware residual transport on cached residual signals.

    This is a generic first implementation of the Category E residual metric:
    it assumes residuals have already been computed upstream and cached in
    DataSample.residuals. The metric then compares the residual distributions
    between setup A and B via either Sinkhorn divergence or RBF-MMD.

    Accepted per-sample residual shapes:
      - (M,)   : one residual channel
      - (M, K) : K residual channels

    The metric concatenates all residual rows from all samples in each setup,
    so the comparison is performed over the empirical residual point clouds.
    Compares residual distributions between setups using a transport-based distance (MMD or Sinkhorn backend).
    """

    backend: BackendKind = "sinkhorn"

    # Sinkhorn params
    sinkhorn_epsilon: float = 0.1
    sinkhorn_max_iters: int = 200
    sinkhorn_tol: float = 1e-6
    sinkhorn_stable: bool = True

    # MMD params
    mmd_sigma2: Optional[float] = None
    mmd_max_pairs: int = 20000
    return_sqrt: bool = False

    # Common
    normalize_by_num_channels: bool = False

    name: str = "Calibration-Residual-Transport"
    required_fields: Sequence[str] = ("residuals",)
    requires_paired: bool = False

    def compute(
        self,
        A: Sequence[DataSample],
        B: Sequence[DataSample],
        pairing: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> MetricResult:
        X_A = _stack_residuals(A)
        X_B = _stack_residuals(B)

        if X_A.shape[1] != X_B.shape[1]:
            raise ValueError(
                f"{self.name}: residual dims differ: A has K={X_A.shape[1]}, B has K={X_B.shape[1]}"
            )

        backend = self.backend.lower()
        if backend == "sinkhorn":
            score_t, info = _sinkhorn_divergence(
                X_A,
                X_B,
                epsilon=float(self.sinkhorn_epsilon),
                max_iters=int(self.sinkhorn_max_iters),
                tol=float(self.sinkhorn_tol),
                stable=bool(self.sinkhorn_stable),
            )
        elif backend == "mmd":
            score_t, info = _mmd2_unbiased_rbf(
                X_A,
                X_B,
                sigma2=self.mmd_sigma2,
                max_pairs=int(self.mmd_max_pairs),
            )
            if self.return_sqrt:
                score_t = torch.sqrt(torch.clamp(score_t, min=0.0))
                info["mmd"] = float(score_t.item())
        else:
            raise ValueError(f"{self.name}: unknown backend '{self.backend}'. Use 'sinkhorn' or 'mmd'.")

        if self.normalize_by_num_channels:
            score_t = score_t / max(X_A.shape[1], 1)

        details: Dict[str, Any] = {
            "backend": backend,
            "N_A_samples": int(len(A)),
            "N_B_samples": int(len(B)),
            "N_A_residuals": int(X_A.shape[0]),
            "N_B_residuals": int(X_B.shape[0]),
            "K": int(X_A.shape[1]),
            "normalize_by_num_channels": bool(self.normalize_by_num_channels),
            "return_sqrt": bool(self.return_sqrt),
            **info,
        }
        return MetricResult(score=float(score_t.item()), details=details)
