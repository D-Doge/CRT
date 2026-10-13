"""Float64 distribution distances; no feature extraction or file I/O."""
import numpy as np
from scipy.linalg import svdvals


def matrices(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] == 0:
        raise ValueError('Expected nonempty (N,D), (M,D) matrices with equal D.')
    if min(len(a), len(b)) < 2:
        raise ValueError('At least two samples per domain are required.')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Features must be finite.')
    return a, b


def frechet_distance(a, b):
    """Squared Gaussian Wasserstein distance with unbiased sample covariances.

    For covariance factors L_A,L_B, the square-root trace equals the
    nuclear norm of L_A.T @ L_B. This avoids constructing a D x D covariance
    when N << D and handles singular covariances without regularization.
    """
    a, b = matrices(a, b)
    ma, mb = a.mean(0), b.mean(0)
    la = (a - ma) / np.sqrt(len(a) - 1)
    lb = (b - mb) / np.sqrt(len(b) - 1)
    mean_term = float(np.sum((ma - mb) ** 2))
    ta, tb = float(np.sum(la ** 2)), float(np.sum(lb ** 2))
    if max(len(a), len(b)) <= a.shape[1]:
        cross = la @ lb.T
    else:
        # QR compresses covariance factors to at most D columns.
        ra = np.linalg.qr(la, mode='reduced')[1]
        rb = np.linalg.qr(lb, mode='reduced')[1]
        cross = ra @ rb.T
    root_trace = float(svdvals(cross).sum())
    covariance_raw = ta + tb - 2 * root_trace
    tolerance = 1e-10 * max(1.0, ta + tb)
    if covariance_raw < -tolerance:
        raise ArithmeticError('Materially negative covariance distance.')
    score = mean_term + max(0.0, covariance_raw)
    return score, dict(mean_term=mean_term, covariance_term_raw=covariance_raw,
                       trace_C_A=ta, trace_C_B=tb, sqrt_product_trace=root_trace,
                       numerical_tolerance=tolerance, N_A=len(a), N_B=len(b),
                       D=a.shape[1], dtype='float64', covariance_regularization=0.0)


def polynomial_mmd(a, b):
    """Unbiased MMD squared: k(x,y) = (x.T y / D + 1)^3.

    Negative estimates are valid and intentionally preserved.
    """
    a, b = matrices(a, b)
    n, m, d = len(a), len(b), a.shape[1]
    with np.errstate(over='raise', invalid='raise'):
        aa = (a @ a.T / d + 1) ** 3
        bb = (b @ b.T / d + 1) ** 3
        ab = (a @ b.T / d + 1) ** 3
        score = ((aa.sum() - np.trace(aa)) / (n * (n - 1))
                 + (bb.sum() - np.trace(bb)) / (m * (m - 1)) - 2 * ab.mean())
    if not np.isfinite(score):
        raise ArithmeticError('Non-finite polynomial MMD.')
    return float(score)


def polynomial_mmd_subsets(a, b, subset_size=1000, subsets=100, seed=0):
    a, b = matrices(a, b)
    if not isinstance(subset_size, int) or subset_size < 2:
        raise ValueError('subset_size must be an integer >= 2.')
    if not isinstance(subsets, int) or subsets < 1:
        raise ValueError('subsets must be an integer >= 1.')
    size = min(subset_size, len(a), len(b))
    rng = np.random.RandomState(seed)
    values = np.array([polynomial_mmd(a[rng.choice(len(a), size, replace=False)],
                                    b[rng.choice(len(b), size, replace=False)])
                       for _ in range(subsets)])
    return float(values.mean()), dict(std=float(values.std(ddof=0)),
        subset_size_requested=subset_size, subset_size_used=size, subsets=subsets,
        seed=seed, subset_scores=values.tolist(), N_A=len(a), N_B=len(b),
        D=a.shape[1], degree=3, gamma=1 / a.shape[1], coef0=1.0,
        estimator='unbiased_mmd_squared', clamped=False, dtype='float64')
