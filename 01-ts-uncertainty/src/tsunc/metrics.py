"""Evaluation metrics for probabilistic forecasts of a time series.

Shape conventions: n forecast origins, horizon H, K quantile levels.
``y_true`` has shape (n, H) (or (n,)); a quantile forecast has shape ``y_true.shape + (K,)``.
All inputs must be finite, except ``y_train`` in the MASE, which may contain gaps (NaN).
"""

import numpy as np

__all__ = ["coverage", "mase", "mean_width", "pinball_loss", "seasonal_naive_scale"]


def _finite(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def pinball_loss(y_true, y_pred, quantiles) -> float:
    """Mean pinball loss, averaged over observations and quantile levels.

    rho_tau(y, q) = max(tau * (y - q), (tau - 1) * (y - q)).
    Its expectation is minimised by the tau-quantile of y: it is the loss of quantile
    regression (the "check function" of ``quantreg::rq`` in R).
    """
    y = _finite(y_true, "y_true")
    q = _finite(y_pred, "y_pred")
    taus = np.asarray(quantiles, dtype=float)
    if taus.ndim != 1 or taus.size == 0 or np.any((taus <= 0) | (taus >= 1)):
        raise ValueError(
            "quantiles must be a non-empty 1-D sequence of levels in (0, 1)"
        )
    expected = y.shape + (taus.size,)
    if q.shape != expected:
        raise ValueError(f"y_pred must have shape {expected}, got {q.shape}")
    diff = y[..., None] - q  # broadcasting: taus (K,) is aligned with the last axis
    return float(np.mean(np.maximum(taus * diff, (taus - 1.0) * diff)))


def seasonal_naive_scale(y_train, m: int = 24) -> float:
    """Mean absolute error of the seasonal naive forecast (lag m) on the training series."""
    y = np.asarray(y_train, dtype=float)
    if y.ndim != 1 or y.size <= m:
        raise ValueError(f"y_train must be 1-D and longer than the period m={m}")
    errors = np.abs(y[m:] - y[:-m])
    errors = errors[~np.isnan(errors)]  # gaps in the training series are skipped
    if errors.size == 0:
        raise ValueError("y_train has no pair of observations m steps apart")
    scale = float(errors.mean())
    if scale == 0.0:
        raise ValueError(
            "the seasonal naive forecast is perfect on y_train: MASE is undefined"
        )
    return scale


def mase(y_true, y_pred, y_train, m: int = 24) -> float:
    """Mean absolute scaled error: MAE divided by the in-sample MAE of the seasonal naive."""
    y = _finite(y_true, "y_true")
    p = _finite(y_pred, "y_pred")
    if y.shape != p.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y.shape} and {p.shape}"
        )
    return float(np.mean(np.abs(y - p)) / seasonal_naive_scale(y_train, m))


def _interval(y_true, lower, upper):
    y = _finite(y_true, "y_true")
    lo = _finite(lower, "lower")
    hi = _finite(upper, "upper")
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError(
            f"y_true, lower and upper must have the same shape, got {y.shape}, {lo.shape}, {hi.shape}"
        )
    if np.any(lo > hi):
        raise ValueError(
            "lower must not exceed upper (fix crossing intervals upstream)"
        )
    return y, lo, hi


def coverage(y_true, lower, upper, axis=None):
    """Share of observations inside [lower, upper] (bounds included).

    ``axis=None`` gives the overall coverage; ``axis=0`` on (n, H) arrays gives the coverage
    at each horizon h, which is not the joint coverage of the whole 24 h trajectory.
    """
    y, lo, hi = _interval(y_true, lower, upper)
    result = ((y >= lo) & (y <= hi)).mean(axis=axis)
    return float(result) if axis is None else result


def mean_width(lower, upper, axis=None):
    """Average interval width (upper - lower), overall or along ``axis``."""
    lo = _finite(lower, "lower")
    hi = _finite(upper, "upper")
    if lo.shape != hi.shape:
        raise ValueError(
            f"lower and upper must have the same shape, got {lo.shape} and {hi.shape}"
        )
    if np.any(lo > hi):
        raise ValueError(
            "lower must not exceed upper (fix crossing intervals upstream)"
        )
    result = (hi - lo).mean(axis=axis)
    return float(result) if axis is None else result
