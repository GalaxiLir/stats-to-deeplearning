import numpy as np
import pytest

from tsunc.metrics import coverage, mase, mean_width, pinball_loss, seasonal_naive_scale


def test_pinball_known_values():
    # tau = 0.9: under-forecast by 1 costs 0.9, over-forecast by 1 costs 0.1
    y = np.array([1.0, 0.0])
    q = np.array([[0.0], [1.0]])
    assert pinball_loss(y, q, [0.9]) == pytest.approx((0.9 + 0.1) / 2)


def test_pinball_median_is_half_the_mae():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    q = rng.normal(size=(200, 1))
    assert pinball_loss(y, q, [0.5]) == pytest.approx(
        0.5 * np.mean(np.abs(y - q[:, 0]))
    )


def test_pinball_averages_over_quantile_levels():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(50, 24))
    q = rng.normal(size=(50, 24, 2))
    both = pinball_loss(y, q, [0.05, 0.95])
    separate = (
        pinball_loss(y, q[..., :1], [0.05]) + pinball_loss(y, q[..., 1:], [0.95])
    ) / 2
    assert both == pytest.approx(separate)


def test_pinball_is_minimised_by_the_true_quantile():
    # Statistical property: the tau-quantile minimises E[rho_tau(Y - c)] over constants c.
    rng = np.random.default_rng(2)
    y = rng.normal(size=20_000)

    def loss(c, tau):
        return pinball_loss(y, np.full((y.size, 1), c), [tau])

    for tau in (0.05, 0.5, 0.95):
        q_star = np.quantile(y, tau)
        assert loss(q_star, tau) < loss(q_star - 0.3, tau)
        assert loss(q_star, tau) < loss(q_star + 0.3, tau)


def test_pinball_validates_inputs():
    y = np.zeros(4)
    with pytest.raises(ValueError, match="shape"):
        pinball_loss(y, np.zeros((4, 3)), [0.5])
    with pytest.raises(ValueError, match="levels in"):
        pinball_loss(y, np.zeros((4, 1)), [1.0])
    with pytest.raises(ValueError, match="NaN"):
        pinball_loss(y, np.full((4, 1), np.nan), [0.5])


def test_mase_is_one_for_the_seasonal_naive_forecast_in_sample():
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(size=300))
    m = 24
    assert mase(y[m:], y[:-m], y_train=y, m=m) == pytest.approx(1.0)


def test_mase_doubles_when_errors_double():
    rng = np.random.default_rng(4)
    y_train = np.cumsum(rng.normal(size=200))
    y_true = rng.normal(size=48)
    one = mase(y_true, y_true + 1.0, y_train, m=24)
    two = mase(y_true, y_true + 2.0, y_train, m=24)
    assert two == pytest.approx(2 * one)


def test_seasonal_naive_scale_skips_gaps_in_training_series():
    rng = np.random.default_rng(5)
    y = np.cumsum(rng.normal(size=120))
    y[[10, 50, 51]] = np.nan
    expected = np.nanmean(np.abs(y[24:] - y[:-24]))
    assert seasonal_naive_scale(y, m=24) == pytest.approx(expected)


def test_mase_rejects_degenerate_training_series():
    with pytest.raises(ValueError, match="undefined"):
        seasonal_naive_scale(np.ones(100), m=24)
    with pytest.raises(ValueError, match="longer"):
        seasonal_naive_scale(np.arange(10.0), m=24)


def test_coverage_includes_bounds_and_works_per_horizon():
    y = np.array([1.0, 2.0, 3.0])
    assert coverage(y, [1.0, 2.5, 0.0], [1.0, 3.0, 2.0]) == pytest.approx(1 / 3)

    y2 = np.array([[0.5, 5.0], [1.5, 1.0]])  # (n=2 origins, H=2 horizons)
    lo = np.zeros((2, 2))
    hi = np.array([[1.0, 2.0], [2.0, 2.0]])
    assert coverage(y2, lo, hi) == pytest.approx(0.75)
    np.testing.assert_allclose(coverage(y2, lo, hi, axis=0), [1.0, 0.5])


def test_width_and_crossing_intervals():
    lo = np.array([0.0, 1.0])
    hi = np.array([2.0, 5.0])
    assert mean_width(lo, hi) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="exceed"):
        mean_width(hi, lo)
    with pytest.raises(ValueError, match="exceed"):
        coverage(np.zeros(2), hi, lo)
