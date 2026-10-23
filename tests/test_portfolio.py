import numpy as np
import pandas as pd
import pytest

from marketlens import portfolio as pf


@pytest.fixture
def two_assets():
    # Uncorrelated assets with vols 10% and 20% (annual), returns 5% and 15%.
    mean = pd.Series([0.05, 0.15], index=["LOW", "HIGH"]) / 252
    cov = pd.DataFrame(np.diag([0.10**2, 0.20**2]) / 252, index=mean.index, columns=mean.index)
    return mean, cov


def test_min_variance_matches_closed_form(two_assets):
    mean, cov = two_assets
    weights, stats = pf.optimize_minimum_variance(mean, cov)
    # For uncorrelated assets w_i is proportional to 1 / sigma_i^2 -> (0.8, 0.2).
    np.testing.assert_allclose(weights, [0.8, 0.2], atol=1e-4)
    assert stats["Volatility"] == pytest.approx(np.sqrt(0.8**2 * 0.01 + 0.2**2 * 0.04), rel=1e-4)


def test_max_sharpe_matches_closed_form(two_assets):
    mean, cov = two_assets
    rf = 0.02
    weights, stats = pf.optimize_sharpe_ratio(mean, cov, risk_free_rate=rf)
    # Unconstrained tangency: w proportional to inv(Sigma) (mu - rf).
    raw = np.array([(0.05 - rf) / 0.01, (0.15 - rf) / 0.04])
    np.testing.assert_allclose(weights, raw / raw.sum(), atol=1e-3)
    assert stats["Sharpe Ratio"] > (0.15 - rf) / 0.20  # beats the best single asset


def test_weights_are_long_only_clean_and_sum_to_one():
    rng = np.random.default_rng(3)
    window = pd.DataFrame(rng.normal(0.0004, 0.012, (400, 5)), columns=list("ABCDE"))
    for fn in (pf.max_sharpe_weights, pf.min_variance_weights, pf.equal_weights):
        w = fn(window)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()
        assert not ((w > 0) & (w < 1e-6)).any()


def test_index_in_universe_absorbs_min_variance_portfolio():
    # Stocks = market factor + large idiosyncratic noise; the index is the
    # market factor with its idiosyncratic risk diversified away. A long-only
    # minimum variance portfolio then piles into the index, which is why the
    # pipeline keeps the benchmark (SPY) out of the investable universe.
    rng = np.random.default_rng(4)
    market = rng.normal(0.0005, 0.01, 750)
    stocks = pd.DataFrame(
        market[:, None] + rng.normal(0, 0.02, (750, 4)), columns=list("ABCD")
    )
    universe = stocks.assign(INDEX=market + rng.normal(0, 0.001, 750))
    weights = pf.min_variance_weights(universe)
    assert weights[-1] > 0.9


def test_efficient_frontier_is_monotonic(two_assets):
    mean, cov = two_assets
    frontier = pf.efficient_frontier(mean, cov, n_points=10)
    assert len(frontier) == 10
    assert frontier["Return"].is_monotonic_increasing
    assert frontier["Volatility"].is_monotonic_increasing
    np.testing.assert_allclose(frontier[["LOW", "HIGH"]].sum(axis=1), 1.0)
