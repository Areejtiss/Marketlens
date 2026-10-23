import numpy as np
import pandas as pd
import pytest

from marketlens import risk_metrics as rm


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2021-01-01", periods=500)
    data = rng.normal(0.0005, 0.01, size=(500, 2))
    return pd.DataFrame(data, index=index, columns=["A", "B"])


def test_annualized_return_of_constant_growth():
    r = pd.Series([0.001] * 252)
    assert rm.annualized_return(r) == pytest.approx(1.001**252 - 1)


def test_annualized_volatility_scales_with_sqrt_time(returns):
    expected = returns["A"].std() * np.sqrt(252)
    assert rm.annualized_volatility(returns["A"]) == pytest.approx(expected)


def test_sharpe_is_zero_when_returns_equal_risk_free():
    rf = 0.02
    r = pd.Series(rf / 252 + np.array([0.01, -0.01] * 50))
    assert rm.sharpe_ratio(r, risk_free_rate=rf) == pytest.approx(0.0, abs=1e-12)


def test_sortino_exceeds_sharpe_for_positive_skew():
    # Small frequent losses, rare large gains: downside deviation < total std.
    r = pd.Series([-0.002] * 90 + [0.03] * 10)
    assert rm.sortino_ratio(r, 0.0) > rm.sharpe_ratio(r, 0.0) > 0


def test_var_and_cvar_on_known_distribution():
    r = pd.Series(np.linspace(-0.10, 0.09, 20))  # -0.10, -0.09, ..., 0.09
    var = rm.historical_var(r, 0.90)
    assert var == pytest.approx(r.quantile(0.10))
    assert rm.historical_cvar(r, 0.90) == pytest.approx(r[r <= var].mean())
    assert rm.historical_cvar(r, 0.90) <= var


def test_cvar_dataframe_matches_per_column(returns):
    frame = rm.historical_cvar(returns)
    for col in returns:
        assert frame[col] == pytest.approx(rm.historical_cvar(returns[col]))


def test_max_drawdown_known_path():
    # Wealth: 1.0 -> 1.2 -> 0.6 -> 0.9 ; peak 1.2, trough 0.6 => -50%
    r = pd.Series([0.2, -0.5, 0.5])
    assert rm.max_drawdown(r) == pytest.approx(-0.5)


def test_max_drawdown_counts_loss_on_first_day():
    assert rm.max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1)


def test_beta_of_scaled_benchmark():
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0, 0.01, 300))
    assert rm.beta(1.5 * bench, bench) == pytest.approx(1.5)
    frame = pd.DataFrame({"x": 2 * bench, "y": -bench})
    np.testing.assert_allclose(rm.beta(frame, bench), [2.0, -1.0])


def test_summary_table_shape(returns):
    table = rm.summary_table(returns)
    assert list(table.index) == ["A", "B"]
    assert {"Sharpe Ratio", "VaR (95%)", "CVaR (95%)"} <= set(table.columns)
    assert (table["Max Drawdown"] <= 0).all()
