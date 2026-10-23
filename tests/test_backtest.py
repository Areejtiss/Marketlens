import numpy as np
import pandas as pd
import pytest

from marketlens import backtest as bt


@pytest.fixture
def returns():
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=400)
    return pd.DataFrame(rng.normal(0.0004, 0.01, (400, 3)), index=index, columns=list("XYZ"))


def test_daily_rebalance_without_costs_matches_weighted_sum(returns):
    w = np.array([0.5, 0.3, 0.2])
    equity = bt.run_backtest(returns, w, 1000.0)
    expected = 1000.0 * (1 + (returns * w).sum(axis=1)).cumprod()
    pd.testing.assert_series_equal(equity, expected, check_names=False)


def test_buy_and_hold_lets_weights_drift():
    index = pd.bdate_range("2024-01-01", periods=2)
    r = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 0.10]}, index=index)
    # Day 1: A doubles -> holdings 100/50 of 150. Day 2: B +10% -> 155.
    hold = bt.run_backtest(r, [0.5, 0.5], 100.0, rebalance_every=None)
    assert hold.iloc[-1] == pytest.approx(155.0)
    # Daily rebalancing resets to 75/75 before day 2 -> 157.5.
    daily = bt.run_backtest(r, [0.5, 0.5], 100.0, rebalance_every=1)
    assert daily.iloc[-1] == pytest.approx(157.5)


def test_transaction_costs_charged_on_turnover():
    index = pd.bdate_range("2024-01-01", periods=2)
    r = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]}, index=index)
    schedule = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=index, columns=["A", "B"])
    result = bt.simulate_portfolio(r, schedule, 10_000.0, cost_bps=10)
    # Day 1: buy from cash, turnover 1. Day 2: switch A -> B, turnover 2.
    np.testing.assert_allclose(result["turnover"], [1.0, 2.0])
    assert result["cost"].iloc[0] == pytest.approx(10.0)
    assert result["equity"].iloc[-1] == pytest.approx(10_000 * (1 - 0.001) * (1 - 0.002))


def test_costs_reduce_performance(returns):
    w = [1 / 3] * 3
    free = bt.run_backtest(returns, w, rebalance_every=5, cost_bps=0)
    costly = bt.run_backtest(returns, w, rebalance_every=5, cost_bps=25)
    assert costly.iloc[-1] < free.iloc[-1]


def test_simulate_requires_weights_on_first_day(returns):
    schedule = pd.DataFrame([[1, 0, 0]], index=[returns.index[5]], columns=returns.columns)
    with pytest.raises(ValueError):
        bt.simulate_portfolio(returns, schedule)


def test_walk_forward_has_no_look_ahead(returns):
    seen = []

    def spy_optimizer(window):
        seen.append((window.index[0], window.index[-1]))
        return np.full(window.shape[1], 1 / window.shape[1])

    result, weights = bt.walk_forward_backtest(returns, spy_optimizer, lookback=100, rebalance_every=50)
    assert result.index[0] == returns.index[100]
    assert len(weights) == len(seen) == 6  # 300 test days / 50
    for (first, last), rebalance_date in zip(seen, weights.index):
        assert last < rebalance_date
        pos = returns.index.get_loc(rebalance_date)
        assert first == returns.index[pos - 100]


def test_walk_forward_rejects_short_history(returns):
    with pytest.raises(ValueError):
        bt.walk_forward_backtest(returns.iloc[:50], lambda w: np.ones(3) / 3, lookback=50)


def test_stats_include_first_day_when_capital_given():
    equity = pd.Series([110.0, 121.0, 108.9], index=pd.bdate_range("2024-01-01", periods=3))
    assert bt.calculate_backtest_stats(equity, initial_capital=100.0)["Total Return"] == pytest.approx(0.089)
    assert bt.calculate_backtest_stats(equity)["Total Return"] == pytest.approx(-0.01)


def test_stats_max_drawdown(returns):
    equity = pd.Series([100.0, 120.0, 60.0, 90.0], index=pd.bdate_range("2024-01-01", periods=4))
    assert bt.calculate_backtest_stats(equity)["Max Drawdown"] == pytest.approx(-0.5)
