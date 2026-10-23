"""Backtesting engine: weight drift, periodic rebalancing, transaction costs
and walk-forward re-optimization."""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from marketlens import risk_metrics as rm


def simulate_portfolio(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 10000.0,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Simulate a portfolio day by day.

    ``target_weights`` is indexed by rebalance dates (a subset of
    ``returns.index``, which must include the first date). On a rebalance
    date the portfolio is traded back to the target weights *before* that
    day's return is earned, paying ``cost_bps`` basis points on the traded
    notional (turnover). Between rebalances the weights drift with prices.

    Returns a DataFrame indexed like ``returns`` with columns ``equity``
    (end-of-day value), ``return`` (net daily return), ``turnover`` and
    ``cost`` (currency paid that day).
    """
    if returns.index[0] not in target_weights.index:
        raise ValueError("target_weights must include the first date of returns")
    targets = target_weights.reindex(columns=returns.columns).to_numpy()
    rebalance_at = dict(zip(target_weights.index, targets))
    cost_rate = cost_bps / 10_000

    value = initial_capital
    weights = np.zeros(returns.shape[1])  # start in cash
    rows = []
    for date, day_returns in zip(returns.index, returns.to_numpy()):
        start_value = value
        turnover = cost = 0.0
        if date in rebalance_at:
            target = rebalance_at[date]
            turnover = float(np.abs(target - weights).sum())
            cost = value * cost_rate * turnover
            value -= cost
            weights = target
        gross = 1.0 + float(weights @ day_returns)
        value *= gross
        weights = weights * (1.0 + day_returns) / gross
        rows.append((value, value / start_value - 1, turnover, cost))
    return pd.DataFrame(rows, index=returns.index, columns=["equity", "return", "turnover", "cost"])


def _rebalance_dates(index: pd.Index, rebalance_every: Optional[int]) -> pd.Index:
    """Every ``rebalance_every``-th date; ``None`` means buy-and-hold."""
    step = len(index) if rebalance_every is None else rebalance_every
    return index[::step]


def run_backtest(
    returns: pd.DataFrame,
    weights: np.ndarray,
    initial_capital: float = 10000.0,
    rebalance_every: Optional[int] = 1,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Equity curve of a fixed-weight portfolio.

    ``rebalance_every`` is the number of trading days between rebalances to
    ``weights`` (1 = daily, 21 ~ monthly, ``None`` = buy-and-hold).
    """
    dates = _rebalance_dates(returns.index, rebalance_every)
    schedule = pd.DataFrame(
        np.tile(np.asarray(weights, dtype=float), (len(dates), 1)),
        index=dates,
        columns=returns.columns,
    )
    return simulate_portfolio(returns, schedule, initial_capital, cost_bps)["equity"]


def walk_forward_backtest(
    returns: pd.DataFrame,
    optimizer: Callable[[pd.DataFrame], np.ndarray],
    lookback: int = 252,
    rebalance_every: int = 21,
    initial_capital: float = 10000.0,
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-optimize on a rolling window and trade the result out of sample.

    At each rebalance date ``t`` the optimizer only sees the ``lookback``
    days strictly before ``t``, so there is no look-ahead. The first
    ``lookback`` days are used purely for estimation.

    Returns ``(result, weights)``: the daily simulation from
    ``simulate_portfolio`` and the target weights chosen on each rebalance date.
    """
    if len(returns) <= lookback:
        raise ValueError("Not enough data: need more rows than the lookback window")
    test_dates = returns.index[lookback:]
    rebalance_dates = _rebalance_dates(test_dates, rebalance_every)
    schedule = []
    for date in rebalance_dates:
        end = returns.index.get_loc(date)
        window = returns.iloc[end - lookback : end]
        schedule.append(optimizer(window))
    weights = pd.DataFrame(schedule, index=rebalance_dates, columns=returns.columns)
    result = simulate_portfolio(returns.loc[test_dates], weights, initial_capital, cost_bps)
    return result, weights


def calculate_backtest_stats(
    equity_curve: pd.Series,
    initial_capital: Optional[float] = None,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> dict[str, float]:
    """Headline performance statistics of an equity curve.

    Pass ``initial_capital`` so the first day's return is included; without
    it the first point of the curve is used as the starting value.
    """
    start = equity_curve.iloc[0] if initial_capital is None else initial_capital
    daily = equity_curve.pct_change()
    daily.iloc[0] = equity_curve.iloc[0] / start - 1
    if initial_capital is None:
        daily = daily.iloc[1:]
    return {
        "Total Return": equity_curve.iloc[-1] / start - 1,
        "CAGR": rm.annualized_return(daily, trading_days),
        "Annualized Volatility": rm.annualized_volatility(daily, trading_days),
        "Sharpe Ratio": rm.sharpe_ratio(daily, risk_free_rate, trading_days),
        "Sortino Ratio": rm.sortino_ratio(daily, risk_free_rate, trading_days),
        "Max Drawdown": rm.max_drawdown(daily),
    }
