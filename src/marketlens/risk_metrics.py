"""Risk and performance metrics computed from periodic (daily) returns.

All functions accept a ``pd.Series`` (one asset or portfolio) or a
``pd.DataFrame`` (one column per asset) and are annualized with
``trading_days`` periods per year. Losses are reported with their natural
sign: VaR, CVaR and max drawdown are negative numbers.
"""

from typing import Union

import numpy as np
import pandas as pd

Returns = Union[pd.Series, pd.DataFrame]

TRADING_DAYS = 252


def annualized_return(returns: Returns, trading_days: int = TRADING_DAYS):
    """Geometric (compounded) annual growth rate implied by the returns."""
    growth = (1 + returns).prod()
    return growth ** (trading_days / len(returns)) - 1


def annualized_volatility(returns: Returns, trading_days: int = TRADING_DAYS):
    """Standard deviation of returns scaled by sqrt(trading_days)."""
    return returns.std() * np.sqrt(trading_days)


def sharpe_ratio(
    returns: Returns,
    risk_free_rate: float = 0.02,
    trading_days: int = TRADING_DAYS,
):
    """Annualized Sharpe ratio from mean daily excess return.

    ``risk_free_rate`` is an annual rate, converted to a per-period rate.
    """
    excess = returns - risk_free_rate / trading_days
    return excess.mean() / returns.std() * np.sqrt(trading_days)


def sortino_ratio(
    returns: Returns,
    risk_free_rate: float = 0.02,
    trading_days: int = TRADING_DAYS,
):
    """Annualized Sortino ratio (downside deviation below the risk-free rate)."""
    excess = returns - risk_free_rate / trading_days
    downside = np.sqrt((excess.clip(upper=0) ** 2).mean())
    return excess.mean() / downside * np.sqrt(trading_days)


def historical_var(returns: Returns, confidence: float = 0.95):
    """Historical one-period Value at Risk: the (1 - confidence) quantile."""
    return returns.quantile(1 - confidence)


def historical_cvar(returns: Returns, confidence: float = 0.95):
    """Expected shortfall: mean return on periods at or below the VaR."""
    var = historical_var(returns, confidence)
    if isinstance(returns, pd.DataFrame):
        return returns[returns.le(var, axis=1)].mean()
    return returns[returns <= var].mean()


def drawdown_series(returns: Returns) -> Returns:
    """Percentage distance of the compounded wealth curve from its running peak."""
    wealth = (1 + returns).cumprod()
    # Include the starting wealth of 1 so a loss on day one counts as drawdown.
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1


def max_drawdown(returns: Returns):
    """Worst peak-to-trough decline of the compounded wealth curve."""
    return drawdown_series(returns).min()


def beta(returns: Returns, benchmark: pd.Series):
    """Sensitivity of returns to a benchmark: cov(r, b) / var(b)."""
    bench_var = benchmark.var()
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: col.cov(benchmark)) / bench_var
    return returns.cov(benchmark) / bench_var


def summary_table(
    returns: Returns,
    risk_free_rate: float = 0.02,
    confidence: float = 0.95,
    trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """One row per asset/strategy with the headline risk metrics."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    pct = f"{confidence:.0%}"
    return pd.DataFrame(
        {
            "Annualized Return": annualized_return(returns, trading_days),
            "Annualized Volatility": annualized_volatility(returns, trading_days),
            "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate, trading_days),
            "Sortino Ratio": sortino_ratio(returns, risk_free_rate, trading_days),
            "Max Drawdown": max_drawdown(returns),
            f"VaR ({pct})": historical_var(returns, confidence),
            f"CVaR ({pct})": historical_cvar(returns, confidence),
        }
    )
