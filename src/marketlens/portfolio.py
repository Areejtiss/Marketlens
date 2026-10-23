"""Mean-variance portfolio optimization (long-only, fully invested)."""

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> tuple[float, float, float]:
    """Annualized return, volatility and Sharpe ratio of a weight vector."""
    weights = np.asarray(weights)
    ann_return = float(np.dot(mean_returns, weights)) * trading_days
    ann_vol = float(np.sqrt(weights @ (np.asarray(cov_matrix) * trading_days) @ weights))
    sharpe_ratio = (ann_return - risk_free_rate) / ann_vol
    return ann_return, ann_vol, sharpe_ratio


def _clean_weights(weights: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Zero out solver noise (e.g. 1e-17) and renormalize to sum to one."""
    weights = np.where(weights < tol, 0.0, weights)
    return weights / weights.sum()


def _solve(
    objective: Callable[[np.ndarray], float],
    num_assets: int,
    extra_constraints: tuple = (),
) -> np.ndarray:
    """Minimize ``objective`` over long-only weights that sum to one."""
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1}, *extra_constraints)
    result = minimize(
        objective,
        np.full(num_assets, 1.0 / num_assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * num_assets,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    return _clean_weights(result.x)


def _with_stats(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[np.ndarray, dict]:
    ret, vol, sharpe = portfolio_performance(
        weights, mean_returns, cov_matrix, risk_free_rate
    )
    return weights, {"Return": ret, "Volatility": vol, "Sharpe Ratio": sharpe}


def optimize_sharpe_ratio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """Weights of the maximum Sharpe ratio (tangency) portfolio."""

    def neg_sharpe(w):
        return -portfolio_performance(w, mean_returns, cov_matrix, risk_free_rate)[2]

    weights = _solve(neg_sharpe, len(mean_returns))
    return _with_stats(weights, mean_returns, cov_matrix, risk_free_rate)


def optimize_minimum_variance(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """Weights of the global minimum variance portfolio."""

    def volatility(w):
        return portfolio_performance(w, mean_returns, cov_matrix)[1]

    weights = _solve(volatility, len(mean_returns))
    return _with_stats(weights, mean_returns, cov_matrix, risk_free_rate)


def efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 40,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Minimum-volatility portfolios for a grid of target annual returns.

    The grid runs from the minimum variance portfolio's return up to the
    highest single-asset return (the long-only frontier's upper end).
    Returns a DataFrame with ``Return``, ``Volatility`` and one weight column
    per asset.
    """
    min_var_w, min_var_stats = optimize_minimum_variance(mean_returns, cov_matrix)
    targets = np.linspace(
        min_var_stats["Return"], mean_returns.max() * trading_days, n_points
    )
    rows = []
    for target in targets:

        def hits_target(w, target=target):
            return np.dot(mean_returns, w) * trading_days - target

        def volatility(w):
            return portfolio_performance(w, mean_returns, cov_matrix)[1]

        weights = _solve(volatility, len(mean_returns), ({"type": "eq", "fun": hits_target},))
        ret, vol, _ = portfolio_performance(weights, mean_returns, cov_matrix)
        rows.append({"Return": ret, "Volatility": vol, **dict(zip(mean_returns.index, weights))})
    return pd.DataFrame(rows)


# Optimizers with the signature expected by ``backtest.walk_forward_backtest``:
# they take a window of daily returns and return a weight vector.


def max_sharpe_weights(window: pd.DataFrame, risk_free_rate: float = 0.02) -> np.ndarray:
    return optimize_sharpe_ratio(window.mean(), window.cov(), risk_free_rate)[0]


def min_variance_weights(window: pd.DataFrame) -> np.ndarray:
    return optimize_minimum_variance(window.mean(), window.cov())[0]


def equal_weights(window: pd.DataFrame) -> np.ndarray:
    return np.full(window.shape[1], 1.0 / window.shape[1])
