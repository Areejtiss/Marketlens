"""Module d'optimisation de portefeuille et calcul de la frontière efficace."""

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
    """Calcule le rendement, la volatilité et le ratio de Sharpe d'un portefeuille."""
    ann_return = np.sum(mean_returns * weights) * trading_days
    ann_vol = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix * trading_days, weights))
    )
    sharpe_ratio = (ann_return - risk_free_rate) / ann_vol
    return ann_return, ann_vol, sharpe_ratio


def optimize_sharpe_ratio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """Trouve les poids optimaux qui maximisent le ratio de Sharpe (Tangency Portfolio)."""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    # Fonction objectif à minimiser (Ratio de Sharpe négatif)
    def min_func_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
        return -portfolio_performance(
            weights, mean_returns, cov_matrix, risk_free_rate
        )[2]

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = num_assets * [1.0 / num_assets]

    result = minimize(
        min_func_sharpe,
        init_guess,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    opt_weights = result.x
    ret, vol, sharpe = portfolio_performance(
        opt_weights, mean_returns, cov_matrix, risk_free_rate
    )

    return opt_weights, {
        "Return": ret,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
    }


def optimize_minimum_variance(
    mean_returns: pd.Series, cov_matrix: pd.DataFrame
) -> tuple[np.ndarray, dict]:
    """Trouve les poids optimaux qui minimisent la volatilité du portefeuille (GMV)."""
    num_assets = len(mean_returns)

    def min_func_volatility(weights):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = num_assets * [1.0 / num_assets]

    result = minimize(
        min_func_volatility,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    opt_weights = result.x
    ret, vol, sharpe = portfolio_performance(
        opt_weights, mean_returns, cov_matrix
    )

    return opt_weights, {
        "Return": ret,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
    }