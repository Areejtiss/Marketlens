"""Module de backtesting pour évaluer la performance hors-échantillon des portefeuilles."""

import numpy as np
import pandas as pd


def run_backtest(
    returns: pd.DataFrame, weights: np.ndarray, initial_capital: float = 10000.0
) -> pd.Series:
    """Simule la valeur temporelle d'un portefeuille (Equity Curve) à partir de poids fixes."""
    portfolio_daily_returns = (returns * weights).sum(axis=1)
    equity_curve = initial_capital * (1 + portfolio_daily_returns).cumprod()
    return equity_curve


def calculate_backtest_stats(
    equity_curve: pd.Series, trading_days: int = 252
) -> dict[str, float]:
    """Calcule les métriques de performance globales d'une courbe de capital."""
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_years = len(equity_curve) / trading_days
    cagr = ((1 + total_return) ** (1 / n_years)) - 1

    daily_returns = equity_curve.pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(trading_days)
    sharpe = (cagr - 0.02) / ann_vol if ann_vol != 0 else 0.0

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }