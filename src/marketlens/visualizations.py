"""Matplotlib charts for the MarketLens report.

Every function returns the ``Figure`` and, if ``save_path`` is given, also
writes it to disk (parent directories are created).
"""

from pathlib import Path
from typing import Mapping, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from marketlens.returns import calculate_normalized_prices, calculate_rolling_volatility
from marketlens.risk_metrics import drawdown_series

PathLike = Optional[Union[str, Path]]


def _finish(fig: Figure, save_path: PathLike) -> Figure:
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_normalized_prices(prices: pd.DataFrame, save_path: PathLike = None) -> Figure:
    """Price paths rebased to 100 on the first date."""
    fig, ax = plt.subplots(figsize=(11, 5))
    calculate_normalized_prices(prices).plot(ax=ax, linewidth=1.2)
    ax.set(title="Normalized performance (base 100)", xlabel="", ylabel="Value")
    ax.grid(alpha=0.3)
    return _finish(fig, save_path)


def plot_correlation_matrix(returns: pd.DataFrame, save_path: PathLike = None) -> Figure:
    """Heatmap of pairwise return correlations with annotated values."""
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Daily return correlation")
    return _finish(fig, save_path)


def plot_rolling_volatility(
    returns: pd.DataFrame, window: int = 63, save_path: PathLike = None
) -> Figure:
    """Annualized rolling volatility per asset."""
    fig, ax = plt.subplots(figsize=(11, 5))
    calculate_rolling_volatility(returns, window=window).plot(ax=ax, linewidth=1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(title=f"Rolling {window}-day annualized volatility", xlabel="")
    ax.grid(alpha=0.3)
    return _finish(fig, save_path)


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    asset_points: pd.DataFrame,
    portfolios: Mapping[str, tuple[float, float]],
    save_path: PathLike = None,
) -> Figure:
    """Efficient frontier with individual assets and highlighted portfolios.

    ``asset_points`` has ``Volatility``/``Return`` columns indexed by ticker;
    ``portfolios`` maps a label to ``(volatility, return)``.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(frontier["Volatility"], frontier["Return"], color="black", lw=2,
            label="Efficient frontier (long-only)")
    ax.scatter(asset_points["Volatility"], asset_points["Return"], color="grey", zorder=3)
    for ticker, row in asset_points.iterrows():
        ax.annotate(ticker, (row["Volatility"], row["Return"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    markers = iter(["*", "D", "s", "^"])
    for label, (vol, ret) in portfolios.items():
        ax.scatter(vol, ret, marker=next(markers), s=180, zorder=4, label=label)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(title="Mean-variance efficient frontier (in-sample)",
           xlabel="Annualized volatility", ylabel="Annualized expected return")
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, save_path)


def plot_equity_curves(
    curves: pd.DataFrame, title: str = "Equity curves", save_path: PathLike = None
) -> Figure:
    """Equity curves (top) and their drawdowns (bottom), one column per strategy."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    curves.plot(ax=top, linewidth=1.4)
    top.set(title=title, ylabel="Portfolio value ($)")
    top.grid(alpha=0.3)
    drawdown_series(curves.pct_change().fillna(0)).plot(ax=bottom, linewidth=1, legend=False)
    bottom.yaxis.set_major_formatter(PercentFormatter(1.0))
    bottom.set(ylabel="Drawdown", xlabel="")
    bottom.grid(alpha=0.3)
    return _finish(fig, save_path)


def plot_weights_over_time(
    weights: pd.DataFrame, title: str = "Portfolio weights", save_path: PathLike = None
) -> Figure:
    """Stacked area chart of target weights chosen at each rebalance date."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(weights.index, weights.T.to_numpy(), labels=weights.columns, step="post")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(title=title, xlabel="")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.margins(x=0)
    return _finish(fig, save_path)


def close_all() -> None:
    """Release figure memory after batch rendering."""
    plt.close("all")


__all__ = [
    "plot_normalized_prices",
    "plot_correlation_matrix",
    "plot_rolling_volatility",
    "plot_efficient_frontier",
    "plot_equity_curves",
    "plot_weights_over_time",
    "close_all",
]
