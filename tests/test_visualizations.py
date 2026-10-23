import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from marketlens import visualizations as viz


@pytest.fixture
def prices():
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2022-01-03", periods=200)
    steps = rng.normal(0.0005, 0.01, (200, 3))
    return pd.DataFrame(100 * np.exp(steps.cumsum(axis=0)), index=index, columns=list("ABC"))


def test_every_chart_renders_and_saves(prices, tmp_path):
    returns = prices.pct_change().dropna()
    frontier = pd.DataFrame({"Volatility": [0.10, 0.12, 0.18], "Return": [0.05, 0.08, 0.12]})
    assets = pd.DataFrame({"Volatility": [0.15, 0.2, 0.25], "Return": [0.06, 0.09, 0.12]}, index=list("ABC"))
    weights = pd.DataFrame([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]], index=returns.index[[0, 100]], columns=list("ABC"))

    charts = {
        "prices.png": lambda p: viz.plot_normalized_prices(prices, save_path=p),
        "corr.png": lambda p: viz.plot_correlation_matrix(returns, save_path=p),
        "vol.png": lambda p: viz.plot_rolling_volatility(returns, window=20, save_path=p),
        "frontier.png": lambda p: viz.plot_efficient_frontier(frontier, assets, {"Max Sharpe": (0.12, 0.08)}, save_path=p),
        "equity.png": lambda p: viz.plot_equity_curves(prices, save_path=p),
        "weights.png": lambda p: viz.plot_weights_over_time(weights, save_path=p),
    }
    for name, draw in charts.items():
        path = tmp_path / "nested" / name
        fig = draw(path)
        assert path.stat().st_size > 0
        assert fig.axes
    viz.close_all()
