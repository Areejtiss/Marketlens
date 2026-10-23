"""Run the full MarketLens pipeline.

Load prices -> risk analysis -> mean-variance optimization -> out-of-sample
backtests (static 70/30 split and rolling walk-forward) -> figures and CSVs
in ``reports/figures``.

Usage:
    python main.py [--refresh] [--cost-bps 10] [--lookback 252] [--rebalance 21]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Allow ``python main.py`` without ``pip install -e .``
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from marketlens import backtest as bt  # noqa: E402
from marketlens import portfolio as pf  # noqa: E402
from marketlens import risk_metrics as rm  # noqa: E402
from marketlens import visualizations as viz  # noqa: E402
from marketlens.returns import calculate_simple_returns, clean_price_data  # noqa: E402

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "SPY"]
BENCHMARK = "SPY"
START, END = "2019-01-01", "2024-01-01"
PRICES_CSV = ROOT / "data" / "processed" / "cleaned_prices.csv"
FIGURES = ROOT / "reports" / "figures"
CAPITAL = 10_000.0
TRAIN_FRACTION = 0.70


def load_prices(refresh: bool) -> pd.DataFrame:
    if PRICES_CSV.exists() and not refresh:
        print(f"Loading cleaned prices from {PRICES_CSV.relative_to(ROOT)}")
        return pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    from marketlens.data_loader import download_market_data

    raw = download_market_data(TICKERS, START, END, output_dir=ROOT / "data" / "raw")
    return clean_price_data(raw, output_dir=PRICES_CSV.parent)


def stats_table(curves: pd.DataFrame, extra: dict[str, dict] | None = None) -> pd.DataFrame:
    rows = {
        name: bt.calculate_backtest_stats(curve, initial_capital=CAPITAL)
        for name, curve in curves.items()
    }
    for name, values in (extra or {}).items():
        rows[name].update(values)
    return pd.DataFrame(rows).T


def static_backtest(returns: pd.DataFrame, assets: list[str], args) -> pd.DataFrame:
    split = int(len(returns) * TRAIN_FRACTION)
    train, test = returns.iloc[:split], returns.iloc[split:]
    mean, cov = train[assets].mean(), train[assets].cov()
    print(f"\n[3/5] Optimizing on {train.index[0]:%Y-%m-%d} -> {train.index[-1]:%Y-%m-%d} "
          f"({len(assets)} assets, {BENCHMARK} excluded)")

    max_sharpe_w, max_sharpe = pf.optimize_sharpe_ratio(mean, cov)
    min_var_w, min_var = pf.optimize_minimum_variance(mean, cov)
    equal_w = pf.equal_weights(train[assets])
    weights = pd.DataFrame(
        {"Max Sharpe": max_sharpe_w, "Min Variance": min_var_w, "Equal Weight": equal_w},
        index=assets,
    )
    weights.to_csv(FIGURES / "05_optimal_weights.csv")
    print(weights.round(3).to_string())

    asset_points = pd.DataFrame(
        {"Return": mean * 252, "Volatility": (train[assets].std() * 252**0.5)}
    )
    viz.plot_efficient_frontier(
        pf.efficient_frontier(mean, cov),
        asset_points,
        {
            "Max Sharpe": (max_sharpe["Volatility"], max_sharpe["Return"]),
            "Min Variance": (min_var["Volatility"], min_var["Return"]),
        },
        save_path=FIGURES / "05_efficient_frontier.png",
    )

    print(f"\n[4/5] Static out-of-sample backtest {test.index[0]:%Y-%m-%d} -> "
          f"{test.index[-1]:%Y-%m-%d} (rebalance every {args.rebalance}d, {args.cost_bps} bps)")
    curves = pd.DataFrame({
        name: bt.run_backtest(test[assets], weights[name], CAPITAL,
                              rebalance_every=args.rebalance, cost_bps=args.cost_bps)
        for name in weights
    })
    curves[f"{BENCHMARK} (buy & hold)"] = bt.run_backtest(
        test[[BENCHMARK]], [1.0], CAPITAL, rebalance_every=None, cost_bps=args.cost_bps
    )
    viz.plot_equity_curves(curves, "Static out-of-sample backtest (70/30 split)",
                           save_path=FIGURES / "06_backtest_equity_curves.png")
    table = stats_table(curves)
    table.to_csv(FIGURES / "06_backtest_metrics.csv")
    return table


def walk_forward(returns: pd.DataFrame, assets: list[str], args) -> pd.DataFrame:
    print(f"\n[5/5] Walk-forward backtest (lookback {args.lookback}d, "
          f"re-optimize every {args.rebalance}d, {args.cost_bps} bps)")
    optimizers = {
        "Max Sharpe": pf.max_sharpe_weights,
        "Min Variance": pf.min_variance_weights,
        "Equal Weight": pf.equal_weights,
    }
    curves, extra = {}, {}
    for name, optimizer in optimizers.items():
        result, weights = bt.walk_forward_backtest(
            returns[assets], optimizer, lookback=args.lookback,
            rebalance_every=args.rebalance, initial_capital=CAPITAL, cost_bps=args.cost_bps,
        )
        curves[name] = result["equity"]
        years = len(result) / 252
        extra[name] = {
            "Annual Turnover": result["turnover"].sum() / years,
            "Total Costs ($)": result["cost"].sum(),
        }
        if name == "Max Sharpe":
            viz.plot_weights_over_time(weights, "Walk-forward Max Sharpe weights",
                                       save_path=FIGURES / "07_walk_forward_weights.png")
    curves = pd.DataFrame(curves)
    curves[f"{BENCHMARK} (buy & hold)"] = bt.run_backtest(
        returns.loc[curves.index, [BENCHMARK]], [1.0], CAPITAL,
        rebalance_every=None, cost_bps=args.cost_bps,
    )
    viz.plot_equity_curves(curves, "Walk-forward out-of-sample backtest",
                           save_path=FIGURES / "07_walk_forward_equity_curves.png")
    table = stats_table(curves, extra)
    table.to_csv(FIGURES / "07_walk_forward_metrics.csv")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download prices from Yahoo Finance")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="transaction cost per unit turnover (bps)")
    parser.add_argument("--lookback", type=int, default=252, help="walk-forward estimation window (days)")
    parser.add_argument("--rebalance", type=int, default=21, help="days between rebalances")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    print("[1/5] Data")
    prices = load_prices(args.refresh)
    returns = calculate_simple_returns(prices)
    assets = [c for c in returns.columns if c != BENCHMARK]
    print(f"      {len(returns)} daily returns, {returns.index[0]:%Y-%m-%d} -> {returns.index[-1]:%Y-%m-%d}")

    print("\n[2/5] Risk analysis (full period)")
    risk = rm.summary_table(returns)
    risk["Beta vs SPY"] = rm.beta(returns, returns[BENCHMARK])
    risk.to_csv(FIGURES / "04_risk_metrics_summary.csv")
    print(risk.round(3).to_string())
    viz.plot_normalized_prices(prices, save_path=FIGURES / "01_normalized_performance.png")
    viz.plot_correlation_matrix(returns, save_path=FIGURES / "02_correlation_matrix.png")
    viz.plot_rolling_volatility(returns, save_path=FIGURES / "03_rolling_volatility.png")

    pct = ["Total Return", "CAGR", "Annualized Volatility", "Max Drawdown"]
    static = static_backtest(returns, assets, args)
    print(static.to_string(formatters={c: "{:.1%}".format for c in pct}, float_format="{:.2f}".format))
    wf = walk_forward(returns, assets, args)
    print(wf.to_string(formatters={c: "{:.1%}".format for c in pct}, float_format="{:.2f}".format))

    viz.close_all()
    print(f"\nFigures and tables written to {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
