"""Script d'exécution principal du pipeline MarketLens.

Exécute l'ensemble du workflow : de l'ingestion à l'optimisation et au
backtesting.
"""

from pathlib import Path
import sys

# Ajouter src au chemin système
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
from marketlens.backtest import calculate_backtest_stats, run_backtest
from marketlens.data_loader import download_market_data, load_local_csv
from marketlens.portfolio import (
    optimize_minimum_variance,
    optimize_sharpe_ratio,
)
from marketlens.returns import calculate_simple_returns


def main() -> None:
    print("=" * 60)
    print("      MARKETLENS : PIPELINE D'ANALYSE ET OPTIMISATION      ")
    print("=" * 60)

    # 1. Chargement / Ingestion des données
    processed_path = Path("data/processed/cleaned_prices.csv")

    if processed_path.exists():
        print(f"\n[1/4] Chargement des prix nettoyés depuis : {processed_path}")
        cleaned_prices = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    else:
        print("\n[1/4] Données nettoyées non trouvées. Téléchargement via yfinance...")
        tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "SPY"]
        raw_data = download_market_data(
            tickers=tickers,
            start_date="2020-01-01",
            end_date="2024-12-31",
            output_dir=Path("data/raw"),
        )
        # Extraction des prix de clôture ajustés
        if isinstance(raw_data.columns, pd.MultiIndex):
            cleaned_prices = raw_data.xs("Adj Close", level=1, axis=1)
        else:
            cleaned_prices = raw_data["Adj Close"]

    print(" ✓ Données de prix chargées avec succès.")

    # 2. Calcul des Rendements
    print("\n[2/4] Calcul des rendements quotidiens...")
    daily_returns = calculate_simple_returns(cleaned_prices)

    # 3. Optimisation In-Sample (70%)
    print("\n[3/4] Exécution de l'optimisation MPT (In-Sample)...")
    split_idx = int(len(daily_returns) * 0.70)
    in_sample = daily_returns.iloc[:split_idx]
    out_sample = daily_returns.iloc[split_idx:]

    mean_ret = in_sample.mean()
    cov_mat = in_sample.cov()

    max_sharpe_w, _ = optimize_sharpe_ratio(mean_ret, cov_mat)
    min_vol_w, _ = optimize_minimum_variance(mean_ret, cov_mat)
    print(" ✓ Poids optimaux calculés (Max Sharpe & Min Variance).")

    # 4. Backtest Out-of-Sample (30%)
    print(
        "\n[4/4] Validation Backtest Out-of-Sample (Capital Initial:"
        " $10,000)..."
    )
    capital = 10000.0
    eq_sharpe = run_backtest(out_sample, max_sharpe_w, capital)
    eq_min_vol = run_backtest(out_sample, min_vol_w, capital)

    spy_w = [1.0 if col == "SPY" else 0.0 for col in cleaned_prices.columns]
    eq_spy = run_backtest(out_sample, spy_w, capital)

    stats_sharpe = calculate_backtest_stats(eq_sharpe)
    stats_spy = calculate_backtest_stats(eq_spy)

    print("\n" + "-" * 50)
    print("RÉSULTATS CLÉS OUT-OF-SAMPLE (VS BENCHMARK SPY)")
    print("-" * 50)
    print(f"Rendement Cumulé (Max Sharpe) : {stats_sharpe['Total Return']:.2%}")
    print(f"Rendement Cumulé (SPY)        : {stats_spy['Total Return']:.2%}")
    print(f"Ratio de Sharpe (Max Sharpe)  : {stats_sharpe['Sharpe Ratio']:.2f}")
    print(f"Ratio de Sharpe (SPY)         : {stats_spy['Sharpe Ratio']:.2f}")
    print("-" * 50)
    print("\n ✓ Pipeline exécuté avec succès !")


if __name__ == "__main__":
    main()