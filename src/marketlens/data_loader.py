"""Data ingestion module for MarketLens pipeline."""

from pathlib import Path
from typing import List, Optional
import pandas as pd
import yfinance as yf


def download_market_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Download daily historical stock data from Yahoo Finance."""
    print(f"Downloading data for {tickers} from {start_date} to {end_date}...")

    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            "No data returned. Check ticker symbols and date range."
        )

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        raw_file = output_path / "raw_stock_data.csv"
        df.to_csv(raw_file)
        print(f"Saved raw snapshot to: {raw_file}")

    return df


def load_local_csv(file_path: Path) -> pd.DataFrame:
    """Load stock data from a local CSV file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Local file not found at {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded data successfully from: {file_path}")
    return df