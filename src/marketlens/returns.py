"""Data cleaning, alignment, and return transformation module for MarketLens."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd


def extract_price_matrix(df: pd.DataFrame, price_type: str = "Adj Close") -> pd.DataFrame:
    """Extract a single price series across all tickers.

    Args:
        df: Raw multi-indexed DataFrame returned by yfinance.
        price_type: Column name to extract ('Adj Close' or 'Close').

    Returns:
        pd.DataFrame: DataFrame indexed by Date with Tickers as columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        if price_type in df.columns.get_level_values(1):
            prices = df.xs(price_type, level=1, axis=1)
        elif price_type in df.columns.get_level_values(0):
            prices = df.xs(price_type, level=0, axis=1)
        else:
            raise KeyError(f"Price field '{price_type}' not found in MultiIndex columns.")
    else:
        if price_type in df.columns:
            prices = df[[price_type]]
        else:
            prices = df.copy()

    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def clean_price_data(
    df: pd.DataFrame,
    price_type: str = "Adj Close",
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Validate, clean, and align historical stock price data.

    Args:
        df: Raw downloaded DataFrame.
        price_type: Price column to extract ('Adj Close' or 'Close').
        output_dir: Optional path to save cleaned data to CSV.

    Returns:
        pd.DataFrame: Cleaned price matrix without NaNs or invalid values.
    """
    prices = extract_price_matrix(df, price_type=price_type)

    # 1. Remove duplicate timestamps
    prices = prices[~prices.index.duplicated(keep="first")]

    # 2. Convert non-positive prices to NaN
    prices = prices.mask(prices <= 0, np.nan)

    # 3. Forward fill small gaps (holiday mismatches), then backward fill
    prices = prices.ffill().bfill()

    # 4. Verify no remaining NaNs
    if prices.isnull().values.any():
        raise ValueError("Data cleaning failed: null values persist after imputation.")

    # Save cleaned data snapshot
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_file = output_path / "cleaned_prices.csv"
        prices.to_csv(save_file)
        print(f"Cleaned price matrix saved to {save_file}")

    return prices