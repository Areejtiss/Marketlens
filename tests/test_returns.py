import numpy as np
import pandas as pd
import pytest

from marketlens import returns as rt
from marketlens.data_loader import download_market_data, load_local_csv


@pytest.fixture
def raw_yfinance():
    """Mimic yfinance's ``group_by='ticker'`` layout: (ticker, field) columns."""
    index = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04", "2024-01-04"])
    columns = pd.MultiIndex.from_product([["AAA", "BBB"], ["Close", "Adj Close"]])
    data = [
        [11, 10.5, 21, 20.5],
        [10, 9.5, 20, 19.5],
        [12, np.nan, 22, -1.0],  # missing and invalid prices
        [99, 99, 99, 99],  # duplicate timestamp, must be dropped
    ]
    return pd.DataFrame(data, index=index, columns=columns, dtype=float)


def test_extract_price_matrix_selects_field_and_sorts(raw_yfinance):
    prices = rt.extract_price_matrix(raw_yfinance, "Close")
    assert list(prices.columns) == ["AAA", "BBB"]
    assert prices.index.is_monotonic_increasing
    assert prices.loc["2024-01-02", "AAA"] == 10


def test_extract_price_matrix_unknown_field(raw_yfinance):
    with pytest.raises(KeyError):
        rt.extract_price_matrix(raw_yfinance, "Open")


def test_clean_price_data(raw_yfinance, tmp_path):
    prices = rt.clean_price_data(raw_yfinance, output_dir=tmp_path)
    assert len(prices) == 3  # duplicate removed
    assert not prices.isna().any().any()
    # NaN and the negative price are forward-filled from the previous day.
    assert prices.loc["2024-01-04", "AAA"] == 10.5
    assert prices.loc["2024-01-04", "BBB"] == 20.5
    saved = pd.read_csv(tmp_path / "cleaned_prices.csv", index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(saved, prices, check_freq=False, check_names=False)


def test_simple_and_log_returns():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    simple = rt.calculate_simple_returns(prices)
    np.testing.assert_allclose(simple["A"], [0.10, -0.10])
    np.testing.assert_allclose(rt.calculate_log_returns(prices)["A"], np.log1p(simple["A"]))


def test_cumulative_and_normalized():
    prices = pd.DataFrame({"A": [50.0, 55.0, 60.5]})
    returns = rt.calculate_simple_returns(prices)
    assert rt.calculate_cumulative_returns(returns)["A"].iloc[-1] == pytest.approx(0.21)
    np.testing.assert_allclose(rt.calculate_normalized_prices(prices)["A"], [100, 110, 121])


def test_rolling_volatility_window():
    returns = pd.DataFrame({"A": np.tile([0.01, -0.01], 20)})
    vol = rt.calculate_rolling_volatility(returns, window=10)
    assert vol["A"].iloc[:9].isna().all()
    assert vol["A"].iloc[-1] == pytest.approx(returns["A"].iloc[-10:].std() * np.sqrt(252))


def test_excess_returns():
    returns = pd.DataFrame({"A": [0.02, 0.01], "SPY": [0.01, 0.01]})
    excess = rt.calculate_excess_returns(returns)
    np.testing.assert_allclose(excess["A"], [0.01, 0.0])
    with pytest.raises(KeyError):
        rt.calculate_excess_returns(returns, benchmark_col="QQQ")


def test_download_market_data_saves_snapshot(monkeypatch, tmp_path, raw_yfinance):
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        return raw_yfinance

    monkeypatch.setattr("marketlens.data_loader.yf.download", fake_download)
    df = download_market_data(["AAA", "BBB"], "2024-01-01", "2024-02-01", output_dir=tmp_path)
    assert df is raw_yfinance
    assert calls["auto_adjust"] is False and calls["group_by"] == "ticker"
    assert (tmp_path / "raw_stock_data.csv").exists()


def test_download_market_data_empty_raises(monkeypatch):
    monkeypatch.setattr("marketlens.data_loader.yf.download", lambda **_: pd.DataFrame())
    with pytest.raises(ValueError):
        download_market_data(["NOPE"], "2024-01-01", "2024-02-01")


def test_load_local_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_local_csv(tmp_path / "missing.csv")
