"""Market data loading and refresh (ports NB01 data extraction and NB02 cleaning)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

TICKERS = {
    "Gold": "GC=F",       # Gold futures; primary alarm asset
    "Brent": "BZ=F",      # Brent crude futures; primary risk-book exposure
    "DXY": "DX-Y.NYB",    # US Dollar Index
    "VIX": "^VIX",        # CBOE Volatility Index
    "US10Y": "^TNX",      # 10Y Treasury yield
}
TICKER_TO_ASSET = {ticker: asset for asset, ticker in TICKERS.items()}

START_DATE = "2007-07-01"
PRICE_FIELD = "Close"

CORE_ASSETS = ["Gold", "Brent", "DXY", "VIX", "US10Y"]
CORE_PRICE_ASSETS = ["Gold", "Brent", "DXY"]
LEVEL_ASSETS = ["VIX", "US10Y"]


def extract_price_panel(raw: pd.DataFrame, price_field: str = PRICE_FIELD) -> pd.DataFrame:
    """Reshape a yfinance multi-ticker download into an asset-named price panel."""
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0)
        if price_field in fields:
            prices = raw[price_field].copy()
        elif "Adj Close" in fields:
            prices = raw["Adj Close"].copy()
        else:
            raise ValueError(f"Could not find {price_field!r} or 'Adj Close' in yfinance output")
    else:
        prices = raw.copy()

    prices = prices.rename(columns=TICKER_TO_ASSET)
    prices = prices[list(TICKERS.keys())]
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def download_prices(start: str = START_DATE, end: str | None = None) -> pd.DataFrame:
    """Download the panel of asset close prices from yfinance."""
    import yfinance as yf

    raw = yf.download(
        tickers=list(TICKERS.values()),
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        threads=True,
    )
    return extract_price_panel(raw)


def compute_market_vars(prices: pd.DataFrame) -> pd.DataFrame:
    """Derive log returns (price assets) and level changes (rate/vol assets) from a price panel."""
    core_prices = prices[CORE_ASSETS].dropna().copy()

    core_log_returns = np.log(core_prices[CORE_PRICE_ASSETS] / core_prices[CORE_PRICE_ASSETS].shift(1))
    core_level_changes = core_prices[LEVEL_ASSETS].diff()

    market_vars = pd.concat(
        [
            core_log_returns.rename(columns={asset: f"r_{asset}" for asset in CORE_PRICE_ASSETS}),
            core_level_changes.rename(columns={asset: f"d_{asset}" for asset in LEVEL_ASSETS}),
        ],
        axis=1,
    ).dropna()
    return market_vars


def load_prices(processed_dir: Path | str = PROCESSED_DIR) -> pd.DataFrame:
    """Load the cleaned core price panel produced by NB02."""
    return pd.read_parquet(Path(processed_dir) / "prices_clean_core.parquet")


def load_market_vars(processed_dir: Path | str = PROCESSED_DIR) -> pd.DataFrame:
    """Load the market variables (r_Gold, r_Brent, r_DXY, d_VIX, d_US10Y) produced by NB02."""
    return pd.read_parquet(Path(processed_dir) / "market_vars_core.parquet")


def refresh_market_data(
    processed_dir: Path | str = PROCESSED_DIR,
    start: str = START_DATE,
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run the NB01 download and NB02 cleaning, overwriting the processed parquet files."""
    prices = download_prices(start=start, end=end)
    core_prices = prices[CORE_ASSETS].dropna().copy()
    market_vars = compute_market_vars(prices)

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    core_prices.to_parquet(processed_dir / "prices_clean_core.parquet")
    core_prices.to_parquet(processed_dir / "prices_clean.parquet")
    market_vars.to_parquet(processed_dir / "market_vars_core.parquet")
    market_vars.to_parquet(processed_dir / "market_vars.parquet")

    return core_prices, market_vars
