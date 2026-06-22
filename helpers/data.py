"""Market data loading and refresh (ports NB01 data extraction and NB02 cleaning)."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
MARKET_DIR = ROOT / "data" / "market"
MARKET_MANIFEST_PATH = MARKET_DIR / "source_manifest.json"

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
DOWNLOAD_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0

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
    """Download the panel of asset close prices from yfinance.

    The download is intentionally sequential and retried per ticker because the
    multi-ticker yfinance path is more prone to partial failures and rate-limit
    issues in interactive environments.
    """
    import yfinance as yf

    series_by_asset: dict[str, pd.Series] = {}

    for asset, ticker in TICKERS.items():
        last_error: Exception | None = None
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                raw = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    group_by="column",
                    threads=False,
                    progress=False,
                )
                series = _extract_single_ticker_close(raw, ticker=ticker, asset=asset)
                if series.dropna().empty:
                    raise ValueError(f"No usable close data returned for {ticker}")
                series_by_asset[asset] = series
                time.sleep(RETRY_SLEEP_SECONDS)
                break
            except Exception as exc:  # pragma: no cover - depends on external API state
                last_error = exc
                if attempt == DOWNLOAD_RETRIES:
                    raise RuntimeError(
                        f"Failed to download {ticker} for asset {asset} after {DOWNLOAD_RETRIES} attempts"
                    ) from exc
                time.sleep(RETRY_SLEEP_SECONDS * attempt)

        if asset not in series_by_asset and last_error is not None:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Ticker {ticker} for asset {asset} did not download successfully") from last_error

    prices = pd.concat(series_by_asset.values(), axis=1)
    prices = prices[CORE_ASSETS]
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def _extract_single_ticker_close(raw: pd.DataFrame, ticker: str, asset: str) -> pd.Series:
    """Extract a single ticker close series from yfinance output."""

    if raw.empty:
        raise ValueError(f"Empty yfinance payload for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if PRICE_FIELD in level0 and ticker in level1:
            series = raw[PRICE_FIELD][ticker].copy()
        elif "Adj Close" in level0 and ticker in level1:
            series = raw["Adj Close"][ticker].copy()
        elif PRICE_FIELD in level1:
            series = raw.xs(PRICE_FIELD, axis=1, level=1).iloc[:, 0].copy()
        elif "Adj Close" in level1:
            series = raw.xs("Adj Close", axis=1, level=1).iloc[:, 0].copy()
        else:
            raise ValueError(f"Could not extract {PRICE_FIELD!r} for {ticker} from yfinance output")
    else:
        if PRICE_FIELD in raw.columns:
            series = raw[PRICE_FIELD].copy()
        elif "Adj Close" in raw.columns:
            series = raw["Adj Close"].copy()
        else:
            raise ValueError(f"Could not find {PRICE_FIELD!r} or 'Adj Close' in yfinance output for {ticker}")

    series.name = asset
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


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
    save_market_surface(prices, source="yfinance", start=start, end=end)
    core_prices.to_parquet(processed_dir / "prices_clean_core.parquet")
    core_prices.to_parquet(processed_dir / "prices_clean.parquet")
    market_vars.to_parquet(processed_dir / "market_vars_core.parquet")
    market_vars.to_parquet(processed_dir / "market_vars.parquet")

    return core_prices, market_vars


def save_market_surface(
    prices: pd.DataFrame,
    source: str = "yfinance",
    start: str = START_DATE,
    end: str | None = None,
    market_dir: Path | str = MARKET_DIR,
) -> Path:
    """Persist the downloaded price surface and a small provenance manifest.

    This keeps the market-data input layer explicit for dashboard use instead of
    hiding all data acquisition inside notebook cells.
    """

    market_dir = Path(market_dir)
    market_dir.mkdir(parents=True, exist_ok=True)
    surface_path = market_dir / "historical_prices.parquet"
    prices.to_parquet(surface_path)

    manifest = {
        "source": source,
        "start": start,
        "end": end,
        "assets": CORE_ASSETS,
        "tickers": TICKERS,
        "rows": int(len(prices)),
        "date_min": str(prices.index.min()) if len(prices.index) else None,
        "date_max": str(prices.index.max()) if len(prices.index) else None,
        "path": str(surface_path),
    }
    MARKET_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return surface_path
