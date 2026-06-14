"""Brent-only physical trading book, expressed as a ledger of futures positions.

The book's history starts on the earliest trade's `entry_date` (no
before-inception padding). Each trade activates `position_bbl` from its
`entry_date` onward and starts contributing to `pnl_usd` the day after entry
(entering at the close has zero P&L on day one). `R_book` is the book's daily
return on the position held overnight (pnl_usd / prior exposure), so adding
or trimming a position does not create an artificial jump in returns -
`nav`/`drawdown`/`realized_vol_20d` track the return on the book's deployed
capital through every trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAV0 = 100.0
REALIZED_VOL_WINDOW = 20

LOT_SIZE_BBL = 1_000  # 1 ICE Brent futures contract = 1,000 barrels

# A logical, simple book: a Brent hedge position built up and partially
# unwound over time.
DEFAULT_TRADES: list[dict] = [
    {
        "trade_id": "BRENT-LONG-01",
        "instrument": "Brent",
        "side": "long",
        "contracts": 300,
        "lot_size_bbl": LOT_SIZE_BBL,
        "entry_date": "2015-01-02",
        "description": "Initial long hedge: 300 Brent futures (300,000 bbl)",
    },
    {
        "trade_id": "BRENT-LONG-02",
        "instrument": "Brent",
        "side": "long",
        "contracts": 200,
        "lot_size_bbl": LOT_SIZE_BBL,
        "entry_date": "2020-03-02",
        "description": "Add-on: 200 more Brent futures (200,000 bbl), total 500,000 bbl",
    },
    {
        "trade_id": "BRENT-REDUCE-01",
        "instrument": "Brent",
        "side": "short",
        "contracts": 150,
        "lot_size_bbl": LOT_SIZE_BBL,
        "entry_date": "2023-01-03",
        "description": "Partial unwind: sell 150 Brent futures (150,000 bbl), net down to 350,000 bbl",
    },
]


def _signed_qty_bbl(trade: dict) -> float:
    qty = trade["contracts"] * trade.get("lot_size_bbl", LOT_SIZE_BBL)
    return qty if trade["side"] == "long" else -qty


def _entry_timestamp(trade: dict, index: pd.DatetimeIndex) -> pd.Timestamp:
    entry_date = trade.get("entry_date")
    return index[0] if entry_date is None else pd.Timestamp(entry_date)


def build_brent_book(
    trades: list[dict] | None,
    prices: pd.DataFrame,
    nav0: float = NAV0,
) -> pd.DataFrame:
    """Mark a Brent futures trade ledger to market and compute book-level risk series.

    Returns a DataFrame indexed like `prices` with:
        position_bbl     - net signed barrels held
        exposure_usd      - position_bbl * Brent close (current notional)
        pnl_usd           - daily mark-to-market P&L in USD
        R_book            - daily book return (pnl_usd / prior exposure)
        nav               - NAV index starting at `nav0`
        drawdown          - drawdown from running NAV peak
        realized_vol_20d  - 20-day annualized realized volatility of R_book
    """
    if trades is None:
        trades = DEFAULT_TRADES

    full_index = prices.index
    start_ts = min(_entry_timestamp(trade, full_index) for trade in trades)

    brent = prices.loc[full_index >= start_ts, "Brent"]
    index = brent.index
    brent_diff = brent.diff()

    position_bbl = pd.Series(0.0, index=index)
    pnl_usd = pd.Series(0.0, index=index)

    for trade in trades:
        qty = _signed_qty_bbl(trade)
        entry_ts = _entry_timestamp(trade, index)
        held = index >= entry_ts
        accruing = index > entry_ts
        position_bbl.loc[held] += qty
        pnl_usd.loc[accruing] += qty * brent_diff.loc[accruing]

    book = pd.DataFrame(index=index)
    book["position_bbl"] = position_bbl
    book["exposure_usd"] = position_bbl * brent
    book["pnl_usd"] = pnl_usd

    prior_exposure = book["exposure_usd"].shift(1)
    book["R_book"] = (book["pnl_usd"] / prior_exposure).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    book["nav"] = nav0 * (1 + book["R_book"]).cumprod()
    book["drawdown"] = book["nav"] / book["nav"].cummax() - 1
    book["realized_vol_20d"] = book["R_book"].rolling(REALIZED_VOL_WINDOW).std() * np.sqrt(252)

    return book


def trade_blotter(
    trades: list[dict] | None,
    prices: pd.DataFrame,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Per-trade snapshot: entry date/price, current price, and unrealized P&L."""
    if trades is None:
        trades = DEFAULT_TRADES

    brent = prices["Brent"]
    as_of_ts = brent.index[-1] if as_of is None else pd.Timestamp(as_of)
    current_price = float(brent.asof(as_of_ts))

    rows = []
    for trade in trades:
        qty = _signed_qty_bbl(trade)
        entry_ts = _entry_timestamp(trade, brent.index)
        entry_price = trade.get("entry_price")
        if entry_price is None:
            entry_price = float(brent.asof(entry_ts))
        rows.append(
            {
                "trade_id": trade["trade_id"],
                "side": trade["side"],
                "contracts": trade["contracts"],
                "position_bbl": qty,
                "entry_date": entry_ts,
                "entry_price": entry_price,
                "as_of_date": as_of_ts,
                "current_price": current_price,
                "unrealized_pnl_usd": qty * (current_price - entry_price),
                "description": trade.get("description", ""),
            }
        )
    return pd.DataFrame(rows)
