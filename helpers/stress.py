"""Brent stress scenarios for the existing risk book."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FIXED_SHOCKS = (-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30)


def _as_of_timestamp(index: pd.DatetimeIndex, as_of: pd.Timestamp | str | None) -> pd.Timestamp:
    if len(index) == 0:
        raise ValueError("Cannot run stress scenarios on an empty book index.")
    return index[-1] if as_of is None else pd.Timestamp(as_of)


def _book_row(book: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    rows = book.loc[:as_of]
    if rows.empty:
        raise ValueError(f"No book data available on or before {as_of.date()}")
    return rows.iloc[-1]


def _shock_label(shock_pct: float) -> str:
    return f"Brent {shock_pct:+.0%}"


def fixed_shocks(shock_pcts: tuple[float, ...] | list[float] = DEFAULT_FIXED_SHOCKS) -> pd.DataFrame:
    """Standard policy-style Brent shock scenarios."""
    return pd.DataFrame(
        {
            "scenario_set": "fixed",
            "scenario_name": [_shock_label(shock) for shock in shock_pcts],
            "shock_pct": list(shock_pcts),
        }
    )


def run_stress_scenarios(
    book: pd.DataFrame,
    prices: pd.DataFrame,
    fixed_shock_pcts: tuple[float, ...] | list[float] | None = DEFAULT_FIXED_SHOCKS,
    loss_limit_usd: float | None = None,
    initial_margin_per_contract_usd: float | None = None,
    margin_multiplier: float | None = None,
    lot_size_bbl: float = 1_000.0,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Apply fixed Brent event shocks to the current book.

    Margin stress is optional because the repo does not contain real clearing
    margin inputs.
    """
    as_of_ts = _as_of_timestamp(book.index, as_of)
    row = _book_row(book, as_of_ts)
    current_price = float(prices["Brent"].loc[:as_of_ts].iloc[-1])
    position_bbl = float(row["position_bbl"])
    exposure_usd = float(row["exposure_usd"])
    nav = float(row["nav"])

    scenario_frames = []
    if fixed_shock_pcts is not None:
        scenario_frames.append(fixed_shocks(fixed_shock_pcts))
    if not scenario_frames:
        return pd.DataFrame()

    scenarios = pd.concat(scenario_frames, ignore_index=True)
    scenarios["as_of"] = as_of_ts
    scenarios["current_price"] = current_price
    scenarios["stressed_price"] = current_price * (1 + scenarios["shock_pct"])
    scenarios["position_bbl"] = position_bbl
    scenarios["exposure_usd"] = exposure_usd
    scenarios["stress_pnl_usd"] = position_bbl * (scenarios["stressed_price"] - current_price)
    scenarios["stress_return"] = scenarios["stress_pnl_usd"] / abs(exposure_usd) if exposure_usd else np.nan
    scenarios["nav_before_stress"] = nav
    scenarios["nav_after_stress"] = nav * (1 + scenarios["stress_return"])
    scenarios["net_contracts"] = position_bbl / lot_size_bbl if lot_size_bbl else np.nan
    scenarios["margin_assumption_used"] = initial_margin_per_contract_usd is not None
    if initial_margin_per_contract_usd is None:
        scenarios["base_margin_usd"] = np.nan
        scenarios["stressed_margin_usd"] = np.nan
        scenarios["incremental_margin_call_usd"] = np.nan
    else:
        multiplier = 1.0 if margin_multiplier is None else margin_multiplier
        base_margin = abs(scenarios["net_contracts"]) * initial_margin_per_contract_usd
        scenarios["base_margin_usd"] = base_margin
        scenarios["stressed_margin_usd"] = base_margin * multiplier
        scenarios["incremental_margin_call_usd"] = scenarios["stressed_margin_usd"] - base_margin
    scenarios["cash_need_usd"] = scenarios["stress_pnl_usd"].clip(upper=0).abs() + scenarios[
        "incremental_margin_call_usd"
    ].fillna(0.0)
    scenarios["loss_limit_usd"] = loss_limit_usd
    scenarios["breach_flag"] = (
        scenarios["stress_pnl_usd"].le(-loss_limit_usd) if loss_limit_usd is not None else False
    )
    return scenarios


def reverse_stress_to_loss_limit(
    book: pd.DataFrame,
    loss_limit_usd: float,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Brent shock needed to lose `loss_limit_usd` on the current signed exposure."""
    as_of_ts = _as_of_timestamp(book.index, as_of)
    row = _book_row(book, as_of_ts)
    exposure_usd = float(row["exposure_usd"])
    if exposure_usd == 0:
        shock_pct = np.nan
    else:
        shock_pct = -loss_limit_usd / exposure_usd
    return pd.DataFrame(
        [
            {
                "as_of": as_of_ts,
                "loss_limit_usd": loss_limit_usd,
                "exposure_usd": exposure_usd,
                "shock_pct": shock_pct,
                "scenario_name": _shock_label(shock_pct) if np.isfinite(shock_pct) else "No exposure",
            }
        ]
    )


def _demo() -> None:
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    prices = pd.DataFrame({"Brent": np.linspace(100, 110, len(dates))}, index=dates)
    book = pd.DataFrame(
        {
            "position_bbl": 100.0,
            "exposure_usd": 11_000.0,
            "nav": 100.0,
        },
        index=dates,
    )
    out = run_stress_scenarios(book, prices, fixed_shock_pcts=[-0.10, 0.10])
    assert out.loc[out["shock_pct"].eq(-0.10), "stress_pnl_usd"].iloc[0] < 0
    assert out.loc[out["shock_pct"].eq(0.10), "stress_pnl_usd"].iloc[0] > 0
    assert out["incremental_margin_call_usd"].isna().all()
    margin = run_stress_scenarios(
        book,
        prices,
        fixed_shock_pcts=[-0.10],
        initial_margin_per_contract_usd=6_000.0,
        margin_multiplier=2.0,
    )
    assert float(margin["incremental_margin_call_usd"].iloc[0]) == 600.0
    reverse = reverse_stress_to_loss_limit(book, 1_100.0)
    assert round(float(reverse["shock_pct"].iloc[0]), 2) == -0.10


if __name__ == "__main__":
    _demo()
