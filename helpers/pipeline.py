"""Single entry point: run the data -> signals -> alarm -> riskbook -> dashboard chain."""

from __future__ import annotations

from pathlib import Path

from . import alarm as alarm_mod
from . import data as data_mod
from . import dashboard as dashboard_mod
from . import riskbook as riskbook_mod
from . import signals as signals_mod
from . import stress as stress_mod


def build_all(
    processed_dir: Path | str | None = None,
    trades: list[dict] | None = None,
    refresh: bool = False,
    stress_loss_limit_usd: float | None = None,
    initial_margin_per_contract_usd: float | None = None,
    margin_multiplier: float | None = None,
) -> dict:
    """Build every helper output and return them in a dict.

    Keys: prices, market_vars, signal_components, alarm_frame, book, dashboard_metrics, stress_results.

    Args:
        processed_dir: directory containing/receiving the processed parquet files
            (defaults to `data/processed` in the repo).
        trades: Brent trade ledger for the risk book (defaults to
            `riskbook.DEFAULT_TRADES`).
        refresh: if True, re-download prices via yfinance and overwrite the
            processed parquet files before building everything else.
    """
    processed_dir = Path(processed_dir) if processed_dir is not None else data_mod.PROCESSED_DIR

    if refresh:
        prices, market_vars = data_mod.refresh_market_data(processed_dir)
    else:
        prices = data_mod.load_prices(processed_dir)
        market_vars = data_mod.load_market_vars(processed_dir)

    signal_components = signals_mod.compute_gold_signal_components(market_vars)
    alarm_frame = alarm_mod.build_gold_alarm_frame(signal_components)
    book = riskbook_mod.build_brent_book(trades, prices)
    dashboard_metrics = dashboard_mod.build_dashboard_metrics(book, alarm_frame)
    stress_results = stress_mod.run_stress_scenarios(
        book,
        prices,
        loss_limit_usd=stress_loss_limit_usd,
        initial_margin_per_contract_usd=initial_margin_per_contract_usd,
        margin_multiplier=margin_multiplier,
    )
    reverse_stress = (
        stress_mod.reverse_stress_to_loss_limit(book, stress_loss_limit_usd)
        if stress_loss_limit_usd is not None
        else None
    )

    return {
        "prices": prices,
        "market_vars": market_vars,
        "signal_components": signal_components,
        "alarm_frame": alarm_frame,
        "book": book,
        "dashboard_metrics": dashboard_metrics,
        "stress_results": stress_results,
        "reverse_stress": reverse_stress,
    }
