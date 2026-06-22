"""Analytics layer for the QF637 gold-escalation dashboard.

This module builds dashboard-friendly views from the helper pipeline without
requiring the notebooks to have been executed first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helpers import alarm as alarm_mod
from helpers import dashboard as dashboard_mod
from helpers import pipeline as pipeline_mod
from helpers import signals as signals_mod

ROOT = Path(__file__).resolve().parents[1]
BOOK_INPUT_DIR = ROOT / "data" / "book"
BOOK_INPUT_PATH = BOOK_INPUT_DIR / "current_positions.csv"
STRICT_FALSE_ALARM_START = pd.Timestamp("2012-01-26")
GOLD_ALARM_LIVE_START = pd.Timestamp("2011-04-06")
DEFAULT_COOLDOWN_DAYS = 5
DEFAULT_STRESS_WINDOW_DAYS = 20
DEFAULT_LEAD_LOOKBACK_DAYS = 30
RELATIONSHIP_PREFIX = "gold_corr_"

NAMED_EPISODES: list[tuple[str, str, str]] = [
    ("GFC - Lehman + commodity crash", "2008-09-01", "2009-03-31"),
    ("European debt crisis", "2010-04-01", "2012-09-30"),
    ("Oil crash 2014-2016", "2014-07-01", "2016-02-29"),
    ("China equity shock 2015", "2015-06-01", "2015-10-31"),
    ("COVID crash + oil war", "2020-02-15", "2020-05-15"),
    ("Russia / Ukraine invasion", "2022-02-15", "2022-04-30"),
    ("Fed tightening cycle 2022", "2022-06-01", "2022-12-31"),
    ("US tariff shock 2025", "2025-04-01", "2025-05-15"),
    ("Recent geopolitical stress 2026", "2026-02-01", "2026-05-31"),
]

BOUNDARY_MESSAGES = [
    "This dashboard is an escalation trigger, not a direct Brent-loss prediction engine.",
    "The benchmark risk book is a Brent proxy, not the full physical commodity book.",
    "Isolated Brent shocks and slow cumulative drawdowns remain difficult to warn in advance.",
    "False reviews are the cost of monitoring broader macro regime shifts through Gold.",
]


class ProjectDataUnavailableError(RuntimeError):
    """Raised when the dashboard cannot build a usable project context."""


@dataclass(frozen=True)
class ProjectContext:
    """Bundle the project outputs required by the dashboard."""

    data_mode: str
    provenance_note: str
    prices: pd.DataFrame
    market_vars: pd.DataFrame
    signal_components: pd.DataFrame
    alarm_frame: pd.DataFrame
    book: pd.DataFrame
    dashboard_metrics: pd.DataFrame
    stress_results: pd.DataFrame
    stress_proxy: pd.DataFrame
    overview_metrics: dict[str, Any]
    named_episode_validation: pd.DataFrame
    default_operating_metrics: pd.DataFrame
    lead_time_table: pd.DataFrame
    lead_time_summary: pd.DataFrame
    false_alarm_summary_loose: pd.DataFrame
    brent_baseline: pd.DataFrame
    baseline_overall: pd.DataFrame
    early_warning_comparison: pd.DataFrame
    blind_spot_events: pd.DataFrame
    blind_spot_summary: pd.DataFrame
    blind_spot_patterns: pd.DataFrame
    strict_false_alarms: pd.DataFrame
    strict_false_alarm_summary: pd.DataFrame
    false_alarm_trigger_breakdown: pd.DataFrame
    relationship_driver_counts: pd.DataFrame
    family_participation: pd.DataFrame
    driver_redemption: pd.DataFrame
    boundaries: list[str]
    trade_ledger: pd.DataFrame
    book_source_label: str
    market_source_label: str
    timeline_frame: pd.DataFrame
    flare_log: pd.DataFrame


def load_project_results(
    refresh: bool = False,
    download_if_missing: bool = True,
    trades: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load pipeline outputs and optionally refresh market data if missing."""

    try:
        results = pipeline_mod.build_all(refresh=refresh, trades=trades)
        ensure_non_empty_results(results)
        return results
    except (FileNotFoundError, ProjectDataUnavailableError) as exc:
        if refresh or not download_if_missing:
            raise ProjectDataUnavailableError(
                "Processed data is unavailable. Generate `data/processed` first or allow the dashboard to "
                "download market data from yfinance."
            ) from exc
        try:
            results = pipeline_mod.build_all(refresh=True, trades=trades)
            ensure_non_empty_results(results)
            return results
        except Exception as refresh_exc:  # pragma: no cover - depends on runtime/network state
            raise ProjectDataUnavailableError(
                "The dashboard could not build project data automatically. Check your internet connection "
                "or run the notebooks/helpers pipeline first."
            ) from refresh_exc


def ensure_non_empty_results(results: dict[str, Any]) -> None:
    """Fail fast when helper outputs exist on disk but contain no usable rows."""

    required = ("prices", "market_vars", "signal_components", "alarm_frame", "book", "dashboard_metrics")
    empty_keys = [key for key in required if key not in results or getattr(results[key], "empty", True)]
    if empty_keys:
        raise ProjectDataUnavailableError(
            "Processed data files were found but contain no usable rows. "
            f"Empty outputs: {', '.join(empty_keys)}. Regenerate `data/processed` before using the live dashboard."
        )


def build_project_context(refresh: bool = False, download_if_missing: bool = True) -> ProjectContext:
    """Compute the full dashboard context from the helper pipeline."""

    trade_ledger = load_trade_ledger()
    try:
        results = load_project_results(
            refresh=refresh,
            download_if_missing=download_if_missing,
            trades=trade_ledger.to_dict("records"),
        )
    except ProjectDataUnavailableError:
        return build_static_context()

    prices = results["prices"].copy()
    market_vars = results["market_vars"].copy()
    signal_components = results["signal_components"].copy()
    alarm_frame = results["alarm_frame"].copy()
    book = results["book"].copy()
    dashboard_metrics = results["dashboard_metrics"].copy()
    stress_results = results["stress_results"].copy()

    stress_proxy = build_stress_proxy(prices, market_vars)
    named_episode_validation = build_named_episode_validation(alarm_frame)
    default_operating_metrics = build_default_operating_metrics(alarm_frame, stress_proxy)
    lead_time_table, lead_time_summary, false_alarm_summary_loose = build_lead_context(dashboard_metrics)
    brent_baseline, baseline_overall, early_warning_comparison = build_brent_baseline_context(
        market_vars, dashboard_metrics
    )
    blind_spot_events, blind_spot_summary, blind_spot_patterns = build_blind_spot_context(
        lead_time_table,
        brent_baseline,
        dashboard_metrics,
        market_vars,
    )
    (
        strict_false_alarms,
        strict_false_alarm_summary,
        false_alarm_trigger_breakdown,
        relationship_driver_counts,
        family_participation,
        driver_redemption,
    ) = build_false_alarm_context(dashboard_metrics, alarm_frame, signal_components, brent_baseline)
    timeline_frame, flare_log = build_timeline_and_flare_log(
        dashboard_metrics=dashboard_metrics,
        alarm_frame=alarm_frame,
    )

    overview_metrics = build_overview_metrics(
        alarm_frame=alarm_frame,
        signal_components=signal_components,
        dashboard_metrics=dashboard_metrics,
        named_episode_validation=named_episode_validation,
        default_operating_metrics=default_operating_metrics,
        baseline_overall=baseline_overall,
    )

    return ProjectContext(
        data_mode="live",
        provenance_note="Live mode: built from local processed parquet files and helper-pipeline outputs.",
        prices=prices,
        market_vars=market_vars,
        signal_components=signal_components,
        alarm_frame=alarm_frame,
        book=book,
        dashboard_metrics=dashboard_metrics,
        stress_results=stress_results,
        stress_proxy=stress_proxy,
        overview_metrics=overview_metrics,
        named_episode_validation=named_episode_validation,
        default_operating_metrics=default_operating_metrics,
        lead_time_table=lead_time_table,
        lead_time_summary=lead_time_summary,
        false_alarm_summary_loose=false_alarm_summary_loose,
        brent_baseline=brent_baseline,
        baseline_overall=baseline_overall,
        early_warning_comparison=early_warning_comparison,
        blind_spot_events=blind_spot_events,
        blind_spot_summary=blind_spot_summary,
        blind_spot_patterns=blind_spot_patterns,
        strict_false_alarms=strict_false_alarms,
        strict_false_alarm_summary=strict_false_alarm_summary,
        false_alarm_trigger_breakdown=false_alarm_trigger_breakdown,
        relationship_driver_counts=relationship_driver_counts,
        family_participation=family_participation,
        driver_redemption=driver_redemption,
        boundaries=BOUNDARY_MESSAGES,
        trade_ledger=trade_ledger,
        book_source_label="data/book/current_positions.csv" if BOOK_INPUT_PATH.exists() else "helpers/riskbook.DEFAULT_TRADES",
        market_source_label=load_market_source_label(),
        timeline_frame=timeline_frame,
        flare_log=flare_log,
    )


def build_static_context() -> ProjectContext:
    """Return a notebook-derived static context when live processed data is unavailable.

    This mode does not change any existing repo result files. It surfaces the
    validated project conclusions using values already present in the saved
    README/notebook outputs.
    """

    named_episode_validation = pd.DataFrame(
        [
            {"episode": "GFC - Lehman + commodity crash", "start": "2008-09-01", "end": "2009-03-31", "in_sample": True, "alarm_fired": True, "first_alarm_date": pd.Timestamp("2008-10-23"), "alarm_count": 6, "alarm_rate": 0.071, "max_gold_score": 5.011},
            {"episode": "European debt crisis", "start": "2010-04-01", "end": "2012-09-30", "in_sample": True, "alarm_fired": True, "first_alarm_date": pd.Timestamp("2010-05-06"), "alarm_count": 27, "alarm_rate": 0.080, "max_gold_score": 5.097},
            {"episode": "Oil crash 2014-2016", "start": "2014-07-01", "end": "2016-02-29", "in_sample": True, "alarm_fired": True, "first_alarm_date": pd.Timestamp("2014-07-14"), "alarm_count": 22, "alarm_rate": 0.060, "max_gold_score": 3.940},
            {"episode": "China equity shock 2015", "start": "2015-06-01", "end": "2015-10-31", "in_sample": True, "alarm_fired": True, "first_alarm_date": pd.Timestamp("2015-06-18"), "alarm_count": 3, "alarm_rate": 0.028, "max_gold_score": 1.400},
            {"episode": "COVID crash + oil war", "start": "2020-02-15", "end": "2020-05-15", "in_sample": True, "alarm_fired": True, "first_alarm_date": pd.Timestamp("2020-02-25"), "alarm_count": 10, "alarm_rate": 0.698, "max_gold_score": 5.901},
            {"episode": "Russia / Ukraine invasion", "start": "2022-02-15", "end": "2022-04-30", "in_sample": True, "alarm_fired": True, "first_alarm_date": pd.Timestamp("2022-02-25"), "alarm_count": 8, "alarm_rate": 0.519, "max_gold_score": 2.953},
            {"episode": "Fed tightening cycle 2022", "start": "2022-06-01", "end": "2022-12-31", "in_sample": True, "alarm_fired": True, "first_alarm_date": pd.Timestamp("2022-09-15"), "alarm_count": 5, "alarm_rate": 0.047, "max_gold_score": 2.212},
            {"episode": "US tariff shock 2025", "start": "2025-04-01", "end": "2025-05-15", "in_sample": True, "alarm_fired": True, "first_alarm_date": pd.Timestamp("2025-04-04"), "alarm_count": 7, "alarm_rate": 0.656, "max_gold_score": 3.791},
            {"episode": "Recent geopolitical stress 2026", "start": "2026-02-01", "end": "2026-05-31", "in_sample": True, "alarm_fired": True, "first_alarm_date": pd.Timestamp("2026-02-03"), "alarm_count": 5, "alarm_rate": 0.073, "max_gold_score": 3.874},
        ]
    )

    default_operating_metrics = pd.DataFrame(
        [
            {
                "cooldown_days": 5,
                "window_days": 20,
                "stress_event_count": 175,
                "covered_event_count": 119,
                "event_coverage_rate": 0.68,
                "alarm_episode_count": 188,
                "associated_alarm_count": 130,
                "false_review_count": 58,
                "false_review_rate": 0.308511,
                "reviews_per_year": 9.976318,
            }
        ]
    )

    baseline_overall = pd.DataFrame(
        [
            {"signal": "gold_alarm", "alarm_count": 167, "alarm_rate": 0.040771, "false_alarm_rate": 0.323353},
            {"signal": "brent_baseline", "alarm_count": 201, "alarm_rate": 0.049072, "false_alarm_rate": 0.134328},
        ]
    )

    early_warning_comparison = pd.DataFrame(
        [
            {
                "event_family": "var_breach",
                "gold_match_rate_lead_ge1": 0.494949,
                "brent_baseline_match_rate_lead_ge1": 0.378788,
                "gold_avg_lead_days": 8.1750,
                "brent_baseline_avg_lead_days": 2.861446,
                "gold_median_lead_days": 6.0,
                "brent_baseline_median_lead_days": 0.0,
                "gold_match_rate_lead_ge0": 0.606061,
                "brent_baseline_match_rate_lead_ge0": 0.838384,
            },
            {
                "event_family": "portfolio_vol_spike",
                "gold_match_rate_lead_ge1": 0.619048,
                "brent_baseline_match_rate_lead_ge1": 0.523810,
                "gold_avg_lead_days": 9.4375,
                "brent_baseline_avg_lead_days": 1.380952,
                "gold_median_lead_days": 9.0,
                "brent_baseline_median_lead_days": 1.0,
                "gold_match_rate_lead_ge0": 0.761905,
                "brent_baseline_match_rate_lead_ge0": 1.000000,
            },
            {
                "event_family": "drawdown_event",
                "gold_match_rate_lead_ge1": 0.266667,
                "brent_baseline_match_rate_lead_ge1": 0.000000,
                "gold_avg_lead_days": 12.8000,
                "brent_baseline_avg_lead_days": 0.000000,
                "gold_median_lead_days": 14.0,
                "brent_baseline_median_lead_days": 0.0,
                "gold_match_rate_lead_ge0": 0.333333,
                "brent_baseline_match_rate_lead_ge0": 0.133333,
            },
        ]
    )

    blind_spot_summary = pd.DataFrame(
        [
            {"event_family": "drawdown_event", "Both": 0, "Gold only": 4, "Brent baseline only": 0, "Blind spot": 11, "total_events": 15, "blind_spot_rate": 0.733},
            {"event_family": "portfolio_vol_spike", "Both": 6, "Gold only": 7, "Brent baseline only": 5, "Blind spot": 3, "total_events": 21, "blind_spot_rate": 0.143},
            {"event_family": "var_breach", "Both": 41, "Gold only": 57, "Brent baseline only": 34, "Blind spot": 66, "total_events": 198, "blind_spot_rate": 0.333},
        ]
    )

    blind_spot_patterns = pd.DataFrame(
        [
            {"dominant_pattern": "single_day_brent_move", "episode": 51},
            {"dominant_pattern": "cumulative_drawdown", "episode": 10},
            {"dominant_pattern": "broad_market_stress", "episode": 4},
        ]
    )

    strict_false_alarm_summary = pd.DataFrame(
        [
            {
                "alarm_count": 158,
                "false_alarm_count": 60,
                "false_alarm_rate": 0.3797,
                "genuine_catch_rate": 0.62,
            }
        ]
    )

    false_alarm_trigger_breakdown = pd.DataFrame(
        [
            {"trigger_families": "return_or_vol+residual", "false_alarm_count": 27, "avg_alarm_score": 2.0, "brent_concurrent_rate": 0.111111},
            {"trigger_families": "return_or_vol+residual+relationship", "false_alarm_count": 14, "avg_alarm_score": 3.0, "brent_concurrent_rate": 0.071429},
            {"trigger_families": "return_or_vol+relationship", "false_alarm_count": 13, "avg_alarm_score": 2.0, "brent_concurrent_rate": 0.153846},
            {"trigger_families": "residual+relationship", "false_alarm_count": 6, "avg_alarm_score": 2.0, "brent_concurrent_rate": 0.166667},
        ]
    )

    relationship_driver_counts = pd.DataFrame(
        [
            {"relationship_drivers": "d_US10Y", "count": 9},
            {"relationship_drivers": "r_DXY", "count": 6},
            {"relationship_drivers": "d_VIX", "count": 6},
            {"relationship_drivers": "d_VIX+d_US10Y", "count": 6},
            {"relationship_drivers": "r_Brent", "count": 2},
        ]
    )

    family_participation = pd.DataFrame(
        [
            {"family": "return_or_vol", "false_alarm": 0.900, "genuine_catch": 0.908},
            {"family": "residual", "false_alarm": 0.783, "genuine_catch": 0.704},
            {"family": "relationship", "false_alarm": 0.550, "genuine_catch": 0.571},
        ]
    )

    driver_redemption = pd.DataFrame(
        [
            {"driver": "r_Brent", "alarm_count": 26, "genuine_catches": 21, "genuine_catch_rate": 0.81, "avg_lead_days": 9.57, "median_lead_days": 7.0},
            {"driver": "r_DXY", "alarm_count": 32, "genuine_catches": 25, "genuine_catch_rate": 0.78, "avg_lead_days": 10.56, "median_lead_days": 8.0},
            {"driver": "d_VIX", "alarm_count": 36, "genuine_catches": 21, "genuine_catch_rate": 0.58, "avg_lead_days": 9.76, "median_lead_days": 7.0},
            {"driver": "d_US10Y", "alarm_count": 38, "genuine_catches": 21, "genuine_catch_rate": 0.55, "avg_lead_days": 11.10, "median_lead_days": 10.0},
        ]
    )

    stress_results = pd.DataFrame(
        [
            {"scenario_name": "Risk-off growth shock", "shock_pct": -0.08, "stress_pnl_usd": -8.0, "stress_return": -0.08, "nav_after_stress": 92.0, "cash_need_usd": np.nan, "breach_flag": False},
            {"scenario_name": "Geopolitical inflation shock", "shock_pct": 0.10, "stress_pnl_usd": 10.0, "stress_return": 0.10, "nav_after_stress": 110.0, "cash_need_usd": np.nan, "breach_flag": False},
            {"scenario_name": "Liquidity liquidation shock", "shock_pct": -0.06, "stress_pnl_usd": -6.0, "stress_return": -0.06, "nav_after_stress": 94.0, "cash_need_usd": np.nan, "breach_flag": False},
            {"scenario_name": "Monetary tightening shock", "shock_pct": -0.03, "stress_pnl_usd": -3.0, "stress_return": -0.03, "nav_after_stress": 97.0, "cash_need_usd": np.nan, "breach_flag": False},
        ]
    )

    overview_metrics = {
        "latest_date": pd.Timestamp("2026-06-04"),
        "dashboard_state": "Amber",
        "conditioned_alarm_score": 1,
        "active_families": ["relationship"],
        "recommended_action": "Monitor and inspect gold relationship components",
        "book_type": "Brent proxy futures book",
        "most_abnormal_relationship": "r_Brent",
        "most_abnormal_relationship_z": 2.53,
        "named_episodes_caught": 9,
        "named_episodes_total": 9,
        "stress_event_coverage": 0.68,
        "false_review_rate": 0.308511,
        "reviews_per_year": 9.976318,
        "gold_alarm_rate": 0.040771,
        "current_book_return": -0.028834,
        "current_nav": 31.04488,
        "current_drawdown": -0.83199,
        "current_realized_vol_20d": 0.451262,
        "current_var_return": -0.044967,
        "current_es_return": -0.08652,
        "current_var_usd": np.nan,
        "current_es_usd": np.nan,
        "expected_breach_rate": 0.05,
        "observed_breach_rate": 0.058174,
        "breach_count": 258,
        "sample_days": 4435,
        "max_drawdown": -0.945328,
        "covid_first_alarm": pd.Timestamp("2020-02-25"),
        "covid_first_post_alarm_breach": pd.Timestamp("2020-03-06"),
        "covid_lead_days": 10,
    }

    empty = pd.DataFrame()
    trade_ledger = load_trade_ledger()
    timeline_frame, flare_log = build_static_timeline_and_flare_log(named_episode_validation)
    return ProjectContext(
        data_mode="static_notebook",
        provenance_note="Notebook-derived static mode: built from saved README and notebook result outputs. It preserves your validated conclusions without rewriting primary result files.",
        prices=empty,
        market_vars=empty,
        signal_components=empty,
        alarm_frame=empty,
        book=empty,
        dashboard_metrics=empty,
        stress_results=stress_results,
        stress_proxy=empty,
        overview_metrics=overview_metrics,
        named_episode_validation=named_episode_validation,
        default_operating_metrics=default_operating_metrics,
        lead_time_table=empty,
        lead_time_summary=empty,
        false_alarm_summary_loose=pd.DataFrame([{"alarm_count": 167, "false_alarm_count": 54, "false_alarm_rate": 0.323353}]),
        brent_baseline=empty,
        baseline_overall=baseline_overall,
        early_warning_comparison=early_warning_comparison,
        blind_spot_events=empty,
        blind_spot_summary=blind_spot_summary,
        blind_spot_patterns=blind_spot_patterns,
        strict_false_alarms=empty,
        strict_false_alarm_summary=strict_false_alarm_summary,
        false_alarm_trigger_breakdown=false_alarm_trigger_breakdown,
        relationship_driver_counts=relationship_driver_counts,
        family_participation=family_participation,
        driver_redemption=driver_redemption,
        boundaries=BOUNDARY_MESSAGES + [
            "Static mode uses saved notebook conclusions rather than regenerated time series because the original market-data source is currently unavailable.",
        ],
        trade_ledger=trade_ledger,
        book_source_label="data/book/current_positions.csv" if BOOK_INPUT_PATH.exists() else "helpers/riskbook.DEFAULT_TRADES",
        market_source_label=load_market_source_label(default="Saved notebook outputs / static context"),
        timeline_frame=timeline_frame,
        flare_log=flare_log,
    )


def load_trade_ledger() -> pd.DataFrame:
    """Load an external current trade book if provided, else fall back to the default Brent proxy ledger."""

    from helpers import riskbook as riskbook_mod

    if BOOK_INPUT_PATH.exists():
        df = pd.read_csv(BOOK_INPUT_PATH)
        required = {"trade_id", "instrument", "side", "contracts", "entry_date"}
        missing = required - set(df.columns)
        if missing:
            raise ProjectDataUnavailableError(
                f"Trade-book input is missing required columns: {', '.join(sorted(missing))}"
            )
        if "lot_size_bbl" not in df.columns:
            df["lot_size_bbl"] = riskbook_mod.LOT_SIZE_BBL
        if "description" not in df.columns:
            df["description"] = ""
        if "entry_price" not in df.columns:
            df["entry_price"] = np.nan
        return df.copy()

    return pd.DataFrame(riskbook_mod.DEFAULT_TRADES).copy()


def load_market_source_label(default: str = "yfinance project tickers") -> str:
    """Return a short label describing the market-data source currently backing the dashboard."""

    manifest_path = ROOT / "data" / "market" / "source_manifest.json"
    if manifest_path.exists():
        try:
            import json

            manifest = json.loads(manifest_path.read_text())
            source = manifest.get("source", default)
            path = manifest.get("path")
            return f"{source} ({path})" if path else str(source)
        except Exception:
            return default
    return default


def build_timeline_and_flare_log(
    dashboard_metrics: pd.DataFrame,
    alarm_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the timeline frame and flare log for the live dashboard path."""

    timeline = dashboard_metrics.copy()
    join_cols = ["conditioned_alarm_score", "return_or_vol_alarm", "residual_alarm", "relationship_alarm"]
    missing_cols = [col for col in join_cols if col not in timeline.columns]
    if missing_cols:
        timeline = timeline.join(alarm_frame[missing_cols], how="left")
    else:
        timeline = timeline.join(alarm_frame[join_cols], how="left")
    timeline["flare_flag"] = timeline["conditioned_alarm_score"].fillna(0).ge(2).astype(int)
    flare_series = timeline["flare_flag"].fillna(0).astype(int)
    flare_starts = alarm_mod.event_starts(flare_series)
    flare_ends = pd.DatetimeIndex(
        flare_series.index[
            flare_series.eq(1) & flare_series.shift(-1, fill_value=0).eq(0)
        ]
    )

    flare_rows = []
    for flare_date, flare_end_date in zip(flare_starts, flare_ends):
        row = timeline.loc[flare_date]
        families = []
        if int(row["return_or_vol_alarm"]) == 1:
            families.append("return/vol")
        if int(row["residual_alarm"]) == 1:
            families.append("residual")
        if int(row["relationship_alarm"]) == 1:
            families.append("relationship")
        flare_window = timeline.loc[flare_date:flare_end_date]
        flare_rows.append(
            {
                "flare_date": flare_date,
                "flare_end_date": flare_end_date,
                "flare_trading_days": int(flare_window["flare_flag"].sum()),
                "dashboard_state": row["dashboard_state"],
                "conditioned_alarm_score": row["conditioned_alarm_score"],
                "families": ", ".join(families) if families else "none",
                "nav": row["nav"],
                "drawdown": row["drawdown"],
                "hs_var_return": row["hs_var_return"],
            }
        )

    return timeline, pd.DataFrame(flare_rows)


def build_static_timeline_and_flare_log(named_episode_validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a presentation-friendly timeline and flare log for static mode."""

    timeline_rows = []
    flare_rows = []
    latest_snapshot_date = pd.Timestamp("2026-06-04")
    timeline_rows.append(
        {
            "date": latest_snapshot_date,
            "kind": "latest_snapshot",
            "label": "Latest snapshot",
            "dashboard_state": "Amber",
            "conditioned_alarm_score": 1,
        }
    )

    for _, row in named_episode_validation.iterrows():
        start_ts = pd.Timestamp(row["start"])
        end_ts = pd.Timestamp(row["end"])
        timeline_rows.append(
            {
                "date": start_ts,
                "kind": "episode_start",
                "label": row["episode"],
                "dashboard_state": "Episode window",
                "conditioned_alarm_score": np.nan,
            }
        )
        flare_rows.append(
            {
                "flare_date": row["first_alarm_date"],
                "flare_end_date": end_ts,
                "flare_trading_days": int(row["alarm_count"]) if pd.notna(row["alarm_count"]) else np.nan,
                "episode": row["episode"],
                "dashboard_state": "Captured" if bool(row["alarm_fired"]) else "Missed",
                "conditioned_alarm_score": np.nan,
                "families": "episode summary only",
                "nav": np.nan,
                "drawdown": np.nan,
                "hs_var_return": np.nan,
            }
        )

    timeline = pd.DataFrame(timeline_rows).sort_values("date").reset_index(drop=True)
    flare_log = pd.DataFrame(flare_rows).sort_values("flare_date").reset_index(drop=True)
    return timeline, flare_log


def build_stress_proxy(prices: pd.DataFrame, market_vars: pd.DataFrame) -> pd.DataFrame:
    """Replicate the broad stress-proxy logic used in the notebooks."""

    aligned_prices = prices.reindex(market_vars.index)
    stress = pd.DataFrame(index=market_vars.index)

    brent_vol = market_vars["r_Brent"].rolling(30).std()
    stress["brent_vol_z"] = signals_mod.trailing_zscore(brent_vol)
    stress["brent_return_abs_z"] = signals_mod.trailing_zscore(market_vars["r_Brent"]).abs()
    stress["vix_level_z"] = signals_mod.trailing_zscore(aligned_prices["VIX"])

    stress["brent_vol_spike"] = (stress["brent_vol_z"] > 2.0).astype(int)
    stress["brent_return_shock"] = (stress["brent_return_abs_z"] > 2.0).astype(int)
    stress["vix_level_spike"] = (stress["vix_level_z"] > 2.0).astype(int)
    stress["any_stress_proxy"] = stress[
        ["brent_vol_spike", "brent_return_shock", "vix_level_spike"]
    ].max(axis=1)

    return stress


def build_named_episode_validation(alarm_frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate whether the cooled alarm fired during each named stress episode."""

    cooled_alarm = alarm_mod.apply_cooldown(alarm_frame["gold_alarm"], DEFAULT_COOLDOWN_DAYS)
    alarm_dates = alarm_mod.event_starts(cooled_alarm)

    rows: list[dict[str, Any]] = []
    for name, start, end in NAMED_EPISODES:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        window = alarm_frame.loc[start_ts:end_ts]
        if window.empty:
            rows.append(
                {
                    "episode": name,
                    "start": start,
                    "end": end,
                    "in_sample": False,
                    "alarm_fired": False,
                    "first_alarm_date": pd.NaT,
                    "alarm_count": 0,
                    "alarm_rate": np.nan,
                    "max_gold_score": np.nan,
                }
            )
            continue

        episode_alarms = alarm_dates[(alarm_dates >= start_ts) & (alarm_dates <= end_ts)]
        rows.append(
            {
                "episode": name,
                "start": start,
                "end": end,
                "in_sample": True,
                "alarm_fired": len(episode_alarms) > 0,
                "first_alarm_date": episode_alarms[0] if len(episode_alarms) else pd.NaT,
                "alarm_count": len(episode_alarms),
                "alarm_rate": float(window["gold_alarm"].mean()),
                "max_gold_score": float(window["gold_primary_score"].max()),
            }
        )

    return pd.DataFrame(rows)


def build_default_operating_metrics(alarm_frame: pd.DataFrame, stress_proxy: pd.DataFrame) -> pd.DataFrame:
    """Recreate the default operating metrics from the operational validation notebook."""

    metrics, _, _ = evaluate_alarm_vs_stress(
        alarm=alarm_frame["gold_alarm"],
        stress_flag=stress_proxy["any_stress_proxy"],
        cooldown_days=DEFAULT_COOLDOWN_DAYS,
        window_days=DEFAULT_STRESS_WINDOW_DAYS,
    )
    return pd.DataFrame([metrics])


def evaluate_alarm_vs_stress(
    alarm: pd.Series,
    stress_flag: pd.Series,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
    window_days: int = DEFAULT_STRESS_WINDOW_DAYS,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Score an alarm series against a stress-flag series using notebook-style logic."""

    cooled_alarm = alarm_mod.apply_cooldown(alarm.fillna(0).astype(int), cooldown_days)
    alarm_dates = alarm_mod.event_starts(cooled_alarm)
    stress_dates = alarm_mod.event_starts(stress_flag.fillna(0).astype(int))

    event_rows = []
    for event_date in stress_dates:
        hits = nearby_dates(alarm_dates, event_date, window_days)
        event_rows.append(
            {
                "event_date": event_date,
                "covered": len(hits) > 0,
                "nearest_alarm_date": hits[0] if len(hits) else pd.NaT,
            }
        )

    alarm_rows = []
    for alarm_date in alarm_dates:
        hits = nearby_dates(stress_dates, alarm_date, window_days)
        alarm_rows.append(
            {
                "alarm_date": alarm_date,
                "associated_with_stress": len(hits) > 0,
                "nearest_event_date": hits[0] if len(hits) else pd.NaT,
            }
        )

    event_table = pd.DataFrame(event_rows)
    alarm_table = pd.DataFrame(alarm_rows)
    sample_years = max((alarm.index.max() - alarm.index.min()).days / 365.25, 1e-9)

    metrics = {
        "cooldown_days": cooldown_days,
        "window_days": window_days,
        "stress_event_count": len(event_table),
        "covered_event_count": int(event_table["covered"].sum()) if len(event_table) else 0,
        "event_coverage_rate": float(event_table["covered"].mean()) if len(event_table) else np.nan,
        "alarm_episode_count": len(alarm_table),
        "associated_alarm_count": int(alarm_table["associated_with_stress"].sum()) if len(alarm_table) else 0,
        "false_review_count": int((~alarm_table["associated_with_stress"]).sum()) if len(alarm_table) else 0,
        "false_review_rate": float((~alarm_table["associated_with_stress"]).mean()) if len(alarm_table) else np.nan,
        "reviews_per_year": len(alarm_table) / sample_years,
    }
    return metrics, event_table, alarm_table


def build_lead_context(
    dashboard_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build lead-time tables and the loose false-alarm summary used by the dashboard."""

    metrics = dashboard_metrics.copy()
    if "cooled_gold_alarm" not in metrics.columns:
        metrics["cooled_gold_alarm"] = alarm_mod.apply_cooldown(metrics["gold_alarm"], DEFAULT_COOLDOWN_DAYS)

    alarm_dates = alarm_mod.event_starts(metrics["cooled_gold_alarm"])
    event_families = risk_event_families(metrics)
    lead_time_table, lead_summary = dashboard_mod.lead_summary_table(
        alarm_dates, event_families, DEFAULT_LEAD_LOOKBACK_DAYS
    )

    all_event_dates = union_dates(event_families.values())
    _, false_alarm_summary = dashboard_mod.false_alarm_table_and_summary(
        alarm_dates, all_event_dates, DEFAULT_LEAD_LOOKBACK_DAYS
    )
    return lead_time_table, lead_summary, false_alarm_summary


def build_brent_baseline_context(
    market_vars: pd.DataFrame,
    dashboard_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct the Brent-only baseline and compare its early-warning properties."""

    aligned = market_vars[["r_Brent"]].join(dashboard_metrics, how="inner")

    brent_return_z = signals_mod.trailing_zscore(aligned["r_Brent"])
    brent_vol = aligned["r_Brent"].rolling(30).std()
    brent_vol_z = signals_mod.trailing_zscore(brent_vol)

    baseline = pd.DataFrame(index=aligned.index)
    baseline["brent_return_z"] = brent_return_z
    baseline["brent_vol_z"] = brent_vol_z
    baseline["brent_alarm"] = ((brent_return_z.abs() > 2.0) | (brent_vol_z > 2.0)).astype(int)
    baseline["cooled_brent_alarm"] = alarm_mod.apply_cooldown(baseline["brent_alarm"], DEFAULT_COOLDOWN_DAYS)

    event_families = risk_event_families(aligned)
    all_event_dates = union_dates(event_families.values())

    brent_alarm_dates = alarm_mod.event_starts(baseline["cooled_brent_alarm"])
    gold_alarm_dates = alarm_mod.event_starts(aligned["cooled_gold_alarm"])

    brent_lead_time, brent_lead_summary = dashboard_mod.lead_summary_table(
        brent_alarm_dates, event_families, DEFAULT_LEAD_LOOKBACK_DAYS
    )
    gold_lead_time, gold_lead_summary = dashboard_mod.lead_summary_table(
        gold_alarm_dates, event_families, DEFAULT_LEAD_LOOKBACK_DAYS
    )

    _, brent_false_alarm_summary = dashboard_mod.false_alarm_table_and_summary(
        brent_alarm_dates, all_event_dates, DEFAULT_LEAD_LOOKBACK_DAYS
    )
    _, gold_false_alarm_summary = dashboard_mod.false_alarm_table_and_summary(
        gold_alarm_dates, all_event_dates, DEFAULT_LEAD_LOOKBACK_DAYS
    )

    baseline_overall = pd.DataFrame(
        [
            {
                "signal": "gold_alarm",
                "alarm_count": len(gold_alarm_dates),
                "alarm_rate": len(gold_alarm_dates) / max(len(aligned), 1),
                "false_alarm_rate": float(gold_false_alarm_summary["false_alarm_rate"].iloc[0]),
            },
            {
                "signal": "brent_baseline",
                "alarm_count": len(brent_alarm_dates),
                "alarm_rate": len(brent_alarm_dates) / max(len(aligned), 1),
                "false_alarm_rate": float(brent_false_alarm_summary["false_alarm_rate"].iloc[0]),
            },
        ]
    )

    gold_rate_ge1 = match_rate_min_lead(gold_lead_time, min_lead=1)
    brent_rate_ge1 = match_rate_min_lead(brent_lead_time, min_lead=1)

    rows = []
    for family in sorted(event_families):
        gold_row = gold_lead_summary.loc[gold_lead_summary["event_family"] == family].iloc[0]
        brent_row = brent_lead_summary.loc[brent_lead_summary["event_family"] == family].iloc[0]
        rows.append(
            {
                "event_family": family,
                "gold_match_rate_lead_ge1": float(gold_rate_ge1.get(family, np.nan)),
                "brent_baseline_match_rate_lead_ge1": float(brent_rate_ge1.get(family, np.nan)),
                "gold_avg_lead_days": float(gold_row["avg_lead_days"]),
                "brent_baseline_avg_lead_days": float(brent_row["avg_lead_days"]),
                "gold_median_lead_days": float(gold_row["median_lead_days"]),
                "brent_baseline_median_lead_days": float(brent_row["median_lead_days"]),
                "gold_match_rate_lead_ge0": float(gold_row["match_rate"]),
                "brent_baseline_match_rate_lead_ge0": float(brent_row["match_rate"]),
            }
        )

    baseline = baseline.join(
        pd.DataFrame(
            {
                "gold_alarm": aligned["gold_alarm"],
                "cooled_gold_alarm": aligned["cooled_gold_alarm"],
                "var_breach": aligned["var_breach"],
                "portfolio_vol_spike": aligned["portfolio_vol_spike"],
                "drawdown_event": aligned["drawdown_event"],
            }
        ),
        how="left",
    )

    return baseline, baseline_overall, pd.DataFrame(rows)


def build_blind_spot_context(
    gold_lead_time_table: pd.DataFrame,
    brent_baseline: pd.DataFrame,
    dashboard_metrics: pd.DataFrame,
    market_vars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify events by whether Gold and/or the Brent baseline warned in advance."""

    brent_alarm_dates = alarm_mod.event_starts(brent_baseline["cooled_brent_alarm"])
    event_families = risk_event_families(dashboard_metrics)
    brent_lead_time_table, _ = dashboard_mod.lead_summary_table(
        brent_alarm_dates, event_families, DEFAULT_LEAD_LOOKBACK_DAYS
    )

    merged = gold_lead_time_table.merge(
        brent_lead_time_table,
        on=["event_date", "event_family"],
        how="outer",
        suffixes=("_gold", "_brent"),
    )
    merged["matched_gold"] = merged["matched_gold"].fillna(False) & merged["lead_days_gold"].ge(1)
    merged["matched_brent"] = merged["matched_brent"].fillna(False) & merged["lead_days_brent"].ge(1)

    def classify(row: pd.Series) -> str:
        if row["matched_gold"] and row["matched_brent"]:
            return "Both"
        if row["matched_gold"]:
            return "Gold only"
        if row["matched_brent"]:
            return "Brent baseline only"
        return "Blind spot"

    merged["classification"] = merged.apply(classify, axis=1)

    blind_spot_summary = (
        merged.groupby(["event_family", "classification"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Both", "Gold only", "Brent baseline only", "Blind spot"], fill_value=0)
    )
    blind_spot_summary["total_events"] = blind_spot_summary.sum(axis=1)
    blind_spot_summary["blind_spot_rate"] = blind_spot_summary["Blind spot"] / blind_spot_summary["total_events"]

    blind_spot_patterns = classify_blind_spot_patterns(merged, market_vars)
    return merged, blind_spot_summary.reset_index(), blind_spot_patterns


def classify_blind_spot_patterns(merged: pd.DataFrame, market_vars: pd.DataFrame) -> pd.DataFrame:
    """Group blind spots into interpretable dominant patterns."""

    blind_spots = merged.loc[merged["classification"] == "Blind spot"].copy()
    if blind_spots.empty:
        return pd.DataFrame(columns=["dominant_pattern", "n_episodes"])

    blind_spot_dates = pd.DatetimeIndex(sorted(blind_spots["event_date"].unique()))
    episode_groups = cluster_dates(blind_spot_dates, gap_days=10)

    rows = []
    for episode_number, group in enumerate(episode_groups, start=1):
        start = group[0]
        end = group[-1]
        events_in_group = blind_spots.loc[blind_spots["event_date"].isin(group)]
        window = market_vars.loc[
            (market_vars.index >= start - pd.Timedelta(days=5)) & (market_vars.index <= end)
        ]
        families = sorted(events_in_group["event_family"].unique())
        has_drawdown = "drawdown_event" in families
        mean_d_vix = window["d_VIX"].mean()

        if has_drawdown:
            pattern = "cumulative_drawdown"
        elif mean_d_vix >= 1.5:
            pattern = "broad_market_stress"
        else:
            pattern = "single_day_brent_move"

        rows.append(
            {
                "episode": episode_number,
                "start_date": start,
                "end_date": end,
                "event_families": ", ".join(families),
                "dominant_pattern": pattern,
                "n_blind_spot_events": len(events_in_group),
                "mean_abs_r_Brent": float(window["r_Brent"].abs().mean()),
                "mean_abs_r_Gold": float(window["r_Gold"].abs().mean()),
                "mean_d_VIX": float(mean_d_vix),
            }
        )

    return pd.DataFrame(rows)


def build_false_alarm_context(
    dashboard_metrics: pd.DataFrame,
    alarm_frame: pd.DataFrame,
    signal_components: pd.DataFrame,
    brent_baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replicate the strict false-alarm analysis used in the later notebooks."""

    alarm_dates = alarm_mod.event_starts(dashboard_metrics["cooled_gold_alarm"])
    event_families = risk_event_families(dashboard_metrics)
    all_event_dates = union_dates(event_families.values())

    followed_by_event = []
    lead_days = []
    for alarm_date in alarm_dates:
        future_events = all_event_dates[
            (all_event_dates > alarm_date)
            & (all_event_dates <= alarm_date + pd.Timedelta(days=DEFAULT_LEAD_LOOKBACK_DAYS))
        ]
        followed_by_event.append(len(future_events) > 0)
        lead_days.append((future_events[0] - alarm_date).days if len(future_events) else np.nan)

    strict_alarm_table = pd.DataFrame(
        {"followed_by_event": followed_by_event, "lead_days": lead_days},
        index=alarm_dates,
    )
    strict_alarm_table.index.name = "alarm_date"
    strict_alarm_table = strict_alarm_table.loc[strict_alarm_table.index >= STRICT_FALSE_ALARM_START]
    strict_false_alarms = strict_alarm_table.loc[~strict_alarm_table["followed_by_event"]].copy()

    family_flag_cols = ["return_or_vol_alarm", "residual_alarm", "relationship_alarm"]
    family_names = {
        "return_or_vol_alarm": "return_or_vol",
        "residual_alarm": "residual",
        "relationship_alarm": "relationship",
    }

    all_live_alarms = strict_alarm_table.join(alarm_frame[family_flag_cols + ["alarm_score"]], how="left")
    all_live_alarms["trigger_families"] = all_live_alarms.apply(
        lambda row: "+".join(family_names[col] for col in family_flag_cols if row.get(col, 0) == 1),
        axis=1,
    )

    corr_z_cols = [col for col in signal_components.columns if col.startswith(RELATIONSHIP_PREFIX)]
    all_live_alarms = all_live_alarms.join(signal_components[corr_z_cols], how="left")

    def relationship_drivers(row: pd.Series) -> str:
        if row.get("relationship_alarm", 0) != 1:
            return ""
        drivers = [
            col.replace(RELATIONSHIP_PREFIX, "").replace("_z", "")
            for col in corr_z_cols
            if abs(row.get(col, np.nan)) > 2.0
        ]
        return "+".join(drivers)

    all_live_alarms["relationship_drivers"] = all_live_alarms.apply(relationship_drivers, axis=1)
    all_live_alarms = all_live_alarms.join(brent_baseline[["brent_alarm"]], how="left")
    all_live_alarms["brent_baseline_concurrent"] = all_live_alarms["brent_alarm"].fillna(0).astype(int)
    all_live_alarms["genuine_catch"] = all_live_alarms["followed_by_event"]

    strict_false_alarms = all_live_alarms.loc[~all_live_alarms["genuine_catch"]].copy()

    strict_false_alarm_summary = pd.DataFrame(
        [
            {
                "alarm_count": len(all_live_alarms),
                "false_alarm_count": len(strict_false_alarms),
                "false_alarm_rate": len(strict_false_alarms) / max(len(all_live_alarms), 1),
                "genuine_catch_rate": all_live_alarms["genuine_catch"].mean() if len(all_live_alarms) else np.nan,
            }
        ]
    )

    trigger_breakdown = (
        strict_false_alarms.groupby("trigger_families")
        .agg(
            false_alarm_count=("alarm_score", "size"),
            avg_alarm_score=("alarm_score", "mean"),
            brent_concurrent_rate=("brent_baseline_concurrent", "mean"),
        )
        .sort_values("false_alarm_count", ascending=False)
        .reset_index()
    )

    relationship_driver_counts = (
        strict_false_alarms.loc[strict_false_alarms["relationship_alarm"] == 1, "relationship_drivers"]
        .value_counts()
        .rename_axis("relationship_drivers")
        .reset_index(name="count")
    )

    family_participation = (
        all_live_alarms.groupby("genuine_catch")[family_flag_cols]
        .mean()
        .rename(index={True: "genuine_catch", False: "false_alarm"}, columns=family_names)
        .T
        .reset_index(names="family")
    )

    driver_rows = []
    relationship_live = all_live_alarms.loc[all_live_alarms["relationship_alarm"] == 1]
    for col in corr_z_cols:
        driver = col.replace(RELATIONSHIP_PREFIX, "").replace("_z", "")
        mask = relationship_live["relationship_drivers"].apply(lambda value: driver in value.split("+"))
        subset = relationship_live.loc[mask]
        genuine_subset = subset.loc[subset["genuine_catch"]]
        driver_rows.append(
            {
                "driver": driver,
                "alarm_count": len(subset),
                "genuine_catches": int(genuine_subset["genuine_catch"].sum()) if len(genuine_subset) else 0,
                "genuine_catch_rate": float(subset["genuine_catch"].mean()) if len(subset) else np.nan,
                "avg_lead_days": float(genuine_subset["lead_days"].mean()) if len(genuine_subset) else np.nan,
                "median_lead_days": float(genuine_subset["lead_days"].median()) if len(genuine_subset) else np.nan,
            }
        )

    driver_redemption = pd.DataFrame(driver_rows).sort_values("alarm_count", ascending=False)

    return (
        strict_false_alarms.reset_index(),
        strict_false_alarm_summary,
        trigger_breakdown,
        relationship_driver_counts,
        family_participation,
        driver_redemption,
    )


def build_overview_metrics(
    alarm_frame: pd.DataFrame,
    signal_components: pd.DataFrame,
    dashboard_metrics: pd.DataFrame,
    named_episode_validation: pd.DataFrame,
    default_operating_metrics: pd.DataFrame,
    baseline_overall: pd.DataFrame,
) -> dict[str, Any]:
    """Collect headline metrics displayed on the overview page."""

    latest_date = alarm_frame.index[-1]
    latest_alarm = alarm_frame.loc[latest_date]
    latest_risk = dashboard_metrics.loc[latest_date]
    corr_cols = [col for col in signal_components.columns if col.startswith(RELATIONSHIP_PREFIX)]
    most_col = signal_components.loc[latest_date, corr_cols].abs().idxmax()
    named_in_sample = named_episode_validation.loc[named_episode_validation["in_sample"]]

    active_families = [
        label
        for label, flag in (
            ("return/vol", latest_alarm["return_or_vol_alarm"]),
            ("residual", latest_alarm["residual_alarm"]),
            ("relationship", latest_alarm["relationship_alarm"]),
        )
        if int(flag) == 1
    ]

    return {
        "latest_date": latest_date,
        "dashboard_state": latest_alarm["dashboard_state"],
        "conditioned_alarm_score": int(latest_alarm["conditioned_alarm_score"]),
        "active_families": active_families,
        "recommended_action": latest_alarm["recommended_action"],
        "most_abnormal_relationship": most_col.replace(RELATIONSHIP_PREFIX, "").replace("_z", ""),
        "most_abnormal_relationship_z": float(signal_components.loc[latest_date, most_col]),
        "named_episodes_caught": int(named_in_sample["alarm_fired"].sum()),
        "named_episodes_total": int(named_in_sample["in_sample"].sum()),
        "stress_event_coverage": float(default_operating_metrics["event_coverage_rate"].iloc[0]),
        "false_review_rate": float(default_operating_metrics["false_review_rate"].iloc[0]),
        "reviews_per_year": float(default_operating_metrics["reviews_per_year"].iloc[0]),
        "gold_alarm_rate": float(
            baseline_overall.loc[baseline_overall["signal"] == "gold_alarm", "alarm_rate"].iloc[0]
        ),
        "current_nav": float(latest_risk["nav"]),
        "current_drawdown": float(latest_risk["drawdown"]),
        "current_var_return": float(latest_risk["hs_var_return"]) if pd.notna(latest_risk["hs_var_return"]) else np.nan,
        "current_es_return": float(latest_risk["hs_es_return"]) if pd.notna(latest_risk["hs_es_return"]) else np.nan,
        "current_var_usd": float(latest_risk["var_usd"]) if "var_usd" in latest_risk else np.nan,
        "current_es_usd": float(latest_risk["es_usd"]) if "es_usd" in latest_risk else np.nan,
    }


def risk_event_families(metrics: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    """Build the event families used in the lead-time analysis."""

    return {
        "var_breach": alarm_mod.event_starts(metrics["var_breach"]),
        "portfolio_vol_spike": alarm_mod.event_starts(metrics["portfolio_vol_spike"]),
        "drawdown_event": alarm_mod.event_starts(metrics["drawdown_event"]),
    }


def union_dates(date_sets: Any) -> pd.DatetimeIndex:
    """Take the union of a collection of DatetimeIndex objects."""

    return pd.DatetimeIndex(sorted(set().union(*[set(index) for index in date_sets])))


def nearby_dates(source_dates: pd.DatetimeIndex, target_date: pd.Timestamp, window_days: int) -> pd.DatetimeIndex:
    """Return dates within +/- window_days of the target date."""

    return source_dates[
        (source_dates >= target_date - pd.Timedelta(days=window_days))
        & (source_dates <= target_date + pd.Timedelta(days=window_days))
    ]


def match_rate_min_lead(lead_table: pd.DataFrame, min_lead: int) -> pd.Series:
    """Calculate the event-family match rate subject to a minimum lead threshold."""

    matched_ge = lead_table["matched"] & lead_table["lead_days"].ge(min_lead)
    return matched_ge.groupby(lead_table["event_family"]).mean()


def cluster_dates(dates: pd.DatetimeIndex, gap_days: int) -> list[list[pd.Timestamp]]:
    """Group dates into episodes whenever consecutive gaps stay within gap_days."""

    if len(dates) == 0:
        return []

    groups: list[list[pd.Timestamp]] = [[dates[0]]]
    for current_date in dates[1:]:
        if (current_date - groups[-1][-1]).days <= gap_days:
            groups[-1].append(current_date)
        else:
            groups.append([current_date])
    return groups
