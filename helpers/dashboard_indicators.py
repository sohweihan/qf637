"""Dashboard metrics assembly and lead-time / false-alarm tables (ports NB09)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import alarm as alarm_mod
from . import signals as signals_mod
from . import var as var_mod

LOOKBACK_DAYS = 30
COOLDOWN_DAYS = 5
DRAWDOWN_EVENT_LEVEL = -0.05
VOL_Z_THRESHOLD = 2.0
PORTFOLIO_VOL_Z_WINDOW = 252


def build_dashboard_metrics(
    book: pd.DataFrame,
    alarm_frame: pd.DataFrame,
    var_window: int = var_mod.VAR_WINDOW,
    var_confidence: float = var_mod.VAR_CONFIDENCE,
) -> pd.DataFrame:
    """Combine the risk book, VaR/ES, and gold alarm into the dashboard-ready table."""
    dashboard = book[
        ["R_book", "nav", "drawdown", "realized_vol_20d", "position_bbl", "exposure_usd", "pnl_usd"]
    ].copy()

    hs_var_return, hs_es_return = var_mod.historical_var_es(
        dashboard["R_book"], window=var_window, confidence=var_confidence
    )
    dashboard["hs_var_return"] = hs_var_return
    dashboard["hs_es_return"] = hs_es_return
    dashboard["var_breach"] = var_mod.var_breach_flags(dashboard["R_book"], hs_var_return)
    dashboard["excess_loss_over_var"] = hs_var_return - dashboard["R_book"]
    dashboard["var_usd"] = hs_var_return * dashboard["exposure_usd"]
    dashboard["es_usd"] = hs_es_return * dashboard["exposure_usd"]

    vol_z = signals_mod.trailing_zscore(dashboard["realized_vol_20d"], PORTFOLIO_VOL_Z_WINDOW)
    dashboard["portfolio_vol_z"] = vol_z
    dashboard["portfolio_vol_spike"] = (vol_z > VOL_Z_THRESHOLD).astype(int)
    dashboard["drawdown_event"] = (dashboard["drawdown"] <= DRAWDOWN_EVENT_LEVEL).astype(int)

    dashboard = dashboard.join(
        alarm_frame[["gold_alarm", "alarm_score", "dashboard_state", "recommended_action"]],
        how="left",
    )
    dashboard["cooled_gold_alarm"] = alarm_mod.apply_cooldown(dashboard["gold_alarm"], COOLDOWN_DAYS)

    return dashboard


def match_leads(
    alarm_dates: pd.DatetimeIndex,
    event_dates: pd.DatetimeIndex,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """For each event date, find the most recent alarm date within `lookback_days` before it."""
    rows = []
    for event_date in event_dates:
        candidates = alarm_dates[
            (alarm_dates <= event_date) & (alarm_dates >= event_date - pd.Timedelta(days=lookback_days))
        ]
        if len(candidates) == 0:
            rows.append({"event_date": event_date, "matched": False, "alarm_date": pd.NaT, "lead_days": np.nan})
        else:
            alarm_date = candidates[-1]
            rows.append(
                {
                    "event_date": event_date,
                    "matched": True,
                    "alarm_date": alarm_date,
                    "lead_days": (event_date - alarm_date).days,
                }
            )
    return pd.DataFrame(rows)


def lead_summary_table(
    alarm_dates: pd.DatetimeIndex,
    event_families: dict[str, pd.DatetimeIndex],
    lookback_days: int = LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-event-family lead-time table and summary (match rate, avg/median lead days)."""
    lead_tables = []
    for family, dates in event_families.items():
        table = match_leads(alarm_dates, dates, lookback_days)
        table["event_family"] = family
        lead_tables.append(table)

    lead_time_table = pd.concat(lead_tables, ignore_index=True) if lead_tables else pd.DataFrame()

    summary = (
        lead_time_table.groupby("event_family")
        .agg(
            event_count=("event_date", "count"),
            matched_count=("matched", "sum"),
            match_rate=("matched", "mean"),
            avg_lead_days=("lead_days", "mean"),
            median_lead_days=("lead_days", "median"),
        )
        .reset_index()
    )
    return lead_time_table, summary


def false_alarm_table_and_summary(
    alarm_dates: pd.DatetimeIndex,
    all_event_dates: pd.DatetimeIndex,
    lookback_days: int = LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag each alarm date as a false alarm if no event date follows within `lookback_days`."""
    rows = []
    for alarm_date in alarm_dates:
        future_events = all_event_dates[
            (all_event_dates >= alarm_date) & (all_event_dates <= alarm_date + pd.Timedelta(days=lookback_days))
        ]
        rows.append(
            {
                "alarm_date": alarm_date,
                "followed_by_event": len(future_events) > 0,
                "next_event_date": future_events[0] if len(future_events) else pd.NaT,
            }
        )

    table = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "alarm_count": len(table),
                "false_alarm_count": int((~table["followed_by_event"]).sum()) if len(table) else 0,
                "false_alarm_rate": float((~table["followed_by_event"]).mean()) if len(table) else np.nan,
            }
        ]
    )
    return table, summary
