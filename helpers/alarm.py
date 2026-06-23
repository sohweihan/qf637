"""Conditioned gold alarm construction (NB06) and event/cooldown utilities (NB09)."""

from __future__ import annotations

import numpy as np
import pandas as pd

CONDITIONED_Z_THRESHOLD = 2.0
PRIMARY_SCORE_THRESHOLD = 1.5
COOLDOWN_DAYS = 5

RED_ACTION = "Review VaR and run stress tests"
AMBER_ACTION = "Inspect signal drivers"
GREEN_ACTION = "Normal monitoring"


def dashboard_state(score: float) -> str:
    """Map a conditioned alarm score (0-3) to a dashboard traffic-light state."""
    if score >= 2:
        return "Red"
    if score == 1:
        return "Amber"
    return "Green"


def recommended_action(dashboard_state_series: pd.Series) -> pd.Series:
    """Map a dashboard-state series to the recommended risk action."""
    return pd.Series(
        np.select(
            [dashboard_state_series.eq("Red"), dashboard_state_series.eq("Amber")],
            [RED_ACTION, AMBER_ACTION],
            default=GREEN_ACTION,
        ),
        index=dashboard_state_series.index,
    )


def build_gold_alarm_frame(signal_components: pd.DataFrame) -> pd.DataFrame:
    """Combine the three signal families into the conditioned ("OR-of-families") gold alarm."""
    alarm_frame = pd.DataFrame(index=signal_components.index)

    alarm_frame["gold_primary_score"] = pd.concat(
        [
            signal_components["gold_return_z"].abs(),
            signal_components["gold_vol_z"].clip(lower=0),
        ],
        axis=1,
    ).mean(axis=1)
    alarm_frame["alarm_score"] = alarm_frame["gold_primary_score"]
    alarm_frame["gold_primary_alarm"] = (alarm_frame["gold_primary_score"] > PRIMARY_SCORE_THRESHOLD).astype(int)

    alarm_frame["return_or_vol_alarm"] = (
        (signal_components["gold_return_z"].abs() > CONDITIONED_Z_THRESHOLD)
        | (signal_components["gold_vol_z"] > CONDITIONED_Z_THRESHOLD)
    ).astype(int)
    alarm_frame["residual_alarm"] = (
        signal_components["gold_residual_z"].abs() > CONDITIONED_Z_THRESHOLD
    ).astype(int)

    corr_z_cols = [col for col in signal_components.columns if col.startswith("gold_corr_")]
    alarm_frame["relationship_alarm"] = (
        signal_components[corr_z_cols].abs().max(axis=1) > CONDITIONED_Z_THRESHOLD
    ).astype(int)

    alarm_frame["conditioned_alarm_score"] = alarm_frame[
        ["return_or_vol_alarm", "residual_alarm", "relationship_alarm"]
    ].sum(axis=1)
    alarm_frame["conditioned_gold_alarm"] = (alarm_frame["conditioned_alarm_score"] >= 2).astype(int)
    alarm_frame["gold_alarm"] = alarm_frame["conditioned_gold_alarm"]
    alarm_frame["dashboard_state"] = alarm_frame["conditioned_alarm_score"].apply(dashboard_state)
    alarm_frame["recommended_action"] = recommended_action(alarm_frame["dashboard_state"])

    return alarm_frame


def event_starts(flag: pd.Series) -> pd.DatetimeIndex:
    """Dates where a 0/1 flag series transitions from 0 to 1."""
    flag = flag.fillna(0).astype(int)
    starts = flag.eq(1) & flag.shift(1, fill_value=0).eq(0)
    return pd.DatetimeIndex(flag.index[starts])


def apply_cooldown(flag: pd.Series, cooldown_days: int = COOLDOWN_DAYS) -> pd.Series:
    """Keep only flag firings that are more than `cooldown_days` after the previous firing."""
    cooled = pd.Series(0, index=flag.index, dtype=int)
    last_fire = None
    for date, value in flag.fillna(0).astype(int).items():
        if value != 1:
            continue
        if last_fire is None or (date - last_fire).days > cooldown_days:
            cooled.loc[date] = 1
            last_fire = date
    return cooled
