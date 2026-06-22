"""Streamlit dashboard for the QF637 gold-escalation project."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.analytics import (
    DEFAULT_LEAD_LOOKBACK_DAYS,
    NAMED_EPISODES,
    ProjectContext,
    ProjectDataUnavailableError,
    build_project_context,
)
from helpers import stress as stress_mod
from helpers import riskbook as riskbook_mod

TITLE = "Gold Escalation Dashboard"
BG = "#0b0a08"
SURFACE = "#17130c"
SURFACE_ALT = "#1f1910"
TEXT = "#f6efdb"
MUTED = "#bcae8d"
GOLD = "#d4af37"
GOLD_LIGHT = "#f0d27a"
RED = "#d95c4f"
GREEN = "#69b779"
AMBER = "#d8a63b"
BLUE = "#7ba7ff"

SCENARIO_PRESETS = {
    "Cross-market liquidation": {
        "shock_pct": -0.10,
        "headline": "A broad risk-off move where the flare turns into a selloff across the book.",
    },
    "Rates / macro tightening": {
        "shock_pct": -0.06,
        "headline": "A macro tightening follow-through that pressures Brent and the book through weaker risk sentiment.",
    },
    "Relationship dislocation": {
        "shock_pct": -0.04,
        "headline": "A smaller but still meaningful move where cross-market relationships stay broken after the flare.",
    },
    "Geopolitical squeeze": {
        "shock_pct": 0.08,
        "headline": "An upside supply-shock path where the same flare resolves into a Brent rally instead of a selloff.",
    },
    "Custom shock": {
        "shock_pct": -0.05,
        "headline": "Use your own Brent move assumption and watch the book reprice immediately.",
    },
}

STATE_COLORS = {"Green": GREEN, "Amber": AMBER, "Red": RED}
EVENT_TITLES = {name: (start, end) for name, start, end in NAMED_EPISODES}
EPISODE_STORIES = {
    "GFC - Lehman + commodity crash": "A severe cross-market dislocation where the dashboard's value proposition is broad regime detection rather than precise day-ahead Brent forecasting.",
    "European debt crisis": "A long stress regime where repeated alerts matter more than one isolated trigger day because the environment deteriorated over an extended horizon.",
    "Oil crash 2014-2016": "A commodity-led downturn that tests whether Gold can stay relevant when Brent weakness is persistent rather than one-off.",
    "China equity shock 2015": "A market-wide risk-off episode that highlights the dashboard's role as a cross-asset stress monitor rather than a pure oil shock detector.",
    "COVID crash + oil war": "The clearest flagship case: the signal escalated before the VaR breach cluster, which is why this episode anchors the project narrative.",
    "Russia / Ukraine invasion": "A geopolitical supply shock where Gold's cross-market role becomes relevant through regime change and stress escalation, not direct Brent prediction.",
    "Fed tightening cycle 2022": "A rates-driven macro regime where the dashboard tests whether Gold's residual and relationship components add information beyond Brent itself.",
    "US tariff shock 2025": "A policy-shock regime that illustrates how broader macro stress can matter for the risk book even when the move is not a simple oil-only story.",
    "Recent geopolitical stress 2026": "A recent validation window used to show that the framework remains interpretable in later sample periods.",
}
PROJECT_LAYERS = pd.DataFrame(
    [
        {
            "Layer": "Signal layer",
            "Inputs": "Gold, Brent, DXY, VIX, US10Y",
            "Output": "Green / Amber / Red alarm",
            "Purpose": "Detect abnormal cross-market regimes",
        },
        {
            "Layer": "Benchmark validation layer",
            "Inputs": "Brent proxy risk book",
            "Output": "VaR, breach, lead-time evidence",
            "Purpose": "Test whether the signal is useful in time, not just direction",
        },
        {
            "Layer": "Implementation layer",
            "Inputs": "Physical-book exposures and desk P&L drivers",
            "Output": "Desk revaluation and stress workflow",
            "Purpose": "Turn the signal into a risk escalation trigger for a real commodity book",
        },
    ]
)


st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="collapsed")


def apply_theme() -> None:
    """Inject a dark black-gold theme with readable text contrast."""

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {BG};
            --surface: {SURFACE};
            --surface-alt: {SURFACE_ALT};
            --text: {TEXT};
            --muted: {MUTED};
            --gold: {GOLD};
            --gold-light: {GOLD_LIGHT};
            --red: {RED};
            --green: {GREEN};
            --amber: {AMBER};
        }}

        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(212, 175, 55, 0.12), transparent 28%),
                radial-gradient(circle at top left, rgba(123, 167, 255, 0.08), transparent 24%),
                linear-gradient(180deg, #090806 0%, #0f0d09 100%);
            color: var(--text);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #110e08 0%, #0c0b08 100%);
            border-right: 1px solid rgba(212, 175, 55, 0.18);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text);
        }}

        [data-testid="stMetric"] {{
            background: rgba(23, 19, 12, 0.88);
            border: 1px solid rgba(212, 175, 55, 0.16);
            border-radius: 14px;
            padding: 10px 12px;
        }}

        [data-testid="stMetricLabel"] {{
            color: var(--muted);
            font-weight: 600;
        }}

        [data-testid="stMetricValue"] {{
            color: var(--text);
        }}

        .card {{
            background: rgba(23, 19, 12, 0.9);
            border: 1px solid rgba(212, 175, 55, 0.18);
            border-radius: 18px;
            padding: 1rem 1rem 0.8rem 1rem;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset;
        }}

        .hero {{
            padding: 1.2rem 1.25rem 1rem 1.25rem;
            border-radius: 20px;
            background:
                linear-gradient(135deg, rgba(212,175,55,0.16), rgba(23,19,12,0.92) 40%),
                rgba(23, 19, 12, 0.95);
            border: 1px solid rgba(212, 175, 55, 0.24);
            margin-bottom: 0.8rem;
        }}

        .hero h1 {{
            margin: 0;
            color: var(--gold-light);
            font-size: 2rem;
            line-height: 1.15;
        }}

        .hero p {{
            color: var(--text);
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
            line-height: 1.55;
        }}

        .state-pill {{
            display: inline-block;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            font-size: 0.92rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-right: 0.45rem;
        }}

        .state-green {{
            background: rgba(105, 183, 121, 0.16);
            color: {GREEN};
            border: 1px solid rgba(105, 183, 121, 0.34);
        }}

        .state-amber {{
            background: rgba(216, 166, 59, 0.16);
            color: {GOLD_LIGHT};
            border: 1px solid rgba(216, 166, 59, 0.34);
        }}

        .state-red {{
            background: rgba(217, 92, 79, 0.16);
            color: #ff9f94;
            border: 1px solid rgba(217, 92, 79, 0.34);
        }}

        .small-note {{
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }}

        .stCaption,
        .stCaption p,
        label,
        label p,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-baseweb="slider"] + div,
        [data-baseweb="select"] label,
        [data-testid="stMarkdownContainer"] small,
        [data-testid="stNumberInput"] label,
        [data-testid="stNumberInput"] label p,
        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] label p,
        [data-testid="stSlider"] label,
        [data-testid="stSlider"] label p,
        [data-testid="stDateInput"] label,
        [data-testid="stDateInput"] label p,
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] label p {{
            color: #eadfbe !important;
            opacity: 1 !important;
            visibility: visible !important;
        }}

        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stDateInput"] input,
        [data-baseweb="select"] > div,
        [data-testid="stSlider"] [role="slider"] {{
            color: var(--text) !important;
        }}

        .boundary-list li {{
            margin-bottom: 0.45rem;
        }}

        .formula {{
            background: rgba(31, 25, 16, 0.92);
            border-left: 3px solid rgba(212, 175, 55, 0.6);
            padding: 0.8rem 0.95rem;
            border-radius: 10px;
            color: var(--text);
            margin-bottom: 0.75rem;
        }}

        .section-title {{
            color: var(--gold-light);
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}

        .story-header {{
            padding: 0.9rem 1rem 0.8rem 1rem;
            border-radius: 16px;
            background:
                linear-gradient(120deg, rgba(212,175,55,0.12), rgba(23,19,12,0.94) 45%),
                rgba(23, 19, 12, 0.94);
            border: 1px solid rgba(212, 175, 55, 0.18);
            margin-bottom: 0.9rem;
        }}

        .story-kicker {{
            color: var(--gold);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.22rem;
        }}

        .story-title {{
            color: var(--gold-light);
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.28rem;
        }}

        .story-subtitle {{
            color: var(--text);
            font-size: 1rem;
            line-height: 1.55;
        }}

        .takeaway {{
            margin-top: 0.75rem;
            background: rgba(31, 25, 16, 0.92);
            border-left: 4px solid rgba(212, 175, 55, 0.65);
            border-radius: 12px;
            padding: 0.8rem 0.95rem;
            color: var(--text);
        }}

        .takeaway strong {{
            color: var(--gold-light);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_context(refresh: bool, download_if_missing: bool) -> ProjectContext:
    """Cache the heavy analytics build behind a simple interface."""

    return build_project_context(refresh=refresh, download_if_missing=download_if_missing)


def render_empty_state(message: str) -> None:
    """Render the dashboard shell even when underlying project data is unavailable."""

    st.markdown(
        f"""
        <div class="hero">
            <h1>{TITLE}</h1>
            <p>
                The dashboard layout is ready, but the project data is not available yet.
                To unlock the live signal, risk, episode, and diagnostics views, generate
                <code>data/processed</code> first or retry automatic download later.
            </p>
            <p class="small-note">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        st.markdown("<div class='section-title'>What this dashboard is designed to show</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card">
                <ul class="boundary-list">
                    <li><strong>Overview:</strong> current Green / Amber / Red state, active families, and recommended action</li>
                    <li><strong>Signal Monitor:</strong> return/vol, residual, and relationship anomalies</li>
                    <li><strong>Risk & Stress:</strong> Brent proxy-book VaR, drawdown, and scenario losses</li>
                    <li><strong>Episode Explorer:</strong> named event studies such as COVID 2020</li>
                    <li><strong>Diagnostics & Method:</strong> baseline comparison, blind spots, false alarms, and formulas</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-title'>Model core</div>", unsafe_allow_html=True)
        st.markdown(
            r"""
            <div class="formula">
                <strong>Alarm logic</strong><br/>
                Family 1: <code>|gold_return_z| &gt; 2</code> or <code>gold_vol_z &gt; 2</code><br/>
                Family 2: <code>|gold_residual_z| &gt; 2</code><br/>
                Family 3: <code>max(|corr_z|) &gt; 2</code><br/>
                Red if at least two families are active.
            </div>
            <div class="formula">
                <strong>Risk validation logic</strong><br/>
                Historical-simulation VaR uses prior returns only:<br/>
                \( VaR^{95\%}_t = Q_{5\%}(R_{book, t-250:t-1}) \)
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div class='section-title'>How to enable live data</div>", unsafe_allow_html=True)
        st.code(
            "1. Run the notebooks or helper pipeline to create data/processed\n"
            "2. Or retry with network access when yfinance rate limits clear\n"
            "3. Then restart:\n"
            "   streamlit run dashboard/app.py",
            language="text",
        )
        st.markdown("<div class='section-title'>Expected inputs</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card">
                <ul class="boundary-list">
                    <li><code>data/processed/prices_clean_core.parquet</code></li>
                    <li><code>data/processed/market_vars_core.parquet</code></li>
                    <li>Optional notebook outputs under <code>outputs/</code> are recomputed inside the dashboard analytics layer.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-title'>Design principles</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card">
                <ul class="boundary-list">
                    <li>State first, not notebook chronology</li>
                    <li>Action second, not raw factor tables</li>
                    <li>Evidence and boundaries always visible</li>
                    <li>Readable black-gold contrast for presentation use</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def fmt_date(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def state_badge(state: str) -> str:
    css_class = {"Green": "state-green", "Amber": "state-amber", "Red": "state-red"}.get(state, "state-amber")
    return f"<span class='state-pill {css_class}'>{state}</span>"


def flare_help_text() -> str:
    return (
        "A flare is the first trading day of a Red-alarm episode: the date when the conditioned Gold alarm "
        "moves from non-Red into Red. It is the start point for drilldown, not every red day inside the same episode."
    )


def flare_label(flare_date: pd.Timestamp, flare_row: pd.Series | dict[str, object]) -> str:
    families = str(flare_row.get("families", ""))
    duration = flare_row.get("flare_trading_days", np.nan)
    duration_text = ""
    if pd.notna(duration):
        duration_text = f" | {int(duration)}d"
    return f"{fmt_date(pd.Timestamp(flare_date))} | {families}{duration_text}"


def format_signed_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+.1%}"


def format_signed_num(value: float | None, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+,.{digits}f}"


def get_static_focus(context: ProjectContext) -> dict[str, object]:
    """Return the current static-mode focus selection."""

    focus_mode = st.session_state.get("static_focus_mode", "Latest snapshot")
    focus_date = st.session_state.get("static_as_of_date", str(context.overview_metrics["latest_date"]))
    focus_episode = st.session_state.get("static_focus_episode")
    return {
        "mode": focus_mode,
        "date": pd.Timestamp(focus_date),
        "episode": focus_episode,
    }


def static_selection_context(context: ProjectContext, as_of: pd.Timestamp) -> dict[str, object]:
    """Map a selected date to the best available static summary.

    Static notebook mode does not have a full daily time series. It supports:
    - one exact latest-state snapshot;
    - named-episode window mapping;
    - a clear fallback message when no saved static summary exists.
    """

    selected = pd.Timestamp(as_of).normalize()
    latest = pd.Timestamp(context.overview_metrics["latest_date"]).normalize()

    if selected == latest:
        return {
            "kind": "latest_snapshot",
            "selected_date": selected,
            "label": "Latest validated snapshot",
            "state": context.overview_metrics["dashboard_state"],
            "score": context.overview_metrics["conditioned_alarm_score"],
            "active_families": context.overview_metrics["active_families"],
            "recommended_action": context.overview_metrics["recommended_action"],
            "episode": None,
            "first_alarm_date": pd.NaT,
            "alarm_count": None,
            "alarm_rate": None,
            "max_gold_score": None,
            "note": "This is the saved latest-state snapshot from the notebook outputs.",
            "strongest_relationship": context.overview_metrics["most_abnormal_relationship"],
            "relationship_z": context.overview_metrics["most_abnormal_relationship_z"],
            "var_return": context.overview_metrics["current_var_return"],
            "es_return": context.overview_metrics["current_es_return"],
            "nav": context.overview_metrics["current_nav"],
            "drawdown": context.overview_metrics["current_drawdown"],
        }

    episodes = context.named_episode_validation.copy()
    episodes["start_ts"] = pd.to_datetime(episodes["start"])
    episodes["end_ts"] = pd.to_datetime(episodes["end"])
    match = episodes.loc[(episodes["start_ts"] <= selected) & (episodes["end_ts"] >= selected) & episodes["in_sample"]]

    if not match.empty:
        row = match.iloc[0]
        lead_days = np.nan
        if str(row["episode"]) == "COVID crash + oil war":
            lead_days = context.overview_metrics.get("covid_lead_days", np.nan)
        return {
            "kind": "episode_window",
            "selected_date": selected,
            "label": "Named episode window",
            "state": "Episode captured" if bool(row["alarm_fired"]) else "Episode missed",
            "score": "Episode-level only",
            "active_families": ["not stored at daily resolution in static mode"],
            "recommended_action": (
                f"Use the episode summary: first alarm {fmt_date(row['first_alarm_date'])}, "
                f"{int(row['alarm_count'])} alarm(s) in window."
            ),
            "episode": row["episode"],
            "episode_start": row["start_ts"],
            "episode_end": row["end_ts"],
            "first_alarm_date": row["first_alarm_date"],
            "lead_days": lead_days,
            "alarm_count": int(row["alarm_count"]),
            "alarm_rate": float(row["alarm_rate"]) if pd.notna(row["alarm_rate"]) else np.nan,
            "max_gold_score": float(row["max_gold_score"]) if pd.notna(row["max_gold_score"]) else np.nan,
            "note": "Static mode maps this date to the containing named stress episode instead of reconstructing a full daily signal state.",
            "strongest_relationship": None,
            "relationship_z": np.nan,
            "var_return": np.nan,
            "es_return": np.nan,
            "nav": np.nan,
            "drawdown": np.nan,
        }

    return {
        "kind": "no_snapshot",
        "selected_date": selected,
        "label": "No saved static snapshot",
        "state": "Unavailable in static mode",
        "score": "N/A",
        "active_families": [],
        "recommended_action": "Static mode cannot reconstruct a true daily state for this date without live processed data.",
        "episode": None,
        "first_alarm_date": pd.NaT,
        "alarm_count": None,
        "alarm_rate": np.nan,
        "max_gold_score": np.nan,
        "note": "Choose the latest snapshot date or a date inside one of the named episode windows, or rebuild live processed data for true day-level state lookup.",
        "strongest_relationship": None,
        "relationship_z": np.nan,
        "var_return": np.nan,
        "es_return": np.nan,
        "nav": np.nan,
        "drawdown": np.nan,
    }


def base_layout(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    """Apply a consistent black-gold layout to Plotly figures."""

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12, 10, 7, 0.85)",
        margin=dict(l=45, r=28, t=56, b=42),
        font=dict(color=TEXT, size=13),
        title=dict(text=title or "", x=0.01, xanchor="left", font=dict(color=GOLD_LIGHT, size=17)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
        ),
        hoverlabel=dict(bgcolor=SURFACE_ALT, bordercolor=GOLD, font=dict(color=TEXT)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color=MUTED)
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.08)",
        color=MUTED,
    )
    return fig


def build_score_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["conditioned_alarm_score"],
            mode="lines",
            name="Conditioned score",
            line=dict(color=GOLD, width=2.4),
            fill="tozeroy",
            fillcolor="rgba(212,175,55,0.12)",
        )
    )
    fig.add_hline(y=2, line_dash="dot", line_color=RED, annotation_text="Red threshold", annotation_position="top left")
    fig.add_hline(y=1, line_dash="dot", line_color=AMBER, annotation_text="Amber threshold", annotation_position="bottom left")
    return base_layout(fig, "Current Alarm Score", height=280)


def build_state_strip(data: pd.DataFrame) -> go.Figure:
    color_map = {"Green": GREEN, "Amber": AMBER, "Red": RED}
    fig = go.Figure(
        data=[
            go.Bar(
                x=data.index,
                y=[1] * len(data),
                marker=dict(color=[color_map.get(state, AMBER) for state in data["dashboard_state"]]),
                hovertext=data["dashboard_state"],
                name="State",
            )
        ]
    )
    fig.update_yaxes(visible=False)
    return base_layout(fig, "State Strip", height=180)


def build_signal_monitor_chart(data: pd.DataFrame, signal_components: pd.DataFrame) -> go.Figure:
    corr_cols = [col for col in signal_components.columns if col.startswith("gold_corr_")]
    relationship_score = signal_components[corr_cols].abs().max(axis=1)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(
        go.Scatter(x=data.index, y=signal_components.loc[data.index, "gold_return_z"].abs(), name="|gold_return_z|", line=dict(color=GOLD)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=signal_components.loc[data.index, "gold_vol_z"].clip(lower=0), name="gold_vol_z", line=dict(color=GOLD_LIGHT)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=signal_components.loc[data.index, "gold_residual_z"].abs(), name="|gold_residual_z|", line=dict(color=BLUE)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=relationship_score.loc[data.index], name="max |corr_z|", line=dict(color=AMBER)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["conditioned_alarm_score"],
            name="Families active",
            marker=dict(color=GOLD),
            opacity=0.8,
        ),
        row=4,
        col=1,
    )

    for row in (1, 2, 3):
        fig.add_hline(y=2, line_dash="dot", line_color=RED, row=row, col=1)

    fig.update_yaxes(title_text="Return / Vol", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_yaxes(title_text="Relationship", row=3, col=1)
    fig.update_yaxes(title_text="Score", row=4, col=1)
    return base_layout(fig, "Signal Families and Alarm Activation", height=760)


def build_relationship_chart(data: pd.DataFrame, signal_components: pd.DataFrame) -> go.Figure:
    corr_cols = [col for col in signal_components.columns if col.startswith("gold_corr_")]
    fig = go.Figure()
    palette = [GOLD, BLUE, AMBER, GREEN]
    for idx, col in enumerate(corr_cols):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=signal_components.loc[data.index, col],
                name=col.replace("gold_corr_", "").replace("_z", ""),
                line=dict(color=palette[idx % len(palette)], width=1.8),
            )
        )
    fig.add_hline(y=2, line_dash="dot", line_color=RED)
    fig.add_hline(y=-2, line_dash="dot", line_color=RED)
    return base_layout(fig, "Relationship Drivers", height=360)


def build_risk_chart(data: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(
        go.Scatter(x=data.index, y=data["R_book"], name="Book return", line=dict(color=GOLD)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["hs_var_return"], name="HS VaR", line=dict(color=RED, dash="dash")),
        row=1,
        col=1,
    )
    if "var_breach" in data.columns:
        breaches = data.loc[data["var_breach"] == 1]
        fig.add_trace(
            go.Scatter(
                x=breaches.index,
                y=breaches["R_book"],
                mode="markers",
                name="VaR breach",
                marker=dict(color=RED, size=7),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(go.Scatter(x=data.index, y=data["nav"], name="NAV", line=dict(color=GOLD_LIGHT)), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["drawdown"], name="Drawdown", line=dict(color=BLUE)), row=3, col=1)
    fig.update_yaxes(title_text="Return", row=1, col=1)
    fig.update_yaxes(title_text="NAV", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown", row=3, col=1, tickformat=".0%")
    return base_layout(fig, "Risk Book and VaR", height=760)


def build_event_study_chart(
    context: ProjectContext,
    episode_name: str,
) -> tuple[go.Figure, pd.DataFrame]:
    start, end = EVENT_TITLES[episode_name]
    window_start = pd.Timestamp(start) - pd.Timedelta(days=25)
    window_end = pd.Timestamp(end) + pd.Timedelta(days=25)

    alarm_slice = context.alarm_frame.loc[window_start:window_end]
    signal_slice = context.signal_components.loc[window_start:window_end]
    risk_slice = context.dashboard_metrics.loc[window_start:window_end]

    corr_cols = [col for col in signal_slice.columns if col.startswith("gold_corr_")]
    relationship_score = signal_slice[corr_cols].abs().max(axis=1)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(
        go.Scatter(x=signal_slice.index, y=signal_slice["gold_return_z"].abs(), name="|return_z|", line=dict(color=GOLD)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=signal_slice.index, y=signal_slice["gold_residual_z"].abs(), name="|residual_z|", line=dict(color=BLUE)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=signal_slice.index, y=relationship_score, name="max|corr_z|", line=dict(color=AMBER)),
        row=1,
        col=1,
    )
    fig.add_hline(y=2, line_dash="dot", line_color=RED, row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=alarm_slice.index,
            y=alarm_slice["conditioned_alarm_score"],
            name="Alarm score",
            marker=dict(color=GOLD),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=2, line_dash="dot", line_color=RED, row=2, col=1)
    fig.add_trace(
        go.Scatter(x=risk_slice.index, y=risk_slice["R_book"], name="Book return", line=dict(color=GOLD)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=risk_slice.index, y=risk_slice["hs_var_return"], name="HS VaR", line=dict(color=RED, dash="dash")),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=risk_slice.index, y=risk_slice["nav"], name="NAV", line=dict(color=GOLD_LIGHT)),
        row=4,
        col=1,
    )

    for boundary in (pd.Timestamp(start), pd.Timestamp(end)):
        for row in range(1, 5):
            fig.add_vline(x=boundary, line_dash="dot", line_color="rgba(240,210,122,0.5)", row=row, col=1)

    fig.update_yaxes(title_text="Signal z", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="Return", row=3, col=1)
    fig.update_yaxes(title_text="NAV", row=4, col=1)

    episode_row = context.named_episode_validation.loc[context.named_episode_validation["episode"] == episode_name]
    return base_layout(fig, f"Episode Study: {episode_name}", height=880), episode_row


def build_baseline_chart(context: ProjectContext) -> go.Figure:
    comparison = context.early_warning_comparison.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comparison["event_family"],
            y=comparison["gold_match_rate_lead_ge1"],
            name="Gold lead>=1",
            marker=dict(color=GOLD),
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparison["event_family"],
            y=comparison["brent_baseline_match_rate_lead_ge1"],
            name="Brent-only lead>=1",
            marker=dict(color=BLUE),
        )
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    return base_layout(fig, "Early-Warning Comparison (Lead >= 1 Day)", height=360)


def build_false_alarm_driver_chart(context: ProjectContext) -> go.Figure:
    drivers = context.relationship_driver_counts.copy()
    fig = go.Figure(
        data=[
            go.Bar(
                x=drivers["relationship_drivers"],
                y=drivers["count"],
                marker=dict(color=AMBER),
                name="False-alarm drivers",
            )
        ]
    )
    fig.update_xaxes(tickangle=-20)
    return base_layout(fig, "False Alarm Drivers (Strict Early-Warning Definition)", height=360)


def build_blind_spot_chart(context: ProjectContext) -> go.Figure:
    patterns = (
        context.blind_spot_patterns.groupby("dominant_pattern")["episode"]
        .count()
        .rename("n_episodes")
        .reset_index()
        .sort_values("n_episodes", ascending=False)
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=patterns["dominant_pattern"],
                y=patterns["n_episodes"],
                marker=dict(color=[RED, BLUE, AMBER][: len(patterns)]),
                name="Blind-spot episodes",
            )
        ]
    )
    return base_layout(fig, "Blind-Spot Pattern Breakdown", height=340)


def build_trade_blotter_frame(context: ProjectContext, as_of: pd.Timestamp) -> pd.DataFrame:
    """Build a trade-blotter table for the current as-of date."""

    if context.data_mode != "live":
        rows = []
        for trade in context.trade_ledger.to_dict("records"):
            signed_bbl = trade["contracts"] * trade.get("lot_size_bbl", riskbook_mod.LOT_SIZE_BBL)
            signed_bbl = signed_bbl if trade["side"] == "long" else -signed_bbl
            rows.append(
                {
                    "trade_id": trade["trade_id"],
                    "side": trade["side"],
                    "contracts": trade["contracts"],
                    "position_bbl": signed_bbl,
                    "entry_date": trade["entry_date"],
                    "entry_price": trade.get("entry_price", "requires live prices") if pd.notna(trade.get("entry_price", np.nan)) else "requires live prices",
                    "current_price": "requires live prices",
                    "unrealized_pnl_usd": "requires live prices",
                    "description": trade.get("description", ""),
                }
            )
        return pd.DataFrame(rows)

    return riskbook_mod.trade_blotter(None, context.prices, as_of=as_of)


def build_headline_bar_chart() -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Named episodes caught", "Stress coverage", "False review rate", "Reviews / year"],
                y=[9, 0.68, 0.309, 10],
                marker=dict(color=[GOLD, GOLD_LIGHT, BLUE, AMBER]),
                text=["9/9", "68.0%", "30.9%", "about 10"],
                textposition="outside",
                name="Headline metrics",
            )
        ]
    )
    fig.update_yaxes(showticklabels=False)
    return base_layout(fig, "Headline Project Results", height=340)


def build_timeline_summary_chart(context: ProjectContext, selected_date: pd.Timestamp) -> go.Figure:
    """Compact timeline summary for the book state and signal flare state."""

    timeline = context.timeline_frame.copy()
    if timeline.empty:
        return base_layout(go.Figure(), "Timeline", height=220)

    if context.data_mode == "live":
        lookback_days = st.session_state.get("timeline_lookback_days", 180)
        start_date = pd.Timestamp(selected_date) - pd.Timedelta(days=int(lookback_days))
        timeline = timeline.loc[start_date:selected_date].copy()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        fig.add_trace(
            go.Scatter(
                x=timeline.index,
                y=timeline["nav"],
                name="NAV",
                line=dict(color=GOLD_LIGHT, width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=timeline.index,
                y=timeline["conditioned_alarm_score"],
                name="Alarm score",
                line=dict(color=GOLD, width=2),
                fill="tozeroy",
                fillcolor="rgba(212,175,55,0.10)",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=2, line_dash="dot", line_color=RED, row=2, col=1)
        x_value = selected_date
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=timeline["date"],
                    y=list(range(1, len(timeline) + 1)),
                    mode="markers+lines+text",
                    text=timeline["label"],
                    textposition="top center",
                    marker=dict(size=12, color=GOLD),
                    line=dict(color=MUTED, width=2),
                    name="Saved milestones",
                )
            ]
        )
        fig.update_yaxes(visible=False)
        x_value = selected_date

    fig.add_vline(x=x_value, line_dash="dash", line_color=BLUE)
    return base_layout(fig, "Book Timeline", height=300 if context.data_mode == "live" else 240)


def build_stress_summary_card(summary: dict[str, object], scenario_label: str, scenario_note: str) -> str:
    return f"""
        <div class="card">
            <strong>Scenario selected:</strong> {scenario_label}<br/><br/>
            {scenario_note}<br/><br/>
            <strong>Stress PnL:</strong> {fmt_num(summary.get('stress_pnl_usd'), 0)} USD<br/>
            <strong>Stressed NAV:</strong> {fmt_num(summary.get('nav_after_stress'), 2)}<br/>
            <strong>MTM-style change:</strong> {fmt_num(summary.get('mtm_style_change_usd'), 0)} USD<br/>
            <strong>Cash need:</strong> {fmt_num(summary.get('cash_need_usd'), 0)} USD
        </div>
    """


def render_scenario_workbench(
    context: ProjectContext,
    as_of: pd.Timestamp,
    key_prefix: str,
    title: str,
    intro: str,
    default_loss_limit: float = 5_000_000.0,
) -> None:
    if context.data_mode != "live":
        st.markdown(intro)
        st.dataframe(
            context.stress_results.rename(
                columns={
                    "scenario_name": "Scenario",
                    "shock_pct": "Brent shock",
                    "stress_pnl_usd": "Stress PnL",
                    "stress_return": "Stress return",
                    "nav_after_stress": "Stressed NAV",
                    "cash_need_usd": "Cash need",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        return

    st.markdown("<div class='section-title'>" + title + "</div>", unsafe_allow_html=True)
    st.markdown(intro)

    ctrl_top_left, ctrl_top_mid, ctrl_top_right = st.columns([1.1, 1.0, 1.0], gap="large")
    preset_name = ctrl_top_left.selectbox(
        "Event scenario",
        options=list(SCENARIO_PRESETS.keys()),
        index=0,
        key=f"{key_prefix}_preset_name",
    )
    preset = SCENARIO_PRESETS[preset_name]
    shock_default = float(preset["shock_pct"])
    shock_value = ctrl_top_mid.slider(
        "Brent move under this event",
        min_value=-0.30,
        max_value=0.30,
        value=float(shock_default),
        step=0.01,
        format="%.2f",
        key=f"{key_prefix}_shock_pct",
    )
    loss_limit = ctrl_top_right.number_input(
        "Loss limit for reverse-stress flag (USD)",
        min_value=0.0,
        value=default_loss_limit,
        step=500_000.0,
        key=f"{key_prefix}_loss_limit",
    )

    ctrl_low_left, ctrl_low_mid, ctrl_low_right = st.columns([1.0, 1.0, 1.0], gap="large")
    initial_margin = ctrl_low_left.number_input(
        "Initial margin per Brent contract (USD, optional)",
        min_value=0.0,
        value=0.0,
        step=500.0,
        key=f"{key_prefix}_initial_margin",
    )
    margin_multiplier = ctrl_low_mid.slider(
        "Stressed margin multiplier",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        key=f"{key_prefix}_margin_multiplier",
    )
    show_full_pack = ctrl_low_right.checkbox(
        "Also show standard scenario pack",
        value=True,
        key=f"{key_prefix}_show_full_pack",
    )

    custom_table = stress_mod.run_stress_scenarios(
        context.book,
        context.prices,
        fixed_shock_pcts=[float(shock_value)],
        loss_limit_usd=loss_limit if loss_limit > 0 else None,
        initial_margin_per_contract_usd=initial_margin if initial_margin > 0 else None,
        margin_multiplier=margin_multiplier if initial_margin > 0 else None,
        as_of=as_of,
    )
    custom_row = custom_table.iloc[0].copy()
    mtm_style_change_usd = float(custom_row["stress_pnl_usd"])
    summary = {
        "stress_pnl_usd": float(custom_row["stress_pnl_usd"]),
        "nav_after_stress": float(custom_row["nav_after_stress"]),
        "mtm_style_change_usd": mtm_style_change_usd,
        "cash_need_usd": float(custom_row["cash_need_usd"]),
        "stress_return": float(custom_row["stress_return"]),
    }

    metric_cols = st.columns(4)
    metric_cols[0].metric("Stress PnL", fmt_num(summary["stress_pnl_usd"], 0))
    metric_cols[1].metric("Stressed NAV", fmt_num(summary["nav_after_stress"], 2))
    metric_cols[2].metric("MTM-style change", fmt_num(summary["mtm_style_change_usd"], 0))
    metric_cols[3].metric("Cash need", fmt_num(summary["cash_need_usd"], 0))

    detail_left, detail_right = st.columns([1.2, 1.0], gap="large")
    with detail_left:
        scenario_frame = custom_table.rename(
            columns={
                "scenario_name": "Scenario",
                "shock_pct": "Brent shock",
                "stress_pnl_usd": "Stress PnL",
                "stress_return": "Stress return",
                "nav_after_stress": "Stressed NAV",
                "cash_need_usd": "Cash need",
                "breach_flag": "Breach flag",
            }
        )[
            ["Scenario", "Brent shock", "Stress PnL", "Stress return", "Stressed NAV", "Cash need", "Breach flag"]
        ]
        st.dataframe(scenario_frame, use_container_width=True, hide_index=True)
    with detail_right:
        st.markdown(
            build_stress_summary_card(
                summary,
                scenario_label=preset_name,
                scenario_note=str(preset["headline"]),
            ),
            unsafe_allow_html=True,
        )

        reverse = stress_mod.reverse_stress_to_loss_limit(
            context.book,
            loss_limit_usd=loss_limit if loss_limit > 0 else 0.0,
            as_of=as_of,
        )
        reverse_row = reverse.iloc[0]
        st.markdown(
            f"""
            <div class="card">
                <strong>Reverse-stress reading</strong><br/><br/>
                To lose {fmt_num(loss_limit, 0)} USD from this book on {fmt_date(as_of)},
                Brent would need to move about <strong>{fmt_pct(reverse_row['shock_pct'])}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if show_full_pack:
        full_pack = stress_mod.run_stress_scenarios(
            context.book,
            context.prices,
            loss_limit_usd=loss_limit if loss_limit > 0 else None,
            initial_margin_per_contract_usd=initial_margin if initial_margin > 0 else None,
            margin_multiplier=margin_multiplier if initial_margin > 0 else None,
            as_of=as_of,
        )
        st.caption(
            "The selected event scenario above reprices the book immediately. The standard scenario pack below stays visible so you can compare that event against the broader stress policy set."
        )
        st.dataframe(
            full_pack.rename(
                columns={
                    "scenario_name": "Scenario",
                    "shock_pct": "Brent shock",
                    "stress_pnl_usd": "Stress PnL",
                    "stress_return": "Stress return",
                    "nav_after_stress": "Stressed NAV",
                    "cash_need_usd": "Cash need",
                    "breach_flag": "Breach flag",
                }
            )[
                ["Scenario", "Brent shock", "Stress PnL", "Stress return", "Stressed NAV", "Cash need", "Breach flag"]
            ],
            use_container_width=True,
            hide_index=True,
        )


def build_method_flow_chart() -> go.Figure:
    fig = go.Figure()
    steps = [
        ("Gold abnormality", 1, GOLD),
        ("3 signal families", 2, GOLD_LIGHT),
        ("Escalation state", 3, AMBER),
        ("VaR review", 4, BLUE),
        ("Stress testing", 5, RED),
    ]
    for label, x, color in steps:
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[1],
                mode="markers+text",
                marker=dict(size=42, color=color, line=dict(color=TEXT, width=1)),
                text=[label],
                textposition="bottom center",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for x0, x1 in zip(range(1, 5), range(2, 6)):
        fig.add_annotation(
            x=x1 - 0.1,
            y=1,
            ax=x0 + 0.1,
            ay=1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor=MUTED,
        )
    fig.update_xaxes(visible=False, range=[0.5, 5.5])
    fig.update_yaxes(visible=False, range=[0.7, 1.3])
    return base_layout(fig, "Project Logic: signal first, action second", height=260)


def build_layer_chart() -> go.Figure:
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(PROJECT_LAYERS.columns),
                    fill_color=SURFACE_ALT,
                    font=dict(color=GOLD_LIGHT, size=13),
                    align="left",
                ),
                cells=dict(
                    values=[PROJECT_LAYERS[col] for col in PROJECT_LAYERS.columns],
                    fill_color=[[SURFACE, SURFACE_ALT, SURFACE]],
                    font=dict(color=TEXT, size=12),
                    align="left",
                    height=34,
                ),
            )
        ]
    )
    return base_layout(fig, "Three-layer research framing", height=300)


def build_key_takeaway_chart() -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Brent-only: false alarms", "Gold: false alarms", "Brent-only: avg lead", "Gold: avg lead"],
                y=[0.134, 0.323, 2.86, 8.18],
                marker=dict(color=[BLUE, GOLD, BLUE, GOLD]),
                text=["13.4%", "32.3%", "2.9d", "8.2d"],
                textposition="outside",
                name="Why Gold adds value",
            )
        ]
    )
    return base_layout(fig, "Cleaner vs earlier: the core trade-off", height=340)


def render_hero(context: ProjectContext) -> None:
    metrics = context.overview_metrics
    mode_label = {
        "live": "Live Mode",
        "static_notebook": "Static Notebook Mode",
    }.get(context.data_mode, "Dashboard Mode")
    st.markdown(
        f"""
        <div class="hero">
            <h1>{TITLE}</h1>
            <p>
                <strong>Current status:</strong> {state_badge(metrics["dashboard_state"])}
                Score {metrics["conditioned_alarm_score"]} / 3 families.
                The dashboard is designed to show <em>state first</em>, then <em>action</em>, then the supporting
                evidence and boundaries.
            </p>
            <p class="small-note">
                <strong>{mode_label}:</strong> {context.provenance_note}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_story_header(step: str, title: str, subtitle: str, takeaway: str) -> None:
    """Render a slide-like section header with a clear takeaway."""

    st.markdown(
        f"""
        <div class="story-header">
            <div class="story-kicker">{step}</div>
            <div class="story-title">{title}</div>
            <div class="story-subtitle">{subtitle}</div>
            <div class="takeaway"><strong>Takeaway:</strong> {takeaway}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, takeaway: str | None = None) -> None:
    """Render a stronger page-level header for the 2-page dashboard."""

    takeaway_html = (
        f"<div class='takeaway'><strong>Takeaway:</strong> {takeaway}</div>" if takeaway else ""
    )
    st.markdown(
        f"""
        <div class="story-header">
            <div class="story-title">{title}</div>
            <div class="story-subtitle">{subtitle}</div>
            {takeaway_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_focus_banner(context: ProjectContext, as_of: pd.Timestamp) -> None:
    """Render a compact focus banner so viewers know what context they are seeing."""

    if context.data_mode == "live":
        st.markdown(
            f"""
            <div class="card">
                <strong>Focus:</strong> Live market snapshot as of {fmt_date(as_of)}.
                The monitor page below is showing the latest dashboard state, trigger logic, and risk translation for that selected date.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    focus = get_static_focus(context)
    selected = static_selection_context(context, focus["date"])
    episode_html = ""
    if selected["episode"]:
        episode_html = f" <strong>Episode:</strong> {selected['episode']}."
    st.markdown(
        f"""
        <div class="card">
            <strong>Focus:</strong> {focus['mode']} for {fmt_date(pd.Timestamp(selected['selected_date']))}.{episode_html}
            <span class="small-note">{selected['note']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_story_intro(context: ProjectContext) -> None:
    render_hero(context)
    render_story_header(
        step="Slide 1",
        title="Why Gold? Reframing the Project as Risk Escalation, Not Price Prediction",
        subtitle=(
            "This project asks whether Gold can warn a commodity risk manager that the current market regime "
            "is changing before traditional Brent-centric risk measures fully react."
        ),
        takeaway=(
            "The project is about earlier risk review and stress escalation, not about claiming Gold directly predicts Brent losses."
        ),
    )
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        st.markdown("<div class='section-title'>Core question</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card">
                Can abnormal Gold behaviour act as a useful cross-market escalation signal
                that prompts timely review of VaR and stress-test exposure in a commodity risk book?
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_method_flow_chart(), use_container_width=True)
    with right:
        st.markdown("<div class='section-title'>What the project does not claim</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card">
                <ul class="boundary-list">
                    <li>It does not claim Gold directly predicts Brent losses.</li>
                    <li>It does not replace a physical-book valuation model.</li>
                    <li>It does not solve isolated Brent shocks or slow cumulative drawdowns.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_layer_chart(), use_container_width=True)


def render_story_method(context: ProjectContext) -> None:
    render_story_header(
        step="Slide 2",
        title="How the Alarm Works",
        subtitle=(
            "The model combines Gold's own abnormal behaviour, Gold's residual surprise, and Gold's cross-market relationship instability into a simple escalation rule."
        ),
        takeaway=(
            "The dashboard turns three interpretable signal families into an action rule: Green for normal monitoring, Amber for inspection, and Red for VaR review plus stress testing."
        ),
    )
    col1, col2 = st.columns([1.1, 0.9], gap="large")
    with col1:
        st.markdown("<div class='section-title'>Inputs and signal families</div>", unsafe_allow_html=True)
        st.markdown(
            r"""
            <div class="formula">
                <strong>Inputs</strong><br/>
                Gold futures, Brent futures, DXY, VIX, US10Y
            </div>
            <div class="formula">
                <strong>Family 1: own-shock</strong><br/>
                <code>|gold_return_z| &gt; 2</code> or <code>gold_vol_z &gt; 2</code>
            </div>
            <div class="formula">
                <strong>Family 2: residual surprise</strong><br/>
                <code>|gold_residual_z| &gt; 2</code>
            </div>
            <div class="formula">
                <strong>Family 3: relationship instability</strong><br/>
                <code>max(|corr_z|) &gt; 2</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("<div class='section-title'>Dashboard rule</div>", unsafe_allow_html=True)
        st.markdown(
            r"""
            <div class="formula">
                <strong>Conditioned score</strong><br/>
                <code>conditioned_alarm_score = 1{return/vol} + 1{residual} + 1{relationship}</code>
            </div>
            <div class="formula">
                <strong>State mapping</strong><br/>
                Green = 0 families<br/>
                Amber = 1 family<br/>
                Red = 2 or more families
            </div>
            <div class="formula">
                <strong>Interpretation</strong><br/>
                This is an escalation rule: it tells the desk when the current environment may no longer be well represented by trailing risk measures.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_story_results(context: ProjectContext) -> None:
    render_story_header(
        step="Slide 3",
        title="Main Results at a Glance",
        subtitle=(
            "Before discussing blind spots or implementation, the audience should see the few numbers that summarize the empirical case for the dashboard."
        ),
        takeaway=(
            "Gold catches the major stress regimes, covers a meaningful share of broader stress events, and does so with a review burden that is operationally plausible."
        ),
    )
    top_left, top_right = st.columns([1.0, 1.0], gap="large")
    with top_left:
        st.plotly_chart(build_headline_bar_chart(), use_container_width=True)
    with top_right:
        st.markdown("<div class='section-title'>Headline results</div>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {"Result": "Named stress episodes caught", "Value": "9 / 9"},
                    {"Result": "Stress-proxy coverage", "Value": "68.0%"},
                    {"Result": "False review rate", "Value": "30.9%"},
                    {"Result": "Reviews per year", "Value": "about 10"},
                    {"Result": "Kupiec p-value", "Value": "0.015"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            """
            <div class="card">
                <strong>Main message</strong><br/>
                Gold is useful because it gives earlier cross-market warning, not because it cleanly predicts every Brent risk event.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_story_comparison(context: ProjectContext) -> None:
    render_story_header(
        step="Slide 4",
        title="Why Gold Adds Value Beyond a Brent-Only Alarm",
        subtitle=(
            "A fair benchmark is a Brent-only warning rule. It is cleaner and closer to the commodity itself, so if Gold still helps, it must help through timing."
        ),
        takeaway=(
            "Brent-only is cleaner for confirmation, but Gold is stronger for early warning because it gives longer lead time before VaR-style events."
        ),
    )
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        st.plotly_chart(build_key_takeaway_chart(), use_container_width=True)
    with right:
        st.dataframe(
            context.early_warning_comparison.rename(
                columns={
                    "event_family": "Event family",
                    "gold_match_rate_lead_ge1": "Gold match rate (lead>=1)",
                    "brent_baseline_match_rate_lead_ge1": "Brent-only match rate (lead>=1)",
                    "gold_avg_lead_days": "Gold avg lead",
                    "brent_baseline_avg_lead_days": "Brent-only avg lead",
                }
            )[
                [
                    "Event family",
                    "Gold match rate (lead>=1)",
                    "Brent-only match rate (lead>=1)",
                    "Gold avg lead",
                    "Brent-only avg lead",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            """
            <div class="card">
                <strong>Interpretation</strong><br/>
                Brent-only is better for contemporaneous confirmation of commodity stress. Gold carries more false reviews, but it gives materially longer lead time and broader macro-regime warning.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_story_limits(context: ProjectContext) -> None:
    render_story_header(
        step="Slide 5",
        title="Where the Dashboard Fails and What the False Alarms Mean",
        subtitle=(
            "The project is stronger when it acknowledges its own boundaries clearly: some events are structurally hard to warn about, and broader monitoring naturally creates review noise."
        ),
        takeaway=(
            "The biggest misses are isolated Brent shocks and slow drawdowns, while many false alarms still reflect real macro relationship breaks rather than meaningless noise."
        ),
    )
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        st.plotly_chart(build_blind_spot_chart(context), use_container_width=True)
        st.markdown(
            """
            <div class="card">
                <strong>Blind-spot message</strong><br/>
                The largest missed-event pattern is the isolated single-day Brent move. These are difficult because the move itself is the event.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.plotly_chart(build_false_alarm_driver_chart(context), use_container_width=True)
        st.markdown(
            """
            <div class="card">
                <strong>False-alarm message</strong><br/>
                Many false alarms are linked to US10Y and VIX relationship breaks. They are not pure noise; they often reflect broader macro stress that does not turn into a Brent-book event within the evaluation window.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_story_conclusion(context: ProjectContext) -> None:
    render_story_header(
        step="Slide 6",
        title="Final Claim and Desk Use",
        subtitle=(
            "The close-out should translate the statistical findings into a practical desk workflow: when the dashboard fires, what exactly should a risk manager do next?"
        ),
        takeaway=(
            "The correct use of this project is as a cross-market escalation trigger that prompts Brent-book VaR review and stress testing, not as a standalone loss predictor."
        ),
    )
    st.markdown(
        """
        <div class="card">
            <strong>Final claim</strong><br/><br/>
            This project does not show that Gold predicts Brent losses directly. It shows that abnormal Gold behaviour can act as a useful cross-market escalation signal. A Brent-only alarm is cleaner and better for contemporaneous confirmation of commodity stress, but it gives little meaningful advance warning because Brent often moves when the risk event is already happening. Gold has a higher false-review burden, but provides materially longer lead time and captures broader macro-regime shifts.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-title'>Desk workflow implication</div>", unsafe_allow_html=True)
    st.dataframe(PROJECT_LAYERS, use_container_width=True, hide_index=True)


def render_overview(context: ProjectContext, as_of: pd.Timestamp, lookback_days: int, show_header: bool = True) -> None:
    if context.data_mode != "live":
        render_static_overview(context)
        return

    if show_header:
        render_story_header(
            step="Support View",
            title="Current Dashboard State",
            subtitle=(
                "This is the monitoring-style support page. It answers the practical follow-up question: what is the dashboard saying right now?"
            ),
            takeaway=(
                "Once the audience understands the project story, this page shows the latest state, the active signal families, and the immediate action recommendation."
            ),
        )

    latest = context.dashboard_metrics.loc[as_of]
    latest_alarm = context.alarm_frame.loc[as_of]
    signal_row = context.signal_components.loc[as_of]
    window_start = as_of - pd.Timedelta(days=lookback_days)
    overview_slice = context.dashboard_metrics.loc[window_start:as_of].join(
        context.alarm_frame[["conditioned_alarm_score", "dashboard_state"]],
        how="left",
    )

    active_families = [
        label
        for label, flag in (
            ("return/vol", latest_alarm["return_or_vol_alarm"]),
            ("residual", latest_alarm["residual_alarm"]),
            ("relationship", latest_alarm["relationship_alarm"]),
        )
        if int(flag) == 1
    ]
    active_families_text = ", ".join(active_families) if active_families else "none"

    left, right = st.columns([1.3, 1.0], gap="large")
    with left:
        st.markdown("<div class='section-title'>Action-first status</div>", unsafe_allow_html=True)
        card_col1, card_col2, card_col3, card_col4 = st.columns(4)
        card_col1.metric("Dashboard state", latest_alarm["dashboard_state"])
        card_col2.metric("Alarm score", int(latest_alarm["conditioned_alarm_score"]))
        card_col3.metric("Active families", active_families_text)
        card_col4.metric("As of", fmt_date(as_of))

        action_col1, action_col2, action_col3 = st.columns(3)
        action_col1.metric("Recommended action", latest_alarm["recommended_action"])
        action_col2.metric("Book NAV", fmt_num(float(latest["nav"]), 2))
        action_col3.metric("Drawdown", fmt_pct(float(latest["drawdown"])))

        st.plotly_chart(build_score_chart(overview_slice), use_container_width=True)
        st.plotly_chart(build_state_strip(overview_slice), use_container_width=True)

    with right:
        st.markdown("<div class='section-title'>What moved this signal?</div>", unsafe_allow_html=True)
        corr_cols = [col for col in context.signal_components.columns if col.startswith("gold_corr_")]
        strongest = signal_row[corr_cols].abs().idxmax()
        strongest_name = strongest.replace("gold_corr_", "").replace("_z", "")

        st.markdown(
            f"""
            <div class="card">
                <p><strong>Active families:</strong> {active_families_text}</p>
                <p><strong>Strongest relationship distortion:</strong> {strongest_name}</p>
                <p><strong>Relationship z-score:</strong> {fmt_num(float(signal_row[strongest]), 2)}</p>
                <p><strong>Current HS VaR (return):</strong> {fmt_pct(float(latest["hs_var_return"]))}</p>
                <p><strong>Current HS ES (return):</strong> {fmt_pct(float(latest["hs_es_return"]))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-title'>Validation headline</div>", unsafe_allow_html=True)
        top1, top2, top3 = st.columns(3)
        top1.metric(
            "Named episodes caught",
            f'{context.overview_metrics["named_episodes_caught"]}/{context.overview_metrics["named_episodes_total"]}',
        )
        top2.metric("Stress coverage", fmt_pct(context.overview_metrics["stress_event_coverage"]))
        top3.metric("False review rate", fmt_pct(context.overview_metrics["false_review_rate"]))
        top4, top5 = st.columns(2)
        top4.metric("Reviews / year", fmt_num(context.overview_metrics["reviews_per_year"], 1))
        top5.metric("Gold alarm rate", fmt_pct(context.overview_metrics["gold_alarm_rate"]))

        st.markdown("<div class='section-title'>Where it works and where it fails</div>", unsafe_allow_html=True)
        success_col, boundary_col = st.columns(2, gap="medium")
        with success_col:
            st.markdown(
                """
                <div class="card">
                    <strong>Best use cases</strong>
                    <ul class="boundary-list">
                        <li>Broad macro risk shifts</li>
                        <li>Cross-market correlation breaks</li>
                        <li>VaR recalibration prompts before full stress arrives</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with boundary_col:
            st.markdown(
                """
                <div class="card">
                    <strong>Known boundaries</strong>
                    <ul class="boundary-list">
                        <li>Isolated single-day Brent shocks</li>
                        <li>Slow cumulative drawdowns</li>
                        <li>False reviews from broader macro monitoring</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_signal_monitor(context: ProjectContext, as_of: pd.Timestamp, lookback_days: int, show_header: bool = True) -> None:
    if context.data_mode != "live":
        render_static_signal_monitor(context)
        return

    if show_header:
        render_story_header(
            step="Support View",
            title="Signal Decomposition",
            subtitle=(
                "This page is the model drill-down: it shows exactly which family is active and whether the dashboard is being driven by Gold's own move, residual surprise, or relationship stress."
            ),
            takeaway=(
                "The audience should come away seeing that the dashboard is interpretable; every escalation can be traced back to a specific signal family."
            ),
        )

    window_start = as_of - pd.Timedelta(days=lookback_days)
    data = context.dashboard_metrics.loc[window_start:as_of].copy()
    if "conditioned_alarm_score" not in data.columns:
        data = data.join(context.alarm_frame[["conditioned_alarm_score"]], how="left")

    st.markdown(
        "The signal monitor explains *why* the dashboard is currently Green, Amber, or Red. "
        "It decomposes the alarm into Gold's own shock, residual surprise, and relationship instability."
    )
    st.markdown(
        """
        <div class="card">
            <strong>How to read these charts</strong><br/><br/>
            The top panel shows the book's NAV over the selected lookback window. The lower panel shows the conditioned alarm score.
            The orange/red dotted threshold at <strong>2</strong> is the escalation threshold:
            when the score reaches 2 or more, at least two signal families are active and the dashboard turns <strong>Red</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_signal_monitor_chart(data, context.signal_components),
        use_container_width=True,
    )

    lower_left, lower_right = st.columns([1.1, 0.9], gap="large")
    with lower_left:
        st.plotly_chart(build_relationship_chart(data, context.signal_components), use_container_width=True)
    with lower_right:
        latest_alarm = context.alarm_frame.loc[as_of]
        st.markdown("<div class='section-title'>Current model logic</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="formula">
                <strong>Signal families</strong><br/>
                1. return/vol: <code>|gold_return_z| &gt; 2</code> or <code>gold_vol_z &gt; 2</code><br/>
                2. residual: <code>|gold_residual_z| &gt; 2</code><br/>
                3. relationship: <code>max(|corr_z|) &gt; 2</code>
            </div>
            <div class="formula">
                <strong>Escalation rule</strong><br/>
                <code>conditioned_alarm_score = 1{{return/vol}} + 1{{residual}} + 1{{relationship}}</code><br/>
                <code>Red if conditioned_alarm_score &gt;= 2</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"family": "return/vol", "active": int(latest_alarm["return_or_vol_alarm"])},
                    {"family": "residual", "active": int(latest_alarm["residual_alarm"])},
                    {"family": "relationship", "active": int(latest_alarm["relationship_alarm"])},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_risk_and_stress(context: ProjectContext, as_of: pd.Timestamp, lookback_days: int, show_header: bool = True) -> None:
    if context.data_mode != "live":
        render_static_risk_and_stress(context)
        return

    if show_header:
        render_story_header(
            step="Support View",
            title="Risk and Stress Translation",
            subtitle=(
                "This page connects the signal to what a risk desk actually does: examine VaR, inspect drawdown and volatility, and run scenario losses."
            ),
            takeaway=(
                "The signal only matters because it changes behaviour: a Red alarm should trigger a concrete risk review workflow."
            ),
        )

    window_start = as_of - pd.Timedelta(days=lookback_days)
    risk_slice = context.dashboard_metrics.loc[window_start:as_of]

    st.markdown(
        "This section ties the signal to action. The dashboard does not stop at saying that Gold looks unusual; "
        "it shows what that means for the Brent proxy risk book, historical-simulation VaR, and scenario losses."
    )
    st.plotly_chart(build_risk_chart(risk_slice), use_container_width=True)
    stress_table = stress_mod.run_stress_scenarios(context.book, context.prices, as_of=as_of)
    summary_cols = st.columns(4)
    down10 = stress_table.loc[stress_table["shock_pct"].eq(-0.10)]
    up10 = stress_table.loc[stress_table["shock_pct"].eq(0.10)]
    worst = stress_table.nsmallest(1, "stress_pnl_usd").iloc[0]
    summary_cols[0].metric("Worst scenario", str(worst["scenario_name"]))
    summary_cols[1].metric("Worst stress PnL", fmt_num(float(worst["stress_pnl_usd"]), 0))
    summary_cols[2].metric(
        "-10% Brent stress",
        fmt_num(float(down10["stress_pnl_usd"].iloc[0]), 0) if not down10.empty else "N/A",
    )
    summary_cols[3].metric(
        "+10% Brent stress",
        fmt_num(float(up10["stress_pnl_usd"].iloc[0]), 0) if not up10.empty else "N/A",
    )
    st.dataframe(
        stress_table.rename(
            columns={
                "scenario_name": "Scenario",
                "shock_pct": "Brent shock",
                "stress_pnl_usd": "Stress PnL",
                "stress_return": "Stress return",
                "nav_after_stress": "Stressed NAV",
                "cash_need_usd": "Cash need",
                "breach_flag": "Breach flag",
            }
        )[
            ["Scenario", "Brent shock", "Stress PnL", "Stress return", "Stressed NAV", "Cash need", "Breach flag"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "This monitor-page snapshot is read-only by design. Use Incident Drilldown for the full interactive event-response workbench."
    )


def render_episode_explorer(context: ProjectContext, show_header: bool = True) -> None:
    if context.data_mode != "live":
        render_static_episode_explorer(context)
        return

    if show_header:
        render_story_header(
            step="Support View",
            title="Named Event Study",
            subtitle=(
                "This is the case-study page. It visualizes the sequence from signal change to alarm escalation to Brent-book stress, with COVID 2020 as the clearest example."
            ),
            takeaway=(
                "Event studies make the timing argument intuitive: the dashboard is useful when it moves before the risk event, not after it."
            ),
        )

    default_episode = "COVID crash + oil war" if "COVID crash + oil war" in EVENT_TITLES else list(EVENT_TITLES)[0]
    episode_name = st.selectbox("Named episode", options=list(EVENT_TITLES), index=list(EVENT_TITLES).index(default_episode))
    chart, episode_summary = build_event_study_chart(context, episode_name)

    st.markdown(
        "Episode Explorer shows the dashboard as a sequence: first the cross-market signal changes, then the alarm "
        "escalates, and only later do risk-book events such as VaR breaches arrive."
    )
    st.plotly_chart(chart, use_container_width=True)

    if not episode_summary.empty:
        row = episode_summary.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alarm fired?", "Yes" if bool(row["alarm_fired"]) else "No")
        c2.metric("First alarm", fmt_date(row["first_alarm_date"]))
        c3.metric("Alarm count", int(row["alarm_count"]))
        c4.metric("Window alarm rate", fmt_pct(float(row["alarm_rate"])))


def render_diagnostics_and_method(context: ProjectContext, show_header: bool = True) -> None:
    if show_header:
        render_story_header(
            step="Appendix",
            title="Diagnostics and Method Appendix",
            subtitle=(
                "This appendix keeps the deeper evidence visible: baseline comparison, false-alarm drivers, blind spots, and the core formulas."
            ),
            takeaway=(
                "The appendix is where you defend the project under questioning: it shows the logic is auditable and the limitations are already understood."
            ),
        )

    top_left, top_right = st.columns([1.0, 1.0], gap="large")
    with top_left:
        st.plotly_chart(build_baseline_chart(context), use_container_width=True)
        st.dataframe(
            context.baseline_overall.rename(
                columns={
                    "signal": "Signal",
                    "alarm_count": "Alarm count",
                    "alarm_rate": "Alarm rate",
                    "false_alarm_rate": "False alarm rate",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with top_right:
        st.plotly_chart(build_blind_spot_chart(context), use_container_width=True)
        st.dataframe(
            context.blind_spot_summary.rename(
                columns={
                    "event_family": "Event family",
                    "Blind spot": "Blind spot",
                    "Gold only": "Gold only",
                    "Brent baseline only": "Brent-only",
                    "Both": "Both",
                    "blind_spot_rate": "Blind-spot rate",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    lower_left, lower_right = st.columns([1.0, 1.0], gap="large")
    with lower_left:
        st.plotly_chart(build_false_alarm_driver_chart(context), use_container_width=True)
        st.dataframe(
            context.false_alarm_trigger_breakdown.rename(
                columns={
                    "trigger_families": "Trigger families",
                    "false_alarm_count": "False alarms",
                    "avg_alarm_score": "Avg score",
                    "brent_concurrent_rate": "Brent concurrent rate",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with lower_right:
        st.dataframe(
            context.driver_redemption.rename(
                columns={
                    "driver": "Driver",
                    "alarm_count": "Alarm count",
                    "genuine_catches": "Genuine catches",
                    "genuine_catch_rate": "Catch rate",
                    "avg_lead_days": "Avg lead days",
                    "median_lead_days": "Median lead days",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<div class='section-title'>Model formulas and interpretation</div>", unsafe_allow_html=True)
    st.markdown(
        r"""
        <div class="formula">
            <strong>Trailing anomaly score</strong><br/>
            \( z_t = \dfrac{x_t - \mu^{(252)}_{t-1}}{\sigma^{(252)}_{t-1}} \)<br/>
            All z-scores use shifted trailing windows to avoid look-ahead bias.
        </div>
        <div class="formula">
            <strong>Residual surprise</strong><br/>
            \( \varepsilon_t = r_{\text{Gold},t} - \hat{r}_{\text{Gold},t \mid \text{Brent}, \text{DXY}, \text{VIX}, \text{US10Y}} \)<br/>
            A large residual means Gold is moving in a way the current macro environment does not explain.
        </div>
        <div class="formula">
            <strong>Alarm rule</strong><br/>
            Red if at least two of the three families are active:<br/>
            return/vol, residual, relationship.
        </div>
        <div class="formula">
            <strong>Risk validation</strong><br/>
            Historical-simulation VaR uses only prior returns:<br/>
            \( VaR^{95\%}_t = Q_{5\%}(R_{book, t-250:t-1}) \)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Boundaries to keep in mind</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card'><ul class='boundary-list'>"
        + "".join(f"<li>{message}</li>" for message in context.boundaries)
        + "</ul></div>",
        unsafe_allow_html=True,
    )


def render_monitor_page(context: ProjectContext, as_of: pd.Timestamp, lookback_days: int) -> None:
    """Book Timeline page: the trade book through time with signal flares."""

    render_page_header(
        title="Book Timeline",
        subtitle=(
            "A time-driven monitor for the Brent proxy trade book: follow positions, risk, and Gold signal flares through time, then drill into any flare as a candidate stress-review event."
        ),
        takeaway=(
            "Read this page left to right: book state, risk state, flare timeline, then action. The signal is useful only when it changes what you do with the book."
        ),
    )
    render_focus_banner(context, as_of)

    reference_as_of = as_of - pd.Timedelta(days=20)
    if context.data_mode == "live":
        reference_as_of = context.dashboard_metrics.index.asof(reference_as_of)
        if pd.isna(reference_as_of):
            reference_as_of = context.dashboard_metrics.index[0]

    st.markdown("<div class='section-title'>Timeline Control</div>", unsafe_allow_html=True)
    if context.data_mode == "live":
        selectable_dates = list(context.timeline_frame.index)
        chosen_date = st.select_slider(
            "Scroll through the book timeline",
            options=selectable_dates,
            value=as_of,
            format_func=lambda x: fmt_date(pd.Timestamp(x)),
        )
        as_of = pd.Timestamp(chosen_date)
    else:
        selectable_dates = [pd.Timestamp(x) for x in context.timeline_frame["date"]]
        default_idx = min(range(len(selectable_dates)), key=lambda i: abs((selectable_dates[i] - as_of).days))
        chosen_date = st.select_slider(
            "Scroll through saved timeline milestones",
            options=selectable_dates,
            value=selectable_dates[default_idx],
            format_func=lambda x: fmt_date(pd.Timestamp(x)),
        )
        st.session_state["static_as_of_date"] = str(pd.Timestamp(chosen_date).date())
        as_of = pd.Timestamp(chosen_date)

    st.session_state["timeline_lookback_days"] = lookback_days
    st.markdown(
        """
        <div class="card">
            <strong>Timeline meaning</strong><br/><br/>
            Use the date slider to move left and right through the book history. The timeline chart below always ends at the selected date
            and shows only the chosen lookback window, so changing the lookback directly changes how much history is visible around that state.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(build_timeline_summary_chart(context, as_of), use_container_width=True)

    if context.data_mode == "live":
        st.markdown("<div class='section-title'>Monitoring Reference</div>", unsafe_allow_html=True)
        ref_col1, ref_col2 = st.columns([1.0, 1.3], gap="large")
        with ref_col1:
            reference_choice = st.select_slider(
                "Reference date for comparison",
                options=list(context.timeline_frame.index),
                value=reference_as_of,
                format_func=lambda x: fmt_date(pd.Timestamp(x)),
                key="reference_as_of_date",
            )
            reference_as_of = pd.Timestamp(reference_choice)
        with ref_col2:
            st.markdown(
                """
                <div class="card">
                    <strong>How to use Selected State below</strong><br/><br/>
                    The selected date is the <strong>current monitoring point</strong>. The reference date is your comparison anchor.
                    A risk analyst should ask: has the book, VaR, drawdown, or signal state worsened relative to the reference point?
                    Scenario changes do <strong>not</strong> alter the signal itself; they only translate the selected state into stress outcomes.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>Red-Alarm Episode Log</div>", unsafe_allow_html=True)
    if context.data_mode == "live":
        flare_log = context.flare_log.copy()
        if flare_log.empty:
            st.info("No Red-alarm episodes are currently available in the live alarm frame.")
        else:
            st.markdown(
                f"""
                <div class="card">
                    <strong>How to use this log</strong><br/><br/>
                    {flare_help_text()} Pick one episode here, then open the <strong>Incident Drilldown</strong> tab to inspect the event response workbench for the same selection.
                </div>
                """,
                unsafe_allow_html=True,
            )
            flare_choices = list(flare_log["flare_date"])
            default_flare = st.session_state.get("selected_flare_date")
            if default_flare not in flare_choices:
                default_flare = flare_choices[-1]
            flare_pick = st.selectbox(
                "Select a Red-alarm episode for drilldown",
                options=flare_choices,
                index=flare_choices.index(default_flare),
                format_func=lambda x: flare_label(pd.Timestamp(x), flare_log.loc[flare_log["flare_date"] == x].iloc[0]),
                key="monitor_flare_pick_live",
                help=flare_help_text(),
            )
            st.session_state["selected_flare_date"] = pd.Timestamp(flare_pick)
            st.dataframe(
                flare_log.rename(
                    columns={
                        "flare_date": "Episode start",
                        "flare_end_date": "Episode end",
                        "flare_trading_days": "Episode length (days)",
                        "dashboard_state": "State",
                        "conditioned_alarm_score": "Score",
                        "families": "Families",
                        "nav": "NAV",
                        "drawdown": "Drawdown",
                        "hs_var_return": "HS VaR",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    if context.data_mode == "live":
        latest = context.dashboard_metrics.loc[as_of]
        latest_alarm = context.alarm_frame.loc[as_of]
        latest_book = context.book.loc[as_of]
        reference = context.dashboard_metrics.loc[reference_as_of]
        reference_alarm = context.alarm_frame.loc[reference_as_of]
        active_families = [
            label
            for label, flag in (
                ("return/vol", latest_alarm["return_or_vol_alarm"]),
                ("residual", latest_alarm["residual_alarm"]),
                ("relationship", latest_alarm["relationship_alarm"]),
            )
            if int(flag) == 1
        ]
    else:
        latest = None
        latest_book = None
        latest_alarm = None
        active_families = context.overview_metrics["active_families"]
        ledger = context.trade_ledger.copy()
        gross_contracts = int(ledger["contracts"].sum()) if "contracts" in ledger else 0
        net_contracts = 0
        net_bbl = 0
        for row in ledger.to_dict("records"):
            signed_contracts = int(row["contracts"]) if row["side"] == "long" else -int(row["contracts"])
            net_contracts += signed_contracts
            lot = row.get("lot_size_bbl", riskbook_mod.LOT_SIZE_BBL)
            net_bbl += signed_contracts * int(lot)

    st.markdown("<div class='section-title'>Selected Book State</div>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    if context.data_mode == "live":
        b1.metric(
            "Net position (bbl)",
            fmt_num(float(latest_book["position_bbl"]), 0),
            delta=format_signed_num(float(latest_book["position_bbl"] - context.book.loc[reference_as_of, "position_bbl"]), 0),
        )
        b2.metric(
            "Exposure (USD)",
            fmt_num(float(latest_book["exposure_usd"]), 0),
            delta=format_signed_num(float(latest_book["exposure_usd"] - context.book.loc[reference_as_of, "exposure_usd"]), 0),
        )
        b3.metric(
            "Daily PnL (USD)",
            fmt_num(float(latest_book["pnl_usd"]), 0),
            delta=format_signed_num(float(latest_book["pnl_usd"] - context.book.loc[reference_as_of, "pnl_usd"]), 0),
        )
        b4.metric(
            "Book NAV",
            fmt_num(float(latest_book["nav"]), 2),
            delta=format_signed_num(float(latest_book["nav"] - context.book.loc[reference_as_of, "nav"]), 2),
        )
    else:
        b1.metric("Net position (bbl)", fmt_num(float(net_bbl), 0))
        b2.metric("Gross contracts", fmt_num(float(gross_contracts), 0))
        b3.metric("Net contracts", fmt_num(float(net_contracts), 0))
        b4.metric("Book NAV", fmt_num(context.overview_metrics["current_nav"], 2))

    if context.data_mode != "live":
        st.markdown(
            """
            <div class="card">
                <strong>Static book assumptions</strong><br/><br/>
                The monitor page is reading the current trade ledger structure available in the repo.
                If <code>data/book/current_positions.csv</code> exists, it will use that file; otherwise it falls back to the default Brent proxy ledger in <code>helpers/riskbook.py</code>.
                Position structure is known from the ledger;
                price-dependent mark-to-market fields require live market data.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="card">
                <strong>Live sources</strong><br/><br/>
                <strong>Book source:</strong> {context.book_source_label}<br/>
                <strong>Market source:</strong> {context.market_source_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Selected Risk State</div>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    if context.data_mode == "live":
        r1.metric(
            "HS VaR",
            fmt_pct(float(latest["hs_var_return"])),
            delta=format_signed_pct(float(latest["hs_var_return"] - reference["hs_var_return"])),
        )
        r2.metric(
            "HS ES",
            fmt_pct(float(latest["hs_es_return"])),
            delta=format_signed_pct(float(latest["hs_es_return"] - reference["hs_es_return"])),
        )
        r3.metric(
            "Drawdown",
            fmt_pct(float(latest["drawdown"])),
            delta=format_signed_pct(float(latest["drawdown"] - reference["drawdown"])),
        )
        r4.metric(
            "Realized vol (20d)",
            fmt_pct(float(latest["realized_vol_20d"])),
            delta=format_signed_pct(float(latest["realized_vol_20d"] - reference["realized_vol_20d"])),
        )
    else:
        r1.metric("HS VaR", fmt_pct(context.overview_metrics["current_var_return"]))
        r2.metric("HS ES", fmt_pct(context.overview_metrics["current_es_return"]))
        r3.metric("Drawdown", fmt_pct(context.overview_metrics["current_drawdown"]))
        r4.metric("Realized vol (20d)", fmt_pct(context.overview_metrics["current_realized_vol_20d"]))

    if context.data_mode != "live":
        risk_left, risk_right = st.columns([1.0, 1.0], gap="large")
        with risk_left:
            st.markdown("<div class='section-title'>Saved risk snapshot</div>", unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Field": "Book return", "Value": fmt_pct(context.overview_metrics["current_book_return"])},
                        {"Field": "NAV", "Value": fmt_num(context.overview_metrics["current_nav"], 2)},
                        {"Field": "Drawdown", "Value": fmt_pct(context.overview_metrics["current_drawdown"])},
                        {"Field": "Realized vol (20d)", "Value": fmt_pct(context.overview_metrics["current_realized_vol_20d"])},
                        {"Field": "HS VaR", "Value": fmt_pct(context.overview_metrics["current_var_return"])},
                        {"Field": "HS ES", "Value": fmt_pct(context.overview_metrics["current_es_return"])},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        with risk_right:
            st.markdown("<div class='section-title'>Saved VaR backtest summary</div>", unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Field": "Expected breach rate", "Value": fmt_pct(context.overview_metrics["expected_breach_rate"])},
                        {"Field": "Observed breach rate", "Value": fmt_pct(context.overview_metrics["observed_breach_rate"])},
                        {"Field": "Breach count", "Value": int(context.overview_metrics["breach_count"])},
                        {"Field": "Coverage sample days", "Value": int(context.overview_metrics["sample_days"])},
                        {"Field": "Max drawdown in sample", "Value": fmt_pct(context.overview_metrics["max_drawdown"])},
                        {"Field": "Coverage test", "Value": "Rejected at 5%"},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("<div class='section-title'>Trade Blotter At The Selected State</div>", unsafe_allow_html=True)
    blotter = build_trade_blotter_frame(context, as_of)
    st.dataframe(blotter, use_container_width=True, hide_index=True)
    if context.data_mode != "live":
        st.markdown(
            """
            <div class="card">
                <strong>Why some trade-level fields still need live prices</strong><br/><br/>
                Entry price, current price, unrealized PnL, and current exposure are mark-to-market fields.
                Their formulas are fully defined by the codebase, but their numeric values require a live or saved Brent price panel.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Signal At The Selected State</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    if context.data_mode == "live":
        s1.metric("Dashboard state", latest_alarm["dashboard_state"], delta=f"vs {fmt_date(reference_as_of)}: {reference_alarm['dashboard_state']}")
        s2.metric(
            "Alarm score",
            int(latest_alarm["conditioned_alarm_score"]),
            delta=f"{int(latest_alarm['conditioned_alarm_score'] - reference_alarm['conditioned_alarm_score']):+d}",
        )
        s3.metric("Active families", ", ".join(active_families) if active_families else "none")
        s4.metric("Recommended action", latest_alarm["recommended_action"])
    else:
        s1.metric("Dashboard state", context.overview_metrics["dashboard_state"])
        s2.metric("Alarm score", context.overview_metrics["conditioned_alarm_score"])
        s3.metric("Active families", ", ".join(active_families) if active_families else "none")
        s4.metric("Recommended action", context.overview_metrics["recommended_action"])

    render_signal_monitor(context, as_of, lookback_days, show_header=False)

    st.markdown("<div class='section-title'>Action Checklist For This State</div>", unsafe_allow_html=True)
    if context.data_mode == "live":
        state = latest_alarm["dashboard_state"]
    else:
        state = context.overview_metrics["dashboard_state"]

    action_items = [
        "Review current positions, exposure, and PnL in the Brent proxy book",
        "Check HS VaR / ES, realized volatility, and drawdown before reacting to the signal",
        f"Interpret the active signal families: {', '.join(active_families) if active_families else 'none'}",
        "Run the core scenario set and inspect stressed NAV / cash need",
        "Escalate to physical-book review only if the signal state and the book risk picture justify it",
    ]
    if state == "Green":
        action_items[-1] = "No immediate escalation; keep monitoring and watch whether another family joins the signal."
    elif state == "Amber":
        action_items[-1] = "No full escalation yet; inspect whether the current family remains isolated or is joined by another family."

    st.markdown(
        "<div class='card'><ul class='boundary-list'>"
        + "".join(f"<li>{item}</li>" for item in action_items)
        + "</ul></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Known Operational Boundaries</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <ul class="boundary-list">
                <li>Isolated Brent shocks can happen before any cross-market warning signal has time to move.</li>
                <li>Slow cumulative drawdowns remain hard because they are not single-day regime breaks.</li>
                <li>Some false reviews are an expected cost of monitoring broader macro stress through Gold.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Quick Stress Snapshot</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            This page is for monitoring and comparison across dates. The snapshot below is fixed to the selected date.
            Use the <strong>Incident Drilldown</strong> tab for the full event-response workbench and detailed scenario interaction.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_risk_and_stress(context, as_of, lookback_days, show_header=False)


def render_validation_page(context: ProjectContext) -> None:
    """Flare Drilldown page: selected flare, stress scenarios, and validation context."""

    render_page_header(
        title="Incident Drilldown",
        subtitle=(
            "Investigate one selected flare like a risk analyst would: identify the event, inspect the book state, review scenario losses, and decide whether escalation is justified."
        ),
        takeaway=(
            "This page is for response, not narration. Its job is to turn a flare into a concrete risk-review decision."
        ),
    )
    st.markdown(
        """
        <div class="card">
            The selected flare below is synchronized with the flare you choose on the <strong>Book Timeline</strong> page.
            Here, a <strong>flare</strong> means the <strong>start date of a Red-alarm episode</strong>: the first day the dashboard turns Red,
            not every red day inside the same episode. Use this page to understand whether that event materially changes the risk picture for the book.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Selected Flare</div>", unsafe_allow_html=True)
    flare_log = context.flare_log.copy()
    if context.data_mode == "live":
        if flare_log.empty:
            st.info("No flares are currently available in the live alarm frame.")
        else:
            flare_options = list(flare_log["flare_date"])
            default_flare = st.session_state.get("selected_flare_date")
            if default_flare not in flare_options:
                default_flare = flare_options[-1]
            flare_date = st.selectbox(
                "Red-alarm episode start",
                options=flare_options,
                index=flare_options.index(default_flare),
                format_func=lambda x: flare_label(pd.Timestamp(x), flare_log.loc[flare_log["flare_date"] == x].iloc[0]),
                key="drilldown_flare_pick_live",
                help=flare_help_text(),
            )
            st.session_state["selected_flare_date"] = pd.Timestamp(flare_date)
            flare_row = flare_log.loc[flare_log["flare_date"] == flare_date].iloc[0]
            alarm_row = context.alarm_frame.loc[pd.Timestamp(flare_date)]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Episode start", fmt_date(pd.Timestamp(flare_date)))
            c2.metric("State", flare_row["dashboard_state"])
            c3.metric("Score", int(flare_row["conditioned_alarm_score"]))
            c4.metric("Families", flare_row["families"])

            state_cols = st.columns(4)
            state_cols[0].metric("NAV at flare", fmt_num(flare_row["nav"], 2))
            state_cols[1].metric("Drawdown at flare", fmt_pct(flare_row["drawdown"]))
            state_cols[2].metric("HS VaR at flare", fmt_pct(flare_row["hs_var_return"]))
            state_cols[3].metric(
                "Episode length",
                f"{int(flare_row['flare_trading_days'])} trading day(s)" if pd.notna(flare_row["flare_trading_days"]) else "N/A",
            )

            action_col1, action_col2 = st.columns([1.2, 1.0], gap="large")
            with action_col1:
                st.markdown(
                    f"""
                    <div class="card">
                        <strong>What you are drilling into</strong><br/><br/>
                        This record is the <strong>start</strong> of one Red-alarm episode. The dashboard turned Red on
                        <strong>{fmt_date(pd.Timestamp(flare_date))}</strong> because the active families were
                        <strong>{flare_row['families']}</strong>. The question for the risk analyst is whether this new Red state
                        changes the book enough to justify immediate stress review and escalation.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with action_col2:
                st.markdown(
                    f"""
                    <div class="card">
                        <strong>Recommended action at episode start</strong><br/><br/>
                        {alarm_row["recommended_action"]}<br/><br/>
                        <strong>Episode end:</strong> {fmt_date(flare_row.get("flare_end_date"))}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            render_scenario_workbench(
                context=context,
                as_of=pd.Timestamp(flare_date),
                key_prefix="drilldown_page",
                title="Event response workbench",
                intro=(
                    "Use the controls below as a flare response area. Start from the selected Red-alarm episode, then vary the event path and watch "
                    "stress PnL, stressed NAV, MTM-style change, and cash need update immediately."
                ),
            )

            st.markdown("<div class='section-title'>Response Interpretation</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="card">
                    <strong>What this flare means</strong><br/><br/>
                    The flare is the start point of a Red-alarm episode, not a random red day. It is classified as
                    <strong>{flare_row['dashboard_state']}</strong> with families <strong>{flare_row['families']}</strong>.
                    The relevant question is not whether the flare is statistically interesting, but whether the book's
                    state at the start of that episode justifies a fuller risk review.
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        episode_options = list(context.named_episode_validation["episode"])
        default_episode = st.session_state.get("selected_flare_episode")
        if default_episode not in episode_options:
            default_episode = episode_options[0]
        episode_name = st.selectbox(
            "Named flare / episode",
            options=episode_options,
            index=episode_options.index(default_episode),
            key="drilldown_flare_pick_static",
            help=flare_help_text(),
        )
        st.session_state["selected_flare_episode"] = episode_name
        row = context.named_episode_validation.loc[context.named_episode_validation["episode"] == episode_name].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Episode", episode_name)
        c2.metric("First alarm", fmt_date(row["first_alarm_date"]))
        c3.metric("Alarm count", int(row["alarm_count"]))
        c4.metric("Alarm rate", fmt_pct(float(row["alarm_rate"])))

        st.markdown("<div class='section-title'>Why This Flare Matters</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card">
                {EPISODE_STORIES.get(episode_name, 'This episode is used as a flare-level validation example.')}<br/><br/>
                In static mode, this page shows the saved flare-level interpretation plus the standard scenario pack,
                rather than reconstructing the full day-level mark-to-market path.
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_scenario_workbench(
            context=context,
            as_of=pd.Timestamp(row["first_alarm_date"]) if pd.notna(row["first_alarm_date"]) else pd.Timestamp(context.overview_metrics["latest_date"]),
            key_prefix="drilldown_page_static",
            title="Event response workbench",
            intro=(
                "Static mode keeps the same event-driven framing: choose a flare-level validation episode, then inspect the saved stress translations as the response layer."
            ),
        )

        st.markdown("<div class='section-title'>Response Interpretation</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card">
                <strong>Operational reading</strong><br/><br/>
                The dashboard saved this episode because it is a good flare-level example of a regime change worth
                stress review. Use the scenario pack to ask: if the book had been escalated at the first alarm, how
                severe would the stress translation have looked?
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Project Context And Validation Evidence"):
        render_story_intro(context)
        render_story_method(context)
        render_story_results(context)
        render_story_comparison(context)
        render_story_limits(context)
        render_story_conclusion(context)
        render_episode_explorer(context, show_header=False)
        render_diagnostics_and_method(context, show_header=False)


def render_static_overview(context: ProjectContext) -> None:
    selected = static_selection_context(context, pd.Timestamp(context.overview_metrics["latest_date"]))
    st.info(
        "Static notebook mode is active. The monitor page is intentionally anchored to the saved latest operational snapshot, so it behaves like a trade-book monitor instead of a research browser."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", "Latest snapshot")
    c2.metric("Displayed state", str(selected["state"]))
    c3.metric("Alarm score", str(selected["score"]))
    c4.metric("As of", fmt_date(pd.Timestamp(selected["selected_date"])))

    c5, c6, c7 = st.columns(3)
    c5.metric("Recommended action", str(selected["recommended_action"]))
    c6.metric("Book NAV", fmt_num(selected["nav"]))
    c7.metric("Drawdown", fmt_pct(selected["drawdown"]))

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("<div class='section-title'>Book context</div>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {"Field": "Book type", "Value": "Brent proxy futures book"},
                    {"Field": "Position build", "Value": "300 long + 200 long - 150 short"},
                    {"Field": "Net barrels", "Value": "350,000"},
                    {"Field": "Use in dashboard", "Value": "Risk monitoring proxy, not the full physical book"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("<div class='section-title'>Current interpretation</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card">
                <p><strong>Selection note:</strong> {selected['note']}</p>
                <p><strong>Strongest relationship distortion:</strong> {selected['strongest_relationship']}</p>
                <p><strong>Relationship z-score:</strong> {fmt_num(selected['relationship_z'], 2)}</p>
                <p><strong>HS VaR:</strong> {fmt_pct(selected['var_return'])}</p>
                <p><strong>HS ES:</strong> {fmt_pct(selected['es_return'])}</p>
                <p><strong>Mode note:</strong> This summary is notebook-derived, not re-downloaded from the market-data API.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Red-Alarm Episode Log</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
            <strong>How to read this log</strong><br/><br/>
            {flare_help_text()}
        </div>
        """,
        unsafe_allow_html=True,
    )
    flare_log = context.flare_log.copy()
    if not flare_log.empty:
        if context.data_mode == "live":
            flare_choices = list(flare_log["flare_date"])
            default_flare = st.session_state.get("selected_flare_date")
            if default_flare not in flare_choices:
                default_flare = flare_choices[-1]
            flare_pick = st.selectbox(
                "Select a Red-alarm episode for drilldown",
                options=flare_choices,
                index=flare_choices.index(default_flare),
                format_func=lambda x: flare_label(pd.Timestamp(x), flare_log.loc[flare_log["flare_date"] == x].iloc[0]),
                key="monitor_flare_pick_live",
                help=flare_help_text(),
            )
            st.session_state["selected_flare_date"] = pd.Timestamp(flare_pick)
        else:
            flare_choices = [ep for ep in flare_log["episode"].dropna().tolist() if isinstance(ep, str)]
            if flare_choices:
                default_episode = st.session_state.get("selected_flare_episode")
                if default_episode not in flare_choices:
                    default_episode = flare_choices[0]
                flare_pick = st.selectbox(
                    "Select a flare / episode for drilldown",
                    options=flare_choices,
                    index=flare_choices.index(default_episode),
                    key="monitor_flare_pick_static",
                )
                st.session_state["selected_flare_episode"] = flare_pick

        if context.data_mode == "live":
            flare_log = flare_log.rename(
                columns={
                    "flare_date": "Episode start",
                    "flare_end_date": "Episode end",
                    "flare_trading_days": "Episode length (days)",
                    "dashboard_state": "State",
                    "conditioned_alarm_score": "Score",
                    "families": "Families",
                    "nav": "NAV",
                    "drawdown": "Drawdown",
                    "hs_var_return": "HS VaR",
                }
            )
        else:
            flare_log = flare_log.rename(
                columns={
                    "flare_date": "First alarm",
                    "episode": "Episode",
                    "dashboard_state": "Capture status",
                    "families": "Saved detail",
                }
            )
        st.dataframe(flare_log, use_container_width=True, hide_index=True)
    else:
        st.info("No flare log is currently available.")


def render_static_signal_monitor(context: ProjectContext) -> None:
    focus = get_static_focus(context)
    selected = static_selection_context(context, focus["date"])
    st.markdown(
        "The formal signal logic is unchanged. Static mode shows the rule set and the latest validated interpretation "
        "from the saved project outputs."
    )
    st.markdown(
        r"""
        <div class="formula">
            <strong>Signal families</strong><br/>
            1. return/vol: <code>|gold_return_z| &gt; 2</code> or <code>gold_vol_z &gt; 2</code><br/>
            2. residual: <code>|gold_residual_z| &gt; 2</code><br/>
            3. relationship: <code>max(|corr_z|) &gt; 2</code>
        </div>
        <div class="formula">
            <strong>Escalation rule</strong><br/>
            <code>conditioned_alarm_score = 1{return/vol} + 1{residual} + 1{relationship}</code><br/>
            <code>Red if conditioned_alarm_score &gt;= 2</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "family": "return/vol",
                    "static view": "quiet" if selected["kind"] == "latest_snapshot" else "not stored at daily resolution",
                },
                {
                    "family": "residual",
                    "static view": "quiet" if selected["kind"] == "latest_snapshot" else "not stored at daily resolution",
                },
                {
                    "family": "relationship",
                    "static view": "active" if selected["kind"] == "latest_snapshot" else "not stored at daily resolution",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_static_risk_and_stress(context: ProjectContext) -> None:
    focus = get_static_focus(context)
    selected = static_selection_context(context, focus["date"])
    st.markdown(
        "Static mode keeps the risk interpretation visible even when live market reconstruction is unavailable. "
        "The scenario table below is the validated stress-action layer from the project."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected VaR breach rate", "5.00%")
    c2.metric("Observed breach rate", "5.82%")
    c3.metric("Kupiec p-value", "0.015")
    c4.metric("Coverage rejected at 5%?", "Yes")
    if selected["kind"] == "episode_window":
        st.markdown(
            f"""
            <div class="card">
                <strong>Episode-focused note</strong><br/>
                The metrics above are full-sample VaR validation results for the Brent proxy book.
                They remain constant across episode focus because they describe the model's overall calibration,
                not the single selected event window.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.dataframe(
        context.stress_results.rename(
            columns={
                "scenario_name": "Scenario",
                "shock_pct": "Brent shock",
                "stress_pnl_usd": "Stress PnL on NAV100",
                "stress_return": "Stress return",
                "nav_after_stress": "NAV after stress",
                "cash_need_usd": "Cash need",
                "breach_flag": "Breach flag",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_static_episode_explorer(context: ProjectContext) -> None:
    focus = get_static_focus(context)
    selected = static_selection_context(context, focus["date"])
    st.markdown(
        "Static mode cannot render the full multi-panel event chart, but it preserves the validated event-study "
        "conclusions from the saved notebook outputs."
    )
    if selected["kind"] == "episode_window":
        episode_name = str(selected["episode"])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Episode", episode_name)
        c2.metric("Window start", fmt_date(selected["episode_start"]))
        c3.metric("Window end", fmt_date(selected["episode_end"]))
        c4.metric("First alarm", fmt_date(selected["first_alarm_date"]))
        c5.metric("Lead time", f"{int(selected['lead_days'])} days" if pd.notna(selected["lead_days"]) else "Not stored in static mode")

        st.markdown(
            f"""
            <div class="card">
                <strong>{episode_name}</strong>
                <ul class="boundary-list">
                    <li>This date was mapped into a named stress episode window in static mode.</li>
                    <li>Saved episode summary is shown instead of a reconstructed daily chart.</li>
                    <li>First alarm is shown relative to the event window, not as the selected focus date.</li>
                    <li>Max Gold score in window: {fmt_num(selected['max_gold_score'], 3)}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("COVID first alarm", fmt_date(context.overview_metrics["covid_first_alarm"]))
        c2.metric("First post-alarm breach", fmt_date(context.overview_metrics["covid_first_post_alarm_breach"]))
        c3.metric("Lead time", f"{context.overview_metrics['covid_lead_days']} days")

        st.markdown(
            """
            <div class="card">
                <strong>COVID 2020 sequence</strong>
                <ul class="boundary-list">
                    <li>Relationship stress appeared before the full commodity shock.</li>
                    <li>The dashboard escalated to Red on 2020-02-25.</li>
                    <li>The first post-alarm VaR breach followed on 2020-03-06.</li>
                    <li>This case is the clearest demonstration that the dashboard is an escalation trigger, not a price predictor.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.dataframe(
        context.named_episode_validation[["episode", "first_alarm_date", "alarm_count", "alarm_rate", "max_gold_score"]],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    apply_theme()
    top_ctrl_left, top_ctrl_mid, top_ctrl_right = st.columns([1.0, 1.0, 1.4], gap="large")
    auto_download = top_ctrl_left.checkbox("Download data if processed files are missing", value=True)
    if top_ctrl_mid.button("Refresh live market data"):
        st.cache_data.clear()
        st.session_state["force_refresh"] = True

    refresh_requested = bool(st.session_state.pop("force_refresh", False))

    try:
        with st.spinner("Building dashboard context..."):
            context = load_context(refresh_requested, auto_download)
    except ProjectDataUnavailableError as exc:
        render_empty_state(str(exc))
        return

    if context.data_mode == "live":
        dates = context.dashboard_metrics.index.sort_values()
        default_as_of = dates[-1]
        as_of = top_ctrl_right.date_input(
            "As-of date",
            value=default_as_of.date(),
            min_value=dates[0].date(),
            max_value=default_as_of.date(),
        )
        as_of_ts = context.dashboard_metrics.index.asof(pd.Timestamp(as_of))
        lookback_days = st.slider("Lookback window (days)", min_value=60, max_value=540, value=180, step=30)
    else:
        default_as_of = pd.Timestamp(context.overview_metrics["latest_date"])
        focus_col1, focus_col2, focus_col3 = st.columns([1.0, 1.0, 1.1], gap="large")
        focus_mode = focus_col1.selectbox(
            "Static focus",
            options=["Latest snapshot", "Named episode", "Custom date"],
            index=0,
        )
        st.session_state["static_focus_mode"] = focus_mode

        if focus_mode == "Latest snapshot":
            static_as_of = default_as_of.date()
            focus_col2.metric("Latest saved snapshot", fmt_date(default_as_of))
            focus_col3.caption("Uses the saved latest-state snapshot from the notebook outputs.")
            st.session_state["static_focus_episode"] = None
        elif focus_mode == "Named episode":
            episode_name = focus_col2.selectbox(
                "Episode",
                options=list(EVENT_TITLES.keys()),
                index=list(EVENT_TITLES.keys()).index("COVID crash + oil war") if "COVID crash + oil war" in EVENT_TITLES else 0,
            )
            st.session_state["static_focus_episode"] = episode_name
            episode_start, episode_end = EVENT_TITLES[episode_name]
            static_as_of = pd.Timestamp(episode_start).date()
            focus_col3.caption("Maps the selected date into the chosen named stress episode window.")
        else:
            static_as_of = focus_col2.date_input(
                "Custom date",
                value=default_as_of.date(),
            )
            st.session_state["static_focus_episode"] = None
            focus_col3.caption("Custom dates use static snapshot/episode mapping if a saved summary exists.")

        st.session_state["static_as_of_date"] = str(static_as_of)
        top_ctrl_right.caption(
            "Static notebook mode uses saved project outputs. It supports latest snapshot, named episodes, and custom-date mapping."
        )
        as_of_ts = pd.Timestamp(static_as_of)
        lookback_days = 180
        top_ctrl_right.metric("Static reference date", fmt_date(pd.Timestamp(static_as_of)))

    tab_monitor, tab_validation = st.tabs(["Book Timeline", "Incident Drilldown"])

    with tab_monitor:
        render_monitor_page(context, as_of_ts, lookback_days)
    with tab_validation:
        render_validation_page(context)


if __name__ == "__main__":
    main()
