"""Gold abnormality signal components (ports NB06's signal-construction step)."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

BASELINE_WIN = 252
CORR_WIN = 60
VOL_WIN = 30
ROLLING_REG_WIN = 252

CONDITIONING_COLS = ["r_Brent", "r_DXY", "d_VIX", "d_US10Y"]


def trailing_zscore(series: pd.Series, window: int = BASELINE_WIN) -> pd.Series:
    """Z-score of `series` against its own trailing (shifted) mean/std, avoiding lookahead."""
    mean = series.rolling(window).mean().shift(1)
    std = series.rolling(window).std().shift(1)
    return (series - mean) / std


def rolling_residual(y: pd.Series, x: pd.DataFrame, window: int = ROLLING_REG_WIN) -> pd.Series:
    """Out-of-sample residual of `y` vs an OLS on `x` fit over the prior `window` observations."""
    data = pd.concat([y, x], axis=1).dropna()
    residuals = pd.Series(index=data.index, dtype=float)

    y_name = y.name
    x_cols = list(x.columns)
    min_obs = int(window * 0.8)

    for i in range(window, len(data)):
        train = data.iloc[i - window:i]
        if len(train) < min_obs:
            continue
        y_train = train[y_name]
        x_train = sm.add_constant(train[x_cols], has_constant="add")
        model = sm.OLS(y_train, x_train).fit()

        x_now = sm.add_constant(data.iloc[[i]][x_cols], has_constant="add")
        fitted = float(model.predict(x_now).iloc[0])
        residuals.loc[data.index[i]] = float(data.iloc[i][y_name] - fitted)

    return residuals


def compute_gold_signal_components(market_vars: pd.DataFrame) -> pd.DataFrame:
    """Build the three gold-alarm signal families: return/vol, residual, and relationship z-scores."""
    signal_components = pd.DataFrame(index=market_vars.index)

    signal_components["gold_return_z"] = trailing_zscore(market_vars["r_Gold"], BASELINE_WIN)

    gold_vol = market_vars["r_Gold"].rolling(VOL_WIN).std()
    signal_components["gold_vol_z"] = trailing_zscore(gold_vol, BASELINE_WIN)

    gold_residual = rolling_residual(market_vars["r_Gold"], market_vars[CONDITIONING_COLS], ROLLING_REG_WIN)
    signal_components["gold_residual_z"] = trailing_zscore(gold_residual, BASELINE_WIN)

    for col in CONDITIONING_COLS:
        rolling_corr = market_vars["r_Gold"].rolling(CORR_WIN).corr(market_vars[col])
        signal_components[f"gold_corr_{col}_z"] = trailing_zscore(rolling_corr, BASELINE_WIN)

    return signal_components
