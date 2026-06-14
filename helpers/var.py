"""Historical Simulation VaR/ES and Kupiec POF backtest (ports NB08)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

VAR_CONFIDENCE = 0.95
VAR_ALPHA = 1 - VAR_CONFIDENCE
VAR_WINDOW = 250


def historical_var_es(
    returns: pd.Series,
    window: int = VAR_WINDOW,
    confidence: float = VAR_CONFIDENCE,
) -> tuple[pd.Series, pd.Series]:
    """Rolling Historical Simulation VaR and ES, using only returns strictly prior to each day."""
    alpha = 1 - confidence
    hist_input = returns.shift(1)

    hs_var_return = hist_input.rolling(window).quantile(alpha)
    hs_es_return = hist_input.rolling(window).apply(
        lambda x: x[x <= np.quantile(x, alpha)].mean(),
        raw=True,
    )
    return hs_var_return, hs_es_return


def var_breach_flags(returns: pd.Series, hs_var_return: pd.Series) -> pd.Series:
    """1 where the realized return breaches (falls below) the VaR estimate, else 0."""
    return (returns < hs_var_return).astype(int)


def kupiec_pof_test(
    breach_flags: pd.Series,
    hs_var_return: pd.Series,
    confidence: float = VAR_CONFIDENCE,
) -> dict:
    """Kupiec Proportion-of-Failures likelihood-ratio test for VaR coverage.

    Restricts to days with a valid `hs_var_return` estimate, matching NB08's
    `coverage_sample = riskbook.dropna(subset=["hs_var_return"])`.
    """
    mask = hs_var_return.notna()
    breaches = breach_flags[mask]

    n_obs = len(breaches)
    n_breaches = int(breaches.sum())
    n_non_breaches = n_obs - n_breaches
    p_hat = n_breaches / n_obs
    p_expected = 1 - confidence

    lr_stat = -2 * (
        n_breaches * np.log(p_expected / p_hat)
        + n_non_breaches * np.log((1 - p_expected) / (1 - p_hat))
    )
    p_value = float(stats.chi2.sf(lr_stat, df=1))

    return {
        "n_obs": n_obs,
        "n_breaches": n_breaches,
        "expected_breach_rate": p_expected,
        "observed_breach_rate": p_hat,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "reject_h0_5pct": p_value < 0.05,
    }
