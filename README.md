# Gold Abnormality as a Nowcasting Escalation Dashboard for Commodity Risk Management

QF637 Commodities Risk Management — SMU MQF

---

## Overview

This project builds a prototype risk dashboard that uses abnormal gold behaviour to nowcast instability in the market environment around a commodity risk book. When gold's return, volatility, residual behaviour, or cross-market relationships become abnormal, the dashboard escalates the book for VaR review and stress testing.

The project is not framed as a pure prediction model. It does not claim that gold reliably forecasts all future commodity losses. Instead, gold is treated as a real-time cross-market instability indicator that may reveal that the current environment is no longer well represented by trailing risk measures.

## Core Research Question

Can abnormal gold behaviour act as a useful nowcasting signal that prompts timely review of VaR and stress-test exposure in a commodity risk book?

## Operational Premise

Traditional VaR reacts through trailing return windows. During regime shifts, VaR may not fully reflect the new environment until volatility, losses, or breaches have already entered the book. This is formally confirmed by the Kupiec Proportion of Failures test on the Brent risk book: the observed VaR breach rate (5.82%) is statistically higher than the declared 95% confidence level (p = 0.015), indicating the HS VaR systematically underestimates risk during regime changes.

Gold may be useful because it sits across several market narratives:

- safe-haven demand,
- USD and real-rate pressure,
- liquidity stress,
- geopolitical or inflation stress,
- changing confidence in monetary assets.

The dashboard asks whether gold is behaving unusually now, not whether gold perfectly predicts future losses.

## Scope

- **Alarm asset:** Gold futures (`GC=F`)
- **Risk book:** Brent crude futures (`BZ=F`)
- **Conditioning variables:** DXY (`DX-Y.NYB`), VIX (`^VIX`), US 10Y yield (`^TNX`)
- **Data:** 2007-07-01 to present via yfinance (4,685 daily observations)

Brent (`BZ=F`) is the binding start-date constraint — data available from 2007-07-30, capturing the full GFC (2008–2009).

> **Note on futures versus physical commodity book.** The risk book in this project uses Brent crude futures as a proxy rather than a physical commodity trading book. A real physical book differs in several ways: P&L is driven by physical price assessments (e.g. Platts dated Brent) rather than exchange-settled futures, carry costs such as storage, freight, and insurance affect returns, basis risk between physical and futures prices is a live exposure, and physical positions cannot be unwound as quickly as futures. These differences affect the specific VaR breach rates and drawdown magnitudes in Notebook 08, but do not affect the signal results or conclusions. The gold alarm is constructed entirely from exchange-traded signals and its coverage rates, lead times, and baseline comparisons are independent of whether the underlying book is futures or physical. The stress episodes it catches would trigger the same gold abnormality regardless of book structure. A physical trading desk would substitute their own book into this framework — the alarm and the VaR-gap argument carry over directly. The intended production implementation of this dashboard would use a physical commodity trading book with actual physical price assessments, carry costs, and basis risk, replacing the futures proxy used here for research purposes.

## Notebook Pipeline

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_DataExtraction.ipynb` | Download and save raw price data |
| 02 | `02_DataCleaning.ipynb` | Strict alignment, log returns, level changes |
| 03 | `03_DescriptiveEvidence.ipynb` | Fat-tail tests, static vs rolling correlations, kill forecasting hypothesis |
| 04 | `04_NowcastingValidation.ipynb` | Validate nowcasting direction against stress proxies and named events |
| 05 | `05_ConditioningVariableSelection.ipynb` | Test Gold-only vs Gold+conditioning variants |
| 06 | `06_GoldCenteredAlarmDesign.ipynb` | Build conditioned challenger alarm (selected default) |
| 07 | `07_OperationalUsefulness.ipynb` | Named episode validation, coverage sweep, baseline comparison |
| 08 | `08_RiskBookVaRStress.ipynb` | Brent risk book, HS VaR, Kupiec test, stress scenarios |
| 09 | `09_LeadTimeDashboard.ipynb` | Lead-time tests, false alarm analysis, dashboard export |
| 10 | `10_Synthesis.ipynb` | Regime interpretation, COVID 2020 case study, current dashboard state, final claim |

## Alarm Design

### Selected default: Conditioned challenger (Notebook 06)

Fires when **2 or more** of 3 signal families simultaneously exceed threshold 2.0 on a 252-day shifted trailing z-score baseline:

- **Family 1:** `|gold_return_z| > 2.0` OR `gold_vol_z > 2.0` — gold's own return/vol abnormality
- **Family 2:** `|gold_residual_z| > 2.0` — gold's OLS residual vs Brent, DXY, VIX, US10Y
- **Family 3:** `max(|corr_z|) > 2.0` — gold's rolling 60d correlation relationships

Dashboard states:
- 🟢 **Green** (0 families): normal monitoring
- 🟡 **Amber** (1 family): inspect gold components and relationships
- 🔴 **Red** (2+ families): review VaR calibration and run stress scenarios

## Key Results

### Named stress episode validation — 9 / 9 caught

| Episode | First alarm |
|---|---|
| GFC — Lehman 2008 | 2008-10-23 |
| European debt crisis 2010–2012 | 2010-05-06 |
| Oil crash 2014–2016 | 2014-07-14 |
| China equity shock 2015 | 2015-06-18 |
| COVID crash + oil war 2020 | 2020-02-25 |
| Russia/Ukraine invasion 2022 | 2022-02-25 |
| Fed tightening cycle 2022 | 2022-09-15 |
| US tariff shock 2025 | 2025-04-04 |
| Recent geopolitical stress 2026 | 2026-02-03 |

### Operational performance

| Metric | Value |
|---|---|
| Stress-proxy coverage (default) | 68.0% |
| False review rate | 30.9% |
| Reviews per year | ~10 |
| At 95% coverage: false review rate | 35.1% |
| Gold vs random (equal frequency) | 35.1% vs 47.9% false reviews |

### VaR backtest (Kupiec POF test)

| Metric | Value |
|---|---|
| Expected breach rate | 5.00% |
| Observed breach rate | 5.82% |
| LR statistic | 5.94 |
| p-value | 0.015 |
| H₀ rejected at 5%? | Yes |

VaR is statistically miscalibrated — directly motivating the cross-asset early warning signal.

### Lead-time results

| Risk event | Match rate | Avg lead | 95% CI |
|---|---|---|---|
| VaR breach (228 episodes) | 60.1% | 8.0 days | 53.6%–66.2% |
| Portfolio vol spike (23 episodes) | 69.6% | 9.4 days | 49.1%–84.4% |
| Drawdown (14 episodes) | 0.0% | — | 0%–21.5% |

False alarm: 31.4% at 30-day window.

## Claim

> Abnormal gold behaviour — detected across multiple signal families including own return/vol, OLS residual, and rolling relationship instability — can provide a useful nowcasting escalation signal for commodity risk management by flagging cross-market instability and prompting VaR review and stress testing. The signal caught all 9 major historical stress episodes in the 2007–2026 dataset and leads VaR breaches with an 8-day average. Its value lies in detecting sudden systemic regime shifts, not in predicting gradual drawdowns or isolated commodity price shocks.

## Known Limitations

- **Drawdown prediction:** 0 of 14 drawdown events matched — the alarm detects sudden regime shifts, not slow cumulative losses.
- **Isolated commodity shocks:** 41 of 56 missed stress events are pure Brent return shocks with no gold safe-haven reaction.
- **Small sample sizes:** Vol spike (23) and drawdown (14) samples produce wide confidence intervals.
- **In-sample calibration:** All thresholds calibrated on the full 2007–2026 sample. No out-of-sample test conducted.

## Requirements

```
pandas
numpy
yfinance
matplotlib
seaborn
scipy
statsmodels
nbformat
```

## Running the Pipeline

Run notebooks in order from `01_DataExtraction.ipynb` to `10_Synthesis.ipynb`. Each notebook reads from `data/processed/` outputs saved by the previous step. `10_Synthesis.ipynb` reads from saved outputs only and does not recompute any signals.
