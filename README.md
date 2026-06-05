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
- **Original benchmark risk book:** Brent crude futures (`BZ=F`)
- **Extended parallel proxy books:** WTI Cushing spot, Henry Hub spot
- **Physical interpretation layers:** Cushing crude stocks, U.S. crude inventories excluding SPR, Lower-48 working gas storage
- **Conditioning variables:** DXY (`DX-Y.NYB`), VIX (`^VIX`), US 10Y yield (`^TNX`)
- **Data:** 2007-07-01 to present via yfinance (4,685 daily observations)

Brent (`BZ=F`) is the binding start-date constraint — data available from 2007-07-30, capturing the full GFC (2008–2009).

> **Note on futures versus physical commodity book.** The risk book in this project uses Brent crude futures as a proxy rather than a physical commodity trading book. A real physical book differs in several ways: P&L is driven by physical price assessments (e.g. Platts dated Brent) rather than exchange-settled futures, carry costs such as storage, freight, and insurance affect returns, basis risk between physical and futures prices is a live exposure, and physical positions cannot be unwound as quickly as futures. These differences affect the specific VaR breach rates and drawdown magnitudes in Notebook 08, but do not affect the signal results or conclusions. The gold alarm is constructed entirely from exchange-traded signals and its coverage rates, lead times, and baseline comparisons are independent of whether the underlying book is futures or physical. The stress episodes it catches would trigger the same gold abnormality regardless of book structure. A physical trading desk would substitute their own book into this framework — the alarm and the VaR-gap argument carry over directly. The intended production implementation of this dashboard would use a physical commodity trading book with actual physical price assessments, carry costs, and basis risk, replacing the futures proxy used here for research purposes.

## Multi-Book Extension

This version extends the downstream risk-book layer from a single benchmark Brent futures book to a **parallel multi-book evaluation template**. The purpose of the extension is to keep the Gold alarm unchanged while testing whether the same escalation logic is useful for several proxy books that are more suggestive of physical trading exposure.

### What the new scheme does

- keeps the **Gold signal construction fixed** from Notebooks `01`–`07`;
- builds several books **in parallel** from Notebook `08` onward;
- evaluates each book with the **same VaR / lead-time template**;
- compares results across books instead of re-tuning the alarm for each market.

### Books included in the extension

- **Brent futures**: original benchmark book retained for continuity.
- **WTI Cushing spot**: physical crude proxy book.
- **Henry Hub spot**: physical gas proxy book.

### Physical interpretation layer

The extension also adds physical-state variables that are **not** treated as P&L legs:

- **WTI context:** Cushing crude stocks and U.S. crude inventories excluding SPR
- **Henry Hub context:** Lower-48 working gas storage

These weekly series are used only as **last-known released context variables** for interpretation. They do **not** enter `R_book`, VaR, ES, breach counts, or lead-time event definitions.

### Design principles of the extension

1. **01–07 are not changed.**
   Gold alarm construction, descriptive evidence, and operating validation remain as originally designed.

2. **Signal timeline and risk-book timeline are separated.**
   `gold_alarm_frame` is preserved. Each book is prepared independently and only then aligned to the signal timeline.

3. **Each book uses its own valid evaluation sample.**
   Alignment is done via **per-book inner join** with the existing Gold alarm outputs.

4. **Weekly physical series are never forced into daily P&L.**
   No interpolation into daily returns, no use of unpublished week-end information, and no inventory terms inside `R_book`.

5. **Direction convention is unified.**
   All books are treated as **long physical inventory / merchant exposure**. Price declines therefore represent downside risk to the book.

6. **Comparison settings are held fixed across books.**
   The extension uses one common template rather than optimizing each book separately:
   - `NAV0 = 100`
   - `VAR_CONFIDENCE = 0.95`
   - `VAR_WINDOW = 250`
   - `REALIZED_VOL_WINDOW = 20`
   - `LOOKBACK_DAYS = 30`
   - `COOLDOWN_DAYS = 5`
   - `VOL_Z_THRESHOLD = 2.0`
   - `DRAWDOWN_EVENT_LEVEL = -0.05`

### Return definitions and publication timing

- **Brent futures:** existing Step 02 return series.
- **WTI Cushing spot:** daily log return on spot proxy series.
- **Henry Hub spot:** daily log return on spot proxy series.

For the weekly physical context layer:

- keep both `week_end_date` and `release_date` conceptually distinct;
- use **release date** as the first date on which the information is allowed to appear in the daily dashboard context;
- forward-fill only the **last known released context value**;
- do not allow unpublished weekly values to influence same-day signal or event interpretation.

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
| 08 | `08_RiskBookVaRStress.ipynb` | Multi-book risk-book construction, HS VaR, Kupiec tests, physical context, stress scenarios |
| 09 | `09_LeadTimeDashboard.ipynb` | Multi-book lead-time tests, false alarm analysis, dashboard comparison, cross-book exports |
| 10 | `10_Synthesis.ipynb` | Multi-book synthesis, regime interpretation, COVID case study, current cross-book state, final claim |

## What Changed In Code

### Not changed

The following notebooks are intentionally preserved so that the Gold alarm itself is unchanged:

- `01_DataExtraction.ipynb`
- `02_DataCleaning.ipynb`
- `03_DescriptiveEvidence.ipynb`
- `04_NowcastingValidation.ipynb`
- `05_ConditioningVariableSelection.ipynb`
- `06_GoldCenteredAlarmDesign.ipynb`
- `07_OperationalUsefulness.ipynb`

### Modified

The downstream book layer is the only part that was redesigned:

- `08_RiskBookVaRStress.ipynb`
  - changed from a single Brent book to a **parameterized multi-book template**;
  - adds WTI and Henry Hub price-leg construction;
  - adds weekly physical context processing;
  - exports cross-book summaries including `book_comparison_summary.csv` and `book_var_backtest_comparison.csv`.

- `09_LeadTimeDashboard.ipynb`
  - changed from single-book lead-time analysis to **parallel lead-time evaluation across books**;
  - exports `book_lead_time_comparison.csv` and multi-book dashboard outputs.

- `10_Synthesis.ipynb`
  - changed from single-book synthesis to **multi-book synthesis**;
  - keeps one common Gold signal interpretation layer while comparing downstream book reactions.

### New output structure introduced by the extension

The extension produces three headline comparison files:

- `book_comparison_summary.csv`
- `book_lead_time_comparison.csv`
- `book_var_backtest_comparison.csv`

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

Unless otherwise stated, the headline results below refer to the **original benchmark Brent futures book**. The later multi-book extension is a **comparative downstream layer** built to test whether the same Gold alarm is useful across several proxy books. It should be read as an extension of the benchmark case, not as a replacement for the Brent headline results.

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

These operating metrics are the benchmark Brent results used to motivate the project claim. The multi-book extension in Notebooks `08`–`10` applies the same evaluation template to WTI Cushing spot and Henry Hub spot for cross-book comparison.

| Metric | Value |
|---|---|
| Stress-proxy coverage (default) | 68.0% |
| False review rate | 30.9% |
| Reviews per year | ~10 |
| At 95% coverage: false review rate | 35.1% |
| Gold vs random (equal frequency) | 35.1% vs 47.9% false reviews |

### VaR backtest (Kupiec POF test)

This VaR backtest table refers to the **benchmark Brent futures book**. In the multi-book extension, equivalent VaR and Kupiec outputs are produced for WTI Cushing spot and Henry Hub spot and saved into the cross-book comparison exports.

| Metric | Value |
|---|---|
| Expected breach rate | 5.00% |
| Observed breach rate | 5.82% |
| LR statistic | 5.94 |
| p-value | 0.015 |
| H₀ rejected at 5%? | Yes |

VaR is statistically miscalibrated — directly motivating the cross-asset early warning signal.

### Lead-time results

These lead-time figures are the benchmark Brent-book results. The multi-book extension keeps the same lead-time template and reports parallel results in `book_lead_time_comparison.csv` for the additional proxy books.

| Risk event | Match rate | Avg lead | 95% CI |
|---|---|---|---|
| VaR breach (228 episodes) | 60.1% | 8.0 days | 53.6%–66.2% |
| Portfolio vol spike (23 episodes) | 69.6% | 9.4 days | 49.1%–84.4% |
| Drawdown (14 episodes) | 0.0% | — | 0%–21.5% |

False alarm: 31.4% at 30-day window.

## Claim

> Abnormal gold behaviour — detected across multiple signal families including own return/vol, OLS residual, and rolling relationship instability — can provide a useful nowcasting escalation signal for commodity risk management by flagging cross-market instability and prompting VaR review and stress testing. The signal caught all 9 major historical stress episodes in the 2007–2026 dataset and leads VaR breaches with an 8-day average. Its value lies in detecting sudden systemic regime shifts, not in predicting gradual drawdowns or isolated commodity price shocks.

The core empirical evidence supporting this claim is the **benchmark Brent futures case**. The later multi-book extension should be read as a **robustness and portability test**: it asks whether the same fixed Gold alarm remains useful when applied to additional proxy books, rather than replacing Brent as the primary evidential base of the project.

## Known Limitations

- **Drawdown prediction:** 0 of 14 drawdown events matched — the alarm detects sudden regime shifts, not slow cumulative losses.
- **Isolated commodity shocks:** 41 of 56 missed stress events are pure Brent return shocks with no gold safe-haven reaction.
- **Small sample sizes:** Vol spike (23) and drawdown (14) samples produce wide confidence intervals.
- **In-sample calibration:** All thresholds calibrated on the full 2007–2026 sample. No out-of-sample test conducted.
- **Proxy-book status:** WTI spot and Henry Hub spot are used as proxy books, not as complete physical trading P&L engines with basis, storage carry, logistics, and optionality.
- **Environment sensitivity:** the multi-book extension depends on external free data sources and local notebook kernel configuration, which may vary across machines.

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

Run notebooks in order from `01_DataExtraction.ipynb` to `10_Synthesis.ipynb`.

Practical dependency notes:

- `08_RiskBookVaRStress.ipynb` expects the processed outputs from at least Steps `02`, `04`, and `06` to exist locally under `data/processed/`.
- `09_LeadTimeDashboard.ipynb` expects the Step 08 multi-book outputs to exist.
- `10_Synthesis.ipynb` expects the saved outputs from Steps `06`, `08`, and `09`.

## Environment And Reproducibility Note

This is a group course project and the working environments are not fully standardized. Different contributors may run different operating systems, Python installations, notebook kernels, package versions, and network conditions. As a result:

- a notebook failing to run immediately on another machine can be normal;
- missing local `data/processed/` files or prior notebook outputs can prevent downstream notebooks from executing;
- external free data downloads (for example FRED / EIA pages) may time out or behave differently across systems;
- notebook kernel metadata may need to be adjusted locally if a saved kernel name does not exist on another contributor's computer.

For this reason, execution issues should first be interpreted as an **environment / dependency alignment issue**, not automatically as a logic error in the research design.
