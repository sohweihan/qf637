# Gold Abnormality as a Nowcasting Escalation Dashboard for Commodity Risk Management

QF637 Commodities Risk Management - SMU MQF

## Overview

This project builds a prototype risk dashboard that uses abnormal gold behaviour to nowcast instability in the market environment around a commodity risk book. When gold's return, volatility, residual behaviour, or cross-market relationships become abnormal, the dashboard escalates the book for VaR review and stress testing.

The project is not framed as a pure prediction model. It does not claim that gold reliably forecasts all future commodity losses. Instead, gold is treated as a real-time cross-market instability indicator that may reveal that the current environment is no longer well represented by trailing risk measures.

## Core Research Question

Can abnormal gold behaviour act as a useful nowcasting signal that prompts timely review of VaR and stress-test exposure in a commodity risk book?

## Implementation Framing

The final implementation separates the signal layer from the valuation layer:

| Layer | Inputs | Output | Purpose |
|---|---|---|---|
| Signal layer | Gold futures, Brent futures, DXY, VIX, US10Y | Green / Amber / Red alarm | Detect abnormal cross-market regimes |
| Benchmark validation layer | Brent futures risk book | VaR, breach, lead-time evidence | Test whether the signal has useful timing |
| Physical-book implementation layer | Physical assessments, basis, storage, freight, inventory, contracts, hedges | Desk P&L, VaR, ES, stress loss, liquidity impact | Apply the trigger to the actual trading book |

The futures-based Gold/Brent signal is not the physical-book valuation model. It is an external market-regime trigger. When the trigger fires, a production desk would revalue its physical commodity book using its own P&L drivers.

## Scope

- **Alarm asset:** Gold futures (`GC=F`)
- **Benchmark risk book:** Brent crude futures (`BZ=F`)
- **Conditioning variables:** DXY (`DX-Y.NYB`), VIX (`^VIX`), US 10Y yield (`^TNX`)
- **Data:** 2007-07-01 to present via yfinance

Brent futures are used first because they provide a liquid, reproducible benchmark for testing the timing value of the Gold alarm. A physical trading desk would replace the benchmark `R_book` with actual physical-book P&L, including basis, storage, freight, inventory, contract terms, and hedge exposures.

## Notebook Pipeline

| # | Notebook | Purpose |
|---|---|---|
| 01 | `notebooks/01_DataExtraction.ipynb` | Download and save raw price data |
| 02 | `notebooks/02_DataCleaning.ipynb` | Strict alignment, log returns, level changes |
| 03 | `notebooks/03_DescriptiveEvidence.ipynb` | Fat-tail tests, static vs rolling correlations, reject a simple forecasting claim |
| 04 | `notebooks/04_NowcastingValidation.ipynb` | Validate the nowcasting direction against stress proxies and named events |
| 05 | `notebooks/05_ConditioningVariableSelection.ipynb` | Test Gold-only versus Gold-plus-conditioning signal variants |
| 06 | `notebooks/06_GoldCenteredAlarmDesign.ipynb` | Build the conditioned challenger alarm selected as the default |
| 07 | `notebooks/07_OperationalUsefulness.ipynb` | Validate the Gold alarm against named events, coverage, review burden, and naive baselines |
| 08 | `notebooks/08_RiskBookVaRStress.ipynb` | Build the Brent futures benchmark book, HS VaR, Kupiec test, and stress scenarios |
| 09 | `notebooks/09_LeadTimeDashboard.ipynb` | Test Gold lead time against Brent-book risk events and export dashboard metrics |
| 10 | `notebooks/10_BrentBaselineComparison.ipynb` | Compare Gold against a Brent-only alarm as a benchmark/control signal |
| 11 | `notebooks/11_BlindSpotAnalysis.ipynb` | Classify events missed by Gold, Brent, both, or neither |
| 12 | `notebooks/12_FalseAlarmJustification.ipynb` | Explain Gold false alarms by trigger family and macro relationship driver |
| 13 | `notebooks/13_Synthesis.ipynb` | Combine the benchmark comparison, limitations, physical-book implementation design, and final claim |

## Alarm Design

The selected default is the conditioned challenger from Notebook 06. It fires when at least two of three signal families exceed threshold 2.0 on a 252-day shifted trailing z-score baseline:

- **Family 1:** `|gold_return_z| > 2.0` or `gold_vol_z > 2.0` - gold's own return/volatility abnormality
- **Family 2:** `|gold_residual_z| > 2.0` - gold's OLS residual versus Brent, DXY, VIX, and US10Y
- **Family 3:** `max(|corr_z|) > 2.0` - instability in gold's rolling 60-day correlations

Dashboard states:

- **Green:** no active signal families, normal monitoring
- **Amber:** one active family, inspect gold components and relationships
- **Red:** two or more active families, review VaR calibration and run stress scenarios

## Key Results

### Gold Alarm Validation

| Metric | Value |
|---|---:|
| Named stress episodes caught | 9 / 9 |
| Stress-proxy coverage at default setting | 68.0% |
| False review rate at default setting | 30.9% |
| Reviews per year | about 10 |
| Gold vs random false reviews at 95% coverage | 35.1% vs 47.9% |

### Brent Futures VaR Backtest

| Metric | Value |
|---|---:|
| Expected breach rate | 5.00% |
| Observed breach rate | 5.82% |
| Kupiec LR statistic | 5.94 |
| Kupiec p-value | 0.015 |
| Reject correct coverage at 5%? | Yes |

The Brent futures benchmark confirms that trailing historical-simulation VaR is statistically miscalibrated during regime changes, motivating a cross-market escalation signal.

### Gold Versus Brent-Only Baseline

The Brent-only baseline is useful as a control signal. It confirms commodity stress with fewer false alarms, but it provides little meaningful lead time because Brent often moves when the risk event is already happening.

| Comparison | Interpretation |
|---|---|
| Brent-only has fewer false reviews | Brent is cleaner for contemporaneous confirmation |
| Gold has higher false reviews | Gold watches broader macro/rates/risk-sentiment stress |
| Gold gives longer lead time | Gold adds early-warning value that Brent-only lacks |
| Drawdowns remain difficult | Slow cumulative losses are hard for both signals |

## Final Claim

This project does not show that Gold predicts Brent losses directly. It shows that abnormal Gold behaviour can act as a useful cross-market escalation signal. A Brent-only alarm is cleaner and better for contemporaneous confirmation of commodity stress, but it gives little meaningful advance warning because Brent often moves when the risk event is already happening. Gold has a higher false-review burden, but provides materially longer lead time and captures broader macro-regime shifts. In production, the futures-based Gold/Brent signal should therefore trigger review of the physical commodity book, whose actual valuation depends on physical assessments, basis, storage, freight, inventory, contracts, and hedges.

## Known Limitations

- **Drawdowns remain hard:** Gold improves the advance-warning framing relative to Brent-only in some tests, but slow cumulative drawdowns remain the weakest event type.
- **Isolated Brent shocks:** Many missed events are sharp Brent moves that are themselves the event, leaving little scope for any one-day-ahead warning signal.
- **False reviews are the cost of broader monitoring:** Gold false alarms often reflect real rates, VIX, DXY, or gold-relationship shifts that do not become Brent-book events within the evaluation window.
- **Proxy-book status:** Brent futures are a validation proxy, not a complete physical trading book.
- **In-sample calibration:** Thresholds are calibrated on the full available sample; a production version should add an out-of-sample monitoring period.

## Requirements

```text
pandas
numpy
yfinance
matplotlib
seaborn
scipy
statsmodels
nbformat
```

## Running The Pipeline

Run notebooks in order from `notebooks/01_DataExtraction.ipynb` to `notebooks/13_Synthesis.ipynb`. The notebooks resolve `data/`, `outputs/`, and `helpers/` from the repository root, so they can be launched from either the repo root or the `notebooks/` folder. The new benchmark and diagnostic notebooks depend on the dashboard outputs from Notebook 09:

- Notebook 10 reads `outputs/step09_lead_time_dashboard/dashboard_metrics.csv`.
- Notebook 11 reads Notebook 09 lead-time outputs and Notebook 10 Brent-baseline outputs.
- Notebook 12 reads Notebook 09 dashboard metrics, the Gold signal components, and the Brent-baseline alarm frame.
- Notebook 13 summarizes outputs from Notebooks 06-12.
