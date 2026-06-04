# Revised Project Proposal

## Project Title

Gold Abnormality as a Nowcasting Escalation Dashboard for Commodity Risk Management

## Revised Objective

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

The core prototype uses:

- Gold futures (`GC=F`) as the alarm asset,
- Brent futures (`BZ=F`) as the primary commodity risk-book exposure,
- DXY, VIX, and US 10Y yield as conditioning variables.

Data runs from 2007-07-01 to present. Brent (`BZ=F`) is the binding start-date constraint on yfinance, with data available from 2007-07-30. This start date captures the full GFC (2008–2009), which is the most important missing regime in shorter datasets.

## Methodology

1. Data extraction and alignment
   - Download Gold, Brent, DXY, VIX, and US10Y from 2007-07-01.
   - Use strict dropna alignment to avoid artificial zero returns from forward-filling closed markets.
   - Log returns for price assets; level changes for VIX and US10Y.

2. Descriptive evidence
   - Study gold return distribution, volatility, static correlations, direct lead-lag checks, and rolling relationship changes.
   - Use this stage to challenge the project premise before constructing the dashboard.
   - The lead-lag test deliberately uses raw returns to kill the forecasting hypothesis; finding no return-level lead-lag is the intended result that motivates the nowcasting pivot.

3. Nowcasting validation
   - Test whether Gold abnormality is better interpreted as a current-state instability signal.
   - Build a preliminary Gold nowcasting score before the final alarm.
   - Compare the preliminary score against simple contemporaneous stress proxies (Brent volatility, Brent return shocks, VIX level spikes).
   - Check known event windows to see whether Gold abnormality behaves sensibly during named market regimes.

4. Conditioning-variable selection
   - Compare Gold-only against Gold plus each conditioning variable using a mean-score approach.
   - Test whether Brent, DXY, VIX, and US10Y improve nowcasting coverage, false review burden, or review frequency.
   - Key finding: averaging conditioning variables into the trigger score reduces coverage (denominator dilution). Variables are retained as diagnostic context and as separate signal families in the conditioned alarm.

5. Gold alarm construction
   - Build shifted trailing z-scores to avoid same-day look-ahead contamination.
   - Apply `clip(lower=0)` to `gold_vol_z` — only vol spikes signal stress; unusually quiet vol is not a stress signal.
   - Build a Gold-primary trigger using gold return and gold volatility abnormality.
   - Build a conditioned challenger alarm using an OR-across-families logic: fires if 2 or more of 3 signal families (return/vol, OLS residual, rolling-correlation relationships) exceed the threshold simultaneously.
   - **The conditioned challenger is the selected default** based on operational evaluation in Notebook 07.

6. Operational usefulness test
   - **Primary evaluation: named stress episode validation.** For each of 9 historical stress episodes, check whether the alarm fired. A miss is a specific named failure, not a fraction of an abstract count.
   - Secondary evaluation: stress-proxy coverage (68%), false review rate (30.9%), and review frequency (~10/year).
   - Coverage-constrained sweep: find the threshold required to achieve 90/95/99/100% stress-proxy coverage and its associated false review cost.
   - Fixed-recall comparison: at equal firing frequency, compare gold's false review rate against a random baseline.
   - Asymmetric cost framing: missing a stress event is a governance failure; an unnecessary review is a low-cost inconvenience. Coverage is a constraint, not a tradeoff.

7. Risk-book and VaR baseline
   - Build a simple Brent-only risk book.
   - Compute historical simulation VaR and ES using prior returns only (no look-ahead).
   - Formally backtest VaR calibration with the Kupiec Proportion of Failures test.
   - Add stress scenarios triggered by dashboard escalation: risk-off growth shock, geopolitical inflation shock, liquidity liquidation, and monetary tightening.

8. Dashboard escalation validation
   - Compare gold alarm episodes against VaR breaches, portfolio volatility spikes, and drawdown events.
   - Report Wilson confidence intervals on all match rates — sample sizes are small.
   - Report false alarm rate at both 30-day and 10-day forward windows.
   - Treat lead time as a useful feature, not the whole premise.

## Notebook Sequence

1. `01_DataExtraction.ipynb`
2. `02_DataCleaning.ipynb`
3. `03_DescriptiveEvidence.ipynb`
4. `04_NowcastingValidation.ipynb`
5. `05_ConditioningVariableSelection.ipynb`
6. `06_GoldCenteredAlarmDesign.ipynb`
7. `07_OperationalUsefulness.ipynb`
8. `08_RiskBookVaRStress.ipynb`
9. `09_LeadTimeDashboard.ipynb`

## Current Prototype Evidence

The pipeline has been fully executed on 4,685 daily observations from 2007-07-31 to 2026-06-04.

**Gold distributional properties (Notebook 03):**
- Skew: -0.53, excess kurtosis: 7.49, Jarque-Bera p-value: 0. Fat tails confirmed.
- Gold-VIX rolling 60d correlation ranges from -0.63 to +0.59 around a near-zero static mean — the relationship is genuinely unstable across regimes.

**Conditioning variable selection (Notebook 05):**
- Gold-only: 56.6% coverage, 28.2% false review rate.
- Adding any conditioning variable to the mean score reduces coverage. Gold-only wins the score-average comparison.
- The conditioned challenger's OR-across-families logic avoids this denominator problem.

**Named stress episode validation — primary result (Notebook 07):**
All 9 named historical stress episodes were caught by the conditioned challenger:

| Episode | First alarm date |
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

**Operational performance — conditioned challenger default (Notebook 07):**
- Stress-proxy coverage: 68.0%, false review rate: 30.9%, reviews per year: ~10.
- At 95% stress-proxy coverage: threshold 0.80, 35.1% false reviews, 24.2 reviews/year.
- Gold at 95% coverage: 35.1% false reviews vs random's 47.9% at equal frequency — gold adds genuine timing discrimination.
- Missed events are dominated by isolated Brent return shocks (41 of 56) where gold showed no safe-haven reaction. This is a bounded, interpretable limitation.

**VaR backtest (Notebook 08):**
- Observed breach rate 5.82% vs 5.00% expected.
- Kupiec POF test: LR = 5.94, p = 0.015 — VaR is statistically miscalibrated. This directly motivates the cross-asset escalation signal.
- Max drawdown: -94.5% (includes GFC negative-price oil event).

**Lead-time results (Notebook 09):**
- VaR breach: 60.1% match rate, 8.0 days avg lead (95% CI: 53.6%–66.2%).
- Portfolio vol spike: 69.6% match rate, 9.4 days avg lead (95% CI: 49.1%–84.4%).
- Drawdown: 0.0% match rate — the alarm does not anticipate slow-developing cumulative losses. This is a structural limitation: the alarm detects sudden regime shifts, not gradual portfolio deterioration.
- False alarm: 31.4% at 30-day window; 54.3% at 10-day window.

This evidence supports an escalation-dashboard framing. It does not support a claim that gold is a reliable standalone predictor of all commodity stress.

## Expected Dashboard Behaviour

The dashboard shows three states based on how many signal families are simultaneously elevated:

- **Green** (0 families firing): normal monitoring.
- **Amber** (1 family firing): inspect gold components and market relationships.
- **Red** (2+ families firing): review VaR calibration and run stress scenarios.

A Red state does not mean that losses are certain. It means that current cross-market conditions are abnormal enough to justify risk review. Protection occurs through the stress testing process that the alarm triggers, not through the alarm itself.

## Final Claim To Defend

The strongest defensible claim is:

> Abnormal gold behaviour — detected across multiple signal families including own return/vol, OLS residual, and rolling relationship instability — can provide a useful nowcasting escalation signal for commodity risk management by flagging cross-market instability and prompting VaR review and stress testing. The signal caught all 9 major historical stress episodes in the 2007–2026 dataset and leads VaR breaches with an 8-day average. Its value lies in detecting sudden systemic regime shifts, not in predicting gradual drawdowns or isolated commodity price shocks.

## Known Limitations

- **Drawdown prediction:** The alarm does not predict slow-developing portfolio drawdowns (0 of 14 drawdown events matched). This is structural, not a calibration failure.
- **Isolated commodity shocks:** 41 of 56 missed stress proxy events are pure Brent return shocks with no gold safe-haven reaction — the alarm's blind spot is well-defined.
- **Small sample sizes:** Vol spike (23 events) and drawdown (14 events) samples are too small for precise inference. Confidence intervals are wide.
- **In-sample calibration:** All thresholds and parameters are calibrated on the full 2007–2026 sample. No out-of-sample test has been conducted.
- **VaR breach base rate:** With ~228 VaR breaches over 19 years, the 30-day false alarm proxy has a high base rate. The lead-time evaluation's primary evidence is the named episode table, not the false alarm rate.

## Next Step

Build the interactive dashboard using the `dashboard_metrics.csv` output from Notebook 09.
