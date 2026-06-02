# Revised Project Proposal

## Project Title

Gold Abnormality as a Nowcasting Escalation Dashboard for Commodity Risk Management

## Revised Objective

This project builds a prototype risk dashboard that uses abnormal gold behaviour to nowcast instability in the market environment around a commodity risk book. When gold's return, volatility, residual behaviour, or cross-market relationships become abnormal, the dashboard escalates the book for VaR review and stress testing.

The project is not framed as a pure prediction model. It does not claim that gold reliably forecasts all future commodity losses. Instead, gold is treated as a real-time cross-market instability indicator that may reveal that the current environment is no longer well represented by trailing risk measures.

## Core Research Question

Can abnormal gold behaviour act as a useful nowcasting signal that prompts timely review of VaR and stress-test exposure in a commodity risk book?

## Operational Premise

Traditional VaR reacts through trailing return windows. During regime shifts, VaR may not fully reflect the new environment until volatility, losses, or breaches have already entered the book.

Gold may be useful because it sits across several market narratives:

- safe-haven demand,
- USD and real-rate pressure,
- liquidity stress,
- geopolitical or inflation stress,
- changing confidence in monetary assets.

The dashboard asks whether gold is behaving unusually now, not whether gold perfectly predicts future losses.

## Scope

The core prototype uses:

- Gold futures as the alarm asset,
- Brent futures as the primary commodity risk-book exposure,
- DXY, VIX, and US 10Y yield as conditioning variables,
- Copper futures as a robustness extension only.

Copper is not part of the primary book yet. It is retained for a later diversified commodity-book robustness check.

## Methodology

1. Data extraction and alignment
   - Download Gold, Brent, DXY, VIX, US10Y, and optional Copper.
   - Use strict aligned daily data to avoid artificial zero returns from forward-filling closed markets.

2. Descriptive evidence
   - Study gold return distribution, volatility, static correlations, direct lead-lag checks, and rolling relationship changes.
   - Use this stage to challenge the project premise before constructing the dashboard.

3. Gold alarm construction
   - Build shifted trailing z-scores to avoid same-day look-ahead contamination.
   - Combine three signal families:
     - gold return or volatility abnormality,
     - gold residual abnormality after conditioning on Brent, DXY, VIX, and US10Y,
     - gold rolling-relationship abnormality.
   - Use a grouped alarm rule where multiple signal families must fire.

4. Risk-book and VaR baseline
   - Build a simple Brent risk book.
   - Compute historical simulation VaR and ES using prior returns only.
   - Add stress scenarios that are triggered by dashboard escalation.

5. Nowcasting and escalation validation
   - Compare gold alarm episodes against VaR breaches, portfolio volatility spikes, and drawdown events.
   - Measure match rate, false-alarm rate, and lead or near-event timing.
   - Treat lead time as a useful performance feature, not the whole premise.

## Current Prototype Evidence

The current compiled notebook outputs suggest the signal is useful but imperfect:

- Gold returns are negatively skewed and fat-tailed.
- Static Gold-VIX correlation is close to zero, but rolling Gold-VIX relationships vary meaningfully.
- The 2.0 threshold creates an alarm-day rate of about 7.75%.
- Gold alarms matched about 60.6% of VaR breach episodes.
- Gold alarms matched about 76.2% of portfolio volatility spike episodes.
- Drawdown matching was weaker, around 33.3%.
- The false-alarm proxy was around 31.9%.

This supports an escalation-dashboard framing. It does not support a claim that gold is a reliable standalone predictor of all commodity stress.

## Expected Dashboard Behaviour

The dashboard should show:

- Green: normal monitoring,
- Amber: inspect gold components and market relationships,
- Red: review VaR calibration and run stress scenarios.

A Red state does not mean that losses are certain. It means that current cross-market conditions are abnormal enough to justify risk review.

## Final Claim To Defend

The strongest defensible claim is:

> Abnormal gold behaviour can provide a useful nowcasting escalation signal for commodity risk management by flagging cross-market instability and prompting VaR review and stress testing before, or around the time, traditional trailing risk measures fully react.

## What Still Needs Work

- Add a contemporaneous nowcasting hit test, not only a lead-time test.
- Compare the gold alarm against naive baselines such as Brent volatility, VIX spikes, and VaR-only triggers.
- Add event-window case studies for COVID 2020, Russia/Ukraine 2022, tariff shock 2025, and recent 2026 stress if data supports it.
- Add a regime interpretation layer that explains whether the alarm likely reflects risk-off, liquidity, geopolitical/inflation, or monetary tightening stress.
- Tune threshold, persistence, and cooldown rules to reduce false alarms.
