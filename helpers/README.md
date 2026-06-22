# Helpers

Reusable code pulled out of the notebooks for the dashboard prototype.

## `data.py`

Loads and refreshes market data.

- `extract_price_panel(raw)`: reshapes yfinance output into asset columns.
- `download_prices(start, end)`: downloads Gold, Brent, DXY, VIX, and US10Y prices.
- `compute_market_vars(prices)`: builds log returns for price assets and level changes for VIX/US10Y.
- `load_prices(processed_dir)`: reads `prices_clean_core.parquet`.
- `load_market_vars(processed_dir)`: reads `market_vars_core.parquet`.
- `refresh_market_data(processed_dir, start, end)`: downloads, cleans, and overwrites processed parquet files.

## `signals.py`

Builds Gold abnormality signal components.

- `trailing_zscore(series, window)`: shifted trailing z-score with no lookahead.
- `rolling_residual(y, x, window)`: one-step-ahead OLS residual against conditioning variables.
- `compute_gold_signal_components(market_vars)`: returns Gold return, volatility, residual, and relationship z-scores.

## `alarm.py`

Turns Gold signal components into dashboard states.

- `dashboard_state(score)`: maps alarm score to `Green`, `Amber`, or `Red`.
- `recommended_action(states)`: maps dashboard state to risk action text.
- `build_gold_alarm_frame(signal_components)`: combines return/vol, residual, and relationship alarms.
- `event_starts(flag)`: returns dates where a flag turns on.
- `apply_cooldown(flag, cooldown_days)`: keeps only separated alarm firings.

## `riskbook.py`

Builds the Brent futures proxy risk book.

- `build_brent_book(trades, prices, nav0)`: marks Brent futures trades to market and returns position, exposure, P&L, return, NAV, drawdown, and realized volatility.
- `trade_blotter(trades, prices, as_of)`: returns per-trade entry price, current price, and unrealized P&L.

The current book is a Brent futures proxy book. It does not model physical basis, freight, storage, counterparty, or cargo timing.

## `var.py`

Historical simulation risk metrics.

- `historical_var_es(returns, window, confidence)`: rolling VaR and expected shortfall using prior returns only.
- `var_breach_flags(returns, hs_var_return)`: flags realized returns below VaR.
- `kupiec_pof_test(breach_flags, hs_var_return, confidence)`: checks whether breach frequency matches the VaR confidence level.

## `stress.py`

Brent event stress for the existing risk book.

- `fixed_shocks(shock_pcts)`: creates fixed Brent price shock scenarios.
- `run_stress_scenarios(book, prices, ...)`: applies fixed shocks and returns chart-ready stress P&L, stressed NAV, breach flags, and optional margin/liquidity fields.
- `reverse_stress_to_loss_limit(book, loss_limit_usd, as_of)`: returns the Brent shock needed to lose the specified amount.

Margin stress is optional. If `initial_margin_per_contract_usd` is not supplied, margin fields stay blank and no assumption is used. If supplied, the helper estimates base margin, stressed margin, incremental margin call, and total cash need.

## `dashboard.py`

Builds dashboard-ready metrics and event matching tables.

- `build_dashboard_metrics(book, alarm_frame)`: joins book metrics, VaR/ES, Gold alarm state, volatility spikes, and drawdown flags.
- `match_leads(alarm_dates, event_dates, lookback_days)`: matches events to prior alarms.
- `lead_summary_table(alarm_dates, event_families, lookback_days)`: summarizes alarm lead time by event family.
- `false_alarm_table_and_summary(alarm_dates, all_event_dates, lookback_days)`: measures alarms not followed by events.

## `pipeline.py`

One-call helper pipeline.

- `build_all(...)`: loads data, builds Gold signals, builds the alarm frame, builds the Brent risk book, builds dashboard metrics, and returns stress outputs.

## `_smoke_test.py`

Runnable sanity check.

Run from the repo root:

```bash
python -m helpers._smoke_test
```

It checks the helper chain against `data/processed` and prints key dashboard, VaR, book, stress, and blotter outputs.
