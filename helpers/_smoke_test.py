"""Run the full helper pipeline against data/processed and print key checks.

Usage: python -m helpers._smoke_test
"""

from __future__ import annotations

from . import riskbook as riskbook_mod
from . import var as var_mod
from .pipeline import build_all


def main() -> None:
    results = build_all()

    market_vars = results["market_vars"]
    signal_components = results["signal_components"]
    alarm_frame = results["alarm_frame"]
    book = results["book"]
    dashboard = results["dashboard_metrics"]
    stress = results["stress_results"]

    print("market_vars:", market_vars.shape, "|", market_vars.index.min().date(), "->", market_vars.index.max().date())
    print("dashboard_metrics:", dashboard.shape)

    kupiec = var_mod.kupiec_pof_test(dashboard["var_breach"], dashboard["hs_var_return"])
    print("\nKupiec POF test:")
    for key, value in kupiec.items():
        print(f"  {key}: {value}")

    latest_date = dashboard.index[-1]
    print("\nLatest day:", latest_date.date())
    print("  dashboard_state:", dashboard.loc[latest_date, "dashboard_state"])
    print("  conditioned_alarm_score:", alarm_frame.loc[latest_date, "conditioned_alarm_score"])
    print("  gold_corr_r_Brent_z:", signal_components.loc[latest_date, "gold_corr_r_Brent_z"])
    print("  recommended_action:", dashboard.loc[latest_date, "recommended_action"])

    print("\nBook snapshot (latest day):")
    print("  position_bbl:", book.loc[latest_date, "position_bbl"])
    print("  exposure_usd: {:,.2f}".format(book.loc[latest_date, "exposure_usd"]))
    print("  pnl_usd: {:,.2f}".format(book.loc[latest_date, "pnl_usd"]))
    print("  nav:", book.loc[latest_date, "nav"])
    print("  drawdown:", book.loc[latest_date, "drawdown"])
    print("  realized_vol_20d:", book.loc[latest_date, "realized_vol_20d"])

    worst = stress.loc[stress["stress_pnl_usd"].idxmin()]
    print("\nWorst stress scenario:")
    print(
        "  {} | shock {} | pnl_usd {:,.2f} | nav_after {:.2f}".format(
            worst["scenario_name"],
            f"{worst['shock_pct']:.2%}",
            worst["stress_pnl_usd"],
            worst["nav_after_stress"],
        )
    )

    print("\nTrade blotter:")
    blotter = riskbook_mod.trade_blotter(None, results["prices"])
    print(blotter[["trade_id", "side", "position_bbl", "entry_date", "entry_price", "current_price", "unrealized_pnl_usd"]].to_string(index=False))


if __name__ == "__main__":
    main()
