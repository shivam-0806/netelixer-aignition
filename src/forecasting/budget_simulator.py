"""Budget Simulator — combines Prophet + Quantile model outputs per channel."""

import pandas as pd
from .prophet_model import forecast_prophet
from .quantile_model import simulate_budget


def run_full_simulation(
    weekly_df:       pd.DataFrame,
    prophet_models:  dict,
    quantile_models: dict,
    budget_inputs:   dict,   # {"Google": 50000, "Bing": 10000, "Meta": 30000}
    horizon_days:    int,
) -> dict:
    """
    Run the full forecast simulation combining both models.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly aggregated data (from prepare_forecast_input).
    prophet_models : dict
        {channel: trained Prophet model}.
    quantile_models : dict
        {channel: {quantile: trained XGBRegressor}}.
    budget_inputs : dict
        {channel: planned_spend}.
    horizon_days : int
        30, 60, or 90.

    Returns
    -------
    dict
        Channel-level + "blended" aggregate forecast results.
    """
    results = {}
    total_spend = sum(budget_inputs.values())

    for channel, planned_spend in budget_inputs.items():
        ch_df = weekly_df[weekly_df["channel"] == channel]

        if len(ch_df) == 0 or planned_spend <= 0:
            results[channel] = {
                "planned_spend": planned_spend,
                "revenue_low": 0, "revenue_median": 0, "revenue_high": 0,
                "roas_low": 0, "roas_median": 0, "roas_high": 0,
            }
            continue

        p_out = forecast_prophet(prophet_models[channel], planned_spend, horizon_days)
        q_out = simulate_budget(
            quantile_models[channel], planned_spend, horizon_days, ch_df
        )

        # Widen bounds to capture uncertainty from both models
        rev_low    = min(p_out["revenue_low"],    q_out["revenue_low"])
        rev_median = (p_out["revenue_median"] + q_out["revenue_median"]) / 2
        rev_high   = max(p_out["revenue_high"],   q_out["revenue_high"])

        results[channel] = {
            "planned_spend":  planned_spend,
            "revenue_low":    round(rev_low, 2),
            "revenue_median": round(rev_median, 2),
            "revenue_high":   round(rev_high, 2),
            "roas_low":       round(rev_low / planned_spend, 2) if planned_spend else 0,
            "roas_median":    round(rev_median / planned_spend, 2) if planned_spend else 0,
            "roas_high":      round(rev_high / planned_spend, 2) if planned_spend else 0,
        }

    # Blended totals (exclude "blended" key itself from sum)
    channel_results = {k: v for k, v in results.items() if k != "blended"}
    results["blended"] = {
        "total_spend":    total_spend,
        "revenue_low":    round(sum(r["revenue_low"] for r in channel_results.values()), 2),
        "revenue_median": round(sum(r["revenue_median"] for r in channel_results.values()), 2),
        "revenue_high":   round(sum(r["revenue_high"] for r in channel_results.values()), 2),
    }
    blended = results["blended"]
    blended["roas_low"]    = round(blended["revenue_low"] / total_spend, 2) if total_spend else 0
    blended["roas_median"] = round(blended["revenue_median"] / total_spend, 2) if total_spend else 0
    blended["roas_high"]   = round(blended["revenue_high"] / total_spend, 2) if total_spend else 0

    return results
