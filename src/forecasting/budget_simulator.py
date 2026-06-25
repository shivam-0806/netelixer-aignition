"""Budget Simulator — combines Prophet + Quantile model outputs per channel."""

import pandas as pd
from .prophet_model import forecast_prophet
from .quantile_model import simulate_budget


def combine_forecast_outputs(
    prophet_output: dict,
    quantile_output: dict,
    min_spread: float = 0.15,
    max_downside: float | None = None,
    max_upside: float | None = None,
) -> dict:
    """
    Combine Prophet's median anchor with XGBoost uncertainty ratios.

    ``min_spread`` enforces a minimum downside/upside width around the median so
    quantile model flatlines do not produce misleadingly narrow ranges.

    ``max_downside`` and ``max_upside`` optionally cap extreme production bands.
    For Google, these caps are justified by the observed 30-day backtest behavior
    where the model had <15% AAPE and over-covered the actuals.
    """
    rev_median = float(prophet_output["revenue_median"])

    # Original production logic:
    # if q_out["revenue_median"] > 0:
    #     pct_low = q_out["revenue_low"] / q_out["revenue_median"]
    #     pct_high = q_out["revenue_high"] / q_out["revenue_median"]
    # else:
    #     pct_low, pct_high = 0.85, 1.15
    # rev_low = rev_median * pct_low
    # rev_high = rev_median * pct_high
    if quantile_output["revenue_median"] > 0:
        raw_pct_low = quantile_output["revenue_low"] / quantile_output["revenue_median"]
        raw_pct_high = quantile_output["revenue_high"] / quantile_output["revenue_median"]
        pct_low = min(raw_pct_low, 1 - min_spread)
        pct_high = max(raw_pct_high, 1 + min_spread)
    else:
        pct_low, pct_high = 1 - min_spread, 1 + min_spread

    rev_low = min(float(prophet_output["revenue_low"]), rev_median * pct_low)
    rev_high = max(float(prophet_output["revenue_high"]), rev_median * pct_high)

    if rev_low > rev_median:
        rev_low = rev_median * (1 - min_spread)
    if rev_high < rev_median:
        rev_high = rev_median * (1 + min_spread)

    # Original:
    # return the fully conservative envelope from Prophet + XGBoost.
    #
    # Corrected: allow channel-specific calibration to prevent Google's
    # production forecast from becoming much wider than its backtest evidence
    # supports. This is intentionally optional so sparse/noisy channels can
    # remain conservative.
    if max_downside is not None:
        rev_low = max(rev_low, rev_median * (1 - max_downside))
    if max_upside is not None:
        rev_high = min(rev_high, rev_median * (1 + max_upside))

    return {
        "revenue_low": round(float(rev_low), 2),
        "revenue_median": round(float(rev_median), 2),
        "revenue_high": round(float(rev_high), 2),
    }


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

        # Original weighted ensemble and hybrid-anchor code is preserved in
        # combine_forecast_outputs(). Use the shared helper here so production
        # simulation and backtesting apply the same minimum spread.
        # Google has complete observed revenue coverage and strong backtest
        # performance. Cap its production band to a business-realistic range
        # while leaving weaker/estimated channels more conservative.
        if channel == "Google":
            combined = combine_forecast_outputs(
                p_out,
                q_out,
                max_downside=0.35,
                max_upside=0.45,
            )
        else:
            combined = combine_forecast_outputs(p_out, q_out)
        rev_low = combined["revenue_low"]
        rev_median = combined["revenue_median"]
        rev_high = combined["revenue_high"]

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
