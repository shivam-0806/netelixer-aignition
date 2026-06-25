"""Prophet-based time-series forecasting for revenue by channel."""

import math
import logging
import pandas as pd
from prophet import Prophet

# Suppress noisy Prophet/cmdstanpy logs
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def train_prophet(weekly_df: pd.DataFrame, channel: str) -> Prophet:
    """
    Train a Prophet model for a single channel using weekly data.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Output of ``prepare_forecast_input()`` with columns: week, channel, spend, revenue.
    channel : str
        Channel name (e.g. "Google", "Bing", "Meta").

    Returns
    -------
    Prophet
        Fitted Prophet model.
    """
    channel_df = weekly_df[weekly_df["channel"] == channel].copy()

    prophet_input = channel_df[["week", "revenue", "spend"]].rename(
        columns={"week": "ds", "revenue": "y"}
    )

    # model = Prophet(
    #     interval_width=0.80,            # 80% confidence interval
    #     yearly_seasonality=True,
    #     weekly_seasonality=False,        # Already weekly-aggregated
    #     daily_seasonality=False,
    #     seasonality_mode="multiplicative",  # Better for marketing data with growth
    # )

    model = Prophet(
        interval_width=0.75,                # Slightly tighter confidence interval
        yearly_seasonality="auto",        # Let it decide if yearly patterns exist
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.5,        # Makes trend more flexible to recent changes
        changepoint_range=0.95, 
        # seasonality_prior_scale=10.0,       # Allows seasonality to have a stronger impact
    )

    # Spend as an external regressor — budget drives revenue
    model.add_regressor("spend")

    model.fit(prophet_input)
    return model


def forecast_prophet(model: Prophet, future_spend: float,
                     horizon_days: int = 90) -> dict:
    """
    Generate a probabilistic forecast for a given horizon and planned spend.

    Parameters
    ----------
    model : Prophet
        Trained Prophet model.
    future_spend : float
        Total planned spend for the forecast period.
    horizon_days : int
        30, 60, or 90.

    Returns
    -------
    dict
        Keys: revenue_low, revenue_median, revenue_high,
              roas_low, roas_median, roas_high
    """
    # Original:
    # n_weeks = max(horizon_days // 7, 1)
    # future = model.make_future_dataframe(periods=n_weeks, freq="W")
    #
    # Corrected: include partial weeks and keep Prophet's future weekly anchor
    # aligned with the Monday week starts used by feature_engineering.py.
    n_weeks = max(math.ceil(horizon_days / 7), 1)
    equivalent_weeks = horizon_days / 7
    horizon_scale = equivalent_weeks / n_weeks
    future = model.make_future_dataframe(periods=n_weeks, freq="W-MON")

    # Original:
    # weekly_spend = future_spend / n_weeks
    #
    # Corrected: convert exact-period spend into a full-week spend rate, then
    # scale forecast totals back to the exact 30/60/90-day window below.
    weekly_spend = future_spend / equivalent_weeks if equivalent_weeks else 0
    future["spend"] = weekly_spend

    forecast = model.predict(future)

    # Extract only the future period rows
    future_forecast = forecast.tail(n_weeks)

    total_low = future_forecast["yhat_lower"].clip(lower=0).sum() * horizon_scale
    total_median = future_forecast["yhat"].clip(lower=0).sum() * horizon_scale
    total_high = future_forecast["yhat_upper"].clip(lower=0).sum() * horizon_scale

    return {
        "revenue_low":    round(float(total_low), 2),
        "revenue_median": round(float(total_median), 2),
        "revenue_high":   round(float(total_high), 2),
        "roas_low":       round(float(total_low) / future_spend, 2) if future_spend else 0,
        "roas_median":    round(float(total_median) / future_spend, 2) if future_spend else 0,
        "roas_high":      round(float(total_high) / future_spend, 2) if future_spend else 0,
    }
