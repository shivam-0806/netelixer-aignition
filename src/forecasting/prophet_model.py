"""Prophet-based time-series forecasting for revenue by channel."""

import pandas as pd
import numpy as np
from prophet import Prophet
import logging

from .features import add_promo_flags

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
    channel_df = add_promo_flags(channel_df)

    prophet_input = channel_df[["week", "revenue", "spend", "is_promo"]].rename(
        columns={"week": "ds", "revenue": "y"}
    )

    model = Prophet(
        interval_width=0.80,            # 80% confidence interval
        yearly_seasonality=True,
        weekly_seasonality=False,        # Already weekly-aggregated
        daily_seasonality=False,
        seasonality_mode="additive",     # Additive is more stable for log-transformed highly volatile data
    )

    # Add built-in holidays
    model.add_country_holidays(country_name='US')

    # Spend and promo flags as external regressors
    model.add_regressor("spend")
    model.add_regressor("is_promo")

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
    n_weeks = max(horizon_days // 7, 1)
    future = model.make_future_dataframe(periods=n_weeks, freq="W")

    # Distribute planned spend evenly across future weeks
    weekly_spend = future_spend / n_weeks
    future["spend"] = weekly_spend
    
    # Add promo flags to future dataframe
    future["week"] = future["ds"]
    future = add_promo_flags(future)
    future = future.drop(columns=["week"])

    forecast = model.predict(future)

    # Extract only the future period rows
    future_forecast = forecast.tail(n_weeks)

    total_low    = future_forecast["yhat_lower"].clip(lower=0).sum()
    total_median = future_forecast["yhat"].clip(lower=0).sum()
    total_high   = future_forecast["yhat_upper"].clip(lower=0).sum()

    return {
        "revenue_low":    round(total_low, 2),
        "revenue_median": round(total_median, 2),
        "revenue_high":   round(total_high, 2),
        "roas_low":       round(total_low / future_spend, 2) if future_spend else 0,
        "roas_median":    round(total_median / future_spend, 2) if future_spend else 0,
        "roas_high":      round(total_high / future_spend, 2) if future_spend else 0,
    }
