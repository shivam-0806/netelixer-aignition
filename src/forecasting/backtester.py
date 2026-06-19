"""Backtesting engine — walk-forward cross-validation for forecast models."""

import logging
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet

from .prophet_model import train_prophet
from .quantile_model import train_quantile_models, build_features, FEATURE_COLS, QUANTILES
from .features import add_promo_flags

logger = logging.getLogger(__name__)

# Suppress noisy Prophet/cmdstanpy logs during backtesting
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def compute_backtest_metrics(fold_results: list[dict]) -> dict:
    """
    Compute aggregate accuracy metrics from a list of backtest folds.

    Parameters
    ----------
    fold_results : list[dict]
        Each dict has keys: actual, predicted, pred_low, pred_high,
        train_start, train_end, test_start, test_end.

    Returns
    -------
    dict
        MAE, MAPE, RMSE, coverage, directional_accuracy, confidence_score,
        confidence_label, n_folds.
    """
    if not fold_results:
        return _empty_metrics()

    actuals = np.array([f["actual"] for f in fold_results])
    predictions = np.array([f["predicted"] for f in fold_results])
    pred_lows = np.array([f["pred_low"] for f in fold_results])
    pred_highs = np.array([f["pred_high"] for f in fold_results])

    # --- Core error metrics ---
    errors = actuals - predictions
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # WMAPE (Weighted MAPE) — robust against small-denominator folds
    # WMAPE = sum(|errors|) / sum(|actuals|), avoids inflated % when some folds are small
    total_actual = np.sum(np.abs(actuals))
    if total_actual > 0:
        mape = float(np.sum(abs_errors) / total_actual * 100)
    else:
        mape = 0.0

    # --- Prediction interval coverage ---
    covered = np.sum((actuals >= pred_lows) & (actuals <= pred_highs))
    coverage = float(covered / len(actuals) * 100)

    # --- Directional accuracy ---
    if len(actuals) >= 2:
        actual_dirs = np.diff(actuals) >= 0
        pred_dirs = np.diff(predictions) >= 0
        directional_accuracy = float(np.mean(actual_dirs == pred_dirs) * 100)
    else:
        directional_accuracy = 50.0  # not enough data, assume chance

    # --- Composite confidence score (0–100) ---
    mape_score = max(0.0, 100.0 - mape * 2)
    coverage_score = min(100.0, coverage * 1.25)
    direction_score = directional_accuracy

    confidence_score = round(
        0.40 * mape_score + 0.35 * coverage_score + 0.25 * direction_score, 1
    )
    confidence_score = max(0.0, min(100.0, confidence_score))

    confidence_label = _score_to_label(confidence_score)

    return {
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "coverage": round(coverage, 1),
        "directional_accuracy": round(directional_accuracy, 1),
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "n_folds": len(fold_results),
        "folds": fold_results,
    }


def _score_to_label(score: float) -> str:
    if score >= 90:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Fair"
    else:
        return "Poor"


def _empty_metrics() -> dict:
    return {
        "mae": 0, "mape": 0, "rmse": 0,
        "coverage": 0, "directional_accuracy": 0,
        "confidence_score": 0, "confidence_label": "Insufficient Data",
        "n_folds": 0, "folds": [],
    }


# ─────────────────────────────────────────────────────────────
# Prophet backtesting
# ─────────────────────────────────────────────────────────────

def backtest_prophet(weekly_df: pd.DataFrame, channel: str,
                     horizon_days: int = 60,
                     min_train_weeks: int = 26) -> list[dict]:
    """
    Walk-forward backtest for the Prophet model on a single channel.

    Returns a list of fold result dicts.
    """
    n_weeks = max(horizon_days // 7, 1)
    ch_df = weekly_df[weekly_df["channel"] == channel].sort_values("week").reset_index(drop=True)

    if len(ch_df) < min_train_weeks + n_weeks:
        logger.warning(f"Not enough data for Prophet backtest on {channel}: "
                       f"{len(ch_df)} weeks < {min_train_weeks + n_weeks} needed")
        return []

    folds = []
    start = min_train_weeks

    while start + n_weeks <= len(ch_df):
        train_data = ch_df.iloc[:start].copy()
        test_data = ch_df.iloc[start:start + n_weeks].copy()

        try:
            # Add promo flags
            train_data = add_promo_flags(train_data)
            
            # Build prophet input from training data
            prophet_input = train_data[["week", "revenue", "spend", "is_promo"]].rename(
                columns={"week": "ds", "revenue": "y"}
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Prophet(
                    interval_width=0.80,
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="additive",
                )
                model.add_country_holidays(country_name='US')
                model.add_regressor("spend")
                model.add_regressor("is_promo")
                model.fit(prophet_input)

            # Create future dataframe and merge spend by date (not position)
            future = model.make_future_dataframe(periods=n_weeks, freq="W")

            # Build a lookup of date -> spend from both train and test data
            spend_lookup = pd.concat([
                train_data[["week", "spend"]].rename(columns={"week": "ds"}),
                test_data[["week", "spend"]].rename(columns={"week": "ds"}),
            ])
            # Normalize dates for matching
            spend_lookup["ds"] = pd.to_datetime(spend_lookup["ds"])
            future["ds"] = pd.to_datetime(future["ds"])

            # Merge spend onto future dataframe by date
            future = future.merge(spend_lookup, on="ds", how="left")
            # Fill any unmatched dates with average spend from training
            avg_spend = train_data["spend"].mean()
            future["spend"] = future["spend"].fillna(avg_spend)

            # Add promo flags to future
            future["week"] = future["ds"]
            future = add_promo_flags(future)

            forecast = model.predict(future)
            future_forecast = forecast.tail(n_weeks)

            predicted = future_forecast["yhat"].clip(lower=0).sum()
            pred_low = future_forecast["yhat_lower"].clip(lower=0).sum()
            pred_high = future_forecast["yhat_upper"].clip(lower=0).sum()
            actual = test_data["revenue"].sum()

            folds.append({
                "actual": round(float(actual), 2),
                "predicted": round(float(predicted), 2),
                "pred_low": round(float(pred_low), 2),
                "pred_high": round(float(pred_high), 2),
                "train_start": str(train_data["week"].iloc[0].date()),
                "train_end": str(train_data["week"].iloc[-1].date()),
                "test_start": str(test_data["week"].iloc[0].date()),
                "test_end": str(test_data["week"].iloc[-1].date()),
                "model": "Prophet",
            })
        except Exception as e:
            logger.warning(f"Prophet backtest fold failed for {channel}: {e}")

        start += n_weeks

    return folds


# ─────────────────────────────────────────────────────────────
# Quantile model backtesting
# ─────────────────────────────────────────────────────────────

def backtest_quantile(weekly_df: pd.DataFrame, channel: str,
                      horizon_days: int = 60,
                      min_train_weeks: int = 26) -> list[dict]:
    """
    Walk-forward backtest for the XGBoost quantile models on a single channel.

    Returns a list of fold result dicts.
    """
    n_weeks = max(horizon_days // 7, 1)
    ch_df = weekly_df[weekly_df["channel"] == channel].sort_values("week").reset_index(drop=True)

    if len(ch_df) < min_train_weeks + n_weeks:
        logger.warning(f"Not enough data for Quantile backtest on {channel}: "
                       f"{len(ch_df)} weeks < {min_train_weeks + n_weeks} needed")
        return []

    folds = []
    start = min_train_weeks

    while start + n_weeks <= len(ch_df):
        train_data = ch_df.iloc[:start].copy()
        test_data = ch_df.iloc[start:start + n_weeks].copy()

        try:
            train_feat = build_features(train_data)
            X_train = train_feat[FEATURE_COLS]
            y_train = train_feat["revenue"]

            # Train quantile models on training window
            import xgboost as xgb  # noqa: local import for backtesting isolation
            models = {}
            for q in QUANTILES:
                m = xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    verbosity=0,
                )
                m.fit(X_train, y_train)
                models[q] = m

            # Predict on test weeks using actual spend
            test_feat = build_features(
                pd.concat([train_data, test_data]).sort_values("week")
            ).tail(n_weeks)
            X_test = test_feat[FEATURE_COLS]

            preds = {q: np.clip(models[q].predict(X_test), 0, None) for q in QUANTILES}

            predicted = float(preds[0.50].sum())
            pred_low = float(preds[0.10].sum())
            pred_high = float(preds[0.90].sum())
            actual = float(test_data["revenue"].sum())

            folds.append({
                "actual": round(actual, 2),
                "predicted": round(predicted, 2),
                "pred_low": round(pred_low, 2),
                "pred_high": round(pred_high, 2),
                "train_start": str(train_data["week"].iloc[0].date()),
                "train_end": str(train_data["week"].iloc[-1].date()),
                "test_start": str(test_data["week"].iloc[0].date()),
                "test_end": str(test_data["week"].iloc[-1].date()),
                "model": "XGBoost Quantile",
            })
        except Exception as e:
            logger.warning(f"Quantile backtest fold failed for {channel}: {e}")

        start += n_weeks

    return folds


# ─────────────────────────────────────────────────────────────
# Channel-level orchestrator
# ─────────────────────────────────────────────────────────────

def backtest_channel(weekly_df: pd.DataFrame, channel: str,
                     horizon_days: int = 60) -> dict:
    """
    Run walk-forward backtests for both Prophet and XGBoost on a channel,
    then compute blended metrics (matching production blending logic).

    Returns
    -------
    dict
        Keys: prophet_metrics, quantile_metrics, blended_metrics,
              blended_folds (list of blended fold results).
    """
    min_weeks = 26  # Consistent across models so folds align
    prophet_folds = backtest_prophet(weekly_df, channel, horizon_days, min_train_weeks=min_weeks)
    quantile_folds = backtest_quantile(weekly_df, channel, horizon_days, min_train_weeks=min_weeks)

    prophet_metrics = compute_backtest_metrics(prophet_folds)
    quantile_metrics = compute_backtest_metrics(quantile_folds)

    # Blend folds (same logic as budget_simulator: avg median, min low, max high)
    blended_folds = []
    n_common = min(len(prophet_folds), len(quantile_folds))

    for i in range(n_common):
        pf = prophet_folds[i]
        qf = quantile_folds[i]
        blended_folds.append({
            "actual": pf["actual"],  # same actual for both
            "predicted": round((pf["predicted"] + qf["predicted"]) / 2, 2),
            "pred_low": round(min(pf["pred_low"], qf["pred_low"]), 2),
            "pred_high": round(max(pf["pred_high"], qf["pred_high"]), 2),
            "train_start": pf["train_start"],
            "train_end": pf["train_end"],
            "test_start": pf["test_start"],
            "test_end": pf["test_end"],
            "model": "Blended (Prophet + XGBoost)",
        })

    blended_metrics = compute_backtest_metrics(blended_folds)

    return {
        "prophet_metrics": prophet_metrics,
        "quantile_metrics": quantile_metrics,
        "blended_metrics": blended_metrics,
        "blended_folds": blended_folds,
    }


def run_backtest_all_channels(weekly_df: pd.DataFrame,
                              channels: list[str],
                              horizon_days: int = 60,
                              progress_callback=None) -> dict:
    """
    Run backtests across all channels and compute an overall confidence score.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly aggregated data.
    channels : list[str]
        Channel names to backtest.
    horizon_days : int
        Forecast horizon.
    progress_callback : callable, optional
        Called with (channel_index, total_channels, channel_name) for UI progress.

    Returns
    -------
    dict
        {channel: backtest_channel result, "overall": aggregated metrics}.
    """
    results = {}

    for i, channel in enumerate(channels):
        if progress_callback:
            progress_callback(i, len(channels), channel)

        ch_weekly = weekly_df[weekly_df["channel"] == channel]
        if len(ch_weekly) < 35:  # Need at least 26 train + ~9 test weeks
            results[channel] = {
                "prophet_metrics": _empty_metrics(),
                "quantile_metrics": _empty_metrics(),
                "blended_metrics": _empty_metrics(),
                "blended_folds": [],
            }
            continue

        results[channel] = backtest_channel(weekly_df, channel, horizon_days)

    # Overall confidence = weighted average by number of folds
    total_folds = 0
    weighted_score = 0.0
    all_blended_folds = []

    for ch, res in results.items():
        bm = res["blended_metrics"]
        n = bm["n_folds"]
        total_folds += n
        weighted_score += bm["confidence_score"] * n
        all_blended_folds.extend(res.get("blended_folds", []))

    if total_folds > 0:
        overall_score = round(weighted_score / total_folds, 1)
    else:
        overall_score = 0.0

    overall_metrics = compute_backtest_metrics(all_blended_folds)
    overall_metrics["confidence_score"] = overall_score
    overall_metrics["confidence_label"] = _score_to_label(overall_score)

    results["overall"] = overall_metrics

    if progress_callback:
        progress_callback(len(channels), len(channels), "Done")

    return results
