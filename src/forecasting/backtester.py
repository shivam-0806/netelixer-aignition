import pandas as pd
from src.forecasting.prophet_model import train_prophet, forecast_prophet
from src.forecasting.quantile_model import train_quantile_models, simulate_budget
from src.forecasting.budget_simulator import combine_forecast_outputs
from src.forecasting import prepare_forecast_input
from src.utils.metrics import calculate_backtest_metrics

def run_channel_backtest(
    df: pd.DataFrame, 
    channel: str, 
    horizon_days: int = 30, 
    windows: int = 3
) -> dict:
    """
    Runs a sliding window backtest for a specific channel.
    """
    channel_df = df[df["channel"] == channel].sort_values("date").copy()
    max_date = channel_df["date"].max()
    
    results = []
    
    # Slide backwards in time to create our testing windows
    for i in range(windows):
        # Calculate the test window dates
        test_end = max_date - pd.Timedelta(days=i * horizon_days)
        # Original:
        # test_start = test_end - pd.Timedelta(days=horizon_days)
        # Corrected: use an exact inclusive horizon, e.g. 30 rows/days for 30.
        test_start = test_end - pd.Timedelta(days=horizon_days - 1)
        
        # Split the data: Train is everything BEFORE test_start (Daily grain)
        train_df = channel_df[channel_df["date"] < test_start].copy()
        test_df = channel_df[(channel_df["date"] >= test_start) & (channel_df["date"] <= test_end)]
        
        if test_df.empty or train_df.empty:
            continue
            
        # 1. What actually happened in this window?
        actual_revenue = float(test_df["revenue"].sum())
        actual_spend = float(test_df["spend"].sum())  # Use actual spend to simulate the budget.
        
        # Aggregate the daily training slice into weekly buckets.
        train_weekly = prepare_forecast_input(train_df, grain="channel")
        
        # 2. Train models on the weekly timeline
        prophet_model = train_prophet(train_weekly, channel)
        quantile_models = train_quantile_models(train_weekly, channel)
        
        # 3. Predict using the actual spend
        p_out = forecast_prophet(prophet_model, actual_spend, horizon_days)
        q_out = simulate_budget(quantile_models, actual_spend, horizon_days, train_weekly)
        
        # 4. Merge bounds using the same policy as production simulation.
        # Original local merge logic enforced a minimum +/- 15% spread here only.
        # It now lives in combine_forecast_outputs() to keep backtest and
        # production behavior consistent.
        combined = combine_forecast_outputs(p_out, q_out)
        pred_low = combined["revenue_low"]
        pred_med = combined["revenue_median"]
        pred_high = combined["revenue_high"]
        
        # 5. Store for the metric calculator
        results.append({
            "window_end": test_end.date(),
            "actual": actual_revenue,
            "p10": pred_low,
            "p50": pred_med,
            "p90": pred_high
        })
        
    # Return the final graded metrics
    return {
        "channel": channel,
        "metrics": calculate_backtest_metrics(results),
        "raw_runs": results
    } 
