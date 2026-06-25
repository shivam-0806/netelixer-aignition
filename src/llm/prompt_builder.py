"""Prompt builder — constructs the analyst prompt from structured data."""


def _format_money(value) -> str:
    """Format a numeric value as whole-dollar text."""
    return f"${float(value):,.0f}"


def _format_validation_summary(validation_report: dict) -> str:
    """Build compact data-quality context for the LLM."""
    # Original:
    # - Zero-revenue campaigns with active spend: {validation_report.get('zero_revenue_campaigns', [])}
    # This dumped very large campaign lists into the prompt. Use summarized
    # validation fields instead so the model focuses on business implications.
    lines = [
        f"- Duplicate rows found: {validation_report.get('duplicate_rows', 0)}",
        f"- Zero-revenue campaigns with active spend: "
        f"{validation_report.get('zero_revenue_campaign_count', 0)} campaigns / "
        f"{validation_report.get('zero_revenue_rows', 0)} rows",
        f"- Zero-spend rows: {validation_report.get('zero_spend_rows', 0)}",
        f"- Extreme row-level ROAS outliers: {validation_report.get('extreme_roas_rows', 0)}",
        f"- Missing observed dates by channel: "
        f"{validation_report.get('missing_dates_by_channel', {})}",
        f"- Estimated revenue rows by channel: "
        f"{validation_report.get('estimated_revenue_rows_by_channel', {})}",
        f"- Revenue sources by channel: "
        f"{validation_report.get('revenue_sources_by_channel', {})}",
    ]

    top_zero_revenue = validation_report.get("zero_revenue_top_spend", [])[:5]
    if top_zero_revenue:
        lines.append("- Top zero-revenue spend concentrations:")
        for row in top_zero_revenue:
            lines.append(
                f"  - {row.get('channel')}: {row.get('campaign_name')} "
                f"spent {_format_money(row.get('spend', 0))} with zero revenue"
            )

    return "\n".join(lines)


def build_forecast_prompt(
    historical_summary: dict,
    forecast_results:   dict,
    validation_report:  dict,
    horizon_days:       int,
) -> str:
    """
    Construct the LLM analyst prompt from structured data.
    Passes only aggregated statistics — never raw row-level data.
    """
    hist = historical_summary
    fc   = forecast_results

    # Build budget allocation text
    budget_lines = []
    for channel, res in fc.items():
        if channel == "blended":
            continue
        budget_lines.append(
            f"  {channel}: ${res['planned_spend']:,.0f}"
        )
    budget_text = "\n".join(budget_lines)

    # Build channel results text
    channel_lines = []
    for channel, res in fc.items():
        if channel == "blended":
            continue
        channel_lines.append(
            f"  {channel}:\n"
            f"    Spend:   ${res['planned_spend']:,.0f}\n"
            f"    Revenue: ${res['revenue_low']:,.0f} — ${res['revenue_median']:,.0f} — ${res['revenue_high']:,.0f}\n"
            f"    ROAS:    {res['roas_low']:.2f}x — {res['roas_median']:.2f}x — {res['roas_high']:.2f}x"
        )
    channel_text = "\n".join(channel_lines)

    blended = fc["blended"]
    validation_text = _format_validation_summary(validation_report)

    return f"""You are a senior digital marketing analyst at a performance agency.
You have been given historical advertising data and a probabilistic forecast for the next {horizon_days} days.
Your job is to generate a concise causal business summary (3-5 paragraphs) that explains the forecast
to a non-technical marketing director.

--- HISTORICAL PERFORMANCE SUMMARY ---
Date Range: {hist['date_range']}
Channels Active: {', '.join(hist['channels'])}

Channel-Level Averages (last 90 days):
{hist['channel_avg_table']}

Known Data Quality Issues:
{validation_text}

Forecast Method Notes:
- The forecast is an ensemble uncertainty band from Prophet and XGBoost, not a guaranteed statistical confidence interval.
- Google uses observed ecommerce revenue and has the strongest data quality in this prototype.
- Google production ranges are calibrated using recent backtest behavior to avoid over-wide bands.
- Any channel listed under estimated revenue must be described as assumption-based, not observed ecommerce revenue.

--- FORECAST INPUTS ---
Planning Horizon: {horizon_days} days
Budget Allocation:
{budget_text}

--- FORECAST OUTPUTS ---
Blended Results:
  Total Spend:     ${blended['total_spend']:,.0f}
  Revenue (Low):   ${blended['revenue_low']:,.0f}
  Revenue (Med):   ${blended['revenue_median']:,.0f}
  Revenue (High):  ${blended['revenue_high']:,.0f}
  ROAS Range:      {blended['roas_low']:.2f}x — {blended['roas_high']:.2f}x

Channel-Level Results:
{channel_text}

--- YOUR TASK ---
1. Summarize the key drivers of the forecast (seasonality, channel efficiency, budget level).
2. Call out the most significant risk factor for each channel.
3. Note any data quality concerns that may affect forecast reliability.
4. Provide one concrete optimization recommendation for the budget allocation.
5. Flag which channel has the widest uncertainty and explain why.

Write in plain English. Be direct. Do not hedge every sentence. Use numbers from the data above.
Do not invent causes that are not supported by the supplied metrics.
When revenue is estimated, explicitly call that out as a limitation.
Format the output using markdown with clear headers and bullet points for readability."""
