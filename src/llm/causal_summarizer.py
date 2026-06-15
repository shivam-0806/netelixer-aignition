"""Causal Summarizer — calls Gemini API for narrative insights."""

import os
import time
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Models to try in order of preference
_MODELS = ["gemini-3.5-flash", "gemini-2.0-flash"]
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2  # seconds


def _get_client():
    """Create and return a Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    client = genai.Client(api_key=api_key)
    return client


def generate_causal_summary(prompt: str) -> str:
    """
    Call Gemini and return the analyst narrative.
    Includes retry logic with exponential backoff for rate limiting.

    Parameters
    ----------
    prompt : str
        The fully constructed analyst prompt.

    Returns
    -------
    str
        Markdown-formatted causal summary.
    """
    client = _get_client()
    last_error = None

    for model in _MODELS:
        for attempt in range(_MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    # Non-rate-limit error — don't retry, try next model
                    break

    # All retries exhausted
    return (
        "⚠️ **AI analysis temporarily unavailable.**\n\n"
        f"The Gemini API returned an error after multiple retries: `{last_error}`\n\n"
        "This is typically caused by API quota limits. The forecast data above "
        "remains fully valid — the AI narrative is supplementary commentary only.\n\n"
        "**To resolve:** Check your Gemini API quota at "
        "[ai.google.dev/rate-limit](https://ai.google.dev/rate-limit) "
        "or try again in a few minutes."
    )


def build_historical_summary(df: pd.DataFrame) -> dict:
    """
    Compute the summary statistics passed into the LLM prompt.

    Parameters
    ----------
    df : pd.DataFrame
        Unified harmonized DataFrame.

    Returns
    -------
    dict
        Keys: date_range, channels, channel_avg_table
    """
    last_90 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]

    channel_avg = (
        last_90.groupby("channel")
        .agg(
            avg_daily_spend=("spend", "mean"),
            avg_daily_revenue=("revenue", "mean"),
            avg_roas=("roas", "mean"),
            avg_cpc=("cpc", "mean"),
        )
        .round(2)
    )

    return {
        "date_range":        f"{df['date'].min().date()} to {df['date'].max().date()}",
        "channels":          df["channel"].unique().tolist(),
        "channel_avg_table": channel_avg.to_string(),
    }
