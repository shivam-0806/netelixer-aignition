"""
AIgnition 2026 — Probabilistic Revenue Forecasting Utility
==========================================================
Streamlit application: multi-channel budget simulation with
probabilistic fan charts, ROAS gauges, and AI-powered insights.
"""

import sys
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from src.ingestion import load_google, load_bing, load_meta, harmonize
from src.forecasting import (
    # Original:
    # prepare_forecast_input, train_prophet, forecast_prophet,
    # train_quantile_models, simulate_budget, run_full_simulation,
    prepare_forecast_input, train_prophet,
    train_quantile_models, run_full_simulation,
)
from src.llm import build_forecast_prompt, generate_causal_summary, build_historical_summary
from src.utils import validate_campaigns, compute_channel_metrics

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIgnition 2026 — Revenue Forecaster",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS for premium look
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Font and Body */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Main background */
    .stApp {
        background-color: #0B0E14;
        background-image: radial-gradient(circle at top left, #161B22 0%, #0B0E14 60%);
        color: #C9D1D9;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0D1117;
        border-right: 1px solid #30363D;
    }

    /* Card-like metric containers */
    div[data-testid="stMetric"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    div[data-testid="stMetric"] label {
        color: #8B949E !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.75rem;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #E6EDF3 !important;
        font-weight: 800;
        font-size: 2rem;
    }

    /* Headers */
    h1, h2, h3 {
        color: #E6EDF3 !important;
        font-weight: 600;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #8B949E !important;
        font-weight: 600;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #58A6FF !important;
        border-bottom-color: #58A6FF !important;
    }

    /* Success/warning/error boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
        background-color: #161B22;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #30363D;
    }

    /* Hide the default hamburger menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Upload area */
    div[data-testid="stFileUploader"] {
        background: #161B22;
        border: 1px dashed #30363D;
        border-radius: 8px;
        padding: 16px;
    }

    /* Inputs */
    div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] > div {
        background: #0D1117 !important;
        color: #E6EDF3 !important;
        border: 1px solid #30363D !important;
        border-radius: 6px !important;
    }

    /* Button styling */
    button[data-testid="stBaseButton-primary"] {
        background: #238636 !important;
        color: #ffffff !important;
        border: 1px solid rgba(240, 246, 252, 0.1) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background: #2EA043 !important;
        border-color: rgba(240, 246, 252, 0.1) !important;
    }

    /* Divider */
    hr {
        border-color: #30363D !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Plotly chart theme (dark to match app)
# ─────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=40, r=40, t=60, b=40),
)

CHANNEL_COLORS = {
    "Google": "#4285F4",
    "Bing":   "#00A4EF",
    "Meta":   "#E1306C",
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return numeric ratio with zero/invalid denominators converted to 0.0."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, float("nan"))
    return (numerator / denominator).fillna(0.0).astype(float)


def build_unified_dataset(channel_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine loaded channel frames and derive shared metrics."""
    frames = [df for df in channel_frames.values() if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()

    # Original:
    # unified_df = harmonize(google_df, bing_df, meta_df)
    #
    # Corrected: Google-first mode may intentionally load only Google. Use the
    # existing harmonizer when all channels are present, otherwise concatenate
    # selected channel frames and derive the same shared metric columns.
    if all(channel in channel_frames and channel_frames[channel] is not None
           for channel in ("Google", "Bing", "Meta")):
        return harmonize(
            channel_frames["Google"],
            channel_frames["Bing"],
            channel_frames["Meta"],
        )

    unified = pd.concat(frames, ignore_index=True)
    unified["roas"] = _safe_divide(unified["revenue"], unified["spend"])
    unified["cpc"] = _safe_divide(unified["spend"], unified["clicks"])
    unified["cvr"] = _safe_divide(unified["conversions"], unified["clicks"])
    return unified.sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Helper: Fan chart
# ─────────────────────────────────────────────────────────────
def plot_revenue_fan(forecast_results: dict, horizon_days: int):
    """Create a probabilistic revenue range chart per channel."""
    channels = [c for c in forecast_results if c != "blended"]

    fig = go.Figure()

    for ch in channels:
        r = forecast_results[ch]
        color = CHANNEL_COLORS.get(ch, "#888888")

        # Uncertainty band (low to high)
        fig.add_trace(go.Bar(
            name=f"{ch} Range",
            x=[ch],
            y=[r["revenue_high"] - r["revenue_low"]],
            base=[r["revenue_low"]],
            marker_color=color,
            opacity=0.3,
            showlegend=True,
        ))

        # Median marker
        fig.add_trace(go.Scatter(
            name=f"{ch} Median",
            x=[ch],
            y=[r["revenue_median"]],
            mode="markers+text",
            text=[f"${r['revenue_median']:,.0f}"],
            textposition="top center",
            textfont=dict(size=13, color=color),
            marker=dict(size=14, color=color, symbol="diamond",
                        line=dict(width=2, color="white")),
        ))

    fig.update_layout(
        title=dict(
            text=f"Forecast Revenue Range — Next {horizon_days} Days",
            font=dict(size=18),
        ),
        yaxis_title="Revenue ($)",
        xaxis_title="Channel",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"),
        **CHART_LAYOUT,
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")

    return fig


# ─────────────────────────────────────────────────────────────
# Helper: ROAS gauge
# ─────────────────────────────────────────────────────────────
def plot_roas_gauge(blended: dict):
    """Create a ROAS gauge indicator."""
    max_val = max(blended["roas_high"] * 1.3, 5.0)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=blended["roas_median"],
        number=dict(suffix="x", font=dict(size=40, color="#e2e8f0")),
        delta=dict(reference=blended["roas_low"], suffix="x",
                   increasing=dict(color="#48bb78"),
                   decreasing=dict(color="#fc8181")),
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor="#a0aec0",
                      tickfont=dict(color="#a0aec0")),
            bar=dict(color="#48bb78", thickness=0.3),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 1.0], color="rgba(252,129,129,0.2)"),
                dict(range=[1.0, blended["roas_low"]], color="rgba(246,173,85,0.2)"),
                dict(range=[blended["roas_low"], blended["roas_high"]],
                     color="rgba(72,187,120,0.15)"),
                dict(range=[blended["roas_high"], max_val],
                     color="rgba(72,187,120,0.05)"),
            ],
            threshold=dict(
                line=dict(color="#e2e8f0", width=3),
                value=blended["roas_median"],
            ),
        ),
        title=dict(text="Blended ROAS", font=dict(size=16, color="#a0aec0")),
    ))

    fig.update_layout(
        height=280,
        **CHART_LAYOUT,
    )

    return fig


# ─────────────────────────────────────────────────────────────
# Helper: Channel ROAS comparison bar chart
# ─────────────────────────────────────────────────────────────
def plot_channel_roas_comparison(forecast_results: dict):
    """Create a grouped bar chart comparing ROAS ranges across channels."""
    channels = [c for c in forecast_results if c != "blended"]
    fig = go.Figure()

    for ch in channels:
        r = forecast_results[ch]
        color = CHANNEL_COLORS.get(ch, "#888")

        fig.add_trace(go.Bar(
            name=ch,
            x=["Low (P10)", "Median (P50)", "High (P90)"],
            y=[r["roas_low"], r["roas_median"], r["roas_high"]],
            marker_color=color,
            opacity=0.85,
            text=[f"{r['roas_low']:.2f}x", f"{r['roas_median']:.2f}x", f"{r['roas_high']:.2f}x"],
            textposition="auto",
            textfont=dict(size=12, color="white"),
        ))

    # Break-even reference line
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(252,129,129,0.6)",
                  annotation_text="Break-even (1.0x)",
                  annotation_font_color="#fc8181")

    fig.update_layout(
        title=dict(text="ROAS by Channel — Probabilistic Range", font=dict(size=18)),
        yaxis_title="ROAS (Revenue / Spend)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"),
        **CHART_LAYOUT,
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")

    return fig


# ─────────────────────────────────────────────────────────────
# Helper: Budget allocation donut
# ─────────────────────────────────────────────────────────────
def plot_budget_donut(budget_inputs: dict):
    """Create a donut chart of budget allocation."""
    labels = list(budget_inputs.keys())
    values = list(budget_inputs.values())
    colors = [CHANNEL_COLORS.get(c, "#888") for c in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#1a1a2e", width=2)),
        textinfo="label+percent",
        textfont=dict(size=13, color="#e2e8f0"),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Budget Allocation", font=dict(size=16)),
        showlegend=False,
        height=300,
        **CHART_LAYOUT,
    )

    return fig


# ─────────────────────────────────────────────────────────────
# Helper: Campaign breakdown
# ─────────────────────────────────────────────────────────────
def show_campaign_breakdown(unified_df: pd.DataFrame, forecast_results: dict):
    """Display campaign-level revenue contribution table."""
    last_period = unified_df[unified_df["date"] >= unified_df["date"].max() - pd.Timedelta(days=90)]

    camp_share = (
        last_period.groupby(["channel", "campaign_name", "campaign_type"])
        .agg(
            historical_revenue=("revenue", "sum"),
            historical_spend=("spend", "sum"),
            estimated_revenue_rows=("revenue_is_estimated", "sum"),
            revenue_source=("revenue_source", lambda values: ", ".join(sorted(map(str, values.dropna().unique())))),
        )
        .reset_index()
    )

    for channel in forecast_results:
        if channel == "blended":
            continue
        ch_mask = camp_share["channel"] == channel
        ch_total_rev = camp_share.loc[ch_mask, "historical_revenue"].sum()
        ch_forecast = forecast_results[channel]["revenue_median"]

        # Original:
        # if ch_total_rev > 0:
        #     camp_share.loc[ch_mask, "forecast_revenue"] = (
        #         (camp_share.loc[ch_mask, "historical_revenue"] / ch_total_rev) * ch_forecast
        #     )
        # else:
        #     camp_share.loc[ch_mask, "forecast_revenue"] = 0
        #
        # Corrected: use revenue-share where possible, then spend-share as a
        # fallback so zero-revenue campaigns/channels still receive a visible
        # allocation basis.
        if ch_total_rev > 0:
            camp_share.loc[ch_mask, "forecast_revenue"] = (
                (camp_share.loc[ch_mask, "historical_revenue"] / ch_total_rev) * ch_forecast
            )
            camp_share.loc[ch_mask, "allocation_basis"] = "Revenue share"
        else:
            ch_total_spend = camp_share.loc[ch_mask, "historical_spend"].sum()
            if ch_total_spend > 0:
                camp_share.loc[ch_mask, "forecast_revenue"] = (
                    (camp_share.loc[ch_mask, "historical_spend"] / ch_total_spend) * ch_forecast
                )
                camp_share.loc[ch_mask, "allocation_basis"] = "Spend share"
            else:
                camp_share.loc[ch_mask, "forecast_revenue"] = 0
                camp_share.loc[ch_mask, "allocation_basis"] = "No spend/revenue"

    camp_share["revenue_is_estimated"] = camp_share["estimated_revenue_rows"] > 0

    display_df = (
        camp_share[["channel", "campaign_name", "campaign_type",
                     "historical_spend", "historical_revenue", "forecast_revenue",
                     "allocation_basis", "revenue_is_estimated", "revenue_source"]]
        .sort_values("forecast_revenue", ascending=False)
        .reset_index(drop=True)
    )

    display_df.columns = ["Channel", "Campaign", "Type",
                          "Hist. Spend ($)", "Hist. Revenue ($)", "Forecast Revenue ($)",
                          "Allocation Basis", "Revenue Estimated", "Revenue Source"]

    st.dataframe(
        display_df.style.format({
            "Hist. Spend ($)": "${:,.0f}",
            "Hist. Revenue ($)": "${:,.0f}",
            "Forecast Revenue ($)": "${:,.0f}",
        }),
        use_container_width=True,
        height=400,
    )


# ─────────────────────────────────────────────────────────────
# Helper: Validation report
# ─────────────────────────────────────────────────────────────
def show_validation_report(report: dict, unified_df: pd.DataFrame):
    """Display data quality validation results."""

    col1, col2, col3 = st.columns(3)

    with col1:
        total_records = sum(report.get("record_counts", {}).values())
        st.metric("Total Records", f"{total_records:,}")

    with col2:
        st.metric("Duplicate Rows", report.get("duplicate_rows", 0))

    with col3:
        # Original:
        # zero_rev = len(report.get("zero_revenue_campaigns", []))
        zero_rev = report.get("zero_revenue_campaign_count", len(report.get("zero_revenue_campaigns", [])))
        st.metric("Zero-Revenue Campaigns", zero_rev)

    if report.get("estimated_revenue_rows", 0) > 0:
        st.warning(
            f"{report['estimated_revenue_rows']:,} rows use estimated revenue. "
            "Treat ROAS/revenue for those channels as assumption-based."
        )

    if report.get("missing_dates_by_channel"):
        missing_dates = {
            channel: count for channel, count in report["missing_dates_by_channel"].items()
            if count > 0
        }
        if missing_dates:
            st.warning(f"Missing observed dates by channel: {missing_dates}")

    # Date range coverage
    if "date_range" in report:
        st.markdown("#### Date Range by Channel")
        date_data = []
        for ch, dr in report["date_range"].items():
            date_data.append({
                "Channel": ch,
                "Start Date": str(dr["min"].date()) if hasattr(dr["min"], "date") else str(dr["min"]),
                "End Date": str(dr["max"].date()) if hasattr(dr["max"], "date") else str(dr["max"]),
            })
        st.dataframe(pd.DataFrame(date_data), use_container_width=True, hide_index=True)

    # Record counts
    if "record_counts" in report:
        st.markdown("#### Records by Channel")
        cols = st.columns(len(report["record_counts"]))
        for i, (ch, cnt) in enumerate(report["record_counts"].items()):
            with cols[i]:
                color = CHANNEL_COLORS.get(ch, "#888")
                st.markdown(
                    f'<div style="text-align:center;padding:12px;background:rgba(255,255,255,0.03);'
                    f'border-radius:10px;border-left:3px solid {color};">'
                    f'<div style="color:#a0aec0;font-size:13px">{ch}</div>'
                    f'<div style="color:#e2e8f0;font-size:24px;font-weight:700">{cnt:,}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    if report.get("revenue_sources_by_channel"):
        st.markdown("#### Revenue Source by Channel")
        source_rows = [
            {"Channel": channel, "Revenue Source": ", ".join(sources)}
            for channel, sources in report["revenue_sources_by_channel"].items()
        ]
        st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)

    # Warnings
    if report.get("zero_revenue_campaigns"):
        with st.expander(f"{zero_rev} campaigns with active spend but zero revenue", expanded=False):
            # Original:
            # for camp in report["zero_revenue_campaigns"][:20]:
            #     st.markdown(f"- `{camp}`")
            # if zero_rev > 20:
            #     st.caption(f"... and {zero_rev - 20} more")
            top_spend = report.get("zero_revenue_top_spend", [])
            if top_spend:
                st.caption("Top zero-revenue campaigns by spend")
                st.dataframe(pd.DataFrame(top_spend), use_container_width=True, hide_index=True)
            else:
                for camp in report["zero_revenue_campaigns"][:20]:
                    st.markdown(f"- `{camp}`")
                if zero_rev > 20:
                    st.caption(f"... and {zero_rev - 20} more")

    if report.get("zero_spend_rows", 0) > 0:
        st.info(
            f"Found {report['zero_spend_rows']:,} zero-spend rows. "
            f"By channel: {report.get('zero_spend_rows_by_channel', {})}"
        )

    if report.get("extreme_roas_rows", 0) > 0:
        with st.expander(f"{report['extreme_roas_rows']:,} extreme row-level ROAS outliers", expanded=False):
            st.dataframe(
                pd.DataFrame(report.get("extreme_roas_top_rows", [])),
                use_container_width=True,
                hide_index=True,
            )

    if report.get("negative_spend_rows", 0) > 0:
        st.error(f"Found {report['negative_spend_rows']} rows with negative spend values.")

    if report.get("negative_revenue_rows", 0) > 0:
        st.warning(f"Found {report['negative_revenue_rows']} rows with negative revenue values.")


# ─────────────────────────────────────────────────────────────
# Helper: AI insights panel
# ─────────────────────────────────────────────────────────────
def show_ai_insights(causal_summary: str, forecast_results: dict):
    """Display AI analyst insights with risk flags."""
    col1, col2 = st.columns([2, 1])

    with col1:
        # Escape dollar signs to prevent Streamlit from rendering them as LaTeX
        safe_summary = causal_summary.replace('$', r'\$')
        st.markdown(safe_summary)

    with col2:
        st.markdown("### Risk Flags")
        blended = forecast_results["blended"]

        uncertainty_pct = (
            (blended["revenue_high"] - blended["revenue_low"])
            / blended["revenue_median"] * 100
        ) if blended["revenue_median"] > 0 else 0

        if uncertainty_pct > 40:
            st.error(
                f"**HIGH UNCERTAINTY** — Forecast range spans ±{uncertainty_pct:.0f}% of median. "
                "Consider reducing budget until more data is available."
            )
        elif uncertainty_pct > 20:
            st.warning(
                f"**MODERATE UNCERTAINTY** — ±{uncertainty_pct:.0f}% spread. "
                "Review channel-level assumptions."
            )
        else:
            st.success(
                f"**LOW UNCERTAINTY** — ±{uncertainty_pct:.0f}% spread. Forecast is reliable."
            )

        st.markdown("---")
        st.markdown("### Channel Alerts")

        for channel, res in forecast_results.items():
            if channel == "blended":
                continue
            color = CHANNEL_COLORS.get(channel, "#888")
            if res["roas_low"] < 1.0:
                st.error(
                    f"**{channel}** — ROAS low-end ({res['roas_low']:.2f}x) "
                    "is below break-even (1.0x)"
                )
            elif res["roas_low"] < 2.0:
                st.warning(
                    f"**{channel}** — ROAS low-end ({res['roas_low']:.2f}x) "
                    "is below typical target (2.0x)"
                )
            else:
                st.success(
                    f"**{channel}** — ROAS low-end ({res['roas_low']:.2f}x) "
                    "is healthy"
                )


# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

def main():
    # ─── Header ───
    st.markdown(
        '<h1 style="text-align:center; background: linear-gradient(135deg, #667eea, #764ba2); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent; '
        'font-size: 2.5rem; font-weight: 800; margin-bottom: 0;">AIgnition 2026</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center; color:#a0aec0; font-size:1.1rem; margin-top:0;">'
        'Probabilistic Revenue Forecasting & Budget Simulation</p>',
        unsafe_allow_html=True,
    )

    # ─── Sidebar ───
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:10px 0 5px;">'
            '<span style="font-size:1.8rem"></span>'
            '<h2 style="margin:0;background:linear-gradient(135deg,#667eea,#764ba2);'
            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
            'AIgnition</h2></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown("#### Forecast Scope")
        forecast_scope = st.radio(
            "Modeling mode",
            options=["Google only (recommended)", "All channels"],
            index=0,
            key="forecast_scope",
        )
        active_channels = ["Google"] if forecast_scope.startswith("Google") else [
            "Google", "Bing", "Meta"
        ]
        if forecast_scope.startswith("Google"):
            st.caption("Recommended path: observed revenue, complete daily coverage, strongest signal.")
        else:
            st.caption("Includes Bing and Meta; Meta revenue is assumption-based unless a revenue field is supplied.")

        st.markdown("---")

        st.markdown("#### 📁 Upload Data")
        google_file = st.file_uploader("Google Ads CSV", type=["csv"], key="google_csv")
        bing_file   = st.file_uploader("Bing / MS Ads CSV", type=["csv"], key="bing_csv")
        meta_file   = st.file_uploader("Meta Ads CSV", type=["csv"], key="meta_csv")

        st.markdown("---")
        st.markdown("#### Budget Inputs")
        google_budget = st.number_input("Google Ads Budget ($)", min_value=0,
                                        value=50000, step=1000, key="google_budget")
        # Original:
        # bing_budget = st.number_input("MS Ads Budget ($)", min_value=0,
        #                               value=10000, step=1000, key="bing_budget")
        # meta_budget = st.number_input("Meta Ads Budget ($)", min_value=0,
        #                               value=30000, step=1000, key="meta_budget")
        if "Bing" in active_channels:
            bing_budget = st.number_input("MS Ads Budget ($)", min_value=0,
                                          value=10000, step=1000, key="bing_budget")
        else:
            bing_budget = 0
        if "Meta" in active_channels:
            meta_budget = st.number_input("Meta Ads Budget ($)", min_value=0,
                                          value=30000, step=1000, key="meta_budget")
        else:
            meta_budget = 0

        st.markdown("---")
        st.markdown("#### Forecast Window")
        horizon = st.selectbox("Planning Horizon (days)", options=[30, 60, 90],
                               index=1, key="horizon")

        st.markdown("---")
        run_button = st.button("Generate Forecast", type="primary",
                               use_container_width=True, key="run_forecast")

        # Auto-load from data/raw if files not uploaded
        use_default = st.checkbox("Use built-in sample data", value=True, key="use_default")

    # ─── Data Loading ───
    data_loaded = False
    forecast_results = None
    budget_inputs = None

    # Original:
    # if run_button or st.session_state.get("forecast_ready"):
    #
    # Corrected: only regenerate models when the button is clicked. Otherwise
    # reuse the previous run from session_state so normal UI interactions do not
    # repeatedly retrain Prophet/XGBoost.
    if run_button:
        with st.spinner("Loading and harmonizing data..."):
            try:
                base_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
                channel_frames = {}

                if google_file:
                    channel_frames["Google"] = load_google(google_file)
                elif use_default and os.path.exists(os.path.join(base_path, "google_ads_campaign_stats.csv")):
                    channel_frames["Google"] = load_google(os.path.join(base_path, "google_ads_campaign_stats.csv"))
                else:
                    st.error("Please upload Google Ads CSV or enable sample data.")
                    return

                # Original:
                # Bing and Meta were always required. In Google-first mode they
                # are intentionally optional, so the app can run the strongest
                # observed-revenue path without noisy/estimated channels.
                if "Bing" in active_channels:
                    if bing_file:
                        channel_frames["Bing"] = load_bing(bing_file)
                    elif use_default and os.path.exists(os.path.join(base_path, "bing_campaign_stats.csv")):
                        channel_frames["Bing"] = load_bing(os.path.join(base_path, "bing_campaign_stats.csv"))
                    else:
                        st.error("Please upload Bing Ads CSV or enable sample data.")
                        return

                if "Meta" in active_channels:
                    if meta_file:
                        channel_frames["Meta"] = load_meta(meta_file)
                    elif use_default and os.path.exists(os.path.join(base_path, "meta_ads_campaign_stats.csv")):
                        channel_frames["Meta"] = load_meta(os.path.join(base_path, "meta_ads_campaign_stats.csv"))
                    else:
                        st.error("Please upload Meta Ads CSV or enable sample data.")
                        return

                unified_df = build_unified_dataset(channel_frames)
                data_loaded = True

            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.exception(e)
                return
    elif st.session_state.get("forecast_ready"):
        unified_df = st.session_state["unified_df"]
        forecast_results = st.session_state["forecast_results"]
        budget_inputs = st.session_state["budget_inputs"]
        horizon = st.session_state.get("forecast_horizon", horizon)
        data_loaded = True

    if not data_loaded:
        # Landing state
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div style="text-align:center;padding:30px;background:rgba(255,255,255,0.03);'
                'border-radius:16px;border:1px solid rgba(255,255,255,0.06);">'
                '<div style="font-size:2.5rem"></div>'
                '<h3 style="color:#e2e8f0">Multi-Channel</h3>'
                '<p style="color:#a0aec0;font-size:0.9rem">Google, Bing, Meta — unified into a '
                'single forecasting pipeline</p></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div style="text-align:center;padding:30px;background:rgba(255,255,255,0.03);'
                'border-radius:16px;border:1px solid rgba(255,255,255,0.06);">'
                '<div style="font-size:2.5rem"></div>'
                '<h3 style="color:#e2e8f0">Probabilistic</h3>'
                '<p style="color:#a0aec0;font-size:0.9rem">Revenue ranges, not point estimates — '
                'powered by Prophet + XGBoost</p></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                '<div style="text-align:center;padding:30px;background:rgba(255,255,255,0.03);'
                'border-radius:16px;border:1px solid rgba(255,255,255,0.06);">'
                '<div style="font-size:2.5rem"></div>'
                '<h3 style="color:#e2e8f0">AI Insights</h3>'
                '<p style="color:#a0aec0;font-size:0.9rem">Gemini-powered causal analysis — '
                'explains the "why" behind the numbers</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.info("**Upload your CSV files** (or use sample data) and click **Generate Forecast** to begin.")
        return

    # ─── Tabs ───
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Validation",
        "Forecast Outputs",
        "Campaign Breakdown",
        "AI Insights",
        "Accuracy Analysis",
    ])

    # ─── Tab 1: Validation ───
    with tab1:
        st.markdown("## Data Quality Report")
        validation_report = validate_campaigns(unified_df)
        show_validation_report(validation_report, unified_df)

        with st.expander("Channel Metrics (last 90 days)", expanded=True):
            channel_metrics = compute_channel_metrics(unified_df)
            st.dataframe(channel_metrics, use_container_width=True, hide_index=True)

        # Quick data overview
        with st.expander("Preview Unified Data (first 100 rows)", expanded=False):
            st.dataframe(unified_df.head(100), use_container_width=True, height=400)

    # ─── Forecasting ───
    if forecast_results is None:
        with st.spinner("🔄 Training models and generating forecasts..."):
            weekly_df = prepare_forecast_input(unified_df, grain="channel")
            channels_present = unified_df["channel"].unique().tolist()

            budget_inputs = {}
            if "Google" in channels_present:
                budget_inputs["Google"] = google_budget
            if "Bing" in channels_present:
                budget_inputs["Bing"] = bing_budget
            if "Meta" in channels_present:
                budget_inputs["Meta"] = meta_budget

            # Train models
            prophet_models = {}
            quantile_models = {}
            for ch in channels_present:
                ch_weekly = weekly_df[weekly_df["channel"] == ch]
                if len(ch_weekly) < 4:
                    st.warning(f"Insufficient data for {ch} (only {len(ch_weekly)} weeks). Skipping.")
                    continue
                prophet_models[ch] = train_prophet(weekly_df, ch)
                quantile_models[ch] = train_quantile_models(weekly_df, ch)

            # Run simulation
            forecast_results = run_full_simulation(
                weekly_df, prophet_models, quantile_models, budget_inputs, horizon
            )

            st.session_state["forecast_ready"] = True
            st.session_state["unified_df"] = unified_df
            st.session_state["forecast_results"] = forecast_results
            st.session_state["budget_inputs"] = budget_inputs
            st.session_state["forecast_horizon"] = horizon
    else:
        st.caption("Showing saved forecast results. Click Generate Forecast to refresh after changing inputs.")

    # ─── Tab 2: Forecast Outputs ───
    with tab2:
        st.markdown("## Probabilistic Forecast Results")

        # Top-level blended KPIs
        blended = forecast_results["blended"]
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Planned Spend", f"${blended['total_spend']:,.0f}")
        with kpi2:
            st.metric("Revenue (Median)", f"${blended['revenue_median']:,.0f}")
        with kpi3:
            st.metric("ROAS (Median)", f"{blended['roas_median']:.2f}x")
        with kpi4:
            spread = blended["revenue_high"] - blended["revenue_low"]
            st.metric("Uncertainty Range", f"${spread:,.0f}")

        if "revenue_is_estimated" in unified_df.columns and unified_df["revenue_is_estimated"].any():
            estimated_channels = sorted(
                unified_df.loc[unified_df["revenue_is_estimated"], "channel"].dropna().unique()
            )
            st.warning(
                "Forecast includes estimated revenue for: "
                f"{', '.join(estimated_channels)}. Interpret revenue and ROAS for those channels as assumption-based."
            )

        st.markdown("---")

        # Charts row
        chart_col1, chart_col2 = st.columns([3, 2])

        with chart_col1:
            st.plotly_chart(plot_revenue_fan(forecast_results, horizon),
                            use_container_width=True)

        with chart_col2:
            st.plotly_chart(plot_roas_gauge(blended), use_container_width=True)

        st.markdown("---")

        # ROAS comparison + budget donut
        roas_col, donut_col = st.columns([3, 2])

        with roas_col:
            st.plotly_chart(plot_channel_roas_comparison(forecast_results),
                            use_container_width=True)

        with donut_col:
            st.plotly_chart(plot_budget_donut(budget_inputs), use_container_width=True)

        # Channel-level detail cards
        st.markdown("---")
        st.markdown("### Channel-Level Forecast Detail")
        ch_cols = st.columns(len(budget_inputs))
        for i, (ch, res) in enumerate(
            [(k, v) for k, v in forecast_results.items() if k != "blended"]
        ):
            with ch_cols[i]:
                color = CHANNEL_COLORS.get(ch, "#888")
                st.markdown(
                    f'<div style="padding:20px;background:rgba(255,255,255,0.03);'
                    f'border-radius:14px;border-top:3px solid {color};">'
                    f'<h3 style="text-align:center;margin:0 0 12px 0">{ch}</h3>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:14px;">'
                    f'<div style="color:#a0aec0">Spend</div><div style="color:#e2e8f0;text-align:right">${res["planned_spend"]:,.0f}</div>'
                    f'<div style="color:#a0aec0">Rev (Low)</div><div style="color:#fc8181;text-align:right">${res["revenue_low"]:,.0f}</div>'
                    f'<div style="color:#a0aec0">Rev (Med)</div><div style="color:#48bb78;text-align:right">${res["revenue_median"]:,.0f}</div>'
                    f'<div style="color:#a0aec0">Rev (High)</div><div style="color:#63b3ed;text-align:right">${res["revenue_high"]:,.0f}</div>'
                    f'<div style="color:#a0aec0">ROAS</div><div style="color:#e2e8f0;text-align:right">{res["roas_low"]:.2f}x — {res["roas_high"]:.2f}x</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    # ─── Tab 3: Campaign Breakdown ───
    with tab3:
        st.markdown("## Campaign-Level Revenue Contribution")
        st.caption(
            # Original:
            # "Forecasted revenue is distributed proportionally across campaigns "
            # "based on their historical revenue share in the last 90 days."
            "Forecasted revenue is allocation-based: historical revenue share is used when available, "
            "with spend share as the fallback for zero-revenue groups."
        )
        show_campaign_breakdown(unified_df, forecast_results)

    # ─── Tab 4: AI Insights ───
    with tab4:
        st.markdown("## AI Analyst Insights")
        st.caption("Powered by Google Gemini — causal analysis of your forecast results")

        # Check for API key
        if not os.environ.get("GEMINI_API_KEY"):
            st.warning("Skipping AI Analysis: GEMINI_API_KEY not found in environment. Add it to `.env` file when ready.")
        else:
            with st.spinner("Generating AI analysis..."):
                try:
                    validation_report = validate_campaigns(unified_df)
                    hist_summary = build_historical_summary(unified_df)
                    prompt = build_forecast_prompt(
                        hist_summary, forecast_results, validation_report, horizon
                    )
                    causal_summary = generate_causal_summary(prompt)
                    show_ai_insights(causal_summary, forecast_results)
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
                    st.exception(e)

            with st.expander("View LLM Prompt (debug)", expanded=False):
                if 'prompt' in locals():
                    st.code(prompt, language="text")

    # ─── Tab 5: Accuracy Analysis ───
    with tab5:
        st.markdown("## Accuracy Analysis & Backtesting")
        st.caption("Run sliding-window backtests on historical data to evaluate model accuracy without leaking future data.")

        # Let the user pick the channel and parameters
        col1, col2 = st.columns(2)
        with col1:
            # Original:
            # test_channel = st.selectbox("Select Channel to Test", ["Google", "Bing", "Meta"])
            available_backtest_channels = unified_df["channel"].dropna().unique().tolist()
            available_backtest_channels = sorted(
                available_backtest_channels,
                key=lambda channel: 0 if channel == "Google" else 1,
            )
            test_channel = st.selectbox("Select Channel to Test", available_backtest_channels)
            if test_channel == "Meta":
                st.caption("Meta backtests use estimated revenue unless the input file contains observed revenue.")
        with col2:
            test_windows = st.number_input("Number of 30-day Windows to Test", min_value=1, max_value=6, value=3)

        if st.button("🚀 Run Accuracy Backtest", type="primary"):
            with st.spinner(f"Running {test_windows} sliding windows for {test_channel}..."):
                try:
                    from src.forecasting.backtester import run_channel_backtest
                    
                    # Run the engine
                    results = run_channel_backtest(unified_df, test_channel, horizon_days=30, windows=test_windows)
                    
                    # Save to session state so it doesn't vanish
                    st.session_state["backtest_results"] = results
                except Exception as e:
                    st.error(f"Backtest failed to execute: {e}")

        # Display results if they exist in session state
        if "backtest_results" in st.session_state:
            res = st.session_state["backtest_results"]
            metrics = res.get("metrics", {})

            st.markdown("---")
            st.markdown(f"### Backtest Results: {res.get('channel', 'Unknown')}")
            
            if not metrics:
                st.warning("Not enough historical data to complete the backtest windows.")
            else:
                m1, m2, m3 = st.columns(3)
                
                # Format AAPE as a percentage
                aape_pct = metrics['aape'] * 100
                with m1:
                    st.metric("AAPE (Error Rate)", f"{aape_pct:.1f}%", 
                              delta="Aim for < 15%", delta_color="off")
                
                # Format Coverage as a percentage
                cov_pct = metrics['coverage_prob'] * 100
                with m2:
                    st.metric("Interval Coverage", f"{cov_pct:.0f}%", 
                              delta="Aim for ~ 80%", delta_color="off")
                
                with m3:
                    st.metric("Windows Tested", metrics["windows_tested"])

                with st.expander("View Raw Window Data", expanded=False):
                    st.dataframe(pd.DataFrame(res["raw_runs"]), use_container_width=True)


if __name__ == "__main__":
    main()
