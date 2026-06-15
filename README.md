# AIgnition 2026 — Probabilistic Revenue Forecasting

AIgnition 2026 is a premium multi-channel budget simulation tool for performance marketing agencies. It generates probabilistic revenue forecasts using **Prophet** (time-series) and **XGBoost** (quantile regression), combined with an **AI-powered causal analysis** engine via Google Gemini.

## 🚀 Features

- **Multi-Channel Ingestion**: Automatically loads, cleans, and harmonizes Google Ads, Bing/MS Ads, and Meta Ads CSV exports. Includes heuristic fallbacks (e.g. 2.0x ROAS assumption for missing Meta revenue).
- **Dual-Model Forecasting**: Uses Meta Prophet to capture seasonality and XGBoost to establish P10, P50, and P90 confidence intervals based on budget sensitivity.
- **Probabilistic Outputs**: Replaces single-point estimates with realistic revenue ranges and blended ROAS projections to manage client expectations.
- **Budget Simulator UI**: Interactive "what-if" budget allocation across channels via a sleek, modern Streamlit dashboard.
- **AI Insights Analyst**: Integrates with `gemini-3.5-flash` to write plain-English, causal summaries of the forecast and flag critical data quality risks.
- **Data Quality Gates**: Surfaces zero-revenue campaigns, duplicate rows, and missing dates before the forecast is run.

## 🏗️ Architecture

```text
[Raw CSVs] → [Ingestion & Harmonization] → [Forecasting Engine] → [Gemini LLM] → [Streamlit UI]
```

| Layer | Responsibility | Key Libraries |
|-------|----------------|---------------|
| **Ingestion** | Data normalization, ID casting, currency conversion | `pandas` |
| **Forecasting**| Weekly aggregation, regressor mapping, quantile bands | `prophet`, `xgboost`, `scikit-learn` |
| **LLM Engine** | Prompt construction, exponential backoff, rate limit handling| `google-genai` |
| **Frontend** | Modern dashboard, Plotly fan charts, session state | `streamlit`, `plotly` |

## 💻 Installation & Setup

1. **Clone & Virtual Environment**
   ```bash
   git clone <repo-url> aignition-2026
   cd aignition-2026
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 📊 Running the Application

Start the Streamlit server:
```bash
streamlit run app/streamlit_app.py
```
The dashboard will be available at `http://localhost:8501`. 

You can either upload your own CSV files via the sidebar or check the "Use built-in sample data" box to simulate a forecast using the files located in `data/raw/`.

## 📁 Project Structure

```
aignition-2026/
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── data/
│   ├── raw/                      # Example input CSVs (Google, Bing, Meta)
│   └── processed/                # Harmonized outputs (auto-generated)
├── src/
│   ├── ingestion/                # CSV loaders and harmonizer logic
│   ├── forecasting/              # Feature engineering, Prophet & XGBoost models
│   ├── llm/                      # Prompt builder & Gemini API summarizer
│   └── utils/                    # Data quality validators and metric math
├── requirements.txt              # Python dependencies
└── .env                          # API configurations
```

## ⚠️ Notes on the Free Tier API
The AI Insights tab utilizes the Google GenAI SDK. If you are using a free-tier API key, you may occasionally encounter rate limits (`429 Resource Exhausted`). The application includes built-in exponential backoff to handle this gracefully, but in cases of complete quota exhaustion, it will display a standard fallback message while preserving all statistical forecast charts.
