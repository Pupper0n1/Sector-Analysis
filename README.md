# Sector Analysis: Interactive Dashboard for Analysis and Forecasting of Sector Performance

**DATA 501 Capstone** — Ashton Frese, Karen Reyes Gonzalez, Wilbur Elbouni

An interactive Dash web dashboard that analyzes five U.S. sector ETFs (Technology, Energy, Financials, Healthcare, Industrials) from 2005-2026, overlaying macroeconomic shocks and combining ARIMA, GARCH, and LSTM models for forecasting.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Pupper0n1/Sector-Analysis.git
cd Sector-Analysis

# 2. Create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
python app.py
```

Open **http://127.0.0.1:8050** in your browser.

## Project Structure

```
Sector-Analysis/
├── app.py                          # Main Dash dashboard application
├── requirements.txt                # Python dependencies (dashboard only)
├── src/
│   └── preprocessing.py            # Raw data -> processed data pipeline
├── data/
│   ├── processed/                  # Pre-built datasets (committed to repo)
│   │   ├── master_sector_data.csv  # All 5 ETFs: close, volume, log returns, FEDFUNDS
│   │   └── model_predictions.csv   # ARIMA, LSTM, and GARCH predictions
│   └── raw/                        # Raw source files (not committed, ~1.6 GB)
├── notebooks/
│   ├── 1_data_acquisition.ipynb    # Data collection and initial processing
│   ├── 2_exploratory_analysis.ipynb# EDA and visualizations
│   └── 3_modeling_forecasting.ipynb# ARIMA, GARCH, LSTM model training
├── papers/                         # LaTeX reports and PDF outputs
├── events.csv                      # Macroeconomic event definitions
├── FEDFUNDS.csv                    # Federal Funds Rate (monthly, 1954-2026)
└── requirements-notebooks.txt      # Extra dependencies for running notebooks
```

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Price Overlay** | Normalized ETF prices (base=100) with macroeconomic event markers |
| **30-day Rolling Volatility** | Annualized rolling volatility with event overlays |
| **Event Drill-Down** | 22-day window analysis of cumulative returns around major shocks |
| **Model Performance** | Backtest: ARIMA/LSTM returns and GARCH volatility vs actuals (2024-2026) |
| **Rolling Correlation with FEDFUNDS** | Pearson correlation between sector returns and the Fed Funds Rate |
| **Sector Relationships** | Correlation heatmap with regime filtering (All / Stable / Shocks) |
| **Future Outlook** | Forward price projections using ARIMA mean + GARCH 95% confidence intervals |

## Data

All processed data needed to run the dashboard is included in `data/processed/`. No additional downloads are required.

### Data Sources

| Source | URL | Usage |
|--------|-----|-------|
| Stooq | https://stooq.com/db/h/ | Historical OHLCV for XLK, XLE, XLF, XLV, XLI |
| FRED | https://fred.stlouisfed.org/series/FEDFUNDS | Federal Funds Effective Rate |

### Reproducing from Raw Data (Optional)

If you want to regenerate the processed datasets from scratch:

1. Download the full Stooq US daily dataset from https://stooq.com/db/h/ and extract it into `data/raw/`
2. Place `FEDFUNDS.csv` (from FRED) and `events.csv` into `data/raw/`
3. Run the preprocessing pipeline:
   ```bash
   python src/preprocessing.py
   ```

## Notebooks

The Jupyter notebooks document the full research pipeline. To run them, install the extended dependencies:

```bash
pip install -r requirements-notebooks.txt
```

| Notebook | Phase |
|----------|-------|
| `1_data_acquisition.ipynb` | Data collection, cleaning, integration |
| `2_exploratory_analysis.ipynb` | Trend analysis, volatility patterns, correlations |
| `3_modeling_forecasting.ipynb` | ARIMA, GARCH, LSTM model training and evaluation |

## Tech Stack

- **Dashboard**: Dash, Plotly, Dash Mantine Components
- **Modeling**: statsmodels (ARIMA), arch (GARCH), TensorFlow/Keras (LSTM)
- **Data**: pandas, NumPy, SciPy
