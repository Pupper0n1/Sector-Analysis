# Sector Analysis: Interactive Dashboard for Analysis and Forecasting of Sector Performance

**DATA 501 Capstone** — Ashton Frese, Karen Reyes Gonzalez, Wilbur Elbouni

## What Is This?

This project is an interactive web dashboard that lets you explore how different sectors of the U.S. stock market have performed over the past 20 years (2005-2026). It tracks five sector ETFs:

- **XLK** — Technology (Apple, Microsoft, Nvidia, etc.)
- **XLE** — Energy (ExxonMobil, Chevron, etc.)
- **XLF** — Financials (JPMorgan, Bank of America, etc.)
- **XLV** — Healthcare (UnitedHealth, Johnson & Johnson, etc.)
- **XLI** — Industrials (Caterpillar, Union Pacific, etc.)

The dashboard overlays major macroeconomic events (e.g. the 2008 financial crisis, COVID-19, the invasion of Ukraine) to show how each sector reacted to shocks. It also includes forecasting models — ARIMA for predicting returns, GARCH for forecasting volatility, and LSTM (a deep learning model) — to project future sector performance.

## How to Run the Dashboard

### Prerequisites

You need **Python 3.10 or newer** installed on your computer. To check, open a terminal and run:

```bash
python3 --version
```

If you don't have Python, download it from [python.org](https://www.python.org/downloads/).

### Step-by-Step Setup

Open a terminal (on Mac: search for "Terminal" in Spotlight; on Windows: search for "Command Prompt" or "PowerShell") and run these commands one at a time:

**1. Download the project**

```bash
git clone https://github.com/Pupper0n1/Sector-Analysis.git
cd Sector-Analysis
```

**2. Create an isolated Python environment**

This keeps the project's packages separate from the rest of your system so nothing conflicts:

```bash
python3 -m venv venv
```

**3. Activate the environment**

On **macOS / Linux**:
```bash
source venv/bin/activate
```

On **Windows**:
```bash
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt. This means the environment is active.

**4. Install the required packages**

```bash
pip install -r requirements.txt
```

This will download and install all the Python libraries the dashboard needs. It may take a minute or two.

**5. Start the dashboard**

```bash
python app.py
```

You should see output like:
```
Loaded model_predictions.csv
Dash is running on http://0.0.0.0:8050/
```

**6. Open the dashboard**

Open your web browser and go to: **http://127.0.0.1:8050**

The dashboard will load with interactive charts you can explore. To stop the dashboard, go back to the terminal and press `Ctrl+C`.

## What You Can Explore

The dashboard has seven tabs, each showing a different aspect of sector performance:

| Tab | What It Shows |
|-----|---------------|
| **Price Overlay** | How each sector's price has moved over time, normalized so they all start at 100 for easy comparison. Dashed vertical lines mark major economic events. |
| **30-day Rolling Volatility** | How "risky" or unstable each sector has been over time. Higher values mean bigger price swings. |
| **Event Drill-Down** | Pick a specific event (e.g. COVID-19) and see how each sector performed in the 22 trading days before and after. |
| **Model Performance** | How well our forecasting models predicted actual returns and volatility during the 2024-2026 test period. |
| **Rolling Correlation with FEDFUNDS** | Whether sector returns move with or against the Federal Reserve's interest rate, measured over 3, 6, and 12-month windows. |
| **Sector Relationships** | A heatmap showing which sectors tend to move together. You can filter by "stable" periods vs "shock" periods to see how relationships change during crises. |
| **Future Outlook** | Forward-looking price projections using ARIMA (predicted trend) and GARCH (uncertainty bands). |

## Project Structure

```
Sector-Analysis/
├── app.py                          # The dashboard application (run this)
├── requirements.txt                # List of Python packages needed
├── src/
│   └── preprocessing.py            # Script that builds the datasets from raw data
├── data/
│   ├── processed/                  # Ready-to-use datasets (included in the repo)
│   │   ├── master_sector_data.csv  # Daily prices, volumes, and returns for all 5 ETFs
│   │   └── model_predictions.csv   # Pre-computed ARIMA, LSTM, and GARCH predictions
│   └── raw/                        # Raw source files (not included, ~1.6 GB)
├── notebooks/
│   ├── 1_data_acquisition.ipynb    # Step 1: Collecting and cleaning the data
│   ├── 2_exploratory_analysis.ipynb# Step 2: Exploring patterns and trends
│   └── 3_modeling_forecasting.ipynb# Step 3: Training the forecasting models
├── papers/                         # Written reports (LaTeX and PDF)
├── events.csv                      # List of macroeconomic events and their dates
└── FEDFUNDS.csv                    # Monthly Federal Funds Rate data (1954-2026)
```

## Data

All the data needed to run the dashboard is already included in the `data/processed/` folder. You do **not** need to download anything extra.

### Where the Data Came From

| Source | URL | What We Used It For |
|--------|-----|---------------------|
| Stooq | https://stooq.com/db/h/ | Daily stock prices and trading volumes for the 5 sector ETFs |
| FRED (Federal Reserve) | https://fred.stlouisfed.org/series/FEDFUNDS | The Federal Funds Effective Rate (the interest rate set by the Fed) |

### Rebuilding the Data from Scratch (Optional)

This is only needed if you want to regenerate `data/processed/` yourself:

1. Download the full Stooq US daily dataset from https://stooq.com/db/h/ and extract it into `data/raw/`
2. Copy `FEDFUNDS.csv` and `events.csv` into `data/raw/`
3. Run:
   ```bash
   python src/preprocessing.py
   ```

## Notebooks

The three Jupyter notebooks walk through the full research pipeline, from data collection to model training. To run them you need additional packages:

```bash
pip install -r requirements-notebooks.txt
```

Then start Jupyter:

```bash
jupyter notebook
```

| Notebook | What It Covers |
|----------|----------------|
| `1_data_acquisition.ipynb` | Downloading, cleaning, and merging the raw data |
| `2_exploratory_analysis.ipynb` | Visualizing trends, volatility patterns, and correlations |
| `3_modeling_forecasting.ipynb` | Training and evaluating ARIMA, GARCH, and LSTM models |

## Tech Stack

- **Dashboard**: [Dash](https://dash.plotly.com/) + [Plotly](https://plotly.com/python/) for interactive charts, [Dash Mantine Components](https://www.dash-mantine-components.com/) for UI styling
- **Time Series Modeling**: [statsmodels](https://www.statsmodels.org/) (ARIMA), [arch](https://arch.readthedocs.io/) (GARCH), [TensorFlow/Keras](https://www.tensorflow.org/) (LSTM)
- **Data Processing**: [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)
