# Project Plan: Sector-Analysis & Forecasting Dashboard

## 1. Project Overview
This project aims to build an interactive dashboard that analyzes the historical performance and volatility of five major US equity sectors (Technology, Energy, Financials, Healthcare, and Industrials) in response to macroeconomic shocks. It combines statistical modeling (ARIMA/GARCH) with deep learning (LSTM) to forecast future trends.

### Target Sectors (ETFs)
- **XLK:** Technology
- **XLE:** Energy
- **XLF:** Financials
- **XLV:** Healthcare
- **XLI:** Industrials

---

## 2. Implementation Phases

### Phase 1: Data Acquisition & Preprocessing (Status: COMPLETED)
- [x] Load raw Stooq OHLCV data for all 5 ETFs.
- [x] Integrate `FEDFUNDS.csv` (Federal Funds Rate) with daily alignment.
- [x] Create a curated `events.csv` for macroeconomic and geopolitical shocks.
- [x] Implement `src/preprocessing.py` to generate `master_sector_data.csv`.

### Phase 2: Exploratory Data Analysis (EDA) (Status: COMPLETED)
- [x] Visualize normalized price trends and annualized volatility.
- [x] Analyze rolling correlations between sectors and interest rates.
- [x] Identify heterogeneous sector responses to curated macro events.

### Phase 3: Modeling & Forecasting (Status: COMPLETED)
- [x] **Baseline Models:** Implement ARIMA (for price trends) and GARCH (for volatility) using `statsmodels`.
- [x] **Deep Learning:** Develop LSTM sequence models using `TensorFlow/Keras`.
- [x] **Validation:** Use a rolling-window approach with a hold-out test set (2024–2025).
- [x] **Evaluation:** Calculate RMSE/MAE for "stable" vs "shock" market regimes.

### Phase 4: Interactive Dashboard Development (Status: IN PROGRESS)
- [ ] **Backend:** Build a `Dash` (Python) application.
- [ ] **Visualization:**
    - **Event Overlay:** Main price chart with vertical markers for curated events.
    - **Drill-Down:** 30-day window analysis around specific shocks.
    - **Forecasting View:** Display model predictions with confidence intervals.
- [ ] **Interactivity:** Allow users to filter by sector and event category (e.g., Geopolitical, Monetary).

### Phase 5: Final Integration & Reporting
- [ ] Consolidate results into a final report (`papers/final_report.tex`).
- [ ] Prepare instructions for running the dashboard.
- [ ] Finalize code documentation and repository structure.

---

## 3. Tech Stack
- **Languages:** Python 3.11
- **Data:** Pandas, NumPy
- **Modeling:** Statsmodels, TensorFlow/Keras
- **Visualization:** Plotly, Dash
- **Environment:** Miniconda/Pip
