#!/usr/bin/env python
# coding: utf-8

# # Phase 3: Modeling & Forecasting
# 
# This notebook focuses on:
# 1. **Baseline Models:** ARIMA (for price trends) and GARCH (for volatility).
# 2. **Deep Learning:** LSTM models for time-series forecasting.
# 3. **Validation:** Rolling-window evaluation (2024-2025).
# 4. **Evaluation:** RMSE/MAE for stable vs. shock market regimes.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DATA_PATH = "../data/processed/master_sector_data.csv"
if os.path.exists(PROCESSED_DATA_PATH):
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    # Ensure business day frequency for statsmodels
    df = df.asfreq('B').ffill()
    print("Data loaded successfully. Shape:", df.shape)
else:
    print(f"Error: File not found at {PROCESSED_DATA_PATH}")

# Use only data until end of 2023 for training, 2024-2025 for validation as per plan
train_df = df[:'2023-12-31']
test_df = df['2024-01-01':]
print(f"Training set: {train_df.index[0]} to {train_df.index[-1]}")
print(f"Test set: {test_df.index[0]} to {test_df.index[-1]}")


# ## 1. Baseline Models: ARIMA & GARCH
# We start by fitting ARIMA models to the log returns (to capture mean-reverting or trending behavior) and GARCH models to capture volatility clustering.

# In[2]:


def fit_baseline_models(series, name):
    # Simple ARIMA(1,0,1) for returns
    arima_model = ARIMA(series, order=(1, 0, 1))
    arima_result = arima_model.fit()

    # GARCH(1,1) for volatility
    garch = arch_model(series, vol='Garch', p=1, q=1, dist='Normal')
    garch_result = garch.fit(disp='off')

    print(f"--- {name} Baseline Results ---")
    print(f"ARIMA Log-Likelihood: {arima_result.llf:.2f}")
    print(f"GARCH Log-Likelihood: {garch_result.loglikelihood:.2f}")

    return arima_result, garch_result

sectors = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI']
baselines = {}

for sector in sectors:
    ret_col = f'{sector}_log_ret'
    series = train_df[ret_col].fillna(0)
    baselines[sector] = fit_baseline_models(series, sector)


# ## 2. Deep Learning: LSTM Models
# We implement a Long Short-Term Memory (LSTM) network to capture non-linear temporal dependencies in sector returns.

# In[3]:


def prepare_lstm_data(df, sector, window_size=60):
    ret_col = f'{sector}_log_ret'
    data = df[ret_col].fillna(0).values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# In[ ]:


window_size = 60
lstm_models = {}
scalers = {}

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

for sector in sectors:
    print(f"Training LSTM for {sector}...")
    X_train, y_train, scaler = prepare_lstm_data(train_df, sector, window_size)

    model = build_lstm_model((X_train.shape[1], 1))
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )
    stopped_epoch = early_stopping.stopped_epoch if early_stopping.stopped_epoch > 0 else 50
    print(f"LSTM for {sector} trained --- stopped at epoch {stopped_epoch}, best val_loss: {min(history.history['val_loss']):.6f}")

    lstm_models[sector] = model
    scalers[sector] = scaler


# ## 3. Validation: One-Step-Ahead Evaluation (2024--2025)
# ARIMA uses `predict()` over the full test period (out-of-sample, no refit). LSTM batches all 60-day windows from the test period at once.

# In[5]:


results = {}

for sector in sectors:
    print(f"Evaluating {sector}...")
    ret_col = f'{sector}_log_ret'
    actuals = test_df[ret_col].fillna(0).values

    # 1. ARIMA: predict over full test period (out-of-sample, no refit)
    arima_res, _ = baselines[sector]
    arima_preds = arima_res.predict(
        start=test_df.index[0],
        end=test_df.index[-1],
        dynamic=False
    ).values

    # 2. LSTM: batch all windows at once
    full_series = df[ret_col].fillna(0).values.reshape(-1, 1)
    scaler = scalers[sector]
    scaled_full = scaler.transform(full_series)

    test_start_idx = len(train_df)
    windows = np.array([
        scaled_full[i - window_size:i, 0]
        for i in range(test_start_idx, len(df))
    ])
    windows = windows.reshape(windows.shape[0], window_size, 1)
    lstm_preds_scaled = lstm_models[sector].predict(windows, verbose=0)
    lstm_preds_inv = scaler.inverse_transform(lstm_preds_scaled).flatten()

    results[sector] = {
        'dates': test_df.index,
        'actual': actuals,
        'arima': arima_preds,
        'lstm': lstm_preds_inv
    }
    print(f"  done --- {len(actuals)} test observations")


# ## 4. Visualizing Forecasts
# Plotting the predicted vs actual log returns for a representative sector.

# In[6]:


sector_to_plot = 'XLK'
data = results[sector_to_plot]

fig = go.Figure()
fig.add_trace(go.Scatter(x=data['dates'], y=data['actual'], name='Actual', line=dict(color='black', width=1, dash='dot')))
fig.add_trace(go.Scatter(x=data['dates'], y=data['arima'], name='ARIMA', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data['dates'], y=data['lstm'], name='LSTM', line=dict(color='red')))

fig.update_layout(title=f"{sector_to_plot} Forecast Comparison (2024-2025)",
                  xaxis_title="Date", yaxis_title="Log Return")
fig.show()


# ## 5. Regime-Based Evaluation: Stable vs. Shock
# We define 'Shock' periods as days where the sector's 30-day rolling volatility is in the top 10% of its historical distribution (calculated from the test set).

# In[7]:


regime_results = []

for sector, data in results.items():
    actual = data['actual']
    arima = data['arima']
    lstm = data['lstm']

    # Calculate rolling volatility for the test set to define regimes
    # We'll use the original df to get enough history for the start of the test set
    full_ret = df[f'{sector}_log_ret'].fillna(0)
    vol = full_ret.rolling(30).std() * np.sqrt(252)
    test_vol = vol.loc[data['dates']]

    threshold = test_vol.quantile(0.90)
    shock_mask = test_vol > threshold

    for regime, mask in [('Stable', ~shock_mask), ('Shock', shock_mask)]:
        r_actual = actual[mask]
        r_arima = arima[mask]
        r_lstm = lstm[mask]

        if len(r_actual) > 0:
            rmse_arima = np.sqrt(mean_squared_error(r_actual, r_arima))
            rmse_lstm = np.sqrt(mean_squared_error(r_actual, r_lstm))

            regime_results.append({
                'Sector': sector,
                'Regime': regime,
                'Samples': len(r_actual),
                'ARIMA_RMSE': rmse_arima,
                'LSTM_RMSE': rmse_lstm,
                'LSTM_Improvement_%': (rmse_arima - rmse_lstm) / rmse_arima * 100
            })

regime_df = pd.DataFrame(regime_results)
print("Regime-Based Performance Comparison:")
display(regime_df.pivot(index='Sector', columns='Regime', values=['ARIMA_RMSE', 'LSTM_RMSE']))


# In[ ]:


## Save predictions for dashboard use
import os
os.makedirs("../data/processed", exist_ok=True)

rows = []
for sector, data in results.items():
    # Refit GARCH on full dataset to get conditional volatility over test period
    full_series = df[f'{sector}_log_ret'].fillna(0)
    garch_full = arch_model(full_series, vol='Garch', p=1, q=1, dist='Normal').fit(disp='off')
    cond_vol_ann = garch_full.conditional_volatility * np.sqrt(252)  # annualize
    garch_vol_test = cond_vol_ann.loc[data['dates']].values

    # Actual 30-day rolling annualized volatility for test period
    actual_vol = full_series.rolling(30).std() * np.sqrt(252)
    actual_vol_test = actual_vol.loc[data['dates']].values

    for i, date in enumerate(data['dates']):
        rows.append({
            'date': date,
            'sector': sector,
            'actual': data['actual'][i],
            'arima_pred': data['arima'][i],
            'lstm_pred': data['lstm'][i],
            'garch_vol': garch_vol_test[i],
            'actual_vol': actual_vol_test[i],
        })

pred_df = pd.DataFrame(rows)
pred_df.to_csv("../data/processed/model_predictions.csv", index=False)
print(f"Saved model_predictions.csv --- {len(pred_df)} rows")

