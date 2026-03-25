#!/usr/bin/env python
# coding: utf-8

# # Phase 1: Data Acquisition & Integration
# 
# This notebook handles the loading and merging of three primary data sources:
# 1. **Market Data (Stooq):** Daily OHLCV for XLK, XLE, XLF, XLV, and XLI.
# 2. **Macroeconomic Indicators (FRED):** Effective Federal Funds Rate.
# 3. **Event Metadata (Manual):** Curated list of macroeconomic and geopolitical shocks.
# 
# The output is a consolidated `master_sector_data.csv` used for modeling and the dashboard.

# In[5]:


import pandas as pd
import numpy as np
import os

# Paths
RAW_DATA_DIR = "../data/raw"
PROCESSED_DATA_DIR = "../data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

ETFS = {
    'XLK': os.path.join(RAW_DATA_DIR, "nyse etfs", "2", "xlk.us.txt"),
    'XLE': os.path.join(RAW_DATA_DIR, "nyse etfs", "2", "xle.us.txt"),
    'XLF': os.path.join(RAW_DATA_DIR, "nyse etfs", "2", "xlf.us.txt"),
    'XLV': os.path.join(RAW_DATA_DIR, "nyse etfs", "2", "xlv.us.txt"),
    'XLI': os.path.join(RAW_DATA_DIR, "nyse etfs", "2", "xli.us.txt")
}
FEDFUNDS_PATH = os.path.join(RAW_DATA_DIR, "FEDFUNDS.csv")
EVENTS_PATH = os.path.join(RAW_DATA_DIR, "events.csv")


# ## 1. Load and Clean ETF Data
# We calculate **daily log returns** for each sector.

# In[6]:


def load_stooq_data(filepath, symbol):
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}")
        return None

    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip('<>') for c in df.columns]
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Calculate daily log returns
    df[f'{symbol}_log_ret'] = np.log(df['close'] / df['close'].shift(1))

    # Select relevant columns and rename them to avoid collisions
    cols = ['close', 'vol', f'{symbol}_log_ret']
    df = df[cols]
    df.columns = [f'{symbol}_{c}' if c != f'{symbol}_log_ret' else c for c in df.columns]
    return df

etf_dfs = []
for symbol, path in ETFS.items():
    df_etf = load_stooq_data(path, symbol)
    if df_etf is not None:
        etf_dfs.append(df_etf)

master_df = etf_dfs[0]
for df_etf in etf_dfs[1:]:
    master_df = master_df.join(df_etf, how='outer')

master_df.head()


# ## 2. Integrate Federal Funds Rate
# The rate is monthly; we forward-fill it to align with daily trading data.

# In[7]:


if os.path.exists(FEDFUNDS_PATH):
    fed_df = pd.read_csv(FEDFUNDS_PATH)
    fed_df['observation_date'] = pd.to_datetime(fed_df['observation_date'])
    fed_df.set_index('observation_date', inplace=True)
    fed_df.sort_index(inplace=True)

    master_df = master_df.join(fed_df, how='left')
    master_df['FEDFUNDS'] = master_df['FEDFUNDS'].ffill()
else:
    print("Warning: FEDFUNDS.csv not found.")

master_df.tail()


# ## 3. Integrate Curated Events
# Markers for macroeconomic and geopolitical shocks.

# In[8]:


if os.path.exists(EVENTS_PATH):
    events_df = pd.read_csv(EVENTS_PATH)
    events_df['date'] = pd.to_datetime(events_df['date'])
    events_map = events_df.set_index('date')['event'].to_dict()

    master_df['event_marker'] = master_df.index.map(lambda d: events_map.get(d, np.nan))
else:
    print("Warning: events.csv not found.")

master_df[master_df['event_marker'].notna()].head()


# ## 4. Export Master Dataset

# In[9]:


output_file = os.path.join(PROCESSED_DATA_DIR, "master_sector_data.csv")
master_df.to_csv(output_file)
print(f"Master dataset saved to {output_file}")
print(f"Dataset shape: {master_df.shape}")

