#!/usr/bin/env python
# coding: utf-8

# # Phase 2: Exploratory Data Analysis (EDA)
# 
# This notebook focuses on:
# 1. Visualizing price trends and log returns for all 5 sectors.
# 2. Analyzing volatility patterns (rolling standard deviation).
# 3. Correlating sector returns with the Federal Funds Rate.
# 4. Overlaying manually curated macroeconomic events on performance charts.
# 5. Measuring heterogeneous sector responses to curated shocks.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

get_ipython().run_line_magic('matplotlib', 'inline')

PROCESSED_DATA_PATH = "../data/processed/master_sector_data.csv"
if os.path.exists(PROCESSED_DATA_PATH):
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    print("Data loaded successfully. Shape:", df.shape)
else:
    print(f"Error: File not found at {PROCESSED_DATA_PATH}")
df.head()


# ## 1. Normalized Sector Price Trends
# * **What it shows:** The price of all five ETFs (XLK, XLE, XLF, XLV, XLI) adjusted to start at a base value of 100.

# In[2]:


price_cols = [c for c in df.columns if '_close' in c]
normalized_df = df[price_cols].divide(df[price_cols].iloc[0]) * 100

fig = px.line(normalized_df, title="Normalized Sector ETF Prices (Start=100)")
fig.update_layout(xaxis_title="Date", yaxis_title="Normalized Price", legend_title="Sector")
fig.show()


# ## 2. 30-Day Rolling Volatility (Annualized)
# * **What it shows:** The standard deviation of daily returns over a moving 30-day window, multiplied by the square root of 252 (trading days) to make it an annual percentage.

# In[3]:


ret_cols = [c for c in df.columns if '_log_ret' in c]
vol_df = df[ret_cols].rolling(window=30).std() * np.sqrt(252) # Annualized

fig = px.line(vol_df, title="30-Day Rolling Volatility (Annualized)")
fig.update_layout(xaxis_title="Date", yaxis_title="Annualized Volatility", legend_title="Sector")
fig.show()


# ## 3. Macro Event Overlay
# * **What it shows:** A price chart for a specific sector (e.g., Tech) with vertical markers indicating the exact dates of your curated events (Lehman collapse, COVID-19, etc.).

# In[4]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['XLK_close'], name="XLK (Tech) Close"))

events = df[df['event_marker'].notna()]
ymax = df['XLK_close'].max()

for i, (idx, row) in enumerate(events.iterrows()):
    fig.add_vline(x=idx, line_dash="dash", line_color="red")
    fig.add_annotation(
        x=idx,
        y=ymax + (i % 5) * (ymax * 0.10),  # stagger in 5 repeating layers
        text=row['event_marker'],
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-20
    )

fig.update_layout(
    title="XLK Price with Macroeconomic Event Overlay",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    margin=dict(t=100)
)

fig.show()


# ## 4. Rolling Correlation with FEDFUNDS
# * **What it shows:** A moving 6-month window showing how closely sector returns move in the same (or opposite) direction as the Federal Funds Rate.

# In[5]:


corr_window = 126 # ~6 months
correlations = {}
ret_cols = [c for c in df.columns if '_log_ret' in c]

for ret_col in ret_cols:
    symbol = ret_col.split('_')[0]
    correlations[symbol] = df[ret_col].rolling(window=corr_window).corr(df['FEDFUNDS'])

corr_df = pd.DataFrame(correlations)
fig = px.line(corr_df, title="6-Month Rolling Correlation with FEDFUNDS", markers=True)
fig.update_layout(xaxis_title="Date", yaxis_title="Correlation Coefficient", legend_title="Sector")
fig.show()


# ## 5. Event Impact Analysis (Heterogeneous Response)
# * **What it shows:** A bar chart comparing the 30-day cumulative returns of every sector immediately following a specific macro event.

# In[6]:


event_dates = df[df['event_marker'].notna()].index
impact_results = []
ret_cols = [c for c in df.columns if '_log_ret' in c]

for event_date in event_dates:
    event_name = df.loc[event_date, 'event_marker']
    # 30-day window (approx 22 trading days)
    post_window = df.loc[event_date:].head(22)

    impact = {'Event': event_name, 'Date': event_date.strftime('%Y-%m-%d')}
    for ret_col in ret_cols:
        symbol = ret_col.split('_')[0]
        # Cumulative return from log returns is sum of log returns
        cum_ret = post_window[ret_col].sum()
        impact[symbol] = cum_ret

    impact_results.append(impact)

impact_df = pd.DataFrame(impact_results)
impact_melted = impact_df.melt(id_vars=['Event', 'Date'], var_name='Sector', value_name='30D_Cum_Return')

fig = px.bar(impact_melted, x='Sector', y='30D_Cum_Return', color='Sector', 
             facet_col='Event', facet_col_wrap=3,
             title="Sector Heterogeneous Response: 30-Day Cumulative Returns Post-Event")
fig.update_layout(showlegend=False)
fig.show()


# ## 6. Sector Correlation Heatmap
# * **What it shows:** Pairwise Pearson correlations between sector log returns over the full dataset (2005–2026). Reveals which sectors move together and which are relatively independent, providing context for diversification analysis and cross-sector transmission patterns.

# In[ ]:


sectors = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI']
ret_cols = [f"{s}_log_ret" for s in sectors]
corr_matrix = df[ret_cols].corr()
corr_matrix.index = sectors
corr_matrix.columns = sectors

fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=sectors,
    y=sectors,
    colorscale="RdBu_r",
    zmin=-1, zmax=1,
    text=np.round(corr_matrix.values, 2),
    texttemplate="%{text}",
    hoverinfo="z",
    colorbar=dict(title="Pearson r"),
))

fig.update_layout(
    title="Sector-to-Sector Log Return Correlation (2005–2026)",
    xaxis_title="Sector",
    yaxis_title="Sector",
    width=600,
    height=550,
    margin=dict(t=80, b=60, l=60, r=60),
)

fig.write_image("sector_correlation_heatmap.png", scale=2)
fig.show()
print("Heatmap saved to sector_correlation_heatmap.png")

