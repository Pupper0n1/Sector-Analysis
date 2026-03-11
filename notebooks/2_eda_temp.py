import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os



PROCESSED_DATA_PATH = "data/processed/master_sector_data.csv"
if os.path.exists(PROCESSED_DATA_PATH):
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    print("Data loaded successfully. Shape:", df.shape)
else:
    print(f"Error: File not found at {PROCESSED_DATA_PATH}")
df.head()
price_cols = [c for c in df.columns if '_close' in c]
normalized_df = df[price_cols].divide(df[price_cols].iloc[0]) * 100

fig = px.line(normalized_df, title="Normalized Sector ETF Prices (Start=100)")
fig.update_layout(xaxis_title="Date", yaxis_title="Normalized Price", legend_title="Sector")
fig.show()
ret_cols = [c for c in df.columns if '_log_ret' in c]
vol_df = df[ret_cols].rolling(window=30).std() * np.sqrt(252) # Annualized

fig = px.line(vol_df, title="30-Day Rolling Volatility (Annualized)")
fig.update_layout(xaxis_title="Date", yaxis_title="Annualized Volatility", legend_title="Sector")
fig.show()
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

# Additional analysis for progress report
print("\n--- Correlation with FEDFUNDS (Full Period) ---")
sectors = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI']
for s in sectors:
    corr = df[f'{s}_log_ret'].corr(df['FEDFUNDS'])
    print(f"{s}: {corr:.4f}")

print("\n--- Correlation with FEDFUNDS (2022-2023 Tightening) ---")
df_tight = df['2022-01-01':'2023-12-31']
for s in sectors:
    corr = df_tight[f'{s}_log_ret'].corr(df_tight['FEDFUNDS'])
    print(f"{s}: {corr:.4f}")

print("\n--- Average Volatility by Sector ---")
for s in sectors:
    vol = df[f'{s}_log_ret'].std() * np.sqrt(252)
    print(f"{s}: {vol:.4f}")
