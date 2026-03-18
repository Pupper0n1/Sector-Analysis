"""
Interactive Dashboard: Sector Analysis & Forecasting
DATA 501 Capstone — Frese, Reyes González, Elbouni

Run: python app.py
Then open http://127.0.0.1:8050
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

warnings.filterwarnings("ignore")

# ── Data paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_PATH = os.path.join(BASE_DIR, "data", "processed", "master_sector_data.csv")
PREDS_PATH  = os.path.join(BASE_DIR, "data", "processed", "model_predictions.csv")

# ── Load master data ──────────────────────────────────────────────────────────
df = pd.read_csv(MASTER_PATH, index_col=0, parse_dates=True)
df = df.asfreq("B").ffill()

# ── Load or generate predictions ─────────────────────────────────────────────
LSTM_AVAILABLE = False
preds_loaded = False

if os.path.exists(PREDS_PATH):
    preds_df = pd.read_csv(PREDS_PATH, parse_dates=["date"])
    if "garch_vol" in preds_df.columns:
        LSTM_AVAILABLE = "lstm_pred" in preds_df.columns and not preds_df["lstm_pred"].isna().all()
        preds_loaded = True
        print("Loaded model_predictions.csv")
    else:
        print("model_predictions.csv missing 'garch_vol' — regenerating on-the-fly.")

if not preds_loaded:
    print("Generating ARIMA + GARCH predictions (LSTM unavailable).")

    SECTORS_LOCAL = ["XLK", "XLE", "XLF", "XLV", "XLI"]
    train_df = df[:"2023-12-31"]
    test_df  = df["2024-01-01":]

    rows = []
    for sector in SECTORS_LOCAL:
        print(f"  Fitting ARIMA + GARCH for {sector}...")
        full_series = df[f"{sector}_log_ret"].fillna(0)
        train_series = train_df[f"{sector}_log_ret"].fillna(0)

        # ARIMA: fit on training data, predict test period
        arima_res = ARIMA(train_series, order=(1, 0, 1)).fit()
        arima_preds = arima_res.predict(
            start=test_df.index[0], end=test_df.index[-1], dynamic=False
        ).values

        # GARCH: refit on full dataset, extract conditional volatility for test period
        garch_res = arch_model(full_series, vol="Garch", p=1, q=1, dist="Normal").fit(disp="off")
        cond_vol_ann = garch_res.conditional_volatility * np.sqrt(252)
        garch_vol_test = cond_vol_ann.loc[test_df.index].values

        # Actual 30-day rolling annualized volatility
        actual_vol_test = (full_series.rolling(30).std() * np.sqrt(252)).loc[test_df.index].values

        actuals = test_df[f"{sector}_log_ret"].fillna(0).values
        for i, date in enumerate(test_df.index):
            rows.append({
                "date": date, "sector": sector,
                "actual": actuals[i],
                "arima_pred": arima_preds[i],
                "lstm_pred": np.nan,
                "garch_vol": garch_vol_test[i],
                "actual_vol": actual_vol_test[i],
            })

    preds_df = pd.DataFrame(rows)
    preds_df.to_csv(PREDS_PATH, index=False)
    print(f"ARIMA + GARCH predictions saved to {PREDS_PATH}")

# ── Event definitions with categories ────────────────────────────────────────
EVENTS = [
    {"date": "2008-09-15", "label": "Lehman Brothers Bankruptcy",       "category": "Geopolitical"},
    {"date": "2011-08-05", "label": "US Credit Rating Downgrade",       "category": "Monetary"},
    {"date": "2014-06-20", "label": "Start of Oil Price Collapse",      "category": "Geopolitical"},
    {"date": "2015-08-11", "label": "China Yuan Devaluation",           "category": "Geopolitical"},
    {"date": "2015-11-27", "label": "Start of 2015–2016 Oil Glut",      "category": "Geopolitical"},
    {"date": "2016-06-23", "label": "Brexit Referendum",                "category": "Geopolitical"},
    {"date": "2020-03-11", "label": "COVID-19 Pandemic Declaration",    "category": "Geopolitical"},
    {"date": "2022-02-24", "label": "Invasion of Ukraine",              "category": "Geopolitical"},
    {"date": "2022-03-16", "label": "First Fed Rate Hike (2022 Cycle)", "category": "Monetary"},
]
events_df = pd.DataFrame(EVENTS)
events_df["date"] = pd.to_datetime(events_df["date"])

SECTORS = ["XLK", "XLE", "XLF", "XLV", "XLI"]
SECTOR_COLORS = {
    "XLK": "#1f77b4", "XLE": "#ff7f0e", "XLF": "#2ca02c",
    "XLV": "#d62728", "XLI": "#9467bd",
}
CATEGORY_COLORS = {"Monetary": "#e377c2", "Geopolitical": "#bcbd22"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_event_vlines(fig, filtered_events, yref="paper"):
    """Add vertical dashed lines + annotations for events."""
    for _, ev in filtered_events.iterrows():
        fig.add_vline(
            x=ev["date"].timestamp() * 1000,
            line_dash="dash",
            line_color=CATEGORY_COLORS.get(ev["category"], "gray"),
            line_width=1.2,
            opacity=0.7,
        )
        fig.add_annotation(
            x=ev["date"],
            yref=yref,
            y=1.01,
            text=ev["label"],
            showarrow=False,
            textangle=-45,
            font=dict(size=9),
            xanchor="left",
        )
    return fig


def shock_mask(sector):
    """Return boolean Series: True on 'Shock' days (rolling vol > 90th pct, test period)."""
    ret = df[f"{sector}_log_ret"].fillna(0)
    vol = ret.rolling(30).std() * np.sqrt(252)
    test_vol = vol["2024-01-01":]
    threshold = test_vol.quantile(0.90)
    return test_vol > threshold


# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Sector Analysis Dashboard"

SECTOR_OPTIONS  = [{"label": s, "value": s} for s in SECTORS]
CATEGORY_OPTIONS = [
    {"label": "All",          "value": "All"},
    {"label": "Monetary",     "value": "Monetary"},
    {"label": "Geopolitical", "value": "Geopolitical"},
]
EVENT_OPTIONS = [{"label": f"{e['label']} ({e['date'][:10]})", "value": e["date"]}
                 for e in EVENTS]

app.layout = dbc.Container(fluid=True, children=[
    html.H2("Sector ETF Analysis & Forecasting Dashboard",
            className="my-3 text-center"),
    html.P("XLK · XLE · XLF · XLV · XLI  |  2005–2026  |  DATA 501 Capstone",
           className="text-center text-muted mb-4"),

    dbc.Tabs([
        # ── Tab 1: Price & Event Overlay ─────────────────────────────────────
        dbc.Tab(label="Price & Event Overlay", children=[
            dbc.Row(className="mt-3 mb-2", children=[
                dbc.Col([
                    html.Label("Sectors"),
                    dcc.Dropdown(
                        id="overlay-sectors",
                        options=SECTOR_OPTIONS,
                        value=SECTORS,
                        multi=True,
                        clearable=False,
                    ),
                ], width=6),
                dbc.Col([
                    html.Label("Event Category"),
                    dbc.RadioItems(
                        id="overlay-category",
                        options=CATEGORY_OPTIONS,
                        value="All",
                        inline=True,
                    ),
                ], width=6),
            ]),
            dcc.Graph(id="overlay-chart", style={"height": "550px"}),
        ]),

        # ── Tab 2: Event Drill-Down ───────────────────────────────────────────
        dbc.Tab(label="Event Drill-Down", children=[
            dbc.Row(className="mt-3 mb-2", children=[
                dbc.Col([
                    html.Label("Select Event"),
                    dcc.Dropdown(
                        id="drilldown-event",
                        options=EVENT_OPTIONS,
                        value=EVENTS[6]["date"],  # COVID default
                        clearable=False,
                    ),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="drilldown-bar",  style={"height": "420px"}), width=6),
                dbc.Col(dcc.Graph(id="drilldown-line", style={"height": "420px"}), width=6),
            ]),
        ]),

        # ── Tab 3: Model Performance ──────────────────────────────────────────
        dbc.Tab(label="Model Performance", children=[
            dbc.Row(className="mt-3 mb-2", children=[
                dbc.Col([
                    html.Label("Sector"),
                    dcc.Dropdown(
                        id="forecast-sector",
                        options=SECTOR_OPTIONS,
                        value="XLK",
                        clearable=False,
                    ),
                ], width=4),
            ]),
            html.Div(id="forecast-warning", className="text-warning mb-2"),
            dcc.Graph(id="forecast-chart", style={"height": "700px"}),
        ]),

        # ── Tab 4: Rolling Correlation ────────────────────────────────────────
        dbc.Tab(label="Rolling Correlation", children=[
            dbc.Row(className="mt-3 mb-2", children=[
                dbc.Col([
                    html.Label("Rolling Window"),
                    dcc.RadioItems(
                        id="corr-window",
                        options=[
                            {"label": "3 months (63 days)",  "value": 63},
                            {"label": "6 months (126 days)", "value": 126},
                            {"label": "12 months (252 days)","value": 252},
                        ],
                        value=126,
                        inline=True,
                    ),
                ], width=8),
            ]),
            dcc.Graph(id="corr-chart", style={"height": "500px"}),
        ]),

        # ── Tab 5: Sector Relationships ──────────────────────────────────────
        dbc.Tab(label="Sector Relationships", children=[
            dbc.Row(className="mt-3 mb-2", children=[
                dbc.Col([
                    html.Label("Correlation Period (Recent History)"),
                    dcc.RadioItems(
                        id="heatmap-window",
                        options=[
                            {"label": "Last 3 Months",  "value": 63},
                            {"label": "Last 6 Months",  "value": 126},
                            {"label": "Last 12 Months", "value": 252},
                        ],
                        value=126,
                        inline=True,
                    ),
                ], width=8),
            ]),
            dcc.Graph(id="sector-heatmap", style={"height": "550px"}),
            html.P("This heatmap shows the Pearson correlation coefficient between sector log returns. Darker red indicates stronger positive correlation, while blue would indicate negative correlation.",
                   className="text-muted small mt-2 text-center"),
        ]),

        # ── Tab 6: Future Outlook ─────────────────────────────────────────────
        dbc.Tab(label="Future Outlook", children=[
            dbc.Row(className="mt-4 mb-2", children=[
                dbc.Col([
                    html.Label("Sector"),
                    dcc.Dropdown(
                        id="future-sector",
                        options=SECTOR_OPTIONS,
                        value="XLK",
                        clearable=False,
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Forecast Horizon (Trading Days)"),
                    dcc.Slider(
                        id="future-horizon",
                        min=20, max=126, step=1, value=63,
                        marks={20: "1 mo", 63: "3 mo", 126: "6 mo"},
                    ),
                ], width=6),
            ]),
            dcc.Graph(id="future-chart", style={"height": "600px"}),
            html.P("This forecast uses ARIMA for the mean price trend and GARCH for the volatility-based confidence intervals. Forecasts are calculated out-of-sample beyond February 2, 2026.",
                   className="text-muted small mt-2 text-center"),
        ]),
    ]),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("overlay-chart", "figure"),
    Input("overlay-sectors",  "value"),
    Input("overlay-category", "value"),
)
def update_overlay(sectors, category):
    if not sectors:
        return go.Figure()

    price_cols = [f"{s}_close" for s in sectors]
    norm = df[price_cols].divide(df[price_cols].iloc[0]) * 100

    fig = go.Figure()
    for s in sectors:
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[f"{s}_close"],
            name=s, line=dict(color=SECTOR_COLORS[s], width=1.5),
        ))

    filtered = events_df if category == "All" else events_df[events_df["category"] == category]
    fig = get_event_vlines(fig, filtered)

    # Legend for event categories
    for cat, col in CATEGORY_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=col, dash="dash", width=1.5),
            name=f"{cat} Event", showlegend=True,
        ))

    fig.update_layout(
        title="Normalized Sector ETF Prices (Base = 100)",
        xaxis_title="Date", yaxis_title="Normalized Price",
        legend_title="", margin=dict(t=80, r=20),
        hovermode="x unified",
        xaxis=dict(range=[norm.index[0], norm.index[-1]])
    )
    return fig


@app.callback(
    Output("drilldown-bar",  "figure"),
    Output("drilldown-line", "figure"),
    Input("drilldown-event", "value"),
)
def update_drilldown(event_date_str):
    event_date = pd.Timestamp(event_date_str)
    label = events_df.loc[events_df["date"] == event_date, "label"].values[0]

    # 22 trading-day window after event
    post = df.loc[event_date:].head(22)

    # Bar: 30-day cumulative log returns
    cum_rets = {s: post[f"{s}_log_ret"].sum() for s in SECTORS}
    bar_fig = go.Figure(go.Bar(
        x=list(cum_rets.keys()),
        y=list(cum_rets.values()),
        marker_color=[SECTOR_COLORS[s] for s in SECTORS],
        text=[f"{v:.3f}" for v in cum_rets.values()],
        textposition="outside",
    ))
    bar_fig.update_layout(
        title=f"30-Day Cumulative Returns<br><sub>{label}</sub>",
        xaxis_title="Sector", yaxis_title="Cumulative Log Return",
        yaxis_zeroline=True, margin=dict(t=80),
    )

    # Line: price indexed to event date = 100
    line_fig = go.Figure()
    window_prices = df.loc[event_date:].head(22)
    for s in SECTORS:
        base = window_prices[f"{s}_close"].iloc[0]
        normed = window_prices[f"{s}_close"] / base * 100
        line_fig.add_trace(go.Scatter(
            x=list(range(len(normed))), y=normed,
            name=s, line=dict(color=SECTOR_COLORS[s], width=2),
        ))
    line_fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
    line_fig.update_layout(
        title=f"Sector Prices (Event Date = 100)<br><sub>{label}</sub>",
        xaxis_title="Trading Days After Event",
        yaxis_title="Normalized Price",
        legend_title="", margin=dict(t=80),
    )
    return bar_fig, line_fig


@app.callback(
    Output("forecast-chart",   "figure"),
    Output("forecast-warning", "children"),
    Input("forecast-sector", "value"),
)
def update_forecast(sector):
    sec_preds = preds_df[preds_df["sector"] == sector].copy()
    sec_preds = sec_preds.sort_values("date")

    # Use exact dates from predictions to define shock mask (top 10% vol)
    full_series = df[f"{sector}_log_ret"].fillna(0)
    actual_vol = (full_series.rolling(30).std() * np.sqrt(252))
    # Filter to the dates we are actually plotting to ensure perfect alignment
    plot_vol = actual_vol.loc[sec_preds["date"]]
    threshold = plot_vol.quantile(0.90)
    mask = plot_vol > threshold

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
        subplot_titles=(
            f"{sector} Returns — Model Performance (ARIMA vs LSTM)",
            f"{sector} Volatility — Model Performance (GARCH vs Actual)",
        ),
    )

    # ── Row 1: Returns ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=sec_preds["date"], y=sec_preds["actual"],
        name="Actual Return", line=dict(color="black", width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sec_preds["date"], y=sec_preds["arima_pred"],
        name="ARIMA(1,0,1)", line=dict(color="#1f77b4", width=1.5),
    ), row=1, col=1)
    if LSTM_AVAILABLE:
        fig.add_trace(go.Scatter(
            x=sec_preds["date"], y=sec_preds["lstm_pred"],
            name="LSTM", line=dict(color="#d62728", width=1.5),
        ), row=1, col=1)

    # ── Row 2: Volatility ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=plot_vol.index, y=plot_vol.values,
        name="Actual Vol (30-day)", line=dict(color="black", width=1, dash="dot"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=sec_preds["date"], y=sec_preds["garch_vol"],
        name="GARCH(1,1)", line=dict(color="#ff7f0e", width=1.5),
    ), row=2, col=1)

    # ── Add Shock highlights (vrect) ──────────────────────────────────────────
    # We do this after adding traces and use strings for safety in subplots
    in_shock = False
    shock_start = None
    for date, is_shock in mask.items():
        dt_str = date.strftime("%Y-%m-%d")
        if is_shock and not in_shock:
            shock_start = dt_str
            in_shock = True
        elif not is_shock and in_shock:
            for row in (1, 2):
                fig.add_vrect(
                    x0=shock_start, x1=dt_str,
                    fillcolor="red", opacity=0.15, layer="above",
                    line_width=0, row=row, col=1,
                )
            in_shock = False
    if in_shock:
        last_str = mask.index[-1].strftime("%Y-%m-%d")
        for row in (1, 2):
            fig.add_vrect(
                x0=shock_start, x1=last_str,
                fillcolor="red", opacity=0.15, layer="above",
                line_width=0, row=row, col=1,
            )

    # Invisible trace for shock regime legend entry
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="rgba(255,0,0,0.3)", symbol="square"),
        name="Shock Regime (top 10% vol)",
    ), row=1, col=1)

    fig.update_yaxes(title_text="Log Return",           row=1, col=1)
    fig.update_yaxes(title_text="Annualized Volatility", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        legend_title="", hovermode="x unified",
        margin=dict(t=80), height=700,
        xaxis=dict(range=[sec_preds["date"].min(), sec_preds["date"].max()])
    )
    warning = "" if LSTM_AVAILABLE else "ℹ LSTM predictions unavailable — run notebook 3 to generate model_predictions.csv."
    return fig, warning


@app.callback(
    Output("corr-chart", "figure"),
    Input("corr-window", "value"),
)
def update_corr(window):
    fig = go.Figure()
    for s in SECTORS:
        corr = df[f"{s}_log_ret"].rolling(window).corr(df["FEDFUNDS"])
        fig.add_trace(go.Scatter(
            x=corr.index, y=corr,
            name=s, line=dict(color=SECTOR_COLORS[s], width=1.5),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    window_label = {63: "3-Month", 126: "6-Month", 252: "12-Month"}[window]
    fig.update_layout(
        title=f"{window_label} Rolling Correlation of Sector Log Returns with FEDFUNDS",
        xaxis_title="Date", yaxis_title="Correlation Coefficient",
        legend_title="", hovermode="x unified", margin=dict(t=60),
        yaxis=dict(range=[-1, 1]),
    )
    return fig


@app.callback(
    Output("future-chart", "figure"),
    Input("future-sector", "value"),
    Input("future-horizon", "value"),
)
def update_future(sector, horizon):
    # 1. Prepare historical data
    full_series = df[f"{sector}_log_ret"].fillna(0)
    last_price = df[f"{sector}_close"].iloc[-1]
    last_date = df.index[-1]

    # 2. Fit models on full available data
    # ARIMA for the mean (drift)
    arima_res = ARIMA(full_series, order=(1, 0, 1)).fit()
    # GARCH for the volatility (uncertainty)
    garch_model_obj = arch_model(full_series, vol="Garch", p=1, q=1, dist="Normal")
    garch_res = garch_model_obj.fit(disp="off")

    # 3. Forecast h-steps ahead
    # Returns forecast
    arima_forecast = arima_res.forecast(steps=horizon)
    # Volatility forecast
    garch_forecast = garch_res.forecast(horizon=horizon)
    # cond_mean is basically 0 for GARCH usually, but we use the variance
    future_var = garch_forecast.variance.values[-1, :]
    future_std = np.sqrt(future_var)

    # 4. Construct Future Date Index
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                periods=horizon, freq="B")

    # 5. Convert Return Forecast to Price Projection
    # price_t = price_{t-1} * exp(log_return_t)
    # Cumulative sum of log returns to get cumulative log growth
    cum_log_ret = np.cumsum(arima_forecast.values)
    price_pred = last_price * np.exp(cum_log_ret)

    # Confidence Intervals (95% = 1.96 * std)
    # This is a simplification: volatility also compounds
    cum_var = np.cumsum(future_var)
    cum_std = np.sqrt(cum_var)
    upper_bound = last_price * np.exp(cum_log_ret + 1.96 * cum_std)
    lower_bound = last_price * np.exp(cum_log_ret - 1.96 * cum_std)

    # 6. Plotting
    fig = go.Figure()

    # FULL history (allows panning back to 2005)
    fig.add_trace(go.Scatter(
        x=df.index, y=df[f"{sector}_close"],
        name="Historical Price", line=dict(color="black", width=1.5),
    ))

    # Prediction
    fig.add_trace(go.Scatter(
        x=future_dates, y=price_pred,
        name="ARIMA Forecast", line=dict(color="#1f77b4", width=2),
    ))

    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="95% Confidence Interval (GARCH)",
    ))

    # Vertical line at end of historical data
    fig.add_vline(x=last_date, line_dash="dash", line_color="gray")
    fig.add_annotation(x=last_date, y=1.05, yref="paper", text="End of Data (Feb 2, 2026)", showarrow=False)

    # Calculate a nice initial view window (2 years back from today)
    start_view = last_date - pd.DateOffset(years=2)

    fig.update_layout(
        title=f"{sector} Future Price Projection ({horizon} Trading Days)",
        xaxis_title="Date",
        yaxis_title="ETF Price ($)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            range=[start_view, future_dates[-1]], 
            rangeslider=dict(
                visible=True,
                range=[df.index[0], future_dates[-1]] # Range slider shows ALL data
            ),
            type="date",
            # This 'constrain' and 'autorange' ensures you can't pan into the void
            autorange=False
        )
    )

    return fig


@app.callback(
    Output("sector-heatmap", "figure"),
    Input("heatmap-window", "value"),
)
def update_heatmap(window):
    # Calculate returns for the most recent N days
    recent_df = df[[f"{s}_log_ret" for s in SECTORS]].tail(window)
    # Rename columns to just sector names for the plot
    recent_df.columns = SECTORS
    
    corr_matrix = recent_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r", # Red for positive, Blue for negative
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        hoverinfo="z",
    ))
    
    window_label = {63: "3-Month", 126: "6-Month", 252: "12-Month"}[window]
    fig.update_layout(
        title=f"Sector-to-Sector Correlation Heatmap ({window_label} Window)",
        xaxis_title="Sector",
        yaxis_title="Sector",
        margin=dict(t=60, b=40, l=40, r=40),
    )
    
    return fig


if __name__ == "__main__":
    app.run(debug=True)
