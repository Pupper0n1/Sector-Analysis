"""
Interactive Dashboard: Sector Analysis & Forecasting
DATA 501 Capstone --- Frese, Reyes Gonzalez, Elbouni

Enhanced with Dash Mantine Components for a professional aesthetic.
"""

# Standard library and data science imports
import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dash imports for building the web app
import dash
from dash import dcc, html, Input, Output, State
import dash_mantine_components as dmc
from dash_iconify import DashIconify

# Time series model imports
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# Suppress convergence warnings from ARIMA/GARCH fitting
warnings.filterwarnings("ignore")

# Build absolute paths so the app works regardless of where it's run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_PATH = os.path.join(
    BASE_DIR, "data", "processed", "master_sector_data.csv")
PREDS_PATH = os.path.join(
    BASE_DIR, "data", "processed", "model_predictions.csv")

# Load master dataset and align to business-day frequency, forward-filling any gaps
df = pd.read_csv(MASTER_PATH, index_col=0, parse_dates=True)
df = df.asfreq("B").ffill()

# Try to load pre-computed model predictions from notebook 3
# If the file doesn't exist or is incomplete, we re-generate on the fly
LSTM_AVAILABLE = False
preds_loaded = False

if os.path.exists(PREDS_PATH):
    preds_df = pd.read_csv(PREDS_PATH, parse_dates=["date"])
    if "garch_vol" in preds_df.columns:
        LSTM_AVAILABLE = "lstm_pred" in preds_df.columns and not preds_df["lstm_pred"].isna(
        ).all()
        preds_loaded = True
        print("Loaded model_predictions.csv")
    else:
        print("model_predictions.csv missing 'garch_vol' --- regenerating on-the-fly.")

if not preds_loaded:
    print("Generating ARIMA + GARCH predictions (LSTM unavailable).")

    SECTORS_LOCAL = ["XLK", "XLE", "XLF", "XLV", "XLI"]
    train_df = df[:"2023-12-31"]
    test_df = df["2024-01-01":]

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
        garch_res = arch_model(full_series, vol="Garch",
                               p=1, q=1, dist="Normal").fit(disp="off")
        cond_vol_ann = garch_res.conditional_volatility * np.sqrt(252)
        garch_vol_test = cond_vol_ann.loc[test_df.index].values

        # Actual 30-day rolling annualized volatility
        actual_vol_test = (full_series.rolling(30).std() *
                           np.sqrt(252)).loc[test_df.index].values

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

# -- Event definitions with categories ----------------------------------------
EVENTS = [
    {"date": "2008-09-15", "label": "Lehman Brothers Bankruptcy",
        "category": "Geopolitical"},
    {"date": "2011-08-05", "label": "US Credit Rating Downgrade",
        "category": "Monetary"},
    {"date": "2014-06-20", "label": "Start of Oil Price Collapse",
        "category": "Geopolitical"},
    {"date": "2015-08-11", "label": "China Yuan Devaluation",
        "category": "Geopolitical"},
    {"date": "2015-11-27", "label": "Start of 2015--2016 Oil Glut",
        "category": "Geopolitical"},
    {"date": "2016-06-23", "label": "Brexit Referendum",
        "category": "Geopolitical"},
    {"date": "2020-03-11", "label": "COVID-19 Pandemic Declaration",
        "category": "Geopolitical"},
    {"date": "2022-02-24", "label": "Invasion of Ukraine",
        "category": "Geopolitical"},
    {"date": "2022-03-16",
        "label": "First Fed Rate Hike (2022 Cycle)", "category": "Monetary"},
]
events_df = pd.DataFrame(EVENTS)
events_df["date"] = pd.to_datetime(events_df["date"])

# The five sector ETFs we're analyzing
SECTORS = ["XLK", "XLE", "XLF", "XLV", "XLI"]

# Consistent colors for each sector across all charts
SECTOR_COLORS = {
    "XLK": "#1f77b4", "XLE": "#ff7f0e", "XLF": "#2ca02c",
    "XLV": "#d62728", "XLI": "#9467bd",
}

# Colors for event category vertical lines (pink = monetary, yellow = geopolitical)
CATEGORY_COLORS = {"Monetary": "#e377c2", "Geopolitical": "#bcbd22"}



def get_event_vlines(fig, filtered_events, yref="paper"):
    """Add vertical dashed lines + annotations for events."""
    y_levels = [0.9, 1.05]
    for i, (_, ev) in enumerate(filtered_events.iterrows()):
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
            y=y_levels[i % len(y_levels)],
            text=ev["label"],
            showarrow=False,
            textangle=-45,
            font=dict(size=9),
            xanchor="left",
        )
    return fig


app = dash.Dash(__name__)
app.title = "Sector Analysis Dashboard"

SECTOR_OPTIONS = [{"label": s, "value": s} for s in SECTORS]
CATEGORY_OPTIONS = [
    {"label": "All",          "value": "All"},
    {"label": "Monetary",     "value": "Monetary"},
    {"label": "Geopolitical", "value": "Geopolitical"},
]
EVENT_OPTIONS = [{"label": f"{e['label']} ({e['date'][:10]})", "value": e["date"]}
                 for e in EVENTS]

app.layout = dmc.MantineProvider(
    theme={
        "primaryColor": "indigo",
        "fontFamily": "'Inter', sans-serif",
    },
    children=[
        dmc.Container(
            size="xl",
            p="md",
            children=[
                dmc.Paper(
                    p="xl",
                    radius="lg",
                    withBorder=True,
                    shadow="sm",
                    mb="xl",
                    style={"backgroundColor": "#ffffff", "overflow": "hidden"},
                    children=[
                        dmc.Group(
                            justify="space-between",
                            align="flex-start",
                            children=[
                                dmc.Group(
                                    children=[
                                        dmc.ThemeIcon(
                                            size="xl",
                                            radius="md",
                                            # variant="gradient",
                                            # gradient={"from": "indigo", "to": "cyan", "deg": 45},
                                            children=DashIconify(
                                                icon="mdi:finance", width=28, color="white"),
                                        ),
                                        dmc.Title(
                                            "Sector Analysis: Interactive Dashboard for Analysis and Forecasting of Sector Performance",
                                            order=1,
                                            # variant="gradient",
                                            # gradient={"from": "indigo", "to": "cyan", "deg": 45},
                                            style={"fontWeight": 800}
                                        ),
                                    ]
                                ),
                                dmc.Group(
                                    children=[
                                        dmc.Badge(
                                            "DATA 501 Capstone", color="indigo", variant="light", size="lg"),
                                        dmc.Badge(
                                            "2005--2026", color="cyan", variant="light", size="lg"),
                                        dmc.Badge("ARIMA", color="teal",
                                                  variant="light", size="lg"),
                                        dmc.Badge("LSTM", color="green",
                                                  variant="light", size="lg"),
                                        dmc.Badge("GARCH", color="violet",
                                                  variant="light", size="lg"),
                                    ]
                                )
                            ]
                        ),
                        dmc.Group(
                            justify="space-between",
                            align="flex-end",
                            children=[
                                dmc.Stack(
                                    gap=4,
                                    children=[
                                        dmc.Text(
                                            "Analyzing historical shocks, cross-sector relationships, and projecting future volatility for XLK, XLE, XLF, XLV, and XLI.",
                                            c="dimmed",
                                            size="md",
                                            mt="md",
                                            style={"maxWidth": "800px"}
                                        ),
                                        dmc.Text(
                                            " All model outputs are for analytical and educational purposes only. They do not constitute trading recommendations. Financial markets are inherently unpredictable and past performance is not indicative of future results.",
                                            c="dimmed",
                                            size="xs",
                                            fs="italic",
                                            style={"maxWidth": "800px"}
                                        ),
                                    ]
                                ),
                                dmc.Text(
                                    "Ashton Frese, Karen Reyes Gonzalez, Wilbur Elbouni",
                                    size="sm",
                                    fw=500,
                                    c="dimmed",
                                ),
                            ]
                        ),
                    ]
                ),

                dmc.Tabs(
                    value="overlay",
                    variant="pills",
                    radius="xl",
                    color="indigo",
                    id="main-tabs",
                    children=[
                        dmc.TabsList(
                            justify="center",
                            mb="xl",
                            style={
                                "backgroundColor": "#f1f3f5", "padding": "4px", "borderRadius": "50px"},
                            children=[
                                dmc.TabsTab("Price Overlay", value="overlay", leftSection=DashIconify(
                                    icon="mdi:layers")),
                                dmc.TabsTab(
                                    "30-day Rolling Volatility", value="volatility", leftSection=DashIconify(icon="mdi:chart-line")),
                                dmc.TabsTab(
                                    "Event Drill-Down", value="events", leftSection=DashIconify(icon="mdi:magnify-expand")),
                                dmc.TabsTab("Model Performance", value="performance", leftSection=DashIconify(
                                    icon="mdi:chart-bell-curve")),
                                dmc.TabsTab("Rolling Correlation with FEDFUNDS", value="corr", leftSection=DashIconify(
                                    icon="mdi:chart-scatter-plot")),
                                dmc.TabsTab("Sector Relationships", value="relationships", leftSection=DashIconify(
                                    icon="mdi:grid")),
                                dmc.TabsTab("Future Outlook", value="future", leftSection=DashIconify(
                                    icon="mdi:crystal-ball")),
                            ]
                        ),

                        # -- Tab 1: Price & Event Overlay ---------------------
                        dmc.TabsPanel(
                            value="overlay",
                            children=[
                                dmc.Grid(
                                    children=[
                                        dmc.GridCol(
                                            span=6,
                                            children=[
                                                dmc.MultiSelect(
                                                    id="overlay-sectors",
                                                    label="Select Sectors",
                                                    placeholder="Choose sectors to compare",
                                                    data=SECTOR_OPTIONS,
                                                    value=SECTORS,
                                                    searchable=True,
                                                )
                                            ]
                                        ),
                                        dmc.GridCol(
                                            span=6,
                                            children=[
                                                dmc.Text(
                                                    "Event Category", size="sm", fw=500, mb=5),
                                                dmc.SegmentedControl(
                                                    id="overlay-category",
                                                    data=CATEGORY_OPTIONS,
                                                    value="All",
                                                    fullWidth=True,
                                                    color="indigo",
                                                )
                                            ]
                                        ),
                                    ]
                                ),
                                dmc.Space(h="md"),
                                dmc.Paper(withBorder=True, shadow="md", p="md", radius="lg", children=[
                                    dcc.Graph(
                                        id="overlay-chart", style={"height": "550px"})
                                ]),
                                dmc.Text(
                                    "This chart shows normalized ETF prices where the first date of the dataset (2005) is indexed to 100. Vertical lines indicate major macroeconomic events.",
                                    size="sm", c="dimmed", ta="center", mt="md"
                                )
                            ]
                        ),

                        dmc.TabsPanel(
                            value="volatility",
                            children=[
                                dmc.Grid(
                                    children=[
                                        dmc.GridCol(
                                            span=6,
                                            children=[
                                                dmc.MultiSelect(
                                                    id="vol-sectors",
                                                    label="Select Sectors",
                                                    placeholder="Choose sectors to compare",
                                                    data=SECTOR_OPTIONS,
                                                    value=SECTORS,
                                                    searchable=True,
                                                )
                                            ]
                                        ),
                                        dmc.GridCol(
                                            span=6,
                                            children=[
                                                dmc.Text(
                                                    "Event Category", size="sm", fw=500, mb=5),
                                                dmc.SegmentedControl(
                                                    id="vol-category",
                                                    data=CATEGORY_OPTIONS,
                                                    value="All",
                                                    fullWidth=True,
                                                    color="indigo",
                                                )
                                            ]
                                        ),
                                    ]
                                ),
                                dmc.Space(h="md"),
                                dmc.Paper(withBorder=True, shadow="md", p="md", radius="lg", children=[
                                    dcc.Graph(id="vol-chart",
                                              style={"height": "550px"})
                                ]),
                                dmc.Text(
                                    "30-day annualized rolling volatility (sigma x sqrt252) for each sector. Vertical lines mark major macroeconomic events. Spikes indicate elevated uncertainty around shocks.",
                                    size="sm", c="dimmed", ta="center", mt="md"
                                )
                            ]
                        ),


                        dmc.TabsPanel(
                            value="events",
                            children=[
                                dmc.Grid(
                                    children=[
                                        dmc.GridCol(
                                            span=6,
                                            children=[
                                                dmc.Select(
                                                    id="drilldown-event",
                                                    label="Select Macroeconomic Event",
                                                    data=EVENT_OPTIONS,
                                                    value=EVENTS[6]["date"],
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                dmc.Space(h="md"),
                                dmc.Grid(
                                    children=[
                                        dmc.GridCol(span=6, children=[
                                            dmc.Paper(withBorder=True, shadow="md", p="md", radius="lg", children=[
                                                dcc.Graph(
                                                    id="drilldown-bar",  style={"height": "420px"})
                                            ])
                                        ]),
                                        dmc.GridCol(span=6, children=[
                                            dmc.Paper(withBorder=True, shadow="md", p="md", radius="lg", children=[
                                                dcc.Graph(
                                                    id="drilldown-line", style={"height": "420px"})
                                            ])
                                        ]),
                                    ]
                                ),
                                dmc.Text(
                                    "The bar chart shows 22-day cumulative log returns after the event. The line chart shows normalized prices 22 trading days before and after (event = day 0, red line), enabling direct comparison of drawdown speed vs. recovery speed.",
                                    size="sm", c="dimmed", ta="center", mt="md"
                                )
                            ]
                        ),


                        # -- Tab 4: Model Performance --------------------------
                        dmc.TabsPanel(
                            value="performance",
                            children=[
                                dmc.Grid(
                                    children=[
                                        dmc.GridCol(span=12, children=[
                                            dmc.Select(
                                                id="forecast-sector",
                                                label="Select Sector for Backtest",
                                                data=SECTORS,
                                                value="XLK",
                                                style={
                                                    "maxWidth": "400px"}
                                            )
                                        ])
                                    ]
                                ),
                                dmc.Space(h="md"),
                                dmc.Paper(withBorder=True, shadow="md", p="md", radius="lg", children=[
                                    dcc.Graph(id="forecast-chart",
                                              style={"height": "700px"})
                                ]),
                                dmc.Text(
                                    "This backtest evaluates model accuracy on the 2024--2026 validation period. Panel 1 compares ARIMA and LSTM return predictions against actual results, while Panel 2 compares GARCH(1,1) volatility with actual 30-day rolling vol.",
                                    size="sm", c="dimmed", ta="center", mt="md"
                                )
                            ]
                        ),

                        # -- Tab 5: Rolling Correlation ------------------------
                        dmc.TabsPanel(
                            value="corr",
                            children=[
                                dmc.Stack(
                                    children=[
                                        dmc.Text(
                                            "Rolling Window vs Fed Funds", size="sm", fw=500, mb=5),
                                        dmc.SegmentedControl(
                                            id="corr-window",
                                            data=[
                                                {"label": "3 months (63 days)",
                                                 "value": "63"},
                                                {"label": "6 months (126 days)",
                                                 "value": "126"},
                                                {"label": "12 months (252 days)",
                                                 "value": "252"},
                                            ],
                                            value="126",
                                            fullWidth=True,
                                            color="indigo",
                                        ),
                                        dmc.Paper(withBorder=True, shadow="md", p="md", radius="lg", children=[
                                            dcc.Graph(id="corr-chart",
                                                      style={"height": "500px"})
                                        ]),
                                        dmc.Text(
                                            "This chart displays the rolling Pearson correlation between sector returns and the Federal Funds rate. Each point on the line represents the relationship over the preceding window (3, 6, or 12 months), illustrating how sector sensitivity to monetary policy shifts over the 2005--2026 period.",
                                            size="sm", c="dimmed", ta="center", mt="md"
                                        )
                                    ]
                                )
                            ]
                        ),

                        # -- Tab 6: Sector Relationships ----------------------
                        dmc.TabsPanel(
                            value="relationships",
                            children=[
                                dmc.Stack(
                                    children=[
                                        dmc.Grid(
                                            children=[
                                                dmc.GridCol(
                                                    span=3,
                                                    children=[
                                                        dmc.Select(
                                                            id="heatmap-start-year",
                                                            label="Start Year",
                                                            data=[{"label": str(y), "value": str(y)} for y in range(
                                                                df.index.min().year, df.index.max().year + 1)],
                                                            value=str(
                                                                df.index.max().year - 1),
                                                        )
                                                    ]
                                                ),
                                                dmc.GridCol(
                                                    span=3,
                                                    children=[
                                                        dmc.Select(
                                                            id="heatmap-start-month",
                                                            label="Start Month",
                                                            data=[{"label": pd.Timestamp(2000, m, 1).strftime(
                                                                "%b"), "value": str(m)} for m in range(1, 13)],
                                                            value="1",
                                                        )
                                                    ]
                                                ),
                                                dmc.GridCol(
                                                    span=3,
                                                    children=[
                                                        dmc.Select(
                                                            id="heatmap-end-year",
                                                            label="End Year",
                                                            data=[{"label": str(y), "value": str(y)} for y in range(
                                                                df.index.min().year, df.index.max().year + 1)],
                                                            value=str(
                                                                df.index.max().year),
                                                        )
                                                    ]
                                                ),
                                                dmc.GridCol(
                                                    span=3,
                                                    children=[
                                                        dmc.Select(
                                                            id="heatmap-end-month",
                                                            label="End Month",
                                                            data=[{"label": pd.Timestamp(2000, m, 1).strftime(
                                                                "%b"), "value": str(m)} for m in range(1, 13)],
                                                            value=str(
                                                                df.index.max().month),
                                                        )
                                                    ]
                                                ),
                                                dmc.GridCol(
                                                    span=4,
                                                    children=[
                                                        dmc.Text(
                                                            "Market Regime Filter", size="sm", fw=500, mb=5),
                                                        dmc.SegmentedControl(
                                                            id="heatmap-regime",
                                                            data=[
                                                                {"label": "All",
                                                                 "value": "all"},
                                                                {"label": "Stable",
                                                                 "value": "stable"},
                                                                {"label": "Shocks",
                                                                 "value": "shock"},
                                                            ],
                                                            value="all",
                                                            fullWidth=True,
                                                            color="indigo",
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                        dmc.Paper(withBorder=True, shadow="md", p="md", radius="lg", children=[
                                            dcc.Graph(
                                                id="sector-heatmap", style={"height": "550px"})
                                        ]),
                                        dmc.Text(
                                            "This heatmap shows the Pearson correlation coefficient between sector log returns. *Note: Average values exclude each sector's correlation with itself. Use the Regime Filter to see how diversification benefits vanish during high-volatility periods.",
                                            size="sm", c="dimmed", ta="center"
                                        )
                                    ]
                                )
                            ]
                        ),

                        # -- Tab 7: Future Outlook -----------------------------
                        dmc.TabsPanel(
                            value="future",
                            children=[
                                dmc.Grid(
                                    children=[
                                        dmc.GridCol(span=4, children=[
                                            dmc.Select(
                                                id="future-sector",
                                                label="Select Sector",
                                                data=SECTORS,
                                                value="XLK",
                                            )
                                        ]),
                                        dmc.GridCol(span=8, children=[
                                            dmc.Text(
                                                "Forecast Horizon (Trading Days)", size="sm", fw=500, mb=5),
                                            dmc.Slider(
                                                id="future-horizon",
                                                min=20, max=126, step=1, value=63,
                                                marks=[
                                                    {"value": 20,
                                                     "label": "1 mo"},
                                                    {"value": 63,
                                                     "label": "3 mo"},
                                                    {"value": 126,
                                                     "label": "6 mo"},
                                                ],
                                                mb="xl"
                                            )
                                        ])
                                    ]
                                ),
                                dmc.Space(h="md"),
                                dmc.Paper(withBorder=True, shadow="md", p="md", radius="lg", children=[
                                    dcc.Graph(id="future-chart",
                                              style={"height": "600px"})
                                ]),
                                dmc.Text(
                                    "Future projections combine ARIMA (for the mean price trend) and GARCH (for the 95% confidence intervals based on forecasted volatility) beyond February 2, 2026.",
                                    size="sm", c="dimmed", ta="center", mt="md"
                                )
                            ]
                        ),
                    ]
                )
            ]
        )
    ]
)


# -- Callbacks -----------------------------------------------------------------

@app.callback(
    Output("overlay-chart", "figure"),
    Input("overlay-sectors",  "value"),
    Input("overlay-category", "value"),
)
def update_overlay(sectors, category):
    if not sectors:
        return go.Figure()

    # Normalize all prices to 100 on the first date so sectors are comparable
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
    # Convert the string date from the dropdown back into a Timestamp
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

    # Line: price indexed to event date = 100, showing 22 days before and after
    line_fig = go.Figure()
    pre  = df.loc[:event_date].tail(23).iloc[:-1]   # 22 trading days before
    post_prices = df.loc[event_date:].head(22+22)       # event day + 21 days after
    window_prices = pd.concat([pre, post_prices])
    base = df.loc[event_date, f"{SECTORS[0]}_close"] # anchor = event day

    for s in SECTORS:
        base_s = df.loc[event_date, f"{s}_close"]
        normed = window_prices[f"{s}_close"] / base_s * 100
        x_vals = list(range(-len(pre), len(post_prices)))
        line_fig.add_trace(go.Scatter(
            x=x_vals, y=normed,
            name=s, line=dict(color=SECTOR_COLORS[s], width=2),
        ))
    line_fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
    line_fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.6)
    line_fig.add_annotation(x=0, y=1.05, yref="paper", text="Event", showarrow=False,
                            font=dict(color="red", size=10))
    line_fig.update_layout(
        title=f"Sector Prices --- 22 Days Before & After Event<br><sub>{label}</sub>",
        xaxis_title="Trading Days Relative to Event",
        yaxis_title="Normalized Price (Event = 100)",
        legend_title="", margin=dict(t=80, b=80, l=110, r=40),
        xaxis=dict(automargin=True, fixedrange=True),
        yaxis=dict(automargin=True, fixedrange=True)
    )
    return bar_fig, line_fig


@app.callback(
    Output("forecast-chart", "figure"),
    Input("forecast-sector", "value"),
)
def update_forecast(sector):
    # Filter predictions to the selected sector and sort by date
    sec_preds = preds_df[preds_df["sector"] == sector].copy()
    sec_preds = sec_preds.sort_values("date")

    # Identify "shock" days --- defined as top 10% of rolling volatility in the test period
    full_series = df[f"{sector}_log_ret"].fillna(0)
    actual_vol = (full_series.rolling(30).std() * np.sqrt(252))
    plot_vol = actual_vol.loc[sec_preds["date"]]
    threshold = plot_vol.quantile(0.90)
    mask = plot_vol > threshold

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
        subplot_titles=(
            f"{sector} Returns --- Model Performance (ARIMA vs LSTM)",
            f"{sector} Volatility --- Model Performance (GARCH vs Actual)",
        ),
    )

    # -- Row 1: Returns --------------------------------------------------------
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

    # -- Row 2: Volatility -----------------------------------------------------
    fig.add_trace(go.Scatter(
        x=plot_vol.index, y=plot_vol.values,
        name="Actual Vol (30-day)", line=dict(color="black", width=1, dash="dot"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=sec_preds["date"], y=sec_preds["garch_vol"],
        name="GARCH(1,1)", line=dict(color="#ff7f0e", width=1.5),
    ), row=2, col=1)

    # Shade shock periods in red on both subplots
    # We loop through the mask to find the start and end of each shock window
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
    # Close out any shock period that extends to the end of the data
    if in_shock:
        last_str = mask.index[-1].strftime("%Y-%m-%d")
        for row in (1, 2):
            fig.add_vrect(
                x0=shock_start, x1=last_str,
                fillcolor="red", opacity=0.15, layer="above",
                line_width=0, row=row, col=1,
            )

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="rgba(255,0,0,0.3)", symbol="square"),
        name="Shock Regime (top 10% vol)",
    ), row=1, col=1)

    fig.update_layout(
        legend_title="", hovermode="x unified",
        margin=dict(t=80, b=40, l=50, r=50), height=700,
        xaxis=dict(range=[sec_preds["date"].min(),
                   sec_preds["date"].max()], automargin=True),
        yaxis=dict(automargin=True),
        yaxis2=dict(automargin=True)
    )
    return fig


@app.callback(
    Output("corr-chart", "figure"),
    Input("corr-window", "value"),
)
def update_corr(window):
    # Convert the window from string (from SegmentedControl) to int
    window = int(window)
    fig = go.Figure()
    for s in SECTORS:
        # Calculate rolling Pearson correlation between sector returns and Fed Funds Rate
        corr = df[f"{s}_log_ret"].rolling(window).corr(df["FEDFUNDS"])
        # Fill NaNs with 0 (NaNs occur when FEDFUNDS is constant, i.e., zero variance)
        corr = corr.fillna(0)
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
    Output("sector-heatmap", "figure"),
    Input("heatmap-start-year",  "value"),
    Input("heatmap-start-month", "value"),
    Input("heatmap-end-year",    "value"),
    Input("heatmap-end-month",   "value"),
    Input("heatmap-regime", "value"),
)
def update_heatmap(start_year, start_month, end_year, end_month, regime):
    # Build the date range from the four dropdowns
    # MonthEnd(0) snaps the end date to the last day of the selected month
    if start_year and start_month and end_year and end_month:
        start_date = f"{start_year}-{int(start_month):02d}-01"
        end_date = (pd.Timestamp(int(end_year), int(end_month), 1) +
                    pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        mask_dates = (df.index >= start_date) & (df.index <= end_date)
        period_df = df.loc[mask_dates]
    else:
        period_df = df

    if len(period_df) < 5:
        return go.Figure().update_layout(title="Not enough data for this period")

    # Define Shock vs Stable across THIS period
    vols = period_df[[f"{s}_log_ret" for s in SECTORS]
                     ].rolling(30).std() * np.sqrt(252)
    avg_vol = vols.mean(axis=1)
    threshold = avg_vol.quantile(0.90)
    is_shock = avg_vol > threshold

    # Filter data by regime
    if regime == "stable":
        target_df = period_df[~is_shock.fillna(False)]
    elif regime == "shock":
        target_df = period_df[is_shock.fillna(False)]
    else:
        target_df = period_df

    if len(target_df) < 5:
        return go.Figure().update_layout(title="No data found for this regime in this period")

    # Calculate Correlation
    recent_df = target_df[[f"{s}_log_ret" for s in SECTORS]]
    recent_df.columns = SECTORS
    corr_matrix = recent_df.corr()

    # Calculate average pairwise correlation (exclude diagonal 1.0s)
    avg_corr = (corr_matrix.values.sum() - len(SECTORS)) / \
        (len(SECTORS)**2 - len(SECTORS))

    # Calculate average correlation PER SECTOR (exclude self-correlation)
    sector_avgs = (corr_matrix.sum(axis=1) - 1) / (len(SECTORS) - 1)
    new_labels = [f"{s}<br>(Avg: {sector_avgs[s]:.2f})" for s in SECTORS]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=new_labels,
        y=new_labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        hoverinfo="z",
    ))

    regime_label = {"all": "Full Period",
                    "stable": "Stable Regimes", "shock": "Shock Regimes"}[regime]
    s_str = str(start_date)[:10]
    e_str = str(end_date)[:10]
    r_str = str(regime)

    fig.update_layout(
        title=f"Sector Correlation: {regime_label}<br><sub>Avg Correlation: {avg_corr:.3f} | Period: {s_str} to {e_str} | Market Regime: {r_str}",
        xaxis_title="Sector",
        yaxis_title="Sector",
        margin=dict(t=100, b=80, l=110, r=80),
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True)
    )
    return fig


@app.callback(
    Output("future-chart", "figure"),
    Input("future-sector", "value"),
    Input("future-horizon", "value"),
)
def update_future(sector, horizon):
    full_series = df[f"{sector}_log_ret"].fillna(0)
    last_price = df[f"{sector}_close"].iloc[-1]
    last_date = df.index[-1]

    # Fit ARIMA on the full dataset (training + test) for the forward projection
    arima_res = ARIMA(full_series, order=(1, 0, 1)).fit()
    garch_model_obj = arch_model(
        full_series, vol="Garch", p=1, q=1, dist="Normal")
    garch_res = garch_model_obj.fit(disp="off")

    # ARIMA gives us the expected return path, GARCH gives us the variance
    arima_forecast = arima_res.forecast(steps=horizon)
    garch_forecast = garch_res.forecast(horizon=horizon)
    future_var = garch_forecast.variance.values[-1, :]

    # Generate business-day dates for the forecast window
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon, freq="B")

    # Convert cumulative log returns back to price level
    cum_log_ret = np.cumsum(arima_forecast.values)
    price_pred = last_price * np.exp(cum_log_ret)

    # Build 95% confidence interval using cumulative GARCH variance
    cum_var = np.cumsum(future_var)
    cum_std = np.sqrt(cum_var)
    upper_bound = last_price * np.exp(cum_log_ret + 1.96 * cum_std)
    lower_bound = last_price * np.exp(cum_log_ret - 1.96 * cum_std)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df[f"{sector}_close"], name="Historical Price", line=dict(color="black", width=1.5)))
    fig.add_trace(go.Scatter(x=future_dates, y=price_pred,
                  name="ARIMA Forecast", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill="toself", fillcolor="rgba(31, 119, 180, 0.15)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="95% CI (GARCH)"
    ))

    fig.add_vline(x=last_date, line_dash="dash", line_color="gray")
    fig.add_annotation(x=last_date, y=1.05, yref="paper",
                       text="End of Data (Feb 2, 2026)", showarrow=False)

    start_view = last_date - pd.DateOffset(years=2)
    fig.update_layout(
        title=f"{sector} Future Price Projection ({horizon} Trading Days)",
        xaxis_title="Date", yaxis_title="ETF Price ($)",
        hovermode="x unified", template="plotly_white",
        margin=dict(r=30),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        xaxis=dict(
            range=[start_view, future_dates[-1]],
            rangeslider=dict(visible=True, range=[
                             df.index[0], future_dates[-1]]),
            type="date", autorange=False,
            automargin=True
        ),
        yaxis=dict(automargin=True)
    )
    return fig


@app.callback(
    Output("vol-chart", "figure"),
    Input("vol-sectors",  "value"),
    Input("vol-category", "value"),
)
def update_vol(sectors, category):
    if not sectors:
        return go.Figure()

    fig = go.Figure()
    for s in sectors:
        # Annualize the 30-day rolling std by multiplying by sqrt(252 trading days)
        vol = df[f"{s}_log_ret"].rolling(30).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=vol.index, y=vol,
            name=s, line=dict(color=SECTOR_COLORS[s], width=1.5),
        ))

    # Overlay event lines, filtered by the selected category
    filtered = events_df if category == "All" else events_df[events_df["category"] == category]
    fig = get_event_vlines(fig, filtered)

    # Add dummy traces so event categories appear in the legend
    for cat, col in CATEGORY_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=col, dash="dash", width=1.5),
            name=f"{cat} Event", showlegend=True,
        ))

    fig.update_layout(
        title="30-Day Annualized Rolling Volatility",
        xaxis_title="Date", yaxis_title="Annualized Volatility",
        legend_title="", hovermode="x unified",
        margin=dict(t=80, r=20),
        yaxis=dict(tickformat=".0%"),
    )
    return fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
