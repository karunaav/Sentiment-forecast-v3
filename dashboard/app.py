import os
import time
import requests
import numpy as np
import pandas as pd
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go

# Try importing both finance libraries
try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from yahooquery import Ticker
except ImportError:
    Ticker = None

# ======================================================
# ✅ CONFIGURATION
# ======================================================
APP_TITLE = os.getenv("APP_TITLE", "Sentiment Forecasting Dashboard")

API_BASE = os.getenv("API_BASE", "https://sentiment-forecast-v3.onrender.com").strip()
if not API_BASE.startswith("http"):
    API_BASE = "https://sentiment-forecast-v3.onrender.com"

FAANG = ["AAPL", "AMZN", "META", "GOOGL", "NFLX"]

# ======================================================
# ✅ HELPER FUNCTIONS
# ======================================================
def safe_get(url, retries=3, delay=5, timeout=60):
    """Try API multiple times with delay and timeout."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp
        except requests.exceptions.RequestException:
            if attempt < retries:
                time.sleep(delay)
    raise Exception(f"API not responding after {retries} tries")


def get_stock_data(ticker: str, period: str = "6mo"):
    """Fetch stock price data with multiple fallbacks."""
    df = None
    # 1️⃣ Try yfinance
    if yf is not None:
        try:
            df = yf.download(ticker, period=period, progress=False)
        except Exception:
            df = None

    # 2️⃣ Try yahooquery fallback
    if (df is None or df.empty) and Ticker is not None:
        try:
            tq = Ticker(ticker)
            hist = tq.history(period=period)
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                if "symbol" in hist.columns:
                    hist = hist.reset_index()
                    hist.rename(columns={"symbol": "Ticker"}, inplace=True)
                    hist.set_index("date", inplace=True)
                df = hist[["open", "high", "low", "close"]]
                df.columns = ["Open", "High", "Low", "Close"]
        except Exception:
            df = None

    # 3️⃣ Synthetic fallback
    if df is None or df.empty:
        print(f"⚠️ Yahoo Finance error for {ticker}: Empty dataframe returned — using synthetic fallback")
        dates = pd.date_range(end=pd.Timestamp.today(), periods=120)
        base = np.linspace(100, 120, len(dates))
        noise = np.random.normal(0, 1, len(dates))
        df = pd.DataFrame({
            "Open": base + noise,
            "High": base + np.abs(noise),
            "Low": base - np.abs(noise),
            "Close": base + noise / 2
        }, index=dates)
    return df


# ======================================================
# ✅ DASH SETUP
# ======================================================
app = dash.Dash(__name__, requests_pathname_prefix="/dashboard/", title=APP_TITLE)
server = app.server

app.layout = html.Div(
    className="glass-container",
    children=[
        html.Div(
            className="header",
            children=[
                html.H1(APP_TITLE, className="title"),
                html.Div(id="api-status", className="status-badge"),
            ],
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
        ),
        html.Div(
            className="controls",
            children=[
                dcc.Dropdown(
                    id="ticker-dropdown",
                    options=[{"label": t, "value": t} for t in FAANG],
                    value="AAPL",
                    clearable=False,
                    className="dropdown",
                ),
                html.Button("Run Analysis", id="analyze-btn", n_clicks=0, className="button"),
            ],
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-pred",
            children=[
                dcc.Tab(label="📈 Prediction", value="tab-pred"),
                dcc.Tab(label="📊 Backtesting", value="tab-back"),
                dcc.Tab(label="🔍 Explainability", value="tab-shap"),
            ],
            className="tabs",
        ),
        html.Div(id="tab-content"),
        dcc.Interval(id="ping-api", interval=60 * 1000, n_intervals=0),
    ],
)

# ======================================================
# ✅ API STATUS INDICATOR
# ======================================================
@app.callback(Output("api-status", "children"), Input("ping-api", "n_intervals"))
def update_status(_):
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        if r.status_code == 200:
            return html.Span("🟢 API Connected", style={"color": "#00e676", "fontWeight": "bold"})
        else:
            return html.Span("🟠 API Slow", style={"color": "#ffb74d", "fontWeight": "bold"})
    except Exception:
        return html.Span("🔴 API Offline", style={"color": "#ef5350", "fontWeight": "bold"})


# ======================================================
# ✅ MAIN CALLBACK LOGIC
# ======================================================
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("analyze-btn", "n_clicks"),
    State("ticker-dropdown", "value"),
)
def update_tabs(tab, n_clicks, ticker):
    if not ticker:
        return html.Div("⚠️ Please select a ticker.")

    # --- Fetch prediction
    try:
        pred = safe_get(f"{API_BASE}/predict/{ticker}").json()
    except Exception as e:
        return html.Div(f"⚠️ Error fetching prediction: {e}")

    # --- Fetch backtest
    try:
        back = safe_get(f"{API_BASE}/backtest/{ticker}").json()
    except Exception:
        back = {}

    # ============ Prediction Tab ============
    if tab == "tab-pred":
        df = get_stock_data(ticker)
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker,
            )
        )
        fig.update_layout(
            template="plotly_dark",
            title=f"{ticker} — Last 6 Months",
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return html.Div(
            [
                html.H4(f"{ticker}: Predicted next-day return = {pred.get('predicted_return',0):.4%}"),
                dcc.Graph(figure=fig),
            ]
        )

    # ============ Backtesting Tab ============
    elif tab == "tab-back":
        if "error" in back:
            return html.Div(f"⚠️ Backtest error: {back['error']}")
        return html.Div(
            [
                html.H3(f"{ticker} — Backtesting Metrics", className="section-title"),
                html.Div(
                    [
                        html.P(f"📅 Samples: {back.get('samples', 0)}"),
                        html.P(f"💡 MSE: {back.get('mse', 0):.6f}"),
                        html.P(f"🚀 CAGR: {back.get('cagr', 0):.2%}"),
                        html.P(f"⚖️ Sharpe: {back.get('sharpe', 0):.2f}"),
                        html.P(f"📉 Max Drawdown: {back.get('max_drawdown', 0):.2%}"),
                    ],
                    style={"textAlign": "center", "fontSize": "1.1rem"},
                ),
            ]
        )

    # ============ Explainability Tab ============
    elif tab == "tab-shap":
        fi = pred.get("feature_importance", {})
        if not fi or "error" in fi:
            return html.Div("⚠️ SHAP explanations not available for this model/environment.")
        features, vals = list(fi.keys()), list(fi.values())
        fig = go.Figure(go.Bar(x=features, y=vals))
        fig.update_layout(template="plotly_dark", title=f"{ticker} — Feature Importance")
        return html.Div([dcc.Graph(figure=fig)])

    return html.Div("⚠️ Invalid tab selection.")


# ======================================================
# ✅ STYLING (Glassmorphism)
# ======================================================
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #fff;
            font-family: 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .glass-container {
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            background-color: rgba(17, 25, 40, 0.55);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.125);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            padding: 30px;
            width: 85%;
            max-width: 1100px;
        }
        .title { font-size: 2rem; font-weight: 700; color: #e0e0e0; }
        .status-badge { font-size: 1rem; margin-right: 15px; }
        .controls { display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 20px; }
        .dropdown { width: 200px; color: #000; }
        .button { background-color: #00bfa6; color: #fff; border: none; border-radius: 8px; padding: 10px 20px; font-size: 1rem; cursor: pointer; transition: all 0.3s ease; }
        .button:hover { background-color: #1de9b6; transform: scale(1.05); }
        .tabs { margin-top: 20px; }
        .section-title { text-align: center; margin-bottom: 15px; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050)
