import os
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import yfinance as yf
import requests

# ===========================
# ✅ App Config
# ===========================
APP_TITLE = os.getenv("APP_TITLE", "Sentiment Forecasting Dashboard")

# Use Render API if available, otherwise local
API_BASE = os.getenv("API_BASE", "https://sentiment-forecast-v3.onrender.com")

FAANG = ["AAPL", "AMZN", "META", "GOOGL", "NFLX"]

app = dash.Dash(__name__, requests_pathname_prefix="/dashboard/", title=APP_TITLE)
server = app.server

# ===========================
# ✅ Layout
# ===========================
app.layout = html.Div(
    className="glass-container",
    children=[
        html.H1(APP_TITLE, className="title"),
        html.Div(className="controls", children=[
            dcc.Dropdown(
                id="ticker-dropdown",
                options=[{"label": t, "value": t} for t in FAANG],
                value="AAPL",
                clearable=False,
                className="dropdown"
            ),
            html.Button("Run Analysis", id="analyze-btn", n_clicks=0, className="button"),
        ]),
        dcc.Tabs(
            id="tabs",
            value="tab-pred",
            children=[
                dcc.Tab(label="📈 Prediction", value="tab-pred", className="tab"),
                dcc.Tab(label="📊 Backtesting", value="tab-back", className="tab"),
                dcc.Tab(label="🔍 Explainability", value="tab-shap", className="tab"),
            ],
            className="tabs"
        ),
        html.Div(id="tab-content"),
    ]
)

# ===========================
# ✅ Callbacks
# ===========================
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("analyze-btn", "n_clicks"),
    State("ticker-dropdown", "value"),
)
def update_tabs(tab, n_clicks, ticker):
    """Update each tab (Prediction, Backtest, SHAP) dynamically."""
    if not ticker:
        return html.Div("⚠️ Please select a ticker.")

    # --- Prediction Fetch ---
    try:
        pred_resp = requests.get(f"{API_BASE}/predict/{ticker}", timeout=15)
        pred_resp.raise_for_status()
        pred = pred_resp.json()
    except Exception as e:
        return html.Div(f"⚠️ Error fetching prediction: {e}")

    # --- Backtest Fetch ---
    try:
        back_resp = requests.get(f"{API_BASE}/backtest/{ticker}", timeout=15)
        back_resp.raise_for_status()
        back = back_resp.json()
    except Exception:
        back = {}

    # ===========================
    # 📈 Prediction Tab
    # ===========================
    if tab == "tab-pred":
        try:
            df = yf.download(ticker, period="6mo", progress=False)
        except Exception:
            df = None

        fig = go.Figure()
        if df is not None and not df.empty:
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name=ticker
            ))
        fig.update_layout(
            template="plotly_dark",
            title=f"{ticker} — Last 6 Months",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        pred_val = pred.get("predicted_return", 0)
        conf = pred.get("confidence", 0)
        return html.Div([
            html.H4(f"{ticker}: Predicted next-day return = {pred_val:.4%} | Confidence = {conf:.2f}"),
            dcc.Graph(figure=fig)
        ])

    # ===========================
    # 📊 Backtesting Tab
    # ===========================
    elif tab == "tab-back":
        if "error" in back:
            return html.Div(f"⚠️ Backtest error: {back['error']}")
        return html.Div([
            html.H3(f"{ticker} — Backtesting Metrics", className="section-title"),
            html.Div([
                html.P(f"📅 Samples: {back.get('samples', 0)}"),
                html.P(f"💡 MSE: {back.get('mse', 0):.6f}"),
                html.P(f"🚀 CAGR: {back.get('cagr', 0):.2%}"),
                html.P(f"⚖️ Sharpe: {back.get('sharpe', 0):.2f}"),
                html.P(f"📉 Max Drawdown: {back.get('max_drawdown', 0):.2%}"),
            ], style={"textAlign": "center", "fontSize": "1.1rem"})
        ])

    # ===========================
    # 🔍 SHAP / Explainability Tab
    # ===========================
    elif tab == "tab-shap":
        fi = pred.get("feature_importance", {})
        if not fi or "error" in fi:
            return html.Div("⚠️ SHAP explanations not available for this model/environment.")
        features = list(fi.keys())
        vals = list(fi.values())
        fig = go.Figure()
        fig.add_trace(go.Bar(x=features, y=vals))
        fig.update_layout(
            template="plotly_dark",
            title=f"{ticker} — Feature Importance",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return html.Div([dcc.Graph(figure=fig)])

    # Default fallback
    return html.Div("⚠️ Please select a valid tab.")

# ===========================
# ✅ Glass UI Template
# ===========================
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
        .title {
            text-align: center;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 20px;
            color: #e0e0e0;
        }
        .controls {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        .dropdown {
            width: 200px;
            color: #000;
        }
        .button {
            background-color: #00bfa6;
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .button:hover {
            background-color: #1de9b6;
            transform: scale(1.05);
        }
        .tabs {
            margin-top: 20px;
        }
        .section-title {
            text-align: center;
            margin-bottom: 15px;
        }
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
    app.run_server(debug=True)
