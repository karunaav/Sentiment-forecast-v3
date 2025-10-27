
import os, datetime as dt
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel

from src.data.prices import fetch_prices_df
from src.model.train import load_model, build_dataset
from src.backtest.engine import simple_backtest
from dashboard.app import app as dash_app

APP_TITLE = os.getenv("APP_TITLE", "Sentiment Forecast API + Dashboard")
MODEL_PATH = os.getenv("MODEL_PATH", "models/latest_model.pkl")

app = FastAPI(title=APP_TITLE, version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/dashboard", WSGIMiddleware(dash_app.server))

class ForecastOut(BaseModel):
    ticker: str
    predicted_return: float
    as_of: str
    shap_ok: bool = False
    feature_importance: Optional[dict] = None

@app.get("/")
def root():
    return {"message":"✅ API up","endpoints":["/predict/{ticker}","/backtest/{ticker}","/dashboard/"]}

@app.get("/predict/{ticker}", response_model=ForecastOut)
def predict(ticker: str):
    model = load_model(MODEL_PATH)
    df = fetch_prices_df(ticker)
    X, y = build_dataset(df)
    if len(X) == 0:
        return ForecastOut(ticker=ticker, predicted_return=0.0, as_of="no data", shap_ok=False, feature_importance={"error":"no data"})

    pred = float(model.predict(X.tail(1))[0])

    shap_ok, shap_vals = False, {}
    if os.getenv("ENABLE_SHAP", "true").lower() == "true":
        try:
            import shap, numpy as np
            background = X.tail(min(200, len(X)))
            explainer = shap.Explainer(model.predict, background)
            sv = explainer(X.tail(1))
            shap_vals = dict(zip(X.columns, np.abs(sv.values[0]).tolist()))
            shap_ok = True
        except Exception as e:
            try:
                import numpy as np
                if hasattr(model, "coef_"):
                    cof = np.abs(model.coef_)
                    shap_vals = dict(zip(X.columns, cof.tolist())); shap_ok = True
                elif hasattr(model, "feature_importances_"):
                    fi = np.abs(model.feature_importances_)
                    shap_vals = dict(zip(X.columns, fi.tolist())); shap_ok = True
                else:
                    shap_vals = {"error": f"SHAP failed and no fallback importances: {e}"}
            except Exception as e2:
                shap_vals = {"error": f"Importance computation failed: {e2}"}

    return ForecastOut(ticker=ticker, predicted_return=pred, as_of=dt.datetime.utcnow().isoformat(), shap_ok=shap_ok, feature_importance=shap_vals)

@app.get("/backtest/{ticker}")
def backtest(ticker: str):
    df = fetch_prices_df(ticker)
    X, y = build_dataset(df)
    return simple_backtest(X, y)
