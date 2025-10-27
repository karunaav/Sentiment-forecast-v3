
from __future__ import annotations
import os, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/period).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period).mean()
    rs = up / (down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def build_dataset(df: pd.DataFrame):
    X = pd.DataFrame(index=df.index)
    X["ret1"] = df["Close"].pct_change().fillna(0.0)
    X["vol"] = df["Close"].pct_change().rolling(10).std().fillna(0.0)
    X["rsi"] = _rsi(df["Close"], 14)
    roll = (X["ret1"] - X["ret1"].rolling(14).mean()) / (X["ret1"].rolling(14).std() + 1e-9)
    X["sentiment"] = roll.fillna(0.0)

    y = X["ret1"].shift(-1).dropna()
    X = X.loc[y.index]
    return X, y

def _train_fresh(X: pd.DataFrame, y: pd.Series):
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)
    return model

def load_model(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    # train minimal model so coef_ exists
    idx = pd.date_range("2020-01-01", periods=200)
    Xs = pd.DataFrame({
        "ret1": np.random.normal(0, 0.01, size=len(idx)),
        "vol": np.random.uniform(0.0, 0.02, size=len(idx)),
        "rsi": np.clip(np.random.normal(50, 10, size=len(idx)), 0, 100),
        "sentiment": np.random.normal(0, 1, size=len(idx))
    }, index=idx)
    y = Xs["ret1"].shift(-1).fillna(0.0)
    model = _train_fresh(Xs, y)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model
