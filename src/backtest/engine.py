
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

def simple_backtest(X: pd.DataFrame, y: pd.Series, splits: int = 5):
    if len(X) < splits + 1:
        return {"samples": int(len(X)), "error": "Not enough samples for backtest"}
    tscv = TimeSeriesSplit(n_splits=splits)
    preds, rets = [], []
    model = Ridge(alpha=1.0, random_state=42)

    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        preds.extend(p.tolist()); rets.extend(yte.tolist())

    preds = np.array(preds); rets = np.array(rets)
    strat = np.where(preds > 0, rets, 0.0)
    mse = float(mean_squared_error(rets, preds))
    cagr = _cagr(strat); sharpe = _sharpe(strat); mdd = _max_drawdown(strat)
    return {"samples": int(len(rets)), "mse": mse, "cagr": float(cagr), "sharpe": float(sharpe), "max_drawdown": float(mdd)}

def _equity_curve(returns: np.ndarray, start: float = 1.0):
    curve = [start]
    for r in returns: curve.append(curve[-1]*(1.0+r))
    return np.array(curve)

def _cagr(returns: np.ndarray, periods_per_year: int = 252):
    curve = _equity_curve(returns); years = max(len(returns)/periods_per_year, 1e-9)
    return (curve[-1]/curve[0])**(1/years) - 1

def _sharpe(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252):
    ex = returns - rf/periods_per_year; std = np.std(ex)
    return 0.0 if std < 1e-9 else np.sqrt(periods_per_year)*np.mean(ex)/std

def _max_drawdown(returns: np.ndarray):
    curve = _equity_curve(returns); peak = np.maximum.accumulate(curve)
    dd = (curve - peak)/peak
    return float(dd.min())
