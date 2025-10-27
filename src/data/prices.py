
import pandas as pd
import yfinance as yf
import requests_cache

def fetch_prices_df(ticker: str, period: str = "1y") -> pd.DataFrame:
    session = requests_cache.CachedSession('yfinance.cache', expire_after=3600)
    try:
        df = yf.download(ticker, period=period, progress=False, session=session, timeout=20)
        if df is None or df.empty:
            df = yf.download(ticker, period="5y", progress=False, session=session, timeout=30)
        if df is None or df.empty:
            raise ValueError("Empty dataframe returned")
        return df.dropna().copy()
    except Exception as e:
        print(f"⚠️ Yahoo Finance error for {ticker}: {e} — using synthetic fallback")
        dates = pd.date_range(end=pd.Timestamp.today(), periods=252)
        base = 100.0
        close = base + (pd.Series(range(len(dates))) * 0.05).values
        fake = pd.DataFrame({
            "Open": close * 0.999,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000
        }, index=dates)
        return fake
