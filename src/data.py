import hashlib
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from .config import DATA_DIR, DEFAULT_START


def _cache_key(tickers: list[str], start: str, end: str) -> str:
    raw = f"{'_'.join(sorted(tickers))}_{start}_{end}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _parquet_path(tickers: list[str], start: str, end: str) -> Path:
    return DATA_DIR / f"prices_{_cache_key(tickers, start, end)}.parquet"


def get_price_data(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    cache_path = _parquet_path(tickers, start, end or "latest")
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))
    prices.to_parquet(cache_path)
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def get_return_data(
    tickers: tuple[str, ...],
    start: str = DEFAULT_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    prices = get_price_data(list(tickers), start, end)
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns
