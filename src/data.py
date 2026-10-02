"""Data layer: download commodity prices and keep a local SQLite copy.

Run from the project root:  python -m src.data
"""
import sqlite3
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.commodities import COMMODITIES

DB_PATH = Path("data/mining.db")


def _table(name: str) -> str:
    return f"{name.lower()}_prices"


def fetch_prices(name: str = "Gold", start: str = "2005-01-01") -> pd.DataFrame:
    df = yf.download(COMMODITIES[name]["ticker"], start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {name}. Check your internet connection.")
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns MultiIndex columns
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "price"}).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df


def save_prices(df: pd.DataFrame, name: str = "Gold", db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql(_table(name), con, if_exists="replace")


def load_prices(name: str = "Gold", db_path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(f"SELECT * FROM {_table(name)}", con, parse_dates=["date"], index_col="date")
    df.columns = ["price"]  # also handles the older gold_usd_oz column name
    return df


def get_prices(name: str = "Gold") -> pd.Series:
    """Fresh download if possible (saved to SQLite); otherwise the saved copy."""
    try:
        df = fetch_prices(name)
        save_prices(df, name)
    except Exception:
        df = load_prices(name)
    return df["price"]


def load_monthly(name: str = "Gold", db_path: Path = DB_PATH) -> pd.Series:
    """Monthly average price from the saved copy, used for forecasting."""
    return load_prices(name, db_path)["price"].resample("MS").mean().dropna()


if __name__ == "__main__":
    for commodity in COMMODITIES:
        prices = fetch_prices(commodity)
        save_prices(prices, commodity)
        print(f"{commodity}: saved {len(prices):,} rows "
              f"({prices.index.min().date()} to {prices.index.max().date()}), "
              f"last {prices['price'].iloc[-1]:,.2f}")
