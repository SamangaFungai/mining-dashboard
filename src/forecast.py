"""Forecasting layer: baselines vs ARIMA with a rolling-origin backtest.

Run from the project root:  python -m src.forecast
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.data import load_monthly

warnings.filterwarnings("ignore")


def forecast_naive(train: pd.Series, h: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], h)


def forecast_drift(train: pd.Series, h: int) -> np.ndarray:
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    return train.iloc[-1] + slope * np.arange(1, h + 1)


def forecast_arima(train: pd.Series, h: int, order=(1, 1, 1), alpha=0.2):
    """ARIMA on log prices. Returns (mean, lower, upper) in USD/oz."""
    model = ARIMA(np.log(train.values), order=order).fit()
    res = model.get_forecast(h)
    mean = np.exp(res.predicted_mean)
    ci = np.exp(res.conf_int(alpha=alpha))
    return mean, ci[:, 0], ci[:, 1]


MODELS = {
    "naive": forecast_naive,
    "drift": forecast_drift,
    "arima": lambda tr, h: forecast_arima(tr, h)[0],
}


def rolling_backtest(series: pd.Series, horizon: int = 6, min_train: int = 120, step: int = 3):
    """Expanding-window evaluation: fit on the past, predict the next `horizon` months."""
    rows = []
    for end in range(min_train, len(series) - horizon + 1, step):
        train, test = series.iloc[:end], series.iloc[end : end + horizon]
        for name, fn in MODELS.items():
            pred = fn(train, horizon)
            err = pred - test.values
            rows.append(
                {
                    "origin": train.index[-1],
                    "model": name,
                    "mape": np.mean(np.abs(err / test.values)) * 100,
                    "rmse": np.sqrt(np.mean(err**2)),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "Gold"
    print(f"Commodity: {name}")
    monthly = load_monthly(name)
    print(f"{len(monthly)} monthly observations, last: {monthly.index[-1].date()}")

    results = rolling_backtest(monthly)
    summary = results.groupby("model")[["mape", "rmse"]].mean().round(2).sort_values("mape")
    print("\nBacktest (6-month horizon, lower is better):")
    print(summary)

    mean, lo, hi = forecast_arima(monthly, 12)
    future = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS")
    out = pd.DataFrame({"forecast": mean, "lower_80": lo, "upper_80": hi}, index=future).round(0)
    print("\n12-month ARIMA forecast (USD/oz):")
    print(out)
