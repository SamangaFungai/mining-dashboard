# Commodity price and mine economics dashboard

An interactive Streamlit dashboard that links commodity price risk to mine economics. Pick gold, platinum or
copper, forecast the price, and see how price uncertainty flows through to NPV, IRR and the probability that a
hypothetical mine loses money.

**Live demo:** _https://mining-dashboard09.streamlit.app_

## What it does

- Downloads daily futures prices (gold `GC=F`, platinum `PL=F`, copper `HG=F`) from Yahoo Finance and stores them in SQLite.
- Forecasts 12 months ahead with ARIMA on log prices, with an 80% interval.
- Models a simple mine: production, operating cost, royalty, tax, straight-line depreciation, capex, discounting.
- Reports NPV, IRR, payback, all-in sustaining cost (AISC) and the breakeven price.
- Runs a Monte Carlo simulation (random-walk prices, volatility from history) to give an NPV distribution
  and P(NPV < 0).
- Shows a sensitivity chart for price, grade, recovery, operating cost and capex.

## Run it locally

```
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional: `python -m src.data` pre-downloads all prices, `python -m src.forecast Gold` runs the forecast backtest,
and `python -m src.economics` prints the base-case economics in the terminal.

## Project structure

```
app.py                  Streamlit dashboard
src/commodities.py      Tickers, units and illustrative mine defaults per commodity
src/data.py             Download and SQLite storage
src/forecast.py         Baselines, ARIMA and rolling-origin backtest
src/economics.py        Cash flows, NPV, IRR, AISC, breakeven, Monte Carlo, sensitivity
```

## Key finding: the forecast barely beats a naive baseline

Rolling-origin backtest on monthly gold prices (6-month horizon, lower is better):

| Model | MAPE (%) | RMSE |
|---|---|---|
| Drift | 6.19 | 157.90 |
| ARIMA(1,1,1) | 6.45 | 165.38 |
| Naive (last value) | 6.47 | 166.63 |

Differences are within noise. Commodity price levels behave close to a random walk, so the dashboard treats the
forecast interval as the useful output and feeds price uncertainty, not a point forecast, into the economics.

## Assumptions and limits

- Mine inputs are illustrative placeholders, not a real project. Royalty and tax rates vary by jurisdiction.
- Annual model with a constant throughput and grade, straight-line depreciation, no working capital or inflation.
- Monte Carlo prices follow a zero-drift random walk and ignore mean reversion and price/cost correlation.
- Copper is priced in USD/lb and grade in % Cu; gold and platinum in USD/oz and g/t.
- Not investment advice.

## Deploy

Push to GitHub, then create a new app at share.streamlit.io pointing at `app.py`. Prices are downloaded
on startup, so no database file needs to be committed.
