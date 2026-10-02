"""Commodity price and mine economics dashboard.

Run from the project root (venv active):  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.commodities import COMMODITIES
from src.data import get_prices
from src.economics import (
    Mine, aisc, breakeven_price, cash_flows, irr, monte_carlo,
    npv, payback_years, sensitivity,
)
from src.forecast import forecast_arima

st.set_page_config(page_title="Mine economics dashboard", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading prices...")
def get_series(name: str) -> pd.Series:
    return get_prices(name)


@st.cache_data(ttl=6 * 3600)
def get_forecast(name: str):
    daily = get_series(name)
    monthly = daily.resample("MS").mean().dropna()
    if daily.index[-1].day < 25:  # drop the incomplete current month
        monthly = monthly.iloc[:-1]
    mean, lo, hi = forecast_arima(monthly, 12)
    idx = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(), periods=12, freq="MS")
    return monthly, pd.DataFrame({"mean": mean, "lo": lo, "hi": hi}, index=idx)


def money(x: float) -> str:
    return f"-${abs(x) / 1e6:,.0f}M" if x < 0 else f"${x / 1e6:,.0f}M"


# ---------------- Sidebar ----------------
name = st.sidebar.selectbox("Commodity", list(COMMODITIES))
cfg, k, unit = COMMODITIES[name], name.lower(), COMMODITIES[name]["price_unit"]
st.title(f"{name} price and mine economics dashboard")

try:
    daily = get_series(name)
except Exception as e:
    st.error(f"Could not load {name} prices: {e}")
    st.stop()
monthly, fc = get_forecast(name)
spot = float(daily.iloc[-1])
hist_vol = float(np.log(daily).diff().dropna().std() * np.sqrt(252))


def px(v: float) -> str:
    return f"${v:,.2f}/{unit}" if unit == "lb" else f"${v:,.0f}/{unit}"


st.sidebar.header("Mine assumptions")
life = st.sidebar.slider("Mine life (years)", 3, 25, 10, key=f"{k}_life")
tonnes = st.sidebar.slider("Ore milled (Mt/year)", 0.2, 20.0, cfg["tonnes_mt"], 0.1, key=f"{k}_t") * 1e6
g_lo, g_hi, g_def, g_step = cfg["grade_range"]
grade = st.sidebar.slider(cfg["grade_label"], g_lo, g_hi, g_def, g_step, key=f"{k}_g")
recovery = st.sidebar.slider("Recovery (%)", 50, 98, cfg["recovery"], key=f"{k}_r") / 100
opex = st.sidebar.slider("Operating cost (USD/t milled)", 10, 200, cfg["opex"], key=f"{k}_o")
capex = st.sidebar.slider("Initial capex (USD M)", 20, 600, cfg["capex_m"], key=f"{k}_c") * 1e6
sust = st.sidebar.slider("Sustaining capex (USD M/year)", 0, 50, cfg["sust_m"], key=f"{k}_s") * 1e6
royalty = st.sidebar.slider("Royalty (% of revenue)", 0.0, 15.0, cfg["royalty"], 0.5, key=f"{k}_ry") / 100
tax = st.sidebar.slider("Tax rate (%)", 0, 40, 25, key=f"{k}_tax") / 100
disc = st.sidebar.slider("Discount rate (%)", 4, 15, 8, key=f"{k}_d") / 100

st.sidebar.header("Price scenario")
p_lo, p_hi, p_step = cfg["price_range"]
p0 = st.sidebar.number_input(
    f"Starting price (USD/{unit})", min_value=p_lo, max_value=p_hi,
    value=float(min(max(round(spot, 2), p_lo), p_hi)), step=p_step,
    format="%.2f" if unit == "lb" else "%.0f", key=f"{k}_p0")
vol = st.sidebar.slider("Annual price volatility (%)", 5, 60,
                        int(min(max(round(hist_vol * 100), 5), 60)), key=f"{k}_v") / 100
n_paths = st.sidebar.select_slider("Monte Carlo paths", [1000, 5000, 10000, 20000], 10000)

mine = Mine(life_years=life, tonnes_per_year=tonnes, head_grade_gpt=grade, recovery=recovery,
            opex_per_tonne=opex, sustaining_capex_per_year=sust, initial_capex=capex,
            royalty=royalty, tax_rate=tax, discount_rate=disc, grade_to_units=cfg["grade_to_units"])
cf = cash_flows(mine, np.full(life, float(p0)))
sim_npv = monte_carlo(mine, float(p0), vol, n_paths)
irr_val = irr(cf)

# ---------------- Headline metrics ----------------
c = st.columns(5)
c[0].metric("NPV (constant price)", money(npv(cf, disc)))
c[1].metric("IRR", "n/a" if np.isnan(irr_val) else f"{irr_val:.0%}")
c[2].metric("AISC", px(aisc(mine, p0)))
c[3].metric("Breakeven price", px(breakeven_price(mine)))
c[4].metric("P(NPV < 0)", f"{(sim_npv < 0).mean():.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["Prices and forecast", "Cash flow", "Risk", "Sensitivity"])

with tab1:
    hist = monthly[monthly.index >= monthly.index[-1] - pd.DateOffset(years=10)]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, mode="lines", name="Monthly average", line=dict(color="#B8860B"))
    fig.add_scatter(x=fc.index, y=fc["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc.index, y=fc["lo"], mode="lines", fill="tonexty", line=dict(width=0),
                    fillcolor="rgba(184,134,11,0.2)", name="80% interval")
    fig.add_scatter(x=fc.index, y=fc["mean"], mode="lines", name="ARIMA forecast",
                    line=dict(dash="dash", color="#8B6508"))
    fig.update_layout(yaxis_title=f"USD per {unit}", height=450, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Point forecasts for commodity prices rarely beat 'next month equals this month'. "
               "Treat the widening uncertainty band as the useful output.")

with tab2:
    years = list(range(life + 1))
    fig = go.Figure()
    fig.add_bar(x=years, y=cf / 1e6, name="Annual cash flow", marker_color="#B8860B")
    fig.add_scatter(x=years, y=np.cumsum(cf) / 1e6, mode="lines", name="Cumulative", line=dict(color="#333"))
    fig.update_layout(xaxis_title="Year", yaxis_title="USD M", height=450, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Production: {mine.units_per_year:,.0f} {unit}/year. "
               f"Payback: {payback_years(cf):.1f} years at a constant {px(p0)}.")

with tab3:
    p10, p50, p90 = np.percentile(sim_npv, [10, 50, 90])
    fig = go.Figure(go.Histogram(x=sim_npv / 1e6, nbinsx=60, marker_color="#B8860B"))
    fig.add_vline(x=0, line_color="red", line_dash="dash")
    fig.update_layout(xaxis_title="NPV (USD M)", yaxis_title="Number of price paths",
                      height=450, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.write(f"**P10 / P50 / P90:** {money(p10)} / {money(p50)} / {money(p90)}. "
             f"**Mean:** {money(sim_npv.mean())}.")
    st.caption("Prices follow a random walk with no drift, using the volatility set in the sidebar.")

with tab4:
    sens = sensitivity(mine, float(p0), 0.20)
    labels = {"gold_price": "Price", "head_grade_gpt": "Head grade", "recovery": "Recovery",
              "opex_per_tonne": "Operating cost", "initial_capex": "Initial capex"}
    order = sorted(sens, key=lambda key: abs(sens[key][1] - sens[key][0]))
    fig = go.Figure()
    fig.add_bar(y=[labels[o] for o in order], x=[sens[o][0] / 1e6 for o in order],
                orientation="h", name="Input -20%", marker_color="#C0392B")
    fig.add_bar(y=[labels[o] for o in order], x=[sens[o][1] / 1e6 for o in order],
                orientation="h", name="Input +20%", marker_color="#2E7D32")
    fig.update_layout(barmode="overlay", xaxis_title="Change in NPV (USD M)", height=400, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each bar shows the NPV change when one input moves 20% and the rest stay fixed. "
               "Price, grade and recovery scale revenue identically, so their bars match.")

st.divider()
st.caption("Illustrative model. Mine assumptions are placeholders, not a real project. Not investment advice.")
