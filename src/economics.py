"""Mine economics model for a hypothetical gold mine.

All defaults are ILLUSTRATIVE placeholders, not a real project. Replace them
with figures from a published technical report when you have one.

Run from the project root:  python -m src.economics
"""
from dataclasses import dataclass, replace

import numpy as np

TROY_OZ_G = 31.1035  # grams per troy ounce


@dataclass
class Mine:
    life_years: int = 10
    tonnes_per_year: float = 1_000_000      # ore milled per year
    head_grade_gpt: float = 1.5             # grams of gold per tonne
    recovery: float = 0.90                  # metallurgical recovery
    opex_per_tonne: float = 65.0            # mining + processing + G&A, USD/t
    sustaining_capex_per_year: float = 5_000_000
    initial_capex: float = 150_000_000
    royalty: float = 0.05                   # share of revenue (check local rate)
    tax_rate: float = 0.25                  # on profit after depreciation
    discount_rate: float = 0.08
    grade_to_units: float = 1 / TROY_OZ_G   # price units per tonne of ore per unit of grade

    @property
    def units_per_year(self) -> float:
        """Payable metal per year, in the price unit (oz for gold/platinum, lb for copper)."""
        return self.tonnes_per_year * self.head_grade_gpt * self.recovery * self.grade_to_units

    @property
    def oz_per_year(self) -> float:  # kept for backward compatibility
        return self.units_per_year


def cash_flows(mine: Mine, prices) -> np.ndarray:
    """Annual cash flows. `prices` has shape (..., life_years), USD/oz.

    Returns shape (..., life_years + 1); index 0 is year 0 (initial capex).
    """
    prices = np.asarray(prices, dtype=float)
    revenue = mine.units_per_year * prices
    opex = mine.tonnes_per_year * mine.opex_per_tonne
    royalty = mine.royalty * revenue
    ebitda = revenue - opex - royalty - mine.sustaining_capex_per_year
    depreciation = mine.initial_capex / mine.life_years  # straight-line
    tax = np.maximum(0.0, ebitda - depreciation) * mine.tax_rate
    operating_cf = ebitda - tax
    year0 = np.full(prices.shape[:-1] + (1,), -mine.initial_capex)
    return np.concatenate([year0, operating_cf], axis=-1)


def npv(cf: np.ndarray, rate: float) -> np.ndarray:
    factors = (1 + rate) ** -np.arange(cf.shape[-1])
    return cf @ factors


def irr(cf: np.ndarray) -> float:
    """IRR for a single cash flow series, by bisection. NaN if undefined."""
    lo, hi = -0.99, 10.0
    if npv(cf, lo) * npv(cf, hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(cf, lo) * npv(cf, mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def payback_years(cf: np.ndarray) -> float:
    cum = np.cumsum(cf)
    idx = np.argmax(cum >= 0) if (cum >= 0).any() else None
    if idx is None:
        return float("nan")
    if idx == 0:
        return 0.0
    return idx - 1 + (-cum[idx - 1]) / cf[idx]  # linear interpolation within the year


def aisc(mine: Mine, price: float) -> float:
    """All-in sustaining cost per ounce at a given gold price."""
    opex = mine.tonnes_per_year * mine.opex_per_tonne
    royalty = mine.royalty * mine.units_per_year * price
    return (opex + royalty + mine.sustaining_capex_per_year) / mine.units_per_year


def breakeven_price(mine: Mine) -> float:
    """Constant gold price (USD/oz) at which NPV = 0."""
    lo, hi = 0.0, 20_000.0

    def f(p):
        return npv(cash_flows(mine, np.full(mine.life_years, p)), mine.discount_rate)

    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate_prices(p0: float, annual_vol: float, years: int, n: int = 10_000, seed: int = 42):
    """Geometric Brownian motion with zero drift. Returns (n, years) prices."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, years))
    log_steps = -0.5 * annual_vol**2 + annual_vol * z
    return p0 * np.exp(np.cumsum(log_steps, axis=1))


def monte_carlo(mine: Mine, p0: float, annual_vol: float, n: int = 10_000, seed: int = 42):
    paths = simulate_prices(p0, annual_vol, mine.life_years, n, seed)
    return npv(cash_flows(mine, paths), mine.discount_rate)


def sensitivity(mine: Mine, price: float, swing: float = 0.20) -> dict:
    """NPV change for a +/- `swing` move in each key input (for a tornado chart)."""
    flat = np.full(mine.life_years, price)
    base = npv(cash_flows(mine, flat), mine.discount_rate)
    out = {}
    for name in ["head_grade_gpt", "recovery", "opex_per_tonne", "initial_capex"]:
        low = replace(mine, **{name: getattr(mine, name) * (1 - swing)})
        high = replace(mine, **{name: getattr(mine, name) * (1 + swing)})
        out[name] = (
            npv(cash_flows(low, flat), mine.discount_rate) - base,
            npv(cash_flows(high, flat), mine.discount_rate) - base,
        )
    p_low = npv(cash_flows(mine, flat * (1 - swing)), mine.discount_rate) - base
    p_high = npv(cash_flows(mine, flat * (1 + swing)), mine.discount_rate) - base
    out["gold_price"] = (p_low, p_high)
    return out


if __name__ == "__main__":
    import numpy as np
    from src.data import get_prices

    daily = get_prices("Gold")
    p0 = float(daily.iloc[-1])
    vol = float(np.log(daily).diff().dropna().std() * np.sqrt(252))

    mine = Mine()
    cf = cash_flows(mine, np.full(mine.life_years, p0))
    m = 1e6

    print(f"Spot gold: ${p0:,.0f}/oz   annualised volatility: {vol:.1%}")
    print(f"Production: {mine.units_per_year:,.0f} oz/year\n")
    print("Base case at constant spot price")
    print(f"  NPV @ {mine.discount_rate:.0%}:    ${npv(cf, mine.discount_rate) / m:,.1f}M")
    print(f"  IRR:          {irr(cf):.1%}")
    print(f"  Payback:      {payback_years(cf):.1f} years")
    print(f"  AISC:         ${aisc(mine, p0):,.0f}/oz")
    print(f"  Breakeven:    ${breakeven_price(mine):,.0f}/oz\n")

    npvs = monte_carlo(mine, p0, vol)
    p10, p50, p90 = np.percentile(npvs, [10, 50, 90]) / m
    print("Monte Carlo (10,000 price paths)")
    print(f"  NPV P10 / P50 / P90: ${p10:,.0f}M / ${p50:,.0f}M / ${p90:,.0f}M")
    print(f"  Mean NPV:            ${npvs.mean() / m:,.0f}M")
    print(f"  Probability NPV < 0: {(npvs < 0).mean():.1%}")
