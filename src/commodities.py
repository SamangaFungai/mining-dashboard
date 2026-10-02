"""Commodity presets: data ticker, units, and illustrative mine defaults.

grade_to_units converts (grade x tonnes x recovery) into price units:
  gold/platinum: grams -> troy oz (price in USD/oz)
  copper:        percent -> pounds (price in USD/lb)
All mine defaults are ILLUSTRATIVE placeholders, not real projects.
"""

COMMODITIES = {
    "Gold": {
        "ticker": "GC=F", "price_unit": "oz", "grade_label": "Head grade (g/t)",
        "grade_to_units": 1 / 31.1035, "grade_range": (0.5, 5.0, 1.5, 0.1),
        "tonnes_mt": 1.0, "recovery": 90, "opex": 65, "capex_m": 150, "sust_m": 5,
        "royalty": 5.0, "price_range": (500.0, 10000.0, 50.0),
    },
    "Platinum": {
        "ticker": "PL=F", "price_unit": "oz", "grade_label": "Head grade (g/t, 4E PGM)",
        "grade_to_units": 1 / 31.1035, "grade_range": (1.0, 8.0, 3.5, 0.1),
        "tonnes_mt": 1.0, "recovery": 85, "opex": 95, "capex_m": 250, "sust_m": 10,
        "royalty": 5.0, "price_range": (500.0, 6000.0, 50.0),
    },
    "Copper": {
        "ticker": "HG=F", "price_unit": "lb", "grade_label": "Head grade (% Cu)",
        "grade_to_units": 2204.62 / 100, "grade_range": (0.2, 3.0, 0.8, 0.05),
        "tonnes_mt": 5.0, "recovery": 88, "opex": 28, "capex_m": 400, "sust_m": 15,
        "royalty": 3.0, "price_range": (1.0, 15.0, 0.05),
    },
}
