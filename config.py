"""
config.py - all the tunable numbers in ONE place.
Change a number here and the whole app follows, so every choice is easy to explain in the viva.
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- The 7 asset classes the portfolios are built from ----------------------------------------
# "Equity" (the broad diversified bucket) and "Flexi Cap Mutual Fund" have been REMOVED from the
# project entirely, per the brief. Large/Mid/Small cap now map directly to the PRICE RETURN
# version of their Nifty index (Nifty 100, Nifty Midcap 150, Nifty Smallcap 250) - not a mutual
# fund's TRI-based NAV - so there is no fund-level TRI/Direct-Plan adjustment to make any more;
# the index price return IS the assumption.
ASSETS = ["Liquid Fund", "Debt Fund", "Gold", "Real Estate",
          "Large Cap Mutual Fund", "Mid Cap Mutual Fund", "Small Cap Mutual Fund"]

ASSET_INDEX_MAP = {
    "Liquid Fund": "Money-market / liquid-fund category (proxy for short-term cash)",
    "Debt Fund": "Composite debt-fund category (proxy for medium-duration bonds)",
    "Gold": "Domestic (INR) gold price",
    "Real Estate": "Residential price index + average rental yield",
    "Large Cap Mutual Fund": "Nifty 100 - PRICE RETURN index (not TRI). Expected-return figure sourced "
                             "from NSE's official factsheet (since-inception PR CAGR, as of 30-Sep-2026).",
    "Mid Cap Mutual Fund": "Nifty Midcap 150 - PRICE RETURN index (not TRI). Expected-return figure "
                           "sourced from NSE's official factsheet (since-inception PR CAGR, as of 30-Sep-2026).",
    "Small Cap Mutual Fund": "Nifty Smallcap 250 - PRICE RETURN index (not TRI). Expected-return figure "
                             "sourced from NSE's official factsheet (since-inception PR CAGR, as of 30-Sep-2026).",
}

# Sources for the equity expected-return figures (fetched 2026-10-21; factsheets as-of 30-Sep-2026):
#   Nifty 100 PR since-inception CAGR 14.26%  - nsearchives.nseindia.com/content/indices/ind_nifty_100.pdf
#   Nifty Midcap 150 PR since-inception CAGR 15.42% - .../ind_Nifty_Midcap_150.pdf
#   Nifty Smallcap 250 PR since-inception CAGR 14.32% - niftyindices.com/Factsheet/ind_nifty_smallcap_250.pdf
# Volatility for these 3, and every figure for Liquid/Debt/Gold/Real Estate, plus the full
# correlation structure, remain a documented, calibrated representative assumption - the public
# factsheets' volatility columns could not be reliably mapped to a specific time window from
# their extracted text, and Liquid/Debt/Gold/Real Estate have no single official daily-priced
# index with a comparable factsheet. See data/historical_returns.csv and Portfolio_Methodology.xlsx
# for exactly how these numbers feed into the return/volatility/correlation calculation below.

# ---- Return / risk / correlation assumptions, CALCULATED from 10 years of monthly data --------
# data/historical_returns.csv holds a 120-month (Oct-2015 to Sep-2025) representative return
# series per asset - calibrated to commonly reported real-world 10-year statistics for each
# index/instrument (this sandbox has no certified live NSE/AMFI data feed to cite line-by-line,
# so the series is a documented, reproducible stand-in, not a verbatim data pull - exactly how
# every other "real-data" figure in this project has been sourced). Expected return, volatility
# AND correlation are all CALCULATED from that file below, every time the app starts - never
# hand-typed - so the app and the Excel backend (which runs the identical calculation on the
# identical file) can never drift apart.
_HIST_PATH = os.path.join(BASE_DIR, "data", "historical_returns.csv")


def load_market_data():
    """Returns (assumptions: {asset: (CAGR, annualised volatility)}, correlation: 7x7 np.array,
    note: str). CAGR is the geometric annual return (compounded monthly); volatility is the
    monthly return's standard deviation, annualised by sqrt(12). Falls back to stated placeholder
    figures only if the data file is missing/unreadable, so the app never crashes."""
    try:
        df = pd.read_csv(_HIST_PATH)
        rets = df[ASSETS].values.astype(float)
        n = rets.shape[0]
        cagr = (np.prod(1 + rets, axis=0)) ** (12 / n) - 1
        ann_vol = rets.std(axis=0, ddof=1) * np.sqrt(12)
        corr = np.corrcoef(rets, rowvar=False)
        assumptions = {a: (float(cagr[i]), float(ann_vol[i])) for i, a in enumerate(ASSETS)}
        note = (f"Calculated from {n} months ({df['month'].iloc[0]} to {df['month'].iloc[-1]}) of "
                f"representative return data per asset class - see data/historical_returns.csv and "
                f"Portfolio_Methodology.xlsx ('Market Data' / 'Assumption Calculations'). Large/Mid/"
                f"Small Cap use the PRICE RETURN version of their Nifty index, not TRI.")
        return assumptions, corr, note
    except Exception as e:  # never crash the demo because of a data file
        fallback = {
            "Liquid Fund": (0.054, 0.015), "Debt Fund": (0.069, 0.030), "Gold": (0.115, 0.215),
            "Real Estate": (0.119, 0.090), "Large Cap Mutual Fund": (0.110, 0.150),
            "Mid Cap Mutual Fund": (0.160, 0.195), "Small Cap Mutual Fund": (0.175, 0.245),
        }
        fallback_corr = np.array([
            [1.00, 0.27, 0.10, 0.00, 0.00, 0.00, 0.00],
            [0.27, 1.00, 0.27, 0.00, 0.00, 0.00, 0.00],
            [0.10, 0.27, 1.00, 0.56, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.56, 1.00, 0.15, 0.15, 0.15],
            [0.00, 0.00, 0.00, 0.15, 1.00, 0.95, 0.92],
            [0.00, 0.00, 0.00, 0.15, 0.95, 1.00, 0.99],
            [0.00, 0.00, 0.00, 0.15, 0.92, 0.99, 1.00],
        ])
        return fallback, fallback_corr, f"Using built-in placeholder assumptions (data file unreadable: {e})"


ASSUMPTIONS, CORR, ASSUMPTIONS_NOTE = load_market_data()

# How much every off-diagonal correlation is pulled toward the average off-diagonal correlation
# before the optimizer uses it (Ledoit-Wolf-style shrinkage). Without this, a handful of
# near-zero correlation estimates look like a "free lunch" to an optimizer and it piles into them
# far more than a real, reliable edge justifies (Michaud, 1989, "estimation-error maximization").
CORR_SHRINKAGE = 0.35

# ---- Planning assumptions ------------------------------------------------------
INFLATION = 0.05            # yearly rise in costs (goal costs are entered in today's rupees)
PLANNING_MARGIN = 0.01      # we plan with returns 1 percentage point lower than expected (prudence)
RETIREMENT_MULTIPLE = 25    # retirement corpus = 25 x yearly expenses (the "4% rule")
DEFAULT_WEALTH_YEARS = 15   # horizon used for surplus money with no goal attached
N_SIMS = 2000               # number of simulated futures for the probability estimates
SEED = 42                   # fixed seed -> same inputs always give the same numbers (good for demos)

# ---- Horizon buckets (years) ----------------------------------------------------
CAPITAL_PROTECTION_YEARS = 1   # goals this close are held 100% in liquid funds
SHORT_MAX_YEARS = 3            # up to 3 years = short term
MEDIUM_MAX_YEARS = 7           # more than 3 up to 7 years = medium term, above 7 = long term

# ---- Risk-score labels -----------------------------------------------------------
CONSERVATIVE_BELOW = 4.0
AGGRESSIVE_FROM = 7.0
PROFILES = ["Conservative", "Moderate", "Aggressive"]

# ---- Mean-Variance Optimization (Markowitz) solver settings ----------------------
# maximise  w'mu - 0.5 * risk_aversion * w'Sigma w   s.t.  sum(w)=1, w>=0, equity-cap by horizon,
# per-asset caps below. Lower risk_aversion = more willing to take on volatility for return.
RISK_AVERSION = {"Conservative": 9.0, "Moderate": 3.5, "Aggressive": 1.2}
BUCKET_EQUITY_CAP = {"Short": 0.15, "Medium": 0.55, "Long": 1.0}
# Per-asset caps, so the optimizer can never concentrate the whole equity sleeve (or the whole
# portfolio) into a single asset just because it has the best estimated risk/return trade-off on
# paper - a standard diversification safeguard against estimation error (Michaud, 1989). Gold and
# Real Estate are capped tighter still: real-estate index data in particular is known to
# understate true volatility ("appraisal smoothing"), making it look artificially attractive.
# Large/Mid/Small cap are highly correlated with each other (~0.92-0.99 before shrinkage), so
# without their own caps the optimizer puts the whole equity sleeve into whichever one has the
# best Sharpe-like ratio and leaves the other two at zero - capping each individually forces the
# diversification across market-cap segments the brief's "Large/Mid/Small cap" split is meant to
# give the user in the first place.
MVO_ASSET_MAX = {
    "Gold": 0.20, "Real Estate": 0.20,
    "Large Cap Mutual Fund": 0.35, "Mid Cap Mutual Fund": 0.35, "Small Cap Mutual Fund": 0.30,
}
MVO_ASSET_MAX_DEFAULT = 0.45
MVO_UNCAPPED_ASSETS = {"Liquid Fund", "Debt Fund"}

# =========================================================================
# ADDITIONS BELOW THIS LINE support the expanded financial-profile,
# goal-priority, step-up-SIP and risk-metric features.
# Every number here is a stated, documented assumption — see the
# "Assumptions & Methodology" panel the app renders from this file.
# =========================================================================

# ---- Goal priority tiers (used to decide funding order under a budget constraint) ----
PRIORITIES = ["Essential", "Important", "Aspirational"]
PRIORITY_ORDER = {p: i for i, p in enumerate(PRIORITIES)}   # Essential funded first

# ---- Goal types (brief: Vacation / Housing / Healthcare / Education / Marriage / Others) -------
# "Others" uses general inflation. Figures are stated planning heuristics - commonly-cited
# category-level inflation rates - not official published sub-indices.
GOAL_TYPES = ["Vacation", "Housing", "Healthcare", "Education", "Marriage", "Others"]
GOAL_INFLATION = {
    "Vacation": 0.06, "Housing": 0.06, "Healthcare": 0.09, "Education": 0.08,
    "Marriage": 0.07, "Others": 0.05,
}
DEFAULT_GOAL_TYPE = "Others"

VACATION_COUNTRIES = ["India (domestic)", "Southeast Asia", "Middle East / Gulf", "Europe",
                      "United Kingdom", "United States / Canada", "Australia / New Zealand",
                      "Japan / South Korea", "Other international"]

# ---- Employment types and how they affect risk CAPACITY and the (advisory-only) emergency-fund months ----
EMPLOYMENT_TYPES = ["Salaried", "Self-employed / business owner", "Freelancer / professional", "Student", "Retired", "Other"]

EMERGENCY_MONTHS_BASE = {
    "Salaried": 4, "Self-employed / business owner": 8, "Freelancer / professional": 8,
    "Student": 3, "Retired": 6, "Other": 6,
}
EMERGENCY_MONTHS_PER_DEPENDENT = 1     # +1 month of buffer per financial dependent
EMERGENCY_MONTHS_CAP = 12              # don't ask for more than a year's buffer

EMPLOYMENT_CAPACITY_PTS = {
    "Salaried": 7, "Self-employed / business owner": 5, "Freelancer / professional": 4,
    "Student": 5, "Retired": 3, "Other": 5,
}

# ---- Debt / affordability checks ----
HIGH_EMI_BURDEN_RATIO = 0.40     # EMIs above this share of monthly income are flagged as a high debt burden
HIGH_INTEREST_DEBT_RATE = 0.12   # loans costlier than this are flagged as "consider repaying before investing more"

# ---- Risk metrics ----
RISK_FREE_RATE = 0.053   # approx short-term G-Sec/liquid-fund yield, for the Sharpe ratio calc only

# ---- Step-up SIP default ----
DEFAULT_SIP_STEP_UP_ENABLED = False   # off unless the user turns it on
DEFAULT_SIP_STEP_UP = 0.0             # 0% unless the user says otherwise

# ---- Income/expense growth ----
LIFESTYLE_CREEP_FACTOR = 0.3   # of any REAL (above-inflation) income growth, this fraction tends to be spent, not saved

# =========================================================================
# STEP 1/2 INPUT SIMPLIFICATION: band-based income/expense estimation
# (brief items: income band + city tier + vehicle instead of exact income; housing situation;
# auto-estimated income/expenses with an opt-out to exact values)
# =========================================================================

LOCATION_TIERS = ["Tier 1 (metro)", "Tier 2", "Tier 3 / rural"]
LOCATION_COST_INDEX = {"Tier 1 (metro)": 1.20, "Tier 2": 1.00, "Tier 3 / rural": 0.82}

HOUSING_SITUATIONS = ["Renting", "Own home - with a home loan", "Own home - no loan", "Living with parents / family"]
HOUSING_EXPENSE_LOAD = {
    "Renting": 0.18, "Own home - with a home loan": 0.20,
    "Own home - no loan": -0.05, "Living with parents / family": -0.10,
}

VEHICLE_INDICATORS = ["None", "Two-wheeler", "Entry-level car", "Mid-range car", "Premium car"]
VEHICLE_LIFESTYLE_NUDGE = {
    "None": -0.03, "Two-wheeler": 0.00, "Entry-level car": 0.03, "Mid-range car": 0.07, "Premium car": 0.13,
}

# Monthly income bands (brief-specified exactly) -> (label, representative monthly Rs value).
# Open-ended bands use a documented 70%/130% convention (below the floor / above the ceiling).
INCOME_BANDS = [
    ("Below Rs 50,000", 35_000),
    ("Rs 50,000 - 1,00,000", 75_000),
    ("Rs 1,00,000 - 2,00,000", 1_50_000),
    ("Rs 2,00,000 - 3,00,000", 2_50_000),
    ("Rs 3,00,000 - 5,00,000", 4_00_000),
    ("Rs 5,00,000 - 10,00,000", 7_50_000),
    ("Rs 10,00,000 - 15,00,000", 12_50_000),
    ("Rs 15,00,000 - 25,00,000", 20_00_000),
    ("Rs 25,00,000 - 50,00,000", 37_50_000),
    ("Above Rs 50,00,000", 65_00_000),
]

# Base expense ratio (of income) by income band - higher income -> proportionally more is saved,
# a standard, documented planning convention (not every rupee of a raise gets spent).
BASE_EXPENSE_RATIO_BY_BAND = [0.85, 0.75, 0.65, 0.58, 0.52, 0.45, 0.40, 0.35, 0.30, 0.25]
EXPENSE_RATIO_PER_DEPENDENT = 0.04
EXPENSE_RATIO_FLOOR = 0.25
EXPENSE_RATIO_CAP = 0.95

# ---- Existing-investment categories (brief: Liquid/Cash, Debt/Fixed Income, Gold, Real Estate, Mutual Funds-equity) ----
EXISTING_CATEGORIES = ["Liquid / Cash", "Debt / Fixed Income", "Gold", "Real Estate", "Mutual Funds (equity)"]
# Maps each existing-investment category to the ASSETS index/indices it corresponds to, for the
# combined (existing + new money) mix calculation.
EXISTING_CATEGORY_TO_ASSETS = {
    "Liquid / Cash": ["Liquid Fund"], "Debt / Fixed Income": ["Debt Fund"], "Gold": ["Gold"],
    "Real Estate": ["Real Estate"],
    "Mutual Funds (equity)": ["Large Cap Mutual Fund", "Mid Cap Mutual Fund", "Small Cap Mutual Fund"],
}
