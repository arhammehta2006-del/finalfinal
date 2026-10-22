"""
market_data.py - keeps data/historical_returns.csv current by (re)fetching real market data.

HOW THIS FITS THE REST OF THE APP
----------------------------------
config.py never changes: it always computes CAGR / volatility / correlation from
data/historical_returns.csv at import time, exactly as before. This module's only job is to
occasionally REGENERATE that CSV so it reflects current prices, using the same exact
moment-matching generator the project has always used (Cholesky whitening against a target
mean/vol/correlation) - so nothing downstream (the planner engine, the Excel backend, the
Methodology doc) needs to know or care whether a given run used live or calibrated numbers.

WHAT IS ACTUALLY "LIVE" HERE
-----------------------------
- Large Cap / Mid Cap / Small Cap (Nifty 100 / Midcap 150 / Smallcap 250 proxies) and Gold are
  fetched from Yahoo Finance (via `yfinance`) as monthly closes, and their CAGR/volatility are
  computed directly from that real price history. The pairwise correlation AMONG these 4 is also
  computed from the real, overlapping price history - so that sub-block of the correlation matrix
  becomes genuinely empirical, not assumed.
- Liquid Fund, Debt Fund and Real Estate have no single, clean, publicly-fetchable daily price
  series (a fund CATEGORY and an illiquid asset class respectively), so they keep the documented,
  calibrated assumptions - unchanged by a refresh. Their correlation with the 4 "live" assets also
  stays at the documented calibrated value.
- IMPORTANT, READ BEFORE DEPLOYING: this sandbox's own network cannot reach Yahoo Finance (every
  outbound request outside a small package-registry allowlist is refused here), so the fetch
  logic below has NOT been exercised against a live connection - only against a mocked response
  (see test_market_data.py). The ticker symbols in TICKERS are a best-effort mapping of each Nifty
  sub-index / Gold to a tradeable, Yahoo-covered instrument; double-check them via the sidebar's
  "Data status" panel once this is actually deployed somewhere with real internet access, and edit
  TICKERS below if a symbol doesn't resolve (Yahoo's exact NSE/index ticker spellings shift
  occasionally).
- KNOWN GOTCHA: a first real deployment (of this project's FastAPI sibling) saw EVERY ticker come
  back "no usable data returned" even with confirmed real internet access - not a per-ticker
  symbol problem (one of the failing tickers, GOLDBEES.NS, is a long-established, highly liquid
  ETF that should resolve easily). The likely cause: an outdated pinned `yfinance` version hitting
  a Yahoo API change (Yahoo has periodically required a session "crumb" that older yfinance
  releases don't fetch correctly). requirements.txt here pins `yfinance>=0.2.54` for this reason -
  if every asset fails identically again after deploying, that is the first thing to check/bump.
"""
import json
import os
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "historical_returns.csv")
META_PATH = os.path.join(DATA_DIR, "market_data_meta.json")

ASSETS = ["Liquid Fund", "Debt Fund", "Gold", "Real Estate",
          "Large Cap Mutual Fund", "Mid Cap Mutual Fund", "Small Cap Mutual Fund"]

# Assets we attempt to fetch live, and which tradeable ticker(s) stand in for each. Each asset
# lists candidates to try in order - the first one that returns enough history wins.
TICKERS = {
    "Large Cap Mutual Fund": ["^CNX100", "ICICINIFTY100.NS", "NIFTY100.NS"],
    "Mid Cap Mutual Fund": ["NIFTYMID150.NS", "MID150BEES.NS", "^NSEMDCP150"],
    "Small Cap Mutual Fund": ["SMALLCAP.NS", "^NSESML250"],
    "Gold": ["GOLDBEES.NS"],
}
LIVE_ASSETS = list(TICKERS.keys())
CALIBRATED_ASSETS = [a for a in ASSETS if a not in LIVE_ASSETS]

# Calibrated fallback assumptions - identical to the project's documented baseline. Used whenever
# a live fetch is unavailable (as it always will be inside this sandbox) or fails for an asset,
# and always used as-is for Liquid Fund / Debt Fund / Real Estate.
CALIBRATED_MEAN = {
    "Liquid Fund": 0.054, "Debt Fund": 0.069, "Gold": 0.115, "Real Estate": 0.119,
    "Large Cap Mutual Fund": 0.1426, "Mid Cap Mutual Fund": 0.1542, "Small Cap Mutual Fund": 0.1432,
}
CALIBRATED_VOL = {
    "Liquid Fund": 0.015, "Debt Fund": 0.030, "Gold": 0.215, "Real Estate": 0.090,
    "Large Cap Mutual Fund": 0.145, "Mid Cap Mutual Fund": 0.190, "Small Cap Mutual Fund": 0.235,
}
CALIBRATED_CORR = pd.DataFrame(
    [
        [1.000000, 0.268688, 0.099785, 0.000000, 0.000000, 0.000000, 0.000000],
        [0.268688, 1.000000, 0.269267, 0.000000, 0.000000, 0.000000, 0.000000],
        [0.099785, 0.269267, 1.000000, 0.563535, 0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.563535, 1.000000, 0.153333, 0.153333, 0.153333],
        [0.000000, 0.000000, 0.000000, 0.153333, 1.000000, 0.945435, 0.921797],
        [0.000000, 0.000000, 0.000000, 0.153333, 0.945435, 1.000000, 0.991152],
        [0.000000, 0.000000, 0.000000, 0.153333, 0.921797, 0.991152, 1.000000],
    ],
    index=ASSETS, columns=ASSETS,
)
N_MONTHS = 120
SEED = 20261021


def _fetch_monthly_closes(ticker, months_needed):
    """One ticker's monthly close series via yfinance, or None if it can't be fetched/parsed."""
    import yfinance as yf  # imported lazily so the rest of the app works even if unavailable
    df = yf.download(ticker, period=f"{months_needed + 6}mo", interval="1mo",
                      progress=False, auto_adjust=True)
    if df is None or df.empty or "Close" not in df:
        return None
    closes = df["Close"].dropna()
    if isinstance(closes, pd.DataFrame):  # yfinance sometimes returns a 1-col DataFrame
        closes = closes.iloc[:, 0]
    if len(closes) < 24:  # need at least 2 years to call this a meaningful estimate
        return None
    return closes.tail(months_needed + 1)


def _fetch_latest_daily_quote(ticker):
    """The single most recent DAILY close for a ticker - used only for the on-screen 'live feed'
    display. The stats fetch above intentionally uses monthly bars (right granularity for
    CAGR/volatility), but a monthly bar is dated to the start of the month, which reads as stale
    on screen even when the fetch just happened. A short daily pull gives a date close to 'today'
    instead, which is what someone cross-checking against Google Finance / NSE actually expects."""
    import yfinance as yf  # imported lazily so the rest of the app works even if unavailable
    df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty or "Close" not in df:
        return None
    closes = df["Close"].dropna()
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    if closes.empty:
        return None
    return float(closes.iloc[-1]), closes.index[-1]


def fetch_live_stats(months_needed=N_MONTHS):
    """
    Attempts a live fetch for every asset in TICKERS.
    Returns (stats, used_ticker, errors, price_series, latest_price):
      stats: {asset: (cagr, annualised_vol)} for assets that were successfully fetched
      used_ticker: {asset: ticker_string} for assets that were successfully fetched
      errors: {asset: "last error message"} for assets where every candidate ticker failed
      price_series: {asset: pandas Series of monthly closes} for successfully fetched assets,
                    used afterwards to compute the real empirical correlation block
      latest_price: {asset: {"ticker", "price", "as_of"}} - the single most recent close actually
                    returned by Yahoo for that ticker, used only to PROVE the fetch is live (this
                    number is never fed into the stats/portfolio math - it's a display-only receipt
                    someone can cross-check against Google Finance / NSE at the same moment)
    """
    stats, used_ticker, errors, price_series, latest_price = {}, {}, {}, {}, {}
    for asset, candidates in TICKERS.items():
        last_err = None
        for ticker in candidates:
            try:
                closes = _fetch_monthly_closes(ticker, months_needed)
                if closes is None or len(closes) < 24:
                    last_err = f"{ticker}: no usable data returned"
                    continue
                monthly_ret = closes.pct_change().dropna()
                n = len(monthly_ret)
                cagr = float((1 + monthly_ret).prod() ** (12 / n) - 1)
                vol = float(monthly_ret.std(ddof=1) * np.sqrt(12))
                stats[asset] = (cagr, vol)
                used_ticker[asset] = ticker
                price_series[asset] = monthly_ret
                # Prefer a fresh daily quote for the on-screen price (reads as genuinely "today",
                # not the 1st-of-the-month date a monthly bar carries); fall back to the monthly
                # close itself if the daily pull fails for some reason.
                try:
                    daily = _fetch_latest_daily_quote(ticker)
                except Exception:  # noqa: BLE001 - this is a display-only extra, never fatal
                    daily = None
                if daily is not None:
                    last_close, last_date = daily
                else:
                    last_close, last_date = closes.iloc[-1], closes.index[-1]
                latest_price[asset] = {
                    "ticker": ticker,
                    "price": round(float(last_close), 2),
                    "as_of": pd.Timestamp(last_date).strftime("%Y-%m-%d"),
                }
                break
            except Exception as e:  # noqa: BLE001 - we want to try the next candidate regardless
                last_err = f"{ticker}: {type(e).__name__}: {e}"
        if asset not in stats:
            errors[asset] = last_err or "no candidates configured"
    return stats, used_ticker, errors, price_series, latest_price


def _empirical_corr_block(price_series):
    """Real pairwise correlation among whichever LIVE_ASSETS were actually fetched, aligned on
    their common dates. Returns a DataFrame (subset x subset), or None if fewer than 2 fetched."""
    if len(price_series) < 2:
        return None
    df = pd.DataFrame(price_series).dropna(how="any")
    if len(df) < 12:
        return None
    return df.corr()


def _generate_series(target_mean, target_vol, target_corr_df, seed=SEED, n_months=N_MONTHS):
    """The project's original exact-moment-matching generator (Cholesky whitening): produces a
    reproducible n_months x len(ASSETS) return series whose realised mean/vol/correlation equal
    the targets to numerical precision, from whatever target_mean/vol/corr it is given."""
    rng = np.random.RandomState(seed)
    monthly_mean = np.array([target_mean[a] for a in ASSETS]) / 12
    monthly_vol = np.array([target_vol[a] for a in ASSETS]) / np.sqrt(12)
    corr = target_corr_df.loc[ASSETS, ASSETS].values
    k = len(ASSETS)
    Z = rng.standard_normal((n_months, k))
    Z = Z - Z.mean(axis=0)
    S = (Z.T @ Z) / (n_months - 1)
    Z_white = Z @ np.linalg.inv(np.linalg.cholesky(S).T)
    L = np.linalg.cholesky(corr)
    X = Z_white @ L.T
    monthly_returns = monthly_mean + X * monthly_vol
    dates = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1),
                           periods=n_months, freq="ME")
    df = pd.DataFrame(monthly_returns, columns=ASSETS)
    df.insert(0, "month", dates.strftime("%Y-%m"))
    df[ASSETS] = df[ASSETS].round(6)
    return df


def refresh(write=True):
    """
    Main entry point. Tries a live fetch for Large/Mid/Small Cap and Gold; whatever succeeds
    overlays the calibrated defaults (return, volatility, and - for pairs that were BOTH fetched -
    their correlation). Always regenerates historical_returns.csv (even on total fetch failure,
    so a fresh deploy with no network yet still has a valid, documented file), and always writes
    market_data_meta.json describing exactly what happened, so GET /api/market-data/status can be
    honest with the user about whether today's numbers are live or calibrated.
    Never raises - a broken data source should degrade the assumptions, not the app.
    """
    started = datetime.now(timezone.utc)
    stats, used_ticker, errors, price_series, latest_price = {}, {}, {}, {}, {}
    fetch_attempted = True
    try:
        stats, used_ticker, errors, price_series, latest_price = fetch_live_stats()
    except Exception:  # noqa: BLE001 - yfinance missing, no network, anything at all
        errors = {a: "fetch_live_stats() raised: " + traceback.format_exc(limit=2) for a in LIVE_ASSETS}

    target_mean = dict(CALIBRATED_MEAN)
    target_vol = dict(CALIBRATED_VOL)
    target_corr = CALIBRATED_CORR.copy()
    for asset, (cagr, vol) in stats.items():
        target_mean[asset] = cagr
        target_vol[asset] = vol
    corr_block = _empirical_corr_block(price_series)
    if corr_block is not None:
        for a in corr_block.index:
            for b in corr_block.columns:
                target_corr.loc[a, b] = float(corr_block.loc[a, b])

    df = _generate_series(target_mean, target_vol, target_corr)
    if write:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(CSV_PATH, index=False)

    meta = {
        "last_refresh_attempt_utc": started.isoformat(),
        "fetch_attempted": fetch_attempted,
        "live_assets_fetched": sorted(stats.keys()),
        "tickers_used": used_ticker,
        "fetch_errors": errors,
        "calibrated_assets": CALIBRATED_ASSETS,
        "any_live_data_used": bool(stats),
        "latest_prices": latest_price,
        "note": ("Large/Mid/Small Cap and Gold are refreshed from live prices when this process "
                 "has real internet access; Liquid Fund, Debt Fund and Real Estate always use the "
                 "documented calibrated assumptions (no single clean public price series exists "
                 "for them). This sandbox used to build the app has no outbound internet access, "
                 "so 'any_live_data_used' will read false until this is deployed somewhere with "
                 "real network access."),
    }
    if write:
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
    return meta


def read_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {"note": "No refresh has run yet in this deployment.", "any_live_data_used": False}
