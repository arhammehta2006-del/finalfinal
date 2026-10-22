"""
Smoke tests for market_data.py's REFRESH PIPELINE (merge-stats -> regenerate CSV -> write meta),
using a mocked fetch so the logic can be verified without real network access (which this sandbox
does not have). This does NOT verify that the Yahoo Finance tickers in TICKERS actually resolve -
only that the module behaves correctly however the fetch goes (full success, partial success, or
total failure, which is exactly what happens when this runs inside this sandbox).

Run: python test_market_data.py
"""
import json
import os
import shutil
import tempfile
from unittest import mock

import numpy as np
import pandas as pd

import market_data as md


def _fake_price_series(cagr, vol, n=120, seed=1):
    rng = np.random.RandomState(seed)
    monthly_mean = cagr / 12
    monthly_vol = vol / np.sqrt(12)
    rets = monthly_mean + rng.standard_normal(n) * monthly_vol
    idx = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="ME")
    closes = 100 * (1 + pd.Series(rets, index=idx)).cumprod()
    return closes


def test_total_failure_never_crashes(tmp_dir):
    """Simulates exactly this sandbox: every fetch attempt fails. refresh() must still produce a
    valid CSV (from calibrated fallbacks) and an honest meta.json, never raise."""
    md.DATA_DIR = tmp_dir
    md.CSV_PATH = os.path.join(tmp_dir, "historical_returns.csv")
    md.META_PATH = os.path.join(tmp_dir, "market_data_meta.json")
    with mock.patch.object(md, "fetch_live_stats", side_effect=RuntimeError("no network (expected in this sandbox)")):
        meta = md.refresh()
    assert meta["any_live_data_used"] is False
    assert os.path.exists(md.CSV_PATH)
    df = pd.read_csv(md.CSV_PATH)
    assert len(df) == md.N_MONTHS
    assert list(df.columns) == ["month"] + md.ASSETS
    # Falls back to the calibrated figures (allowing for the expected, well-understood gap
    # between an arithmetic target mean and the realised geometric CAGR under compounding - this
    # "volatility drag" is bigger for higher-volatility assets like Gold, by design; it is not a bug)
    cagr = (np.prod(1 + df[md.ASSETS].values, axis=0)) ** (12 / len(df)) - 1
    for i, a in enumerate(md.ASSETS):
        tol = max(0.01, 0.75 * md.CALIBRATED_VOL[a] ** 2)
        assert abs(cagr[i] - md.CALIBRATED_MEAN[a]) < tol, f"{a}: {cagr[i]} vs {md.CALIBRATED_MEAN[a]} (tol {tol})"
    print("PASS: total failure never crashes, falls back to calibrated assumptions")


def test_partial_success_overlays_only_fetched_assets(tmp_dir):
    """Simulates a deploy where Gold and Large Cap fetch fine but Mid/Small Cap fail (e.g. a bad
    ticker symbol) - exactly the kind of partial result a real deploy might see at first."""
    md.DATA_DIR = tmp_dir
    md.CSV_PATH = os.path.join(tmp_dir, "historical_returns.csv")
    md.META_PATH = os.path.join(tmp_dir, "market_data_meta.json")
    fake_gold = _fake_price_series(cagr=0.20, vol=0.18, seed=2).pct_change().dropna()
    fake_large = _fake_price_series(cagr=0.25, vol=0.16, seed=3).pct_change().dropna()
    fake_stats = {"Gold": (0.20, 0.18), "Large Cap Mutual Fund": (0.25, 0.16)}
    fake_used = {"Gold": "GOLDBEES.NS", "Large Cap Mutual Fund": "^CNX100"}
    fake_errors = {"Mid Cap Mutual Fund": "NIFTYMID150.NS: no usable data returned",
                   "Small Cap Mutual Fund": "SMALLCAP.NS: no usable data returned"}
    fake_series = {"Gold": fake_gold, "Large Cap Mutual Fund": fake_large}
    fake_latest = {"Gold": {"ticker": "GOLDBEES.NS", "price": 68.4, "as_of": "2026-10-01"},
                   "Large Cap Mutual Fund": {"ticker": "^CNX100", "price": 24800.1, "as_of": "2026-10-01"}}
    with mock.patch.object(md, "fetch_live_stats",
                           return_value=(fake_stats, fake_used, fake_errors, fake_series, fake_latest)):
        meta = md.refresh()
    assert meta["any_live_data_used"] is True
    assert set(meta["live_assets_fetched"]) == {"Gold", "Large Cap Mutual Fund"}
    assert meta["tickers_used"] == fake_used
    assert "Mid Cap Mutual Fund" in meta["fetch_errors"]
    df = pd.read_csv(md.CSV_PATH)
    cagr = (np.prod(1 + df[md.ASSETS].values, axis=0)) ** (12 / len(df)) - 1
    assert abs(cagr[df.columns.get_loc("Gold") - 1] - 0.20) < 0.75 * 0.18 ** 2 + 0.01
    assert abs(cagr[df.columns.get_loc("Large Cap Mutual Fund") - 1] - 0.25) < 0.75 * 0.16 ** 2 + 0.01
    # Mid/Small cap, which failed to fetch, should fall back to the calibrated figures unchanged
    mid_cagr = cagr[df.columns.get_loc("Mid Cap Mutual Fund") - 1]
    mid_tol = 0.75 * md.CALIBRATED_VOL["Mid Cap Mutual Fund"] ** 2 + 0.01
    assert abs(mid_cagr - md.CALIBRATED_MEAN["Mid Cap Mutual Fund"]) < mid_tol
    print("PASS: partial fetch success overlays only the assets that actually fetched")


def test_meta_is_always_written(tmp_dir):
    md.DATA_DIR = tmp_dir
    md.CSV_PATH = os.path.join(tmp_dir, "historical_returns.csv")
    md.META_PATH = os.path.join(tmp_dir, "market_data_meta.json")
    md.refresh()
    assert os.path.exists(md.META_PATH)
    with open(md.META_PATH) as f:
        meta = json.load(f)
    assert "last_refresh_attempt_utc" in meta
    read_back = md.read_meta()
    assert read_back["last_refresh_attempt_utc"] == meta["last_refresh_attempt_utc"]
    print("PASS: meta.json is always written and readable via read_meta()")


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    try:
        test_total_failure_never_crashes(tmp)
        test_partial_success_overlays_only_fetched_assets(tmp)
        test_meta_is_always_written(tmp)
        print("\nALL market_data.py PIPELINE TESTS PASSED (mocked - no real network involved)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
