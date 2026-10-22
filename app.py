"""
app.py - interactive front-end (Streamlit).
Run with:   streamlit run app.py

Flow:  How it works -> About you -> Financial life -> Money & goals -> Risk comfort -> Review -> Your plan
"""
import html
import importlib
import json
import os
import re
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Goal-Based Portfolio Planner", page_icon="📈", layout="wide")

# market_data.py must be imported (and, if the cached refresh is due, run) BEFORE config.py - config
# computes its return/volatility/correlation assumptions from data/historical_returns.csv the moment
# it is first imported, so a refresh that happens after that point would only take effect on the
# NEXT rerun. st.cache_resource makes this run at most once per server process per `ttl` (here, once
# a day) rather than on every page interaction - see market_data.py for what "live" actually covers
# and how it degrades safely when there's no network access (as in this sandbox).
import market_data as MD


@st.cache_resource(ttl=24 * 3600, show_spinner=False)
def _bootstrap_market_data():
    return MD.refresh()


_bootstrap_market_data()

import advisor as D
import allocation as A
import config as C
import excel_export as X
import risk as R
from utils import fmt_inr

STEPS = ["How it works", "About you", "Financial life", "Money & goals", "Risk comfort", "Review", "Your plan"]
# The wizard steps shown in the horizontal stepper (the intro and results screens get their own treatment).
WIZARD_STEPS = ["About you", "Financial life", "Money & goals", "Risk comfort", "Review"]

# Liquid / Debt / Gold / Real Estate keep realistic, distinct colors; Large/Mid/Small cap run as a
# light-to-dark indigo ramp (matching the app's accent) so they read as one equity "family",
# graduated by market-cap segment.
ASSET_COLORS = ["#94A3B8", "#64748B", "#D4A72C", "#EA7B3C", "#A5B4FC", "#6366F1", "#3730A3"]
ASSET_SCALE = alt.Scale(domain=C.ASSETS, range=ASSET_COLORS)
DISCLAIMER = ("Educational guidance only, not personalised financial advice. Figures are estimates based on stated "
              "assumptions and are not guaranteed. Please consult a SEBI-registered advisor before investing.")

HEALTH_COLOR = {"Good": "🟢", "Strong": "🟢", "On Track": "🟢", "Healthy": "🟢",
                "Needs Attention": "🟡", "Behind Target": "🟡", "Moderate": "🟡", "Needs Review": "🟡",
                "At Risk": "🔴", "Significantly Behind": "🔴", "Low": "🔴",
                "Not assessed": "⚪"}

# ---- Clean modern SaaS theme: white/light-grey, one confident indigo accent, soft shadows, ----
# ---- generously rounded cards - a modern fintech-dashboard feel. Inter throughout. ----
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --accent: #4F46E5; --accent-dark: #3730A3; --accent-soft: #EEF0FD;
  --teal: #0D9488; --teal-soft: #ECFBF9;
  --bg: #F6F7FB; --card: #FFFFFF; --line: #E7E9F2; --ink: #1A1D29; --muted: #70768C;
  --shadow: 0 1px 2px rgba(16,24,40,0.04), 0 1px 8px rgba(16,24,40,0.05);
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
.stApp { background: var(--bg); }
.block-container { padding-top: 1.6rem; max-width: 1080px; }
h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: -0.01em; color: var(--ink); }
h1 { font-size: 2rem !important; }
p, li, span, label { font-family: 'Inter', sans-serif; }

/* ---- stepper ---- */
.pp-stepper { display: flex; align-items: center; margin: 4px 0 28px; }
.pp-step { display: flex; flex-direction: column; align-items: center; flex: 1; position: relative; }
.pp-step .dot {
  width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.9rem; border: 1.5px solid var(--line);
  background: var(--card); color: var(--muted); z-index: 2;
}
.pp-step.done .dot { background: var(--accent); border-color: var(--accent); color: #fff; }
.pp-step.active .dot { background: #fff; border-color: var(--accent); color: var(--accent); box-shadow: 0 0 0 4px var(--accent-soft); }
.pp-step .lbl { font-size: 0.72rem; letter-spacing: 0.01em; color: var(--muted); margin-top: 7px; text-align: center; font-weight: 600; }
.pp-step.active .lbl, .pp-step.done .lbl { color: var(--ink); }
.pp-step:not(:last-child)::after {
  content: ""; position: absolute; top: 15px; left: 50%; width: 100%; height: 2px; background: var(--line); z-index: 1;
}
.pp-step.done:not(:last-child)::after { background: var(--accent); }

/* ---- metrics ---- */
/* Streamlit clips a metric's value with an ellipsis if it's wider than the card (common with big
   rupee amounts). Force it to wrap onto a second line instead of ever hiding digits. */
div[data-testid="stMetricValue"] {
    font-size: 1.3rem; font-weight: 700; color: var(--ink);
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
    overflow-wrap: break-word; word-break: break-word; line-height: 1.25;
}
div[data-testid="stMetricValue"] > div {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}
div[data-testid="stMetricLabel"] {
    font-weight: 600; color: var(--muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.03em;
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}
div[data-testid="stMetric"] {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px 18px;
    box-shadow: var(--shadow); min-height: 0;
}

.reason-chip {
    background: var(--accent-soft); border-left: 3px solid var(--accent); border-radius: 10px;
    padding: 12px 16px; margin-bottom: 10px; font-size: 0.97rem; color: var(--ink);
}
.goal-card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; box-shadow: var(--shadow); }

.pp-hero {
  background: linear-gradient(135deg, var(--accent-dark), var(--accent)); color: #fff; border-radius: 18px;
  padding: 40px 44px; margin-bottom: 10px; box-shadow: var(--shadow);
}
.pp-hero h1 { color: #fff !important; margin-bottom: 6px; }
.pp-hero p { color: #E4E3FB; font-size: 1.05rem; max-width: 640px; }
.pp-hero-steps { display: flex; gap: 28px; margin-top: 22px; flex-wrap: wrap; }
.pp-hero-steps .n { font-size: 1.5rem; color: #C7D2FE; font-weight: 800; }
.pp-hero-steps .t { font-size: 0.82rem; color: #E4E3FB; max-width: 150px; }

/* ---- existing vs target mix bars (plain-language combined-mix section) ---- */
.mix-row { margin-bottom: 16px; }
.mix-row-head { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
.mix-cat { font-weight: 700; font-size: 0.92rem; color: var(--ink); }
.mix-badge {
  font-size: 0.68rem; font-weight: 700; color: var(--teal); background: var(--teal-soft);
  border-radius: 999px; padding: 2px 9px; text-transform: uppercase; letter-spacing: 0.02em;
}
.mix-bar-track {
  position: relative; height: 14px; background: var(--bg); border: 1px solid var(--line);
  border-radius: 999px; overflow: visible;
}
.mix-bar-fill {
  position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent); border-radius: 999px;
}
.mix-bar-target {
  position: absolute; top: -4px; bottom: -4px; width: 3px; background: #1A1D29; border-radius: 2px;
  transform: translateX(-1.5px);
}
.mix-row-foot { display: flex; justify-content: space-between; gap: 10px; margin-top: 5px; font-size: 0.78rem; color: var(--muted); flex-wrap: wrap; }
.mix-row-foot b { color: var(--ink); }
.mix-target-lbl { color: var(--ink); }

/* ---- alternative-allocation comparison cards ---- */
.alt-card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; margin-bottom: 14px; box-shadow: var(--shadow); }
.alt-card-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; margin-bottom: 12px; }
.alt-name { font-weight: 700; font-size: 1.05rem; color: var(--ink); }
.alt-stats { font-size: 0.82rem; color: var(--muted); }
.alt-pill-row { display: flex; flex-wrap: wrap; gap: 8px; }
.alt-pill {
  display: flex; align-items: center; gap: 7px; background: var(--bg); border: 1px solid var(--line);
  border-radius: 999px; padding: 6px 12px; font-size: 0.82rem;
}
.alt-pill .a { font-weight: 600; color: var(--ink); }
.alt-pill .v { color: var(--muted); }
.alt-pill .d { font-weight: 700; padding: 1px 7px; border-radius: 999px; font-size: 0.76rem; }
.alt-pill .d.up { color: #0D9488; background: var(--teal-soft); }
.alt-pill .d.down { color: #B91C1C; background: #FEF2F2; }
.alt-pill .d.flat { color: var(--muted); background: var(--bg); }

button[kind="primary"], button[kind="primaryFormSubmit"] {
  background-color: var(--accent); border-color: var(--accent); border-radius: 10px; font-weight: 600;
}
button[kind="primary"]:hover { background-color: var(--accent-dark); border-color: var(--accent-dark); }
button[kind="secondary"] { border-radius: 10px; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h1 { font-size: 1.3rem !important; }
div[data-testid="stProgress"] div[role="progressbar"] > div:first-child { background-color: var(--accent); }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: var(--card); }
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent); border-bottom-color: var(--accent) !important; }
hr { border-color: var(--line) !important; }
</style>
"""


def stepper(active_label):
    """A horizontal numbered stepper shown at the top of every wizard page, in place of the old
    sidebar checklist - replaces WIZARD_STEPS progress with styled circles + connecting line."""
    idx = WIZARD_STEPS.index(active_label)
    cells = []
    for i, label in enumerate(WIZARD_STEPS):
        cls = "done" if i < idx else "active" if i == idx else ""
        mark = "✓" if i < idx else str(i + 1)
        cells.append(f'<div class="pp-step {cls}"><div class="dot">{mark}</div><div class="lbl">{label}</div></div>')
    st.markdown(f'<div class="pp-stepper">{"".join(cells)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ state helpers
def _blank_goal():
    return {"name": "", "goal_type": "Others", "vacation_country": C.VACATION_COUNTRIES[0],
            "cost": 0.0, "years": 1.0, "priority": "Important"}


def init_state():
    ss = st.session_state
    if "step" not in ss:
        ss.step = 0
    if "inputs" not in ss:
        ss.inputs = {
            "age": 28, "dependents": 0, "has_retirement": True, "retirement_age": 60,
            "location_tier": C.LOCATION_TIERS[0], "income_band": C.INCOME_BANDS[4][0],
            "vehicle_indicator": C.VEHICLE_INDICATORS[1],
            "mode": "SIP (monthly)", "amount": 10000, "sip_step_up_enabled": False, "sip_step_up": 0.0,
            "income_growth": 8.0, "stated_risk": "Moderate",
            "quiz_idx": [None, None, None],
            "goals_list": [{"name": "Vacation", "goal_type": "Vacation", "vacation_country": "Southeast Asia",
                            "cost": 150000.0, "years": 2.0, "priority": "Important"}],
            "employment": "Salaried", "emi_total": 0, "emi_high_interest": False,
            "housing_situation": C.HOUSING_SITUATIONS[0],
            "use_exact_financials": False, "monthly_income": 0, "monthly_expenses": 0,
            "existing_by_category": {c: 0 for c in C.EXISTING_CATEGORIES},
            "emergency_savings": 0,
            "has_health_insurance": True, "has_life_insurance": True, "sole_earner": False,
            "liquidity_need": "Some",
        }


def _num(x):
    return 0.0 if x is None or pd.isna(x) else float(x)


def _txt(x, default=""):
    s = "" if x is None or pd.isna(x) else str(x).strip()
    return s or default


def resolve_income_expenses(i):
    """Brief items 1-2: income/expenses come from the easy categorical questions by default - never
    asked as exact numbers unless the user explicitly opts in via 'I'd rather enter exact values'."""
    if i.get("use_exact_financials"):
        return float(i.get("monthly_income") or 0), float(i.get("monthly_expenses") or 0), False, None
    mi, me, note = R.estimate_income_expenses(i)
    return (mi or 0.0), (me or 0.0), True, note


def engine_inputs():
    """Convert what the user typed into the plain dict the engine expects."""
    i = st.session_state.inputs
    goals = [{
        "name": _txt(g.get("name")), "cost": _num(g.get("cost")), "years": _num(g.get("years")),
        "priority": _txt(g.get("priority"), "Important"), "goal_type": _txt(g.get("goal_type"), C.DEFAULT_GOAL_TYPE),
        "vacation_country": g.get("vacation_country"),
    } for g in i.get("goals_list", [])]
    quiz = [R.QUIZ[k]["options"][idx][1] for k, idx in enumerate(i["quiz_idx"]) if idx is not None]
    monthly_income, monthly_expenses, is_estimated, note = resolve_income_expenses(i)
    return {
        "age": int(i["age"]), "dependents": int(i["dependents"]), "has_retirement": bool(i["has_retirement"]),
        "retirement_age": int(i["retirement_age"]), "monthly_expenses": float(monthly_expenses),
        "mode": i["mode"], "amount": float(i["amount"]),
        "sip_step_up_enabled": bool(i.get("sip_step_up_enabled", False)),
        "sip_step_up": float(i.get("sip_step_up") or 0.0) / 100.0,
        "income_growth": float(i.get("income_growth") or 0.0) / 100.0,
        "goals": goals, "stated_risk": i["stated_risk"], "quiz": quiz,
        "employment": i["employment"], "monthly_income": float(monthly_income), "emi_total": float(i["emi_total"]),
        "emi_high_interest": bool(i["emi_high_interest"]),
        "existing_by_category": dict(i.get("existing_by_category") or {}),
        "emergency_savings": float(i["emergency_savings"]),
        "has_health_insurance": bool(i["has_health_insurance"]), "has_life_insurance": bool(i["has_life_insurance"]),
        "sole_earner": bool(i["sole_earner"]), "liquidity_need": i["liquidity_need"],
        "location_tier": i["location_tier"], "housing_situation": i["housing_situation"],
        "vehicle_indicator": i["vehicle_indicator"], "income_band": i["income_band"],
        "income_expense_estimated": is_estimated, "income_expense_note": note,
    }


@st.cache_data(show_spinner=False)
def cached_plan(sig, override, mult):
    return A.build_plan(json.loads(sig), override=override, amount_mult=mult)


def go(step):
    st.session_state.step = step
    st.rerun()


def suffix(mode):
    return "/month" if mode.startswith("SIP") else ""


# ------------------------------------------------------------------ chart helpers (kept minimal per brief: "remove unnecessary graphs")
def donut(alloc):
    df = pd.DataFrame({"Asset": list(alloc.keys()), "Percent": list(alloc.values())})
    df = df[df["Percent"] > 0]
    return (alt.Chart(df).mark_arc(innerRadius=55)
            .encode(theta=alt.Theta("Percent:Q"),
                    color=alt.Color("Asset:N", scale=ASSET_SCALE, legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Asset:N", alt.Tooltip("Percent:Q", format=".1f")])
            .properties(height=240))


def net_worth_chart(projection):
    df = pd.DataFrame(projection)
    return (alt.Chart(df).mark_line(point=True, color="#1B4971")
            .encode(x=alt.X("age:Q", title="Age"),
                    y=alt.Y("wealth:Q", title="Total wealth (Rs)"),
                    tooltip=[alt.Tooltip("age:Q", title="Age"), alt.Tooltip("wealth:Q", title="Wealth", format=",.0f")])
            .properties(height=260, width="container"))


# ------------------------------------------------------------------ table / text helpers
def goal_status(g):
    if g.get("target") is None:
        return "On Track"
    fr = g.get("funded_ratio")
    if fr is None or g.get("funding", 0) <= 0:
        return "Currently Unfunded"
    return "On Track" if fr >= 0.95 else "Needs Attention"


GOAL_STATUS_COLOR = {"On Track": "🟢", "Needs Attention": "🟡", "Currently Unfunded": "🔴"}


def goals_table(plan):
    opt = plan["options"][plan["recommended"]]
    rows = []
    for g, r in zip(plan["goals"], opt["goals"]):
        status = goal_status(g)
        rows.append({
            "Goal": g["name"], "Type": g.get("goal_type", "-"), "Priority": g.get("priority", "-"),
            "Years": f"{g['years']:g}",
            "Target (future)": fmt_inr(g["target"]) if g["target"] else "Open-ended",
            "Allocated": fmt_inr(g["funding"]) + suffix(plan["mode"]),
            "Funded": f"{g['funded_ratio'] * 100:.0f}%" if g.get("funded_ratio") is not None else "-",
            "Equity share": f"{r['equity_pct']:.0f}%",
            "Status": f"{GOAL_STATUS_COLOR[status]} {status}",
        })
    return pd.DataFrame(rows)


def per_goal_alloc_table(plan):
    opt = plan["options"][plan["recommended"]]
    rows = []
    for r in opt["goals"]:
        row = {"Goal": r["goal"]}
        row.update({a: f"{v:.0f}%" for a, v in r["alloc"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def alt_comparison_df(plan, alt_name):
    """Alternative allocation shown as weight CHANGES vs the current (recommended) allocation."""
    rec = plan["recommended"]
    rec_alloc = plan["options"][rec]["overall_alloc"]
    alt_alloc = plan["options"][alt_name]["overall_alloc"]
    rows = []
    for a in C.ASSETS:
        r, v = rec_alloc[a], alt_alloc[a]
        if r < 0.5 and v < 0.5:
            continue
        delta = v - r
        change = "—" if abs(delta) < 0.5 else (f"↑{delta:.0f} pts" if delta > 0 else f"↓{abs(delta):.0f} pts")
        rows.append({"Asset": a, f"{rec} (current)": f"{r:.0f}%", alt_name: f"{v:.0f}%", "Change": change})
    return pd.DataFrame(rows)


def alt_comparison_rows(plan, alt_name):
    """Same data as alt_comparison_df, as plain rows for the card-style comparison layout."""
    rec = plan["recommended"]
    rec_alloc = plan["options"][rec]["overall_alloc"]
    alt_alloc = plan["options"][alt_name]["overall_alloc"]
    rows = []
    for a in C.ASSETS:
        r, v = rec_alloc[a], alt_alloc[a]
        if r < 0.5 and v < 0.5:
            continue
        delta = v - r
        rows.append({"asset": a, "rec": r, "alt": v, "delta": delta})
    return rows


def live_feed_card():
    """A prominent, always-visible (not tucked in a collapsed sidebar expander) card showing the
    actual Yahoo Finance ticker, last price and as-of date this deployment fetched - meant to be
    pulled up next to Google Finance / NSE on another tab so someone can visually confirm the
    number is real and current, and to show the refresh happening live with a fresh timestamp."""
    meta = MD.read_meta()
    live = meta.get("any_live_data_used")
    prices = meta.get("latest_prices") or {}
    when = meta.get("last_refresh_attempt_utc")
    with st.container(border=True):
        h = st.columns([4, 1])
        h[0].markdown("#### 📡 Live market feed")
        if h[1].button("🔄 Refresh now", key="live_feed_refresh"):
            with st.spinner("Fetching live prices from Yahoo Finance..."):
                MD.refresh()
            importlib.reload(C)
            importlib.reload(A)
            st.cache_data.clear()
            st.rerun()
        if when:
            local_when = datetime.fromisoformat(when).strftime("%d %b %Y, %H:%M:%S UTC")
            st.caption(f"Last fetch attempt: {local_when}")
        if live and prices:
            st.success(f"Live prices fetched from Yahoo Finance for {len(prices)} of "
                       f"{len(MD.TICKERS)} assets - compare these against Google Finance / NSE "
                       "right now to confirm.", icon="✅")
            rows = [{"Asset": asset, "Ticker": p["ticker"], "Last price": f"{p['price']:,.2f}",
                     "As of": p["as_of"]} for asset, p in prices.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        elif when:
            st.warning("No live prices came back on the last fetch - showing calibrated "
                      "assumptions instead. This happens when the deployment has no outbound "
                      "internet access (e.g. inside a sandboxed dev environment) or Yahoo "
                      "Finance blocked/rejected the request.", icon="⚠️")
            errs = meta.get("fetch_errors") or {}
            if errs:
                with st.popover("Why did these fail?"):
                    for asset, err in errs.items():
                        st.caption(f"**{asset}:** {err}")
        else:
            st.info("No refresh has run yet in this session - click 'Refresh now' above.", icon="ℹ️")


def combined_mix_section(plan, opt):
    combined = opt.get("combined")
    if not combined or combined["existing_total"] <= 0:
        st.caption("No existing investments entered in Step 2 - the recommended mix applies directly to your new investment.")
        return

    existing_total, new_amount, combined_total = combined["existing_total"], combined["new_amount"], combined["combined_total"]
    period_note = "(one month's SIP)" if plan["mode"].startswith("SIP") else "(this lumpsum)"
    st.markdown(
        f'<div class="reason-chip">You already have <b>{fmt_inr(existing_total)}</b> invested. '
        f"This period you're adding <b>{fmt_inr(new_amount)}</b> {period_note}. "
        f"Added together, that's <b>{fmt_inr(existing_total)} + {fmt_inr(new_amount)} = {fmt_inr(combined_total)}</b>.</div>",
        unsafe_allow_html=True,
    )

    # Figure out where this period's new money is actually going, for one plain-English callout -
    # avoids making the person read a whole row-by-row table just to find out.
    new_money_alloc = combined.get("new_money_alloc") or {}
    going_to = sorted(((cat, v) for cat, v in new_money_alloc.items() if v >= 0.5),
                      key=lambda x: -x[1])
    if going_to:
        going_str = ", ".join(f"{cat} ({v:.0f}%)" for cat, v in going_to)
        st.markdown(f'<div class="reason-chip">💰 This period\'s {fmt_inr(new_amount)} will go toward: '
                   f"<b>{going_str}</b> - the category(ies) you're currently furthest below target in.</div>",
                   unsafe_allow_html=True)

    bar_rows = []
    for cat in C.EXISTING_CATEGORIES:
        cur = (combined["current_mix"] or {}).get(cat, 0.0)
        tgt = combined["target_mix"].get(cat, 0.0)
        new = new_money_alloc.get(cat, 0.0)
        comb = combined["combined_mix"].get(cat, 0.0)
        if max(cur, tgt, new, comb) < 0.5:
            continue
        bar_rows.append((cat, cur, tgt, new, comb))

    bars_html = []
    for cat, cur, tgt, new, comb in bar_rows:
        bars_html.append(f"""
        <div class="mix-row">
          <div class="mix-cat">{cat}</div>
          <div class="mix-bar-track">
            <div class="mix-bar-fill" style="width:{min(cur, 100):.1f}%"></div>
            <div class="mix-bar-target" style="left:{min(tgt, 100):.1f}%" title="Target: {tgt:.0f}%"></div>
          </div>
          <div class="mix-row-foot"><span>You: <b>{cur:.0f}%</b></span><span class="mix-target-lbl">◆ Target: <b>{tgt:.0f}%</b></span></div>
        </div>
        """)
    st.markdown("".join(bars_html), unsafe_allow_html=True)
    st.caption("Each bar shows where you are today. The ◆ mark shows where you ideally should be. New "
              "contributions are steered toward whichever categories are furthest below their mark.")

    with st.expander("See the exact numbers"):
        rows = [{"Category": cat, "Existing mix": f"{cur:.1f}%", "Target mix": f"{tgt:.1f}%",
                "New money →": f"{new:.1f}%", "Combined mix": f"{comb:.1f}%"}
                for cat, cur, tgt, new, comb in bar_rows]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("'Combined mix' = existing holdings + new money blended by rupee value. 'New money' is how "
                  "this period's investment should be split so the combined portfolio drifts toward the target "
                  "mix over time - existing holdings are never 'sold' to get there.")


def action_plan(plan):
    tips = []
    for g in plan["goals"]:
        if g.get("required") and g.get("funded_ratio") is not None and g["funded_ratio"] < 0.999:
            gap = max(0.0, g["required"] - g["funding"])
            tips.append(f"'{g['name']}': increase funding by about {fmt_inr(gap)}{suffix(plan['mode'])}, "
                        "or push the goal date out, to fully fund it.")
    snap = plan["snapshot"]
    if snap["emergency_shortfall"] > 0:
        tips.append(f"Advisory: build your emergency fund by about {fmt_inr(snap['emergency_shortfall'])} more, "
                    "kept separately in a liquid fund (this does not affect the plan above).")
    if snap["high_interest_debt"]:
        tips.append("Prioritise repaying high-interest debt before increasing investments further.")
    if not snap["has_health_insurance"]:
        tips.append("Consider adequate health insurance coverage as part of your overall plan.")
    if plan["financial_health"].get("Insurance Protection") == "Needs Review" and snap["has_health_insurance"]:
        tips.append("Consider adequate life insurance cover, since you are a primary earner with dependents.")
    if not tips:
        tips.append("Your plan looks on track based on what you've entered. Revisit it if your goals, income or dependents change.")
    return tips


def plan_markdown(plan, explanation):
    prof = plan["profile"]
    pc = plan["projected_corpus"]
    lines = ["# My goal-based investment plan", "", f"_{DISCLAIMER}_", "",
             f"**Investing:** {fmt_inr(plan['amount'])}{suffix(plan['mode'])} ({plan['mode']})  ",
             f"**Risk score:** {prof['score']}/10 - suggested profile: **{plan['recommended']}**  ",
             f"**Projected value:** {fmt_inr(pc['median'])} (weak market {fmt_inr(pc['weak_market'])} - "
             f"strong market {fmt_inr(pc['strong_market'])})  ",
             "", "## Explanation", explanation, "", "## Financial health"]
    for k, v in plan["financial_health"].items():
        lines.append(f"- **{k}:** {v}")
    lines += ["", "## Goals"]
    for _, row in goals_table(plan).iterrows():
        lines.append(f"- **{row['Goal']}** ({row['Type']}, {row['Priority']}, {row['Years']} yrs): target "
                     f"{row['Target (future)']}, allocated {row['Allocated']}, funded {row['Funded']}, "
                     f"equity {row['Equity share']}, {row['Status']}")
    lines += ["", "## The three options (overall mix)"]
    for n, o in plan["options"].items():
        mix = ", ".join(f"{a} {v:.0f}%" for a, v in o["overall_alloc"].items() if v > 0)
        lines.append(f"- **{n}{' (suggested)' if n == plan['recommended'] else ''}:** {mix} | expected return "
                     f"{o['exp_return'] * 100:.1f}% | Sharpe {o['sharpe']:.2f}")
    lines += ["", "## Action plan"] + [f"- {t}" for t in action_plan(plan)]
    lines += ["", "## Assumptions", plan["assumptions_note"]]
    return "\n".join(lines)


# ------------------------------------------------------------------ sidebar
def sidebar():
    st.sidebar.markdown('<div style="font-family:Inter,sans-serif; font-size:1.3rem; font-weight:800; '
                        'color:#4F46E5;">◆ Portfolio Planner</div>', unsafe_allow_html=True)
    st.sidebar.caption("A goal-based plan you can understand, with the reasoning shown alongside it.")
    st.sidebar.divider()
    with st.sidebar.expander("📡 Data status", expanded=False):
        meta = MD.read_meta()
        if meta.get("any_live_data_used"):
            st.success("Using live market prices", icon="✅")
        else:
            st.info("Using calibrated assumptions (no live fetch yet)", icon="ℹ️")
        when = meta.get("last_refresh_attempt_utc")
        if when:
            st.caption(f"Last refresh attempt: {when} UTC")
        live = meta.get("live_assets_fetched") or []
        if live:
            st.caption("Fetched live: " + ", ".join(live))
        errs = meta.get("fetch_errors") or {}
        if errs:
            st.caption("Not fetched (using calibrated values instead): " + ", ".join(errs.keys()))
            with st.popover("Why did these fail?"):
                for asset, err in errs.items():
                    st.caption(f"**{asset}:** {err}")
        st.caption(meta.get("note", ""))
        if st.button("🔄 Refresh market data now", key="manual_refresh"):
            with st.spinner("Fetching live prices..."):
                MD.refresh()
            importlib.reload(C)
            importlib.reload(A)
            st.cache_data.clear()
            st.rerun()
    with st.sidebar.expander("Assumptions & methodology"):
        st.caption("Return & volatility, calculated from 10 years of representative monthly data per asset class:")
        st.dataframe(pd.DataFrame([{"Asset": a, "Exp. return": f"{v[0] * 100:.1f}%", "Volatility": f"{v[1] * 100:.1f}%"}
                                   for a, v in C.ASSUMPTIONS.items()]), hide_index=True)
        st.caption(C.ASSUMPTIONS_NOTE)
        st.caption("Inflation by goal type:")
        st.dataframe(pd.DataFrame([{"Goal type": k, "Assumed inflation": f"{v*100:.0f}%"} for k, v in C.GOAL_INFLATION.items()]),
                    hide_index=True)
        st.caption(f"Planning margin {C.PLANNING_MARGIN * 100:.0f} pt | risk-free rate (for Sharpe) {C.RISK_FREE_RATE*100:.1f}% | "
                   f"{C.N_SIMS:,} simulations | correlation shrinkage {C.CORR_SHRINKAGE*100:.0f}% | emergency-fund "
                   "months and EMI-burden threshold are stated planning heuristics - see config.py. The emergency "
                   "fund is advisory only and never reduces the amount invested.")
    st.sidebar.caption(DISCLAIMER)


# ------------------------------------------------------------------ step 0: how it works
def step_intro():
    steps_html = "".join(
        f'<div><div class="n">{n:02d}</div><div class="t">{label}</div></div>'
        for n, label in enumerate([
            "Tell us about your finances", "Add your financial goals", "Understand your risk profile",
            "Get your personalised portfolio", "Compare alternative strategies"], start=1))
    st.markdown(f"""
    <div class="pp-hero">
      <h1>Build your plan around your life.</h1>
      <p>Tell us what you're investing for, and we'll turn it into a portfolio you can understand -
      not just a number, but the reasoning behind it.</p>
      <div class="pp-hero-steps">{steps_html}</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("Build my plan →", type="primary", key="intro_start"):
        go(1)


# ------------------------------------------------------------------ step 1: about you + income
def step_about():
    inp = st.session_state.inputs
    stepper("About you")
    st.subheader("About you")
    c1, c2 = st.columns(2)
    age = c1.number_input("Your age", 18, 70, int(inp["age"]), key="w_age")
    deps = c2.number_input("People who depend on you financially (spouse, children, parents)", 0, 10,
                           int(inp["dependents"]), key="w_deps",
                           help="Household responsibilities affect how much risk your situation can really afford.")
    has_ret = st.checkbox("I also want to plan for retirement", value=bool(inp["has_retirement"]), key="w_has_ret")
    ret_age = inp["retirement_age"]
    if has_ret:
        ret_age = st.number_input("Retirement age", 35, 80, max(int(inp["retirement_age"]), 35), key="w_ret_age")
    if deps > 0:
        sole = st.checkbox("I am the sole or primary income earner in my household", value=bool(inp["sole_earner"]), key="w_sole")
    else:
        sole = False

    st.markdown("**Your income** (instead of typing an exact figure, a few easy questions estimate it for you)")
    c3, c4, c5 = st.columns(3)
    location_tier = c3.selectbox("Where do you live?", C.LOCATION_TIERS,
                                 index=C.LOCATION_TIERS.index(inp["location_tier"]) if inp["location_tier"] in C.LOCATION_TIERS else 0,
                                 key="w_location")
    band_labels = [b[0] for b in C.INCOME_BANDS]
    income_band = c4.selectbox("Monthly income band", band_labels,
                               index=band_labels.index(inp["income_band"]) if inp["income_band"] in band_labels else 4,
                               key="w_income_band")
    vehicle = c5.selectbox("Vehicle", C.VEHICLE_INDICATORS,
                           index=C.VEHICLE_INDICATORS.index(inp["vehicle_indicator"]) if inp["vehicle_indicator"] in C.VEHICLE_INDICATORS else 1,
                           key="w_vehicle")
    employment = st.selectbox("Employment type", C.EMPLOYMENT_TYPES,
                              index=C.EMPLOYMENT_TYPES.index(inp["employment"]) if inp["employment"] in C.EMPLOYMENT_TYPES else 0,
                              key="w_employment")

    if st.button("Next →", type="primary", key="next0"):
        inp.update(age=int(age), dependents=int(deps), has_retirement=bool(has_ret), retirement_age=int(ret_age),
                   sole_earner=bool(sole), location_tier=location_tier, income_band=income_band,
                   vehicle_indicator=vehicle, employment=employment)
        go(2)


# ------------------------------------------------------------------ step 2: financial life
def step_financial_life():
    inp = st.session_state.inputs
    stepper("Financial life")
    st.subheader("Your financial life")
    st.caption("This helps us check what you can actually afford to invest, and whether anything needs attention "
               "before building a long-term portfolio.")

    housing = st.selectbox("Housing situation", C.HOUSING_SITUATIONS,
                           index=C.HOUSING_SITUATIONS.index(inp["housing_situation"]) if inp["housing_situation"] in C.HOUSING_SITUATIONS else 0,
                           key="w_housing")

    use_exact = st.checkbox("I'd rather enter my exact income/expenses", value=bool(inp.get("use_exact_financials")), key="w_use_exact")

    preview_inp = {**inp, "housing_situation": housing}
    if use_exact:
        c1, c2 = st.columns(2)
        income = c1.number_input("Your monthly take-home income (Rs)", 0, 10_000_000,
                                 int(inp.get("monthly_income") or 0), step=1000, key="w_income_exact")
        expenses = c2.number_input("Monthly household expenses (Rs)", 0, 10_000_000,
                                   int(inp.get("monthly_expenses") or 0), step=1000, key="w_expenses_exact")
    else:
        est_income, est_expenses, note = R.estimate_income_expenses(preview_inp)
        income, expenses = est_income, est_expenses
        c1, c2 = st.columns(2)
        c1.metric("Estimated monthly income", fmt_inr(income))
        c2.metric("Estimated monthly expenses", fmt_inr(expenses))
        with st.expander("How was this estimated?"):
            st.caption(note)

    income_growth = st.slider("Expected annual increase in your income (%)", 0, 20, int(inp.get("income_growth") or 0), 1,
                              key="w_income_growth",
                              help="If your income is likely to grow, your disposable income and affordable SIP can grow with it.")
    if income_growth > 0:
        exp_growth = A.expense_growth_rate(income_growth / 100.0)
        st.caption(f"We'll assume your expenses grow too - at {C.INFLATION*100:.0f}%/yr inflation, plus a bit more "
                   f"({C.LIFESTYLE_CREEP_FACTOR*100:.0f}% of any raise above inflation) - about {exp_growth*100:.1f}%/yr in total.")

    c3, c4 = st.columns(2)
    emi = c3.number_input("Total monthly EMIs / loan payments (Rs)", 0, 10_000_000, int(inp["emi_total"]), step=500, key="w_emi")
    emi_high = c4.checkbox("Some of this debt is high-interest (e.g. credit card, personal loan)",
                           value=bool(inp["emi_high_interest"]), key="w_emi_high") if emi > 0 else False

    st.markdown("**Existing investments** (so we don't recommend a portfolio that ignores what you already hold)")
    ex = dict(inp.get("existing_by_category") or {})
    cols = st.columns(len(C.EXISTING_CATEGORIES))
    existing_by_category = {}
    for col, cat in zip(cols, C.EXISTING_CATEGORIES):
        existing_by_category[cat] = col.number_input(cat, 0, 100_000_000, int(ex.get(cat, 0)), step=5000,
                                                      key=f"w_existing_{cat}")

    st.markdown("**Emergency fund** (advisory only - shown as a recommendation, never subtracted from your invested amount)")
    emergency_savings = st.number_input("Money already set aside as an emergency fund (Rs)", 0, 100_000_000,
                                        int(inp["emergency_savings"]), step=5000, key="w_emergency")
    months, target = R.emergency_fund_target({**inp, "employment": inp["employment"], "monthly_expenses": expenses})
    if expenses > 0:
        st.caption(f"Based on your employment type and dependents, a buffer of about {months:g} months of expenses "
                   f"(~{fmt_inr(target)}) is a reasonable target. This is a recommendation only.")

    liquidity = st.radio("How much of this investment might you need to access unexpectedly?",
                         ["Very little", "Some", "Significant"],
                         index=["Very little", "Some", "Significant"].index(inp["liquidity_need"]), horizontal=True, key="w_liquidity")

    if inp["dependents"] > 0:
        st.markdown("**Protection check** (we don't recommend insurance products - just flag a gap if one exists)")
        c7, c8 = st.columns(2)
        health = c7.checkbox("I have health insurance", value=bool(inp["has_health_insurance"]), key="w_health_ins")
        life = c8.checkbox("I have life insurance / term cover", value=bool(inp["has_life_insurance"]), key="w_life_ins")
    else:
        health, life = inp["has_health_insurance"], inp["has_life_insurance"]

    def _save():
        inp.update(housing_situation=housing, use_exact_financials=bool(use_exact),
                   monthly_income=float(income), monthly_expenses=float(expenses),
                   income_growth=float(income_growth), emi_total=float(emi), emi_high_interest=bool(emi_high),
                   existing_by_category=existing_by_category, emergency_savings=float(emergency_savings),
                   liquidity_need=liquidity, has_health_insurance=bool(health), has_life_insurance=bool(life))

    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("← Back", key="back_fl"):
        _save()
        go(1)
    if c2.button("Next →", type="primary", key="next_fl"):
        _save()
        go(3)


# ------------------------------------------------------------------ step 3: money & goals
def step_money():
    inp = st.session_state.inputs
    stepper("Money & goals")
    st.subheader("Your money and goals")
    mode = st.radio("How will you invest?", ["SIP (monthly)", "Lumpsum"], horizontal=True, key="w_mode",
                    index=0 if inp["mode"].startswith("SIP") else 1)
    label = "Monthly SIP amount (Rs)" if mode.startswith("SIP") else "One-time amount to invest (Rs)"
    amount = st.number_input(label, 0, 100_000_000, int(inp["amount"]), step=1000, key="w_amount")
    step_up_enabled = inp.get("sip_step_up_enabled", False)
    step_up = inp["sip_step_up"]
    if mode.startswith("SIP"):
        step_up_enabled = st.checkbox("Increase my SIP every year", value=bool(step_up_enabled), key="w_stepup_on")
        if step_up_enabled:
            step_up = st.slider("Annual increase (%)", 0, 25, int(inp["sip_step_up"]), 1, key="w_stepup")

    st.markdown("**Your goals**")
    st.caption("Enter the cost in *today's* rupees; we adjust for inflation using a rate that depends on the goal "
               "type. Use cost 0 for an open-ended goal. Set priority so we know what to fund first if your amount "
               "can't cover everything.")

    goals_list = list(inp.get("goals_list") or [])
    if not goals_list:
        goals_list = [_blank_goal()]
    current = []
    remove_idx = None
    type_counts = {}
    for i, g in enumerate(goals_list):
        with st.container(border=True):
            c2, c3 = st.columns([1.5, 1])
            goal_type = c2.selectbox("Goal type", C.GOAL_TYPES,
                                     index=C.GOAL_TYPES.index(g.get("goal_type")) if g.get("goal_type") in C.GOAL_TYPES else len(C.GOAL_TYPES) - 1,
                                     key=f"g_type_{i}")
            priority = c3.selectbox("Priority", C.PRIORITIES,
                                    index=C.PRIORITIES.index(g.get("priority")) if g.get("priority") in C.PRIORITIES else 1,
                                    key=f"g_prio_{i}")
            vacation_country = g.get("vacation_country") or C.VACATION_COUNTRIES[0]
            if goal_type == "Vacation":
                c4, c5, c6 = st.columns([1.3, 1, 1])
                vacation_country = c4.selectbox("Country / region", C.VACATION_COUNTRIES,
                                                index=C.VACATION_COUNTRIES.index(vacation_country) if vacation_country in C.VACATION_COUNTRIES else 0,
                                                key=f"g_country_{i}")
                cost = c5.number_input("Cost today (Rs)", 0, step=10000, value=int(g.get("cost", 0)), key=f"g_cost_{i}")
                years = c6.number_input("Years to go", 0.5, 50.0, float(g.get("years", 1.0)), step=0.5, key=f"g_years_{i}")
            else:
                c5, c6 = st.columns(2)
                cost = c5.number_input("Cost today (Rs, 0 = open-ended)", 0, step=10000, value=int(g.get("cost", 0)), key=f"g_cost_{i}")
                years = c6.number_input("Years to go", 0.5, 50.0, float(g.get("years", 1.0)), step=0.5, key=f"g_years_{i}")
            # Goal name is auto-derived from the goal type (no free-text field) - "Vacation" uses the
            # chosen country, and repeated types get a counter suffix so each goal stays distinguishable.
            base_name = f"{goal_type} ({vacation_country})" if goal_type == "Vacation" else goal_type
            type_counts[base_name] = type_counts.get(base_name, 0) + 1
            name = base_name if type_counts[base_name] == 1 else f"{base_name} #{type_counts[base_name]}"
            current.append({"name": name, "goal_type": goal_type, "vacation_country": vacation_country,
                            "cost": float(cost), "years": float(years), "priority": priority})
            if len(goals_list) > 1 and st.button("🗑️ Remove this goal", key=f"g_remove_{i}"):
                remove_idx = i

    if remove_idx is not None:
        current.pop(remove_idx)
        inp["goals_list"] = current
        st.rerun()

    if st.button("➕ Add another goal", key="add_goal"):
        inp["goals_list"] = current + [_blank_goal()]
        st.rerun()

    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("← Back", key="back1"):
        inp["goals_list"] = current
        go(2)
    if c2.button("Next →", type="primary", key="next1"):
        inp.update(mode=mode, amount=int(amount), sip_step_up_enabled=bool(step_up_enabled),
                   sip_step_up=float(step_up), goals_list=current)
        go(4)


# ------------------------------------------------------------------ step 4: risk comfort
def step_risk():
    inp = st.session_state.inputs
    stepper("Risk comfort")
    st.subheader("How do you feel about risk?")
    levels = ["Low", "Moderate", "High"]
    stated = st.radio("In general, how would you describe your risk appetite?", levels,
                      index=levels.index(inp["stated_risk"]), horizontal=True, key="w_stated")
    st.markdown("Now three quick scenarios. Your answers tell us more than a label does.")
    idxs = []
    for k, q in enumerate(R.QUIZ):
        opts = [o[0] for o in q["options"]]
        choice = st.radio(f"{k + 1}. {q['q']}", opts, index=inp["quiz_idx"][k], key=f"w_q{k}")
        idxs.append(None if choice is None else opts.index(choice))
    complete = all(i is not None for i in idxs)
    if not complete:
        st.caption("Please answer all three scenarios to continue.")
    c1, c2, _ = st.columns([1, 2, 3])
    if c1.button("← Back", key="back2"):
        inp.update(stated_risk=stated, quiz_idx=idxs)
        go(3)
    if c2.button("Review my profile →", type="primary", key="next2", disabled=not complete):
        inp.update(stated_risk=stated, quiz_idx=idxs)
        go(5)


# ------------------------------------------------------------------ step 5: review
def step_review():
    einp = engine_inputs()
    errors, warnings = R.validate(einp)
    stepper("Review")
    st.subheader("Review · here is what we understood")
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)
    if not errors:
        goals = R.build_goals(einp)
        prof = R.risk_profile(einp, goals)
        c = st.columns(4)
        c[0].metric("Risk appetite", f"{prof['willingness']}/10")
        c[1].metric("Risk capacity", f"{prof['capacity']}/10")
        c[2].metric("Suggested profile", prof["label"])
        c[3].metric("Your score", f"{prof['score']}/10")
        st.progress(min(prof["score"] / 10, 1.0))
        st.markdown(f"**Investing:** {fmt_inr(einp['amount'])}{suffix(einp['mode'])} ({einp['mode']})")
        if einp.get("income_expense_estimated"):
            st.caption(f"Estimated income Rs {einp['monthly_income']:,.0f}/month, estimated expenses "
                       f"Rs {einp['monthly_expenses']:,.0f}/month (from your Step 2 answers).")
        if goals:
            st.dataframe(pd.DataFrame([{
                "Goal": g["name"], "Type": g.get("goal_type", "-"), "Priority": g.get("priority", "-"),
                "Years to go": f"{g['years']:g}",
                "Target (future Rs)": fmt_inr(g["target"]) if g["target"] else "Open-ended"} for g in goals]),
                hide_index=True)
        preview = cached_plan(json.dumps(einp, sort_keys=True, default=str), None, 1.0)
        for n in preview["notes"][:4]:
            st.info(n)
    c1, c2, _ = st.columns([1, 2, 3])
    if c1.button("← Edit answers", key="back3"):
        go(4)
    if c2.button("Generate my plan →", type="primary", key="next3", disabled=bool(errors)):
        go(6)


# ------------------------------------------------------------------ step 6: results
def get_explanation(plan, sig, api_key, model, force=False):
    ex = st.session_state.get("expl")
    if force or not ex or ex["sig"] != sig:
        with st.spinner("Writing your personalised explanation..."):
            text, used, err = D.generate_explanation(plan, api_key, model)
        st.session_state.expl = {"sig": sig, "text": text, "used_ai": used, "err": err}
    return st.session_state.expl


def render_reasoning_chips(markdown_text):
    lines = markdown_text.split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().startswith("### Why this fits you")) + 1
    except StopIteration:
        return "", markdown_text
    end = next((i for i in range(start, len(lines)) if lines[i].strip().startswith("###")), len(lines))
    bullets = [l.strip("- ").strip() for l in lines[start:end] if l.strip().startswith("-")]
    rest = "\n".join(lines[end:])

    def _chip_inner(b):
        escaped = html.escape(b)
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)

    chips_html = "".join(f'<div class="reason-chip">{_chip_inner(b)}</div>' for b in bullets) if bullets else ""
    return chips_html, rest


def step_results(api_key, model):
    einp = engine_inputs()
    sig = json.dumps(einp, sort_keys=True, default=str)
    plan = cached_plan(sig, None, 1.0)
    prof, rec = plan["profile"], plan["recommended"]
    opt = plan["options"][rec]
    pc = plan["projected_corpus"]
    if st.session_state.get("chat_sig") != sig:
        st.session_state.chat, st.session_state.chat_sig = [], sig

    top = st.columns([5, 1, 1])
    top[0].title("Your plan is ready.")
    if top[1].button("✏️ Edit", key="edit_answers"):
        go(1)
    if top[2].button("↺ Restart", key="restart"):
        for k in ["step", "inputs", "expl", "chat", "chat_sig"]:
            st.session_state.pop(k, None)
        st.rerun()

    if einp.get("income_expense_estimated") and einp.get("income_expense_note"):
        st.caption(f"ℹ️ {einp['income_expense_note']}")

    live_feed_card()

    # ===================== PORTFOLIO AT A GLANCE (crisp, essentials first) =====================
    st.markdown("### Portfolio at a glance")
    g1 = st.columns(4)
    g1[0].metric("Profile", rec)
    g1[1].metric("Investing", fmt_inr(plan["amount"]) + suffix(plan["mode"]))
    g1[2].metric("Horizon", f"{prof['weighted_horizon_years']:g} years")
    g1[3].metric("Risk score", f"{prof['score']}/10")
    g2 = st.columns(3)
    g2[0].metric("Expected return / yr", f"{opt['exp_return']*100:.1f}%")
    g2[1].metric("Volatility / yr", f"{opt['volatility']*100:.1f}%")
    n_on_track = sum(1 for g in plan["goals"] if goal_status(g) == "On Track")
    g2[2].metric("Goals on track", f"{n_on_track} of {len(plan['goals'])}")

    st.markdown(f"##### Corpus impact in {pc.get('horizon_years', prof['weighted_horizon_years']):g} years - a range, not a single number")
    r1, r2, r3 = st.columns(3)
    r1.metric("Weak market", fmt_inr(pc["weak_market"]))
    r2.metric("Expected", fmt_inr(pc["median"]))
    r3.metric("Strong market", fmt_inr(pc["strong_market"]))
    st.caption(pc["definition"])

    # ===================== PORTFOLIO MIX (existing + combined) =====================
    st.markdown("### Your portfolio mix")
    left, right = st.columns(2)
    left.altair_chart(donut(opt["overall_alloc"]))
    left.caption(f"Recommended ({rec}) mix for new money, across {len(C.ASSETS)} asset classes.")
    with right:
        combined_mix_section(plan, opt)

    show_mvo = st.checkbox("Show the portfolio optimisation (MVO) calculation", key="show_mvo")
    if show_mvo:
        st.caption("Solved by Mean-Variance Optimization: maximise `w'μ - 0.5·λ·w'Σw` subject to weights summing "
                   "to 100%, no negative weights, an equity-exposure cap for this goal's horizon, and per-asset "
                   "caps (so no single asset or market-cap segment dominates). λ (risk aversion) = "
                   f"{C.RISK_AVERSION[rec]:g} for the {rec} profile. The identical calculation is reproduced with "
                   "formulas in Portfolio_Methodology.xlsx.")
        st.dataframe(pd.DataFrame([{
            "Asset": row["asset"], "Expected return": f"{row['expected_return']*100:.1f}%",
            "Volatility": f"{row['volatility']*100:.1f}%", "Solved weight": f"{row['weight']*100:.1f}%",
        } for row in A.solver_table(rec)]), hide_index=True)

    # ===================== WHY THIS PORTFOLIO =====================
    st.markdown("### Why this portfolio?")
    ex = get_explanation(plan, sig, api_key, model)
    chips_html, rest_md = render_reasoning_chips(ex["text"])
    if chips_html:
        st.markdown(CUSTOM_CSS + chips_html, unsafe_allow_html=True)
    else:
        st.markdown(rest_md)
    with st.expander("Full explanation"):
        if chips_html:
            st.markdown(rest_md)
        st.caption("Explanation written by AI from the tool's calculated numbers." if ex["used_ai"]
                  else "Built-in explanation (no live AI connection configured in this deployment).")
        if ex["err"]:
            st.warning(ex["err"])
        if st.button("Regenerate explanation", key="regen"):
            get_explanation(plan, sig, api_key, model, force=True)
            st.rerun()

    # ===================== GOALS =====================
    st.markdown("### Your goals")
    st.dataframe(goals_table(plan), hide_index=True)
    if plan["surplus"] > 0.5:
        st.info(f"After fully funding your goals, {fmt_inr(plan['surplus'])}{suffix(plan['mode'])} was left over "
               "and has been placed in open-ended long-term wealth building.")
    with st.expander("See the asset mix for each goal"):
        st.dataframe(per_goal_alloc_table(plan), hide_index=True)

    # ===================== ALTERNATIVE ALLOCATION =====================
    st.markdown("### Alternative allocation")
    st.caption("Weight changes versus your current (recommended) allocation.")
    alt_names = [n for n in C.PROFILES if n != rec]
    rec_metrics = plan["options"][rec]
    for name in alt_names:
        o = plan["options"][name]
        ret_delta = (o["exp_return"] - rec_metrics["exp_return"]) * 100
        vol_delta = (o["volatility"] - rec_metrics["volatility"]) * 100
        pills = "".join(
            f'<div class="alt-pill"><span class="a">{r["asset"]}</span>'
            f'<span class="v">{r["rec"]:.0f}% → {r["alt"]:.0f}%</span>'
            f'<span class="d {"up" if r["delta"] > 0.5 else "down" if r["delta"] < -0.5 else "flat"}">'
            f'{"—" if abs(r["delta"]) < 0.5 else f"{r["delta"]:+.0f} pts"}</span></div>'
            for r in alt_comparison_rows(plan, name)
        )
        st.markdown(f"""
        <div class="alt-card">
          <div class="alt-card-head">
            <span class="alt-name">{name}</span>
            <span class="alt-stats">Return {o['exp_return']*100:.1f}%/yr ({ret_delta:+.1f} pts)
            &nbsp;·&nbsp; Volatility {o['volatility']*100:.1f}%/yr ({vol_delta:+.1f} pts)</span>
          </div>
          <div class="alt-pill-row">{pills}</div>
        </div>
        """, unsafe_allow_html=True)

    # ===================== ACTION PLAN =====================
    st.markdown("### Action plan")
    for tip in action_plan(plan):
        st.markdown(f"- {tip}")

    # ===================== DETAILS (collapsed - keeps the main page crisp) =====================
    with st.expander("More detail: financial foundation, wealth over time, risk profile"):
        health = plan["financial_health"]
        cols = st.columns(len(health))
        for col, (k, v) in zip(cols, health.items()):
            col.markdown(f"{HEALTH_COLOR.get(v, '⚪')} **{k}**")
            col.caption(v)
        st.markdown(f"- **Risk appetite** {prof['willingness']}/10 · **Risk capacity** {prof['capacity']}/10 · "
                   f"**Required risk** {prof['required_risk']['label']} (what your goals, as entered, would "
                   "actually need - shown for awareness; the recommendation always follows the lower of "
                   "appetite/capacity).")
        st.markdown("##### Your wealth over time")
        nw_proj = A.net_worth_projection(einp, plan)
        st.altair_chart(net_worth_chart(nw_proj))
        st.caption(f"Existing investments plus everything invested through this plan, compounding at the "
                   f"recommended {rec} portfolio's expected return. A single simplified projection, not a guarantee.")

    st.divider()
    _secondary_tabs(plan, einp, sig, api_key, model, opt, rec)
    st.divider()
    st.caption(DISCLAIMER)


def _secondary_tabs(plan, einp, sig, api_key, model, opt, rec):
    t_whatif, t_chat, t_dl = st.tabs(["What-if", "Ask the advisor", "Download"])

    with t_whatif:
        st.markdown("Change the inputs and see the effect instantly - this never changes your saved plan.")
        w1, w2 = st.columns(2)
        mult = w1.slider("Amount, as % of what you entered", 25, 300, 100, 5, key="wi_mult") / 100
        choice = w2.selectbox("Risk profile to test", ["Suggested"] + C.PROFILES, key="wi_prof")
        wi = cached_plan(sig, None if choice == "Suggested" else choice, mult)
        wi_opt = wi["options"][wi["recommended"]]
        wi_pc = wi["projected_corpus"]
        base_prob = {r["goal"]: r["prob"] for r in opt["goals"]}
        rows = []
        for g, r in zip(wi["goals"], wi_opt["goals"]):
            b = base_prob.get(g["name"])
            rows.append({
                "Goal": g["name"],
                "Funded": f"{g['funded_ratio'] * 100:.0f}%" if g.get("funded_ratio") is not None else "-",
                "Chance of reaching target": "n/a" if r["prob"] is None else f"{r['prob'] * 100:.0f}%",
                "Change vs your plan": "-" if (r["prob"] is None or b is None) else f"{(r['prob'] - b) * 100:+.0f} pts",
                "Typical outcome": fmt_inr(r["median"]),
            })
        k = st.columns(4)
        k[0].metric("Amount tested", fmt_inr(wi["amount"]) + suffix(wi["mode"]))
        k[1].metric("Profile tested", wi["recommended"])
        k[2].metric("Expected return / year", f"{wi_opt['exp_return'] * 100:.1f}%")
        k[3].metric("Projected value", fmt_inr(wi_pc["median"]))
        st.caption(f"Corpus range tested: {fmt_inr(wi_pc['weak_market'])} (weak) - {fmt_inr(wi_pc['strong_market'])} (strong)")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.altair_chart(donut(wi_opt["overall_alloc"]))

    with t_chat:
        st.markdown("Ask anything about *your* plan - the advisor answers instantly from the numbers calculated above.")
        suggestions = ["Why is my equity share what it is?", "How is each of my goals doing?",
                      "What could I change to reach my goals?"]
        sc = st.columns(3)
        for i, s in enumerate(suggestions):
            if sc[i].button(s, key=f"sq{i}"):
                st.session_state.pending_q = s
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        q = st.chat_input("Ask about your plan...") or st.session_state.pop("pending_q", None)
        if q:
            history = list(st.session_state.chat)
            st.session_state.chat.append({"role": "user", "content": q})
            with st.spinner("Thinking..."):
                answer, used, err = D.answer_question(plan, history, q, api_key, model)
            st.session_state.chat.append({"role": "assistant", "content": answer})
            if err:
                st.warning(err)
            st.rerun()

    with t_dl:
        xlsx_bytes = X.build_plan_workbook(plan, einp)
        st.download_button("⬇️ Download Excel workbook (your plan + full MVO calculation)", xlsx_bytes,
                           file_name="my_portfolio_plan.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary")
        st.caption("Includes a 'MVO Calculation' sheet with the complete mean-variance optimization behind your "
                   "recommended portfolio - expected returns, the covariance matrix, the solved weights, and "
                   "live formulas that independently verify the result, compared against two alternative "
                   "portfolios to confirm it's the better choice.")
        ex = st.session_state.get("expl") or {"text": ""}
        st.download_button("⬇️ Download my plan (Markdown)", plan_markdown(plan, ex["text"]),
                           file_name="my_investment_plan.md", mime="text/markdown")
        st.download_button("⬇️ Download goal table (CSV)", goals_table(plan).to_csv(index=False),
                           file_name="my_goal_plan.csv", mime="text/csv")


# ------------------------------------------------------------------ main
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
init_state()
sidebar()
# No API key is collected from the user in this deployment - AI explanations stay fully optional and are
# only ever switched on via a server-side ANTHROPIC_API_KEY, never a public input box. With no key set,
# the app runs entirely on its own built-in rule-based explanations and Q&A (see advisor.py).
api_key, model = os.environ.get("ANTHROPIC_API_KEY"), D.DEFAULT_MODEL
step = st.session_state.step
if step > 0:
    st.progress((step - 1) / (len(STEPS) - 2), text=f"{STEPS[step]}")
{0: step_intro, 1: step_about, 2: step_financial_life, 3: step_money, 4: step_risk, 5: step_review,
 6: lambda: step_results(api_key, model)}[step]()
