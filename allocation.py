"""
allocation.py - the calculation engine (plain Python + numpy/scipy, NO AI).
Steps:  goals -> per-goal MVO-based allocation -> funding split (by priority) ->
        3 portfolio options (existing-investment-aware) -> simulated outcomes -> financial health.
The AI layer only ever explains numbers produced here.

The emergency fund is ADVISORY ONLY (brief): risk.emergency_fund_target()/financial_snapshot()
compute a recommended target and shortfall for display, but NOTHING in this file ever subtracts
it from the investable amount or any calculation below - it only affects the risk CAPACITY score
(see risk.risk_profile()), which is a judgement signal, not a cashflow.
"""
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

import config as C
import risk as R

# Index positions of the equity-type sleeves (Large/Mid/Small Cap) within C.ASSETS.
EQUITY_IDX = [i for i, a in enumerate(C.ASSETS) if a in
              ("Large Cap Mutual Fund", "Mid Cap Mutual Fund", "Small Cap Mutual Fund")]


def alloc_dict(weights):
    return {a: float(round(float(w) * 100, 1)) for a, w in zip(C.ASSETS, weights)}


def equity_share(weights):
    return sum(weights[i] for i in EQUITY_IDX) * 100


# ---- Correlation shrinkage + covariance (calculated once at import, from config's 10-yr data) ----
def _shrunk_correlation(corr, alpha):
    n = corr.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    avg_corr = corr[off_diag_mask].mean()
    shrunk = (1 - alpha) * corr + alpha * avg_corr
    np.fill_diagonal(shrunk, 1.0)
    return shrunk


_MU = np.array([C.ASSUMPTIONS[a][0] for a in C.ASSETS])
_SD = np.array([C.ASSUMPTIONS[a][1] for a in C.ASSETS])
_CORR_RAW = np.array(C.CORR)
_CORR = _shrunk_correlation(_CORR_RAW, C.CORR_SHRINKAGE)
_COV = np.outer(_SD, _SD) * _CORR


def portfolio_stats(w):
    """Expected yearly return and volatility (typical ups and downs) of a mix."""
    w = np.array(w, dtype=float)
    return float(w @ _MU), float(np.sqrt(w @ _COV @ w))


def sharpe_ratio(mu, sd):
    """Return per unit of risk, relative to a risk-free rate (config.RISK_FREE_RATE)."""
    return float((mu - C.RISK_FREE_RATE) / sd) if sd > 1e-9 else 0.0


def planning_return(w):
    """Prudent return used to size the required saving: expected - volatility drag - safety margin."""
    mu, sd = portfolio_stats(w)
    return max(0.0, mu - 0.5 * sd ** 2 - C.PLANNING_MARGIN)


# ---- Mean-Variance Optimization solver ------------------------------------------------------
def _solve_mvo(mu, cov, risk_aversion, equity_cap):
    """maximise w'mu - 0.5*risk_aversion*w'cov*w  s.t. sum(w)=1, w>=0, equity exposure <= equity_cap,
    per-asset caps from config (Gold/Real Estate tighter than the rest; Liquid/Debt uncapped)."""
    n = len(mu)

    def neg_utility(w):
        return -(w @ mu - 0.5 * risk_aversion * (w @ cov @ w))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: equity_cap - sum(w[i] for i in EQUITY_IDX)},
    ]
    bounds = [(0.0, 1.0 if a in C.MVO_UNCAPPED_ASSETS else C.MVO_ASSET_MAX.get(a, C.MVO_ASSET_MAX_DEFAULT))
              for a in C.ASSETS]
    w0 = np.full(n, 1.0 / n)
    res = minimize(neg_utility, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 200, "ftol": 1e-10})
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return (w / s) if s > 1e-9 else w0


@lru_cache(maxsize=32)
def _goal_allocation_cached(bucket, profile):
    w = _solve_mvo(_MU, _COV, C.RISK_AVERSION[profile], C.BUCKET_EQUITY_CAP[bucket])
    return tuple(w)


def goal_allocation(years, profile):
    """Weights (fractions, same order as C.ASSETS) for a goal `years` away under a risk profile,
    solved by Mean-Variance Optimization for that (horizon bucket, risk profile) pair."""
    if years <= C.CAPITAL_PROTECTION_YEARS:
        return [1.0] + [0.0] * (len(C.ASSETS) - 1)      # goal is very close: 100% Liquid Fund
    return list(_goal_allocation_cached(R.bucket_for(years), profile))


def solver_table(profile):
    """Per-asset (expected return, volatility, solved weight) for the LONG-horizon bucket under a
    risk profile - what the 'show the MVO calculation' panel displays, and what the Excel backend
    reproduces with formulas against the same data."""
    w = goal_allocation(C.MEDIUM_MAX_YEARS + 1, profile)
    return [{"asset": a, "expected_return": _MU[i], "volatility": _SD[i], "weight": w[i]}
            for i, a in enumerate(C.ASSETS)]


def tilt_for_existing(weights, existing_equity_share):
    """
    Don't blindly add more equity if the client's EXISTING holdings are already equity-heavy (or
    the reverse). Shifts new-money equity weight toward/away from debt+liquid, proportionally to
    how far the existing portfolio sits from this mix's own equity target.
    """
    if existing_equity_share is None:
        return weights
    w = list(weights)
    target_equity = sum(w[i] for i in EQUITY_IDX)
    diff = existing_equity_share - target_equity
    if abs(diff) < 1e-6 or target_equity <= 0:
        return w
    shift = 0.5 * diff * target_equity
    shift = max(-target_equity, min(target_equity, shift))
    for i in EQUITY_IDX:
        w[i] -= shift * (w[i] / target_equity)
    w[1] += shift          # absorb into debt first
    if w[1] < 0:
        w[0] += w[1]        # spill any remainder into liquid
        w[1] = 0.0
    w = [max(0.0, x) for x in w]
    s = sum(w)
    return [x / s for x in w] if s > 0 else w


# ---- Existing + new money "combined mix" --------------------------------------------------
def existing_mix_vector(existing_by_category, target_weights_7):
    """Returns (total_value, 7-asset weight vector or None). The one existing category that
    doesn't map 1:1 to a single asset ('Mutual Funds (equity)') is split across Large/Mid/Small
    cap in proportion to the TARGET mix's own split among them - a documented modelling choice
    (we aren't told how the user's existing equity fund is actually split), used ONLY to make the
    combined-portfolio maths work; the on-screen 'existing mix' always shows the 5 categories the
    user actually entered, never this invented split."""
    total = sum(existing_by_category.values())
    if total <= 0:
        return 0.0, None
    w = np.zeros(len(C.ASSETS))
    eq_total_target = sum(target_weights_7[i] for i in EQUITY_IDX)
    for cat, assets in C.EXISTING_CATEGORY_TO_ASSETS.items():
        val = existing_by_category.get(cat, 0.0)
        if val <= 0:
            continue
        if len(assets) == 1:
            w[C.ASSETS.index(assets[0])] += val
        else:
            idxs = [C.ASSETS.index(a) for a in assets]
            if eq_total_target > 1e-9:
                for i in idxs:
                    w[i] += val * (target_weights_7[i] / eq_total_target)
            else:
                for i in idxs:
                    w[i] += val / len(idxs)
    return total, (w / total)


def combined_portfolio(existing_by_category, new_amount, target_weights):
    """Given existing holdings (by the 5 EXISTING_CATEGORIES), new money available this period,
    and the target 7-asset mix, solve how to split ONLY the new money so the combined portfolio
    (existing + new) moves toward the target - existing holdings are never 'sold' or rebalanced."""
    existing_by_category = dict(existing_by_category or {})
    for cat in C.EXISTING_CATEGORIES:
        existing_by_category.setdefault(cat, 0.0)
    target = np.array([target_weights.get(a, 0.0) / 100.0 for a in C.ASSETS])
    existing_total, existing_w = existing_mix_vector(existing_by_category, target)
    new_amount = max(0.0, new_amount or 0.0)
    combined_total = existing_total + new_amount

    if new_amount <= 0:
        new_alloc = np.zeros(len(C.ASSETS))
    elif existing_total <= 0:
        new_alloc = target.copy()
    else:
        existing_rupees = existing_w * existing_total

        def obj(new_w):
            combined = (existing_rupees + new_w * new_amount) / combined_total
            return np.sum((combined - target) ** 2)

        n = len(C.ASSETS)
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        res = minimize(obj, target.copy(), method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-12})
        new_alloc = np.clip(res.x, 0.0, None)
        s = new_alloc.sum()
        new_alloc = (new_alloc / s) if s > 1e-9 else target.copy()

    combined_rupees = (existing_w * existing_total if existing_w is not None else np.zeros(len(C.ASSETS))) + new_alloc * new_amount
    combined_mix_7 = (combined_rupees / combined_total) if combined_total > 0 else target.copy()

    def fold_to_categories(weight_vec_7):
        out = {}
        for cat, assets in C.EXISTING_CATEGORY_TO_ASSETS.items():
            out[cat] = float(sum(weight_vec_7[C.ASSETS.index(a)] for a in assets) * 100)
        return out

    return {
        "existing_total": existing_total, "new_amount": new_amount, "combined_total": combined_total,
        "current_mix": fold_to_categories(existing_w) if existing_w is not None else None,
        "target_mix": fold_to_categories(target),
        "new_money_alloc": fold_to_categories(new_alloc) if new_amount > 0 else None,
        "new_money_alloc_rupees": {a: round(float(new_alloc[i] * new_amount), 2) for i, a in enumerate(C.ASSETS)} if new_amount > 0 else None,
        "combined_mix": fold_to_categories(combined_mix_7),
    }


# ---- Income/expense growth projection -------------------------------------------------------
def expense_growth_rate(income_growth):
    g = income_growth or 0.0
    return C.INFLATION + C.LIFESTYLE_CREEP_FACTOR * max(0.0, g - C.INFLATION)


def project_disposable_income(inp, years):
    income_growth = inp.get("income_growth") or 0.0
    exp_growth = expense_growth_rate(income_growth)
    income0 = inp.get("monthly_income") or 0.0
    expenses0 = inp.get("monthly_expenses") or 0.0
    emi0 = inp.get("emi_total") or 0.0
    n = max(1, round(years or 10))
    rows = []
    for y in range(n + 1):
        income = income0 * (1 + income_growth) ** y
        expenses = expenses0 * (1 + exp_growth) ** y
        rows.append({"year": y, "income": income, "expenses": expenses,
                     "disposable": max(0.0, income - expenses - emi0)})
    return rows


def net_worth_projection(inp, plan):
    """Age-by-age projection of TOTAL wealth: existing investments (today) plus every rupee
    invested through this plan, compounding at the recommended portfolio's expected return."""
    snap = plan["snapshot"]
    rec_opt = plan["options"][plan["recommended"]]
    r = rec_opt["exp_return"]
    sip = plan["mode"].startswith("SIP")
    amount = plan["amount"]
    step_up = plan["sip_step_up"] if sip else 0.0

    age0 = int(inp.get("age") or 0)
    goal_years = [g["years"] for g in plan["goals"] if g.get("years")]
    retire_years = (inp["retirement_age"] - inp["age"]) if inp.get("has_retirement") and inp.get("retirement_age", 0) > age0 else 0
    horizon = max([retire_years] + goal_years + [plan["profile"]["weighted_horizon_years"]])
    horizon = int(min(50, max(5, round(horizon))))

    wealth = snap["existing_total"] + (0.0 if sip else amount)
    rows = [{"age": age0, "year": 0, "wealth": wealth}]
    for j in range(1, horizon + 1):
        contribution = amount * 12 * ((1 + step_up) ** (j - 1)) if sip else 0.0
        wealth = wealth * (1 + r) + contribution
        rows.append({"age": age0 + j, "year": j, "wealth": wealth})
    return rows


def _sip_future_value(c0, months, rm, step_up):
    v = 0.0
    for t in range(months):
        year_idx = t // 12
        v += c0 * (1 + step_up) ** year_idx
        v *= (1 + rm)
    return v


def required_funding(target, years, g, sip, step_up=0.0):
    if sip:
        months = max(1, round(years * 12))
        rm = (1 + g) ** (1 / 12) - 1
        if step_up <= 1e-9:
            return target / months if rm < 1e-9 else target * rm / ((1 + rm) ** months - 1)
        lo, hi = 0.0, max(target, 1.0)
        for _ in range(60):
            mid = (lo + hi) / 2
            fv = _sip_future_value(mid, months, rm, step_up)
            if fv < target:
                lo = mid
            else:
                hi = mid
        return hi
    return target / (1 + g) ** years


def required_return(amount, target, years, sip, step_up=0.0):
    if amount <= 0 or target <= 0 or years <= 0:
        return 0.0
    lo, hi = -0.5, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        rm = (1 + mid) ** (1 / 12) - 1 if sip else None
        fv = _sip_future_value(amount, max(1, round(years * 12)), rm, step_up) if sip else amount * (1 + mid) ** years
        if fv < target:
            lo = mid
        else:
            hi = mid
    return hi


def simulate(z, mu, sd, funding, sip, target, step_up=0.0):
    n, sims = z.shape
    if funding <= 0:
        return {"median": 0.0, "p10": 0.0, "p90": 0.0, "prob": 0.0 if target else None,
                "median_pct": 0.0 if target else None}
    v = np.zeros(sims) if sip else np.full(sims, float(funding))
    mm, sm = mu / 12, sd / np.sqrt(12)
    for t in range(n):
        if sip:
            year_idx = t // 12
            v = v + funding * (1 + step_up) ** year_idx
        v = v * (1 + np.maximum(mm + sm * z[t], -0.95))
    return {
        "median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
        "prob": float((v >= target).mean()) if target else None,
        "median_pct": float(np.median(v) / target) if target else None,
    }


def financial_health(inp, snap, goals):
    """Labelled scorecard. Any input the user skipped comes back as 'Not assessed', never guessed."""
    card = {}

    if snap["emergency_target"] <= 0:
        card["Emergency Fund (advisory)"] = "Not assessed"
    elif snap["emergency_shortfall"] <= 0:
        card["Emergency Fund (advisory)"] = "Good"
    elif snap["emergency_current"] / snap["emergency_target"] >= 0.5:
        card["Emergency Fund (advisory)"] = "Needs Attention"
    else:
        card["Emergency Fund (advisory)"] = "At Risk"

    if snap["emi_ratio"] is None:
        card["Debt Level"] = "Not assessed"
    elif snap["high_emi_burden"]:
        card["Debt Level"] = "Needs Attention"
    else:
        card["Debt Level"] = "Healthy"

    if snap["savings_rate"] is None:
        card["Savings Rate"] = "Not assessed"
    elif snap["savings_rate"] >= 0.3:
        card["Savings Rate"] = "Strong"
    elif snap["savings_rate"] >= 0.15:
        card["Savings Rate"] = "Good"
    else:
        card["Savings Rate"] = "Needs Attention"

    ret = next((g for g in goals if g["kind"] == "retirement"), None)
    if ret is None:
        card["Retirement Planning"] = "Not assessed"
    elif ret.get("funded_ratio") is None:
        card["Retirement Planning"] = "On Track"
    elif ret["funded_ratio"] >= 0.95:
        card["Retirement Planning"] = "On Track"
    elif ret["funded_ratio"] >= 0.7:
        card["Retirement Planning"] = "Behind Target"
    else:
        card["Retirement Planning"] = "Significantly Behind"

    needs_life_cover = inp.get("dependents", 0) > 0 and snap["sole_earner"] and not snap["has_life_insurance"]
    if not snap["has_health_insurance"] or needs_life_cover:
        card["Insurance Protection"] = "Needs Review"
    else:
        card["Insurance Protection"] = "Good"

    return card


# ---- The full plan ---------------------------------------------------------------
def build_plan(inp, override=None, amount_mult=1.0):
    """
    inp: validated inputs dict. override: force a risk profile (what-if). amount_mult: scale the money (what-if).
    Returns one big dict that both the UI and the AI layer read from.
    """
    sip = inp["mode"].startswith("SIP")
    step_up = float(inp.get("sip_step_up") or 0.0) if sip and inp.get("sip_step_up_enabled") else 0.0
    goals = R.build_goals(inp)
    prof = R.risk_profile(inp, goals)
    rec = override or prof["label"]
    amount = inp["amount"] * amount_mult
    snap = R.financial_snapshot(inp)

    # 1) plan returns from the RECOMMENDED profile's allocation for each goal
    for g in goals:
        w = goal_allocation(g["years"], rec)
        g["plan_return"] = planning_return(w)

    # 2) the FULL investable amount is available to fund goals - the emergency fund is
    #    advisory-only (brief) and is never subtracted here.
    avail = amount

    # 3) split the money: fund goals by PRIORITY first, then nearest goal within a tier
    targeted = [x for x in goals if x["target"] is not None]
    targeted.sort(key=lambda x: (C.PRIORITY_ORDER.get(x.get("priority", "Important"), 1), x["years"]))
    for g in targeted:
        g["required"] = required_funding(g["target"], g["years"], g["plan_return"], sip, step_up)
        g["funding"] = min(g["required"], avail)
        g["funded_ratio"] = g["funding"] / g["required"] if g["required"] > 0 else 1.0
        avail -= g["funding"]
    for g in goals:
        if g["target"] is None:
            g.update(required=None, funding=0.0, funded_ratio=None)

    # 4) anything left over goes to open-ended goals, or to an automatic wealth-building bucket
    surplus = max(avail, 0.0)
    if surplus > 0.5:
        open_goals = [g for g in goals if g["target"] is None]
        if open_goals:
            for g in open_goals:
                g["funding"] += surplus / len(open_goals)
        else:
            longest = max([g["years"] for g in goals], default=0)
            yrs = longest if longest >= C.MEDIUM_MAX_YEARS else float(C.DEFAULT_WEALTH_YEARS)
            goals.append({"name": "Long-term wealth building (surplus)", "years": yrs, "kind": "wealth",
                          "today_value": None, "target": None, "bucket": R.bucket_for(yrs),
                          "priority": "Aspirational", "goal_type": "Others", "inflation_rate": C.GOAL_INFLATION["Others"],
                          "vacation_country": None,
                          "required": None, "funding": surplus, "funded_ratio": None, "plan_return": 0.0})

    # 5) build the three portfolio options and simulate every goal under each
    z_list = []
    for i, g in enumerate(goals):
        rng = np.random.default_rng(C.SEED + i)
        z_list.append(rng.standard_normal((max(1, round(g["years"] * 12)), C.N_SIMS)))

    options = {}
    for name in C.PROFILES:
        rows, agg = [], np.zeros(len(C.ASSETS))
        for g, z in zip(goals, z_list):
            w = goal_allocation(g["years"], name)
            mu, sd = portfolio_stats(w)
            res = simulate(z, mu, sd, g["funding"], sip, g["target"], step_up)
            rows.append({"goal": g["name"], "alloc": alloc_dict(w), "equity_pct": round(equity_share(w), 1),
                         "exp_return": mu, "volatility": sd, **res})
            agg += g["funding"] * np.array(w)
        if agg.sum() > 0:
            overall = agg / agg.sum()
        elif goals:
            overall = np.mean([goal_allocation(g["years"], name) for g in goals], axis=0)
        else:
            overall = np.zeros(len(C.ASSETS))
            overall[EQUITY_IDX[0]] = 1.0
        overall = np.array(tilt_for_existing(list(overall), snap["existing_equity_share"]))
        mu, sd = portfolio_stats(overall)
        combined = combined_portfolio(snap["existing_by_category"], avail if name == rec else 0.0,
                                      alloc_dict(overall)) if name == rec else None
        options[name] = {
            "overall_alloc": alloc_dict(overall), "exp_return": mu, "volatility": sd,
            "bad_year": mu - 1.645 * sd,
            "stress_decline": mu - 2.0 * sd,
            "sharpe": round(sharpe_ratio(mu, sd), 2),
            "goals": rows,
            "combined": combined,
        }
    # combined-portfolio figures for the recommended option only (what the user actually invests
    # new money into); computed against the FULL investable amount (pre-goal-split), since that is
    # "the new money this period" from the brief's point of view.
    options[rec]["combined"] = combined_portfolio(snap["existing_by_category"], amount, options[rec]["overall_alloc"])

    # 6) required risk - a single, documented simplification: what annual return would the
    #    money-weighted horizon + total investable amount need to hit the money-weighted total of
    #    all dated goals? Shown for awareness; does NOT override the appetite/capacity recommendation.
    weighted_horizon = prof["weighted_horizon_years"]
    total_target = sum(g["target"] for g in goals if g.get("target"))
    req_g = required_return(amount, total_target, weighted_horizon, sip, step_up) if total_target > 0 else 0.0
    bucket_here = R.bucket_for(weighted_horizon)
    cons_pr = planning_return(goal_allocation(weighted_horizon, "Conservative")) if weighted_horizon > C.CAPITAL_PROTECTION_YEARS else 0.0
    mod_pr = planning_return(goal_allocation(weighted_horizon, "Moderate")) if weighted_horizon > C.CAPITAL_PROTECTION_YEARS else 0.0
    aggr_pr = planning_return(goal_allocation(weighted_horizon, "Aggressive")) if weighted_horizon > C.CAPITAL_PROTECTION_YEARS else 0.0
    if total_target <= 0:
        required_label = "Not applicable"
    elif req_g <= cons_pr:
        required_label = "Low"
    elif req_g <= mod_pr:
        required_label = "Moderate"
    elif req_g <= aggr_pr:
        required_label = "Moderate-High"
    else:
        required_label = "High"
    prof["required_risk"] = {"label": required_label, "required_return": req_g, "bucket": bucket_here}

    # 7) plain-English notes about mismatches and gaps
    notes = []
    if override and override != prof["label"]:
        notes.append(f"You are viewing the {override} portfolio; based on your answers the suggested profile is {prof['label']}.")
    gap = prof["willingness"] - prof["capacity"]
    if gap >= 2.5:
        notes.append(f"You are more comfortable with risk (willingness {prof['willingness']}/10) than your situation "
                     f"can comfortably afford (capacity {prof['capacity']}/10), so your score follows the lower figure.")
    elif gap <= -2.5:
        notes.append(f"Your situation could support more risk (capacity {prof['capacity']}/10) than you are comfortable "
                     f"with (willingness {prof['willingness']}/10), so your score follows your comfort level. "
                     "Long goals usually need some growth, so consider the higher-risk options too.")
    if abs(prof["stated"] - prof["quiz"]) >= 4:
        notes.append("Your stated risk appetite and your answers to the scenario questions point in different directions, "
                     "so we gave more weight to the scenario answers.")
    if required_label in ("Moderate-High", "High") and prof["label"] != "Aggressive":
        notes.append(f"To reach your goals as entered, your money would need to grow at roughly {req_g*100:.1f}% a year "
                      f"(a 'Required Risk' of {required_label}) - higher than your recommended {prof['label']} profile "
                      "targets. Consider a larger amount, a later date, lower targets, or a higher-risk option.")
    if snap["emergency_shortfall"] > 0:
        notes.append(f"Advisory: your emergency fund is about Rs {snap['emergency_shortfall']:,.0f} short of the "
                     f"recommended {snap['emergency_months_target']:g}-month target. This is a recommendation only "
                     "- it is not deducted from the amount invested below.")
    if snap["high_interest_debt"]:
        notes.append("You reported some high-interest debt. Its cost likely exceeds typical investment returns, "
                     "so consider repaying it before increasing investments further.")
    for g in goals:
        if g["years"] <= C.CAPITAL_PROTECTION_YEARS:
            notes.append(f"'{g['name']}' is {g['years']:g} year or less away, so it is kept 100% in liquid funds.")
        elif g["bucket"] == "Short" and rec != "Conservative":
            eq = equity_share(goal_allocation(g["years"], rec))
            notes.append(f"'{g['name']}' is only {g['years']:g} years away, so equity is limited to {eq:.0f}% "
                         f"for this goal even though your profile is {rec}.")
        if g["kind"] == "retirement" and g["years"] <= 10 and rec == "Aggressive":
            notes.append(f"Retirement is only {g['years']:g} years away, yet the Aggressive option still carries high "
                         "equity exposure. Consider the Moderate option as retirement gets closer.")
    for g in goals:
        if g.get("funded_ratio") is not None and g["funded_ratio"] < 0.999:
            shortfall_rupees = (g["required"] or 0) - g["funding"]
            notes.append(f"'{g['name']}' is only {g['funded_ratio']*100:.0f}% funded at your current amount "
                         f"(about Rs {shortfall_rupees:,.0f} short per {'month' if sip else 'the plan'}). "
                         "Consider a larger amount, a later date, or a lower target.")
    if snap["existing_equity_share"] is not None and snap["existing_equity_share"] >= 0.75:
        notes.append(f"Your existing investments are already about {snap['existing_equity_share']*100:.0f}% equity, "
                     "so new money has been tilted a little more toward debt/liquid to avoid over-concentration.")

    health = financial_health(inp, snap, goals)
    overall_mix = options[rec]["overall_alloc"]
    distinct_classes = sum(1 for v in overall_mix.values() if v >= 5.0)
    health["Investment Diversification"] = "Low" if distinct_classes <= 2 else "Moderate" if distinct_classes == 3 else "Good"

    # 8) income/expense growth projection - informational only
    income_growth = inp.get("income_growth") or 0.0
    expense_growth = expense_growth_rate(income_growth)
    proj_years = min(15, max(5, round(weighted_horizon or 10)))
    disposable_projection = project_disposable_income(inp, proj_years)

    # Corpus impact shown as a RANGE (brief): weak-market (p10) to strong-market (p90) across all
    # goals under the recommended option, from the same Monte Carlo simulation used per goal.
    # IMPORTANT: g["funding"] above is only ever drawn from `avail = amount` (this period's new
    # money) - existing investments are never mixed into per-goal funding, because they aren't
    # tracked goal-by-goal, only as a lump total by category. So the per-goal simulation alone
    # would silently EXCLUDE existing investments despite this card's own definition promising
    # "current investments + future contributions". We fix that honestly here: simulate the
    # existing lump sum growing on its own, under the recommended portfolio's own expected
    # return/volatility over the plan's weighted horizon, and add its median/p10/p90 into the
    # per-goal totals below (summing percentiles across independent simulations is the same
    # practical approximation already used across goals, not a precise joint percentile - it's
    # a brief estimate, not a guarantee, exactly as the definition text says).
    rec_rows = options[rec]["goals"]
    existing_total = snap.get("existing_total", 0.0)
    horizon_years = max(1, round(weighted_horizon or 10))
    if existing_total > 0:
        rng_e = np.random.default_rng(C.SEED + 9973)
        z_e = rng_e.standard_normal((max(1, horizon_years * 12), C.N_SIMS))
        existing_sim = simulate(z_e, options[rec]["exp_return"], options[rec]["volatility"],
                                existing_total, sip=False, target=0.0)
    else:
        existing_sim = {"median": 0.0, "p10": 0.0, "p90": 0.0}
    projected_corpus = {
        "median": sum(r["median"] for r in rec_rows) + existing_sim["median"],
        "weak_market": sum(r["p10"] for r in rec_rows) + existing_sim["p10"],
        "strong_market": sum(r["p90"] for r in rec_rows) + existing_sim["p90"],
        "existing_included": existing_total > 0,
        "horizon_years": horizon_years,
        "definition": ("Current investments (if any) + future contributions to your goals + estimated growth "
                       f"under the recommended portfolio, projected {horizon_years} years ahead (your plan's "
                       "goal-amount-weighted average goal horizon - individual goals maturing sooner or later "
                       "are still simulated on their own actual timelines underneath this). 'Weak market' and "
                       f"'strong market' are the 10th/90th-percentile outcomes across {C.N_SIMS:,} simulated "
                       "futures - not single-point forecasts."),
    }

    return {
        "mode": "SIP (monthly)" if sip else "Lumpsum", "amount": amount, "surplus": surplus,
        "sip_step_up": step_up,
        "profile": prof, "recommended": rec, "goals": goals, "options": options, "notes": notes,
        "assumptions": {a: {"expected_return": C.ASSUMPTIONS[a][0], "volatility": C.ASSUMPTIONS[a][1]} for a in C.ASSETS},
        "assumptions_note": C.ASSUMPTIONS_NOTE,
        "inflation": C.INFLATION,
        "income_growth": income_growth, "expense_growth": expense_growth,
        "disposable_projection": disposable_projection,
        "projected_corpus": projected_corpus,
        "snapshot": snap, "financial_health": health,
    }
