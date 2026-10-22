"""
advisor.py - the GenAI layer.
The AI NEVER calculates. It receives the numbers computed by allocation.py (as JSON) and explains them
in simple language. If no API key is set (or the call fails) a rule-based explanation is used instead,
so the demo never breaks.
"""
import json
import os
from utils import fmt_inr

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are the explanation layer of an educational, goal-based portfolio planning tool for \
first-time retail investors in India (age 22-40).

All numbers (risk scores, allocations, projections, probabilities, funding gaps, financial-health flags) were \
computed by the tool's own code and are given to you as JSON. Your job is to EXPLAIN them, not to compute anything.

Rules:
1. Use ONLY numbers that appear in the JSON. Never calculate new figures, invent returns, or estimate anything yourself.
2. Never promise or guarantee returns and never promise "maximum returns". Say the figures are estimates from stated assumptions.
3. Talk about asset classes only (liquid funds, debt funds, gold, real estate, and large/mid/small cap mutual funds - based on the Nifty 100 / Nifty Midcap 150 / Nifty Smallcap 250 price-return indices). Do not name specific funds, stocks or companies.
4. Write in simple, friendly English with short sentences. Explain any jargon in a few words.
5. Always explain WHY the allocation fits THIS client: weave together their goals' timelines, risk appetite vs capacity vs
   required risk, dependents, income/debt situation, emergency-fund status and funding gaps - reference several of these
   together, the way a human adviser would, rather than a single generic reason.
6. If risk appetite, risk capacity and required risk point in different directions for this client, say so plainly and
   explain which one the recommendation followed and why (the tool always follows the LOWER of appetite/capacity; it
   never silently overrides that with required risk).
7. This is educational guidance, not personalised financial advice.
8. If the client asks something the JSON cannot answer, say what the tool can and cannot tell them.
Amounts are in Indian rupees."""

EXPLAIN_INSTRUCTION = """Write the client's plan explanation using exactly these markdown headings, in this order:

### Why this fits you
3 to 6 short bullet points, each in the form "**<their specific input>** - <the one portfolio decision it drove>".
Example style: "**Marriage in 4 years** - limits how much high-volatility equity goes toward that goal." Pick the
inputs that actually mattered most for THIS client (don't force a bullet for something that had no effect).

### Where your wishes and your situation pull in different directions
(include this section only if the JSON 'notes' list is not empty - turn the relevant notes into plain sentences)

### How the three options compare
### What to keep in mind

Keep the whole thing under 320 words. The first section must stay as short bullets, not paragraphs."""


def plan_context(plan):
    """Compact, rounded JSON-safe summary of the plan for the AI."""
    prof = plan["profile"]
    snap = plan["snapshot"]
    goals = []
    for g in plan["goals"]:
        goals.append({
            "name": g["name"], "priority": g.get("priority"), "years_to_go": g["years"], "horizon_bucket": g["bucket"],
            "goal_type": g.get("goal_type"),
            "target_in_future_rupees": round(g["target"]) if g["target"] else None,
            "needed_per_month_or_lumpsum": round(g["required"]) if g.get("required") else None,
            "allocated_amount": round(g["funding"]),
            "funded_percent": round(g["funded_ratio"] * 100) if g.get("funded_ratio") is not None else None,
        })
    options = {}
    for name, o in plan["options"].items():
        options[name] = {
            "overall_allocation_percent": o["overall_alloc"],
            "expected_yearly_return_percent": round(o["exp_return"] * 100, 1),
            "typical_yearly_ups_and_downs_percent": round(o["volatility"] * 100, 1),
            "return_in_a_bad_year_percent": round(o["bad_year"] * 100, 1),
            "sharpe_ratio": o["sharpe"],
            "per_goal": [{
                "goal": r["goal"], "equity_percent": r["equity_pct"],
                "chance_of_reaching_target_percent": None if r["prob"] is None else round(r["prob"] * 100),
                "typical_outcome_rupees": round(r["median"]),
                "typical_outcome_as_percent_of_target": None if r["median_pct"] is None else round(r["median_pct"] * 100),
            } for r in o["goals"]],
        }

    financial_profile = {
        "employment_income_known": snap["income_known"],
        "monthly_disposable_income_rupees": round(snap["disposable_income"]) if snap["income_known"] else None,
        "savings_rate_percent": round(snap["savings_rate"] * 100) if snap["savings_rate"] is not None else None,
        "emergency_fund_target_months": snap["emergency_months_target"],
        "emergency_fund_shortfall_rupees": round(snap["emergency_shortfall"]),
        "emergency_fund_is_advisory_only": "This is a RECOMMENDATION only - it is never subtracted from the "
                                            "client's invested amount or any portfolio calculation.",
        "high_emi_debt_burden": snap["high_emi_burden"],
        "high_interest_debt_reported": snap["high_interest_debt"],
        "existing_equity_share_percent": round(snap["existing_equity_share"] * 100) if snap["existing_equity_share"] is not None else None,
        "has_health_insurance": snap["has_health_insurance"],
        "has_life_insurance": snap["has_life_insurance"],
    }

    return {
        "mode": plan["mode"], "amount_rupees": round(plan["amount"]), "sip_step_up_percent": round(plan["sip_step_up"] * 100, 1),
        "risk": {"score_out_of_10": prof["score"], "label": prof["label"], "willingness": prof["willingness"],
                 "capacity": prof["capacity"], "required_risk_label": prof["required_risk"]["label"],
                 "money_weighted_horizon_years": prof["weighted_horizon_years"]},
        "recommended_option": plan["recommended"], "goals": goals, "options": options,
        "financial_profile": financial_profile, "financial_health_scorecard": plan["financial_health"],
        "notes": plan["notes"], "assumed_inflation": plan["inflation"],
        "return_assumptions": plan["assumptions"], "assumptions_note": plan["assumptions_note"],
    }


def _call_llm(system, messages, api_key, model, max_tokens=900):
    import anthropic  # imported here so the app still runs without the package
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(model=model, max_tokens=max_tokens, system=system, messages=messages)
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def generate_explanation(plan, api_key=None, model=None):
    """Returns (markdown_text, used_ai, error_message)."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return fallback_explanation(plan), False, None
    try:
        ctx = json.dumps(plan_context(plan), indent=1)
        msg = f"Here is the client's plan as JSON:\n{ctx}\n\n{EXPLAIN_INSTRUCTION}"
        text = _call_llm(SYSTEM_PROMPT, [{"role": "user", "content": msg}], api_key, model or DEFAULT_MODEL)
        return text, True, None
    except Exception as e:
        return fallback_explanation(plan), False, f"AI call failed ({type(e).__name__}: {e}). Showing the built-in explanation instead."


def answer_question(plan, history, question, api_key=None, model=None):
    """history: list of {'role','content'}. Returns (answer, used_ai, error). Always answers - with a
    live AI model when a key is configured, otherwise with a grounded, rule-based answer built from
    the plan's own numbers (never a generic 'not enabled' non-answer), so the chat box always works."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            system = SYSTEM_PROMPT + "\n\nThe client's plan (JSON):\n" + json.dumps(plan_context(plan), indent=1)
            msgs = list(history) + [{"role": "user", "content": question}]
            return _call_llm(system, msgs, api_key, model or DEFAULT_MODEL, max_tokens=600), True, None
        except Exception as e:
            return fallback_answer(plan, question), False, f"AI call failed ({type(e).__name__}: {e}). Showing a built-in answer instead."
    return fallback_answer(plan, question), False, None


# ---------- rule-based Q&A fallback (no AI needed - keyword-matched against the plan's own numbers) ----------
def fallback_answer(plan, question):
    """A simple, always-available keyword-matched answer grounded in this client's own plan, used
    whenever no live AI connection is configured (or a live call fails)."""
    q = (question or "").lower()
    prof, rec, snap = plan["profile"], plan["recommended"], plan["snapshot"]
    opt = plan["options"][rec]
    pc = plan.get("projected_corpus", {})

    def goal_lines():
        out = []
        for g, r in zip(plan["goals"], opt["goals"]):
            out.append(f"- **{g['name']}** ({g['years']:g} yrs, {g.get('priority','-')}): "
                       f"{r['equity_pct']:.0f}% equity, funded {('%.0f%%' % (g['funded_ratio']*100)) if g.get('funded_ratio') is not None else 'open-ended'}, "
                       + ("chance of reaching target %.0f%%" % (r['prob']*100) if r.get('prob') is not None else "no fixed target to hit"))
        return "\n".join(out)

    if any(k in q for k in ["equity", "stock", "large cap", "mid cap", "small cap"]):
        return (f"Your recommended **{rec}** portfolio holds about "
                f"**{sum(opt['overall_alloc'].get(a,0) for a in ('Large Cap Mutual Fund','Mid Cap Mutual Fund','Small Cap Mutual Fund')):.0f}%** "
                f"across Large/Mid/Small cap combined. This comes from solving for the mix that gives the best "
                f"expected return for your risk level (a mean-variance optimisation), capped so no single market-cap "
                f"segment or asset dominates the portfolio. Your risk score is {prof['score']}/10 ({prof['label']}).")
    if any(k in q for k in ["goal", "target", "funded", "chance"]):
        return "Here's how each of your goals stands under the recommended portfolio:\n\n" + goal_lines()
    if any(k in q for k in ["risk", "score", "profile", "aggressive", "conservative", "moderate"]):
        return (f"Your risk score is **{prof['score']}/10 ({prof['label']})** - the LOWER of your risk appetite "
                f"({prof['willingness']}/10, how you say you feel about ups and downs) and your risk capacity "
                f"({prof['capacity']}/10, what your age, dependents, horizon and emergency-fund status can really "
                f"afford). We never recommend more risk than either of these supports.")
    if any(k in q for k in ["emergency", "buffer", "liquid fund"]):
        if snap["emergency_shortfall"] > 0:
            return (f"We'd recommend keeping about **Rs {snap['emergency_target']:,.0f}** "
                    f"({snap['emergency_months_target']:g} months of expenses) in a liquid fund as an emergency "
                    f"buffer. You currently have about Rs {snap['emergency_current']:,.0f}, so you're about "
                    f"Rs {snap['emergency_shortfall']:,.0f} short. This is advisory only - it does not reduce "
                    "the amount being invested in your plan.")
        return "Your emergency fund already meets the recommended target, based on what you entered - nothing to flag here."
    if any(k in q for k in ["corpus", "projected", "worth", "how much will i have"]):
        return (f"Under the recommended **{rec}** portfolio, your projected investment value is about "
                f"**Rs {pc.get('median', 0):,.0f}**, with a plausible range from about Rs {pc.get('weak_market', 0):,.0f} "
                f"in a weak market to about Rs {pc.get('strong_market', 0):,.0f} in a strong one. These come from "
                f"{plan.get('mode','')} simulations, not a single guaranteed number.")
    if any(k in q for k in ["change", "improve", "increase", "reach my goal", "shortfall"]):
        under = [g for g in plan["goals"] if g.get("funded_ratio") is not None and g["funded_ratio"] < 0.999]
        if under:
            lines = "\n".join(f"- Increase funding to **{g['name']}**, or push its date out." for g in under)
            return "A few goals aren't fully funded at your current amount:\n\n" + lines
        return "All your dated goals are fully funded at your current amount and recommended portfolio."
    return (f"Your plan: investing {plan['mode'].lower()} toward {len(plan['goals'])} goal(s), on the "
            f"**{rec}** portfolio (risk score {prof['score']}/10). Ask me about your equity split, a "
            "specific goal, your risk score, your emergency fund, or your projected corpus, and I'll answer "
            "from the numbers already calculated for your plan.")


# ---------- rule-based fallback (no AI needed) ----------
_BUCKET_REASON = {
    "Short": "the money is needed soon, so it is kept mostly in safer funds (a market fall right before the date could derail the goal)",
    "Medium": "there is some time to recover from dips, so it mixes safer funds with some growth assets",
    "Long": "there is plenty of time to ride out market ups and downs, so growth assets can carry more of the load",
}


def fallback_explanation(plan):
    prof, rec, snap = plan["profile"], plan["recommended"], plan["snapshot"]
    opt = plan["options"][rec]
    lines = ["### Why this fits you",
             f"- **Risk score {prof['score']}/10 ({prof['label']})** - the lower of your comfort with risk "
             f"({prof['willingness']}/10) and what your situation can afford ({prof['capacity']}/10); this profile "
             "drives the mix below."]
    for g, row in zip(plan["goals"], opt["goals"]):
        lines.append(f"- **{g['name']}** ({g['years']:g} years, {g['bucket'].lower()}-term, {g.get('priority', 'Important')} priority) "
                     f"- about {row['equity_pct']:.0f}% in equity, because {_BUCKET_REASON[g['bucket']]}.")
    if snap["emergency_shortfall"] > 0:
        lines.append(f"- **Emergency fund below target** - we'd recommend about Rs {snap['emergency_shortfall']:,.0f} "
                     "more in a liquid fund, kept separate from this plan; this is advisory only and doesn't change "
                     "the amount invested below.")
    if snap["high_emi_burden"]:
        lines.append("- **High EMI burden** - this reduces how much market risk your situation can really afford, "
                     "pulling your recommended profile toward the safer side.")
    if plan["notes"]:
        lines += ["", "### Where your wishes and your situation pull in different directions"]
        lines += [f"- {n}" for n in plan["notes"]]
    lines += ["", "### How the three options compare"]
    equity_keys = ["Large Cap Mutual Fund", "Mid Cap Mutual Fund", "Small Cap Mutual Fund"]
    for name, o in plan["options"].items():
        tag = " (suggested)" if name == rec else ""
        eq = sum(o["overall_alloc"].get(k, 0.0) for k in equity_keys)
        lines.append(f"- **{name}{tag}:** about {eq:.0f}% in equity overall, expected return around {o['exp_return']*100:.1f}% a year, "
                     f"Sharpe ratio {o['sharpe']:.2f}, and in a bad year roughly {o['bad_year']*100:.1f}%.")
    lines += ["", "### What to keep in mind",
              "These figures are estimates based on stated assumptions, not guarantees. Markets can do better or worse. "
              "This tool is for education and does not replace a registered financial advisor."]
    return "\n".join(lines)
