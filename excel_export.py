"""
excel_export.py - builds a personalised Excel workbook for the user's own generated plan, with the
FULL Mean-Variance Optimization calculation reproduced as live formulas (not just the final weights),
so the "how we got to this portfolio" math is visible and independently checkable in Excel.

Design note: the solver itself (scipy.optimize.minimize, SLSQP, a constrained nonlinear program) can't
be re-run as a plain spreadsheet formula - Excel has no native constrained-QP function. So this sheet
does the honest, standard thing: it copies in the solver's own solved weights as a clearly-labelled
input, then recomputes portfolio return, variance, volatility and the objective function directly from
the same raw inputs (expected returns, volatilities, correlation matrix) using ordinary SUMPRODUCT
formulas - an independent, formula-driven verification that those weights really do produce the return/
risk/objective values the app reports. A couple of alternative portfolios are added alongside with the
identical formulas, so the solved portfolio's higher objective value is visible side-by-side, not asserted.
"""
import io

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import allocation as A
import config as C
from utils import fmt_inr

FONT_NAME = "Arial"
HEAD_FILL = PatternFill("solid", fgColor="0E2A43")
SECTION_FILL = PatternFill("solid", fgColor="E8EEF5")
INPUT_FONT = Font(name=FONT_NAME, size=10, color="0000FF")          # blue = hardcoded input
FORMULA_FONT = Font(name=FONT_NAME, size=10, color="000000")        # black = calculated
HEAD_FONT = Font(name=FONT_NAME, size=11, bold=True, color="FFFFFF")
TITLE_FONT = Font(name=FONT_NAME, size=14, bold=True, color="0E2A43")
SECTION_FONT = Font(name=FONT_NAME, size=11, bold=True, color="0E2A43")
LABEL_FONT = Font(name=FONT_NAME, size=10, bold=True)
NOTE_FONT = Font(name=FONT_NAME, size=9, italic=True, color="5B6B7C")
THIN = Side(style="thin", color="D7E1EC")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
PCT_FMT = "0.00%"
RS_FMT = '[$-4009]"Rs "#,##0'
ASSETS = C.ASSETS
N = len(ASSETS)
SHORT_NAME = {a: a.replace(" Mutual Fund", "").replace("Fund", "Fd") for a in ASSETS}


def _set(ws, row, col, value, font=FORMULA_FONT, fmt=None, fill=None, align=None, border=True):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = font
    if fmt:
        cell.number_format = fmt
    if fill:
        cell.fill = fill
    if align:
        cell.alignment = align
    if border:
        cell.border = BORDER
    return cell


def _section(ws, row, text, span=9):
    _set(ws, row, 1, text, font=SECTION_FONT, fill=SECTION_FILL, border=False)
    for c in range(2, span + 1):
        ws.cell(row=row, column=c).fill = SECTION_FILL
    return row + 1


def _col(i):
    """Asset i (0-indexed) lives in column B+i."""
    return get_column_letter(2 + i)


def _asset_header_row(ws, row, bold=True):
    for i, a in enumerate(ASSETS):
        c = _set(ws, row, 2 + i, SHORT_NAME[a], font=LABEL_FONT if bold else FORMULA_FONT,
                 align=Alignment(horizontal="center", wrap_text=True))
    ws.row_dimensions[row].height = 28
    return row + 1


# ------------------------------------------------------------------ MVO Calculation sheet
def _build_mvo_sheet(wb, plan, rec):
    ws = wb.create_sheet("MVO Calculation")
    for i in range(1, 10):
        ws.column_dimensions[get_column_letter(i)].width = 15
    ws.column_dimensions["A"].width = 34

    row = 1
    _set(ws, row, 1, "Mean-Variance Optimization — the full calculation behind your recommended portfolio",
         font=TITLE_FONT, border=False); row += 1
    _set(ws, row, 1, f"Recommended profile: {rec}  |  Shown for the Long-horizon bucket (goals more than "
                      f"{C.MEDIUM_MAX_YEARS} years away) — the same inputs, formulas and methodology below apply "
                      "to every (horizon bucket × risk profile) combination; only the equity cap and the solved "
                      "weights change. Blue cells are inputs (sourced as noted); black cells are formulas.",
         font=NOTE_FONT, border=False); row += 2

    mu = A._MU
    sd = A._SD
    corr = A._CORR

    # ---- 1. Return & risk inputs ----
    row = _section(ws, row, "1. Return & risk inputs (calculated from historical return data — see 'Assumptions' sheet)")
    row = _asset_header_row(ws, row)
    mu_row = row
    _set(ws, row, 1, "Expected annual return (μ)", font=LABEL_FONT)
    for i in range(N):
        _set(ws, row, 2 + i, float(mu[i]), font=INPUT_FONT, fmt=PCT_FMT)
    row += 1
    sd_row = row
    _set(ws, row, 1, "Annual volatility (σ)", font=LABEL_FONT)
    for i in range(N):
        _set(ws, row, 2 + i, float(sd[i]), font=INPUT_FONT, fmt=PCT_FMT)
    row += 2

    # ---- 2. Correlation matrix ----
    row = _section(ws, row, f"2. Shrunk correlation matrix (ρ), used by the optimizer "
                            f"(shrinkage toward the average off-diagonal correlation, α={C.CORR_SHRINKAGE:.0%})")
    row = _asset_header_row(ws, row)
    corr_first_row = row
    for i in range(N):
        _set(ws, row, 1, SHORT_NAME[ASSETS[i]], font=LABEL_FONT)
        for j in range(N):
            _set(ws, row, 2 + j, float(corr[i, j]), font=INPUT_FONT, fmt="0.000")
        row += 1
    row += 1

    # ---- 3. Covariance matrix (formula-driven from sections 1 & 2) ----
    row = _section(ws, row, "3. Covariance matrix  Σ = σ · σᵀ ∘ ρ  (calculated — not re-entered)")
    row = _asset_header_row(ws, row)
    cov_first_row = row
    for i in range(N):
        _set(ws, row, 1, SHORT_NAME[ASSETS[i]], font=LABEL_FONT)
        vol_i_ref = f"${_col(i)}${sd_row}"
        for j in range(N):
            vol_j_ref = f"{_col(j)}${sd_row}"
            corr_ref = f"{_col(j)}{corr_first_row + i}"
            formula = f"={vol_i_ref}*{vol_j_ref}*{corr_ref}"
            _set(ws, row, 2 + j, formula, font=FORMULA_FONT, fmt="0.00000")
        row += 1
    row += 1

    # ---- 4. Optimizer setup ----
    row = _section(ws, row, "4. Optimizer setup for this portfolio")
    lam = C.RISK_AVERSION[rec]
    _set(ws, row, 1, "Risk aversion λ (profile: " + rec + ")", font=LABEL_FONT)
    lam_cell = f"B{row}"
    _set(ws, row, 2, float(lam), font=INPUT_FONT, fmt="0.00")
    row += 1
    cap = C.BUCKET_EQUITY_CAP["Long"]
    _set(ws, row, 1, "Equity exposure cap (Long-horizon bucket)", font=LABEL_FONT)
    cap_cell = f"B{row}"
    _set(ws, row, 2, float(cap), font=INPUT_FONT, fmt=PCT_FMT)
    row += 1
    cap_assets = ["Gold", "Real Estate", "Large Cap Mutual Fund", "Mid Cap Mutual Fund", "Small Cap Mutual Fund"]
    for k, a in enumerate(cap_assets):
        _set(ws, row, 2 + k, SHORT_NAME[a], font=LABEL_FONT, align=Alignment(horizontal="center", wrap_text=True))
    row += 1
    _set(ws, row, 1, "Per-asset caps (Liquid & Debt are uncapped)", font=LABEL_FONT)
    asset_cap_row = row
    for k, a in enumerate(cap_assets):
        _set(ws, row, 2 + k, C.MVO_ASSET_MAX[a], font=INPUT_FONT, fmt=PCT_FMT)
    row += 1
    _set(ws, row, 1, "A per-asset cap prevents the optimizer concentrating the whole equity sleeve (or the whole "
                      "portfolio) into a single asset just because it looks best on paper — see config.py, "
                      "MVO_ASSET_MAX / Michaud (1989) on estimation-error maximization.",
         font=NOTE_FONT, border=False); row += 1
    _set(ws, row, 1, "Solved by scipy.optimize.minimize (method='SLSQP') outside Excel: "
                      "maximise  w'μ − 0.5·λ·w'Σw   subject to  Σw = 1,  w ≥ 0,  equity exposure ≤ cap,  and the "
                      "per-asset caps above (see allocation.py, _solve_mvo()). Excel has no native constrained "
                      "quadratic-program solver, so the solved weights below are copied in as that optimizer's "
                      "exact, documented output for this plan — every formula in sections 6–7 then recomputes "
                      "return, risk and the objective value independently from the raw inputs above, to verify "
                      "those weights really do produce the numbers this plan reports.",
         font=NOTE_FONT, border=False)
    ws.row_dimensions[row].height = 42
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=9)
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
    row += 2

    # ---- 5. Solved weights ----
    row = _section(ws, row, "5. Solved (optimal) weights w*")
    row = _asset_header_row(ws, row)
    w_row = row
    table = A.solver_table(rec)
    _set(ws, row, 1, "Weight", font=LABEL_FONT)
    for i, r in enumerate(table):
        _set(ws, row, 2 + i, float(r["weight"]), font=INPUT_FONT, fmt=PCT_FMT)
    row += 2

    # ---- 6. Verification ----
    row = _section(ws, row, "6. Verification — recomputing return, risk and the objective from the raw inputs")
    _set(ws, row, 1, "Sum of weights (should equal 100%)", font=LABEL_FONT)
    sum_w_cell = f"B{row}"
    _set(ws, row, 2, f"=SUM({_col(0)}{w_row}:{_col(N-1)}{w_row})", fmt=PCT_FMT); row += 1

    _set(ws, row, 1, "Portfolio expected return  E[w] = Σ wᵢ·μᵢ", font=LABEL_FONT)
    ret_cell = f"B{row}"
    _set(ws, row, 2, f"=SUMPRODUCT({_col(0)}{w_row}:{_col(N-1)}{w_row},{_col(0)}{mu_row}:{_col(N-1)}{mu_row})", fmt=PCT_FMT)
    row += 2

    _set(ws, row, 1, "(Σw)ᵢ — covariance matrix × weight vector, one value per asset", font=NOTE_FONT, border=False)
    row += 1
    row = _asset_header_row(ws, row, bold=False)
    sigma_w_row = row
    for i in range(N):
        cov_row_range = f"{_col(0)}{cov_first_row + i}:{_col(N-1)}{cov_first_row + i}"
        w_range = f"{_col(0)}{w_row}:{_col(N-1)}{w_row}"
        _set(ws, row, 2 + i, f"=SUMPRODUCT({cov_row_range},{w_range})", fmt="0.00000")
    row += 2

    _set(ws, row, 1, "Portfolio variance  w'Σw = Σᵢ wᵢ·(Σw)ᵢ", font=LABEL_FONT)
    var_cell = f"B{row}"
    _set(ws, row, 2, f"=SUMPRODUCT({_col(0)}{w_row}:{_col(N-1)}{w_row},{_col(0)}{sigma_w_row}:{_col(N-1)}{sigma_w_row})",
         fmt="0.00000"); row += 1
    _set(ws, row, 1, "Portfolio volatility  √(w'Σw)", font=LABEL_FONT)
    vol_cell = f"B{row}"
    _set(ws, row, 2, f"=SQRT({var_cell})", fmt=PCT_FMT); row += 1
    _set(ws, row, 1, "Objective  U(w) = E[w] − 0.5·λ·Var(w)", font=LABEL_FONT)
    obj_cell = f"B{row}"
    _set(ws, row, 2, f"={ret_cell}-0.5*{lam_cell}*{var_cell}", fmt="0.0000"); row += 2

    _set(ws, row, 1, "Equity exposure (Large + Mid + Small Cap)", font=LABEL_FONT)
    equity_cell = f"B{row}"
    idx = [ASSETS.index(a) for a in ("Large Cap Mutual Fund", "Mid Cap Mutual Fund", "Small Cap Mutual Fund")]
    _set(ws, row, 2, f"={'+'.join(_col(i) + str(w_row) for i in idx)}", fmt=PCT_FMT)
    _set(ws, row, 3, "vs cap", font=NOTE_FONT, border=False)
    _set(ws, row, 4, f"={cap_cell}", fmt=PCT_FMT)
    _set(ws, row, 5, f'=IF({equity_cell}<={cap_cell}+0.0005,"OK","BREACH")', font=FORMULA_FONT)
    row += 1
    for k, a in enumerate(cap_assets):
        w_cell = f"{_col(ASSETS.index(a))}{w_row}"
        cap_ref = f"{_col(k)}{asset_cap_row}"
        _set(ws, row, 1, f"{a} vs its cap", font=NOTE_FONT, border=False)
        _set(ws, row, 2, f"={w_cell}", fmt=PCT_FMT)
        _set(ws, row, 3, "cap", font=NOTE_FONT, border=False)
        _set(ws, row, 4, f"={cap_ref}", fmt=PCT_FMT)
        _set(ws, row, 5, f'=IF({w_cell}<={cap_ref}+0.0005,"OK","BREACH")')
        row += 1
    row += 1

    # ---- 7. Alternative portfolios, same formulas, for comparison ----
    row = _section(ws, row, "7. Is this really (close to) optimal? Two alternative portfolios, identical formulas")
    row = _asset_header_row(ws, row)
    alt_rows = []
    equal_w = 1.0 / N
    alt_weights_1 = [equal_w] * N
    # a small, deliberately sub-optimal nudge away from the solved weights: shift 5pp from Liquid Fund
    # into Large Cap, renormalised - shows a *nearby* portfolio, not a strawman
    base_w = [r["weight"] for r in table]
    nudge = [w for w in base_w]
    # Pick a donor asset that actually HAS weight to move (the largest-weight asset), and a distinct
    # target asset (the smallest-weight one) - guarantees a genuinely different, non-trivial alternative
    # regardless of which assets the solver zeroed out for this particular plan.
    donor_idx = max(range(N), key=lambda i: base_w[i])
    target_idx = min((i for i in range(N) if i != donor_idx), key=lambda i: base_w[i])
    shift = min(0.05, nudge[donor_idx])
    nudge[donor_idx] -= shift
    nudge[target_idx] += shift
    nudge_label = f"Nearby alternative ({shift*100:.0f}pp shifted: {SHORT_NAME[ASSETS[donor_idx]]} → {SHORT_NAME[ASSETS[target_idx]]})"
    for label, wts in [("Equal-weight (1/7 each)", alt_weights_1),
                        (nudge_label, nudge)]:
        _set(ws, row, 1, label, font=LABEL_FONT)
        alt_w_row = row
        for i in range(N):
            _set(ws, row, 2 + i, float(wts[i]), font=INPUT_FONT, fmt=PCT_FMT)
        row += 1
        sw_row = row
        _set(ws, row, 1, "  (Σw)ᵢ for this alternative", font=NOTE_FONT, border=False)
        for i in range(N):
            cov_row_range = f"{_col(0)}{cov_first_row + i}:{_col(N-1)}{cov_first_row + i}"
            w_range = f"{_col(0)}{alt_w_row}:{_col(N-1)}{alt_w_row}"
            _set(ws, row, 2 + i, f"=SUMPRODUCT({cov_row_range},{w_range})", fmt="0.00000")
        row += 1
        a_ret = f"B{row}"
        _set(ws, row, 1, "  Return", font=NOTE_FONT, border=False)
        _set(ws, row, 2, f"=SUMPRODUCT({_col(0)}{alt_w_row}:{_col(N-1)}{alt_w_row},{_col(0)}{mu_row}:{_col(N-1)}{mu_row})", fmt=PCT_FMT)
        row += 1
        a_var = f"B{row}"
        _set(ws, row, 1, "  Variance", font=NOTE_FONT, border=False)
        _set(ws, row, 2, f"=SUMPRODUCT({_col(0)}{alt_w_row}:{_col(N-1)}{alt_w_row},{_col(0)}{sw_row}:{_col(N-1)}{sw_row})", fmt="0.00000")
        row += 1
        a_obj = f"B{row}"
        _set(ws, row, 1, "  Objective U(w)", font=LABEL_FONT)
        _set(ws, row, 2, f"={a_ret}-0.5*{lam_cell}*{a_var}", fmt="0.0000")
        alt_rows.append(a_obj)
        row += 2

    _set(ws, row, 1, f"Compare: solved portfolio's objective ({obj_cell}) should be the HIGHEST of the three — "
                      "that is what 'optimal under these constraints' means here.", font=NOTE_FONT, border=False)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=9)
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)
    row += 1
    _set(ws, row, 1, "Solved objective", font=LABEL_FONT)
    _set(ws, row, 2, f"={obj_cell}", fmt="0.0000")
    _set(ws, row, 3, "Equal-weight objective", font=LABEL_FONT)
    _set(ws, row, 4, f"={alt_rows[0]}", fmt="0.0000")
    _set(ws, row, 5, "Nearby-alt objective", font=LABEL_FONT)
    _set(ws, row, 6, f"={alt_rows[1]}", fmt="0.0000")
    row += 1
    _set(ws, row, 1, "Solved is best?", font=LABEL_FONT)
    _set(ws, row, 2, f'=IF(AND(B{row-1}>=D{row-1},B{row-1}>=F{row-1}),"YES","CHECK")', font=FORMULA_FONT)

    ws.freeze_panes = "B5"
    return ws


# ------------------------------------------------------------------ Your Plan sheet
def _build_plan_sheet(wb, plan, einp, rec):
    ws = wb.create_sheet("Your Plan", 0)
    ws.column_dimensions["A"].width = 30
    for c in "BCDEFG":
        ws.column_dimensions[c].width = 18

    row = 1
    _set(ws, row, 1, "Your goal-based investment plan", font=TITLE_FONT, border=False); row += 1
    _set(ws, row, 1, "Educational guidance only, not personalised financial advice. Figures are estimates based "
                      "on stated assumptions and are not guaranteed.", font=NOTE_FONT, border=False); row += 2

    opt = plan["options"][rec]
    pc = plan["projected_corpus"]
    prof = plan["profile"]
    summary = [
        ("Recommended profile", rec),
        ("Risk score", f"{prof['score']}/10"),
        ("Investing", fmt_inr(plan["amount"]) + ("/month" if plan["mode"].startswith("SIP") else "")),
        ("Mode", plan["mode"]),
        ("Expected return / year", f"{opt['exp_return']*100:.2f}%"),
        ("Volatility / year", f"{opt['volatility']*100:.2f}%"),
        ("Projected corpus — weak market", fmt_inr(pc["weak_market"])),
        ("Projected corpus — expected", fmt_inr(pc["median"])),
        ("Projected corpus — strong market", fmt_inr(pc["strong_market"])),
    ]
    row = _section(ws, row, "Summary")
    for label, val in summary:
        _set(ws, row, 1, label, font=LABEL_FONT)
        _set(ws, row, 2, val, font=FORMULA_FONT)
        row += 1
    row += 1

    row = _section(ws, row, "Recommended mix (new money)")
    _set(ws, row, 1, "Asset", font=LABEL_FONT)
    _set(ws, row, 2, "Weight", font=LABEL_FONT); row += 1
    for a in ASSETS:
        v = opt["overall_alloc"].get(a, 0.0)
        if v <= 0.05:
            continue
        _set(ws, row, 1, a)
        _set(ws, row, 2, v / 100.0, fmt=PCT_FMT)
        row += 1
    row += 1

    row = _section(ws, row, "Your goals")
    headers = ["Goal", "Type", "Priority", "Years", "Target (future Rs)", "Allocated", "Funded %", "Equity %"]
    for k, h in enumerate(headers):
        _set(ws, row, 1 + k, h, font=LABEL_FONT, fill=SECTION_FILL)
    row += 1
    for g, r in zip(plan["goals"], opt["goals"]):
        _set(ws, row, 1, g["name"])
        _set(ws, row, 2, g.get("goal_type", "-"))
        _set(ws, row, 3, g.get("priority", "-"))
        _set(ws, row, 4, g["years"], fmt="0.0")
        _set(ws, row, 5, g["target"] if g["target"] else None, fmt=RS_FMT)
        _set(ws, row, 6, g["funding"], fmt=RS_FMT)
        _set(ws, row, 7, g["funded_ratio"] if g.get("funded_ratio") is not None else None, fmt=PCT_FMT)
        _set(ws, row, 8, r["equity_pct"] / 100.0, fmt=PCT_FMT)
        row += 1
    return ws


# ------------------------------------------------------------------ Assumptions sheet
def _build_assumptions_sheet(wb):
    ws = wb.create_sheet("Assumptions")
    ws.column_dimensions["A"].width = 30
    for c in "BCD":
        ws.column_dimensions[c].width = 18
    row = 1
    _set(ws, row, 1, "Return & volatility assumptions", font=TITLE_FONT, border=False); row += 2
    _set(ws, row, 1, "Asset", font=LABEL_FONT, fill=SECTION_FILL)
    _set(ws, row, 2, "Expected return", font=LABEL_FONT, fill=SECTION_FILL)
    _set(ws, row, 3, "Volatility", font=LABEL_FONT, fill=SECTION_FILL); row += 1
    for a in ASSETS:
        mu, sd = C.ASSUMPTIONS[a]
        _set(ws, row, 1, a)
        _set(ws, row, 2, mu, fmt=PCT_FMT)
        _set(ws, row, 3, sd, fmt=PCT_FMT)
        row += 1
    row += 1
    _set(ws, row, 1, C.ASSUMPTIONS_NOTE, font=NOTE_FONT, border=False)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)
    row += 3
    _set(ws, row, 1, "Inflation by goal type", font=SECTION_FONT, border=False); row += 1
    for k, v in C.GOAL_INFLATION.items():
        _set(ws, row, 1, k)
        _set(ws, row, 2, v, fmt=PCT_FMT)
        row += 1
    return ws


def build_plan_workbook(plan, einp):
    """Returns an in-memory .xlsx (BytesIO) for the given plan — the personalised 'Your Plan' summary,
    the full MVO calculation with live verification formulas, and the assumptions it was built from."""
    rec = plan["recommended"]
    wb = Workbook()
    wb.remove(wb.active)
    _build_plan_sheet(wb, plan, einp, rec)
    _build_mvo_sheet(wb, plan, rec)
    _build_assumptions_sheet(wb)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf
