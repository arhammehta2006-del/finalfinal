"""utils.py - small formatting helpers.
Global rule: every rupee figure uses Indian (lakh/crore) comma grouping, e.g. Rs 12,50,000."""
import re


def _indian_grouping(int_str):
    """Group a non-negative integer's digit string the Indian way: last 3 digits, then pairs of 2
    (e.g. 1234567 -> '12,34,567')."""
    if len(int_str) <= 3:
        return int_str
    head, tail = int_str[:-3], int_str[-3:]
    head = re.sub(r"(\d)(?=(\d{2})+(?!\d))", r"\1,", head)
    return f"{head},{tail}"


def fmt_inr(x):
    """The default rupee format everywhere in the app: full Indian-grouped figure, e.g.
    Rs 12,50,000 / Rs 1,22,07,030 - per the brief's global formatting rule."""
    if x is None:
        return "-"
    x = float(x)
    sign = "-" if x < 0 else ""
    return f"Rs {sign}{_indian_grouping(f'{abs(x):.0f}')}"


def fmt_inr_compact(x):
    """Compact Rs __L / Rs __Cr form, used ONLY as a secondary, parenthetical hint next to a very
    large full-grouped figure (never as a replacement for it)."""
    if x is None:
        return "-"
    x = float(x)
    sign = "-" if x < 0 else ""
    ax = abs(x)
    if ax >= 1e7:
        return f"{sign}Rs {ax / 1e7:.2f} Cr"
    if ax >= 1e5:
        return f"{sign}Rs {ax / 1e5:.2f} L"
    return fmt_inr(x)


def pct(x, digits=0):
    return "-" if x is None else f"{x * 100:.{digits}f}%"
