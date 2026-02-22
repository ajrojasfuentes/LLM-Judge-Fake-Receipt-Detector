"""
forensics_analysis.semantic_checks — OCR-based accounting validation.

Performs heuristic extraction of monetary amounts from OCR text lines
and validates basic accounting relationships such as:

    subtotal + tax ≈ total

The extraction relies on keyword matching (e.g. "Total", "Subtotal",
"Tax", "VAT", "GST") and regex-based monetary-value parsing.  Results
are reported as a list of named check dictionaries that include pass /
fail / uncertain status along with supporting evidence lines.

Important implementation note
-----------------------------
The keyword scan intentionally checks **subtotal** and **tax** patterns
*before* the generic **total** pattern.  This prevents "Sub Total" or
"Subtotal" lines from being incorrectly matched by the ``\\btotal\\b``
regex inside the total pattern, which would misclassify them.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Monetary token: optional sign, 1-7 leading digits, optional decimal
# part with 2-3 fractional digits.  Negative look-ahead excludes
# percentage values (e.g. "7.5 %").
_MONEY_TOKEN_RE = re.compile(r"(-?\d{1,7}(?:[.,]\d{2,3})?)(?!\s*%)")

# Keyword patterns — order matters (see module docstring).
_KEY_SUBTOTAL_RE = re.compile(r"\b(subtotal|sub total|sub-total)\b", re.I)
_KEY_TAX_RE = re.compile(r"\b(tax|vat|gst)\b", re.I)
_KEY_TOTAL_RE = re.compile(r"\b(total|amount due|grand total|balance due)\b", re.I)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_decimal(s: str) -> Optional[Decimal]:
    """Parse a string as a Decimal, treating commas as decimal separators.

    Parameters
    ----------
    s : str
        Raw numeric string (e.g. ``"12.99"`` or ``"12,99"``).

    Returns
    -------
    Decimal or None
        Parsed value, or ``None`` if the string is not a valid number.
    """
    s = s.strip().replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


# ---------------------------------------------------------------------------
# Amount extraction
# ---------------------------------------------------------------------------

def extract_amounts(lines: List[str]) -> Dict[str, Any]:
    """Heuristically extract subtotal, tax, and total from OCR lines.

    Each line is scanned for monetary tokens.  When a keyword match is
    found, the **last** numeric token on that line is taken as the
    value (this heuristic works well for receipt layouts where the
    amount appears at the end of the line).

    **Scan order**: subtotal -> tax -> total.  This ensures that "Sub
    Total" lines are not consumed by the generic "total" pattern.

    Parameters
    ----------
    lines : list of str
        OCR text lines (one string per line).

    Returns
    -------
    dict
        Keys ``"total"``, ``"subtotal"``, ``"tax"`` (each a
        ``Decimal`` or ``None``) and ``"total_line"``,
        ``"subtotal_line"``, ``"tax_line"`` (each the raw OCR line
        string or ``None``).
    """
    best: Dict[str, Optional[Decimal]] = {
        "total": None,
        "subtotal": None,
        "tax": None,
    }
    evidence: Dict[str, Optional[str]] = {
        "total_line": None,
        "subtotal_line": None,
        "tax_line": None,
    }

    for ln in lines:
        l = ln.strip()
        if not l:
            continue

        m = _MONEY_TOKEN_RE.findall(l)
        if not m:
            continue
        # Take the last numeric token on the line (usually the amount)
        val = _to_decimal(m[-1])
        if val is None:
            continue

        # IMPORTANT: check subtotal BEFORE total to avoid "Sub Total"
        # being consumed by the generic total pattern (which contains
        # \btotal\b and would match the "total" part of "sub total").
        if best["subtotal"] is None and _KEY_SUBTOTAL_RE.search(l):
            best["subtotal"] = val
            evidence["subtotal_line"] = l
            continue
        if best["tax"] is None and _KEY_TAX_RE.search(l):
            best["tax"] = val
            evidence["tax_line"] = l
            continue
        if best["total"] is None and _KEY_TOTAL_RE.search(l):
            best["total"] = val
            evidence["total_line"] = l
            continue

    return {**best, **evidence}


# ---------------------------------------------------------------------------
# Semantic checks
# ---------------------------------------------------------------------------

def semantic_checks(lines: List[str]) -> List[Dict[str, Any]]:
    """Run accounting cross-checks on OCR-extracted monetary values.

    Currently implements a single check:

    * **subtotal + tax == total** — Verifies that the extracted
      subtotal and tax sum to the stated total within a tolerance of
      +/-0.02 (to accommodate minor rounding differences).

    Parameters
    ----------
    lines : list of str
        OCR text lines (one string per line).

    Returns
    -------
    list of dict
        Each dict has keys:

        * ``"name"`` — Check identifier string.
        * ``"passed"`` — ``True``, ``False``, or ``None`` (if the
          required fields could not be extracted).
        * ``"details"`` — Human-readable explanation.
        * ``"evidence"`` — Dict of supporting OCR line strings.
    """
    amt = extract_amounts(lines)
    total = amt.get("total")
    subtotal = amt.get("subtotal")
    tax = amt.get("tax")

    out: List[Dict[str, Any]] = []
    if total is not None and subtotal is not None and tax is not None:
        lhs = subtotal + tax
        passed = abs(lhs - total) <= Decimal("0.02")
        out.append({
            "name": "subtotal+tax==total",
            "passed": bool(passed),
            "details": (
                f"{subtotal}+{tax}={'%.2f' % float(lhs)} "
                f"vs total={'%.2f' % float(total)}"
            ),
            "evidence": {
                "subtotal_line": amt.get("subtotal_line"),
                "tax_line": amt.get("tax_line"),
                "total_line": amt.get("total_line"),
            },
        })
    else:
        out.append({
            "name": "subtotal+tax==total",
            "passed": None,
            "details": "missing subtotal/tax/total from OCR",
            "evidence": {
                "subtotal_line": amt.get("subtotal_line"),
                "tax_line": amt.get("tax_line"),
                "total_line": amt.get("total_line"),
            },
        })
    return out
