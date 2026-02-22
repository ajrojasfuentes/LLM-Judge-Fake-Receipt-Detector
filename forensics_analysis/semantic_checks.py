from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

# Basic money token (avoid %)
_MONEY_TOKEN_RE = re.compile(r"(-?\d{1,7}(?:[.,]\d{2,3})?)(?!\s*%)")
_KEY_TOTAL_RE = re.compile(r"\b(total|amount due|grand total|balance due)\b", re.I)
_KEY_SUBTOTAL_RE = re.compile(r"\b(subtotal|sub total)\b", re.I)
_KEY_TAX_RE = re.compile(r"\b(tax|vat|gst)\b", re.I)


def _to_decimal(s: str) -> Optional[Decimal]:
    s = s.strip().replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def extract_amounts(lines: List[str]) -> Dict[str, Any]:
    """
    Heuristic extraction of subtotal/tax/total from OCR lines.
    """
    best = {"total": None, "subtotal": None, "tax": None}
    evidence = {"total_line": None, "subtotal_line": None, "tax_line": None}

    for ln in lines:
        l = ln.strip()
        if not l:
            continue

        m = _MONEY_TOKEN_RE.findall(l)
        if not m:
            continue
        # take last number in line (often the value)
        val = _to_decimal(m[-1])
        if val is None:
            continue

        if best["total"] is None and _KEY_TOTAL_RE.search(l):
            best["total"] = val
            evidence["total_line"] = l
            continue
        if best["subtotal"] is None and _KEY_SUBTOTAL_RE.search(l):
            best["subtotal"] = val
            evidence["subtotal_line"] = l
            continue
        if best["tax"] is None and _KEY_TAX_RE.search(l):
            best["tax"] = val
            evidence["tax_line"] = l
            continue

    return {**best, **evidence}


def semantic_checks(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Run simple accounting checks if possible: subtotal + tax ≈ total
    """
    amt = extract_amounts(lines)
    total = amt.get("total")
    subtotal = amt.get("subtotal")
    tax = amt.get("tax")

    out: List[Dict[str, Any]] = []
    if total is not None and subtotal is not None and tax is not None:
        lhs = subtotal + tax
        # tolerance: 0.02
        passed = abs(lhs - total) <= Decimal("0.02")
        out.append({
            "name": "subtotal+tax==total",
            "passed": bool(passed),
            "details": f"{subtotal}+{tax}={'%.2f' % float(lhs)} vs total={'%.2f' % float(total)}",
            "evidence": {
                "subtotal_line": amt.get("subtotal_line"),
                "tax_line": amt.get("tax_line"),
                "total_line": amt.get("total_line"),
            }
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
            }
        })
    return out