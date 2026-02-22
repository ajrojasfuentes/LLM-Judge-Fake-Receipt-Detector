# forensics.semantic_check.py
"""
Semantic checks (accounting validation) built on top of ocr_postprocess.py.

Goals:
- Provide the convenient [{name, passed, details, evidence}] output style
  from the original semantic_checks.py
- BUT reuse the robust parsing and structured representation from
  ocr_postprocess.py (MoneyToken, OCRLine, OCRStructured, ArithmeticReport)
- Preserve the important "subtotal before total" / "avoid sub-total being
  matched as total" behavior using explicit excludes.

This module does NOT run OCR. It consumes OCR text or OCRExtractionResult.

Example:
    from pathlib import Path
    from semantic_checks_v2 import semantic_checks_from_txt

    checks = semantic_checks_from_txt(Path("receipt.txt"))
    for c in checks:
        print(c["name"], c["passed"], c["details"])
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ocr_postprocess import (
    OCRParseConfig,
    OCRExtractionResult,
    OCRStructured,
    extract_ocr_from_txt,
    extract_ocr_from_text,
)


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class SemanticChecksConfig:
    # strict check tolerance for subtotal+tax ~= total
    strict_tol_abs: Decimal = Decimal("0.02")
    # if you want a percent-based strict tol too (optional)
    strict_tol_pct: Optional[Decimal] = None  # e.g. Decimal("0.5") for 0.5%

    # how many evidence lines to include
    evidence_max_len: int = 160


# -----------------------------
# Regex patterns (like semantic_checks.py, but used for line selection)
# -----------------------------

_KEY_SUBTOTAL_RE = re.compile(r"\b(subtotal|sub total|sub-total)\b", re.I)
_KEY_TAX_RE = re.compile(r"\b(tax|vat|gst|sst|iva|igv)\b", re.I)
_KEY_TOTAL_RE = re.compile(r"\b(total|amount due|grand total|balance due|amount payable|total due)\b", re.I)


def _fmt(d: Optional[Decimal]) -> Optional[str]:
    if d is None:
        return None
    return str(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _line_amount_rightmost(ln) -> Optional[Decimal]:
    # OCRLine.money is already parsed; pick rightmost token like receipts typically do.
    if not getattr(ln, "money", None):
        return None
    return ln.money[-1].value


def _score_line(norm: str, idx: int, priority_phrases: Sequence[str]) -> int:
    """
    Scoring heuristic:
    - keyword/phrase priority
    - later lines get a small boost (totals usually appear near the bottom)
    """
    s = 0
    for i, phr in enumerate(priority_phrases):
        if phr in norm:
            s += 200 - i * 20
            break
    s += min(idx, 80)  # light late-line preference
    return s


def _pick_best_line(
    parsed_lines,
    include_re: re.Pattern,
    exclude_res: Sequence[re.Pattern] = (),
    priority_phrases: Sequence[str] = (),
) -> Tuple[Optional[Decimal], Optional[str]]:
    best_score = None
    best_amt: Optional[Decimal] = None
    best_text: Optional[str] = None

    # precompute uppercase phrases for norm matching
    pri = [p.upper() for p in priority_phrases]

    for ln in parsed_lines:
        text = ln.text
        norm = ln.norm
        if not ln.money:
            continue

        if not include_re.search(text):
            continue
        if any(ex.search(text) for ex in exclude_res):
            continue

        amt = _line_amount_rightmost(ln)
        if amt is None:
            continue

        score = _score_line(norm, ln.idx, pri)
        if best_score is None or score > best_score:
            best_score = score
            best_amt = amt
            best_text = text

    return best_amt, best_text


def extract_amounts_from_structured(structured: OCRStructured) -> Dict[str, Any]:
    """
    Similar spirit to semantic_checks.extract_amounts(), but:
    - uses parsed money tokens from ocr_postprocess
    - explicitly prevents subtotal lines from being treated as total
    - returns evidence lines
    """
    lines = structured.parsed_lines

    subtotal, subtotal_line = _pick_best_line(
        lines,
        include_re=_KEY_SUBTOTAL_RE,
        exclude_res=(),
        priority_phrases=("SUBTOTAL", "SUB TOTAL", "SUB-TOTAL"),
    )

    tax, tax_line = _pick_best_line(
        lines,
        include_re=_KEY_TAX_RE,
        exclude_res=(),
        priority_phrases=("GST", "VAT", "TAX", "SST", "IVA", "IGV"),
    )

    # IMPORTANT: when extracting TOTAL, explicitly EXCLUDE subtotal lines
    # to avoid "SUB TOTAL" being matched as total.
    total, total_line = _pick_best_line(
        lines,
        include_re=_KEY_TOTAL_RE,
        exclude_res=(_KEY_SUBTOTAL_RE,),
        priority_phrases=("GRAND TOTAL", "TOTAL DUE", "AMOUNT DUE", "AMOUNT PAYABLE", "BALANCE DUE", "TOTAL"),
    )

    return {
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "subtotal_line": subtotal_line,
        "tax_line": tax_line,
        "total_line": total_line,
    }


def _passes_strict(lhs: Decimal, rhs: Decimal, cfg: SemanticChecksConfig) -> bool:
    diff = abs(lhs - rhs)

    tol = cfg.strict_tol_abs
    if cfg.strict_tol_pct is not None and rhs != 0:
        pct_tol = (cfg.strict_tol_pct * abs(rhs)) / Decimal("100")
        tol = max(tol, pct_tol)

    return diff <= tol


def _truncate(s: Optional[str], n: int) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def semantic_checks_from_structured(
    structured: OCRStructured,
    arithmetic_report: Optional[Dict[str, Any]] = None,
    cfg: Optional[SemanticChecksConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a list of semantic check dicts.

    - Includes a strict 'subtotal + tax ~= total' check (like semantic_checks.py)
    - Optionally also includes converted checks from ocr_postprocess ArithmeticReport
      if you pass arithmetic_report (e.g., res.arithmetic_report.to_dict()).
    """
    cfg = cfg or SemanticChecksConfig()
    amt = extract_amounts_from_structured(structured)

    subtotal: Optional[Decimal] = amt["subtotal"]
    tax: Optional[Decimal] = amt["tax"]
    total: Optional[Decimal] = amt["total"]

    out: List[Dict[str, Any]] = []

    # 1) strict subtotal+tax ~= total
    if total is not None and subtotal is not None and tax is not None:
        lhs = (subtotal + tax).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        passed = _passes_strict(lhs, total, cfg)
        out.append({
            "name": "subtotal+tax≈total_strict",
            "passed": bool(passed),
            "details": f"{_fmt(subtotal)} + {_fmt(tax)} = {_fmt(lhs)} vs total={_fmt(total)} "
                       f"(tol_abs={str(cfg.strict_tol_abs)})",
            "evidence": {
                "subtotal_line": _truncate(amt.get("subtotal_line"), cfg.evidence_max_len),
                "tax_line": _truncate(amt.get("tax_line"), cfg.evidence_max_len),
                "total_line": _truncate(amt.get("total_line"), cfg.evidence_max_len),
            },
            "values": {
                "subtotal": _fmt(subtotal),
                "tax": _fmt(tax),
                "lhs": _fmt(lhs),
                "total": _fmt(total),
            }
        })
    else:
        out.append({
            "name": "subtotal+tax≈total_strict",
            "passed": None,
            "details": "missing subtotal/tax/total from OCR (strict check)",
            "evidence": {
                "subtotal_line": _truncate(amt.get("subtotal_line"), cfg.evidence_max_len),
                "tax_line": _truncate(amt.get("tax_line"), cfg.evidence_max_len),
                "total_line": _truncate(amt.get("total_line"), cfg.evidence_max_len),
            },
            "values": {
                "subtotal": _fmt(subtotal) if subtotal is not None else None,
                "tax": _fmt(tax) if tax is not None else None,
                "total": _fmt(total) if total is not None else None,
            }
        })

    # 2) Convert ocr_postprocess arithmetic_report checks (optional)
    if arithmetic_report:
        for chk in arithmetic_report.get("checks", []):
            # chk is expected to look like:
            # {name, lhs, rhs, diff, diff_pct, passes, details}
            out.append({
                "name": f"arith::{chk.get('name')}",
                "passed": chk.get("passes"),
                "details": (
                    f"lhs={chk.get('lhs')} rhs={chk.get('rhs')} diff={chk.get('diff')} "
                    f"diff_pct={chk.get('diff_pct')} details={chk.get('details')}"
                ),
                "evidence": {},
                "values": {
                    "lhs": chk.get("lhs"),
                    "rhs": chk.get("rhs"),
                    "diff": chk.get("diff"),
                    "diff_pct": chk.get("diff_pct"),
                }
            })

        # Also surface the final decision if present
        out.append({
            "name": "arith::overall",
            "passed": arithmetic_report.get("arithmetic_consistent"),
            "details": f"best_explanation={arithmetic_report.get('best_explanation')}",
            "evidence": {},
            "values": {
                "arithmetic_consistent": arithmetic_report.get("arithmetic_consistent"),
                "best_explanation": arithmetic_report.get("best_explanation"),
            }
        })

    return out


def semantic_checks_from_result(
    res: OCRExtractionResult,
    cfg: Optional[SemanticChecksConfig] = None,
) -> List[Dict[str, Any]]:
    # OCRExtractionResult has structured + arithmetic_report objects.
    # We convert arithmetic_report to dict if it has to_dict().
    ar = res.arithmetic_report
    ar_dict = ar.to_dict() if hasattr(ar, "to_dict") else ar  # type: ignore[assignment]
    return semantic_checks_from_structured(res.structured, arithmetic_report=ar_dict, cfg=cfg)


def semantic_checks_from_text(
    ocr_text: str,
    parse_cfg: Optional[OCRParseConfig] = None,
    sem_cfg: Optional[SemanticChecksConfig] = None,
) -> List[Dict[str, Any]]:
    res = extract_ocr_from_text(ocr_text, parse_cfg)
    return semantic_checks_from_result(res, sem_cfg)


def semantic_checks_from_txt(
    txt_path: Path,
    parse_cfg: Optional[OCRParseConfig] = None,
    sem_cfg: Optional[SemanticChecksConfig] = None,
) -> List[Dict[str, Any]]:
    res = extract_ocr_from_txt(txt_path, parse_cfg)
    return semantic_checks_from_result(res, sem_cfg)