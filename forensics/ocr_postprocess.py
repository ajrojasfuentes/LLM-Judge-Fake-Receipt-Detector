# forensics.ocr_postprocess.py
"""
OCR post-processing + arithmetic verification (NO OCR engine).

This module is meant to consume a paired OCR transcription (.txt) and produce:
- cleaned text
- structured candidates (company/date/amounts/totals/items/payments)
- arithmetic verification report (subtotal/tax/total/paid/change consistency)

It is a robust replacement / extension of the minimal helpers previously embedded
inside a pipeline file.

Usage:
    from pathlib import Path
    from ocr_postprocess import extract_ocr_from_txt

    res = extract_ocr_from_txt(Path("receipt.txt"))
    print(res.to_dict())

Notes:
- This module does NOT run PaddleOCR/Tesseract/EasyOCR.
- It parses and verifies what OCR already produced.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import re


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class OCRParseConfig:
    # Prompt-size cap / safety
    max_cleaned_chars: int = 4000

    # Line processing
    max_lines_for_company: int = 8
    max_company_lines: int = 5
    max_date_candidates: int = 8
    max_total_candidates: int = 12
    max_item_candidates: int = 30
    max_amounts: int = 40

    # Arithmetic tolerance
    # A check passes if abs(diff) <= max(tolerance_abs, tolerance_pct * ref / 100)
    tolerance_abs: Decimal = Decimal("0.10")
    tolerance_pct: Decimal = Decimal("10.0")

    # If total is very small/zero, only abs tolerance is used.
    min_ref_for_pct: Decimal = Decimal("1.00")

    # If an item line has multiple amounts, how to pick the "line amount"
    # "rightmost" is typically best for receipts.
    item_amount_policy: str = "rightmost"  # one of {"rightmost", "largest"}

    # Currency codes/tokens frequently seen in OCR
    currency_tokens: Tuple[str, ...] = (
        "RM", "MYR", "USD", "SGD", "EUR", "GBP", "COP", "MXN", "BRL", "PEN",
        "$", "€", "£"
    )

    # Keep at most N amounts per line (for performance / noise)
    max_amounts_per_line: int = 6


# -----------------------------
# Dataclasses: structured output
# -----------------------------

@dataclass
class MoneyToken:
    raw: str
    value: Decimal
    currency: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["value"] = str(self.value)
        return d


@dataclass
class OCRLine:
    idx: int
    text: str
    norm: str
    money: List[MoneyToken] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "text": self.text,
            "norm": self.norm,
            "money": [m.to_dict() for m in self.money],
            "keywords": self.keywords,
        }


@dataclass
class OCRStructured:
    quality_score: float
    cleaned_text: str
    company_lines: List[str]
    date_candidates: List[str]

    # Candidate lines (raw snippets)
    total_candidates: List[str]
    item_candidates: List[str]
    payment_candidates: List[str]

    # Parsed amounts (unique, sorted)
    all_amounts: List[str]  # decimal strings

    # Rich parsed lines (optional, useful for debugging)
    parsed_lines: List[OCRLine] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["parsed_lines"] = [ln.to_dict() for ln in self.parsed_lines]
        return d


@dataclass
class ArithmeticCheck:
    name: str
    lhs: Optional[str]
    rhs: Optional[str]
    diff: Optional[str]
    diff_pct: Optional[float]
    passes: Optional[bool]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArithmeticReport:
    item_count: int
    item_sum: str
    subtotal: Optional[str]
    tax: Optional[str]
    discount: Optional[str]
    rounding: Optional[str]
    service_charge: Optional[str]
    total: Optional[str]
    paid: Optional[str]
    change: Optional[str]

    checks: List[ArithmeticCheck]
    arithmetic_consistent: Optional[bool]
    best_explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_count": self.item_count,
            "item_sum": self.item_sum,
            "subtotal": self.subtotal,
            "tax": self.tax,
            "discount": self.discount,
            "rounding": self.rounding,
            "service_charge": self.service_charge,
            "total": self.total,
            "paid": self.paid,
            "change": self.change,
            "checks": [c.to_dict() for c in self.checks],
            "arithmetic_consistent": self.arithmetic_consistent,
            "best_explanation": self.best_explanation,
        }


@dataclass
class OCRExtractionResult:
    structured: OCRStructured
    arithmetic_report: ArithmeticReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structured": self.structured.to_dict(),
            "arithmetic_report": self.arithmetic_report.to_dict(),
        }


# -----------------------------
# Keywords / Regex
# -----------------------------

# Normalize whitespace but keep line breaks
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

# Date patterns (several common variants)
_DATE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b"),  # 12/03/2024
    re.compile(r"\b\d{2,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}\b"),  # 2024-03-12
    re.compile(r"\b\d{1,2}\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\s*\d{2,4}\b", re.I),
]

# Important line keyword groups (uppercase matching on norm)
KW_TOTAL = (
    "GRAND TOTAL", "TOTAL DUE", "AMOUNT DUE", "AMOUNT PAYABLE", "TOTAL",
    "NETT", "NET", "BALANCE DUE", "TO PAY"
)
KW_SUBTOTAL = ("SUBTOTAL", "SUB TOTAL", "SUB-TOTAL")
KW_TAX = ("TAX", "GST", "VAT", "SST", "IVA", "IGV", "TAX AMOUNT")
KW_DISCOUNT = ("DISCOUNT", "DISC", "LESS", "PROMO", "COUPON")
KW_ROUNDING = ("ROUNDING", "ROUND OFF", "ROUNDED", "RND")
KW_SERVICE = ("SERVICE", "SVC", "SERVICE CHARGE", "SERVICECHARGE")

KW_PAID = ("PAID", "PAYMENT", "CASH", "TENDERED", "RECEIVED", "CARD", "VISA", "MASTERCARD", "DEBIT", "CREDIT")
KW_CHANGE = ("CHANGE", "BALANCE", "RETURN", "REFUND", "CASHBACK")

# Lines to ignore as company candidates
COMPANY_STOPWORDS = (
    "RECEIPT", "TAX INVOICE", "INVOICE", "CUSTOMER COPY", "MERCHANT COPY",
    "THANK YOU", "WELCOME", "TEL", "PHONE", "FAX", "EMAIL", "WWW", "HTTP"
)


def _norm(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s


def _contains_any(norm_line: str, phrases: Sequence[str]) -> bool:
    return any(p in norm_line for p in phrases)


def _find_date_candidates(lines: Sequence[str], max_n: int) -> List[str]:
    out: List[str] = []
    for ln in lines:
        for pat in _DATE_PATTERNS:
            if pat.search(ln):
                out.append(ln[:80])
                break
        if len(out) >= max_n:
            break
    return out


# -----------------------------
# Money parsing
# -----------------------------

# Money-like token finder:
# - optional currency prefix
# - number with optional thousand separators
# - optional decimals (2-3 typically)
# - optional trailing minus
# - exclude percentages (handled by check)
_MONEY_TOKEN_RE = re.compile(
    r"""
    (?P<cur>RM|MYR|USD|SGD|EUR|GBP|COP|MXN|BRL|PEN|\$|€|£)?\s*
    (?P<num>
        [\(\-]?\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2,3})?[\)\-]?
        |
        [\(\-]?\d+(?:[.,]\d{2,3})?[\)\-]?
    )
    (?!\s*%)   # not a percentage
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Detect span of date to avoid parsing date fragments as money
def _date_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _parse_decimal(num_raw: str) -> Optional[Decimal]:
    """
    Robust parse of OCR numeric formats:
    - handles thousand separators and decimal separators: 1,234.56 / 1.234,56 / 1234,56 / 1234.56
    - handles parentheses negatives: (12.34)
    - handles trailing minus: 12.34-
    - handles spaces as thousand separators: 1 234,56
    """
    s = num_raw.strip()

    # Remove surrounding currency junk already stripped by regex group.
    # Keep parentheses and minus for sign detection.
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()

    if s.endswith("-") and s[:-1].strip().replace(".", "").replace(",", "").isdigit():
        negative = True
        s = s[:-1].strip()

    if s.startswith("-"):
        negative = True
        s = s[1:].strip()

    # normalize internal spaces
    s = re.sub(r"\s+", "", s)

    # Identify decimal separator:
    # - if both '.' and ',', last occurrence is decimal sep
    # - if only one of them appears: if exactly 2-3 digits after it => decimal sep else thousand sep
    last_dot = s.rfind(".")
    last_com = s.rfind(",")

    dec_sep: Optional[str] = None
    thou_sep: Optional[str] = None

    if last_dot != -1 and last_com != -1:
        # both present
        dec_sep = "." if last_dot > last_com else ","
        thou_sep = "," if dec_sep == "." else "."
    elif last_dot != -1:
        # only dot
        if len(s) - last_dot - 1 in (2, 3):
            dec_sep = "."
        else:
            thou_sep = "."
    elif last_com != -1:
        # only comma
        if len(s) - last_com - 1 in (2, 3):
            dec_sep = ","
        else:
            thou_sep = ","

    if thou_sep:
        s = s.replace(thou_sep, "")

    if dec_sep and dec_sep != ".":
        s = s.replace(dec_sep, ".")

    # now s should be a plain decimal or integer
    try:
        val = Decimal(s)
    except InvalidOperation:
        return None

    if negative:
        val = -val

    return val


def extract_money_tokens(text: str, config: OCRParseConfig) -> List[MoneyToken]:
    spans = _date_spans(text)
    out: List[MoneyToken] = []
    for m in _MONEY_TOKEN_RE.finditer(text):
        # skip if overlaps a detected date span
        token_span = (m.start(), m.end())
        if any(_overlaps(token_span, ds) for ds in spans):
            continue

        cur = m.group("cur")
        num_raw = m.group("num")
        raw = (cur or "") + num_raw
        raw = raw.strip()

        # Quick reject: avoid tiny tokens like "1" unless it has decimals or currency
        # (helps reduce noise in quantity columns).
        has_decimal = bool(re.search(r"[.,]\d{2,3}\b", num_raw))
        if not has_decimal and not cur:
            # allow integers only if they look like plausible totals (>= 10)
            digits_only = re.sub(r"[^\d]", "", num_raw)
            if not digits_only or int(digits_only) < 10:
                continue

        val = _parse_decimal(num_raw)
        if val is None:
            continue

        # cap number of tokens per line
        out.append(MoneyToken(raw=raw, value=val, currency=cur.upper() if cur else None))
        if len(out) >= config.max_amounts_per_line:
            break

    return out


def _pick_line_amount(tokens: List[MoneyToken], policy: str) -> Optional[Decimal]:
    if not tokens:
        return None
    if policy == "largest":
        return max((t.value for t in tokens), key=lambda d: abs(d))
    # default "rightmost": take last extracted token
    return tokens[-1].value


def _unique_sorted_amounts(lines: Sequence[OCRLine], limit: int) -> List[str]:
    seen: Dict[str, Decimal] = {}
    for ln in lines:
        for t in ln.money:
            # normalize to 2 decimals for uniqueness
            q = t.value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            seen[str(q)] = q
    vals = sorted(seen.values())
    return [str(v) for v in vals[:limit]]


# -----------------------------
# Cleaning + quality scoring
# -----------------------------

def clean_ocr_text(raw: str) -> str:
    # Remove control chars
    s = _CTRL_CHARS_RE.sub(" ", raw)
    # Normalize spaces but keep line breaks
    # - collapse multiple spaces/tabs
    s = re.sub(r"[ \t]{2,}", " ", s)
    # normalize weird unicode spaces
    s = s.replace("\u00a0", " ")
    return s.strip()


def compute_quality_score(cleaned: str, lines: Sequence[str]) -> float:
    """
    Simple heuristic quality score in [0, 1].
    Uses:
    - ratio of useful chars (alnum + common receipt punctuation)
    - presence of dates
    - presence of money tokens
    - structural richness (#lines)
    """
    total_chars = max(len(cleaned), 1)
    useful_chars = sum(1 for c in cleaned if c.isalnum() or c in ".,:-/$%()")
    base = min((useful_chars / total_chars) * 1.4, 1.0)

    has_date = any(any(p.search(cleaned) for p in _DATE_PATTERNS) for _ in [0])
    money_hits = len(re.findall(r"[.,]\d{2}\b", cleaned))
    line_count = len(lines)

    bonus = 0.0
    if has_date:
        bonus += 0.06
    if money_hits >= 3:
        bonus += 0.08
    elif money_hits >= 1:
        bonus += 0.04
    if line_count >= 10:
        bonus += 0.06
    elif line_count >= 5:
        bonus += 0.03

    return float(min(base + bonus, 1.0))


# -----------------------------
# Structuring logic
# -----------------------------

def _extract_company_lines(lines: Sequence[str], parsed: Sequence[OCRLine], config: OCRParseConfig) -> List[str]:
    """
    Heuristic: take first N lines until we see strong signs of dates/money,
    skipping generic receipt headers.
    """
    out: List[str] = []
    n = min(config.max_lines_for_company, len(lines))

    for i in range(n):
        ln = lines[i].strip()
        if not ln:
            continue
        norm = _norm(ln)
        if _contains_any(norm, COMPANY_STOPWORDS):
            continue

        # stop if current line already has money/date
        if parsed[i].money:
            break
        if any(p.search(ln) for p in _DATE_PATTERNS):
            break

        out.append(ln[:80])
        if len(out) >= config.max_company_lines:
            break

    return out


def _line_keywords(norm_line: str) -> List[str]:
    kws: List[str] = []
    # Order matters: more specific first
    if _contains_any(norm_line, KW_TOTAL):
        kws.append("TOTAL")
    if _contains_any(norm_line, KW_SUBTOTAL):
        kws.append("SUBTOTAL")
    if _contains_any(norm_line, KW_TAX):
        kws.append("TAX")
    if _contains_any(norm_line, KW_DISCOUNT):
        kws.append("DISCOUNT")
    if _contains_any(norm_line, KW_ROUNDING):
        kws.append("ROUNDING")
    if _contains_any(norm_line, KW_SERVICE):
        kws.append("SERVICE")
    if _contains_any(norm_line, KW_PAID):
        kws.append("PAID")
    if _contains_any(norm_line, KW_CHANGE):
        kws.append("CHANGE")
    return kws


def _score_total_line(ln: OCRLine, total_priority: Sequence[str]) -> int:
    """
    Score candidate total lines:
    - keyword priority
    - position (later lines) should typically matter, but we keep scoring simple.
    """
    score = 0
    norm = ln.norm
    # priority phrases (more specific)
    for i, phrase in enumerate(total_priority):
        if phrase in norm:
            score += 200 - i * 20
            break
    # generic total indicator
    if "TOTAL" in norm:
        score += 40
    # taxes etc often appear near totals but we still allow them (lower)
    if "TAX" in norm or "GST" in norm or "VAT" in norm:
        score -= 10
    # prefer lines with exactly 1-2 money tokens (less noisy)
    if len(ln.money) == 1:
        score += 20
    elif len(ln.money) >= 3:
        score -= 10
    return score


def structure_ocr_text(ocr_text: str, config: Optional[OCRParseConfig] = None) -> OCRStructured:
    cfg = config or OCRParseConfig()

    cleaned = clean_ocr_text(ocr_text)
    raw_lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]

    # Parse each line into OCRLine + money tokens
    parsed_lines: List[OCRLine] = []
    for idx, ln in enumerate(raw_lines):
        n = _norm(ln)
        money = extract_money_tokens(ln, cfg)
        kws = _line_keywords(n)
        parsed_lines.append(OCRLine(idx=idx, text=ln, norm=n, money=money, keywords=kws))

    quality = compute_quality_score(cleaned, raw_lines)

    # Company lines
    company_lines = _extract_company_lines(raw_lines, parsed_lines, cfg)

    # Dates
    date_candidates = _find_date_candidates(raw_lines, cfg.max_date_candidates)

    # Total/payment/item candidates (raw snippet lists)
    total_candidates: List[str] = []
    item_candidates: List[str] = []
    payment_candidates: List[str] = []

    # Lines that should NOT be items
    item_exclude_if_has = ("TOTAL", "SUBTOTAL", "TAX", "DISCOUNT", "ROUNDING", "SERVICE", "PAID", "CHANGE")

    for ln in parsed_lines:
        if not ln.money:
            continue

        # classify
        has_any_totalish = any(k in item_exclude_if_has for k in ln.keywords)

        if "PAID" in ln.keywords or "CHANGE" in ln.keywords:
            payment_candidates.append(ln.text[:120])
        if has_any_totalish:
            total_candidates.append(ln.text[:120])
        else:
            # Many receipts have item lines w/ money; filter obvious non-items
            if _contains_any(ln.norm, COMPANY_STOPWORDS):
                continue
            item_candidates.append(ln.text[:120])

    # Deduplicate while preserving order
    def _dedupe(seq: Sequence[str], limit: int) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= limit:
                break
        return out

    total_candidates = _dedupe(total_candidates, cfg.max_total_candidates)
    item_candidates = _dedupe(item_candidates, cfg.max_item_candidates)
    payment_candidates = _dedupe(payment_candidates, cfg.max_total_candidates)

    all_amounts = _unique_sorted_amounts(parsed_lines, cfg.max_amounts)

    # Cap cleaned text for prompt usage
    cleaned_cap = cleaned[: cfg.max_cleaned_chars]

    return OCRStructured(
        quality_score=quality,
        cleaned_text=cleaned_cap,
        company_lines=company_lines,
        date_candidates=date_candidates,
        total_candidates=total_candidates,
        item_candidates=item_candidates,
        payment_candidates=payment_candidates,
        all_amounts=all_amounts,
        parsed_lines=parsed_lines,
    )


# -----------------------------
# Arithmetic verification
# -----------------------------

def _fmt_money(d: Optional[Decimal]) -> Optional[str]:
    if d is None:
        return None
    return str(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _tol_pass(diff: Decimal, ref: Decimal, cfg: OCRParseConfig) -> Tuple[bool, Optional[float]]:
    """
    Returns (passes, diff_pct)
    """
    ad = abs(diff)
    if ref >= cfg.min_ref_for_pct:
        diff_pct = float((ad / ref) * Decimal("100"))
        tol = max(cfg.tolerance_abs, (cfg.tolerance_pct * ref) / Decimal("100"))
        return (ad <= tol), float(Decimal(str(diff_pct)).quantize(Decimal("0.1")))
    else:
        return (ad <= cfg.tolerance_abs), None


def _best_amount_from_lines(
    parsed_lines: Sequence[OCRLine],
    want: str,
    priority_phrases: Sequence[str],
) -> Optional[Decimal]:
    """
    Generic extractor for subtotal/tax/total/paid/change etc:
    - choose best line that contains relevant keyword(s)
    - pick rightmost/most plausible amount on that line
    """
    candidates: List[Tuple[int, int, Decimal]] = []

    for ln in parsed_lines:
        if not ln.money:
            continue
        if want not in ln.keywords and not any(p in ln.norm for p in priority_phrases):
            continue

        # Score: priority phrase match + line index weight
        score = 0
        for i, phrase in enumerate(priority_phrases):
            if phrase in ln.norm:
                score += 200 - i * 20
                break
        if want in ln.keywords:
            score += 50

        # pick last amount (usually total printed at end)
        amount = ln.money[-1].value
        candidates.append((score, ln.idx, amount))

    if not candidates:
        return None

    # Prefer: higher score; if tie, later line (higher idx)
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


def compute_arithmetic_report(structured: OCRStructured, config: Optional[OCRParseConfig] = None) -> ArithmeticReport:
    cfg = config or OCRParseConfig()

    parsed_lines = structured.parsed_lines

    # Item amounts: pick a line amount from each item candidate line by matching OCRLine.text
    item_amounts: List[Decimal] = []
    item_text_set = set(structured.item_candidates)

    for ln in parsed_lines:
        if not ln.money:
            continue
        if ln.text[:120] not in item_text_set:
            continue
        amt = _pick_line_amount(ln.money, cfg.item_amount_policy)
        if amt is not None:
            item_amounts.append(amt)

    item_sum = sum(item_amounts, Decimal("0.00")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # Extract core fields
    subtotal = _best_amount_from_lines(parsed_lines, "SUBTOTAL", ("SUBTOTAL", "SUB TOTAL", "SUB-TOTAL"))
    tax = _best_amount_from_lines(parsed_lines, "TAX", ("GST", "VAT", "TAX", "SST", "IVA", "IGV"))
    discount = _best_amount_from_lines(parsed_lines, "DISCOUNT", ("DISCOUNT", "DISC", "LESS", "PROMO", "COUPON"))
    rounding = _best_amount_from_lines(parsed_lines, "ROUNDING", ("ROUNDING", "ROUND OFF", "ROUNDED", "RND"))
    service_charge = _best_amount_from_lines(parsed_lines, "SERVICE", ("SERVICE CHARGE", "SERVICE", "SVC"))

    # Total: use a priority list (more specific first)
    total_priority = (
        "GRAND TOTAL", "TOTAL DUE", "AMOUNT DUE", "AMOUNT PAYABLE",
        "BALANCE DUE", "TOTAL", "NETT", "NET", "TO PAY"
    )
    total = _best_amount_from_lines(parsed_lines, "TOTAL", total_priority)

    paid = _best_amount_from_lines(parsed_lines, "PAID", ("TENDERED", "RECEIVED", "CASH", "PAID", "PAYMENT"))
    change = _best_amount_from_lines(parsed_lines, "CHANGE", ("CHANGE", "CASHBACK", "RETURN", "REFUND", "BALANCE"))

    # Build checks
    checks: List[ArithmeticCheck] = []

    # Check A: item_sum ~ subtotal (if both available)
    if subtotal is not None and item_sum != 0:
        diff = (subtotal - item_sum).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        passes, diff_pct = _tol_pass(diff, subtotal, cfg)
        checks.append(
            ArithmeticCheck(
                name="items_vs_subtotal",
                lhs=_fmt_money(item_sum),
                rhs=_fmt_money(subtotal),
                diff=_fmt_money(diff),
                diff_pct=diff_pct,
                passes=passes,
                details={"meaning": "subtotal - sum(items)"},
            )
        )

    # Check B: subtotal + tax + service - discount + rounding ~ total
    # Use subtotal if present, else item_sum as base.
    base = subtotal if subtotal is not None else (item_sum if item_sum != 0 else None)
    if base is not None and total is not None:
        expected = base
        if tax is not None:
            expected += tax
        if service_charge is not None:
            expected += service_charge
        if discount is not None:
            expected -= abs(discount)  # discount often printed positive; treat as subtraction
        if rounding is not None:
            expected += rounding

        expected = expected.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        diff = (total - expected).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        passes, diff_pct = _tol_pass(diff, total, cfg)
        checks.append(
            ArithmeticCheck(
                name="expected_total_vs_total",
                lhs=_fmt_money(expected),
                rhs=_fmt_money(total),
                diff=_fmt_money(diff),
                diff_pct=diff_pct,
                passes=passes,
                details={
                    "meaning": "total - expected",
                    "base": _fmt_money(base),
                    "tax": _fmt_money(tax),
                    "service_charge": _fmt_money(service_charge),
                    "discount": _fmt_money(discount),
                    "rounding": _fmt_money(rounding),
                },
            )
        )

    # Check C: paid - change ~ total
    if paid is not None and change is not None and total is not None:
        net_paid = (paid - change).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        diff = (net_paid - total).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        passes, diff_pct = _tol_pass(diff, total, cfg)
        checks.append(
            ArithmeticCheck(
                name="paid_minus_change_vs_total",
                lhs=_fmt_money(net_paid),
                rhs=_fmt_money(total),
                diff=_fmt_money(diff),
                diff_pct=diff_pct,
                passes=passes,
                details={"meaning": "(paid - change) - total"},
            )
        )

    # Determine overall consistency:
    # - True if any check passes
    # - False if we have enough info to test and all checks fail
    # - None if insufficient info
    arithmetic_consistent: Optional[bool]
    best_explanation: Optional[str] = None

    pass_checks = [c for c in checks if c.passes is True]
    fail_checks = [c for c in checks if c.passes is False]
    unknown_checks = [c for c in checks if c.passes is None]

    if pass_checks:
        arithmetic_consistent = True
        # choose best: smallest abs diff, prefer total-related checks
        def _rank(c: ArithmeticCheck) -> Tuple[int, Decimal]:
            total_related = 0 if "total" in c.name else 1
            diff = Decimal(c.diff) if c.diff is not None else Decimal("999999")
            return (total_related, abs(diff))
        best = sorted(pass_checks, key=_rank)[0]
        best_explanation = best.name
    else:
        # If we managed to build at least one check and all are false => inconsistent
        if checks and not unknown_checks and fail_checks:
            arithmetic_consistent = False
            # pick most relevant failure
            best_explanation = fail_checks[0].name
        else:
            arithmetic_consistent = None

    return ArithmeticReport(
        item_count=len(item_amounts),
        item_sum=_fmt_money(item_sum) or "0.00",
        subtotal=_fmt_money(subtotal),
        tax=_fmt_money(tax),
        discount=_fmt_money(discount),
        rounding=_fmt_money(rounding),
        service_charge=_fmt_money(service_charge),
        total=_fmt_money(total),
        paid=_fmt_money(paid),
        change=_fmt_money(change),
        checks=checks,
        arithmetic_consistent=arithmetic_consistent,
        best_explanation=best_explanation,
    )


# -----------------------------
# High-level API
# -----------------------------

def extract_ocr_from_txt(txt_path: Path, config: Optional[OCRParseConfig] = None) -> OCRExtractionResult:
    cfg = config or OCRParseConfig()
    raw = txt_path.read_text(encoding="utf-8", errors="replace")
    structured = structure_ocr_text(raw, cfg)
    report = compute_arithmetic_report(structured, cfg)
    return OCRExtractionResult(structured=structured, arithmetic_report=report)


def extract_ocr_from_text(ocr_text: str, config: Optional[OCRParseConfig] = None) -> OCRExtractionResult:
    cfg = config or OCRParseConfig()
    structured = structure_ocr_text(ocr_text, cfg)
    report = compute_arithmetic_report(structured, cfg)
    return OCRExtractionResult(structured=structured, arithmetic_report=report)


# -----------------------------
# CLI (optional)
# -----------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Post-process OCR text and compute arithmetic verification.")
    ap.add_argument("txt", type=str, help="Path to OCR .txt file")
    ap.add_argument("--tol_abs", type=str, default=None, help="Absolute tolerance (e.g. 0.10)")
    ap.add_argument("--tol_pct", type=str, default=None, help="Percent tolerance (e.g. 10.0)")
    args = ap.parse_args()

    cfg = OCRParseConfig(
        tolerance_abs=Decimal(args.tol_abs) if args.tol_abs else OCRParseConfig().tolerance_abs,
        tolerance_pct=Decimal(args.tol_pct) if args.tol_pct else OCRParseConfig().tolerance_pct,
    )

    res = extract_ocr_from_txt(Path(args.txt), cfg)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))