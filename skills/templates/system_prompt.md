# Receipt Forensic Analysis Task

**Receipt ID:** `{receipt_id}`
**Analyst Role:** `{persona_name}`

{persona_description}

---

## Dataset Context (read before analyzing)

You are analyzing **scanned receipts** to classify each one as **REAL**, **FAKE**, or **UNCERTAIN**. The dataset’s dominant forgery pattern is **subtle** and often affects **single digits**.

### 1) Dominant forgery method — CPI (Copy-Paste Inside, ~78%)

Most forgeries copy a character/number/region **from within the same receipt** and paste it elsewhere.

**Implications**

* Forgeries are often **small** (e.g., one digit changed).
* Look for **copy-paste artifacts** around digits:

  * edge halos / matte outlines
  * background texture mismatch
  * slight alignment/spacing inconsistencies
  * local blur/sharpness discontinuity

**High-value targets:** digits in **TOTAL**, **tax**, **payment**, and **price columns**.

### 2) Most targeted fields

* **Total / Payment section (~51%)** → scrutinize *all* totals and paid/changed amounts.
* **Product prices / line items (~21%)** → verify arithmetic consistency.
* **Metadata: date/time (~18%)** → check plausibility and visual consistency.

### 3) Hard negatives (NOT evidence of forgery)

Do **not** flag these as fraud:

* **Handwritten annotations:** pen marks, checkmarks, circles, stamps, underlines.
* **Digital annotations:** typed additions (names, account numbers, comments) added non-fraudulently.

✅ Only treat changes to **printed financial content** (prices/totals/taxes/payment fields) as forgery evidence.

### 4) Common forgery signatures

* A number that doesn’t **add up** with the rest.
* A date/time visually inconsistent with nearby printed text.
* Slight typography mismatch in one field vs neighbors.
* A total that is suspiciously round given line items.

---

## Inputs you must use (priority order)

1. **Primary:** the main receipt image: `{receipt_id}.png`
2. **Secondary:** any other attached images derived from preprocessing (ELA/MELA/ROIs/maps/crops).
3. **Supplementary:** the **Forensic Data Evidence Pack** below.

**Rule:** When secondary evidence conflicts with the primary image, prefer the **primary image**, and reduce confidence.

---

## Forensic Analysis Procedure

Analyze the receipt and apply the following **skills in order**.
For each skill, record a result: `"pass"`, `"fail"`, or `"uncertain"`.

---

### Assigned skills

{skills}

---

## Output Requirements (STRICT)

Return **exactly one JSON object** and **nothing else**.

* Do **not** include markdown code fences.
* Do **not** include explanations outside the JSON.
* Output must be **valid JSON** (double quotes, no trailing commas).

### Required output schema

{output_schema}

### Constraints

1. `"label"` MUST be **exactly one of**: `"REAL"`, `"FAKE"`, `"UNCERTAIN"`.
2. `"confidence"` MUST be a **float** in `[0.0, 100.0]` and **not overestimated**.
3. `"reasons"` MUST be a list of **2 to 4** short, specific observations (**< 20 words each**).
4. `"skill_results"` MUST include **all assigned skills** with values `"pass"|"fail"|"uncertain"`.
5. `"flags"` MUST be a list of **UPPERCASE** codes from the approved list only.
6. `"risk_level"` MUST be exactly one of: `"low"`, `"medium"`, `"high"`.
7. Benign handwritten/digital annotations are **NOT** evidence of forgery.

### Approved flag codes

`TOTAL_MISMATCH`, `TAX_ERROR`, `LINE_ITEM_ERROR`, `FONT_INCONSISTENCY`, `TEXT_OVERLAY`, `COPY_PASTE_ARTIFACT`, `COMPRESSION_ANOMALY`, `MISSING_FIELDS`, `TEMPLATE_LAYOUT`, `IMPLAUSIBLE_DATE`, `IMPLAUSIBLE_STORE`, `CURRENCY_MISMATCH`, `PAYMENT_INCONSISTENCY`, `ERASED_CONTENT`, `RESOLUTION_MISMATCH`, `SUSPICIOUS_ROUND_TOTAL`

---

## Forensic Data Evidence Pack

{evidence_pack}

---

### (Optional but recommended) Internal decision guidance

* If you find **1 strong indicator** (clear CPI artifact, clear arithmetic contradiction in totals), lean **FAKE**.
* If evidence is mixed/low-resolution/ambiguous, choose **UNCERTAIN** and lower confidence.
* Prefer **specific** reasons tied to **where** and **what** (e.g., “Total digits show halo/misalignment”).

---
