[VISUAL_AUTHENTICITY]

Apply the checks below to detect **visual / image-level forgeries**.

**Key context**

* The dominant forgery method is **CPI (Copy-Paste Inside, ~77%)**: duplicating a region from the same receipt and placing it elsewhere.
* Secondary methods include **PIX** (pixel edits), **CUT** (deletion), and **CPO** (external paste).
  Implications:
* Most edits are **subtle** (often a single digit).
* Highest-risk zones are **price fields**, **tax**, and **totals/payment**.
* CPI often creates regions with **unnaturally repeated pixel patterns** or boundary artifacts.

**Hard negative (do not mislabel as forgery)**

* Handwritten notes (checkmarks, circles, underlines, stamps) and benign digital notes are **not** forgery evidence.
  ✅ Focus on fraudulent changes to **printed financial/identity content**.

**Checks to perform**

* **Copy-paste traces (CPI)**

  * Look for hard edges, halos, slight color fringing, or sharpness boundaries around digits/characters.
  * Watch for subtle background shade differences behind a pasted digit.
  * Prioritize totals and price columns.

* **Local compression / block inconsistency**

  * Even in PNG, edited receipts may have passed through JPEG at some stage.
  * Flag regions where blockiness/ringing/noise patterns differ from adjacent areas (suggesting recompression or localized edits).

* **Pixel-level digit edits (PIX)**

  * Inspect individual digits for “stroke surgery” (e.g., `1→7`, `0→8`).
  * Look for jagged edges, mismatched pixel density, inconsistent stroke width, or unnatural corners on a single character.

* **Erased / whitened areas (CUT or cleanup)**

  * Flag rectangular white patches, smudged zones, or overly uniform color regions where content may have been removed.
  * Check whether the erased area breaks surrounding noise/texture continuity.

* **Text overlay / digital insertion**

  * Flag text that appears “on top” of the paper:

    * missing paper/thermal texture beneath the glyphs
    * different anti-aliasing / subpixel rendering
    * unnatural crispness compared to nearby printed text

* **Resolution / sharpness mismatches**

  * Check if a small region (often totals/prices) appears at a different resolution, blur level, or edge sharpness than the rest.
  * Copy-pasted content may carry different DPI or anti-aliasing characteristics.

* **Paper texture consistency**

  * Verify paper grain, scanner noise, fold lines, and background texture are coherent across the receipt.
  * Pasted regions may look unnaturally clean or “too uniform”.

* **Lighting & shadow uniformity**

  * Ensure lighting gradients and exposure are consistent.
  * Edited patches often lack the natural brightness falloff or shadow continuity present elsewhere.

**Result**
Record as `"pass"`, `"fail"`, or `"uncertain"` (use `"uncertain"` only if the image is too low-quality to judge artifacts reliably).
