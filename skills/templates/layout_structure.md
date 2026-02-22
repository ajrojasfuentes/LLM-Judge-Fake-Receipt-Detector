[LAYOUT_STRUCTURE]

Apply the checks below to detect **structural / layout forgeries** (misalignments, broken grids, duplicated rows, template-like artifacts).

**Key context**

* **CPI (Copy-Paste Inside, ~78%)** can subtly disrupt the layout by inserting duplicated regions that don’t perfectly match the original grid.
* The **Total / Payment** area is the most targeted and often shows spacing/alignment anomalies when numbers are edited without adjusting surrounding layout.

**Hard negative (do not mislabel as forgery)**

* Handwritten notes, stamps, circles, or underlines may look “off-grid” but are benign.
  ✅ Evaluate only the **printed** layout structure.

**Checks to perform**

* **Standard receipt flow**

  * Verify the receipt follows a coherent order (allowing minor variations):
    `Header (store identity + date/time)` → `Item list` → `Subtotal/Tax/Total` → `Payment` → `Footer`
  * Flag suspiciously missing core elements (when expected for a real receipt):

    * no store name/identity
    * no date/time
    * no total due / amount due
    * no item list (unless clearly a single-item receipt)

* **Header block coherence**

  * Store identity block (name/address/phone) should appear as a grouped header region.
  * Date/time should be positioned plausibly (often near header or near transaction details).
  * Flag header text that looks “floating”, mis-grouped, or oddly spaced compared to surrounding lines.

* **Item table structure and column integrity**

  * Check for consistent column roles across the item list (common patterns):

    * `item name | qty | unit price | line total`
    * `item name | amount | quantity` (simplified)
  * **Price/amount column alignment:** amounts should generally be **right-aligned** or consistently aligned to a clear column edge.
  * A CPI-pasted amount may be shifted slightly left/right or not match the column baseline.

* **Decimal and currency alignment**

  * Within the same column, decimal points (or last digits) often align visually.
  * Flag single values that break the alignment pattern (especially in totals/prices).

* **Total section spatial separation**

  * Subtotal/tax/total should be visually distinct from item rows (often separated by:

    * whitespace gap
    * divider line
    * bold/uppercase labels like “TOTAL”
  * Flag cases where:

    * totals look embedded in the item list without separation
    * grand total placement is inconsistent with the structure
    * the total amount is misaligned with the rest of the total block labels

* **Vertical spacing irregularities**

  * Look for unusual gaps or compressed spacing between specific rows:

    * **CUT** may leave odd blank space where a line was removed.
    * **CPI** may introduce an unnatural tight/loose gap around the pasted region.
  * Pay special attention around:

    * the last few items before subtotal
    * the transition into the total/payment section

* **Duplicate line / repeated structure detection**

  * Scan for duplicated or near-duplicated line items:

    * identical descriptions/prices appearing twice in suspicious proximity
    * repeated formatting blocks (same spacing and token pattern)
  * CPI often introduces “echo” patterns: repeated fragments that look too similar.

* **Grid continuity and edge alignment**

  * Check whether text blocks follow consistent left margins and column edges.
  * Flag local “jumps” where one line’s start position is offset compared to adjacent lines.
  * In thermal receipts, small natural drift is normal—flag only *localized* discontinuities.

* **Template / synthetic receipt cues**

  * Flag layouts that resemble generic receipt generators:

    * perfectly uniform spacing everywhere
    * overly clean background with no scan/print noise
    * placeholder-like content (“ITEM 1”, “STORE NAME”, “ADDRESS HERE”)
    * unnaturally consistent font rendering with no print artifacts

* **Payment section coherence and placement**

  * Payment lines should appear **after** the totals section and be spatially grouped.
  * Payment method content should match the receipt structure (card/cash/change lines).
  * Flag payment blocks that appear out of order (e.g., payment above subtotal) or misaligned with totals.

* **Footer appropriateness**

  * Footer (thanks message, return policy, barcode/QR, membership prompts) should be plausible for the store type/locale.
  * Flag footers that look inconsistent with the rest of the receipt (e.g., abrupt style change, missing expected footer elements, or strange unrelated content).

**Result**
Record as:

* `"pass"`: layout is coherent and consistent with a real receipt.
* `"fail"`: at least one strong structural anomaly suggests tampering (localized misalignment, broken grid, suspicious duplication).
* `"uncertain"`: layout cannot be judged reliably (crop, blur, occlusion, or insufficient resolution).
