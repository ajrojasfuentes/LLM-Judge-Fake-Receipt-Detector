[MATH_CONSISTENCY]

Apply the checks below to detect **numerical forgeries**.

**Key context**

* **Total / Payment** fields are the **most targeted** entities.
* **Line-item prices** are the second most targeted.
  If a receipt is forged, it very often involves **numbers**. Scrutinize **all monetary values** with maximum precision.

**Checks to perform**

* **Line-item arithmetic**

  * For each product line, verify: `quantity × unit_price = line_total` (when all fields exist).
  * A single CPI-edited digit often creates a mismatch here.

* **Subtotal verification**

  * Verify the sum of line totals (or item prices, depending on layout) equals the stated **subtotal**.
  * Allow for common rounding conventions (rounding at item-level vs at subtotal-level).

* **Tax calculation**

  * Verify: `subtotal + tax = grand_total`.
  * If a tax rate can be inferred, it should be plausible. Common rates include:
    `5%, 6%, 7%, 8%, 9%, 10%, 12%, 15%, 18%, 20%, 23%, 25%`
  * Flag tax values/rates that are inconsistent with the subtotal or appear implausible.

* **Grand total consistency**

  * Cross-check the grand total against all related fields:

    * `TOTAL DUE`, `AMOUNT DUE`, `BALANCE`, `NET`, `TOTAL`, etc.
  * If stated total contradicts items + tax (or items − discounts + tax), treat as strong forgery evidence.

* **Payment arithmetic**

  * **Cash:** verify `cash_tendered ≥ total` and `change = cash_tendered − total`.
  * **Card:** charged amount should equal the total due (unless split payment is explicitly shown).
  * Flag mismatches between payment lines and the stated total.

* **Suspiciously round totals**

  * Flag totals that are unusually round (e.g., `100.00`, `50.00`) when line items suggest an irregular result.
  * Round totals are a common sign of “edited total without fixing line items”.

* **Discount / coupon arithmetic**

  * Verify discounts are applied correctly: `subtotal − discount + tax = total` (or equivalent layout).
  * Flag impossible discounts:

    * discount > eligible amount
    * negative totals created by discounts
    * discounts that don’t affect the final total at all

* **Impossible or malformed values**

  * Negative item prices (unless clearly a refund/return).
  * Zero-price items with normal descriptions (unless clearly “FREE”/promo).
  * Missing numeric fields where a value is expected.
  * Non-integer quantities for clearly countable items (when it doesn’t make sense).

* **Suspicious duplicates / copy patterns**

  * Identical monetary values appearing in suspicious positions:

    * two different items with exactly the same uncommon price
    * a line total matching the grand total
    * repeated totals across unrelated fields
  * These can indicate copy-paste reuse of a number.

**Result**

* `"pass"`: all checks pass with no contradictions.
* `"fail"`: at least one clear arithmetic inconsistency is found.
* `"uncertain"`: numbers are not readable enough (image quality/occlusion) to verify.
