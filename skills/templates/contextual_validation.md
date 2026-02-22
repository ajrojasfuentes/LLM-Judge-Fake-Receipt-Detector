[CONTEXTUAL_VALIDATION]

Apply the checks below to detect **contextual/semantic forgeries** (changes that “make sense visually” but break real-world consistency).

**Key context**

* **Metadata fields (date/time)** are a common modification target (~18%).
* **Company/store identity** is also targeted (~6%).
  Forgers often change **date** or **store name** to fit a fraudulent expense claim. These edits can be semantically inconsistent even if visually subtle.

**Hard negative (do not mislabel as forgery)**

* Handwritten notes (circles, checkmarks, names, account numbers) and typed annotations can be benign.
  ✅ Focus only on whether the **printed receipt content** is internally consistent.

**Checks to perform**

* **Date plausibility**

  * Not in the future (relative to today).
  * Not implausibly old (≈ >10 years is suspicious).
  * Date format matches the apparent locale/country.
  * If day-of-week is shown, it must match the calendar date.
  * Time of purchase is plausible for the store type (e.g., unusual late-night hours).

* **Metadata field forgery (date/time)**

  * Inspect date/time text for visual consistency with nearby printed text:

    * font style/weight, alignment, spacing, ink rendering, baseline, and placement in the field.
  * Look for subtle signs of IMI/PIX edits (local texture mismatch, misaligned glyphs).

* **Store name and identity coherence**

  * Placeholder/generic names (e.g., “Store Name”, “My Shop”, “Company”) ⇒ suspicious.
  * Store name should match the apparent business type and locale.
  * Phone format should match the apparent country/region.
  * Address format (if present) should look realistic for the region.

* **Item plausibility for store type**

  * Items should match the store category (grocery/pharmacy/restaurant/hardware, etc.).
  * Cross-category mixes that don’t fit (e.g., groceries + electronics at a small market) are suspicious.
  * Brand names should be plausible (avoid invented-looking brands or nonsense tokens).

* **Currency and locale consistency**

  * Currency symbol/code must match the apparent country/locale.
  * Prices, tax style/rates, and payment method conventions should be regionally plausible.

* **Payment method consistency**

  * If **card**: look for card type and/or last-4 digits (if shown) and ensure formatting is coherent.
  * If **cash**: change should be present and arithmetically consistent with paid vs total.
  * If hints of card number format exist, they should not contradict the card type.

* **Internal contradictions**

  * Any field contradicting another is suspicious, for example:

    * Day-of-week doesn’t match the printed date.
    * “Tax Invoice” but no tax lines anywhere.
    * Subtotal equals total despite non-zero tax/fees.
    * Store hours implied by the receipt contradict the transaction time.

* **Receipt / transaction number plausibility**

  * Should not be zero or obviously “template-like” (e.g., “0001”, “1234”).
  * Should look consistent with the receipt’s overall formatting and metadata style.

**Result**
Record as `"pass"`, `"fail"`, or `"uncertain"` based on the overall strength and consistency of findings.
