[TYPOGRAPHY_ANALYSIS]

Apply the checks below to detect **typographic forgeries** (font/spacing/rendering anomalies).

**Key context**
Common typography-related forgery methods:

* **IMI (Imitation Insertion, ~8%)**: a number/character is retyped to match the font → expect subtle size/weight/spacing differences.
* **CPI (Copy-Paste Inside, ~78%)**: a character/value is copied from elsewhere in the same receipt → font may match, but alignment/context often won’t.
* **PIX (Pixel editing, ~7%)**: a stroke is edited to change a digit (e.g., `1→7`, `0→8`) → expect unnatural stroke artifacts.

**Hard negative (do not mislabel as forgery)**

* Handwritten marks (checkmarks, circles, underlines, stamps, pen-written notes) are **not** typographic forgeries.
  ✅ Focus primarily on **printed** text.

**Checks to perform**

* **Font consistency**

  * Printed sections that should share a font (totals, prices, date/time, item rows) should look consistent.
  * Flag localized areas where font rendering differs from neighbors.

* **Serif / sans-serif mixing**

  * Unexpected mixing (serif vs sans-serif) within the same logical region (e.g., one item line in a different style) is suspicious.

* **Character spacing and baseline alignment**

  * Look for irregular kerning/letter spacing in a single word/number group.
  * Baseline shifts are high-signal for CPI/IMI (a digit slightly above/below neighbors).
  * Pay extra attention to numeric fields: totals, taxes, line totals, dates/times.

* **Weight and size inconsistencies**

  * Flag characters that appear slightly bolder/lighter or at a different point size than surrounding printed text.
  * This is a key indicator for **IMI**.

* **Rendering / resolution mismatch**

  * Detect text that has different sharpness, edge behavior, or anti-aliasing compared to adjacent text.
  * Copy-pasted text from a different source may carry different DPI/hinting or compression artifacts.

* **Text overlay “halo” and background mismatch**

  * Flag digits/words that look like they “float” above the paper:

    * halo outlines
    * background color patching
    * missing paper/thermal texture beneath the glyphs

* **Pixel-stroke artifacts (PIX edits)**

  * Look for single digits with strokes that appear digitally drawn rather than printed:

    * jagged pixel edges
    * inconsistent stroke width
    * pixel-art look in an otherwise smooth print region

* **Capitalization and encoding anomalies**

  * Inconsistent capitalization patterns within the same style block (e.g., one line differs oddly).
  * Garbled symbols/replacement characters can indicate re-rendered text or pipeline artifacts (treat as suspicious if localized).

**Result**
Record as `"pass"`, `"fail"`, or `"uncertain"` (use `"uncertain"` if typography cannot be judged due to blur/low resolution/occlusion).
