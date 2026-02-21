# LLM-Judge Fake Receipt Detector

A tool that uses multiple LLM "judges" to vote whether a receipt is FAKE or REAL, with short rationales.

---

## Architecture

The system is a **multi-judge voting pipeline** built around three vision-language models accessed via the HuggingFace Inference API:

```
Dataset → Sampler (20 receipts) → Image Loader
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
     Judge 1 (Qwen2.5-VL)     Judge 2 (Qwen2.5-VL)     Judge 3 (GLM-4.5V)
     "Forensic Accountant"    "Document Examiner"       "Holistic Auditor"
     T=0.2, math/context      T=0.6, visual/typo        T=0.3, all skills
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        ▼
                                  VotingEngine
                            (dynamic uncertainty-weighted)
                                        │
                                  Final Verdict
                             FAKE/REAL/UNCERTAIN
```

Each judge receives the receipt image plus a **structured system prompt** that includes:
- A unique forensic persona description
- 5 ordered "skills" (rubric checklist) to apply
- A rigid JSON output schema to enforce structured responses

The **VotingEngine** aggregates the three `JudgeResult` objects via dynamic uncertainty-weighted voting (default), with majority and confidence-weighted as alternative strategies.

**Key design decisions:**
- No OCR required: models reason directly from the image, which avoids an extra dependency and lets the judge flag visual anomalies that OCR would miss.
- Strict JSON output template + multi-attempt retry + validation pipeline eliminates most hallucination and free-text noise.
- Two Qwen2.5-VL personas + one GLM-4.5V provide model diversity without requiring three separate API accounts.

---

## Judge Prompts (Core)

Each judge's system prompt is assembled by `skills/rubric.py` from modular template files. The core structure is:

```
=== RECEIPT FORENSIC ANALYSIS TASK ===
Receipt ID: {receipt_id}
Analyst Role: {persona_name}

{persona_description}

[SKILL 1 - MATH_CONSISTENCY]
- Verify sum of items = subtotal
- Verify subtotal + tax = total
- Flag impossible rounding, suspicious round totals
- ...

[SKILL 2 - TYPOGRAPHY_ANALYSIS]
- Flag mixed font families
- Detect irregular spacing, baseline anomalies
- ...

[SKILL 3 - VISUAL_AUTHENTICITY]
- Detect JPEG compression inconsistencies
- Flag copy-paste artifacts, halos, hard edges
- ...

[SKILL 4 - LAYOUT_STRUCTURE]
- Verify header → items → total → payment structure
- Flag generic template layouts
- ...

[SKILL 5 - CONTEXTUAL_VALIDATION]
- Verify date plausibility
- Verify store name/address coherence
- Flag payment method inconsistencies
- ...

Output ONLY the following JSON (no markdown, no extra text):
{ "label": "FAKE|REAL|UNCERTAIN", "confidence": 0-100, "reasons": [...], ... }
```

**Why this structure:**
- Ordered skills force the model to reason step-by-step before reaching a verdict, reducing anchoring on superficial cues.
- Separating skills by focus area maps to the two Qwen personas (one numeric, one visual), which improves signal quality per judge.
- The strict JSON-only instruction with a concrete example schema dramatically reduces free-text hallucination and makes parsing reliable.

---

## Output Schema (Extended)

```json
{
  "label": "FAKE|REAL|UNCERTAIN",
  "confidence": 87.5,
  "reasons": ["specific observation 1", "specific observation 2"],
  "skill_results": {
    "math_consistency": "pass|fail|uncertain",
    "typography": "pass|fail|uncertain",
    "visual_authenticity": "pass|fail|uncertain",
    "layout_structure": "pass|fail|uncertain",
    "contextual_validation": "pass|fail|uncertain"
  },
  "flags": ["TOTAL_MISMATCH", "FONT_INCONSISTENCY"],
  "risk_level": "low|medium|high"
}
```

---

## Sampling

- **Random seed:** 42
- **Method:** Stratified random sample — `random.sample()` called separately on sorted REAL list and sorted FAKE list, then combined and shuffled (all with seed 42).
- **Selected 20 receipts:** See `outputs/samples.json` (generated after running `python main.py sample`).

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your HuggingFace token
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_...

# 3. Download and extract dataset
python main.py download

# 4. Select 20-receipt sample
python main.py sample

# 5. Run all 3 judges on all 20 receipts
python main.py run

# 6. Evaluate results
python main.py evaluate

# 7. Quick demo on a single receipt
python main.py demo <receipt_id>
```

---

## AI Tools Used

- **Claude (claude-sonnet-4-6 via Claude Code):** Used as the primary coding assistant to scaffold the project architecture, implement all modules, design the skill templates and prompt structure, and write tests.
- **HuggingFace InferenceClient:** Runtime API for all three judge models (Qwen2.5-VL-72B-Instruct x2 + GLM-4.5V).

---

## Project Structure

```
├── configs/          # Judge and sampling configuration (YAML)
├── data/             # Raw dataset + selected samples
├── forensic/         # Modular forensic analysis tools (MELA, CPI, DCT, noise)
├── judges/           # LLM judge implementations + voting engine
├── skills/           # Rubric system: 5 forensic skill templates
├── pipeline/         # Dataset loading, sampling, evaluation, forensic orchestration
├── notebooks/        # EDA + evaluation analysis notebooks
├── outputs/          # samples.json + per-receipt result JSONs
├── tests/            # Unit tests for judges, voting, pipeline
└── main.py           # CLI entry point
```
