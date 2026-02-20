"""
Streamlit app (optional extension) — LLM-Judge Fake Receipt Detector.

Run with:
    streamlit run app/main.py
"""

# TODO: Implement Streamlit UI after core pipeline is validated.
# Suggested components:
#   - Receipt browser (list of 20 samples with labels)
#   - Receipt viewer (image + metadata)
#   - "Run judges" button (triggers main.py run for selected receipt)
#   - Results panel (per-judge JSON + final vote)

import streamlit as st

st.title("LLM-Judge Fake Receipt Detector")
st.info("Optional UI — to be implemented after core pipeline validation.")
