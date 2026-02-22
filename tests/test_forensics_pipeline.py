from pathlib import Path

import numpy as np
from PIL import Image

from forensics import pipeline as fp


def _make_img(path: Path) -> None:
    arr = np.full((128, 128, 3), 245, dtype=np.uint8)
    arr[30:34, 20:108] = 0
    arr[60:64, 20:108] = 0
    Image.fromarray(arr).save(path)


def test_build_evidence_pack_full_uses_new_reading_stack(tmp_path, monkeypatch):
    img = tmp_path / "receipt.png"
    txt = tmp_path / "receipt.txt"
    _make_img(img)
    txt.write_text("SUBTOTAL 10.00\nTAX 1.00\nTOTAL 11.00\nPAID 20.00\nCHANGE 9.00\n", encoding="utf-8")

    monkeypatch.setattr(fp, "BASE_EVIDENCE_DIR", tmp_path / "evidence")
    pack = fp.build_evidence_pack(img, output_dir=None, ocr_txt=txt, mode="FULL")

    assert pack["mode"] == "FULL"
    assert pack["reading"]["source"] == "ocr_postprocess"
    assert pack["reading"]["arithmetic_report"] is not None
    assert isinstance(pack["reading"]["semantic_checks"], list)
    assert Path(pack["evidence_json"]).exists()
    assert str(tmp_path / "evidence" / "receipt") in pack["output_dir"]


def test_build_evidence_pack_reading_mode_shape(tmp_path):
    img = tmp_path / "receipt2.png"
    txt = tmp_path / "receipt2.txt"
    _make_img(img)
    txt.write_text("TOTAL 12.00\n", encoding="utf-8")

    pack = fp.build_evidence_pack(img, output_dir=tmp_path / "out", ocr_txt=txt, mode="READING")
    assert pack["mode"] == "READING"
    assert "reading" in pack
    assert "global" not in pack
    assert "forensic" not in pack


def test_forensic_pipeline_context_evidence_pack_is_json_serializable(tmp_path):
    from pipeline.forensic_pipeline import ForensicPipeline
    import json

    img = tmp_path / "receipt3.png"
    txt = tmp_path / "receipt3.txt"
    _make_img(img)
    txt.write_text("TOTAL 12.00\n", encoding="utf-8")

    fp_ctx = ForensicPipeline(output_dir=tmp_path / "evidence")
    ctx = fp_ctx.analyze(img, ocr_txt_path=txt, mode="FULL")

    # Regression: prompt builder json.dumps() must not fail on ROI dataclasses
    json.dumps(ctx.evidence_pack, ensure_ascii=False)
