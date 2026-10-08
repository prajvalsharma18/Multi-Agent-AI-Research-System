from pathlib import Path

import export
from pypdf import PdfReader


def test_report_exports_markdown_and_readable_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "OUTPUT_DIR", tmp_path)
    report = "## Key Findings\n\nA finding with citation [Source](https://example.org).\n\n| A | B |\n|---|---|\n| 1 | 2 |"
    paths = export.save_report("A topic & research", report, "Revision Required: NO")
    md = Path(paths["markdown"]).read_text(encoding="utf-8")
    pdf_path = Path(paths["pdf"])
    pdf = pdf_path.read_bytes()
    assert "## Key Findings" in md
    assert "https://example.org" in md
    assert "Critic Review" in md
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 500
    pdf_reader = PdfReader(str(pdf_path))
    pdf_text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
    assert "�" not in pdf_text
    assert "https://example.org" in {
        str(annotation.get_object().get("/A", {}).get("/URI"))
        for page in pdf_reader.pages
        for annotation in page.get("/Annots", [])
        if annotation.get_object().get("/A", {}).get("/URI")
    }


def test_empty_and_special_character_report_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "OUTPUT_DIR", tmp_path)
    paths = export.save_report("Résumé <topic>", "", None)
    assert Path(paths["markdown"]).exists()
    pdf_path = Path(paths["pdf"])
    assert pdf_path.read_bytes().startswith(b"%PDF")
    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
    assert "Résumé" in pdf_text
    assert "�" not in pdf_text


def test_slugify_has_fallback_and_length_limit():
    assert export._slugify("!!!") == "report"
    assert len(export._slugify("x" * 100)) == 60
