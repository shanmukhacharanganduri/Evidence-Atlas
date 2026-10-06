import io
import zipfile

from exports import export_docx, export_pdf

ROWS = [
    {"Criterion": "Attendance <b>Policy", "Evidence strength": "Located", "Review": "Not reviewed", "Color": "strong",
     "Finding": "Minimum attendance is 75% & ₹5,00,000 <i>unclosed", "Citation": "[Source: a.pdf, Page: 2]", "synthetic": True},
    {"Criterion": "Library", "Evidence strength": "No evidence located", "Review": "Not reviewed", "Color": "none",
     "Finding": "None found.", "Citation": "—"},
]


def test_pdf_export_survives_markup_and_unicode():
    assert export_pdf(ROWS, "KMEC (synthetic demo)").startswith(b"%PDF")


def test_docx_export_carries_scope_and_synthetic_status():
    text = zipfile.ZipFile(io.BytesIO(export_docx(ROWS, "KMIT"))).read("word/document.xml").decode("utf-8")
    assert "Scope: KMIT" in text and "synthetic demonstration source" in text and "KMEC" not in text


def test_exports_accept_empty_coverage():
    assert export_pdf([], "All institutions").startswith(b"%PDF")
    assert export_docx([], "All institutions")
