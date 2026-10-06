import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def make_pdf(path: Path, pages: list[list[str]]) -> None:
    """Write a small text PDF; each page is a list of lines (an empty list gives a blank page)."""
    from reportlab.pdfgen import canvas
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path))
    for lines in pages:
        y = 800
        for line in lines:
            pdf.drawString(50, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()
