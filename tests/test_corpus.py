import json

import corpus
from conftest import make_pdf
from corpus import clean_pages, load_corpus, split_page, split_sentences


def test_sentence_split_keeps_abbreviations_and_numbering():
    parts = split_sentences("2. Minimum CGPA Requirement\nA student must secure a CGPA of 6.50 or higher to qualify. "
                            "Dr. S. Raghavan chaired the Ph.D. committee meeting today. Another full sentence follows here.")
    assert parts[0] == "2. Minimum CGPA Requirement"
    assert any(p.startswith("Dr. S. Raghavan chaired the Ph.D. committee") for p in parts)
    assert len(parts) == 4


def test_chunks_follow_sentence_boundaries_and_overlap():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(60))
    chunks = split_page(text, "a.pdf", 3, size=50, overlap=14)
    assert len(chunks) > 1
    assert all(c.text.rstrip().endswith(".") for c in chunks)
    assert all(c.page == 3 and c.source == "a.pdf" for c in chunks)
    assert chunks[0].text.splitlines()[-1] in chunks[1].text  # sentence-aligned overlap


def test_clean_pages_removes_numbers_timestamps_and_repeated_furniture():
    pages = [f"Official Header\nFirst body line of page {i}.\nSecond body line is useful.\nThird body line.\n"
             f"Page {i}/9 21-11-2024 01:31:35\nOfficial Footer Line\n{i}" for i in range(1, 6)]
    for page in clean_pages(pages):
        assert "Official Footer" not in page and "Official Header" not in page and "21-11-2024" not in page
        assert "Second body line is useful." in page  # repeated body text is not furniture


def test_every_pdf_is_registered_including_scanned(tmp_path):
    docs = tmp_path / "docs"
    make_pdf(docs / "kmec_archive" / "kmit" / "real.pdf", [["Attendance of 75% is mandatory for every student."]])
    make_pdf(docs / "scan.pdf", [[]])
    chunks, records, _ = load_corpus(docs, tmp_path / "cache")
    by_source = {r.source: r for r in records}
    assert set(by_source) == {"kmec_archive/kmit/real.pdf", "scan.pdf"}
    assert by_source["scan.pdf"].status == "no_text" and by_source["scan.pdf"].zero_text_pages == [1]
    assert by_source["kmec_archive/kmit/real.pdf"].institution == "KMIT"
    assert all(c.institution == "KMIT" for c in chunks)


def test_same_chunk_count_edit_changes_fingerprint_and_unchanged_corpus_skips_extraction(tmp_path, monkeypatch):
    docs, cache = tmp_path / "docs", tmp_path / "cache"
    make_pdf(docs / "a.pdf", [["The minimum attendance is 75% in each course."]])
    _, _, first = load_corpus(docs, cache)

    calls = []
    original = corpus.extract_document
    monkeypatch.setattr(corpus, "extract_document", lambda *a: calls.append(a) or original(*a))
    _, _, again = load_corpus(docs, cache)
    assert again == first and calls == []  # unchanged corpus: no PDF extraction

    make_pdf(docs / "a.pdf", [["The minimum attendance is 65% in each course."]])  # same chunk count, new content
    _, _, edited = load_corpus(docs, cache)
    assert edited != first and len(calls) == 1


def test_registry_sets_institution_and_synthetic_flag(tmp_path):
    docs = tmp_path / "docs"
    make_pdf(docs / "demo.pdf", [["Demonstration policy text with enough words to form a chunk."]])
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"demo.pdf": {"institution": "KMEC (synthetic demo)", "synthetic": True}}))
    chunks, records, _ = load_corpus(docs, tmp_path / "cache", registry)
    assert records[0].synthetic and chunks[0].synthetic and chunks[0].institution == "KMEC (synthetic demo)"


def test_corrupt_pdf_is_recorded_not_raised(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "broken.pdf").write_bytes(b"not a pdf")
    chunks, records, _ = load_corpus(docs, tmp_path / "cache")
    assert chunks == [] and records[0].status == "error" and records[0].error


def test_empty_corpus_does_not_raise(tmp_path):
    chunks, records, _ = load_corpus(tmp_path / "missing", tmp_path / "cache")
    assert chunks == [] and records == []
