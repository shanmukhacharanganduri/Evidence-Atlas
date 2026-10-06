import pytest

import rag_pipeline
from conftest import make_pdf
from rag_pipeline import AccreditationRAG


@pytest.fixture
def rag(tmp_path):
    docs = tmp_path / "docs"
    make_pdf(docs / "kmec_archive" / "kmit" / "att.pdf", [["A minimum attendance of 75% is mandatory for every examination at KMIT."]])
    make_pdf(docs / "kmec_archive" / "ngit" / "att.pdf", [["A minimum attendance of 65% is mandatory for every examination at NGIT."]])
    make_pdf(docs / "scan.pdf", [[]])
    for index in range(6):  # BM25 needs a few unrelated documents for positive term weights
        make_pdf(docs / "kmec_archive" / "kmit" / f"filler{index}.pdf", [[f"Canteen hours and parking notice number {index} for visitors."]])
    return AccreditationRAG(docs_dir=docs, chroma_dir=tmp_path / "chroma", cache_dir=tmp_path / "cache", registry_path=None)


def test_same_basename_in_different_institutions_stays_distinct(rag):
    assert {d.source for d in rag.documents} >= {"kmec_archive/kmit/att.pdf", "kmec_archive/ngit/att.pdf"}
    assert rag.institutions == ["KMIT", "NGIT"]


def test_institution_scope_never_returns_another_institutions_policy(rag):
    results = rag.search("minimum attendance mandatory examination", "BM25", institution="NGIT")
    assert results and {r.chunk.institution for r in results} == {"NGIT"}
    assert "65%" in results[0].chunk.text


def test_scanned_document_is_listed_and_reported_as_unsearchable(rag):
    assert [d.source for d in rag.unsearchable] == ["scan.pdf"]


def test_reranker_failure_is_disclosed_and_not_retried(rag, monkeypatch):
    attempts = []

    def failing(name):
        attempts.append(name)
        raise OSError("offline")

    monkeypatch.setattr(rag_pipeline, "_load_cross_encoder", failing)
    for _ in range(5):
        rag.bm25_rerank_search("minimum attendance", institution="KMIT")
    assert len(attempts) == 1
    assert any("reranking model is unavailable" in note for note in rag.notices())


def test_answer_without_models_uses_lexical_fallback_and_cites_the_page(rag, monkeypatch):
    monkeypatch.setattr(rag_pipeline, "_load_cross_encoder", lambda name: (_ for _ in ()).throw(OSError("offline")))
    result = rag.answer("What minimum attendance is mandatory?", "BM25", institution="KMIT")
    assert result.status == "answered" and "75%" in result.statement
    assert result.citation.source.endswith("kmit/att.pdf")
    assert any("reranking model is unavailable" in note for note in result.notices)


def test_unknown_entity_gives_insufficient_evidence_not_an_error(rag):
    result = rag.answer("Which department owns the Novalux asteroid laboratory?", "BM25")
    assert result.status == "insufficient_evidence" and result.formatted().startswith("Evidence not found")


def test_search_failure_becomes_a_processing_error_state(rag, monkeypatch):
    monkeypatch.setattr(rag, "search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    result = rag.answer("anything")
    assert result.status == "processing_error" and "boom" in result.error


def test_governance_alerts_are_scoped_per_institution(rag):
    # 75% (KMIT) and 65% (NGIT) are different institutions, so they must not be reported as a conflict.
    assert not any(a["level"] == "Potential variation" for a in rag.governance_gaps())


def test_governance_alert_detects_two_values_inside_one_institution(tmp_path):
    docs = tmp_path / "docs"
    make_pdf(docs / "kmec_archive" / "kmit" / "a.pdf", [["Minimum attendance is 75% for examinations."]])
    make_pdf(docs / "kmec_archive" / "kmit" / "b.pdf", [["Minimum attendance is 80% for laboratory courses."]])
    rag = AccreditationRAG(docs_dir=docs, chroma_dir=tmp_path / "chroma", cache_dir=tmp_path / "cache", registry_path=None)
    alerts = rag.governance_gaps("KMIT")
    assert any(a["level"] == "Potential variation" and "75%" in a["message"] and "80%" in a["message"] for a in alerts)


def test_faithfulness_check_with_no_evidence_does_not_crash(rag):
    out = rag.faithfulness_check("The attendance rule is 75%. [Source: a.pdf, Page: 1]", [])
    assert out and not out[0]["supported"]


def test_coverage_requires_topic_and_value_in_one_statement(rag, monkeypatch):
    monkeypatch.setattr(rag_pipeline, "_load_cross_encoder", lambda name: (_ for _ in ()).throw(OSError("offline")))
    rows = {r["Criterion"]: r for r in rag.criterion_coverage("KMIT")}
    assert rows["Attendance Policy"]["Citation"].startswith("[Source: kmec_archive/kmit/att.pdf")
    assert rows["Attendance Policy"]["Evidence strength"] == "Partial match"  # no reranker: never "Located"
    assert rows["Library Resources"]["Evidence strength"] == "No evidence located"
    assert all(r["Review"] == "Not reviewed" for r in rows.values())
