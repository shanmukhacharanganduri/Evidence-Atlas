"""Streamlit AppTest checks against a counted fake backend: no models, no PDFs, deterministic."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import rag_pipeline
from answering import AnswerResult
from corpus import Chunk, DocumentRecord

SUPPORT = Chunk("a.pdf::p2::c0", "A minimum attendance of 75% in each registered course is mandatory.", "a.pdf", 2, "KMEC (synthetic demo)", True)
OTHER = Chunk("b.pdf::p1::c0", "Attendance is recorded for every session.", "b.pdf", 1, "KMEC (synthetic demo)", True)


class FakeRAG:
    def __init__(self):
        self.calls = {"answer": 0, "coverage": 0}
        self.chunks = [SUPPORT, OTHER]
        self.documents = [DocumentRecord("a.pdf", "KMEC (synthetic demo)", True, chunks=1, pages=3, status="ok"),
                          DocumentRecord("scan.pdf", "KMIT", False, chunks=0, pages=2, zero_text_pages=[1, 2], status="no_text")]
        self.institutions = ["KMEC (synthetic demo)", "KMIT"]
        self.index_version = "abc123"

    def answer(self, query, method="", institution=None):
        self.calls["answer"] += 1
        candidates = [SimpleNamespace(chunk=OTHER, score=1.0), SimpleNamespace(chunk=SUPPORT, score=0.9)]
        return AnswerResult(query=query, status="answered", statement=SUPPORT.text, citation=SUPPORT, support_rank=2,
                            candidates=candidates, method=method, institution=institution)

    def criterion_coverage(self, institution=None):
        self.calls["coverage"] += 1
        return [{"Criterion": "Attendance Policy", "Evidence strength": "Located", "Review": "Not reviewed", "Institution": "KMEC (synthetic demo)",
                 "Finding": SUPPORT.text, "Citation": "[Source: a.pdf, Page: 2]", "source": "a.pdf", "page": "2", "Color": "strong", "synthetic": True}]

    def governance_gaps(self, institution=None):
        return []

    def render_cited_page(self, *a):
        return b""

    def page_count(self, source):
        return 3

    def page_text(self, source, page):
        return "text"

    def source_bytes(self, source):
        return b"%PDF"


@pytest.fixture
def app(monkeypatch):
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()
    fake = FakeRAG()
    monkeypatch.setattr(rag_pipeline, "build_rag", lambda: fake)
    test = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    test.fake = fake
    return test.run()


def ask(app, question):
    app.text_input(key="question").set_value(question)
    app.button[0].click()
    return app.run()


def test_startup_runs_neither_search_nor_coverage(app):
    assert not app.exception
    assert app.fake.calls == {"answer": 0, "coverage": 0}


def test_submitting_a_question_calls_the_backend_exactly_once(app):
    app = ask(app, "What attendance percentage is mandatory?")
    assert app.fake.calls["answer"] == 1
    assert any("75%" in m.value for m in app.markdown)


def test_scope_filter_and_viewer_buttons_do_not_retrieve_again(app):
    app = ask(app, "What attendance percentage is mandatory?")
    app.selectbox(key="scope").select("KMIT").run()
    app.text_input(key="corpus_filter").set_value("scan").run()
    open_buttons = [b for b in app.button if b.label.startswith("Open cited page")]
    assert open_buttons, "the supporting passage must offer a direct source action"
    open_buttons[0].click().run()
    assert app.fake.calls["answer"] == 1
    others = [b for b in app.button if b.label.startswith("Open page")]
    assert len({b.label for b in others}) == len(others)  # distinct accessible names


def test_empty_question_shows_validation_and_does_not_search(app):
    app.text_input(key="question").set_value("   ")
    app.button[0].click()
    app.run()
    assert app.fake.calls["answer"] == 0
    assert any("Enter a question" in w.value for w in app.warning)


def test_coverage_runs_only_on_demand_and_is_cached(app):
    assert app.fake.calls["coverage"] == 0
    [b for b in app.button if b.key == "run_coverage"][0].click()
    app.run()
    assert app.fake.calls["coverage"] == 1
    app.selectbox(key="scope").select("KMIT").run()  # a different scope needs its own run
    assert app.fake.calls["coverage"] == 1


def test_scanned_documents_stay_visible_in_the_corpus_tab(app):
    frames = [d.value for d in app.dataframe]
    assert any("scan.pdf" in set(frame["Document"]) for frame in frames if "Document" in frame)
