# -*- coding: utf-8 -*-
"""Evidence Atlas — accreditation evidence workspace.

Retrieval runs only on an explicit question submission, and its structured result is kept in
session state. Appearance, filter, source-viewer and download actions never call the backend.
"""
from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from answering import tokens
from rag_pipeline import ALL_INSTITUTIONS, DEFAULT_METHOD, ACCREDITATION_CRITERIA, build_rag, corpus_stamp

st.set_page_config(page_title="Evidence Atlas", layout="wide", initial_sidebar_state="collapsed")

def inject_theme() -> None:
    """Colour tokens live in theme.css (light-dark()), so they track Streamlit's native theme without a rerun."""
    css = (Path(__file__).resolve().parent / "theme.css").read_text("utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_rag(stamp: str):
    return build_rag()


@st.cache_data(show_spinner=False, max_entries=64)
def render_page(_rag, version: str, source: str, page: int, terms: tuple[str, ...]) -> bytes:
    return _rag.render_cited_page(source, page, list(terms))


@st.cache_data(show_spinner=False, max_entries=64)
def page_text(_rag, version: str, source: str, page: int) -> str:
    return _rag.page_text(source, page)


@st.cache_data(show_spinner=False, max_entries=32)
def review_alerts(_rag, version: str, scope: str) -> list[dict]:
    return _rag.governance_gaps(scope)


def short(value: str, length: int = 54) -> str:
    return value if len(value) <= length else value[: length - 1] + "…"


def highlight(text: str, query: str) -> str:
    output = html.escape(text)
    terms = sorted({t for t in tokens(query) if len(t) > 3}, key=len, reverse=True)
    if terms:
        output = re.sub("(" + "|".join(map(re.escape, terms)) + ")", r"<mark>\1</mark>", output, flags=re.I)
    return output


# ---------------------------------------------------------------- source viewer

def request_viewer(source: str, page: int, terms: list[str]) -> None:
    """Button callback: runs before the next render, so no explicit rerun is needed."""
    st.session_state["open_viewer"] = {"source": source, "page": page, "terms": tuple(terms)}


@st.dialog("Source page", width="large")
def source_dialog(rag, source: str, page: int, terms: tuple[str, ...]) -> None:
    key = "viewer_page"
    st.session_state.setdefault(key, page)
    try:
        total = rag.page_count(source)
    except FileNotFoundError:
        st.error("This source file is no longer in the corpus folder. Rescan the corpus from the Corpus tab.")
        return
    except Exception as error:
        st.error(f"Unable to open this source file: {error}")
        return
    current = min(max(1, st.session_state[key]), total)
    st.markdown(f'<div class="atlas"><h2>{html.escape(source)}</h2><p class="reference">PAGE {current} OF {total}</p></div>', unsafe_allow_html=True)
    prev_col, next_col, download_col = st.columns([1, 1, 2])
    if prev_col.button("Previous page", disabled=current <= 1, key="viewer_prev"):
        st.session_state[key] = current - 1
        st.rerun(scope="fragment")
    if next_col.button("Next page", disabled=current >= total, key="viewer_next"):
        st.session_state[key] = current + 1
        st.rerun(scope="fragment")
    download_col.download_button("Download original PDF", rag.source_bytes(source), file_name=Path(source).name,
                                 mime="application/pdf", key="viewer_download")
    try:
        st.image(render_page(rag, rag.index_version, source, current, terms), width="stretch",
                 caption=f"{source}, page {current}. Highlighted terms locate the supporting passage.")
    except Exception as error:
        st.error(f"Unable to render the selected page: {error}")
    with st.expander("Page text (selectable, screen-reader friendly)"):
        text = page_text(rag, rag.index_version, source, current)
        st.text(text.strip() or "No extractable text on this page (it may be a scan).")


# ---------------------------------------------------------------- search tab

def submit_question() -> None:
    question = st.session_state.get("question", "").strip()
    if not question:
        st.session_state["form_message"] = "Enter a question to search the records."
        return
    st.session_state["form_message"] = ""
    st.session_state["pending_query"] = {"query": question, "scope": st.session_state.get("scope", ALL_INSTITUTIONS)}


def clear_result() -> None:
    for key in ("result", "pending_query", "question", "form_message"):
        st.session_state.pop(key, None)


def evidence_card(item, query: str, label: str, key_prefix: str) -> None:
    chunk = item.chunk
    badge = '<span class="badge">SYNTHETIC</span>' if chunk.synthetic else ""
    st.markdown(
        f'<article class="evidence-card"><div class="evidence-meta"><span>{html.escape(short(chunk.source))}{badge}</span>'
        f'<span>PAGE {chunk.page}</span></div><p>{highlight(chunk.text, query)}</p></article>', unsafe_allow_html=True)
    digest = hashlib.sha256(f"{chunk.id}|{key_prefix}".encode()).hexdigest()[:12]
    st.button(f"{label}: page {chunk.page} of {short(chunk.source, 36)}", key=f"open_{digest}",
              on_click=request_viewer, args=(chunk.source, chunk.page, tokens(query)))


def render_result(result) -> None:
    query = html.escape(result.query)
    st.markdown(f'<div class="atlas"><div class="article-rule"></div><p class="kicker">Submitted question</p><h1 class="question-headline">{query}</h1></div>', unsafe_allow_html=True)
    reading, rail = st.columns([7, 4], gap="large")
    with reading:
        st.markdown('<p class="kicker">Answer</p>', unsafe_allow_html=True)
        for note in result.notices:
            st.warning(note)
        if result.status == "processing_error":
            st.error(f"Search could not be completed: {result.error}. This is a processing failure, not an absence of evidence.")
        elif result.status == "answered" and result.citation:
            chunk = result.citation
            badge = '<span class="badge">SYNTHETIC DEMO RECORD</span>' if chunk.synthetic else ""
            st.markdown(
                f'<article class="answer-copy"><p>{html.escape(result.statement)}</p></article>'
                f'<div class="citation-line">{html.escape(chunk.institution)} · {html.escape(short(chunk.source, 60))} · page {chunk.page}{badge}</div>',
                unsafe_allow_html=True)
            st.markdown('<aside class="notice limitation atlas"><h2>Evidence limitation</h2><p><strong>This is an extract from the indexed records, not an accreditation decision.</strong> Read the cited page in context before relying on it.</p></aside>', unsafe_allow_html=True)
        else:
            reason = f" ({html.escape(result.reason)})" if result.reason else ""
            st.markdown(f'<aside class="notice unable atlas"><h2>Unable to verify</h2><p><strong>No supporting passage answers this question in the searchable records{reason}.</strong> Try a narrower question or another institution scope. Documents without searchable text are listed in the Corpus tab.</p></aside>', unsafe_allow_html=True)
    with rail:
        if result.answered and result.citation:
            st.markdown('<p class="kicker">Supporting passage</p>', unsafe_allow_html=True)
            support = next((c for c in result.candidates if c.chunk.id == result.citation.id), None)
            if support:
                evidence_card(support, result.query, "Open cited page", "support")
            others = [c for c in result.candidates if c.chunk.id != result.citation.id][:3]
            if others:
                st.markdown('<p class="kicker">Other retrieved passages (not used for the answer)</p>', unsafe_allow_html=True)
                for index, item in enumerate(others):
                    evidence_card(item, result.query, "Open page", f"other{index}")
        elif result.candidates:
            st.markdown('<p class="kicker">Closest passages (do not answer the question)</p>', unsafe_allow_html=True)
            for index, item in enumerate(result.candidates[:3]):
                evidence_card(item, result.query, "Open page", f"near{index}")


def search_tab(rag) -> None:
    st.markdown('<div class="atlas"><div class="section-rule"></div><p class="kicker">Ask the corpus</p><h1>Find evidence, not just documents.</h1><p class="lede">Ask about SSR, AQAR, IQAC minutes, NIRF, NBA, and the records supplied to this workspace.</p></div>', unsafe_allow_html=True)
    with st.form("evidence-question", clear_on_submit=False):
        st.text_input("Ask a question about the corpus", key="question", placeholder="For example: What attendance percentage is mandatory?")
        left, right = st.columns([1, 5])
        left.form_submit_button("Ask the corpus", type="primary", on_click=submit_question, disabled=not rag.chunks)
        if st.session_state.get("result") or st.session_state.get("form_message"):
            right.form_submit_button("Clear", on_click=clear_result)
    if st.session_state.get("form_message"):
        st.warning(st.session_state["form_message"])
    st.caption("Answers are extracted from the indexed records. Demonstration records are labelled SYNTHETIC. Choose an institution above to scope the search.")

    pending = st.session_state.pop("pending_query", None)
    if pending:
        scope = pending["scope"]
        with st.spinner("Preparing search models and matching the question to evidence…"):
            st.session_state["result"] = rag.answer(pending["query"], DEFAULT_METHOD, scope)
    result = st.session_state.get("result")
    if result:
        render_result(result)
    else:
        st.markdown('<div class="atlas"><section class="initial-state"><p class="kicker">Ready for review</p><h2>Every claim stays connected to a page.</h2><p>Submit a question to read a concise answer alongside the passage that supports it.</p></section></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- coverage tab

def coverage_tab(rag, scope: str) -> None:
    st.markdown('<div class="atlas"><div class="section-rule"></div><p class="kicker">Coverage review</p><h1>Evidence coverage</h1><p class="lede">Located passages for each accreditation topic. Locating a passage is not verification: every row starts as not reviewed.</p></div>', unsafe_allow_html=True)
    key = f"coverage::{rag.index_version}::{scope}"
    if st.button("Run coverage review", type="primary", key="run_coverage"):
        with st.spinner(f"Searching {len(ACCREDITATION_CRITERIA)} criteria…"):
            st.session_state[key] = rag.criterion_coverage(scope)
    rows = st.session_state.get(key)
    if rows is None:
        st.info("Run the coverage review to search the selected records for each criterion. It is not run automatically.")
        return
    gaps = [d for d in rag.documents if (scope == ALL_INSTITUTIONS or d.institution == scope) and d.status in ("no_text", "error")]
    if gaps:
        st.warning(f"{len(gaps)} document(s) in this scope have no searchable text and were not searched, so “No evidence located” may reflect an ingestion gap. See the Corpus tab.")
    frame = pd.DataFrame([{"Criterion": r["Criterion"], "Located": r["Evidence strength"], "Review": r["Review"],
                           "Institution": r["Institution"], "Finding": r["Finding"], "Citation": r["Citation"]} for r in rows])
    st.dataframe(frame, width="stretch", hide_index=True)
    cited = [r for r in rows if r["source"]]
    if cited:
        choice = st.selectbox("Open a cited page", [f"{r['Criterion']} — {short(r['source'], 40)} p.{r['page']}" for r in cited], key="cov_open")
        index = [f"{r['Criterion']} — {short(r['source'], 40)} p.{r['page']}" for r in cited].index(choice)
        row = cited[index]
        st.button("Open cited page", key="cov_open_btn", on_click=request_viewer, args=(row["source"], int(row["page"]), tokens(row["Finding"])[:6]))
    from exports import export_docx, export_pdf
    export_scope = scope
    left, right = st.columns(2)
    left.download_button("Download evidence pack (DOCX)", export_docx(rows, export_scope), "evidence_pack.docx",
                         "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    right.download_button("Download evidence pack (PDF)", export_pdf(rows, export_scope), "evidence_pack.pdf", "application/pdf")
    alerts = review_alerts(rag, rag.index_version, scope)
    with st.expander(f"Review alerts ({len(alerts)})"):
        st.caption("Values that differ within one institution, flagged for human review. Not findings of non-compliance.")
        for alert in alerts or []:
            st.markdown(f"**{alert['level']} · {alert['institution']}** — {alert['message']}")
        if not alerts:
            st.write("No alerts.")


# ---------------------------------------------------------------- evaluation tab

def evaluation_tab(rag) -> None:
    from evaluate import BENCHMARK, DEFAULT_SCOPE, METHODS, run_evaluation, summarise
    positives = sum(1 for case in BENCHMARK if case[3] == "Evidence expected")
    scope = DEFAULT_SCOPE if DEFAULT_SCOPE in rag.institutions else None
    st.markdown(f'<div class="atlas"><div class="section-rule"></div><p class="kicker">Labeled test suite · maintainer tool</p><h1>Evaluation</h1><p class="lede">{positives} evidence-required questions and {len(BENCHMARK) - positives} no-evidence controls, scored separately for retrieval, answer correctness and abstention. Scope: {html.escape(scope or "all institutions")}.</p></div>', unsafe_allow_html=True)
    if st.button("Run backend evaluation", type="primary"):
        with st.spinner("Running retrieval and answer checks (this can take a minute)…"):
            st.session_state["evaluation"] = run_evaluation(rag, scope)
    evaluation = st.session_state.get("evaluation")
    if evaluation is None:
        st.info("Run the evaluation to calculate current results. Nothing is prefilled.")
        return
    summary = summarise(evaluation)
    table = pd.DataFrame([{"Method": METHODS[name], "Recall@5": summary[name]["recall_at_5"], "MRR": summary[name]["mrr"],
                           "Answer correct": summary[name]["answer_correct"], "Correct abstentions": f"{summary[name]['correct_abstentions']}/{summary['negative_cases']}"}
                          for name in METHODS])
    st.caption(f"Run {evaluation.attrs.get('run_id', '(unsaved)')} · positive-case recall excludes the no-evidence controls.")
    st.dataframe(table, width="stretch", hide_index=True)
    st.dataframe(evaluation, width="stretch", hide_index=True)
    st.download_button("Download evaluation CSV", evaluation.to_csv(index=False), f"evaluation_{evaluation.attrs.get('run_id', 'run')}.csv", "text/csv")


# ---------------------------------------------------------------- corpus tab

def rescan() -> None:
    get_rag.clear()


def corpus_tab(rag) -> None:
    ok = sum(1 for d in rag.documents if d.status == "ok")
    st.markdown(f'<div class="atlas"><div class="section-rule"></div><p class="kicker">Supplied records</p><h1>Corpus</h1><p class="lede">{len(rag.documents)} documents registered, {ok} fully searchable, {len(rag.chunks)} indexed passages. Index version {rag.index_version or "—"}.</p></div>', unsafe_allow_html=True)
    left, right = st.columns([3, 1])
    filter_text = left.text_input("Filter documents", placeholder="Filter by filename or institution", key="corpus_filter")
    status_filter = right.selectbox("Status", ["All", "ok", "partial", "no_text", "error"])
    st.button("Rescan corpus folder", on_click=rescan, help="Re-read changed PDFs and rebuild the index if the documents changed.")
    visible = [d for d in rag.documents
               if (filter_text.lower() in d.source.lower() or filter_text.lower() in d.institution.lower())
               and (status_filter == "All" or d.status == status_filter)]
    labels = {"ok": "Searchable", "partial": "Partly searchable", "no_text": "No searchable text (scan)", "error": "Unreadable"}
    st.dataframe(pd.DataFrame([{"Document": d.source, "Institution": d.institution, "Status": labels.get(d.status, d.status),
                                "Pages": d.pages, "Pages without text": len(d.zero_text_pages), "Passages": d.chunks,
                                "Synthetic": "Yes" if d.synthetic else ""} for d in visible]),
                 width="stretch", hide_index=True)
    if visible:
        pick = st.selectbox("Open a document", [d.source for d in visible], key="corpus_open")
        st.button("Open first page", key="corpus_open_btn", on_click=request_viewer, args=(pick, 1, []))
    st.markdown('<aside class="notice atlas"><h2>Important notice</h2><p>Documents marked as having no searchable text are registered but cannot be searched until they are OCR-processed; absence of evidence from them is not evidence of absence.</p></aside>', unsafe_allow_html=True)


# ---------------------------------------------------------------- page

inject_theme()
st.markdown('<header class="masthead"><div class="wordmark">Evidence Atlas</div><p>Accreditation evidence workspace for reviewers and IQAC staff</p></header>', unsafe_allow_html=True)

try:
    with st.spinner("Loading the evidence index (cached after the first run)…"):
        rag = get_rag(corpus_stamp())
except Exception as error:
    st.error(f"The evidence index could not be loaded: {error}")
    st.button("Try again", on_click=get_rag.clear)
    st.stop()

if not rag.chunks:
    st.warning("No searchable PDF text was found in the docs folder. Add PDFs (text-based, or OCR-processed scans) and rescan from the Corpus tab.")

scope_options = [ALL_INSTITUTIONS] + rag.institutions
scope = st.selectbox("Institution scope", scope_options, key="scope", help="Searches, coverage and exports are limited to this institution's records.")

open_request = st.session_state.pop("open_viewer", None)
if open_request:
    st.session_state["viewer_page"] = open_request["page"]
    source_dialog(rag, open_request["source"], open_request["page"], open_request["terms"])

search, coverage, evaluation, corpus = st.tabs(["Search", "Coverage", "Evaluation", "Corpus"])
with search:
    search_tab(rag)
with coverage:
    coverage_tab(rag, scope)
with evaluation:
    evaluation_tab(rag)
with corpus:
    corpus_tab(rag)
