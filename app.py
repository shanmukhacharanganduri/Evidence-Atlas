# -*- coding: utf-8 -*-
"""Evidence Atlas — editorial accreditation evidence workspace."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from evaluate import BENCHMARK, run_evaluation
from rag_pipeline import ACCREDITATION_CRITERIA, NO_EVIDENCE_RESPONSE, build_rag, tokens

st.set_page_config(page_title="Evidence Atlas", layout="wide", initial_sidebar_state="collapsed")

THEME = Path(__file__).resolve().parent / "theme.css"
if THEME.exists():
    st.markdown(f"<style>{THEME.read_text('utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Indexing the evidence corpus…")
def get_rag():
    return build_rag()


@st.cache_data(show_spinner="Evaluating corpus coverage…")
def get_coverage(_rag):
    return _rag.criterion_coverage()


def _anchor(source: str, page: int) -> str:
    return f"evidence-{hashlib.sha256(f'{source}|{page}'.encode()).hexdigest()[:12]}"


def _short(value: str, length: int = 54) -> str:
    return value if len(value) <= length else value[: length - 1] + "…"


def _highlight(text: str, query: str) -> str:
    output = html.escape(text)
    terms = sorted({term for term in tokens(query) if len(term) > 3}, key=len, reverse=True)
    if terms:
        output = re.sub("(" + "|".join(map(re.escape, terms)) + ")", r"<mark>\1</mark>", output, flags=re.I)
    return output


def install_following_magnifier() -> None:
    """Attach a decorative, measured magnifier to Streamlit's native text input."""
    st.html(
        """
        <script>
        (() => {
          const root = window.parent.document;
          const input = [...root.querySelectorAll('input')].find((el) => el.getAttribute('aria-label') === 'Ask a question about the corpus');
          if (!input || root.getElementById('atlas-following-search')) return;
          const icon = root.createElement('span');
          icon.id = 'atlas-following-search'; icon.setAttribute('aria-hidden', 'true');
          icon.innerHTML = '&#128269;'; icon.style.cssText = 'position:fixed;z-index:1000;pointer-events:none;font-size:15px;line-height:1;color:#245847;transition:left 125ms linear,top 125ms linear;';
          root.body.appendChild(icon);
          const canvas = root.createElement('canvas'); const context = canvas.getContext('2d');
          const update = () => {
            const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            icon.style.transition = reduced ? 'none' : 'left 125ms linear,top 125ms linear';
            const style = window.getComputedStyle(input); context.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
            const beforeCaret = input.value.slice(0, input.selectionStart ?? input.value.length);
            const width = context.measureText(beforeCaret).width;
            const rect = input.getBoundingClientRect(); const pad = parseFloat(style.paddingLeft) || 16;
            const minimum = rect.left + 34; const maximum = rect.right - 58;
            const left = Math.max(minimum, Math.min(rect.left + pad + width - input.scrollLeft + 8, maximum));
            icon.style.left = `${left}px`; icon.style.top = `${rect.top + (rect.height - 16) / 2}px`;
          };
          ['input','keyup','click','focus','scroll'].forEach((event) => input.addEventListener(event, update));
          new ResizeObserver(update).observe(input); root.fonts?.ready.then(update); update();
        })();
        </script>
        """,
    )


def evidence_card(result, query: str) -> None:
    source, page = result.chunk.source, result.chunk.page
    st.markdown(
        f'<article class="evidence-card" id="{_anchor(source, page)}">'
        f'<div class="evidence-meta"><span>{html.escape(_short(source))}</span><span>PAGE {page}</span></div>'
        f'<p>{_highlight(result.chunk.text, query)}</p></article>',
        unsafe_allow_html=True,
    )
    key = hashlib.sha256(f"{source}|{page}|{result.chunk.text}".encode()).hexdigest()[:16]
    if st.button("Open source page", key=f"source_{key}"):
        st.session_state["pdf_viewer"] = {"source": source, "page": page, "terms": tokens(query)}
        st.rerun()


def show_viewer(rag) -> None:
    viewer = st.session_state.get("pdf_viewer")
    if not viewer:
        return
    st.markdown('<section class="document-viewer"><div class="kicker">Selected evidence</div>'
                f'<h2>{html.escape(viewer["source"])}</h2><p class="reference">PAGE {viewer["page"]} · original rendering</p></section>', unsafe_allow_html=True)
    left, right = st.columns([5, 1])
    with right:
        if st.button("Close document"):
            del st.session_state["pdf_viewer"]
            st.rerun()
    with left:
        try:
            image = rag.render_cited_page(viewer["source"], viewer["page"], viewer.get("terms", []))
            st.image(image, width="stretch", caption="Highlighted terms locate the supporting passage; the scan itself is not recolored.")
        except FileNotFoundError:
            st.error("Unable to open this source file. It may have been moved from the corpus.")
        except Exception as error:
            st.error(f"Unable to render the selected page: {error}")


rag = get_rag()
sources = sorted({chunk.source for chunk in rag.chunks})

st.markdown('<header class="masthead"><div><div class="wordmark">Evidence Atlas</div><p>Accreditation evidence workspace for reviewers and IQAC staff</p></div></header>', unsafe_allow_html=True)
nav, appearance = st.columns([5, 2])
with nav:
    st.caption("SEARCH · COVERAGE · EVALUATION · CORPUS")
with appearance:
    saved_theme = st.query_params.get("theme", "System").title()
    theme_options = ["System", "Light", "Dark"]
    theme_choice = st.selectbox("Appearance", theme_options, index=theme_options.index(saved_theme) if saved_theme in theme_options else 0, key="appearance", label_visibility="collapsed")
if theme_choice == "System":
    if "theme" in st.query_params:
        del st.query_params["theme"]
else:
    st.query_params["theme"] = theme_choice.lower()
st.markdown(f'<div class="theme-flag" data-theme-choice="{theme_choice.lower()}"></div>', unsafe_allow_html=True)

search_tab, coverage_tab, evaluation_tab, corpus_tab = st.tabs(["Search", "Coverage", "Evaluation", "Corpus"])

with search_tab:
    st.markdown('<div class="section-rule"></div><p class="kicker">Ask the corpus</p><h1>Find evidence, not just documents.</h1><p class="lede">Ask about SSR, AQAR, IQAC minutes, NIRF, NBA, and the records supplied to this workspace.</p>', unsafe_allow_html=True)
    with st.form("evidence-question", clear_on_submit=False):
        question = st.text_input("Ask a question about the corpus", key="question", placeholder="For example: What evidence supports faculty development activity?")
        ask = st.form_submit_button("Ask the corpus", type="primary")
    install_following_magnifier()
    st.caption("Demonstration corpus. Answers are extractive and only include evidence located in supplied records.")

    if ask and question.strip():
        st.session_state["submitted_question"] = question.strip()
    submitted = st.session_state.get("submitted_question", "")

    if submitted:
        try:
            with st.spinner("Matching the question to supporting evidence…"):
                answer, evidence = rag.answer(submitted, "Hybrid + Reranker")
        except Exception as err:
            st.error(f"Retrieval failed: {err}")
            answer, evidence = NO_EVIDENCE_RESPONSE, []
        st.markdown('<div class="article-rule"></div><p class="kicker">Submitted question</p>'
                    f'<h1 class="question-headline">{html.escape(submitted)}</h1>', unsafe_allow_html=True)
        reading, rail = st.columns([7, 4], gap="large")
        with reading:
            st.markdown('<p class="kicker">Answer</p>', unsafe_allow_html=True)
            if answer == NO_EVIDENCE_RESPONSE:
                st.markdown('<aside class="notice unable"><h2>Unable to verify</h2><p><strong>No supporting passage was found in the supplied records.</strong> Try a narrower question, add the governing document, or inspect the corpus.</p></aside>', unsafe_allow_html=True)
            else:
                citation = re.search(r"\[Source: (.*?), Page: (\d+)\]$", answer)
                fact = re.sub(r"\s*\[Source: .*?, Page: \d+\]$", "", answer)
                citation_text = f"{citation.group(1)} · page {citation.group(2)}" if citation else "Citation unavailable"
                st.markdown(f'<article class="answer-copy"><p>{html.escape(fact)}</p></article><div class="citation-line">EVIDENCE LINKED · {html.escape(citation_text)}</div>', unsafe_allow_html=True)
                st.markdown('<aside class="notice limitation"><h2>Evidence limitation</h2><p><strong>This response is an extract from the indexed records, not an accreditation decision.</strong> Review the cited page in context before relying on it.</p></aside>', unsafe_allow_html=True)
        with rail:
            st.markdown('<p class="kicker">Supporting evidence</p>', unsafe_allow_html=True)
            if evidence:
                for item in evidence[:3]:
                    evidence_card(item, submitted)
            else:
                st.markdown('<aside class="notice"><h2>Important notice</h2><p>No supporting evidence is available for this question.</p></aside>', unsafe_allow_html=True)
        show_viewer(rag)
    else:
        st.markdown('<section class="initial-state"><p class="kicker">Ready for review</p><h2>Every claim stays connected to a page.</h2><p>Submit a question to read a concise answer alongside the passages that support it.</p></section>', unsafe_allow_html=True)

with coverage_tab:
    st.markdown('<div class="section-rule"></div><p class="kicker">Coverage review</p><h1>Evidence coverage</h1><p class="lede">A document-led view of accreditation topics identified in the supplied corpus.</p>', unsafe_allow_html=True)
    coverage = get_coverage(rag)
    frame = pd.DataFrame([{ "Criterion": row["Criterion"], "Evidence": row["Evidence strength"], "Citation": row["Citation"] } for row in coverage])
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.markdown('<aside class="notice"><h2>Important notice</h2><p><strong>Coverage indicates located passages, not a compliance finding.</strong> Authority, currency, and applicability require reviewer judgement.</p></aside>', unsafe_allow_html=True)

with evaluation_tab:
    st.markdown(f'<div class="section-rule"></div><p class="kicker">Labeled test suite</p><h1>Evaluation</h1><p class="lede">{len(BENCHMARK)} questions: evidence-required cases and explicit no-evidence controls.</p>', unsafe_allow_html=True)
    if st.button("Run backend evaluation", type="primary"):
        with st.spinner("Running retrieval and citation validation…"):
            st.session_state["evaluation"] = run_evaluation(rag)
    evaluation = st.session_state.get("evaluation")
    if evaluation is None:
        st.info("Run the evaluation to calculate current results. No results are invented or prefilled.")
    else:
        st.dataframe(evaluation, use_container_width=True, hide_index=True)
        st.download_button("Download evaluation CSV", evaluation.to_csv(index=False), "evaluation_results.csv", "text/csv")

with corpus_tab:
    st.markdown(f'<div class="section-rule"></div><p class="kicker">Supplied records</p><h1>Corpus</h1><p class="lede">{len(sources)} documents and {len(rag.chunks)} indexed passages.</p>', unsafe_allow_html=True)
    filter_text = st.text_input("Filter documents", placeholder="Filter by filename")
    visible_sources = [source for source in sources if filter_text.lower() in source.lower()]
    st.dataframe(pd.DataFrame({"Document": visible_sources}), use_container_width=True, hide_index=True)
    st.markdown('<aside class="notice"><h2>Important notice</h2><p>Files displayed here are demonstration and supplied records. Source scans remain visually faithful when opened.</p></aside>', unsafe_allow_html=True)