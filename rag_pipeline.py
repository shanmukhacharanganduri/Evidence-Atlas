"""Grounded, citation-first retrieval for the Accreditation Evidence Assistant.

Heavy ML dependencies (chromadb, sentence-transformers) are imported lazily so the app shell and
the unit tests do not pay for them. Corpus extraction is cached by content fingerprint.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

import exports
from answering import (  # noqa: F401  (re-exported for callers)
    NO_EVIDENCE_RESPONSE, RERANK_THRESHOLD, AnswerResult, candidate_statements, select_answer, tokens,
)
from corpus import Chunk, DocumentRecord, load_corpus

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "chroma_db"
CACHE_DIR = ROOT / "index_cache"
REGISTRY_PATH = ROOT / "document_registry.json"
COLLECTION = "governance_evidence"
MODEL_NAME = "all-MiniLM-L6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"
RETRIEVAL_METHODS = ("Dense (Chroma)", "BM25", "BM25 + Reranker", "Hybrid + Reranker")
DEFAULT_METHOD = "Hybrid + Reranker"
ALL_INSTITUTIONS = "All institutions"
EMBED_BATCH = 128

# (criterion, query, topic pattern, value pattern). A criterion counts as located only when one
# statement matches BOTH the topic and the value pattern, so a loosely related sentence cannot qualify.
ACCREDITATION_CRITERIA = [
    ("Student-Faculty Ratio", "What student-to-faculty ratio applies to academic departments?", r"student[- ]?(?:to[- ])?faculty|faculty ratio", r"\b\d+\s*:\s*\d+\b"),
    ("Attendance Policy", "What attendance threshold is mandatory for examinations?", r"attendance", r"\b\d{2,3}(?:\.\d+)?\s*%"),
    ("Minimum CGPA for Graduation", "What minimum CGPA is required for graduation?", r"\bcgpa\b|grade point average", r"\b\d+\.\d+\b"),
    ("Examination Malpractice Policy", "Which committee reviews examination malpractice?", r"malpractice", r"committee"),
    ("Medical Condonation", "What medical proof is needed for attendance condonation?", r"condon", r"medical"),
    ("Senate/BoG Minutes", "Who chaired the most recent Senate or BoG meeting?", r"\bchair(?:ed|man|person)?\b|presided", r"senate|board of (?:governors|management)|governing (?:body|council)|academic council"),
    ("IQAC Functioning", "What quality initiatives did IQAC recommend or implement?", r"\biqac\b|internal quality assurance", r"recommend|implement|initiative|resolved|decided"),
    ("Anti-Ragging Committee", "Who is on the anti-ragging committee and who chairs it?", r"anti[- ]ragging", r"committee|chair"),
    ("Grievance Redressal", "What is the grievance resolution timeline?", r"grievance", r"\b\d+[- ]days?\b|\b(?:seven|three|ten|fifteen|thirty) (?:working )?days"),
    ("Research Seed Grants", "What research seed grants were approved for faculty research?", r"seed (?:grant|fund)", r"₹|\brs\.?\s*\d|\blakh|\bcrore"),
    ("Research Promotion Policy", "What incentives exist for faculty publishing in top-tier journals?", r"research promotion|incentive", r"journal|publication"),
    ("Ph.D / Doctoral Programme", "Does the institution have a doctoral or Ph.D programme?", r"\bph\.?\s?d\b|doctoral", r"programme|program|admitted|enrol|awarded"),
    ("Campus Safety Protocol", "What is the campus safety protocol or duty officer contact?", r"campus safety|duty officer|helpline", r"\+?\d[\d\s\-]{6,}|contact"),
    ("Library Resources", "What is the library collection size or budget?", r"library", r"\d[\d,]*\s*(?:books|volumes|titles|journals|e-?resources)|budget|₹"),
    ("Scholarship & Financial Aid", "What scholarships or financial aid schemes are available to students?", r"scholarship|financial aid|fee waiver", r"\d|₹|%|eligib"),
    ("Placement & Industry Connect", "What is the campus placement record or industry tie-up policy?", r"placement", r"\d+"),
]


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


def _all_zero(statement: str) -> bool:
    """True when a statement's numeric values are all zero (e.g. an enrolment form with 0 students)."""
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", statement)
    return bool(numbers) and all(float(n.replace(",", "")) == 0 for n in numbers)


class AccreditationRAG:
    def __init__(self, docs_dir: Path = DOCS_DIR, chroma_dir: Path = CHROMA_DIR, cache_dir: Path = CACHE_DIR,
                 registry_path: Path | None = REGISTRY_PATH) -> None:
        self.chroma_dir, self.cache_dir = Path(chroma_dir), Path(cache_dir)
        self.chunks, self.documents, self.fingerprint = load_corpus(Path(docs_dir), self.cache_dir, registry_path)
        self.index_version = self.fingerprint[:12]
        self._by_id = {chunk.id: chunk for chunk in self.chunks}
        self.bm25 = BM25Okapi([tokens(chunk.text) for chunk in self.chunks]) if self.chunks else None
        self._institution_index: dict[str, np.ndarray] = {}
        for index, chunk in enumerate(self.chunks):
            self._institution_index.setdefault(chunk.institution, []).append(index)  # type: ignore[arg-type]
        self._institution_index = {k: np.array(v) for k, v in self._institution_index.items()}
        self._lock = threading.Lock()
        self._model: SentenceTransformer | None = None
        self._collection = None
        self._dense_error = ""
        self._reranker: CrossEncoder | None = None
        self._reranker_error = ""
        self._nli: CrossEncoder | None = None

    # ------------------------------------------------------------------ status

    @property
    def institutions(self) -> list[str]:
        return sorted({chunk.institution for chunk in self.chunks})

    @property
    def unsearchable(self) -> list[DocumentRecord]:
        return [d for d in self.documents if d.status in ("no_text", "error") or d.zero_text_pages or d.error_pages]

    def notices(self) -> list[str]:
        notes = []
        if self._dense_error:
            notes.append(f"Semantic (dense) search is unavailable, so lexical search was used: {self._dense_error}")
        if self._reranker_error:
            notes.append(f"The reranking model is unavailable, so passages are ordered by first-stage score: {self._reranker_error}")
        return notes

    def reset_models(self) -> None:
        """Allow a retry after a model-load failure (failures are otherwise cached, not retried)."""
        self._dense_error = self._reranker_error = ""

    # ------------------------------------------------------------------ models

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(MODEL_NAME, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(MODEL_NAME, local_files_only=False)
        return self._model

    def _get_reranker(self) -> "CrossEncoder | None":
        if self._reranker is None and not self._reranker_error:
            with self._lock:
                if self._reranker is None and not self._reranker_error:
                    try:
                        self._reranker = _load_cross_encoder(RERANKER_MODEL)
                    except Exception as error:
                        self._reranker_error = f"{type(error).__name__}: {error}"[:200]
        return self._reranker

    def score_statements(self, query: str, statements: Sequence[str]) -> Sequence[float] | None:
        reranker = self._get_reranker()
        if reranker is None or not statements:
            return None
        try:
            return [float(s) for s in reranker.predict([(query, s) for s in statements], show_progress_bar=False)]
        except Exception as error:
            self._reranker_error = f"{type(error).__name__}: {error}"[:200]
            return None

    # ------------------------------------------------------------------ dense index

    def _ensure_dense(self):
        """Build or reuse the dense index for the current corpus fingerprint.

        A new, fingerprint-named collection is built completely before older ones are removed,
        so an interrupted build leaves the previous good index in place.
        """
        if self._collection is not None:
            return self._collection
        if self._dense_error or not self.chunks:
            return None
        with self._lock:
            if self._collection is not None:
                return self._collection
            try:
                import chromadb

                client = chromadb.PersistentClient(path=str(self.chroma_dir))
                name = f"{COLLECTION}_{self.index_version}"
                existing = {getattr(c, "name", c) for c in client.list_collections()}
                if name in existing:
                    collection = client.get_collection(name)
                    if collection.count() == len(self.chunks):
                        self._collection = collection
                if self._collection is None:
                    if name in existing:
                        client.delete_collection(name)
                    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
                    model = self._get_model()
                    for start in range(0, len(self.chunks), EMBED_BATCH):
                        batch = self.chunks[start:start + EMBED_BATCH]
                        collection.add(
                            ids=[c.id for c in batch], documents=[c.text for c in batch],
                            metadatas=[{"source": c.source, "page": c.page, "institution": c.institution} for c in batch],
                            embeddings=model.encode([c.text for c in batch], normalize_embeddings=True, show_progress_bar=False).tolist(),
                        )
                    self._collection = collection
                for stale in existing - {name}:
                    if str(stale).startswith(COLLECTION):
                        client.delete_collection(str(stale))
            except Exception as error:
                self._dense_error = f"{type(error).__name__}: {error}"[:200]
                self._collection = None
        return self._collection

    def prepare(self) -> None:
        """Warm the dense index and reranker (call from a controlled setup step)."""
        self._ensure_dense()
        self._get_reranker()

    # ------------------------------------------------------------------ retrieval

    def dense_search(self, query: str, top_k: int = 5, institution: str | None = None) -> list[RetrievalResult]:
        collection = self._ensure_dense()
        if collection is None:
            return []
        kwargs = {"where": {"institution": institution}} if institution else {}
        found = collection.query(
            query_embeddings=self._get_model().encode([query], normalize_embeddings=True).tolist(),
            n_results=min(top_k, len(self.chunks)), include=["distances"], **kwargs,
        )
        ids, distances = (found.get("ids") or [[]])[0], (found.get("distances") or [[]])[0]
        return [RetrievalResult(self._by_id[i], max(0.0, 1 - float(d))) for i, d in zip(ids, distances) if i in self._by_id]

    def lexical_search(self, query: str, top_k: int = 5, institution: str | None = None) -> list[RetrievalResult]:
        """Pure BM25: no semantic model involved."""
        if self.bm25 is None:
            return []
        scores = np.array(self.bm25.get_scores(tokens(query)), dtype=float)
        if institution:
            mask = np.zeros(len(scores), dtype=bool)
            mask[self._institution_index.get(institution, np.array([], dtype=int))] = True
            scores = np.where(mask, scores, 0.0)
        order = np.argsort(-scores)[:top_k]
        return [RetrievalResult(self.chunks[int(i)], float(scores[int(i)])) for i in order if scores[int(i)] > 0]

    def _rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        unique: dict[str, RetrievalResult] = {}
        for item in candidates:
            unique.setdefault(item.chunk.id, item)
        items = list(unique.values())
        if not items:
            return []
        scores = self.score_statements(query, [item.chunk.text for item in items])
        if scores is None:  # disclosed through notices(); keep first-stage order
            return items[:top_k]
        ranked = sorted(zip(items, scores), key=lambda pair: pair[1], reverse=True)
        return [RetrievalResult(item.chunk, float(score)) for item, score in ranked[:top_k]]

    def bm25_rerank_search(self, query: str, top_k: int = 5, institution: str | None = None) -> list[RetrievalResult]:
        candidates = self.lexical_search(query, max(20, top_k * 4), institution)
        if not candidates:  # a paraphrase may share no vocabulary with the record
            candidates = self.dense_search(query, max(20, top_k * 4), institution)
        return self._rerank(query, candidates, top_k)

    def hybrid_search(self, query: str, top_k: int = 5, institution: str | None = None) -> list[RetrievalResult]:
        pool = max(20, top_k * 4)
        return self._rerank(query, self.dense_search(query, pool, institution) + self.lexical_search(query, pool, institution), top_k)

    def search(self, query: str, method: str, top_k: int = 5, institution: str | None = None) -> list[RetrievalResult]:
        institution = None if institution in (None, "", ALL_INSTITUTIONS) else institution
        if method == "Dense (Chroma)":
            results = self.dense_search(query, top_k, institution)
            return results or self.lexical_search(query, top_k, institution)
        if method == "BM25":
            return self.lexical_search(query, top_k, institution)
        if method == "BM25 + Reranker":
            return self.bm25_rerank_search(query, top_k, institution)
        if method == "Hybrid + Reranker":
            return self.hybrid_search(query, top_k, institution)
        raise ValueError(f"Unknown retrieval method: {method}")

    # ------------------------------------------------------------------ answers

    def _idf(self, term: str) -> float:
        """Inverse document frequency of a query term in this corpus (rare terms weigh more)."""
        if self.bm25 is None:
            return 1.0
        return max(0.1, float(self.bm25.idf.get(term, max(self.bm25.idf.values(), default=1.0))))

    def answer_from_results(self, query: str, results: list[RetrievalResult], method: str = "",
                            institution: str | None = None) -> AnswerResult:
        started = time.perf_counter()
        base = dict(query=query, institution=institution, method=method, index_version=self.index_version)
        selection, reason = select_answer(query, results, scorer=self.score_statements, term_weight=self._idf)
        notices = self.notices()
        seconds = time.perf_counter() - started
        if selection is None:
            return AnswerResult(status="insufficient_evidence", candidates=list(results), notices=notices,
                                reason=reason, seconds=seconds, **base)
        scope_institutions = sorted({r.chunk.institution for r in results})
        if institution is None and len(scope_institutions) > 1:
            notices.append("The retrieved passages come from several institutions (" + ", ".join(scope_institutions)
                           + "). Choose an institution to scope the answer.")
        return AnswerResult(status="answered", statement=selection.statement, citation=selection.chunk,
                            support_rank=selection.rank, score=selection.score, candidates=list(results),
                            notices=notices, seconds=seconds, **base)

    def answer(self, query: str, method: str = DEFAULT_METHOD, institution: str | None = None) -> AnswerResult:
        started = time.perf_counter()
        institution = None if institution in (None, "", ALL_INSTITUTIONS) else institution
        try:
            results = self.search(query, method, top_k=5, institution=institution)
            result = self.answer_from_results(query, results, method, institution)
        except Exception as error:  # typed error state, distinct from "no evidence"
            return AnswerResult(query=query, status="processing_error", error=f"{type(error).__name__}: {error}",
                                method=method, institution=institution, index_version=self.index_version,
                                notices=self.notices())
        result.seconds = time.perf_counter() - started
        return result

    # ------------------------------------------------------------------ coverage

    def criterion_coverage(self, institution: str | None = None) -> list[dict]:
        """Locate a statement for each criterion whose text matches BOTH its topic and value patterns.

        "Located" means the best matching statement is relevant to the criterion question; "Partial"
        means a statement matches the patterns but is weakly relevant. Neither is a compliance
        finding and every row starts as "Not reviewed".
        """
        institution = None if institution in (None, "", ALL_INSTITUTIONS) else institution
        scope_docs = [d for d in self.documents if institution is None or d.institution == institution]
        gaps = [d for d in scope_docs if d.status in ("no_text", "error")]
        rows = []
        for criterion, query, topic, value in ACCREDITATION_CRITERIA:
            topic_re, value_re = re.compile(topic, re.I), re.compile(value, re.I)
            results = self.hybrid_search(query, top_k=10, institution=institution)
            matches: list[tuple[str, Chunk]] = []
            for result in results:
                for statement in candidate_statements(result.chunk.text):
                    if topic_re.search(statement) and value_re.search(statement) and not _all_zero(statement):
                        matches.append((statement, result.chunk))
            best, score = None, None
            if matches:
                scores = self.score_statements(query, [m[0] for m in matches])
                if scores is None:  # no reranker: cannot judge relevance, so never report "Located"
                    best, score = matches[0], None
                else:
                    index = int(np.argmax(scores))
                    best, score = matches[index], float(scores[index])
            row = {"Criterion": criterion, "Review": "Not reviewed", "Institution": "", "synthetic": False}
            if best is None:
                note = "No statement matching this criterion was located in the searchable records."
                if gaps:
                    note += f" {len(gaps)} document(s) in this scope have no searchable text and were not searched."
                row.update({"Evidence strength": "No evidence located", "Coverage score": 0, "Color": "none",
                            "Finding": note, "Citation": "—", "source": "", "page": ""})
            else:
                statement, chunk = best
                located = score is not None and score >= RERANK_THRESHOLD
                row.update({
                    "Evidence strength": "Located" if located else "Partial match",
                    "Coverage score": 3 if located else 2, "Color": "strong" if located else "thin",
                    "Finding": statement, "Citation": f"[Source: {chunk.source}, Page: {chunk.page}]",
                    "source": chunk.source, "page": str(chunk.page),
                    "Institution": chunk.institution, "synthetic": chunk.synthetic,
                })
            rows.append(row)
        return rows

    # ------------------------------------------------------------------ review alerts

    def governance_gaps(self, institution: str | None = None) -> list[dict[str, str]]:
        """Flag differing values for the same norm WITHIN one institution, for human review."""
        patterns = [
            ("Attendance threshold", r"attendance", r"\b\d{2,3}(?:\.\d+)?\s?%"),
            ("Student-faculty ratio", r"student[- ]to[- ]faculty|faculty ratio", r"\b\d+\s*:\s*\d+\b"),
            ("Grievance timeline", r"grievance|resolution timeline", r"\b\d+[- ]days?\b"),
        ]
        institution = None if institution in (None, "", ALL_INSTITUTIONS) else institution
        by_institution: dict[str, list[Chunk]] = {}
        for chunk in self.chunks:
            if institution is None or chunk.institution == institution:
                by_institution.setdefault(chunk.institution, []).append(chunk)
        alerts: list[dict[str, str]] = []
        for owner, chunks in sorted(by_institution.items()):
            for label, topic, value in patterns:
                seen: dict[str, tuple[str, int]] = {}
                for chunk in chunks:
                    if re.search(topic, chunk.text, re.I):
                        for found in re.findall(value, chunk.text, re.I):
                            seen.setdefault(found.replace(" ", "").lower(), (chunk.source, chunk.page))
                if len(seen) > 1:
                    cited = "; ".join(f"{v} [Source: {s}, Page: {p}]" for v, (s, p) in seen.items())
                    alerts.append({"level": "Potential variation", "institution": owner,
                                   "message": f"{label} appears with several values within {owner}: {cited}. "
                                              "These may be different cases (e.g. bands) or a real conflict; check authority and effective date."})
            for chunk in chunks:
                if re.search(r"\bresolution\s*:|\bapproved\b", chunk.text, re.I) and re.search(r"ratio|attendance|grant", chunk.text, re.I):
                    alerts.append({"level": "Resolution review", "institution": owner,
                                   "message": f"A potentially controlling resolution was found; compare it with older regulations before relying on it. [Source: {chunk.source}, Page: {chunk.page}]"})
                    break
        return alerts

    # ------------------------------------------------------------------ exports

    def export_evidence_pack_docx(self, coverage: list[dict], scope: str, criterion: str | None = None) -> bytes:
        return exports.export_docx(coverage, scope, criterion)

    def export_evidence_pack_pdf(self, coverage: list[dict], scope: str, criterion: str | None = None) -> bytes:
        return exports.export_pdf(coverage, scope, criterion)

    # ------------------------------------------------------------------ NLI (optional, not used by the UI)

    def faithfulness_check(self, answer: str, evidence: list[RetrievalResult]) -> list[dict]:
        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=\[Source:|[A-Z])", answer) if s.strip()]
        claims = [c for c in (re.sub(r"\s*\[Source: .*?, Page: \d+\]", "", c).strip() for c in claims) if c]
        if not evidence:
            return [{"claim": c, "supported": False, "score": 0.0, "source": "", "page": 0, "method": "NLI entailment"} for c in claims]
        if self._nli is None:
            self._nli = _load_cross_encoder(NLI_MODEL_NAME)
        id2label = getattr(getattr(getattr(self._nli, "model", None), "config", None), "id2label", None)
        if id2label is None:
            raise RuntimeError("NLI model config missing id2label mapping.")
        entail = next((int(k) for k, v in id2label.items() if "entail" in str(v).lower()), 2)
        output = []
        for claim in claims:
            probs = np.atleast_2d(self._nli.predict([(e.chunk.text, claim) for e in evidence], apply_softmax=True))
            best = int(np.argmax([float(r[entail]) for r in probs]))
            score = float(probs[best][entail])
            output.append({"claim": claim, "supported": score >= 0.5, "score": score,
                           "source": evidence[best].chunk.source, "page": evidence[best].chunk.page, "method": "NLI entailment"})
        return output

    # ------------------------------------------------------------------ source pages

    def _source_path(self, source: str) -> Path:
        path = (DOCS_DIR / source).resolve()
        if not path.is_relative_to(DOCS_DIR.resolve()) or not path.is_file():
            raise FileNotFoundError(source)
        return path

    def page_count(self, source: str) -> int:
        import pymupdf
        with pymupdf.open(self._source_path(source)) as document:
            return document.page_count

    def page_text(self, source: str, page: int) -> str:
        import pymupdf
        with pymupdf.open(self._source_path(source)) as document:
            return document[page - 1].get_text()

    def source_bytes(self, source: str) -> bytes:
        return self._source_path(source).read_bytes()

    def render_cited_page(self, source: str, page: int, highlight_terms: list[str]) -> bytes:
        """Render a cited page, highlighting terms. Safe to cache by (source, page, terms, index_version)."""
        import pymupdf as fitz
        document = fitz.open(self._source_path(source))
        try:
            pdf_page = document[page - 1]
            for term in sorted({t for t in highlight_terms if len(t) >= 4}):
                for rectangle in pdf_page.search_for(term, quads=False)[:8]:
                    annotation = pdf_page.add_highlight_annot(rectangle)
                    annotation.set_colors(stroke=(1, 0.79, 0.15))
                    annotation.update()
            return pdf_page.get_pixmap(matrix=fitz.Matrix(1.65, 1.65), alpha=False, annots=True).tobytes("png")
        finally:
            document.close()


def _load_cross_encoder(model_name: str) -> "CrossEncoder":
    from sentence_transformers import CrossEncoder
    try:
        return CrossEncoder(model_name, local_files_only=True)
    except Exception:
        return CrossEncoder(model_name, local_files_only=False)


def build_rag() -> AccreditationRAG:
    return AccreditationRAG()



def corpus_stamp() -> str:
    """Cheap change detector (file names, sizes, mtimes) used to key the cached backend."""
    parts = []
    for path in sorted(DOCS_DIR.rglob("*.pdf")) if DOCS_DIR.exists() else []:
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    if REGISTRY_PATH.exists():
        parts.append(f"registry:{REGISTRY_PATH.stat().st_mtime_ns}")
    return str(hash(tuple(parts)))
