"""Grounded, citation-first RAG for the Accreditation Evidence Assistant."""
from __future__ import annotations

import io
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import chromadb
import pymupdf as fitz
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "governance_evidence"
MODEL_NAME = "all-MiniLM-L6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"
STOPWORDS = {"a", "an", "the", "is", "are", "what", "which", "who", "when", "where", "how", "for", "to", "of", "in", "on", "and", "with", "does", "do", "was", "were", "it", "that", "this"}
RETRIEVAL_METHODS = ("Dense (Chroma)", "BM25", "Hybrid + Reranker")
NO_EVIDENCE_RESPONSE = "Evidence not found in provided governance records."

# ---------------------------------------------------------------------------
# Expanded NAAC/NBA accreditation criteria catalogue (Pick #1)
# ---------------------------------------------------------------------------
ACCREDITATION_CRITERIA = [
    ("Student-Faculty Ratio",          "What student-to-faculty ratio applies to academic departments?",          r"\b\d+\s*:\s*\d+\b"),
    ("Attendance Policy",              "What attendance threshold is mandatory for examinations?",                 r"\b(?:\d{2,3})%"),
    ("Minimum CGPA for Graduation",    "What minimum CGPA is required for graduation?",                           r"\bCGPA\b.*\d+\.\d+|\d+\.\d+.*\bCGPA\b"),
    ("Examination Malpractice Policy", "Which committee reviews examination malpractice?",                        r"malpractice|disciplinary committee"),
    ("Medical Condonation",            "What medical proof is needed for attendance condonation?",                r"medical|condon"),
    ("Senate/BoG Minutes",             "Who chaired the most recent Senate or BoG meeting?",                     r"chair(?:man|person|ed)|senate|governing body"),
    ("IQAC Functioning",               "What quality initiatives did IQAC recommend or implement?",              r"IQAC|quality assurance|best practices"),
    ("Anti-Ragging Committee",         "Who is on the anti-ragging committee and who chairs it?",                r"anti[- ]ragging.*committee|committee.*anti[- ]ragging"),
    ("Grievance Redressal",            "What is the grievance resolution timeline?",                             r"\b\d+[- ]day|\b\d+\s+days|grievance"),
    ("Research Seed Grants",           "What research seed grants were approved for faculty research?",           r"(?:\u20b9|rs\.|grant|seed)"),
    ("Research Promotion Policy",      "What incentives exist for faculty publishing in top-tier journals?",      r"incentive|publication|journal|research promotion"),
    ("Ph.D / Doctoral Programme",      "Does the institution have a doctoral or Ph.D programme?",                r"\bph\.?d\b|doctoral|research programme"),
    ("Campus Safety Protocol",         "What is the campus safety protocol or duty officer contact?",            r"duty officer|contact|campus safety|helpline"),
    ("Library Resources",              "What is the library collection size or budget?",                         r"library|journals|books|database"),
    ("Scholarship & Financial Aid",    "What scholarships or financial aid schemes are available to students?",  r"scholarship|financial aid|stipend|fee waiver"),
    ("Placement & Industry Connect",   "What is the campus placement record or industry tie-up policy?",         r"placement|industry|recruiter|internship"),
]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


def tokens(text: str) -> list[str]:
    """Return consistent lexical terms for BM25 and answer matching."""
    return [word for word in re.findall(r"[\w%]+", text.lower(), flags=re.UNICODE) if word not in STOPWORDS]


def split_page(text: str, source: str, page: int, size: int = 180, overlap: int = 40) -> list[Chunk]:
    """Create overlapping chunks while preserving source and one-indexed page metadata."""
    words = text.split()
    output: list[Chunk] = []
    for start in range(0, len(words), size - overlap):
        passage = " ".join(words[start : start + size]).strip()
        if not passage:
            break
        output.append(Chunk(f"{source}::p{page}::c{len(output)}", passage, source, page))
        if start + size >= len(words):
            break
    return output


def ingest_all_pdfs() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(DOCS_DIR.rglob("*.pdf")):
        try:
            reader = PdfReader(str(path))
            source = path.relative_to(DOCS_DIR).as_posix()
            for page_number, pdf_page in enumerate(reader.pages, start=1):
                chunks.extend(split_page(pdf_page.extract_text() or "", source, page_number))
        except Exception as error:
            print(f"Skipping unreadable PDF {path.name}: {error}")
    if not chunks:
        raise RuntimeError("No extractable PDF text found in ./docs.")
    return chunks


def load_sentence_transformer(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name, local_files_only=False)


def load_cross_encoder(model_name: str) -> CrossEncoder:
    try:
        return CrossEncoder(model_name, local_files_only=True)
    except Exception:
        return CrossEncoder(model_name, local_files_only=False)


class AccreditationRAG:
    def __init__(self) -> None:
        self.chunks = ingest_all_pdfs()
        self.model = load_sentence_transformer(MODEL_NAME)
        self.bm25 = BM25Okapi([tokens(chunk.text) for chunk in self.chunks])
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        
        # Only re-embed if the chunk count has changed
        if self.collection.count() != len(self.chunks):
            try:
                self.client.delete_collection(COLLECTION)
            except Exception:
                pass
            self.collection = self.client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
            embeddings = self.model.encode([chunk.text for chunk in self.chunks], normalize_embeddings=True, show_progress_bar=False).tolist()
            self.collection.add(
                ids=[chunk.id for chunk in self.chunks],
                documents=[chunk.text for chunk in self.chunks],
                metadatas=[{"source": chunk.source, "page": chunk.page} for chunk in self.chunks],
                embeddings=embeddings,
            )
        self._nli: CrossEncoder | None = None
        self._reranker: CrossEncoder | None = None

    def dense_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        result = self.collection.query(
            query_embeddings=self.model.encode([query], normalize_embeddings=True).tolist(),
            n_results=min(top_k, len(self.chunks)), include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"]
        metadatas = result["metadatas"]
        distances = result["distances"]
        if documents is None or metadatas is None or distances is None:
            return []
        results: list[RetrievalResult] = []
        for text, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
            page_val = metadata.get("page", 0)
            page_num = page_val if isinstance(page_val, int) else 0
            results.append(RetrievalResult(
                Chunk("", text, str(metadata["source"]), page_num),
                max(0.0, 1 - float(distance)),
            ))
        return results

    def _bm25_candidates(self, query: str, limit: int) -> list[RetrievalResult]:
        """Use BM25 only to create a broad candidate set, never as final proof ranking."""
        scores = self.bm25.get_scores(tokens(query))
        order = np.argsort(-scores)[:limit]
        return [
            RetrievalResult(self.chunks[int(index)], float(scores[int(index)]))
            for index in order
            if scores[int(index)] > 0
        ]

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = load_cross_encoder(RERANKER_MODEL)
        return self._reranker

    def _semantic_rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        """Rank candidate passages by query-to-passage meaning, not word count."""
        unique: dict[tuple[str, int, str], RetrievalResult] = {}
        for item in candidates:
            unique.setdefault((item.chunk.source, item.chunk.page, item.chunk.text), item)
        items = list(unique.values())
        if not items:
            return []
        try:
            scores = self._get_reranker().predict([(query, item.chunk.text) for item in items])
            ranked = sorted(zip(items, scores), key=lambda pair: float(pair[1]), reverse=True)
            return [RetrievalResult(item.chunk, float(score)) for item, score in ranked[:top_k]]
        except Exception:
            # Dense similarity is a semantic fallback if the local reranker is unavailable.
            return items[:top_k]

    def bm25_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        candidates = self._bm25_candidates(query, limit=max(20, top_k * 4))
        # A paraphrase may contain none of the record's exact vocabulary.
        if not candidates:
            candidates = self.dense_search(query, top_k=max(20, top_k * 4))
        return self._semantic_rerank(query, candidates, top_k)

    def hybrid_search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Combine lexical and dense candidates, then select only semantic matches."""
        dense = self.dense_search(query, top_k=max(20, top_k * 4))
        bm25 = self._bm25_candidates(query, limit=max(20, top_k * 4))
        return self._semantic_rerank(query, dense + bm25, top_k)

    def search(self, query: str, method: str, top_k: int = 5) -> list[RetrievalResult]:
        if method == "Dense (Chroma)":
            return self.dense_search(query, top_k)
        if method == "BM25":
            return self.bm25_search(query, top_k)
        if method == "Hybrid + Reranker":
            return self.hybrid_search(query, top_k)
        raise ValueError(f"Unknown retrieval method: {method}")

    @staticmethod
    def _candidate_units(text: str) -> list[str]:
        """Break a retrieved passage into answer-sized extractive units."""
        units = re.split(r"(?<=[.!?])\s+|(?=\b(?:Resolution|Decision|Requirement|Eligibility|Approval item)\b)", text)
        return [unit.strip() for unit in units if len(unit.strip()) >= 25]

    def _best_fact(self, query: str, results: list[RetrievalResult]) -> tuple[str, Chunk] | None:
        query_terms = set(tokens(query))
        query_lower = query.lower()
        # Accreditation questions commonly ask for a specific numeric rule. Extract the
        # complete record sentence around that rule before applying generic QA ranking.
        targeted_patterns: list[str] = []
        if "cgpa" in query_lower or "gpa" in query_lower:
            targeted_patterns.append(r"(?:cgpa|cumulative grade point average).*?(?:\d+\.\d+|\d+)")
        if "attendance" in query_lower:
            targeted_patterns.append(r"(?:75%|65%|attendance)")
        if "grant" in query_lower or "seed" in query_lower or "journal" in query_lower:
            targeted_patterns.append(r"(?:\u20b9\s?[\d,]+|rs\.\s?[\d,]+)")
        if "ratio" in query_lower or "faculty" in query_lower:
            targeted_patterns.append(r"\b\d+\s*:\s*\d+\b")
        if "timeline" in query_lower or "resolution" in query_lower or "days" in query_lower:
            targeted_patterns.append(r"(?:7-day|seven days|resolution timeline)")
        for result in results:
            record_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", result.chunk.text)
            for record_sentence in record_sentences:
                for pattern in targeted_patterns:
                    if re.search(pattern, record_sentence, flags=re.IGNORECASE):
                        # Strip a structural heading when the factual sentence follows it.
                        record_sentence = re.sub(r"^.*?(?=(?:A student|The Senate|The University|A minimum|Students))", "", record_sentence).strip()
                        return record_sentence, result.chunk
        candidates: list[tuple[float, str, Chunk]] = []
        for rank, result in enumerate(results):
            for unit in self._candidate_units(result.chunk.text):
                unit_terms = set(tokens(unit))
                matches = len(query_terms & unit_terms)
                if not matches:
                    continue
                coverage = matches / max(1, len(query_terms))
                precision = matches / max(1, len(unit_terms))
                # Favor direct, concise facts without introducing any model-generated wording.
                score = (coverage * 1.8) + (precision * 0.35) + (0.25 / (rank + 1)) - (len(unit) / 9000)
                candidates.append((score, unit, result.chunk))
        if not candidates:
            return None
        best_score, sentence, chunk = max(candidates, key=lambda item: item[0])
        # Zero-hallucination gate: reject weak matches that lack meaningful overlap.
        # Targeted-pattern matches (ratios, CGPA, grants, etc.) bypass this check
        # because they return directly from the pattern-matching loop above.
        if best_score < 0.20:
            return None
        return sentence, chunk

    def answer_from_results(self, query: str, results: list[RetrievalResult]) -> tuple[str, list[RetrievalResult]]:
        """Create a citation-only answer from already retrieved evidence.

        This lets the evaluation backend validate every retriever without
        running retrieval a second time, while preserving the same abstention
        guardrail used by the Streamlit search UI.
        """
        if not results:
            return NO_EVIDENCE_RESPONSE, []
        best = self._best_fact(query, results)
        if best is None:
            return NO_EVIDENCE_RESPONSE, []
        fact, chunk = best
        citation = f"[Source: {chunk.source}, Page: {chunk.page}]"
        return f"{fact} {citation}", results

    def answer(self, query: str, method: str) -> tuple[str, list[RetrievalResult]]:
        """Retrieve evidence and return a grounded answer or the abstention sentinel."""
        results = self.search(query, method, top_k=5)
        return self.answer_from_results(query, results)

    def accreditation_audit(self) -> list[dict[str, str | bool]]:
        """Create a compact NAAC/NBA evidence dossier from the complete PDF corpus."""
        dossier: list[dict[str, str | bool]] = []
        for criterion, query, proof_pattern in ACCREDITATION_CRITERIA:
            results = self.bm25_search(query, top_k=10)
            verified = next((item for item in results if re.search(proof_pattern, item.chunk.text, re.IGNORECASE)), None)
            if verified:
                fact = self._best_fact(query, [verified])
                finding = fact[0] if fact else verified.chunk.text[:260].rstrip() + "..."
                dossier.append({
                    "Criterion": criterion,
                    "Status": "Verified",
                    "Finding": finding,
                    "Citation": f"[Source: {verified.chunk.source}, Page: {verified.chunk.page}]",
                    "source": verified.chunk.source,
                    "page": str(verified.chunk.page),
                })
            else:
                dossier.append({
                    "Criterion": criterion,
                    "Status": "Missing",
                    "Finding": "No supporting evidence was located in the provided governance records.",
                    "Citation": "—",
                    "source": "",
                    "page": "",
                })
        return dossier

    def criterion_coverage(self) -> list[dict]:
        """Return all criteria with evidence-strength rating for the heatmap dashboard."""
        coverage = []
        for row in self.accreditation_audit():
            verified = row["Status"] == "Verified"
            if not verified:
                strength, score, color = "No evidence", 0, "none"
            else:
                citation = str(row["Citation"])
                source = str(row["source"])
                finding_len = len(str(row.get("Finding", "")))
                if citation != "—" and source and finding_len > 60:
                    strength, score, color = "Strong evidence", 3, "strong"
                else:
                    strength, score, color = "Thin evidence", 2, "thin"
            coverage.append({
                "Criterion": str(row["Criterion"]),
                "Evidence strength": strength,
                "Coverage score": score,
                "Color": color,
                "Finding": str(row["Finding"]),
                "Citation": str(row["Citation"]),
                "source": str(row.get("source", "")),
                "page": str(row.get("page", "")),
            })
        return coverage

    # ------------------------------------------------------------------
    # Pick #4 – Evaluation helper (Recall@k, MRR, side-by-side)
    # ------------------------------------------------------------------

    def eval_query(
        self, query: str, expected_source: str, expected_page: int, top_k: int = 5
    ) -> dict:
        """Run one query through all three methods and return structured metrics."""
        results_map = {
            "Dense (Chroma)": self.dense_search(query, top_k),
            "BM25": self.bm25_search(query, top_k),
            "Hybrid + Reranker": self.hybrid_search(query, top_k),
        }
        row: dict = {"query": query}
        for method, res in results_map.items():
            hit_rank = next(
                (rank for rank, r in enumerate(res, 1)
                 if r.chunk.source == expected_source and r.chunk.page == expected_page),
                None,
            )
            key = method.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
            row[f"{key}_hit"] = int(hit_rank is not None)
            row[f"{key}_rank"] = hit_rank
            row[f"{key}_mrr"] = (1.0 / hit_rank) if hit_rank else 0.0
            row[f"{key}_results"] = res
        return row

    # ------------------------------------------------------------------
    # Pick #5 – Evidence pack export (DOCX + PDF)
    # ------------------------------------------------------------------

    def export_evidence_pack_docx(self, criterion: str | None = None) -> bytes:
        """Generate a DOCX evidence pack for a given criterion (or all)."""
        try:
            from docx import Document
            from docx.shared import Pt, RGBColor
            from docx.enum.text import WD_ALIGN_PARAGRAPH
        except ImportError:
            raise ImportError("python-docx is required. Run: pip install python-docx")
        doc = Document()
        title = doc.add_heading("Accreditation Evidence Pack", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub = doc.add_paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}  |  "
            f"Institution: KMEC  |  Framework: NAAC / NBA"
        )
        sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()
        coverage = self.criterion_coverage()
        rows = [r for r in coverage if criterion is None or r["Criterion"] == criterion]
        rgb_map = {"strong": RGBColor(0x1B, 0x6E, 0x3E), "thin": RGBColor(0x92, 0x59, 0x00), "none": RGBColor(0x9B, 0x1C, 0x1C)}
        for row in rows:
            h = doc.add_heading(row["Criterion"], level=1)
            if h.runs:
                h.runs[0].font.color.rgb = rgb_map.get(row["Color"], RGBColor(0, 0, 0))
            sp = doc.add_paragraph()
            sr = sp.add_run(f"Status: {row['Evidence strength']}")
            sr.bold = True
            if row["Finding"] and "No supporting" not in row["Finding"]:
                doc.add_paragraph(f"Finding: {row['Finding']}")
            if row["Citation"] and row["Citation"] != "—":
                cp = doc.add_paragraph()
                cr = cp.add_run(f"Citation: {row['Citation']}")
                cr.italic = True
                cr.font.color.rgb = RGBColor(0x33, 0x56, 0xD8)
            doc.add_paragraph("─" * 80)
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return buf.read()

    def export_evidence_pack_pdf(self, criterion: str | None = None) -> bytes:
        """Generate a PDF evidence pack using ReportLab."""
        try:
            from reportlab.lib import colors  # type: ignore[import-untyped]
            from reportlab.lib.pagesizes import A4  # type: ignore[import-untyped]
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore[import-untyped]
            from reportlab.lib.units import cm  # type: ignore[import-untyped]
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("reportlab is required. Run: pip install reportlab")
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story = []
        header_style = ParagraphStyle("H", parent=styles["Title"], fontSize=20,
                                      textColor=colors.HexColor("#172b4d"), spaceAfter=4)
        story.append(Paragraph("Accreditation Evidence Pack", header_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')} | Institution: KMEC | Framework: NAAC / NBA",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.5*cm))
        color_map = {"strong": colors.HexColor("#1B6E3E"), "thin": colors.HexColor("#925900"), "none": colors.HexColor("#9B1C1C")}
        coverage = self.criterion_coverage()
        rows = [r for r in coverage if criterion is None or r["Criterion"] == criterion]
        for row in rows:
            c = color_map.get(row["Color"], colors.black)
            crit_style = ParagraphStyle("CH", parent=styles["Heading2"], textColor=c, spaceBefore=10)
            story.append(Paragraph(row["Criterion"], crit_style))
            story.append(Paragraph(f"<b>Status:</b> {row['Evidence strength']}", styles["Normal"]))
            if row["Finding"] and "No supporting" not in row["Finding"]:
                finding_short = textwrap.shorten(row["Finding"], width=300, placeholder="…")
                story.append(Paragraph(f"<b>Finding:</b> {finding_short}", styles["Normal"]))
            if row["Citation"] and row["Citation"] != "—":
                cite_style = ParagraphStyle("Cite", parent=styles["Normal"], textColor=colors.HexColor("#3356D8"))
                story.append(Paragraph(f"<i>Citation: {row['Citation']}</i>", cite_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dde5f0"), spaceAfter=6))
        doc.build(story)
        buf.seek(0)
        return buf.read()

    def _get_nli(self) -> CrossEncoder:
        if self._nli is None:
            # Model is downloaded during setup, then loaded from the local cache.
            self._nli = load_cross_encoder(NLI_MODEL_NAME)
        return self._nli

    def faithfulness_check(self, answer: str, evidence: list[RetrievalResult]) -> list[dict[str, str | float | bool]]:
        """Use an NLI cross-encoder to test whether each answer sentence is entailed by evidence."""
        claims = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+(?=\[Source:|[A-Z])", answer) if segment.strip()]
        output = []
        nli = self._get_nli()
        model = getattr(nli, "model", None)
        if model is None:
            raise RuntimeError("NLI cross-encoder model failed to load.")
        config = getattr(model, "config", None)
        if config is None:
            raise RuntimeError("NLI model config is unavailable.")
        id2label = getattr(config, "id2label", None)
        if id2label is None:
            raise RuntimeError("NLI model config missing id2label mapping.")
        labels = {str(value).lower(): int(key) for key, value in id2label.items()}
        entailment_index = next((index for label, index in labels.items() if "entail" in label), 2)
        for claim in claims:
            claim_text = re.sub(r"\s*\[Source: .*?, Page: \d+\]", "", claim).strip()
            if not claim_text:
                continue
            probabilities = nli.predict([(item.chunk.text, claim_text) for item in evidence], apply_softmax=True)
            probabilities = np.atleast_2d(probabilities)
            best_index = int(np.argmax([float(row[entailment_index]) for row in probabilities]))
            entailment = float(probabilities[best_index][entailment_index])
            supported = entailment >= 0.50
            best = evidence[best_index]
            output.append({
                "claim": claim_text,
                "supported": supported,
                "score": entailment,
                "source": best.chunk.source,
                "page": best.chunk.page,
                "method": "NLI entailment",
            })
        return output

    def render_cited_page(self, source: str, page: int, highlight_terms: list[str]) -> bytes:
        """Render a cited source page and highlight matching evidence terms with PyMuPDF."""
        pdf_path = DOCS_DIR / source
        if not pdf_path.exists():
            raise FileNotFoundError(source)
        document = fitz.open(pdf_path)
        try:
            pdf_page = document[page - 1]
            for term in {term for term in highlight_terms if len(term) >= 4}:
                for rectangle in pdf_page.search_for(term, quads=False)[:8]:
                    annotation = pdf_page.add_highlight_annot(rectangle)
                    annotation.set_colors(stroke=(1, 0.79, 0.15))
                    annotation.update()
            pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(1.65, 1.65), alpha=False, annots=True)
            return pixmap.tobytes("png")
        finally:
            document.close()

    def governance_gaps(self) -> list[dict[str, str]]:
        """Flag possible conflicting norms or later resolutions for human review.

        These are deliberately labelled as review alerts, not automated findings of
        non-compliance: source authority and effective date require human judgement.
        """
        patterns = [
            ("Attendance threshold", r"attendance", r"\b\d{2,3}%\b"),
            ("Student-faculty ratio", r"student[- ]to[- ]faculty|faculty ratio", r"\b\d+\s*:\s*\d+\b"),
            ("Grievance timeline", r"grievance|resolution timeline", r"\b\d+[- ]day|\b\d+\s+days"),
        ]
        alerts: list[dict[str, str]] = []
        for label, topic_pattern, value_pattern in patterns:
            observations: dict[str, list[tuple[str, int]]] = {}
            for chunk in self.chunks:
                if not re.search(topic_pattern, chunk.text, re.IGNORECASE):
                    continue
                for value in re.findall(value_pattern, chunk.text, re.IGNORECASE):
                    observations.setdefault(value.replace(" ", ""), []).append((chunk.source, chunk.page))
            if len(observations) > 1:
                evidence = []
                for value, refs in observations.items():
                    source, page = refs[0]
                    evidence.append(f"{value} [Source: {source}, Page: {page}]")
                alerts.append({
                    "level": "Potential variation",
                    "message": f"{label} appears with multiple values across the corpus: " + "; ".join(evidence) + ". Review the governing document and effective dates.",
                })
        # Senate-style resolutions may supersede or refine a standing rule, but require
        # an explicit reviewer decision before treating them as an override.
        for chunk in self.chunks:
            if re.search(r"\bresolution:\b|\bapproved\b", chunk.text, re.IGNORECASE) and re.search(r"ratio|attendance|grant", chunk.text, re.IGNORECASE):
                alerts.append({
                    "level": "Resolution review",
                    "message": f"A potentially controlling resolution was found; compare it with older regulations before relying on it. [Source: {chunk.source}, Page: {chunk.page}]",
                })
                break
        return alerts


def build_rag() -> AccreditationRAG:
    return AccreditationRAG()
