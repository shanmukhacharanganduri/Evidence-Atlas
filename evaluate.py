# -*- coding: utf-8 -*-
"""Labelled benchmark: retrieval, answer correctness, citation validity and abstention, scored separately.

Each run writes an immutable, run-ID-named CSV plus a JSON summary (corpus/model fingerprints) under
eval_runs/. Retrieval recall is computed over evidence-required cases only; abstention over the
no-evidence controls only.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

import pandas as pd

from rag_pipeline import MODEL_NAME, RERANKER_MODEL, ROOT, build_rag

RUNS_DIR = ROOT / "eval_runs"
METHODS = {"Dense": "Dense (Chroma)", "BM25": "BM25", "BM25+Rerank": "BM25 + Reranker", "Hybrid": "Hybrid + Reranker"}
REPORTED_METHOD = "Hybrid"  # what the UI uses
# The labelled facts come from the three synthetic sample documents, so the benchmark is scoped to them.
DEFAULT_SCOPE = "KMEC (synthetic demo)"

# (source, page, acceptable fact fragments, questions). A statement is correct if it contains ANY fragment.
POSITIVE_BENCHMARK_GROUPS = [
    ("Academic_Regulations_2024.pdf", 1, ["6.50"], [
        "What minimum CGPA is required for graduation?", "What CGPA is needed to receive a degree?",
        "State the graduation CGPA requirement.", "Which CGPA threshold qualifies a student for the award of a degree?"]),
    ("Academic_Regulations_2024.pdf", 2, ["75%"], [
        "What attendance percentage is mandatory to appear in end-semester examinations?", "What is the examination attendance threshold?",
        "How much attendance is required for semester exams?", "State the minimum attendance rule for end-semester examinations."]),
    ("Academic_Regulations_2024.pdf", 2, ["medical proof", "physician"], [
        "What medical proof is needed for attendance condonation?", "Which document supports a medical attendance condonation request?",
        "What evidence is required for medical condonation?", "What proof must a student submit for medical attendance relief?"]),
    ("Academic_Regulations_2024.pdf", 3, ["Disciplinary Committee"], [
        "Which committee reviews examination malpractice?", "Who handles examination malpractice cases?",
        "What body considers academic examination malpractice?", "Which committee is responsible for malpractice review?"]),
    ("Senate_Meeting_Minutes_Nov2024.pdf", 1, ["Vice-Chancellor"], [
        "Who chaired the November 2024 Senate meeting?", "Name the chair of the November Senate meeting.",
        "Who presided over the 2024 November Senate session?", "Identify the Senate meeting chair recorded for November 2024."]),
    ("Senate_Meeting_Minutes_Nov2024.pdf", 2, ["5,00,000"], [
        "What Research Seed Grant amount was approved for faculty publishing in top journals?", "How much seed funding was approved for top-tier journal publications?",
        "State the approved faculty research seed grant amount.", "What publication seed grant did the Senate approve?"]),
    ("Senate_Meeting_Minutes_Nov2024.pdf", 3, ["15:1"], [
        "What student-to-faculty ratio was approved for Computer Science?", "State the approved Computer Science student faculty ratio.",
        "Which student-faculty ratio applies to Computer Science?", "What ratio did the Senate approve for the Computer Science department?"]),
    ("Campus_Safety_and_AntiRagging_Policy.pdf", 1, ["Raghavan"], [
        "Who is the chair of the anti-ragging committee?", "Name the anti-ragging committee chair.",
        "Who heads the anti-ragging committee?", "Identify the chairperson of the anti-ragging committee."]),
    ("Campus_Safety_and_AntiRagging_Policy.pdf", 1, ["11002"], [
        "What is the anti-ragging squad duty officer contact number?", "Give the anti-ragging duty officer phone number.",
        "How can the anti-ragging squad duty officer be contacted?", "What contact number is listed for the anti-ragging duty officer?"]),
    ("Campus_Safety_and_AntiRagging_Policy.pdf", 2, ["7-day", "7 days", "seven days"], [
        "What is the strict grievance resolution timeline?", "Within how many days must grievances be resolved?",
        "State the grievance redressal timeline.", "What resolution period applies to student grievances?"]),
]

NO_EVIDENCE_QUESTIONS = [
    "What is the Zephyrian tritium calibration protocol?",
    "Who leads the Xylophor interplanetary observatory?",
    "What is the Quantarion warp-core maintenance schedule?",
    "Which department owns the Novalux asteroid laboratory?",
    "What is the Aetherion quantum-teleportation safety code?",
    # Realistic near-misses: plausible for a university, absent from the sample records.
    "What is the hostel curfew time for residents?",
    "What is the maximum tuition fee for the MBA programme?",
    "How many days of maternity leave do staff receive?",
    "Which body approves the university budget?",
    "What is the library collection size?",
]

# (query, source, page, label, acceptable fragments)
BENCHMARK = [
    (query, source, page, "Evidence expected", facts)
    for source, page, facts, questions in POSITIVE_BENCHMARK_GROUPS for query in questions
] + [(query, "", 0, "No evidence expected", []) for query in NO_EVIDENCE_QUESTIONS]


def rank_of(results, source: str, page: int) -> int | None:
    return next((rank for rank, r in enumerate(results, 1) if r.chunk.source == source and r.chunk.page == page), None)


def citation_reference_valid(result) -> bool:
    """The cited source/page is one of the retrieved passages. This does NOT show the answer is right."""
    return bool(result.answered and result.citation and any(
        r.chunk.source == result.citation.source and r.chunk.page == result.citation.page for r in result.candidates))


def answer_is_correct(result, facts: list[str]) -> bool:
    """The statement itself contains an expected fact (invented text with a valid citation fails)."""
    statement = result.statement.lower()
    return result.answered and any(fact.lower() in statement for fact in facts)


def citation_supports_answer(result, source: str, page: int) -> bool:
    return bool(result.answered and result.citation and result.citation.source == source and result.citation.page == page)


def summarise(frame: pd.DataFrame) -> dict:
    positive = frame[frame["Label"] == "Evidence expected"]
    negative = frame[frame["Label"] == "No evidence expected"]
    summary = {"positive_cases": len(positive), "negative_cases": len(negative)}
    for name in METHODS:
        summary[name] = {
            "recall_at_5": round(float(positive[f"{name} Hit"].mean()), 4),
            "mrr": round(float(positive[f"{name} MRR"].mean()), 4),
            "answer_correct": round(float(positive[f"{name} Answer Correct"].mean()), 4),
            "correct_abstentions": int(negative[f"{name} Abstained"].sum()),
        }
    return summary


def run_evaluation(rag=None, institution: str | None = DEFAULT_SCOPE, save: bool = True) -> pd.DataFrame:
    rag = rag or build_rag()
    rows = []
    for number, (query, source, page, label, facts) in enumerate(BENCHMARK, 1):
        expected_abstention = label == "No evidence expected"
        row = {"#": number, "Accreditation query": query, "Label": label,
               "Expected evidence": "No evidence should be returned" if expected_abstention else f"{source}, p.{page}"}
        for name, method in METHODS.items():
            results = rag.search(query, method, institution=institution)
            rank = rank_of(results, source, page) if not expected_abstention else None
            answer = rag.answer_from_results(query, results, method, institution)
            row[f"{name} Hit"] = int(rank is not None)
            row[f"{name} MRR"] = round(1.0 / rank, 4) if rank else 0.0
            row[f"{name} Answer Correct"] = int(answer_is_correct(answer, facts)) if not expected_abstention else 0
            row[f"{name} Citation Valid"] = int(citation_reference_valid(answer))
            row[f"{name} Cites Expected Page"] = int(citation_supports_answer(answer, source, page)) if not expected_abstention else 0
            row[f"{name} Abstained"] = int(not answer.answered) if expected_abstention else 0
            if name == REPORTED_METHOD:
                row["Reported Answer"] = answer.formatted()
                row["Reported Outcome"] = (
                    ("Abstained correctly" if not answer.answered else "Incorrectly answered") if expected_abstention
                    else ("Correct" if row[f"{name} Answer Correct"] else ("Abstained" if not answer.answered else "Wrong answer")))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if save:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
        RUNS_DIR.mkdir(exist_ok=True)
        frame.to_csv(RUNS_DIR / f"{run_id}.csv", index=False)
        meta = {"run_id": run_id, "corpus_fingerprint": rag.fingerprint, "index_version": rag.index_version,
                "embedding_model": MODEL_NAME, "reranker_model": RERANKER_MODEL, "institution_scope": institution,
                "summary": summarise(frame)}
        (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(meta, indent=2), "utf-8")
        frame.attrs["run_id"] = run_id
    return frame


if __name__ == "__main__":
    results = run_evaluation()
    print(f"Run {results.attrs.get('run_id')}: {len(results)} cases")
    print(json.dumps(summarise(results), indent=2))
