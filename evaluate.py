# -*- coding: utf-8 -*-
"""Ten-query evaluation for Track C: dense, BM25, hybrid+reranker, and citation-grounded answers."""
from __future__ import annotations


import re

import pandas as pd

from rag_pipeline import NO_EVIDENCE_RESPONSE, ROOT, build_rag

METHODS = ["Dense (Chroma)", "BM25", "Hybrid + Reranker"]


POSITIVE_BENCHMARK_GROUPS = [
    ("Academic_Regulations_2024.pdf", 1, [
        "What minimum CGPA is required for graduation?", "What CGPA is needed to receive a degree?", "State the graduation CGPA requirement.", "Which CGPA threshold qualifies a student for the award of a degree?",
    ]),
    ("Academic_Regulations_2024.pdf", 2, [
        "What attendance percentage is mandatory to appear in end-semester examinations?", "What is the examination attendance threshold?", "How much attendance is required for semester exams?", "State the minimum attendance rule for end-semester examinations.",
        "What medical proof is needed for attendance condonation?", "Which document supports a medical attendance condonation request?", "What evidence is required for medical condonation?", "What proof must a student submit for medical attendance relief?",
    ]),
    ("Academic_Regulations_2024.pdf", 3, [
        "Which committee reviews examination malpractice?", "Who handles examination malpractice cases?", "What body considers academic examination malpractice?", "Which committee is responsible for malpractice review?",
    ]),
    ("Senate_Meeting_Minutes_Nov2024.pdf", 1, [
        "Who chaired the November 2024 Senate meeting?", "Name the chair of the November Senate meeting.", "Who presided over the 2024 November Senate session?", "Identify the Senate meeting chair recorded for November 2024.",
    ]),
    ("Senate_Meeting_Minutes_Nov2024.pdf", 2, [
        "What Research Seed Grant amount was approved for faculty publishing in top journals?", "How much seed funding was approved for top-tier journal publications?", "State the approved faculty research seed grant amount.", "What publication seed grant did the Senate approve?",
    ]),
    ("Senate_Meeting_Minutes_Nov2024.pdf", 3, [
        "What student-to-faculty ratio was approved for Computer Science?", "State the approved Computer Science student faculty ratio.", "Which student-faculty ratio applies to Computer Science?", "What ratio did the Senate approve for the Computer Science department?",
    ]),
    ("Campus_Safety_and_AntiRagging_Policy.pdf", 1, [
        "Who is the chair of the anti-ragging committee?", "Name the anti-ragging committee chair.", "Who heads the anti-ragging committee?", "Identify the chairperson of the anti-ragging committee.",
        "What is the anti-ragging squad duty officer contact number?", "Give the anti-ragging duty officer phone number.", "How can the anti-ragging squad duty officer be contacted?", "What contact number is listed for the anti-ragging duty officer?",
    ]),
    ("Campus_Safety_and_AntiRagging_Policy.pdf", 2, [
        "What is the strict grievance resolution timeline?", "Within how many days must grievances be resolved?", "State the grievance redressal timeline.", "What resolution period applies to student grievances?",
    ]),
]

NO_EVIDENCE_QUESTIONS = [
    "What is the Zephyrian tritium calibration protocol?",
    "Who leads the Xylophor interplanetary observatory?",
    "What is the Quantarion warp-core maintenance schedule?",
    "Which department owns the Novalux asteroid laboratory?",
    "What is the Aetherion quantum-teleportation safety code?",
]

# 40 evidence-backed questions + 5 abstention controls = 45 labelled cases.
BENCHMARK = [
    (query, source, page, "Evidence expected")
    for source, page, questions in POSITIVE_BENCHMARK_GROUPS
    for query in questions
] + [
    (query, "", 0, "No evidence expected") for query in NO_EVIDENCE_QUESTIONS
]


def rank_of(results, source: str, page: int) -> int | None:
    return next((rank for rank, result in enumerate(results, 1) if result.chunk.source == source and result.chunk.page == page), None)


def status(rank: int | None) -> str:
    return f"✅ Passed / Hit @ {rank}" if rank else "⚠️ Not retrieved"


def citation_is_grounded(answer: str, results) -> bool:
    match = re.search(r"\[Source: (.*?), Page: (\d+)\]$", answer)
    if not match:
        return False
    source, page = match.group(1), int(match.group(2))
    return any(item.chunk.source == source and item.chunk.page == page for item in results)


def run_evaluation(rag=None) -> pd.DataFrame:
    rag = rag or build_rag()
    rows = []
    for number, (query, source, page, label) in enumerate(BENCHMARK, 1):
        dense_results  = rag.search(query, "Dense (Chroma)")
        bm25_results   = rag.search(query, "BM25")
        hybrid_results = rag.search(query, "Hybrid + Reranker")

        dense_rank  = rank_of(dense_results,  source, page)
        bm25_rank   = rank_of(bm25_results,   source, page)
        hybrid_rank = rank_of(hybrid_results, source, page)

        expected_abstention = label == "No evidence expected"
        answer_map = {
            "Dense": rag.answer_from_results(query, dense_results),
            "BM25": rag.answer_from_results(query, bm25_results),
            "Hybrid": rag.answer_from_results(query, hybrid_results),
        }
        valid_map = {
            name: (answer == NO_EVIDENCE_RESPONSE if expected_abstention else citation_is_grounded(answer, evidence))
            for name, (answer, evidence) in answer_map.items()
        }
        answer, answer_evidence = answer_map["BM25"]
        abstained = answer == NO_EVIDENCE_RESPONSE
        grounded = int(valid_map["BM25"])

        rows.append({
            "#": number,
            "Accreditation query": query,
            "Label": label,
            "Expected evidence": "No evidence should be returned" if expected_abstention else f"{source}, p.{page}",
            "Dense Hit":   int(dense_rank  is not None),
            "BM25 Hit":    int(bm25_rank   is not None),
            "Hybrid Hit":  int(hybrid_rank is not None),
            "Dense MRR":   round(1.0 / dense_rank,  4) if dense_rank  else 0.0,
            "BM25 MRR":    round(1.0 / bm25_rank,   4) if bm25_rank   else 0.0,
            "Hybrid MRR":  round(1.0 / hybrid_rank, 4) if hybrid_rank else 0.0,
            "Grounded Faithfulness": grounded,
            "Dense Answer Valid": int(valid_map["Dense"]),
            "BM25 Answer Valid": int(valid_map["BM25"]),
            "Hybrid Answer Valid": int(valid_map["Hybrid"]),
            "Backend Validation": "Passed" if all(valid_map.values()) else "Failed",
            "Dense Status":   status(dense_rank),
            "BM25 Status":    status(bm25_rank),
            "Hybrid Status":  status(hybrid_rank),
            "Grounded Status": ("Abstained correctly" if abstained else "Incorrectly answered") if expected_abstention else ("Citation grounded" if grounded else "Review needed"),
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "evaluation_results.csv", index=False)
    return frame


if __name__ == "__main__":
    results = run_evaluation()
    print(f"Saved {len(results)} evaluation rows to evaluation_results.csv")
    for method, hit_col, mrr_col in [
        ("Dense",  "Dense Hit",  "Dense MRR"),
        ("BM25",   "BM25 Hit",   "BM25 MRR"),
        ("Hybrid", "Hybrid Hit", "Hybrid MRR"),
    ]:
        print(f"{method}: Recall@5={results[hit_col].mean():.0%}  MRR={results[mrr_col].mean():.3f}")
    print(f"Grounded Faithfulness: {results['Grounded Faithfulness'].mean():.0%}")
