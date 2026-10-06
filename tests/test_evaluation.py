import pandas as pd

import evaluate as ev
from answering import AnswerResult
from corpus import Chunk


class R:
    def __init__(self, chunk):
        self.chunk = chunk


def answered(statement, source="a.pdf", page=1):
    chunk = Chunk("id", "passage", source, page)
    return AnswerResult(query="q", status="answered", statement=statement, citation=chunk, candidates=[R(chunk)])


def test_invented_sentence_with_a_valid_citation_fails_correctness():
    forged = answered("The Moon is made of cheese.")
    assert ev.citation_reference_valid(forged)  # the reference is valid ...
    assert not ev.answer_is_correct(forged, ["75%"])  # ... but the answer is wrong


def test_wrong_attendance_sentence_fails_even_though_citation_is_valid():
    wrong = answered("Attendance and Condonation Attendance is recorded for every scheduled lecture.")
    assert ev.citation_reference_valid(wrong) and not ev.answer_is_correct(wrong, ["75%"])


def test_correct_statement_passes():
    assert ev.answer_is_correct(answered("A minimum attendance of 75% is mandatory."), ["75%"])


def test_abstention_is_not_a_correct_positive_answer():
    assert not ev.answer_is_correct(AnswerResult(query="q", status="insufficient_evidence"), ["75%"])


def test_summary_excludes_negative_cases_from_positive_recall():
    def row(label, hit, abstained):
        return {"Label": label, **{f"{n} {m}": v for n in ev.METHODS for m, v in
                                    (("Hit", hit), ("MRR", float(hit)), ("Answer Correct", hit), ("Abstained", abstained))}}
    summary = ev.summarise(pd.DataFrame([row("Evidence expected", 1, 0), row("No evidence expected", 0, 1)]))
    assert summary["Hybrid"]["recall_at_5"] == 1.0 and summary["Hybrid"]["correct_abstentions"] == 1
    assert summary["positive_cases"] == 1 and summary["negative_cases"] == 1


def test_benchmark_has_labelled_facts_and_realistic_negatives():
    assert all(case[4] for case in ev.BENCHMARK if case[3] == "Evidence expected")
    assert sum(1 for case in ev.BENCHMARK if case[3] == "No evidence expected") >= 10
