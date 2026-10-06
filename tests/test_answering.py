from dataclasses import dataclass

from answering import NO_EVIDENCE_RESPONSE, AnswerResult, entity_terms, required_value_patterns, select_answer
from corpus import Chunk


@dataclass
class R:
    chunk: Chunk
    score: float = 1.0


def result(text, n=1, source="a.pdf", page=1):
    return R(Chunk(f"{source}::p{page}::c{n}", text, source, page))


ATTENDANCE_CHUNK = ("Attendance and Condonation Attendance is recorded for every scheduled lecture, tutorial, laboratory and studio session. "
                    "A minimum attendance of 75% in each registered course is mandatory for permission to appear in the end-semester examination.")


def test_attendance_percentage_answer_contains_the_percentage():
    selection, _ = select_answer("What attendance percentage is mandatory?", [result(ATTENDANCE_CHUNK)])
    assert selection and "75%" in selection.statement and "recorded for every" not in selection.statement


def test_faculty_development_question_never_selects_a_timestamp_or_ratio():
    furniture = result("All of the above Page 25/138 21-11-2024 01:31:35 and the ratio 3:1 appears in an unrelated annexure listing.")
    selection, reason = select_answer("What evidence supports faculty development activity?", [furniture])
    assert selection is None and reason


def test_unknown_entity_abstains_even_with_high_scores():
    chunk = result("The research training department conducts laboratory sessions for students every semester without exception.")
    selection, reason = select_answer("Which department owns the Novalux asteroid laboratory?", [chunk], scorer=lambda q, s: [9.0] * len(s))
    assert selection is None and "novalux" in reason.lower()


def test_low_relevance_scores_abstain():
    chunk = result("The library is open from nine in the morning until five in the evening on all working days.")
    selection, _ = select_answer("What is the library budget?", [chunk], scorer=lambda q, s: [-9.0] * len(s))
    assert selection is None


def test_unavailable_scorer_falls_back_to_lexical_selection():
    selection, _ = select_answer("What attendance percentage is mandatory?", [result(ATTENDANCE_CHUNK)], scorer=lambda q, s: None)
    assert selection and "75%" in selection.statement


def test_answer_from_fourth_result_reports_its_rank():
    results = [result("Unrelated sentence about the canteen timings of the university campus.", i, page=i) for i in range(1, 4)]
    results.append(result("The Senate approved a Student-to-Faculty Ratio of 15:1 for the Computer Science Department.", 4, "b.pdf", 7))
    selection, _ = select_answer("What student-to-faculty ratio was approved?", results)
    assert selection and selection.rank == 4 and selection.chunk.page == 7


def test_value_requirements_by_question_type():
    assert required_value_patterns("What attendance percentage is mandatory?")
    assert required_value_patterns("What is the duty officer phone number?")
    assert required_value_patterns("Who chaired the Senate meeting?") == []


def test_entity_terms_skip_sentence_start_and_acronyms():
    assert entity_terms("What is the Zephyrian tritium protocol?") == ["zephyrian"]
    assert entity_terms("Who chaired the IQAC meeting?") == []


def test_answer_result_formatting_round_trip():
    chunk = Chunk("x", "text", "a.pdf", 2)
    answered = AnswerResult(query="q", status="answered", statement="Fact.", citation=chunk)
    assert answered.formatted() == "Fact. [Source: a.pdf, Page: 2]"
    assert AnswerResult(query="q", status="insufficient_evidence").formatted() == NO_EVIDENCE_RESPONSE
