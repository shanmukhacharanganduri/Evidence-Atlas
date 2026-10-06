"""Real-model checks of the audited failure cases: `python -m pytest -m integration` (models must be cached)."""
import pytest

from rag_pipeline import build_rag

pytestmark = pytest.mark.integration
SCOPE = "KMEC (synthetic demo)"


@pytest.fixture(scope="module")
def rag():
    return build_rag()


def test_attendance_answer_contains_75_percent(rag):
    result = rag.answer("What attendance percentage is mandatory?", institution=SCOPE)
    assert result.answered and "75%" in result.statement


def test_faculty_development_question_does_not_return_a_timestamp_or_ratio(rag):
    result = rag.answer("What evidence supports faculty development activity?")
    assert "01:31:35" not in result.statement and "Page 25/138" not in result.statement


def test_nonexistent_entity_abstains(rag):
    assert rag.answer("Which department owns the Novalux asteroid laboratory?").status == "insufficient_evidence"


def test_scoped_search_never_leaves_the_institution(rag):
    results = rag.search("minimum attendance required for examinations", "Hybrid + Reranker", institution="KMIT")
    assert results and {r.chunk.institution for r in results} == {"KMIT"}
