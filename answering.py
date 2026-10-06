"""Question-aware extractive answer selection with relevance-based abstention.

Pure Python: a sentence scorer (e.g. a cross-encoder) is injected, so the logic is testable
without models. A statement is returned only if it is relevant to the question, carries the kind
of value the question asks for, and the question's named entities appear in the evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from corpus import Chunk, split_sentences

NO_EVIDENCE_RESPONSE = "Evidence not found in provided governance records."

STOPWORDS = {"a", "an", "the", "is", "are", "what", "which", "who", "when", "where", "how", "for", "to", "of", "in",
             "on", "and", "with", "does", "do", "was", "were", "it", "that", "this", "be", "by", "at", "as", "or",
             "required", "state", "name", "give", "identify", "applies", "apply"}
# A cross-encoder (ms-marco) logit below this is treated as "not an answer to the question".
RERANK_THRESHOLD = -6.0
LEXICAL_THRESHOLD = 0.5
COVERAGE_WEIGHT = 8.0  # how strongly rare-term coverage re-ranks statements that already pass the relevance gate

Scorer = Callable[[str, Sequence[str]], Sequence[float]]


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[\w%]+", text.lower(), flags=re.UNICODE) if w not in STOPWORDS]


@dataclass(frozen=True)
class Selection:
    statement: str
    chunk: Chunk
    rank: int  # 1-based position of the supporting chunk among the retrieved results
    score: float


@dataclass
class AnswerResult:
    """Structured outcome of a question. `status` is one of answered, insufficient_evidence,
    ambiguous_scope, processing_error."""
    query: str
    status: str
    statement: str = ""
    citation: Chunk | None = None
    support_rank: int = 0
    score: float = 0.0
    candidates: list = field(default_factory=list)  # RetrievalResult list, supporting chunk included
    notices: list[str] = field(default_factory=list)
    institution: str | None = None
    method: str = ""
    index_version: str = ""
    seconds: float = 0.0
    error: str = ""
    reason: str = ""

    @property
    def answered(self) -> bool:
        return self.status == "answered"

    def formatted(self) -> str:
        """Legacy single-string form used by the benchmark: statement plus citation, or the sentinel."""
        if not self.answered or self.citation is None:
            return NO_EVIDENCE_RESPONSE
        return f"{self.statement} [Source: {self.citation.source}, Page: {self.citation.page}]"


# ---------------------------------------------------------------------------
# Question analysis
# ---------------------------------------------------------------------------

_NUMBER_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"


def required_value_patterns(query: str) -> list[re.Pattern]:
    """Value shapes an answer sentence must contain for this kind of question (all must match)."""
    q = query.lower()
    needs: list[str] = []
    if re.search(r"\bratio\b", q):
        needs.append(r"\d+\s*:\s*\d+")
    elif re.search(r"\bcgpa\b|\bgpa\b|grade point", q) and re.search(r"minimum|required|threshold|needed|qualif|\bwhat\b", q):
        needs.append(r"\d+\.\d+|\b\d+\b")
    elif re.search(r"percent|attendance (?:threshold|rule|required)|how much attendance|attendance.*(?:mandatory|required|minimum)", q):
        needs.append(r"\d+(?:\.\d+)?\s*%|percent")
    elif re.search(r"phone|contact number|telephone|mobile number|helpline", q):
        needs.append(r"\+?\d[\d\s\-]{6,}")
    elif re.search(r"how many days|within how many|timeline|time ?frame|resolution period|\bdays\b", q):
        needs.append(rf"\b\d+[- ]?days?\b|\b{_NUMBER_WORDS}[- ]days?\b|\bwithin\s+\d+\s*(?:hours|days)")
    elif re.search(r"how much|amount|budget|grant", q):
        needs.append(r"₹|rs\.?\s*\d|\binr\b|\b\d[\d,]*(?:\.\d+)?\b")
    return [re.compile(p, re.I) for p in needs]


_PREDICATE_SYNONYMS = {"presid": ("chair", "presid"), "head": ("chair", "head")}
_ROLE_NOUNS = {"chair": ("chair",), "chairperson": ("chair",), "chairman": ("chair",), "head": ("chair", "head"), "heads": ("chair", "head")}
_NOT_PREDICATES = {"required", "needed", "provided", "supplied", "indexed", "related", "recorded", "listed", "named", "noted", "applied"}


def predicate_stems(query: str) -> list[str]:
    """Stems of past-participle verbs in the question (chaired -> chair), with a few synonyms."""
    stems: list[str] = []
    for word in re.findall(r"[a-z]+", query.lower()):
        if word in _ROLE_NOUNS:
            stems.extend(_ROLE_NOUNS[word])
        elif len(word) >= 6 and word.endswith("ed") and word not in _NOT_PREDICATES:
            stem = word[:-2]
            stems.extend(_PREDICATE_SYNONYMS.get(stem, (stem[:5],)))
    return stems


def entity_terms(query: str) -> list[str]:
    """Mid-question capitalised words (named entities such as 'Novalux') that must appear in evidence."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", query)
    terms = []
    for index, word in enumerate(words):
        if index == 0 or len(word) < 5 or not word[0].isupper():
            continue
        if word.isupper() and len(word) <= 5:
            continue  # acronyms are matched by the lexical gate instead
        terms.append(word.lower())
    return terms


# ---------------------------------------------------------------------------
# Candidate statements
# ---------------------------------------------------------------------------

def candidate_statements(text: str, max_chars: int = 320) -> list[str]:
    """Answer-sized units of a passage. Over-long units (tables) are windowed by their lines."""
    output: list[str] = []
    # Chunks store one ingestion-time unit per line, so re-split line by line (not across lines).
    for sentence in (s for line in text.splitlines() for s in split_sentences(line)):
        if len(sentence) < 25 or (len(sentence) < 40 and not re.search(r"[.!?]$", sentence)):
            continue  # headings and fragments are not statements
        if len(sentence) <= max_chars:
            output.append(sentence)
            continue
        words = sentence.split()
        window, step = 40, 25
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + window])
            if len(piece) >= 25:
                output.append(piece)
            if start + window >= len(words):
                break
    return output


def _lexical_score(query_terms: set[str], statement: str) -> float:
    terms = set(tokens(statement))
    if not query_terms:
        return 0.0
    return len(query_terms & terms) / len(query_terms)


def select_answer(query: str, results: Sequence, scorer: Scorer | None = None,
                  term_weight: Callable[[str], float] | None = None) -> tuple[Selection | None, str]:
    """Pick the statement that answers `query`, or return (None, reason) to abstain.

    `results` are RetrievalResult-like objects with a `.chunk` attribute.
    """
    if not results:
        return None, "no passages were retrieved"
    evidence_text = " ".join(item.chunk.text.lower() for item in results)
    missing = [term for term in entity_terms(query) if term not in evidence_text]
    if missing:
        return None, f"the question names {', '.join(missing)}, which does not appear in the retrieved passages"

    requirements = required_value_patterns(query)
    query_terms = set(tokens(query))
    stems = predicate_stems(query)
    pool: list[tuple[str, Chunk, int]] = []
    for rank, item in enumerate(results, start=1):
        for statement in candidate_statements(item.chunk.text):
            if requirements and not all(pattern.search(statement) for pattern in requirements):
                continue
            has_predicate = any(stem in statement.lower() for stem in stems)
            if _lexical_score(query_terms, statement) == 0.0 and not has_predicate:
                continue
            pool.append((statement, item.chunk, rank))
    if not pool:
        return None, "no retrieved statement contains the value the question asks for"

    raw = None
    if scorer is not None:
        scored = scorer(query, [p[0] for p in pool])  # None means the scorer is unavailable
        raw = None if scored is None else [float(v) for v in scored]
    weights = {term: (term_weight(term) if term_weight else 1.0) for term in query_terms}
    total = sum(weights.values()) or 1.0

    def weighted_coverage(statement: str) -> float:
        present = set(tokens(statement))
        return sum(w for term, w in weights.items() if term in present) / total

    if raw is None:  # lexical fallback: coverage is the relevance signal
        relevance = [_lexical_score(query_terms, p[0]) for p in pool]
        threshold, ranking = LEXICAL_THRESHOLD, [r - 0.02 * (p[2] - 1) for r, p in zip(relevance, pool)]
    else:  # relevance gate on the cross-encoder; ranking adds rare-term coverage
        relevance, threshold = raw, RERANK_THRESHOLD
        ranking = [r + COVERAGE_WEIGHT * weighted_coverage(p[0]) for r, p in zip(raw, pool)]
    allowed = [i for i in range(len(pool)) if relevance[i] >= threshold]
    if stems:  # the question's action (e.g. "chaired") must appear when any relevant statement has it
        with_predicate = [i for i in allowed if any(stem in pool[i][0].lower() for stem in stems)]
        allowed = with_predicate or allowed
    if not allowed:
        return None, "the retrieved statements are not relevant enough to the question"
    best = max(allowed, key=lambda i: (ranking[i], -pool[i][2]))
    statement, chunk, rank = pool[best]
    return Selection(statement.strip(), chunk, rank, relevance[best]), ""
