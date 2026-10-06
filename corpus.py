"""Corpus ingestion: document registry, text cleaning, sentence-aware chunking and a versioned cache.

This module has no machine-learning dependencies, so the app shell and unit tests can use it
without loading models. Every PDF is registered, including files that yield no searchable text.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

PARSER_VERSION = "3"
CHUNK_WORDS = 180
OVERLAP_WORDS = 40

INSTITUTION_LABELS = {
    "cbit": "CBIT", "iith": "IIT Hyderabad", "kmec": "KMEC", "kmit": "KMIT",
    "ngit": "NGIT", "nitw": "NIT Warangal",
}
UNSPECIFIED_INSTITUTION = "Unspecified"


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int
    institution: str = UNSPECIFIED_INSTITUTION
    synthetic: bool = False


@dataclass
class DocumentRecord:
    source: str
    institution: str = UNSPECIFIED_INSTITUTION
    synthetic: bool = False
    sha256: str = ""
    pages: int = 0
    zero_text_pages: list[int] = field(default_factory=list)
    error_pages: list[int] = field(default_factory=list)
    chunks: int = 0
    status: str = "ok"  # ok | partial | no_text | error
    error: str = ""
    effective_date: str = ""
    authority: str = ""
    origin: str = ""

    @property
    def searchable(self) -> bool:
        return self.chunks > 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def load_registry(path: Path) -> dict[str, dict]:
    """Optional per-document metadata overrides keyed by relative source path."""
    try:
        data = json.loads(path.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def default_institution(source: str) -> str:
    parts = source.split("/")
    if len(parts) >= 3 and parts[0] == "kmec_archive":
        return INSTITUTION_LABELS.get(parts[1].lower(), parts[1].upper())
    return UNSPECIFIED_INSTITUTION


def apply_registry(record: DocumentRecord, registry: dict[str, dict]) -> DocumentRecord:
    meta = registry.get(record.source, {})
    record.institution = str(meta.get("institution") or default_institution(record.source))
    record.synthetic = bool(meta.get("synthetic", False))
    record.effective_date = str(meta.get("effective_date", ""))
    record.authority = str(meta.get("authority", ""))
    record.origin = str(meta.get("origin", ""))
    return record


# ---------------------------------------------------------------------------
# Text cleaning and chunking
# ---------------------------------------------------------------------------

_FURNITURE = [
    re.compile(r"^\s*page\s+\d+\s*(?:/|of)?\s*\d*\s*$", re.I),
    re.compile(r"^\s*\d{1,4}\s*$"),
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?\b"),
]


def _normalise_line(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def clean_pages(pages: list[str], head: int = 2, tail: int = 3) -> list[str]:
    """Remove page furniture: page numbers, print timestamps and header/footer lines repeated across pages.

    Only the first `head` and last `tail` lines of a page can be dropped as repeated furniture, so
    legitimately repeated body lines (e.g. table rows) are kept.
    """
    split = [[line.strip() for line in page.splitlines() if line.strip()] for page in pages]
    counts: Counter[str] = Counter()
    for lines in split:
        edge = lines[:head] + lines[max(head, len(lines) - tail):]
        counts.update({_normalise_line(line) for line in edge})
    threshold = max(3, int(len(pages) * 0.4))
    cleaned = []
    for lines in split:
        kept = []
        for position, line in enumerate(lines):
            at_edge = position < head or position >= len(lines) - tail
            if any(pattern.search(line) for pattern in _FURNITURE):
                continue
            if at_edge and len(line) < 100 and counts[_normalise_line(line)] >= threshold:
                continue
            kept.append(line)
        cleaned.append("\n".join(kept))
    return cleaned


_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "sri", "smt", "st", "no", "nos", "rs", "fig", "vs",
                  "etc", "sr", "jr", "dept", "univ", "approx", "ph", "d", "e.g", "i.e", "viz", "ie", "eg"}


def _blocks(text: str) -> list[str]:
    """Split text into blocks, isolating short unpunctuated heading lines that precede a prose line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks: list[str] = []
    buffer: list[str] = []
    for index, line in enumerate(lines):
        following = lines[index + 1] if index + 1 < len(lines) else ""
        is_heading = (len(line) <= 70 and not re.search(r"[.!?:;,]$", line)
                      and len(following) >= 50 and following[:1].isupper())
        if is_heading:
            if buffer:
                blocks.append(" ".join(buffer))
                buffer = []
            blocks.append(line)
            continue
        buffer.append(line)
    if buffer:
        blocks.append(" ".join(buffer))
    return blocks


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation without breaking abbreviations, initials, numbering or Ph.D."""
    output: list[str] = []
    for block in _blocks(text):
        flat = re.sub(r"\s+", " ", block).strip()
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9₹\"'(\[])", flat)
        merged: list[str] = []
        for part in parts:
            if merged and merged[-1].endswith("."):
                last_word = merged[-1].rstrip(".").split(" ")[-1].lower().strip("()")
                if last_word in _ABBREVIATIONS or last_word.isdigit() or (len(last_word) == 1 and last_word.isalpha()):
                    merged[-1] = f"{merged[-1]} {part}"
                    continue
            merged.append(part)
        output.extend(m for m in merged if m)
    return output


def split_page(text: str, source: str, page: int, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS,
               institution: str = UNSPECIFIED_INSTITUTION, synthetic: bool = False) -> list[Chunk]:
    """Pack whole sentences into chunks of about `size` words with a sentence-aligned overlap."""
    units: list[str] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        if len(words) <= size:
            units.append(sentence)
        else:  # a very long unit (e.g. an unpunctuated table) is windowed by words
            step = max(1, size - overlap)
            units.extend(" ".join(words[i:i + size]) for i in range(0, len(words), step))
    chunks: list[Chunk] = []
    current: list[str] = []
    count = 0
    fresh = False  # whether `current` holds a unit not already emitted via the overlap

    def emit() -> None:
        passage = "\n".join(current).strip()
        if passage:
            chunks.append(Chunk(f"{source}::p{page}::c{len(chunks)}", passage, source, page, institution, synthetic))

    for unit in units:
        n = len(unit.split())
        if current and count + n > size:
            emit()
            tail: list[str] = []
            tail_count = 0
            for previous in reversed(current):
                pn = len(previous.split())
                if tail_count + pn > overlap:
                    break
                tail.insert(0, previous)
                tail_count += pn
            current, count, fresh = tail, tail_count, False
        current.append(unit)
        count += n
        fresh = True
    if current and fresh:
        emit()
    return chunks


# ---------------------------------------------------------------------------
# Ingestion with a versioned, incremental cache
# ---------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_document(path: Path, source: str) -> tuple[DocumentRecord, list[tuple[int, str]]]:
    """Return a record and cleaned (page, text) pairs. Never raises: failures are recorded."""
    from pypdf import PdfReader

    record = DocumentRecord(source=source)
    raw: list[str] = []
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                pass
        for number, pdf_page in enumerate(reader.pages, start=1):
            try:
                raw.append(pdf_page.extract_text() or "")
            except Exception:
                raw.append("")
                record.error_pages.append(number)
    except Exception as error:
        record.status, record.error = "error", f"{type(error).__name__}: {error}"
        return record, []
    record.pages = len(raw)
    cleaned = clean_pages(raw)
    pages = []
    for number, text in enumerate(cleaned, start=1):
        if text.strip():
            pages.append((number, text))
        elif number not in record.error_pages:
            record.zero_text_pages.append(number)
    if not pages:
        record.status = "no_text"
    elif record.error_pages:
        record.status = "partial"
    return record, pages


def _chunks_to_json(chunks: list[Chunk]) -> list[list]:
    return [[c.id, c.text, c.page] for c in chunks]


def load_corpus(docs_dir: Path, cache_dir: Path, registry_path: Path | None = None
                ) -> tuple[list[Chunk], list[DocumentRecord], str]:
    """Return (chunks, document records, corpus fingerprint), re-extracting only changed files."""
    registry = load_registry(registry_path) if registry_path else {}
    manifest_path = cache_dir / "corpus_manifest.json"
    cached: dict = {}
    try:
        stored = json.loads(manifest_path.read_text("utf-8"))
        if stored.get("parser_version") == PARSER_VERSION and stored.get("chunker") == [CHUNK_WORDS, OVERLAP_WORDS]:
            cached = stored.get("documents", {})
    except (OSError, ValueError):
        pass

    paths = sorted(docs_dir.rglob("*.pdf")) if docs_dir.exists() else []
    documents: dict[str, dict] = {}
    records: list[DocumentRecord] = []
    chunks: list[Chunk] = []
    for path in paths:
        source = path.relative_to(docs_dir).as_posix()
        stat = path.stat()
        entry = cached.get(source)
        if entry and entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
            sha = entry["sha256"]
        else:
            sha = _file_sha256(path)
        if entry and entry.get("sha256") == sha:
            base = DocumentRecord(**{**entry["record"]})
            pages = entry["pages"]
        else:
            base, extracted = extract_document(path, source)
            base.sha256 = sha
            pages = [[number, text] for number, text in extracted]
        base.sha256 = sha
        record = apply_registry(base, registry)
        doc_chunks: list[Chunk] = []
        for number, text in pages:
            doc_chunks.extend(split_page(text, source, number, institution=record.institution, synthetic=record.synthetic))
        record.chunks = len(doc_chunks)
        records.append(record)
        chunks.extend(doc_chunks)
        documents[source] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha,
                             "record": asdict(record), "pages": pages}

    fingerprint = hashlib.sha256(json.dumps(
        [PARSER_VERSION, CHUNK_WORDS, OVERLAP_WORDS]
        + [[r.source, r.sha256, r.institution, r.synthetic] for r in records]
    ).encode()).hexdigest()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"parser_version": PARSER_VERSION, "chunker": [CHUNK_WORDS, OVERLAP_WORDS],
                                   "fingerprint": fingerprint, "documents": documents}), "utf-8")
        tmp.replace(manifest_path)
    except OSError:
        pass  # the cache is an optimisation; failing to write it must not break the app
    return chunks, records, fingerprint
