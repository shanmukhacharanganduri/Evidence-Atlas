# Evidence Atlas

Accreditation evidence workspace: ask a question over the PDFs in `docs/` and get an extracted statement with a
citation to the exact page, or an explicit "unable to verify" when the records do not answer it.

## Setup

Tested on Python 3.12 (exact tested versions: `requirements.lock`).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # adds pytest and Playwright
python -m streamlit run app.py
```

On first run the app downloads two small Hugging Face models (`all-MiniLM-L6-v2` embeddings and the
`ms-marco-MiniLM-L-2-v2` reranker) and builds the index. Later starts reuse `index_cache/` (extracted text, keyed by file
content hash) and `chroma_db/` (vectors, keyed by a corpus fingerprint); unchanged PDFs are not re-parsed or re-embedded.
Both folders are generated and git-ignored. If a model cannot be downloaded the app says so and falls back to lexical
search instead of failing silently.

## How it works

- **Registry** (`corpus.py`, optional `document_registry.json`): every PDF is registered with institution, a synthetic
  flag, page counts, and extraction status (searchable / partly / no text / unreadable). Institution defaults to the
  `docs/kmec_archive/<institution>/` folder; edit the registry to set institution, `synthetic`, `effective_date`,
  `authority` and `origin`. The three PDFs made by `generate_sample_docs.py` are marked synthetic and say so in their footer.
- **Retrieval** (`rag_pipeline.py`): dense (Chroma), BM25 (pure lexical), BM25 + reranker, and the default
  Hybrid + reranker. Searches can be scoped to one institution from the UI.
- **Answers** (`answering.py`): a statement is returned only if it carries the value the question asks for (a percentage, a
  ratio, a phone number, ...), the question's named entities appear in the evidence, and a cross-encoder rates it
  relevant. Otherwise the app abstains. The result records which retrieved passage supports the answer, and the UI links it
  directly.
- **Coverage**: a criterion is "Located" only when one statement matches both its topic and its value pattern. Rows start as
  "Not reviewed"; locating a passage is not a compliance finding. Coverage runs on demand and exports (DOCX/PDF) reuse the
  displayed result and inherit the selected scope.
- **Evaluation** (`evaluate.py`): 40 evidence questions with labelled expected facts plus 10 no-evidence controls (5
  realistic near-misses). Retrieval recall, answer correctness, citation validity and abstention are scored separately and each
  run is saved immutably under `eval_runs/` with corpus and model fingerprints. Run it with `python evaluate.py` or from the
  Evaluation tab. The benchmark is scoped to the synthetic KMEC documents its labels refer to.

## Tests

```powershell
python -m pytest                 # unit + Streamlit AppTest suites, no model downloads (this is what CI runs)
python -m pytest -m integration  # real models, audited failure cases
python -m playwright install chromium; python -m pytest -m e2e   # browser tests against the real app
```

## Known limitations

- Scanned PDFs have no OCR: they are listed in the Corpus tab as "No searchable text", and Coverage warns when a scope
  contains such files. 11 of the 49 bundled PDFs are in this state.
- Tables are kept as text rows, not structured cells. The small reranker is weak on paraphrased "who/what proof" questions
  and can abstain or choose a neighbouring sentence; scoped benchmark answer accuracy is about 87.5%, not 100%.
- Index builds are guarded by an in-process lock; running several separate processes against one `chroma_db/` is not coordinated.
- There is no authentication. Do not expose a deployment with real institutional records without adding access control.
