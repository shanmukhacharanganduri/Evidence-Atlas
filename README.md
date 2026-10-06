# Track C - Accreditation Evidence Assistant

The application ingests **every PDF in `docs/`**, chunks extracted text with source/page metadata, and builds:

- ChromaDB dense retrieval with local `all-MiniLM-L6-v2` embeddings;
- BM25 keyword retrieval with `rank-bm25`.

The UI offers a retrieval-method toggle. Answers are extractive: every displayed factual sentence has a `[Source: filename, Page: number]` citation. If no cited sentence supports a question, the app returns `Evidence not found in provided governance records.`

```powershell
python generate_sample_docs.py  # optional: regenerate the three sample PDFs
python evaluate.py
python -m streamlit run app.py
```

The evaluation runs ten accreditation queries, compares Dense and BM25 retrieval hit rate and answer faithfulness, and writes `evaluation_results.csv`.
