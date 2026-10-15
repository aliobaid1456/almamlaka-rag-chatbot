# Almamlaka TV Document Assistant

A bilingual (Arabic / English) RAG chatbot that answers questions about the Almamlaka TV **Digital Expansion Initiative** using only three supplied PDFs, cites the document and page for every answer, flags conflicts between documents, says so when the documents don't contain the answer, and resists prompt injection.

- **Live demo:** _add your deployment URL here_
- **Stack:** Streamlit · PyMuPDF · FastEmbed (multilingual MiniLM, ONNX) · FAISS + BM25 hybrid retrieval · any Anthropic or OpenAI-compatible LLM · pytest

> Independent technical demo built for an Almamlaka TV assessment; not an official Almamlaka TV service. The logo was supplied with the assessment.

---

## How it works

```
PDF ─► page-by-page extraction ─► section chunks (1 chunk = 1 section on 1 page) ─► embeddings + BM25 ─► FAISS index (persisted)
                                                                                                    │
question ─► input checks ─► injection screen ─► LLM query analysis ─► hybrid retrieval ─► evidence E1..En
            (length,         (EN/AR regex,       (intent, standalone   (dense + BM25, RRF,   │
             control chars)   no model call)      question, English     cross-document         ▼
                                                  search queries)       coverage)       LLM answer as JSON
                                                                                         (markers [E#], conflicts,
                                                                                          calculations)
                                                                                               │
                         render with trusted metadata ◄── code validation (IDs, figures, language, conflicts)
                         or abstain                        ─ fail → one corrective retry ─ fail → abstain
```

### 1. PDF extraction (`backend/pdf_parser.py`)
PyMuPDF reads each page's text with font information. Headings are detected from size/weight (18 pt bold vs 10.5 pt body); the three tables are detected with PyMuPDF's table finder and emitted one row per line **with their column headers** (`Quarter: Q2 2027; Development: $500,000; Marketing: $150,000; …`), so a row stays meaningful on its own. All six pages have a native text layer, so OCR isn't needed; empty pages are logged rather than skipped silently. Page numbers are physical 1-based indices (the PDFs define no page labels). Full inventory: [`evaluation/extraction_report.md`](evaluation/extraction_report.md).

### 2. Chunking (`backend/chunking.py`)
**One chunk per numbered section, never crossing a page boundary**, so every citation maps to exactly one verifiable page and section. The documents are short (16 sections, 30–188 tokens each), so the usual 500–800-token windows would merge unrelated sections (e.g. budget and data privacy) and blur citations. Sections longer than 500 tokens are split on sentence/row boundaries with 80 tokens of overlap (exercised in tests, not triggered by these PDFs). Each chunk stores `document_id`, `document_name`, `page_number`, `section_title`, `chunk_id`, `content_hash`; the document title and section heading are prepended for indexing only.

### 3. Embeddings and vector store (`backend/embeddings.py`, `backend/vector_store.py`)
- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0) via **FastEmbed** (ONNX Runtime, quantized, ~0.22 GB, no PyTorch). It was trained by multilingual knowledge distillation on parallel sentences, which is what this corpus needs: the PDFs are English, users may ask in Arabic. Configurable via `EMBEDDING_MODEL` (e.g. `intfloat/multilingual-e5-large` on bigger hardware).
- **Vector store:** FAISS `IndexFlatIP` on L2-normalized vectors (exact cosine). With 16 chunks an exact index is the right choice: no approximation error and no database service to run. The index, chunk metadata and a manifest are persisted in `index/`.
- **Deterministic, idempotent ingestion:** the manifest fingerprints the PDF hashes, chunking settings and embedding model; the index is rebuilt only when one of these changes.
- If the embedding model can't be downloaded, the app falls back to BM25-only retrieval and shows this in the sidebar.

### 4. Retrieval (`backend/retrieval.py`, `backend/query_rewriting.py`)
1. An LLM call classifies intent (`document_question`, `greeting`, `out_of_scope`, `prompt_injection`), rewrites follow-ups into a **standalone question** using the last 6 turns, and produces 1–3 **English search queries**.
2. Dense search (on the original question, the rewrite and the English queries) and **BM25** (Arabic-normalized; numbers indexed without separators so `$2,600,000` matches `2600000`) are fused with Reciprocal Rank Fusion; top-k = 6.
3. **Cross-document coverage:** a document missing from the top-k gets its best chunk added if it ranks just below the cut-off, so both sides of a conflict reach the generator.

Retrieval scores are only a ranking signal. Whether the evidence answers the question is decided by the generator and checked by the validators.

### 5. Grounded generation and citation validation (`backend/generation.py`, `backend/citations.py`)
The model receives a server-side system prompt, the question, and passages wrapped as `<evidence id="E3" …>` data blocks. It must reply with JSON:

```json
{"answerable": true, "answer": "… [E1] … [E3]", "evidence_ids": ["E1"],
 "conflicts": [{"topic": "…", "sides": [{"claim": "…", "evidence_id": "E1"}, {"claim": "…", "evidence_id": "E4"}]}],
 "calculations": [{"expression": "…", "result": "…"}]}
```

The model never writes a filename or page number. Code then checks:

| Check | Rejects |
|---|---|
| Evidence IDs | Any `[E#]` not in this retrieval |
| Citation presence | Answers without markers; any line stating a figure without a marker |
| Numeric grounding | Any number (≥2 digits) not in the **cited** passages, the user's question, or a verified calculation. `$2.4 million` is accepted for `$2,400,000`; Arabic-Indic digits are normalized |
| Calculations | Re-computed with a safe arithmetic parser; operands must come from evidence |
| Language | Arabic question → Arabic answer, English → English |
| Prompt leak | A canary token in the system prompt must never appear in output |

On failure the model gets one corrective retry with the validator's errors; if it still fails, the app **abstains** instead of showing an unverified answer. Citations are rendered from the stored chunk metadata, with expandable verbatim excerpts.

Abstention text is fixed in code:
- EN: *I couldn't find sufficient information to answer this question in the three supplied PDF documents.*
- AR: *لم أجد معلومات كافية للإجابة عن هذا السؤال ضمن ملفات PDF الثلاثة المقدّمة.*

### 6. Conflicts (`backend/conflict_detection.py`)
The PDFs contain two genuine cross-document conflicts, both handled:

| Topic | Side A | Side B |
|---|---|---|
| Total budget | $2,400,000 — Project Overview p.1 (also the original table total, Budget & Timeline p.1) | $2,600,000 after the Finance Committee revision — Budget & Timeline p.1 |
| Public launch date | March 15, 2027 — Project Overview p.1, Budget & Timeline p.2 | April 1, 2027 — Team & Governance p.1 |

Reported conflicts are validated in code before display: ≥2 sides, distinct passages, every figure in a side present in that side's passage, and different values on each side. Unsupported conflicts are dropped and logged. The UI shows them in a separate "The documents disagree" panel with a citation per side and does not pick a winner (it may note that a passage describes itself as a later revision).

### 7. Multi-turn and language
- History lives in the Streamlit session (one per browser tab) and is passed explicitly to a stateless pipeline, so sessions can't see each other.
- History is used **only** to rewrite the question; the generator sees the standalone question and freshly retrieved passages, never earlier answers.
- The answer language is detected from the **current** message (Arabic if ≥30% of letters are Arabic), so users can switch language mid-conversation. Arabic messages render RTL.

### 8. Prompt-injection and application security (`backend/security.py`)
- **Architecture first:** the model has no tools, no file system, no network; it only produces JSON that code validates.
- **Input:** NFKC normalization, control/bidi-override characters stripped, 1000-character limit.
- **Direct injection:** English and Arabic patterns (ignore instructions, reveal prompt/secrets, role change, run commands, other users' sessions, jokes/poems) are refused **before any model call**. Subtler attempts are classified by the query-analysis step. Both are tested for false positives on legitimate questions ("Who will act as the Project Lead?", "ما قصة تغيير موعد الإطلاق؟").
- **Indirect injection:** evidence is wrapped as data; tag-like text in a passage is neutralized so it can't close its own block; passages containing instruction-like text are flagged to the model; an answer that obeys such text fails grounding validation.
- **Secrets:** read from environment / platform secrets only; `.env` and `secrets.toml` are git-ignored; logs record lengths, IDs and chunk IDs, and a redactor masks key-shaped strings. Provider errors are mapped to safe messages (no keys, no raw bodies, no stack traces in the UI).
- Pinned dependencies.

**Limits:** pattern lists can be paraphrased around and the classifier is itself an LLM. The strongest guarantees are the structural ones: answers must cite retrieved passages and survive numeric validation, so a successful injection can at most produce a refusal or an abstention, not fabricated content with citations.

---

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env    # set LLM_API_KEY (and LLM_PROVIDER / LLM_MODEL / LLM_BASE_URL if not Anthropic)
set -a && source .env && set +a
python scripts/ingest_documents.py      # optional: the app builds the index on first start
streamlit run app.py
```

LLM configuration (`.env.example` has examples):

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | or `openai_compatible` (OpenAI, Groq, Gemini, OpenRouter, vLLM…) |
| `LLM_API_KEY` | — | required |
| `LLM_MODEL` | `claude-haiku-5-5` for Anthropic | required for `openai_compatible` |
| `LLM_BASE_URL` | — | required for `openai_compatible` |
| `LLM_REASONING_EFFORT` | — | e.g. `low` for reasoning models such as Groq `openai/gpt-oss-120b` |
| `EMBEDDING_BACKEND` | `fastembed` | `none` = BM25 only |

Requests use temperature 0; the model only ever sees the server prompt, the question and retrieved passages.

## Tests and evaluation

```bash
pytest                                   # 105 offline tests, scripted model
python scripts/run_evaluation.py         # 40-case end-to-end evaluation with the real model
python scripts/run_evaluation.py --retrieval-only
```

**Unit/integration tests** (`tests/`, model replaced by a scripted fake) cover the 18 required categories: supported answers, multi-chunk and multi-document answers, exact amounts/dates/names, unanswerable questions, false premises, English and Arabic follow-ups, language switching, cross-language retrieval, both real conflicts (and rejection of unsupported ones), direct and Arabic prompt injection, malicious instructions inside a passage, invalid evidence IDs, fabricated figures and page numbers, citation pages verified against the PDFs, missing documents and corrupt indexes, session isolation, and provider failures (429 retry, 5xx, timeout, bad credentials).

**End-to-end evaluation** (`evaluation/test_set.json`, 40 cases, all derived from the PDFs) records expected vs actual behaviour, retrieval recall, fact accuracy, citation validity (re-checked against the PDF page), abstention correctness, conflict detection and false-conflict rate, language match and security cases. Results: [`evaluation/report.md`](evaluation/report.md) and `evaluation/results.json`.

Measured results:

| Run | Result |
|---|---|
| `pytest` | 105 passed (earlier suite also verified from a fresh virtualenv built from `requirements-dev.txt`) |
| Retrieval-only, BM25 only, raw question (no rewriting) | English recall 1.00; Arabic recall 0.125 — Arabic questions have no lexical overlap with English PDFs, which is why the pipeline uses English search queries from the rewriter plus multilingual embeddings |
| End-to-end, Groq `openai/gpt-oss-120b`, hybrid retrieval (`--delay 6`) | **36/40 passed**; retrieval recall 1.00 (EN and AR); citation validity 1.00 on 27 answers; language match 40/40; security 5/5; conflicts 3/4. All 4 failures were safe (abstain/out-of-scope or a scoring-normalization miss), none fabricated. Details: [`evaluation/report.md`](evaluation/report.md) |
| Fixes after that run | `(E1)`-style markers accepted, partial-list answers, out-of-scope label overridden when the question names document entities, duplicate conflict sides merged, evaluation text normalization. Covered by new unit tests; end-to-end re-run pending |

## Deploy

**Streamlit Community Cloud** (simplest)
1. Push this repository to GitHub.
2. On share.streamlit.io: *Create app* → pick the repo, branch, `app.py`; under *Advanced settings* choose Python 3.12 and paste secrets:
   ```toml
   LLM_PROVIDER = "anthropic"
   LLM_API_KEY = "…"
   LLM_MODEL = "claude-haiku-5-5"
   ```
3. Deploy. The first start downloads the embedding model (~220 MB) and builds the index (a few seconds).

**Hugging Face Spaces (Docker)** — create a Docker Space, push the repo, add the same variables as *Secrets*. The `Dockerfile` downloads the embedding model and builds the index at build time, and runs as a non-root user on port 8501 (set `app_port: 8501` in the Space's README metadata).

**Any Docker host:** `docker build -t almamlaka-rag . && docker run -p 8501:8501 --env-file .env almamlaka-rag`

## Project layout

```
app.py                      Streamlit UI (branding, RTL/LTR, citations, conflict panel)
backend/config.py           settings from env / Streamlit secrets; abstention strings
backend/pdf_parser.py       PyMuPDF extraction: headings, tables, page metadata
backend/chunking.py         section-per-page chunks with metadata and hashes
backend/embeddings.py       FastEmbed multilingual embedder (+ BM25-only fallback)
backend/vector_store.py     FAISS index + metadata + manifest persistence
backend/ingestion.py        idempotent build/load of the index
backend/retrieval.py        hybrid dense + BM25 retrieval, RRF, cross-document coverage
backend/query_rewriting.py  intent, standalone question, English search queries
backend/generation.py       system prompt, evidence formatting, retry loop
backend/citations.py        evidence IDs, validation, numeric grounding, rendering
backend/conflict_detection.py  conflict validation
backend/security.py         input checks, injection heuristics, evidence sanitizing, redaction
backend/llm.py              Anthropic / OpenAI-compatible HTTP adapters with retries
backend/rag.py              pipeline orchestration
scripts/                    ingest_documents.py, run_evaluation.py
tests/                      pytest suite
evaluation/                 test set, extraction report, evaluation results
```

## Known limitations

- **Small corpus.** Three two-page documents; the design (exact FAISS, section chunks) is chosen for that. Larger corpora would need token-window chunking, an ANN index and a reranker.
- **Numeric grounding is not semantic grounding.** It catches invented figures, dates and page numbers, not a correct number attached to the wrong entity; the citation-per-claim requirement and the visible excerpts are the safeguard there.
- **Conflict detection depends on the model noticing the disagreement.** Code verifies every reported conflict and retrieval ensures both sides are present, but a conflict the model fails to report is not detected by rules.
- **Two model calls per question** (analysis + answer), plus one retry on validation failure; latency depends on the provider.
- **Arabic depends on the rewriter or embeddings** because the source PDFs are English; with both unavailable, Arabic questions abstain rather than guess.
- **Prompt-injection defences are best-effort** (see above).
- Names are transliterated by the model in Arabic answers; the original spelling is kept in parentheses where the model follows the instruction.
