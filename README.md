# Enterprise Agentic RAG & Knowledge Platform

A backend platform (+ a Streamlit UI) for document-grounded question answering: upload
PDFs/DOCX/CSV/TXT, ask questions in natural language, and get answers with inline
citations back to the exact source chunk. Includes agentic query routing (documents vs.
deterministic tools), hybrid retrieval, reranking, JWT auth, and a retrieval/answer
evaluation pipeline.

Runnable **end-to-end with zero external services or API keys** - a local,
dependency-free embedding provider and a deterministic extractive answer generator are
the defaults with a one-line env-var swap to real hosted embeddings/LLMs (OpenAI) for
production use.

![Ask your documents — chat UI](assets/screenshot_chat.png)

## Problem statement

Most internal knowledge (policies, handbooks, runbooks, contracts) sits in PDFs and
Word docs that are slow to search and easy to misquote. Pasting documents into a raw
LLM prompt doesn't scale past a handful of files and gives no way to verify an answer
against its source. This project builds the backend a real "chat with your company
docs" product needs: ingestion, retrieval, grounded generation, citations back to the
exact chunk, and a way to measure whether retrieval is actually working ; not just a
script that embeds one PDF and calls an LLM once.

## Features

- **Document ingestion** : PDF, DOCX, TXT, CSV, chunked with overlap-aware recursive
  splitting so answer-relevant text doesn't get cut mid-sentence.
- **Agentic query routing** : a router decides per-query whether to hit the document
  index, run a deterministic tool (calculator, structured lookup), or both, instead of
  sending every query through the same path.
- **Hybrid retrieval** : dense vector similarity + BM25 keyword search, fused with
  Reciprocal Rank Fusion, so both exact-term queries and paraphrased queries are
  covered.
- **Reranking** before generation to push the most relevant chunks to the top.
- **Grounded generation with citations** : every answer is either backed by a specific
  retrieved chunk (`[source: N]`) or explicitly says it doesn't know; no free-form
  hallucinated answers.
- **Swappable providers** : embeddings and the LLM are both abstract interfaces with a
  local, offline default and a real OpenAI implementation already wired in.
- **JWT auth, conversation memory** (Redis, with in-process fallback), and a
  **retrieval/answer evaluation pipeline** with precision/recall/MRR/hit-rate and
  citation-rate/keyword-coverage metrics.
- **Streamlit UI** : upload documents, ask questions, see citations and routing/latency
  metadata, all against the real API.

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI + Uvicorn |
| ORM / DB | SQLAlchemy, SQLite (dev) / PostgreSQL (prod, via Docker) |
| Auth | JWT (`python-jose`) + bcrypt password hashing (`passlib`) |
| Embeddings | scikit-learn `HashingVectorizer` (default, offline) or OpenAI `text-embedding-3-small` |
| Keyword search | `rank_bm25` (BM25Okapi) |
| Generation | Deterministic extractive mock (default, offline) or OpenAI chat completions |
| Conversation memory | Redis, with automatic in-process fallback |
| UI | Streamlit |
| Testing | pytest (37 tests: unit + full-stack API integration) |
| Packaging | Docker + docker-compose (API + UI + Postgres + Redis) |

## How it works

1. **Ingest** - a document is loaded (`app/ingestion/loaders.py`), split into
   overlapping chunks (`chunking.py`), embedded, and written to the vector store.
2. **Route** - an incoming query is classified by `app/routing/query_router.py` into
   `retrieval`, `tool:calculator`, `tool:sql`, or `hybrid`.
3. **Retrieve** - for retrieval/hybrid routes, `HybridRetriever` runs dense vector
   search and BM25 in parallel and fuses the two ranked lists with RRF.
4. **Rerank** - the fused candidates are reranked by lexical overlap with the query
   before being truncated to the top-K sent to the LLM.
5. **Generate** - the LLM (mock or OpenAI) answers strictly from the provided context
   and marks which numbered source block it used.
6. **Respond** - the API returns the answer, resolved citations (chunk id + snippet),
   the route taken, and latency; the Streamlit UI renders all of it.

## Architecture

![Architecture diagram](assets/architecture.png)

```
Document → Chunk → Embed → Vector Store
                                 │
Query → Agent Router → Hybrid Retrieve → Rerank → LLM → Answer + Citations
              │                                     ▲
              └──────────────→ Tools ────────────────┘
```

Every arrow with more than one implementation (embedding provider, LLM provider,
vector store, reranker) is a swap point defined by an abstract base class - see
`app/*/base.py` and the `factory.py` files. That's the seam a production deployment
upgrades through without a rewrite.

## What's real vs. what's a documented stand-in

Being upfront about this because it's the difference between an honest portfolio piece
and an inflated one:

| Component | This implementation | Production would use |
|---|---|---|
| Vector search | Brute-force cosine similarity in SQL, O(n) per query | pgvector, Chroma, Pinecone, or similar ANN index |
| Embeddings (default) | Local hashing vectorizer (`HashingVectorizer`), term-hashing, no semantic understanding | A real embedding model - **OpenAI provider is implemented, tested, one env var away** |
| Generation (default) | Deterministic extractive sentence-selection, not free generation | A real LLM - **OpenAI provider implemented, same swap** |
| Keyword search stemming | Hand-rolled suffix-stripping heuristic | `nltk.PorterStemmer` or a language-aware analyzer |
| Query routing | Rule-based regex classifier | Same rules to start, graduating ambiguous cases to an LLM classifier as real query logs accumulate |
| Reranker | Lexical term-overlap blended with fused retrieval score | A cross-encoder model or hosted rerank API |

None of these are hidden ; each has a comment in the code explaining the trade-off and
naming the upgrade path. The **architecture and interfaces are the deliverable**; the
zero-dependency defaults exist so the whole thing runs and is testable without any
external account.

## Setup

### Local (no Docker, no external services)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for interactive API docs (Swagger UI).

In a second terminal, launch the UI:

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, register an account, upload a document, and ask a
question.

Seed demo data and run the eval pipeline instead of uploading manually:

```bash
python scripts/seed_demo_data.py
python scripts/run_eval.py
```

### Docker (one command: API + UI + Postgres + Redis)

```bash
cd docker
docker compose up --build
```

- API: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

`DATABASE_URL`/`REDIS_URL`/`API_BASE_URL` are all set automatically in
`docker-compose.yml` so the three services find each other.

### Switching to real embeddings / LLM generation

```bash
# in .env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
```

No code changes needed — `app/embeddings/factory.py` and `app/generation/factory.py`
read these at startup.

## Sample query / output

Request:

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "How many days do I have to request a refund?"}'
```

Response:

```json
{
  "answer": "Customers may request a refund within 30 days of purchase if the product is unused and in its original packaging. [source: 1]",
  "route": "retrieval",
  "citations": [
    {
      "marker": 1,
      "chunk_id": "5c1e2f0a-...",
      "document_id": "9b32d2c8-...",
      "snippet": "Customers may request a refund within 30 days of purchase if the product is unused and in its original packaging..."
    }
  ],
  "latency_ms": 3.1,
  "conversation_id": "a4e6...",
  "debug": {}
}
```

Same round trip in the UI, upload panel and citations expanded:

![Upload and document panel](assets/screenshot_upload.png)

**Demo (query → routing → retrieval → grounded answer with citation):**

![Demo GIF](assets/demo.gif)

> The screenshots and GIF above are rendered mockups of the real Streamlit UI (this
> development sandbox has no display/browser to capture live screen recordings). Every
> value shown — the routes, latencies, and citation text — matches real output from
> `scripts/run_eval.py` against `data/sample_support_handbook.txt`. Swap in real
> screenshots with `streamlit run streamlit_app.py` once you have a display.

## Evaluation

`scripts/run_eval.py` runs 19 hand-labeled queries against
`data/sample_support_handbook.txt` (all 5 sections: refunds, shipping, account
security, billing, support escalation) and reports:

- **Retrieval**: precision@k, recall@k, MRR, hit rate
- **Answer**: citation rate, keyword coverage against expected terms, latency

Latest run:

```
=== Retrieval metrics ===
  precision@8: 1.0
  recall@8:    1.0
  MRR:         1.0
  hit rate:    1.0

=== Answer metrics ===
  citation rate:        1.0
  avg keyword coverage: 0.895
  avg latency (ms):     ~3
```

Retrieval is perfect against this single-document eval set expected, since it's 5
sections in one file and the point of this eval is regression coverage, not a
challenging benchmark (see Limitations). The keyword-coverage gap (89.5%, not 100%) is
real and comes from the default **mock** extractive generator occasionally selecting an
adjacent sentence instead of the exact target sentence; visible per-query in the
script's output, and one of the honest reasons to swap in a real LLM (`LLM_PROVIDER=openai`)
for production quality.

Historical eval runs are also queryable via `GET /eval/runs`.

## Bugs found and fixed during development

Listed because "I tested it and found real bugs" is a more credible engineering story
than "it worked first try":

1. **Calculator tool regex bug**: the original expression-extraction regex matched a
   single stray whitespace character instead of the arithmetic substring, so
   `"what is 45 * 12 + 8"` failed to evaluate. Fixed by requiring the regex to match a
   proper `number (operator number)+` pattern.
2. **Query misrouting on math regex**: the router's math-detection pattern matched a
   bare `-`, `+`, `*`, `/`, or `^` **anywhere** in the query; so any hyphenated word
   ("two-factor", "follow-up") got misrouted to the calculator tool instead of document
   retrieval. Found by adding "Is two-factor authentication required?" to the eval set
   and seeing it come back with `"Could not evaluate expression"`. Fixed by requiring
   an operator to sit **between two digits** (`\d\s*[-+*/^]\s*\d`).
3. **Query misrouting on SQL intent**: `"How many days do I have to request a
   refund?"` was incorrectly routed to the SQL/structured-data tool because it contains
   "how many". Fixed by requiring both an aggregation phrase *and* a structured-data
   noun (documents/records/rows/table) before routing to the SQL tool.
4. **Stemming gap causing zero BM25 matches**: "refund" (query) vs. "refunds"
   (document) scored zero keyword overlap with naive `.split()` tokenization; a
   textbook BM25 failure mode. Fixed by adding a lightweight stemmer used consistently
   across BM25 search, the reranker, and the mock LLM's sentence scoring.
5. **DB engine bound at import time broke test isolation**: `app/db/session.py` builds
   its SQLAlchemy engine once at module import, so per-test `DATABASE_URL` overrides
   via env vars silently had no effect. Fixed properly using FastAPI's
   `dependency_overrides` in tests rather than patching around the root cause (a known
   limitation of the current app structure - see below).
6. **RRF tie-breaking bug**: when a vector-search false positive (embedding hash
   collision) and a real BM25 keyword match tied on Reciprocal Rank Fusion score, sort
   stability silently favored whichever ranked list was merged first. Fixed by adding
   raw combined score as a tiebreaker.
7. **`email-validator` missing from `requirements.txt`**: `pydantic.EmailStr` (used in
   `UserCreate`) raises `ImportError` at request time without it - a clean `pip
   install -r requirements.txt` looked fine until the first `/auth/register` call. Fixed
   by adding it explicitly instead of relying on a transitive extra.
8. **`bcrypt>=4.1` broke password hashing outright**: `passlib` 1.7.4 (last released
   2020) reads a `bcrypt.__about__.__version__` attribute that was removed in
   `bcrypt` 4.1+, so `pip install -r requirements.txt` on a fresh machine today
   installs a `bcrypt`/`passlib` combination where every `hash_password()` call raises
   `ValueError`. This is exactly the kind of transitive-dependency rot that "works on my
   machine" hides ; fixed by pinning `bcrypt<4.1` with a comment explaining why.

## Known limitations (things I'd flag in an interview unprompted)

- **Default embeddings are lexical, not semantic** : the offline `HashingVectorizer`
  matches on term overlap rather than meaning, so paraphrased queries with no shared
  vocabulary can miss relevant chunks. A real embedding model
  (`text-embedding-3-small`) is implemented and unit-tested behind the same interface
  - set `EMBEDDING_PROVIDER=openai` to use it but hasn't been run end-to-end here
  since this sandbox has no external network access.
- **Vector search doesn't scale** : brute-force O(n) cosine similarity. Fine for a
  demo, wrong past maybe 50k chunks. `VectorStore` is an abstract interface
  specifically so this is a one-file swap to pgvector/Chroma, not a rewrite.
- **Single-node, no real distributed concerns** : no sharding, no replication, no
  multi-worker cache coherence beyond what Redis already gives conversation memory.
- **Eval set is single-document (19 queries, 1 source file)** : enough to prove the
  pipeline works and catch regressions like the two routing bugs above, not a
  statistically meaningful retrieval benchmark. A real eval run needs 50–200+ labeled
  cases across many documents with genuine near-duplicate/distractor chunks.
- **No LLM-as-judge evaluation** : citation rate and keyword coverage are proxies, not
  a real correctness/faithfulness judgment, because that needs a real LLM call to be
  meaningful.
- **Docker Compose is written but not build-verified** : this development environment
  has no Docker daemon. The Dockerfile and compose file follow standard patterns and
  the app itself is fully tested; run `docker compose up --build` yourself before
  considering deployment done, and fix anything that surfaces.
- **Screenshots/GIF in this README are mockups**, not live captures : see the note in
  Sample query/output above.

## Future scope

- Swap brute-force vector search for pgvector once chunk volume justifies it
- LLM-as-judge answer evaluation (faithfulness/relevance scoring)
- Streaming responses (SSE) for the `/query` endpoint, and streaming tokens into the
  Streamlit UI
- Multi-turn context injection into retrieval (use conversation memory to rewrite
  follow-up queries, not just log them)
- Replace the rule-based router with a hybrid rule+LLM router once real query logs
  exist to identify misrouted edge cases
- Document-level access control (currently any authenticated user sees all documents)
- Expand the eval set to 50–200+ labeled queries across multiple documents

## Project structure

```
app/
  main.py                 FastAPI app, middleware, router registration
  config.py                 Settings (env-var driven)
  auth.py                    JWT auth
  schemas.py                  Pydantic request/response models
  db/
    models.py                  SQLAlchemy ORM models
    session.py                   Engine/session setup
  ingestion/
    loaders.py                    PDF/DOCX/TXT/CSV -> plain text
    chunking.py                     Recursive chunking with overlap
    pipeline.py                       Wires loaders -> chunking -> embedding -> DB
  embeddings/
    base.py / local_provider.py / openai_provider.py / factory.py
  retrieval/
    vector_store.py                    Cosine-similarity search (SQL-backed)
    hybrid_retriever.py                  Vector + BM25 fused with RRF
    reranker.py                            Lexical-overlap reranking
    text_utils.py                            Shared stemming tokenizer
  routing/
    query_router.py                           Rule-based agentic router
  tools/
    registry.py                                 Calculator + document-count tools
  generation/
    llm_provider.py                               Mock (extractive) generator
    openai_provider.py                              Real LLM provider
    factory.py
  agent/
    orchestrator.py                                   Ties routing/retrieval/tools/generation together
  memory/
    conversation.py                                     Redis-backed conv. memory, in-process fallback
  evaluation/
    retrieval_eval.py                                     Precision/recall/MRR/hit-rate
    answer_eval.py                                          Citation rate, keyword coverage
  api/
    routes_auth.py / routes_documents.py / routes_query.py / routes_eval.py
streamlit_app.py          Streamlit UI (upload, query, citations) — thin client over the API
tests/                     37 tests: unit + full-stack API integration
scripts/
  seed_demo_data.py            Ingests the sample handbook
  run_eval.py                    Runs the eval pipeline (19 queries), saves results
data/
  sample_support_handbook.txt      Demo document used by seed/eval scripts
assets/                             README images (architecture diagram, screenshots, demo GIF)
docker/
  Dockerfile / docker-compose.yml    API + UI + Postgres + Redis, one command
```

## Running tests

```bash
pytest tests/ -v
```

37 tests: chunking, embeddings, routing, tools, hybrid retrieval, reranking, the full
agent orchestrator, and full-stack API integration tests (auth, upload, query, citation
correctness). All passing as of this polish pass ; see "Bugs found and fixed" for two
real regressions caught and fixed while re-verifying this suite (a routing misfire on
hyphenated words, and a `bcrypt`/`passlib` version incompatibility that silently broke
every password hash).

## API overview

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | No | Liveness check |
| `/auth/register` | POST | No | Create account, returns JWT |
| `/auth/login` | POST | No | Login, returns JWT |
| `/documents/upload` | POST | Yes | Upload PDF/DOCX/TXT/CSV, triggers ingestion |
| `/documents` | GET | Yes | List ingested documents |
| `/documents/{id}` | DELETE | Yes | Delete a document and its chunks |
| `/query` | POST | Yes | Ask a question, returns answer + citations + route + latency |
| `/eval/runs` | GET | Yes | Historical evaluation run results |

Full interactive schema at `/docs` once the server is running.
