# FinLens — Technical Architect's Getting Started Guide

> This is the order I would actually build this in. Not the "clean" order you'd present in a README. The real order — the one that surfaces problems early, validates assumptions before you build on them, and never lets you paint yourself into a corner.

---

## The Core Principle Before Anything Else

**Build vertically, not horizontally.**

Most engineers build horizontally: set up all the infra, all the tooling, all the services, then wire them together at the end and discover nothing works. That is the path to a half-finished project.

You build vertically: one thin slice of the real pipeline end-to-end as fast as possible. A single document, a single query, a real answer. Then you harden each layer. Then you expand the corpus. Then you add the observability. Then the CI gate.

The first question you should be able to answer within your first working session is: **"Given a real financial PDF, can I get a real grounded answer out of this pipeline?"**

Everything else — the full corpus, the CI eval gate, the React dashboard, the deployment — is scaffolding on top of that one proven vertical slice.

---

## Phase 0 — Validate Your Assumptions Before Writing a Single Line of Application Code

This is the phase most engineers skip. It is the most important phase.

You have five external dependencies that are not in your control. Every one of them needs to be validated in isolation before you build anything that depends on them. If any of them behave differently than assumed, you need to know now — not after you've built three layers on top of them.

### 0.1 — Validate Docling on a Real FinanceBench PDF

Docling is the entire foundation of your ingestion quality story. Before you build a single integration:

**What to do:**
```bash
pip install docling
```

Then write a standalone 30-line script:
```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("AAPL_2022_10K.pdf")
doc = result.document

# Print the first 20 elements with their types
for i, element in enumerate(doc.texts[:20]):
    print(f"[{element.label}] page={element.prov[0].page_num}: {element.text[:100]}")

# Specifically look for tables
tables = [el for el in doc.texts if el.label == "table"]
print(f"\nFound {len(tables)} tables")
if tables:
    print(tables[0].text[:500])
```

**What you're verifying:**
- Does Docling actually detect tables as `label="table"` or does it flatten them to text?
- Are page numbers populated on every element's `.prov[0].page_num`?
- Are footnotes preserved and linked?
- How long does it take to parse one document? (This matters for your ingestion pipeline design)

**What failure looks like:** Tables come out as a flat string of numbers with no structure. Page numbers are `None`. Footnotes are merged into adjacent paragraphs. If this happens, your entire ingestion story changes — you need to know this before you build the SemanticSplitter integration.

**What you need to see before continuing:** Tables extracted as distinct elements with cell structure. Headers tagged as headers. Page numbers on every element. If the output looks like garbage on a known complex PDF (Apple 10-K has extremely dense tables), switch to a simpler document first to confirm Docling works, then escalate.

---

### 0.2 — Validate the LlamaIndex DoclingReader Integration

Docling working in isolation is one thing. Docling producing output that LlamaIndex can ingest as proper `Document` nodes is another.

**What to do:**
```bash
pip install llama-index llama-index-readers-docling
```

```python
from llama_index.readers.docling import DoclingReader

reader = DoclingReader()
documents = reader.load_data("AAPL_2022_10K.pdf")

print(f"Number of documents produced: {len(documents)}")
for doc in documents[:3]:
    print(f"Metadata: {doc.metadata}")
    print(f"Text preview: {doc.text[:200]}")
    print("---")
```

**What you're verifying:**
- Does the reader produce one `Document` per element, or one `Document` per page, or one per file?
- Is the metadata (`page_number`, `element_type`, `source_file`) populated on each node?
- Is the text coherent or is it garbage?

**Critical note:** The `llama-index-readers-docling` package is relatively new. It may not be on PyPI as a stable release. You may need to install from source or use the LlamaIndex `DoclingNodeParser` instead. Check the exact package name against the current LlamaIndex docs before assuming it installs cleanly.

---

### 0.3 — Validate SemanticSplitterNodeParser

This is the chunking step. It needs an embedding model. Run it on the Docling output from the previous step.

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model
)

nodes = splitter.get_nodes_from_documents(documents)  # documents from step 0.2

print(f"Number of chunks: {len(nodes)}")
for node in nodes[:5]:
    print(f"Chunk length: {len(node.text)} chars")
    print(f"Metadata: {node.metadata}")
    print(f"Text: {node.text[:300]}")
    print("---")
```

**What you're verifying:**
- Are chunks semantically coherent? Does a table stay in one chunk or get split across two?
- Are the metadata fields (`page_number`, `element_type`) still present after splitting?
- What's the average chunk length? (too short = noisy retrieval; too long = loses precision)
- How long does chunking take per document?

**What failure looks like:** Metadata is dropped during splitting (the splitter creates new nodes without copying metadata from the source document). This is a known issue — you need to confirm metadata propagates or write the propagation logic yourself.

---

### 0.4 — Validate Your Qdrant Setup and Round-Trip

Before you ingest anything real, confirm Qdrant works with your metadata schema.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)
client.create_collection(
    collection_name="finlens_test",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)  # bge-large = 1024 dims
)

# Insert one fake point with your metadata schema
client.upsert(
    collection_name="finlens_test",
    points=[
        PointStruct(
            id=1,
            vector=[0.1] * 1024,
            payload={
                "company": "AAPL",
                "year": "2022",
                "doc_type": "10K",
                "page_number": 47,
                "element_type": "table",
                "source_file": "AAPL_2022_10K.pdf",
                "text": "Apple total net revenue was $394.3 billion"
            }
        )
    ]
)

# Test a payload filter query
from qdrant_client.models import Filter, FieldCondition, MatchValue
results = client.search(
    collection_name="finlens_test",
    query_vector=[0.1] * 1024,
    query_filter=Filter(
        must=[
            FieldCondition(key="company", match=MatchValue(value="AAPL")),
            FieldCondition(key="year", match=MatchValue(value="2022"))
        ]
    ),
    limit=5
)
print(results)
```

**What you're verifying:**
- Qdrant starts and is accessible
- Your metadata schema inserts and is queryable
- Payload filter syntax works as expected
- `bge-large` produces 1024-dimensional vectors (critical — if you create a collection with the wrong dimension, you'll need to drop and recreate it)

---

### 0.5 — Validate OpenRouter + LiteLLM

```bash
pip install litellm
export OPENROUTER_API_KEY="your-key"
```

```python
import litellm

response = litellm.completion(
    model="openrouter/free",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    stream=False
)
print(response.choices[0].message.content)
print(f"Tokens used: {response.usage}")
```

**What you're verifying:**
- Your API key works
- The model name format for OpenRouter via LiteLLM (`openrouter/` prefix)
- Streaming works (test with `stream=True` and iterate `response` as a generator)
- Token costs per request (so you can budget your $10)

---

### 0.6 — Validate the Cross-Encoder

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "What was Apple's total revenue in fiscal year 2022?"
candidates = [
    "Apple's net revenue for fiscal 2022 was $394.3 billion.",
    "The company increased its dividend by 5% in Q2.",
    "Revenue from iPhone was $205.5 billion in fiscal 2022.",
    "Apple operates retail stores in 25 countries."
]

scores = model.predict([[query, c] for c in candidates])
ranked = sorted(zip(scores, candidates), reverse=True)
for score, text in ranked:
    print(f"{score:.4f}: {text}")
```

**What you're verifying:**
- The model downloads and loads correctly (~85MB)
- The scoring is intuitive — the most relevant chunk should rank highest
- Inference latency for 10 pairs (your production load per query)

---

## Phase 1 — Project Structure and Environment Setup

Only after all Phase 0 validations pass do you touch project structure.

### 1.1 — Repository Layout

```
finlens/
├── backend/
│   ├── ingestion/          # Everything that touches documents offline
│   │   ├── parse.py        # Docling → LlamaIndex Document objects
│   │   ├── chunk.py        # SemanticSplitter → Nodes
│   │   ├── embed.py        # bge-large embedding
│   │   ├── index.py        # Qdrant + BM25 index building
│   │   └── run_ingestion.py  # CLI entrypoint: python -m ingestion.run_ingestion
│   ├── retrieval/          # Online query-time retrieval
│   │   ├── hybrid.py       # BM25 + Qdrant + RRF
│   │   ├── rerank.py       # Cross-encoder scoring
│   │   └── pipeline.py     # Assembles hybrid + rerank into one callable
│   ├── generation/
│   │   ├── prompt.py       # Prompt template (version-controlled here)
│   │   └── generate.py     # LiteLLM call + citation assembly
│   ├── observability/
│   │   └── tracing.py      # Langfuse span wrappers
│   ├── api/
│   │   └── main.py         # FastAPI app
│   └── pyproject.toml      # uv-managed dependencies
├── eval/
│   ├── ragas_eval.py       # Loads FinDER, runs pipeline, scores with RAGAS
│   └── thresholds.yaml     # faithfulness: 0.80, context_recall: 0.75, etc.
├── frontend/               # React app
├── .github/
│   └── workflows/
│       └── eval.yml        # GitHub Actions CI eval gate
├── docker-compose.yml
└── data/
    └── financebench/       # PDFs go here (gitignored)
```

**Why this structure matters:** The `ingestion/` directory is entirely offline — it never runs during query time. This separation prevents you from accidentally importing ingestion dependencies (Docling, which is heavy) into the hot path.

---

### 1.2 — Environment with uv

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize the project
cd backend
uv init
uv add \
  llama-index \
  llama-index-vector-stores-qdrant \
  llama-index-retrievers-bm25 \
  llama-index-embeddings-huggingface \
  llama-index-readers-docling \
  docling \
  sentence-transformers \
  langchain \
  langchain-core \
  litellm \
  langfuse \
  fastapi \
  uvicorn \
  qdrant-client \
  ragas \
  datasets  # for loading FinDER from HuggingFace
```

**Lock immediately:**
```bash
uv lock
```

Do not upgrade any of these dependencies mid-project. Pin them now. LlamaIndex and LangChain both break things across minor versions.

---

### 1.3 — Docker Compose

Write this before you run anything else. You want all services to start with one command from day one.

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse_db:5432/langfuse
      NEXTAUTH_SECRET: dev-secret
      NEXTAUTH_URL: http://localhost:3000
      SALT: dev-salt
    depends_on:
      - langfuse_db

  langfuse_db:
    image: postgres:15
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
    volumes:
      - langfuse_db_data:/var/lib/postgresql/data

volumes:
  qdrant_data:
  langfuse_db_data:
```

Run it: `docker compose up -d` — verify Qdrant is at `http://localhost:6333` and Langfuse is at `http://localhost:3000` before continuing.

---

## Phase 2 — The Ingestion Pipeline (Build and Validate)

This is the first real code you write. Everything downstream depends on the quality of what comes out of here.

### 2.1 — Write `parse.py`

This is the Docling adapter. Its job: take a PDF path, return a list of LlamaIndex `Document` objects with your full metadata schema populated.

```python
# ingestion/parse.py
from pathlib import Path
from docling.document_converter import DocumentConverter
from llama_index.core import Document

def parse_pdf(pdf_path: str, company: str, year: str, doc_type: str) -> list[Document]:
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    documents = []

    for element in doc.texts:
        if not element.text.strip():
            continue

        page_num = element.prov[0].page_num if element.prov else None

        documents.append(Document(
            text=element.text,
            metadata={
                "company": company,
                "year": year,
                "doc_type": doc_type,
                "page_number": page_num,
                "element_type": str(element.label),
                "source_file": Path(pdf_path).name
            }
        ))

    return documents
```

**Validate this immediately:** Call it on one real PDF. Print the first 10 documents and their metadata. Confirm every field is populated. This is your checkpoint before you write anything else.

---

### 2.2 — Write `chunk.py`

```python
# ingestion/chunk.py
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document

def build_splitter() -> SemanticSplitterNodeParser:
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    return SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model
    )

def chunk_documents(documents: list[Document], splitter: SemanticSplitterNodeParser):
    nodes = splitter.get_nodes_from_documents(documents)
    # CRITICAL: verify metadata propagated
    for node in nodes:
        assert "page_number" in node.metadata, f"Metadata lost during chunking: {node.node_id}"
        assert "company" in node.metadata
    return nodes
```

**The assertion block is not optional.** Run this and confirm metadata is preserved. If the assert fails, the fix is to manually copy metadata from source documents to output nodes before returning.

---

### 2.3 — Write `index.py`

Two indexes get built from the same set of nodes: Qdrant for vector search, BM25 for lexical search.

```python
# ingestion/index.py
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from qdrant_client import QdrantClient
import pickle

def build_qdrant_index(nodes, collection_name: str = "finlens"):
    client = QdrantClient("localhost", port=6333)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index

def build_bm25_index(nodes, persist_path: str = "data/bm25_index.pkl"):
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    # Persist to disk
    with open(persist_path, "wb") as f:
        pickle.dump(retriever, f)
    return retriever
```

---

### 2.4 — Write `run_ingestion.py` and Run It on 3 Documents

Do NOT ingest all 150 documents on your first run. Ingest exactly 3: one 10-K, one 10-Q, one earnings release, from different companies. This gives you a corpus that tests your payload filters without wasting an hour on a full ingestion.

```python
# ingestion/run_ingestion.py
from parse import parse_pdf
from chunk import build_splitter, chunk_documents
from index import build_qdrant_index, build_bm25_index

DOCUMENTS = [
    ("data/financebench/AAPL_2022_10K.pdf", "AAPL", "2022", "10K"),
    ("data/financebench/MSFT_2022_10Q.pdf", "MSFT", "2022", "10Q"),
    ("data/financebench/AMZN_2022_earnings.pdf", "AMZN", "2022", "earnings"),
]

splitter = build_splitter()
all_nodes = []

for pdf_path, company, year, doc_type in DOCUMENTS:
    print(f"Parsing {pdf_path}...")
    docs = parse_pdf(pdf_path, company, year, doc_type)
    print(f"  → {len(docs)} elements extracted")

    nodes = chunk_documents(docs, splitter)
    print(f"  → {len(nodes)} chunks produced")

    all_nodes.extend(nodes)

print(f"\nTotal chunks: {len(all_nodes)}")
print("Building Qdrant index...")
build_qdrant_index(all_nodes)

print("Building BM25 index...")
build_bm25_index(all_nodes)

print("Done.")
```

Run this. Verify in the Qdrant UI (`http://localhost:6333/dashboard`) that you can see your collection with the correct number of points and that payload fields are populated.

---

## Phase 3 — The Retrieval Pipeline (Build and Validate)

### 3.1 — Write and Test `hybrid.py`

Before wiring this into anything, write a standalone test that queries your 3-document index and prints results.

```python
# retrieval/hybrid.py
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import MetadataFilters, FilterCondition, MetadataFilter
import pickle

def build_hybrid_retriever(vector_index, bm25_path: str = "data/bm25_index.pkl", top_k: int = 10):
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)

    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=top_k)

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=1,  # don't generate query variants
        mode="reciprocal_rerank",
        use_async=False
    )

    return hybrid_retriever

def retrieve_with_filters(retriever, query: str, company: str = None, year: str = None):
    # Apply metadata filters if provided
    filters = []
    if company:
        filters.append(MetadataFilter(key="company", value=company))
    if year:
        filters.append(MetadataFilter(key="year", value=year))

    # Set filters on the retriever
    retriever._retrievers[0]._filters = MetadataFilters(filters=filters) if filters else None

    nodes = retriever.retrieve(query)
    return nodes
```

**Run a test query immediately:**
```python
results = retrieve_with_filters(retriever, "What was total revenue?", company="AAPL", year="2022")
for r in results:
    print(f"Score: {r.score:.4f} | Page: {r.metadata.get('page_number')} | {r.text[:150]}")
```

You need to see results that are obviously related to the query. If you're getting unrelated chunks back, your index is broken, not your retrieval code. Fix upstream.

---

### 3.2 — Write and Test `rerank.py`

```python
# retrieval/rerank.py
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore

_model = None  # Lazy load — don't load at import time

def get_reranker():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model

def rerank(query: str, nodes: list[NodeWithScore], top_n: int = 3) -> list[NodeWithScore]:
    model = get_reranker()
    pairs = [[query, node.text] for node in nodes]
    scores = model.predict(pairs)

    scored_nodes = sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)

    # Return top_n with updated scores
    reranked = []
    for score, node in scored_nodes[:top_n]:
        node.score = float(score)
        reranked.append(node)

    return reranked
```

**Test this in isolation** using your retrieval output. Print the top 3 reranked results and manually verify they are more relevant than what BM25+semantic returned alone. This is your human spot-check of the pipeline's precision.

---

### 3.3 — Write `pipeline.py` — The First Full Vertical Slice

```python
# retrieval/pipeline.py
from hybrid import build_hybrid_retriever, retrieve_with_filters
from rerank import rerank

def retrieve_and_rerank(query: str, vector_index, company: str = None, year: str = None):
    retriever = build_hybrid_retriever(vector_index)
    candidates = retrieve_with_filters(retriever, query, company=company, year=year)
    top_chunks = rerank(query, candidates, top_n=3)
    return top_chunks
```

This function is the bridge between ingestion and generation. Test it with a real question from your 3-document corpus. Print the 3 chunks. They should be the correct source text to answer the question.

---

## Phase 4 — The Generation Layer

### 4.1 — Write and Lock the Prompt Template

The prompt is a first-class artifact in this system. It lives in a file. It is not a string inside a function. It is version-controlled and every change to it will later trigger a RAGAS eval.

```python
# generation/prompt.py

SYSTEM_PROMPT = """You are a precise financial document analyst. You answer questions
strictly based on the provided source excerpts from SEC filings.

Rules you must follow:
1. Answer only from the provided context. Do not use outside knowledge.
2. Every factual claim must be followed by a citation in this exact format: [SOURCE: {company}, {year}, {doc_type}, page {page_number}]
3. If the context does not contain enough information to answer, say exactly: "The provided documents do not contain sufficient information to answer this question."
4. For numerical figures, quote them exactly as they appear in the source. Do not round or paraphrase numbers.
5. If multiple source chunks support the answer, cite each one where relevant."""

USER_PROMPT_TEMPLATE = """Answer the following question using only the source excerpts below.

Question: {query}

Source Excerpts:
{context}

Answer:"""

def build_prompt(query: str, chunks) -> list[dict]:
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        context_parts.append(
            f"[Excerpt {i+1} | {meta.get('company')} {meta.get('year')} {meta.get('doc_type')} "
            f"page {meta.get('page_number')}]\n{chunk.text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query, context=context)}
    ]
```

**Why this matters for your resume:** You can point to this file in a code review and say "this is the prompt, it's versioned, every change to it goes through a RAGAS eval gate." That is not something most candidates can say.

---

### 4.2 — Write `generate.py`

```python
# generation/generate.py
import litellm
from prompt import build_prompt

def generate_answer(query: str, chunks, model: str = "openrouter/free", stream: bool = False):
    messages = build_prompt(query, chunks)

    response = litellm.completion(
        model=model,
        messages=messages,
        max_tokens=1000,
        stream=stream
    )

    if stream:
        return response  # caller iterates

    return {
        "answer": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_cost_usd": litellm.completion_cost(response)
        },
        "chunks": [
            {
                "text": c.text,
                "metadata": c.metadata,
                "reranker_score": c.score
            }
            for c in chunks
        ]
    }
```

---

### 4.3 — Run Your First Full End-to-End Query

Write a single test script:

```python
# test_pipeline.py  (not a unit test, a manual validation script)
from ingestion.parse import parse_pdf
from retrieval.pipeline import retrieve_and_rerank
from generation.generate import generate_answer
# (Assumes index already built from Phase 2)

query = "What was Apple's total net revenue in fiscal year 2022?"

print("Retrieving...")
chunks = retrieve_and_rerank(query, vector_index, company="AAPL", year="2022")

print(f"\nTop {len(chunks)} chunks selected by cross-encoder:")
for i, c in enumerate(chunks):
    print(f"\n[{i+1}] Score: {c.score:.4f} | {c.metadata}")
    print(c.text[:300])

print("\nGenerating answer...")
result = generate_answer(query, chunks)

print("\n=== ANSWER ===")
print(result["answer"])
print(f"\nTokens: {result['usage']}")
```

**This is your go/no-go checkpoint.** Before you write a single line of FastAPI, Langfuse, or React code, you need this test to produce a factually correct answer with a real citation pointing to the right page. If it doesn't, there is something wrong in the pipeline that needs to be fixed before you build anything else on top of it.

---

## Phase 5 — Observability (Langfuse)

Only add Langfuse after the pipeline works. Adding observability to a broken pipeline makes debugging harder, not easier.

### 5.1 — Create a Langfuse Account and Get Keys

Go to `http://localhost:3000` (your Docker Langfuse instance). Create an account. Go to Settings → API Keys. Copy `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.

### 5.2 — Write `tracing.py`

```python
# observability/tracing.py
import os
from langfuse import Langfuse
from functools import wraps
import time

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host="http://localhost:3000"  # change to cloud URL in prod
)

def trace_pipeline(query: str, company: str, year: str):
    """Context manager that wraps a full pipeline execution in a Langfuse trace."""
    return langfuse.trace(
        name="rag_query",
        input={"query": query, "company": company, "year": year}
    )
```

### 5.3 — Instrument the Pipeline

Add spans to each step. The key steps to instrument are:
- `hybrid_retrieval` — log number of candidates returned, latency
- `cross_encoder_rerank` — log scores per chunk, which 3 were selected
- `llm_generation` — log prompt, response, token count, cost

Do this in the `pipeline.py` and `generate.py` files by accepting an optional `trace` parameter and calling `trace.span(...)` around each step.

---

## Phase 6 — FastAPI Backend

Only after the full pipeline is traced and working in the test script.

### 6.1 — Write `api/main.py`

```python
# api/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    company: str | None = None
    year: str | None = None

@app.post("/chat")
async def chat(request: QueryRequest):
    # Non-streaming first — get this working before adding streaming
    chunks = retrieve_and_rerank(request.query, vector_index, request.company, request.year)
    result = generate_answer(request.query, chunks, stream=False)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}
```

Test with `curl` before touching the frontend:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple revenue in 2022?", "company": "AAPL", "year": "2022"}'
```

Only once this returns correct JSON do you add streaming.

---

## Phase 7 — RAGAS Evaluation Setup

### 7.1 — Write the Eval Script Before Writing the CI Config

The eval script needs to work locally first. Get it running manually before you touch GitHub Actions.

```python
# eval/ragas_eval.py
from datasets import load_dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, answer_relevancy
from ragas.llms import LiteLLMWrapper

# Load FinDER — start with 20 questions, not 200
dataset = load_dataset("Linq-AI-Research/FinDER", split="train[:20]")

# CRITICAL: Configure RAGAS to use a cheap judge model, NOT GPT-4
judge_llm = LiteLLMWrapper("openrouter/free")

results = []
for item in dataset:
    question = item["question"]
    ground_truth = item["answer"]

    # Run your pipeline
    chunks = retrieve_and_rerank(question, vector_index)
    result = generate_answer(question, chunks)

    results.append({
        "question": question,
        "answer": result["answer"],
        "contexts": [c["text"] for c in result["chunks"]],
        "ground_truth": ground_truth
    })

# Score
scores = evaluate(
    dataset=results,
    metrics=[faithfulness, context_recall, answer_relevancy],
    llm=judge_llm
)
print(scores)
```

### 7.2 — Check the Thresholds.yaml Against Realistic Scores

Before setting your CI gate thresholds, run the eval manually on 20 questions and look at what scores you actually get. If your pipeline is producing faithfulness of 0.65 on the first run, setting a gate at 0.80 means your CI is permanently red from day one. The gates should be set at "current baseline minus a small regression tolerance," not aspirational targets that you haven't hit yet.

Set aspirational targets in the vision doc. Set realistic, passing thresholds in `thresholds.yaml` based on what your actual pipeline produces.

---

## Phase 8 — React Frontend

Build the frontend last. It is the most visible part of the project but the least technically interesting. Build it in this order:

1. **Chat view first** — static, hardcoded query, just displays the API response with citations. No streaming yet.
2. **Add streaming** — SSE from FastAPI, token-by-token rendering
3. **Add company/year filters** — dropdown selectors passed to `/chat`
4. **Metrics dashboard** — pulls from Langfuse API, shows faithfulness trend over deployments
5. **Eval history table** — shows CI run results with score deltas

Don't design the frontend until the backend is working. Many engineers waste two days making the UI beautiful before the API it calls is functional.

---

## Phase 9 — GitHub Actions CI Gate

Only write the CI workflow after you have:
- A working eval script that runs locally
- Langfuse receiving eval scores
- A known baseline score to compare against

```yaml
# .github/workflows/eval.yml
name: RAGAS Eval Gate

on:
  pull_request:
    paths:
      - "backend/generation/prompt.py"    # Only trigger on prompt changes
      - "backend/retrieval/**"            # Or retrieval changes
      - "eval/**"

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run RAGAS eval
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
          QDRANT_URL: ${{ secrets.QDRANT_CLOUD_URL }}
        run: |
          pip install uv
          uv sync
          python eval/ragas_eval.py --threshold-file eval/thresholds.yaml

      # The eval script exits with code 1 if any metric is below threshold
      # GitHub Actions will mark the check as failed automatically
```

**Critical:** The Qdrant index used in CI must be your pre-built cloud index, not rebuilt from scratch on every run. Building the index in CI on 150 documents would take too long and cost too much. Point CI at Qdrant Cloud which holds your pre-ingested index.

---

## Pre-Implementation Checklist

Before writing any application code, you should be able to check every box:

**Data:**
- [ ] FinanceBench PDFs downloaded and stored in `data/financebench/`
- [ ] You've opened 3 real PDFs and confirmed they have the table/footnote structure you expect
- [ ] FinDER dataset loads via `load_dataset("Linq-AI-Research/FinDER")` without error
- [ ] You've spot-checked 10 FinDER questions and confirmed their source companies exist in your FinanceBench corpus

**Models:**
- [ ] Docling parses a real 10-K and produces typed elements with page numbers
- [ ] `bge-large-en-v1.5` loads locally, produces 1024-dim vectors
- [ ] `cross-encoder/ms-marco-MiniLM-L-6-v2` loads locally, ranks a test pair correctly
- [ ] LiteLLM → OpenRouter → Mistral returns a response with correct token tracking

**Infrastructure:**
- [ ] `docker compose up -d` starts Qdrant and Langfuse cleanly
- [ ] Qdrant dashboard accessible at `:6333/dashboard`
- [ ] Langfuse UI accessible at `:3000`
- [ ] Qdrant Cloud free cluster created (for CI and production)
- [ ] OpenRouter key active with available balance

**Code:**
- [ ] `uv lock` file committed
- [ ] `.env.example` with all required environment variable names (never commit actual keys)
- [ ] `.gitignore` includes `data/financebench/` (PDFs are large), `.env`, `__pycache__/`

---

## The One Thing That Will Determine If This Project Is Good or Mediocre

Run this question through your pipeline once you have the ingestion done:

> "What was Apple's total net revenue for fiscal year 2022, and how did it compare to fiscal year 2021?"

This question requires your system to:
- Retrieve the right table (not just any revenue mention)
- Handle cross-year comparison (tests metadata filtering)
- Return an exact number, not a paraphrase (tests hallucination resistance)
- Cite a specific page (tests citation assembly)

If your system answers this correctly with a source citation pointing to the actual page in the actual document, your pipeline is working. Everything else is polish.
