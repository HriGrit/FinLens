# FinLens — Data Flow & API Contracts

> **Learning goal:** Trace exactly what data looks like at every stage of the pipeline. Know the shape of every major data structure. This helps answer "what happens if X is missing?" type interview questions.

---

## 1. Ingestion Data Flow

### Stage 0: Input

```
File: data/financebench/pdfs/3M_2022_10K_10.pdf
Parsed filename into DocSpec:
  company  = "3M"
  year     = "2022"
  doc_type = "10-K"
  filename = "3M_2022_10K_10.pdf"
```

`run_ingestion.py` parses the filename to extract company, year, and doc_type. This is fragile — if the filename format changes (e.g., `3M_10K_2022.pdf`), parsing fails. A more robust approach would be a metadata sidecar file (JSON) per PDF.

---

### Stage 1: Docling → Raw Elements

Docling produces a `DoclingDocument`, which is a tree of elements. Each element has:
- `label`: `text`, `section_header`, `table`, `figure`, `list_item`
- `text`: raw text content
- `prov`: list of provenance objects, each with `.page` (1-indexed page number)

```python
# Conceptual Docling element structure
element = {
    "label": "text",
    "text": "Net sales for the year ended December 31, 2022 were $35.4 billion.",
    "prov": [{"page": 42, "bbox": {...}}]
}
```

---

### Stage 2: `doc_to_nodes()` → TextNodes

Each Docling element becomes one `TextNode`. Label is normalized:

| Docling label | Normalized `element_type` |
|---|---|
| `section_header` | `heading` |
| `text` | `paragraph` |
| `table` | `table` |
| others | passed through as-is |

Tables are special — the text is replaced with markdown export:
```
# Raw Docling table text: "Net Sales 35.4 32.2 ..."
# After export_to_markdown():
| Metric | 2022 | 2021 |
|---|---|---|
| Net Sales | 35.4 | 32.2 |
```

**TextNode shape after `doc_to_nodes()`:**
```python
TextNode(
    node_id="uuid-...",
    text="## Revenue from Operations\nNet sales for 2022 were $35.4B",
    metadata={
        "element_type": "paragraph",  # str
        "page_number": 42,            # int (1-indexed) or None
        "filename": "3M_2022_10K_10.pdf",  # str
        "company": "3M",              # str
        "year": "2022",               # str
        "doc_type": "10-K",           # str
    }
)
```

---

### Stage 3: `inject_heading_context()` → Enriched TextNodes

- Heading nodes are consumed and their text is stored as `current_heading`
- Each subsequent paragraph/table node gets heading text prepended
- Heading nodes are **not emitted** — they're dropped

```python
# Before:
[
    TextNode(text="Revenue from Operations", metadata={"element_type": "heading"}),
    TextNode(text="Net sales were $35.4B", metadata={"element_type": "paragraph"}),
    TextNode(text="Operating income was $4.2B", metadata={"element_type": "paragraph"}),
]

# After inject_heading_context():
[
    TextNode(text="## Revenue from Operations\nNet sales were $35.4B", ...),
    TextNode(text="## Revenue from Operations\nOperating income was $4.2B", ...),
]
# Heading node is gone
```

---

### Stage 4: `split_paragraph_nodes()` → Chunked TextNodes

Only `element_type == "paragraph"` nodes are semantically split. Others pass through.

If a paragraph is short enough (or SemanticSplitter determines it's one topic), it may not be split at all.

If split, the resulting chunks inherit all metadata from the parent node. The `chunk_index` metadata field is added:

```python
# Parent paragraph → 2 semantic chunks:
TextNode(text="## Section\nThe company operates safety products...", metadata={..., "chunk_index": 0})
TextNode(text="## Section\nThe industrial segment focuses on...",   metadata={..., "chunk_index": 1})
```

---

### Stage 5: Qdrant Upsert

Each TextNode is converted to a Qdrant `PointStruct`:

```python
PointStruct(
    id="uuid-...",           # node_id
    vector=[0.021, -0.14, ...],  # 1024-dim bge-large embedding
    payload={
        "text": "## Revenue from Operations\n...",
        "element_type": "paragraph",
        "page_number": 42,
        "filename": "3M_2022_10K_10.pdf",
        "company": "3M",
        "year": "2022",
        "doc_type": "10-K",
    }
)
```

The `payload` is what Qdrant stores alongside the vector. Payload fields are filterable.

---

## 2. Query Data Flow

### Stage 1: HTTP Request

```http
POST /chat
Content-Type: application/json

{
    "query": "What was 3M's net sales in 2022?",
    "company": "3M",
    "year": "2022",
    "model": "openrouter/stepfun/step-3.5-flash:free",
    "top_k": 10,
    "rerank_top_k": 3
}
```

---

### Stage 2: Hybrid Retrieval

**Qdrant query:**
```python
client.search(
    collection_name="finlens",
    query_vector=query_embedding,  # 768-dim gte-modernbert (⚠️ mismatch!)
    query_filter=Filter(must=[
        FieldCondition(key="company", match=MatchValue(value="3M")),
        FieldCondition(key="year",    match=MatchValue(value="2022")),
    ]),
    limit=10
)
```

Returns 10 `ScoredPoint` objects with their payload and similarity score.

**BM25 query:**
```python
bm25_retriever.retrieve("What was 3M's net sales in 2022?")
```
Returns up to 10 TextNode objects. **No company/year filter.**

**RRF fusion:**
```python
# Each result list has items ranked 1..10
# RRF score = sum(1 / (60 + rank)) across lists
# Deduplicate by text, keep highest-scoring metadata on collision
```

Output: up to 10 unique TextNodes sorted by RRF score.

---

### Stage 3: Cross-Encoder Reranking

Input: 10 nodes from RRF
Process:
```python
pairs = [(query, node.text) for node in nodes]
scores = cross_encoder.predict(pairs)  # shape: (10,)
# Sort by score descending, take top rerank_top_k=3
```

Output: 3 TextNodes in order of cross-encoder relevance score.

---

### Stage 4: Prompt Construction

```python
# build_prompt(query, nodes) produces:
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT  # "Answer only from context, cite everything..."
    },
    {
        "role": "user",
        "content": """
Context:
[1] 3M | 2022 | 10-K | page 42
## Revenue from Operations
Net sales for the year ended December 31, 2022 were $35.4 billion, compared with...

[2] 3M | 2022 | 10-K | page 43
## Revenue by Segment
Safety & Industrial segment net sales were $11.2 billion...

[3] 3M | 2022 | 10-K | page 44
## Geographic Revenue
U.S. net sales were $15.1 billion...

Question: What was 3M's net sales in 2022?
"""
    }
]
```

---

### Stage 5: LLM Response

OpenRouter returns:

```python
response = {
    "choices": [{
        "message": {
            "content": "3M's net sales in 2022 were $35.4 billion [1], ..."
        }
    }],
    "usage": {
        "prompt_tokens": 620,
        "completion_tokens": 85,
        "total_tokens": 705
    },
    "model": "stepfun/step-3.5-flash"
}
```

---

### Stage 6: HTTP Response

```json
{
    "answer": "3M's net sales in 2022 were $35.4 billion [1], representing a 2% increase from 2021's $35.2 billion [1].",
    "citations": [
        {
            "index": 1,
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
            "page_number": 42,
            "filename": "3M_2022_10K_10.pdf",
            "excerpt": "Net sales for the year ended December 31, 2022 were $35.4 billion..."
        },
        {
            "index": 2,
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
            "page_number": 43,
            "filename": "3M_2022_10K_10.pdf",
            "excerpt": "Safety & Industrial segment net sales were $11.2 billion..."
        },
        {
            "index": 3,
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
            "page_number": 44,
            "filename": "3M_2022_10K_10.pdf",
            "excerpt": "U.S. net sales were $15.1 billion..."
        }
    ],
    "usage": {
        "prompt_tokens": 620,
        "completion_tokens": 85,
        "total_tokens": 705,
        "cost_usd": 0.0
    },
    "latency_ms": 1843.2,
    "trace_id": "trace-uuid-...",
    "reasoning": {
        "summary": "FinLens answered using 3 sources from 3M 2022 10-K",
        "retrieval": {
            "strategy": "hybrid_bm25_dense_rrf",
            "top_k": 10,
            "filters": {"company": "3M", "year": "2022"},
            "candidates_found": 10
        },
        "reranking": {
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "selected_nodes": 3
        },
        "generation": {
            "model": "openrouter/stepfun/step-3.5-flash:free",
            "cost_usd": 0.0,
            "latency_ms": 1240
        },
        "top_sources": [
            "3M_2022_10K_10.pdf p.42",
            "3M_2022_10K_10.pdf p.43",
            "3M_2022_10K_10.pdf p.44"
        ]
    }
}
```

---

## 3. Error Cases

### No documents indexed

```json
// GET /status/ingestion
{
    "total_chunks": 0,
    "documents": []
}
// POST /chat → 404
{
    "detail": "No relevant context found"
}
```

### Qdrant unreachable

```json
// GET /status/services
{
    "qdrant": {"status": "unhealthy", "error": "Connection refused"},
    "langfuse": {"status": "healthy"}
}
// POST /chat → 500 (Qdrant client throws)
```

### BM25 pickle not found

BM25 retrieval silently returns empty list (graceful degradation). Only Qdrant dense results are used. The system does not return an error — this is arguably a bug, since you'd want to know your BM25 index is missing.

### OpenRouter rate limit (429)

LiteLLM propagates the 429 exception. FastAPI returns 500. The frontend shows an error state. There is no retry logic — a production system would implement exponential backoff.

---

## 4. Langfuse Trace Structure

```
Trace: {
    id: "trace-uuid",
    name: "chat query: What was 3M's net sales?",
    input: { query: "...", company: "3M", year: "2022" },
    output: { answer: "...", citations: [...] },
    metadata: { model: "stepfun/...", latency_ms: 1843 }
}
  ├── Span: "hybrid_retrieve" {
  │     input: { query: "...", company: "3M", year: "2022", top_k: 10 },
  │     output: { nodes_count: 10 },
  │     latency_ms: 45
  │   }
  ├── Span: "rerank" {
  │     input: { nodes_count: 10 },
  │     output: { nodes_count: 3 },
  │     latency_ms: 120
  │   }
  └── Span: "generate" {
        input: { model: "...", messages_count: 2 },
        output: { answer: "...", tokens: 705 },
        latency_ms: 1240
      }
```

---

## 5. Environment Variables Reference

| Variable | Used by | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | `generate.py` | LiteLLM → OpenRouter auth |
| `QDRANT_URL` | `index.py`, `hybrid.py` | Qdrant server address |
| `QDRANT_API_KEY` | same | Qdrant Cloud auth (optional for local) |
| `LANGFUSE_PUBLIC_KEY` | `tracing.py` | Langfuse auth |
| `LANGFUSE_SECRET_KEY` | `tracing.py` | Langfuse auth |
| `LANGFUSE_HOST` | `tracing.py` | Langfuse URL (default: localhost:3000) |
| `EMBED_MODEL_NAME` | `embed.py` | Override default embedding model |
| `DEFAULT_LLM_MODEL` | `generate.py` | Override default LLM |

All loaded via `python-dotenv` from `.env` file (gitignored). `.env.example` is committed as a template.
