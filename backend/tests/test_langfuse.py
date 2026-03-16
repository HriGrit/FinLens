"""
test_langfuse.py — Integration smoke test for Langfuse end-to-end tracing.

Requires:
  - Langfuse running at http://localhost:3000
  - Qdrant running at http://localhost:6333 with indexed corpus
  - .env with LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, OPENROUTER_API_KEY

Run with:
  cd backend && uv run pytest tests/test_langfuse.py -v -m integration
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_rag_trace_visible_in_langfuse() -> None:
    """Full RAG call produces a trace named 'rag_query' visible via Langfuse SDK."""
    from dotenv import load_dotenv
    load_dotenv()

    from observability.tracing import create_trace, flush
    from retrieval.pipeline import retrieve_and_rerank
    from generation.generate import generate

    query = "What was 3M's total revenue in 2022?"

    trace = create_trace("rag_query", metadata={"company": "3M", "year": "2022"})
    retrieval = retrieve_and_rerank(query, company="3M", year="2022", trace=trace)
    nodes = retrieval.nodes
    assert nodes, "No nodes returned — is the corpus indexed?"

    result = generate(query=query, context_nodes=nodes, trace=trace)
    trace.update(output=result["answer"])

    flush()

    # Verify the trace exists via the Langfuse v2 SDK
    import os
    from langfuse import Langfuse
    lf = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
    )

    response = lf.fetch_traces(name="rag_query", limit=5)
    assert response.data, "No traces found with name 'rag_query' in Langfuse"

    latest = response.data[0]
    assert latest.output is not None, "Trace output is null — trace.update() may have failed"
