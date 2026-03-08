from __future__ import annotations

import os

import pytest

from generation.generate import generate
from retrieval import hybrid
from retrieval.pipeline import retrieve_and_rerank


@pytest.mark.live
def test_full_rag_query_uses_live_openrouter(
    monkeypatch,
    requires_openrouter,
    seeded_qdrant_collection,
    qdrant_client,
):
    query = "What were 3M's 2022 net sales?"
    monkeypatch.setattr(hybrid, "QDRANT_COLLECTION", seeded_qdrant_collection)

    nodes = retrieve_and_rerank(query, retrieval_top_k=5, rerank_top_k=2, company="3M", year="2022")
    if not nodes:
        pytest.skip("No live retrieval candidates were returned.")

    output = generate(query=query, context_nodes=nodes, model=os.getenv("OPENROUTER_TEST_MODEL", "openrouter/stepfun/step-3.5-flash:free"))

    assert output["answer"]
    assert output["citations"]
