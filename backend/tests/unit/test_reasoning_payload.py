from __future__ import annotations

from llama_index.core.schema import TextNode

from api.main import ChatRequest, _build_reasoning


def test_build_reasoning_payload_includes_summary_and_sources() -> None:
    request = ChatRequest(
        query="What was net sales?",
        company="3M",
        year="2022",
        retrieval_top_k=15,
        rerank_top_k=3,
    )
    nodes = [
        TextNode(
            text="Chunk 1",
            metadata={
                "filename": "3M_2022_10K.pdf",
                "page_number": 10,
                "company": "3M",
                "year": "2022",
            },
        ),
        TextNode(
            text="Chunk 2",
            metadata={
                "filename": "3M_2022_10K.pdf",
                "page_number": 11,
                "company": "3M",
                "year": "2022",
            },
        ),
    ]

    payload = _build_reasoning(
        query=request.query,
        trace_id="trace-123",
        request=request,
        context_nodes=nodes,
        n_candidates=20,
        generation_model="mock",
        usage={"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11, "cost_usd": 0.01},
        retrieval_ms=40,
        generation_ms=80,
    )

    assert payload["summary"].startswith("FinLens answered 'What was net sales?'")
    assert payload["retrieval"]["retrieved_candidates"] == 20
    assert payload["retrieval"]["retrieval_top_k"] == 15
    assert payload["rerank"]["selected_nodes"] == 2
    assert payload["generation"]["model"] == "mock"
    assert payload["generation"]["cost_usd"] == 0.01
    assert payload["rerank"]["top_sources"][1]["page_number"] == 11
