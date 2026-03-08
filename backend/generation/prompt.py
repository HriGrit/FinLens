"""
prompt.py — System prompt and message builder (version-controlled).

Changing this file triggers the RAGAS eval CI workflow.
"""
from __future__ import annotations

from llama_index.core.schema import TextNode

SYSTEM_PROMPT = """\
You are FinLens, a financial document analysis assistant.

Answer questions using ONLY the provided context passages from SEC filings.
If the answer cannot be found in the context, say "I don't have enough information \
to answer this question from the provided documents."

Rules:
- Cite the specific document (company, year, page) for every factual claim.
- Use exact figures when available; do not round or estimate.
- Do not use prior knowledge outside the provided context.
- Keep answers concise and structured.
"""


def build_prompt(query: str, context_nodes: list[TextNode]) -> list[dict[str, str]]:
    """Construct the messages list for the LLM call."""
    context_blocks: list[str] = []
    for i, node in enumerate(context_nodes, 1):
        meta = node.metadata
        citation = (
            f"{meta.get('company', '?')} {meta.get('year', '?')} "
            f"{meta.get('doc_type', '10-K')} p.{meta.get('page_number', '?')}"
        )
        context_blocks.append(f"[{i}] ({citation})\n{node.text}")

    context_str = "\n\n---\n\n".join(context_blocks)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n\n{context_str}\n\nQuestion: {query}",
        },
    ]
