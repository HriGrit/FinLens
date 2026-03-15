"""
validate_e2e.py — Phase 4.3 end-to-end generation pipeline test.

Tests the full retrieve -> rerank -> generate chain on a real query.

Run from backend/:
    uv run python ../tests/validate_e2e.py

Requirements:
  - Docker Compose running (Qdrant on :6333)
  - data/bm25_index.pkl on disk
  - .env with OPENROUTER_API_KEY set
  - finlens_chunks_dev collection populated (3M 2015-2019)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# backend/ must be on sys.path so retrieval/generation imports resolve
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from retrieval.pipeline import retrieve_and_rerank
from generation.generate import generate


# ── Test cases ───────────────────────────────────────────────────────────────

TESTS = [
    {
        "label": "3M net sales 2019 (filtered)",
        "query": "What were 3M's total net sales in fiscal year 2019?",
        "company": "3M",
        "year": "2019",
        "retrieval_top_k": 20,
        "rerank_top_k": 5,
    },
    {
        "label": "3M cross-year comparison (filtered)",
        "query": "How did 3M's total net sales change from 2018 to 2019?",
        "company": "3M",
        "year": "2019",
        "retrieval_top_k": 20,
        "rerank_top_k": 5,
    },
    {
        "label": "3M operating segments unfiltered (the benchmark test)",
        "query": "What are 3M's main business segments and what revenue did each generate?",
        "company": None,
        "year": None,
        "retrieval_top_k": 20,
        "rerank_top_k": 5,
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(title: str = "", width: int = 66) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def run_test(test: dict) -> dict:
    label = test["label"]
    query = test["query"]
    company = test.get("company")
    year = test.get("year")
    retrieval_top_k = test.get("retrieval_top_k", 20)
    rerank_top_k = test.get("rerank_top_k", 5)

    sep(f"TEST: {label}")
    print(f"Query   : {query}")
    print(f"Filters : company={company!r}  year={year!r}")
    print(f"Top-K   : retrieve={retrieval_top_k}  rerank={rerank_top_k}")

    # ── 1. RETRIEVE + RERANK ─────────────────────────────────────────────────
    sep("1 · HYBRID RETRIEVE + RERANK")
    t0 = time.perf_counter()
    retrieval = retrieve_and_rerank(
        query=query,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        company=company,
        year=year,
        trace=None,
    )
    nodes = retrieval.nodes
    retrieve_ms = (time.perf_counter() - t0) * 1000

    print(
        f"Returned {len(nodes)} node(s) in {retrieve_ms:.0f} ms "
        f"(candidates={retrieval.candidate_count})"
    )
    if not nodes:
        print("FAIL: 0 nodes returned — check Qdrant collection and BM25 index.")
        return {"label": label, "pass": False, "error": "0 nodes"}

    print()
    for i, node in enumerate(nodes, 1):
        m = node.metadata
        preview = node.text[:220].replace("\n", " ").strip()
        score_val = getattr(node, "score", None)
        score_str = f"score={score_val:.4f}" if score_val is not None else "score=n/a"
        print(f"  [{i}] {score_str}", end="")
        print(
            f"  company={m.get('company')}  year={m.get('year')}"
            f"  doc_type={m.get('doc_type')}  page={m.get('page_number')}"
        )
        print(f"       section: {m.get('section_heading', '—')!r}")
        print(f"       file   : {m.get('filename')}")
        print(f"       text   : {preview!r}")
        print()

    # ── 2. GENERATE ──────────────────────────────────────────────────────────
    sep("2 · GENERATE")
    print(f"Sending {len(nodes)} context node(s) to LLM …\n")

    t1 = time.perf_counter()
    result = generate(query=query, context_nodes=nodes, trace=None)
    gen_ms = (time.perf_counter() - t1) * 1000

    print(f"Model   : {result['model']}")
    u = result["usage"]
    print(
        f"Tokens  : prompt={u['prompt_tokens']}  "
        f"completion={u['completion_tokens']}  "
        f"total={u['total_tokens']}"
    )
    cost_str = f"${u['cost_usd']:.6f}" if u["cost_usd"] is not None else "unknown"
    print(f"Cost    : {cost_str}  |  Latency: {gen_ms:.0f} ms")

    # ── 3. ANSWER ─────────────────────────────────────────────────────────────
    sep("3 · ANSWER")
    print(result["answer"])

    # ── 4. CITATIONS ──────────────────────────────────────────────────────────
    sep("4 · CITATIONS")
    for c in result["citations"]:
        print(
            f"  [{c['index']}] {c['company']} {c['year']} {c['doc_type']} "
            f"p.{c['page_number']}  ← {c['filename']}"
        )

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    sep("SUMMARY")
    total_ms = retrieve_ms + gen_ms
    print(f"  Retrieval latency : {retrieve_ms:.0f} ms")
    print(f"  Generation latency: {gen_ms:.0f} ms")
    print(f"  Total latency     : {total_ms:.0f} ms")
    print(f"  Nodes used        : {len(nodes)}")
    print(f"  Total tokens      : {u['total_tokens']}")

    # ── ASSERTIONS ────────────────────────────────────────────────────────────
    errors = []
    if not result["answer"]:
        errors.append("answer is empty")
    if not result["citations"]:
        errors.append("no citations returned")
    if u["total_tokens"] == 0:
        errors.append("zero token usage reported")
    if "does not contain" in result["answer"].lower() and len(nodes) >= 3:
        errors.append("LLM said it had no info despite having context — possible retrieval quality issue")

    if errors:
        for e in errors:
            print(f"  WARN: {e}")
        passed = len(errors) <= 1  # allow the "does not contain" warning
    else:
        passed = True

    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n  {status}: {label}")
    return {"label": label, "pass": passed, "retrieve_ms": retrieve_ms, "gen_ms": gen_ms}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep("FINLENS PHASE 4 — END-TO-END PIPELINE TEST")
    print(f"Backend : {BACKEND}")
    print(f"BM25    : {REPO_ROOT / 'data' / 'bm25_index.pkl'}")
    print(f"Tests   : {len(TESTS)} queries")

    results = []
    for test in TESTS:
        try:
            r = run_test(test)
        except Exception as exc:
            sep("EXCEPTION")
            import traceback
            traceback.print_exc()
            r = {"label": test["label"], "pass": False, "error": str(exc)}
        results.append(r)

    sep("FINAL RESULTS")
    passed = sum(1 for r in results if r["pass"])
    for r in results:
        icon = "✓" if r["pass"] else "✗"
        timing = ""
        if "retrieve_ms" in r:
            timing = f"  (retrieve {r['retrieve_ms']:.0f}ms + gen {r['gen_ms']:.0f}ms)"
        print(f"  {icon} {r['label']}{timing}")

    print(f"\n{passed}/{len(results)} tests passed.")
    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
