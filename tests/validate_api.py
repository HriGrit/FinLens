"""
validate_api.py — Integration test for the FastAPI endpoints.

Run with: cd backend && uv run python ../tests/validate_api.py
Requires: API server running (uvicorn api.main:app --reload from backend/)
          Qdrant running and ingestion already completed.
"""
import sys

import httpx

BASE_URL = "http://localhost:8000"


def check_health() -> None:
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    assert r.json()["status"] == "ok"
    print(f"GET /health -> {r.json()}")


def check_chat() -> None:
    payload = {
        "query": "What were 3M's total net sales in 2022?",
        "company": "3M",
        "year": "2022",
        "rerank_top_k": 3,
    }
    r = httpx.post(f"{BASE_URL}/chat", json=payload, timeout=60)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    body = r.json()
    assert body["answer"], "Expected non-empty answer"
    assert isinstance(body["citations"], list), "Expected citations list"
    assert body["usage"]["total_tokens"] > 0, "Expected non-zero token usage"

    print(f"POST /chat -> answer snippet: {body['answer'][:120]!r}")
    print(f"           -> citations: {len(body['citations'])}")
    print(f"           -> tokens: {body['usage']['total_tokens']}  cost: ${body['usage']['cost_usd']:.6f}")


def main() -> None:
    failed = []

    for name, fn in [("GET /health", check_health), ("POST /chat", check_chat)]:
        try:
            fn()
            print(f"PASS: {name}")
        except Exception as e:
            print(f"FAIL: {name} — {e}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} test(s) failed: {failed}")
        sys.exit(1)
    else:
        print("\nAll API endpoint tests passed.")


if __name__ == "__main__":
    main()
