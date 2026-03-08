"""
Milestone M0.6 — Cross-encoder reranker validation.

Run with: cd backend && uv run python ../validate_reranker.py
Requires: sentence-transformers installed (included in backend/pyproject.toml).
"""
from sentence_transformers import CrossEncoder

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Loading cross-encoder: {MODEL} ...")
model = CrossEncoder(MODEL)

query = "What were 3M's total net sales in 2022?"
candidates = [
    "3M Company reported total net sales of $35.4 billion for fiscal year 2022.",  # most relevant
    "The company operates across four business segments worldwide.",                # somewhat relevant
    "3M's stock closed at $118.32 on December 31, 2022.",                         # tangentially related
    "The annual report was filed with the SEC on February 9, 2023.",               # least relevant
]

pairs = [(query, c) for c in candidates]
scores = model.predict(pairs)
ranked = sorted(zip(scores, candidates), reverse=True)

print(f"\nQuery: {query!r}\n")
print("Ranked candidates (highest score = most relevant):")
for rank, (score, text) in enumerate(ranked, 1):
    marker = " <-- BEST" if rank == 1 else ""
    print(f"  #{rank}  score={score:.4f}  {text[:70]!r}{marker}")

best = ranked[0][1]
assert best == candidates[0], f"Expected most relevant candidate to rank #1, got:\n  {best!r}"

print("\nM0.6 PASS: Cross-encoder reranker ranks correctly.")
