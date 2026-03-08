"""
Milestone M0.5 — LiteLLM -> OpenRouter -> Mistral validation.

Run with: cd backend && uv run python ../validate_llm.py
Requires: OPENROUTER_API_KEY set in .env (copy from .env.example and fill in).
"""
import os

from dotenv import load_dotenv

load_dotenv()

import litellm  # noqa: E402

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not set. Copy .env.example -> .env and fill in values.")

response = litellm.completion(
    model="openrouter/stepfun/step-3.5-flash:free",
    messages=[
        {"role": "user", "content": "What is 2+2? Answer in one word."},
    ],
    api_key=api_key,
    api_base="https://openrouter.ai/api/v1",
)

content = response.choices[0].message.content
usage = response.usage
try:
    cost = litellm.completion_cost(completion_response=response)
except Exception:
    cost = 0.0  # free/unmapped models have no pricing entry

print(f"Response: {content!r}")
print(f"Usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
print(f"Estimated cost: ${cost:.6f} (0.0 = free/unmapped model)")

assert content, "Expected non-empty response content"
assert usage.total_tokens > 0, "Expected non-zero token usage"

print("\nM0.5 PASS: LiteLLM -> OpenRouter -> Mistral works.")
