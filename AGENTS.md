# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python workspace for document parsing and extraction experiments.

- `main.py`: current entry point for Docling/LlamaIndex workflows.
- `data/`: place input datasets and intermediate artifacts.
- `3M_2022_10K*.pdf`: sample source documents used by the pipeline.
- `output.md`: generated markdown output from conversion steps.
- `pyproject.toml` and `uv.lock`: dependency and environment definitions.

As the codebase grows, prefer moving reusable logic into a package directory (for example, `finops/`) and keep `main.py` as a thin runner.

## Build, Test, and Development Commands
- `uv sync`: install/update dependencies from `pyproject.toml`/`uv.lock`.
- `uv run python main.py`: run the current extraction/conversion workflow.
- `uv run python -m pip list`: inspect installed runtime packages.

If you add tooling, expose it through `uv run ...` so commands stay consistent across environments.

## Coding Style & Naming Conventions
- Target Python `>=3.12` (see `pyproject.toml`).
- Use 4-space indentation and PEP 8 naming:
  - `snake_case` for functions/variables
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants
- Keep functions focused and side effects explicit (file writes, prints, network/API calls).
- Add type hints for new/edited functions where practical.

## Testing Guidelines
No test framework is configured yet. For new logic, add `pytest` tests under `tests/` with names like `test_<feature>.py`.

- Run tests with `uv run pytest` once `pytest` is added.
- Prefer deterministic unit tests over PDF-heavy end-to-end tests.
- For parsing changes, include at least one regression test for expected extracted text/metadata.

## Commit & Pull Request Guidelines
This branch currently has no commit history, so use the following baseline:

- Commit messages: imperative, concise subject line (for example, `Add table node filtering`).
- Keep commits scoped to one logical change.
- PRs should include:
  - What changed and why
  - How it was validated (commands run)
  - Sample output diffs when extraction behavior changes

## Security & Configuration Tips
- Do not commit secrets or API keys.
- Keep large raw documents in `data/` and avoid unnecessary duplicates at repo root.
- Validate file paths and input file types before processing untrusted documents.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
