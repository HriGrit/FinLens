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
