# Repository Guidelines

## Project Structure & Module Organization
- Core runtime lives in `multi_agent_research_system/` with domain packages such as `agents/`, `core/`, `config/`, and `hooks/`; CLI entrypoints `run_research.py` and `main.py` orchestrate agent workflows.
- Supporting SDK integrations live under `src/claude_agent_sdk/`; keep mirrored changes in sync with upstream SDK docs before modifying.
- Testing assets span root-level `test_*.py`, `tests/` for SDK compatibility, and scenario folders (`integration/`, `e2e-tests/`, `validation_results/`); place new fixtures in `tests/conftest.py`.
- Research output is persisted under `KEVIN/sessions/<id>/`; avoid committing session artifacts unless referenced in documentation.

## Build, Test, and Development Commands
- Install tooling with `pip install -e .[dev]` (requires Python 3.10+) or `uv sync` when using the provided `uv.lock`.
- Run the default agent flow via `python run_research.py "topic"`; `python main_comprehensive_research.py --enhanced-editorial-workflow` exercises the full multi-agent pipeline.
- Execute the suite with `pytest --cov=multi_agent_research_system` and inspect HTML coverage in `htmlcov/index.html`; target ≥90% coverage on touched modules.
- Lint and type-check before pushing: `ruff check .`, `black .`, and `mypy multi_agent_research_system`.

## Coding Style & Naming Conventions
- Use 4-space indentation, dataclasses or Pydantic models for structured payloads, and favor dependency injection via `config/` factories.
- Modules and functions follow snake_case, classes CamelCase, and constants UPPER_SNAKE_CASE; new agents should use `AgentNameAgent`.
- Keep functions under ~40 lines, and document agent entrypoints with docstrings that describe expected message schemas.
- `ruff` enforces import sorting and bugbear rules; run `ruff format` (or `black`) after large edits to satisfy the 88-char line limit.

## Testing Guidelines
- Co-locate fast unit tests with the code in `multi_agent_research_system/tests/`; heavier workflows belong in top-level `tests/` or `integration/`.
- Name files `test_<component>.py` and async tests `async def test_<behavior>` to cooperate with `pytest-asyncio`.
- When adding agents, include golden-path and failure-path tests plus fixtures that mimic session folders under `KEVIN/sessions/`.
- Regenerate coverage reports (`pytest --cov ... --cov-report=xml`) so CI surfaces deltas; attach summaries in PRs for major features.

## Commit & Pull Request Guidelines
- Follow Conventional Commits; recent history includes `feat:`, `docs:`, and `cleanup:` prefixes—keep scopes short and action-oriented.
- Reference issues or ADRs in the commit body, note API or message schema changes, and update `CHANGELOG.md` when behavior shifts.
- PRs must describe the change, the verification steps (commands run, screenshots for any `ui/` updates), and new configuration or env requirements.
- Request reviews from agent-platform maintainers for cross-cutting changes, and wait for green CI before merging via squash.
