# Repository Guidelines

This guide orients contributors to the agent research stack, highlighting where core components live and how to keep changes consistent with team practices.

## Project Structure & Module Organization
- Runtime logic sits in `multi_agent_research_system/` with domain packages like `agents/`, `core/`, `config/`, and `hooks/`. CLI drivers `run_research.py`, `run_enhanced_research.py`, and `main_comprehensive_research.py` orchestrate workflows.
- Supporting SDK integrations reside in `src/claude_agent_sdk/`; confirm deviations against upstream SDK docs before committing.
- Test assets include root `test_*.py` helpers, targeted suites in `tests/`, and scenario folders (`integration/`, `e2e-tests/`, `validation_results/`). Place new fixtures in `tests/conftest.py`.
- Research session artifacts live under `KEVIN/sessions/<id>/`. Keep them out of commits unless cited in documentation.

## Build, Test, and Development Commands
- `pip install -e .[dev]` or `uv sync` prepares the virtualenv (Python 3.10+).
- `python run_research.py "topic"` starts the standard agent flow; use `python main_comprehensive_research.py --enhanced-editorial-workflow` for the full multi-agent pipeline.
- `pytest --cov=multi_agent_research_system` enforces ≥90 % coverage on touched modules; view reports via `htmlcov/index.html`.
- `ruff check .`, `black .`, and `mypy multi_agent_research_system` keep style, formatting, and types aligned before PRs.

## Coding Style & Naming Conventions
- Follow 4-space indentation, favor dataclasses or Pydantic models for structured payloads, and inject dependencies through `config/` factories.
- Modules, functions, and variables use snake_case; classes employ CamelCase; constants are UPPER_SNAKE_CASE; new agents follow `AgentNameAgent`.
- Document agent entrypoints with docstrings outlining expected message schemas, and keep functions under ~40 lines. Run `ruff format` if reflowing large sections.

## Testing Guidelines
- Unit tests co-locate in `multi_agent_research_system/tests/`; integration or SDK compatibility cases belong in `tests/` or `integration/`.
- Name tests `test_<component>.py`; async tests should be `async def test_<behavior>` for `pytest-asyncio`.
- Include golden-path and failure-path scenarios for new agents, mirroring `KEVIN/sessions/` structures when fixtures are required. Regenerate coverage via `pytest --cov ... --cov-report=xml` ahead of review.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (e.g., `feat: agent logging hook`, `docs: update SDK setup`); keep scopes concise.
- Reference issues or ADRs in commit bodies, note schema or API shifts, and update `CHANGELOG.md` when behavior changes.
- Pull requests must state intent, validation commands, and any configuration updates. Solicit reviews from agent-platform maintainers for cross-cutting work and wait for green CI before squashing.

## Agent-Specific Practices
- Reuse lifecycle hooks in `hooks/` rather than duplicating logic, and register new behaviors through existing config factories.
