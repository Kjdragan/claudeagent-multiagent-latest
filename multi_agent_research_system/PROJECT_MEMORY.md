# Project Memory & Guidelines

## 🚨 IMPORTANT: Package Management

**ALWAYS use UV Package Manager for this project - NEVER use pip install**

### Installing Dependencies
```bash
# ✅ CORRECT - Use uv add for new dependencies
uv add package-name

# ✅ CORRECT - Use uv add --dev for development dependencies
uv add --dev pytest

# ❌ NEVER use pip install
pip install package-name  # DON'T DO THIS
```

### Why We Use UV
- Faster dependency resolution
- Better virtual environment management
- Consistent with our pyproject.toml setup
- Maintains clean .venv structure

### Managing Dependencies
- All dependencies go into pyproject.toml
- Use `uv sync` to ensure all dependencies are installed
- Use `uv add` for new packages, `uv remove` to remove packages
- Development dependencies use `--dev` flag

### Virtual Environment
- UV automatically manages .venv
- Never manually activate/deactivate venv
- Use `uv run` to execute commands in the venv

## Project Architecture

This is a Multi-Agent Research System built with the Claude Agent SDK.

### Core Components
- **Orchestrator**: Manages workflow coordination
- **Agents**: Research, Report, Editor, UI Coordinator
- **Tools**: Web search, analysis, report generation
- **UI**: Streamlit web interface

### Testing Strategy
- Unit tests for individual components
- Integration tests for workflow
- Functional tests with real API calls

### Running the System
1. Web Interface: `uv run streamlit run ui/streamlit_app.py`
2. Testing: `uv run pytest tests/`
3. Examples: `uv run python example_usage.py`

---

**Remember: UV first, never pip!** 🚀