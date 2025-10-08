# Suggested Commands - Multi-Agent Research System Development

## Essential Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Configure environment variables (create .env file)
cp .env.example .env
# Edit .env with your API keys
```

### Required Environment Variables
```bash
# Required API Keys
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"  
export SERP_API_KEY="your-serp-key"

# Optional Configuration
export LOGFIRE_TOKEN="your-logfire-token"
export RESEARCH_QUALITY_THRESHOLD="0.8"
export MAX_SEARCH_RESULTS="10"
export MAX_CONCURRENT_AGENTS="3"
export DEBUG_MODE=false
export LOG_LEVEL=INFO
```

### Code Quality Commands
```bash
# Run linting and formatting
ruff check multi_agent_research_system/
ruff format multi_agent_research_system/

# Run type checking
mypy multi_agent_research_system/

# Run all quality checks together
ruff check multi_agent_research_system/ && ruff format multi_agent_research_system/ && mypy multi_agent_research_system/
```

### Testing Commands
```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=multi_agent_research_system --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/functional/

# Run tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_orchestrator.py
```

### Running the System

### Command Line Interface
```bash
# Main research execution
python multi_agent_research_system/run_research.py "your research query"

# Simple research interface
python simple_research.py "your research query"

# Run with specific parameters
python multi_agent_research_system/run_research.py "query" --depth "Comprehensive Analysis" --audience "Academic"

# Run with debug mode
python multi_agent_research_system/run_research.py "query" --debug

# Main system entry point
python multi_agent_research_system/main.py
```

### Web Interface
```bash
# Start Streamlit web interface
python multi_agent_research_system/start_ui.py

# Access at http://localhost:8501
```

### Testing and Debugging Commands
```bash
# Run integration tests
python -m pytest e2e-tests/

# Test specific components
python test_file_operations.py
python test_quality_system_integration.py
python test_config_debug.py

# Debug with logging
export LOG_LEVEL=DEBUG
python multi_agent_research_system/run_research.py "test query"

# Monitor system performance
python multi_agent_research_system/test_monitoring.py
```

### File System Commands (Linux)
```bash
# List files in session directory
ls -la KEVIN/sessions/

# Find specific file types
find KEVIN/sessions/ -name "*.md" | head -10

# Search for content in files
grep -r "specific term" KEVIN/sessions/

# Monitor log files
tail -f KEVIN/logs/research_system.log

# Clean up old sessions (older than 7 days)
find KEVIN/sessions/ -type d -mtime +7 -exec rm -rf {} \;
```

### Git Commands
```bash
# Check status
git status

# Add changes
git add .

# Commit with conventional format
git commit -m "feat: add new research enhancement feature"

# Push changes
git push

# Check recent commits
git log --oneline -10

# Create new branch
git checkout -b feature/new-feature
```

### MCP Server Commands
```bash
# Start MCP server (if configured)
python -m mcp_tools.servers.research_server

# Test MCP tools
python -m pytest tests/test_mcp_integration.py
```

### Monitoring Commands
```bash
# Monitor active sessions
python multi_agent_research_system/tools/session_monitor

# Check system health
python multi_agent_research_system/monitoring/system_health.py

# Analyze performance
python multi_agent_research_system/utils/performance_timers.py
```

### Development Workflow Commands
```bash
# Full development workflow
git status
ruff check multi_agent_research_system/
ruff format multi_agent_research_system/
mypy multi_agent_research_system/
pytest tests/
git add .
git commit -m "feat: implementation with quality checks"
git push

# Quick development cycle
python -m pytest tests/unit/test_orchestrator.py -v
python multi_agent_research_system/run_research.py "test query" --debug
```

### Utility Commands
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(anthropic|claude|pydantic)"

# Check environment variables
env | grep -E "(ANTHROPIC|OPENAI|SERP)"

# Find Python files in project
find . -name "*.py" | grep -v __pycache__ | wc -l

# Check disk usage of KEVIN directory
du -sh KEVIN/
```

### Pre-commit Checklist Commands
```bash
# Run complete pre-commit checks
echo "Running pre-commit checks..."
ruff check multi_agent_research_system/ && echo "✅ Ruff check passed"
ruff format multi_agent_research_system/ && echo "✅ Ruff format passed"
mypy multi_agent_research_system/ && echo "✅ MyPy check passed"
pytest tests/ && echo "✅ Tests passed"
echo "All checks passed! Ready to commit."
```

### Session Management Commands
```bash
# List active sessions
ls KEVIN/sessions/

# View session details
ls -la KEVIN/sessions/35c79207-e852-4a5e-9c56-3a516735e0cc/

# Clean up old sessions
find KEVIN/sessions/ -type d -mtime +30 -exec rm -rf {} \;

# Archive completed sessions
mkdir -p KEVIN/archive/
mv KEVIN/sessions/2025-10-* KEVIN/archive/ 2>/dev/null || true
```

### Troubleshooting Commands
```bash
# Check API key configuration
python -c "import os; print('ANTHROPIC_API_KEY:', 'SET' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET')"
python -c "import os; print('SERP_API_KEY:', 'SET' if os.getenv('SERP_API_KEY') else 'NOT SET')"

# Test import of main modules
python -c "from multi_agent_research_system.core.orchestrator import ResearchOrchestrator; print('✅ Orchestrator import OK')"
python -c "from multi_agent_research_system.agents.research_agent import ResearchAgent; print('✅ ResearchAgent import OK')"

# Check file permissions
ls -la KEVIN/
chmod -R 755 KEVIN/sessions/ 2>/dev/null || true

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

## Common Development Tasks

### Adding a New Agent
1. Create agent file in `multi_agent_research_system/agents/`
2. Inherit from `BaseAgent` class
3. Define tools and message handlers
4. Add agent definition in `config/agents.py`
5. Add tests in `tests/unit/`
6. Update documentation

### Adding a New Tool
1. Define tool function in appropriate `core/` module
2. Use `@tool` decorator with proper schema
3. Add tool to agent configuration in `config/agents.py`
4. Add tests in `tests/unit/test_tools.py`
5. Update tool documentation

### Debugging a Session
1. Get session ID from `ls KEVIN/sessions/`
2. Examine session files: `ls -la KEVIN/sessions/{session_id}/`
3. Check logs: `tail -f KEVIN/logs/research_system.log`
4. Run debug mode: `python multi_agent_research_system/run_research.py "query" --debug`

### Performance Optimization
1. Monitor with: `python multi_agent_research_system/test_monitoring.py`
2. Check memory usage: `ps aux | grep python`
3. Analyze bottlenecks: `python -m cProfile -o profile.stats multi_agent_research_system/run_research.py "query"`
4. Review settings in `config/settings.py`