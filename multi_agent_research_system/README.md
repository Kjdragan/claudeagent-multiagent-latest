# Multi-Agent Research System

A comprehensive research workflow system built with the Claude Agent SDK that coordinates multiple specialized AI agents to conduct research, generate reports, and provide editorial review.

## Features

- **Multi-Agent Coordination**: Research, Report Generation, and Editorial agents working in sequence
- **Streamlit Web Interface**: User-friendly interface for managing research projects
- **Session Management**: Track progress and maintain state across research workflows
- **File Management**: Automatic saving of research materials and reports
- **Quality Control**: Built-in editorial review and revision cycles

## System Architecture

### Agents
1. **Research Agent**: Conducts comprehensive web research and source validation
2. **Report Agent**: Transforms research findings into structured reports
3. **Editor Agent**: Reviews reports for quality, accuracy, and completeness
4. **UI Coordinator**: Manages workflow orchestration and user interaction

### Workflow Stages
1. **Research**: Comprehensive information gathering and source validation
2. **Report Generation**: Creating structured reports from research findings
3. **Editorial Review**: Quality assessment and feedback generation
4. **Finalization**: Revisions and final report delivery

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface
```bash
cd ui
streamlit run streamlit_app.py
```

### Programmatic Usage
```python
from multi_agent_research_system import ResearchOrchestrator

# Initialize the orchestrator
orchestrator = ResearchOrchestrator()
await orchestrator.initialize()

# Start a research session
session_id = await orchestrator.start_research_session(
    topic="Artificial Intelligence in Healthcare",
    user_requirements={
        "depth": "Comprehensive Analysis",
        "audience": "Technical",
        "format": "Technical Documentation"
    }
)
```

## File Structure

```
multi_agent_research_system/
├── core/
│   ├── orchestrator.py      # Main workflow coordination
│   └── research_tools.py    # Custom tools for agents
├── config/
│   └── agents.py           # Agent definitions
├── ui/
│   └── streamlit_app.py    # Web interface
└── researchmaterials/
    └── sessions/           # Research session data
```

## Configuration

Agents are configured using the Claude Agent SDK pattern with `AgentDefinition` objects. Each agent has:

- **System Prompt**: Defines the agent's role and behavior
- **Tools**: Available tools and capabilities
- **Model**: Claude model to use (sonnet, haiku, etc.)

## Development

To run linting and type checking:

```bash
# Lint and format
python -m ruff check . --fix
python -m ruff format .

# Type checking
python -m mypy .
```

## License

This project is part of the Claude Agent SDK ecosystem.