# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent Logging System Architecture

This directory contains a comprehensive logging infrastructure for the multi-agent research system, providing structured monitoring and analysis capabilities for all agent activities.

### Core Components

#### Base Infrastructure
- **`base.py`** - Base `AgentLogger` class that all specialized loggers inherit from, providing common session management, activity tracking, and export capabilities
- **`structured_logger.py`** - Core structured logging system with JSON formatting, correlation IDs, and workflow tracking
- **`__init__.py`** - Module exports and factory functions for easy logger creation

#### Specialized Loggers
- **`agent_logger.py`** - Generic agent-specific logger for initialization, health checks, queries, tool usage, errors, and performance metrics
- **`agent_loggers.py`** - Specialized loggers for each agent type:
  - `ResearchAgentLogger` - Search execution, source analysis, data extraction, research synthesis
  - `ReportAgentLogger` - Content generation, section progress, citation processing, report structure
  - `EditorAgentLogger` - Review initiation, quality assessment, fact-checking, feedback generation
  - `UICoordinatorLogger` - Workflow management, user interactions, agent handoffs, progress monitoring
- **`hook_logger.py`** - Hook execution monitoring including:
  - `ToolUseLogger` - Pre/post tool use monitoring, execution tracking, blocked operations
  - `AgentCommunicationLogger` - Message passing, handoffs, prompt submissions
  - `SessionLifecycleLogger` - Session creation, resumption, pause, termination, error handling
  - `WorkflowLogger` - Stage progression, decision points, workflow errors

### Key Architectural Patterns

#### Session Management
All loggers support session-based tracking with unique IDs:
```python
logger = ResearchAgentLogger(session_id="unique-session-id")
# All activities are correlated to this session
logger.log_search_initiation(query="AI trends", session_id="unique-session-id")
```

#### Structured Data Collection
Each logger maintains specialized metrics and history:
- Search history with relevance scores and success rates
- Quality assessments with normalized scoring
- Performance metrics with execution times
- User interaction tracking with satisfaction indicators

#### Export and Analysis
Built-in data export capabilities:
```python
# Export session data for analysis
file_path = logger.export_session_data()
summary = logger.get_session_summary()
```

## Development Commands

### Code Quality
```bash
# Lint and format (run from project root)
python -m ruff check multi_agent_research_system/agent_logging/ --fix
python -m ruff format multi_agent_research_system/agent_logging/

# Type checking
python -m mypy multi_agent_research_system/agent_logging/
```

### Testing
```bash
# Run tests (from project root)
python -m pytest multi_agent_research_system/agent_logging/tests/ -v

# Run with coverage
python -m pytest multi_agent_research_system/agent_logging/tests/ --cov=multi_agent_research_system.agent_logging
```

## Usage Patterns

### Creating Agent Loggers
```python
from multi_agent_research_system.agent_logging import (
    ResearchAgentLogger,
    ReportAgentLogger,
    create_agent_logger
)

# Direct instantiation
research_logger = ResearchAgentLogger(session_id="session-123")

# Factory pattern
logger = create_agent_logger("research_agent", session_id="session-123")
```

### Structured Logging
```python
from multi_agent_research_system.agent_logging import get_logger

# Get structured logger
logger = get_logger("my_component", log_dir=Path("logs"))

# Log with structured data
logger.info("Search completed",
           event_type="search_complete",
           query="artificial intelligence",
           results_count=25,
           execution_time=2.5)
```

### Hook Integration
```python
from multi_agent_research_system.agent_logging import ToolUseLogger

tool_logger = ToolUseLogger()

# Log tool usage
tool_logger.log_pre_tool_use(
    tool_name="web_search",
    tool_use_id="tool-123",
    session_id="session-456",
    input_data={"query": "AI trends"},
    agent_context={"agent_name": "research_agent", "agent_type": "research"}
)
```

## Configuration

### Log Directory Structure
```
logs/
├── research_agent/
│   ├── activities_session-id.jsonl
│   └── research_agent_logger.json
├── report_agent/
│   ├── activities_session-id.jsonl
│   └── report_agent_logger.json
├── hook.tool_use_monitor/
│   └── tool_use_monitor.json
└── multi_agent_system.json
```

### Log Levels and Formats
- **JSON Structure**: Timestamp, level, logger, message, correlation ID, event type, metadata
- **Console Output**: Human-readable format for development
- **File Rotation**: Automatic log rotation with size limits (10MB default, 5 backups)

## Performance Considerations

### Memory Management
- Activity history is maintained in memory for session duration
- Automatic cleanup on session termination
- Export capabilities for long-term storage

### Async Safety
- All logging operations are thread-safe
- Correlation ID context management for async workflows
- Non-blocking file operations where possible

## Integration Examples

### Multi-Agent Workflow Logging
```python
# Research phase
research_logger = ResearchAgentLogger(session_id)
research_logger.log_search_initiation(query, params, topic, estimated_results)
research_logger.log_search_results(search_id, count, results, scores, duration)

# Report generation phase
report_logger = ReportAgentLogger(session_id)
report_logger.log_section_generation_start(section_name, section_type, sources_count, target_words)
report_logger.log_section_generation_complete(section_name, word_count, gen_time, citations, coherence, quality)

# Editorial review phase
editor_logger = EditorAgentLogger(session_id)
editor_logger.log_review_initiation(title, document_type, word_count, focus_areas)
editor_logger.log_quality_assessment(review_id, category, score, max_score, issues, strengths)
```

### Hook System Integration
```python
# Tool use monitoring
tool_logger = ToolUseLogger()
tool_logger.log_pre_tool_use(tool_name, tool_use_id, session_id, input_data, agent_context)
# ... tool execution ...
tool_logger.log_post_tool_use(tool_name, tool_use_id, session_id, input_data, result, exec_time, success, agent_context)

# Session lifecycle monitoring
session_logger = SessionLifecycleLogger()
session_logger.log_session_creation(session_id, config, user_context)
session_logger.log_session_termination(session_id, reason, final_state)
```

## Best Practices

### Logger Usage
1. Use appropriate logger types for each agent
2. Include structured metadata for better analysis
3. Use session IDs for correlation tracking
4. Export session data for long-term analysis

### Performance Optimization
1. Monitor memory usage with long-running sessions
2. Use appropriate log levels to reduce noise
3. Implement cleanup strategies for completed sessions
4. Consider batch logging for high-frequency events

### Error Handling
1. Log errors with full context information
2. Include correlation IDs for debugging
3. Use structured error data for analysis
4. Implement error recovery logging