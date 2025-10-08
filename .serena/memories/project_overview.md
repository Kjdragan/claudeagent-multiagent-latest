# Multi-Agent Research System - Project Overview

## Project Purpose
The Multi-Agent Research System is a sophisticated AI-powered research automation platform that delivers comprehensive, high-quality research outputs through coordinated multi-agent workflows. This is a production-ready MVP system that combines advanced orchestration, quality management, and intelligent research automation.

## Key Capabilities
- **Multi-Agent Coordination**: Specialized agents working in sophisticated workflows
- **Quality-Gated Processes**: Progressive enhancement with comprehensive quality assessment
- **Advanced Research Automation**: Anti-bot detection, AI content cleaning, intelligent search
- **MCP Integration**: Full Claude Agent SDK integration with intelligent token management
- **Enterprise-Grade Features**: Comprehensive error recovery, logging, and monitoring

## System Architecture

### High-Level Component Structure
```
multi_agent_research_system/
├── core/           # Orchestration, quality management, error recovery
├── agents/         # Specialized AI agents (research, report, editorial, quality)
├── tools/          # High-level research tools and search interfaces
├── utils/          # Web crawling, content processing, anti-bot detection
├── config/         # Agent definitions and system configuration
├── mcp_tools/      # Claude SDK integration with intelligent token management
├── agent_logging/  # Comprehensive monitoring and debugging infrastructure
├── ui/             # Streamlit web interface
└── KEVIN/          # Data storage and session management directory
```

### Core Components

**core/** - System Orchestration & Quality Management
- Advanced orchestrator with gap research coordination
- Quality framework with progressive enhancement
- Error recovery and resilience patterns
- Session lifecycle management

**agents/** - Specialized AI Agents
- Research Agent: Multi-source data collection and analysis
- Report Agent: Structured report generation and formatting
- Editorial Agent: Quality assessment and gap identification
- Content Cleaner: AI-powered content enhancement
- Quality Judge: Final quality validation and approval

**tools/** - High-Level Research Interfaces
- Intelligent Research Tool: Complete z-playground1 methodology implementation
- Advanced Scraping Tool: Multi-stage extraction with AI cleaning
- SERP Search Tool: High-performance Google search

**utils/** - Web Crawling & Content Processing
- AI-Powered Content Pipeline: Raw HTML → AI cleaning → agent consumption
- 4-Level Anti-Bot System: Basic → Enhanced → Advanced → Stealth escalation
- Intelligent Search Strategy: AI-driven search engine selection
- Media Optimization: 3-4x performance improvement

## Tech Stack

### Core Technologies
- **Python 3.10+** with async/await patterns
- **Claude Agent SDK (v0.1.1)** for agent management
- **Pydantic AI (v1.0.2)** for AI agent integration
- **MCP (Model Context Protocol)** for tool integration
- **Crawl4AI (v0.7.4)** for web crawling
- **Playwright (v1.55.0)** for browser automation

### Key Dependencies
- **anthropic>=0.69.0** - Claude API integration
- **httpx>=0.28.1** - Async HTTP client
- **psutil>=7.1.0** - System monitoring
- **logfire>=0.1.0** - Observability and logging
- **streamlit>=1.50.0** - Web UI interface

### Development Tools
- **pytest>=8.4.2** with pytest-asyncio for testing
- **ruff>=0.13.2** for linting and formatting
- **mypy>=1.18.2** for type checking
- **uv** for package management

## Workflow Process

### 4-Stage Research Workflow
1. **Research Stage**: Multi-source data collection using intelligent search strategies
2. **Report Stage**: Structured report generation with professional formatting
3. **Editorial Stage**: Quality assessment and gap identification with progressive enhancement
4. **Quality Enhancement**: AI-powered content improvement and final validation

### Multi-Agent Coordination
- **Quality-Gated Workflows**: Progressive enhancement with quality assessment
- **Gap Research Coordination**: Intelligent control handoff for additional research
- **Error Recovery**: Comprehensive resilience patterns with checkpointing
- **Session Management**: Full lifecycle tracking with persistence

## Entry Points

### Command Line Interface
```bash
# Main research execution
python multi_agent_research_system/run_research.py "query"

# Simple research interface
python simple_research.py "query"

# Web interface
python multi_agent_research_system/start_ui.py

# Main system entry point
python multi_agent_research_system/main.py
```

### Programmatic Interface
```python
from multi_agent_research_system.core.orchestrator import ResearchOrchestrator

orchestrator = ResearchOrchestrator()
result = await orchestrator.run_research(
    query="Latest developments in quantum computing",
    max_sources=10,
    quality_threshold=0.8
)
```

## Data Organization

### KEVIN Directory Structure
```
KEVIN/
├── sessions/
│   └── {session_id}/
│       ├── working/          # Work-in-progress files
│       │   ├── RESEARCH_*.md
│       │   ├── REPORT_*.md
│       │   ├── EDITORIAL_*.md
│       │   └── FINAL_*.md
│       └── complete/         # Final outputs
│           └── FINAL_ENHANCED_*.md
├── logs/                    # System logs
├── work_products/          # Research work products
└── temp/                   # Temporary files
```

## Environment Requirements

### Required API Keys
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export SERP_API_KEY="your-serp-key"
```

### Optional Configuration
```bash
export LOGFIRE_TOKEN="your-logfire-token"
export RESEARCH_QUALITY_THRESHOLD="0.8"
export MAX_SEARCH_RESULTS="10"
```

## Development Status
- **System Status**: Production-ready MVP with enterprise-grade features
- **Documentation**: Comprehensive documentation system with CLAUDE.md files
- **Testing**: Unit, integration, functional, and performance tests
- **Quality**: Type hints, linting, and comprehensive error handling
- **Monitoring**: Built-in logging, performance tracking, and debugging tools