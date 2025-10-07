# Multi-Agent Research System - Comprehensive Guide

## Overview

The Multi-Agent Research System is a sophisticated AI-powered research automation platform that delivers comprehensive, high-quality research outputs through coordinated multi-agent workflows. This production-ready MVP system combines advanced orchestration, quality management, and intelligent research automation to provide reliable, scalable research with built-in quality assurance.

**Key Capabilities:**
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
└── agent_logging/  # Comprehensive monitoring and debugging infrastructure
```

### Core Directory Functions

**🎯 core/** - *System Orchestration & Quality Management*
- Advanced orchestrator with gap research coordination
- Quality framework with progressive enhancement
- Error recovery and resilience patterns
- Session lifecycle management

**🤖 agents/** - *Specialized AI Agents*
- **Research Agent**: Multi-source data collection and analysis
- **Report Agent**: Structured report generation and formatting
- **Editorial Agent**: Quality assessment and gap identification
- **Content Cleaner**: AI-powered content enhancement
- **Quality Judge**: Final quality validation and approval

**🔧 tools/** - *High-Level Research Interfaces*
- **Intelligent Research Tool**: Complete z-playground1 methodology implementation
- **Advanced Scraping Tool**: Multi-stage extraction with AI cleaning (35-40s savings per URL)
- **SERP Search Tool**: High-performance Google search (10x improvement over MCP search)

**⚙️ utils/** - *Web Crawling & Content Processing*
- **AI-Powered Content Pipeline**: Raw HTML → AI cleaning → agent consumption
- **4-Level Anti-Bot System**: Basic → Enhanced → Advanced → Stealth escalation
- **Intelligent Search Strategy**: AI-driven search engine selection
- **Media Optimization**: 3-4x performance improvement for media-rich content

**📋 config/** - *System Configuration & Agent Definitions*
- **Agent Definition Architecture**: Claude Agent SDK integration patterns
- **Enhanced Search Configuration**: Anti-bot levels and target-based scraping
- **Settings Management**: Environment variable overrides and validation
- **Quality Framework Configuration**: Customizable quality thresholds

**🔌 mcp_tools/** - *Claude SDK Integration*
- **Multi-Server Architecture**: Enhanced search and zplayground1 servers
- **Adaptive Content Chunking**: Intelligent token management
- **Progressive Anti-Bot Detection**: 4-level escalation system
- **Session-Based Workproduct Management**: Environment-aware path detection

**📊 agent_logging/** - *Monitoring & Debugging Infrastructure*
- **Session-Based Correlation**: Cross-agent tracking with unique session IDs
- **Structured Data Collection**: Detailed metrics for each agent type
- **Hook System Integration**: Tool usage, agent communication, workflow monitoring
- **Export and Analysis**: Built-in data export with session summaries

## Quick Start Guide

### Prerequisites

1. **Python 3.10+** with required dependencies
2. **API Keys** for external services:
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export OPENAI_API_KEY="your-openai-key"
   export SERP_API_KEY="your-serp-key"
   ```
3. **Node.js** for Claude Code CLI (optional for MCP integration)

### Basic Usage

```python
from multi_agent_research_system.core.orchestrator import ResearchOrchestrator

# Initialize the orchestrator
orchestrator = ResearchOrchestrator()

# Run a comprehensive research query
result = await orchestrator.run_research(
    query="Latest developments in quantum computing",
    max_sources=10,
    quality_threshold=0.8
)

# Access the results
print(f"Status: {result.status}")
print(f"Report: {result.final_report_path}")
print(f"Quality Score: {result.quality_score}")
```

### Expected Results

Research outputs are organized in the `KEVIN/` directory:
```
KEVIN/
├── sessions/
│   └── {session_id}/
│       ├── working/
│       │   ├── RESEARCH_*.md
│       │   ├── REPORT_*.md
│       │   ├── EDITORIAL_*.md
│       │   └── FINAL_*.md
│       └── complete/
│           └── FINAL_ENHANCED_*.md
```

## Core Features & Capabilities

### Multi-Agent Workflows

The system implements a sophisticated 4-stage workflow:

1. **Research Stage**: Multi-source data collection using intelligent search strategies
2. **Report Stage**: Structured report generation with professional formatting
3. **Editorial Stage**: Quality assessment and gap identification with progressive enhancement
4. **Quality Enhancement**: AI-powered content improvement and final validation

### Quality Management System

**Progressive Enhancement Pipeline:**
- **Quality Assessment**: Comprehensive quality scoring across multiple dimensions
- **Gap Research**: Intelligent identification of content gaps and missing information
- **Content Enhancement**: AI-powered improvement of clarity, depth, and accuracy
- **Final Validation**: Quality judge approval for production-ready outputs

**Quality Framework Features:**
- Configurable quality thresholds and standards
- Multi-dimensional quality assessment (accuracy, completeness, clarity)
- Progressive enhancement with measurable improvement tracking
- Error recovery and fallback mechanisms

### Advanced Research Capabilities

**Intelligent Search & Discovery:**
- AI-driven search engine selection based on query analysis
- Progressive anti-bot detection with 4-level escalation
- SERP integration with automatic content extraction
- Relevance filtering with domain authority scoring

**AI-Powered Content Processing:**
- GPT-5-nano content cleaning with cleanliness assessment
- Media optimization for 3-4x performance improvement
- Intelligent content chunking and token management
- Automatic source attribution and citation management

### MCP Integration

**Claude Agent SDK Integration:**
- Multi-server architecture with specialized tool servers
- Intelligent token management and content allocation
- Adaptive content chunking for optimal performance
- Comprehensive error handling and fail-fast validation

## Development Guidelines

### System Design Principles

1. **Quality-First Architecture**: Every component includes quality assessment and enhancement
2. **Resilience & Recovery**: Comprehensive error handling and recovery mechanisms
3. **Scalability**: Async-first design with resource management
4. **Observability**: Extensive logging and monitoring capabilities

### Extension Patterns

**Adding New Agents:**
```python
from multi_agent_research_system.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    agent_type = "custom"

    async def process_task(self, task_data):
        # Custom agent implementation
        return await self.execute_with_quality_check(task_data)
```

**Adding New Tools:**
```python
from claude_agent_sdk import tool

@tool("custom_tool", "Custom tool description", {"param": str})
async def custom_tool(args):
    # Tool implementation with MCP compliance
    return {"content": [{"type": "text", "text": "Result"}]}
```

### Testing Approach

- **Unit Tests**: Individual component testing with mocked dependencies
- **Integration Tests**: End-to-end workflow testing
- **Quality Tests**: Quality framework validation and enhancement testing
- **Performance Tests**: Load testing and optimization validation

## Configuration & Customization

### Environment Setup

```bash
# Required API keys
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export SERP_API_KEY="your-serp-key"

# Optional configuration
export LOGFIRE_TOKEN="your-logfire-token"  # For enhanced monitoring
export RESEARCH_QUALITY_THRESHOLD="0.8"    # Default quality threshold
export MAX_SEARCH_RESULTS="10"              # Default search result limit
```

### Agent Configuration

Customize agent behavior through the configuration system:

```python
from multi_agent_research_system.config.settings import SettingsManager

settings = SettingsManager()

# Configure quality thresholds
settings.quality_threshold = 0.9
settings.enhancement_iterations = 3

# Configure search behavior
settings.max_search_results = 15
settings.anti_bot_level = "advanced"
```

### Quality Framework Settings

Adjust quality assessment criteria:

```python
# Quality dimensions weighting
quality_weights = {
    "accuracy": 0.3,
    "completeness": 0.25,
    "clarity": 0.2,
    "depth": 0.15,
    "source_quality": 0.1
}
```

## Usage Examples & Integration

### Basic Research Query

```python
# Simple research query
result = await orchestrator.run_research(
    query="Impact of AI on healthcare",
    max_sources=5
)
```

### Advanced Research with Custom Quality

```python
# High-quality research with custom requirements
result = await orchestrator.run_research(
    query="Quantum computing breakthroughs 2024",
    max_sources=20,
    quality_threshold=0.9,
    enhancement_iterations=5,
    require_peer_reviewed=True
)
```

### Programmatic Integration

```python
from multi_agent_research_system import ResearchSystem

# Initialize system with custom configuration
system = ResearchSystem(
    quality_threshold=0.85,
    max_concurrent_agents=3,
    enable_gap_research=True
)

# Process multiple research queries
queries = [
    "Renewable energy trends",
    "Space exploration developments",
    "Biotechnology innovations"
]

results = await system.process_batch(queries)
```

### MCP Tool Usage

```python
from claude_agent_sdk import ClaudeSDKClient
from multi_agent_research_system.mcp_tools.servers import create_research_server

# Create MCP server with research tools
research_server = create_research_server()

# Use with Claude SDK
options = ClaudeAgentOptions(
    mcp_servers={"research": research_server},
    allowed_tools=["mcp__research__comprehensive_search"]
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Research the latest AI developments")
    # Claude can now use the research tools directly
```

## System Requirements & Dependencies

### Prerequisites

- **Python 3.10+** with async support
- **Node.js 16+** for Claude Code CLI (optional)
- **8GB+ RAM** recommended for optimal performance
- **Stable internet connection** for external API access

### Required Dependencies

```bash
# Core dependencies
pip install claude-agent-sdk
pip install pydantic-ai
pip install crawl4ai
pip install logfire

# Development dependencies
pip install pytest pytest-asyncio
pip install ruff mypy
pip install pre-commit
```

### External Services

- **Anthropic Claude API**: For AI agent interactions
- **OpenAI API**: For content cleaning and enhancement
- **SERP API**: For high-performance search capabilities
- **Logfire**: For enhanced monitoring (optional)

## Monitoring & Debugging

### Logging System

The system provides comprehensive logging infrastructure:

```python
from multi_agent_research_system.agent_logging.logger import AgentLogger

# Initialize logger for your component
logger = AgentLogger("custom_component")

# Log structured data
await logger.log_info("Component started", {
    "session_id": session_id,
    "configuration": config,
    "performance_metrics": metrics
})
```

### Performance Monitoring

Key metrics to monitor:
- **Agent Execution Time**: Track performance of each agent
- **Quality Improvement Scores**: Monitor enhancement effectiveness
- **Error Recovery Rates**: Track system resilience
- **Resource Utilization**: Monitor memory and processing usage

### Debug Tools

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with trace output
python -m logfire install

# Monitor active sessions
python -m multi_agent_research_system.tools.session_monitor
```

### Common Issues & Solutions

**Issue: Slow research processing**
- Solution: Adjust `max_concurrent_agents` and `batch_size` settings
- Check anti-bot level configuration
- Verify API key quotas and rate limits

**Issue: Low quality scores**
- Solution: Increase `quality_threshold` gradually
- Enable additional enhancement iterations
- Check source quality settings

**Issue: Memory usage high**
- Solution: Reduce `max_concurrent_agents`
- Enable content chunking optimization
- Monitor session cleanup processes

## Best Practices & Guidelines

### Research Query Design

- **Specificity**: Use clear, specific queries for better results
- **Scope**: Balance breadth and depth for optimal coverage
- **Keywords**: Include relevant technical terms and industry jargon
- **Timeframes**: Specify date ranges for time-sensitive research

### Quality Optimization

- **Threshold Settings**: Start with moderate thresholds (0.7-0.8) and adjust based on results
- **Enhancement Iterations**: Use 2-3 iterations for balance of quality and speed
- **Source Diversity**: Enable multiple search strategies for comprehensive coverage
- **Validation**: Always review final outputs for critical applications

### Resource Management

- **Concurrent Agents**: Limit to 2-3 agents for optimal performance
- **Session Cleanup**: Regular cleanup of old research sessions
- **API Quotas**: Monitor external API usage and implement rate limiting
- **Storage**: Manage KEVIN directory size with archival policies

### Security Considerations

- **API Key Management**: Store API keys securely using environment variables
- **Data Privacy**: Review data handling policies for sensitive research
- **Access Control**: Implement appropriate access controls for research outputs
- **Audit Trails**: Use logging system for compliance and audit requirements

## Directory Navigation

This root documentation provides an overview of the entire system. For detailed information about specific components, refer to the individual directory documentation:

- **[core/CLAUDE.md](multi_agent_research_system/core/CLAUDE.md)** - Advanced orchestrator and quality management
- **[agents/CLAUDE.md](multi_agent_research_system/agents/CLAUDE.md)** - Specialized AI agents and workflows
- **[tools/CLAUDE.md](multi_agent_research_system/tools/CLAUDE.md)** - High-level research tools and interfaces
- **[utils/CLAUDE.md](multi_agent_research_system/utils/CLAUDE.md)** - Web crawling and content processing utilities
- **[config/CLAUDE.md](multi_agent_research_system/config/CLAUDE.md)** - System configuration and agent definitions
- **[mcp_tools/CLAUDE.md](multi_agent_research_system/mcp_tools/CLAUDE.md)** - Claude SDK integration and MCP tools
- **[agent_logging/CLAUDE.md](multi_agent_research_system/agent_logging/CLAUDE.md)** - Monitoring and debugging infrastructure

## Contributing

### Development Setup

1. **Clone the repository** and install in development mode:
   ```bash
   git clone <repository>
   cd multi-agent-research-system
   pip install -e ".[dev]"
   ```

2. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

3. **Configure environment variables** for development
4. **Run tests** to verify setup:
   ```bash
   pytest tests/
   ```

### Code Quality Standards

- **Code Formatting**: Use ruff for consistent formatting
- **Type Checking**: Enforce type hints with mypy
- **Testing**: Maintain >90% code coverage
- **Documentation**: Update documentation for all changes

### Pre-commit Checklist

- [ ] Code passes `ruff check` and `ruff format`
- [ ] Code passes `mypy multi_agent_research_system/` with no errors
- [ ] All tests pass: `pytest tests/`
- [ ] Documentation updated for affected components
- [ ] Integration tests pass for workflow changes
- [ ] Performance impact assessed and documented

---

**System Status**: Production-ready MVP with enterprise-grade features
**Documentation Version**: Current implementation reflecting actual system state
**Last Updated**: Comprehensive documentation revamp capturing current MVP capabilities