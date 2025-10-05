# Multi-Agent Research System - Main Directory

This directory serves as the root of the multi-agent research system, providing entry points, coordination logic, and system-level utilities.

## Directory Purpose

The multi_agent_research_system directory is the main system container that orchestrates sophisticated AI-powered research workflows using multiple specialized agents. It provides comprehensive research capabilities through web search, content analysis, report generation, and quality enhancement.

## Key Components

### System Entry Points
- **`main.py`** - Main CLI entry point with logging configuration and demo functionality
- **`run_research.py`** - Primary research execution script with CLI interface
- **`start_ui.py`** - Streamlit web interface launcher

### Core System Directories
- **`utils/`** - Web crawling, content processing, and research utilities
- **`tools/`** - High-level research tools and search interfaces
- **`mcp_tools/`** - Model Context Protocol implementations for Claude integration
- **`core/`** - System orchestration and foundational components
- **`config/`** - Agent definitions and system configuration
- **`agents/`** - Specialized AI agent implementations
- **`tests/`** - Comprehensive test suite

### Supporting Infrastructure
- **`KEVIN/`** - Data storage and session management directory
- **`monitoring/`** - System monitoring and performance tracking
- **`hooks/`** - System hooks and extension points
- **`ui/`** - User interface components

## System Architecture

### Multi-Agent Workflow
```
User Request → Research Agent → Report Agent → Editorial Agent → Quality Enhancement → Final Output
```

### Core Components Interaction
```
Orchestrator
├── Agent Management (research, report, editorial, quality)
├── MCP Server (tool integration)
├── Quality Framework (assessment and enhancement)
├── Session Management (state tracking)
└── Error Recovery (resilience mechanisms)
```

### Data Flow Architecture
```
Input → Search & Research → Content Processing → Report Generation → Quality Enhancement → Output
```

## Quick Start

### Basic Usage
```bash
# Run research with default settings
python multi_agent_research_system/run_research.py "your research topic"

# Run with specific depth and audience
python multi_agent_research_system/run_research.py "topic" --depth "Comprehensive Analysis" --audience "Academic"

# Run with debug mode
python multi_agent_research_system/run_research.py "topic" --debug
```

### Development Mode
```bash
# Run main system with debugging
python multi_agent_research_system/main.py

# Start web interface
python multi_agent_research_system/start_ui.py
```

### Testing
```bash
# Run all tests
python multi_agent_research_system/tests/run_tests.py

# Run specific test categories
python multi_agent_research_system/tests/run_tests.py --category integration
python multi_agent_research_system/tests/run_tests.py --category functional
```

## Development Guidelines

### System Design Principles
1. **Agent Specialization**: Each agent has distinct responsibilities and expertise
2. **Quality-First**: Built-in quality assessment and enhancement at every stage
3. **Resilience**: Comprehensive error handling and recovery mechanisms
4. **Scalability**: Designed for both small and large-scale research operations

### Configuration Management
```python
# Example: System configuration
SYSTEM_CONFIG = {
    "research": {
        "max_sources": 20,
        "search_depth": "comprehensive",
        "quality_threshold": 0.7
    },
    "agents": {
        "max_concurrent": 4,
        "timeout": 300,
        "retry_attempts": 3
    },
    "output": {
        "format": "markdown",
        "include_citations": True,
        "quality_enhancement": True
    }
}
```

### Agent Coordination Patterns
```python
# Example: Agent orchestration
class ResearchOrchestrator:
    def __init__(self):
        self.agents = {
            "research": ResearchAgent(),
            "report": ReportAgent(),
            "editorial": EditorialAgent(),
            "quality": QualityJudge()
        }

    async def execute_research(self, topic: str, config: dict):
        # Sequential agent execution with quality gates
        research_data = await self.agents["research"].execute(topic, config)
        report = await self.agents["report"].execute(research_data, config)
        enhanced_report = await self.agents["editorial"].execute(report, research_data, config)
        final_quality = await self.agents["quality"].evaluate(enhanced_report)

        return self.format_output(enhanced_report, final_quality)
```

## Usage Examples

### Basic Research Query
```bash
python multi_agent_research_system/run_research.py "latest developments in artificial intelligence"
```

### Advanced Research with Parameters
```bash
python multi_agent_research_system/run_research.py \
  "climate change impacts on global agriculture" \
  --depth "Comprehensive Analysis" \
  --audience "Academic" \
  --format "Academic Paper"
```

### Programmatic Usage
```python
from multi_agent_research_system.core.orchestrator import ResearchOrchestrator

orchestrator = ResearchOrchestrator()
await orchestrator.initialize()

session_id = await orchestrator.start_research_session(
    "quantum computing applications in healthcare",
    {
        "depth": "Standard Research",
        "audience": "Technical",
        "format": "Detailed Report"
    }
)

# Monitor progress and get results
status = await orchestrator.get_session_status(session_id)
results = await orchestrator.get_session_results(session_id)
```

### Custom Agent Configuration
```python
from multi_agent_research_system.config.agents import get_research_agent_definition

# Get standard agent definition
research_agent = get_research_agent_definition()

# Customize agent behavior
custom_config = {
    "max_sources": 30,
    "search_strategy": "comprehensive",
    "quality_threshold": 0.8
}

# Use agent with custom configuration
orchestrator = ResearchOrchestrator(agent_config=custom_config)
```

## System Features

### Multi-Agent Collaboration
- **Research Agent**: Web search, source validation, information synthesis
- **Report Agent**: Content structuring, report generation, formatting
- **Editorial Agent**: Quality enhancement, gap analysis, content improvement
- **Quality Judge**: Assessment, scoring, recommendation

### Quality Management
- **Progressive Enhancement**: Iterative quality improvement
- **Quality Gates**: Minimum quality standards enforcement
- **Content Validation**: Fact-checking and source verification
- **Style Consistency**: Format and tone standardization

### Search Capabilities
- **Multi-Source Research**: Searches across multiple data sources
- **Intelligent Querying**: Optimized search strategies
- **Anti-Detection**: Advanced bot detection avoidance
- **Content Extraction**: Sophisticated information extraction

### MCP Integration
- **Claude SDK Integration**: Seamless Claude model integration
- **Tool Exposure**: Research capabilities exposed through MCP
- **Protocol Compliance**: Full MCP standard compliance
- **Token Management**: Intelligent content optimization

## Performance Considerations

### Optimization Strategies
1. **Concurrent Processing**: Multiple agents work in parallel where possible
2. **Intelligent Caching**: Cache frequently accessed data and results
3. **Resource Management**: Monitor and manage system resources
4. **Quality vs. Speed**: Configurable trade-offs between quality and performance

### Scaling Recommendations
- Use appropriate research depth for your needs
- Configure agent timeouts and retry logic
- Monitor system performance and adjust accordingly
- Consider distributed processing for large-scale operations

## Monitoring and Debugging

### Logging System
```python
# Example: Comprehensive logging
import logging
from multi_agent_research_system.core.logging_config import get_logger

logger = get_logger("research_system")
logger.info("Starting research session")
logger.debug(f"Configuration: {config}")
logger.warning("Quality threshold not met, applying enhancement")
logger.error("Research failed: {error}")
```

### Performance Monitoring
- Session tracking and state management
- Agent performance metrics
- Quality assessment statistics
- Error rate monitoring

### Debugging Tools
- Verbose logging modes
- Agent execution traces
- Quality assessment reports
- Session state inspection

## Configuration

### Environment Variables
```bash
# Required API keys
ANTHROPIC_API_KEY=your_anthropic_key
SERPER_API_KEY=your_serp_key

# Optional configuration
DEFAULT_RESEARCH_DEPTH=Standard Research
MAX_CONCURRENT_AGENTS=5
QUALITY_THRESHOLD=0.7

# Development settings
DEBUG_MODE=false
LOG_LEVEL=INFO
```

### System Settings
```python
# research_system_config.py
SYSTEM_CONFIG = {
    "research": {
        "default_depth": "Standard Research",
        "max_sources": 20,
        "quality_threshold": 0.7
    },
    "agents": {
        "timeout": 300,
        "retry_attempts": 3,
        "max_concurrent": 4
    },
    "output": {
        "directory": "KEVIN",
        "format": "markdown",
        "include_metadata": True
    }
}
```

## Testing

### Test Structure
```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
├── functional/     # End-to-end functional tests
└── fixtures/       # Test data and utilities
```

### Running Tests
```bash
# All tests
python multi_agent_research_system/tests/run_tests.py

# Specific categories
python multi_agent_research_system/tests/run_tests.py --unit
python multi_agent_research_system/tests/run_tests.py --integration
python multi_agent_research_system/tests/run_tests.py --functional

# With coverage
python multi_agent_research_system/tests/run_tests.py --coverage
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure ANTHROPIC_API_KEY is set
   - Verify key has sufficient permissions
   - Check network connectivity

2. **Research Failures**
   - Check search service availability
   - Verify internet connectivity
   - Review search query complexity

3. **Quality Issues**
   - Adjust quality thresholds
   - Verify source diversity
   - Check content length requirements

4. **Performance Issues**
   - Reduce concurrent agent count
   - Optimize search depth settings
   - Monitor system resources

### Debug Mode
```bash
# Enable comprehensive debugging
python multi_agent_research_system/run_research.py "topic" --debug --log-level DEBUG
```

### Log Analysis
```bash
# View recent logs
tail -f KEVIN/logs/research_system.log

# Search for errors
grep "ERROR" KEVIN/logs/research_system.log

# Monitor agent activity
grep "agent" KEVIN/logs/research_system.log
```

## Integration Examples

### CLI Integration
```python
# Example: CLI command integration
import subprocess
import asyncio

async def run_research_query(topic: str, depth: str = "Standard"):
    cmd = [
        "python", "multi_agent_research_system/run_research.py",
        topic, "--depth", depth
    ]

    result = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await result.communicate()
    return result.returncode == 0, stdout.decode(), stderr.decode()
```

### Python API Integration
```python
# Example: Direct API usage
from multi_agent_research_system.core.orchestrator import ResearchOrchestrator

class ResearchService:
    def __init__(self):
        self.orchestrator = ResearchOrchestrator()

    async def research(self, topic: str, **kwargs):
        await self.orchestrator.initialize()

        session_id = await self.orchestrator.start_research_session(
            topic, kwargs
        )

        # Wait for completion
        while True:
            status = await self.orchestrator.get_session_status(session_id)
            if status["status"] in ["completed", "error"]:
                break
            await asyncio.sleep(1)

        return await self.orchestrator.get_session_results(session_id)
```

## Best Practices

### Research Query Design
- Be specific and focused in your queries
- Use appropriate research depth for your needs
- Consider target audience when formulating queries
- Include relevant timeframes if needed

### Quality Optimization
- Set appropriate quality thresholds
- Use progressive enhancement for better results
- Monitor quality assessment scores
- Adjust agent behavior based on results

### Resource Management
- Monitor system resource usage
- Configure appropriate timeouts
- Use caching for repeated queries
- Scale resources based on demand

### Security Considerations
- Protect API keys and sensitive configuration
- Use secure connections for external services
- Monitor for unusual activity patterns
- Regularly update dependencies

## Future Development

### Planned Enhancements
- Additional specialized agents
- Enhanced quality assessment algorithms
- Improved search strategies
- Extended format support

### Extension Points
- Custom agent development
- Additional search sources
- Quality assessment plugins
- Output format extensions

### Contributing
- Follow established code patterns
- Add comprehensive tests
- Update documentation
- Ensure quality standards