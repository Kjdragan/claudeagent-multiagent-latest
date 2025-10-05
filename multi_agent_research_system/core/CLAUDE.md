# Core Directory - Multi-Agent Research System

This directory contains the system orchestration and foundational components that coordinate the entire multi-agent research workflow.

## Directory Purpose

The core directory provides the central nervous system of the multi-agent research system, including orchestration, quality management, agent coordination, and workflow state management. These components ensure reliable, scalable, and intelligent research operations.

## Key Components

### System Orchestration
- **`orchestrator.py`** - Main system orchestrator that coordinates all agents and workflow (217KB - central component)
- **`workflow_state.py`** - Workflow state management and session tracking
- **`error_recovery.py`** - Error handling and recovery mechanisms
- **`progressive_enhancement.py`** - Progressive enhancement pipeline for research quality

### Quality Management
- **`quality_framework.py`** - Quality control and validation framework
- **`quality_gates.py`** - Quality gate implementations and thresholds
- **`agent_logger.py`** - Comprehensive agent activity logging system

### Agent Foundation
- **`base_agent.py`** - Base agent class and common functionality
- **`research_tools.py`** - Core research tool implementations and MCP integration
- **`search_analysis_tools.py`** - Search result analysis and processing tools
- **`simple_research_tools.py`** - Simplified research tool implementations

### System Infrastructure
- **`logging_config.py`** - Centralized logging configuration and setup
- **`cli_parser.py`** - Command-line interface parsing and configuration
- **`llm_utils.py`** - LLM integration utilities and helper functions

## Core Architecture

### Orchestrator Architecture
```python
class ResearchOrchestrator:
    """Central coordination point for the multi-agent research system"""

    def __init__(self, debug_mode=False):
        self.agents = {}  # Agent registry
        self.sessions = {}  # Active research sessions
        self.quality_framework = QualityFramework()
        self.mcp_server = None

    async def start_research_session(self, topic: str, config: dict) -> str:
        """Initiate a new research session"""

    async def orchestrate_research(self, session_id: str) -> dict:
        """Coordinate the research workflow"""
```

### Quality Framework Integration
```
Research Input → Quality Gates → Agent Processing → Quality Assessment → Progressive Enhancement → Final Output
```

### Error Recovery Flow
```
Error Detection → Error Classification → Recovery Strategy → Retry Logic → Fallback Handling → Status Reporting
```

## Development Guidelines

### Core System Patterns
1. **Async/Await Architecture**: All core operations should be asynchronous
2. **Error-First Design**: Implement comprehensive error handling at every level
3. **State Management**: Maintain clear, serializable state for all operations
4. **Quality Integration**: Build quality assessment into every stage

### Orchestrator Design Principles
```python
# Example: Orchestrator coordination pattern
class ResearchOrchestrator:
    async def orchestrate_research(self, session_id: str):
        try:
            # Initialize workflow state
            workflow_state = await self._initialize_workflow(session_id)

            # Execute research pipeline
            research_results = await self._execute_research_pipeline(workflow_state)

            # Apply quality gates
            quality_results = await self._apply_quality_gates(research_results)

            # Progressive enhancement if needed
            if not quality_results.meets_standards:
                enhanced_results = await self._progressive_enhancement(quality_results)
                return enhanced_results

            return quality_results

        except Exception as e:
            return await self._handle_orchestration_error(e, session_id)
```

### Quality Management Patterns
```python
# Example: Quality gate implementation
class QualityGate:
    def __init__(self, name: str, threshold: float, validator: callable):
        self.name = name
        self.threshold = threshold
        self.validator = validator

    def evaluate(self, content: dict) -> tuple[bool, dict]:
        score = self.validator(content)
        passes = score >= self.threshold
        return passes, {"score": score, "threshold": self.threshold}
```

### Error Recovery Strategies
```python
# Example: Error recovery implementation
class ErrorRecoveryManager:
    def __init__(self):
        self.recovery_strategies = {
            "network_error": self._recover_from_network_error,
            "content_extraction_error": self._recover_from_extraction_error,
            "quality_failure": self._recover_from_quality_failure
        }

    async def handle_error(self, error: Exception, context: dict) -> dict:
        error_type = self._classify_error(error)
        recovery_func = self.recovery_strategies.get(error_type)

        if recovery_func:
            return await recovery_func(error, context)
        else:
            return await self._default_recovery(error, context)
```

## Testing & Debugging

### Core System Testing
1. **Orchestrator Testing**: Test complete research workflows end-to-end
2. **Quality Framework Testing**: Verify quality gates and enhancement work correctly
3. **Error Recovery Testing**: Test various error scenarios and recovery mechanisms
4. **State Management Testing**: Ensure workflow state is maintained correctly

### Debugging Core Components
1. **Comprehensive Logging**: Enable detailed logging for all core operations
2. **State Inspection**: Monitor workflow state changes and transitions
3. **Performance Monitoring**: Track orchestration performance and bottlenecks
4. **Quality Metrics**: Monitor quality gate performance and enhancement effectiveness

### Common Core Issues & Solutions
- **Orchestration Failures**: Implement better error handling and recovery mechanisms
- **Quality Gate Failures**: Adjust thresholds and improve validation logic
- **State Inconsistencies**: Implement better state synchronization and validation
- **Performance Bottlenecks**: Optimize async operations and resource management

## Dependencies & Interactions

### Core Dependencies
- **claude-agent-sdk**: Claude Agent SDK for agent management
- **asyncio**: Async programming support
- **pydantic**: Data validation and serialization
- **logfire**: Structured logging and observability

### Internal System Dependencies
- **Agent System**: Core orchestrates and manages all agents
- **Utils Layer**: Core components use utilities for low-level operations
- **MCP Tools**: Core manages MCP server lifecycle and tool registration
- **Config System**: Core uses configuration for behavior control

### Data Flow Architecture
```
User Request → Orchestrator → Agent Coordination → Quality Management → State Updates → Response Generation
```

## Usage Examples

### Basic Orchestration
```python
from core.orchestrator import ResearchOrchestrator

orchestrator = ResearchOrchestrator(debug_mode=True)
await orchestrator.initialize()

# Start a research session
session_id = await orchestrator.start_research_session(
    "artificial intelligence in healthcare",
    {
        "depth": "comprehensive",
        "audience": "technical",
        "format": "detailed_report"
    }
)

# Monitor progress
status = await orchestrator.get_session_status(session_id)
print(f"Session status: {status['status']}")
```

### Quality Framework Usage
```python
from core.quality_framework import QualityFramework

framework = QualityFramework()

# Define quality gates
framework.add_gate("content_completeness", 0.8, completeness_validator)
framework.add_gate("source_credibility", 0.7, credibility_validator)

# Evaluate content
results = await framework.evaluate_content(research_data)
if results.passes_all_gates:
    print("Content meets quality standards")
else:
    print("Content needs enhancement")
```

### Error Recovery Setup
```python
from core.error_recovery import ErrorRecoveryManager

recovery_manager = ErrorRecoveryManager()

# Configure custom recovery strategies
recovery_manager.add_strategy("custom_error", custom_recovery_handler)

# Handle errors in orchestrator
try:
    results = await orchestrator.orchestrate_research(session_id)
except Exception as e:
    recovered_results = await recovery_manager.handle_error(e, {
        "session_id": session_id,
        "operation": "research_orchestration"
    })
```

### Progressive Enhancement
```python
from core.progressive_enhancement import ProgressiveEnhancementPipeline

enhancement_pipeline = ProgressiveEnhancementPipeline()

# Configure enhancement stages
enhancement_pipeline.add_stage("additional_research", research_enhancer)
enhancement_pipeline.add_stage("content_improvement", content_enhancer)
enhancement_pipeline.add_stage("quality_boost", quality_enhancer)

# Apply enhancement if needed
if not quality_results.meets_standards:
    enhanced_results = await enhancement_pipeline.enhance(
        quality_results,
        target_quality=0.9
    )
```

## Performance Considerations

### Core System Optimization
1. **Async Optimization**: Use efficient async patterns and avoid blocking operations
2. **Resource Management**: Monitor and manage memory, CPU, and network usage
3. **Caching Strategy**: Implement intelligent caching for frequently accessed data
4. **Connection Pooling**: Use connection pooling for external service calls

### Scaling Recommendations
- Implement horizontal scaling for orchestrator instances
- Use distributed state management for large-scale deployments
- Implement load balancing for agent coordination
- Monitor and optimize quality gate performance

### Monitoring and Observability
- Implement comprehensive metrics collection
- Use distributed tracing for workflow monitoring
- Set up alerts for performance degradation
- Monitor quality gate effectiveness and enhancement success rates

## Configuration Management

### Core Configuration
```python
# Example: Core system configuration
CORE_CONFIG = {
    "orchestrator": {
        "max_concurrent_sessions": 10,
        "session_timeout": 3600,
        "retry_attempts": 3
    },
    "quality": {
        "default_threshold": 0.7,
        "enhancement_enabled": True,
        "max_enhancement_cycles": 3
    },
    "logging": {
        "level": "INFO",
        "structured": True,
        "include_agent_logs": True
    }
}
```

### Quality Framework Configuration
```python
# Example: Quality framework setup
QUALITY_CONFIG = {
    "gates": {
        "content_completeness": {"threshold": 0.8, "weight": 0.3},
        "source_credibility": {"threshold": 0.7, "weight": 0.3},
        "analytical_depth": {"threshold": 0.6, "weight": 0.2},
        "clarity_coherence": {"threshold": 0.8, "weight": 0.2}
    },
    "enhancement": {
        "enabled": True,
        "max_cycles": 3,
        "improvement_threshold": 0.1
    }
}
```

## Integration Patterns

### Agent Integration
```python
# Example: Core-actor integration
class ResearchOrchestrator:
    def __init__(self):
        self.agents = {
            "research": ResearchAgent(),
            "report": ReportAgent(),
            "editorial": EditorialAgent()
        }
        self.agent_logger = AgentLogger()

    async def coordinate_agents(self, session_id: str):
        # Research phase
        research_results = await self.agents["research"].execute(session_id)
        self.agent_logger.log_agent_activity("research", research_results)

        # Report phase
        report_results = await self.agents["report"].execute(research_results)
        self.agent_logger.log_agent_activity("report", report_results)

        # Editorial phase
        editorial_results = await self.agents["editorial"].execute(report_results)
        self.agent_logger.log_agent_activity("editorial", editorial_results)

        return editorial_results
```

### MCP Integration
```python
# Example: Core-MCP integration
class ResearchOrchestrator:
    async def setup_mcp_server(self):
        self.mcp_server = create_sdk_mcp_server(
            tools=self._get_core_tools(),
            name="research-core",
            description="Core research orchestration tools"
        )

    def _get_core_tools(self):
        return [
            self.research_tools.get_session_data,
            self.research_tools.create_research_report,
            self.search_analysis_tools.analyze_search_results
        ]
```