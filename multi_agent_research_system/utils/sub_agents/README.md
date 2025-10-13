# Sub-Agent Architecture for Multi-Agent Research System

## Overview

This comprehensive sub-agent architecture enhances the multi-agent research system with specialized, context-isolated agents that can work in parallel while maintaining proper communication protocols and performance monitoring. The system is built using Claude Agent SDK patterns and provides enterprise-grade reliability, scalability, and observability.

## Architecture Components

### Core Components

1. **Sub-Agent Factory** (`sub_agent_factory.py`)
   - Creates and manages sub-agent instances with proper lifecycle management
   - Handles context isolation and configuration
   - Provides resource cleanup and expiration management

2. **Sub-Agent Types** (`sub_agent_types.py`)
   - Defines 10 specialized agent types with distinct capabilities
   - Provides comprehensive configuration management
   - Includes quality standards and expertise definitions

3. **Context Isolation Manager** (`context_isolation.py`)
   - Ensures proper data isolation between sub-agents
   - Manages data sharing rules and permissions
   - Provides secure communication channels

4. **Communication Protocols** (`communication_protocols.py`)
   - Implements message passing between sub-agents
   - Provides priority-based message routing
   - Handles broadcast and direct communication

5. **Performance Monitor** (`performance_monitor.py`)
   - Tracks execution metrics and resource usage
   - Provides performance analytics and alerting
   - Enables optimization recommendations

6. **Sub-Agent Coordinator** (`sub_agent_coordinator.py`)
   - Orchestrates complex multi-agent workflows
   - Manages task dependencies and execution order
   - Provides workflow status monitoring

7. **Integration Layer** (`integration.py`)
   - Seamlessly integrates with existing agent system
   - Provides fallback mechanisms for reliability
   - Enables hybrid orchestration approaches

## Sub-Agent Types

### Available Agent Types

1. **Researcher** (`SubAgentType.RESEARCHER`)
   - Expert in web research and source discovery
   - Capabilities: web search, source validation, information synthesis
   - Tools: WebSearch, WebFetch, source analysis tools

2. **Report Writer** (`SubAgentType.REPORT_WRITER`)
   - Specializes in transforming research into structured reports
   - Capabilities: content structuring, audience adaptation, formatting
   - Tools: content organization, template management

3. **Editorial Reviewer** (`SubAgentType.EDITORIAL_REVIEWER`)
   - Conducts comprehensive editorial review with gap research
   - Capabilities: quality assessment, gap identification, enhancement coordination
   - Tools: quality assessment, gap research coordination

4. **Quality Assessor** (`SubAgentType.QUALITY_ASSESSOR`)
   - Provides multi-dimensional quality assessment
   - Capabilities: content evaluation, scoring, feedback generation
   - Tools: quality metrics, assessment frameworks

5. **Gap Researcher** (`SubAgentType.GAP_RESEARCHER`)
   - Specializes in targeted research for information gaps
   - Capabilities: focused research, quick retrieval, targeted analysis
   - Tools: specialized search, gap-specific tools

6. **Content Enhancer** (`SubAgentType.CONTENT_ENHANCER`)
   - Applies progressive enhancement techniques
   - Capabilities: content improvement, deepening analysis, optimization
   - Tools: enhancement algorithms, quality improvement

7. **Style Editor** (`SubAgentType.STYLE_EDITOR`)
   - Ensures style consistency and formatting
   - Capabilities: style optimization, presentation enhancement
   - Tools: style checkers, formatting tools

8. **Fact Checker** (`SubAgentType.FACT_CHECKER`)
   - Validates factual accuracy and claims
   - Capabilities: fact verification, source validation, accuracy assessment
   - Tools: fact-checking databases, validation frameworks

9. **Source Validator** (`SubAgentType.SOURCE_VALIDATOR`)
   - Assesses source credibility and reliability
   - Capabilities: source evaluation, bias detection, authority assessment
   - Tools: credibility analysis, source ranking

10. **Coordinator** (`SubAgentType.COORDINATOR`)
    - Manages sub-agent workflows and coordination
    - Capabilities: task distribution, workflow orchestration, resource management
    - Tools: coordination protocols, workflow management

## Quick Start

### Basic Usage

```python
import asyncio
from utils.sub_agents import (
    SubAgentCoordinator, SubAgentType,
    get_sub_agent_factory
)

async def main():
    # Initialize the coordinator
    coordinator = SubAgentCoordinator()
    await coordinator.initialize()

    # Create a coordinated workflow
    workflow_id = await coordinator.create_coordinated_workflow(
        session_id="research_session_001",
        topic="Latest developments in quantum computing",
        description="Comprehensive research on quantum computing advances",
        workflow_type="standard_research"
    )

    # Start the workflow
    await coordinator.start_workflow(workflow_id)

    # Monitor progress
    while True:
        status = await coordinator.get_workflow_status(workflow_id)
        print(f"Status: {status['status']} - Progress: {status['progress_percentage']:.1f}%")

        if status['status'] in ['completed', 'failed']:
            break

        await asyncio.sleep(5)

    # Get results
    if status['status'] == 'completed':
        results = await coordinator.get_workflow_results(workflow_id)
        print(f"Research completed: {results['total_duration']:.2f}s")

    # Cleanup
    await coordinator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Individual Sub-Agents

```python
from utils.sub_agents import get_sub_agent_factory
from utils.sub_agents.sub_agent_types import SubAgentRequest, SubAgentType

async def use_individual_agents():
    factory = get_sub_agent_factory()
    await factory.initialize()

    # Create a researcher sub-agent
    request = SubAgentRequest(
        agent_type=SubAgentType.RESEARCHER,
        task_description="Research AI applications in healthcare",
        session_id="session_001",
        parent_agent="main_system"
    )

    result = await factory.create_and_execute(
        request,
        "Please conduct comprehensive research on AI applications in healthcare, focusing on recent breakthroughs and clinical implementations."
    )

    if result.success:
        print(f"Research completed in {result.execution_time:.2f}s")
        print(f"Quality score: {result.quality_score}")
    else:
        print(f"Research failed: {result.error_message}")

    await factory.shutdown()
```

### Integration with Existing System

```python
from utils.sub_agents.integration import (
    get_sub_agent_integration, create_legacy_adapter
)

async def enhanced_research():
    # Get integration instance
    integration = await get_sub_agent_integration()

    # Execute enhanced research
    result = await integration.execute_research_with_sub_agents(
        session_id="enhanced_session",
        topic="Climate change impacts on agriculture",
        depth="comprehensive",
        focus_areas=["crop_yield", "farming_technology", "policy_impact"]
    )

    if result["success"]:
        print(f"Enhanced research completed using {result['execution_method']}")
        return result["results"]
    else:
        print("Enhanced research failed, fallback was used")
        return None

# Create legacy adapter for existing agents
async def enhance_existing_agents():
    integration = await get_sub_agent_integration()
    adapter = create_legacy_adapter()

    # Enhance existing research agent
    enhanced_research_agent = await adapter.enhance_research_agent(existing_research_agent)

    # Use enhanced capabilities
    result = await enhanced_research_agent.enhanced_research(
        "Renewable energy trends",
        session_id="hybrid_session"
    )
```

## Workflow Types

### Standard Research Workflow
- **Stages**: Research → Report Generation → Editorial Review → Quality Assessment
- **Duration**: ~30 minutes
- **Quality Requirements**: 80% success rate, 70% quality score
- **Use Case**: Standard research tasks with comprehensive analysis

### Quick Analysis Workflow
- **Stages**: Combined Research & Analysis → Quality Assessment
- **Duration**: ~10 minutes
- **Quality Requirements**: 75% success rate, 60% quality score
- **Use Case**: Rapid insights and preliminary analysis

### Comprehensive Report Workflow
- **Stages**: Research → Report Generation → Editorial Review → Gap Research → Content Enhancement → Style Editing → Final Quality Assessment
- **Duration**: ~60 minutes
- **Quality Requirements**: 90% success rate, 85% quality score
- **Use Case**: Detailed reports requiring full enhancement cycle

### Editorial Review Workflow
- **Stages**: Editorial Analysis → Gap Research (if needed) → Enhancement → Quality Assessment
- **Duration**: ~20 minutes
- **Quality Requirements**: 85% success rate, 80% quality score
- **Use Case**: Content review and improvement

### Gap Research Workflow
- **Stages**: Targeted Research → Analysis → Integration
- **Duration**: ~15 minutes
- **Quality Requirements**: 80% success rate, 70% quality score
- **Use Case**: Filling specific information gaps

## Configuration

### Sub-Agent Configuration

Each sub-agent type has a comprehensive configuration including:

```python
# Example: Researcher configuration
config = create_sub_agent_config(SubAgentType.RESEARCHER)

# Access configuration components
persona = config.persona  # Agent persona and behavior
capabilities = config.capabilities  # Tools and capabilities
claude_options = config.claude_options  # Claude SDK options
isolation_level = config.isolation_level  # Context isolation
```

### Custom Configuration

```python
# Create custom configuration
custom_config = create_sub_agent_config(
    SubAgentType.RESEARCHER,
    max_turns=100,
    timeout_seconds=600,
    isolation_level="strict",
    quality_standards={
        "source_credibility": 0.9,
        "information_accuracy": 0.95
    }
)
```

### System Configuration

```python
integration_config = {
    "enable_sub_agents": True,
    "fallback_to_legacy": True,
    "sub_agent_timeout_minutes": 30,
    "max_concurrent_workflows": 5,
    "quality_gate_threshold": 0.75,
    "enable_progressive_enhancement": True
}
```

## Context Isolation

### Isolation Levels

1. **Strict**: Only explicitly allowed data keys can be accessed
2. **Moderate**: Access allowed unless explicitly forbidden
3. **Permissive**: Full access with minimal restrictions

### Data Sharing Rules

```python
# Create sharing rule
from utils.sub_agents.context_isolation import DataSharingRule

rule = DataSharingRule(
    rule_id="research_to_report",
    source_context_pattern="*researcher*",
    target_context_pattern="*report_writer*",
    data_key_pattern="research_findings",
    sharing_allowed=True
)

await isolation_manager.add_sharing_rule(rule)
```

### Context Usage

```python
# Store data in context
await isolation_manager.store_data_in_context(
    context_id, "research_findings", findings_data
)

# Retrieve data from context
findings = await isolation_manager.retrieve_data_from_context(
    context_id, "research_findings"
)

# Share data between contexts
await isolation_manager.share_data_between_contexts(
    source_context, target_context, "research_findings"
)
```

## Communication Protocols

### Message Types

- `DIRECT_MESSAGE`: Point-to-point communication
- `REQUEST`: Request requiring response
- `RESPONSE`: Response to a request
- `NOTIFICATION`: One-way information
- `TASK_ASSIGNMENT`: Assigning tasks to agents
- `TASK_COMPLETION`: Task completion notification
- `WORKFLOW_HANDOFF`: Handing off workflow control
- `DATA_SHARE`: Sharing data between agents

### Message Priorities

1. `CRITICAL`: System-critical messages
2. `HIGH`: High-priority workflow messages
3. `NORMAL`: Standard priority messages
4. `LOW`: Low-priority background messages
5. `BULK`: Bulk data transfer messages

### Sending Messages

```python
# Direct message
message_id = await communication_manager.send_direct_message(
    sender_id="researcher_001",
    sender_type="researcher",
    recipient_id="writer_001",
    recipient_type="report_writer",
    session_id="session_001",
    payload={"research_data": findings},
    priority=MessagePriority.HIGH
)

# Broadcast message
broadcast_id = await communication_manager.broadcast_message(
    sender_id="coordinator",
    sender_type="coordinator",
    session_id="session_001",
    payload={"announcement": "Research phase completed"}
)
```

## Performance Monitoring

### Metrics Tracked

- Execution time and success rates
- Resource usage (memory, CPU)
- Quality scores and assessments
- Error rates and types
- Communication patterns

### Performance Alerts

```python
# Configure performance thresholds
performance_config = {
    "performance_thresholds": {
        "execution_time_warning": 30.0,
        "execution_time_critical": 60.0,
        "memory_usage_warning": 512,
        "success_rate_warning": 80,
        "quality_score_warning": 60
    }
}
```

### Monitoring Usage

```python
# Get system performance
system_perf = await performance_monitor.get_system_performance()

# Get agent-specific performance
agent_perf = await performance_monitor.get_agent_performance("agent_001")

# Get performance trends
trends = await performance_monitor.get_performance_trends(
    "agent_001", hours=24
)
```

## Error Handling and Recovery

### Automatic Retry

The system provides automatic retry mechanisms for failed tasks:

```python
# Configure retry behavior
task = WorkflowTask(
    task_id="task_001",
    max_retries=3,
    retry_count=0,
    timeout_seconds=300
)
```

### Fallback Mechanisms

When sub-agent system is unavailable, the system can fall back to legacy agents:

```python
# Enable fallback
integration_config = {
    "enable_sub_agents": True,
    "fallback_to_legacy": True
}

# The system will automatically use legacy agents if sub-agents fail
```

### Error Recovery

```python
# Handle specific error types
try:
    result = await coordinator.execute_workflow(workflow_id)
except SubAgentError as e:
    logger.error(f"Sub-agent error: {e}")
    # Implement recovery strategy
except WorkflowTimeoutError as e:
    logger.error(f"Workflow timeout: {e}")
    # Handle timeout
```

## Testing

### Running Tests

```bash
# Run all tests
python test_sub_agents.py

# Run specific test categories
python -m pytest test_sub_agents.py::TestSubAgentTypes -v
python -m pytest test_sub_agents.py::TestIntegration -v
```

### Demo Script

```bash
# Run comprehensive demo
python demo_sub_agents.py
```

### Test Coverage

The test suite covers:
- Unit tests for individual components
- Integration tests for component interaction
- End-to-end workflow tests
- Performance monitoring validation
- Error handling and recovery testing

## Best Practices

### Performance Optimization

1. **Choose appropriate workflow types** based on requirements
2. **Monitor resource usage** and adjust concurrency limits
3. **Use context isolation efficiently** to avoid unnecessary restrictions
4. **Set appropriate timeouts** for different task types
5. **Enable performance monitoring** to identify bottlenecks

### Quality Assurance

1. **Set quality thresholds** appropriate to your use case
2. **Monitor quality scores** and address degradation
3. **Use progressive enhancement** for iterative improvement
4. **Validate outputs** before using in downstream processes
5. **Track quality trends** over time

### Resource Management

1. **Limit concurrent workflows** to prevent resource exhaustion
2. **Cleanup completed contexts** and agent instances
3. **Monitor memory usage** and implement appropriate limits
4. **Use efficient data structures** for large datasets
5. **Implement proper caching** where appropriate

### Security Considerations

1. **Use strict context isolation** for sensitive data
2. **Validate data sharing rules** before implementation
3. **Monitor access logs** for unauthorized access attempts
4. **Implement proper authentication** for agent communication
5. **Secure persistent storage** if enabled

## Troubleshooting

### Common Issues

1. **Sub-agent creation fails**
   - Check Claude SDK configuration
   - Verify API credentials
   - Ensure available resources

2. **Workflow timeouts**
   - Increase timeout settings
   - Check task complexity
   - Monitor resource usage

3. **Context isolation errors**
   - Verify isolation level settings
   - Check data sharing rules
   - Review permission configurations

4. **Communication failures**
   - Verify message handler registration
   - Check network connectivity
   - Review message formats

5. **Performance degradation**
   - Monitor resource usage
   - Check for memory leaks
   - Review agent configurations

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode in components
coordinator = SubAgentCoordinator(debug_mode=True)
factory = SubAgentFactory(debug_mode=True)
```

### Performance Profiling

Use the performance monitor to identify bottlenecks:

```python
# Get detailed performance metrics
status = coordinator.get_coordinator_status()
factory_status = factory.get_factory_status()
monitoring_status = performance_monitor.get_monitoring_status()
```

## Migration Guide

### From Legacy System

1. **Install sub-agent architecture** alongside existing system
2. **Enable integration layer** with fallback enabled
3. **Test specific workflows** with sub-agents
4. **Gradually increase usage** as confidence builds
5. **Monitor performance** and quality improvements
6. **Fully migrate** when ready

### Configuration Migration

```python
# Legacy configuration
legacy_config = {
    "research_depth": "comprehensive",
    "max_sources": 20,
    "quality_threshold": 0.75
}

# Sub-agent configuration
sub_agent_config = {
    "workflow_type": "standard_research",
    "quality_requirements": {
        "min_success_rate": 80,
        "min_quality_score": 75
    },
    "enable_fallback": True
}
```

## Future Enhancements

### Planned Features

1. **Dynamic workflow composition** - AI-driven workflow creation
2. **Advanced load balancing** - Intelligent resource allocation
3. **Cross-session learning** - Knowledge transfer between sessions
4. **Real-time collaboration** - Interactive multi-user workflows
5. **Advanced analytics** - Predictive performance optimization

### Extension Points

1. **Custom agent types** - Domain-specific specializations
2. **Workflow templates** - Industry-specific patterns
3. **Quality criteria** - Custom assessment frameworks
4. **Communication protocols** - Specialized message types
5. **Integration adapters** - Additional system connections

## Support and Contributing

### Getting Help

- Review the documentation and examples
- Check the test cases for usage patterns
- Enable debug logging for detailed troubleshooting
- Review performance metrics for optimization opportunities

### Contributing

1. **Follow established patterns** in existing code
2. **Add comprehensive tests** for new features
3. **Update documentation** for API changes
4. **Ensure backward compatibility** when possible
5. **Test integration** with existing systems

---

**Version**: 2.0.0
**Last Updated**: October 13, 2025
**Compatibility**: Claude Agent SDK 1.0+, Python 3.8+
**License**: See project license file