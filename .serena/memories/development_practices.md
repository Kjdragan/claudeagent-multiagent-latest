# Development Practices - Multi-Agent Research System

## Code Style and Conventions

### Python Style Guidelines
- **Python Version**: 3.10+ with strict type hints
- **Line Length**: 88 characters (ruff configuration)
- **Type Hints**: Mandatory for all function signatures and class attributes
- **Docstrings**: Comprehensive docstrings for all public functions and classes
- **Naming Conventions**: 
  - Functions and variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_single_underscore_prefix`

### Async/Await Patterns
- **Async-First Design**: All I/O operations use async/await
- **Error Handling**: Comprehensive try/catch blocks with specific exception types
- **Resource Management**: Proper cleanup and context managers
- **Concurrency**: Use asyncio.gather() for parallel operations when appropriate

### Agent Development Patterns
```python
from multi_agent_research_system.core.base_agent import BaseAgent
from typing import Any, Dict, Optional

class CustomAgent(BaseAgent):
    """Custom agent implementation following established patterns."""
    
    def __init__(self):
        super().__init__("custom_agent", "custom_type")
        self.register_message_handler("task_type", self.handle_task)
    
    def get_system_prompt(self) -> str:
        """Return detailed system prompt for the agent."""
        return """You are a specialized AI agent with specific capabilities...
        
        CRITICAL INSTRUCTION: You MUST execute actual tools rather than just documenting intentions.
        """
    
    async def handle_task(self, message: Message) -> Optional[Message]:
        """Handle incoming messages with proper error handling."""
        try:
            # Process task
            result = await self.execute_task_logic(message.payload)
            
            # Send response
            return Message(
                sender=self.name,
                recipient=message.sender,
                message_type="task_result",
                payload=result,
                session_id=message.session_id,
                correlation_id=message.correlation_id
            )
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return await self.handle_error(e, message)
```

## Quality Standards

### Code Quality Requirements
- **Type Safety**: 100% type coverage with mypy strict mode
- **Linting**: Pass ruff with all rules enabled
- **Testing**: >90% code coverage with comprehensive test suite
- **Documentation**: Updated CLAUDE.md files for all components
- **Error Handling**: Comprehensive error recovery and resilience patterns

### Agent Quality Standards
- **System Prompts**: Detailed, specific instructions with mandatory execution requirements
- **Tool Integration**: Proper Claude Agent SDK tool registration and usage
- **Message Handling**: Robust message processing with correlation IDs
- **Session Management**: Proper session lifecycle management
- **Quality Gates**: Built-in quality assessment and enhancement

## Testing Approach

### Test Structure
```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
├── functional/     # End-to-end functional tests
├── e2e-tests/      # End-to-end tests in separate directory
└── fixtures/       # Test data and utilities
```

### Test Categories
1. **Unit Tests**: Individual component testing with mocked dependencies
2. **Integration Tests**: Component interaction and workflow testing
3. **Functional Tests**: End-to-end workflow validation
4. **Performance Tests**: Load testing and optimization validation
5. **Quality Tests**: Quality framework validation and enhancement testing

### Test Patterns
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from multi_agent_research_system.core.orchestrator import ResearchOrchestrator

class TestResearchOrchestrator:
    """Test orchestrator functionality with comprehensive coverage."""
    
    @pytest.mark.asyncio
    async def test_research_workflow_success(self):
        """Test successful research workflow execution."""
        # Setup
        orchestrator = ResearchOrchestrator()
        orchestrator.research_agent = AsyncMock()
        orchestrator.report_agent = AsyncMock()
        
        # Mock successful research
        orchestrator.research_agent.execute.return_value = {
            "success": True,
            "findings": "Research data",
            "sources": 10
        }
        
        # Execute
        result = await orchestrator.execute_research_workflow("test query")
        
        # Assert
        assert result["success"] is True
        assert result["findings"] == "Research data"
```

## Pre-commit Checklist

### Required Checks Before Commit
- [ ] Code passes `ruff check` with no errors
- [ ] Code passes `ruff format` with consistent formatting
- [ ] Code passes `mypy multi_agent_research_system/` with no errors
- [ ] All tests pass: `pytest tests/`
- [ ] Integration tests pass: `pytest e2e-tests/`
- [ ] Documentation updated for affected components
- [ ] CLAUDE.md files updated in modified directories
- [ ] Performance impact assessed and documented
- [ ] Memory usage reviewed for new features

### Quality Gates
- **Type Safety**: No mypy errors allowed
- **Code Style**: Zero ruff violations
- **Test Coverage**: Minimum 90% coverage
- **Documentation**: All public APIs documented
- **Performance**: No performance regression >5%

## Design Patterns

### Agent Architecture Patterns
1. **Base Agent Inheritance**: All agents inherit from BaseAgent
2. **Message-Based Communication**: Structured message passing with correlation IDs
3. **Tool Registration**: Proper Claude Agent SDK tool integration
4. **Quality Integration**: Built-in quality assessment and enhancement
5. **Error Recovery**: Comprehensive resilience patterns

### System Design Patterns
1. **Quality-First Architecture**: Quality assessment built into every stage
2. **Progressive Enhancement**: Iterative quality improvement
3. **Gap Research Coordination**: Intelligent control handoff for additional research
4. **Session Management**: Comprehensive lifecycle tracking with persistence
5. **MCP Integration**: Full Model Context Protocol compliance

### Error Handling Patterns
```python
async def execute_with_recovery(self, operation: str, *args, **kwargs) -> dict:
    """Execute operation with comprehensive error recovery."""
    for attempt in range(self.max_retry_attempts):
        try:
            result = await self._execute_operation(operation, *args, **kwargs)
            await self._create_checkpoint(operation, result)
            return result
            
        except RecoverableError as e:
            self.logger.warning(f"Recoverable error in {operation}: {e}")
            if attempt < self.max_retry_attempts - 1:
                await self._recover_from_error(e, operation, *args, **kwargs)
                continue
            else:
                return await self._execute_fallback_strategy(operation, *args, **kwargs)
                
        except CriticalError as e:
            self.logger.error(f"Critical error in {operation}: {e}")
            return await self._handle_critical_error(e, operation, *args, **kwargs)
```

## Configuration Management

### Environment Variables
```bash
# Required API Keys
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
SERP_API_KEY=your_serp_key

# Optional Configuration
LOGFIRE_TOKEN=your-logfire-token
RESEARCH_QUALITY_THRESHOLD=0.8
MAX_SEARCH_RESULTS=10
MAX_CONCURRENT_AGENTS=3
DEBUG_MODE=false
LOG_LEVEL=INFO
```

### Configuration Files
- **pyproject.toml**: Project dependencies and tool configuration
- **CLAUDE.md**: System documentation and usage guidelines
- **.gitignore**: Git ignore rules for sensitive data and build artifacts
- **.env.example**: Environment variable template

## Documentation Standards

### CLAUDE.md Documentation
Every directory must have a CLAUDE.md file containing:
- Directory purpose and scope
- Key components and their responsibilities
- Usage examples and integration patterns
- Development guidelines and best practices
- Dependencies and system interactions
- Testing approaches and quality standards

### Code Documentation
- **Function Docstrings**: Purpose, parameters, return values, examples
- **Class Docstrings**: Purpose, responsibilities, usage patterns
- **Inline Comments**: Complex logic explanations and architectural decisions
- **Type Hints**: Complete type annotations for all signatures

## Performance Guidelines

### Optimization Targets
- **Memory Usage**: Monitor and manage agent memory consumption
- **Concurrent Operations**: Limit to 2-3 agents for optimal performance
- **API Rate Limits**: Implement proper rate limiting and quota management
- **Content Processing**: Use cleanliness assessment to skip unnecessary AI cleaning
- **Search Optimization**: Intelligent search strategy selection

### Monitoring Requirements
- **Agent Performance**: Track execution time and success rates
- **Quality Metrics**: Monitor quality assessment scores and improvement trends
- **Resource Utilization**: Monitor memory, CPU, and network usage
- **Error Rates**: Track error frequency and recovery success rates

## Security Considerations

### API Key Management
- **Environment Variables**: Store API keys in environment variables
- **No Hardcoded Keys**: Never commit API keys to version control
- **Key Rotation**: Regular rotation of API keys
- **Access Control**: Limit access to API keys based on roles

### Data Privacy
- **Sensitive Information**: No logging of sensitive research data
- **Session Isolation**: Proper session data isolation and cleanup
- **Audit Trails**: Comprehensive logging for compliance requirements
- **Data Retention**: Regular cleanup of old research sessions