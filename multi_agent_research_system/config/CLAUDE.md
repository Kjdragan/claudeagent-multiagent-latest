# Config Directory - Multi-Agent Research System

This directory contains system configuration, agent definitions, and settings that control the behavior of the multi-agent research system.

## Directory Purpose

The config directory provides centralized configuration management for the entire multi-agent research system, including agent definitions, behavior settings, and system parameters. This configuration enables flexible system behavior without code changes.

## Key Components

### Agent Configuration
- **`agents.py`** - Comprehensive agent definitions, prompts, and tool configurations (26KB)
- **`settings.py`** - System settings, environment variables, and global configuration

## Configuration Architecture

### Agent Definition Structure
```python
def get_research_agent_definition() -> AgentDefinition:
    """Define the Research Agent using SDK AgentDefinition pattern."""
    return AgentDefinition(
        description="Expert Research Agent specializing in comprehensive web research...",
        prompt="""You are a Research Agent, an expert in conducting comprehensive, high-quality research...""",
        tools=["conduct_research", "analyze_sources", "save_research_findings"],
        model="sonnet"
    )
```

### Settings Management
```python
# Environment-based configuration
class Settings:
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY')
    SERPER_API_KEY: str = os.getenv('SERPER_API_KEY')
    DEFAULT_RESEARCH_DEPTH: str = "Standard Research"
    MAX_CONCURRENT_AGENTS: int = 5
    QUALITY_THRESHOLD: float = 0.7
```

## Agent Definitions

### Available Agents
1. **Research Agent** - Web research, source validation, information synthesis
2. **Report Agent** - Report generation, content structuring, formatting
3. **Editorial Agent** - Content enhancement, quality improvement, gap analysis
4. **Content Cleaner Agent** - Content cleaning, standardization, quality assessment
5. **Content Quality Judge** - Quality evaluation, scoring, recommendation

### Agent Configuration Patterns
Each agent definition includes:
- **Description**: Clear purpose and capabilities
- **Prompt**: Detailed behavior instructions and guidelines
- **Tools**: Available MCP tools and utilities
- **Quality Criteria**: Specific quality standards and evaluation metrics
- **Workflow Integration**: How the agent fits into the overall system

### Tool Configuration
```python
# Example: Agent tool configuration
RESEARCH_AGENT_TOOLS = [
    "conduct_research",
    "analyze_sources",
    "save_research_findings",
    "get_session_data",
    "create_research_report"
]

EDITORIAL_AGENT_TOOLS = [
    "get_session_data",
    "revise_report",
    "identify_research_gaps",
    "conduct_research",
    "analyze_sources",
    "create_research_report"
]
```

## Development Guidelines

### Configuration Management
1. **Environment Variables**: Use environment variables for sensitive configuration
2. **Default Values**: Provide sensible defaults for all configuration options
3. **Validation**: Validate configuration values at startup
4. **Documentation**: Document all configuration options and their effects

### Agent Definition Standards
```python
# Example: Standard agent definition structure
def get_agent_definition() -> AgentDefinition:
    return AgentDefinition(
        description="Clear, concise description of agent purpose",
        prompt="""
        Detailed behavior instructions including:
        - Core responsibilities and priorities
        - Quality criteria and success metrics
        - Tool usage guidelines and workflows
        - Error handling and fallback strategies
        - Integration patterns with other agents
        """,
        tools=["tool1", "tool2", "tool3"],
        model="sonnet"
    )
```

### Settings Patterns
```python
# Example: Settings configuration
class Settings:
    # API Configuration
    ANTHROPIC_BASE_URL: str = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY')

    # Research Configuration
    DEFAULT_RESEARCH_DEPTH: str = "Standard Research"
    MAX_SEARCH_RESULTS: int = 20
    CONTENT_QUALITY_THRESHOLD: float = 0.7

    # Agent Configuration
    MAX_CONCURRENT_AGENTS: int = 5
    AGENT_TIMEOUT: int = 300  # 5 minutes

    # Output Configuration
    DEFAULT_OUTPUT_FORMAT: str = "markdown"
    MAX_CONTENT_LENGTH: int = 50000

    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required")
        # Additional validation logic
```

## Configuration Sections

### Research Configuration
```python
RESEARCH_CONFIG = {
    "depths": {
        "Quick Overview": {
            "max_sources": 5,
            "content_length": "short",
            "analysis_depth": "basic"
        },
        "Standard Research": {
            "max_sources": 15,
            "content_length": "medium",
            "analysis_depth": "detailed"
        },
        "Comprehensive Analysis": {
            "max_sources": 30,
            "content_length": "long",
            "analysis_depth": "thorough"
        }
    },
    "audiences": {
        "General Public": {"language_level": "simple", "technical_depth": "low"},
        "Academic": {"language_level": "formal", "technical_depth": "high"},
        "Business": {"language_level": "professional", "technical_depth": "medium"},
        "Technical": {"language_level": "technical", "technical_depth": "high"},
        "Policy Makers": {"language_level": "formal", "technical_depth": "medium"}
    }
}
```

### Quality Configuration
```python
QUALITY_CONFIG = {
    "thresholds": {
        "content_completeness": 0.8,
        "source_credibility": 0.7,
        "analytical_depth": 0.6,
        "clarity_coherence": 0.8,
        "factual_accuracy": 0.9
    },
    "enhancement": {
        "enabled": True,
        "max_cycles": 3,
        "improvement_threshold": 0.1,
        "gap_filling_enabled": True
    }
}
```

### Agent Behavior Configuration
```python
AGENT_BEHAVIOR = {
    "research_agent": {
        "search_strategy": "adaptive",
        "source_validation": "strict",
        "content_extraction": "comprehensive",
        "max_retry_attempts": 3
    },
    "report_agent": {
        "structure": "standard_report",
        "citation_style": "informal",
        "length_adaptation": True,
        "quality_threshold": 0.7
    },
    "editorial_agent": {
        "enhancement_focus": "data_integration",
        "gap_filling_strategy": "targeted",
        "style_consistency": True,
        "length_optimization": True
    }
}
```

## Testing & Debugging

### Configuration Testing
1. **Validation Testing**: Test configuration validation logic
2. **Default Value Testing**: Verify defaults work correctly
3. **Environment Variable Testing**: Test environment variable loading
4. **Agent Configuration Testing**: Test agent definitions load correctly

### Debugging Configuration Issues
1. **Configuration Logging**: Log configuration values at startup
2. **Validation Errors**: Provide clear error messages for configuration issues
3. **Environment Monitoring**: Monitor environment variable changes
4. **Agent Behavior**: Monitor agent behavior changes due to configuration

### Common Configuration Issues
- **Missing Environment Variables**: Provide clear setup instructions
- **Invalid Values**: Implement comprehensive validation with helpful error messages
- **Agent Definition Errors**: Validate agent definitions before use
- **Configuration Conflicts**: Detect and resolve conflicting settings

## Usage Examples

### Basic Configuration Usage
```python
from config.settings import Settings
from config.agents import get_research_agent_definition

# Load configuration
settings = Settings()
settings.validate()

# Get agent definition
research_agent = get_research_agent_definition()
print(f"Research agent: {research_agent.description}")
```

### Custom Configuration
```python
# Custom settings for specific deployment
class CustomSettings(Settings):
    CUSTOM_RESEARCH_DEPTH = "Comprehensive Analysis"
    CUSTOM_MAX_SOURCES = 50
    CUSTOM_QUALITY_THRESHOLD = 0.9

# Custom agent configuration
def get_custom_research_agent() -> AgentDefinition:
    base_agent = get_research_agent_definition()
    base_agent.prompt += "\n\nAdditional custom instructions..."
    return base_agent
```

### Environment-Based Configuration
```python
# .env file
ANTHROPIC_API_KEY=your_api_key_here
SERPER_API_KEY=your_serp_key_here
DEFAULT_RESEARCH_DEPTH=Standard Research
MAX_CONCURRENT_AGENTS=10
QUALITY_THRESHOLD=0.8

# Loading in code
from dotenv import load_dotenv
load_dotenv()

settings = Settings()
print(f"Research depth: {settings.DEFAULT_RESEARCH_DEPTH}")
```

### Runtime Configuration Updates
```python
# Dynamic configuration updates
class ConfigurationManager:
    def __init__(self):
        self.settings = Settings()
        self.agent_configs = {}

    def update_quality_threshold(self, new_threshold: float):
        self.settings.QUALITY_THRESHOLD = new_threshold
        self._notify_agents_of_config_change()

    def update_agent_prompt(self, agent_name: str, additional_instructions: str):
        if agent_name in self.agent_configs:
            self.agent_configs[agent_name].prompt += f"\n\n{additional_instructions}"
```

## Best Practices

### Configuration Management
1. **Separation of Concerns**: Keep different types of configuration separate
2. **Environment Awareness**: Use different configurations for different environments
3. **Security**: Never commit sensitive configuration values to version control
4. **Documentation**: Document all configuration options and their effects

### Agent Configuration
1. **Clear Instructions**: Provide clear, specific instructions in agent prompts
2. **Tool Alignment**: Ensure agent tools match their described capabilities
3. **Quality Standards**: Define clear quality criteria for each agent
4. **Integration Patterns**: Document how agents should work together

### Performance Configuration
1. **Resource Limits**: Set appropriate resource limits for agents
2. **Timeout Configuration**: Configure reasonable timeouts for operations
3. **Concurrency Settings**: Optimize concurrency for your deployment environment
4. **Caching Configuration**: Configure caching appropriately for your use case

## Integration Patterns

### Configuration Loading
```python
# Example: Configuration loading pattern
class ConfigManager:
    def __init__(self):
        self.settings = self._load_settings()
        self.agent_definitions = self._load_agent_definitions()

    def _load_settings(self) -> Settings:
        settings = Settings()
        settings.validate()
        return settings

    def _load_agent_definitions(self) -> dict:
        return {
            "research": get_research_agent_definition(),
            "report": get_report_agent_definition(),
            "editorial": get_editorial_agent_definition()
        }
```

### Agent Factory Pattern
```python
# Example: Agent factory with configuration
class AgentFactory:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager

    def create_agent(self, agent_type: str, **kwargs):
        if agent_type not in self.config.agent_definitions:
            raise ValueError(f"Unknown agent type: {agent_type}")

        definition = self.config.agent_definitions[agent_type]
        return ClaudeAgent(
            definition=definition,
            settings=self.config.settings,
            **kwargs
        )
```

### Configuration Validation
```python
# Example: Comprehensive configuration validation
def validate_configuration(settings: Settings, agent_definitions: dict) -> list[str]:
    errors = []

    # Validate settings
    if not settings.ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is required")

    if settings.QUALITY_THRESHOLD < 0 or settings.QUALITY_THRESHOLD > 1:
        errors.append("QUALITY_THRESHOLD must be between 0 and 1")

    # Validate agent definitions
    for name, definition in agent_definitions.items():
        if not definition.description:
            errors.append(f"Agent {name} missing description")

        if not definition.tools:
            errors.append(f"Agent {name} has no tools configured")

    return errors
```