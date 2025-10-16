# Configuration System - Multi-Agent Research System

This directory contains the actual configuration management system for the multi-agent research system, based on analysis of the real implementation files.

## Directory Purpose

The config directory provides centralized configuration management for the multi-agent research system, including search parameters, agent definitions, SDK configuration, and environment-based settings management.

## Key Components

### Core Configuration Files

- **`settings.py`** - Main settings management using Pydantic BaseSettings with environment variable support (240 lines)
- **`agents.py`** - Agent definitions using Claude Agent SDK AgentDefinition pattern with detailed prompts and tool configurations (520 lines)
- **`sdk_config.py`** - Comprehensive Claude Agent SDK configuration with hooks, observability, and sub-agent coordination (536 lines)
- **`enhanced_agents.py`** - Enhanced agent definitions with flow adherence enforcement and quality gates (818 lines)
- **`config_manager.py`** - Unified configuration manager integrating legacy and enhanced SDK configurations (428 lines)
- **`research_targets.py`** - Research target configuration for different scopes and stages (133 lines)
- **`settings_broken.py`** - Alternative configuration implementation with enhanced search config (414 lines)

## Actual Configuration Architecture

### Multi-Layer Configuration System

The system implements a multi-layered configuration approach:

1. **Legacy Settings (settings.py)** - Pydantic-based configuration with environment variables
2. **SDK Configuration (sdk_config.py)** - Claude Agent SDK integration with advanced features
3. **Enhanced Agents (enhanced_agents.py)** - Flow adherence enforcement and quality gates
4. **Unified Manager (config_manager.py)** - Integration layer for all configuration types

### Core Settings Implementation

#### Pydantic Settings Base (settings.py)

The main configuration system uses Pydantic BaseSettings for environment variable management:

```python
class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # AI Model APIs
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")

    # External Services
    serper_api_key: str = Field(..., env="SERPER_API_KEY")
    youtube_api_key: Optional[str] = Field(None, env="YOUTUBE_API_KEY")

    # System Settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_tokens: int = Field(8192, env="MAX_TOKENS")
    temperature: float = Field(0.7, env="TEMPERATURE")

    # Agent Configuration
    orchestrator_model: str = Field("openai:gpt-5-mini", env="ORCHESTRATOR_MODEL")
    youtube_model: str = Field("openai:gpt-5-mini", env="YOUTUBE_MODEL")
    tool_specialty_model: str = Field("openai:gpt-5-mini", env="TOOL_SPECIALTY_MODEL")

    # Crawling Configuration
    target_successful_scrapes: int = Field(15, env="TARGET_SUCCESSFUL_SCRAPES")
    max_total_urls_to_process: int = Field(50, env="MAX_TOTAL_URLS_TO_PROCESS")
    enable_success_based_termination: bool = Field(True, env="ENABLE_SUCCESS_BASED_TERMINATION")
    primary_batch_size: int = Field(16, env="PRIMARY_BATCH_SIZE")
    secondary_batch_size: int = Field(16, env="SECONDARY_BATCH_SIZE")
    pdf_processing_enabled: bool = Field(True, env="PDF_PROCESSING_ENABLED")

    # Original z-playground1 settings
    auto_crawl_top_default: int = Field(10, env="AUTO_CRAWL_TOP_DEFAULT")
    crawl_relevance_threshold: float = Field(0.15, env="CRAWL_RELEVANCE_THRESHOLD")
    concurrent_crawl_limit: int = Field(16, env="CONCURRENT_CRAWL_LIMIT")
    crawl_success_target: int = Field(15, env="CRAWL_SUCCESS_TARGET")
    crawl_timeout_seconds: float = Field(180.0, env="CRAWL_TIMEOUT_SECONDS")
    crawl_max_retries: int = Field(1, env="CRAWL_MAX_RETRIES")
    url_selection_limit_default: int = Field(10, env="URL_SELECTION_LIMIT_DEFAULT")

    # Rate Limiting
    max_requests_per_minute: int = Field(5000, env="MAX_REQUESTS_PER_MINUTE")
    max_tokens_per_minute: int = Field(4000000, env="MAX_TOKENS_PER_MINUTE")

    # Cache Settings
    cache_ttl: int = Field(300, env="CACHE_TTL")
    enable_caching: bool = Field(True, env="ENABLE_CACHING")

    # HTTP Client Settings
    http_timeout: float = Field(30.0, env="HTTP_TIMEOUT")
    http_max_retries: int = Field(3, env="HTTP_MAX_RETRIES")
    http_rate_limit: float = Field(83.3, env="HTTP_RATE_LIMIT")

    # Development Settings
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    development_mode: bool = Field(False, env="DEVELOPMENT_MODE")
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
```

### Enhanced Search Configuration (settings_broken.py)

Alternative configuration implementation with comprehensive search settings:

```python
@dataclass
class EnhancedSearchConfig:
    """Configuration for enhanced search functionality."""

    # Search settings
    default_num_results: int = 15
    default_auto_crawl_top: int = 10
    default_crawl_threshold: float = 0.3  # Fixed at 0.3 for better success rates
    default_anti_bot_level: int = 1
    default_max_concurrent: int = 0  # 0 => unbounded concurrency

    # Anti-bot levels
    anti_bot_levels = {
        0: "basic",      # 6/10 sites success
        1: "enhanced",   # 8/10 sites success
        2: "advanced",   # 9/10 sites success
        3: "stealth"     # 9.5/10 sites success
    }

    # Success-based scraping termination settings
    target_successful_scrapes: int = 15
    max_total_urls_to_process: int = 50
    enable_success_based_termination: bool = True
    url_deduplication_enabled: bool = True
    progressive_retry_enabled: bool = False
    max_retry_attempts: int = 1

    # Concurrent scraping batch configuration
    primary_batch_size: int = 16
    secondary_batch_size: int = 16
    batch_size_calculation_method: str = "adaptive"
    adaptive_batch_multiplier: float = 1.5

    # PDF processing configuration
    pdf_processing_enabled: bool = True
    pdf_extract_images: bool = False
    pdf_save_images_locally: bool = False
    pdf_batch_size: int = 4
    pdf_timeout: int = 30000
    pdf_content_min_length: int = 100

    # Token management
    max_response_tokens: int = 20000
    content_summary_threshold: int = 20000

    # Content cleaning settings
    default_cleanliness_threshold: float = 0.7
    min_content_length_for_cleaning: int = 500
    min_cleaned_content_length: int = 200

    # Crawl settings
    default_crawl_timeout: int = 30000
    max_concurrent_crawls: int = 0
    crawl_retry_attempts: int = 1
```

### Agent Definitions

#### Claude Agent SDK Integration (agents.py)

The system uses Claude Agent SDK AgentDefinition pattern:

```python
def get_research_agent_definition() -> AgentDefinition:
    """Define the Research Agent using SDK AgentDefinition pattern."""
    return AgentDefinition(
        description="Expert Research Agent specializing in comprehensive web research, source validation, and information synthesis",
        prompt="""You are a Research Agent, an expert in conducting comprehensive, high-quality research...

MANDATORY RESEARCH PROCESS:
1. IMMEDIATELY execute: conduct_research with the research topic
2. Set num_results to 15 for comprehensive coverage
3. Set auto_crawl_top to 8 for detailed content extraction
4. Set crawl_threshold to 0.3 for relevance filtering
5. Save research findings using save_report
6. Create structured search results using create_research_report""",
        tools=[
            "conduct_research",
            "analyze_sources",
            "generate_report",
            "save_report",
            "get_session_data",
            "create_research_report",
            "Read", "Write", "Edit", "Bash"
        ],
        model="sonnet"
    )
```

#### Enhanced Agent Definitions (enhanced_agents.py)

Advanced agent definitions with flow adherence enforcement:

```python
@dataclass
class EnhancedAgentDefinition:
    """Enhanced agent definition with SDK integration."""

    # Basic agent information
    agent_type: AgentType
    name: str
    description: str
    version: str = "1.0.0"

    # Model configuration
    model: str = "sonnet"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    # Tool configuration
    tools: List[ToolConfiguration] = field(default_factory=list)
    tool_access_level: str = "full"

    # Hooks configuration
    hooks: AgentHooksConfiguration = field(default_factory=AgentHooksConfiguration)

    # Quality configuration
    quality_gates: QualityGateConfiguration = field(default_factory=QualityGateConfiguration)

    # Flow adherence configuration
    flow_adherence: FlowAdherenceConfiguration = field(default_factory=FlowAdherenceConfiguration)
```

### SDK Configuration (sdk_config.py)

Comprehensive Claude Agent SDK configuration:

```python
@dataclass
class ClaudeAgentSDKConfig:
    """Main configuration class for Claude Agent SDK integration."""

    # Core SDK configuration
    claude_sdk_version: str = "0.1.3"
    default_model: str = "sonnet"
    max_tokens: int = 8192
    temperature: float = 0.7

    # Hooks configuration
    hooks: HooksConfiguration = field(default_factory=HooksConfiguration)

    # Observability configuration
    observability: ObservabilityConfiguration = field(default_factory=ObservabilityConfiguration)

    # Message processing configuration
    message_processing: MessageProcessingConfiguration = field(default_factory=MessageProcessingConfiguration)

    # Sub-agent configuration
    sub_agents: SubAgentConfiguration = field(default_factory=SubAgentConfiguration)

    # Enhanced search configuration
    search: EnhancedSearchConfiguration = field(default_factory=EnhancedSearchConfiguration)

    # Environment settings
    environment: Literal["development", "testing", "staging", "production"] = "development"
    debug_mode: bool = False
    enable_experimental_features: bool = False

    # API configuration
    anthropic_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
```

### Configuration Manager Integration

The unified configuration manager integrates all configuration types:

```python
class ConfigurationManager:
    """Unified configuration manager for the multi-agent research system."""

    def __init__(self):
        self._legacy_settings: Optional[Settings] = None
        self._sdk_config: Optional[ClaudeAgentSDKConfig] = None
        self._enhanced_agents: Optional[Dict[str, EnhancedAgentDefinition]] = None
        self._config_dir: Optional[Path] = None
        self._environment: Optional[str] = None

    def initialize(self, config_dir: Optional[Union[str, Path]] = None,
                  environment: Optional[str] = None) -> None:
        """Initialize the configuration manager."""

        # Load configurations
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load all configuration files."""
        # Load legacy settings
        self._load_legacy_settings()
        # Load SDK configuration
        self._load_sdk_config()
        # Load enhanced agent definitions
        self._load_enhanced_agents()
        # Apply environment-specific overrides
        self._apply_environment_overrides()
        # Validate configurations
        self._validate_configurations()
```

## Research Targets Configuration

### Scope-Based Target Configuration (research_targets.py)

```python
RESEARCH_TARGETS = {
    "brief": {
        "primary_clean_scrape_target": 10,
        "editorial_clean_scrape_target": 4,
        "attempt_multiplier": 1.5
    },
    "default": {
        "primary_clean_scrape_target": 15,
        "editorial_clean_scrape_target": 6,
        "attempt_multiplier": 1.5
    },
    "comprehensive": {
        "primary_clean_scrape_target": 20,
        "editorial_clean_scrape_target": 8,
        "attempt_multiplier": 1.5
    }
}
```

## Environment Variables

### Required Environment Variables

```bash
# API Keys (Required)
OPENAI_API_KEY=your-openai-key
SERPER_API_KEY=your-serper-key

# Optional API Keys
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
YOUTUBE_API_KEY=your-youtube-key

# System Configuration
LOG_LEVEL=INFO
MAX_TOKENS=8192
TEMPERATURE=0.7
DEBUG_MODE=false
DEVELOPMENT_MODE=false
ENABLE_METRICS=true

# Agent Models
ORCHESTRATOR_MODEL=openai:gpt-5-mini
YOUTUBE_MODEL=openai:gpt-5-mini
TOOL_SPECIALTY_MODEL=openai:gpt-5-mini
SEARCH_MODEL=openai:gpt-5-mini
CRAWL4AI_MODEL=openai:gpt-5-mini

# Crawling Configuration
TARGET_SUCCESSFUL_SCRAPES=15
MAX_TOTAL_URLS_TO_PROCESS=50
ENABLE_SUCCESS_BASED_TERMINATION=true
PRIMARY_BATCH_SIZE=16
SECONDARY_BATCH_SIZE=16
PDF_PROCESSING_ENABLED=true

# Original z-playground1 settings
AUTO_CRAWL_TOP_DEFAULT=10
CRAWL_RELEVANCE_THRESHOLD=0.15
CONCURRENT_CRAWL_LIMIT=16
CRAWL_SUCCESS_TARGET=15
CRAWL_TIMEOUT_SECONDS=180.0
CRAWL_MAX_RETRIES=1
URL_SELECTION_LIMIT_DEFAULT=10

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=5000
MAX_TOKENS_PER_MINUTE=4000000

# Cache Settings
CACHE_TTL=300
ENABLE_CACHING=true

# HTTP Client Settings
HTTP_TIMEOUT=30.0
HTTP_MAX_RETRIES=3
HTTP_RATE_LIMIT=83.3

# Enhanced Search Configuration
ENHANCED_SEARCH_NUM_RESULTS=15
ENHANCED_SEARCH_AUTO_CRAWL_TOP=10
ENHANCED_SEARCH_CRAWL_THRESHOLD=0.3
ENHANCED_SEARCH_ANTI_BOT_LEVEL=1
ENHANCED_SEARCH_MAX_CONCURRENT=0

# Path Configuration
KEVIN_BASE_DIR=/path/to/KEVIN
KEVIN_SESSIONS_DIR=/path/to/KEVIN/sessions
KEVIN_LOGS_DIR=/path/to/KEVIN/logs
KEVIN_WORKPRODUCTS_DIR=/path/to/KEVIN/work_products

# SDK Configuration
CLAUDE_DEFAULT_MODEL=sonnet
CLAUDE_MAX_TOKENS=8192
CLAUDE_TEMPERATURE=0.7
ENVIRONMENT=development
```

## Path Management

### KEVIN Directory Structure

The configuration system implements environment-aware path management:

```python
def _get_base_repo_dir(self) -> str:
    """Get base repository directory with environment-aware detection."""
    current_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if "claudeagent-multiagent-latest" in current_repo:
        return current_repo
    else:
        return "/home/kjdragan/lrepos/claudeagent-multiagent-latest"

@property
def kevin_base_dir(self) -> str:
    """Get KEVIN base directory with environment override support."""
    if hasattr(self, '_kevin_base_dir_override') and self._kevin_base_dir_override:
        return self._kevin_base_dir_override
    return f"{self._get_base_repo_dir()}/KEVIN"

def get_session_dir(self, session_id: str) -> str:
    """Get session directory path."""
    return f"{self.kevin_sessions_dir}/{session_id}"

def ensure_session_directory(self, session_id: str) -> dict[str, Path]:
    """Ensure all session directories exist and return paths."""
    base_dir = Path(self.get_session_dir(session_id))
    working_dir = Path(self.get_session_working_dir(session_id))
    research_dir = Path(self.get_session_research_dir(session_id))
    final_dir = Path(self.get_session_final_dir(session_id))

    # Create all directories
    base_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    research_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    return {
        "session": base_dir,
        "working": working_dir,
        "research": research_dir,
        "final": final_dir
    }
```

## Configuration Loading and Validation

### Settings Loading

```python
# Global settings instance
try:
    settings = Settings()
    settings.setup_logging()

    # Log configuration status
    logger = logging.getLogger(__name__)
    logger.info("Settings loaded successfully")

    missing_keys = settings.get_missing_api_keys()
    if missing_keys:
        logger.warning(f"Missing required API keys: {missing_keys}")
    else:
        logger.info("All required API keys are configured")

except Exception as e:
    # Fallback settings for development
    print(f"Warning: Could not load settings from environment: {e}")
    print("Using default settings. Please configure your .env file.")

    class DefaultSettings:
        openai_api_key = "not_configured"
        serper_api_key = "not_configured"
        log_level = "INFO"
        debug_mode = True
        development_mode = True
        target_successful_scrapes = 15
        concurrent_crawl_limit = 16

        def setup_logging(self):
            logging.basicConfig(level=logging.INFO)

        def get_missing_api_keys(self):
            return ["OPENAI_API_KEY", "SERPER_API_KEY"]

    settings = DefaultSettings()
    settings.setup_logging()
```

### Configuration Validation

```python
def _validate_configuration(self):
    """Validate configuration values."""
    # Validate temperature
    if not 0.0 <= self.temperature <= 2.0:
        raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")

    # Validate max_tokens
    if self.max_tokens <= 0:
        raise ValueError(f"Max tokens must be positive, got {self.max_tokens}")

    # Validate search thresholds
    if not 0.0 <= self.search.default_crawl_threshold <= 1.0:
        raise ValueError(f"Crawl threshold must be between 0.0 and 1.0, got {self.search.default_crawl_threshold}")

    # Validate required API keys based on environment
    if self.environment == "production" and self._is_active_instance():
        missing_keys = []
        if not self.anthropic_api_key:
            missing_keys.append("ANTHROPIC_API_KEY")
        if not self.serper_api_key:
            missing_keys.append("SERPER_API_KEY")

        if missing_keys:
            raise ValueError(f"Missing required API keys for production: {missing_keys}")
```

## Usage Examples

### Basic Configuration Usage

```python
from config.settings import get_settings, get_enhanced_search_config
from config.agents import get_research_agent_definition, get_all_agent_definitions

# Load configuration
settings = get_settings()
search_config = get_enhanced_search_config()

# Get agent definition
research_agent = get_research_agent_definition()
print(f"Research agent: {research_agent.description}")
print(f"Available tools: {research_agent.tools}")

# Get all agent definitions
all_agents = get_all_agent_definitions()
for name, definition in all_agents.items():
    print(f"{name}: {definition.description}")
```

### SDK Configuration Usage

```python
from config.sdk_config import get_sdk_config, ClaudeAgentSDKConfig

# Get SDK configuration
sdk_config = get_sdk_config()

# Create custom configuration
custom_config = ClaudeAgentSDKConfig(
    environment="production",
    debug_mode=False,
    default_model="sonnet",
    max_tokens=16384
)

# Load configuration from file
loaded_config = ClaudeAgentSDKConfig.load_from_file("config.json")
```

### Enhanced Agent Configuration

```python
from config.enhanced_agents import create_enhanced_agent, AgentType

# Create enhanced research agent
research_agent = create_enhanced_agent(AgentType.RESEARCH)

# Create enhanced editorial agent with flow adherence
editorial_agent = create_enhanced_agent(AgentType.EDITORIAL)

# Get all enhanced agents
all_enhanced_agents = get_all_enhanced_agent_definitions()
```

### Unified Configuration Manager

```python
from config.config_manager import initialize_configuration, get_config_manager

# Initialize configuration system
config_manager = initialize_configuration(
    config_dir="/path/to/config",
    environment="production"
)

# Get configuration summary
summary = config_manager.get_configuration_summary()
print(f"Environment: {summary['environment']}")
print(f"SDK Model: {summary['sdk_config']['model']}")

# Validate system configuration
validation_result = validate_system_configuration()
if validation_result['compatible']:
    print("System configuration is valid")
else:
    print(f"Configuration issues: {validation_result['issues']}")
```

## Environment-Specific Configuration

### Development Configuration

```python
DEVELOPMENT_CONFIG = ClaudeAgentSDKConfig(
    environment="development",
    debug_mode=True,
    observability=ObservabilityConfiguration(
        log_level=LogLevel.DEBUG,
        enable_detailed_hooks_logging=True,
        enable_performance_metrics=True
    ),
    search=EnhancedSearchConfiguration(
        default_num_results=5,  # Reduced for development
        target_successful_scrapes=3
    )
)
```

### Production Configuration

```python
PRODUCTION_CONFIG = ClaudeAgentSDKConfig(
    environment="production",
    debug_mode=False,
    observability=ObservabilityConfiguration(
        log_level=LogLevel.INFO,
        enable_detailed_hooks_logging=False,
        enable_performance_metrics=True
    ),
    search=EnhancedSearchConfiguration(
        default_num_results=20,
        target_successful_scrapes=25,
        default_anti_bot_level=AntiBotLevel.ADVANCED
    )
)
```

### Testing Configuration

```python
TESTING_CONFIG = ClaudeAgentSDKConfig(
    environment="testing",
    debug_mode=True,
    observability=ObservabilityConfiguration(
        log_level=LogLevel.DEBUG,
        enable_performance_metrics=True
    ),
    search=EnhancedSearchConfiguration(
        default_num_results=3,
        target_successful_scrapes=2,
        default_anti_bot_level=AntiBotLevel.BASIC
    )
)
```

## Configuration Debugging

### Debug Information

```python
def get_debug_info(self) -> dict[str, Any]:
    """Get debug information about current configuration."""
    return {
        "enhanced_search_config": {
            "default_num_results": self._enhanced_search_config.default_num_results,
            "default_auto_crawl_top": self._enhanced_search_config.default_auto_crawl_top,
            "default_crawl_threshold": self._enhanced_search_config.default_crawl_threshold,
            "default_anti_bot_level": self._enhanced_search_config.default_anti_bot_level,
            "default_max_concurrent": self._enhanced_search_config.default_max_concurrent,
            "workproduct_dir": self._enhanced_search_config.default_workproduct_dir,
        },
        "environment_variables": {
            "SERP_API_KEY": "SET" if os.getenv('SERP_API_KEY') else "NOT_SET",
            "OPENAI_API_KEY": "SET" if os.getenv('OPENAI_API_KEY') else "NOT_SET",
            "ANTHROPIC_API_KEY": "SET" if os.getenv('ANTHROPIC_API_KEY') else "NOT_SET",
            "KEVIN_BASE_DIR": os.getenv('KEVIN_BASE_DIR', 'NOT_SET'),
        },
        "path_validation": {
            "kevin_base_exists": Path(self.get_kevin_base_dir()).exists(),
            "sessions_dir_exists": Path(self.get_kevin_sessions_dir()).exists(),
            "logs_dir_exists": Path(self.get_kevin_logs_dir()).exists(),
        }
    }
```

## Configuration Validation

### Agent Compatibility Validation

```python
def validate_agent_compatibility(self) -> Dict[str, Any]:
    """Validate that enhanced agents are compatible with current configuration."""
    validation_results = {
        "compatible": True,
        "issues": [],
        "warnings": [],
        "agent_status": {}
    }

    for agent_name, agent_def in self._enhanced_agents.items():
        agent_status = {
            "compatible": True,
            "issues": [],
            "warnings": []
        }

        # Check model compatibility
        if agent_def.model not in ["sonnet", "haiku", "opus"]:
            agent_status["warnings"].append(f"Unusual model specified: {agent_def.model}")

        # Check tool availability
        required_tools = [tool.tool_name for tool in agent_def.tools if tool.execution_policy.value == "mandatory"]
        if required_tools:
            agent_status["required_tools"] = required_tools

        # Check timeout configuration
        if agent_def.timeout_seconds > 600:  # 10 minutes
            agent_status["warnings"].append(f"Long timeout: {agent_def.timeout_seconds}s")

        validation_results["agent_status"][agent_name] = agent_status

    return validation_results
```

## Integration with System Components

### Core System Integration

```python
# In core/orchestrator.py
from config.agents import get_all_agent_definitions
from config.settings import get_settings

class ResearchOrchestrator:
    def __init__(self, debug_mode: bool = False):
        self.settings = get_settings()
        self.agent_definitions = get_all_agent_definitions()

    async def initialize_agent(self, agent_type: str):
        """Initialize agent with configuration."""
        if agent_type in self.agent_definitions:
            definition = self.agent_definitions[agent_type]
            return await self._create_sdk_agent(definition)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
```

### Enhanced Search Integration

```python
# In utils/enhanced_search_utils.py
from config.settings_broken import get_enhanced_search_config

def get_search_parameters():
    """Get search parameters from configuration."""
    config = get_enhanced_search_config()
    return {
        "num_results": config.default_num_results,
        "auto_crawl_top": config.default_auto_crawl_top,
        "crawl_threshold": config.default_crawl_threshold,
        "anti_bot_level": config.default_anti_bot_level,
        "max_concurrent": config.default_max_concurrent,
    }
```

## Best Practices

### Configuration Management

1. **Environment Variables First**: Always use environment variables for deployment-specific configuration
2. **Default Values**: Provide sensible defaults for all configuration options
3. **Validation**: Validate configuration values at startup and during runtime
4. **Type Safety**: Use proper type hints and validation for configuration values
5. **Fallback Handling**: Implement graceful fallbacks when configuration loading fails

### Path Management

1. **Environment-Aware Paths**: Use environment-aware path detection for different deployment scenarios
2. **Session-Based Organization**: Organize files by session ID for better management
3. **Override Support**: Allow path overrides through environment variables
4. **Directory Creation**: Automatically create necessary directories

### Agent Configuration

1. **SDK Integration**: Use Claude Agent SDK patterns for agent definitions
2. **Tool Configuration**: Properly configure tool access and execution policies
3. **Flow Adherence**: Implement flow adherence validation where needed
4. **Quality Gates**: Configure quality assessment and enhancement

## Troubleshooting

### Common Configuration Issues

1. **Missing Environment Variables**
   ```python
   debug_info = settings.get_debug_info()
   missing_vars = [k for k, v in debug_info["environment_variables"].items() if v == "NOT_SET"]
   print(f"Missing environment variables: {missing_vars}")
   ```

2. **Invalid Configuration Values**
   ```python
   level = 5  # Invalid anti-bot level
   validated = settings.validate_anti_bot_level(level)
   print(f"Validated level: {validated}")  # Will be clamped to 3
   ```

3. **Path Issues**
   ```python
   try:
       work_dir = settings.ensure_session_directory("test-session")
       print(f"Session directory: {work_dir}")
   except Exception as e:
       print(f"Path creation failed: {e}")
   ```

4. **API Key Issues**
   ```python
   missing_keys = settings.get_missing_api_keys()
   if missing_keys:
       print(f"Missing API keys: {missing_keys}")
   ```

## Testing Configuration

### Configuration Testing

```python
def test_configuration_loading():
    """Test configuration loading and validation."""
    settings = get_settings()

    # Test enhanced search config
    assert settings.enhanced_search.default_num_results > 0
    assert 0 <= settings.enhanced_search.default_crawl_threshold <= 1
    assert 0 <= settings.enhanced_search.default_anti_bot_level <= 3

def test_environment_overrides():
    """Test environment variable overrides."""
    import os

    # Set test environment variables
    os.environ['ENHANCED_SEARCH_NUM_RESULTS'] = '25'
    os.environ['ENHANCED_SEARCH_ANTI_BOT_LEVEL'] = '2'

    # Reload configuration
    settings = SettingsManager()

    assert settings.enhanced_search.default_num_results == 25
    assert settings.enhanced_search.default_anti_bot_level == 2
```

## Critical Configuration Issues

### Tool Registration Problems ❌

**Issue**: Corpus tools are defined in `enhanced_agents.py:285-425` but never registered with the SDK client

**Root Causes**:
- Tools are behind an `if SDK_AVAILABLE` guard that prevents registration
- Missing MCP server creation with `create_sdk_mcp_server()`
- Missing `mcp_servers` and `allowed_tools` configuration in SDK options
- Tool definitions exist but are not integrated into the runtime

**Impact**: Enhanced report agent fails hook validation because required corpus tools are not available

**Files Affected**:
- `config/enhanced_agents.py` (tool definitions exist but not registered)
- Missing: `mcp_tools/corpus_tools.py` (proper MCP server implementation)
- Missing: `core/sdk_client_manager.py` (SDK client with corpus server registration)

### SDK Integration Gaps ❌

**Current Status**:
- ✅ Search tools properly registered and functional
- ❌ Corpus tools defined but never registered
- ❌ Agent definitions reference non-existent tools
- ❌ No SDK client manager for proper tool registration

### Hook Validation Configuration Issues ❌

**Problem**: Hook validation system requires tools that agents don't have access to

**Required Tools (Missing from Agent Toolkit)**:
- `build_research_corpus`
- `analyze_research_corpus`
- `synthesize_from_corpus`
- `generate_comprehensive_report`

**Available Tools (Used Instead)**:
- `get_session_data`
- `create_research_report`
- `Write`

## System Status

### Current Implementation Status: ⚠️ Partially Functional Configuration System

- **Multi-Layer Configuration**: ✅ Working integration of legacy, SDK, and enhanced configurations
- **Environment Variable Support**: ✅ Comprehensive environment variable management
- **Path Management**: ✅ Environment-aware path detection and session-based organization
- **Agent Definitions**: ⚠️ Claude Agent SDK integration with tool configuration (tools missing registration)
- **Validation System**: ⚠️ Configuration validation works, but hook validation fails due to missing tools
- **Debug Support**: ✅ Comprehensive debugging information and validation tools

### Critical Issues Requiring Immediate Fix

1. **Tool Registration Failure**: Corpus tools exist but aren't registered with SDK client
2. **SDK Integration Gap**: Missing MCP server creation and SDK options configuration
3. **Hook Validation Mismatch**: Required tools don't exist in agent toolkits
4. **Coroutine Usage**: Tool wrappers call async functions without await
5. **No Error Recovery**: System cannot proceed when validation fails

### Known Limitations

- **Multiple Configuration Systems**: Multiple overlapping configuration implementations
- **Complex Integration**: Complex integration between legacy and enhanced configurations
- **Tool Registration Gap**: Critical missing piece preventing end-to-end workflows
- **Testing Coverage**: Limited test coverage for all configuration scenarios

### Next Steps for Configuration System

1. **Implement Corpus MCP Server**: Create proper MCP server for corpus tools
2. **Fix SDK Client Registration**: Register corpus server with SDK options
3. **Update Agent Definitions**: Include corpus tools in agent configurations
4. **Add Error Recovery**: Implement fallback strategies for validation failures
5. **Test Integration**: Validate that all tools are properly registered and functional

This documentation reflects the actual configuration system implementation based on analysis of the real code files and critical issues identified in the current system state.