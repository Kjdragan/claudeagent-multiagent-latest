"""
Enhanced Claude Agent SDK Configuration System

This module provides comprehensive configuration management for the Claude Agent SDK
integration, including advanced options patterns, hooks configuration, and observability
settings based on the redesign plan specifications.

Key Features:
- ClaudeAgentPatterns configuration
- Comprehensive hooks system setup
- Rich message processing configuration
- Observability and monitoring settings
- Sub-agent architecture configuration
- Development and production environment support

Based on Redesign Plan PLUS SDK Implementation (October 13, 2025)
"""

import os
from typing import Optional, Dict, Any, List, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json


class LogLevel(str, Enum):
    """Supported log levels for the system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AntiBotLevel(int, Enum):
    """Anti-bot detection levels for web scraping."""
    BASIC = 0      # 6/10 sites success
    ENHANCED = 1   # 8/10 sites success
    ADVANCED = 2   # 9/10 sites success
    STEALTH = 3    # 9.5/10 sites success


class MessageDisplayFormat(str, Enum):
    """Supported message display formats."""
    RICH_MARKDOWN = "rich_markdown"
    PLAIN_TEXT = "plain_text"
    STRUCTURED = "structured"
    DEBUG = "debug"


@dataclass
class HooksConfiguration:
    """Configuration for comprehensive hooks system."""

    # Pre-execution hooks
    pre_tool_hooks: List[str] = field(default_factory=lambda: [
        "validate_tool_parameters",
        "check_resource_availability",
        "log_tool_execution_start",
        "enforce_flow_adherence"
    ])

    # Post-execution hooks
    post_tool_hooks: List[str] = field(default_factory=lambda: [
        "analyze_tool_results",
        "log_tool_execution_complete",
        "update_performance_metrics",
        "validate_quality_standards"
    ])

    # Message processing hooks
    message_hooks: List[str] = field(default_factory=lambda: [
        "parse_message_structure",
        "extract_key_information",
        "monitor_message_flow",
        "track_agent_reasoning"
    ])

    # Error handling hooks
    error_hooks: List[str] = field(default_factory=lambda: [
        "log_error_details",
        "attempt_error_recovery",
        "escalate_if_needed",
        "update_error_metrics"
    ])

    # Sub-agent coordination hooks
    sub_agent_hooks: List[str] = field(default_factory=lambda: [
        "coordinate_agent_handoff",
        "validate_agent_completion",
        "aggregate_agent_results",
        "manage_agent_state"
    ])


@dataclass
class ObservabilityConfiguration:
    """Configuration for comprehensive observability infrastructure."""

    # Metrics collection
    enable_performance_metrics: bool = True
    enable_quality_metrics: bool = True
    enable_flow_compliance_metrics: bool = True
    metrics_collection_interval: float = 1.0  # seconds

    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    enable_structured_logging: bool = True
    enable_detailed_hooks_logging: bool = True
    log_agent_reasoning: bool = True
    log_tool_execution_details: bool = True

    # Message processing tracking
    enable_message_tracking: bool = True
    message_storage_format: MessageDisplayFormat = MessageDisplayFormat.RICH_MARKDOWN
    retain_message_history: bool = True
    max_message_history_size: int = 10000

    # Performance monitoring
    enable_resource_monitoring: bool = True
    enable_memory_tracking: bool = True
    enable_execution_time_tracking: bool = True
    performance_alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "tool_execution_time": 30.0,  # seconds
        "memory_usage_mb": 1024.0,    # MB
        "error_rate": 0.05           # 5%
    })

    # Export and analysis
    enable_metrics_export: bool = True
    metrics_export_format: Literal["json", "prometheus", "csv"] = "json"
    export_interval: float = 60.0  # seconds


@dataclass
class MessageProcessingConfiguration:
    """Configuration for rich message processing."""

    # Processing options
    enable_rich_parsing: bool = True
    enable_content_extraction: bool = True
    enable_metadata_extraction: bool = True
    enable_reasoning_analysis: bool = True

    # Display options
    default_display_format: MessageDisplayFormat = MessageDisplayFormat.RICH_MARKDOWN
    enable_interactive_elements: bool = True
    enable_syntax_highlighting: bool = True

    # Analysis options
    enable_content_analysis: bool = True
    enable_sentiment_analysis: bool = False
    enable_entity_extraction: bool = True
    enable_topic_modeling: bool = False

    # Storage options
    store_processed_messages: bool = True
    message_storage_path: Optional[str] = None
    compress_old_messages: bool = True
    message_retention_days: int = 30


@dataclass
class SubAgentConfiguration:
    """Configuration for sub-agent architecture."""

    # Agent coordination
    enable_sub_agent_coordination: bool = True
    max_concurrent_sub_agents: int = 5
    sub_agent_timeout: float = 300.0  # seconds

    # Agent handoff
    enable_automatic_handoff: bool = True
    handoff_validation_required: bool = True
    handoff_timeout: float = 30.0  # seconds

    # Result aggregation
    enable_result_aggregation: bool = True
    aggregation_strategy: Literal["merge", "vote", "priority", "custom"] = "merge"
    conflict_resolution: Literal["first", "best", "merge", "manual"] = "best"

    # Agent specialization
    enable_specialized_agents: bool = True
    specializations: List[str] = field(default_factory=lambda: [
        "research",
        "analysis",
        "synthesis",
        "validation",
        "formatting"
    ])


@dataclass
class EnhancedSearchConfiguration:
    """Enhanced search configuration with anti-bot and target-based scraping."""

    # Search parameters
    default_num_results: int = 15
    default_auto_crawl_top: int = 10
    default_crawl_threshold: float = 0.3
    default_max_concurrent: int = 10

    # Anti-bot configuration
    default_anti_bot_level: AntiBotLevel = AntiBotLevel.ENHANCED
    progressive_anti_bot_enabled: bool = True
    max_anti_bot_retry_attempts: int = 3

    # Target-based scraping
    target_successful_scrapes: int = 15
    max_total_urls_to_process: int = 50
    enable_success_based_termination: bool = True
    url_deduplication_enabled: bool = True

    # Content processing
    enable_ai_content_cleaning: bool = True
    content_cleanliness_threshold: float = 0.7
    min_content_length_for_cleaning: int = 500

    # Budget management
    total_budget_scrapes: int = 30
    total_budget_queries: int = 10
    emergency_reserve_scrapes: int = 5
    emergency_reserve_queries: int = 2


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

    # Paths and directories
    workproducts_directory: Optional[str] = None
    logs_directory: Optional[str] = None
    session_data_directory: Optional[str] = None

    def __post_init__(self):
        """Post-initialization setup and validation."""
        # Load environment variables
        self._load_environment_variables()

        # Set default paths
        self._set_default_paths()

        # Validate configuration
        self._validate_configuration()

    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        # API keys
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or self.anthropic_api_key
        self.serper_api_key = os.getenv("SERPER_API_KEY") or self.serper_api_key
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or self.openai_api_key

        # Environment
        env = os.getenv("ENVIRONMENT", "").lower()
        if env in ["development", "testing", "staging", "production"]:
            self.environment = env

        # Debug mode
        if os.getenv("DEBUG_MODE", "").lower() in ["true", "1", "yes"]:
            self.debug_mode = True

        # Log level
        log_level = os.getenv("LOG_LEVEL", "").upper()
        if log_level in [level.value for level in LogLevel]:
            self.observability.log_level = LogLevel(log_level)

        # Model settings
        if os.getenv("CLAUDE_DEFAULT_MODEL"):
            self.default_model = os.getenv("CLAUDE_DEFAULT_MODEL")

        if os.getenv("CLAUDE_MAX_TOKENS"):
            try:
                self.max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS"))
            except ValueError:
                pass

        if os.getenv("CLAUDE_TEMPERATURE"):
            try:
                self.temperature = float(os.getenv("CLAUDE_TEMPERATURE"))
            except ValueError:
                pass

        # Search settings
        if os.getenv("ENHANCED_SEARCH_NUM_RESULTS"):
            try:
                self.search.default_num_results = int(os.getenv("ENHANCED_SEARCH_NUM_RESULTS"))
            except ValueError:
                pass

        if os.getenv("ENHANCED_SEARCH_ANTI_BOT_LEVEL"):
            try:
                level = int(os.getenv("ENHANCED_SEARCH_ANTI_BOT_LEVEL"))
                if 0 <= level <= 3:
                    self.search.default_anti_bot_level = AntiBotLevel(level)
            except ValueError:
                pass

    def _set_default_paths(self):
        """Set default paths based on environment."""
        if not self.workproducts_directory:
            # Default to KEVIN/workproducts
            base_dir = Path.cwd()
            if "claudeagent-multiagent-latest" in str(base_dir):
                self.workproducts_directory = str(base_dir / "KEVIN" / "workproducts")
            else:
                self.workproducts_directory = str(base_dir / "KEVIN" / "workproducts")

        if not self.logs_directory:
            base_dir = Path.cwd()
            self.logs_directory = str(base_dir / "KEVIN" / "logs")

        if not self.session_data_directory:
            base_dir = Path.cwd()
            self.session_data_directory = str(base_dir / "KEVIN" / "sessions")

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

        if not 0.0 <= self.search.content_cleanliness_threshold <= 1.0:
            raise ValueError(f"Content cleanliness threshold must be between 0.0 and 1.0, got {self.search.content_cleanliness_threshold}")

        # Validate required API keys based on environment (only for active instances)
        if self.environment == "production" and self._is_active_instance():
            missing_keys = []
            if not self.anthropic_api_key:
                missing_keys.append("ANTHROPIC_API_KEY")
            if not self.serper_api_key:
                missing_keys.append("SERPER_API_KEY")

            if missing_keys:
                raise ValueError(f"Missing required API keys for production: {missing_keys}")

    def _is_active_instance(self) -> bool:
        """Check if this is an active instance being used (not just a module import)."""
        # Simple heuristic: if we're in a setup script or testing, don't enforce API key validation
        import sys
        return "setup_development_environment.py" not in " ".join(sys.argv)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}

        # Convert simple fields
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                result[key] = value
            elif isinstance(value, (LogLevel, AntiBotLevel, MessageDisplayFormat)):
                result[key] = value.value
            elif isinstance(value, (HooksConfiguration, ObservabilityConfiguration,
                                  MessageProcessingConfiguration, SubAgentConfiguration,
                                  EnhancedSearchConfiguration)):
                result[key] = value.__dict__

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaudeAgentSDKConfig":
        """Create configuration from dictionary."""
        # Extract nested configurations
        hooks_data = data.pop("hooks", {})
        observability_data = data.pop("observability", {})
        message_processing_data = data.pop("message_processing", {})
        sub_agents_data = data.pop("sub_agents", {})
        search_data = data.pop("search", {})

        # Create configuration
        config = cls(**data)

        # Update nested configurations
        for key, value in hooks_data.items():
            if hasattr(config.hooks, key):
                setattr(config.hooks, key, value)

        for key, value in observability_data.items():
            if hasattr(config.observability, key):
                if key == "log_level" and isinstance(value, str):
                    setattr(config.observability, key, LogLevel(value.upper()))
                elif key == "message_storage_format" and isinstance(value, str):
                    setattr(config.observability, key, MessageDisplayFormat(value))
                else:
                    setattr(config.observability, key, value)

        for key, value in message_processing_data.items():
            if hasattr(config.message_processing, key):
                if key == "default_display_format" and isinstance(value, str):
                    setattr(config.message_processing, key, MessageDisplayFormat(value))
                else:
                    setattr(config.message_processing, key, value)

        for key, value in sub_agents_data.items():
            if hasattr(config.sub_agents, key):
                setattr(config.sub_agents, key, value)

        for key, value in search_data.items():
            if hasattr(config.search, key):
                if key == "default_anti_bot_level" and isinstance(value, int):
                    setattr(config.search, key, AntiBotLevel(value))
                else:
                    setattr(config.search, key, value)

        return config

    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "ClaudeAgentSDKConfig":
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)


# Global configuration instance
_global_config: Optional[ClaudeAgentSDKConfig] = None


def get_sdk_config() -> ClaudeAgentSDKConfig:
    """Get the global SDK configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ClaudeAgentSDKConfig()
    return _global_config


def set_sdk_config(config: ClaudeAgentSDKConfig):
    """Set the global SDK configuration instance."""
    global _global_config
    _global_config = config


def load_sdk_config_from_file(file_path: Union[str, Path]) -> ClaudeAgentSDKConfig:
    """Load SDK configuration from file and set as global instance."""
    config = ClaudeAgentSDKConfig.load_from_file(file_path)
    set_sdk_config(config)
    return config


def save_sdk_config_to_file(file_path: Union[str, Path]):
    """Save current SDK configuration to file."""
    config = get_sdk_config()
    config.save_to_file(file_path)


# Environment-specific configuration presets
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