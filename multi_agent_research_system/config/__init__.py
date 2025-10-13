"""Configuration modules for the multi-agent research system."""

from .settings import Settings, get_settings

# Legacy agent definitions
from .agents import (
    create_agent_config_file,
    get_all_agent_definitions,
    get_editor_agent_definition,
    get_report_agent_definition,
    get_research_agent_definition,
    get_ui_coordinator_definition,
)

# Enhanced SDK-based configuration
from .sdk_config import (
    ClaudeAgentSDKConfig,
    get_sdk_config,
    set_sdk_config,
    load_sdk_config_from_file,
    save_sdk_config_to_file,
    DEVELOPMENT_CONFIG,
    PRODUCTION_CONFIG,
    TESTING_CONFIG,
)

from .enhanced_agents import (
    EnhancedAgentDefinition,
    EnhancedAgentFactory,
    AgentType,
    ToolConfiguration,
    create_enhanced_agent,
    get_all_enhanced_agent_definitions,
)

from .config_manager import (
    ConfigurationManager,
    get_config_manager,
    initialize_configuration,
    get_configuration_summary,
    validate_system_configuration,
)

__all__ = [
    # Legacy configuration
    "Settings", "get_settings",
    "get_research_agent_definition", "get_report_agent_definition",
    "get_editor_agent_definition", "get_ui_coordinator_definition",
    "get_all_agent_definitions", "create_agent_config_file",

    # Enhanced SDK configuration
    "ClaudeAgentSDKConfig", "get_sdk_config", "set_sdk_config",
    "load_sdk_config_from_file", "save_sdk_config_to_file",
    "DEVELOPMENT_CONFIG", "PRODUCTION_CONFIG", "TESTING_CONFIG",

    # Enhanced agents
    "EnhancedAgentDefinition", "EnhancedAgentFactory", "AgentType",
    "ToolConfiguration", "create_enhanced_agent", "get_all_enhanced_agent_definitions",

    # Configuration management
    "ConfigurationManager", "get_config_manager", "initialize_configuration",
    "get_configuration_summary", "validate_system_configuration",
]
