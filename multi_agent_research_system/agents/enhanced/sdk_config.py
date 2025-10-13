"""Comprehensive SDK Options Configuration System

This module provides comprehensive configuration management for Claude Agent SDK
options, supporting agent-specific configurations, presets, and dynamic
configuration management.

Key Features:
- Comprehensive SDK Options Management
- Agent-Specific Configuration
- Configuration Presets and Templates
- Dynamic Configuration Updates
- Environment Variable Integration
- Configuration Validation
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from claude_agent_sdk import ClaudeAgentOptions


class LogLevel(Enum):
    """Logging levels for SDK configuration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ExecutionMode(Enum):
    """Execution modes for agents."""
    STANDARD = "standard"
    FAST = "fast"
    THOROUGH = "thorough"
    DEBUG = "debug"
    PERFORMANCE = "performance"


@dataclass
class PerformanceConfig:
    """Performance-related configuration."""
    max_concurrent_requests: int = 5
    request_timeout_seconds: int = 300
    connection_pool_size: int = 10
    keep_alive_timeout: int = 30
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    circuit_breaker_threshold: float = 0.5
    circuit_breaker_timeout: int = 300
    enable_compression: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class SecurityConfig:
    """Security-related configuration."""
    enable_encryption: bool = False
    api_key_rotation_enabled: bool = False
    request_signing: bool = False
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    max_request_size_mb: int = 10
    rate_limit_requests_per_minute: int = 60
    enable_audit_logging: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = True
    tracing_sample_rate: float = 0.1
    log_level: LogLevel = LogLevel.INFO
    enable_profiling: bool = False
    health_check_interval_seconds: int = 30
    performance_report_interval_seconds: int = 300


@dataclass
class ToolConfig:
    """Tool-specific configuration."""
    enabled_tools: List[str] = field(default_factory=list)
    disabled_tools: List[str] = field(default_factory=list)
    tool_timeouts: Dict[str, int] = field(default_factory=dict)
    tool_retry_attempts: Dict[str, int] = field(default_factory=dict)
    custom_tool_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class MCPConfig:
    """Model Context Protocol configuration."""
    enabled_servers: List[str] = field(default_factory=list)
    server_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_timeout_seconds: int = 60
    connection_retry_attempts: int = 3
    enable_server_discovery: bool = True
    server_health_check_interval: int = 60


@dataclass
class ComprehensiveSDKConfig:
    """Comprehensive SDK configuration."""
    # Basic SDK Options
    max_turns: int = 50
    continue_conversation: bool = True
    include_partial_messages: bool = True
    enable_hooks: bool = True
    system_prompt: Optional[str] = None

    # Execution Configuration
    execution_mode: ExecutionMode = ExecutionMode.STANDARD
    timeout_seconds: int = 300
    parallel_execution: bool = False
    max_parallel_tasks: int = 3

    # Tool Configuration
    tools: ToolConfig = field(default_factory=ToolConfig)

    # MCP Configuration
    mcp: MCPConfig = field(default_factory=MCPConfig)

    # Sub-Configuration Objects
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Agent-Specific Settings
    agent_type: str = "generic"
    agent_id: Optional[str] = None
    session_management_enabled: bool = True
    state_persistence_enabled: bool = False

    # Quality and Validation
    quality_threshold: float = 0.75
    enable_content_validation: bool = True
    enable_output_filtering: bool = False
    max_response_length: Optional[int] = None

    # Development and Debugging
    debug_mode: bool = False
    enable_request_logging: bool = False
    enable_response_logging: bool = False
    save_conversation_history: bool = False
    custom_options: Dict[str, Any] = field(default_factory=dict)

    def to_sdk_options(self) -> ClaudeAgentOptions:
        """Convert to ClaudeAgentOptions."""
        return ClaudeAgentOptions(
            max_turns=self.max_turns,
            continue_conversation=self.continue_conversation,
            include_partial_messages=self.include_partial_messages,
            enable_hooks=self.enable_hooks,
            system_prompt=self.system_prompt,
            timeout=self.timeout_seconds,
            allowed_tools=self.tools.enabled_tools,
            blocked_tools=self.tools.disabled_tools,
            mcp_servers=self.mcp.server_configs if self.mcp.enabled_servers else None,
            **self.custom_options
        )

    def merge_with(self, other: "ComprehensiveSDKConfig") -> "ComprehensiveSDKConfig":
        """Merge this configuration with another, with other taking precedence."""
        result = ComprehensiveSDKConfig()

        # Merge basic fields
        for field_name in self.__dataclass_fields__:
            current_value = getattr(self, field_name)
            other_value = getattr(other, field_name)

            if isinstance(current_value, dict) and isinstance(other_value, dict):
                # Merge dictionaries
                merged = current_value.copy()
                merged.update(other_value)
                setattr(result, field_name, merged)
            elif hasattr(current_value, 'merge_with') and hasattr(other_value, 'merge_with'):
                # Merge nested config objects
                setattr(result, field_name, current_value.merge_with(other_value))
            elif other_value is not None:
                # Use other value if not None
                setattr(result, field_name, other_value)
            else:
                # Use current value
                setattr(result, field_name, current_value)

        return result

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Basic validation
        if self.max_turns <= 0:
            issues.append("max_turns must be positive")

        if self.timeout_seconds <= 0:
            issues.append("timeout_seconds must be positive")

        if not 0 <= self.quality_threshold <= 1:
            issues.append("quality_threshold must be between 0 and 1")

        if self.max_parallel_tasks <= 0:
            issues.append("max_parallel_tasks must be positive")

        # Performance validation
        if self.performance.max_concurrent_requests <= 0:
            issues.append("performance.max_concurrent_requests must be positive")

        if self.performance.request_timeout_seconds <= 0:
            issues.append("performance.request_timeout_seconds must be positive")

        # Security validation
        if self.security.max_request_size_mb <= 0:
            issues.append("security.max_request_size_mb must be positive")

        if self.security.rate_limit_requests_per_minute <= 0:
            issues.append("security.rate_limit_requests_per_minute must be positive")

        # Tool validation
        disabled_tools_set = set(self.tools.disabled_tools)
        enabled_tools_set = set(self.tools.enabled_tools)

        overlap = disabled_tools_set.intersection(enabled_tools_set)
        if overlap:
            issues.append(f"Tools both enabled and disabled: {overlap}")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComprehensiveSDKConfig":
        """Create from dictionary."""
        # Handle nested config objects
        if "performance" in data:
            data["performance"] = PerformanceConfig(**data["performance"])
        if "security" in data:
            data["security"] = SecurityConfig(**data["security"])
        if "monitoring" in data:
            data["monitoring"] = MonitoringConfig(**data["monitoring"])
        if "tools" in data:
            data["tools"] = ToolConfig(**data["tools"])
        if "mcp" in data:
            data["mcp"] = MCPConfig(**data["mcp"])

        # Handle enums
        if "execution_mode" in data:
            data["execution_mode"] = ExecutionMode(data["execution_mode"])
        if "monitoring" in data and hasattr(data["monitoring"], "log_level"):
            data["monitoring"].log_level = LogLevel(data["monitoring"].log_level)

        return cls(**data)


class SDKConfigManager:
    """Manager for SDK configurations with presets and environment integration."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.logger = logging.getLogger("sdk_config_manager")
        self.config_dir = config_dir or Path("config/sdk_configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Configuration storage
        self.presets: Dict[str, ComprehensiveSDKConfig] = {}
        self.agent_configs: Dict[str, ComprehensiveSDKConfig] = {}
        self.global_config: Optional[ComprehensiveSDKConfig] = None

        # Load configurations
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load all configurations from files and environment."""
        # Load global configuration
        self._load_global_config()

        # Load presets
        self._load_presets()

        # Load agent-specific configs
        self._load_agent_configs()

        # Apply environment overrides
        self._apply_environment_overrides()

    def _load_global_config(self) -> None:
        """Load global configuration."""
        config_file = self.config_dir / "global.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                self.global_config = ComprehensiveSDKConfig.from_dict(data)
                self.logger.info("Loaded global SDK configuration")
            except Exception as e:
                self.logger.error(f"Failed to load global config: {e}")
                self.global_config = ComprehensiveSDKConfig()
        else:
            self.global_config = ComprehensiveSDKConfig()
            self._save_global_config()

    def _load_presets(self) -> None:
        """Load configuration presets."""
        presets_dir = self.config_dir / "presets"
        presets_dir.mkdir(exist_ok=True)

        # Default presets
        self._create_default_presets()

        # Load custom presets
        for preset_file in presets_dir.glob("*.json"):
            try:
                with open(preset_file, 'r') as f:
                    data = json.load(f)
                preset_name = preset_file.stem
                self.presets[preset_name] = ComprehensiveSDKConfig.from_dict(data)
                self.logger.debug(f"Loaded preset: {preset_name}")
            except Exception as e:
                self.logger.error(f"Failed to load preset {preset_file}: {e}")

    def _create_default_presets(self) -> None:
        """Create default configuration presets."""
        # Fast preset
        fast_config = ComprehensiveSDKConfig(
            execution_mode=ExecutionMode.FAST,
            max_turns=20,
            timeout_seconds=120,
            parallel_execution=True,
            max_parallel_tasks=5,
            quality_threshold=0.6,
            performance=PerformanceConfig(
                max_concurrent_requests=10,
                request_timeout_seconds=120,
                cache_enabled=True
            ),
            monitoring=MonitoringConfig(
                enable_metrics=False,
                enable_tracing=False
            )
        )
        self.presets["fast"] = fast_config

        # Thorough preset
        thorough_config = ComprehensiveSDKConfig(
            execution_mode=ExecutionMode.THOROUGH,
            max_turns=100,
            timeout_seconds=600,
            quality_threshold=0.9,
            enable_content_validation=True,
            enable_output_filtering=True,
            performance=PerformanceConfig(
                max_concurrent_requests=3,
                request_timeout_seconds=600,
                retry_attempts=5
            ),
            monitoring=MonitoringConfig(
                enable_metrics=True,
                enable_tracing=True,
                tracing_sample_rate=1.0,
                enable_profiling=True
            )
        )
        self.presets["thorough"] = thorough_config

        # Debug preset
        debug_config = ComprehensiveSDKConfig(
            execution_mode=ExecutionMode.DEBUG,
            debug_mode=True,
            enable_request_logging=True,
            enable_response_logging=True,
            save_conversation_history=True,
            monitoring=MonitoringConfig(
                log_level=LogLevel.DEBUG,
                enable_metrics=True,
                enable_tracing=True,
                tracing_sample_rate=1.0,
                enable_profiling=True
            )
        )
        self.presets["debug"] = debug_config

        # Performance preset
        performance_config = ComprehensiveSDKConfig(
            execution_mode=ExecutionMode.PERFORMANCE,
            max_turns=30,
            timeout_seconds=180,
            parallel_execution=True,
            max_parallel_tasks=10,
            performance=PerformanceConfig(
                max_concurrent_requests=20,
                request_timeout_seconds=180,
                enable_compression=True,
                cache_enabled=True,
                cache_ttl_seconds=7200
            ),
            monitoring=MonitoringConfig(
                enable_metrics=True,
                enable_tracing=False,
                health_check_interval_seconds=10
            )
        )
        self.presets["performance"] = performance_config

    def _load_agent_configs(self) -> None:
        """Load agent-specific configurations."""
        agents_dir = self.config_dir / "agents"
        agents_dir.mkdir(exist_ok=True)

        for agent_file in agents_dir.glob("*.json"):
            try:
                with open(agent_file, 'r') as f:
                    data = json.load(f)
                agent_type = agent_file.stem
                self.agent_configs[agent_type] = ComprehensiveSDKConfig.from_dict(data)
                self.logger.debug(f"Loaded agent config: {agent_type}")
            except Exception as e:
                self.logger.error(f"Failed to load agent config {agent_file}: {e}")

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Global overrides
        if self.global_config:
            self._apply_env_overrides_to_config(self.global_config, "SDK_")

        # Agent-specific overrides
        for agent_type, config in self.agent_configs.items():
            prefix = f"SDK_{agent_type.upper()}_"
            self._apply_env_overrides_to_config(config, prefix)

    def _apply_env_overrides_to_config(self, config: ComprehensiveSDKConfig, prefix: str) -> None:
        """Apply environment overrides to a specific configuration."""
        # Basic settings
        if f"{prefix}MAX_TURNS" in os.environ:
            config.max_turns = int(os.environ[f"{prefix}MAX_TURNS"])

        if f"{prefix}TIMEOUT_SECONDS" in os.environ:
            config.timeout_seconds = int(os.environ[f"{prefix}TIMEOUT_SECONDS"])

        if f"{prefix}QUALITY_THRESHOLD" in os.environ:
            config.quality_threshold = float(os.environ[f"{prefix}QUALITY_THRESHOLD"])

        if f"{prefix}DEBUG_MODE" in os.environ:
            config.debug_mode = os.environ[f"{prefix}DEBUG_MODE"].lower() == "true"

        # Performance settings
        perf_prefix = f"{prefix}PERFORMANCE_"
        if f"{perf_prefix}MAX_CONCURRENT_REQUESTS" in os.environ:
            config.performance.max_concurrent_requests = int(os.environ[f"{perf_prefix}MAX_CONCURRENT_REQUESTS"])

        if f"{perf_prefix}REQUEST_TIMEOUT_SECONDS" in os.environ:
            config.performance.request_timeout_seconds = int(os.environ[f"{perf_prefix}REQUEST_TIMEOUT_SECONDS"])

        if f"{perf_prefix}CACHE_ENABLED" in os.environ:
            config.performance.cache_enabled = os.environ[f"{perf_prefix}CACHE_ENABLED"].lower() == "true"

        # Monitoring settings
        mon_prefix = f"{prefix}MONITORING_"
        if f"{mon_prefix}ENABLE_METRICS" in os.environ:
            config.monitoring.enable_metrics = os.environ[f"{mon_prefix}ENABLE_METRICS"].lower() == "true"

        if f"{mon_prefix}LOG_LEVEL" in os.environ:
            config.monitoring.log_level = LogLevel(os.environ[f"{mon_prefix}LOG_LEVEL"].lower())

    def get_config_for_agent(self, agent_type: str, preset_name: Optional[str] = None,
                           overrides: Optional[Dict[str, Any]] = None) -> ComprehensiveSDKConfig:
        """Get configuration for a specific agent."""
        # Start with global config
        config = self.global_config.copy() if self.global_config else ComprehensiveSDKConfig()

        # Apply agent-specific config
        if agent_type in self.agent_configs:
            config = config.merge_with(self.agent_configs[agent_type])

        # Apply preset
        if preset_name and preset_name in self.presets:
            config = config.merge_with(self.presets[preset_name])

        # Set agent type
        config.agent_type = agent_type

        # Apply overrides
        if overrides:
            override_config = ComprehensiveSDKConfig.from_dict(overrides)
            config = config.merge_with(override_config)

        # Validate
        issues = config.validate()
        if issues:
            self.logger.warning(f"Configuration validation issues for {agent_type}: {issues}")

        return config

    def save_agent_config(self, agent_type: str, config: ComprehensiveSDKConfig) -> None:
        """Save configuration for a specific agent."""
        self.agent_configs[agent_type] = config

        # Save to file
        agents_dir = self.config_dir / "agents"
        agents_dir.mkdir(exist_ok=True)
        config_file = agents_dir / f"{agent_type}.json"

        try:
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            self.logger.info(f"Saved configuration for agent: {agent_type}")
        except Exception as e:
            self.logger.error(f"Failed to save agent config {agent_type}: {e}")

    def save_preset(self, preset_name: str, config: ComprehensiveSDKConfig) -> None:
        """Save a configuration preset."""
        self.presets[preset_name] = config

        # Save to file
        presets_dir = self.config_dir / "presets"
        presets_dir.mkdir(exist_ok=True)
        preset_file = presets_dir / f"{preset_name}.json"

        try:
            with open(preset_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            self.logger.info(f"Saved preset: {preset_name}")
        except Exception as e:
            self.logger.error(f"Failed to save preset {preset_name}: {e}")

    def _save_global_config(self) -> None:
        """Save global configuration."""
        config_file = self.config_dir / "global.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(self.global_config.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save global config: {e}")

    def list_presets(self) -> List[str]:
        """List all available presets."""
        return list(self.presets.keys())

    def list_agent_configs(self) -> List[str]:
        """List all agent-specific configurations."""
        return list(self.agent_configs.keys())

    def get_preset(self, preset_name: str) -> Optional[ComprehensiveSDKConfig]:
        """Get a specific preset."""
        return self.presets.get(preset_name)

    def delete_preset(self, preset_name: str) -> bool:
        """Delete a preset."""
        if preset_name in self.presets:
            del self.presets[preset_name]

            # Delete file
            preset_file = self.config_dir / "presets" / f"{preset_name}.json"
            if preset_file.exists():
                preset_file.unlink()

            self.logger.info(f"Deleted preset: {preset_name}")
            return True
        return False

    def reload_configurations(self) -> None:
        """Reload all configurations from files."""
        self.presets.clear()
        self.agent_configs.clear()
        self._load_configurations()
        self.logger.info("Reloaded all SDK configurations")


# Global config manager instance
_config_manager: Optional[SDKConfigManager] = None


def get_sdk_config_manager(config_dir: Optional[Path] = None) -> SDKConfigManager:
    """Get or create the global SDK config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = SDKConfigManager(config_dir)
    return _config_manager


def get_agent_sdk_config(agent_type: str, preset: Optional[str] = None,
                        overrides: Optional[Dict[str, Any]] = None) -> ClaudeAgentOptions:
    """Convenience function to get SDK options for an agent."""
    manager = get_sdk_config_manager()
    config = manager.get_config_for_agent(agent_type, preset, overrides)
    return config.to_sdk_options()