"""
Configuration Manager - Unified Configuration Management

This module provides unified configuration management that integrates the legacy settings
system with the enhanced SDK configuration patterns, providing backward compatibility
while enabling advanced features.

Key Features:
- Unified configuration interface
- Legacy and enhanced configuration integration
- Environment-based configuration loading
- Configuration validation and migration
- Hot reloading capabilities
- Configuration export/import

Based on Redesign Plan PLUS SDK Implementation (October 13, 2025)
"""

import os
import json
from typing import Dict, Any, Optional, Union, List, Type
from pathlib import Path
from dataclasses import asdict
import logging

from .settings import Settings, get_settings
from .sdk_config import (
    ClaudeAgentSDKConfig, get_sdk_config, set_sdk_config,
    DEVELOPMENT_CONFIG, PRODUCTION_CONFIG, TESTING_CONFIG
)
from .enhanced_agents import (
    EnhancedAgentDefinition, get_all_enhanced_agent_definitions,
    create_enhanced_agent, AgentType
)


logger = logging.getLogger(__name__)


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
        """Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (development, testing, production)
        """
        # Set configuration directory
        if config_dir is None:
            # Default to KEVIN/config or current directory/config
            base_dir = Path.cwd()
            possible_dirs = [
                base_dir / "KEVIN" / "config",
                base_dir / "config",
                base_dir / ".config"
            ]
            for dir_path in possible_dirs:
                if dir_path.exists():
                    self._config_dir = dir_path
                    break
            else:
                self._config_dir = base_dir / "KEVIN" / "config"
                self._config_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._config_dir = Path(config_dir)
            self._config_dir.mkdir(parents=True, exist_ok=True)

        # Set environment
        if environment is None:
            self._environment = os.getenv("ENVIRONMENT", "development").lower()
        else:
            self._environment = environment.lower()

        # Validate environment
        if self._environment not in ["development", "testing", "staging", "production"]:
            logger.warning(f"Unknown environment '{self._environment}', defaulting to 'development'")
            self._environment = "development"

        logger.info(f"Initializing configuration manager for environment: {self._environment}")
        logger.info(f"Configuration directory: {self._config_dir}")

        # Load configurations
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load all configuration files."""
        try:
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

            logger.info("All configurations loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise

    def _load_legacy_settings(self) -> None:
        """Load legacy settings with fallback to defaults."""
        try:
            self._legacy_settings = get_settings()
            logger.info("Legacy settings loaded from environment/defaults")
        except Exception as e:
            logger.warning(f"Could not load legacy settings: {e}")
            # Create minimal fallback settings
            self._legacy_settings = self._create_fallback_settings()

    def _create_fallback_settings(self) -> Settings:
        """Create fallback settings when configuration loading fails."""
        class FallbackSettings:
            def __init__(self):
                self.openai_api_key = os.getenv("OPENAI_API_KEY", "not_configured")
                self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                self.serper_api_key = os.getenv("SERPER_API_KEY", "not_configured")
                self.log_level = os.getenv("LOG_LEVEL", "INFO")
                self.debug_mode = self._environment == "development"
                self.development_mode = self._environment == "development"
                self.target_successful_scrapes = 15
                self.concurrent_crawl_limit = 16

            def setup_logging(self):
                import logging
                logging.basicConfig(level=getattr(logging, self.log_level))

            def get_missing_api_keys(self):
                missing = []
                if not self.openai_api_key or self.openai_api_key == "not_configured":
                    missing.append("OPENAI_API_KEY")
                if not self.serper_api_key or self.serper_api_key == "not_configured":
                    missing.append("SERPER_API_KEY")
                return missing

        return FallbackSettings()

    def _load_sdk_config(self) -> None:
        """Load SDK configuration from file or use defaults."""
        sdk_config_file = self._config_dir / "sdk_config.json"

        if sdk_config_file.exists():
            try:
                self._sdk_config = ClaudeAgentSDKConfig.load_from_file(sdk_config_file)
                logger.info(f"SDK configuration loaded from: {sdk_config_file}")
            except Exception as e:
                logger.warning(f"Could not load SDK config from file: {e}")
                self._sdk_config = self._get_default_sdk_config()
        else:
            self._sdk_config = self._get_default_sdk_config()
            logger.info("Using default SDK configuration")

        # Set as global instance
        set_sdk_config(self._sdk_config)

    def _get_default_sdk_config(self) -> ClaudeAgentSDKConfig:
        """Get default SDK configuration based on environment."""
        if self._environment == "production":
            return PRODUCTION_CONFIG
        elif self._environment == "testing":
            return TESTING_CONFIG
        else:
            return DEVELOPMENT_CONFIG

    def _load_enhanced_agents(self) -> None:
        """Load enhanced agent definitions."""
        agents_config_file = self._config_dir / "enhanced_agents.json"

        if agents_config_file.exists():
            try:
                with open(agents_config_file, 'r') as f:
                    agents_data = json.load(f)

                self._enhanced_agents = {}
                for agent_name, agent_data in agents_data.items():
                    self._enhanced_agents[agent_name] = EnhancedAgentDefinition.from_dict(agent_data)

                logger.info(f"Enhanced agents loaded from: {agents_config_file}")
            except Exception as e:
                logger.warning(f"Could not load enhanced agents from file: {e}")
                self._enhanced_agents = get_all_enhanced_agent_definitions()
        else:
            self._enhanced_agents = get_all_enhanced_agent_definitions()
            logger.info("Using default enhanced agent definitions")

    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides."""
        # SDK config overrides
        self._sdk_config.environment = self._environment
        self._sdk_config.debug_mode = (self._environment == "development")

        # Legacy settings overrides
        if hasattr(self._legacy_settings, 'debug_mode'):
            self._legacy_settings.debug_mode = (self._environment == "development")
        if hasattr(self._legacy_settings, 'development_mode'):
            self._legacy_settings.development_mode = (self._environment == "development")

        # Environment-specific logging
        if self._environment == "production":
            self._sdk_config.observability.log_level.value = "INFO"
            self._sdk_config.observability.enable_detailed_hooks_logging = False
        elif self._environment == "development":
            self._sdk_config.observability.log_level.value = "DEBUG"
            self._sdk_config.observability.enable_detailed_hooks_logging = True

    def _validate_configurations(self) -> None:
        """Validate loaded configurations."""
        # Validate SDK configuration
        try:
            self._sdk_config._validate_configuration()
        except Exception as e:
            logger.error(f"SDK configuration validation failed: {e}")
            raise

        # Validate required API keys
        missing_keys = self._legacy_settings.get_missing_api_keys()
        if missing_keys:
            if self._environment == "production":
                logger.error(f"Missing required API keys for production: {missing_keys}")
                raise ValueError(f"Missing required API keys: {missing_keys}")
            else:
                logger.warning(f"Missing API keys (system may have limited functionality): {missing_keys}")

    def get_legacy_settings(self) -> Settings:
        """Get legacy settings instance."""
        return self._legacy_settings

    def get_sdk_config(self) -> ClaudeAgentSDKConfig:
        """Get SDK configuration instance."""
        return self._sdk_config

    def get_enhanced_agent(self, agent_type: Union[str, AgentType]) -> EnhancedAgentDefinition:
        """Get enhanced agent definition by type."""
        if isinstance(agent_type, str):
            agent_key = agent_type.lower()
        else:
            agent_key = agent_type.value

        if agent_key in self._enhanced_agents:
            return self._enhanced_agents[agent_key]
        else:
            # Create on-demand if not found
            return create_enhanced_agent(agent_type)

    def get_all_enhanced_agents(self) -> Dict[str, EnhancedAgentDefinition]:
        """Get all enhanced agent definitions."""
        return self._enhanced_agents.copy()

    def save_configurations(self) -> None:
        """Save all configurations to files."""
        if self._config_dir is None:
            raise ValueError("Configuration manager not initialized")

        try:
            # Save SDK configuration
            sdk_config_file = self._config_dir / "sdk_config.json"
            self._sdk_config.save_to_file(sdk_config_file)
            logger.info(f"SDK configuration saved to: {sdk_config_file}")

            # Save enhanced agents
            agents_config_file = self._config_dir / "enhanced_agents.json"
            agents_data = {name: agent.to_dict() for name, agent in self._enhanced_agents.items()}
            with open(agents_config_file, 'w') as f:
                json.dump(agents_data, f, indent=2, default=str)
            logger.info(f"Enhanced agents saved to: {agents_config_file}")

        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
            raise

    def reload_configurations(self) -> None:
        """Reload all configurations from files."""
        logger.info("Reloading configurations...")
        self._load_configurations()
        logger.info("Configurations reloaded successfully")

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration state."""
        return {
            "environment": self._environment,
            "config_directory": str(self._config_dir),
            "sdk_config": {
                "version": self._sdk_config.claude_sdk_version,
                "model": self._sdk_config.default_model,
                "max_tokens": self._sdk_config.max_tokens,
                "temperature": self._sdk_config.temperature,
                "debug_mode": self._sdk_config.debug_mode,
                "observability_enabled": self._sdk_config.observability.enable_performance_metrics
            },
            "legacy_settings": {
                "log_level": self._legacy_settings.log_level,
                "debug_mode": getattr(self._legacy_settings, 'debug_mode', False),
                "target_successful_scrapes": getattr(self._legacy_settings, 'target_successful_scrapes', 15)
            },
            "enhanced_agents": {
                "total_agents": len(self._enhanced_agents),
                "agent_types": list(self._enhanced_agents.keys())
            },
            "api_keys_status": {
                "anthropic": bool(self._sdk_config.anthropic_api_key),
                "serper": bool(self._sdk_config.serper_api_key),
                "openai": bool(self._sdk_config.openai_api_key)
            }
        }

    def export_configuration(self, export_path: Union[str, Path],
                           include_sensitive: bool = False) -> None:
        """Export configuration to a file (excluding sensitive data by default)."""
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        config_summary = self.get_configuration_summary()

        if not include_sensitive:
            # Remove sensitive information
            config_summary["api_keys_status"] = {
                "anthropic": "configured" if self._sdk_config.anthropic_api_key else "not_configured",
                "serper": "configured" if self._sdk_config.serper_api_key else "not_configured",
                "openai": "configured" if self._sdk_config.openai_api_key else "not_configured"
            }

        with open(export_path, 'w') as f:
            json.dump(config_summary, f, indent=2, default=str)

        logger.info(f"Configuration exported to: {export_path}")

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

            # Check flow adherence configuration
            if agent_def.flow_adherence.enabled and not agent_def.flow_adherence.mandatory_steps:
                agent_status["issues"].append("Flow adherence enabled but no mandatory steps defined")

            validation_results["agent_status"][agent_name] = agent_status

            if agent_status["issues"]:
                validation_results["compatible"] = False
                validation_results["issues"].extend([f"{agent_name}: {issue}" for issue in agent_status["issues"]])

            if agent_status["warnings"]:
                validation_results["warnings"].extend([f"{agent_name}: {warning}" for warning in agent_status["warnings"]])

        return validation_results


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def initialize_configuration(config_dir: Optional[Union[str, Path]] = None,
                           environment: Optional[str] = None) -> ConfigurationManager:
    """Initialize the global configuration manager."""
    manager = get_config_manager()
    manager.initialize(config_dir, environment)
    return manager


def get_configuration_summary() -> Dict[str, Any]:
    """Get configuration summary."""
    return get_config_manager().get_configuration_summary()


def validate_system_configuration() -> Dict[str, Any]:
    """Validate the entire system configuration."""
    try:
        manager = get_config_manager()
        return manager.validate_agent_compatibility()
    except Exception as e:
        # Return error result if validation fails
        return {
            "compatible": False,
            "issues": [f"Configuration validation failed: {str(e)}"],
            "warnings": [],
            "agent_status": {},
            "error": str(e)
        }