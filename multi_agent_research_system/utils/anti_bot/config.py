"""
Anti-Bot Escalation Configuration Management

This module provides comprehensive configuration management for the anti-bot escalation
system, integrating with the existing settings infrastructure from Phase 1.1 and
providing environment-based overrides with validation.

Key Features:
- Integration with existing settings system
- Environment variable overrides
- Configuration validation and defaults
- Performance tuning parameters
- Learning and optimization settings
- Monitoring and debugging options

Based on Phase 1.1 enhanced configuration foundation
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from ...config.settings import get_settings


@dataclass
class AntiBotSystemConfig:
    """Configuration for anti-bot escalation system."""

    # Core escalation settings
    max_attempts_per_url: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0
    default_start_level: int = 1  # Enhanced level by default
    max_escalation_level: int = 3

    # Learning and optimization
    enable_domain_learning: bool = True
    enable_auto_optimization: bool = True
    min_attempts_for_learning: int = 3
    learning_confidence_threshold: float = 0.7
    learning_data_retention_days: int = 30

    # Cooldown management
    enable_cooldown_management: bool = True
    base_cooldown_minutes: int = 5
    max_cooldown_minutes: int = 60
    cooldown_multiplier_rate_limit: float = 2.0
    cooldown_multiplier_bot_detection: float = 3.0
    cooldown_multiplier_captcha: float = 4.0

    # Performance settings
    concurrent_limit: int = 5
    request_timeout_base: int = 10
    request_timeout_stealth: int = 120
    keep_alive_enabled: bool = True
    cache_enabled: bool = True

    # Browser and rendering settings
    enable_javascript_enhanced: bool = True
    enable_stealth_mode_advanced: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_agent_rotation_enabled: bool = True

    # Human-like behavior
    enable_mouse_movements: bool = True
    enable_keyboard_typing: bool = True
    enable_scrolling_behavior: bool = True
    human_like_delay_range: tuple = (0.5, 2.0)

    # Content filtering and quality
    enable_content_filtering: bool = True
    content_filter_threshold: float = 0.4
    min_content_length: int = 100
    max_content_length: int = 100000

    # Detection and monitoring
    detection_markers_enabled: bool = True
    performance_monitoring: bool = True
    detailed_logging: bool = False
    export_statistics: bool = True

    # Security and privacy
    verify_ssl: bool = True
    proxy_rotation_enabled: bool = False
    max_redirects: int = 5

    # Data persistence
    save_domain_profiles: bool = True
    profile_save_interval: int = 10  # Save every 10 updates
    profile_backup_enabled: bool = True

    # Level-specific configurations
    level_configs: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration values."""
        if not 1 <= self.max_attempts_per_url <= 10:
            raise ValueError(f"max_attempts_per_url must be 1-10, got {self.max_attempts_per_url}")

        if not 0 <= self.default_start_level <= self.max_escalation_level <= 3:
            raise ValueError(f"Levels must be 0-3, got start={self.default_start_level}, max={self.max_escalation_level}")

        if not 0.1 <= self.base_delay <= 60.0:
            raise ValueError(f"base_delay must be 0.1-60.0 seconds, got {self.base_delay}")

        if not 1.0 <= self.max_delay <= 300.0:
            raise ValueError(f"max_delay must be 1.0-300.0 seconds, got {self.max_delay}")

        if not 1 <= self.concurrent_limit <= 20:
            raise ValueError(f"concurrent_limit must be 1-20, got {self.concurrent_limit}")

        if not 0.0 <= self.learning_confidence_threshold <= 1.0:
            raise ValueError(f"learning_confidence_threshold must be 0.0-1.0, got {self.learning_confidence_threshold}")

        if self.human_like_delay_range[0] >= self.human_like_delay_range[1]:
            raise ValueError(f"Invalid delay range: {self.human_like_delay_range}")

        # Initialize default level configs if not provided
        if not self.level_configs:
            self.level_configs = self._get_default_level_configs()

    def _get_default_level_configs(self) -> Dict[int, Dict[str, Any]]:
        """Get default configurations for each anti-bot level."""
        return {
            0: {  # Basic
                'timeout': 10,
                'javascript': False,
                'stealth': False,
                'wait_strategy': 'minimal',
                'retry_delays': [1.0, 2.0],
                'cool_down_period': 0,
                'user_agent_rotation': False,
                'verify_ssl': True,
                'max_redirects': 3
            },
            1: {  # Enhanced
                'timeout': 30,
                'javascript': True,
                'stealth': False,
                'wait_strategy': 'smart',
                'retry_delays': [2.0, 5.0, 10.0],
                'cool_down_period': 30,
                'user_agent_rotation': True,
                'verify_ssl': True,
                'max_redirects': 5
            },
            2: {  # Advanced
                'timeout': 60,
                'javascript': True,
                'stealth': True,
                'wait_strategy': 'adaptive',
                'retry_delays': [5.0, 15.0, 30.0, 60.0],
                'cool_down_period': 120,
                'user_agent_rotation': True,
                'verify_ssl': True,
                'max_redirects': 5,
                'viewport': 'width=device-width, initial-scale=1.0',
                'mouse_movements': True,
                'scrolling_behavior': True
            },
            3: {  # Stealth
                'timeout': 120,
                'javascript': True,
                'stealth': True,
                'wait_strategy': 'human_like',
                'retry_delays': [10.0, 30.0, 60.0, 120.0, 300.0],
                'cool_down_period': 300,
                'user_agent_rotation': True,
                'verify_ssl': True,
                'max_redirects': 10,
                'viewport': 'width=device-width, initial-scale=1.0',
                'proxy_rotation': True,
                'mouse_movements': True,
                'keyboard_typing': True,
                'scrolling_behavior': True
            }
        }


class AntiBotConfigManager:
    """Manager for anti-bot configuration with environment overrides."""

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """Initialize configuration manager.

        Args:
            custom_config: Optional custom configuration to merge
        """
        self.base_config = AntiBotSystemConfig()
        self.environment_overrides = self._load_environment_overrides()
        self.custom_config = custom_config or {}

        # Create final configuration by merging all sources
        self.config = self._merge_configurations()

    def _load_environment_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}

        # Core escalation settings
        if os.getenv('ANTI_BOT_MAX_ATTEMPTS'):
            overrides['max_attempts_per_url'] = int(os.getenv('ANTI_BOT_MAX_ATTEMPTS'))

        if os.getenv('ANTI_BOT_BASE_DELAY'):
            overrides['base_delay'] = float(os.getenv('ANTI_BOT_BASE_DELAY'))

        if os.getenv('ANTI_BOT_MAX_DELAY'):
            overrides['max_delay'] = float(os.getenv('ANTI_BOT_MAX_DELAY'))

        if os.getenv('ANTI_BOT_DEFAULT_LEVEL'):
            overrides['default_start_level'] = int(os.getenv('ANTI_BOT_DEFAULT_LEVEL'))

        if os.getenv('ANTI_BOT_MAX_LEVEL'):
            overrides['max_escalation_level'] = int(os.getenv('ANTI_BOT_MAX_LEVEL'))

        # Learning settings
        if os.getenv('ANTI_BOT_ENABLE_LEARNING'):
            overrides['enable_domain_learning'] = os.getenv('ANTI_BOT_ENABLE_LEARNING').lower() == 'true'

        if os.getenv('ANTI_BOT_ENABLE_OPTIMIZATION'):
            overrides['enable_auto_optimization'] = os.getenv('ANTI_BOT_ENABLE_OPTIMIZATION').lower() == 'true'

        if os.getenv('ANTI_BOT_MIN_ATTEMPTS_LEARNING'):
            overrides['min_attempts_for_learning'] = int(os.getenv('ANTI_BOT_MIN_ATTEMPTS_LEARNING'))

        if os.getenv('ANTI_BOT_LEARNING_THRESHOLD'):
            overrides['learning_confidence_threshold'] = float(os.getenv('ANTI_BOT_LEARNING_THRESHOLD'))

        # Cooldown settings
        if os.getenv('ANTI_BOT_ENABLE_COOLDOWN'):
            overrides['enable_cooldown_management'] = os.getenv('ANTI_BOT_ENABLE_COOLDOWN').lower() == 'true'

        if os.getenv('ANTI_BOT_BASE_COOLDOWN'):
            overrides['base_cooldown_minutes'] = int(os.getenv('ANTI_BOT_BASE_COOLDOWN'))

        if os.getenv('ANTI_BOT_MAX_COOLDOWN'):
            overrides['max_cooldown_minutes'] = int(os.getenv('ANTI_BOT_MAX_COOLDOWN'))

        # Performance settings
        if os.getenv('ANTI_BOT_CONCURRENT_LIMIT'):
            overrides['concurrent_limit'] = int(os.getenv('ANTI_BOT_CONCURRENT_LIMIT'))

        if os.getenv('ANTI_BOT_REQUEST_TIMEOUT_BASE'):
            overrides['request_timeout_base'] = int(os.getenv('ANTI_BOT_REQUEST_TIMEOUT_BASE'))

        if os.getenv('ANTI_BOT_REQUEST_TIMEOUT_STEALTH'):
            overrides['request_timeout_stealth'] = int(os.getenv('ANTI_BOT_REQUEST_TIMEOUT_STEALTH'))

        if os.getenv('ANTI_BOT_ENABLE_CACHE'):
            overrides['cache_enabled'] = os.getenv('ANTI_BOT_ENABLE_CACHE').lower() == 'true'

        # Browser settings
        if os.getenv('ANTI_BOT_ENABLE_JS_ENHANCED'):
            overrides['enable_javascript_enhanced'] = os.getenv('ANTI_BOT_ENABLE_JS_ENHANCED').lower() == 'true'

        if os.getenv('ANTI_BOT_ENABLE_STEALTH_ADVANCED'):
            overrides['enable_stealth_mode_advanced'] = os.getenv('ANTI_BOT_ENABLE_STEALTH_ADVANCED').lower() == 'true'

        if os.getenv('ANTI_BOT_VIEWPORT_WIDTH'):
            overrides['viewport_width'] = int(os.getenv('ANTI_BOT_VIEWPORT_WIDTH'))

        if os.getenv('ANTI_BOT_VIEWPORT_HEIGHT'):
            overrides['viewport_height'] = int(os.getenv('ANTI_BOT_VIEWPORT_HEIGHT'))

        # Human-like behavior
        if os.getenv('ANTI_BOT_ENABLE_MOUSE_MOVEMENTS'):
            overrides['enable_mouse_movements'] = os.getenv('ANTI_BOT_ENABLE_MOUSE_MOVEMENTS').lower() == 'true'

        if os.getenv('ANTI_BOT_ENABLE_KEYBOARD_TYPING'):
            overrides['enable_keyboard_typing'] = os.getenv('ANTI_BOT_ENABLE_KEYBOARD_TYPING').lower() == 'true'

        if os.getenv('ANTI_BOT_ENABLE_SCROLLING'):
            overrides['enable_scrolling_behavior'] = os.getenv('ANTI_BOT_ENABLE_SCROLLING').lower() == 'true'

        # Content filtering
        if os.getenv('ANTI_BOT_ENABLE_CONTENT_FILTERING'):
            overrides['enable_content_filtering'] = os.getenv('ANTI_BOT_ENABLE_CONTENT_FILTERING').lower() == 'true'

        if os.getenv('ANTI_BOT_CONTENT_FILTER_THRESHOLD'):
            overrides['content_filter_threshold'] = float(os.getenv('ANTI_BOT_CONTENT_FILTER_THRESHOLD'))

        # Monitoring and debugging
        if os.getenv('ANTI_BOT_PERFORMANCE_MONITORING'):
            overrides['performance_monitoring'] = os.getenv('ANTI_BOT_PERFORMANCE_MONITORING').lower() == 'true'

        if os.getenv('ANTI_BOT_DETAILED_LOGGING'):
            overrides['detailed_logging'] = os.getenv('ANTI_BOT_DETAILED_LOGGING').lower() == 'true'

        if os.getenv('ANTI_BOT_EXPORT_STATISTICS'):
            overrides['export_statistics'] = os.getenv('ANTI_BOT_EXPORT_STATISTICS').lower() == 'true'

        # Security settings
        if os.getenv('ANTI_BOT_VERIFY_SSL'):
            overrides['verify_ssl'] = os.getenv('ANTI_BOT_VERIFY_SSL').lower() == 'true'

        if os.getenv('ANTI_BOT_ENABLE_PROXY_ROTATION'):
            overrides['proxy_rotation_enabled'] = os.getenv('ANTI_BOT_ENABLE_PROXY_ROTATION').lower() == 'true'

        # Data persistence
        if os.getenv('ANTI_BOT_SAVE_PROFILES'):
            overrides['save_domain_profiles'] = os.getenv('ANTI_BOT_SAVE_PROFILES').lower() == 'true'

        if os.getenv('ANTI_BOT_PROFILE_SAVE_INTERVAL'):
            overrides['profile_save_interval'] = int(os.getenv('ANTI_BOT_PROFILE_SAVE_INTERVAL'))

        return overrides

    def _merge_configurations(self) -> AntiBotSystemConfig:
        """Merge base configuration with environment and custom overrides."""
        # Start with base config as dictionary
        config_dict = self.base_config.__dict__.copy()

        # Apply environment overrides
        config_dict.update(self.environment_overrides)

        # Apply custom configuration
        config_dict.update(self.custom_config)

        # Create new configuration instance
        try:
            return AntiBotSystemConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"Failed to create anti-bot configuration: {e}")

    def get_config(self) -> AntiBotSystemConfig:
        """Get the final merged configuration."""
        return self.config

    def get_level_config(self, level: int) -> Dict[str, Any]:
        """Get configuration for a specific anti-bot level.

        Args:
            level: Anti-bot level (0-3)

        Returns:
            Level-specific configuration dictionary
        """
        if level not in self.config.level_configs:
            raise ValueError(f"Invalid anti-bot level: {level}")

        level_config = self.config.level_configs[level].copy()

        # Apply global settings to level config
        level_config.update({
            'user_agent_rotation': self.config.user_agent_rotation_enabled,
            'verify_ssl': self.config.verify_ssl,
            'performance_monitoring': self.config.performance_monitoring,
            'detailed_logging': self.config.detailed_logging
        })

        # Level-specific overrides
        if level >= 2:  # Advanced and Stealth
            level_config.update({
                'viewport': f'width={self.config.viewport_width}, height={self.config.viewport_height}',
                'mouse_movements': self.config.enable_mouse_movements,
                'scrolling_behavior': self.config.enable_scrolling_behavior
            })

        if level == 3:  # Stealth only
            level_config.update({
                'proxy_rotation': self.config.proxy_rotation_enabled,
                'keyboard_typing': self.config.enable_keyboard_typing,
                'content_filtering': self.config.enable_content_filtering,
                'content_filter_threshold': self.config.content_filter_threshold
            })

        return level_config

    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about environment configuration.

        Returns:
            Dictionary with environment configuration info
        """
        return {
            'environment_variables': {
                var: os.getenv(var) for var in [
                    'ANTI_BOT_MAX_ATTEMPTS',
                    'ANTI_BOT_ENABLE_LEARNING',
                    'ANTI_BOT_ENABLE_COOLDOWN',
                    'ANTI_BOT_CONCURRENT_LIMIT',
                    'ANTI_BOT_PERFORMANCE_MONITORING',
                    'ANTI_BOT_DETAILED_LOGGING'
                ] if os.getenv(var)
            },
            'overrides_applied': len(self.environment_overrides),
            'custom_config_applied': len(self.custom_config),
            'config_source': 'merged' if (self.environment_overrides or self.custom_config) else 'default'
        }

    def validate_for_environment(self) -> Dict[str, Any]:
        """Validate configuration for current environment.

        Returns:
            Validation result with warnings and recommendations
        """
        warnings = []
        recommendations = []

        # Check for production environment concerns
        if self.config.detailed_logging and not os.getenv('DEBUG'):
            warnings.append("Detailed logging enabled in non-debug environment")

        if self.config.concurrent_limit > 10 and not os.getenv('ANTI_BOT_HIGH_CONCURRENCY'):
            recommendations.append("Consider reducing concurrent limit for stability")

        if self.config.base_delay < 0.5 and not os.getenv('ANTI_BOT_AGGRESSIVE'):
            recommendations.append("Very short base delay may trigger rate limiting")

        # Check for development environment optimizations
        if os.getenv('DEVELOPMENT_MODE') == 'true':
            if self.config.enable_cooldown_management:
                recommendations.append("Consider disabling cooldown in development for faster testing")

            if self.config.save_domain_profiles:
                recommendations.append("Consider disabling profile saving in development")

        # Check for performance impacts
        if self.config.enable_domain_learning and self.config.min_attempts_for_learning > 5:
            recommendations.append("High learning threshold may reduce optimization effectiveness")

        if self.config.performance_monitoring and self.config.concurrent_limit > 15:
            warnings.append("High concurrency with performance monitoring may impact performance")

        return {
            'valid': True,
            'warnings': warnings,
            'recommendations': recommendations,
            'environment': os.getenv('ENVIRONMENT', 'development')
        }

    def export_config(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Export configuration for debugging or backup.

        Args:
            include_sensitive: Whether to include sensitive information

        Returns:
            Exportable configuration dictionary
        """
        config_dict = {
            'anti_bot_config': self.config.__dict__.copy(),
            'environment_info': self.get_environment_info(),
            'validation_result': self.validate_for_environment()
        }

        # Remove sensitive information if requested
        if not include_sensitive:
            sensitive_keys = ['proxy_rotation_enabled', 'verify_ssl']
            for key in sensitive_keys:
                if key in config_dict['anti_bot_config']:
                    config_dict['anti_bot_config'][key] = '[REDACTED]'

        return config_dict

    def update_setting(self, key: str, value: Any) -> bool:
        """Update a specific configuration setting.

        Args:
            key: Configuration key to update
            value: New value

        Returns:
            True if update successful, False otherwise
        """
        if not hasattr(self.config, key):
            return False

        try:
            # Create new config with updated value
            config_dict = self.config.__dict__.copy()
            config_dict[key] = value

            new_config = AntiBotSystemConfig(**config_dict)
            self.config = new_config

            return True

        except Exception:
            return False

    def reset_to_defaults(self) -> AntiBotSystemConfig:
        """Reset configuration to defaults.

        Returns:
            New default configuration
        """
        self.config = AntiBotSystemConfig()
        return self.config


# Global configuration manager instance
_global_config_manager: Optional[AntiBotConfigManager] = None


def get_anti_bot_config(custom_config: Optional[Dict[str, Any]] = None) -> AntiBotSystemConfig:
    """Get global anti-bot configuration.

    Args:
        custom_config: Optional custom configuration to merge

    Returns:
        Anti-bot system configuration
    """
    global _global_config_manager

    if _global_config_manager is None or custom_config is not None:
        _global_config_manager = AntiBotConfigManager(custom_config)

    return _global_config_manager.get_config()


def get_anti_bot_config_manager() -> AntiBotConfigManager:
    """Get the global configuration manager instance.

    Returns:
        Configuration manager instance
    """
    global _global_config_manager

    if _global_config_manager is None:
        _global_config_manager = AntiBotConfigManager()

    return _global_config_manager


def configure_anti_bot_system(config: Dict[str, Any]) -> bool:
    """Configure anti-bot system with custom settings.

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration successful, False otherwise
    """
    try:
        global _global_config_manager
        _global_config_manager = AntiBotConfigManager(config)
        return True
    except Exception:
        return False


def get_level_configuration(level: int) -> Dict[str, Any]:
    """Get configuration for a specific anti-bot level.

    Args:
        level: Anti-bot level (0-3)

    Returns:
        Level-specific configuration
    """
    manager = get_anti_bot_config_manager()
    return manager.get_level_config(level)


def is_anti_bot_enabled() -> bool:
    """Check if anti-bot system is enabled.

    Returns:
        True if anti-bot system is enabled
    """
    try:
        config = get_anti_bot_config()
        return config.max_escalation_level > 0
    except Exception:
        return False


def get_anti_bot_environment_info() -> Dict[str, Any]:
    """Get environment information for anti-bot system.

    Returns:
        Environment information dictionary
    """
    manager = get_anti_bot_config_manager()
    return manager.get_environment_info()