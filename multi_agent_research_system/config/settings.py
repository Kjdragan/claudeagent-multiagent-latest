"""
Configuration settings for the multi-agent research system.

This module contains configuration for enhanced search, crawling,
and content cleaning functionality.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EnhancedSearchConfig:
    """Configuration for enhanced search functionality."""

    # Search settings
    default_num_results: int = 15
    default_auto_crawl_top: int = 10
    default_crawl_threshold: float = 0.3  # Fixed at 0.3 for better success rates
    default_anti_bot_level: int = 1
    default_max_concurrent: int = 15

    # Anti-bot levels
    anti_bot_levels = {
        0: "basic",      # 6/10 sites success
        1: "enhanced",   # 8/10 sites success
        2: "advanced",   # 9/10 sites success
        3: "stealth"     # 9.5/10 sites success
    }

    # Target-based scraping settings
    target_successful_scrapes: int = 15  # Target number of successful scrapes per search
    url_deduplication_enabled: bool = True  # Prevent duplicate URL crawling
    progressive_retry_enabled: bool = True  # Retry failed URLs with higher anti-bot levels

    # Retry logic settings
    max_retry_attempts: int = 3
    progressive_timeout_multiplier: float = 1.5

    # Token management
    max_response_tokens: int = 20000
    content_summary_threshold: int = 20000

    # Centralized Path Configuration
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

    @property
    def kevin_sessions_dir(self) -> str:
        """Get KEVIN sessions directory."""
        return f"{self.kevin_base_dir}/sessions"

    @property
    def kevin_logs_dir(self) -> str:
        """Get KEVIN logs directory."""
        return f"{self.kevin_base_dir}/logs"

    @property
    def kevin_workproducts_dir(self) -> str:
        """Get KEVIN work products directory with environment override support."""
        if hasattr(self, '_kevin_workproducts_dir_override') and self._kevin_workproducts_dir_override:
            return self._kevin_workproducts_dir_override
        return f"{self.kevin_base_dir}/work_products"

    def get_session_dir(self, session_id: str) -> str:
        """Get session directory path."""
        return f"{self.kevin_sessions_dir}/{session_id}"

    def get_session_working_dir(self, session_id: str) -> str:
        """Get session working directory path."""
        return f"{self.get_session_dir(session_id)}/working"

    def get_session_research_dir(self, session_id: str) -> str:
        """Get session research directory path."""
        return f"{self.get_session_dir(session_id)}/research"

    def get_session_final_dir(self, session_id: str) -> str:
        """Get session final directory path."""
        return f"{self.get_session_dir(session_id)}/final"

    def get_final_report_path(self, session_id: str, filename: str) -> str:
        """Get final report path - CRITICAL: Always use session working directory."""
        return f"{self.get_session_working_dir(session_id)}/{filename}"

    def validate_session_path(self, session_id: str, file_path: str) -> bool:
        """Validate that file path belongs to session directory."""
        session_dir = self.get_session_dir(session_id)
        return file_path.startswith(session_dir)

    # Legacy work product directories - use environment-aware path detection
    def _get_default_workproduct_dir(self) -> str:
        current_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if "claudeagent-multiagent-latest" in current_repo:
            return f"{current_repo}/KEVIN/work_products"
        else:
            return "/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/work_products"

    @property
    def default_workproduct_dir(self) -> str:
        return self._get_default_workproduct_dir()

    # Content cleaning settings
    default_cleanliness_threshold: float = 0.7
    min_content_length_for_cleaning: int = 500
    min_cleaned_content_length: int = 200

    # Crawl settings
    default_crawl_timeout: int = 30000
    max_concurrent_crawls: int = 15
    crawl_retry_attempts: int = 2

    def __post_init__(self):
        """Initialize derived settings with environment variable overrides."""
        # Set KEVIN directory paths from environment if available
        if os.getenv('KEVIN_BASE_DIR'):
            self._kevin_base_dir_override = os.getenv('KEVIN_BASE_DIR')
        else:
            self._kevin_base_dir_override = None

        # Legacy support for workproducts directory override
        if os.getenv('KEVIN_WORKPRODUCTS_DIR'):
            self._kevin_workproducts_dir_override = os.getenv('KEVIN_WORKPRODUCTS_DIR')
        else:
            self._kevin_workproducts_dir_override = None


class SettingsManager:
    """Manages configuration settings for the research system."""

    def __init__(self):
        self._enhanced_search_config = EnhancedSearchConfig()
        self._load_environment_overrides()

    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""

        # Enhanced search settings
        if os.getenv('ENHANCED_SEARCH_NUM_RESULTS'):
            try:
                self._enhanced_search_config.default_num_results = int(os.getenv('ENHANCED_SEARCH_NUM_RESULTS'))
            except ValueError:
                pass

        if os.getenv('ENHANCED_SEARCH_AUTO_CRAWL_TOP'):
            try:
                self._enhanced_search_config.default_auto_crawl_top = int(os.getenv('ENHANCED_SEARCH_AUTO_CRAWL_TOP'))
            except ValueError:
                pass

        if os.getenv('ENHANCED_SEARCH_CRAWL_THRESHOLD'):
            try:
                self._enhanced_search_config.default_crawl_threshold = float(os.getenv('ENHANCED_SEARCH_CRAWL_THRESHOLD'))
            except ValueError:
                pass

        if os.getenv('ENHANCED_SEARCH_ANTI_BOT_LEVEL'):
            try:
                level = int(os.getenv('ENHANCED_SEARCH_ANTI_BOT_LEVEL'))
                if 0 <= level <= 3:
                    self._enhanced_search_config.default_anti_bot_level = level
            except ValueError:
                pass

        if os.getenv('ENHANCED_SEARCH_MAX_CONCURRENT'):
            try:
                self._enhanced_search_config.default_max_concurrent = int(os.getenv('ENHANCED_SEARCH_MAX_CONCURRENT'))
            except ValueError:
                pass

        # Path configuration settings
        self._load_path_environment_overrides()

    def _load_path_environment_overrides(self):
        """Load path configuration overrides from environment variables."""

        # KEVIN directory path overrides
        if os.getenv('KEVIN_BASE_DIR'):
            self._enhanced_search_config._kevin_base_dir_override = os.getenv('KEVIN_BASE_DIR')

        if os.getenv('KEVIN_WORKPRODUCTS_DIR'):
            self._enhanced_search_config._kevin_workproducts_dir_override = os.getenv('KEVIN_WORKPRODUCTS_DIR')

        # Session directory overrides
        if os.getenv('KEVIN_SESSIONS_DIR'):
            self._kevin_sessions_dir_override = os.getenv('KEVIN_SESSIONS_DIR')

        if os.getenv('KEVIN_LOGS_DIR'):
            self._kevin_logs_dir_override = os.getenv('KEVIN_LOGS_DIR')

    # Path configuration management
    def get_kevin_base_dir(self) -> str:
        """Get KEVIN base directory with environment override support."""
        return self._enhanced_search_config.kevin_base_dir

    def get_kevin_sessions_dir(self) -> str:
        """Get KEVIN sessions directory with environment override support."""
        if hasattr(self, '_kevin_sessions_dir_override'):
            return self._kevin_sessions_dir_override
        return self._enhanced_search_config.kevin_sessions_dir

    def get_kevin_logs_dir(self) -> str:
        """Get KEVIN logs directory with environment override support."""
        if hasattr(self, '_kevin_logs_dir_override'):
            return self._kevin_logs_dir_override
        return self._enhanced_search_config.kevin_logs_dir

    def get_session_dir(self, session_id: str) -> str:
        """Get session directory path."""
        return f"{self.get_kevin_sessions_dir()}/{session_id}"

    def get_session_working_dir(self, session_id: str) -> str:
        """Get session working directory path."""
        return f"{self.get_session_dir(session_id)}/working"

    def get_session_research_dir(self, session_id: str) -> str:
        """Get session research directory path."""
        return f"{self.get_session_dir(session_id)}/research"

    def get_session_final_dir(self, session_id: str) -> str:
        """Get session final directory path."""
        return f"{self.get_session_dir(session_id)}/final"

    def get_final_report_path(self, session_id: str, filename: str) -> str:
        """Get final report path - CRITICAL: Always use session working directory."""
        return f"{self.get_session_working_dir(session_id)}/{filename}"

    def validate_session_path(self, session_id: str, file_path: str) -> bool:
        """Validate that file path belongs to session directory."""
        session_dir = self.get_session_dir(session_id)
        return file_path.startswith(session_dir)

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

    @property
    def enhanced_search(self) -> EnhancedSearchConfig:
        """Get enhanced search configuration."""
        return self._enhanced_search_config

    def get_anti_bot_description(self, level: int) -> str:
        """Get description for anti-bot level."""
        return self._enhanced_search_config.anti_bot_levels.get(level, "unknown")

    def validate_anti_bot_level(self, level: int) -> int:
        """Validate and clamp anti-bot level to valid range."""
        return max(0, min(3, level))

    def validate_crawl_threshold(self, threshold: float) -> float:
        """Validate and clamp crawl threshold to valid range."""
        return max(0.0, min(1.0, threshold))

    def ensure_workproduct_directory(self, custom_dir: str = None, session_id: str = None, category: str = "research") -> Path:
        """Ensure workproduct directory exists and return path with centralized session-based organization."""
        if custom_dir:
            workproduct_dir = Path(custom_dir)
        elif session_id:
            # Use centralized session-based directory structure
            session_dir = Path(self.get_session_dir(session_id))
            workproduct_dir = session_dir / category
        elif hasattr(self, '_kevin_workproducts_dir_override') and self._kevin_workproducts_dir_override:
            workproduct_dir = Path(self._kevin_workproducts_dir_override)
        else:
            workproduct_dir = Path(self._enhanced_search_config.kevin_workproducts_dir)

        workproduct_dir.mkdir(parents=True, exist_ok=True)
        return workproduct_dir

    def get_default_search_params(self) -> dict[str, Any]:
        """Get default search parameters."""
        return {
            "num_results": self._enhanced_search_config.default_num_results,
            "auto_crawl_top": self._enhanced_search_config.default_auto_crawl_top,
            "crawl_threshold": self._enhanced_search_config.default_crawl_threshold,
            "anti_bot_level": self._enhanced_search_config.default_anti_bot_level,
            "max_concurrent": self._enhanced_search_config.default_max_concurrent,
            "session_id": "default"
        }

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
                "kevin_workproducts_dir": self._enhanced_search_config.kevin_workproducts_dir
            },
            "path_configuration": {
                "kevin_base_dir": self.get_kevin_base_dir(),
                "kevin_sessions_dir": self.get_kevin_sessions_dir(),
                "kevin_logs_dir": self.get_kevin_logs_dir(),
                "kevin_workproducts_dir": self._enhanced_search_config.kevin_workproducts_dir,
                "session_working_dir_example": self.get_session_working_dir("example-session-id"),
                "final_report_path_example": self.get_final_report_path("example-session-id", "FINAL_report.md")
            },
            "environment_variables": {
                "SERP_API_KEY": "SET" if os.getenv('SERP_API_KEY') else "NOT_SET",
                "OPENAI_API_KEY": "SET" if os.getenv('OPENAI_API_KEY') else "NOT_SET",
                "ANTHROPIC_API_KEY": "SET" if os.getenv('ANTHROPIC_API_KEY') else "NOT_SET",
                "KEVIN_BASE_DIR": os.getenv('KEVIN_BASE_DIR', 'NOT_SET'),
                "KEVIN_SESSIONS_DIR": os.getenv('KEVIN_SESSIONS_DIR', 'NOT_SET'),
                "KEVIN_LOGS_DIR": os.getenv('KEVIN_LOGS_DIR', 'NOT_SET'),
                "KEVIN_WORKPRODUCTS_DIR": os.getenv('KEVIN_WORKPRODUCTS_DIR', 'NOT_SET')
            },
            "path_validation": {
                "kevin_base_exists": Path(self.get_kevin_base_dir()).exists(),
                "sessions_dir_exists": Path(self.get_kevin_sessions_dir()).exists(),
                "logs_dir_exists": Path(self.get_kevin_logs_dir()).exists(),
                "workproducts_dir_exists": Path(self._enhanced_search_config.kevin_workproducts_dir).exists()
            }
        }


# Global settings instance
_settings_manager = None


def get_settings() -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager


def get_enhanced_search_config() -> EnhancedSearchConfig:
    """Get enhanced search configuration."""
    return get_settings().enhanced_search


# Export functions
__all__ = [
    'EnhancedSearchConfig',
    'SettingsManager',
    'get_settings',
    'get_enhanced_search_config'
]
