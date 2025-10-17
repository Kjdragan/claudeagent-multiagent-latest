"""
Configuration Settings for Pydantic AI Multi-Agent System

This module handles all configuration settings including environment variables,
API keys, agent configurations, and system parameters.

Key Features:
- Environment variable management
- API key validation
- Agent-specific configurations
- Development/production settings
- Logging configuration
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
import logging


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # AI Model APIs
    openai_api_key: str = Field(..., env="OPENAI_API_KEY", description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY", description="Anthropic API key")
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY", description="Google AI API key")

    # External Services
    serp_api_key: str = Field(..., env="SERP_API_KEY", description="SERP API key")
    youtube_api_key: Optional[str] = Field(None, env="YOUTUBE_API_KEY", description="YouTube Data API key")

    # System Settings
    log_level: str = Field("INFO", env="LOG_LEVEL", description="Logging level")
    max_tokens: int = Field(8192, env="MAX_TOKENS", description="Maximum tokens per request")
    temperature: float = Field(0.7, env="TEMPERATURE", description="Default model temperature")

    # Agent Configuration
    orchestrator_model: str = Field("openai:gpt-5-mini", env="ORCHESTRATOR_MODEL", description="Orchestrator agent model")
    youtube_model: str = Field("openai:gpt-5-mini", env="YOUTUBE_MODEL", description="YouTube agent model")
    tool_specialty_model: str = Field("openai:gpt-5-mini", env="TOOL_SPECIALTY_MODEL", description="Tool specialist model")
    search_model: str = Field("openai:gpt-5-mini", env="SEARCH_MODEL", description="Search agent model")
    crawl4ai_model: str = Field("openai:gpt-5-mini", env="CRAWL4AI_MODEL", description="Crawl4AI agent model")

    # Crawling Configuration (from our enhancements)
    target_successful_cleans: int = Field(10, env="TARGET_SUCCESSFUL_CLEANS", description="Target number of successfully cleaned articles")
    scrape_attempt_multiplier: float = Field(2.0, env="SCRAPE_ATTEMPT_MULTIPLIER", description="Multiplier for scrape attempts (scrape_attempts = target_cleans * multiplier)")
    early_cutoff_multiplier: float = Field(1.25, env="EARLY_CUTOFF_MULTIPLIER", description="Multiplier for early scraping cutoff (cutoff = target_cleans * multiplier)")
    max_total_urls_to_process: int = Field(50, env="MAX_TOTAL_URLS_TO_PROCESS", description="Maximum total URLs to process")
    enable_success_based_termination: bool = Field(True, env="ENABLE_SUCCESS_BASED_TERMINATION", description="Use success-based termination")
    primary_batch_size: int = Field(16, env="PRIMARY_BATCH_SIZE", description="Primary batch size for concurrent scraping")
    secondary_batch_size: int = Field(16, env="SECONDARY_BATCH_SIZE", description="Secondary batch size for concurrent scraping")
    pdf_processing_enabled: bool = Field(True, env="PDF_PROCESSING_ENABLED", description="Enable PDF processing")
    
    @property
    def max_scrape_attempts(self) -> int:
        """Calculate maximum scrape attempts based on target cleans and multiplier."""
        return int(self.target_successful_cleans * self.scrape_attempt_multiplier)
    
    @property
    def early_cutoff_threshold(self) -> int:
        """Calculate early cutoff threshold for successful scrapes (rounded up)."""
        import math
        return math.ceil(self.target_successful_cleans * self.early_cutoff_multiplier)

    # Original z-playground1 settings
    auto_crawl_top_default: int = Field(10, env="AUTO_CRAWL_TOP_DEFAULT", description="Default number of URLs to auto-crawl per round")
    crawl_relevance_threshold: float = Field(0.15, env="CRAWL_RELEVANCE_THRESHOLD", description="Minimum relevance score for crawling")
    concurrent_crawl_limit: int = Field(16, env="CONCURRENT_CRAWL_LIMIT", description="Maximum concurrent crawling operations")  # Updated to 16
    crawl_success_target: int = Field(15, env="CRAWL_SUCCESS_TARGET", description="Target number of successful crawls before early termination")  # Updated to 15
    crawl_timeout_seconds: float = Field(180.0, env="CRAWL_TIMEOUT_SECONDS", description="Timeout for crawling operations")
    crawl_max_retries: int = Field(1, env="CRAWL_MAX_RETRIES", description="Maximum retry attempts per URL")  # Updated to 1 per user directive
    url_selection_limit_default: int = Field(10, env="URL_SELECTION_LIMIT_DEFAULT", description="Default limit for URL selection")

    # Rate Limiting
    max_requests_per_minute: int = Field(5000, env="MAX_REQUESTS_PER_MINUTE", description="Rate limit per minute")
    max_tokens_per_minute: int = Field(4000000, env="MAX_TOKENS_PER_MINUTE", description="Token limit per minute")

    # Cache Settings
    cache_ttl: int = Field(300, env="CACHE_TTL", description="Cache TTL in seconds")
    enable_caching: bool = Field(True, env="ENABLE_CACHING", description="Enable response caching")

    # HTTP Client Settings
    http_timeout: float = Field(30.0, env="HTTP_TIMEOUT", description="HTTP request timeout")
    http_max_retries: int = Field(3, env="HTTP_MAX_RETRIES", description="Maximum HTTP retries")
    http_rate_limit: float = Field(83.3, env="HTTP_RATE_LIMIT", description="HTTP requests per second")

    # Development Settings
    debug_mode: bool = Field(False, env="DEBUG_MODE", description="Enable debug mode")
    development_mode: bool = Field(False, env="DEVELOPMENT_MODE", description="Enable development features")
    enable_metrics: bool = Field(True, env="ENABLE_METRICS", description="Enable performance metrics")

    # LLM Gap Research Evaluator Settings
    llm_gap_research_enabled: bool = Field(True, env="LLM_GAP_RESEARCH_ENABLED", description="Enable LLM-based gap research evaluation")
    llm_gap_research_model: str = Field("gpt-5-nano", env="LLM_GAP_RESEARCH_MODEL", description="Model for LLM gap research evaluation")
    llm_gap_research_strictness: str = Field("standard", env="LLM_GAP_RESEARCH_STRICTNESS", description="Gap research evaluation strictness (lenient/standard/strict)")
    llm_gap_research_timeout: int = Field(30, env="LLM_GAP_RESEARCH_TIMEOUT", description="Timeout for LLM gap research evaluation (seconds)")
    llm_gap_research_max_tokens: int = Field(500, env="LLM_GAP_RESEARCH_MAX_TOKENS", description="Max tokens for LLM gap research evaluation")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Validate model temperature."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        """Validate max tokens."""
        if v <= 0:
            raise ValueError("Max tokens must be positive")
        return v

    def get_model_for_agent(self, agent_name: str) -> str:
        """Get the configured model for a specific agent."""
        model_mapping = {
            "orchestrator": self.orchestrator_model,
            "youtube": self.youtube_model,
            "tool_specialty": self.tool_specialty_model,
            "search": self.search_model,
            "crawl4ai": self.crawl4ai_model
        }
        return model_mapping.get(agent_name, self.orchestrator_model)

    def get_api_key_for_model(self, model: str) -> Optional[str]:
        """Get the appropriate API key for a model."""
        if model.startswith("openai:"):
            return self.openai_api_key
        elif model.startswith("anthropic:"):
            return self.anthropic_api_key
        elif model.startswith("google:"):
            return self.google_api_key
        else:
            # Default to OpenAI for unknown providers
            return self.openai_api_key

    def is_api_key_available(self, service: str) -> bool:
        """Check if an API key is available for a service."""
        key_mapping = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "serper": self.serper_api_key,
            "youtube": self.youtube_api_key
        }
        key = key_mapping.get(service.lower())
        return key is not None and key.strip() != ""

    def get_missing_api_keys(self) -> List[str]:
        """Get list of missing required API keys."""
        missing = []

        # Required keys
        if not self.openai_api_key or self.openai_api_key.strip() == "your_openai_key_here":
            missing.append("OPENAI_API_KEY")

        if not self.serp_api_key or self.serp_api_key.strip() == "your_serp_key_here":
            missing.append("SERP_API_KEY")

        return missing

    def setup_logging(self) -> None:
        """Setup logging configuration with timestamped log files."""
        import os
        from datetime import datetime

        # Create Logs directory if it doesn't exist
        logs_dir = os.path.join(os.getcwd(), 'Logs')
        os.makedirs(logs_dir, exist_ok=True)

        # Generate timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"multi_agent_system_{timestamp}.log"
        log_path = os.path.join(logs_dir, log_filename)

        # Configure logging with fresh file for each run
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, mode='w') if not self.development_mode else logging.NullHandler()
            ],
            force=True  # Force reconfiguration to ensure fresh setup
        )

        # Log the log file location for reference
        logger = logging.getLogger(__name__)
        if not self.development_mode:
            logger.info(f"📝 Session log file: {log_path}")

        # Set specific logger levels
        if self.debug_mode:
            logging.getLogger("agents").setLevel(logging.DEBUG)
            logging.getLogger("utils").setLevel(logging.DEBUG)
        else:
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)


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
        target_successful_cleans = 10
        scrape_attempt_multiplier = 2.0
        early_cutoff_multiplier = 1.25
        concurrent_crawl_limit = 16
        
        @property
        def max_scrape_attempts(self):
            return int(self.target_successful_cleans * self.scrape_attempt_multiplier)
        
        @property
        def early_cutoff_threshold(self):
            import math
            return math.ceil(self.target_successful_cleans * self.early_cutoff_multiplier)

        def setup_logging(self):
            logging.basicConfig(level=logging.INFO)

        def get_missing_api_keys(self):
            return ["OPENAI_API_KEY", "SERP_API_KEY"]

    settings = DefaultSettings()
    settings.setup_logging()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings