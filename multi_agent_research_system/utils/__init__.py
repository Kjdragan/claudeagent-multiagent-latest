"""
Utility functions for the multi-agent research system.

This module provides core utility functions for web crawling, content processing,
intelligent search strategy selection, anti-bot detection, and research data
standardization that power the multi-agent research system.

Enhanced Two-Module Scraping Architecture (Phase 1.4):
- Pydantic data contracts for strict validation
- Async orchestrator with 40/20 worker pools
- Comprehensive error handling and recovery mechanisms
"""

from .port_manager import (
    ensure_port_available,
    find_process_using_port,
    get_available_port,
    kill_process_using_port,
)

# Import scraping pipeline (Phase 1.4)
try:
    from .scraping_pipeline import (
        # Core API
        ScrapingPipelineAPI,

        # Convenience functions
        quick_scrape_and_clean,
        batch_process_urls,

        # Data contracts
        TaskContext,
        ScrapingRequest,
        ScrapingResult,
        CleaningRequest,
        CleaningResult,
        PipelineConfig,

        # Async orchestrator
        AsyncScrapingOrchestrator,
        managed_orchestrator,

        # Validation and recovery
        ErrorRecoveryManager,
        ValidationLevel,

        # Factory functions
        create_scraping_request,
        create_cleaning_request,
        create_pipeline_config
    )
    SCRAPING_PIPELINE_AVAILABLE = True
except ImportError as e:
    SCRAPING_PIPELINE_AVAILABLE = False
    print(f"Warning: Scraping pipeline not available: {e}")

__all__ = [
    # Port management
    "find_process_using_port",
    "kill_process_using_port",
    "ensure_port_available",
    "get_available_port",

    # Scraping pipeline (Phase 1.4) - only if available
    "ScrapingPipelineAPI",
    "quick_scrape_and_clean",
    "batch_process_urls",
    "TaskContext",
    "ScrapingRequest",
    "ScrapingResult",
    "CleaningRequest",
    "CleaningResult",
    "PipelineConfig",
    "AsyncScrapingOrchestrator",
    "managed_orchestrator",
    "ErrorRecoveryManager",
    "ValidationLevel",
    "create_scraping_request",
    "create_cleaning_request",
    "create_pipeline_config"
]

# Only include scraping pipeline exports if available
if not SCRAPING_PIPELINE_AVAILABLE:
    __all__ = __all__[:4]  # Only keep port management exports
