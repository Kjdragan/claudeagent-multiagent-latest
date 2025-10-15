"""
Enhanced Two-Module Scraping Architecture

Phase 1.4: Build Enhanced Two-Module Scraping Architecture with data contracts

This module provides a comprehensive async scraping pipeline with worker pools,
data contracts validation, and integration with anti-bot and content cleaning systems.

Key Features:
- AsyncScrapingOrchestrator with 40/20 worker pools (scrape/clean)
- Pydantic data contracts for strict validation
- Integration with anti-bot escalation system (Phase 1.2)
- Integration with content cleaning pipeline (Phase 1.3)
- Comprehensive error handling and recovery mechanisms
- Performance monitoring and optimization
- Backpressure management and queue control

Architecture:
    Data Contracts → Async Orchestrator → Worker Pools → Integration Layer
"""

from .data_contracts import (
    # Enums
    TaskStatus,
    PipelineStage,
    ErrorType,
    Priority,
    ValidationLevel,

    # Core Models
    TaskContext,
    ScrapingRequest,
    ScrapingResult,
    CleaningRequest,
    CleaningResult,

    # Configuration and Statistics
    PipelineConfig,
    PipelineStatistics,

    # Validation
    DataContractValidator,
    ValidationError,
    DataValidator,

    # Factory Functions
    create_scraping_request,
    create_cleaning_request,
    create_pipeline_config,
)

from .integration import ScrapingPipelineAPI, quick_scrape_and_clean, batch_process_urls

# Import async orchestrator and other components
try:
    from .orchestrator import AsyncScrapingOrchestrator, managed_orchestrator
    from .error_recovery import ErrorRecoveryManager
except ImportError:
    # Fallback for standalone usage
    AsyncScrapingOrchestrator = None
    managed_orchestrator = None
    ErrorRecoveryManager = None

# Add optional components to __all__ if they were imported successfully
if AsyncScrapingOrchestrator is not None:
    __all__.extend(['AsyncScrapingOrchestrator', 'managed_orchestrator'])
if ErrorRecoveryManager is not None:
    __all__.append('ErrorRecoveryManager')

__version__ = "1.4.0"
__author__ = "Multi-Agent Research System"

__all__ = [
    # Enums
    'TaskStatus',
    'PipelineStage',
    'ErrorType',
    'Priority',
    'ValidationLevel',

    # Core Models
    'TaskContext',
    'ScrapingRequest',
    'ScrapingResult',
    'CleaningRequest',
    'CleaningResult',

    # Configuration and Statistics
    'PipelineConfig',
    'PipelineStatistics',

    # Validation
    'DataContractValidator',
    'ValidationError',
    'DataValidator',

    # Factory Functions
    'create_scraping_request',
    'create_cleaning_request',
    'create_pipeline_config',

    # API
    'ScrapingPipelineAPI',

    # Convenience Functions
    'quick_scrape_and_clean',
    'batch_process_urls',
]