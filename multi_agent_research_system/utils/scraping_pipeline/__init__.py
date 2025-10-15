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

from .integration import ScrapingPipelineAPI

__version__ = "1.4.0"
__author__ = "Multi-Agent Research System"

__all__ = [
    # Enums
    'TaskStatus',
    'PipelineStage',
    'ErrorType',
    'Priority',

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
]