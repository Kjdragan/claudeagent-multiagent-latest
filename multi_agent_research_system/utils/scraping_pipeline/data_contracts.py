"""
Enhanced Two-Module Scraping Architecture - Data Contracts

Phase 1.4.1: Create Pydantic data contracts for scraper→cleaner pipeline

This module defines the core data contracts and validation schemas for the
enhanced two-module scraping architecture. It provides strict type safety and
validation to prevent the "'str' object has no attribute 'get'" failures.

Key Features:
- ScrapingResult and CleaningResult Pydantic models with strict validation
- TaskContext and workflow state management
- Pipeline configuration and performance tracking
- Comprehensive error handling and recovery strategies
- Integration with anti-bot escalation system (Phase 1.2)
- Integration with content cleaning pipeline (Phase 1.3)

Based on Technical Enhancements Section 3: Explicit Data Contracts & Validation Schemas
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Literal, Set,
    Protocol, runtime_checkable
)
from uuid import uuid4

from pydantic import (
    BaseModel, Field, validator, ValidationError, conlist,
    constr, confloat, conint, HttpUrl, root_validator
)

# Import existing types from previous phases
try:
    from ..anti_bot.escalation_manager import EscalationResult, AntiBotLevel
    from ..content_cleaning.content_cleaning_pipeline import CleaningResult as ContentCleaningResult
    from ..search_types import SearchQuery, SearchResult
    PHASE_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phase integration not fully available: {e}")
    PHASE_INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class PipelineStage(str, Enum):
    """Pipeline stage enumeration."""
    INITIALIZATION = "initialization"
    SEARCH = "search"
    SCRAPING = "scraping"
    CLEANING = "cleaning"
    VALIDATION = "validation"
    COMPLETED = "completed"


class ErrorType(str, Enum):
    """Error type classification."""
    NETWORK_ERROR = "network_error"
    ANTI_BOT_DETECTION = "anti_bot_detection"
    CONTENT_EXTRACTION_ERROR = "content_extraction_error"
    CLEANING_ERROR = "cleaning_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    UNKNOWN_ERROR = "unknown_error"


class Priority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskContext(BaseModel):
    """Context information for scraping/cleaning tasks."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    priority: Priority = Field(default=Priority.NORMAL)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Workflow tracking
    pipeline_stage: PipelineStage = Field(default=PipelineStage.INITIALIZATION)
    previous_stages: List[PipelineStage] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)

    # Performance tracking
    estimated_duration: Optional[float] = Field(None, description="Estimated duration in seconds")
    actual_duration: Optional[float] = Field(None, description="Actual duration in seconds")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ScrapingRequest(BaseModel):
    """Request model for scraping operations."""

    url: HttpUrl
    search_query: Optional[str] = Field(None, description="Original search query for context")
    context: TaskContext = Field(default_factory=TaskContext)

    # Scraping configuration
    anti_bot_level: Optional[int] = Field(None, ge=0, le=3, description="Starting anti-bot level")
    max_anti_bot_level: int = Field(default=3, ge=0, le=3, description="Maximum anti-bot level")
    timeout_seconds: int = Field(default=120, ge=5, le=600, description="Request timeout")
    enable_media_optimization: bool = Field(default=True, description="Enable media optimization")

    # Content requirements
    min_content_length: int = Field(default=500, ge=0, description="Minimum content length")
    max_content_length: Optional[int] = Field(None, ge=0, description="Maximum content length")
    require_clean_content: bool = Field(default=True, description="Require content cleaning")

    # Quality thresholds
    min_quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    content_cleanliness_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @validator('url')
    def validate_url(cls, v):
        """Validate URL format and accessibility."""
        url_str = str(v)
        if not url_str.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v

    @validator('max_anti_bot_level')
    def validate_anti_bot_levels(cls, v, values):
        """Validate anti-bot level consistency."""
        if 'anti_bot_level' in values and values['anti_bot_level'] is not None:
            if v < values['anti_bot_level']:
                raise ValueError("max_anti_bot_level must be >= anti_bot_level")
        return v


class ScrapingResult(BaseModel):
    """Result model for scraping operations with comprehensive validation."""

    # Core result data
    url: str
    domain: str
    success: bool
    content: Optional[str] = None

    # Performance metrics
    duration: float = Field(ge=0.0, description="Total duration in seconds")
    attempts_made: int = Field(ge=1, description="Number of attempts made")
    word_count: int = Field(default=0, ge=0, description="Word count of content")
    char_count: int = Field(default=0, ge=0, description="Character count of content")

    # Anti-bot escalation data
    final_anti_bot_level: int = Field(ge=0, le=3, description="Final anti-bot level used")
    escalation_used: bool = Field(default=False, description="Whether escalation was used")
    escalation_triggers: List[str] = Field(default_factory=list, description="Escalation triggers detected")

    # Quality metrics
    content_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Content quality score")
    cleanliness_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Content cleanliness score")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance to search query")

    # Context and metadata
    context: TaskContext
    search_query: Optional[str] = None
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Error handling
    error_message: Optional[str] = None
    error_type: Optional[ErrorType] = None
    retry_recommendation: bool = Field(default=False, description="Whether retry is recommended")

    # Content analysis
    content_type: Optional[str] = Field(None, description="Detected content type (article, blog, etc.)")
    language_detected: Optional[str] = Field(None, description="Detected content language")
    has_structured_data: bool = Field(default=False, description="Whether structured data was found")

    @validator('domain')
    def extract_domain(cls, v, values):
        """Ensure domain is extracted from URL."""
        if 'url' in values and not v:
            from urllib.parse import urlparse
            parsed = urlparse(values['url'])
            return parsed.netloc
        return v

    @validator('word_count', 'char_count')
    def calculate_content_metrics(cls, v, values):
        """Calculate content metrics if content is provided."""
        if 'content' in values and values['content'] and v == 0:
            content = values['content']
            if cls.__name__ == 'ScrapingResult' or 'word_count' in cls.__fields__:
                return len(content.split())
            else:
                return len(content)
        return v

    @validator('content_quality_score', 'cleanliness_score', 'relevance_score')
    def validate_score_ranges(cls, v):
        """Validate score ranges."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Scores must be between 0.0 and 1.0")
        return v

    @root_validator(skip_on_failure=True)
    def validate_result_consistency(cls, values):
        """Validate overall result consistency."""
        success = values.get('success', False)
        content = values.get('content')
        error_message = values.get('error_message')

        if success and not content:
            raise ValueError("Successful result must have content")

        if not success and content:
            # Allow content on failed results for debugging
            logger.debug("Failed result has content - may be partial extraction")

        if not success and not error_message:
            raise ValueError("Failed result must have error message")

        return values

    def is_high_quality(self, min_quality: float = 0.7) -> bool:
        """Check if result meets quality thresholds."""
        if not self.success or not self.content_quality_score:
            return False
        return (
            self.content_quality_score >= min_quality and
            (self.cleanliness_score or 0.0) >= 0.6 and
            self.word_count >= 100
        )

    def should_retry(self) -> bool:
        """Determine if this result should be retried."""
        if self.success or not self.retry_recommendation:
            return False

        # Don't retry certain error types
        if self.error_type in [ErrorType.CONTENT_EXTRACTION_ERROR, ErrorType.VALIDATION_ERROR]:
            return False

        # Check retry count
        return self.context.retry_count < self.context.max_retries


class CleaningRequest(BaseModel):
    """Request model for content cleaning operations."""

    content: str = Field(min_length=10, description="Content to clean")
    url: str = Field(description="Source URL for context")
    search_query: Optional[str] = Field(None, description="Original search query")
    context: TaskContext = Field(default_factory=TaskContext)

    # Cleaning configuration
    cleaning_intensity: Literal["light", "medium", "aggressive"] = Field(default="medium")
    enable_ai_cleaning: bool = Field(default=True, description="Enable AI-powered cleaning")
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Target quality threshold")

    # Content preservation
    preserve_links: bool = Field(default=True, description="Preserve hyperlinks")
    preserve_formatting: bool = Field(default=False, description="Preserve original formatting")
    max_length_reduction: float = Field(default=0.5, ge=0.0, le=1.0, description="Maximum length reduction")

    # Quality assessment
    assess_quality: bool = Field(default=True, description="Perform quality assessment")
    generate_suggestions: bool = Field(default=True, description="Generate enhancement suggestions")

    @validator('content')
    def validate_content_length(cls, v):
        """Validate minimum content length."""
        if len(v.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        return v


class CleaningResult(BaseModel):
    """Result model for content cleaning operations."""

    # Core result data
    original_content: str
    cleaned_content: str
    url: str
    success: bool = True

    # Cleaning metrics
    cleaning_performed: bool = Field(default=False, description="Whether cleaning was performed")
    quality_improvement: float = Field(default=0.0, description="Quality improvement score")
    length_reduction: float = Field(default=0.0, description="Content length reduction ratio")

    # Quality assessment
    original_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    final_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    cleanliness_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Context and metadata
    context: TaskContext
    search_query: Optional[str] = None
    cleaning_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Processing information
    cleaning_stage: str = Field(default="none", description="Cleaning stage used")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Processing time in milliseconds")

    # Enhancement information
    editorial_recommendation: str = Field(default="UNKNOWN", description="Editorial assessment")
    enhancement_suggestions: List[str] = Field(default_factory=list, description="Enhancement suggestions")

    # Error handling
    error_message: Optional[str] = None
    error_type: Optional[ErrorType] = None
    fallback_used: bool = Field(default=False, description="Whether fallback cleaning was used")

    @validator('final_quality_score', 'cleanliness_score')
    def validate_final_scores(cls, v):
        """Validate final score ranges."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Final scores must be between 0.0 and 1.0")
        return v

    @validator('length_reduction')
    def validate_length_reduction(cls, v):
        """Validate length reduction range."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Length reduction must be between 0.0 and 1.0")
        return v

    def is_improvement(self) -> bool:
        """Check if cleaning improved content quality."""
        if not self.cleaning_performed:
            return False
        return (
            self.quality_improvement > 0.1 or
            (self.final_quality_score or 0.0) > (self.original_quality_score or 0.0) + 0.1
        )

    def needs_enhancement(self, threshold: float = 0.8) -> bool:
        """Check if content needs further enhancement."""
        return (self.final_quality_score or 0.0) < threshold


class PipelineConfig(BaseModel):
    """Configuration for the scraping pipeline."""

    # Concurrency settings
    max_scrape_workers: int = Field(default=40, ge=1, le=100, description="Maximum scrape workers")
    max_clean_workers: int = Field(default=20, ge=1, le=50, description="Maximum clean workers")
    worker_timeout_seconds: int = Field(default=300, ge=30, le=1800, description="Worker timeout")

    # Queue management
    max_queue_size: int = Field(default=1000, ge=10, le=10000, description="Maximum queue size")
    backpressure_threshold: float = Field(default=0.8, ge=0.1, le=1.0, description="Backpressure threshold")

    # Quality thresholds
    default_quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    min_acceptable_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_quality_gates: bool = Field(default=True, description="Enable quality gates")

    # Retry and error handling
    default_max_retries: int = Field(default=3, ge=0, le=10, description="Default max retries")
    retry_delays: List[float] = Field(default=[1.0, 2.0, 5.0], description="Retry delays in seconds")
    enable_circuit_breaker: bool = Field(default=True, description="Enable circuit breaker")

    # Performance optimization
    enable_batch_processing: bool = Field(default=True, description="Enable batch processing")
    batch_size: int = Field(default=10, ge=1, le=100, description="Batch size for processing")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=3600, ge=60, description="Cache TTL in seconds")

    # Integration settings
    enable_anti_bot: bool = Field(default=True, description="Enable anti-bot escalation")
    enable_content_cleaning: bool = Field(default=True, description="Enable content cleaning")
    enable_quality_assessment: bool = Field(default=True, description="Enable quality assessment")

    # Monitoring and logging
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    log_level: str = Field(default="INFO", description="Logging level")
    enable_metrics_export: bool = Field(default=False, description="Enable metrics export")

    @validator('retry_delays')
    def validate_retry_delays(cls, v):
        """Validate retry delays are positive and increasing."""
        if any(d <= 0 for d in v):
            raise ValueError("Retry delays must be positive")
        if v != sorted(v):
            raise ValueError("Retry delays must be in ascending order")
        return v


class PipelineStatistics(BaseModel):
    """Pipeline performance and statistics tracking."""

    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Task counts
    total_tasks: int = Field(default=0, ge=0)
    completed_tasks: int = Field(default=0, ge=0)
    failed_tasks: int = Field(default=0, ge=0)
    cancelled_tasks: int = Field(default=0, ge=0)

    # Performance metrics
    total_duration: float = Field(default=0.0, ge=0.0, description="Total duration in seconds")
    avg_task_duration: float = Field(default=0.0, ge=0.0, description="Average task duration")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")

    # Quality metrics
    avg_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average quality score")
    high_quality_results: int = Field(default=0, ge=0, description="High quality results count")

    # Worker metrics
    active_scrape_workers: int = Field(default=0, ge=0)
    active_clean_workers: int = Field(default=0, ge=0)
    queue_size: int = Field(default=0, ge=0)

    # Error metrics
    error_counts: Dict[ErrorType, int] = Field(default_factory=dict)
    escalation_count: int = Field(default=0, ge=0, description="Number of escalations")
    retry_count: int = Field(default=0, ge=0, description="Number of retries")

    def update_success_rate(self):
        """Update success rate calculation."""
        if self.total_tasks > 0:
            self.success_rate = self.completed_tasks / self.total_tasks

    def add_task_result(self, duration: float, success: bool, quality_score: Optional[float] = None):
        """Add a task result to statistics."""
        self.total_tasks += 1
        self.total_duration += duration
        self.avg_task_duration = self.total_duration / self.total_tasks

        if success:
            self.completed_tasks += 1
            if quality_score and quality_score >= 0.8:
                self.high_quality_results += 1
            if quality_score:
                # Update average quality score
                self.avg_quality_score = (
                    (self.avg_quality_score * (self.completed_tasks - 1) + quality_score) /
                    self.completed_tasks
                )
        else:
            self.failed_tasks += 1

        self.update_success_rate()
        self.updated_at = datetime.now()


class ValidationError(Exception):
    """Custom validation error for pipeline data contracts."""
    pass


@runtime_checkable
class DataValidator(Protocol):
    """Protocol for data validation implementations."""

    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """Validate data and return (is_valid, error_message)."""
        ...


class DataContractValidator:
    """Validator for pipeline data contracts."""

    def __init__(self, config: PipelineConfig):
        """Initialize the validator with pipeline configuration."""
        self.config = config
        self.custom_validators: Dict[str, List[DataValidator]] = {}

    def validate_scraping_request(self, request: ScrapingRequest) -> Tuple[bool, Optional[str]]:
        """Validate a scraping request."""
        try:
            # Pydantic validation
            ScrapingRequest.parse_obj(request.dict())

            # Custom validation
            if request.min_content_length > request.max_content_length if request.max_content_length else False:
                return False, "min_content_length cannot exceed max_content_length"

            # Check domain blacklist/whitelist if configured
            if self._is_domain_restricted(request.url):
                return False, f"Domain {request.url} is restricted"

            return True, None

        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected validation error: {str(e)}"

    def validate_scraping_result(self, result: ScrapingResult) -> Tuple[bool, Optional[str]]:
        """Validate a scraping result."""
        try:
            # Pydantic validation
            ScrapingResult.parse_obj(result.dict())

            # Business logic validation
            if result.success and not result.content:
                return False, "Successful result must contain content"

            if result.content_quality_score and (result.content_quality_score < 0 or result.content_quality_score > 1):
                return False, "Content quality score must be between 0 and 1"

            # Check if result meets minimum quality standards
            if self.config.enable_quality_gates:
                if not result.is_high_quality(self.config.min_acceptable_quality):
                    logger.debug(f"Result below quality threshold: {result.url}")

            return True, None

        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected validation error: {str(e)}"

    def validate_cleaning_request(self, request: CleaningRequest) -> Tuple[bool, Optional[str]]:
        """Validate a cleaning request."""
        try:
            # Pydantic validation
            CleaningRequest.parse_obj(request.dict())

            # Content validation
            if len(request.content.strip()) < 10:
                return False, "Content must be at least 10 characters long"

            # Check if content needs cleaning
            if request.cleanliness_score and request.cleanliness_score >= 0.9:
                logger.debug(f"Content already clean: {request.url}")

            return True, None

        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected validation error: {str(e)}"

    def validate_cleaning_result(self, result: CleaningResult) -> Tuple[bool, Optional[str]]:
        """Validate a cleaning result."""
        try:
            # Pydantic validation
            CleaningResult.parse_obj(result.dict())

            # Business logic validation
            if result.cleaning_performed and result.cleaned_content == result.original_content:
                logger.warning("Cleaning performed but content unchanged")

            if result.final_quality_score and (result.final_quality_score < 0 or result.final_quality_score > 1):
                return False, "Final quality score must be between 0 and 1"

            return True, None

        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected validation error: {str(e)}"

    def _is_domain_restricted(self, url: str) -> bool:
        """Check if domain is restricted (placeholder for implementation)."""
        # This could be extended to check against blacklists/whitelists
        return False

    def register_validator(self, data_type: str, validator: DataValidator):
        """Register a custom validator for a data type."""
        if data_type not in self.custom_validators:
            self.custom_validators[data_type] = []
        self.custom_validators[data_type].append(validator)


# Factory functions for easy object creation
def create_scraping_request(
    url: str,
    search_query: Optional[str] = None,
    **kwargs
) -> ScrapingRequest:
    """Create a scraping request with sensible defaults."""
    return ScrapingRequest(
        url=url,
        search_query=search_query,
        **kwargs
    )


def create_cleaning_request(
    content: str,
    url: str,
    search_query: Optional[str] = None,
    **kwargs
) -> CleaningRequest:
    """Create a cleaning request with sensible defaults."""
    return CleaningRequest(
        content=content,
        url=url,
        search_query=search_query,
        **kwargs
    )


def create_pipeline_config(**overrides) -> PipelineConfig:
    """Create a pipeline configuration with optional overrides."""
    return PipelineConfig(**overrides)


# Export main classes and functions
__all__ = [
    # Enums
    'TaskStatus', 'PipelineStage', 'ErrorType', 'Priority',

    # Core Models
    'TaskContext', 'ScrapingRequest', 'ScrapingResult',
    'CleaningRequest', 'CleaningResult',

    # Configuration and Statistics
    'PipelineConfig', 'PipelineStatistics',

    # Validation
    'DataContractValidator', 'ValidationError', 'DataValidator',

    # Factory Functions
    'create_scraping_request', 'create_cleaning_request', 'create_pipeline_config',
]