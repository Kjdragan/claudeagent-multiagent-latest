"""
Enhanced Two-Module Scraping Architecture - Validation & Error Recovery

Phase 1.4.3: Build data contract validation and error recovery mechanisms

This module implements comprehensive validation and error recovery mechanisms for the
enhanced two-module scraping architecture, ensuring data integrity and system resilience.

Key Features:
- Data contract validation with Pydantic models
- Sophisticated error recovery strategies
- Fallback mechanisms for failed operations
- Performance monitoring and optimization
- Integration with anti-bot and content cleaning systems
- Comprehensive logging and debugging support

Based on Technical Enhancements Section 3: Explicit Data Contracts & Validation Schemas
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable,
    Set, Type, Protocol, runtime_checkable
)
from uuid import uuid4

from .data_contracts import (
    TaskContext, ScrapingRequest, ScrapingResult,
    CleaningRequest, CleaningResult, PipelineConfig,
    PipelineStatistics, TaskStatus, PipelineStage,
    ErrorType, Priority, DataContractValidator
)

# Import enhanced logging from Phase 1.1
try:
    from ...agent_logging.enhanced_logger import (
        get_enhanced_logger, LogLevel, LogCategory, AgentEventType
    )
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation level enumeration."""
    BASIC = "basic"           # Basic structure validation
    STANDARD = "standard"     # Standard business logic validation
    STRICT = "strict"         # Strict validation with all checks
    CUSTOM = "custom"         # Custom validation rules


class RecoveryStrategy(str, Enum):
    """Recovery strategy enumeration."""
    RETRY = "retry"                      # Simple retry with same parameters
    ESCALATE = "escalate"                # Escalate anti-bot level or cleaning intensity
    FALLBACK = "fallback"                # Use fallback implementation
    SKIP = "skip"                        # Skip and continue
    ABORT = "abort"                      # Abort the entire operation


class ValidationError(Exception):
    """Custom validation error with detailed information."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.field = field
        self.value = value
        self.error_code = error_code
        self.timestamp = datetime.now()


class RecoveryAction:
    """Represents a recovery action to be taken."""

    def __init__(
        self,
        strategy: RecoveryStrategy,
        parameters: Optional[Dict[str, Any]] = None,
        delay_seconds: float = 0.0,
        max_attempts: int = 3
    ):
        self.strategy = strategy
        self.parameters = parameters or {}
        self.delay_seconds = delay_seconds
        self.max_attempts = max_attempts
        self.attempt_count = 0


class ValidationResult:
    """Result of a validation operation."""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[ValidationError]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def add_error(self, error: ValidationError):
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning to the validation result."""
        self.warnings.append(warning)

    def has_critical_errors(self) -> bool:
        """Check if there are critical validation errors."""
        return any(
            error.error_code and error.error_code.startswith('CRITICAL_')
            for error in self.errors
        )


@runtime_checkable
class Validator(Protocol):
    """Protocol for validator implementations."""

    def validate(self, data: Any, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate data and return validation result."""
        ...


class BaseValidator(ABC):
    """Base class for validators."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def validate(self, data: Any, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate data and return validation result."""
        pass

    def _create_error(self, message: str, field: Optional[str] = None, error_code: Optional[str] = None) -> ValidationError:
        """Create a validation error."""
        return ValidationError(
            message=message,
            field=field,
            error_code=error_code
        )


class ScrapingRequestValidator(BaseValidator):
    """Validator for scraping requests."""

    def __init__(self, config: PipelineConfig):
        super().__init__("ScrapingRequestValidator")
        self.config = config
        self.data_validator = DataContractValidator(config)

    def validate(self, request: ScrapingRequest, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate a scraping request."""
        result = ValidationResult(is_valid=True)

        try:
            # Basic validation (always performed)
            self._validate_basic_structure(request, result)

            if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.CUSTOM]:
                self._validate_business_logic(request, result)

            if level in [ValidationLevel.STRICT, ValidationLevel.CUSTOM]:
                self._validate_strict_requirements(request, result)

            if level == ValidationLevel.CUSTOM:
                self._validate_custom_rules(request, result)

        except Exception as e:
            result.add_error(self._create_error(f"Validation error: {str(e)}", error_code="CRITICAL_VALIDATION_ERROR"))

        return result

    def _validate_basic_structure(self, request: ScrapingRequest, result: ValidationResult):
        """Validate basic request structure."""
        # URL validation
        if not request.url:
            result.add_error(self._create_error("URL is required", field="url", error_code="MISSING_URL"))
        else:
            url_str = str(request.url)
            if not url_str.startswith(('http://', 'https://')):
                result.add_error(self._create_error("URL must start with http:// or https://", field="url", error_code="INVALID_URL_FORMAT"))

        # Task context validation
        if not request.context:
            result.add_error(self._create_error("Task context is required", field="context", error_code="MISSING_CONTEXT"))

    def _validate_business_logic(self, request: ScrapingRequest, result: ValidationResult):
        """Validate business logic rules."""
        # Anti-bot level validation
        if request.anti_bot_level is not None:
            if request.anti_bot_level < 0 or request.anti_bot_level > 3:
                result.add_error(self._create_error("Anti-bot level must be between 0 and 3", field="anti_bot_level", error_code="INVALID_ANTI_BOT_LEVEL"))

        if request.max_anti_bot_level < 0 or request.max_anti_bot_level > 3:
            result.add_error(self._create_error("Max anti-bot level must be between 0 and 3", field="max_anti_bot_level", error_code="INVALID_MAX_ANTI_BOT_LEVEL"))

        if request.anti_bot_level is not None and request.max_anti_bot_level < request.anti_bot_level:
            result.add_error(self._create_error("Max anti-bot level must be >= anti-bot level", field="max_anti_bot_level", error_code="ANTI_BOT_LEVEL_MISMATCH"))

        # Content length validation
        if request.min_content_length < 0:
            result.add_error(self._create_error("Min content length must be non-negative", field="min_content_length", error_code="INVALID_MIN_LENGTH"))

        if request.max_content_length is not None and request.max_content_length < request.min_content_length:
            result.add_error(self._create_error("Max content length must be >= min content length", field="max_content_length", error_code="INVALID_LENGTH_RANGE"))

        # Quality threshold validation
        if request.min_quality_score < 0.0 or request.min_quality_score > 1.0:
            result.add_error(self._create_error("Min quality score must be between 0.0 and 1.0", field="min_quality_score", error_code="INVALID_QUALITY_THRESHOLD"))

    def _validate_strict_requirements(self, request: ScrapingResult, result: ValidationResult):
        """Validate strict requirements."""
        # Timeout validation
        if request.timeout_seconds < 5 or request.timeout_seconds > 600:
            result.add_error(self._create_error("Timeout must be between 5 and 600 seconds", field="timeout_seconds", error_code="INVALID_TIMEOUT"))

        # Domain blacklist check (placeholder)
        url_str = str(request.url)
        if self._is_blacklisted_domain(url_str):
            result.add_error(self._create_error(f"Domain is blacklisted: {url_str}", field="url", error_code="BLACKLISTED_DOMAIN"))

    def _validate_custom_rules(self, request: ScrapingRequest, result: ValidationResult):
        """Validate custom business rules."""
        # Add custom validation rules here
        pass

    def _is_blacklisted_domain(self, url: str) -> bool:
        """Check if domain is blacklisted (placeholder implementation)."""
        # This could be extended to check against actual blacklists
        return False


class ScrapingResultValidator(BaseValidator):
    """Validator for scraping results."""

    def __init__(self, config: PipelineConfig):
        super().__init__("ScrapingResultValidator")
        self.config = config
        self.data_validator = DataContractValidator(config)

    def validate(self, result: ScrapingResult, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate a scraping result."""
        validation_result = ValidationResult(is_valid=True)

        try:
            # Basic validation
            self._validate_basic_structure(result, validation_result)

            if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.CUSTOM]:
                self._validate_content_quality(result, validation_result)

            if level in [ValidationLevel.STRICT, ValidationLevel.CUSTOM]:
                self._validate_performance_metrics(result, validation_result)

            if level == ValidationLevel.CUSTOM:
                self._validate_custom_rules(result, validation_result)

        except Exception as e:
            validation_result.add_error(self._create_error(f"Validation error: {str(e)}", error_code="CRITICAL_VALIDATION_ERROR"))

        return validation_result

    def _validate_basic_structure(self, result: ScrapingResult, validation_result: ValidationResult):
        """Validate basic result structure."""
        # Consistency validation
        if result.success and not result.content:
            validation_result.add_error(self._create_error("Successful result must have content", field="content", error_code="MISSING_CONTENT"))

        if not result.success and not result.error_message:
            validation_result.add_error(self._create_error("Failed result must have error message", field="error_message", error_code="MISSING_ERROR_MESSAGE"))

        # URL and domain validation
        if not result.url:
            validation_result.add_error(self._create_error("URL is required", field="url", error_code="MISSING_URL"))

        if result.success and not result.domain:
            validation_result.add_warning("Domain not provided for successful result")

    def _validate_content_quality(self, result: ScrapingResult, validation_result: ValidationResult):
        """Validate content quality metrics."""
        if result.success and result.content:
            # Quality score validation
            if result.content_quality_score is not None:
                if result.content_quality_score < 0.0 or result.content_quality_score > 1.0:
                    validation_result.add_error(self._create_error("Content quality score must be between 0.0 and 1.0", field="content_quality_score", error_code="INVALID_QUALITY_SCORE"))

                if result.content_quality_score < self.config.min_acceptable_quality:
                    validation_result.add_warning(f"Content quality below threshold: {result.content_quality_score:.2f}")

            # Content length validation
            if result.word_count < 10:
                validation_result.add_warning("Very short content detected")

            # Anti-bot level validation
            if result.final_anti_bot_level < 0 or result.final_anti_bot_level > 3:
                validation_result.add_error(self._create_error("Final anti-bot level must be between 0 and 3", field="final_anti_bot_level", error_code="INVALID_FINAL_LEVEL"))

    def _validate_performance_metrics(self, result: ScrapingResult, validation_result: ValidationResult):
        """Validate performance metrics."""
        # Duration validation
        if result.duration < 0:
            validation_result.add_error(self._create_error("Duration cannot be negative", field="duration", error_code="INVALID_DURATION"))

        if result.duration > 300:  # 5 minutes
            validation_result.add_warning(f"Very long duration: {result.duration:.1f}s")

        # Attempts validation
        if result.attempts_made < 1:
            validation_result.add_error(self._create_error("Attempts made must be at least 1", field="attempts_made", error_code="INVALID_ATTEMPTS"))

        if result.attempts_made > 10:
            validation_result.add_warning(f"High number of attempts: {result.attempts_made}")

    def _validate_custom_rules(self, result: ScrapingResult, validation_result: ValidationResult):
        """Validate custom business rules."""
        # Add custom validation rules here
        pass


class CleaningRequestValidator(BaseValidator):
    """Validator for cleaning requests."""

    def __init__(self, config: PipelineConfig):
        super().__init__("CleaningRequestValidator")
        self.config = config

    def validate(self, request: CleaningRequest, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate a cleaning request."""
        result = ValidationResult(is_valid=True)

        try:
            # Basic validation
            self._validate_basic_structure(request, result)

            if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.CUSTOM]:
                self._validate_business_logic(request, result)

            if level in [ValidationLevel.STRICT, ValidationLevel.CUSTOM]:
                self._validate_strict_requirements(request, result)

            if level == ValidationLevel.CUSTOM:
                self._validate_custom_rules(request, result)

        except Exception as e:
            result.add_error(self._create_error(f"Validation error: {str(e)}", error_code="CRITICAL_VALIDATION_ERROR"))

        return result

    def _validate_basic_structure(self, request: CleaningRequest, result: ValidationResult):
        """Validate basic request structure."""
        # Content validation
        if not request.content or len(request.content.strip()) < 10:
            result.add_error(self._create_error("Content must be at least 10 characters", field="content", error_code="INSUFFICIENT_CONTENT"))

        # URL validation
        if not request.url:
            result.add_error(self._create_error("URL is required", field="url", error_code="MISSING_URL"))

    def _validate_business_logic(self, request: CleaningRequest, result: ValidationResult):
        """Validate business logic rules."""
        # Quality threshold validation
        if request.quality_threshold < 0.0 or request.quality_threshold > 1.0:
            result.add_error(self._create_error("Quality threshold must be between 0.0 and 1.0", field="quality_threshold", error_code="INVALID_QUALITY_THRESHOLD"))

        # Length reduction validation
        if request.max_length_reduction < 0.0 or request.max_length_reduction > 1.0:
            result.add_error(self._create_error("Max length reduction must be between 0.0 and 1.0", field="max_length_reduction", error_code="INVALID_REDUCTION_THRESHOLD"))

    def _validate_strict_requirements(self, request: CleaningRequest, result: ValidationResult):
        """Validate strict requirements."""
        # Content analysis
        if len(request.content) > 100000:  # 100KB
            result.add_warning("Very large content for cleaning")

        # Check if content is mostly HTML tags
        import re
        html_tags = re.findall(r'<[^>]+>', request.content)
        if len(html_tags) > len(request.content.split()) * 2:
            result.add_warning("Content appears to be mostly HTML tags")

    def _validate_custom_rules(self, request: CleaningRequest, result: ValidationResult):
        """Validate custom business rules."""
        # Add custom validation rules here
        pass


class CleaningResultValidator(BaseValidator):
    """Validator for cleaning results."""

    def __init__(self, config: PipelineConfig):
        super().__init__("CleaningResultValidator")
        self.config = config

    def validate(self, result: CleaningResult, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate a cleaning result."""
        validation_result = ValidationResult(is_valid=True)

        try:
            # Basic validation
            self._validate_basic_structure(result, validation_result)

            if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.CUSTOM]:
                self._validate_cleaning_quality(result, validation_result)

            if level in [ValidationLevel.STRICT, ValidationLevel.CUSTOM]:
                self._validate_performance_metrics(result, validation_result)

            if level == ValidationLevel.CUSTOM:
                self._validate_custom_rules(result, validation_result)

        except Exception as e:
            validation_result.add_error(self._create_error(f"Validation error: {str(e)}", error_code="CRITICAL_VALIDATION_ERROR"))

        return validation_result

    def _validate_basic_structure(self, result: CleaningResult, validation_result: ValidationResult):
        """Validate basic result structure."""
        # Content validation
        if not result.original_content:
            validation_result.add_error(self._create_error("Original content is required", field="original_content", error_code="MISSING_ORIGINAL_CONTENT"))

        if not result.cleaned_content:
            validation_result.add_error(self._create_error("Cleaned content is required", field="cleaned_content", error_code="MISSING_CLEANED_CONTENT"))

        # URL validation
        if not result.url:
            validation_result.add_error(self._create_error("URL is required", field="url", error_code="MISSING_URL"))

    def _validate_cleaning_quality(self, result: CleaningResult, validation_result: ValidationResult):
        """Validate cleaning quality metrics."""
        # Quality scores validation
        if result.final_quality_score is not None:
            if result.final_quality_score < 0.0 or result.final_quality_score > 1.0:
                validation_result.add_error(self._create_error("Final quality score must be between 0.0 and 1.0", field="final_quality_score", error_code="INVALID_FINAL_QUALITY"))

        if result.cleanliness_score is not None:
            if result.cleanliness_score < 0.0 or result.cleanliness_score > 1.0:
                validation_result.add_error(self._create_error("Cleanliness score must be between 0.0 and 1.0", field="cleanliness_score", error_code="INVALID_CLEANLINESS_SCORE"))

        # Length reduction validation
        if result.length_reduction < 0.0 or result.length_reduction > 1.0:
            validation_result.add_error(self._create_error("Length reduction must be between 0.0 and 1.0", field="length_reduction", error_code="INVALID_LENGTH_REDUCTION"))

        # Check if cleaning was effective
        if result.cleaning_performed and result.cleaned_content == result.original_content:
            validation_result.add_warning("Cleaning performed but content unchanged")

    def _validate_performance_metrics(self, result: CleaningResult, validation_result: ValidationResult):
        """Validate performance metrics."""
        # Processing time validation
        if result.processing_time_ms < 0:
            validation_result.add_error(self._create_error("Processing time cannot be negative", field="processing_time_ms", error_code="INVALID_PROCESSING_TIME"))

        if result.processing_time_ms > 30000:  # 30 seconds
            validation_result.add_warning(f"Very long processing time: {result.processing_time_ms:.1f}ms")

    def _validate_custom_rules(self, result: CleaningResult, validation_result: ValidationResult):
        """Validate custom business rules."""
        # Add custom validation rules here
        pass


class ErrorRecoveryManager:
    """Manages error recovery strategies and execution."""

    def __init__(self, config: PipelineConfig):
        """Initialize the error recovery manager."""
        self.config = config
        self.recovery_history: Dict[str, List[RecoveryAction]] = {}
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'recovery_by_strategy': defaultdict(int)
        }

        # Setup enhanced logging
        self._setup_logging()

        # Initialize validators
        self.validators = {
            'scraping_request': ScrapingRequestValidator(config),
            'scraping_result': ScrapingResultValidator(config),
            'cleaning_request': CleaningRequestValidator(config),
            'cleaning_result': CleaningResultValidator(config)
        }

    def _setup_logging(self):
        """Setup enhanced logging integration."""
        if ENHANCED_LOGGING_AVAILABLE:
            self.enhanced_logger = get_enhanced_logger("error_recovery_manager")
        else:
            self.enhanced_logger = None

    def validate(
        self,
        data_type: str,
        data: Any,
        level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ValidationResult:
        """Validate data using appropriate validator."""
        validator = self.validators.get(data_type)
        if not validator:
            error_msg = f"No validator found for data type: {data_type}"
            logger.error(error_msg)
            return ValidationResult(is_valid=False, errors=[ValidationError(error_msg)])

        try:
            return validator.validate(data, level)
        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            logger.error(error_msg)
            return ValidationResult(is_valid=False, errors=[ValidationError(error_msg, error_code="VALIDATION_EXCEPTION")])

    def determine_recovery_strategy(
        self,
        error: Exception,
        context: TaskContext,
        data_type: str,
        attempt_count: int = 0
    ) -> Optional[RecoveryAction]:
        """Determine the appropriate recovery strategy for an error."""
        error_type = self._classify_error(error)

        # Don't recover if max attempts exceeded
        if attempt_count >= context.max_retries:
            return None

        # Determine strategy based on error type and context
        if error_type == ErrorType.NETWORK_ERROR:
            if attempt_count < 2:
                return RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    delay_seconds=2.0 ** attempt_count,  # Exponential backoff
                    max_attempts=3
                )
            else:
                return RecoveryAction(
                    strategy=RecoveryStrategy.ESCALATE,
                    parameters={'escalate_anti_bot': True},
                    delay_seconds=5.0
                )

        elif error_type == ErrorType.ANTI_BOT_DETECTION:
            return RecoveryAction(
                strategy=RecoveryStrategy.ESCALATE,
                parameters={'escalate_anti_bot': True},
                delay_seconds=1.0 * (attempt_count + 1)
            )

        elif error_type == ErrorType.CONTENT_EXTRACTION_ERROR:
            if attempt_count < 2:
                return RecoveryAction(
                    strategy=RecoveryStrategy.ESCALATE,
                    parameters={'enable_javascript': True},
                    delay_seconds=2.0
                )
            else:
                return RecoveryAction(strategy=RecoveryStrategy.FALLBACK)

        elif error_type == ErrorType.CLEANING_ERROR:
            if attempt_count < 1:
                return RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    parameters={'cleaning_intensity': 'light'},
                    delay_seconds=1.0
                )
            else:
                return RecoveryAction(strategy=RecoveryStrategy.FALLBACK)

        elif error_type == ErrorType.TIMEOUT_ERROR:
            return RecoveryAction(
                strategy=RecoveryStrategy.ESCALATE,
                parameters={'increase_timeout': True},
                delay_seconds=3.0
            )

        elif error_type == ErrorType.RATE_LIMIT_ERROR:
            return RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                delay_seconds=30.0 * (attempt_count + 1),  # Longer delays for rate limiting
                max_attempts=2
            )

        else:
            # Unknown error - try simple retry first
            if attempt_count < 1:
                return RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    delay_seconds=1.0
                )

        return None

    async def execute_recovery(
        self,
        recovery_action: RecoveryAction,
        original_func: Callable,
        *args,
        **kwargs
    ) -> Tuple[bool, Any]:
        """Execute a recovery action.

        Args:
            recovery_action: Recovery action to execute
            original_func: Original function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (success, result)
        """
        recovery_action.attempt_count += 1
        self.recovery_stats['total_recoveries'] += 1
        self.recovery_stats['recovery_by_strategy'][recovery_action.strategy] += 1

        logger.info(f"Executing recovery strategy: {recovery_action.strategy.value} (attempt {recovery_action.attempt_count})")

        # Apply recovery parameters
        if recovery_action.parameters:
            kwargs.update(recovery_action.parameters)

        # Wait before retry if specified
        if recovery_action.delay_seconds > 0:
            await asyncio.sleep(recovery_action.delay_seconds)

        try:
            # Execute the function with recovery parameters
            result = await original_func(*args, **kwargs)

            self.recovery_stats['successful_recoveries'] += 1

            if self.enhanced_logger:
                self.enhanced_logger.log_event(
                    LogLevel.INFO,
                    LogCategory.ERROR,
                    AgentEventType.MESSAGE_PROCESSED,
                    f"Recovery successful using {recovery_action.strategy.value}",
                    strategy=recovery_action.strategy.value,
                    attempt_count=recovery_action.attempt_count,
                    parameters=recovery_action.parameters
                )

            logger.info(f"Recovery successful with strategy: {recovery_action.strategy.value}")
            return True, result

        except Exception as e:
            self.recovery_stats['failed_recoveries'] += 1

            if self.enhanced_logger:
                self.enhanced_logger.log_event(
                    LogLevel.WARNING,
                    LogCategory.ERROR,
                    AgentEventType.ERROR,
                    f"Recovery failed with {recovery_action.strategy.value}",
                    strategy=recovery_action.strategy.value,
                    attempt_count=recovery_action.attempt_count,
                    error=str(e)
                )

            logger.warning(f"Recovery failed with strategy {recovery_action.strategy.value}: {e}")
            return False, e

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify an error into an ErrorType."""
        error_message = str(error).lower()
        error_type_name = type(error).__name__.lower()

        if 'timeout' in error_message or 'timeout' in error_type_name:
            return ErrorType.TIMEOUT_ERROR
        elif 'network' in error_message or 'connection' in error_message or 'http' in error_type_name:
            return ErrorType.NETWORK_ERROR
        elif 'bot' in error_message or 'captcha' in error_message or 'blocked' in error_message:
            return ErrorType.ANTI_BOT_DETECTION
        elif 'rate limit' in error_message or '429' in error_message:
            return ErrorType.RATE_LIMIT_ERROR
        elif 'extraction' in error_message or 'parse' in error_message:
            return ErrorType.CONTENT_EXTRACTION_ERROR
        elif 'cleaning' in error_message or 'process' in error_type_name:
            return ErrorType.CLEANING_ERROR
        elif 'validation' in error_message or 'invalid' in error_message:
            return ErrorType.VALIDATION_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR

    async def validate_and_recover(
        self,
        data_type: str,
        data: Any,
        original_func: Callable,
        *args,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        **kwargs
    ) -> Tuple[bool, Any]:
        """Validate data and attempt recovery if validation fails.

        Args:
            data_type: Type of data being validated
            data: Data to validate
            original_func: Original function to call if recovery needed
            *args: Function arguments
            validation_level: Validation level to apply
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (success, result)
        """
        # First validation
        validation_result = self.validate(data_type, data, validation_level)

        if validation_result.is_valid:
            return True, data

        # Check for critical errors
        if validation_result.has_critical_errors():
            logger.error("Critical validation errors detected, no recovery attempt")
            return False, validation_result.errors

        # Attempt recovery
        context = getattr(data, 'context', None)
        if not context:
            logger.error("No task context available for recovery")
            return False, validation_result.errors

        # Create a synthetic error for recovery strategy determination
        synthetic_error = ValidationError(f"Validation failed: {validation_result.errors}")
        recovery_action = self.determine_recovery_strategy(
            synthetic_error, context, data_type, context.retry_count
        )

        if not recovery_action:
            logger.error("No recovery strategy available")
            return False, validation_result.errors

        # Execute recovery
        recovery_success, result = await self.execute_recovery(
            recovery_action, original_func, *args, **kwargs
        )

        if recovery_success:
            # Validate the recovered result
            if hasattr(result, 'context'):
                result.context.retry_count += 1

            recovered_validation = self.validate(data_type, result, validation_level)
            if recovered_validation.is_valid:
                return True, result
            else:
                logger.warning("Recovered data still fails validation")
                return False, recovered_validation.errors

        return False, validation_result.errors

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        total = self.recovery_stats['total_recoveries']
        success_rate = (
            self.recovery_stats['successful_recoveries'] / total
            if total > 0 else 0.0
        )

        return {
            'total_recoveries': total,
            'successful_recoveries': self.recovery_stats['successful_recoveries'],
            'failed_recoveries': self.recovery_stats['failed_recoveries'],
            'success_rate': success_rate,
            'recovery_by_strategy': dict(self.recovery_stats['recovery_by_strategy'])
        }

    def reset_statistics(self):
        """Reset recovery statistics."""
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'recovery_by_strategy': defaultdict(int)
        }

        logger.info("Recovery statistics reset")


# Factory functions and utilities
def create_validation_system(config: PipelineConfig) -> Tuple[Dict[str, BaseValidator], ErrorRecoveryManager]:
    """Create a complete validation and recovery system.

    Args:
        config: Pipeline configuration

    Returns:
        Tuple of (validators_dict, recovery_manager)
    """
    validators = {
        'scraping_request': ScrapingRequestValidator(config),
        'scraping_result': ScrapingResultValidator(config),
        'cleaning_request': CleaningRequestValidator(config),
        'cleaning_result': CleaningResultValidator(config)
    }

    recovery_manager = ErrorRecoveryManager(config)

    return validators, recovery_manager


# Export main classes and functions
__all__ = [
    # Enums and Classes
    'ValidationLevel', 'RecoveryStrategy', 'ValidationError',
    'ValidationResult', 'RecoveryAction',

    # Validators
    'BaseValidator', 'ScrapingRequestValidator', 'ScrapingResultValidator',
    'CleaningRequestValidator', 'CleaningResultValidator',

    # Recovery
    'ErrorRecoveryManager',

    # Factory Functions
    'create_validation_system',
]