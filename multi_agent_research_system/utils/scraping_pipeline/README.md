# Enhanced Two-Module Scraping Architecture - Phase 1.4

## Overview

This module implements the enhanced two-module scraping architecture with comprehensive data contracts, async orchestration, and error recovery mechanisms. It addresses the critical `'str' object has no attribute 'get'` failure from peer review by implementing strict data validation and type safety throughout the pipeline.

## Architecture

### Core Components

```
Phase 1.4.1: Data Contracts
├── ScrapingRequest/Result (Pydantic models)
├── CleaningRequest/Result (Pydantic models)
├── TaskContext and workflow state
├── PipelineConfig and statistics
└── Validation schemas

Phase 1.4.2: Async Orchestrator
├── AsyncScrapingOrchestrator (40/20 worker pools)
├── AsyncTaskQueue with backpressure
├── WorkerPool management
└── Performance monitoring

Phase 1.4.3: Validation & Recovery
├── DataContractValidator
├── ErrorRecoveryManager
├── Sophisticated error handling
└── Fallback mechanisms
```

### Integration Points

- **Phase 1.1**: Enhanced logging and monitoring
- **Phase 1.2**: Anti-bot escalation system
- **Phase 1.3**: Content cleaning pipeline
- **Phase 1.4**: Data contracts and orchestration

## Key Features

### 1. Pydantic Data Contracts (Phase 1.4.1)

**Strict Type Safety**: Prevents data structure mismatches with comprehensive validation

```python
from .data_contracts import ScrapingRequest, ScrapingResult, CleaningRequest, CleaningResult

# Create validated requests
scrape_request = ScrapingRequest(
    url="https://example.com",
    search_query="research topic",
    anti_bot_level=1,
    min_quality_score=0.7
)

clean_request = CleaningRequest(
    content="Raw content to clean",
    url="https://example.com",
    cleaning_intensity="aggressive"
)
```

**Comprehensive Validation**: Multi-level validation with business logic checks

```python
from .data_contracts import ValidationLevel, DataContractValidator

validator = DataContractValidator(config)
is_valid, error = validator.validate_scraping_request(scrape_request)
if not is_valid:
    print(f"Validation failed: {error}")
```

### 2. Async Orchestrator (Phase 1.4.2)

**High-Concurrency Processing**: 40 scrape workers, 20 clean workers with proper backpressure

```python
from .async_orchestrator import AsyncScrapingOrchestrator, managed_orchestrator

# Use context manager for automatic lifecycle management
async with managed_orchestrator(config) as orchestrator:
    # Submit scraping task
    success = await orchestrator.submit_scraping_task(scrape_request)

    # Submit cleaning task
    success = await orchestrator.submit_cleaning_task(clean_request)

    # Get statistics
    stats = orchestrator.get_statistics()
    health = await orchestrator.get_health_status()
```

**Queue Management**: Priority queues with backpressure control

```python
from .async_orchestrator import AsyncTaskQueue

queue = AsyncTaskQueue(
    max_size=1000,
    backpressure_threshold=0.8
)

# Add tasks with priority
await queue.put(task, priority=1)  # Lower number = higher priority
task, task_id, wait_time = await queue.get()
```

### 3. Validation & Error Recovery (Phase 1.4.3)

**Sophisticated Validation**: Multi-level validation with custom rules

```python
from .validation_recovery import ErrorRecoveryManager, ValidationLevel

recovery_manager = ErrorRecoveryManager(config)

# Validate with different levels
result = recovery_manager.validate(
    'scraping_request',
    request,
    ValidationLevel.STRICT
)

if not result.is_valid:
    print(f"Validation errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
```

**Intelligent Error Recovery**: Automatic recovery strategies based on error types

```python
# Automatic recovery with strategy selection
success, result = await recovery_manager.validate_and_recover(
    'scraping_request',
    request,
    original_scraping_function,
    validation_level=ValidationLevel.STANDARD
)
```

## Quick Start

### Basic Usage

```python
from .integration import ScrapingPipelineAPI

# Create API with default configuration
api = ScrapingPipelineAPI()

try:
    await api.initialize()

    # Scrape and clean a single URL
    success, result = await api.scrape_and_clean(
        url="https://example.com",
        search_query="research topic",
        validation_level=ValidationLevel.STANDARD
    )

    if success:
        print(f"Quality score: {result.final_quality_score}")
        print(f"Content: {result.cleaned_content[:200]}...")
    else:
        print(f"Errors: {result}")

finally:
    await api.shutdown()
```

### Batch Processing

```python
from .integration import batch_process_urls

urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
]

result = await batch_process_urls(
    urls=urls,
    search_query="batch research",
    max_scrape_workers=10,
    max_clean_workers=5
)

print(f"Processed: {result['success_rate']:.1%} success rate")
print(f"Results: {len(result['results'])} URLs processed")
```

### Advanced Configuration

```python
from .data_contracts import PipelineConfig
from .integration import ScrapingPipelineAPI

config = PipelineConfig(
    # Concurrency settings
    max_scrape_workers=40,
    max_clean_workers=20,
    max_queue_size=1000,
    backpressure_threshold=0.8,

    # Quality settings
    default_quality_threshold=0.7,
    min_acceptable_quality=0.5,
    enable_quality_gates=True,

    # Performance settings
    enable_batch_processing=True,
    batch_size=10,
    enable_caching=True,

    # Integration settings
    enable_anti_bot=True,
    enable_content_cleaning=True,
    enable_quality_assessment=True
)

api = ScrapingPipelineAPI(config)
```

## Data Models

### ScrapingRequest

```python
ScrapingRequest(
    url: HttpUrl,                    # Required URL to scrape
    search_query: Optional[str],      # Original search query
    context: TaskContext,             # Task context and metadata

    # Scraping configuration
    anti_bot_level: Optional[int],    # Starting anti-bot level (0-3)
    max_anti_bot_level: int,          # Maximum anti-bot level
    timeout_seconds: int,             # Request timeout

    # Content requirements
    min_content_length: int,          # Minimum content length
    require_clean_content: bool,      # Require content cleaning

    # Quality thresholds
    min_quality_score: float,         # Minimum quality score
    content_cleanliness_threshold: float  # Cleanliness threshold
)
```

### ScrapingResult

```python
ScrapingResult(
    url: str,                         # Source URL
    domain: str,                      # Extracted domain
    success: bool,                    # Success status
    content: Optional[str],           # Scraped content

    # Performance metrics
    duration: float,                  # Total duration
    attempts_made: int,               # Number of attempts
    word_count: int,                  # Word count
    char_count: int,                  # Character count

    # Anti-bot data
    final_anti_bot_level: int,        # Final anti-bot level used
    escalation_used: bool,            # Whether escalation was used
    escalation_triggers: List[str],   # Detection triggers

    # Quality metrics
    content_quality_score: Optional[float],  # Content quality
    cleanliness_score: Optional[float],      # Cleanliness score
    relevance_score: Optional[float],        # Relevance score

    # Error handling
    error_message: Optional[str],     # Error message if failed
    error_type: Optional[ErrorType],  # Error classification
    retry_recommendation: bool,       # Whether retry is recommended
)
```

### CleaningRequest

```python
CleaningRequest(
    content: str,                     # Content to clean
    url: str,                         # Source URL
    search_query: Optional[str],      # Original search query
    context: TaskContext,             # Task context

    # Cleaning configuration
    cleaning_intensity: str,          # light/medium/aggressive
    enable_ai_cleaning: bool,         # Enable AI cleaning
    quality_threshold: float,         # Target quality

    # Content preservation
    preserve_links: bool,             # Preserve hyperlinks
    preserve_formatting: bool,        # Preserve formatting
    max_length_reduction: float,      # Maximum reduction ratio
)
```

### CleaningResult

```python
CleaningResult(
    original_content: str,            # Original content
    cleaned_content: str,             # Cleaned content
    url: str,                         # Source URL
    success: bool,                    # Success status

    # Cleaning metrics
    cleaning_performed: bool,         # Whether cleaning was performed
    quality_improvement: float,       # Quality improvement score
    length_reduction: float,          # Content reduction ratio

    # Quality assessment
    original_quality_score: Optional[float],  # Original quality
    final_quality_score: Optional[float],      # Final quality
    cleanliness_score: Optional[float],        # Cleanliness score

    # Processing info
    cleaning_stage: str,              # Cleaning stage used
    processing_time_ms: float,        # Processing time

    # Enhancement info
    editorial_recommendation: str,    # Editorial assessment
    enhancement_suggestions: List[str] # Enhancement suggestions
)
```

## Error Handling

### Error Types

```python
class ErrorType(str, Enum):
    NETWORK_ERROR = "network_error"
    ANTI_BOT_DETECTION = "anti_bot_detection"
    CONTENT_EXTRACTION_ERROR = "content_extraction_error"
    CLEANING_ERROR = "cleaning_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    UNKNOWN_ERROR = "unknown_error"
```

### Recovery Strategies

```python
class RecoveryStrategy(str, Enum):
    RETRY = "retry"                    # Simple retry with same parameters
    ESCALATE = "escalate"              # Escalate anti-bot level or intensity
    FALLBACK = "fallback"              # Use fallback implementation
    SKIP = "skip"                      # Skip and continue
    ABORT = "abort"                    # Abort the entire operation
```

### Error Recovery Example

```python
from .validation_recovery import ErrorRecoveryManager

recovery_manager = ErrorRecoveryManager(config)

# Automatic error classification and recovery
network_error = Exception("Connection failed")
strategy = recovery_manager.determine_recovery_strategy(
    network_error,
    task_context,
    'scraping_request'
)

# Execute recovery
success, result = await recovery_manager.execute_recovery(
    strategy,
    original_scraping_function,
    request
)
```

## Performance Monitoring

### Statistics

```python
# Get comprehensive statistics
stats = await api.get_pipeline_status()

# Pipeline statistics
pipeline_stats = stats['pipeline_stats']
print(f"Success rate: {pipeline_stats['success_rate']:.1%}")
print(f"Average quality: {pipeline_stats['avg_quality_score']:.2f}")
print(f"Total tasks: {pipeline_stats['total_tasks']}")

# Queue statistics
queue_stats = stats['scrape_queue']
print(f"Queue size: {queue_stats['current_size']}")
print(f"Backpressure active: {queue_stats['backpressure_active']}")

# Worker pool statistics
worker_stats = stats['scrape_pool']
print(f"Active workers: {worker_stats['active_workers']}")
print(f"Success rate: {worker_stats['success_rate']:.1%}")
```

### Health Monitoring

```python
health = await api.get_health_status()

print(f"Overall health: {health['overall_health']}")
if health['health_issues']:
    print("Health issues:")
    for issue in health['health_issues']:
        print(f"  - {issue}")

print(f"Circuit breaker: {health['circuit_breaker_state']}")
print(f"Queue sizes: scrape={health['queue_sizes']['scrape']}, clean={health['queue_sizes']['clean']}")
```

## Testing

### Running Tests

```python
# Run comprehensive test suite
python -m pytest test_phase_1_4.py -v

# Run specific test classes
python -m pytest test_phase_1_4.py::TestDataContracts -v
python -m pytest test_phase_1_4.py::TestAsyncOrchestrator -v

# Run performance tests
python -m pytest test_phase_1_4.py::TestPerformance -v -s
```

### Test Coverage

- **Data Contracts**: Pydantic model validation and factory functions
- **Async Orchestrator**: Worker pools, queues, and lifecycle management
- **Validation & Recovery**: Error handling, recovery strategies, and validation levels
- **Integration**: End-to-end API functionality
- **Performance**: Concurrent processing, memory usage, and backpressure

## Configuration

### Environment Variables

```bash
# Performance settings
SCRAPE_MAX_WORKERS=40
CLEAN_MAX_WORKERS=20
MAX_QUEUE_SIZE=1000
BACKPRESSURE_THRESHOLD=0.8

# Quality settings
DEFAULT_QUALITY_THRESHOLD=0.7
MIN_ACCEPTABLE_QUALITY=0.5
ENABLE_QUALITY_GATES=true

# Feature flags
ENABLE_ANTI_BOT=true
ENABLE_CONTENT_CLEANING=true
ENABLE_QUALITY_ASSESSMENT=true
ENABLE_PERFORMANCE_MONITORING=true

# Logging
LOG_LEVEL=INFO
ENABLE_METRICS_EXPORT=false
```

### Configuration File

```python
# scraping_config.py
from .data_contracts import PipelineConfig

PRODUCTION_CONFIG = PipelineConfig(
    max_scrape_workers=40,
    max_clean_workers=20,
    max_queue_size=1000,
    backpressure_threshold=0.8,

    default_quality_threshold=0.7,
    min_acceptable_quality=0.5,
    enable_quality_gates=True,

    enable_batch_processing=True,
    batch_size=10,
    enable_caching=True,
    cache_ttl_seconds=3600,

    enable_anti_bot=True,
    enable_content_cleaning=True,
    enable_quality_assessment=True,

    enable_performance_monitoring=True,
    log_level="INFO"
)

DEVELOPMENT_CONFIG = PipelineConfig(
    max_scrape_workers=5,
    max_clean_workers=3,
    max_queue_size=50,
    backpressure_threshold=0.7,

    default_quality_threshold=0.6,
    min_acceptable_quality=0.4,
    enable_quality_gates=False,  # More lenient in development

    enable_performance_monitoring=False,
    log_level="DEBUG"
)
```

## Best Practices

### 1. Error Handling

- Always check validation results before processing
- Implement proper error recovery strategies
- Monitor error rates and patterns
- Use circuit breakers for external dependencies

### 2. Performance Optimization

- Use appropriate worker pool sizes
- Monitor queue sizes and backpressure
- Implement proper caching strategies
- Balance quality vs. speed requirements

### 3. Resource Management

- Always shutdown orchestrators properly
- Use context managers for lifecycle management
- Monitor memory usage with large content
- Implement proper cleanup on errors

### 4. Quality Assurance

- Set appropriate quality thresholds
- Use multi-level validation
- Monitor quality scores and trends
- Implement enhancement feedback loops

## Troubleshooting

### Common Issues

**High Memory Usage**
```python
# Reduce worker pool sizes
config = PipelineConfig(max_scrape_workers=20, max_clean_workers=10)

# Enable content length limits
request = ScrapingRequest(
    url="https://example.com",
    max_content_length=50000  # 50KB limit
)
```

**Queue Backpressure**
```python
# Increase queue size or reduce backpressure threshold
config = PipelineConfig(
    max_queue_size=2000,
    backpressure_threshold=0.9
)

# Monitor queue statistics
stats = await api.get_pipeline_status()
if stats['scrape_queue']['current_size'] > stats['scrape_queue']['max_size'] * 0.8:
    print("Approaching queue capacity")
```

**Low Success Rates**
```python
# Lower quality thresholds temporarily
config = PipelineConfig(
    default_quality_threshold=0.5,
    min_acceptable_quality=0.3
)

# Enable more aggressive error recovery
config.enable_circuit_breaker = False
config.default_max_retries = 5
```

### Debug Mode

```python
# Enable comprehensive logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use strict validation for debugging
success, result = await api.scrape_and_clean(
    url="https://example.com",
    validation_level=ValidationLevel.STRICT
)

# Get detailed status
status = await api.get_pipeline_status()
print("Full status:", status)
```

## Migration Guide

### From Previous Implementation

1. **Replace string-based data structures with Pydantic models**
```python
# Old approach
data = {
    'url': 'https://example.com',
    'content': 'scraped content',
    'success': True
}

# New approach
result = ScrapingResult(
    url="https://example.com",
    domain="example.com",
    success=True,
    content="scraped content",
    duration=1.5,
    attempts_made=1,
    context=TaskContext()
)
```

2. **Use async orchestrator instead of direct function calls**
```python
# Old approach
content = await scrape_url(url)
cleaned = await clean_content(content)

# New approach
api = ScrapingPipelineAPI()
await api.initialize()
success, result = await api.scrape_and_clean(url)
await api.shutdown()
```

3. **Implement proper error handling**
```python
# Old approach
try:
    result = await scrape_url(url)
except Exception as e:
    print(f"Error: {e}")

# New approach
success, result = await api.scrape_and_clean(url)
if not success:
    print(f"Errors: {result}")  # Detailed error list
    # Automatic recovery attempted
```

## Future Enhancements

### Planned Features

1. **Advanced Caching**: Intelligent caching with invalidation strategies
2. **Distributed Processing**: Multi-node orchestration capabilities
3. **Advanced Analytics**: ML-based quality prediction and optimization
4. **Plugin System**: Extensible validation and recovery plugins
5. **Real-time Monitoring**: Live dashboard with metrics and alerts

### Extension Points

- Custom validator implementations
- Additional recovery strategies
- Custom performance metrics
- Enhanced quality assessment algorithms
- Specialized content extraction rules

---

**Phase 1.4 Implementation Status**: ✅ Complete

All three phases have been successfully implemented:
- ✅ Phase 1.4.1: Pydantic data contracts for scraper→cleaner pipeline
- ✅ Phase 1.4.2: AsyncScrapingOrchestrator with worker pools (40/20 concurrency)
- ✅ Phase 1.4.3: Data contract validation and error recovery mechanisms

The enhanced two-module scraping architecture is now ready for production use with comprehensive data validation, async processing, and error recovery capabilities.