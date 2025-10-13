# Two-Module Scraping System - Configuration Reference

## Overview

This comprehensive configuration reference covers all available settings, parameters, and options for the Two-Module Scraping System. Configuration is managed through environment variables, configuration files, and programmatic settings to provide maximum flexibility for different deployment scenarios and use cases.

## Configuration Architecture

### Configuration Hierarchy

```
1. Environment Variables (Highest Priority)
   ├── System Environment Variables
   ├── .env Files
   └── Runtime Overrides

2. Configuration Files
   ├── settings.py (Default Configuration)
   ├── custom_config.py (User Overrides)
   └── domain_configs.json (Domain-Specific Settings)

3. Programmatic Configuration (Lowest Priority)
   ├── In-Code Settings
   ├── Runtime Parameters
   └── Dynamic Adjustments
```

### Configuration Loading Order

```python
# Configuration loading follows this priority order:
1. Environment variables (ANTIBOT_LEVEL, TARGET_SUCCESSFUL_SCRAPES, etc.)
2. .env file in project root
3. settings.py default values
4. Runtime parameters passed to functions
5. Dynamic adjustments during execution
```

## Core Configuration Settings

### Anti-Bot Configuration

#### Anti-Bot Levels

```python
ANTI_BOT_LEVELS = {
    0: {
        "name": "Basic",
        "description": "Standard HTTP requests with basic headers",
        "success_rate": 0.6,  # 6/10 sites
        "speed": "Fast",
        "detection_risk": "Low",
        "use_cases": ["Simple blogs", "Documentation sites", "News sites without protection"]
    },
    1: {
        "name": "Enhanced",
        "description": "Enhanced headers with JavaScript rendering",
        "success_rate": 0.8,  # 8/10 sites
        "speed": "Medium",
        "detection_risk": "Medium",
        "use_cases": ["Most websites", "Corporate sites", "News sites with basic protection"]
    },
    2: {
        "name": "Advanced",
        "description": "Browser automation with advanced anti-detection",
        "success_rate": 0.9,  # 9/10 sites
        "speed": "Slow",
        "detection_risk": "Low",
        "use_cases": ["Social media", "E-commerce", "Sites with moderate protection"]
    },
    3: {
        "name": "Stealth",
        "description": "Full browser simulation with comprehensive anti-detection",
        "success_rate": 0.95,  # 9.5/10 sites
        "speed": "Very Slow",
        "detection_risk": "Very Low",
        "use_cases": ["Highly protected sites", "LinkedIn", "Major social platforms"]
    }
}
```

#### Environment Variables

```bash
# Anti-Bot Configuration
ANTI_BOT_LEVEL=1                          # Default anti-bot level (0-3)
ANTI_BOT_AUTO_LEARNING=true               # Enable auto-learning of difficult sites
ANTI_BOT_MIN_ESCALATIONS=3                # Minimum escalations before auto-learning
ANTI_BOT_MAX_ATTEMPTS=4                   # Maximum attempts per URL
ANTI_BOT_BASE_DELAY=1.0                   # Base delay between attempts (seconds)
ANTI_BOT_MAX_DELAY=30.0                   # Maximum delay between attempts (seconds)
ANTI_BOT_DEBUG=false                      # Enable detailed anti-bot logging

# Difficult Sites Configuration
DIFFICULT_SITES_CONFIG_PATH=/path/to/difficult_sites.json
DIFFICULT_SITES_AUTO_ADD=true             # Automatically add learned difficult sites
DIFFICULT_SITES_CONFIDENCE_THRESHOLD=0.8  # Confidence threshold for auto-addition
```

#### Configuration Class

```python
@dataclass
class AntiBotConfig:
    """Configuration for anti-bot escalation system."""

    # Default settings
    default_level: int = 1
    max_level: int = 3
    auto_learning: bool = True
    min_escalations_for_learning: int = 3

    # Retry settings
    max_attempts_per_url: int = 4
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    delay_multiplier: float = 2.0

    # Escalation thresholds
    escalation_thresholds: Dict[int, float] = field(default_factory=lambda: {
        0: 0.7,  # Escalate at 70% failure rate
        1: 0.5,  # Escalate at 50% failure rate
        2: 0.3,  # Escalate at 30% failure rate
    })

    # Performance settings
    timeout_per_level: Dict[int, int] = field(default_factory=lambda: {
        0: 15000,   # 15 seconds
        1: 30000,   # 30 seconds
        2: 45000,   # 45 seconds
        3: 60000,   # 60 seconds
    })
```

### Content Cleaning Configuration

#### Cleanliness Assessment

```bash
# Content Cleaning Configuration
CLEANLINESS_THRESHOLD=0.7                  # Skip cleaning if content is 70% clean
CONTENT_CLEANING_MODEL=gpt-5-nano         # Model for content cleaning
CLEANLINESS_JUDGE_MODEL=gpt-5-nano        # Model for cleanliness assessment
MIN_CONTENT_LENGTH_FOR_CLEANING=500       # Minimum content length for cleaning
MAX_CONTENT_LENGTH_FOR_CLEANING=150000    # Maximum content length for cleaning
MIN_CLEANED_CONTENT_LENGTH=200            # Minimum length after cleaning
CONTENT_CLEANING_DEBUG=false              # Enable detailed cleaning logs
```

#### AI Cleaning Settings

```python
@dataclass
class ContentCleaningConfig:
    """Configuration for AI-powered content cleaning."""

    # Model settings
    cleaning_model: str = "gpt-5-nano"
    judge_model: str = "gpt-5-nano"

    # Thresholds
    cleanliness_threshold: float = 0.7
    min_content_length: int = 500
    max_content_length: int = 150000
    min_cleaned_length: int = 200

    # Quality criteria
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        "content_completeness": 0.30,
        "relevance_score": 0.25,
        "readability_score": 0.20,
        "technical_accuracy": 0.15,
        "source_credibility": 0.10
    })

    # Content type detection
    enable_content_type_detection: bool = True
    preserve_technical_content: bool = True
    preserve_code_examples: bool = True

    # Performance optimization
    enable_judge_optimization: bool = True
    batch_cleaning_enabled: bool = True
    max_concurrent_cleaning: int = 6
```

#### Technical Content Preservation

```python
TECHNICAL_CONTENT_CONFIG = {
    "preserve_code_blocks": True,
    "preserve_installation_commands": True,
    "preserve_api_documentation": True,
    "preserve_configuration_examples": True,
    "preserve_error_messages": True,
    "preserve_version_information": True,

    "validation_rules": {
        "code_blocks_preserved": lambda cleaned, original: '```' in original and '```' in cleaned,
        "commands_preserved": lambda cleaned, original: any(cmd in cleaned for cmd in ['pip install', 'npm install', 'go get']),
        "technical_accuracy": lambda cleaned: validate_technical_accuracy(cleaned)
    }
}
```

### Search & Target Configuration

#### Search Settings

```bash
# Search Configuration
DEFAULT_NUM_RESULTS=15                    # Default search results count
DEFAULT_AUTO_CRAWL_TOP=10                 # Number of top results to crawl
DEFAULT_CRAWL_THRESHOLD=0.3               # Minimum relevance score for crawling
DEFAULT_MAX_CONCURRENT=0                  # Max concurrent operations (0 = unbounded)

# Multi-query Configuration
ENABLE_MULTI_QUERY_STRATEGY=true          # Enable multi-query URL selection
MAX_QUERIES_PER_SEARCH=3                  # Maximum query variations
QUERY_ENHANCEMENT_MODEL=gpt-5-nano        # Model for query enhancement
```

#### Target-Based Scraping

```bash
# Target-Based Scraping Configuration
TARGET_SUCCESSFUL_SCRAPES=15              # Target number of successful extractions
MAX_TOTAL_URLS_TO_PROCESS=50              # Maximum total URLs to process
ENABLE_SUCCESS_BASED_TERMINATION=true     # Stop when target achieved
PRIMARY_BATCH_SIZE=16                     # Primary batch size for processing
SECONDARY_BATCH_SIZE=16                   # Secondary batch size for processing
URL_DEDUPLICATION_ENABLED=true            # Enable URL deduplication
PROGRESSIVE_RETRY_ENABLED=true            # Enable progressive retry logic
```

#### Configuration Class

```python
@dataclass
class TargetBasedScrapingConfig:
    """Configuration for target-based scraping system."""

    # Success targets
    target_successful_scrapes: int = 15
    max_total_urls_to_process: int = 50
    enable_success_based_termination: bool = True

    # Batching configuration
    primary_batch_size: int = 16
    secondary_batch_size: int = 16
    fallback_batch_size: int = 8

    # URL selection
    crawl_threshold: float = 0.3
    enable_deduplication: bool = True
    deduplication_window_hours: int = 24

    # Retry logic
    max_retry_attempts: int = 3
    progressive_timeout_multiplier: float = 1.5
    enable_progressive_retry: bool = True

    # Quality filtering
    min_content_length: int = 500
    max_content_length: int = 150000
    quality_threshold: float = 0.7
```

### Performance Configuration

#### Concurrency Settings

```bash
# Concurrency Configuration
MAX_CONCURRENT_SCRAPES=15                 # Maximum concurrent scraping operations
MAX_CONCURRENT_CLEANS=6                   # Maximum concurrent cleaning operations
MAX_CONCURRENT_SEARCHES=5                 # Maximum concurrent search operations
CONCURRENCY_ADAPTIVE=true                 # Enable adaptive concurrency control

# Resource Management
CPU_THRESHOLD_HIGH=0.8                    # High CPU usage threshold
CPU_THRESHOLD_LOW=0.4                     # Low CPU usage threshold
MEMORY_THRESHOLD_HIGH=0.85                # High memory usage threshold
MEMORY_THRESHOLD_LOW=0.6                  # Low memory usage threshold
ENABLE_RESOURCE_MONITORING=true           # Enable resource monitoring
```

#### Performance Optimization

```python
@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    # Concurrency limits
    max_concurrent_scrapes: int = 15
    max_concurrent_cleans: int = 6
    max_concurrent_searches: int = 5

    # Resource thresholds
    cpu_threshold_high: float = 0.8
    cpu_threshold_low: float = 0.4
    memory_threshold_high: float = 0.85
    memory_threshold_low: float = 0.6

    # Adaptive settings
    enable_adaptive_concurrency: bool = True
    enable_resource_monitoring: bool = True
    enable_performance_optimization: bool = True

    # Optimization features
    enable_media_optimization: bool = True
    enable_judge_optimization: bool = True
    enable_streaming_pipeline: bool = True

    # Timeout settings
    default_timeout: int = 30000  # 30 seconds
    max_timeout: int = 120000    # 2 minutes
    cleanup_timeout: int = 10000  # 10 seconds
```

#### Media Optimization

```python
MEDIA_OPTIMIZATION_CONFIG = {
    "enabled": True,
    "text_mode": True,
    "exclude_all_images": True,
    "exclude_external_images": True,
    "light_mode": True,
    "disable_css": False,
    "disable_javascript": False,

    "performance_impact": {
        "speed_improvement": "3-4x faster",
        "memory_reduction": "50-70% less memory",
        "bandwidth_savings": "80-90% less bandwidth",
        "success_rate_impact": "Minimal (for text content)"
    }
}
```

### Quality Management Configuration

#### Quality Thresholds

```bash
# Quality Configuration
MIN_QUALITY_THRESHOLD=70                  # Minimum quality score (0-100)
CONTENT_QUALITY_WEIGHTS=                  # Quality dimension weights
ENABLE_QUALITY_ENHANCEMENT=true           # Enable quality enhancement
MAX_ENHANCEMENT_CYCLES=3                  # Maximum enhancement cycles
ENHANCEMENT_IMPROVEMENT_THRESHOLD=0.1     # Minimum improvement per cycle
```

#### Quality Assessment Configuration

```python
@dataclass
class QualityConfig:
    """Configuration for quality assessment and enhancement."""

    # Thresholds
    min_quality_threshold: int = 70  # 0-100 scale
    enhancement_threshold: float = 0.1  # 10% improvement minimum
    max_enhancement_cycles: int = 3

    # Quality weights (should sum to 1.0)
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        "content_completeness": 0.30,
        "relevance_score": 0.25,
        "readability_score": 0.20,
        "technical_accuracy": 0.15,
        "source_credibility": 0.10
    })

    # Enhancement settings
    enable_enhancement: bool = True
    enhancement_strategies: List[str] = field(default_factory=lambda: [
        "content_expansion",
        "fact_verification",
        "structure_improvement",
        "readability_enhancement"
    ])

    # Validation criteria
    min_word_count: int = 100
    min_sentence_count: int = 3
    min_paragraph_count: int = 2
    max_repetition_ratio: float = 0.3
```

### Session Management Configuration

#### Session Settings

```bash
# Session Management Configuration
SESSION_TIMEOUT_MINUTES=120               # Session timeout in minutes
SESSION_AUTO_CLEANUP_DAYS=30              # Auto-cleanup sessions after days
SESSION_COMPRESSION_ENABLED=true          # Enable session compression
SESSION_BASED_ORGANIZATION=true           # Use session-based file organization

# Directory Structure
KEVIN_WORKPRODUCTS_DIR=./KEVIN            # Base directory for outputs
SESSIONS_DIR=KEVIN/sessions               # Sessions directory
WORKING_DIR=working                       # Working subdirectory
RESEARCH_DIR=research                     # Research subdirectory
COMPLETE_DIR=complete                     # Complete subdirectory
```

#### Session Configuration Class

```python
@dataclass
class SessionConfig:
    """Configuration for session management."""

    # Timing settings
    timeout_minutes: int = 120
    auto_cleanup_days: int = 30
    compression_enabled: bool = True

    # Directory structure
    base_workproducts_dir: str = "KEVIN"
    sessions_dir: str = "KEVIN/sessions"
    working_subdir: str = "working"
    research_subdir: str = "research"
    complete_subdir: str = "complete"

    # File organization
    session_based_organization: bool = True
    stage_based_prefixes: bool = True
    timestamped_files: bool = True
    metadata_files: bool = True

    # Workproduct naming
    research_prefix: str = "RESEARCH"
    report_prefix: str = "REPORT"
    editorial_prefix: str = "EDITORIAL"
    final_prefix: str = "FINAL"
    gap_research_prefix: str = "EDITOR_RESEARCH"
```

### Monitoring & Logging Configuration

#### Logging Settings

```bash
# Logging Configuration
LOG_LEVEL=INFO                            # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
ENABLE_STRUCTURED_LOGGING=true            # Enable structured logging
LOG_FILE_PATH=KEVIN/logs/scraping.log    # Log file path
LOG_ROTATION_SIZE=10MB                    # Log file rotation size
LOG_RETENTION_DAYS=30                     # Log retention period

# Performance Monitoring
ENABLE_PERFORMANCE_MONITORING=true        # Enable performance monitoring
METRICS_COLLECTION_INTERVAL=60            # Metrics collection interval (seconds)
ENABLE_PERFORMANCE_DASHBOARD=true         # Enable performance dashboard
PERFORMANCE_REPORT_INTERVAL=300           # Performance report interval (seconds)
```

#### Monitoring Configuration

```python
@dataclass
class MonitoringConfig:
    """Configuration for monitoring and observability."""

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_structured_logging: bool = True
    log_file_path: str = "KEVIN/logs/scraping.log"
    log_rotation_size: str = "10MB"
    log_retention_days: int = 30

    # Performance monitoring
    enable_performance_monitoring: bool = True
    metrics_collection_interval: int = 60  # seconds
    enable_performance_dashboard: bool = True
    performance_report_interval: int = 300  # seconds

    # Alerting
    enable_alerting: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "success_rate_low": 0.5,
        "quality_score_low": 50,
        "processing_time_high": 120,  # seconds
        "error_rate_high": 0.3
    })

    # Analytics
    enable_analytics: bool = True
    analytics_retention_days: int = 90
    enable_performance_reports: bool = True
```

## Domain-Specific Configuration

### Difficult Sites Configuration

#### JSON Configuration Format

```json
{
  "metadata": {
    "version": "1.0",
    "last_updated": "2024-01-15T10:30:00Z",
    "description": "Difficult sites database for anti-bot optimization"
  },
  "difficult_sites": {
    "linkedin.com": {
      "level": 3,
      "reason": "Heavy bot protection and JavaScript requirements",
      "last_updated": "2024-01-15T10:30:00Z",
      "success_rate": 0.95,
      "recommended_approach": "stealth_mode_with_full_browser"
    },
    "medium.com": {
      "level": 2,
      "reason": "JavaScript heavy content loading",
      "last_updated": "2024-01-14T15:45:00Z",
      "success_rate": 0.88,
      "recommended_approach": "advanced_browser_automation"
    },
    "instagram.com": {
      "level": 3,
      "reason": "Social media with strict anti-bot measures",
      "last_updated": "2024-01-13T09:20:00Z",
      "success_rate": 0.92,
      "recommended_approach": "stealth_mode_with_custom_headers"
    },
    "twitter.com": {
      "level": 3,
      "reason": "Rate limiting and sophisticated bot detection",
      "last_updated": "2024-01-12T14:15:00Z",
      "success_rate": 0.90,
      "recommended_approach": "stealth_with_rate_limiting"
    }
  },
  "auto_learning": {
    "enabled": true,
    "min_escalations": 3,
    "confidence_threshold": 0.8,
    "auto_add_domains": true
  }
}
```

#### Domain Configuration Class

```python
@dataclass
class DomainConfig:
    """Configuration for specific domains."""

    domain: str
    anti_bot_level: int
    reason: str
    last_updated: str
    success_rate: float
    recommended_approach: str

    # Custom settings
    custom_headers: Optional[Dict[str, str]] = None
    custom_delays: Optional[Dict[str, float]] = None
    custom_selectors: Optional[List[str]] = None
    javascript_required: Optional[bool] = None
    wait_strategy: Optional[str] = None

class DomainConfigManager:
    """Manages domain-specific configurations."""

    def __init__(self, config_file: str = None):
        self.config_file = config_file or "difficult_sites.json"
        self.domain_configs: Dict[str, DomainConfig] = {}
        self.load_configurations()

    def load_configurations(self):
        """Load domain configurations from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)

            for domain, config_data in data.get("difficult_sites", {}).items():
                self.domain_configs[domain] = DomainConfig(
                    domain=domain,
                    anti_bot_level=config_data["level"],
                    reason=config_data["reason"],
                    last_updated=config_data["last_updated"],
                    success_rate=config_data.get("success_rate", 0.0),
                    recommended_approach=config_data.get("recommended_approach", "standard"),
                    custom_headers=config_data.get("custom_headers"),
                    custom_delays=config_data.get("custom_delays"),
                    custom_selectors=config_data.get("custom_selectors"),
                    javascript_required=config_data.get("javascript_required"),
                    wait_strategy=config_data.get("wait_strategy")
                )
        except Exception as e:
            logger.error(f"Failed to load domain configurations: {e}")
```

### Content Type Configuration

#### Content Type Specific Settings

```python
CONTENT_TYPE_CONFIGS = {
    "technical_documentation": {
        "cleaning_strategy": "preserve_technical_integrity",
        "anti_bot_level": 1,  # Usually cooperative
        "preserve_elements": [
            "code_blocks", "installation_commands", "api_documentation",
            "configuration_examples", "error_messages", "version_information"
        ],
        "quality_weights": {
            "technical_accuracy": 0.40,
            "content_completeness": 0.30,
            "readability_score": 0.20,
            "relevance_score": 0.10
        }
    },

    "news_article": {
        "cleaning_strategy": "extract_main_article",
        "anti_bot_level": 1,  # Usually accessible
        "preserve_elements": [
            "headline", "publication_date", "author", "main_content",
            "quotes", "key_facts", "source_attribution"
        ],
        "quality_weights": {
            "content_completeness": 0.35,
            "relevance_score": 0.30,
            "readability_score": 0.25,
            "source_credibility": 0.10
        }
    },

    "academic_paper": {
        "cleaning_strategy": "academic_structure_preservation",
        "anti_bot_level": 2,  # Often behind paywalls or complex layouts
        "preserve_elements": [
            "abstract", "introduction", "methodology", "results",
            "conclusion", "references", "citations", "data_tables"
        ],
        "quality_weights": {
            "content_completeness": 0.30,
            "technical_accuracy": 0.25,
            "source_credibility": 0.20,
            "readability_score": 0.15,
            "relevance_score": 0.10
        }
    },

    "product_page": {
        "cleaning_strategy": "extract_product_information",
        "anti_bot_level": 2,  # Often has protection
        "preserve_elements": [
            "product_name", "specifications", "description", "pricing",
            "customer_reviews", "technical_details", "availability"
        ],
        "quality_weights": {
            "content_completeness": 0.30,
            "technical_accuracy": 0.25,
            "readability_score": 0.20,
            "relevance_score": 0.15,
            "source_credibility": 0.10
        }
    }
}
```

## Environment-Specific Configuration

### Development Environment

#### .env.development

```bash
# Development Configuration
DEBUG_MODE=true
LOG_LEVEL=DEBUG
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_DETAILED_LOGGING=true

# Relaxed settings for development
ANTI_BOT_LEVEL=1
TARGET_SUCCESSFUL_SCRAPES=5
MAX_CONCURRENT_SCRAPES=3
CLEANLINESS_THRESHOLD=0.6

# Development paths
KEVIN_WORKPRODUCTS_DIR=./dev_KEVIN
LOG_FILE_PATH=./dev_KEVIN/logs/scraping.log

# API limits (conservative for development)
API_RATE_LIMIT_DELAY=2.0
ENABLE_RATE_LIMITING=true
```

### Production Environment

#### .env.production

```bash
# Production Configuration
DEBUG_MODE=false
LOG_LEVEL=INFO
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_DETAILED_LOGGING=false

# Optimized settings for production
ANTI_BOT_LEVEL=2
TARGET_SUCCESSFUL_SCRAPES=15
MAX_CONCURRENT_SCRAPES=12
CLEANLINESS_THRESHOLD=0.7

# Production paths
KEVIN_WORKPRODUCTS_DIR=/app/KEVIN
LOG_FILE_PATH=/app/KEVIN/logs/scraping.log

# Performance optimization
ENABLE_MEDIA_OPTIMIZATION=true
ENABLE_JUDGE_OPTIMIZATION=true
ENABLE_STREAMING_PIPELINE=true

# Resource limits
CPU_THRESHOLD_HIGH=0.8
MEMORY_THRESHOLD_HIGH=0.85
ENABLE_RESOURCE_MONITORING=true
```

### Testing Environment

#### .env.testing

```bash
# Testing Configuration
DEBUG_MODE=true
LOG_LEVEL=DEBUG
ENABLE_PERFORMANCE_MONITORING=false

# Minimal settings for testing
ANTI_BOT_LEVEL=0
TARGET_SUCCESSFUL_SCRAPES=2
MAX_CONCURRENT_SCRAPES=1
CLEANLINESS_THRESHOLD=0.5

# Testing paths
KEVIN_WORKPRODUCTS_DIR=./test_KEVIN
LOG_FILE_PATH=./test_KEVIN/logs/scraping.log

# Mock/fake settings for testing
USE_MOCK_SERVICES=true
ENABLE_FAKE_SCRAPING=true
SKIP_RATE_LIMITING=true
```

## Configuration Validation

### Validation Rules

```python
class ConfigurationValidator:
    """Validates configuration settings."""

    def validate_anti_bot_config(self, config: AntiBotConfig) -> ValidationResult:
        """Validate anti-bot configuration."""
        errors = []
        warnings = []

        # Validate anti-bot level
        if not 0 <= config.default_level <= 3:
            errors.append("Anti-bot level must be between 0 and 3")

        # Validate thresholds
        for level, threshold in config.escalation_thresholds.items():
            if not 0 <= threshold <= 1:
                errors.append(f"Escalation threshold for level {level} must be between 0 and 1")

        # Validate delays
        if config.base_delay < 0:
            errors.append("Base delay must be positive")

        if config.max_delay < config.base_delay:
            warnings.append("Max delay should be greater than base delay")

        return ValidationResult(errors=errors, warnings=warnings)

    def validate_performance_config(self, config: PerformanceConfig) -> ValidationResult:
        """Validate performance configuration."""
        errors = []
        warnings = []

        # Validate concurrency limits
        if config.max_concurrent_scrapes < 1:
            errors.append("Max concurrent scrapes must be at least 1")

        if config.max_concurrent_cleans < 1:
            errors.append("Max concurrent cleans must be at least 1")

        # Validate thresholds
        if not 0 <= config.cpu_threshold_high <= 1:
            errors.append("CPU threshold must be between 0 and 1")

        if not 0 <= config.memory_threshold_high <= 1:
            errors.append("Memory threshold must be between 0 and 1")

        # Performance warnings
        if config.max_concurrent_scrapes > 20:
            warnings.append("High concurrency may cause rate limiting or resource issues")

        return ValidationResult(errors=errors, warnings=warnings)

    def validate_quality_config(self, config: QualityConfig) -> ValidationResult:
        """Validate quality configuration."""
        errors = []
        warnings = []

        # Validate thresholds
        if not 0 <= config.min_quality_threshold <= 100:
            errors.append("Quality threshold must be between 0 and 100")

        # Validate weights sum
        total_weight = sum(config.quality_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Quality weights must sum to 1.0, current sum: {total_weight}")

        # Validate enhancement settings
        if config.max_enhancement_cycles < 1:
            errors.append("Max enhancement cycles must be at least 1")

        if config.enhancement_threshold <= 0:
            errors.append("Enhancement threshold must be positive")

        return ValidationResult(errors=errors, warnings=warnings)

@dataclass
class ValidationResult:
    """Result of configuration validation."""
    errors: List[str]
    warnings: List[str]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def format_errors(self) -> str:
        return "\n".join(f"❌ {error}" for error in self.errors)

    def format_warnings(self) -> str:
        return "\n".join(f"⚠️ {warning}" for warning in self.warnings)
```

### Configuration Loading

```python
class ConfigurationManager:
    """Manages loading and validation of configuration."""

    def __init__(self):
        self.config_cache: Dict[str, Any] = {}
        self.validator = ConfigurationValidator()

    def load_configuration(self, environment: str = "production") -> ConfigurationSet:
        """Load configuration for specific environment."""

        # Load base configuration
        base_config = self._load_base_configuration()

        # Load environment-specific overrides
        env_config = self._load_environment_configuration(environment)

        # Load environment variables
        env_vars = self._load_environment_variables()

        # Merge configurations (priority: env_vars > env_config > base_config)
        merged_config = self._merge_configurations(base_config, env_config, env_vars)

        # Validate configuration
        validation_results = self._validate_all_configs(merged_config)

        if not validation_results.is_valid:
            raise ConfigurationError(f"Invalid configuration: {validation_results.format_errors()}")

        if validation_results.warnings:
            logger.warning(f"Configuration warnings: {validation_results.format_warnings()}")

        return ConfigurationSet(
            anti_bot=merged_config["anti_bot"],
            performance=merged_config["performance"],
            quality=merged_config["quality"],
            content_cleaning=merged_config["content_cleaning"],
            session_management=merged_config["session_management"],
            monitoring=merged_config["monitoring"]
        )

    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # Anti-bot settings
        if os.getenv('ANTI_BOT_LEVEL'):
            env_config['anti_bot_level'] = int(os.getenv('ANTI_BOT_LEVEL'))

        if os.getenv('TARGET_SUCCESSFUL_SCRAPES'):
            env_config['target_successful_scrapes'] = int(os.getenv('TARGET_SUCCESSFUL_SCRAPES'))

        # Performance settings
        if os.getenv('MAX_CONCURRENT_SCRAPES'):
            env_config['max_concurrent_scrapes'] = int(os.getenv('MAX_CONCURRENT_SCRAPES'))

        # Quality settings
        if os.getenv('CLEANLINESS_THRESHOLD'):
            env_config['cleanliness_threshold'] = float(os.getenv('CLEANLINESS_THRESHOLD'))

        return env_config

@dataclass
class ConfigurationSet:
    """Complete configuration set."""
    anti_bot: AntiBotConfig
    performance: PerformanceConfig
    quality: QualityConfig
    content_cleaning: ContentCleaningConfig
    session_management: SessionConfig
    monitoring: MonitoringConfig

class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass
```

## Configuration Best Practices

### Security Best Practices

```python
# Secure configuration management
class SecureConfigurationManager:
    """Manages configuration with security best practices."""

    @staticmethod
    def mask_sensitive_values(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive values in configuration for logging."""
        sensitive_keys = ['api_key', 'password', 'token', 'secret']
        masked_config = {}

        for key, value in config_dict.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                masked_config[key] = '***MASKED***'
            else:
                masked_config[key] = value

        return masked_config

    @staticmethod
    def validate_api_keys(config: ConfigurationSet) -> ValidationResult:
        """Validate that required API keys are present and valid."""
        errors = []

        # Check for required API keys
        required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'SERPER_API_KEY']

        for key in required_keys:
            if not os.getenv(key):
                errors.append(f"Required API key missing: {key}")

        return ValidationResult(errors=errors, warnings=[])
```

### Performance Tuning Guide

```python
PERFORMANCE_TUNING_GUIDE = {
    "high_throughput": {
        "description": "Optimize for maximum throughput",
        "settings": {
            "max_concurrent_scrapes": 20,
            "max_concurrent_cleans": 10,
            "anti_bot_level": 1,
            "enable_judge_optimization": True,
            "enable_media_optimization": True,
            "target_successful_scrapes": 25
        },
        "trade_offs": "May trigger rate limiting on some sites"
    },

    "high_quality": {
        "description": "Optimize for highest content quality",
        "settings": {
            "max_concurrent_scrapes": 6,
            "max_concurrent_cleans": 4,
            "anti_bot_level": 2,
            "cleanliness_threshold": 0.8,
            "min_quality_threshold": 85,
            "enable_enhancement": True,
            "max_enhancement_cycles": 5
        },
        "trade_offs": "Slower processing, higher resource usage"
    },

    "balanced": {
        "description": "Balance speed and quality",
        "settings": {
            "max_concurrent_scrapes": 12,
            "max_concurrent_cleans": 6,
            "anti_bot_level": 1,
            "cleanliness_threshold": 0.7,
            "min_quality_threshold": 70,
            "enable_judge_optimization": True,
            "target_successful_scrapes": 15
        },
        "trade_offs": "Good balance for most use cases"
    },

    "resource_constrained": {
        "description": "Optimize for low resource usage",
        "settings": {
            "max_concurrent_scrapes": 4,
            "max_concurrent_cleans": 2,
            "anti_bot_level": 1,
            "enable_media_optimization": True,
            "enable_streaming_pipeline": False,
            "target_successful_scrapes": 8
        },
        "trade_offs": "Slower processing, lower resource usage"
    }
}
```

### Troubleshooting Configuration

```python
class ConfigurationTroubleshooter:
    """Troubleshooting guide for configuration issues."""

    @staticmethod
    def diagnose_performance_issues(config: ConfigurationSet) -> List[str]:
        """Diagnose performance-related configuration issues."""
        issues = []

        if config.performance.max_concurrent_scrapes > 15:
            issues.append("High concurrency may cause rate limiting")

        if config.anti_bot.default_level == 0:
            issues.append("Low anti-bot level may cause many failures")

        if not config.content_cleaning.enable_judge_optimization:
            issues.append("Judge optimization disabled - slower processing")

        if config.quality.min_quality_threshold > 90:
            issues.append("Very high quality threshold may reduce results")

        return issues

    @staticmethod
    def suggest_optimizations(config: ConfigurationSet) -> List[str]:
        """Suggest configuration optimizations."""
        suggestions = []

        if not config.performance.enable_adaptive_concurrency:
            suggestions.append("Enable adaptive concurrency for better resource utilization")

        if config.content_cleaning.cleanliness_threshold < 0.6:
            suggestions.append("Consider increasing cleanliness threshold for better quality")

        if config.performance.max_concurrent_scrapes < 8:
            suggestions.append("Consider increasing concurrency for better throughput")

        if not config.monitoring.enable_performance_monitoring:
            suggestions.append("Enable performance monitoring to identify bottlenecks")

        return suggestions
```

This comprehensive configuration reference provides all the necessary information to configure, optimize, and troubleshoot the Two-Module Scraping System for any use case or deployment scenario.