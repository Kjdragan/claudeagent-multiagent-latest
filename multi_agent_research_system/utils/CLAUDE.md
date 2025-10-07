# Utils Directory - Multi-Agent Research System

This directory contains core utility functions for web crawling, content processing, intelligent search strategy selection, anti-bot detection, and research data standardization that power the multi-agent research system.

## Directory Purpose

The utils directory provides the foundational infrastructure for intelligent web data collection, AI-powered content cleaning, adaptive search optimization, and research data processing. These utilities are designed to work together to provide reliable, scalable web research capabilities with built-in anti-detection mechanisms and performance optimization.

## Key Components

### Web Crawling & Data Collection
- **`crawl4ai_z_playground.py`** - Production-ready web crawler using Crawl4AI with direct implementation, multimedia exclusion, and Logfire integration
- **`crawl4ai_utils.py`** - Core crawling utilities and helper functions with comprehensive error handling
- **`crawl4ai_optimized.py`** - Performance-optimized crawling implementation for high-throughput scenarios
- **`crawl4ai_media_optimized.py`** - Advanced media handling and optimization utilities (3-4x performance improvement)
- **`crawl_enhancement.py`** - Advanced crawling enhancements with anti-bot integration
- **`z_crawl4ai_utils.py`** - Enhanced crawling utilities with additional optimization features
- **`crawl_utils.py`** - General-purpose crawling utilities and helper functions

### Content Processing & AI-Powered Cleaning
- **`content_cleaning.py`** - AI-powered content cleaning using GPT-5-nano with cleanliness assessment optimization
- **`z_content_cleaning.py`** - Enhanced content cleaning with better text extraction and noise removal
- **`modern_content_cleaner.py`** - Modern approach to content cleaning and normalization using advanced NLP techniques
- **`research_data_standardizer.py`** - Standardizes research data into structured session data for seamless agent integration

### Intelligent Search & Discovery
- **`search_strategy_selector.py`** - AI-driven search strategy selection based on query analysis, timing, and topic characteristics
- **`serp_search_utils.py`** - SERP API integration with 10x performance improvement and automatic content extraction
- **`z_search_crawl_utils.py`** - Integrated search and crawling utilities with optimized workflows
- **`enhanced_relevance_scorer.py`** - Sophisticated relevance scoring with domain authority boosting (Position 40% + Title 30% + Snippet 30%)
- **`query_intent_analyzer.py`** - Analyzes user queries to determine appropriate report formats and processing approaches

### Anti-Detection & Reliability
- **`anti_bot_escalation.py`** - 4-level progressive anti-bot escalation system (Basic → Enhanced → Advanced → Stealth)
- **`port_manager.py`** - Cross-platform network port management for crawling operations
- **`url_tracker.py`** - URL deduplication system with progressive retry logic and comprehensive tracking

### Testing & Optimization
- **`test_media_optimization.py`** - Testing utilities for media optimization performance validation
- **`MEDIA_OPTIMIZATION_GUIDE.md`** - Comprehensive implementation guide for media handling optimization

## Architecture & Workflow Integration

### Multi-Agent Research Pipeline
```
User Query → Query Intent Analysis → Search Strategy Selection → Web Crawling → AI Content Cleaning → Data Standardization → Agent Processing
```

### Progressive Anti-Bot Escalation System
```
Initial Request → Detection Level 0 (Basic) → Level 1 (Enhanced) → Level 2 (Advanced) → Level 3 (Stealth) → Success/Failure
```

### AI-Powered Content Processing Pipeline
```
Raw HTML → Cleanliness Assessment → AI Cleaning (GPT-5-nano) → Relevance Scoring → Standardization → Quality Assessment → Agent Consumption
```

### Search Strategy Decision Flow
```
Query Analysis → Time Factor Assessment → Topic Classification → Strategy Selection (Google/SERP/Hybrid) → Performance Optimization
```

## Core Implementation Patterns

### Async-First Architecture
```python
# All crawling operations follow async patterns
async def crawl_with_escalation(url: str, max_level: int = 3) -> CrawlResult:
    escalator = AntiBotEscalator()
    return await escalator.attempt_crawl(url, max_level)
```

### AI-Enhanced Content Processing
```python
# Intelligent content cleaning with optimization
async def clean_content_optimized(content: str, url: str) -> dict:
    # First assess if cleaning is needed
    is_clean, score = await assess_content_cleanliness(content, url)
    if is_clean:
        return {"content": content, "cleaned": False}

    # Apply AI cleaning only when necessary
    return await clean_with_ai(content, url)
```

### Progressive Retry Logic
```python
# Smart retry with escalation
async def crawl_with_retry(url: str, session_id: str = None) -> CrawlResult:
    tracker = get_url_tracker()

    for level in range(4):  # 0-3 escalation levels
        if tracker.should_retry(url, level):
            result = await crawl_with_anti_bot(url, level)
            tracker.record_attempt(url, result, level)
            if result.success:
                return result

    return CrawlResult(url, False, error="All escalation levels failed")
```

### Intelligent Search Strategy Selection
```python
# AI-driven search strategy selection
def select_optimal_strategy(query: str) -> StrategyAnalysis:
    selector = SearchStrategySelector()
    return selector.analyze_and_recommend(query)
```

## Testing & Quality Assurance

### Comprehensive Testing Strategy
1. **Unit Tests**: Individual utility function testing with edge cases
2. **Integration Tests**: Pipeline interaction testing across utilities
3. **Anti-Bot Validation**: Progressive escalation system verification
4. **Content Quality Assessment**: AI cleaning effectiveness validation
5. **Performance Benchmarking**: Media optimization and crawling speed tests
6. **URL Tracking Tests**: Deduplication and retry logic validation

### Development & Debugging Tools
1. **Logfire Integration**: Comprehensive observability and tracing
2. **Verbose Logging**: Structured logging at multiple levels
3. **Content Inspection**: Intermediate processing result preservation
4. **Performance Monitoring**: Real-time speed and success rate tracking
5. **Anti-Bot Diagnostics**: Detection and escalation event logging
6. **Query Intent Analysis**: Format decision process transparency

### Production Issue Resolution
- **Detection/Bot Blocking**: Automatic escalation through 4 anti-bot levels
- **Poor Content Extraction**: AI-powered cleaning with cleanliness assessment
- **Performance Bottlenecks**: Media optimization (3-4x improvement) and concurrent processing
- **Memory Constraints**: Streaming processing and token limit management
- **Search Inefficiency**: Intelligent strategy selection (Google vs SERP vs Hybrid)

## Dependencies & System Integration

### External Dependencies
- **crawl4ai**: Core web crawling framework with multimedia optimization
- **aiohttp**: Async HTTP client for high-performance web requests
- **beautifulsoup4**: HTML parsing and content extraction
- **logfire**: Advanced observability and structured logging
- **pydantic-ai**: AI agents for content cleaning and assessment
- **httpx**: Modern HTTP client for SERP API integration

### Core System Dependencies
- **Agent System**: Research, Report, Editorial, and Quality agents consume processed data
- **MCP Integration**: Utilities provide data to Model Context Protocol implementations
- **Configuration System**: Environment variables and settings control utility behavior
- **Session Management**: Research session tracking and state persistence

### Data Flow Architecture
```
External Sources → Search/Crawling Utils → AI Cleaning Utils → Standardization Utils → Quality Assessment → Agent Processing → Final Output
```

### Module Exports
```python
# Main utilities exported through __init__.py
from utils.port_manager import (
    ensure_port_available,
    find_process_using_port,
    get_available_port,
    kill_process_using_port
)
```

## Usage Examples & Best Practices

### AI-Powered Web Crawling with Anti-Bot Protection
```python
from utils.crawl4ai_z_playground import ZPlaygroundCrawler
from utils.anti_bot_escalation import AntiBotEscalator

# High-performance crawling with automatic escalation
crawler = ZPlaygroundCrawler()
escalator = AntiBotEscalator()

result = await escalator.attempt_crawl(
    url="https://example.com",
    max_level=3  # Progressive anti-bot escalation
)

if result.success:
    print(f"Content: {result.content[:200]}...")
    print(f"Anti-bot level used: {result.final_level}")
```

### Intelligent Search Strategy Selection
```python
from utils.search_strategy_selector import SearchStrategySelector
from utils.query_intent_analyzer import analyze_query_intent

# AI-driven search optimization
selector = SearchStrategySelector()
query = "latest developments in quantum computing 2024"

# Get optimal search strategy
strategy = selector.analyze_and_recommend(query)
print(f"Strategy: {strategy.recommended_strategy.value}")
print(f"Confidence: {strategy.confidence}")
print(f"Reasoning: {strategy.reasoning}")

# Analyze query intent for format selection
intent = analyze_query_intent(query)
print(f"Recommended format: {intent['format']}")
```

### AI-Enhanced Content Processing Pipeline
```python
from utils.content_cleaning import assess_content_cleanliness, clean_content
from utils.research_data_standardizer import standardize_research_data
from utils.enhanced_relevance_scorer import calculate_relevance_score

# Complete content processing pipeline
async def process_research_content(raw_html: str, url: str) -> dict:
    # 1. Assess if cleaning is needed (performance optimization)
    is_clean, cleanliness_score = await assess_content_cleanliness(raw_html, url)

    if not is_clean:
        # 2. Apply AI-powered cleaning only when necessary
        cleaned_content = await clean_content(raw_html, url)
    else:
        cleaned_content = raw_html

    # 3. Calculate relevance with domain authority boosting
    relevance_score = calculate_relevance_score(cleaned_content, url)

    # 4. Standardize for agent consumption
    standardized = standardize_research_data({
        "content": cleaned_content,
        "url": url,
        "relevance_score": relevance_score,
        "cleanliness_score": cleanliness_score
    })

    return standardized
```

### Media-Optimized High-Performance Crawling
```python
from utils.crawl4ai_media_optimized import MediaOptimizedCrawler

# 3-4x performance improvement with media exclusion
crawler = MediaOptimizedCrawler()
config = crawler.get_optimized_config(
    text_mode=True,           # Disable images and heavy content
    exclude_all_images=True,  # Remove all images completely
    light_mode=True,          # Disable background features
    page_timeout=20000        # Faster timeout
)

result = await crawler.crawl_with_config(url, config)
```

### URL Tracking and Progressive Retry Logic
```python
from utils.url_tracker import get_url_tracker

# Intelligent URL management with deduplication
tracker = get_url_tracker()

# Check if URL was recently processed
if not tracker.should_process(url):
    print("URL recently processed, skipping...")
    return

# Record attempt with comprehensive tracking
tracker.record_attempt(
    url=url,
    success=True,
    anti_bot_level=2,
    content_length=len(content),
    duration=2.5,
    session_id="research_session_123"
)
```

### Query Intent Analysis for Format Selection
```python
from utils.query_intent_analyzer import get_query_intent_analyzer

analyzer = get_query_intent_analyzer()

# Analyze multiple queries for batch processing
queries = [
    "brief overview of machine learning",
    "comprehensive analysis of climate change impacts",
    "quick summary of latest tech news"
]

for query in queries:
    intent = analyzer.analyze_query_intent(query)
    format_type = analyzer.suggest_format(query)

    print(f"Query: {query}")
    print(f"Format: {format_type} (confidence: {intent['confidence']})")
    print(f"Reasoning: {intent['reasoning']}")
    print("---")
```

## Performance Optimization Guidelines

### Media Optimization (3-4x Performance Improvement)
```python
# Recommended configuration for text-only research
config = CrawlerRunConfig(
    text_mode=True,                    # Disable images and heavy content
    exclude_all_images=True,           # Remove all images completely
    exclude_external_images=True,      # Block external domain images
    light_mode=True,                   # Disable background features
    wait_for="body",                   # Faster than domcontentloaded
    page_timeout=20000                 # Shorter timeout for better responsiveness
)
```

### Anti-Bot Escalation Configuration
```python
# Progressive escalation levels
ANTI_BOT_CONFIGS = {
    0: {"headers": "basic", "delay": 1, "js_rendering": False},
    1: {"headers": "enhanced", "delay": 2, "js_rendering": True},
    2: {"headers": "advanced", "delay": 5, "proxy_rotation": True},
    3: {"headers": "stealth", "delay": 10, "full_browser_simulation": True}
}
```

### Content Processing Optimization
- Use cleanliness assessment to skip unnecessary AI cleaning
- Implement token limit management for large content
- Cache cleaning results for frequently accessed content
- Use concurrent processing for multiple URLs

### Scaling Best Practices
- Implement intelligent caching strategies
- Use connection pooling for HTTP requests
- Monitor success rates and adjust anti-bot strategies
- Apply progressive retry logic with backoff
- Use distributed processing for large-scale operations

## Configuration & Environment Setup

### Required Environment Variables
```bash
# API Keys
ANTHROPIC_API_KEY=your_anthropic_key          # For AI content cleaning
SERPER_API_KEY=your_serp_key                  # For search operations

# Performance Configuration
DEFAULT_CRAWL_TIMEOUT=30000                   # Crawl timeout in milliseconds
MAX_CONCURRENT_CRAWLS=10                      # Concurrent crawl limit
CLEANLINESS_THRESHOLD=0.7                     # Content cleanliness threshold
ANTI_BOT_MAX_LEVEL=3                          # Maximum anti-bot escalation level

# Development Settings
UTILS_LOG_LEVEL=INFO                          # Logging level for utils
DEBUG_CRAWLING=false                          # Enable detailed crawling logs
MEDIA_OPTIMIZATION=true                       # Enable media optimization
```

### Utility-Specific Configuration
```python
# Content Cleaning Configuration
CONTENT_CLEANING_CONFIG = {
    "cleanliness_threshold": 0.7,
    "ai_model": "gpt-5-nano",
    "max_content_length": 50000,
    "cache_cleaned_content": True
}

# Search Strategy Configuration
SEARCH_STRATEGY_CONFIG = {
    "time_sensitive_keywords": ["latest", "news", "breaking", "today"],
    "authority_domains": ["gov", "edu", "nature.com", "science.org"],
    "default_confidence_threshold": 0.6
}

# Anti-Bot Configuration
ANTI_BOT_CONFIG = {
    "max_escalation_level": 3,
    "base_delay": 1.0,
    "delay_multiplier": 2.0,
    "success_rate_threshold": 0.8
}
```

## Monitoring & Observability

### Logfire Integration
```python
# Automatic observability with Logfire
import logfire

# Spans are automatically created for major operations
async def crawl_with_monitoring(url: str):
    with logfire.span("crawling_operation", url=url):
        result = await crawler.crawl_url(url)
        logfire.info("crawl_completed", success=result.success, duration=result.duration)
        return result
```

### Key Metrics to Monitor
- **Crawl Success Rate**: Percentage of successful crawling operations
- **Anti-Bot Escalation Frequency**: How often escalation is triggered
- **Content Cleaning Performance**: Time spent on AI-powered cleaning
- **Search Strategy Effectiveness**: Success rates by strategy type
- **URL Deduplication Efficiency**: Duplicate detection and prevention

### Performance Dashboards
- Real-time crawling speed and success rates
- Anti-bot escalation level distribution
- Content quality scores and cleaning effectiveness
- Search strategy performance comparison
- System resource utilization (memory, CPU, network)

## Troubleshooting Guide

### Common Issues and Solutions

#### Crawling Failures
```python
# Symptom: High failure rates
# Solution: Check anti-bot escalation logs
from utils.anti_bot_escalation import AntiBotEscalator

escalator = AntiBotEscalator()
escalation_stats = escalator.get_escalation_statistics()
print(f"Most used level: {escalation_stats['most_common_level']}")
print(f"Success rate by level: {escalation_stats['success_by_level']}")
```

#### Content Quality Issues
```python
# Symptom: Poor content extraction
# Solution: Check cleanliness assessment and cleaning logs
from utils.content_cleaning import assess_content_cleanliness

is_clean, score = await assess_content_cleanliness(content, url)
if not is_clean:
    print(f"Content needs cleaning (score: {score})")
    # Review AI cleaning effectiveness
```

#### Performance Bottlenecks
```python
# Symptom: Slow processing
# Solution: Enable media optimization
from utils.crawl4ai_media_optimized import MediaOptimizedCrawler

crawler = MediaOptimizedCrawler()
# 3-4x performance improvement with media exclusion
optimized_config = crawler.get_optimized_config()
```

#### Search Strategy Issues
```python
# Symptom: Ineffective search results
# Solution: Analyze strategy selection effectiveness
from utils.search_strategy_selector import SearchStrategySelector

selector = SearchStrategySelector()
strategy_stats = selector.get_performance_stats()
print(f"Strategy effectiveness: {strategy_stats}")
```

### Debug Mode Activation
```bash
# Enable comprehensive debugging
export UTILS_LOG_LEVEL=DEBUG
export DEBUG_CRAWLING=true
export ANTI_BOT_DEBUG=true

# Run with verbose output
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from utils.crawl4ai_z_playground import ZPlaygroundCrawler
# ... your code here
"
```

## Integration Examples

### Integration with Research Agents
```python
from utils.serp_search_utils import search_with_serp_api
from utils.research_data_standardizer import standardize_research_data

async def research_agent_workflow(query: str):
    # 1. Get search results with SERP API (10x faster)
    search_results = await search_with_serp_api(query, max_results=10)

    # 2. Standardize data for agent consumption
    standardized_data = standardize_research_data({
        "query": query,
        "sources": search_results,
        "timestamp": datetime.now().isoformat()
    })

    # 3. Return data ready for report generation
    return standardized_data
```

### Integration with MCP Tools
```python
from utils.crawl4ai_z_playground import ZPlaygroundCrawler
from utils.content_cleaning import clean_content

@mcp_tool()
async def web_research_tool(url: str, clean_content: bool = True):
    """MCP tool for web research with content cleaning."""
    crawler = ZPlaygroundCrawler()
    result = await crawler.crawl_url(url)

    if result.success and clean_content:
        cleaned = await clean_content(result.content, url)
        result.content = cleaned

    return {
        "url": url,
        "content": result.content,
        "success": result.success,
        "word_count": result.word_count
    }
```

## Best Practices & Guidelines

### Code Quality Standards
- Use comprehensive error handling with fallback mechanisms
- Implement proper logging at appropriate levels
- Follow async/await patterns for all I/O operations
- Use type hints for better code documentation and IDE support
- Write docstrings for all public functions and classes

### Performance Optimization
- Always use media optimization for text-only research
- Implement intelligent caching to avoid redundant operations
- Use cleanliness assessment to skip unnecessary AI cleaning
- Apply concurrent processing for independent operations
- Monitor and optimize token usage for AI operations

### Security Considerations
- Rotate user agents and implement anti-bot detection avoidance
- Use rate limiting to avoid overwhelming target websites
- Implement proper error handling to avoid information leakage
- Use secure storage for API keys and sensitive configuration
- Monitor for unusual activity patterns

### Maintainability Guidelines
- Keep utility functions focused and single-purpose
- Use consistent naming conventions across all utilities
- Implement comprehensive testing for all utility functions
- Document all configuration options and their effects
- Use dependency injection for better testability

## Future Development Roadmap

### Planned Enhancements
1. **Advanced AI Integration**: More sophisticated content analysis and extraction
2. **Enhanced Search Strategies**: Additional search sources and intelligent query optimization
3. **Performance Improvements**: Further optimization of crawling and cleaning operations
4. **Expanded Anti-Bot Features**: More sophisticated detection avoidance mechanisms
5. **Better Caching Strategies**: Intelligent caching with expiration and invalidation

### Extension Points
- Custom content cleaning algorithms
- Additional search source integrations
- Specialized anti-bot strategies for specific domains
- Custom relevance scoring algorithms
- Domain-specific content extraction rules

### Contributing Guidelines
- Follow established code patterns and conventions
- Add comprehensive tests for new functionality
- Update documentation for all configuration options
- Ensure performance impact is measured and optimized
- Maintain backward compatibility when possible