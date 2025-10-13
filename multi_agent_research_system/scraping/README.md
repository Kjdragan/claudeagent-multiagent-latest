# Two-Module Scraping System - Quick Start Guide

## Overview

The Two-Module Scraping System provides high-quality, research-ready web content through intelligent anti-bot escalation and AI-powered content cleaning. This guide will help you get started quickly with basic usage examples and common configurations.

## Quick Installation

### Prerequisites

```bash
# Required Python packages
pip install crawl4ai
pip install pydantic-ai
pip install httpx
pip install asyncio
pip install beautifulsoup4
```

### Environment Setup

```bash
# Set required API keys
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export SERPER_API_KEY="your-serp-api-key"

# Optional performance settings
export ANTI_BOT_LEVEL=1
export TARGET_SUCCESSFUL_SCRAPES=15
export CLEANLINESS_THRESHOLD=0.7
```

## Basic Usage

### 1. Simple URL Scraping

```python
from multi_agent_research_system.utils.anti_bot_escalation import get_escalation_manager

# Initialize escalation manager
escalation_manager = get_escalation_manager()

# Scrape a single URL with anti-bot protection
async def scrape_url(url: str):
    result = await escalation_manager.crawl_with_escalation(
        url=url,
        initial_level=1,  # Start with enhanced headers
        max_level=3,      # Allow escalation to stealth mode
        session_id="my_session"
    )

    if result.success:
        print(f"✅ Success: {result.char_count} characters extracted")
        print(f"Content preview: {result.content[:200]}...")
        print(f"Anti-bot level used: {result.final_level}")
    else:
        print(f"❌ Failed: {result.error}")
        print(f"Attempts made: {result.attempts_made}")

# Usage
await scrape_url("https://example.com/article")
```

### 2. Content Cleaning with AI

```python
from multi_agent_research_system.utils.content_cleaning import (
    clean_content_with_judge_optimization,
    assess_content_cleanliness
)

# Clean scraped content with AI optimization
async def clean_content(raw_content: str, url: str, search_query: str):
    # Step 1: Check if content needs cleaning (saves 35-40s if clean)
    is_clean, score = await assess_content_cleanliness(raw_content, url, 0.7)

    if is_clean:
        print(f"✅ Content already clean (score: {score:.2f})")
        return raw_content
    else:
        print(f"🧹 Cleaning needed (score: {score:.2f})")

        # Step 2: Clean with AI
        cleaned_content, metadata = await clean_content_with_judge_optimization(
            content=raw_content,
            url=url,
            search_query=search_query,
            cleanliness_threshold=0.7
        )

        print(f"✅ Content cleaned successfully")
        print(f"Original: {len(raw_content)} chars → Cleaned: {len(cleaned_content)} chars")
        print(f"Processing time: {metadata['processing_time']:.2f}s")

        return cleaned_content

# Usage
raw_html = "<html>...</html>"  # Your scraped content
clean_content = await clean_content(raw_html, "https://example.com", "artificial intelligence")
```

### 3. Batch Processing with Streaming Pipeline

```python
from multi_agent_research_system.utils.streaming_scrape_clean_pipeline import StreamingScrapeCleanPipeline

# Process multiple URLs with streaming (recommended for efficiency)
async def batch_scrape_and_clean(urls: list[str], search_query: str):
    # Initialize streaming pipeline
    pipeline = StreamingScrapeCleanPipeline(
        max_concurrent_scrapes=8,    # Limit concurrent scrapes
        max_concurrent_cleans=6      # Limit concurrent cleaning
    )

    # Process all URLs
    results = await pipeline.process_urls_streaming(
        urls=urls,
        search_query=search_query,
        session_id="batch_session",
        initial_level=1,
        max_level=3
    )

    # Filter successful results
    successful_results = [
        result for result in results
        if result.scrape_success and result.clean_success
    ]

    print(f"✅ Processed {len(successful_results)}/{len(urls)} URLs successfully")

    # Display results
    for i, result in enumerate(successful_results, 1):
        print(f"\n{i}. {result.url}")
        print(f"   Quality Score: {result.quality_score}/100")
        print(f"   Processing Time: {result.total_time:.2f}s")
        print(f"   Content Preview: {result.cleaned_content[:150]}...")

    return successful_results

# Usage
urls = [
    "https://example.com/article1",
    "https://example.com/article2",
    "https://example.com/article3"
]
results = await batch_scrape_and_clean(urls, "latest technology trends")
```

## Configuration Examples

### Basic Configuration

```python
from multi_agent_research_system.config.settings import get_enhanced_search_config

# Load default configuration
config = get_enhanced_search_config()

print(f"Default search results: {config.default_num_results}")
print(f"Default anti-bot level: {config.default_anti_bot_level}")
print(f"Target successful scrapes: {config.target_successful_scrapes}")
print(f"Cleanliness threshold: {config.default_cleanliness_threshold}")
```

### Custom Configuration

```python
# Custom scraping session
async def custom_scraping_session():
    # High-quality scraping for important research
    important_config = {
        "anti_bot_level": 2,           # Start with advanced protection
        "target_successful_scrapes": 25,  # More content needed
        "cleanliness_threshold": 0.8,  # Higher quality standard
        "max_concurrent_scrapes": 5     # Conservative concurrency
    }

    # Quick scraping for basic research
    quick_config = {
        "anti_bot_level": 1,           # Enhanced headers only
        "target_successful_scrapes": 8,   # Less content needed
        "cleanliness_threshold": 0.6,  # Lower quality threshold
        "max_concurrent_scrapes": 12    # Higher concurrency
    }

    # Use configuration based on your needs
    return important_config  # or quick_config
```

### Environment-Based Configuration

```bash
# Create .env file for your project
cat > .env << EOF
# API Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
SERPER_API_KEY=your-serp-key

# Scraping Configuration
ANTI_BOT_LEVEL=2
TARGET_SUCCESSFUL_SCRAPES=20
CLEANLINESS_THRESHOLD=0.75
MAX_CONCURRENT_SCRAPES=10

# Performance Settings
ENABLE_SUCCESS_BASED_TERMINATION=true
ANTI_BOT_AUTO_LEARNING=true
LOG_LEVEL=INFO
EOF

# Load environment variables in Python
import os
from dotenv import load_dotenv

load_dotenv()

anti_bot_level = int(os.getenv('ANTI_BOT_LEVEL', '1'))
target_scrapes = int(os.getenv('TARGET_SUCCESSFUL_SCRAPES', '15'))
```

## Integration Examples

### Integration with Research Workflow

```python
from multi_agent_research_system.utils.serp_search_utils import search_with_serp_api
from multi_agent_research_system.utils.streaming_scrape_clean_pipeline import StreamingScrapeCleanPipeline

async def research_workflow(query: str, max_results: int = 15):
    """Complete research workflow from search to clean content."""

    print(f"🔍 Searching for: {query}")

    # Step 1: Search for relevant URLs
    search_results = await search_with_serp_api(query, max_results)
    urls = [result['url'] for result in search_results]

    print(f"📋 Found {len(urls)} URLs to process")

    # Step 2: Scrape and clean content
    pipeline = StreamingScrapeCleanPipeline()
    results = await pipeline.process_urls_streaming(
        urls=urls,
        search_query=query,
        session_id=f"research_{query.replace(' ', '_')}"
    )

    # Step 3: Filter and format results
    successful_results = [
        {
            'url': result.url,
            'content': result.cleaned_content,
            'quality_score': result.quality_score,
            'word_count': len(result.cleaned_content.split())
        }
        for result in results
        if result.scrape_success and result.clean_success
    ]

    print(f"✅ Successfully processed {len(successful_results)} URLs")

    # Step 4: Return formatted results
    return {
        'query': query,
        'total_urls_found': len(urls),
        'successful_extractions': len(successful_results),
        'results': successful_results
    }

# Usage
research_results = await research_workflow("artificial intelligence in healthcare 2024")
print(f"Found {len(research_results['results'])} high-quality articles")
```

### Integration with Multi-Agent System

```python
# Custom agent that uses scraping system
class ResearchAgent:
    def __init__(self):
        self.pipeline = StreamingScrapeCleanPipeline()

    async def research_topic(self, topic: str, depth: str = "standard"):
        """Research a topic with configurable depth."""

        # Configure based on depth
        if depth == "comprehensive":
            max_results = 25
            target_scrapes = 20
            anti_bot_level = 2
        elif depth == "quick":
            max_results = 8
            target_scrapes = 5
            anti_bot_level = 1
        else:  # standard
            max_results = 15
            target_scrapes = 12
            anti_bot_level = 1

        # Search and scrape
        search_results = await search_with_serp_api(topic, max_results)

        results = await self.pipeline.process_urls_streaming(
            urls=[r['url'] for r in search_results],
            search_query=topic,
            session_id=f"agent_research_{topic.replace(' ', '_')}",
            initial_level=anti_bot_level
        )

        # Filter high-quality results
        high_quality = [
            r for r in results
            if r.scrape_success and r.clean_success and r.quality_score >= 70
        ]

        return {
            'topic': topic,
            'depth': depth,
            'sources_found': len(search_results),
            'high_quality_sources': len(high_quality),
            'content': [r.cleaned_content for r in high_quality],
            'sources': [r.url for r in high_quality]
        }

# Usage
agent = ResearchAgent()
research = await agent.research_topic("quantum computing applications", depth="comprehensive")
```

## Common Troubleshooting

### Fixing Low Success Rates

```python
# Diagnose and fix low success rates
async def improve_success_rates():
    """Improve scraping success rates through configuration."""

    # Enable more aggressive anti-bot measures
    escalation_manager = get_escalation_manager()

    # Check current performance
    stats = escalation_manager.get_stats()
    print(f"Current success rate: {stats['overall_success_rate']:.1%}")

    if stats['overall_success_rate'] < 0.7:
        print("💡 Recommendations to improve success rate:")
        print("1. Increase default anti-bot level to 2 (Advanced)")
        print("2. Enable auto-learning for difficult sites")
        print("3. Reduce concurrent scraping to avoid rate limiting")
        print("4. Add specific domains to difficult sites database")

    # Apply improvements
    improved_config = {
        'initial_level': 2,  # Start with advanced protection
        'max_concurrent': 5,  # Reduce concurrency
        'enable_learning': True  # Enable auto-learning
    }

    return improved_config
```

### Performance Optimization

```python
# Optimize for speed vs quality
async def optimize_performance(priority: str = "balanced"):
    """Optimize scraping performance based on priority."""

    if priority == "speed":
        # Fastest configuration
        config = {
            "anti_bot_level": 1,
            "cleanliness_threshold": 0.6,  # Skip more cleaning
            "max_concurrent_scrapes": 15,
            "max_concurrent_cleans": 12,
            "media_optimization": True
        }
        print("⚡ Optimized for speed (may reduce quality)")

    elif priority == "quality":
        # Highest quality configuration
        config = {
            "anti_bot_level": 2,
            "cleanliness_threshold": 0.8,  # Strict quality
            "max_concurrent_scrapes": 6,
            "max_concurrent_cleans": 4,
            "media_optimization": False
        }
        print("🎯 Optimized for quality (slower processing)")

    else:  # balanced
        # Balanced configuration
        config = {
            "anti_bot_level": 1,
            "cleanliness_threshold": 0.7,
            "max_concurrent_scrapes": 8,
            "max_concurrent_cleans": 6,
            "media_optimization": True
        }
        print("⚖️ Balanced speed and quality")

    return config
```

### Content Quality Issues

```python
# Diagnose content quality problems
def diagnose_quality_issues(results: list) -> dict:
    """Diagnose common content quality issues."""

    issues = {
        "problems": [],
        "recommendations": []
    }

    # Check for short content
    short_content = [r for r in results if len(r.cleaned_content) < 500]
    if len(short_content) > len(results) * 0.3:
        issues["problems"].append("Many results have very short content")
        issues["recommendations"].append("Reduce crawl threshold to include more content")

    # Check for low quality scores
    low_quality = [r for r in results if r.quality_score < 50]
    if len(low_quality) > len(results) * 0.4:
        issues["problems"].append("Many results have low quality scores")
        issues["recommendations"].append("Check search query relevance and URL selection")

    # Check for cleaning failures
    cleaning_failures = [r for r in results if not r.clean_success]
    if len(cleaning_failures) > len(results) * 0.2:
        issues["problems"].append("High content cleaning failure rate")
        issues["recommendations"].append("Check OpenAI API access and content cleaning prompts")

    return issues

# Usage
results = await batch_scrape_and_clean(urls, "your query")
issues = diagnose_quality_issues(results)
if issues["problems"]:
    print("⚠️ Quality Issues Detected:")
    for problem in issues["problems"]:
        print(f"  • {problem}")
    print("\n💡 Recommendations:")
    for rec in issues["recommendations"]:
        print(f"  • {rec}")
```

## Best Practices

### 1. Start with Conservative Settings

```python
# Good starting configuration for most use cases
starter_config = {
    "anti_bot_level": 1,        # Enhanced headers (good balance)
    "target_successful_scrapes": 12,  # Reasonable content amount
    "cleanliness_threshold": 0.7,    # Standard quality
    "max_concurrent_scrapes": 8      # Moderate concurrency
}
```

### 2. Monitor Performance

```python
# Always monitor your scraping performance
async def monitor_session(session_id: str):
    """Monitor scraping session performance."""

    escalation_manager = get_escalation_manager()
    stats = escalation_manager.get_stats()

    print(f"📊 Session Performance for {session_id}:")
    print(f"  Success Rate: {stats['overall_success_rate']:.1%}")
    print(f"  Escalation Rate: {stats['escalation_rate']:.1%}")
    print(f"  Avg Attempts per URL: {stats['avg_attempts_per_url']:.1f}")
    print(f"  Domains Learned: {len(stats['difficult_sites_stats']['configured_sites'])}")

    # Provide recommendations
    if stats['overall_success_rate'] < 0.8:
        print("⚠️ Consider increasing anti-bot level for better success rates")

    if stats['escalation_rate'] > 0.5:
        print("💡 Many sites require escalation - consider updating difficult sites database")
```

### 3. Handle Errors Gracefully

```python
# Robust error handling
async def safe_scrape_with_retry(url: str, max_retries: int = 3):
    """Scrape URL with comprehensive error handling."""

    escalation_manager = get_escalation_manager()

    for attempt in range(max_retries):
        try:
            result = await escalation_manager.crawl_with_escalation(
                url=url,
                initial_level=min(attempt + 1, 3),  # Escalate with retries
                session_id=f"retry_session_{attempt}"
            )

            if result.success:
                return result

            # Wait between retries with exponential backoff
            wait_time = 2 ** attempt
            print(f"⏳ Attempt {attempt + 1} failed, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)

        except Exception as e:
            print(f"❌ Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                # Return failure result on final attempt
                from multi_agent_research_system.utils.anti_bot_escalation import EscalationResult
                return EscalationResult(
                    url=url,
                    success=False,
                    error=f"All {max_retries} attempts failed: {str(e)}"
                )

    return None
```

### 4. Use Content Filtering

```python
# Filter content for quality and relevance
def filter_high_quality_content(results: list, min_quality: int = 70, min_length: int = 1000):
    """Filter results for high-quality content."""

    high_quality = []

    for result in results:
        if not (result.scrape_success and result.clean_success):
            continue

        # Quality filters
        if result.quality_score < min_quality:
            continue

        if len(result.cleaned_content) < min_length:
            continue

        # Content quality checks
        if not _contains_meaningful_content(result.cleaned_content):
            continue

        high_quality.append(result)

    return high_quality

def _contains_meaningful_content(content: str) -> bool:
    """Check if content contains meaningful information."""

    # Simple heuristics for meaningful content
    word_count = len(content.split())
    sentence_count = content.count('.') + content.count('!') + content.count('?')

    # Should have multiple sentences and reasonable word count
    return word_count >= 100 and sentence_count >= 3
```

## Advanced Tips

### Custom Anti-Bot Strategies

```python
# Configure custom strategies for specific domains
def configure_domain_strategies():
    """Configure custom anti-bot strategies for difficult domains."""

    escalation_manager = get_escalation_manager()

    # Add known difficult sites
    difficult_sites = [
        ("linkedin.com", 3, "Heavy bot protection, requires stealth mode"),
        ("medium.com", 2, "JavaScript heavy, requires browser automation"),
        ("instagram.com", 3, "Social media with strict anti-bot"),
        ("twitter.com", 3, "Rate limiting and bot detection"),
    ]

    for domain, level, reason in difficult_sites:
        escalation_manager.difficult_sites_manager.add_difficult_site(
            domain=domain,
            level=level,
            reason=reason
        )

    print("✅ Configured custom strategies for difficult domains")
```

### Batch Processing for Large Research

```python
# Process large research projects efficiently
async def large_research_project(query: str, target_content_count: int = 50):
    """Handle large research projects with batch processing."""

    all_results = []
    batch_size = 20
    processed_count = 0

    print(f"🎯 Target: {target_content_count} high-quality articles")

    while len(all_results) < target_content_count:
        # Search for more URLs
        search_results = await search_with_serp_api(
            query,
            max_results=batch_size,
            start=processed_count
        )

        if not search_results:
            print("📋 No more search results available")
            break

        # Process batch
        pipeline = StreamingScrapeCleanPipeline()
        batch_results = await pipeline.process_urls_streaming(
            urls=[r['url'] for r in search_results],
            search_query=query,
            session_id=f"large_research_batch_{processed_count//batch_size}"
        )

        # Filter and add successful results
        successful = [
            r for r in batch_results
            if r.scrape_success and r.clean_success and r.quality_score >= 70
        ]

        all_results.extend(successful)
        processed_count += len(search_results)

        print(f"📊 Progress: {len(all_results)}/{target_content_count} articles collected")

        # Avoid getting stuck in loop
        if len(search_results) < batch_size:
            break

    print(f"✅ Research complete: {len(all_results)} articles collected")
    return all_results
```

This quick start guide provides the essential information needed to get started with the Two-Module Scraping System. For more detailed technical information, see the [CLAUDE.md](CLAUDE.md) technical guide.