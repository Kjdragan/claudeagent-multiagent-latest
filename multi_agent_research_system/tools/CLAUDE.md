# Tools Directory - Multi-Agent Research System

This directory contains high-level research tools and search interfaces built as MCP (Model Context Protocol) tools that provide specialized capabilities for intelligent information discovery, web scraping, and research orchestration.

## Directory Purpose

The tools directory provides sophisticated research orchestration and search capabilities that build upon the utilities layer. These tools are implemented as MCP tools using the Claude SDK, offering intelligent research automation, advanced scraping functionality, and search engine result processing to support the multi-agent research system. Each tool is designed as a self-contained, reusable component that can be exposed through MCP interfaces or used directly in agent workflows.

## Key Components

### Core Research Tools
- **`intelligent_research_tool.py`** - Complete intelligent research system implementing proven z-playground1 methodology with search → relevance filtering → parallel crawl → AI cleaning → MCP-compliant results
- **`advanced_scraping_tool.py`** - Advanced web scraping using Crawl4AI with browser automation, AI content cleaning, and technical content preservation
- **`serp_search_tool.py`** - High-performance Google search using SERP API with automatic content extraction and work product generation

## Tool Capabilities

### Intelligent Research Tool (`intelligent_research_tool.py`)
**Purpose**: Complete intelligent research system implementing proven z-playground1 methodology

**Core Responsibilities**:
- Search 15 URLs with redundancy for expected failures
- Apply enhanced relevance scoring (position 40% + title 30% + snippet 30%)
- Filter by relevance threshold (default 0.3)
- Execute parallel crawling with anti-bot escalation
- Perform AI content cleaning with search query filtering
- Generate smart content compression for MCP compliance
- Create complete work products with full research data

**Key Features**:
- **Enhanced Relevance Scoring**: Proven formula combining position, title matching, and snippet relevance
- **Threshold-Based Selection**: Intelligent URL filtering based on relevance scores
- **Multi-Level Compression**: Smart content allocation (Priority 1: full detail, Priority 2: summarized, Priority 3: references)
- **MCP Compliance**: Stays within 25K token limits while preserving research value
- **Work Product Generation**: Complete research data saved with full content and metadata
- **Error Recovery**: Graceful handling of failures with fallback strategies

**MCP Tool Function**: `intelligent_research_with_advanced_scraping`

### Advanced Scraping Tool (`advanced_scraping_tool.py`)
**Purpose**: Advanced web scraping with Crawl4AI browser automation and AI content cleaning

**Core Responsibilities**:
- Multi-stage extraction with fallback strategies (CSS selector → universal extraction → AI cleaning)
- Browser automation for JavaScript-heavy sites
- AI-powered content cleaning using GPT-5-nano
- Judge optimization for speed (saves 35-40s per URL)
- Technical content preservation (code blocks, installation commands)
- Parallel processing of multiple URLs with anti-bot detection

**Key Features**:
- **Multi-Stage Extraction**: Fast CSS selector extraction with robust fallback strategies
- **Judge Optimization**: Speed optimization that saves 35-40 seconds per URL
- **Technical Content Preservation**: Maintains code blocks, commands, and technical information
- **High Success Rates**: Achieves 70-100% success rates with advanced anti-detection
- **Large Content Extraction**: Extracts 30K-58K characters (vs 2K limit in basic scraping)
- **Progressive Anti-Bot**: 4-level anti-bot detection escalation system

**MCP Tool Functions**: `advanced_scrape_url`, `advanced_scrape_multiple_urls`

### SERP Search Tool (`serp_search_tool.py`)
**Purpose**: High-performance Google search using SERP API with automatic content extraction

**Core Responsibilities**:
- Execute high-performance Google searches using SERP API
- Provide 10x faster search performance compared to MCP-based search
- Automatic content extraction with relevance scoring
- Work product generation with search metadata
- Configurable search parameters and thresholds

**Key Features**:
- **High Performance**: 10x faster than MCP search systems
- **Relevance Scoring**: Built-in relevance assessment and ranking
- **Auto-Crawl Capability**: Automatically crawls top results based on thresholds
- **Work Product Generation**: Saves complete search data with metadata
- **Flexible Search Types**: Supports various search modes (search, news, etc.)
- **Configurable Parameters**: Adjustable result counts and thresholds

**MCP Tool Function**: `serp_search`

## Tool Workflow Integration

### Complete Research Pipeline (Intelligent Research Tool)
```
User Query → SERP Search (15 URLs) → Enhanced Relevance Scoring → Threshold Filtering (0.3+) → Parallel Crawl → AI Content Cleaning → Smart Compression → MCP Response + Work Product
```

### Advanced Scraping Pipeline
```
URL Input → Multi-Stage Extraction (CSS → Universal → AI) → Content Cleaning → Technical Preservation → Quality Assessment → Structured Output
```

### SERP Search Pipeline
```
Query Input → SERP API → Results Parsing → Relevance Scoring → Auto-Crawl (Optional) → Content Extraction → Work Product Generation
```

### Tool Integration Patterns
```
Agent Request → Tool Selection → Parameter Processing → Tool Execution → Result Formatting → Agent Consumption
```

## MCP Tool Development Guidelines

### Tool Design Patterns
1. **MCP Tool Decorator**: Use the `@tool` decorator from Claude SDK for MCP compliance
2. **Self-Contained Processing**: Each tool handles complete workflows internally
3. **Error Recovery**: Implement graceful error handling with informative messages
4. **Parameter Validation**: Validate input parameters and provide clear error messages
5. **Async Operations**: Use async/await patterns for all network operations
6. **Work Product Generation**: Save detailed work products for complex operations

### MCP Tool Interface Standards
```python
# Standard MCP tool implementation pattern
from claude_agent_sdk import tool

@tool(
    "tool_name",
    "Tool description for MCP discovery",
    {
        "param1": str,
        "param2": int,
        "optional_param": bool
    }
)
async def tool_function(args):
    """Tool function implementing specific capability."""

    # Parameter extraction with defaults
    param1 = args.get("param1")
    param2 = args.get("param2", 10)
    optional_param = args.get("optional_param", True)

    try:
        # Core processing logic
        result = await _process_tool_logic(param1, param2, optional_param)

        # Return MCP-compliant response
        return {
            "content": [{"type": "text", "text": result}],
            "metadata": {"processing_info": "additional_data"}
        }

    except Exception as e:
        # Error handling with helpful messages
        error_msg = f"Tool execution failed: {str(e)}"
        return {
            "content": [{"type": "text", "text": error_msg}],
            "is_error": True
        }
```

### Tool Configuration Patterns
```python
# Tool configuration with environment awareness
def get_tool_configuration():
    return {
        "intelligent_research": {
            "max_urls": 10,
            "relevance_threshold": 0.3,
            "max_concurrent": 10,
            "compression_tokens": 20000
        },
        "advanced_scraping": {
            "default_timeout": 30,
            "preserve_technical": True,
            "max_concurrent": 5
        },
        "serp_search": {
            "num_results": 15,
            "auto_crawl_top": 5,
            "crawl_threshold": 0.3
        }
    }
```

### Error Handling Standards
```python
# Comprehensive error handling pattern
async def handle_tool_errors(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Check for common issues and provide specific guidance
            error_msg = str(e)

            if "SERP_API_KEY" in error_msg:
                error_msg += "\n\n⚠️ **SERP_API_KEY not found**\nAdd SERP_API_KEY to your .env file."
            elif "OPENAI_API_KEY" in error_msg:
                error_msg += "\n\n⚠️ **OPENAI_API_KEY not found**\nAdd OPENAI_API_KEY to your .env file for AI content cleaning."
            elif "playwright" in error_msg.lower():
                error_msg += "\n\n⚠️ **Playwright not installed**\nRun: uv run playwright install chromium"

            return {
                "content": [{"type": "text", "text": error_msg}],
                "is_error": True
            }
    return wrapper
```

## Testing & Debugging

### MCP Tool Testing Strategies
1. **Unit Testing**: Test individual tool functions in isolation with mocked dependencies
2. **Integration Testing**: Test tool interactions with utilities and external services
3. **MCP Protocol Testing**: Verify MCP compliance and proper response formatting
4. **Performance Testing**: Ensure tools perform within acceptable time limits
5. **Error Scenario Testing**: Test error handling and recovery mechanisms

### Debugging MCP Tools
1. **Verbose Logging**: Enable detailed logging with structured information
2. **Parameter Inspection**: Log input parameters and processing steps
3. **Response Validation**: Verify MCP response format and content
4. **Work Product Analysis**: Examine saved work products for completeness
5. **Performance Monitoring**: Track execution times and resource usage

### Common Tool Issues & Solutions
- **API Key Errors**: Check environment variables and service availability
- **Scraping Failures**: Verify anti-bot escalation and target site accessibility
- **Content Quality Issues**: Adjust cleaning parameters and relevance thresholds
- **MCP Compliance**: Ensure response format matches MCP standards
- **Token Limit Exceeded**: Implement proper content compression strategies

### Debug Commands
```bash
# Test tool functionality directly
python -c "
import asyncio
from tools.intelligent_research_tool import intelligent_research_with_advanced_scraping

async def test():
    result = await intelligent_research_with_advanced_scraping({
        'query': 'test query',
        'session_id': 'debug_session'
    })
    print(result)

asyncio.run(test())
"

# Check API key configuration
python -c "
import os
print(f'SERP_API_KEY: {os.getenv('SERP_API_KEY', 'NOT SET')}')
print(f'OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY', 'NOT SET')}')
"
```

## Dependencies & Interactions

### External Dependencies
- **claude-agent-sdk**: Core MCP framework and tool decorators
- **crawl4ai**: Advanced web crawling with browser automation
- **openai**: AI content cleaning and optimization
- **serpapi**: High-performance search engine API
- **playwright**: Browser automation for JavaScript-heavy sites
- **beautifulsoup4**: HTML parsing and content extraction
- **aiohttp**: Async HTTP client for web requests

### Internal Dependencies
- **Utils Layer**: Core crawling, content cleaning, and search utilities
  - `utils.crawl4ai_utils`: Multi-URL crawling with anti-bot features
  - `utils.content_cleaning`: AI-powered content cleaning and optimization
  - `utils.serp_search_utils`: Search engine result processing and analysis
  - `utils.anti_bot_escalation`: Progressive anti-bot detection strategies
- **Agent System**: Tools provide research data to agents for processing
- **KEVIN Directory**: Work product storage and session management
- **Config System**: Environment variables and system settings

### Tool Dependency Flow
```
MCP Tool Request → Parameter Validation → Utils Layer Processing → External Service Integration → Content Processing → Work Product Generation → MCP Response
```

### Import Patterns
```python
# Standard import pattern for tools
from claude_agent_sdk import tool

# Import utilities with fallback for different execution contexts
try:
    from ..utils.crawl4ai_utils import crawl_multiple_urls_with_cleaning
    from ..utils.content_cleaning import format_cleaned_results
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.crawl4ai_utils import crawl_multiple_urls_with_cleaning
    from utils.content_cleaning import format_cleaned_results
```

## Usage Examples

### Intelligent Research Tool Usage
```python
from tools.intelligent_research_tool import intelligent_research_with_advanced_scraping

# Complete research workflow in single tool call
result = await intelligent_research_with_advanced_scraping({
    "query": "latest developments in artificial intelligence",
    "session_id": "research_session_001",
    "max_urls": 10,
    "relevance_threshold": 0.3,
    "max_concurrent": 8
})

# Access research results
if not result.get("is_error"):
    content = result["content"][0]["text"]
    metadata = result.get("metadata", {})

    print(f"Search results found: {metadata.get('search_results_found', 0)}")
    print(f"Successfully crawled: {metadata.get('successful_crawls', 0)}")
    print(f"Total content extracted: {metadata.get('total_content_chars', 0):,} characters")
    print(f"Work product saved: {metadata.get('work_product_path', 'N/A')}")

    # Process the compressed research content
    print(content)
else:
    print(f"Research failed: {result['content'][0]['text']}")
```

### Advanced Scraping Tool Usage
```python
from tools.advanced_scraping_tool import advanced_scrape_url, advanced_scrape_multiple_urls

# Single URL scraping with AI cleaning
result = await advanced_scrape_url({
    "url": "https://example.com/technical-article",
    "session_id": "scraping_session_001",
    "search_query": "machine learning algorithms",
    "preserve_technical": True
})

if not result.get("is_error"):
    content = result["content"][0]["text"]
    print(f"Scraping successful: {content}")
else:
    print(f"Scraping failed: {result['content'][0]['text']}")

# Multiple URL parallel scraping
results = await advanced_scrape_multiple_urls({
    "urls": [
        "https://example.com/article1",
        "https://example.com/article2",
        "https://example.com/article3"
    ],
    "session_id": "parallel_scraping_001",
    "search_query": "data science trends",
    "max_concurrent": 3
})

content = results["content"][0]["text"]
print(f"Parallel scraping results: {content}")
```

### SERP Search Tool Usage
```python
from tools.serp_search_tool import serp_search

# High-performance Google search with auto-crawling
result = await serp_search({
    "query": "quantum computing applications in healthcare",
    "search_type": "search",
    "num_results": 15,
    "auto_crawl_top": 5,
    "crawl_threshold": 0.3,
    "session_id": "search_session_001"
})

if not result.get("is_error"):
    search_content = result["content"][0]["text"]
    print(f"Search completed: {search_content}")
else:
    print(f"Search failed: {result['content'][0]['text']}")
```

### Tool Integration in Agent Workflows
```python
# Example: Using tools within an agent
class ResearchAgent:
    def __init__(self):
        self.session_id = "agent_research_session"

    async def research_topic(self, topic: str, depth: str = "comprehensive"):
        # Use intelligent research for complete workflow
        research_result = await intelligent_research_with_advanced_scraping({
            "query": topic,
            "session_id": self.session_id,
            "max_urls": 15 if depth == "comprehensive" else 8,
            "relevance_threshold": 0.3,
            "max_concurrent": 10
        })

        if research_result.get("is_error"):
            raise Exception(f"Research failed: {research_result['content'][0]['text']}")

        return {
            "content": research_result["content"][0]["text"],
            "metadata": research_result.get("metadata", {}),
            "session_id": self.session_id
        }

    async def scrape_specific_sources(self, urls: list[str], context: str):
        # Use advanced scraping for specific URLs
        scraping_result = await advanced_scrape_multiple_urls({
            "urls": urls,
            "session_id": self.session_id,
            "search_query": context,
            "max_concurrent": 5
        })

        return scraping_result
```

## Performance Considerations

### Tool Optimization Strategies
1. **Intelligent Relevance Filtering**: Use enhanced scoring to process only high-quality sources
2. **Parallel Processing**: Concurrent crawling and content processing (configurable limits)
3. **Smart Content Compression**: Multi-level compression to stay within MCP token limits
4. **Judge Optimization**: AI judge optimization saves 35-40 seconds per URL in content cleaning
5. **Progressive Anti-Bot**: 4-level escalation system for reliable access without detection

### MCP Compliance Optimization
- **Token Management**: Smart content allocation (Priority 1-2-3 system) to stay under 25K limits
- **Response Formatting**: Standardized MCP response structure with metadata
- **Work Product Offloading**: Complete data saved to files, summaries returned in responses
- **Error Handling**: Comprehensive error messages with specific guidance for common issues

### Scaling Recommendations
```python
# Performance configuration for different scales
PERFORMANCE_CONFIG = {
    "small_scale": {
        "max_urls": 5,
        "max_concurrent": 3,
        "relevance_threshold": 0.4
    },
    "medium_scale": {
        "max_urls": 10,
        "max_concurrent": 6,
        "relevance_threshold": 0.3
    },
    "large_scale": {
        "max_urls": 20,
        "max_concurrent": 10,
        "relevance_threshold": 0.25
    }
}
```

### Quality vs. Performance Trade-offs
- **High Quality Mode**: Lower relevance threshold (0.25), more URLs, comprehensive processing
- **Balanced Mode**: Standard threshold (0.3), moderate URL count, optimized processing
- **Fast Mode**: Higher threshold (0.4), fewer URLs, minimal processing

### Resource Management
- **Memory Usage**: Monitor content size and implement streaming for large results
- **CPU Usage**: Configure appropriate concurrency limits based on available resources
- **Network Bandwidth**: Use connection pooling and rate limiting
- **API Limits**: Monitor external API usage and implement backoff strategies

## MCP Integration Patterns

### Direct Tool Usage
```python
# Tools can be used directly in Python code
from tools.intelligent_research_tool import intelligent_research_with_advanced_scraping
from tools.advanced_scraping_tool import advanced_scrape_url
from tools.serp_search_tool import serp_search

# Direct function calls
research_result = await intelligent_research_with_advanced_scraping({
    "query": "artificial intelligence trends",
    "session_id": "direct_usage_session"
})
```

### MCP Server Integration
```python
# Tools are automatically exposed through MCP when imported
from claude_agent_sdk import create_sdk_mcp_server
from tools.intelligent_research_tool import intelligent_research_with_advanced_scraping
from tools.advanced_scraping_tool import advanced_scrape_url, advanced_scrape_multiple_urls
from tools.serp_search_tool import serp_search

# Create MCP server with all tools
research_mcp_server = create_sdk_mcp_server(
    "research_tools",
    tools=[
        intelligent_research_with_advanced_scraping,
        advanced_scrape_url,
        advanced_scrape_multiple_urls,
        serp_search
    ]
)
```

### Agent Integration Patterns
```python
# Example: Research agent using tools
class MultiModalResearchAgent:
    def __init__(self, config: dict):
        self.config = config
        self.session_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    async def comprehensive_research(self, query: str, requirements: dict) -> dict:
        # Phase 1: Initial broad search
        search_result = await serp_search({
            "query": query,
            "search_type": "search",
            "num_results": requirements.get("search_results", 15),
            "session_id": self.session_id
        })

        if search_result.get("is_error"):
            return {"error": f"Search failed: {search_result['content'][0]['text']}"}

        # Phase 2: Deep research with intelligent analysis
        research_result = await intelligent_research_with_advanced_scraping({
            "query": query,
            "session_id": self.session_id,
            "max_urls": requirements.get("max_sources", 10),
            "relevance_threshold": requirements.get("relevance_threshold", 0.3),
            "max_concurrent": requirements.get("max_concurrent", 8)
        })

        return {
            "search_phase": search_result,
            "research_phase": research_result,
            "session_id": self.session_id,
            "requirements": requirements
        }

    async def targeted_scraping(self, urls: list[str], context: str) -> dict:
        # Scrape specific URLs with context awareness
        scraping_result = await advanced_scrape_multiple_urls({
            "urls": urls,
            "session_id": self.session_id,
            "search_query": context,
            "max_concurrent": min(5, len(urls)),
            "preserve_technical": True
        })

        return scraping_result
```

### Workflow Orchestration
```python
# Example: Research workflow orchestrator
class ResearchWorkflow:
    def __init__(self):
        self.session_counter = 0

    async def execute_research_workflow(self, query: str, workflow_config: dict) -> dict:
        self.session_counter += 1
        session_id = f"workflow_{self.session_counter}"

        workflow_steps = workflow_config.get("steps", ["search", "research", "scrape"])
        results = {}

        for step in workflow_steps:
            if step == "search":
                results["search"] = await serp_search({
                    "query": query,
                    "session_id": session_id,
                    **workflow_config.get("search_config", {})
                })

            elif step == "research":
                results["research"] = await intelligent_research_with_advanced_scraping({
                    "query": query,
                    "session_id": session_id,
                    **workflow_config.get("research_config", {})
                })

            elif step == "scrape" and "search" in results:
                # Extract URLs from search results for targeted scraping
                urls = self._extract_top_urls(results["search"], workflow_config.get("scrape_count", 5))
                results["scrape"] = await advanced_scrape_multiple_urls({
                    "urls": urls,
                    "session_id": session_id,
                    "search_query": query,
                    **workflow_config.get("scrape_config", {})
                })

        return {
            "workflow_results": results,
            "session_id": session_id,
            "workflow_config": workflow_config
        }
```

## Configuration & Environment Setup

### Required Environment Variables
```bash
# API Keys (Required)
SERP_API_KEY=your_serp_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional Configuration
DEFAULT_MAX_URLS=10
DEFAULT_RELEVANCE_THRESHOLD=0.3
DEFAULT_MAX_CONCURRENT=8
CLEANING_JUDGE_SCORE_THRESHOLD=0.6

# Development Settings
DEBUG_MODE=false
LOG_LEVEL=INFO
ENABLE_WORK_PRODUCTS=true
KEVIN_DIR=/path/to/kevin/directory
```

### Tool Configuration
```python
# tools_config.py
TOOLS_CONFIG = {
    "intelligent_research": {
        "default_max_urls": 10,
        "default_relevance_threshold": 0.3,
        "default_max_concurrent": 8,
        "compression_tokens": 20000,
        "enable_work_products": True
    },
    "advanced_scraping": {
        "default_timeout": 30,
        "preserve_technical_content": True,
        "default_extraction_mode": "article",
        "enable_judge_optimization": True
    },
    "serp_search": {
        "default_num_results": 15,
        "default_search_type": "search",
        "auto_crawl_top": 5,
        "default_crawl_threshold": 0.3
    }
}
```

## Best Practices & Guidelines

### Tool Usage Best Practices
1. **Session Management**: Use unique session IDs for tracking and debugging
2. **Parameter Tuning**: Adjust relevance thresholds and concurrency based on use case
3. **Error Handling**: Always check for `is_error` in tool responses
4. **Work Products**: Enable work products for complex research operations
5. **Resource Monitoring**: Monitor API usage and system resources

### Development Guidelines
1. **Follow MCP Standards**: Use proper tool decorators and response formats
2. **Implement Async Patterns**: All tool operations should be asynchronous
3. **Provide Clear Error Messages**: Include specific guidance for common issues
4. **Document Parameters**: Clearly document required and optional parameters
5. **Test Thoroughly**: Test both success and error scenarios

### Performance Optimization
1. **Use Relevance Filtering**: Avoid processing low-quality sources
2. **Configure Appropriate Concurrency**: Balance speed with resource usage
3. **Enable Judge Optimization**: Reduce latency in content cleaning
4. **Monitor Token Usage**: Stay within MCP token limits
5. **Implement Caching**: Cache frequently accessed data when appropriate

### Quality Assurance
1. **Validate Content Quality**: Use AI cleaning and relevance scoring
2. **Preserve Technical Content**: Maintain code blocks and technical information
3. **Cross-Reference Sources**: Use multiple sources for validation
4. **Implement Fallback Strategies**: Handle failures gracefully
5. **Monitor Success Rates**: Track and optimize tool performance

## Future Development & Extensions

### Potential Tool Enhancements
- **Multi-Language Support**: Extend tools to work with non-English content
- **Real-Time Data Integration**: Add support for real-time data sources
- **Custom Search Engines**: Support for specialized search engines
- **Advanced Content Analysis**: Deeper content understanding and extraction
- **Distributed Processing**: Support for distributed tool execution

### Extension Points
- **Custom Tool Development**: Framework for adding new research tools
- **Plugin Architecture**: Support for third-party tool integrations
- **Configuration Presets**: Pre-configured tool combinations for specific use cases
- **Performance Profiles**: Optimized settings for different performance requirements

### Integration Opportunities
- **External Research APIs**: Integration with academic and professional databases
- **Collaborative Research**: Multi-user research session support
- **Real-Time Collaboration**: Live research sharing and collaboration
- **Advanced Analytics**: Research trend analysis and reporting