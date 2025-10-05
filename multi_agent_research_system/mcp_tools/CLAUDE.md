# MCP Tools Directory - Multi-Agent Research System

This directory contains Model Context Protocol (MCP) implementations that enable seamless integration between the multi-agent research system and Claude AI models.

## Directory Purpose

The mcp_tools directory provides MCP-compliant tool implementations that expose the research system's capabilities to Claude agents through standardized interfaces. These tools enable Claude agents to perform web research, content extraction, and data analysis using the system's sophisticated infrastructure.

## Key Components

### Core MCP Implementations
- **`enhanced_search_scrape_clean.py`** - Integrated search, scraping, and content cleaning pipeline
- **`zplayground1_search.py`** - Specialized search implementation with advanced features
- **`mcp_compliance_manager.py`** - MCP protocol compliance and token management

## Tool Capabilities

### Enhanced Search Scrape Clean Tool
The `enhanced_search_scrape_clean.py` provides:
- **Integrated Pipeline**: Combines search, scraping, and content cleaning in one tool
- **Multi-Source Search**: Searches across multiple data sources and search engines
- **Intelligent Scraping**: Uses advanced anti-bot strategies and content extraction
- **Content Cleaning**: Applies sophisticated content cleaning and standardization
- **MCP Compliance**: Handles MCP protocol requirements and token limits

### ZPlayground1 Search Tool
The `zplayground1_search.py` offers:
- **Specialized Search**: Optimized search implementation for specific use cases
- **Quality Assurance**: Built-in quality assessment and filtering
- **Performance Optimization**: Optimized for speed and reliability
- **Error Handling**: Robust error handling and recovery mechanisms

### MCP Compliance Manager
The `mcp_compliance_manager.py` handles:
- **Token Management**: Manages MCP token limits and allocation
- **Protocol Compliance**: Ensures MCP protocol requirements are met
- **Content Optimization**: Optimizes content for MCP constraints
- **Error Reporting**: Provides detailed error reporting for debugging

## MCP Integration Architecture

### MCP Tool Structure
```python
# Standard MCP tool pattern
@mcp_tool("tool_name", "Tool description", {
    "parameter1": "type",
    "parameter2": "type"
})
async def tool_function(args: dict) -> dict:
    # Tool implementation
    result = await process_request(args)
    return format_for_mcp(result)
```

### Tool Registration Flow
```
Tool Definition → MCP Registration → Claude Discovery → Tool Invocation → Result Processing
```

### Content Processing Pipeline
```
User Input → Tool Invocation → Research Processing → Content Optimization → MCP Formatting → Claude Response
```

## Development Guidelines

### MCP Tool Development Standards
1. **Standard Interface**: Follow MCP tool interface patterns consistently
2. **Error Handling**: Implement comprehensive error handling with MCP-compliant error responses
3. **Token Management**: Monitor and respect MCP token limits
4. **Content Optimization**: Optimize content for Claude's context window

### Tool Definition Patterns
```python
# Example: MCP tool definition
from mcp import tool

@tool(
    name="search_and_extract",
    description="Search for information and extract relevant content",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer", "description": "Maximum results to return", "default": 10},
        "content_type": {"type": "string", "description": "Type of content to extract", "default": "text"}
    }
)
async def search_and_extract(args: dict) -> dict:
    query = args.get("query")
    max_results = args.get("max_results", 10)
    content_type = args.get("content_type", "text")

    try:
        results = await perform_search(query, max_results)
        cleaned_results = clean_content(results, content_type)
        return {"success": True, "results": cleaned_results}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### Token Management Strategies
```python
# Example: Token-aware content processing
def optimize_for_mcp(content: str, max_tokens: int = 8000) -> str:
    if estimate_tokens(content) <= max_tokens:
        return content

    # Implement smart truncation
    return truncate_intelligently(content, max_tokens)

def estimate_tokens(text: str) -> int:
    # Rough estimation: ~1 token per 4 characters
    return len(text) // 4
```

## Testing & Debugging

### MCP Testing Strategies
1. **Protocol Compliance**: Verify tools follow MCP standards
2. **Token Management**: Test token limit handling
3. **Error Scenarios**: Test various error conditions and recovery
4. **Performance Testing**: Ensure tools perform within acceptable limits

### Debugging MCP Tools
1. **Verbose Logging**: Enable detailed logging for MCP interactions
2. **Token Tracking**: Monitor token usage and optimization
3. **Error Analysis**: Analyze error patterns and optimize handling
4. **Performance Monitoring**: Track tool execution times and resource usage

### Common MCP Issues & Solutions
- **Token Limit Exceeded**: Implement better content optimization and truncation
- **Protocol Violations**: Ensure proper MCP message formatting
- **Slow Response Times**: Optimize tool algorithms and caching
- **Poor Tool Discovery**: Verify tool registration and metadata

## Dependencies & Interactions

### MCP Dependencies
- **mcp Python package**: Core MCP protocol implementation
- **claude-agent-sdk**: Claude Agent SDK integration
- **anthropic**: Anthropic API client for Claude integration

### Internal Dependencies
- **Utils Layer**: MCP tools use utilities for core functionality
- **Tools Layer**: Higher-level tools exposed through MCP interfaces
- **Core System**: Orchestrator manages MCP server lifecycle
- **Agent System**: Claude agents use MCP tools for research

### Data Flow
```
Claude Agent → MCP Tool Invocation → Research Processing → Content Optimization → MCP Response → Claude Agent
```

## Usage Examples

### Basic MCP Tool Usage
```python
# Example: Using MCP tools from Claude
"Please search for information about renewable energy trends in 2024"

# Claude would invoke:
search_and_extract({
    "query": "renewable energy trends 2024",
    "max_results": 10,
    "content_type": "text"
})
```

### Advanced Search with Filtering
```python
# Example: Complex search with parameters
"Find recent academic papers about quantum computing applications"

# Claude would invoke:
enhanced_search_scrape_clean({
    "query": "quantum computing applications",
    "search_mode": "academic",
    "anti_bot_level": 1,
    "content_filter": "scholarly",
    "max_urls": 15
})
```

### Content Analysis and Extraction
```python
# Example: Content analysis request
"Analyze the content from these URLs and extract key insights"

# Claude would invoke:
zplayground1_search({
    "urls": ["https://example1.com", "https://example2.com"],
    "analysis_type": "key_insights",
    "content_summary": True
})
```

## Performance Optimization

### MCP-Specific Optimizations
1. **Token Efficiency**: Optimize content to maximize information density
2. **Caching**: Cache frequently requested results to improve response times
3. **Batch Processing**: Process multiple requests efficiently when possible
4. **Smart Filtering**: Filter and prioritize content before returning to Claude

### Scaling Considerations
- Implement connection pooling for web requests
- Use intelligent caching strategies
- Monitor and optimize token usage patterns
- Implement graceful degradation under load

### Quality Assurance
- Validate tool outputs before returning to Claude
- Implement quality scoring for search results
- Provide confidence metrics where appropriate
- Use multiple sources for cross-validation

## MCP Server Configuration

### Server Setup
```python
# Example: MCP server configuration
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("research-tools")

# Register tools
@app.tool("search_web")
async def search_web(args: dict) -> dict:
    # Tool implementation
    pass

@app.tool("extract_content")
async def extract_content(args: dict) -> dict:
    # Tool implementation
    pass

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

### Tool Discovery and Metadata
```python
# Example: Tool metadata for Claude discovery
TOOLS_METADATA = {
    "search_web": {
        "description": "Search the web for information",
        "parameters": {
            "query": {"type": "string", "required": True},
            "max_results": {"type": "integer", "default": 10}
        },
        "examples": [
            {"query": "artificial intelligence trends", "max_results": 5}
        ]
    }
}
```

## Integration with Claude Agent SDK

### SDK Integration Pattern
```python
# Example: Integration with Claude Agent SDK
from claude_agent_sdk import create_sdk_mcp_server

# Create MCP server with tools
server = create_sdk_mcp_server(
    tools=[search_and_extract, enhanced_search_scrape_clean],
    name="research-tools",
    description="Multi-agent research system tools"
)

# Server can now be used with Claude agents
```

### Agent Tool Usage
```python
# Example: Agent using MCP tools
class ResearchAgent:
    def __init__(self):
        self.mcp_client = create_mcp_client("research-tools")

    async def research_topic(self, topic: str):
        results = await self.mcp_client.call_tool(
            "search_and_extract",
            {"query": topic, "max_results": 10}
        )
        return self.process_results(results)
```