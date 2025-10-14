# MCP Tools Directory - Multi-Agent Research System

This directory contains Model Context Protocol (MCP) implementations that provide seamless integration between the multi-agent research system and Claude AI models through the Claude Agent SDK, enabling sophisticated research capabilities with intelligent token management and content optimization.

## Directory Purpose

The mcp_tools directory provides production-ready MCP-compliant tool implementations that expose the research system's advanced capabilities to Claude agents through standardized interfaces. These tools enable Claude agents to perform comprehensive web research, intelligent content extraction, AI-powered content cleaning, and sophisticated data analysis while maintaining strict token efficiency and MCP protocol compliance. The implementation focuses on enterprise-grade reliability, intelligent content allocation, and seamless integration with the Claude Agent SDK.

## Key Components

### Advanced Search & Content Extraction Tools
- **`enhanced_search_scrape_clean.py`** - Production-grade multi-tool MCP server with three distinct search capabilities (enhanced search, news search, expanded query search), featuring adaptive content chunking, comprehensive error handling, and intelligent token management with automatic workproduct tracking
- **`zplayground1_search.py`** - Single comprehensive MCP tool implementing the complete zPlayground1 topic-based search, scrape, and clean workflow with exact implementation fidelity, progressive anti-bot detection (levels 0-3), AI content cleaning via GPT-5-nano, and fail-fast parameter validation

### Enhanced Editorial Workflow MCP Tools (NEW in v3.2)
- **`enhanced_editorial_mcp_tools.py`** - Advanced editorial workflow MCP server with comprehensive editorial decision engine, gap research coordination, and evidence-based recommendations with intelligent token management and session coordination
- **`editorial_analysis_mcp_tools.py`** - Editorial analysis MCP tools for quality assessment, coverage analysis, and gap identification with multi-dimensional confidence scoring
- **`gap_research_mcp_tools.py`** - Gap research coordination MCP tools for sub-session management, resource allocation, and result integration with parent-child session coordination
- **`editorial_recommendations_mcp_tools.py`** - Evidence-based editorial recommendations MCP tools with ROI estimation, implementation planning, and priority-based ranking
- **`quality_enhancement_mcp_tools.py`** - Quality enhancement MCP tools for multi-dimensional quality assessment, progressive enhancement, and quality metrics tracking
- **`editorial_workflow_orchestrator_mcp.py`** - Complete editorial workflow orchestration MCP server for end-to-end editorial process management with tool coordination and quality gate management

### MCP Compliance & Token Management
- **`mcp_compliance_manager.py`** - Sophisticated MCP compliance system with multi-level content allocation (70/30 split), intelligent compression with quality preservation, priority-based content distribution, comprehensive metadata generation, and detailed token usage analytics with performance tracking

### Package Management
- **`__init__.py`** - Package initialization with version management and clean import structure

## Architecture & Implementation

### Multi-Server Architecture Pattern

The mcp_tools directory implements a sophisticated multi-server architecture with distinct capabilities:

```python
# Enhanced Search Server - Multi-Tool Approach
enhanced_search_server = create_enhanced_search_mcp_server()
# Tools: enhanced_search_scrape_clean, enhanced_news_search, expanded_query_search_and_extract

# ZPlayground1 Server - Single Comprehensive Tool
zplayground1_server = create_zplayground1_mcp_server()
# Tool: zplayground1_search_scrape_clean (complete workflow in one call)

# Enhanced Editorial Workflow Servers (NEW in v3.2)
enhanced_editorial_server = create_enhanced_editorial_mcp_server()
# Tools: enhanced_editorial_analysis, gap_research_decision, editorial_recommendations

editorial_analysis_server = create_editorial_analysis_mcp_server()
# Tools: comprehensive_editorial_analysis, quality_assessment, gap_identification

gap_research_server = create_gap_research_mcp_server()
# Tools: coordinate_gap_research, manage_sub_sessions, integrate_gap_results

editorial_recommendations_server = create_editorial_recommendations_mcp_server()
# Tools: generate_editorial_recommendations, estimate_roi, plan_implementation

quality_enhancement_server = create_quality_enhancement_mcp_server()
# Tools: enhance_content_quality, assess_quality_dimensions, plan_quality_improvements

editorial_orchestrator_server = create_editorial_workflow_orchestrator_mcp_server()
# Tools: orchestrate_editorial_workflow, manage_quality_gates, coordinate_editorial_tools
```

### MCP Tool Implementation Patterns

#### Enhanced Search Multi-Tool Pattern
```python
@tool(
    "enhanced_search_scrape_clean",
    "Advanced topic-based search with parallel crawling and AI content cleaning...",
    {
        "query": {"type": "string", "description": "Search query or topic to research"},
        "search_type": {"type": "string", "enum": ["search", "news"], "default": "search"},
        "num_results": {"type": "integer", "default": 15, "minimum": 1, "maximum": 50},
        "auto_crawl_top": {"type": "integer", "default": 10, "minimum": 0, "maximum": 20},
        "anti_bot_level": {"type": "integer", "default": 1, "minimum": 0, "maximum": 3},
        "max_concurrent": {"type": "integer", "default": 15, "minimum": 1, "maximum": 20},
        "session_id": {"type": "string", "default": "default"},
        "workproduct_prefix": {"type": "string", "default": ""}
    }
)
async def enhanced_search_scrape_clean(args: dict[str, Any]) -> dict[str, Any]:
    """Enhanced search with parallel crawling and adaptive chunking."""

    # Execute search with comprehensive error handling
    result = await search_crawl_and_clean_direct(
        query=args["query"],
        search_type=args["search_type"],
        num_results=args["num_results"],
        auto_crawl_top=args["auto_crawl_top"],
        anti_bot_level=args["anti_bot_level"],
        session_id=args["session_id"],
        workproduct_dir=workproduct_dir
    )

    # Intelligent token management with adaptive chunking
    if len(result) > 20000:
        content_blocks = create_adaptive_chunks(result, query)
        return {"content": content_blocks, "metadata": {..., "chunked_content": True}}
    else:
        return {"content": [{"type": "text", "text": result}], "metadata": {...}}
```

#### ZPlayground1 Single-Tool Pattern
```python
@tool(
    "zplayground1_search_scrape_clean",
    "Complete zPlayground1 topic-based search, scrape, and clean functionality...",
    {
        "query": {"type": "string", "description": "Search query, topic, or news topic"},
        "search_mode": {"type": "string", "enum": ["web", "news"], "default": "web"},
        "num_results": {"type": "integer", "default": 15, "minimum": 1, "maximum": 50},
        "anti_bot_level": {"type": "integer", "default": 1, "minimum": 0, "maximum": 3},
        "session_id": {"type": "string", "default": "default"},
        # ... comprehensive parameter set
    }
)
async def zplayground1_search_scrape_clean(args: dict[str, Any]) -> dict[str, Any]:
    """Complete zPlayground1 workflow with fail-fast validation and MCP compliance."""

    # FAIL-FAST parameter validation with detailed error reporting
    query = args.get("query")
    if not query:
        return {"content": [{"type": "text", "text": "❌ CRITICAL ERROR: 'query' parameter required"}], "is_error": True}

    # Execute exact zPlayground1 implementation
    result = await search_crawl_and_clean_direct(...)

    # Apply MCP compliance with intelligent content allocation
    allocation = mcp_manager.allocate_content(result, metadata, context)

    return {
        "content": [{"type": "text", "text": allocation.primary_content + allocation.metadata_content}],
        "metadata": {..., "mcp_compliance": True, "token_usage": allocation.token_usage}
    }
```

#### Enhanced Editorial Workflow Multi-Tool Pattern (NEW in v3.2)
```python
@tool(
    "enhanced_editorial_analysis",
    "Advanced editorial analysis with multi-dimensional confidence scoring and gap research decisions...",
    {
        "report_content": {"type": "string", "description": "First draft report content for editorial analysis"},
        "research_corpus": {"type": "object", "description": "Research corpus data including sources and metadata"},
        "session_id": {"type": "string", "default": "default", "description": "Session identifier for tracking"},
        "analysis_depth": {"type": "string", "enum": ["standard", "comprehensive", "deep"], "default": "comprehensive"},
        "confidence_threshold": {"type": "number", "default": 0.7, "minimum": 0.5, "maximum": 0.9},
        "max_gap_topics": {"type": "integer", "default": 2, "minimum": 1, "maximum": 5}
    }
)
async def enhanced_editorial_analysis(args: dict[str, Any]) -> dict[str, Any]:
    """Enhanced editorial analysis with multi-dimensional confidence scoring."""

    # Parameter validation with detailed error reporting
    report_content = args.get("report_content")
    if not report_content:
        return {"content": [{"type": "text", "text": "❌ ERROR: 'report_content' parameter required"}], "is_error": True}

    session_id = args.get("session_id", "default")
    analysis_depth = args.get("analysis_depth", "comprehensive")
    confidence_threshold = args.get("confidence_threshold", 0.7)

    try:
        # Execute enhanced editorial analysis
        decision_engine = EnhancedEditorialDecisionEngine()
        corpus_analyzer = ResearchCorpusAnalyzer()

        # Analyze research corpus
        corpus_analysis = await corpus_analyzer.analyze_research_corpus(args["research_corpus"])

        # Assess editorial decisions with confidence scoring
        editorial_decision = await decision_engine.assess_editorial_decision(
            report_content, args["research_corpus"]
        )

        # Apply MCP compliance for editorial results
        mcp_manager = get_mcp_compliance_manager()
        allocation = mcp_manager.allocate_content_for_editorial(
            editorial_decision, corpus_analysis, {"analysis_depth": analysis_depth}
        )

        return {
            "content": [{"type": "text", "text": allocation.primary_content + allocation.metadata_content}],
            "metadata": {
                "session_id": session_id,
                "analysis_depth": analysis_depth,
                "confidence_threshold": confidence_threshold,
                "gap_research_decision": editorial_decision.gap_research_decision.should_execute,
                "confidence_scores": editorial_decision.confidence_scores,
                "mcp_compliance": True,
                "token_usage": allocation.token_usage
            }
        }

    except Exception as e:
        error_msg = f"❌ **Enhanced Editorial Analysis Error**: {str(e)}"
        return {"content": [{"type": "text", "text": error_msg}], "is_error": True}

@tool(
    "coordinate_gap_research",
    "Gap research coordination with sub-session management and result integration...",
    {
        "gap_topics": {"type": "array", "description": "List of gap topics for research", "items": {"type": "string"}},
        "parent_session_id": {"type": "string", "description": "Parent session ID for coordination"},
        "max_concurrent_sub_sessions": {"type": "integer", "default": 3, "minimum": 1, "maximum": 5},
        "session_id": {"type": "string", "default": "default"},
        "integration_quality_threshold": {"type": "number", "default": 0.7, "minimum": 0.5, "maximum": 0.9}
    }
)
async def coordinate_gap_research(args: dict[str, Any]) -> dict[str, Any]:
    """Gap research coordination with sub-session management."""

    # Parameter validation
    gap_topics = args.get("gap_topics", [])
    if not gap_topics:
        return {"content": [{"type": "text", "text": "❌ ERROR: 'gap_topics' parameter required with at least one topic"}], "is_error": True}

    parent_session_id = args.get("parent_session_id")
    if not parent_session_id:
        return {"content": [{"type": "text", "text": "❌ ERROR: 'parent_session_id' parameter required"}], "is_error": True}

    session_id = args.get("session_id", "default")
    max_concurrent = args.get("max_concurrent_sub_sessions", 3)

    try:
        # Initialize gap research coordinator
        coordinator = GapResearchCoordinator()

        # Coordinate gap research through sub-sessions
        gap_results = await coordinator.coordinate_gap_research(
            gap_topics, parent_session_id, max_concurrent
        )

        # Apply MCP compliance for gap research results
        mcp_manager = get_mcp_compliance_manager()
        allocation = mcp_manager.allocate_content_for_gap_research(
            gap_results, {"max_concurrent": max_concurrent}
        )

        return {
            "content": [{"type": "text", "text": allocation.primary_content + allocation.metadata_content}],
            "metadata": {
                "session_id": session_id,
                "parent_session_id": parent_session_id,
                "total_gap_topics": len(gap_topics),
                "successful_researches": gap_results.successful_researches,
                "integrated_analysis": gap_results.integrated_analysis,
                "mcp_compliance": True,
                "token_usage": allocation.token_usage
            }
        }

    except Exception as e:
        error_msg = f"❌ **Gap Research Coordination Error**: {str(e)}"
        return {"content": [{"type": "text", "text": error_msg}], "is_error": True}

@tool(
    "generate_editorial_recommendations",
    "Evidence-based editorial recommendations with ROI estimation and implementation planning...",
    {
        "report_content": {"type": "string", "description": "Report content for recommendations"},
        "editorial_analysis": {"type": "object", "description": "Editorial analysis results"},
        "gap_results": {"type": "object", "description": "Gap research results (optional)"},
        "session_id": {"type": "string", "default": "default"},
        "include_roi_analysis": {"type": "boolean", "default": True},
        "priority_threshold": {"type": "number", "default": 0.8, "minimum": 0.5, "maximum": 1.0}
    }
)
async def generate_editorial_recommendations(args: dict[str, Any]) -> dict[str, Any]:
    """Generate evidence-based editorial recommendations with ROI estimation."""

    # Parameter validation
    report_content = args.get("report_content")
    if not report_content:
        return {"content": [{"type": "text", "text": "❌ ERROR: 'report_content' parameter required"}], "is_error": True}

    editorial_analysis = args.get("editorial_analysis")
    if not editorial_analysis:
        return {"content": [{"type": "text", "text": "❌ ERROR: 'editorial_analysis' parameter required"}], "is_error": True}

    session_id = args.get("session_id", "default")
    include_roi = args.get("include_roi_analysis", True)

    try:
        # Initialize recommendations engine
        recommendations_engine = EditorialRecommendationsEngine()

        # Generate evidence-based recommendations
        recommendations = await recommendations_engine.generate_editorial_recommendations(
            report_content, editorial_analysis, args.get("gap_results")
        )

        # Apply MCP compliance for recommendations
        mcp_manager = get_mcp_compliance_manager()
        allocation = mcp_manager.allocate_content_for_recommendations(
            recommendations, {"include_roi": include_roi}
        )

        return {
            "content": [{"type": "text", "text": allocation.primary_content + allocation.metadata_content}],
            "metadata": {
                "session_id": session_id,
                "total_recommendations": recommendations.total_recommendations,
                "high_priority_count": recommendations.high_priority_count,
                "estimated_quality_improvement": recommendations.estimated_quality_improvement,
                "roi_analysis_included": include_roi,
                "mcp_compliance": True,
                "token_usage": allocation.token_usage
            }
        }

    except Exception as e:
        error_msg = f"❌ **Editorial Recommendations Error**: {str(e)}"
        return {"content": [{"type": "text", "text": error_msg}], "is_error": True}
```

### Adaptive Content Chunking System

The enhanced search server implements intelligent content chunking for token management:

```python
def create_adaptive_chunks(content: str, query: str, max_chunk_size: int = 18000) -> list[dict[str, Any]]:
    """Create adaptive chunks for large content to avoid token limits."""

    content_length = len(content)
    if content_length <= max_chunk_size:
        return [{"type": "text", "text": content}]

    # Intelligent chunking with logical break points
    content_blocks = []
    lines = content.split("\n")
    current_chunk = ""
    chunk_number = 1
    total_chunks = (len(content) // max_chunk_size) + 1

    for line in lines:
        test_chunk = current_chunk + line + "\n"
        if len(test_chunk) > max_chunk_size:
            # Finalize current chunk with continuation indicator
            content_blocks.append({"type": "text", "text": current_chunk.rstrip()})
            chunk_number += 1
            current_chunk = f"# Search Results - Part {chunk_number} of {total_chunks}\n\n" + line + "\n"
        else:
            current_chunk += line + "\n"

    # Add completion metadata to chunks
    for i in range(len(content_blocks) - 1):
        content_blocks[i]["text"] += f"\n\n---\n*Part {i + 1} of {len(content_blocks)} - Continued in next part*"

    if content_blocks:
        content_blocks[-1]["text"] += f"\n\n---\n*Complete search results for query: '{query}'*"

    return content_blocks
```

### MCP Compliance Management

The compliance manager provides sophisticated content allocation and token optimization:

```python
class MCPComplianceManager:
    """Advanced MCP compliance with multi-level content allocation."""

    def __init__(self, max_tokens: int = 25000):
        self.max_tokens = max_tokens
        self.primary_content_ratio = 0.7  # 70% for cleaned content
        self.metadata_ratio = 0.3         # 30% for metadata

    def allocate_content(self, raw_content: str, metadata: dict, context: dict) -> ContentAllocation:
        """Allocate content according to MCP compliance standards."""

        # Analyze and prioritize content
        content_analysis = self._analyze_content(raw_content, context)

        # Allocate primary content based on priority
        primary_content = self._allocate_primary_content(
            raw_content, content_analysis, primary_tokens
        )

        # Generate enhanced metadata
        metadata_content = self._generate_enhanced_metadata(
            metadata, content_analysis, context, metadata_tokens
        )

        return ContentAllocation(
            primary_content=primary_content,
            metadata_content=metadata_content,
            token_usage=self._estimate_token_usage(primary_content, metadata_content),
            priority_distribution=content_analysis['priority_distribution']
        )
```

## Advanced Features

### Progressive Anti-Bot Detection Integration

Both MCP servers implement the sophisticated 4-level anti-bot escalation system:

```python
# Anti-Bot Level Configuration
ANTI_BOT_LEVELS = {
    0: {"name": "basic", "description": "Standard crawling with basic headers"},
    1: {"name": "enhanced", "description": "Enhanced headers and request timing"},
    2: {"name": "advanced", "description": "Advanced stealth techniques and proxy rotation"},
    3: {"name": "stealth", "description": "Maximum stealth with advanced evasion"}
}

# Tool parameter validation includes anti-bot level validation
anti_bot_level = int(args.get("anti_bot_level", 1))
if not (0 <= anti_bot_level <= 3):
    raise ValueError(f"Invalid anti_bot_level '{anti_bot_level}'. Must be between 0 and 3")
```

### Session-Based Workproduct Management

Comprehensive workproduct tracking with session-based organization:

```python
# Environment-aware workproduct directory setup
current_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "claudeagent-multiagent-latest" in current_repo:
    base_session_dir = f"{current_repo}/KEVIN/sessions/{session_id}"
else:
    base_session_dir = f"/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions/{session_id}"

workproduct_dir = f"{base_session_dir}/research"
Path(workproduct_dir).mkdir(parents=True, exist_ok=True)
```

### Comprehensive Error Handling & Recovery

Sophisticated error handling with fail-fast validation and detailed error reporting:

```python
# Enhanced Search Error Handling
try:
    result = await search_crawl_and_clean_direct(...)
except Exception as e:
    error_msg = f"""❌ **Enhanced Search Error**

Failed to execute search and content extraction: {str(e)}

Please check:
- SERP_API_KEY is configured
- Network connectivity
- Query parameters are valid
"""
    return {"content": [{"type": "text", "text": error_msg}], "is_error": True}

# ZPlayground1 Fail-Fast Error Handling
try:
    # Validate parameters with detailed error reporting
    anti_bot_level_raw = args.get("anti_bot_level", 1)
    if isinstance(anti_bot_level_raw, str) and anti_bot_level_raw.isdigit():
        anti_bot_level = int(anti_bot_level_raw)
        logger.info(f"🔄 Converted numeric string '{anti_bot_level_raw}' to integer {anti_bot_level}")
    # ... comprehensive validation
except ValueError as param_error:
    error_msg = f"❌ **CRITICAL PARAMETER VALIDATION ERROR**: {param_error}"
    return {"content": [{"type": "text", "text": error_msg}], "is_error": True}
```

### Performance Monitoring & Statistics

Comprehensive performance tracking for optimization and debugging:

```python
# Chunking Statistics Tracking
chunking_stats = {
    "total_calls": 0,
    "chunking_triggered": 0,
    "total_content_chars": 0,
    "total_chunks_created": 0,
}

def get_chunking_stats() -> dict[str, Any]:
    """Get comprehensive chunking performance statistics."""
    chunking_rate = (chunking_stats["chunking_triggered"] / chunking_stats["total_calls"]) * 100
    return {
        "total_calls": chunking_stats["total_calls"],
        "chunking_rate_percent": round(chunking_rate, 1),
        "total_content_processed": f"{chunking_stats['total_content_chars']:,} chars",
        "average_chunks_when_chunking": round(
            chunking_stats["total_chunks_created"] / max(chunking_stats["chunking_triggered"], 1), 1
        )
    }
```

## Tool Capabilities & Usage Patterns

### Enhanced Search Server Tools

#### 1. Enhanced Search Scrape Clean (`enhanced_search_scrape_clean`)
- **Purpose**: Advanced topic-based search with parallel crawling and AI content cleaning
- **Features**: Progressive anti-bot detection, adaptive content chunking, workproduct tracking
- **Use Cases**: Comprehensive research, content analysis, competitive intelligence
- **Token Management**: Intelligent chunking for large content (>20,000 chars)

#### 2. Enhanced News Search (`enhanced_news_search`)
- **Purpose**: Specialized news search with enhanced content extraction
- **Features**: News-focused query optimization, article content extraction, temporal relevance
- **Use Cases**: Current events research, news analysis, trend monitoring
- **Token Management**: Same adaptive chunking as enhanced search

#### 3. Expanded Query Search (`expanded_query_search_and_extract`)
- **Purpose**: Query expansion workflow with master result consolidation
- **Features**: Multi-query generation, result deduplication, relevance ranking, budget control
- **Use Cases**: Comprehensive coverage research, exploratory analysis, gap discovery
- **Token Management**: Consolidated results with intelligent allocation

### ZPlayground1 Server Tool

#### ZPlayground1 Search Scrape Clean (`zplayground1_search_scrape_clean`)
- **Purpose**: Complete zPlayground1 workflow in single tool call
- **Features**: Exact implementation fidelity, fail-fast validation, MCP compliance integration
- **Use Cases**: Production deployment, simplified integration, maximum reliability
- **Token Management**: 70/30 content/metadata split with priority-based allocation

### Enhanced Editorial Workflow MCP Tools (NEW in v3.2)

#### 1. Enhanced Editorial Analysis (`enhanced_editorial_analysis`)
- **Purpose**: Advanced editorial analysis with multi-dimensional confidence scoring and gap research decisions
- **Features**: Multi-dimensional confidence scoring, gap research decision logic, cost-benefit analysis
- **Use Cases**: Editorial decision making, gap research assessment, quality evaluation
- **Token Management**: Intelligent allocation for editorial analysis results and confidence scores

#### 2. Gap Research Coordination (`coordinate_gap_research`)
- **Purpose**: Gap research coordination with sub-session management and result integration
- **Features**: Sub-session creation and management, resource allocation, result integration
- **Use Cases**: Gap research execution, sub-session coordination, result integration
- **Token Management**: Efficient allocation for gap research results and integration analysis

#### 3. Editorial Recommendations (`generate_editorial_recommendations`)
- **Purpose**: Evidence-based editorial recommendations with ROI estimation and implementation planning
- **Features**: ROI estimation, implementation planning, priority-based ranking
- **Use Cases**: Editorial improvement planning, quality enhancement, implementation guidance
- **Token Management**: Optimized allocation for recommendations and implementation plans

#### 4. Quality Assessment (`assess_quality_dimensions`)
- **Purpose**: Multi-dimensional quality assessment with progressive enhancement
- **Features**: 8+ dimension quality assessment, progressive enhancement, quality metrics
- **Use Cases**: Quality evaluation, improvement planning, metrics tracking
- **Token Management**: Efficient allocation for quality assessment results and recommendations

#### 5. Research Corpus Analysis (`analyze_research_corpus`)
- **Purpose**: Comprehensive research corpus analysis for quality and coverage assessment
- **Features**: Quality assessment, coverage analysis, gap identification, source diversity
- **Use Cases**: Corpus evaluation, coverage analysis, gap identification
- **Token Management**: Intelligent allocation for corpus analysis results and metrics

#### 6. Editorial Workflow Orchestration (`orchestrate_editorial_workflow`)
- **Purpose**: Complete editorial workflow orchestration with tool coordination and quality gate management
- **Features**: End-to-end orchestration, tool coordination, quality gate management
- **Use Cases**: Complete editorial workflow management, tool coordination, quality assurance
- **Token Management**: Optimized allocation for workflow results and coordination data

## Integration with Claude Agent SDK

### Server Registration Pattern
```python
from claude_agent_sdk import create_sdk_mcp_server, tool

# Enhanced Search Server Registration
enhanced_search_server = create_sdk_mcp_server(
    name="enhanced_search_scrape_clean",
    version="1.0.0",
    tools=[enhanced_search_scrape_clean, enhanced_news_search, expanded_query_search_and_extract_tool]
)

# ZPlayground1 Server Registration
zplayground1_server = create_sdk_mcp_server(
    name="zplayground1_search_scrape_clean",
    version="1.0.0",
    tools=[zplayground1_search_scrape_clean]
)
```

### Agent Integration Pattern
```python
# Using MCP tools in agent definitions
from claude_agent_sdk import AgentDefinition

research_agent = AgentDefinition(
    name="research_agent",
    model="claude-3-5-sonnet-20241022",
    instructions="You are a research agent with access to advanced search and content extraction tools.",
    tools=["enhanced_search_scrape_clean", "enhanced_news_search"],
    mcp_servers={
        "enhanced_search": enhanced_search_server,
        "zplayground1": zplayground1_server
    }
)
```

## Development Guidelines

### MCP Tool Development Standards

1. **Claude Agent SDK Integration**: Use `create_sdk_mcp_server` and `@tool` decorators consistently
2. **Comprehensive Parameter Validation**: Implement fail-fast validation with detailed error messages
3. **Intelligent Token Management**: Use adaptive chunking and content allocation strategies
4. **Session-Based Organization**: Use session IDs for workproduct tracking and state management
5. **Error Resilience**: Implement comprehensive error handling with informative error messages

### Code Quality Standards
```python
# Standard MCP Tool Implementation Pattern
@tool(
    "tool_name",
    "Comprehensive tool description with clear usage guidelines and examples",
    {
        "required_param": {"type": "string", "description": "Clear parameter description"},
        "optional_param": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50}
    }
)
async def tool_function(args: dict[str, Any]) -> dict[str, Any]:
    """Tool function with comprehensive docstring."""

    # 1. Parameter validation with fail-fast errors
    required_param = args.get("required_param")
    if not required_param:
        return {
            "content": [{"type": "text", "text": "❌ Error: required_param is required"}],
            "is_error": True
        }

    # 2. Environment-aware session setup
    session_id = args.get("session_id", "default")
    workproduct_dir = _setup_workproduct_directory(session_id)

    try:
        # 3. Core functionality execution
        result = await core_functionality(required_param, args)

        # 4. Token management and content optimization
        if len(result) > TOKEN_LIMIT:
            content_blocks = create_adaptive_chunks(result, required_param)
            return {"content": content_blocks, "metadata": {..., "chunked_content": True}}
        else:
            return {"content": [{"type": "text", "text": result}], "metadata": {...}}

    except Exception as e:
        # 5. Comprehensive error handling
        error_msg = f"❌ **Tool Error**: Failed to execute {str(e)}"
        logger.error(f"Tool execution failed: {e}")
        return {"content": [{"type": "text", "text": error_msg}], "is_error": True}
```

### Token Management Best Practices

```python
# Intelligent Content Allocation
def optimize_content_for_mcp(content: str, context: dict) -> dict:
    """Optimize content for MCP token limits with intelligent allocation."""

    if len(content) <= SAFE_TOKEN_LIMIT:
        return {"content": content, "chunked": False}

    # Use adaptive chunking
    chunks = create_adaptive_chunks(content, context["query"])
    return {"content": chunks, "chunked": True, "total_chunks": len(chunks)}

# MCP Compliance Integration
def apply_mcp_compliance(content: str, metadata: dict, context: dict) -> dict:
    """Apply MCP compliance with content allocation."""

    mcp_manager = get_mcp_compliance_manager()
    allocation = mcp_manager.allocate_content(content, metadata, context)

    return {
        "primary_content": allocation.primary_content,
        "metadata_content": allocation.metadata_content,
        "token_usage": allocation.token_usage,
        "compression_applied": allocation.compression_applied
    }
```

## Testing & Debugging

### MCP Tool Testing Strategy

1. **Protocol Compliance Testing**: Verify tools follow MCP standards and Claude SDK patterns
2. **Parameter Validation Testing**: Test fail-fast validation with comprehensive edge cases
3. **Token Management Testing**: Verify chunking behavior and content allocation efficiency
4. **Error Recovery Testing**: Test various error scenarios and recovery mechanisms
5. **Integration Testing**: Test tool integration with Claude Agent SDK and agent workflows

### Debugging MCP Tools

```python
# Comprehensive Debugging Setup
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for MCP interactions
logger = logging.getLogger("mcp_tools")

# Performance tracking
performance_stats = {
    "tool_calls": 0,
    "chunking_events": 0,
    "error_events": 0,
    "avg_execution_time": 0
}

def debug_tool_execution(tool_name: str, args: dict, execution_time: float, result: dict):
    """Debug tool execution with comprehensive tracking."""

    performance_stats["tool_calls"] += 1

    if result.get("chunked_content"):
        performance_stats["chunking_events"] += 1

    if result.get("is_error"):
        performance_stats["error_events"] += 1

    logger.debug(f"Tool {tool_name} executed in {execution_time:.2f}s - "
                f"Chunked: {result.get('chunked_content', False)}, "
                f"Error: {result.get('is_error', False)}")
```

### Common Issues & Solutions

1. **Token Limit Exceeded**: Implement better content optimization and use adaptive chunking
2. **Parameter Validation Failures**: Use fail-fast validation with clear error messages
3. **SDK Integration Issues**: Ensure proper import patterns and server registration
4. **Workproduct Path Issues**: Use environment-aware path detection for cross-repo compatibility
5. **Performance Issues**: Monitor chunking statistics and optimize content processing

## Dependencies & Interactions

### External Dependencies
- **claude-agent-sdk**: Claude Agent SDK for MCP server creation and tool registration
- **anthropic**: Anthropic API client for Claude integration
- **pydantic**: Data validation and serialization for complex data structures
- **asyncio**: Async programming support with proper resource management

### Internal System Dependencies
- **Utils Layer**: Core search and crawling functionality (`serp_search_utils`, `z_search_crawl_utils`)
- **Tools Layer**: Higher-level research tools and orchestration logic
- **Core System**: Orchestrator manages MCP server lifecycle and agent coordination
- **Config System**: Configuration management for tool behavior and limits

### Data Flow Architecture
```
Claude Agent → MCP Tool Invocation → Parameter Validation → Research Processing →
Content Cleaning → Token Management → MCP Compliance → Structured Response → Claude Agent
```

## Performance Considerations

### Token Optimization Strategies

1. **Adaptive Chunking**: Intelligently split content at logical break points to preserve context
2. **Content Allocation**: Use 70/30 split for primary content vs metadata
3. **Priority-Based Selection**: Prioritize critical content when token limits are reached
4. **Compression Tracking**: Monitor compression ratios and effectiveness

### Performance Monitoring
```python
# Comprehensive Performance Tracking
class MCPPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "tool_executions": {},
            "chunking_efficiency": {},
            "token_utilization": {},
            "error_rates": {}
        }

    def track_tool_execution(self, tool_name: str, execution_time: float,
                           chunked: bool, token_usage: int, error: bool):
        """Track comprehensive tool performance metrics."""

        if tool_name not in self.metrics["tool_executions"]:
            self.metrics["tool_executions"][tool_name] = []

        self.metrics["tool_executions"][tool_name].append({
            "execution_time": execution_time,
            "chunked": chunked,
            "token_usage": token_usage,
            "error": error,
            "timestamp": datetime.now()
        })
```

### Scaling Recommendations
- Use connection pooling for web requests to external APIs
- Implement intelligent caching for frequently requested content
- Monitor and optimize token usage patterns across different query types
- Use async/await patterns efficiently for concurrent processing

## Configuration Management

### MCP Tools Configuration
```python
# Production Configuration
MCP_TOOLS_CONFIG = {
    "enhanced_search": {
        "max_results": 50,
        "max_concurrent": 15,
        "chunking_threshold": 20000,
        "anti_bot_default": 1,
        "workproduct_tracking": True
    },
    "zplayground1": {
        "max_results": 50,
        "auto_crawl_top": 20,
        "anti_bot_levels": [0, 1, 2, 3],
        "mcp_compliance": True,
        "fail_fast_validation": True
    },
    "compliance_manager": {
        "max_tokens": 25000,
        "primary_content_ratio": 0.7,
        "metadata_ratio": 0.3,
        "compression_enabled": True,
        "priority_allocation": True
    }
}
```

### Environment Configuration
```bash
# Required Environment Variables
ANTHROPIC_API_KEY=your_anthropic_key
SERP_API_KEY=your_serp_api_key

# Optional Configuration
KEVIN_WORKPRODUCTS_DIR=/path/to/KEVIN/sessions
MCP_TOKEN_LIMIT=25000
CHUNKING_THRESHOLD=20000
DEBUG_MODE=false
```

## Usage Examples

### Basic Research Workflow
```python
# Claude agent using enhanced search
"Research the latest developments in quantum computing applications for healthcare"

# Tool invocation
enhanced_search_scrape_clean({
    "query": "quantum computing applications healthcare",
    "search_type": "search",
    "num_results": 15,
    "auto_crawl_top": 10,
    "anti_bot_level": 1,
    "session_id": "research_session_001"
})
```

### News Analysis Workflow
```python
# Claude agent using news search
"Analyze recent news about artificial intelligence regulation in Europe"

# Tool invocation
enhanced_news_search({
    "query": "artificial intelligence regulation Europe",
    "num_results": 20,
    "auto_crawl_top": 15,
    "anti_bot_level": 2,
    "session_id": "news_analysis_001"
})
```

### Comprehensive Research Workflow
```python
# Claude agent using expanded query search
"Provide comprehensive coverage of renewable energy trends and technologies"

# Tool invocation
expanded_query_search_and_extract({
    "query": "renewable energy trends technologies",
    "search_type": "search",
    "num_results": 15,
    "max_expanded_queries": 3,
    "session_id": "comprehensive_research_001"
})
```

### ZPlayground1 Production Workflow
```python
# Claude agent using zplayground1 implementation
"Research machine learning applications in financial services"

# Tool invocation
zplayground1_search_scrape_clean({
    "query": "machine learning applications financial services",
    "search_mode": "web",
    "num_results": 20,
    "anti_bot_level": 1,
    "session_id": "ml_finance_research",
    "workproduct_prefix": "financial_analysis"
})
```

### Enhanced Editorial Workflow MCP Tools Usage (NEW in v3.2)

#### Enhanced Editorial Analysis Workflow
```python
# Claude agent using enhanced editorial analysis
"Perform comprehensive editorial analysis of this research report on AI trends"

# Tool invocation
enhanced_editorial_analysis({
    "report_content": "AI trends research report content...",
    "research_corpus": {
        "sources": ["source1", "source2", "source3"],
        "content": ["content1", "content2", "content3"],
        "metadata": {"quality_scores": [0.8, 0.7, 0.9]}
    },
    "session_id": "editorial_analysis_001",
    "analysis_depth": "comprehensive",
    "confidence_threshold": 0.7,
    "max_gap_topics": 2
})
```

#### Gap Research Coordination Workflow
```python
# Claude agent coordinating gap research
"Coordinate gap research for identified gaps in AI healthcare applications"

# Tool invocation
coordinate_gap_research({
    "gap_topics": [
        "recent AI developments in healthcare",
        "regulatory considerations for AI in medicine",
        "AI diagnostic tool accuracy studies"
    ],
    "parent_session_id": "ai_healthcare_research",
    "max_concurrent_sub_sessions": 3,
    "session_id": "gap_research_coordination_001",
    "integration_quality_threshold": 0.7
})
```

#### Editorial Recommendations Workflow
```python
# Claude agent generating editorial recommendations
"Generate evidence-based editorial recommendations with ROI analysis"

# Tool invocation
generate_editorial_recommendations({
    "report_content": "Healthcare AI research report content...",
    "editorial_analysis": {
        "quality_assessment": {"overall_score": 0.72, "dimensions": {...}},
        "corpus_analysis": {"quality_metrics": {...}, "coverage_analysis": {...}},
        "gap_research_decision": {"should_execute": True, "confidence": 0.85}
    },
    "gap_results": {
        "successful_researches": 2,
        "integrated_analysis": {"quality_improvement": 0.15, "coverage_enhancement": 0.20}
    },
    "session_id": "editorial_recommendations_001",
    "include_roi_analysis": True,
    "priority_threshold": 0.8
})
```

#### Quality Assessment Workflow
```python
# Claude agent performing multi-dimensional quality assessment
"Assess quality across multiple dimensions for this research report"

# Tool invocation
assess_quality_dimensions({
    "content": "Research report content to assess...",
    "dimensions": [
        "accuracy", "completeness", "coherence", "relevance",
        "depth", "clarity", "source_quality", "objectivity"
    ],
    "session_id": "quality_assessment_001",
    "include_enhancement_recommendations": True,
    "quality_threshold": 0.75
})
```

#### Research Corpus Analysis Workflow
```python
# Claude agent analyzing research corpus
"Analyze this research corpus for quality and coverage assessment"

# Tool invocation
analyze_research_corpus({
    "research_corpus": {
        "articles": [
            {"content": "article 1 content", "date": "2024-01-01", "source": "source1"},
            {"content": "article 2 content", "date": "2024-02-01", "source": "source2"},
            {"content": "article 3 content", "date": "2024-03-01", "source": "source3"}
        ],
        "query": "artificial intelligence trends",
        "metadata": {"search_depth": "comprehensive"}
    },
    "analysis_dimensions": ["quality", "coverage", "temporal", "source_diversity"],
    "session_id": "corpus_analysis_001",
    "include_gap_identification": True,
    "coverage_aspects": ["temporal", "geographical", "topical"]
})
```

#### Complete Editorial Workflow Orchestration
```python
# Claude agent orchestrating complete editorial workflow
"Orchestrate complete editorial workflow for this AI research report"

# Tool invocation
orchestrate_editorial_workflow({
    "workflow_type": "comprehensive_editorial_analysis",
    "research_data": {
        "report_content": "AI research report content...",
        "research_corpus": {"sources": [...], "content": [...], "metadata": {...}},
        "requirements": {"quality_threshold": 0.8, "depth": "comprehensive"}
    },
    "config": {
        "analysis_depth": "comprehensive",
        "quality_thresholds": {"overall": 0.8, "dimensions": {...}},
        "enable_gap_research": True,
        "enable_roi_analysis": True
    },
    "session_id": "editorial_workflow_orchestration_001",
    "tool_sequence": [
        "enhanced_editorial_analysis",
        "gap_research_coordination",
        "editorial_recommendations",
        "quality_enhancement"
    ]
})
```

## Integration Examples

### Agent Definition with MCP Tools
```python
from claude_agent_sdk import AgentDefinition
from mcp_tools.enhanced_search_scrape_clean import enhanced_search_server
from mcp_tools.zplayground1_search import zplayground1_server

# Research agent with MCP tools
research_agent = AgentDefinition(
    name="research_agent",
    model="claude-3-5-sonnet-20241022",
    instructions="""You are a research agent with access to advanced search and content extraction tools.
    Use the enhanced search tools for comprehensive research and the zplayground1 tool for production workflows.""",
    tools=[
        "enhanced_search_scrape_clean",
        "enhanced_news_search",
        "expanded_query_search_and_extract",
        "zplayground1_search_scrape_clean"
    ],
    mcp_servers={
        "enhanced_search": enhanced_search_server,
        "zplayground1": zplayground1_server
    }
)
```

### Orchestration Integration
```python
# Using MCP tools in orchestrator
class ResearchOrchestrator:
    def __init__(self):
        self.mcp_servers = {
            "enhanced_search": enhanced_search_server,
            "zplayground1": zplayground1_server
        }

    async def execute_research_stage(self, session_id: str, topic: str):
        """Execute research stage using MCP tools."""

        # Choose appropriate tool based on research requirements
        if self._is_news_focused(topic):
            tool_name = "enhanced_news_search"
        elif self._needs_comprehensive_coverage(topic):
            tool_name = "expanded_query_search_and_extract"
        else:
            tool_name = "enhanced_search_scrape_clean"

        # Execute tool through SDK
        result = await self.client.call_tool(tool_name, {
            "query": topic,
            "session_id": session_id,
            "num_results": 15,
            "anti_bot_level": 1
        })

        return self._process_research_results(result)
```

This comprehensive MCP tools implementation provides enterprise-grade research capabilities with intelligent token management, sophisticated error handling, and seamless integration with the Claude Agent SDK, enabling reliable and scalable multi-agent research workflows.