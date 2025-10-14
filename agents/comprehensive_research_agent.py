#!/usr/bin/env python3
"""
Comprehensive Research Agent - Claude Agent SDK Implementation

This module defines the comprehensive research agent that leverages the existing
zplayground1_search_scrape_clean MCP tool to conduct advanced research with
50+ URL processing capability, concurrent cleaning, and progressive anti-bot detection.

Key Features:
- Access to zplayground1_search_scrape_clean MCP tool
- Intelligent query analysis and optimization
- Comprehensive research workflow orchestration
- Quality assessment and enhancement
- KEVIN session management integration
- Real-time progress tracking and reporting

Agent Capabilities:
- Process 50+ target URLs with concurrent scraping
- Progressive anti-bot detection (4-level escalation)
- AI-powered content cleaning (GPT-5-nano)
- Early termination with success tracking
- Session-based file organization
- Quality assessment and progressive enhancement
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions
    CLAUDE_SDK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Claude Agent SDK not available: {e}")
    CLAUDE_SDK_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Tool definitions for comprehensive research agent
COMPREHENSIVE_RESEARCH_TOOLS = [
    {
        "name": "zplayground1_search_scrape_clean",
        "description": "Advanced web research with 50+ URL processing, concurrent scraping, and AI-powered content cleaning",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Research query or topic to investigate"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of search results to target (default: 50)",
                "default": 50
            },
            "auto_crawl_top": {
                "type": "integer",
                "description": "Number of top URLs to crawl concurrently (default: 20)",
                "default": 20
            },
            "anti_bot_level": {
                "type": "integer",
                "description": "Anti-bot detection level (0-3, default: 1)",
                "default": 1,
                "minimum": 0,
                "maximum": 3
            },
            "session_prefix": {
                "type": "string",
                "description": "Session prefix for organizing research results",
                "default": "comprehensive_research"
            }
        }
    },
    {
        "name": "get_session_data",
        "description": "Retrieve session data and research context from KEVIN directory",
        "parameters": {
            "session_id": {
                "type": "string",
                "description": "Session ID to retrieve data for"
            },
            "data_type": {
                "type": "string",
                "description": "Type of data to retrieve",
                "enum": ["research", "report", "editorial", "all"],
                "default": "all"
            }
        }
    }
]

# Agent instructions for comprehensive research
COMPREHENSIVE_RESEARCH_INSTRUCTIONS = """
You are a Comprehensive Research Specialist with access to advanced web scraping and content analysis tools. Your role is to conduct thorough, high-quality research using the comprehensive research capabilities available to you.

## Your Core Responsibilities

### 1. Research Excellence
- Use the zplayground1_search_scrape_clean tool to conduct comprehensive research
- Target 50+ high-quality URLs for extensive data collection
- Leverage concurrent processing for efficient research
- Apply progressive anti-bot detection as needed
- Ensure content quality through AI-powered cleaning

### 2. Query Optimization
- Analyze user queries to determine optimal research strategies
- Reformulate queries for better search results
- Generate orthogonal queries for comprehensive coverage
- Balance depth vs. breadth based on research requirements

### 3. Quality Assurance
- Assess source credibility and relevance
- Ensure content usefulness and quality
- Apply progressive enhancement when needed
- Generate comprehensive research reports with proper citations

### 4. Session Management
- Organize research using KEVIN directory structure
- Maintain proper session metadata and tracking
- Generate workproducts with standardized naming
- Provide real-time progress updates

## Research Workflow

### Initial Research Phase
1. **Query Analysis**: Carefully analyze the user's research query
2. **Strategy Planning**: Determine optimal search parameters and approach
3. **Execute Research**: Use zplayground1_search_scrape_clean with appropriate parameters:
   - Start with default settings (50 results, 20 concurrent crawls)
   - Adjust parameters based on query complexity and requirements
   - Monitor progress and adjust anti-bot levels as needed
4. **Quality Assessment**: Evaluate research results and identify gaps
5. **Enhancement**: Apply progressive enhancement if quality is insufficient

### Research Parameters
- **num_results**: Target 50 URLs for comprehensive coverage (adjust based on requirements)
- **auto_crawl_top**: Process 20 URLs concurrently for efficiency
- **anti_bot_level**: Start at level 1, escalate to 2-3 if needed
- **session_prefix**: Use descriptive prefixes for organization

### Content Analysis
- Focus on credible, authoritative sources
- Prioritize recent content for temporal relevance
- Ensure diverse perspectives and comprehensive coverage
- Apply critical thinking to source evaluation

## Quality Standards

### Source Quality Criteria
- Authority: Credible publications and expert sources
- Accuracy: Factual and verifiable information
- Relevance: Direct relation to research query
- Currency: Recent and up-to-date content
- Diversity: Multiple perspectives and sources

### Research Output Standards
- Comprehensive coverage of the research topic
- Proper source attribution and citation
- Clear organization and logical structure
- Actionable insights and analysis
- Identification of research gaps or limitations

## Interaction Patterns

### When Receiving a Research Query
1. Acknowledge and clarify the research scope
2. Plan your research strategy based on query complexity
3. Execute comprehensive research using available tools
4. Provide real-time progress updates when processing large research tasks
5. Deliver comprehensive results with proper organization

### Research Progress Updates
For large research tasks, provide updates:
- "Searching and analyzing 50+ sources on [topic]..."
- "Processing URLs with concurrent scraping and AI content cleaning..."
- "Found X high-quality sources, continuing analysis..."
- "Generating comprehensive research report..."

### Error Handling and Recovery
- If initial research is insufficient, adjust search parameters
- Escalate anti-bot levels if encountering access issues
- Provide alternative approaches if primary research method fails
- Always prioritize user experience and research quality

## Example Research Execution

**User Query**: "latest developments in quantum computing"

**Your Approach**:
1. Set research parameters: num_results=50, auto_crawl_top=20, anti_bot_level=1
2. Execute zplayground1_search_scrape_clean with optimized query
3. Monitor progress and adjust parameters as needed
4. Analyze results for quality and completeness
5. Generate comprehensive report with:
   - Executive summary of key developments
   - Technical breakthroughs and applications
   - Industry implications and market trends
   - Expert opinions and future outlook
   - Proper source attribution

## Special Instructions

### For Complex Research Topics
- Break down complex queries into focused sub-queries
- Use multiple research passes with different angles
- Synthesize information from diverse sources
- Provide structured analysis with clear categories

### For Time-Sensitive Queries
- Prioritize recent sources and breaking news
- Use real-time search capabilities when available
- Provide ongoing updates for developing situations
- Balance speed with comprehensive coverage

### For Academic or Technical Research
- Prioritize peer-reviewed and authoritative sources
- Include technical details and specific terminology
- Provide proper academic citation formats
- Identify research gaps and future directions

Remember: You are conducting comprehensive research at scale. Your goal is to provide thorough, well-sourced, and actionable research that fully addresses the user's information needs while maintaining high quality standards throughout the research process.
"""


def create_comprehensive_research_agent() -> Dict[str, Any]:
    """
    Create a comprehensive research agent definition with proper tool permissions.

    Returns:
        Dict[str, Any]: Agent configuration ready for integration

    Note:
        Returns a dict-based agent configuration for now until Claude Agent SDK
        constructor parameters are fully documented
    """

    logger.info("Creating comprehensive research agent with advanced tool permissions")

    # Create agent configuration
    agent_config = {
        "name": "comprehensive_research_agent",
        "description": "Advanced research specialist with 50+ URL processing capability, concurrent scraping, and AI-powered content analysis",
        "instructions": COMPREHENSIVE_RESEARCH_INSTRUCTIONS,
        "tools": COMPREHENSIVE_RESEARCH_TOOLS,
        "options": {
            "max_turns": 50,
            "continue_conversation": True,
            "include_partial_messages": True
        },
        "capabilities": [
            "comprehensive_web_research",
            "concurrent_url_processing",
            "ai_content_cleaning",
            "progressive_anti_bot_detection",
            "session_management",
            "quality_assessment"
        ]
    }

    logger.info("✅ Comprehensive research agent configuration created")
    logger.info(f"🔧 Agent tools: {[tool['name'] for tool in COMPREHENSIVE_RESEARCH_TOOLS]}")
    logger.info(f"⚙️  Agent options: max_turns={agent_config['options']['max_turns']}")

    return agent_config


def create_fallback_research_agent() -> Dict[str, Any]:
    """
    Create a fallback agent definition for when Claude Agent SDK is not available.

    Returns:
        Dict[str, Any]: Fallback agent configuration
    """

    logger.warning("Creating fallback research agent (Claude Agent SDK not available)")

    fallback_agent = {
        "name": "comprehensive_research_agent_fallback",
        "description": "Fallback research agent for when Claude Agent SDK is not available",
        "instructions": COMPREHENSIVE_RESEARCH_INSTRUCTIONS,
        "tools": COMPREHENSIVE_RESEARCH_TOOLS,
        "capabilities": {
            "research_execution": True,
            "quality_assessment": True,
            "session_management": True,
            "progress_tracking": True
        },
        "limitations": [
            "Claude Agent SDK integration not available",
            "MCP tool access limited",
            "Session management handled by fallback system"
        ]
    }

    logger.info("⚠️  Fallback research agent created")
    return fallback_agent


def validate_agent_configuration() -> Dict[str, Any]:
    """
    Validate agent configuration and return status report.

    Returns:
        Dict[str, Any]: Configuration validation report
    """

    validation_report = {
        "timestamp": datetime.now().isoformat(),
        "claude_sdk_available": CLAUDE_SDK_AVAILABLE,
        "tools_configured": len(COMPREHENSIVE_RESEARCH_TOOLS),
        "tool_details": {},
        "configuration_status": "unknown",
        "recommendations": []
    }

    # Validate tool definitions
    for tool in COMPREHENSIVE_RESEARCH_TOOLS:
        tool_name = tool["name"]
        tool_validation = {
            "has_parameters": "parameters" in tool,
            "parameter_count": len(tool.get("parameters", {})),
            "has_description": bool(tool.get("description")),
            "parameters_valid": True
        }

        # Check parameter schemas
        for param_name, param_config in tool.get("parameters", {}).items():
            if param_name not in ["query", "num_results", "auto_crawl_top", "anti_bot_level", "session_prefix", "session_id", "data_type"]:
                tool_validation["parameters_valid"] = False
                break

        validation_report["tool_details"][tool_name] = tool_validation

    # Determine overall status
    if CLAUDE_SDK_AVAILABLE:
        validation_report["configuration_status"] = "ready"
        validation_report["recommendations"].append("Agent ready for Claude SDK registration")
    else:
        validation_report["configuration_status"] = "fallback_only"
        validation_report["recommendations"].append("Install Claude Agent SDK: pip install claude-agent-sdk")

    # Check tool completeness
    required_tools = ["zplayground1_search_scrape_clean", "get_session_data"]
    available_tools = [tool["name"] for tool in COMPREHENSIVE_RESEARCH_TOOLS]

    missing_tools = [tool for tool in required_tools if tool not in available_tools]
    if missing_tools:
        validation_report["configuration_status"] = "incomplete"
        validation_report["recommendations"].append(f"Missing required tools: {missing_tools}")

    logger.info(f"Agent configuration validation: {validation_report['configuration_status']}")
    return validation_report


# Main agent creation function with fallback handling
def create_agent() -> Dict[str, Any]:
    """
    Create agent configuration for comprehensive research.

    Returns:
        Dict[str, Any]: Agent configuration
    """

    try:
        return create_comprehensive_research_agent()
    except Exception as e:
        logger.error(f"Unexpected error creating agent: {e}")
        return create_fallback_research_agent()


if __name__ == "__main__":
    # Test agent creation and validation
    print("🔧 Testing Comprehensive Research Agent Configuration")
    print("=" * 60)

    # Validate configuration
    validation = validate_agent_configuration()
    print(f"📋 Configuration Status: {validation['configuration_status']}")
    print(f"🔧 Claude SDK Available: {validation['claude_sdk_available']}")
    print(f"🛠️  Tools Configured: {validation['tools_configured']}")

    # Display tool details
    print("\n📦 Tool Configuration:")
    for tool_name, details in validation["tool_details"].items():
        status_icon = "✅" if details["parameters_valid"] else "❌"
        print(f"  {status_icon} {tool_name}: {details['parameter_count']} parameters")

    # Show recommendations
    if validation["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in validation["recommendations"]:
            print(f"  • {rec}")

    # Test agent creation
    print("\n🚀 Testing Agent Creation:")
    try:
        agent = create_agent()
        if isinstance(agent, dict) and 'name' in agent:
            print(f"  ✅ Agent configuration created: {agent['name']}")
            print(f"  🔧 Tools configured: {len(agent.get('tools', []))}")
            print(f"  ⚙️  Max turns: {agent.get('options', {}).get('max_turns', 'N/A')}")
        else:
            print(f"  ⚠️  Unexpected agent format: {type(agent)}")
    except Exception as e:
        print(f"  ❌ Agent creation failed: {e}")

    print("\n🎉 Agent configuration test completed")