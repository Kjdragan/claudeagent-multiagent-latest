"""Research Agent Integration Configuration

This module provides the integration configuration to properly connect
the enhanced research agent with the working MCP search tools.

Key Features:
- Proper MCP server registration for search tools
- Enhanced research agent configuration with real tool access
- Tool mapping between agent capabilities and MCP implementations
- Session management integration with threshold tracking
"""

import logging
from typing import Dict, List, Any

# Initialize servers as None and try to import them
enhanced_search_server = None
zplayground1_server = None

try:
    from ..agents.enhanced_research_agent import EnhancedResearchAgent
except ImportError:
    # Handle relative imports for different execution contexts
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from agents.enhanced_research_agent import EnhancedResearchAgent

try:
    from ..mcp_tools.enhanced_search_scrape_clean import enhanced_search_server
except ImportError:
    try:
        from mcp_tools.enhanced_search_scrape_clean import enhanced_search_server
    except ImportError:
        # Try direct import
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_tools"))
        from enhanced_search_scrape_clean import enhanced_search_server

try:
    from ..mcp_tools.zplayground1_search import zplayground1_server
except ImportError:
    try:
        from mcp_tools.zplayground1_search import zplayground1_server
    except ImportError:
        # Try direct import
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_tools"))
        from zplayground1_search import zplayground1_server


class ResearchAgentIntegration:
    """Integration configuration for enhanced research agent with real search tools."""

    def __init__(self):
        self.logger = logging.getLogger("research_agent_integration")
        self.enhanced_research_agent = EnhancedResearchAgent()

    def get_mcp_servers(self) -> Dict[str, Any]:
        """Get properly configured MCP servers for research operations."""
        return {
            "enhanced_search": enhanced_search_server,
            "zplayground1": zplayground1_search_server
        }

    def get_agent_definition(self) -> Dict[str, Any]:
        """Get the enhanced research agent definition with proper tool access."""
        return {
            "name": "enhanced_research_agent",
            "agent_class": EnhancedResearchAgent,
            "model": "claude-3-5-sonnet-20241022",
            "tools": [
                # Real search tools from enhanced_search MCP server
                "enhanced_search_scrape_clean",
                "enhanced_news_search",
                "expanded_query_search_and_extract",

                # Comprehensive search from zplayground1 MCP server
                "zplayground1_search_scrape_clean",

                # Enhanced agent analysis tools
                "real_web_research",
                "comprehensive_search_analysis",
                "gap_research_execution"
            ],
            "mcp_servers": self.get_mcp_servers(),
            "system_prompt": self.enhanced_research_agent.get_system_prompt(),
            "description": "Enhanced research agent with real search tool integration",
            "capabilities": [
                "Real web search using SERP API integration",
                "Content crawling with anti-bot protection",
                "AI-powered content cleaning and analysis",
                "Threshold monitoring for search optimization",
                "Gap research execution for comprehensive coverage",
                "Source credibility assessment",
                "Research synthesis from multiple sources"
            ]
        }

    def get_tool_mappings(self) -> Dict[str, str]:
        """Get mappings between agent tools and MCP implementations."""
        return {
            # Agent tool -> MCP tool mapping
            "real_web_research": {
                "standard_search": "enhanced_search_scrape_clean",
                "news_search": "enhanced_news_search",
                "comprehensive_search": "expanded_query_search_and_extract",
                "all_in_one_search": "zplayground1_search_scrape_clean"
            },
            "comprehensive_search_analysis": {
                "source": "internal_agent_capability"
            },
            "gap_research_execution": {
                "targeted_search": "enhanced_search_scrape_clean",
                "focused_search": "zplayground1_search_scrape_clean"
            }
        }

    def get_search_strategy_config(self) -> Dict[str, Any]:
        """Get configuration for intelligent search strategy selection."""
        return {
            "search_types": {
                "standard": {
                    "tool": "enhanced_search_scrape_clean",
                    "default_params": {
                        "search_type": "search",
                        "num_results": 15,
                        "auto_crawl_top": 10,
                        "anti_bot_level": 1
                    }
                },
                "news": {
                    "tool": "enhanced_news_search",
                    "default_params": {
                        "num_results": 15,
                        "auto_crawl_top": 10,
                        "anti_bot_level": 1
                    }
                },
                "comprehensive": {
                    "tool": "expanded_query_search_and_extract",
                    "default_params": {
                        "search_type": "search",
                        "num_results": 20,
                        "auto_crawl_top": 15,
                        "max_expanded_queries": 3,
                        "anti_bot_level": 2
                    }
                },
                "all_in_one": {
                    "tool": "zplayground1_search_scrape_clean",
                    "default_params": {
                        "search_mode": "web",
                        "num_results": 15,
                        "auto_crawl_top": 10,
                        "anti_bot_level": 1
                    }
                }
            },
            "research_depth_mapping": {
                "basic": {"search_type": "standard", "anti_bot_level": 0},
                "medium": {"search_type": "standard", "anti_bot_level": 1},
                "comprehensive": {"search_type": "comprehensive", "anti_bot_level": 2},
                "deep": {"search_type": "comprehensive", "anti_bot_level": 3}
            },
            "topic_detection": {
                "news_keywords": ["news", "latest", "recent", "breaking", "current", "today"],
                "academic_keywords": ["research", "study", "academic", "paper", "journal"],
                "technical_keywords": ["technical", "implementation", "code", "api", "technology"]
            }
        }

    def recommend_search_strategy(self, topic: str, research_depth: str = "medium") -> Dict[str, Any]:
        """Recommend optimal search strategy based on topic and requirements."""
        topic_lower = topic.lower()
        config = self.get_search_strategy_config()

        # Detect topic type
        search_type = "standard"
        if any(keyword in topic_lower for keyword in config["topic_detection"]["news_keywords"]):
            search_type = "news"
        elif any(keyword in topic_lower for keyword in config["topic_detection"]["academic_keywords"]):
            search_type = "comprehensive"
        elif any(keyword in topic_lower for keyword in config["topic_detection"]["technical_keywords"]):
            search_type = "comprehensive"

        # Map research depth to strategy
        depth_config = config["research_depth_mapping"].get(research_depth, config["research_depth_mapping"]["medium"])

        # Get tool configuration
        tool_config = config["search_types"][search_type]

        return {
            "recommended_tool": tool_config["tool"],
            "search_type": search_type,
            "parameters": {
                **tool_config["default_params"],
                "anti_bot_level": depth_config["anti_bot_level"]
            },
            "reasoning": f"Topic detected as {search_type} type with {research_depth} depth requirement"
        }

    def validate_search_configuration(self) -> Dict[str, Any]:
        """Validate that all search components are properly configured."""
        validation_results = {
            "mcp_servers": {},
            "agent_tools": {},
            "integrations": {},
            "overall_status": "unknown"
        }

        # Check MCP servers
        try:
            mcp_servers = self.get_mcp_servers()
            validation_results["mcp_servers"] = {
                "enhanced_search_server": "available" if enhanced_search_server else "missing",
                "zplayground1_server": "available" if zplayground1_search_server else "missing",
                "total_servers": len(mcp_servers)
            }
        except Exception as e:
            validation_results["mcp_servers"] = {"error": str(e)}

        # Check agent tools
        try:
            agent_tools = self.enhanced_research_agent.get_tools()
            validation_results["agent_tools"] = {
                "total_tools": len(agent_tools),
                "tool_names": [tool.__name__ for tool in agent_tools],
                "status": "available"
            }
        except Exception as e:
            validation_results["agent_tools"] = {"error": str(e)}

        # Check integration mappings
        try:
            tool_mappings = self.get_tool_mappings()
            validation_results["integrations"] = {
                "mappings_count": len(tool_mappings),
                "agent_tools_mapped": len(tool_mappings),
                "status": "configured"
            }
        except Exception as e:
            validation_results["integrations"] = {"error": str(e)}

        # Determine overall status
        server_errors = [k for k, v in validation_results["mcp_servers"].items() if isinstance(v, str) and v == "missing"]
        tool_errors = "error" in validation_results["agent_tools"]
        integration_errors = "error" in validation_results["integrations"]

        if server_errors or tool_errors or integration_errors:
            validation_results["overall_status"] = "configuration_errors"
            validation_results["issues"] = {
                "missing_servers": server_errors,
                "agent_tool_errors": tool_errors,
                "integration_errors": integration_errors
            }
        else:
            validation_results["overall_status"] = "ready"

        return validation_results

    def get_integration_instructions(self) -> str:
        """Get instructions for integrating the enhanced research agent."""
        return """
# Enhanced Research Agent Integration Instructions

## Overview
The enhanced research agent provides real web search capabilities by integrating with the working MCP search tools. This replaces the placeholder research agent with actual search functionality.

## MCP Server Registration
```python
from multi_agent_research_system.core.research_agent_integration import ResearchAgentIntegration

# Initialize integration
integration = ResearchAgentIntegration()

# Get MCP servers
mcp_servers = integration.get_mcp_servers()

# Register with Claude Agent SDK
client = ClaudeSDKClient(mcp_servers=mcp_servers)
```

## Agent Definition
```python
# Get enhanced research agent definition
agent_def = integration.get_agent_definition()

# Register agent
agent = await client.create_agent(
    name=agent_def["name"],
    model=agent_def["model"],
    tools=agent_def["tools"],
    system_prompt=agent_def["system_prompt"]
)
```

## Search Strategy Selection
```python
# Get optimal search strategy for a topic
strategy = integration.recommend_search_strategy(
    topic="latest developments in quantum computing",
    research_depth="comprehensive"
)

# Use the recommended tool
result = await client.call_tool(
    strategy["recommended_tool"],
    strategy["parameters"]
)
```

## Configuration Validation
```python
# Validate the integration setup
validation = integration.validate_search_configuration()

if validation["overall_status"] == "ready":
    print("Enhanced research agent is ready for use")
else:
    print("Configuration issues found:", validation["issues"])
```

## Key Features
1. **Real Search Tools**: Integration with enhanced_search_scrape_clean and zplayground1_search
2. **Intelligent Strategy Selection**: Automatic tool selection based on topic and requirements
3. **Threshold Monitoring**: Prevents excessive searching through intervention system
4. **Gap Research Execution**: Targeted research to fill identified gaps
5. **Source Credibility Assessment**: Analysis of source reliability from real search results

## Tool Capabilities
- `enhanced_search_scrape_clean`: Advanced topic-based search with parallel crawling
- `enhanced_news_search`: Specialized news search with content extraction
- `expanded_query_search_and_extract`: Query expansion with result consolidation
- `zplayground1_search_scrape_clean`: Complete search workflow in one tool
- `real_web_research`: Agent tool that selects optimal search strategy
- `comprehensive_search_analysis`: Synthesis of search results
- `gap_research_execution`: Targeted research for specific gaps

## Session Management
The enhanced research agent integrates with the KEVIN directory structure and threshold tracking system for efficient research session management.
"""


def create_enhanced_research_integration() -> ResearchAgentIntegration:
    """Factory function to create the enhanced research integration."""
    return ResearchAgentIntegration()


def get_research_agent_integration_instructions() -> str:
    """Get the integration instructions for the enhanced research agent."""
    integration = ResearchAgentIntegration()
    return integration.get_integration_instructions()


def validate_enhanced_research_setup() -> Dict[str, Any]:
    """Validate the enhanced research agent setup."""
    integration = ResearchAgentIntegration()
    return integration.validate_search_configuration()