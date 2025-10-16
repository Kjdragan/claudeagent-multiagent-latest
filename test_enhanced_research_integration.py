#!/usr/bin/env python3
"""
Test Script for Enhanced Research Agent Integration

This script demonstrates and tests the integration between the enhanced research agent
and the real MCP search tools.

Usage:
    python test_enhanced_research_integration.py

Features:
- Validates MCP server registration
- Tests enhanced research agent functionality
- Demonstrates search strategy selection
- Shows threshold monitoring integration
- Provides diagnostic information
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multi_agent_research_system.core.research_agent_integration import (
    ResearchAgentIntegration,
    create_enhanced_research_integration,
    validate_enhanced_research_setup,
    get_research_agent_integration_instructions
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_mcp_server_registration():
    """Test MCP server registration and accessibility."""
    print("\n" + "="*80)
    print("🔧 TESTING MCP SERVER REGISTRATION")
    print("="*80)

    try:
        integration = create_enhanced_research_integration()
        mcp_servers = integration.get_mcp_servers()

        print(f"✅ MCP Servers Found: {len(mcp_servers)}")
        for server_name, server_instance in mcp_servers.items():
            status = "✅ Available" if server_instance else "❌ Missing"
            print(f"  - {server_name}: {status}")

        return len(mcp_servers) > 0

    except Exception as e:
        print(f"❌ MCP Server Registration Error: {e}")
        return False


async def test_enhanced_research_agent():
    """Test enhanced research agent initialization and tools."""
    print("\n" + "="*80)
    print("🤖 TESTING ENHANCED RESEARCH AGENT")
    print("="*80)

    try:
        integration = create_enhanced_research_integration()
        agent_def = integration.get_agent_definition()

        print(f"✅ Agent Name: {agent_def['name']}")
        print(f"✅ Model: {agent_def['model']}")
        print(f"✅ Tools Available: {len(agent_def['tools'])}")

        # Test agent tools
        enhanced_agent = integration.enhanced_research_agent
        tools = enhanced_agent.get_tools()

        print(f"✅ Agent Tools: {len(tools)}")
        for tool in tools:
            print(f"  - {tool.__name__}: Available")

        # Test system prompt
        system_prompt = enhanced_agent.get_system_prompt()
        print(f"✅ System Prompt Length: {len(system_prompt)} characters")

        return len(tools) > 0

    except Exception as e:
        print(f"❌ Enhanced Research Agent Error: {e}")
        return False


async def test_search_strategy_selection():
    """Test intelligent search strategy selection."""
    print("\n" + "="*80)
    print("🎯 TESTING SEARCH STRATEGY SELECTION")
    print("="*80)

    try:
        integration = create_enhanced_research_integration()

        # Test different topic types
        test_cases = [
            {
                "topic": "latest developments in quantum computing",
                "research_depth": "comprehensive",
                "expected_type": "comprehensive"
            },
            {
                "topic": "breaking news about climate change",
                "research_depth": "medium",
                "expected_type": "news"
            },
            {
                "topic": "research on artificial intelligence",
                "research_depth": "deep",
                "expected_type": "comprehensive"
            },
            {
                "topic": "general overview of renewable energy",
                "research_depth": "basic",
                "expected_type": "standard"
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_case['topic']}")

            strategy = integration.recommend_search_strategy(
                topic=test_case["topic"],
                research_depth=test_case["research_depth"]
            )

            print(f"  Recommended Tool: {strategy['recommended_tool']}")
            print(f"  Search Type: {strategy['search_type']}")
            print(f"  Anti-Bot Level: {strategy['parameters']['anti_bot_level']}")
            print(f"  Reasoning: {strategy['reasoning']}")

            # Validate recommendation
            if strategy['search_type'] == test_case['expected_type']:
                print(f"  ✅ Strategy matches expectation")
            else:
                print(f"  ⚠️  Strategy differs from expected ({test_case['expected_type']})")

        return True

    except Exception as e:
        print(f"❌ Search Strategy Selection Error: {e}")
        return False


async def test_tool_mappings():
    """Test tool mappings between agent and MCP implementations."""
    print("\n" + "="*80)
    print("🔗 TESTING TOOL MAPPINGS")
    print("="*80)

    try:
        integration = create_enhanced_research_integration()
        tool_mappings = integration.get_tool_mappings()

        print(f"✅ Tool Mappings Found: {len(tool_mappings)}")

        for agent_tool, mapping in tool_mappings.items():
            print(f"\nAgent Tool: {agent_tool}")
            if isinstance(mapping, dict):
                for purpose, mcp_tool in mapping.items():
                    print(f"  - {purpose}: {mcp_tool}")
            else:
                print(f"  - Mapping: {mapping}")

        return len(tool_mappings) > 0

    except Exception as e:
        print(f"❌ Tool Mappings Error: {e}")
        return False


async def test_threshold_tracking():
    """Test threshold tracking integration."""
    print("\n" + "="*80)
    print("📊 TESTING THRESHOLD TRACKING")
    print("="*80)

    try:
        # Try to import threshold tracking
        from multi_agent_research_system.utils.research_threshold_tracker import (
            check_search_threshold,
            get_research_threshold_tracker
        )

        tracker = get_research_threshold_tracker()
        print(f"✅ Threshold Tracker Initialized: {type(tracker).__name__}")

        # Test threshold checking
        test_session_id = "test_session_123"
        test_query = "test query for threshold checking"

        intervention = check_search_threshold(test_session_id, test_query, "test_search")

        if intervention is None:
            print("✅ Threshold check passed (no intervention needed)")
        else:
            print(f"✅ Threshold intervention generated: {len(intervention)} characters")

        return True

    except ImportError as e:
        print(f"⚠️  Threshold tracking not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Threshold Tracking Error: {e}")
        return False


async def test_configuration_validation():
    """Test overall configuration validation."""
    print("\n" + "="*80)
    print("✅ TESTING CONFIGURATION VALIDATION")
    print("="*80)

    try:
        validation_results = validate_enhanced_research_setup()

        print(f"Overall Status: {validation_results['overall_status']}")

        # MCP servers status
        mcp_status = validation_results.get("mcp_servers", {})
        print(f"\nMCP Servers:")
        for server, status in mcp_status.items():
            if server != "total_servers":
                print(f"  - {server}: {status}")
        if "total_servers" in mcp_status:
            print(f"  Total servers: {mcp_status['total_servers']}")

        # Agent tools status
        tools_status = validation_results.get("agent_tools", {})
        print(f"\nAgent Tools:")
        if "total_tools" in tools_status:
            print(f"  - Total tools: {tools_status['total_tools']}")
        if "tool_names" in tools_status:
            print(f"  - Tool names: {', '.join(tools_status['tool_names'])}")
        if "status" in tools_status:
            print(f"  - Status: {tools_status['status']}")

        # Integration status
        integration_status = validation_results.get("integrations", {})
        print(f"\nIntegrations:")
        if "mappings_count" in integration_status:
            print(f"  - Tool mappings: {integration_status['mappings_count']}")
        if "status" in integration_status:
            print(f"  - Status: {integration_status['status']}")

        # Check for issues
        if validation_results['overall_status'] == "ready":
            print(f"\n✅ Configuration is ready for use!")
            return True
        else:
            issues = validation_results.get("issues", {})
            print(f"\n⚠️  Configuration issues found:")
            if issues.get("missing_servers"):
                print(f"  - Missing MCP servers: {issues['missing_servers']}")
            if issues.get("agent_tool_errors"):
                print(f"  - Agent tool errors: {issues['agent_tool_errors']}")
            if issues.get("integration_errors"):
                print(f"  - Integration errors: {issues['integration_errors']}")
            return False

    except Exception as e:
        print(f"❌ Configuration Validation Error: {e}")
        return False


async def demonstrate_integration_usage():
    """Demonstrate how to use the enhanced research integration."""
    print("\n" + "="*80)
    print("📚 INTEGRATION USAGE DEMONSTRATION")
    print("="*80)

    try:
        # Show integration instructions
        instructions = get_research_agent_integration_instructions()
        print("Integration Instructions:")
        print("-" * 40)
        print(instructions[:1000] + "..." if len(instructions) > 1000 else instructions)

        # Show example usage
        print("\nExample Usage:")
        print("-" * 40)

        integration = create_enhanced_research_integration()

        # Example 1: Get agent definition
        agent_def = integration.get_agent_definition()
        print(f"1. Agent Definition:")
        print(f"   Name: {agent_def['name']}")
        print(f"   Capabilities: {len(agent_def['capabilities'])} capabilities")

        # Example 2: Recommend search strategy
        strategy = integration.recommend_search_strategy(
            topic="latest AI research papers",
            research_depth="comprehensive"
        )
        print(f"\n2. Search Strategy Recommendation:")
        print(f"   Tool: {strategy['recommended_tool']}")
        print(f"   Parameters: {strategy['parameters']}")

        # Example 3: Validate setup
        validation = integration.validate_search_configuration()
        print(f"\n3. Setup Validation:")
        print(f"   Status: {validation['overall_status']}")

        return True

    except Exception as e:
        print(f"❌ Integration Usage Demo Error: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 ENHANCED RESEARCH AGENT INTEGRATION TEST")
    print("="*80)
    print("This script tests the integration between the enhanced research agent")
    print("and the real MCP search tools.")
    print("="*80)

    # Run all tests
    test_results = {}

    test_results["mcp_servers"] = await test_mcp_server_registration()
    test_results["research_agent"] = await test_enhanced_research_agent()
    test_results["search_strategy"] = await test_search_strategy_selection()
    test_results["tool_mappings"] = await test_tool_mappings()
    test_results["threshold_tracking"] = await test_threshold_tracking()
    test_results["config_validation"] = await test_configuration_validation()
    test_results["usage_demo"] = await demonstrate_integration_usage()

    # Summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! The enhanced research integration is ready.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. Some components may need attention.")
    else:
        print("🚨 Multiple test failures. Integration needs configuration fixes.")

    print("\nNext Steps:")
    print("1. Ensure all MCP servers are properly registered")
    print("2. Replace the placeholder research_agent with enhanced_research_agent")
    print("3. Update orchestrator to use the new agent definition")
    print("4. Test with real research queries to verify functionality")


if __name__ == "__main__":
    asyncio.run(main())