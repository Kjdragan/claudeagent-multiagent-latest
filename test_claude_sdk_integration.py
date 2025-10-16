#!/usr/bin/env python3
"""
Simple Claude Agent SDK Integration Test
This script validates that your environment is properly configured for Claude Agent SDK usage
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_environment():
    """Test basic environment setup"""
    print("🔍 Testing Environment Setup")
    print("=" * 50)

    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ Environment variables loaded from .env file")
        else:
            print("⚠️  .env file not found - using system environment")
    except ImportError:
        print("⚠️  python-dotenv not installed - using system environment")

    results = {}

    # Test 1: Check environment variables
    print("\n1. Environment Variables:")
    env_vars = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ZAI_API_ENDPOINT": os.getenv("ZAI_API_ENDPOINT")
    }

    for var, value in env_vars.items():
        status = "✅ SET" if value else "❌ NOT SET"
        print(f"   {var}: {status}")
        results[f"env_{var}"] = bool(value)

    # Test 2: Check Python version
    print(f"\n2. Python Version: {sys.version}")
    results["python_version"] = sys.version_info

    # Test 3: Check project structure
    print("\n3. Project Structure:")
    important_dirs = [
        "multi_agent_research_system",
        "multi_agent_research_system/mcp_tools",
        "multi_agent_research_system/config",
        "multi_agent_research_system/core",
        "multi_agent_research_system/agents",
        "multi_agent_research_system/utils",
        "KEVIN"
    ]

    for dir_path in important_dirs:
        exists = Path(dir_path).exists()
        status = "✅ EXISTS" if exists else "❌ NOT FOUND"
        print(f"   {dir_path}: {status}")
        results[f"dir_{dir_path.replace('/', '_')}"] = exists

    # Test 4: Check Claude Agent SDK
    print("\n4. Claude Agent SDK:")
    try:
        # Try different import patterns based on SDK version
        try:
            from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition
            print("   ✅ Claude Agent SDK imported successfully (new style)")
        except ImportError:
            from claude_agent_sdk import ClaudeSDKClient, AgentDefinition
            print("   ✅ Claude Agent SDK imported successfully (legacy style)")

        results["claude_sdk"] = True

        # Test SDK version
        import claude_agent_sdk
        print(f"   Version: {getattr(claude_agent_sdk, '__version__', 'Unknown')}")

        # Test available classes
        available_classes = [attr for attr in dir(claude_agent_sdk) if not attr.startswith('_')]
        print(f"   Available classes: {', '.join(available_classes[:5])}{'...' if len(available_classes) > 5 else ''}")

    except ImportError as e:
        print(f"   ❌ Claude Agent SDK import failed: {str(e)}")
        results["claude_sdk"] = False

    return results

def test_configuration():
    """Test configuration files"""
    print("\n🔧 Testing Configuration")
    print("=" * 50)

    results = {}

    # Test 1: Check settings file
    settings_file = Path("~/.claude/settings.local.json").expanduser()
    print(f"\n1. Settings File: {settings_file}")

    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                config = json.load(f)

            print("   ✅ Settings file loaded successfully")
            results["settings_file"] = True
            results["settings_content"] = config

            # Check for MCP servers
            if "mcpServers" in config:
                mcp_count = len(config["mcpServers"])
                print(f"   MCP Servers: {mcp_count} configured")
                results["mcp_servers_count"] = mcp_count

                for name, server_config in config["mcpServers"].items():
                    print(f"     - {name}: {server_config.get('command', 'python')}")
            else:
                print("   ❌ No MCP servers configured")
                results["mcp_servers_count"] = 0

        except Exception as e:
            print(f"   ❌ Failed to load settings file: {str(e)}")
            results["settings_file"] = False
    else:
        print("   ❌ Settings file not found")
        results["settings_file"] = False

    return results

async def test_mcp_server_creation():
    """Test MCP server creation"""
    print("\n🔧 Testing MCP Server Creation")
    print("=" * 50)

    results = {}

    # Test 1: Try to create enhanced search server
    print("\n1. Enhanced Search Server:")
    try:
        from multi_agent_research_system.mcp_tools.enhanced_search_scrape_clean import create_enhanced_search_mcp_server

        server = create_enhanced_search_mcp_server()

        if server:
            print("   ✅ Enhanced search server created successfully")
            results["enhanced_search_server"] = True
        else:
            print("   ❌ Enhanced search server creation returned None")
            results["enhanced_search_server"] = False

    except Exception as e:
        print(f"   ❌ Enhanced search server creation failed: {str(e)}")
        results["enhanced_search_server"] = False

    # Test 2: Try to create zplayground1 server
    print("\n2. zPlayground1 Server:")
    try:
        from multi_agent_research_system.mcp_tools.zplayground1_search import create_zplayground1_mcp_server

        server = create_zplayground1_mcp_server()

        if server:
            print("   ✅ zPlayground1 server created successfully")
            results["zplayground1_server"] = True
        else:
            print("   ❌ zPlayground1 server creation returned None")
            results["zplayground1_server"] = False

    except Exception as e:
        print(f"   ❌ zPlayground1 server creation failed: {str(e)}")
        results["zplayground1_server"] = False

    return results

async def test_agent_creation():
    """Test agent creation with MCP tools"""
    print("\n🤖 Testing Agent Creation")
    print("=" * 50)

    results = {}

    # Test 1: Try to create enhanced research agent
    print("\n1. Enhanced Research Agent:")
    try:
        from multi_agent_research_system.agents.enhanced_research_agent import EnhancedResearchAgent

        agent = EnhancedResearchAgent()
        print("   ✅ Enhanced research agent created successfully")
        results["enhanced_research_agent"] = True

        # Test agent configuration
        if hasattr(agent, 'agent_config'):
            print(f"   Tools configured: {len(agent.agent_config.tools)}")
            print(f"   MCP servers: {len(agent.agent_config.mcp_servers)}")
            print(f"   Force tool usage: {agent.agent_config.force_tool_usage}")
            results["agent_config"] = True
        else:
            print("   ❌ Agent configuration not found")
            results["agent_config"] = False

    except Exception as e:
        print(f"   ❌ Enhanced research agent creation failed: {str(e)}")
        results["enhanced_research_agent"] = False

    return results

async def test_simple_tool_call():
    """Test simple tool invocation"""
    print("\n🔧 Testing Simple Tool Invocation")
    print("=" * 50)

    results = {}

    # Test direct MCP tool call if available
    print("\n1. Direct MCP Tool Call:")
    try:
        from multi_agent_research_system.mcp_tools.zplayground1_search import create_zplayground1_mcp_server

        # Create server
        server = create_zplayground1_mcp_server()

        if server:
            print(f"   ✅ Server created successfully")
            print(f"   Server type: {type(server)}")

            # Try different ways to access the tool
            tool_found = False
            tool_name = None

            # Method 1: get_tool method
            if hasattr(server, 'get_tool'):
                try:
                    tool = server.get_tool("zplayground1_search_scrape_clean")
                    if tool:
                        tool_found = True
                        tool_name = getattr(tool, 'name', 'Unknown')
                        print(f"   ✅ Tool found via get_tool: {tool_name}")
                except Exception as e:
                    print(f"   ⚠️  get_tool method failed: {str(e)}")

            # Method 2: tools attribute
            if not tool_found and hasattr(server, 'tools'):
                tools = server.tools
                print(f"   ✅ Found {len(tools)} tools in tools attribute")
                for tool in tools:
                    if hasattr(tool, 'name') and 'search' in tool.name:
                        tool_found = True
                        tool_name = tool.name
                        print(f"   ✅ Search tool found: {tool_name}")
                        break

            # Method 3: dict-like access
            if not tool_found and isinstance(server, dict):
                if 'tools' in server:
                    tools = server['tools']
                    print(f"   ✅ Found {len(tools)} tools in server dict")
                    for tool_name, tool in tools.items():
                        if 'search' in tool_name:
                            tool_found = True
                            print(f"   ✅ Search tool found: {tool_name}")
                            break

            if tool_found:
                results["tool_found"] = True
                results["tool_name"] = tool_name or "Found"
            else:
                print("   ❌ Tool not found with any access method")
                results["tool_found"] = False
        else:
            print("   ❌ Server creation failed")
            results["tool_found"] = False

    except Exception as e:
        print(f"   ❌ Tool lookup failed: {str(e)}")
        results["tool_found"] = False

    return results

def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)

    # Count passed tests
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} ❌")

    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")

    # Print detailed results
    print(f"\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")

    # Recommendations
    print(f"\nRecommendations:")
    if not results.get("env_ANTHROPIC_API_KEY"):
        print("   ⚠️  Set ANTHROPIC_API_KEY environment variable")
    if not results.get("env_SERPER_API_KEY"):
        print("   ⚠️  Set SERPER_API_KEY environment variable")
    if not results.get("settings_file"):
        print("   ⚠️  Create ~/.claude/settings.local.json configuration file")
    if not results.get("mcp_servers_count", 0):
        print("   ⚠️  Configure MCP servers in settings.local.json")
    if not results.get("claude_sdk"):
        print("   ⚠️  Install Claude Agent SDK: pip install claude-agent-sdk")
    if not results.get("enhanced_search_server"):
        print("   ⚠️  Check enhanced_search_scrape_clean.py for import issues")
    if not results.get("enhanced_research_agent"):
        print("   ⚠️  Check enhanced_research_agent.py for import issues")

    # Overall status
    if success_rate >= 80:
        print(f"\n🎉 EXCELLENT: Integration looks good! ({success_rate:.1f}% success rate)")
    elif success_rate >= 60:
        print(f"\n👍 GOOD: Integration mostly working ({success_rate:.1f}% success rate)")
    elif success_rate >= 40:
        print(f"\n⚠️  FAIR: Integration has some issues ({success_rate:.1f}% success rate)")
    else:
        print(f"\n❌ POOR: Integration has significant issues ({success_rate:.1f}% success rate)")

    return success_rate

async def main():
    """Main test function"""
    print("🚀 Claude Agent SDK Integration Test")
    print("Testing your environment for proper Claude Agent SDK integration")
    print("This will help identify why your research agents are generating template responses")

    # Run all tests
    env_results = test_environment()
    config_results = test_configuration()
    mcp_results = await test_mcp_server_creation()
    agent_results = await test_agent_creation()
    tool_results = await test_simple_tool_call()

    # Combine all results
    all_results = {
        **env_results,
        **config_results,
        **mcp_results,
        **agent_results,
        **tool_results
    }

    # Print summary
    success_rate = print_summary(all_results)

    # Exit with appropriate code
    if success_rate >= 80:
        print(f"\n✅ Integration test PASSED! Your environment is properly configured.")
        print("   The enhanced research agent should work correctly now.")
        sys.exit(0)
    elif success_rate >= 60:
        print(f"\n⚠️  Integration test PARTIAL. Some issues found but basic functionality should work.")
        print("   Try the enhanced research agent to see if it improves tool invocation.")
        sys.exit(0)
    else:
        print(f"\n❌ Integration test FAILED. Multiple critical issues found.")
        print("   Please address the recommendations above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())