#!/usr/bin/env python3
"""Test script to verify the search pipeline fix works."""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.append('/home/kjdragan/lrepos/claudeagent-multiagent-latest')

async def test_search_pipeline():
    """Test that the search pipeline now works with real MCP tools."""

    print("🔍 Testing Search Pipeline Fix")
    print("=" * 50)

    try:
        # Test 1: Check if MCP tools are properly configured
        print("1. Checking MCP tool registration...")
        from multi_agent_research_system.mcp_tools.zplayground1_search import create_zplayground1_mcp_server

        server = create_zplayground1_mcp_server()
        print(f"   ✅ MCP server created successfully")

        # Check server structure - it might be a dict
        if isinstance(server, dict):
            tools = server.get('tools', [])
            server_name = server.get('name', 'Unknown')
            print(f"   ✅ Server: {server_name}")
        else:
            tools = getattr(server, 'tools', [])
            print(f"   ✅ Server object created")

        print(f"   ✅ Available tools: {len(tools)}")

        # Test 2: Check agent configuration
        print("\n2. Checking agent configuration...")
        from multi_agent_research_system.config.agents import get_research_agent_definition

        research_agent = get_research_agent_definition()
        print(f"   ✅ Research agent definition loaded")
        print(f"   ✅ Tools configured: {research_agent.tools}")

        # Verify the MCP tool is in the list
        mcp_tool = "mcp__zplayground1_search__zplayground1_search_scrape_clean"
        if mcp_tool in research_agent.tools:
            print(f"   ✅ MCP tool found in agent configuration: {mcp_tool}")
        else:
            print(f"   ❌ MCP tool NOT found in agent configuration")
            print(f"   Available tools: {research_agent.tools}")
            return False

        # Test 3: Test actual search functionality (simple test)
        print("\n3. Testing actual search tool...")
        from multi_agent_research_system.utils.z_search_crawl_utils import search_crawl_and_clean_direct

        # Test with a simple query
        test_query = "artificial intelligence"
        test_session = "test_session_" + str(int(asyncio.get_event_loop().time()))

        print(f"   Testing query: '{test_query}'")
        print(f"   Session ID: {test_session}")

        # This should trigger real search if everything is working
        result = await search_crawl_and_clean_direct(
            query=test_query,
            search_type="web",
            num_results=3,  # Small number for testing
            auto_crawl_top=2,
            anti_bot_level=1,
            session_id=test_session
        )

        print(f"   ✅ Search completed successfully")
        print(f"   ✅ Result length: {len(result)} characters")
        print(f"   ✅ Result preview: {result[:200]}...")

        # Clean up test session
        import shutil
        test_session_dir = f"/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions/{test_session}"
        if os.path.exists(test_session_dir):
            shutil.rmtree(test_session_dir)
            print(f"   ✅ Test session cleaned up")

        print("\n" + "=" * 50)
        print("🎉 SEARCH PIPELINE FIX VERIFICATION COMPLETE")
        print("✅ All tests passed - the search pipeline should now work!")
        print("\nWhat was fixed:")
        print("- Updated agent configuration to use real MCP tools")
        print("- Replaced template-based 'conduct_research' with actual search pipeline")
        print("- Connected agents to working SERP API + crawling + content cleaning")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_search_pipeline())
    if success:
        print("\n✅ Fix verification successful!")
        sys.exit(0)
    else:
        print("\n❌ Fix verification failed!")
        sys.exit(1)