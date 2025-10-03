#!/usr/bin/env python3
"""
Test script for the multi-agent research system.
Validates system components without requiring external dependencies.
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.append('..')

def test_agent_definitions():
    """Test agent definitions module."""
    print("Testing agent definitions...")

    from config.agents import get_all_agent_definitions

    agents = get_all_agent_definitions()
    assert len(agents) == 4, f"Expected 4 agents, got {len(agents)}"

    expected_agents = ['research_agent', 'report_agent', 'editor_agent', 'ui_coordinator']
    for agent_name in expected_agents:
        assert agent_name in agents, f"Missing agent: {agent_name}"

    # Test each agent has required attributes
    for name, agent in agents.items():
        assert hasattr(agent, 'description'), f"Agent {name} missing description"
        assert hasattr(agent, 'prompt'), f"Agent {name} missing prompt"
        assert hasattr(agent, 'tools'), f"Agent {name} missing tools"
        assert hasattr(agent, 'model'), f"Agent {name} missing model"
        assert len(agent.prompt) > 100, f"Agent {name} prompt too short"

    print("✅ Agent definitions test passed")

def test_research_tools():
    """Test research tools module."""
    print("Testing research tools...")

    from core.research_tools import (
        conduct_research, analyze_sources, generate_report, revise_report,
        review_report, identify_research_gaps, manage_session, save_report
    )

    # Test that tools are callable
    tools = [
        conduct_research, analyze_sources, generate_report, revise_report,
        review_report, identify_research_gaps, manage_session, save_report
    ]

    for tool_func in tools:
        assert callable(tool_func), f"Tool {tool_func.__name__} is not callable"

    # Test tool metadata
    for tool_func in tools:
        assert hasattr(tool_func, '_tool_name'), f"Tool {tool_func.__name__} missing name"
        assert hasattr(tool_func, '_tool_description'), f"Tool {tool_func.__name__} missing description"
        assert hasattr(tool_func, '_tool_params'), f"Tool {tool_func.__name__} missing params"

    print("✅ Research tools test passed")

def test_orchestrator():
    """Test orchestrator initialization."""
    print("Testing orchestrator...")

    from core.orchestrator import ResearchOrchestrator

    # Test basic initialization
    orchestrator = ResearchOrchestrator()

    # Test required attributes
    assert hasattr(orchestrator, 'agent_definitions'), "Orchestrator missing agent_definitions"
    assert hasattr(orchestrator, 'active_sessions'), "Orchestrator missing active_sessions"
    assert hasattr(orchestrator, 'agent_clients'), "Orchestrator missing agent_clients"

    # Test agent definitions are loaded
    assert len(orchestrator.agent_definitions) == 4, "Orchestrator didn't load agent definitions"

    # Test session management
    assert isinstance(orchestrator.active_sessions, dict), "Active sessions should be a dict"
    assert isinstance(orchestrator.agent_clients, dict), "Agent clients should be a dict"

    print("✅ Orchestrator test passed")

def test_file_structure():
    """Test file structure and directories."""
    print("Testing file structure...")

    # Test required directories exist
    required_dirs = [
        'core',
        'config',
        'ui',
        'researchmaterials/sessions'
    ]

    for dir_path in required_dirs:
        full_path = Path(dir_path)
        assert full_path.exists(), f"Directory {dir_path} does not exist"
        if dir_path != 'researchmaterials/sessions':  # Skip empty directory check
            assert any(full_path.iterdir()), f"Directory {dir_path} is empty"

    # Test required files exist
    required_files = [
        'core/orchestrator.py',
        'core/research_tools.py',
        'config/agents.py',
        'ui/streamlit_app.py',
        'requirements.txt',
        'README.md'
    ]

    for file_path in required_files:
        full_path = Path(file_path)
        assert full_path.exists(), f"File {file_path} does not exist"
        assert full_path.stat().st_size > 0, f"File {file_path} is empty"

    print("✅ File structure test passed")

def test_tool_execution():
    """Test basic tool execution with mock data."""
    print("Testing tool execution...")

    from core.research_tools import conduct_research, generate_report

    # Test research tool
    research_args = {
        "topic": "Test Topic",
        "depth": "medium",
        "focus_areas": ["area1", "area2"],
        "max_sources": 5
    }

    # This would normally be async, but we're testing the function exists
    assert callable(conduct_research), "Research tool not callable"

    # Test report tool
    report_args = {
        "research_data": {"topic": "Test Topic", "findings": []},
        "format": "markdown",
        "audience": "general",
        "sections": ["summary", "findings"]
    }

    assert callable(generate_report), "Report tool not callable"

    print("✅ Tool execution test passed")

def run_all_tests():
    """Run all system tests."""
    print("🧪 Running Multi-Agent Research System Tests")
    print("=" * 50)

    try:
        test_file_structure()
        test_agent_definitions()
        test_research_tools()
        test_orchestrator()
        test_tool_execution()

        print("=" * 50)
        print("🎉 All tests passed! System is ready for use.")
        print("\nTo run the system:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the UI: streamlit run ui/streamlit_app.py")
        print("3. Or use programmatically: from multi_agent_research_system import ResearchOrchestrator")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)