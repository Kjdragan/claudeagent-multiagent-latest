"""Unit tests for agent definitions and configurations."""

import os
import sys
from unittest.mock import patch

import pytest

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.agents import (
    create_agent_config_file,
    get_all_agent_definitions,
    get_editor_agent_definition,
    get_report_agent_definition,
    get_research_agent_definition,
    get_ui_coordinator_definition,
)


class TestAgentDefinitions:
    """Test agent definition functions."""

    @pytest.mark.unit
    def test_research_agent_definition(self):
        """Test research agent definition structure."""
        agent = get_research_agent_definition()

        # Test basic attributes
        assert hasattr(agent, 'description')
        assert hasattr(agent, 'prompt')
        assert hasattr(agent, 'tools')
        assert hasattr(agent, 'model')

        # Test content quality
        assert len(agent.description) > 50
        assert len(agent.prompt) > 500  # Should be comprehensive
        assert "research" in agent.description.lower()
        assert "research" in agent.prompt.lower()

        # Test tools
        expected_tools = ["WebSearch", "WebFetch", "Read", "Write", "Edit", "Bash"]
        for tool in expected_tools:
            assert tool in agent.tools

        # Test model
        assert agent.model == "sonnet"

    @pytest.mark.unit
    def test_report_agent_definition(self):
        """Test report agent definition structure."""
        agent = get_report_agent_definition()

        # Test basic attributes
        assert hasattr(agent, 'description')
        assert hasattr(agent, 'prompt')
        assert hasattr(agent, 'tools')
        assert hasattr(agent, 'model')

        # Test content quality
        assert len(agent.description) > 50
        assert len(agent.prompt) > 500
        assert "report" in agent.description.lower()
        assert "report" in agent.prompt.lower()

        # Test tools
        expected_tools = ["Read", "Write", "Edit", "WebFetch", "Bash"]
        for tool in expected_tools:
            assert tool in agent.tools

        # Test model
        assert agent.model == "sonnet"

    @pytest.mark.unit
    def test_editor_agent_definition(self):
        """Test editor agent definition structure."""
        agent = get_editor_agent_definition()

        # Test basic attributes
        assert hasattr(agent, 'description')
        assert hasattr(agent, 'prompt')
        assert hasattr(agent, 'tools')
        assert hasattr(agent, 'model')

        # Test content quality
        assert len(agent.description) > 50
        assert len(agent.prompt) > 500
        assert "editor" in agent.description.lower()
        assert "review" in agent.prompt.lower()

        # Test tools
        expected_tools = ["Read", "Write", "Edit", "WebSearch", "WebFetch", "Bash"]
        for tool in expected_tools:
            assert tool in agent.tools

        # Test model
        assert agent.model == "sonnet"

    @pytest.mark.unit
    def test_ui_coordinator_definition(self):
        """Test UI coordinator definition structure."""
        agent = get_ui_coordinator_definition()

        # Test basic attributes
        assert hasattr(agent, 'description')
        assert hasattr(agent, 'prompt')
        assert hasattr(agent, 'tools')
        assert hasattr(agent, 'model')

        # Test content quality
        assert len(agent.description) > 50
        assert len(agent.prompt) > 500
        assert "coordinator" in agent.description.lower()
        assert "workflow" in agent.prompt.lower()

        # Test tools
        expected_tools = ["Read", "Write", "Edit", "WebSearch", "WebFetch", "Bash"]
        for tool in expected_tools:
            assert tool in agent.tools

        # Test model
        assert agent.model == "sonnet"

    @pytest.mark.unit
    def test_get_all_agent_definitions(self):
        """Test getting all agent definitions."""
        agents = get_all_agent_definitions()

        # Test structure
        assert isinstance(agents, dict)
        assert len(agents) == 4

        # Test expected agents
        expected_agents = [
            "research_agent",
            "report_agent",
            "editor_agent",
            "ui_coordinator"
        ]
        for agent_name in expected_agents:
            assert agent_name in agents
            assert hasattr(agents[agent_name], 'description')
            assert hasattr(agents[agent_name], 'prompt')
            assert hasattr(agents[agent_name], 'tools')
            assert hasattr(agents[agent_name], 'model')

    @pytest.mark.unit
    def test_create_agent_config_file(self):
        """Test creating agent configuration file."""
        config_json = create_agent_config_file()

        # Test it's valid JSON
        import json
        config = json.loads(config_json)

        # Test structure
        assert "agents" in config
        assert isinstance(config["agents"], dict)
        assert len(config["agents"]) == 4

        # Test each agent in config
        for agent_name, agent_config in config["agents"].items():
            assert "description" in agent_config
            assert "prompt" in agent_config
            assert "tools" in agent_config
            assert "model" in agent_config

            # Test content quality
            assert len(agent_config["description"]) > 50
            assert len(agent_config["prompt"]) > 500
            assert isinstance(agent_config["tools"], list)
            assert len(agent_config["tools"]) > 0

    @pytest.mark.unit
    def test_agent_prompt_quality(self):
        """Test that agent prompts have sufficient quality and detail."""
        agents = get_all_agent_definitions()

        for name, agent in agents.items():
            prompt = agent.prompt

            # Test prompt length (should be comprehensive)
            assert len(prompt) > 500, f"Agent {name} prompt too short"

            # Test for key sections
            assert "responsibilities" in prompt.lower() or "core" in prompt.lower()
            assert "available" in prompt.lower() and "tools" in prompt.lower()

            # Test for role clarity
            assert name.replace("_", " ") in prompt.lower() or name in prompt.lower()

            # Test for instructions
            assert "when" in prompt.lower()  # Should have conditional instructions

    @pytest.mark.unit
    def test_agent_tool_assignments(self):
        """Test that agents have appropriate tool assignments."""
        agents = get_all_agent_definitions()

        # Research agent should have research tools
        research_tools = agents["research_agent"].tools
        assert "WebSearch" in research_tools
        assert "WebFetch" in research_tools

        # Report agent should have writing tools
        report_tools = agents["report_agent"].tools
        assert "Write" in report_tools
        assert "Edit" in report_tools

        # Editor agent should have editing tools
        editor_tools = agents["editor_agent"].tools
        assert "Edit" in editor_tools
        assert "Read" in editor_tools

        # All agents should have basic tools
        common_tools = ["Read", "Bash"]
        for agent_name, agent in agents.items():
            for tool in common_tools:
                assert tool in agent.tools, f"Agent {agent_name} missing common tool {tool}"

    @pytest.mark.unit
    def test_agent_model_consistency(self):
        """Test that all agents use consistent models."""
        agents = get_all_agent_definitions()

        for name, agent in agents.items():
            assert hasattr(agent, 'model')
            assert agent.model in ["sonnet", "haiku", "opus"]
            # Currently all should be sonnet for consistency
            assert agent.model == "sonnet"

    @pytest.mark.unit
    @patch('config.agents.AgentDefinition')
    def test_fallback_agent_definition(self, mock_agent_def):
        """Test fallback behavior when SDK is not available."""
        # Mock the AgentDefinition to not be available
        mock_agent_def.side_effect = ImportError("No module named 'claude_agent_sdk'")

        # This should use the fallback class
        try:
            from config.agents import AgentDefinition as FallbackAgent

            # Test fallback agent works
            fallback = FallbackAgent(
                description="Test",
                prompt="Test prompt",
                tools=["test"],
                model="sonnet"
            )

            assert fallback.description == "Test"
            assert fallback.prompt == "Test prompt"
            assert fallback.tools == ["test"]
            assert fallback.model == "sonnet"

        except ImportError:
            pytest.skip("Cannot test fallback when real SDK is available")
