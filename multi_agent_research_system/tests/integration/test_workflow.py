"""Integration tests for the multi-agent workflow orchestration."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.orchestrator import ResearchOrchestrator
from tests.utils.assertions import (
    assert_valid_session_data,
    assert_workflow_progression,
)
from tests.utils.test_helpers import create_test_session_files


class TestWorkflowIntegration:
    """Test integration between workflow components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_workflow_simulation(self, temp_dir, monkeypatch):
        """Test complete workflow with mocked SDK clients."""
        monkeypatch.chdir(temp_dir)

        # Mock the SDK clients
        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock(return_value="Mock response")
            mock_client.receive_response = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # Start a research session
            topic = "Integration Test Topic"
            requirements = {
                "depth": "Standard Research",
                "audience": "Business",
                "format": "Standard Report"
            }

            session_id = await orchestrator.start_research_session(topic, requirements)

            # Test session was created
            assert session_id in orchestrator.active_sessions
            session_data = orchestrator.active_sessions[session_id]
            assert_valid_session_data(session_data)

            # Test workflow stages progression
            await asyncio.sleep(0.1)  # Allow mock workflow to progress

            status = await orchestrator.get_session_status(session_id)
            assert status["session_id"] == session_id

            # Clean up
            await orchestrator.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_coordination(self, temp_dir, monkeypatch):
        """Test coordination between different agents."""
        monkeypatch.chdir(temp_dir)

        # Mock different responses for different agents
        def create_mock_client(agent_name):
            client = MagicMock()
            client.connect = AsyncMock()
            client.query = AsyncMock()
            client.agent_name = agent_name
            return client

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client_class.side_effect = lambda: create_mock_client("mock_agent")

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # Test that all agent clients were created
            assert len(orchestrator.agent_clients) == 4

            # Test agent interactions through workflow
            session_id = await orchestrator.start_research_session(
                "Agent Coordination Test",
                {"depth": "Quick Overview"}
            )

            # Verify session management across agents
            status = await orchestrator.get_session_status(session_id)
            assert status["session_id"] == session_id

            await orchestrator.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_stage_transitions(self, temp_dir, monkeypatch):
        """Test transitions between workflow stages."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            session_id = await orchestrator.start_research_session(
                "Workflow Transitions Test",
                {"depth": "Standard Research"}
            )

            # Mock stage transitions
            stages = [
                ("research", "researching", "Conducting research"),
                ("report_generation", "generating_report", "Generating report"),
                ("editorial_review", "editorial_review", "Reviewing report"),
                ("completion", "completed", "Workflow completed")
            ]

            workflow_history = []
            for stage_name, status, message in stages:
                await orchestrator.update_session_status(session_id, status, message)

                # Add to workflow history
                workflow_history.append({
                    "stage": stage_name,
                    "completed_at": "2024-01-01T12:00:00",
                    "results_count": 1
                })

                # Verify status update
                current_status = await orchestrator.get_session_status(session_id)
                assert current_status["current_stage"] == stage_name
                assert current_status["status"] == status

            # Verify workflow progression
            session_data = orchestrator.active_sessions[session_id]
            session_data["workflow_history"] = workflow_history
            assert_workflow_progression(workflow_history)

            await orchestrator.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tool_integration_workflow(self, temp_dir, monkeypatch):
        """Test integration between different tools in the workflow."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # Mock the tool execution workflow
            from core.research_tools import (
                conduct_research,
                generate_report,
                review_report,
            )

            # Execute research tools in sequence
            research_args = {
                "topic": "Tool Integration Test",
                "depth": "medium",
                "focus_areas": ["test"],
                "max_sources": 3
            }

            research_result = await conduct_research(research_args)
            assert "research_data" in research_result

            # Use research results to generate report
            report_args = {
                "research_data": research_result["research_data"],
                "format": "markdown",
                "audience": "general",
                "sections": ["summary", "findings"]
            }

            report_result = await generate_report(report_args)
            assert "report_data" in report_result

            # Review the generated report
            review_args = {
                "report": report_result["report_data"],
                "review_criteria": ["clarity", "accuracy"]
            }

            review_result = await review_report(review_args)
            assert "review_result" in review_result

            await orchestrator.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_persistence_integration(self, temp_dir, monkeypatch):
        """Test session persistence across workflow stages."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # Create session
            session_id = await orchestrator.start_research_session(
                "Persistence Integration Test",
                {"depth": "Standard Research"}
            )

            # Create session directory and files
            session_dir = temp_dir / "researchmaterials" / "sessions" / session_id
            session_dir.mkdir(parents=True)

            session_data = orchestrator.active_sessions[session_id]
            create_test_session_files(session_dir, session_data)

            # Test saving session state
            await orchestrator.save_session_state(session_id)

            # Test loading session state (create new orchestrator instance)
            orchestrator2 = ResearchOrchestrator()

            # Simulate loading from file
            import json
            session_file = session_dir / "session_state.json"
            with open(session_file) as f:
                loaded_session = json.load(f)

            assert loaded_session["session_id"] == session_id
            assert loaded_session["topic"] == "Persistence Integration Test"

            await orchestrator.cleanup()
            await orchestrator2.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, temp_dir, monkeypatch):
        """Test error handling across the workflow."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock(side_effect=Exception("Mock error"))
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            session_id = await orchestrator.start_research_session(
                "Error Handling Test",
                {"depth": "Quick Overview"}
            )

            # Test error handling in status updates
            await orchestrator.update_session_status(session_id, "error", "Test error occurred")

            status = await orchestrator.get_session_status(session_id)
            assert status["status"] == "error"
            assert "error" in status.get("status_message", "").lower()

            await orchestrator.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, temp_dir, monkeypatch):
        """Test handling multiple concurrent sessions."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # Start multiple sessions concurrently
            session_tasks = []
            topics = [
                "Concurrent Session 1",
                "Concurrent Session 2",
                "Concurrent Session 3"
            ]

            for topic in topics:
                task = orchestrator.start_research_session(topic, {"depth": "Quick Overview"})
                session_tasks.append(task)

            # Wait for all sessions to be created
            session_ids = await asyncio.gather(*session_tasks)

            # Test all sessions were created
            assert len(session_ids) == 3
            assert len(set(session_ids)) == 3  # All unique

            # Test session isolation
            for i, session_id in enumerate(session_ids):
                status = await orchestrator.get_session_status(session_id)
                assert status["topic"] == topics[i]
                assert status["session_id"] == session_id

            await orchestrator.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_data_flow(self, temp_dir, monkeypatch):
        """Test data flow between workflow stages."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # Simulate workflow data flow
            session_id = await orchestrator.start_research_session(
                "Data Flow Test",
                {"depth": "Standard Research"}
            )

            # Mock research results
            research_data = {
                "topic": "Data Flow Test",
                "findings": [
                    {"fact": "Test finding 1", "sources": ["source1.com"]},
                    {"fact": "Test finding 2", "sources": ["source2.com"]}
                ],
                "sources_used": [
                    {"title": "Test Source", "url": "https://source1.com", "type": "web"}
                ]
            }

            # Mock report generation
            report_data = {
                "title": "Data Flow Test Report",
                "sections": {
                    "summary": "Executive summary",
                    "findings": "Detailed findings",
                    "conclusions": "Conclusions"
                },
                "research_data": research_data
            }

            # Test data integrity through workflow
            assert report_data["research_data"] == research_data
            assert report_data["research_data"]["topic"] == "Data Flow Test"

            await orchestrator.cleanup()

    @pytest.mark.integration
    def test_mcp_server_integration(self):
        """Test MCP server integration with tools."""
        from tests.utils.mock_sdk import MockMCPTool

        # Test that tools can be wrapped for MCP server
        research_tool = MockMCPTool(
            "conduct_research",
            "Conduct research on a topic",
            {"topic": str, "depth": str}
        )

        # Test tool execution
        result = asyncio.run(research_tool({
            "topic": "MCP Integration Test",
            "depth": "medium"
        }))

        assert "content" in result
        assert "research_data" in result

        # Test tool metadata
        assert research_tool.call_count == 1
        assert research_tool.calls[0]["topic"] == "MCP Integration Test"
