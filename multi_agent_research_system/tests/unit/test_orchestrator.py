"""Unit tests for the ResearchOrchestrator class."""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.orchestrator import ResearchOrchestrator


class TestResearchOrchestrator:
    """Test ResearchOrchestrator functionality."""

    @pytest.mark.unit
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = ResearchOrchestrator()

        # Test required attributes
        assert hasattr(orchestrator, 'agent_definitions')
        assert hasattr(orchestrator, 'active_sessions')
        assert hasattr(orchestrator, 'agent_clients')
        assert hasattr(orchestrator, 'mcp_server')

        # Test initial state
        assert isinstance(orchestrator.agent_definitions, dict)
        assert isinstance(orchestrator.active_sessions, dict)
        assert isinstance(orchestrator.agent_clients, dict)
        assert len(orchestrator.active_sessions) == 0
        assert len(orchestrator.agent_clients) == 0

        # Test agent definitions loaded
        assert len(orchestrator.agent_definitions) == 4
        expected_agents = ['research_agent', 'report_agent', 'editor_agent', 'ui_coordinator']
        for agent in expected_agents:
            assert agent in orchestrator.agent_definitions

    @pytest.mark.unit
    @patch('core.orchestrator.create_sdk_mcp_server')
    @patch('core.orchestrator.ClaudeSDKClient')
    def test_initialization_with_mock_sdk(self, mock_client_class, mock_mcp_server):
        """Test initialization with mocked SDK."""
        # Mock MCP server
        mock_mcp = MagicMock()
        mock_mcp_server.return_value = mock_mcp

        # Mock client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.connect = AsyncMock()

        orchestrator = ResearchOrchestrator()

        # Test that MCP server was created with correct tools
        mock_mcp_server.assert_called_once()
        call_args = mock_mcp_server.call_args
        assert call_args[1]['name'] == "research_tools"

    @pytest.mark.unit
    @patch('core.orchestrator.create_sdk_mcp_server')
    @patch('core.orchestrator.ClaudeSDKClient')
    @pytest.mark.asyncio
    async def test_initialize_orchestrator(self, mock_client_class, mock_mcp_server):
        """Test orchestrator initialization with agent clients."""
        # Mock MCP server
        mock_mcp = MagicMock()
        mock_mcp_server.return_value = mock_mcp

        # Mock client
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client_class.return_value = mock_client

        orchestrator = ResearchOrchestrator()
        await orchestrator.initialize()

        # Test that clients were created for all agents
        assert len(orchestrator.agent_clients) == 4
        expected_agents = ['research_agent', 'report_agent', 'editor_agent', 'ui_coordinator']
        for agent in expected_agents:
            assert agent in orchestrator.agent_clients
            mock_client_class.assert_any_call()

        # Test that clients were connected
        assert mock_client.connect.call_count == 4

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_start_research_session(self, temp_dir, monkeypatch):
        """Test starting a research session."""
        # Mock the file operations to use temp directory
        monkeypatch.chdir(temp_dir)

        orchestrator = ResearchOrchestrator()

        topic = "Test Research Topic"
        user_requirements = {
            "depth": "Standard Research",
            "audience": "Business",
            "format": "Standard Report"
        }

        session_id = await orchestrator.start_research_session(topic, user_requirements)

        # Test session ID format
        assert isinstance(session_id, str)
        assert len(session_id) > 20  # UUID length

        # Test session was created
        assert session_id in orchestrator.active_sessions

        # Test session data
        session_data = orchestrator.active_sessions[session_id]
        assert session_data["session_id"] == session_id
        assert session_data["topic"] == topic
        assert session_data["user_requirements"] == user_requirements
        assert session_data["status"] == "initialized"
        assert session_data["current_stage"] == "research"

        # Test session directory was created
        session_dir = Path(f"researchmaterials/sessions/{session_id}")
        assert session_dir.exists()

        # Test session state file was created
        session_file = session_dir / "session_state.json"
        assert session_file.exists()

        # Clean up
        import shutil
        if session_dir.parent.exists():
            shutil.rmtree(session_dir.parent)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_start_research_session_with_mock_workflow(self, temp_dir, monkeypatch):
        """Test starting research session with mocked workflow execution."""
        monkeypatch.chdir(temp_dir)

        orchestrator = ResearchOrchestrator()

        # Mock the workflow execution to prevent actual execution
        with patch.object(orchestrator, 'execute_research_workflow') as mock_workflow:
            mock_workflow = AsyncMock()

            topic = "AI in Healthcare"
            user_requirements = {"depth": "Standard Research"}

            session_id = await orchestrator.start_research_session(topic, user_requirements)

            # Test workflow was scheduled
            assert asyncio.create_task.called  # Workflow task was created

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_session_status(self):
        """Test updating session status."""
        orchestrator = ResearchOrchestrator()

        # Create a test session
        session_id = "test-session-123"
        orchestrator.active_sessions[session_id] = {
            "session_id": session_id,
            "topic": "Test Topic",
            "status": "initialized",
            "current_stage": "research"
        }

        await orchestrator.update_session_status(session_id, "researching", "Conducting research")

        # Test status was updated
        session_data = orchestrator.active_sessions[session_id]
        assert session_data["status"] == "researching"
        assert session_data["status_message"] == "Conducting research"
        assert "last_updated" in session_data

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session_status(self):
        """Test getting session status."""
        orchestrator = ResearchOrchestrator()

        # Create a test session
        session_id = "test-session-123"
        orchestrator.active_sessions[session_id] = {
            "session_id": session_id,
            "topic": "Test Topic",
            "status": "completed",
            "current_stage": "finalization",
            "created_at": "2024-01-01T12:00:00",
            "workflow_history": [
                {"stage": "research", "completed_at": "2024-01-01T12:30:00"}
            ]
        }

        status = await orchestrator.get_session_status(session_id)

        # Test status response
        assert status["session_id"] == session_id
        assert status["status"] == "completed"
        assert status["current_stage"] == "finalization"
        assert status["topic"] == "Test Topic"
        assert status["created_at"] == "2024-01-01T12:00:00"
        assert len(status["workflow_history"]) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_session_status_not_found(self):
        """Test getting status for non-existent session."""
        orchestrator = ResearchOrchestrator()

        status = await orchestrator.get_session_status("non-existent-session")

        # Test error response
        assert "error" in status
        assert status["error"] == "Session not found"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_session_state(self, temp_dir, monkeypatch):
        """Test saving session state to file."""
        monkeypatch.chdir(temp_dir)

        orchestrator = ResearchOrchestrator()

        # Create a test session
        session_id = "test-session-123"
        session_data = {
            "session_id": session_id,
            "topic": "Test Topic",
            "status": "in_progress",
            "workflow_history": []
        }
        orchestrator.active_sessions[session_id] = session_data

        # Create session directory
        session_dir = Path(f"researchmaterials/sessions/{session_id}")
        session_dir.mkdir(parents=True)

        await orchestrator.save_session_state(session_id)

        # Test file was created
        session_file = session_dir / "session_state.json"
        assert session_file.exists()

        # Test file contents
        import json
        with open(session_file) as f:
            saved_data = json.load(f)

        assert saved_data["session_id"] == session_id
        assert saved_data["topic"] == "Test Topic"
        assert saved_data["status"] == "in_progress"

        # Clean up
        import shutil
        if session_dir.parent.exists():
            shutil.rmtree(session_dir.parent)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test orchestrator cleanup."""
        orchestrator = ResearchOrchestrator()

        # Mock some agent clients
        mock_client1 = MagicMock()
        mock_client1.disconnect = AsyncMock()
        mock_client2 = MagicMock()
        mock_client2.disconnect = AsyncMock()

        orchestrator.agent_clients = {
            "agent1": mock_client1,
            "agent2": mock_client2
        }
        orchestrator.active_sessions = {"session1": {}, "session2": {}}

        await orchestrator.cleanup()

        # Test clients were disconnected
        mock_client1.disconnect.assert_called_once()
        mock_client2.disconnect.assert_called_once()

        # Test collections were cleared
        assert len(orchestrator.agent_clients) == 0
        assert len(orchestrator.active_sessions) == 0

    @pytest.mark.unit
    def test_workflow_stages(self):
        """Test that workflow stages are properly defined."""
        from core.orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator()

        # Test that workflow methods exist
        assert hasattr(orchestrator, 'execute_research_workflow')
        assert hasattr(orchestrator, 'stage_conduct_research')
        assert hasattr(orchestrator, 'stage_generate_report')
        assert hasattr(orchestrator, 'stage_editorial_review')
        assert hasattr(orchestrator, 'stage_finalize')
        assert hasattr(orchestrator, 'complete_session')

        # Test that methods are async
        import inspect
        for method_name in [
            'execute_research_workflow',
            'stage_conduct_research',
            'stage_generate_report',
            'stage_editorial_review',
            'stage_finalize',
            'complete_session'
        ]:
            method = getattr(orchestrator, method_name)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"

    @pytest.mark.unit
    def test_session_data_structure(self):
        """Test that session data follows expected structure."""
        orchestrator = ResearchOrchestrator()

        # Test creating a session with expected structure
        session_id = "test-session-456"
        session_data = {
            "session_id": session_id,
            "topic": "AI in Healthcare",
            "user_requirements": {
                "depth": "Standard Research",
                "audience": "Technical",
                "format": "Technical Documentation"
            },
            "status": "initialized",
            "created_at": "2024-01-01T12:00:00",
            "current_stage": "research",
            "workflow_history": [],
            "final_report": None
        }

        orchestrator.active_sessions[session_id] = session_data

        # Validate session structure
        from tests.utils.assertions import assert_valid_session_data
        assert_valid_session_data(session_data)

    @pytest.mark.unit
    @patch('core.orchestrator.create_sdk_mcp_server')
    def test_mcp_server_tools_configuration(self, mock_mcp_server):
        """Test that MCP server is configured with correct tools."""
        mock_mcp = MagicMock()
        mock_mcp_server.return_value = mock_mcp

        orchestrator = ResearchOrchestrator()

        # Test MCP server creation
        mock_mcp_server.assert_called_once()
        call_args = mock_mcp_server.call_args

        # Test tools parameter includes all expected tools
        tools_arg = call_args[1]['tools']
        expected_tools = [
            'conduct_research', 'analyze_sources', 'generate_report',
            'revise_report', 'review_report', 'identify_research_gaps',
            'manage_session', 'save_report'
        ]

        for tool in expected_tools:
            assert tool in str(tools_arg), f"Missing tool: {tool}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stage_generate_report_bug_fix(self):
        """Test that the topic variable bug is fixed in stage_generate_report."""
        orchestrator = ResearchOrchestrator()

        # Create a test session with research data
        session_id = "test-session-789"
        topic = "Test Research Topic"

        session_data = {
            "session_id": session_id,
            "topic": topic,
            "user_requirements": {
                "depth": "Standard Research",
                "audience": "Business",
                "format": "Standard Report"
            },
            "status": "research_completed",
            "current_stage": "report_generation",
            "research_results": {
                "success": True,
                "data": {
                    "research_count": 2,
                    "files": ["test_file1.md", "test_file2.md"]
                }
            },
            "workflow_history": [
                {"stage": "research", "completed_at": "2024-01-01T12:00:00"}
            ]
        }

        orchestrator.active_sessions[session_id] = session_data

        # Mock the report scope determination
        mock_report_config = {
            "scope": "Standard Report",
            "llm_confidence": 0.9,
            "llm_reasoning": "Standard report appropriate for business audience",
            "style_instructions": "Professional business tone",
            "editing_rigor": "Standard",
            "special_requirements": ""
        }

        with patch.object(orchestrator, '_determine_report_scope_with_llm_judge', return_value=mock_report_config):
            with patch.object(orchestrator, 'execute_agent_query') as mock_query:
                # Mock successful report generation
                mock_query.return_value = {
                    "success": True,
                    "substantive_responses": 2,
                    "tool_executions": ["create_research_report", "Write"],
                    "report_quality": "high"
                }

                with patch.object(orchestrator, 'save_session_state') as mock_save:
                    with patch.object(orchestrator, 'start_work_product', return_value=2):
                        with patch.object(orchestrator, 'complete_work_product'):
                            with patch.object(orchestrator, 'update_session_status') as mock_status:

                                # This should not raise a NameError anymore
                                await orchestrator.stage_generate_report(session_id)

                                # Verify the method completed successfully
                                mock_query.assert_called_once()

                                # Verify the report prompt was created correctly with the topic
                                call_args = mock_query.call_args
                                report_prompt = call_args[0][1]  # Second argument is the prompt

                                # Check that the topic is correctly referenced in the prompt
                                assert topic in report_prompt
                                assert f'"{topic}"' in report_prompt

                                # Verify no NameError was raised by checking method completion
                                assert session_data["current_stage"] == "editorial_review"
                                assert "report_results" in session_data

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_stage_generate_report_error_handling(self):
        """Test error handling in stage_generate_report method."""
        orchestrator = ResearchOrchestrator()

        session_id = "test-session-error"
        topic = "Error Test Topic"

        session_data = {
            "session_id": session_id,
            "topic": topic,
            "user_requirements": {"depth": "Standard Research"},
            "status": "research_completed",
            "current_stage": "report_generation",
            "research_results": {"success": True},
            "workflow_history": []
        }

        orchestrator.active_sessions[session_id] = session_data

        # Mock report scope determination
        mock_report_config = {
            "scope": "Standard Report",
            "llm_confidence": 0.9,
            "llm_reasoning": "Test reasoning",
            "style_instructions": "Test style",
            "editing_rigor": "Standard",
            "special_requirements": ""
        }

        with patch.object(orchestrator, '_determine_report_scope_with_llm_judge', return_value=mock_report_config):
            with patch.object(orchestrator, 'execute_agent_query') as mock_query:
                # Mock report generation failure
                mock_query.side_effect = Exception("Test error")

                with patch.object(orchestrator, 'save_session_state'):
                    with patch.object(orchestrator, 'start_work_product', return_value=2):

                        # This should raise an error after retries
                        with pytest.raises(RuntimeError, match="Report generation failed after 3 attempts"):
                            await orchestrator.stage_generate_report(session_id)

                        # Verify error logging occurred (should have been called 3 times)
                        assert mock_query.call_count == 3

    @pytest.mark.unit
    def test_stage_generate_report_method_exists(self):
        """Test that stage_generate_report method exists and is async."""
        orchestrator = ResearchOrchestrator()

        # Test method exists
        assert hasattr(orchestrator, 'stage_generate_report')

        # Test method is async
        import inspect
        method = orchestrator.stage_generate_report
        assert inspect.iscoroutinefunction(method)
