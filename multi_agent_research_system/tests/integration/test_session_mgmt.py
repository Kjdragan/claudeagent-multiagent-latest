"""Integration tests for session management functionality."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from core.orchestrator import ResearchOrchestrator
from tests.utils.assertions import (
    assert_file_exists_and_not_empty,
    assert_session_directory_structure,
    assert_valid_session_data,
)
from tests.utils.test_helpers import create_mock_session, create_test_session_files


class TestSessionManagementIntegration:
    """Test session management integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_lifecycle(self, temp_dir, monkeypatch):
        """Test complete session lifecycle."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # 1. Create session
            topic = "Session Lifecycle Test"
            requirements = {"depth": "Standard Research", "audience": "Business"}

            session_id = await orchestrator.start_research_session(topic, requirements)

            # Verify session creation
            assert session_id in orchestrator.active_sessions
            session_data = orchestrator.active_sessions[session_id]
            assert_valid_session_data(session_data)

            # 2. Update session status
            await orchestrator.update_session_status(session_id, "researching", "Conducting research")
            await orchestrator.update_session_status(session_id, "generating_report", "Creating report")
            await orchestrator.update_session_status(session_id, "completed", "Research completed")

            # Verify status updates
            final_status = await orchestrator.get_session_status(session_id)
            assert final_status["status"] == "completed"
            assert final_status["current_stage"] in ["finalization", "completed"]

            # 3. Clean up session
            await orchestrator.cleanup()

            # Verify cleanup
            assert len(orchestrator.active_sessions) == 0
            assert len(orchestrator.agent_clients) == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_file_operations(self, temp_dir, monkeypatch):
        """Test session file operations."""
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
                "File Operations Test",
                {"depth": "Quick Overview"}
            )

            # Create session directory structure
            session_dir = temp_dir / "researchmaterials" / "sessions" / session_id
            session_dir.mkdir(parents=True)

            session_data = orchestrator.active_sessions[session_id]
            create_test_session_files(session_dir, session_data)

            # Test session state saving
            await orchestrator.save_session_state(session_id)

            # Verify file exists and has content
            session_file = session_dir / "session_state.json"
            assert_file_exists_and_not_empty(session_file)

            # Test loading session state
            with open(session_file) as f:
                loaded_data = json.load(f)

            assert loaded_data["session_id"] == session_id
            assert loaded_data["topic"] == "File Operations Test"

            # Test directory structure
            assert_session_directory_structure(session_dir)

            await orchestrator.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_recovery(self, temp_dir, monkeypatch):
        """Test session recovery from saved state."""
        monkeypatch.chdir(temp_dir)

        # Create session files manually
        session_id = "recovery-test-session"
        session_dir = temp_dir / "researchmaterials" / "sessions" / session_id
        session_dir.mkdir(parents=True)

        session_data = create_mock_session(
            session_id=session_id,
            topic="Session Recovery Test",
            status="researching",
            current_stage="research"
        )

        create_test_session_files(session_dir, session_data)

        # Test recovery
        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client_class.return_value = mock_client

            # Simulate loading session from file
            session_file = session_dir / "session_state.json"
            with open(session_file) as f:
                recovered_session = json.load(f)

            # Verify recovered data
            assert recovered_session["session_id"] == session_id
            assert recovered_session["topic"] == "Session Recovery Test"
            assert recovered_session["status"] == "researching"

            # Test continuing session
            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # Add recovered session to active sessions
            orchestrator.active_sessions[session_id] = recovered_session

            # Test continuing workflow
            await orchestrator.update_session_status(session_id, "generating_report", "Continuing from recovery")

            status = await orchestrator.get_session_status(session_id)
            assert status["status"] == "generating_report"

            await orchestrator.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_isolation(self, temp_dir, monkeypatch):
        """Test isolation between multiple sessions."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # Create multiple sessions with different topics
            sessions = []
            topics = [
                ("AI in Healthcare", {"depth": "Comprehensive", "audience": "Medical"}),
                ("Remote Work Trends", {"depth": "Standard", "audience": "Business"}),
                ("Climate Change Impact", {"depth": "Quick", "audience": "General"})
            ]

            for topic, requirements in topics:
                session_id = await orchestrator.start_research_session(topic, requirements)
                sessions.append((session_id, topic, requirements))

            # Test session isolation
            for session_id, topic, requirements in sessions:
                status = await orchestrator.get_session_status(session_id)
                assert status["topic"] == topic
                assert status["user_requirements"] == requirements
                assert status["session_id"] == session_id

            # Test independent status updates
            await orchestrator.update_session_status(sessions[0][0], "researching", "Research in progress")
            await orchestrator.update_session_status(sessions[1][0], "completed", "Research completed")
            await orchestrator.update_session_status(sessions[2][0], "error", "Error occurred")

            # Verify updates didn't affect other sessions
            for session_id, topic, requirements in sessions:
                status = await orchestrator.get_session_status(session_id)
                assert status["topic"] == topic  # Topic unchanged
                assert status["user_requirements"] == requirements  # Requirements unchanged

            await orchestrator.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_workflow_history(self, temp_dir, monkeypatch):
        """Test workflow history tracking in sessions."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            session_id = await orchestrator.start_research_session(
                "Workflow History Test",
                {"depth": "Standard Research"}
            )

            # Simulate workflow progression
            workflow_stages = [
                ("research", 5),
                ("report_generation", 1),
                ("editorial_review", 3),
                ("finalization", 1)
            ]

            for stage, results_count in workflow_stages:
                # Update session status
                await orchestrator.update_session_status(session_id, stage, f"Executing {stage}")

                # Add to workflow history
                session_data = orchestrator.active_sessions[session_id]
                session_data["workflow_history"].append({
                    "stage": stage,
                    "completed_at": "2024-01-01T12:00:00",
                    "results_count": results_count
                })

            # Verify workflow history
            final_status = await orchestrator.get_session_status(session_id)
            history = final_status["workflow_history"]

            assert len(history) == len(workflow_stages)

            for i, (stage, expected_count) in enumerate(workflow_stages):
                assert history[i]["stage"] == stage
                assert history[i]["results_count"] == expected_count

            await orchestrator.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_cleanup(self, temp_dir, monkeypatch):
        """Test session cleanup and resource management."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # Create multiple sessions
            session_ids = []
            for i in range(3):
                session_id = await orchestrator.start_research_session(
                    f"Cleanup Test {i}",
                    {"depth": "Quick Overview"}
                )
                session_ids.append(session_id)

            # Verify sessions are active
            assert len(orchestrator.active_sessions) == 3
            assert len(orchestrator.agent_clients) == 4

            # Test individual session cleanup
            # Note: Individual session cleanup would need to be implemented
            # For now, we test full orchestrator cleanup

            # Test full cleanup
            await orchestrator.cleanup()

            # Verify all resources were cleaned up
            assert len(orchestrator.active_sessions) == 0
            assert len(orchestrator.agent_clients) == 0

            # Verify clients were disconnected
            assert mock_client.disconnect.call_count == 4

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_error_recovery(self, temp_dir, monkeypatch):
        """Test session recovery from error states."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            session_id = await orchestrator.start_research_session(
                "Error Recovery Test",
                {"depth": "Standard Research"}
            )

            # Simulate error during workflow
            await orchestrator.update_session_status(
                session_id,
                "error",
                "Research failed due to network issue"
            )

            # Verify error state
            status = await orchestrator.get_session_status(session_id)
            assert status["status"] == "error"
            assert "network issue" in status.get("status_message", "").lower()

            # Test recovery from error
            await orchestrator.update_session_status(
                session_id,
                "researching",
                "Resuming research after error recovery"
            )

            # Verify recovery
            recovered_status = await orchestrator.get_session_status(session_id)
            assert recovered_status["status"] == "researching"
            assert "resuming" in recovered_status.get("status_message", "").lower()

            await orchestrator.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_session_performance(self, temp_dir, monkeypatch):
        """Test session performance with many operations."""
        monkeypatch.chdir(temp_dir)

        with patch('core.orchestrator.ClaudeSDKClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.query = AsyncMock()
            mock_client_class.return_value = mock_client

            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # Test performance with many status updates
            session_id = await orchestrator.start_research_session(
                "Performance Test",
                {"depth": "Quick Overview"}
            )

            # Perform many status updates
            update_count = 50
            start_time = asyncio.get_event_loop().time()

            tasks = []
            for i in range(update_count):
                task = orchestrator.update_session_status(
                    session_id,
                    "researching",
                    f"Update {i}"
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            # Performance assertion - should handle 50 updates quickly
            assert duration < 5.0, f"Too slow: {duration:.2f}s for {update_count} updates"

            # Verify final state
            final_status = await orchestrator.get_session_status(session_id)
            assert final_status["status"] == "researching"

            await orchestrator.cleanup()
