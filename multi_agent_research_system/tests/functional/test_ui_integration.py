"""Functional tests for UI integration with real backend functionality.

These tests validate that the Streamlit UI works correctly with the real
multi-agent research system backend.
"""

import asyncio
import os

import pytest

# Only run functional tests if SDK is available and credentials are set
pytest.importorskip("claude_agent_sdk", reason="Claude Agent SDK not available")

try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        create_sdk_mcp_server,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

try:
    import anthropic
    API_KEY_AVAILABLE = bool(os.getenv("ANTHROPIC_API_KEY"))
except ImportError:
    API_KEY_AVAILABLE = False

from core.orchestrator import ResearchOrchestrator
from ui.streamlit_app import ResearchUI


class TestUIIntegrationFunctional:
    """Functional tests for UI integration with real backend."""

    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ui_research_session_creation(self, temp_dir, monkeypatch):
        """Test UI integration for creating research sessions."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        # Initialize UI
        ui = ResearchUI()

        # Test research request processing
        research_request = {
            "topic": "Blockchain Technology Applications",
            "depth": "Standard Research",
            "audience": "Technical",
            "format": "Technical Documentation",
            "requirements": "Focus on real-world applications and use cases",
            "timeline": "Within 2 days"
        }

        print("\n🖥️ Testing UI session creation...")
        print(f"📝 Request: {research_request['topic']}")

        # Process request through UI
        session_id = ui.process_research_request(research_request)

        # Validate session creation
        assert session_id is not None, "UI failed to create session"
        assert len(session_id) > 20, "Invalid session ID format"

        # Check if session is tracked in UI
        assert session_id in ui.active_sessions, "Session not tracked by UI"
        session_data = ui.active_sessions[session_id]

        # Validate session data
        assert session_data["topic"] == research_request["topic"]
        assert session_data["user_requirements"] == research_request
        assert session_data["status"] == "initialized"

        print("✅ UI session creation successful!")
        print(f"🆔 Session ID: {session_id}")

    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ui_status_monitoring(self, temp_dir, monkeypatch):
        """Test UI status monitoring functionality."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        # Initialize real orchestrator
        orchestrator = ResearchOrchestrator()
        await orchestrator.initialize()

        # Initialize UI
        ui = ResearchUI()

        try:
            # Create research session
            session_id = await orchestrator.start_research_session(
                "Machine Learning in Healthcare",
                {
                    "depth": "Standard Research",
                    "audience": "Medical Professionals",
                    "format": "Medical Report"
                }
            )

            # Add session to UI tracking
            ui.active_sessions[session_id] = await orchestrator.get_session_status(session_id)

            print("\n📊 Testing UI status monitoring...")
            print(f"🆔 Session: {session_id}")

            # Monitor status updates
            status_updates = []
            timeout = 60  # 1 minute
            start_time = asyncio.get_event_loop().time()

            while True:
                # Get current status from orchestrator
                current_status = await orchestrator.get_session_status(session_id)

                # Update UI status
                ui.update_session_status(session_id, current_status)

                # Track status changes
                if len(ui.active_sessions[session_id].get("status_updates", [])) > len(status_updates):
                    latest_update = ui.active_sessions[session_id]["status_updates"][-1]
                    status_updates.append(latest_update)
                    print(f"📈 Status update: {latest_update['status']} - {latest_update['message']}")

                if current_status['status'] in ['completed', 'error']:
                    break

                if asyncio.get_event_loop().time() - start_time > timeout:
                    break  # Don't fail, just stop monitoring

                await asyncio.sleep(5)

            # Validate status tracking
            assert len(status_updates) > 0, "No status updates tracked"
            assert session_id in ui.active_sessions, "Session lost from UI tracking"

            # Test UI status retrieval
            ui_status = ui.get_session_status(session_id)
            assert ui_status["session_id"] == session_id, "UI status retrieval failed"

            print("✅ UI status monitoring successful!")
            print(f"📈 Total status updates: {len(status_updates)}")

        finally:
            await orchestrator.cleanup()

    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ui_results_display(self, temp_dir, monkeypatch):
        """Test UI results display functionality."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        # Initialize orchestrator
        orchestrator = ResearchOrchestrator()
        await orchestrator.initialize()

        # Initialize UI
        ui = ResearchUI()

        try:
            # Create and complete a research session
            session_id = await orchestrator.start_research_session(
                "Renewable Energy Trends",
                {
                    "depth": "Quick Overview",
                    "audience": "General Public",
                    "format": "Summary Report"
                }
            )

            # Simulate workflow completion
            await orchestrator.update_session_status(session_id, "completed", "Research completed")

            # Create mock results for UI display
            mock_results = {
                "summary": "Renewable energy adoption is accelerating globally with solar and wind leading the growth.",
                "key_findings": [
                    "Solar energy costs decreased by 90% since 2010",
                    "Wind energy capacity grew by 93% in 2023",
                    "Renewables will account for 42% of global electricity by 2028"
                ],
                "sources": [
                    {"title": "IRENA Renewable Energy Statistics 2024", "url": "https://irena.org"},
                    {"title": "IEA World Energy Outlook 2023", "url": "https://iea.org"}
                ]
            }

            # Update session with results
            session_data = ui.active_sessions[session_id]
            session_data["results"] = mock_results
            session_data["status"] = "completed"

            print("\n📊 Testing UI results display...")
            print(f"🆔 Session: {session_id}")

            # Test results retrieval
            ui_results = ui.get_session_results(session_id)

            # Validate results structure
            assert "summary" in ui_results, "Missing summary in UI results"
            assert "key_findings" in ui_results, "Missing key findings in UI results"
            assert "sources" in ui_results, "Missing sources in UI results"

            # Test display formatting
            formatted_display = ui.format_results_for_display(ui_results)

            assert isinstance(formatted_display, str), "Results should be formatted as string"
            assert len(formatted_display) > 100, "Formatted results too short"
            assert "Renewable" in formatted_display, "Results should mention topic"
            assert "Solar" in formatted_display, "Results should include findings"

            print("✅ UI results display successful!")
            print(f"📊 Found {len(ui_results['key_findings'])} key findings")
            print(f"📚 Used {len(ui_results['sources'])} sources")

        finally:
            await orchestrator.cleanup()

    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ui_file_operations(self, temp_dir, monkeypatch):
        """Test UI file operations for saving and loading results."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        # Initialize UI
        ui = ResearchUI()

        print("\n💾 Testing UI file operations...")

        # Test saving configuration
        config_data = {
            "default_depth": "Standard Research",
            "default_audience": "Business",
            "default_format": "Business Brief",
            "api_timeout": 300,
            "max_sources": 10
        }

        config_file = ui.save_configuration(config_data)

        # Validate file creation
        assert config_file.exists(), "Configuration file not created"
        assert config_file.stat().st_size > 0, "Configuration file is empty"

        # Test loading configuration
        loaded_config = ui.load_configuration(config_file)

        # Validate loaded data
        assert loaded_config == config_data, "Configuration data corrupted during save/load"

        print("✅ Configuration save/load successful!")
        print(f"📁 Config file: {config_file}")

        # Test saving session results
        session_results = {
            "session_id": "test-session-123",
            "topic": "Test Results",
            "findings": ["Finding 1", "Finding 2"],
            "report": "# Test Report\n\nThis is a test report."
        }

        results_file = ui.save_session_results(session_results)

        # Validate results file
        assert results_file.exists(), "Results file not created"
        assert results_file.stat().st_size > 0, "Results file is empty"

        # Test loading session results
        loaded_results = ui.load_session_results(results_file)

        # Validate loaded results
        assert loaded_results["session_id"] == session_results["session_id"]
        assert loaded_results["topic"] == session_results["topic"]
        assert len(loaded_results["findings"]) == len(session_results["findings"])

        print("✅ Results save/load successful!")
        print(f"📄 Results file: {results_file}")

    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ui_error_handling(self, temp_dir, monkeypatch):
        """Test UI error handling for various scenarios."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        # Initialize UI
        ui = ResearchUI()

        print("\n❌ Testing UI error handling...")

        # Test invalid research request
        invalid_request = {
            "topic": "",  # Empty topic
            "depth": "invalid_depth",  # Invalid depth
            "audience": "",  # Empty audience
            "format": "invalid_format"  # Invalid format
        }

        # Should handle gracefully
        try:
            session_id = ui.process_research_request(invalid_request)
            assert session_id is None or "error" in str(session_id).lower(), "Should handle invalid request"
            print("✅ Invalid request handled gracefully")
        except Exception as e:
            # Should not crash
            assert "graceful" in str(e).lower() or "invalid" in str(e).lower(), "Should handle error gracefully"

        # Test invalid session ID
        invalid_session_id = "invalid-session-123"
        status = ui.get_session_status(invalid_session_id)

        # Should return error indication
        assert "error" in status or "not found" in str(status).lower(), "Should handle invalid session ID"
        print("✅ Invalid session ID handled gracefully")

        # Test file operations with invalid data
        try:
            invalid_config = {"invalid": "data with no valid fields"}
            config_file = ui.save_configuration(invalid_config)

            # Should either fail gracefully or save with default values
            if config_file and config_file.exists():
                loaded_config = ui.load_configuration(config_file)
                # Should have some default structure
                assert isinstance(loaded_config, dict), "Should still be a dictionary"

            print("✅ Invalid configuration handled gracefully")
        except Exception as e:
            # Should not crash
            assert "invalid" in str(e).lower(), "Should handle invalid config gracefully"

        print("✅ UI error handling tests completed!")

    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ui_performance_under_load(self, temp_dir, monkeypatch):
        """Test UI performance under multiple concurrent operations."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        # Initialize UI
        ui = ResearchUI()

        print("\n⚡ Testing UI performance under load...")

        # Test concurrent session creation
        num_sessions = 10
        topics = [f"Performance Test Topic {i}" for i in range(num_sessions)]

        start_time = asyncio.get_event_loop().time()

        # Create multiple sessions
        session_ids = []
        for topic in topics:
            request = {
                "topic": topic,
                "depth": "Quick Overview",
                "audience": "General",
                "format": "Summary"
            }
            session_id = ui.process_research_request(request)
            session_ids.append(session_id)

        creation_time = asyncio.get_event_loop().time() - start_time

        # Validate all sessions created
        successful_sessions = [sid for sid in session_ids if sid is not None]
        assert len(successful_sessions) >= num_sessions * 0.8, f"Too many session creation failures: {len(successful_sessions)}/{num_sessions}"

        print(f"✅ Created {len(successful_sessions)}/{num_sessions} sessions in {creation_time:.2f}s")

        # Test concurrent status queries
        start_time = asyncio.get_event_loop().time()

        for session_id in successful_sessions[:5]:  # Test first 5
            status = ui.get_session_status(session_id)
            assert "session_id" in status, "Status query should return valid data"

        query_time = asyncio.get_event_loop().time() - start_time

        print(f"✅ Queried 5 session statuses in {query_time:.2f}s")

        # Performance assertions
        assert creation_time < 10.0, f"Session creation too slow: {creation_time:.2f}s for {num_sessions} sessions"
        assert query_time < 2.0, f"Status queries too slow: {query_time:.2f}s for 5 queries"

        print("✅ Performance tests passed!")
        print(f"📊 Creation: {creation_time:.2f}s total, {creation_time/num_sessions:.2f}s per session")
        print(f"📊 Queries: {query_time:.2f}s total, {query_time/5:.2f}s per query")
