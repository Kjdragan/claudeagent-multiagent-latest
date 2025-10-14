"""
Comprehensive Test Suite for KEVIN Session Management System.

Phase 3.5: Implement Session Management with KEVIN directory structure

This test suite validates the enhanced session management system including:
- KEVIN directory structure creation and management
- Session lifecycle management with persistence
- Sub-session coordination for gap research
- File management and tracking
- Session metadata handling
- Integration with workflow state management
"""

import asyncio
import json
import os
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Import the KEVIN session management components
from multi_agent_research_system.core.kevin_session_manager import (
    KevinSessionManager,
    SessionMetadata,
    SubSessionInfo,
    SessionStatus,
    DataType,
    create_kevin_session
)

from multi_agent_research_system.core.workflow_state import WorkflowStage, StageStatus


class TestKevinSessionManager:
    """Test cases for the KEVIN Session Management System."""

    @pytest.fixture
    def temp_kevin_dir(self):
        """Create a temporary directory for KEVIN structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def kevin_manager(self, temp_kevin_dir):
        """Create a KEVIN session manager instance for testing."""
        manager = KevinSessionManager(temp_kevin_dir)
        return manager

    @pytest.fixture
    def sample_user_requirements(self):
        """Sample user requirements for testing."""
        return {
            "depth": "Comprehensive Analysis",
            "audience": "Academic",
            "format": "Detailed Report",
            "quality_threshold": 0.8,
            "max_sources": 20,
            "include_gap_research": True
        }

    @pytest.fixture
    def sample_content(self):
        """Sample content for testing."""
        return """
        # Test Research Content

        This is a sample research content for testing the KEVIN session management system.

        ## Introduction
        The KEVIN system provides comprehensive session management capabilities.

        ## Key Features
        - Session-based organization
        - Sub-session coordination
        - File management and tracking
        - Quality assurance integration

        ## Conclusion
        The system ensures organized and efficient research workflow management.
        """

    def test_manager_initialization(self, temp_kevin_dir):
        """Test KEVIN session manager initialization."""
        manager = KevinSessionManager(temp_kevin_dir)

        # Verify directory structure creation
        assert manager.kevin_base_path == Path(temp_kevin_dir)
        assert manager.sessions_path.exists()
        assert manager.logs_path.exists()
        assert manager.reports_path.exists()
        assert manager.work_products_path.exists()
        assert manager.monitoring_path.exists()

        # Verify internal state
        assert isinstance(manager.active_sessions, dict)
        assert isinstance(manager.session_workflows, dict)
        assert isinstance(manager.sub_sessions, dict)
        assert isinstance(manager.parent_child_links, dict)
        assert isinstance(manager.file_mappings, dict)

    @pytest.mark.asyncio
    async def test_create_session(self, kevin_manager, sample_user_requirements):
        """Test session creation functionality."""
        topic = "artificial intelligence in healthcare"

        session_id = await kevin_manager.create_session(topic, sample_user_requirements)

        # Verify session creation
        assert session_id is not None
        assert session_id in kevin_manager.active_sessions

        # Verify session metadata
        metadata = kevin_manager.active_sessions[session_id]
        assert metadata.session_id == session_id
        assert metadata.topic == topic
        assert metadata.user_requirements == sample_user_requirements
        assert metadata.status == SessionStatus.ACTIVE
        assert metadata.workflow_stage == WorkflowStage.RESEARCH
        assert metadata.parent_session_id is None
        assert metadata.sub_session_count == 0

        # Verify workflow session creation
        assert session_id in kevin_manager.session_workflows
        workflow_session = kevin_manager.session_workflows[session_id]
        assert workflow_session.session_id == session_id
        assert workflow_session.topic == topic
        assert workflow_session.current_stage == WorkflowStage.RESEARCH

        # Verify directory structure creation
        session_path = kevin_manager.sessions_path / session_id
        assert session_path.exists()
        assert (session_path / "working").exists()
        assert (session_path / "research").exists()
        assert (session_path / "complete").exists()
        assert (session_path / "agent_logs").exists()
        assert (session_path / "quality_reports").exists()
        assert (session_path / "sub_sessions").exists()

        # Verify metadata file creation
        metadata_file = session_path / "session_metadata.json"
        assert metadata_file.exists()

    @pytest.mark.asyncio
    async def test_create_session_with_custom_id(self, kevin_manager, sample_user_requirements):
        """Test session creation with custom session ID."""
        custom_session_id = "test_session_123"
        topic = "quantum computing applications"

        session_id = await kevin_manager.create_session(
            topic, sample_user_requirements, custom_session_id
        )

        assert session_id == custom_session_id
        assert session_id in kevin_manager.active_sessions

    @pytest.mark.asyncio
    async def test_update_session_status(self, kevin_manager, sample_user_requirements):
        """Test session status updates."""
        session_id = await kevin_manager.create_session("test topic", sample_user_requirements)

        # Update to completed
        await kevin_manager.update_session_status(
            session_id, SessionStatus.COMPLETED, WorkflowStage.FINAL_OUTPUT
        )

        # Verify status update
        metadata = kevin_manager.active_sessions[session_id]
        assert metadata.status == SessionStatus.COMPLETED
        assert metadata.workflow_stage == WorkflowStage.FINAL_OUTPUT

        # Verify workflow session update
        workflow_session = kevin_manager.session_workflows[session_id]
        assert workflow_session.overall_status == StageStatus.COMPLETED
        assert workflow_session.end_time is not None

    @pytest.mark.asyncio
    async def test_create_sub_session(self, kevin_manager, sample_user_requirements):
        """Test sub-session creation for gap research."""
        # Create parent session
        parent_session_id = await kevin_manager.create_session("main research", sample_user_requirements)

        # Create sub-session
        gap_topic = "temporal gaps in AI research"
        sub_session_id = await kevin_manager.create_sub_session(gap_topic, parent_session_id)

        # Verify sub-session creation
        assert sub_session_id is not None
        assert sub_session_id in kevin_manager.active_sessions
        assert sub_session_id in kevin_manager.sub_sessions

        # Verify sub-session metadata
        sub_metadata = kevin_manager.active_sessions[sub_session_id]
        assert sub_metadata.session_id == sub_session_id
        assert sub_metadata.topic == gap_topic
        assert sub_metadata.parent_session_id == parent_session_id
        assert sub_metadata.status == SessionStatus.ACTIVE

        # Verify sub-session info
        sub_info = kevin_manager.sub_sessions[sub_session_id]
        assert sub_info.sub_session_id == sub_session_id
        assert sub_info.parent_session_id == parent_session_id
        assert sub_info.gap_topic == gap_topic
        assert sub_info.status == SessionStatus.ACTIVE

        # Verify parent-child link
        assert parent_session_id in kevin_manager.parent_child_links
        assert sub_session_id in kevin_manager.parent_child_links[parent_session_id]

        # Verify parent session update
        parent_metadata = kevin_manager.active_sessions[parent_session_id]
        assert parent_metadata.sub_session_count == 1

        # Verify sub-session directory structure
        sub_session_path = kevin_manager.sessions_path / sub_session_id
        assert sub_session_path.exists()
        assert (sub_session_path / "research" / "gap_research").exists()

    @pytest.mark.asyncio
    async def test_store_session_file(self, kevin_manager, sample_user_requirements, sample_content):
        """Test file storage in session directories."""
        session_id = await kevin_manager.create_session("test topic", sample_user_requirements)

        # Store different types of files
        working_file = await kevin_manager.store_session_file(
            session_id, DataType.WORKING, sample_content, "test_working.md"
        )

        research_file = await kevin_manager.store_session_file(
            session_id, DataType.RESEARCH, sample_content, "test_research.md"
        )

        # Verify file storage
        assert working_file is not None
        assert research_file is not None
        assert Path(working_file).exists()
        assert Path(research_file).exists()

        # Verify file paths are in correct directories
        session_path = kevin_manager.sessions_path / session_id
        assert Path(working_file).parent == session_path / "working"
        assert Path(research_file).parent == session_path / "research"

        # Verify file mappings
        assert session_id in kevin_manager.file_mappings
        assert "test_working.md" in kevin_manager.file_mappings[session_id]
        assert "test_research.md" in kevin_manager.file_mappings[session_id]

    @pytest.mark.asyncio
    async def test_store_session_file_with_metadata(self, kevin_manager, sample_user_requirements, sample_content):
        """Test file storage with metadata."""
        session_id = await kevin_manager.create_session("test topic", sample_user_requirements)

        file_metadata = {
            "author": "test_agent",
            "quality_score": 0.85,
            "processing_time": 2.5,
            "stage": "research"
        }

        file_path = await kevin_manager.store_session_file(
            session_id, DataType.WORKING, sample_content, "test_with_metadata.md", file_metadata
        )

        # Verify file storage
        assert file_path is not None
        assert Path(file_path).exists()

        # Verify metadata file creation
        metadata_file = Path(file_path).with_suffix('.json')
        assert metadata_file.exists()

        # Load and verify metadata
        with open(metadata_file, 'r') as f:
            stored_metadata = json.load(f)

        assert stored_metadata["session_id"] == session_id
        assert stored_metadata["data_type"] == DataType.WORKING.value
        assert stored_metadata["metadata"] == file_metadata

    @pytest.mark.asyncio
    async def test_get_session_file(self, kevin_manager, sample_user_requirements, sample_content):
        """Test file retrieval from session directories."""
        session_id = await kevin_manager.create_session("test topic", sample_user_requirements)

        # Store a file
        original_content = sample_content
        filename = "test_retrieval.md"
        file_path = await kevin_manager.store_session_file(
            session_id, DataType.RESEARCH, original_content, filename
        )

        # Retrieve the file
        retrieved_content = await kevin_manager.get_session_file(session_id, filename)

        # Verify retrieval
        assert retrieved_content is not None
        assert retrieved_content == original_content

    @pytest.mark.asyncio
    async def test_get_session_files(self, kevin_manager, sample_user_requirements, sample_content):
        """Test getting all files for a session."""
        session_id = await kevin_manager.create_session("test topic", sample_user_requirements)

        # Store multiple files of different types
        await kevin_manager.store_session_file(session_id, DataType.WORKING, sample_content, "working_file.md")
        await kevin_manager.store_session_file(session_id, DataType.RESEARCH, sample_content, "research_file.md")
        await kevin_manager.store_session_file(session_id, DataType.COMPLETE, sample_content, "complete_file.md")

        # Get all files
        all_files = await kevin_manager.get_session_files(session_id)

        # Verify file retrieval
        assert len(all_files) >= 3
        assert "working_file.md" in all_files
        assert "research_file.md" in all_files
        assert "complete_file.md" in all_files

        # Get files by type
        research_files = await kevin_manager.get_session_files(session_id, DataType.RESEARCH)
        assert len(research_files) >= 1
        assert "research_file.md" in research_files

    @pytest.mark.asyncio
    async def test_integrate_sub_session_results(self, kevin_manager, sample_user_requirements):
        """Test sub-session result integration."""
        # Create parent session
        parent_session_id = await kevin_manager.create_session("main research", sample_user_requirements)

        # Create sub-sessions
        sub_session1_id = await kevin_manager.create_sub_session("gap topic 1", parent_session_id)
        sub_session2_id = await kevin_manager.create_sub_session("gap topic 2", parent_session_id)

        # Prepare integration data
        integration_data = {
            sub_session1_id: {
                "quality_score": 0.85,
                "research_findings": "Gap research findings for topic 1",
                "additional_sources": 3
            },
            sub_session2_id: {
                "quality_score": 0.78,
                "research_findings": "Gap research findings for topic 2",
                "additional_sources": 2
            }
        }

        # Integrate results
        integration_result = await kevin_manager.integrate_sub_session_results(parent_session_id, integration_data)

        # Verify integration
        assert integration_result["parent_session_id"] == parent_session_id
        assert len(integration_result["integrated_results"]) == 2
        assert integration_result["total_sub_sessions"] == 2
        assert integration_result["successful_integrations"] == 2

        # Verify quality analysis
        quality_analysis = integration_result["quality_analysis"]
        assert len(quality_analysis) == 2
        assert all(item["quality_score"] > 0 for item in quality_analysis)

        # Verify integration quality metrics
        integration_quality = integration_result["integration_quality"]
        assert "overall_score" in integration_quality
        assert "average_quality" in integration_quality
        assert "completion_rate" in integration_quality

        # Verify sub-session status updates
        sub_info1 = kevin_manager.sub_sessions[sub_session1_id]
        sub_info2 = kevin_manager.sub_sessions[sub_session2_id]
        assert sub_info1.status == SessionStatus.COMPLETED
        assert sub_info2.status == SessionStatus.COMPLETED
        assert sub_info1.integration_status == "completed"
        assert sub_info2.integration_status == "completed"

    @pytest.mark.asyncio
    async def test_get_active_sessions(self, kevin_manager, sample_user_requirements):
        """Test getting active sessions."""
        # Create sessions with different statuses
        session1_id = await kevin_manager.create_session("topic 1", sample_user_requirements)
        session2_id = await kevin_manager.create_session("topic 2", sample_user_requirements)

        # Update one session status
        await kevin_manager.update_session_status(session1_id, SessionStatus.COMPLETED)

        # Get all active sessions
        all_sessions = await kevin_manager.get_active_sessions()
        assert len(all_sessions) == 2

        # Get sessions filtered by status
        active_sessions = await kevin_manager.get_active_sessions(SessionStatus.ACTIVE)
        completed_sessions = await kevin_manager.get_active_sessions(SessionStatus.COMPLETED)

        assert len(active_sessions) == 1
        assert len(completed_sessions) == 1
        assert active_sessions[0].session_id == session2_id
        assert completed_sessions[0].session_id == session1_id

    @pytest.mark.asyncio
    async def test_get_session_summary(self, kevin_manager, sample_user_requirements, sample_content):
        """Test getting session summary."""
        session_id = await kevin_manager.create_session("test topic", sample_user_requirements)

        # Store some files
        await kevin_manager.store_session_file(session_id, DataType.WORKING, sample_content, "summary_test.md")
        await kevin_manager.store_session_file(session_id, DataType.RESEARCH, sample_content, "research_summary.md")

        # Create a sub-session
        sub_session_id = await kevin_manager.create_sub_session("gap topic", session_id)

        # Get session summary
        summary = await kevin_manager.get_session_summary(session_id)

        # Verify summary structure
        assert "session_metadata" in summary
        assert "file_statistics" in summary
        assert "sub_sessions" in summary
        assert "workflow_stage" in summary
        assert "session_duration" in summary
        assert "last_activity" in summary

        # Verify session metadata
        metadata = summary["session_metadata"]
        assert metadata["session_id"] == session_id
        assert metadata["topic"] == "test topic"

        # Verify file statistics
        file_stats = summary["file_statistics"]
        assert file_stats["total_files"] >= 2
        assert file_stats["working_files"] >= 1
        assert file_stats["research_files"] >= 1

        # Verify sub-session information
        sub_sessions = summary["sub_sessions"]
        assert len(sub_sessions) == 1
        assert sub_sessions[0]["sub_session_id"] == sub_session_id

    @pytest.mark.asyncio
    async def test_archive_session(self, kevin_manager, sample_user_requirements):
        """Test session archiving."""
        session_id = await kevin_manager.create_session("test topic", sample_user_requirements)

        # Complete the session
        await kevin_manager.update_session_status(session_id, SessionStatus.COMPLETED)

        # Archive the session
        archive_result = await kevin_manager.archive_session(session_id)

        # Verify archiving
        assert archive_result is True
        assert session_id not in kevin_manager.active_sessions
        assert session_id not in kevin_manager.session_workflows

        # Verify archived directory
        archive_path = kevin_manager.sessions_path / "archived" / session_id
        assert archive_path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, kevin_manager, sample_user_requirements):
        """Test cleanup of expired sessions."""
        # Create sessions
        session1_id = await kevin_manager.create_session("topic 1", sample_user_requirements)
        session2_id = await kevin_manager.create_session("topic 2", sample_user_requirements)

        # Complete sessions
        await kevin_manager.update_session_status(session1_id, SessionStatus.COMPLETED)
        await kevin_manager.update_session_status(session2_id, SessionStatus.FAILED)

        # Mock expired sessions by manually setting old creation times
        old_time = datetime.now() - timedelta(hours=25)
        kevin_manager.active_sessions[session1_id].created_at = old_time
        kevin_manager.active_sessions[session2_id].created_at = old_time

        # Clean up expired sessions
        cleaned_count = await kevin_manager.cleanup_expired_sessions(max_age_hours=24)

        # Verify cleanup
        assert cleaned_count == 2
        assert session1_id not in kevin_manager.active_sessions
        assert session2_id not in kevin_manager.active_sessions

    @pytest.mark.asyncio
    async def test_get_kevin_statistics(self, kevin_manager, sample_user_requirements):
        """Test KEVIN system statistics."""
        # Create sessions
        session1_id = await kevin_manager.create_session("topic 1", sample_user_requirements)
        session2_id = await kevin_manager.create_session("topic 2", sample_user_requirements)

        # Create sub-sessions
        sub_session1_id = await kevin_manager.create_sub_session("gap 1", session1_id)
        sub_session2_id = await kevin_manager.create_sub_session("gap 2", session1_id)

        # Get statistics
        stats = await kevin_manager.get_kevin_statistics()

        # Verify statistics
        assert "total_sessions" in stats
        assert "status_distribution" in stats
        assert "stage_distribution" in stats
        assert "total_files" in stats
        assert "total_size_bytes" in stats
        assert "sub_sessions" in stats
        assert "parent_sessions" in stats
        assert "kevin_directory" in stats
        assert "sessions_directory" in stats
        assert "last_updated" in stats

        assert stats["total_sessions"] == 2
        assert stats["sub_sessions"] == 2
        assert stats["parent_sessions"] == 1

    @pytest.mark.asyncio
    async def test_load_session_metadata(self, kevin_manager, sample_user_requirements):
        """Test loading session metadata from file."""
        # Create session
        session_id = await kevin_manager.create_session("test topic", sample_user_requirements)

        # Modify metadata in memory
        metadata = kevin_manager.active_sessions[session_id]
        metadata.session_state["test_field"] = "test_value"

        # Create new manager instance to simulate restart
        new_manager = KevinSessionManager(str(kevin_manager.kevin_base_path))

        # Load metadata from file
        loaded_metadata = await new_manager.load_session_metadata(session_id)

        # Verify loaded metadata
        assert loaded_metadata is not None
        assert loaded_metadata.session_id == session_id
        assert loaded_metadata.topic == "test topic"
        assert loaded_metadata.status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_error_handling(self, kevin_manager):
        """Test error handling in various scenarios."""
        # Test getting non-existent session
        non_existent_summary = await kevin_manager.get_session_summary("non_existent")
        assert non_existent_summary is None

        # Test storing file for non-existent session
        with pytest.raises(ValueError):
            await kevin_manager.store_session_file(
                "non_existent", DataType.WORKING, "content"
            )

        # Test creating sub-session for non-existent parent
        with pytest.raises(ValueError):
            await kevin_manager.create_sub_session("gap topic", "non_existent")

        # Test integrating sub-sessions for non-existent parent
        result = await kevin_manager.integrate_sub_session_results("non_existent", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_convenience_function(self, temp_kevin_dir, sample_user_requirements):
        """Test the convenience function for session creation."""
        topic = "convenience test topic"
        session_id = await create_kevin_session(topic, sample_user_requirements, temp_kevin_dir)

        # Verify convenience function worked
        assert session_id is not None
        assert len(session_id) > 0

        # Verify manager was created and session exists
        manager = KevinSessionManager(temp_kevin_dir)
        assert session_id in manager.active_sessions
        assert manager.active_sessions[session_id].topic == topic

    @pytest.mark.asyncio
    async def test_persistence_across_manager_instances(self, temp_kevin_dir, sample_user_requirements):
        """Test that session data persists across manager instances."""
        # Create session with first manager
        manager1 = KevinSessionManager(temp_kevin_dir)
        session_id = await manager1.create_session("persistence test", sample_user_requirements)

        # Store some data
        await manager1.store_session_file(
            session_id, DataType.WORKING, "test content", "persistence_test.md"
        )

        # Create second manager instance
        manager2 = KevinSessionManager(temp_kevin_dir)

        # Load session metadata
        metadata = await manager2.load_session_metadata(session_id)

        # Verify persistence
        assert metadata is not None
        assert metadata.session_id == session_id
        assert metadata.topic == "persistence test"

        # Verify file persistence
        content = await manager2.get_session_file(session_id, "persistence_test.md")
        assert content == "test content"

    @pytest.mark.asyncio
    async def test_sub_session_workflow_integration(self, kevin_manager, sample_user_requirements):
        """Test complete sub-session workflow integration."""
        # Create parent session
        parent_session_id = await kevin_manager.create_session("parent research", sample_user_requirements)

        # Create multiple sub-sessions for different gap topics
        gap_topics = ["temporal gaps", "comparative analysis", "methodological gaps"]
        sub_session_ids = []

        for gap_topic in gap_topics:
            sub_id = await kevin_manager.create_sub_session(gap_topic, parent_session_id)
            sub_session_ids.append(sub_id)

            # Store research data for each sub-session
            research_content = f"Gap research content for {gap_topic}"
            await kevin_manager.store_session_file(
                sub_id, DataType.RESEARCH, research_content, f"gap_research_{gap_topic.replace(' ', '_')}.md"
            )

        # Simulate sub-session completion and quality assessment
        integration_data = {}
        for i, sub_id in enumerate(sub_session_ids):
            integration_data[sub_id] = {
                "quality_score": 0.75 + (i * 0.05),  # Different quality scores
                "gap_findings": f"Key findings for {gap_topics[i]}",
                "research_duration": 300 + (i * 60),
                "sources_found": 5 + i
            }

        # Integrate all sub-session results
        integration_result = await kevin_manager.integrate_sub_session_results(parent_session_id, integration_data)

        # Verify complete integration
        assert integration_result["successful_integrations"] == 3
        assert integration_result["total_sub_sessions"] == 3

        # Get comprehensive session summary
        summary = await kevin_manager.get_session_summary(parent_session_id)

        # Verify summary includes sub-session information
        assert len(summary["sub_sessions"]) == 3
        assert summary["session_metadata"]["sub_session_count"] == 3

        # Verify all sub-sessions are properly tracked
        for sub_id in sub_session_ids:
            assert sub_id in kevin_manager.sub_sessions
            assert kevin_manager.sub_sessions[sub_id].status == SessionStatus.COMPLETED


class TestKevinSessionManagerIntegration:
    """Integration tests for KEVIN Session Manager with other system components."""

    @pytest.mark.asyncio
    async def test_workflow_state_integration(self, temp_kevin_dir):
        """Test integration with workflow state management."""
        from multi_agent_research_system.core.workflow_state import WorkflowSession

        manager = KevinSessionManager(temp_kevin_dir)

        user_requirements = {
            "depth": "Standard Research",
            "audience": "Technical",
            "format": "Detailed Report"
        }

        # Create session through KEVIN manager
        session_id = await manager.create_session("integration test", user_requirements)

        # Verify workflow session integration
        assert session_id in manager.session_workflows
        workflow_session = manager.session_workflows[session_id]

        # Update workflow through KEVIN manager
        await manager.update_session_status(
            session_id, SessionStatus.ACTIVE, WorkflowStage.REPORT_GENERATION
        )

        # Verify workflow state update
        assert workflow_session.current_stage == WorkflowStage.REPORT_GENERATION
        assert workflow_session.overall_status == StageStatus.PENDING

        # Complete workflow
        await manager.update_session_status(
            session_id, SessionStatus.COMPLETED, WorkflowStage.COMPLETED
        )

        # Verify workflow completion
        assert workflow_session.overall_status == StageStatus.COMPLETED
        assert workflow_session.is_completed is True

    @pytest.mark.asyncio
    async def test_quality_assurance_integration(self, temp_kevin_dir):
        """Test integration with quality assurance framework."""
        from multi_agent_research_system.core.quality_assurance_framework import QualityAssuranceFramework

        manager = KevinSessionManager(temp_kevin_dir)
        qa_framework = QualityAssuranceFramework()

        # Create session
        user_requirements = {"quality_threshold": 0.8, "enable_enhancement": True}
        session_id = await manager.create_session("QA integration test", user_requirements)

        # Create quality report
        quality_report = await qa_framework.generate_comprehensive_quality_report(session_id)

        # Store quality report in KEVIN structure
        report_content = json.dumps(quality_report.to_dict(), indent=2)
        report_path = await manager.store_session_file(
            session_id, DataType.QUALITY_REPORTS, report_content, "quality_assessment_report.json"
        )

        # Verify quality report storage
        assert report_path is not None
        assert Path(report_path).exists()

        # Verify report retrieval
        retrieved_report = await manager.get_session_file(
            session_id, "quality_assessment_report.json", DataType.QUALITY_REPORTS
        )
        assert retrieved_report is not None
        assert json.loads(retrieved_report)["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_gap_research_enforcement_integration(self, temp_kevin_dir):
        """Test integration with gap research enforcement system."""
        manager = KevinSessionManager(temp_kevin_dir)

        # Create parent session
        user_requirements = {"gap_research_enabled": True, "max_gap_topics": 2}
        parent_session_id = await manager.create_session("gap research test", user_requirements)

        # Simulate gap research enforcement identifying gaps
        identified_gaps = ["recent developments", "comparative analysis"]

        # Create sub-sessions for each gap
        sub_sessions = []
        for gap_topic in identified_gaps:
            sub_id = await manager.create_sub_session(gap_topic, parent_session_id)
            sub_sessions.append(sub_id)

            # Store gap research enforcement data
            enforcement_data = {
                "gap_type": gap_topic,
                "enforcement_action": "AUTO_EXECUTION",
                "compliance_level": "HIGH",
                "quality_requirements": ["completeness", "accuracy", "relevance"],
                "enforcement_timestamp": datetime.now().isoformat()
            }

            enforcement_content = json.dumps(enforcement_data, indent=2)
            await manager.store_session_file(
                sub_id, DataType.RESEARCH, enforcement_content,
                f"gap_enforcement_{gap_topic.replace(' ', '_')}.json"
            )

        # Verify gap research enforcement integration
        assert len(sub_sessions) == 2
        assert manager.active_sessions[parent_session_id].sub_session_count == 2

        # Verify enforcement data storage
        for sub_id in sub_sessions:
            enforcement_files = await manager.get_session_files(sub_id, DataType.RESEARCH)
            assert len(enforcement_files) >= 1

    @pytest.mark.asyncio
    async def test_progressive_enhancement_integration(self, temp_kevin_dir):
        """Test integration with progressive enhancement system."""
        from multi_agent_research_system.core.progressive_enhancement import ProgressiveEnhancementPipeline

        manager = KevinSessionManager(temp_kevin_dir)
        enhancement_pipeline = ProgressiveEnhancementPipeline()

        # Create session
        user_requirements = {"progressive_enhancement": True, "target_quality": 85}
        session_id = await manager.create_session("enhancement test", user_requirements)

        # Store original content
        original_content = "This is basic content that needs enhancement."
        await manager.store_session_file(
            session_id, DataType.WORKING, original_content, "original_content.md"
        )

        # Apply progressive enhancement (simulated)
        enhanced_content = """
        # Enhanced Content Analysis

        This is enhanced content that has been improved through progressive enhancement.

        ## Introduction
        The content has been systematically improved through multiple enhancement stages.

        ## Key Improvements
        - Enhanced structure with proper headings
        - Improved clarity and readability
        - Added comprehensive analysis
        - Expanded coverage of the topic

        ## Conclusion
        The progressive enhancement process has significantly improved content quality.
        """

        # Store enhanced content
        await manager.store_session_file(
            session_id, DataType.WORKING, enhanced_content, "enhanced_content.md"
        )

        # Store enhancement log
        enhancement_log = {
            "original_score": 65,
            "enhanced_score": 85,
            "improvement": 20,
            "stages_applied": ["structural_enhancement", "clarity_enhancement", "depth_enhancement"],
            "processing_time": 12.5,
            "enhancement_timestamp": datetime.now().isoformat()
        }

        enhancement_content = json.dumps(enhancement_log, indent=2)
        await manager.store_session_file(
            session_id, DataType.WORKING, enhancement_content, "enhancement_log.json"
        )

        # Verify enhancement integration
        original_file = await manager.get_session_file(session_id, "original_content.md")
        enhanced_file = await manager.get_session_file(session_id, "enhanced_content.md")
        enhancement_log_file = await manager.get_session_file(session_id, "enhancement_log.json")

        assert original_file is not None
        assert enhanced_file is not None
        assert enhancement_log_file is not None
        assert len(enhanced_file) > len(original_file)  # Enhanced content should be longer

    @pytest.mark.asyncio
    async def test_end_to_end_research_workflow(self, temp_kevin_dir):
        """Test complete end-to-end research workflow integration."""
        manager = KevinSessionManager(temp_kevin_dir)

        # Research requirements
        user_requirements = {
            "depth": "Comprehensive Analysis",
            "audience": "Academic",
            "format": "Detailed Report",
            "quality_threshold": 0.8,
            "gap_research_enabled": True,
            "progressive_enhancement": True
        }

        # Stage 1: Create main research session
        session_id = await manager.create_session(
            "artificial intelligence in healthcare 2024", user_requirements
        )

        # Stage 2: Store initial research data
        research_data = """
        # Initial Research Findings

        ## Key Research Areas
        1. AI in medical diagnosis
        2. Machine learning for treatment optimization
        3. Natural language processing in healthcare
        4. Computer vision for medical imaging
        5. Predictive analytics for patient care

        ## Current Findings
        AI technologies are rapidly transforming healthcare delivery...
        """
        await manager.store_session_file(
            session_id, DataType.RESEARCH, research_data, "initial_research.md"
        )

        # Stage 3: Generate first draft
        draft_content = """
        # AI in Healthcare: Analysis Report

        ## Executive Summary
        Artificial intelligence is revolutionizing healthcare delivery systems across multiple domains.

        ## Introduction
        Healthcare organizations are increasingly adopting AI technologies...

        ## Main Analysis
        Based on initial research findings, several key areas emerge...

        ## Conclusion
        AI presents significant opportunities for healthcare improvement.
        """
        await manager.store_session_file(
            session_id, DataType.WORKING, draft_content, "first_draft.md"
        )

        # Stage 4: Identify gaps and create sub-sessions
        identified_gaps = ["regulatory compliance", "implementation challenges"]
        sub_session_ids = []

        for gap_topic in identified_gaps:
            sub_id = await manager.create_sub_session(gap_topic, session_id)
            sub_session_ids.append(sub_id)

            # Store gap research results
            gap_research = f"""
            # Gap Research: {gap_topic}

            ## Key Findings for {gap_topic}
            Detailed analysis of {gap_topic} challenges and opportunities...

            ## Implementation Considerations
            Strategic recommendations for addressing {gap_topic}...

            ## Conclusion
            {gap_topic} requires careful consideration in AI healthcare implementation.
            """
            await manager.store_session_file(
                sub_id, DataType.RESEARCH, gap_research, f"gap_research_{gap_topic.replace(' ', '_')}.md"
            )

        # Stage 5: Integrate sub-session results
        integration_data = {
            sub_session_ids[0]: {
                "quality_score": 0.88,
                "key_insights": "Regulatory compliance frameworks for healthcare AI",
                "implementation_roadmap": "Step-by-step compliance guidance"
            },
            sub_session_ids[1]: {
                "quality_score": 0.82,
                "key_insights": "Common implementation challenges and solutions",
                "best_practices": "Proven strategies for successful deployment"
            }
        }

        integration_result = await manager.integrate_sub_session_results(session_id, integration_data)

        # Stage 6: Generate final report
        final_report = f"""
        # Comprehensive Analysis: AI in Healthcare 2024

        ## Executive Summary
        This comprehensive analysis examines the current state and future prospects of artificial intelligence in healthcare, drawing on extensive research and gap analysis.

        {draft_content}

        ## Gap Research Integration
        Based on targeted gap research, the following critical areas have been addressed:

        ### Regulatory Compliance
        {integration_data['quality_analysis']['average_quality']:.1f}% quality score achieved
        - Compliance frameworks and regulatory requirements
        - Risk assessment and mitigation strategies
        - Implementation roadmaps for healthcare organizations

        ### Implementation Challenges
        - Common barriers and solutions
        - Best practices from successful implementations
        - Change management considerations

        ## Conclusions and Recommendations
        The integration of AI in healthcare presents transformative potential while requiring careful attention to compliance and implementation challenges.

        **Quality Assessment**: {integration_result['integration_quality']['overall_score']:.2f}
        **Research Coverage**: Comprehensive with targeted gap research integration
        **Implementation Ready**: Actionable recommendations with compliance guidance
        """

        await manager.store_session_file(
            session_id, DataType.COMPLETE, final_report, "final_comprehensive_report.md"
        )

        # Stage 7: Complete session
        await manager.update_session_status(
            session_id, SessionStatus.COMPLETED, WorkflowStage.COMPLETED
        )

        # Verify complete workflow
        summary = await manager.get_session_summary(session_id)

        assert summary["session_metadata"]["status"] == SessionStatus.COMPLETED.value
        assert summary["file_statistics"]["total_files"] >= 5  # Multiple files created
        assert summary["file_statistics"]["complete_files"] >= 1  # Final report stored
        assert len(summary["sub_sessions"]) == 2  # Gap research sub-sessions
        assert summary["session_metadata"]["sub_session_count"] == 2

        # Verify end-to-end workflow files
        files = await manager.get_session_files(session_id)
        assert "initial_research.md" in files
        assert "first_draft.md" in files
        assert "final_comprehensive_report.md" in files

        # Verify sub-session files
        for gap_topic in identified_gaps:
            expected_filename = f"gap_research_{gap_topic.replace(' ', '_')}.md"
            sub_files = await manager.get_session_files(session_id, DataType.SUB_SESSIONS)
            assert expected_filename in sub_files.values()

        self.logger.info(f"Successfully completed end-to-end research workflow for session {session_id}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])