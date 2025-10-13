"""
Comprehensive Testing Suite for Enhanced Research Orchestrator

This module provides comprehensive testing and validation for the enhanced orchestrator,
including unit tests, integration tests, and functional tests for all Phase 2.2 components.

Phase 2.2 Testing: Comprehensive testing and validation for enhanced orchestrator
"""

import asyncio
import json
import logging
import pytest
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import enhanced orchestrator components
from core.enhanced_orchestrator import (
    EnhancedResearchOrchestrator,
    EnhancedOrchestratorConfig,
    RichMessage,
    MessageType,
    WorkflowHookContext,
    EnhancedHookManager,
    RichMessageProcessor,
    create_enhanced_orchestrator
)
from core.enhanced_system_integration import EnhancedSystemIntegrator, create_enhanced_system_integrator

# Import base components for testing
from core.workflow_state import WorkflowStage, StageStatus
from core.quality_framework import QualityAssessment, QualityCriterion


class MockQualityAssessment(QualityAssessment):
    """Mock quality assessment for testing."""
    def __init__(self, score: float = 80.0):
        super().__init__(
            overall_score=score,
            criteria_results={},
            strengths=["Test strength"],
            weaknesses=["Test weakness"],
            recommendations=["Test recommendation"],
            quality_level="good" if score >= 70 else "needs_improvement"
        )


class TestEnhancedOrchestratorConfig:
    """Test cases for EnhancedOrchestratorConfig."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = EnhancedOrchestratorConfig()

        assert config.enable_hooks is True
        assert config.enable_rich_messages is True
        assert config.enable_sub_agents is True
        assert config.enable_quality_gates is True
        assert config.enable_error_recovery is True
        assert config.enable_performance_monitoring is True

    def test_custom_configuration(self):
        """Test custom configuration values."""
        custom_config = {
            "enable_hooks": False,
            "max_concurrent_workflows": 10,
            "workflow_timeout": 7200
        }

        config = EnhancedOrchestratorConfig(**custom_config)

        assert config.enable_hooks is False
        assert config.max_concurrent_workflows == 10
        assert config.workflow_timeout == 7200

    def test_performance_thresholds(self):
        """Test performance threshold configuration."""
        config = EnhancedOrchestratorConfig()

        thresholds = config.performance_thresholds
        assert "max_stage_duration" in thresholds
        assert "min_quality_score" in thresholds
        assert "max_error_rate" in thresholds
        assert thresholds["min_quality_score"] == 0.7


class TestRichMessage:
    """Test cases for RichMessage class."""

    def test_rich_message_creation(self):
        """Test basic rich message creation."""
        message = RichMessage(
            id="test_001",
            message_type=MessageType.TEXT,
            content="Test message content",
            session_id="session_123",
            agent_name="test_agent",
            stage="test_stage"
        )

        assert message.id == "test_001"
        assert message.message_type == MessageType.TEXT
        assert message.content == "Test message content"
        assert message.session_id == "session_123"
        assert message.agent_name == "test_agent"
        assert message.stage == "test_stage"

    def test_rich_message_with_metadata(self):
        """Test rich message with metadata."""
        metadata = {"priority": "high", "tags": ["test", "message"]}
        message = RichMessage(
            id="test_002",
            message_type=MessageType.PROGRESS,
            content="Progress update",
            metadata=metadata
        )

        assert message.metadata == metadata
        assert message.message_type == MessageType.PROGRESS

    def test_rich_message_to_dict(self):
        """Test rich message serialization."""
        message = RichMessage(
            id="test_003",
            message_type=MessageType.ERROR,
            content="Error message",
            confidence_score=0.8,
            quality_metrics={"score": 75}
        )

        message_dict = message.to_dict()

        assert message_dict["id"] == "test_003"
        assert message_dict["message_type"] == "error"
        assert message_dict["content"] == "Error message"
        assert message_dict["confidence_score"] == 0.8
        assert message_dict["quality_metrics"]["score"] == 75
        assert "timestamp" in message_dict


class TestWorkflowHookContext:
    """Test cases for WorkflowHookContext."""

    def test_workflow_hook_context_creation(self):
        """Test workflow hook context creation."""
        context = WorkflowHookContext(
            session_id="session_123",
            workflow_stage=WorkflowStage.RESEARCH,
            agent_name="research_agent",
            operation="test_operation",
            start_time=datetime.now()
        )

        assert context.session_id == "session_123"
        assert context.workflow_stage == WorkflowStage.RESEARCH
        assert context.agent_name == "research_agent"
        assert context.operation == "test_operation"

    def test_workflow_hook_context_duration(self):
        """Test workflow hook context duration calculation."""
        start_time = datetime.now()
        context = WorkflowHookContext(
            session_id="session_123",
            workflow_stage=WorkflowStage.RESEARCH,
            agent_name="research_agent",
            operation="test_operation",
            start_time=start_time
        )

        # Duration should be positive
        duration = context.get_duration()
        assert duration >= 0

    def test_workflow_hook_context_with_metadata(self):
        """Test workflow hook context with metadata."""
        metadata = {"test_key": "test_value", "operation_id": 123}
        context = WorkflowHookContext(
            session_id="session_123",
            workflow_stage=WorkflowStage.RESEARCH,
            agent_name="research_agent",
            operation="test_operation",
            start_time=datetime.now(),
            metadata=metadata
        )

        assert context.metadata == metadata


class TestEnhancedHookManager:
    """Test cases for EnhancedHookManager."""

    @pytest.fixture
    def hook_manager(self):
        """Create hook manager for testing."""
        logger = logging.getLogger("test_logger")
        return EnhancedHookManager(logger)

    def test_hook_manager_initialization(self, hook_manager):
        """Test hook manager initialization."""
        assert hook_manager.logger is not None
        assert len(hook_manager.hooks) > 0
        assert "workflow_start" in hook_manager.hooks
        assert "workflow_stage_start" in hook_manager.hooks

    @pytest.mark.asyncio
    async def test_register_hook(self, hook_manager):
        """Test hook registration."""
        async def test_hook(context):
            return {"hook_executed": True}

        hook_manager.register_hook("test_event", test_hook)

        assert "test_event" in hook_manager.hooks
        assert test_hook in hook_manager.hooks["test_event"]

    @pytest.mark.asyncio
    async def test_execute_hooks_success(self, hook_manager):
        """Test successful hook execution."""
        async def test_hook(context):
            return {"hook_result": "success"}

        hook_manager.register_hook("test_event", test_hook)

        context = WorkflowHookContext(
            session_id="test_session",
            workflow_stage=WorkflowStage.RESEARCH,
            agent_name="test_agent",
            operation="test_operation",
            start_time=datetime.now()
        )

        results = await hook_manager.execute_hooks("test_event", context)

        assert "test_hook" in results
        assert results["test_hook"]["success"] is True
        assert results["test_hook"]["result"]["hook_result"] == "success"

    @pytest.mark.asyncio
    async def test_execute_hooks_with_error(self, hook_manager):
        """Test hook execution with error."""
        async def failing_hook(context):
            raise ValueError("Test error")

        hook_manager.register_hook("test_event", failing_hook)

        context = WorkflowHookContext(
            session_id="test_session",
            workflow_stage=WorkflowStage.RESEARCH,
            agent_name="test_agent",
            operation="test_operation",
            start_time=datetime.now()
        )

        results = await hook_manager.execute_hooks("test_event", context)

        assert "failing_hook" in results
        assert results["failing_hook"]["success"] is False
        assert "error" in results["failing_hook"]

    def test_get_hook_statistics(self, hook_manager):
        """Test hook statistics retrieval."""
        stats = hook_manager.get_hook_statistics()

        assert "registered_hooks" in stats
        assert "execution_stats" in stats
        assert isinstance(stats["registered_hooks"], dict)
        assert isinstance(stats["execution_stats"], dict)


class TestRichMessageProcessor:
    """Test cases for RichMessageProcessor."""

    @pytest.fixture
    def message_processor(self):
        """Create message processor for testing."""
        logger = logging.getLogger("test_logger")
        return RichMessageProcessor(logger)

    def test_message_processor_initialization(self, message_processor):
        """Test message processor initialization."""
        assert message_processor.logger is not None
        assert len(message_processor.message_processors) > 0
        assert MessageType.TEXT in message_processor.message_processors
        assert MessageType.ERROR in message_processor.message_processors

    @pytest.mark.asyncio
    async def test_process_text_message(self, message_processor):
        """Test text message processing."""
        message = RichMessage(
            id="test_001",
            message_type=MessageType.TEXT,
            content="This is a test message with multiple words for analysis.",
            session_id="test_session",
            agent_name="test_agent"
        )

        processed_message = await message_processor.process_message(message)

        assert processed_message.id == "test_001"
        assert "style" in processed_message.formatting
        assert "word_count" in processed_message.metadata
        assert "char_count" in processed_message.metadata
        assert processed_message.metadata["word_count"] == 10  # Count the words

    @pytest.mark.asyncio
    async def test_process_error_message(self, message_processor):
        """Test error message processing."""
        message = RichMessage(
            id="error_001",
            message_type=MessageType.ERROR,
            content="Critical error occurred in the system",
            session_id="test_session",
            agent_name="test_agent"
        )

        processed_message = await message_processor.process_message(message)

        assert processed_message.message_type == MessageType.ERROR
        assert "style" in processed_message.formatting
        assert "severity" in processed_message.metadata
        assert "error_category" in processed_message.metadata
        assert processed_message.metadata["severity"] == "high"  # Contains "critical"

    @pytest.mark.asyncio
    async def test_process_quality_assessment_message(self, message_processor):
        """Test quality assessment message processing."""
        message = RichMessage(
            id="quality_001",
            message_type=MessageType.QUALITY_ASSESSMENT,
            content="Quality assessment completed",
            session_id="test_session",
            agent_name="quality_agent",
            quality_metrics={"overall_score": 85}
        )

        processed_message = await message_processor.process_message(message)

        assert processed_message.message_type == MessageType.QUALITY_ASSESSMENT
        assert "style" in processed_message.formatting
        assert "color_scheme" in processed_message.formatting
        assert processed_message.formatting["color_scheme"] == "green"  # Score >= 85
        assert processed_message.metadata["quality_level"] == "excellent"

    @pytest.mark.asyncio
    async def test_process_tool_use_message(self, message_processor):
        """Test tool use message processing."""
        message = RichMessage(
            id="tool_001",
            message_type=MessageType.TOOL_USE,
            content="Tool execution initiated",
            session_id="test_session",
            agent_name="test_agent",
            metadata={"tool_name": "search_tool"}
        )

        processed_message = await message_processor.process_message(message)

        assert processed_message.message_type == MessageType.TOOL_USE
        assert "style" in processed_message.formatting
        assert "tool_category" in processed_message.metadata
        assert "estimated_duration" in processed_message.metadata
        assert processed_message.metadata["tool_category"] == "search"

    def test_get_message_statistics(self, message_processor):
        """Test message statistics retrieval."""
        # Add some messages to history
        message1 = RichMessage("id1", MessageType.TEXT, "Content", session_id="session1")
        message2 = RichMessage("id2", MessageType.ERROR, "Error", session_id="session1")
        message3 = RichMessage("id3", MessageType.TEXT, "More content", session_id="session2")

        message_processor.message_history = [message1, message2, message3]

        stats = message_processor.get_message_statistics()

        assert stats["total_messages"] == 3
        assert "by_type" in stats
        assert "by_agent" in stats
        assert "by_stage" in stats
        assert "recent_messages" in stats
        assert stats["by_type"]["text"] == 2
        assert stats["by_type"]["error"] == 1


class TestEnhancedResearchOrchestrator:
    """Test cases for EnhancedResearchOrchestrator."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return EnhancedOrchestratorConfig(
            enable_hooks=True,
            enable_rich_messages=True,
            enable_sub_agents=False,  # Disable for unit testing
            enable_quality_gates=True,
            enable_error_recovery=True
        )

    @pytest.fixture
    def orchestrator(self, mock_config):
        """Create enhanced orchestrator for testing."""
        with patch('core.enhanced_orchestrator.ResearchOrchestrator.__init__', return_value=None):
            orchestrator = EnhancedResearchOrchestrator(config=mock_config, debug_mode=True)
            # Mock the parent class attributes
            orchestrator.logger = logging.getLogger("test_orchestrator")
            orchestrator.workflow_state_manager = Mock()
            orchestrator.quality_framework = Mock()
            orchestrator.progressive_enhancement_pipeline = Mock()
            orchestrator.decoupled_editorial_agent = Mock()
            return orchestrator

    def test_enhanced_orchestrator_initialization(self, orchestrator):
        """Test enhanced orchestrator initialization."""
        assert orchestrator.config is not None
        assert orchestrator.hook_manager is not None
        assert orchestrator.message_processor is not None
        assert orchestrator.quality_gate_manager is not None
        assert orchestrator.error_recovery_manager is not None

    def test_create_workflow_context(self, orchestrator):
        """Test workflow context creation."""
        session_id = "test_session"
        context = orchestrator._create_workflow_context(session_id)

        assert context.session_id == session_id
        assert context.workflow_stage == WorkflowStage.INITIALIZATION
        assert context.agent_name == "enhanced_orchestrator"
        assert context.operation == "workflow_execution"

    @pytest.mark.asyncio
    async def test_initialize_enhanced_session(self, orchestrator):
        """Test enhanced session initialization."""
        session_id = "test_session"
        await orchestrator._initialize_enhanced_session(session_id)

        assert session_id in orchestrator.enhanced_session_data
        session_data = orchestrator.enhanced_session_data[session_id]
        assert session_data["session_id"] == session_id
        assert "start_time" in session_data
        assert "performance_metrics" in session_data
        assert "stage_history" in session_data

    @pytest.mark.asyncio
    async def test_extract_gap_research_requests(self, orchestrator):
        """Test gap research request extraction."""
        editorial_result = {
            "gap_research_requests": ["quantum computing applications", "AI ethics"],
            "content": "Some editorial content"
        }

        gap_requests = orchestrator._extract_gap_research_requests(editorial_result)

        assert len(gap_requests) == 2
        assert "quantum computing applications" in gap_requests
        assert "AI ethics" in gap_requests

    @pytest.mark.asyncio
    async def test_get_workflow_performance(self, orchestrator):
        """Test workflow performance metrics retrieval."""
        session_id = "test_session"

        # Initialize session data
        await orchestrator._initialize_enhanced_session(session_id)

        performance = orchestrator._get_workflow_performance(session_id)

        assert "total_duration" in performance
        assert "stage_performance" in performance
        assert "quality_progression" in performance
        assert "hook_performance" in performance
        assert "message_performance" in performance

    @pytest.mark.asyncio
    async def test_hook_workflow_start(self, orchestrator):
        """Test workflow start hook."""
        context = WorkflowHookContext(
            session_id="test_session",
            workflow_stage=WorkflowStage.INITIALIZATION,
            agent_name="enhanced_orchestrator",
            operation="workflow_start",
            start_time=datetime.now()
        )

        result = await orchestrator._hook_workflow_start(context)

        assert result["event"] == "workflow_start"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_hook_workflow_stage_complete(self, orchestrator):
        """Test workflow stage complete hook."""
        context = WorkflowHookContext(
            session_id="test_session",
            workflow_stage=WorkflowStage.RESEARCH,
            agent_name="research_agent",
            operation="research_complete",
            start_time=datetime.now()
        )

        result = await orchestrator._hook_workflow_stage_complete(context)

        assert result["event"] == "stage_complete"
        assert result["stage"] == "research"
        assert "duration" in result


class TestEnhancedSystemIntegrator:
    """Test cases for EnhancedSystemIntegrator."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator for testing."""
        orchestrator = Mock()
        orchestrator.logger = logging.getLogger("test_orchestrator")
        orchestrator.hook_manager = Mock()
        orchestrator.message_processor = Mock()
        return orchestrator

    @pytest.fixture
    def system_integrator(self, mock_orchestrator):
        """Create system integrator for testing."""
        with patch('core.enhanced_system_integration.PHASE1_SYSTEMS_AVAILABLE', False):
            integrator = EnhancedSystemIntegrator(mock_orchestrator)
            return integrator

    def test_system_integrator_initialization(self, system_integrator):
        """Test system integrator initialization."""
        assert system_integrator.orchestrator is not None
        assert system_integrator.logger is not None
        assert isinstance(system_integrator.integration_metrics, dict)
        assert system_integrator.integration_metrics["total_integrations"] == 0

    def test_get_integration_metrics(self, system_integrator):
        """Test integration metrics retrieval."""
        metrics = system_integrator.get_integration_metrics()

        assert "integration_metrics" in metrics
        assert "phase1_systems_available" in metrics
        assert "available_systems" in metrics
        assert "total_integrations" in metrics
        assert "integration_breakdown" in metrics

    def test_get_system_status(self, system_integrator):
        """Test system status retrieval."""
        status = system_integrator.get_system_status()

        assert "integration_available" in status
        assert "systems_status" in status
        assert isinstance(status["systems_status"], dict)

    @pytest.mark.asyncio
    async def test_extract_urls_from_content(self, system_integrator):
        """Test URL extraction from content."""
        content = "Check out https://example.com and https://test.org for more information."
        urls = system_integrator._extract_urls_from_content(content)

        assert len(urls) == 2
        assert "https://example.com" in urls
        assert "https://test.org" in urls

    def test_calculate_readability_score(self, system_integrator):
        """Test readability score calculation."""
        # Good readability (15-20 words per sentence)
        good_content = "This is a good sentence. It has about fifteen words per sentence. The readability should be high."
        good_score = system_integrator._calculate_readability_score(good_content)
        assert good_score == 1.0

        # Poor readability (too many words per sentence)
        poor_content = "This is a very long sentence that contains way too many words and should result in a lower readability score because it exceeds the optimal range of fifteen to twenty words per sentence."
        poor_score = system_integrator._calculate_readability_score(poor_content)
        assert poor_score < 1.0

    def test_calculate_source_diversity(self, system_integrator):
        """Test source diversity calculation."""
        sources = [
            {"url": "https://example1.com/page1"},
            {"url": "https://example1.com/page2"},
            {"url": "https://example2.com/page1"},
            {"url": "https://example3.com/page1"}
        ]

        diversity = system_integrator._calculate_source_diversity(sources)
        assert diversity == 0.75  # 3 unique domains out of 4 sources

        # Single domain
        single_domain_sources = [
            {"url": "https://example1.com/page1"},
            {"url": "https://example1.com/page2"}
        ]

        single_diversity = system_integrator._calculate_source_diversity(single_domain_sources)
        assert single_diversity == 0.5  # 1 unique domain out of 2 sources


class TestIntegrationScenarios:
    """Integration test scenarios for the enhanced orchestrator."""

    @pytest.mark.asyncio
    async def test_enhanced_research_workflow_integration(self):
        """Test complete enhanced research workflow integration."""
        # This is a high-level integration test
        # In a real implementation, this would test the complete workflow

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            config = EnhancedOrchestratorConfig(
                enable_hooks=True,
                enable_rich_messages=True,
                enable_sub_agents=False,  # Disable for simpler testing
                enable_quality_gates=True,
                enable_error_recovery=True
            )

            # Mock the base orchestrator to avoid complex dependencies
            with patch('core.enhanced_orchestrator.ResearchOrchestrator') as mock_base:
                mock_instance = Mock()
                mock_instance.stage_conduct_research = AsyncMock(return_value={"content": "Research results"})
                mock_instance.stage_generate_report = AsyncMock(return_value={"content": "Generated report"})
                mock_instance.stage_conduct_editorial_review = AsyncMock(return_value={"content": "Editorial review"})
                mock_base.return_value = mock_instance

                # Create enhanced orchestrator
                orchestrator = EnhancedResearchOrchestrator(config=config, debug_mode=True)

                # Mock dependencies
                orchestrator.quality_framework = Mock()
                orchestrator.quality_framework.assess_content = AsyncMock(return_value=MockQualityAssessment(85))
                orchestrator.progressive_enhancement_pipeline = Mock()
                orchestrator.progressive_enhancement_pipeline.enhance_content = AsyncMock(return_value={"content": "Enhanced content"})

                # Test workflow execution (without actual SDK dependencies)
                session_id = "test_session_123"

                # Verify initialization
                assert orchestrator.hook_manager is not None
                assert orchestrator.message_processor is not None
                assert orchestrator.config.enable_hooks is True

                # Verify session initialization
                await orchestrator._initialize_enhanced_session(session_id)
                assert session_id in orchestrator.enhanced_session_data

    @pytest.mark.asyncio
    async def test_system_integration_workflow(self):
        """Test system integration workflow with Phase 1 systems."""
        # Create mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.logger = logging.getLogger("test_orchestrator")
        mock_orchestrator.hook_manager = Mock()
        mock_orchestrator.message_processor = Mock()

        # Create system integrator
        with patch('core.enhanced_system_integration.PHASE1_SYSTEMS_AVAILABLE', False):
            integrator = EnhancedSystemIntegrator(mock_orchestrator)

        # Test research enhancement
        session_id = "test_session"
        research_params = {
            "query": "test query",
            "max_sources": 10
        }

        enhanced_params = await integrator.enhance_research_execution(session_id, research_params)

        assert isinstance(enhanced_params, dict)
        assert "query" in enhanced_params

        # Test content processing enhancement
        content_data = {
            "sources": [
                {"url": "https://example.com", "content": "Test content"}
            ]
        }

        enhanced_content = await integrator.enhance_content_processing(session_id, content_data)

        assert isinstance(enhanced_content, dict)
        assert "sources" in enhanced_content

        # Verify metrics
        metrics = integrator.get_integration_metrics()
        assert metrics["total_integrations"] > 0


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_enhanced_orchestrator(self):
        """Test enhanced orchestrator factory function."""
        config = EnhancedOrchestratorConfig(enable_hooks=False)

        with patch('core.enhanced_orchestrator.ResearchOrchestrator.__init__', return_value=None):
            orchestrator = create_enhanced_orchestrator(config=config, debug_mode=True)

        assert isinstance(orchestrator, EnhancedResearchOrchestrator)
        assert orchestrator.config == config

    def test_create_enhanced_system_integrator(self):
        """Test enhanced system integrator factory function."""
        mock_orchestrator = Mock()
        mock_orchestrator.logger = logging.getLogger("test_orchestrator")

        with patch('core.enhanced_system_integration.PHASE1_SYSTEMS_AVAILABLE', False):
            integrator = create_enhanced_system_integrator(mock_orchestrator)

        assert isinstance(integrator, EnhancedSystemIntegrator)
        assert integrator.orchestrator == mock_orchestrator


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_hook_execution_error_handling(self):
        """Test error handling in hook execution."""
        logger = logging.getLogger("test_logger")
        hook_manager = EnhancedHookManager(logger)

        # Register a failing hook
        async def failing_hook(context):
            raise ValueError("Test hook error")

        hook_manager.register_hook("test_event", failing_hook)

        context = WorkflowHookContext(
            session_id="test_session",
            workflow_stage=WorkflowStage.RESEARCH,
            agent_name="test_agent",
            operation="test_operation",
            start_time=datetime.now()
        )

        # Execute hooks - should not raise exception
        results = await hook_manager.execute_hooks("test_event", context)

        assert "failing_hook" in results
        assert results["failing_hook"]["success"] is False
        assert "error" in results["failing_hook"]

    @pytest.mark.asyncio
    async def test_message_processor_error_handling(self):
        """Test error handling in message processing."""
        logger = logging.getLogger("test_logger")
        processor = RichMessageProcessor(logger)

        # Create a message that might cause issues
        message = RichMessage(
            id="test_error",
            message_type=MessageType.ERROR,
            content="Error message",
            session_id="test_session",
            agent_name="test_agent"
        )

        # Process message - should not raise exception
        processed_message = await processor.process_message(message)

        assert processed_message.id == "test_error"
        assert processed_message.message_type == MessageType.ERROR


# Performance Tests
class TestPerformance:
    """Performance tests for enhanced orchestrator components."""

    @pytest.mark.asyncio
    async def test_hook_manager_performance(self):
        """Test hook manager performance with multiple hooks."""
        logger = logging.getLogger("test_logger")
        hook_manager = EnhancedHookManager(logger)

        # Register multiple hooks
        for i in range(10):
            async def test_hook(context, index=i):
                await asyncio.sleep(0.01)  # Simulate work
                return {"hook_index": index}

            hook_manager.register_hook("performance_test", test_hook)

        context = WorkflowHookContext(
            session_id="perf_test",
            workflow_stage=WorkflowStage.RESEARCH,
            agent_name="test_agent",
            operation="performance_test",
            start_time=datetime.now()
        )

        # Measure execution time
        start_time = asyncio.get_event_loop().time()
        results = await hook_manager.execute_hooks("performance_test", context)
        end_time = asyncio.get_event_loop().time()

        execution_time = end_time - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert execution_time < 2.0  # 2 seconds
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_message_processor_performance(self):
        """Test message processor performance with multiple messages."""
        logger = logging.getLogger("test_logger")
        processor = RichMessageProcessor(logger)

        # Create multiple messages
        messages = []
        for i in range(100):
            message = RichMessage(
                id=f"perf_test_{i}",
                message_type=MessageType.TEXT,
                content=f"Test message {i} with some content for processing.",
                session_id="perf_test_session",
                agent_name="perf_test_agent"
            )
            messages.append(message)

        # Process all messages
        start_time = asyncio.get_event_loop().time()
        for message in messages:
            await processor.process_message(message)
        end_time = asyncio.get_event_loop().time()

        execution_time = end_time - start_time

        # Should process 100 messages in reasonable time
        assert execution_time < 5.0  # 5 seconds
        assert len(processor.message_history) == 100


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])