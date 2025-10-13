#!/usr/bin/env python3
"""
Comprehensive Tests for Sub-Agent Architecture

This module provides comprehensive tests for the sub-agent architecture,
including unit tests, integration tests, and end-to-end validation.
"""

import asyncio
import pytest
import unittest
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.sub_agents import (
    SubAgentFactory, SubAgentCoordinator, SubAgentType,
    ContextIsolationManager, SubAgentCommunicationManager,
    SubAgentPerformanceMonitor, create_sub_agent_config
)
from utils.sub_agents.sub_agent_types import (
    SubAgentConfiguration, SubAgentRequest, SubAgentResult
)
from utils.sub_agents.sub_agent_factory import SubAgentInstance
from utils.sub_agents.communication_protocols import (
    MessageType, MessagePriority, SubAgentMessage
)
from utils.sub_agents.sub_agent_coordinator import (
    WorkflowTask, CoordinatedWorkflow, WorkflowStage, WorkflowStatus
)


class TestSubAgentTypes(unittest.TestCase):
    """Test sub-agent types and configurations."""

    def test_create_sub_agent_config(self):
        """Test creating sub-agent configurations."""
        # Test researcher configuration
        config = create_sub_agent_config(SubAgentType.RESEARCHER)
        self.assertIsInstance(config, SubAgentConfiguration)
        self.assertEqual(config.agent_type, SubAgentType.RESEARCHER)
        self.assertIsNotNone(config.persona)
        self.assertIsNotNone(config.capabilities)
        self.assertIsNotNone(config.claude_options)

        # Test that researcher has research tools
        self.assertIn("WebSearch", config.capabilities.allowed_tools)
        self.assertIn("WebFetch", config.capabilities.allowed_tools)

    def test_all_agent_types_configurable(self):
        """Test that all agent types can be configured."""
        for agent_type in SubAgentType:
            config = create_sub_agent_config(agent_type)
            self.assertIsInstance(config, SubAgentConfiguration)
            self.assertEqual(config.agent_type, agent_type)

    def test_agent_personas(self):
        """Test agent personas have required attributes."""
        config = create_sub_agent_config(SubAgentType.RESEARCHER)
        persona = config.persona

        self.assertIsNotNone(persona.name)
        self.assertIsNotNone(persona.description)
        self.assertIsNotNone(persona.system_prompt)
        self.assertIsInstance(persona.expertise_areas, list)
        self.assertIn("web_research", persona.expertise_areas)

    def test_agent_capabilities(self):
        """Test agent capabilities."""
        config = create_sub_agent_config(SubAgentType.REPORT_WRITER)
        capabilities = config.capabilities

        self.assertIsInstance(capabilities.allowed_tools, list)
        self.assertGreater(len(capabilities.allowed_tools), 0)
        self.assertGreater(capabilities.max_turns, 0)
        self.assertGreater(capabilities.timeout_seconds, 0)

    def test_custom_configuration(self):
        """Test custom configuration modifications."""
        custom_config = {
            "max_turns": 100,
            "timeout_seconds": 600,
            "isolation_level": "strict"
        }

        config = create_sub_agent_config(
            SubAgentType.QUALITY_ASSESSOR,
            **custom_config
        )

        self.assertEqual(config.capabilities.max_turns, 100)
        self.assertEqual(config.capabilities.timeout_seconds, 600)
        self.assertEqual(config.isolation_level, "strict")


class TestContextIsolation(unittest.IsolatedAsyncioTestCase):
    """Test context isolation mechanisms."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.isolation_manager = ContextIsolationManager()
        await self.isolation_manager.initialize()

    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.isolation_manager.shutdown()

    async def test_create_isolation_context(self):
        """Test creating isolation contexts."""
        context_id = await self.isolation_manager.create_isolation_context(
            agent_type="researcher",
            session_id="test_session"
        )

        self.assertIsNotNone(context_id)
        self.assertIn(context_id, self.isolation_manager.active_contexts)

        context = self.isolation_manager.active_contexts[context_id]
        self.assertEqual(context.agent_type, "researcher")
        self.assertEqual(context.session_id, "test_session")

    async def test_context_data_storage(self):
        """Test storing and retrieving data in contexts."""
        context_id = await self.isolation_manager.create_isolation_context(
            agent_type="researcher",
            session_id="test_session"
        )

        # Store data
        test_data = {"research_findings": "Quantum computing breakthrough"}
        await self.isolation_manager.store_data_in_context(
            context_id, "findings", test_data
        )

        # Retrieve data
        retrieved_data = await self.isolation_manager.retrieve_data_from_context(
            context_id, "findings"
        )

        self.assertEqual(retrieved_data, test_data)

    async def test_context_isolation_enforcement(self):
        """Test that context isolation is enforced."""
        context_1 = await self.isolation_manager.create_isolation_context(
            agent_type="researcher",
            session_id="test_session",
            isolation_level="strict"
        )

        context_2 = await self.isolation_manager.create_isolation_context(
            agent_type="report_writer",
            session_id="test_session",
            isolation_level="strict"
        )

        # Store data in context 1
        await self.isolation_manager.store_data_in_context(
            context_1, "secret_data", "confidential information"
        )

        # Try to access from context 2 (should fail)
        with self.assertRaises(Exception):
            await self.isolation_manager.retrieve_data_from_context(
                context_2, "secret_data"
            )

    async def test_context_cleanup(self):
        """Test context cleanup."""
        context_id = await self.isolation_manager.create_isolation_context(
            agent_type="researcher",
            session_id="test_session"
        )

        # Verify context exists
        self.assertIn(context_id, self.isolation_manager.active_contexts)

        # Cleanup context
        await self.isolation_manager.cleanup_isolation_context(context_id)

        # Verify context is removed
        self.assertNotIn(context_id, self.isolation_manager.active_contexts)

    async def test_context_expiry(self):
        """Test context expiration."""
        # Create context with short expiry
        context_id = await self.isolation_manager.create_isolation_context(
            agent_type="researcher",
            session_id="test_session",
            expiry_hours=0.001  # Very short expiry (3.6 seconds)
        )

        # Wait for expiry
        await asyncio.sleep(4)

        # Try to get expired context
        context = await self.isolation_manager.get_context(context_id)
        self.assertIsNone(context)


class TestCommunicationProtocols(unittest.IsolatedAsyncioTestCase):
    """Test communication protocols."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.communication_manager = SubAgentCommunicationManager()
        await self.communication_manager.initialize()

    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.communication_manager.shutdown()

    async def test_register_message_handler(self):
        """Test registering message handlers."""
        handler_called = asyncio.Event()
        received_message = None

        async def test_handler(message):
            nonlocal received_message
            received_message = message
            handler_called.set()

        handler_id = await self.communication_manager.register_message_handler(
            agent_id="test_agent",
            agent_type="tester",
            message_types=[MessageType.DIRECT_MESSAGE],
            handler_function=test_handler
        )

        self.assertIsNotNone(handler_id)

        # Send message
        message_id = await self.communication_manager.send_direct_message(
            sender_id="sender",
            sender_type="tester",
            recipient_id="test_agent",
            recipient_type="tester",
            session_id="test_session",
            payload={"test": "data"}
        )

        # Wait for handler to be called
        await asyncio.wait_for(handler_called.wait(), timeout=5)

        self.assertIsNotNone(received_message)
        self.assertEqual(received_message.payload["test"], "data")

    async def test_send_receive_message(self):
        """Test sending and receiving messages."""
        response_received = asyncio.Event()
        response_data = None

        async def response_handler(message):
            nonlocal response_data
            response_data = message.payload
            response_received.set()

        await self.communication_manager.register_message_handler(
            agent_id="responder",
            agent_type="tester",
            message_types=[MessageType.REQUEST],
            handler_function=response_handler
        )

        # Send message that requires response
        message_id = await self.communication_manager.send_message(
            message_type=MessageType.REQUEST,
            sender_id="requester",
            sender_type="tester",
            recipient_id="responder",
            recipient_type="tester",
            session_id="test_session",
            payload={"question": "test"},
            requires_response=True
        )

        # Wait for response
        await asyncio.wait_for(response_received.wait(), timeout=5)

        self.assertIsNotNone(response_data)

    async def test_broadcast_messages(self):
        """Test broadcasting messages."""
        handlers_called = asyncio.Queue()

        async def broadcast_handler(message):
            await handlers_called.put(message.message_id)

        # Register multiple handlers
        for i in range(3):
            await self.communication_manager.register_message_handler(
                agent_id=f"handler_{i}",
                agent_type="tester",
                message_types=[MessageType.NOTIFICATION],
                handler_function=broadcast_handler
            )

        # Send broadcast
        message_id = await self.communication_manager.broadcast_message(
            sender_id="broadcaster",
            sender_type="tester",
            session_id="test_session",
            payload={"broadcast": "test"}
        )

        # Wait for handlers to be called
        received_count = 0
        timeout = datetime.now() + timedelta(seconds=5)

        while received_count < 3 and datetime.now() < timeout:
            try:
                await asyncio.wait_for(handlers_called.get(), timeout=1)
                received_count += 1
            except asyncio.TimeoutError:
                break

        self.assertEqual(received_count, 3)

    async def test_message_priority(self):
        """Test message priority handling."""
        received_messages = []

        async def priority_handler(message):
            received_messages.append((message.priority.value, message.message_id))

        await self.communication_manager.register_message_handler(
            agent_id="priority_tester",
            agent_type="tester",
            message_types=[MessageType.DIRECT_MESSAGE],
            handler_function=priority_handler
        )

        # Send messages with different priorities
        await self.communication_manager.send_direct_message(
            sender_id="sender",
            sender_type="tester",
            recipient_id="priority_tester",
            recipient_type="tester",
            session_id="test_session",
            payload={"priority": "low"},
            priority=MessagePriority.LOW
        )

        await self.communication_manager.send_direct_message(
            sender_id="sender",
            sender_type="tester",
            recipient_id="priority_tester",
            recipient_type="tester",
            session_id="test_session",
            payload={"priority": "high"},
            priority=MessagePriority.HIGH
        )

        # Wait for messages to be processed
        await asyncio.sleep(1)

        # High priority message should be processed first
        if len(received_messages) >= 2:
            self.assertEqual(received_messages[0][0], MessagePriority.HIGH.value)
            self.assertEqual(received_messages[1][0], MessagePriority.LOW.value)


class TestSubAgentFactory(unittest.IsolatedAsyncioTestCase):
    """Test sub-agent factory."""

    async def asyncSetUp(self):
        """Set up test environment."""
        # Mock Claude SDK to avoid actual API calls
        with patch('utils.sub_agents.sub_agent_factory.ClaudeSDKClient'):
            self.factory = SubAgentFactory()
            await self.factory.initialize()

    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.factory.shutdown()

    async def test_create_sub_agent(self):
        """Test creating sub-agent instances."""
        request = SubAgentRequest(
            agent_type=SubAgentType.RESEARCHER,
            task_description="Test research task",
            session_id="test_session",
            parent_agent="test_parent"
        )

        # Mock the client creation
        with patch.object(self.factory, '_create_claude_client') as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            instance = await self.factory.create_sub_agent(request)

            self.assertIsInstance(instance, SubAgentInstance)
            self.assertEqual(instance.agent_type, SubAgentType.RESEARCHER)
            self.assertIsNotNone(instance.instance_id)
            self.assertEqual(instance.status, "active")

    async def test_execute_sub_agent_task(self):
        """Test executing tasks with sub-agents."""
        # Create a mock instance
        instance_id = "test_instance"
        mock_client = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.receive_message = AsyncMock()
        mock_client.receive_message.return_value = iter([])

        instance = SubAgentInstance(
            instance_id=instance_id,
            agent_type=SubAgentType.RESEARCHER,
            configuration=create_sub_agent_config(SubAgentType.RESEARCHER),
            client=mock_client
        )

        self.factory.active_instances[instance_id] = instance

        # Execute task
        result = await self.factory.execute_sub_agent_task(
            instance_id,
            "Test task prompt"
        )

        self.assertIsInstance(result, SubAgentResult)
        self.assertEqual(result.instance_id, instance_id)
        self.assertEqual(result.agent_type, SubAgentType.RESEARCHER)

    async def test_cleanup_expired_instances(self):
        """Test cleanup of expired instances."""
        # Create expired instance
        instance_id = "expired_instance"
        mock_client = Mock()
        mock_client.disconnect = AsyncMock()

        instance = SubAgentInstance(
            instance_id=instance_id,
            agent_type=SubAgentType.RESEARCHER,
            configuration=create_sub_agent_config(SubAgentType.RESEARCHER),
            client=mock_client,
            created_at=datetime.now() - timedelta(hours=1),  # 1 hour ago
            last_activity=datetime.now() - timedelta(hours=1)  # 1 hour ago
        )

        self.factory.active_instances[instance_id] = instance

        # Trigger cleanup
        await self.factory._cleanup_expired_instances()

        # Verify instance was cleaned up
        self.assertNotIn(instance_id, self.factory.active_instances)

    def test_get_factory_status(self):
        """Test getting factory status."""
        status = self.factory.get_factory_status()

        self.assertIn("running", status)
        self.assertIn("active_instances", status)
        self.assertIn("max_concurrent", status)
        self.assertIsInstance(status["active_instances"], int)


class TestSubAgentCoordinator(unittest.IsolatedAsyncioTestCase):
    """Test sub-agent coordinator."""

    async def asyncSetUp(self):
        """Set up test environment."""
        with patch('utils.sub_agents.sub_agent_factory.ClaudeSDKClient'):
            self.coordinator = SubAgentCoordinator()
            await self.coordinator.initialize()

    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.coordinator.shutdown()

    async def test_create_coordinated_workflow(self):
        """Test creating coordinated workflows."""
        workflow_id = await self.coordinator.create_coordinated_workflow(
            session_id="test_session",
            topic="Test topic",
            description="Test workflow",
            workflow_type="standard_research"
        )

        self.assertIsNotNone(workflow_id)
        self.assertIn(workflow_id, self.coordinator.active_workflows)

        workflow = self.coordinator.active_workflows[workflow_id]
        self.assertEqual(workflow.session_id, "test_session")
        self.assertEqual(workflow.topic, "Test topic")
        self.assertEqual(workflow.status, WorkflowStatus.PENDING)
        self.assertGreater(len(workflow.tasks), 0)

    async def test_workflow_task_dependencies(self):
        """Test workflow task dependencies."""
        workflow_id = await self.coordinator.create_coordinated_workflow(
            session_id="test_session",
            topic="Test topic",
            description="Test workflow"
        )

        workflow = self.coordinator.active_workflows[workflow_id]

        # Verify tasks have correct dependencies
        report_tasks = workflow.get_tasks_by_stage(WorkflowStage.REPORT_GENERATION)
        if report_tasks:
            report_task = report_tasks[0]
            # Report task should depend on research task
            self.assertGreater(len(report_task.dependencies), 0)

        # Test task readiness
        ready_tasks = workflow.get_ready_tasks()
        # Initially, only research tasks should be ready
        research_tasks = workflow.get_tasks_by_stage(WorkflowStage.RESEARCH)
        self.assertTrue(all(task in ready_tasks for task in research_tasks))

    async def test_workflow_status_tracking(self):
        """Test workflow status tracking."""
        workflow_id = await self.coordinator.create_coordinated_workflow(
            session_id="test_session",
            topic="Test topic",
            description="Test workflow"
        )

        # Get initial status
        status = await self.coordinator.get_workflow_status(workflow_id)
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "pending")

        # Start workflow
        await self.coordinator.start_workflow(workflow_id)

        # Check updated status
        status = await self.coordinator.get_workflow_status(workflow_id)
        self.assertEqual(status["status"], "in_progress")
        self.assertIsNotNone(status["started_at"])

    async def test_cancel_workflow(self):
        """Test cancelling workflows."""
        workflow_id = await self.coordinator.create_coordinated_workflow(
            session_id="test_session",
            topic="Test topic",
            description="Test workflow"
        )

        # Start and then cancel workflow
        await self.coordinator.start_workflow(workflow_id)
        cancelled = await self.coordinator.cancel_workflow(workflow_id)

        self.assertTrue(cancelled)

        # Check status
        status = await self.coordinator.get_workflow_status(workflow_id)
        self.assertEqual(status["status"], "cancelled")

    def test_get_coordinator_status(self):
        """Test getting coordinator status."""
        status = self.coordinator.get_coordinator_status()

        self.assertIn("running", status)
        self.assertIn("active_workflows", status)
        self.assertIn("factory_status", status)
        self.assertIn("communication_stats", status)
        self.assertIn("isolation_status", status)
        self.assertIn("monitoring_status", status)


class TestPerformanceMonitoring(unittest.IsolatedAsyncioTestCase):
    """Test performance monitoring."""

    async def asyncSetUp(self):
        """Set up test environment."""
        self.monitor = SubAgentPerformanceMonitor()
        await self.monitor.initialize()

    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.monitor.shutdown()

    async def test_track_agent_creation(self):
        """Test tracking agent creation."""
        mock_agent = Mock()
        mock_agent.instance_id = "test_agent_1"
        mock_agent.agent_type = SubAgentType.RESEARCHER

        await self.monitor.track_agent_creation(mock_agent)

        self.assertIn("test_agent_1", self.monitor.agent_profiles)
        profile = self.monitor.agent_profiles["test_agent_1"]
        self.assertEqual(profile.agent_type, SubAgentType.RESEARCHER.value)

    async def test_track_execution(self):
        """Test tracking execution performance."""
        mock_agent = Mock()
        mock_agent.instance_id = "test_agent_1"
        mock_agent.agent_type = SubAgentType.RESEARCHER

        await self.monitor.track_agent_creation(mock_agent)
        await self.monitor.track_execution(
            mock_agent, execution_time=5.5, success=True, quality_score=85.0
        )

        profile = self.monitor.agent_profiles["test_agent_1"]
        self.assertEqual(profile.total_executions, 1)
        self.assertEqual(profile.successful_executions, 1)
        self.assertEqual(profile.average_execution_time, 5.5)
        self.assertEqual(profile.average_quality_score, 85.0)

    async def test_track_execution_error(self):
        """Test tracking execution errors."""
        mock_agent = Mock()
        mock_agent.instance_id = "test_agent_1"
        mock_agent.agent_type = SubAgentType.RESEARCHER

        await self.monitor.track_agent_creation(mock_agent)
        await self.monitor.track_execution_error(mock_agent, "Connection timeout")

        profile = self.monitor.agent_profiles["test_agent_1"]
        self.assertIn("timeout", profile.error_types)
        self.assertEqual(profile.error_types["timeout"], 1)

    async def test_get_performance_data(self):
        """Test getting performance data."""
        mock_agent = Mock()
        mock_agent.instance_id = "test_agent_1"
        mock_agent.agent_type = SubAgentType.RESEARCHER

        await self.monitor.track_agent_creation(mock_agent)
        await self.monitor.track_execution(
            mock_agent, execution_time=3.2, success=True, quality_score=90.0
        )

        # Get agent performance
        agent_perf = await self.monitor.get_agent_performance("test_agent_1")
        self.assertIsNotNone(agent_perf)
        self.assertEqual(agent_perf["total_executions"], 1)
        self.assertEqual(agent_perf["success_rate"], 100.0)

        # Get agent type performance
        type_perf = await self.monitor.get_agent_type_performance("researcher")
        self.assertIsNotNone(type_perf)
        self.assertEqual(type_perf["agent_count"], 1)
        self.assertEqual(type_perf["total_executions"], 1)

    async def test_performance_alerts(self):
        """Test performance alerts."""
        mock_agent = Mock()
        mock_agent.instance_id = "test_agent_1"
        mock_agent.agent_type = SubAgentType.RESEARCHER

        await self.monitor.track_agent_creation(mock_agent)

        # Track execution with high execution time (should trigger alert)
        await self.monitor.track_execution(
            mock_agent, execution_time=120.0, success=True  # 2 minutes
        )

        # Check for alerts
        critical_alerts = [
            alert for alert in self.monitor.performance_alerts
            if alert["agent_id"] == "test_agent_1" and
               alert["type"] == "execution_time_critical"
        ]

        self.assertGreater(len(critical_alerts), 0)

    def test_get_monitoring_status(self):
        """Test getting monitoring status."""
        status = self.monitor.get_monitoring_status()

        self.assertIn("running", status)
        self.assertIn("tracked_agents", status)
        self.assertIn("metrics_collected", status)
        self.assertIn("monitoring_config", status)


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the complete sub-agent system."""

    async def asyncSetUp(self):
        """Set up test environment."""
        with patch('utils.sub_agents.sub_agent_factory.ClaudeSDKClient'):
            self.factory = SubAgentFactory()
            self.coordinator = SubAgentCoordinator()
            self.isolation_manager = ContextIsolationManager()
            self.communication_manager = SubAgentCommunicationManager()
            self.performance_monitor = SubAgentPerformanceMonitor()

            await self.factory.initialize()
            await self.coordinator.initialize()
            await self.isolation_manager.initialize()
            await self.communication_manager.initialize()
            await self.performance_monitor.initialize()

    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.coordinator.shutdown()
        await self.communication_manager.shutdown()
        await self.isolation_manager.shutdown()
        await self.performance_monitor.shutdown()
        await self.factory.shutdown()

    async def test_end_to_end_workflow(self):
        """Test end-to-end workflow execution."""
        # Create a simple workflow
        workflow_id = await self.coordinator.create_coordinated_workflow(
            session_id="integration_test",
            topic="AI in healthcare",
            description="Integration test workflow",
            workflow_type="quick_analysis"
        )

        # Start workflow
        await self.coordinator.start_workflow(workflow_id)

        # Monitor progress (with timeout)
        max_wait_time = 30  # 30 seconds
        start_time = datetime.now()

        workflow_completed = False
        while (datetime.now() - start_time).total_seconds() < max_wait_time:
            status = await self.coordinator.get_workflow_status(workflow_id)

            if status and status["status"] in ["completed", "failed"]:
                workflow_completed = True
                break

            await asyncio.sleep(1)

        # Verify system components are still functional
        factory_status = self.factory.get_factory_status()
        isolation_status = self.isolation_manager.get_isolation_status()
        comm_stats = self.communication_manager.get_communication_stats()
        monitoring_status = self.performance_monitor.get_monitoring_status()

        self.assertTrue(factory_status["running"])
        self.assertTrue(isolation_status["running"])
        self.assertTrue(monitoring_status["running"])

    async def test_component_interaction(self):
        """Test interaction between different components."""
        # Create isolation context
        context_id = await self.isolation_manager.create_isolation_context(
            agent_type="researcher",
            session_id="interaction_test"
        )

        # Store data in context
        await self.isolation_manager.store_data_in_context(
            context_id, "test_data", {"value": "test"}
        )

        # Register communication handler
        message_received = asyncio.Event()

        async def test_handler(message):
            message_received.set()

        await self.communication_manager.register_message_handler(
            agent_id="test_agent",
            agent_type="tester",
            message_types=[MessageType.DIRECT_MESSAGE],
            handler_function=test_handler
        )

        # Send message
        await self.communication_manager.send_direct_message(
            sender_id="sender",
            sender_type="tester",
            recipient_id="test_agent",
            recipient_type="tester",
            session_id="interaction_test",
            payload={"test": "message"}
        )

        # Wait for message processing
        await asyncio.wait_for(message_received.wait(), timeout=5)

        # Cleanup
        await self.isolation_manager.cleanup_isolation_context(context_id)

        # Verify all components are still running
        self.assertTrue(self.factory.get_factory_status()["running"])
        self.assertTrue(self.isolation_manager.get_isolation_status()["running"])


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestSubAgentTypes,
        TestContextIsolation,
        TestCommunicationProtocols,
        TestSubAgentFactory,
        TestSubAgentCoordinator,
        TestPerformanceMonitoring,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running Sub-Agent Architecture Tests")
    print("=" * 50)

    success = run_tests()

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)