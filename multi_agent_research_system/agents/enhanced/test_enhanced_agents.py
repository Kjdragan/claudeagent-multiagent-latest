"""Comprehensive Testing and Validation for Enhanced Agent System

This module provides comprehensive testing and validation for the enhanced agent
system, including unit tests, integration tests, performance tests, and
validation scenarios.

Key Features:
- Unit Testing for Enhanced Agents
- Integration Testing for Agent Communication
- Performance Testing and Benchmarking
- Configuration Validation Testing
- Lifecycle Management Testing
- End-to-End Workflow Testing
"""

import asyncio
import json
import logging
import time
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import enhanced agent components
from .base_agent import EnhancedBaseAgent, AgentConfiguration, AgentStatus, RichMessage
from .agent_factory import EnhancedAgentFactory, AgentCreationRequest, AgentType
from .sdk_config import ComprehensiveSDKConfig, SDKConfigManager
from .lifecycle_manager import AgentLifecycleManager
from .communication import AgentCommunicationManager, CommunicationProtocol
from .performance_monitor import AgentPerformanceMonitor


class MockEnhancedAgent(EnhancedBaseAgent):
    """Mock enhanced agent for testing."""

    def __init__(self, config: AgentConfiguration):
        super().__init__(config)
        self.test_data = {}
        self.mock_responses = {}

    def get_system_prompt(self) -> str:
        return f"Mock {self.agent_type} agent for testing"

    def get_default_tools(self) -> List[str]:
        return ["test_tool"]

    async def initialize(self, agent_registry=None) -> None:
        """Mock initialization."""
        self.status = AgentStatus.READY

    async def shutdown(self) -> None:
        """Mock shutdown."""
        self.status = AgentStatus.TERMINATED


class TestEnhancedBaseAgent(unittest.TestCase):
    """Test cases for EnhancedBaseAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AgentConfiguration(
            agent_type="test",
            agent_id="test_agent_001",
            max_turns=10,
            timeout_seconds=60,
            performance_monitoring=True
        )
        self.agent = MockEnhancedAgent(self.config)

    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.agent_id, "test_agent_001")
        self.assertEqual(self.agent.agent_type, "test")
        self.assertEqual(self.agent.status, AgentStatus.INITIALIZING)
        self.assertTrue(self.agent.monitoring_enabled)

    def test_agent_configuration_validation(self):
        """Test agent configuration validation."""
        # Test valid configuration
        self.assertGreater(self.config.max_turns, 0)
        self.assertGreater(self.config.timeout_seconds, 0)
        self.assertTrue(0 <= self.config.quality_threshold <= 1)

        # Test invalid configuration
        with self.assertRaises(ValueError):
            invalid_config = AgentConfiguration(
                agent_type="test",
                agent_id="invalid",
                max_turns=-1
            )

    def test_message_handling(self):
        """Test message handling."""
        # Test message registration
        def test_handler(message):
            pass

        self.agent.register_message_handler("test_message", test_handler)
        self.assertIn("test_message", self.agent.message_handlers)

    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        # Initial metrics should be empty
        self.assertEqual(len(self.agent.performance_metrics), 0)

        # Test metrics update
        session_id = "test_session"
        self.agent._update_execution_metrics(session_id, 1.0, True)

        self.assertIn(session_id, self.agent.performance_metrics)
        metrics = self.agent.performance_metrics[session_id]
        self.assertEqual(metrics.total_requests, 1)
        self.assertEqual(metrics.successful_requests, 1)

    def test_health_check(self):
        """Test health check functionality."""
        health_status = self.agent._calculate_error_rate()
        self.assertEqual(health_status, 0.0)  # No errors initially


class TestAgentFactory(unittest.IsolatedAsyncioTestCase):
    """Test cases for AgentFactory."""

    async def asyncSetUp(self):
        """Set up async test fixtures."""
        self.factory = EnhancedAgentFactory()
        await self.factory.start()

    async def asyncTearDown(self):
        """Clean up async test fixtures."""
        await self.factory.stop()

    async def test_agent_creation(self):
        """Test agent creation from request."""
        request = AgentCreationRequest(
            agent_type=AgentType.RESEARCH,
            agent_id="test_research_001",
            auto_initialize=False
        )

        agent = await self.factory.create_agent(request)
        self.assertIsNotNone(agent)
        self.assertEqual(agent.agent_id, "test_research_001")
        self.assertEqual(agent.agent_type, "research")

    async def test_agent_workflow_creation(self):
        """Test creation of agent workflows."""
        workflow_config = {
            "agents": [
                {
                    "type": "research",
                    "agent_id": "workflow_research",
                    "config_overrides": {"max_turns": 5}
                },
                {
                    "type": "report",
                    "agent_id": "workflow_report",
                    "config_overrides": {"timeout_seconds": 120}
                }
            ]
        }

        agents = await self.factory.create_agent_workflow(workflow_config)
        self.assertEqual(len(agents), 2)

        research_agent = next(a for a in agents if a.agent_id == "workflow_research")
        report_agent = next(a for a in agents if a.agent_id == "workflow_report")

        self.assertEqual(research_agent.agent_type, "research")
        self.assertEqual(report_agent.agent_type, "report")

    def test_template_management(self):
        """Test template management."""
        templates = self.factory.list_templates()
        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 0)

        # Test getting specific template
        research_template = self.factory.registry.agent_templates.get("research_default")
        self.assertIsNotNone(research_template)
        self.assertEqual(research_template.agent_type, AgentType.RESEARCH)

    def test_factory_status(self):
        """Test factory status reporting."""
        status = self.factory.get_factory_status()
        self.assertIn("total_agents", status)
        self.assertIn("agent_types", status)
        self.assertIn("available_templates", status)


class TestSDKConfig(unittest.TestCase):
    """Test cases for SDK configuration management."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = SDKConfigManager(Path("/tmp/test_sdk_config"))

    def test_config_creation(self):
        """Test SDK configuration creation."""
        config = ComprehensiveSDKConfig(
            max_turns=20,
            timeout_seconds=120,
            debug_mode=True
        )

        self.assertEqual(config.max_turns, 20)
        self.assertEqual(config.timeout_seconds, 120)
        self.assertTrue(config.debug_mode)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = ComprehensiveSDKConfig()
        issues = valid_config.validate()
        self.assertEqual(len(issues), 0)

        # Invalid config
        invalid_config = ComprehensiveSDKConfig(
            max_turns=-1,
            quality_threshold=1.5
        )
        issues = invalid_config.validate()
        self.assertGreater(len(issues), 0)

    def test_config_merge(self):
        """Test configuration merging."""
        base_config = ComprehensiveSDKConfig(
            max_turns=10,
            timeout_seconds=60
        )

        override_config = ComprehensiveSDKConfig(
            max_turns=20,
            debug_mode=True
        )

        merged = base_config.merge_with(override_config)
        self.assertEqual(merged.max_turns, 20)  # Override applied
        self.assertEqual(merged.timeout_seconds, 60)  # Original preserved
        self.assertTrue(merged.debug_mode)  # New value added

    def test_agent_specific_config(self):
        """Test agent-specific configuration retrieval."""
        config = self.config_manager.get_config_for_agent(
            "research",
            preset_name="fast",
            overrides={"max_turns": 15}
        )

        self.assertEqual(config.agent_type, "research")
        self.assertEqual(config.max_turns, 15)

    def test_preset_management(self):
        """Test preset management."""
        presets = self.config_manager.list_presets()
        self.assertIn("fast", presets)
        self.assertIn("debug", presets)

        fast_preset = self.config_manager.get_preset("fast")
        self.assertIsNotNone(fast_preset)
        self.assertEqual(fast_preset.execution_mode.value, "fast")


class TestAgentCommunication(unittest.IsolatedAsyncioTestCase):
    """Test cases for agent communication."""

    async def asyncSetUp(self):
        """Set up async test fixtures."""
        self.comm_manager = AgentCommunicationManager("test_agent")
        await self.comm_manager.start()

    async def asyncTearDown(self):
        """Clean up async test fixtures."""
        await self.comm_manager.stop()

    async def test_message_creation(self):
        """Test rich message creation."""
        message = RichMessage(
            sender="agent_a",
            recipient="agent_b",
            message_type="test_request",
            payload={"data": "test"},
            session_id="session_001"
        )

        self.assertEqual(message.sender, "agent_a")
        self.assertEqual(message.recipient, "agent_b")
        self.assertEqual(message.message_type, "test_request")
        self.assertIsNotNone(message.correlation_id)

    async def test_message_sending(self):
        """Test message sending."""
        message = RichMessage(
            sender="test_agent",
            recipient="target_agent",
            message_type="ping",
            payload={},
            session_id="test_session"
        )

        message_id = await self.comm_manager.send_message(
            message,
            protocol=CommunicationProtocol.ASYNCHRONOUS
        )

        self.assertIsNotNone(message_id)
        self.assertIn(message_id, self.comm_manager.sent_messages)

    async def test_message_handling(self):
        """Test message handling."""
        handled_messages = []

        async def test_handler(message):
            handled_messages.append(message)

        self.comm_manager.register_message_handler("test_message", test_handler)

        message = RichMessage(
            sender="sender",
            recipient="test_agent",
            message_type="test_message",
            payload={"test": "data"},
            session_id="test_session"
        )

        # Simulate receiving message
        from .communication import MessageEnvelope
        envelope = MessageEnvelope(message=message)
        await self.comm_manager.receive_message(envelope)

        # Give processing time
        await asyncio.sleep(0.1)

        self.assertEqual(len(handled_messages), 1)

    def test_communication_statistics(self):
        """Test communication statistics."""
        stats = self.comm_manager.get_communication_statistics()
        self.assertIn("agent_id", stats)
        self.assertIn("metrics", stats)
        self.assertIn("queue_size", stats)
        self.assertEqual(stats["agent_id"], "test_agent")


class TestLifecycleManager(unittest.IsolatedAsyncioTestCase):
    """Test cases for agent lifecycle management."""

    async def asyncSetUp(self):
        """Set up async test fixtures."""
        self.factory = EnhancedAgentFactory()
        await self.factory.start()
        self.lifecycle_manager = AgentLifecycleManager(
            self.factory,
            Path("/tmp/test_lifecycle")
        )

    async def asyncTearDown(self):
        """Clean up async test fixtures."""
        await self.lifecycle_manager.stop()
        await self.factory.stop()

    async def test_agent_registration(self):
        """Test agent registration."""
        request = AgentCreationRequest(
            agent_type=AgentType.RESEARCH,
            agent_id="lifecycle_test_agent",
            auto_initialize=False
        )

        agent = await self.factory.create_agent(request)
        success = await self.lifecycle_manager.register_agent(agent)

        self.assertTrue(success)
        self.assertIn(agent.agent_id, self.lifecycle_manager.managed_agents)
        self.assertIn(agent.agent_id, self.lifecycle_manager.agent_states)

    async def test_agent_health_monitoring(self):
        """Test agent health monitoring."""
        request = AgentCreationRequest(
            agent_type=AgentType.RESEARCH,
            agent_id="health_test_agent",
            auto_initialize=False
        )

        agent = await self.factory.create_agent(request)
        await self.lifecycle_manager.register_agent(agent, health_check_interval=1)

        # Wait for health check
        await asyncio.sleep(2)

        state = self.lifecycle_manager.get_agent_status(agent.agent_id)
        self.assertIsNotNone(state)
        self.assertIsNotNone(state.health_status)

    async def test_agent_restart(self):
        """Test agent restart functionality."""
        request = AgentCreationRequest(
            agent_type=AgentType.RESEARCH,
            agent_id="restart_test_agent",
            auto_initialize=False
        )

        agent = await self.factory.create_agent(request)
        await self.lifecycle_manager.register_agent(agent)

        # Simulate error condition
        state = self.lifecycle_manager.agent_states[agent.agent_id]
        state.consecutive_errors = 3

        # Attempt restart
        success = await self.lifecycle_manager.restart_agent(agent.agent_id)
        self.assertTrue(success)
        self.assertGreater(state.total_restarts, 0)

    def test_lifecycle_statistics(self):
        """Test lifecycle statistics."""
        stats = self.lifecycle_manager.get_lifecycle_statistics()
        self.assertIn("total_agents_created", stats)
        self.assertIn("currently_managed_agents", stats)
        self.assertIn("health_percentage", stats)


class TestPerformanceMonitor(unittest.IsolatedAsyncioTestCase):
    """Test cases for performance monitoring."""

    async def asyncSetUp(self):
        """Set up async test fixtures."""
        self.monitor = AgentPerformanceMonitor(Path("/tmp/test_performance"))
        await self.monitor.start()

    async def asyncTearDown(self):
        """Clean up async test fixtures."""
        await self.monitor.stop()

    def test_agent_registration(self):
        """Test agent registration for monitoring."""
        config = AgentConfiguration(
            agent_type="test",
            agent_id="perf_test_agent"
        )
        agent = MockEnhancedAgent(config)

        self.monitor.register_agent(agent)
        self.assertIn(agent.agent_id, self.monitor.monitored_agents)

    def test_metrics_collection(self):
        """Test metrics collection."""
        config = AgentConfiguration(
            agent_type="test",
            agent_id="metrics_test_agent"
        )
        agent = MockEnhancedAgent(config)
        self.monitor.register_agent(agent)

        # Collect metrics
        metrics = self.monitor._collect_agent_metrics(agent)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.agent_id, agent.agent_id)

    def test_resource_snapshot(self):
        """Test resource snapshot collection."""
        snapshot = self.monitor._collect_resource_snapshot()
        self.assertIsNotNone(snapshot)
        self.assertIsInstance(snapshot.timestamp, datetime)

    def test_performance_analysis(self):
        """Test performance analysis."""
        # Create mock metrics history
        from .base_agent import AgentPerformanceMetrics
        metrics_history = [
            AgentPerformanceMetrics(
                agent_id="test_agent",
                session_id="session_1",
                start_time=datetime.now(),
                total_requests=10,
                successful_requests=9,
                failed_requests=1,
                average_response_time=2.0,
                memory_usage_mb=100.0,
                cpu_usage_percent=50.0,
                error_count=1,
                last_activity=datetime.now()
            )
        ]

        analysis = self.monitor.analyzer.analyze_performance(
            "test_agent",
            metrics_history,
            []
        )

        self.assertIn("agent_id", analysis)
        self.assertIn("performance_level", analysis)
        self.assertIn("recommendations", analysis)

    def test_performance_summary(self):
        """Test performance summary generation."""
        config = AgentConfiguration(
            agent_type="test",
            agent_id="summary_test_agent"
        )
        agent = MockEnhancedAgent(config)
        self.monitor.register_agent(agent)

        summary = self.monitor.get_agent_performance_summary(agent.agent_id)
        self.assertIn("agent_id", summary)

    def test_system_overview(self):
        """Test system performance overview."""
        overview = self.monitor.get_system_performance_overview()
        self.assertIn("monitoring_active", overview)
        self.assertIn("total_agents_monitored", overview)


class TestIntegrationScenarios(unittest.IsolatedAsyncioTestCase):
    """Integration test scenarios for the enhanced agent system."""

    async def asyncSetUp(self):
        """Set up integration test environment."""
        self.factory = EnhancedAgentFactory()
        await self.factory.start()

        self.lifecycle_manager = AgentLifecycleManager(self.factory)
        await self.lifecycle_manager.start()

        self.performance_monitor = AgentPerformanceMonitor()
        await self.performance_monitor.start()

    async def asyncTearDown(self):
        """Clean up integration test environment."""
        await self.performance_monitor.stop()
        await self.lifecycle_manager.stop()
        await self.factory.stop()

    async def test_complete_agent_lifecycle(self):
        """Test complete agent lifecycle from creation to shutdown."""
        # Create agent
        request = AgentCreationRequest(
            agent_type=AgentType.RESEARCH,
            agent_id="lifecycle_integration_agent",
            auto_initialize=True
        )

        agent = await self.factory.create_agent(request)

        # Register for lifecycle management
        await self.lifecycle_manager.register_agent(agent)

        # Register for performance monitoring
        self.performance_monitor.register_agent(agent)

        # Verify agent is ready
        self.assertEqual(agent.status, AgentStatus.READY)

        # Wait for monitoring to collect data
        await asyncio.sleep(2)

        # Check lifecycle state
        lifecycle_state = self.lifecycle_manager.get_agent_status(agent.agent_id)
        self.assertIsNotNone(lifecycle_state)
        self.assertEqual(lifecycle_state.current_status, AgentStatus.READY)

        # Check performance monitoring
        perf_summary = self.performance_monitor.get_agent_performance_summary(agent.agent_id)
        self.assertIn("agent_id", perf_summary)

        # Shutdown agent
        success = await self.lifecycle_manager.unregister_agent(agent.agent_id)
        self.assertTrue(success)

    async def test_agent_communication_workflow(self):
        """Test agent communication workflow."""
        # Create two agents
        agent_a_request = AgentCreationRequest(
            agent_type=AgentType.RESEARCH,
            agent_id="comm_agent_a",
            auto_initialize=False
        )
        agent_b_request = AgentCreationRequest(
            agent_type=AgentType.REPORT,
            agent_id="comm_agent_b",
            auto_initialize=False
        )

        agent_a = await self.factory.create_agent(agent_a_request)
        agent_b = await self.factory.create_agent(agent_b_request)

        # Setup communication managers
        comm_a = AgentCommunicationManager(agent_a.agent_id)
        comm_b = AgentCommunicationManager(agent_b.agent_id)

        await comm_a.start()
        await comm_b.start()

        try:
            # Send message from A to B
            message = RichMessage(
                sender=agent_a.agent_id,
                recipient=agent_b.agent_id,
                message_type="test_request",
                payload={"data": "integration_test"},
                session_id="integration_session"
            )

            message_id = await comm_a.send_message(message)
            self.assertIsNotNone(message_id)

            # Verify message was sent
            self.assertIn(message_id, comm_a.sent_messages)

        finally:
            await comm_a.stop()
            await comm_b.stop()

    async def test_configuration_driven_behavior(self):
        """Test configuration-driven agent behavior."""
        # Create agent with specific configuration
        request = AgentCreationRequest(
            agent_type=AgentType.RESEARCH,
            agent_id="config_test_agent",
            config_overrides={
                "max_turns": 5,
                "timeout_seconds": 30,
                "debug_mode": True,
                "performance_monitoring": True
            },
            auto_initialize=False
        )

        agent = await self.factory.create_agent(request)

        # Verify configuration was applied
        self.assertEqual(agent.config.max_turns, 5)
        self.assertEqual(agent.config.timeout_seconds, 30)
        self.assertTrue(agent.config.debug_mode)
        self.assertTrue(agent.monitoring_enabled)

    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Create agent
        request = AgentCreationRequest(
            agent_type=AgentType.RESEARCH,
            agent_id="error_test_agent",
            auto_initialize=False
        )

        agent = await self.factory.create_agent(request)
        await self.lifecycle_manager.register_agent(agent)

        # Simulate error condition
        state = self.lifecycle_manager.agent_states[agent.agent_id]
        state.consecutive_errors = 3
        state.last_error = datetime.now()

        # Verify error tracking
        self.assertEqual(state.consecutive_errors, 3)
        self.assertIsNotNone(state.last_error)

        # Test recovery (if enabled)
        if state.auto_restart_enabled:
            # In a real scenario, this would restart the agent
            # For testing, we verify the recovery logic exists
            self.assertLessEqual(state.recovery_attempts, state.max_recovery_attempts)


class TestPerformanceBenchmarks(unittest.IsolatedAsyncioTestCase):
    """Performance benchmarks for the enhanced agent system."""

    async def asyncSetUp(self):
        """Set up benchmark environment."""
        self.factory = EnhancedAgentFactory()
        await self.factory.start()

    async def asyncTearDown(self):
        """Clean up benchmark environment."""
        await self.factory.stop()

    async def test_agent_creation_performance(self):
        """Benchmark agent creation performance."""
        num_agents = 10
        start_time = time.time()

        agents = []
        for i in range(num_agents):
            request = AgentCreationRequest(
                agent_type=AgentType.RESEARCH,
                agent_id=f"perf_agent_{i}",
                auto_initialize=False
            )
            agent = await self.factory.create_agent(request)
            agents.append(agent)

        creation_time = time.time() - start_time
        avg_time_per_agent = creation_time / num_agents

        # Performance assertions (adjust based on expected performance)
        self.assertLess(avg_time_per_agent, 1.0)  # Should create agents in < 1 second each
        self.assertEqual(len(agents), num_agents)

        # Cleanup
        for agent in agents:
            await self.factory.shutdown_agent(agent.agent_id)

    async def test_message_throughput(self):
        """Benchmark message throughput."""
        agent_a_request = AgentCreationRequest(
            agent_type=AgentType.RESEARCH,
            agent_id="throughput_agent_a",
            auto_initialize=False
        )
        agent_b_request = AgentCreationRequest(
            agent_type=AgentType.REPORT,
            agent_id="throughput_agent_b",
            auto_initialize=False
        )

        agent_a = await self.factory.create_agent(agent_a_request)
        agent_b = await self.factory.create_agent(agent_b_request)

        comm_a = AgentCommunicationManager(agent_a.agent_id)
        comm_b = AgentCommunicationManager(agent_b.agent_id)

        await comm_a.start()
        await comm_b.start()

        try:
            num_messages = 100
            start_time = time.time()

            message_ids = []
            for i in range(num_messages):
                message = RichMessage(
                    sender=agent_a.agent_id,
                    recipient=agent_b.agent_id,
                    message_type="benchmark_message",
                    payload={"index": i},
                    session_id=f"benchmark_session_{i}"
                )

                message_id = await comm_a.send_message(message)
                message_ids.append(message_id)

            send_time = time.time() - start_time
            messages_per_second = num_messages / send_time

            # Performance assertions
            self.assertGreater(messages_per_second, 50)  # Should handle > 50 messages/second
            self.assertEqual(len(message_ids), num_messages)

        finally:
            await comm_a.stop()
            await comm_b.stop()

    async def test_concurrent_agent_operations(self):
        """Benchmark concurrent agent operations."""
        num_concurrent = 5
        operations_per_agent = 10

        async def agent_operations(agent_id: str):
            """Perform operations for a single agent."""
            request = AgentCreationRequest(
                agent_type=AgentType.RESEARCH,
                agent_id=agent_id,
                auto_initialize=False
            )

            agent = await self.factory.create_agent(request)

            # Simulate some operations
            for i in range(operations_per_agent):
                # Mock operation
                await asyncio.sleep(0.01)

            await self.factory.shutdown_agent(agent_id)
            return agent_id

        start_time = time.time()

        # Run concurrent operations
        tasks = [
            agent_operations(f"concurrent_agent_{i}")
            for i in range(num_concurrent)
        ]

        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        total_operations = num_concurrent * operations_per_agent
        operations_per_second = total_operations / total_time

        # Performance assertions
        self.assertGreater(operations_per_second, 20)  # Should handle > 20 ops/second
        self.assertEqual(len(results), num_concurrent)


def run_enhanced_agent_tests():
    """Run all enhanced agent tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestEnhancedBaseAgent,
        TestAgentFactory,
        TestSDKConfig,
        TestAgentCommunication,
        TestLifecycleManager,
        TestPerformanceMonitor,
        TestIntegrationScenarios,
        TestPerformanceBenchmarks
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise during tests
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run tests
    success = run_enhanced_agent_tests()
    exit(0 if success else 1)