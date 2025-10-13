"""
Phase 3.1 Comprehensive Hooks System Demo and Integration Test

This module demonstrates the complete Phase 3.1 implementation including:
- Comprehensive hooks system with Claude Agent SDK integration
- Real-time monitoring infrastructure
- Performance analytics and optimization
- Integration with existing Phase 1 & 2 systems
- Rich messaging and notifications

Usage:
    python phase3_1_demo.py [--demo-type basic|standard|comprehensive] [--session-id <id>]

Demo Types:
- basic: Essential hooks only
- standard: Full hooks with monitoring
- comprehensive: All hooks with analytics and optimization (default)
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import system components
try:
    from core.logging_config import setup_logging, get_logger
    from core.enhanced_orchestrator import EnhancedResearchOrchestrator, WorkflowHookContext, WorkflowStage
    from utils.message_processing.main import MessageProcessor, MessageType
    from hooks.enhanced_integration import (
        EnhancedHooksIntegrator,
        IntegrationConfig,
        IntegrationLevel,
        create_enhanced_hooks_integration
    )
    from hooks.comprehensive_hooks import ComprehensiveHookManager, HookCategory, HookPriority
    from hooks.hook_analytics import HookAnalyticsEngine, create_hook_analytics_engine
    from hooks.real_time_monitoring import create_real_time_monitoring
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running this from the correct directory structure")
    sys.exit(1)


class Phase31Demo:
    """
    Demonstration of Phase 3.1 Comprehensive Hooks System
    """

    def __init__(self, demo_type: str = "comprehensive", session_id: str = None):
        """Initialize the demo."""
        self.demo_type = demo_type
        self.session_id = session_id or f"demo_{int(time.time())}"
        self.logger = get_logger("phase31_demo")

        # Configure integration based on demo type
        if demo_type == "basic":
            self.config = IntegrationConfig(
                integration_level=IntegrationLevel.BASIC,
                enable_analytics=False,
                enable_real_time_monitoring=False,
                enable_performance_optimization=False,
                enable_rich_messaging=False
            )
        elif demo_type == "standard":
            self.config = IntegrationConfig(
                integration_level=IntegrationLevel.STANDARD,
                enable_analytics=True,
                enable_real_time_monitoring=True,
                enable_performance_optimization=False,
                enable_rich_messaging=True
            )
        else:  # comprehensive
            self.config = IntegrationConfig(
                integration_level=IntegrationLevel.COMPREHENSIVE,
                enable_analytics=True,
                enable_real_time_monitoring=True,
                enable_performance_optimization=True,
                enable_rich_messaging=True,
                analytics_window_minutes=30,
                metrics_retention_hours=2
            )

        # Components
        self.integrator: EnhancedHooksIntegrator = None
        self.orchestrator: EnhancedResearchOrchestrator = None
        self.message_processor: MessageProcessor = None

        # Demo metrics
        self.demo_stats = {
            "hooks_executed": 0,
            "mock_research_operations": 0,
            "performance_samples": 0,
            "alerts_triggered": 0,
            "optimizations_applied": 0
        }

    async def run_demo(self):
        """Run the complete Phase 3.1 demonstration."""
        print(f"\n🚀 Phase 3.1 Comprehensive Hooks System Demo")
        print(f"Demo Type: {self.demo_type.upper()}")
        print(f"Session ID: {self.session_id}")
        print(f"Integration Level: {self.config.integration_level.value}")
        print("=" * 60)

        try:
            # Initialize components
            await self._initialize_demo()

            # Run demonstration scenarios
            await self._demo_basic_hooks()
            await self._demo_phase1_integration()
            await self._demo_phase2_integration()
            await self._demo_orchestrator_integration()

            if self.config.enable_analytics:
                await self._demo_analytics_system()

            if self.config.enable_real_time_monitoring:
                await self._demo_real_time_monitoring()

            if self.config.enable_performance_optimization:
                await self._demo_performance_optimization()

            # Show final results
            await self._show_demo_results()

        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")
            raise

        finally:
            await self._cleanup_demo()

    async def _initialize_demo(self):
        """Initialize demo components."""
        print("\n📋 Initializing Demo Components...")

        # Setup logging
        setup_logging(level="INFO", console_output=True)

        # Initialize message processor
        self.message_processor = MessageProcessor()

        # Create mock orchestrator for demo
        self.orchestrator = MockEnhancedOrchestrator(self.session_id, self.logger)

        # Initialize enhanced hooks integration
        self.integrator = create_enhanced_hooks_integration(self.config)

        success = await self.integrator.initialize(self.orchestrator)
        if not success:
            raise RuntimeError("Failed to initialize hooks integration")

        print("✅ Demo components initialized successfully")

    async def _demo_basic_hooks(self):
        """Demonstrate basic hook functionality."""
        print("\n🔗 Testing Basic Hook Functionality...")

        hook_manager = self.integrator.hook_manager

        # Test hook registration and execution
        test_results = await self._execute_test_hooks(hook_manager)

        print(f"✅ Basic hooks test completed: {len(test_results)} hooks executed")
        self.demo_stats["hooks_executed"] += len(test_results)

    async def _execute_test_hooks(self, hook_manager) -> list:
        """Execute test hooks for demonstration."""
        test_context = WorkflowHookContext(
            hook_name="test_hook",
            hook_type="test_execution",
            session_id=self.session_id,
            agent_name="demo_agent",
            workflow_stage="testing",
            operation="demo_test",
            start_time=datetime.now(),
            metadata={"demo_type": self.demo_type}
        )

        # Execute various hook types
        hook_types = [
            "workflow_start",
            "workflow_stage_start",
            "quality_assessment_completed",
            "performance_metrics_collection",
            "error_recovery_completed"
        ]

        results = []
        for hook_type in hook_types:
            if hook_type in hook_manager.hooks:
                hook_results = await hook_manager.execute_hooks(hook_type, test_context)
                results.extend(hook_results)
                await asyncio.sleep(0.1)  # Small delay between executions

        return results

    async def _demo_phase1_integration(self):
        """Demonstrate Phase 1 system integration."""
        print("\n🏗️ Testing Phase 1 System Integration...")

        # Simulate anti-bot escalation
        await self._simulate_anti_bot_escalation()

        # Simulate content cleaning
        await self._simulate_content_cleaning()

        # Simulate scraping pipeline
        await self._simulate_scraping_pipeline()

        # Simulate workflow management
        await self._simulate_workflow_management()

        print("✅ Phase 1 integration tests completed")

    async def _simulate_anti_bot_escalation(self):
        """Simulate anti-bot escalation system."""
        context = WorkflowHookContext(
            hook_name="anti_bot_demo",
            hook_type="anti_bot_escalation_started",
            session_id=self.session_id,
            agent_name="anti_bot_system",
            workflow_stage="research",
            operation="escalation",
            start_time=datetime.now(),
            metadata={
                "domain": "example.com",
                "escalation_level": 2,
                "detection_confidence": 0.85
            }
        )

        results = await self.integrator.hook_manager.execute_hooks(
            "anti_bot_escalation_started", context
        )

        # Simulate escalation completion
        await asyncio.sleep(0.5)
        context.metadata.update({
            "success": True,
            "final_level": 3,
            "requests_completed": 15
        })

        completion_results = await self.integrator.hook_manager.execute_hooks(
            "anti_bot_escalation_completed", context
        )

        self.demo_stats["mock_research_operations"] += 1

    async def _simulate_content_cleaning(self):
        """Simulate content cleaning system."""
        context = WorkflowHookContext(
            hook_name="content_cleaning_demo",
            hook_type="content_cleaning_started",
            session_id=self.session_id,
            agent_name="content_cleaner",
            workflow_stage="processing",
            operation="cleaning",
            start_time=datetime.now(),
            metadata={
                "content_count": 25,
                "cleaning_method": "gpt5nano",
                "quality_threshold": 0.7
            }
        )

        # Start cleaning
        await self.integrator.hook_manager.execute_hooks("content_cleaning_started", context)

        # Simulate GPT-5-nano scoring
        await asyncio.sleep(0.3)
        scoring_context = WorkflowHookContext(
            hook_name="gpt5nano_scoring_demo",
            hook_type="gpt5nano_scoring_completed",
            session_id=self.session_id,
            agent_name="gpt5nano_scorer",
            workflow_stage="processing",
            operation="scoring",
            start_time=datetime.now(),
            metadata={
                "average_score": 0.82,
                "cache_hit_rate": 0.75,
                "total_processed": 25
            }
        )

        await self.integrator.hook_manager.execute_hooks("gpt5nano_scoring_completed", scoring_context)

        # Simulate quality judgments
        for i in range(5):
            await asyncio.sleep(0.1)
            judgment_context = WorkflowHookContext(
                hook_name=f"quality_judgment_{i}",
                hook_type="content_quality_judgment",
                session_id=self.session_id,
                agent_name="quality_judge",
                workflow_stage="processing",
                operation="judgment",
                start_time=datetime.now(),
                metadata={
                    "useful": i % 4 != 0,  # 75% useful
                    "confidence": 0.7 + (i * 0.05),
                    "content_id": f"content_{i}"
                }
            )

            await self.integrator.hook_manager.execute_hooks("content_quality_judgment", judgment_context)

    async def _simulate_scraping_pipeline(self):
        """Simulate scraping pipeline system."""
        context = WorkflowHookContext(
            hook_name="scraping_demo",
            hook_type="async_orchestrator_started",
            session_id=self.session_id,
            agent_name="scraping_orchestrator",
            workflow_stage="research",
            operation="scraping",
            start_time=datetime.now(),
            metadata={
                "concurrent_scrapes": 40,
                "concurrent_cleans": 20,
                "target_urls": 50
            }
        )

        await self.integrator.hook_manager.execute_hooks("async_orchestrator_started", context)

        # Simulate batch processing
        for batch in range(3):
            await asyncio.sleep(0.2)
            batch_context = WorkflowHookContext(
                hook_name=f"batch_{batch}",
                hook_type="scraping_batch_completed",
                session_id=self.session_id,
                agent_name="batch_processor",
                workflow_stage="research",
                operation="batch_processing",
                start_time=datetime.now(),
                metadata={
                    "batch_size": 15,
                    "success_count": 12 + batch,
                    "batch_number": batch + 1
                }
            )

            await self.integrator.hook_manager.execute_hooks("scraping_batch_completed", batch_context)

        # Simulate data contract validation
        validation_context = WorkflowHookContext(
            hook_name="contract_validation",
            hook_type="data_contract_validated",
            session_id=self.session_id,
            agent_name="contract_validator",
            workflow_stage="validation",
            operation="validation",
            start_time=datetime.now(),
            metadata={
                "valid": True,
                "contract_type": "scraped_content_v2",
                "validations_passed": 45,
                "validations_failed": 0
            }
        )

        await self.integrator.hook_manager.execute_hooks("data_contract_validated", validation_context)

        self.demo_stats["mock_research_operations"] += 1

    async def _simulate_workflow_management(self):
        """Simulate workflow management system."""
        # Simulate success tracking updates
        for progress in range(5):
            await asyncio.sleep(0.15)
            success_count = 2 + progress * 2
            target_count = 10

            tracking_context = WorkflowHookContext(
                hook_name=f"success_update_{progress}",
                hook_type="success_tracker_updated",
                session_id=self.session_id,
                agent_name="success_tracker",
                workflow_stage="research",
                operation="tracking",
                start_time=datetime.now(),
                metadata={
                    "current_successes": success_count,
                    "target_count": target_count,
                    "completion_reached": success_count >= target_count
                }
            )

            await self.integrator.hook_manager.execute_hooks("success_tracker_updated", tracking_context)

        # Simulate early termination
        termination_context = WorkflowHookContext(
            hook_name="early_termination",
            hook_type="early_termination_triggered",
            session_id=self.session_id,
            agent_name="workflow_manager",
            workflow_stage="completion",
            operation="termination",
            start_time=datetime.now(),
            metadata={
                "reason": "target_reached",
                "resources_saved": 15,
                "time_saved": 45.2
            }
        )

        await self.integrator.hook_manager.execute_hooks("early_termination_triggered", termination_context)

    async def _demo_phase2_integration(self):
        """Demonstrate Phase 2 system integration."""
        print("\n🤖 Testing Phase 2 System Integration...")

        # Simulate sub-agent operations
        await self._simulate_sub_agent_operations()

        # Simulate message processing
        await self._simulate_message_processing()

        # Simulate enhanced agent operations
        await self._simulate_enhanced_agent_operations()

        print("✅ Phase 2 integration tests completed")

    async def _simulate_sub_agent_operations(self):
        """Simulate sub-agent system operations."""
        # Create sub-agents
        agent_types = ["research_specialist", "content_analyzer", "quality_evaluator"]
        for agent_type in agent_types:
            context = WorkflowHookContext(
                hook_name=f"sub_agent_creation_{agent_type}",
                hook_type="sub_agent_created",
                session_id=self.session_id,
                agent_name="sub_agent_factory",
                workflow_stage="initialization",
                operation="agent_creation",
                start_time=datetime.now(),
                metadata={
                    "agent_type": agent_type,
                    "agent_id": f"{agent_type}_{self.session_id[:8]}",
                    "specialization": agent_type.replace("_", " ")
                }
            )

            await self.integrator.hook_manager.execute_hooks("sub_agent_created", context)
            await asyncio.sleep(0.1)

        # Simulate coordination
        coordination_context = WorkflowHookContext(
            hook_name="coordination_demo",
            hook_type="sub_agent_coordination_started",
            session_id=self.session_id,
            agent_name="coordination_manager",
            workflow_stage="coordination",
            operation="multi_agent_coordination",
            start_time=datetime.now(),
            metadata={
                "coordination_type": "research_pipeline",
                "participant_count": len(agent_types),
                "coordination_pattern": "pipeline"
            }
        )

        await self.integrator.hook_manager.execute_hooks("sub_agent_coordination_started", coordination_context)

        # Simulate context isolation
        isolation_context = WorkflowHookContext(
            hook_name="context_isolation_demo",
            hook_type="context_isolation_enforced",
            session_id=self.session_id,
            agent_name="context_manager",
            workflow_stage="security",
            operation="isolation_enforcement",
            start_time=datetime.now(),
            metadata={
                "isolation_level": "strict",
                "data_leak_prevented": True,
                "isolation_mechanism": "context_partitioning"
            }
        )

        await self.integrator.hook_manager.execute_hooks("context_isolation_enforced", isolation_context)

    async def _simulate_message_processing(self):
        """Simulate message processing system."""
        message_types = ["research_update", "quality_alert", "coordination_request", "status_report"]

        for i, msg_type in enumerate(message_types):
            await asyncio.sleep(0.2)

            # Simulate rich message processing
            processing_context = WorkflowHookContext(
                hook_name=f"message_processing_{i}",
                hook_type="rich_message_processed",
                session_id=self.session_id,
                agent_name="message_processor",
                workflow_stage="communication",
                operation="message_processing",
                start_time=datetime.now(),
                metadata={
                    "message_type": msg_type,
                    "processing_time": 0.05 + (i * 0.02),
                    "cache_used": i % 2 == 0,
                    "message_size": 1024 + (i * 256)
                }
            )

            await self.integrator.hook_manager.execute_hooks("rich_message_processed", processing_context)

            # Simulate cache hits for some messages
            if i % 2 == 0:
                cache_context = WorkflowHookContext(
                    hook_name=f"cache_hit_{i}",
                    hook_type="message_cache_hit",
                    session_id=self.session_id,
                    agent_name="cache_manager",
                    workflow_stage="caching",
                    operation="cache_retrieval",
                    start_time=datetime.now(),
                    metadata={
                        "cache_key": f"msg_{msg_type}_{self.session_id[:8]}",
                        "hit_rate": 0.75 + (i * 0.05)
                    }
                )

                await self.integrator.hook_manager.execute_hooks("message_cache_hit", cache_context)

            # Simulate message formatting
            formatting_context = WorkflowHookContext(
                hook_name=f"formatting_{i}",
                hook_type="message_formatting_applied",
                session_id=self.session_id,
                agent_name="message_formatter",
                workflow_stage="formatting",
                operation="format_application",
                start_time=datetime.now(),
                metadata={
                    "format_type": "rich_markdown",
                    "enhancement_level": ["low", "medium", "high"][i % 3],
                    "formatting_features": ["emoji", "markdown", "syntax_highlight"]
                }
            )

            await self.integrator.hook_manager.execute_hooks("message_formatting_applied", formatting_context)

    async def _simulate_enhanced_agent_operations(self):
        """Simulate enhanced agent operations."""
        # Agent lifecycle events
        lifecycle_events = ["initialized", "active", "processing", "idle", "cleanup"]
        for event in lifecycle_events:
            context = WorkflowHookContext(
                hook_name=f"lifecycle_{event}",
                hook_type="agent_lifecycle_event",
                session_id=self.session_id,
                agent_name="lifecycle_manager",
                workflow_stage="lifecycle",
                operation="lifecycle_management",
                start_time=datetime.now(),
                metadata={
                    "event": event,
                    "agent_type": "enhanced_research_agent",
                    "state_change": True
                }
            )

            await self.integrator.hook_manager.execute_hooks("agent_lifecycle_event", context)
            await asyncio.sleep(0.1)

        # SDK options application
        sdk_context = WorkflowHookContext(
            hook_name="sdk_options_demo",
            hook_type="sdk_options_applied",
            session_id=self.session_id,
            agent_name="sdk_manager",
            workflow_stage="configuration",
            operation="sdk_configuration",
            start_time=datetime.now(),
            metadata={
                "options_count": 12,
                "custom_options": True,
                "key_options": ["max_tokens", "temperature", "top_p", "hooks_enabled"]
            }
        )

        await self.integrator.hook_manager.execute_hooks("sdk_options_applied", sdk_context)

        # Factory pattern execution
        factory_context = WorkflowHookContext(
            hook_name="factory_demo",
            hook_type="factory_pattern_executed",
            session_id=self.session_id,
            agent_name="agent_factory",
            workflow_stage="creation",
            operation="agent_creation",
            start_time=datetime.now(),
            metadata={
                "pattern_type": "abstract_factory",
                "instance_created": True,
                "product_type": "enhanced_research_agent"
            }
        )

        await self.integrator.hook_manager.execute_hooks("factory_pattern_executed", factory_context)

    async def _demo_orchestrator_integration(self):
        """Demonstrate orchestrator integration."""
        print("\n🎯 Testing Orchestrator Integration...")

        # Simulate quality gate enforcement
        await self._simulate_quality_gate_enforcement()

        # Simulate flow adherence validation
        await self._simulate_flow_adherence_validation()

        # Simulate gap research enforcement
        await self._simulate_gap_research_enforcement()

        print("✅ Orchestrator integration tests completed")

    async def _simulate_quality_gate_enforcement(self):
        """Simulate quality gate enforcement."""
        quality_scores = [65, 78, 82, 91, 88]  # Various quality scores

        for i, score in enumerate(quality_scores):
            await asyncio.sleep(0.2)

            threshold = 75
            gate_passed = score >= threshold

            context = WorkflowHookContext(
                hook_name=f"quality_gate_{i}",
                hook_type="quality_gate_enforcement",
                session_id=self.session_id,
                agent_name="quality_gate_manager",
                workflow_stage="quality_check",
                operation="gate_enforcement",
                start_time=datetime.now(),
                metadata={
                    "gate_passed": gate_passed,
                    "quality_score": score,
                    "threshold": threshold,
                    "gate_type": "content_quality"
                }
            )

            await self.integrator.hook_manager.execute_hooks("quality_gate_enforcement", context)

    async def _simulate_flow_adherence_validation(self):
        """Simulate flow adherence validation."""
        # Simulate various adherence scenarios
        scenarios = [
            {"adherence_level": "full_compliance", "violations": 0, "actions": 0},
            {"adherence_level": "minor_violation", "violations": 2, "actions": 2},
            {"adherence_level": "full_compliance", "violations": 0, "actions": 0},
            {"adherence_level": "major_violation", "violations": 5, "actions": 5},
            {"adherence_level": "full_compliance", "violations": 0, "actions": 0}
        ]

        for scenario in scenarios:
            await asyncio.sleep(0.3)

            context = WorkflowHookContext(
                hook_name=f"flow_adherence_{scenario['adherence_level']}",
                hook_type="flow_adherence_validation",
                session_id=self.session_id,
                agent_name="flow_validator",
                workflow_stage="validation",
                operation="adherence_check",
                start_time=datetime.now(),
                metadata=scenario
            )

            await self.integrator.hook_manager.execute_hooks("flow_adherence_validation", context)

    async def _simulate_gap_research_enforcement(self):
        """Simulate gap research enforcement."""
        # Simulate editorial decisions
        editorial_scenarios = [
            {"enforcement_needed": False, "gaps": 0, "triggered": False},
            {"enforcement_needed": True, "gaps": 2, "triggered": True},
            {"enforcement_needed": True, "gaps": 1, "triggered": True},
            {"enforcement_needed": False, "gaps": 0, "triggered": False}
        ]

        for i, scenario in enumerate(editorial_scenarios):
            await asyncio.sleep(0.4)

            context = WorkflowHookContext(
                hook_name=f"gap_research_enforcement_{i}",
                hook_type="gap_research_enforcement",
                session_id=self.session_id,
                agent_name="gap_research_enforcer",
                workflow_stage="editorial_review",
                operation="gap_enforcement",
                start_time=datetime.now(),
                metadata={
                    "enforcement_needed": scenario["enforcement_needed"],
                    "gaps_identified": scenario["gaps"],
                    "research_triggered": scenario["triggered"],
                    "editorial_decision": f"decision_{i}"
                }
            )

            await self.integrator.hook_manager.execute_hooks("gap_research_enforcement", context)

            if scenario["triggered"]:
                self.demo_stats["mock_research_operations"] += 1

    async def _demo_analytics_system(self):
        """Demonstrate analytics system."""
        if not self.config.enable_analytics:
            return

        print("\n📊 Testing Analytics System...")

        # Wait for some analytics data to be collected
        await asyncio.sleep(2)

        # Get analytics summary
        analytics_summary = self.integrator.analytics_engine.get_performance_summary()
        print(f"📈 Analytics Summary:")
        print(f"   Total hooks monitored: {analytics_summary.get('total_hooks', 0)}")
        print(f"   Total executions: {analytics_summary.get('total_executions', 0)}")
        print(f"   Overall success rate: {analytics_summary.get('overall_success_rate', 0):.1%}")
        print(f"   Average efficiency: {analytics_summary.get('average_efficiency', 0):.1f}")

        # Show performance distribution
        performance_dist = analytics_summary.get('performance_distribution', {})
        if performance_dist:
            print(f"   Performance distribution:")
            for level, count in performance_dist.items():
                print(f"     {level}: {count}")

        # Show top recommendations
        recommendations = analytics_summary.get('top_recommendations', [])
        if recommendations:
            print(f"   Top optimization recommendations:")
            for i, rec in enumerate(recommendations[:3]):
                print(f"     {i+1}. {rec.get('description', 'No description')} (Priority: {rec.get('priority', 'N/A')})")

        self.demo_stats["performance_samples"] += 1

    async def _demo_real_time_monitoring(self):
        """Demonstrate real-time monitoring."""
        if not self.config.enable_real_time_monitoring:
            return

        print("\n📡 Testing Real-Time Monitoring...")

        # Get system health snapshot
        health_snapshot = self.integrator.real_time_monitor.create_health_snapshot()
        print(f"🏥 System Health:")
        print(f"   Overall health: {health_snapshot.overall_health.value}")
        print(f"   Active sessions: {health_snapshot.active_sessions}")
        print(f"   Error rate: {health_snapshot.error_rate:.1%}")
        print(f"   Average response time: {health_snapshot.average_response_time:.2f}s")

        # Show active alerts
        active_alerts = self.integrator.real_time_monitor.get_active_alerts()
        if active_alerts:
            print(f"   Active alerts: {len(active_alerts)}")
            for alert in active_alerts[:3]:
                print(f"     - {alert.title}: {alert.description}")
        else:
            print(f"   Active alerts: None")

        # Simulate some metrics to trigger alerts
        await self._simulate_monitoring_scenarios()

        self.demo_stats["alerts_triggered"] = len(active_alerts)

    async def _simulate_monitoring_scenarios(self):
        """Simulate monitoring scenarios to demonstrate alerting."""
        if not self.integrator.metrics_collector:
            return

        from hooks.real_time_monitoring import MetricValue, MetricType

        # Simulate some metrics that might trigger alerts
        test_metrics = [
            ("hook_execution_time", 8.5, MetricType.TIMER),  # Slow execution
            ("error_rate", 0.15, MetricType.GAUGE),          # High error rate
            ("memory_usage", 0.92, MetricType.GAUGE),         # High memory usage
            ("cpu_usage", 0.88, MetricType.GAUGE),           # High CPU usage
        ]

        for metric_name, value, metric_type in test_metrics:
            metric = MetricValue(
                name=metric_name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(),
                labels={"session_id": self.session_id, "demo": "true"}
            )

            self.integrator.metrics_collector.record_metric(metric)
            await asyncio.sleep(0.2)

    async def _demo_performance_optimization(self):
        """Demonstrate performance optimization."""
        if not self.config.enable_performance_optimization:
            return

        print("\n⚡ Testing Performance Optimization...")

        # Wait for optimization recommendations
        await asyncio.sleep(3)

        # Check for optimization recommendations
        if self.integrator.analytics_engine:
            recommendations = self.integrator.analytics_engine._recommendations_cache

            if recommendations:
                print(f"🔧 Optimization Recommendations Available:")
                for i, rec in enumerate(recommendations[:5]):
                    print(f"   {i+1}. {rec.description}")
                    print(f"      Type: {rec.optimization_type.value}")
                    print(f"      Priority: {rec.priority}")
                    print(f"      Expected improvement: {rec.expected_improvement}")
                    print(f"      Complexity: {rec.implementation_complexity}")
                    print()

                # Simulate applying some optimizations
                auto_applicable = [r for r in recommendations if r.priority <= 2 and r.implementation_complexity == "low"]
                for rec in auto_applicable[:2]:  # Apply up to 2 optimizations
                    await self.integrator._apply_optimization_recommendation(rec)
                    self.demo_stats["optimizations_applied"] += 1
            else:
                print("   No optimization recommendations available at this time")

    async def _show_demo_results(self):
        """Show comprehensive demo results."""
        print("\n" + "=" * 60)
        print("🎉 PHASE 3.1 COMPREHENSIVE HOOKS SYSTEM DEMO RESULTS")
        print("=" * 60)

        # Demo statistics
        print(f"\n📊 Demo Statistics:")
        print(f"   Hooks executed: {self.demo_stats['hooks_executed']}")
        print(f"   Mock research operations: {self.demo_stats['mock_research_operations']}")
        print(f"   Performance samples: {self.demo_stats['performance_samples']}")
        print(f"   Alerts triggered: {self.demo_stats['alerts_triggered']}")
        print(f"   Optimizations applied: {self.demo_stats['optimizations_applied']}")

        # Integration status
        integration_status = self.integrator.get_integration_status()
        print(f"\n🔗 Integration Status:")
        print(f"   Integration active: {integration_status['integration_active']}")
        print(f"   Integration level: {integration_status['config']['integration_level'].value}")

        print(f"\n🏗️ Component Status:")
        for component, status in integration_status['components'].items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}: {'Active' if status else 'Inactive'}")

        # Hook statistics
        if 'hook_statistics' in integration_status:
            hook_stats = integration_status['hook_statistics']
            print(f"\n🔗 Hook Statistics:")
            print(f"   Total hook types: {len(hook_stats.get('hook_types', []))}")
            print(f"   Category distribution: {hook_stats.get('category_counts', {})}")

        # Analytics summary
        if 'analytics_summary' in integration_status:
            analytics = integration_status['analytics_summary']
            print(f"\n📈 Analytics Summary:")
            print(f"   Total executions: {analytics.get('total_executions', 0)}")
            print(f"   Average efficiency: {analytics.get('average_efficiency', 0):.1f}")

        # Monitoring status
        if 'monitoring_status' in integration_status:
            monitoring = integration_status['monitoring_status']
            print(f"\n📡 Monitoring Status:")
            print(f"   Active alerts: {monitoring.get('active_alerts', 0)}")
            print(f"   System health: {monitoring.get('system_health', 'Unknown')}")

        # Performance metrics
        if self.integrator.real_time_monitor:
            health = self.integrator.real_time_monitor.create_health_snapshot()
            print(f"\n🏥 Current System Health:")
            print(f"   Overall: {health.overall_health.value}")
            print(f"   Error rate: {health.error_rate:.1%}")
            print(f"   Response time: {health.average_response_time:.2f}s")

        print(f"\n✨ Phase 3.1 Demo completed successfully!")
        print(f"   Session ID: {self.session_id}")
        print(f"   Demo Type: {self.demo_type}")
        print(f"   Duration: {(datetime.now() - datetime.fromtimestamp(int(self.session_id.split('_')[1]))).total_seconds():.1f}s")

    async def _cleanup_demo(self):
        """Cleanup demo resources."""
        print("\n🧹 Cleaning up demo resources...")

        if self.integrator:
            await self.integrator.shutdown()

        print("✅ Cleanup completed")


class MockEnhancedOrchestrator:
    """Mock enhanced orchestrator for demo purposes."""

    def __init__(self, session_id: str, logger):
        self.session_id = session_id
        self.logger = logger
        self.hook_manager = None

    async def execute_enhanced_research_workflow(self, session_id: str):
        """Mock workflow execution."""
        self.logger.info(f"Mock workflow execution for session {session_id}")
        await asyncio.sleep(0.1)  # Simulate some work
        return {"status": "completed", "session_id": session_id}


async def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(description="Phase 3.1 Comprehensive Hooks System Demo")
    parser.add_argument(
        "--demo-type",
        choices=["basic", "standard", "comprehensive"],
        default="comprehensive",
        help="Type of demo to run (default: comprehensive)"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        help="Custom session ID for the demo"
    )

    args = parser.parse_args()

    try:
        # Run the demo
        demo = Phase31Demo(args.demo_type, args.session_id)
        await demo.run_demo()

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())