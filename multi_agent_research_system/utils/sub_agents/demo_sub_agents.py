#!/usr/bin/env python3
"""
Sub-Agent Architecture Demo

This script demonstrates the comprehensive sub-agent architecture with
specialized roles, context isolation, communication protocols, and
performance monitoring.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.sub_agents import (
    SubAgentFactory, SubAgentCoordinator, SubAgentType,
    ContextIsolationManager, SubAgentCommunicationManager,
    SubAgentPerformanceMonitor, create_sub_agent_config
)
from utils.sub_agents.sub_agent_types import SubAgentRequest


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SubAgentDemo:
    """Demonstration of the sub-agent architecture."""

    def __init__(self):
        self.factory = SubAgentFactory()
        self.coordinator = SubAgentCoordinator()
        self.isolation_manager = ContextIsolationManager()
        self.communication_manager = SubAgentCommunicationManager()
        self.performance_monitor = SubAgentPerformanceMonitor()

    async def initialize(self):
        """Initialize all components."""
        logger.info("=== Initializing Sub-Agent Architecture Demo ===")

        await self.factory.initialize()
        await self.coordinator.initialize()
        await self.isolation_manager.initialize()
        await self.communication_manager.initialize()
        await self.performance_monitor.initialize()

        logger.info("✅ All components initialized successfully")

    async def shutdown(self):
        """Shutdown all components."""
        logger.info("=== Shutting Down Sub-Agent Architecture Demo ===")

        await self.coordinator.shutdown()
        await self.communication_manager.shutdown()
        await self.isolation_manager.shutdown()
        await self.performance_monitor.shutdown()
        await self.factory.shutdown()

        logger.info("✅ All components shutdown successfully")

    async def demo_individual_sub_agents(self):
        """Demo individual sub-agent creation and execution."""
        logger.info("\n=== Demo 1: Individual Sub-Agent Creation and Execution ===")

        # Demo researcher sub-agent
        logger.info("🔍 Creating researcher sub-agent...")
        research_request = SubAgentRequest(
            agent_type=SubAgentType.RESEARCHER,
            task_description="Research the latest developments in quantum computing",
            session_id="demo_session_001",
            parent_agent="demo",
            priority=1
        )

        research_result = await self.factory.create_and_execute(
            research_request,
            "Please conduct comprehensive research on quantum computing developments in 2024. Focus on breakthrough applications, major research papers, and industry adoption trends."
        )

        logger.info(f"📊 Research completed: Success={research_result.success}, Time={research_result.execution_time:.2f}s")

        # Demo quality assessor sub-agent
        logger.info("🎯 Creating quality assessor sub-agent...")
        quality_request = SubAgentRequest(
            agent_type=SubAgentType.QUALITY_ASSESSOR,
            task_description="Assess the quality of research findings",
            session_id="demo_session_001",
            parent_agent="demo",
            priority=2
        )

        quality_result = await self.factory.create_and_execute(
            quality_request,
            "Please assess the quality of the research findings with focus on accuracy, completeness, and relevance. Provide a detailed quality report with scores and recommendations."
        )

        logger.info(f"📊 Quality assessment completed: Success={quality_result.success}, Time={quality_result.execution_time:.2f}s")

    async def demo_context_isolation(self):
        """Demo context isolation between sub-agents."""
        logger.info("\n=== Demo 2: Context Isolation Between Sub-Agents ===")

        # Create two separate contexts
        context_1 = await self.isolation_manager.create_isolation_context(
            agent_type="researcher",
            session_id="demo_session_002",
            isolation_level="strict"
        )

        context_2 = await self.isolation_manager.create_isolation_context(
            agent_type="report_writer",
            session_id="demo_session_002",
            isolation_level="strict"
        )

        logger.info(f"🔒 Created isolation contexts: {context_1[:8]}..., {context_2[:8]}...")

        # Store data in first context
        await self.isolation_manager.store_data_in_context(
            context_1, "research_findings", {"quantum_breakthrough": "New quantum algorithm discovered"}
        )

        # Try to access data from wrong context (should fail)
        try:
            await self.isolation_manager.retrieve_data_from_context(context_2, "research_findings")
            logger.error("❌ Context isolation failed - data was accessible from wrong context")
        except Exception as e:
            logger.info("✅ Context isolation working - data correctly blocked")

        # Share data between contexts with proper rules
        sharing_success = await self.isolation_manager.share_data_between_contexts(
            context_1, context_2, "research_findings"
        )

        logger.info(f"📡 Data sharing result: {'Success' if sharing_success else 'Blocked'}")

        # Cleanup contexts
        await self.isolation_manager.cleanup_isolation_context(context_1)
        await self.isolation_manager.cleanup_isolation_context(context_2)

    async def demo_communication_protocols(self):
        """Demo inter-agent communication protocols."""
        logger.info("\n=== Demo 3: Inter-Agent Communication Protocols ===")

        # Register message handlers
        await self.communication_manager.register_message_handler(
            agent_id="demo_researcher",
            agent_type="researcher",
            message_types=["request", "notification"],
            handler_function=self._handle_demo_message
        )

        await self.communication_manager.register_message_handler(
            agent_id="demo_writer",
            agent_type="report_writer",
            message_types=["request", "notification"],
            handler_function=self._handle_demo_message
        )

        # Send direct message
        message_id = await self.communication_manager.send_direct_message(
            sender_id="demo_researcher",
            sender_type="researcher",
            recipient_id="demo_writer",
            recipient_type="report_writer",
            session_id="demo_session_003",
            payload={
                "action": "generate_report",
                "topic": "quantum computing research",
                "priority": "high"
            }
        )

        logger.info(f"📨 Sent direct message: {message_id[:8]}...")

        # Send broadcast message
        broadcast_id = await self.communication_manager.broadcast_message(
            sender_id="coordinator",
            sender_type="coordinator",
            session_id="demo_session_003",
            payload={
                "announcement": "Research phase completed",
                "next_phase": "report_generation"
            }
        )

        logger.info(f"📢 Sent broadcast message: {broadcast_id[:8]}...")

        # Wait a bit for message processing
        await asyncio.sleep(1)

        # Show communication stats
        stats = self.communication_manager.get_communication_stats()
        logger.info(f"📊 Communication stats: {stats['messages_sent']} sent, {stats['messages_received']} received")

    async def _handle_demo_message(self, message):
        """Handle demo messages."""
        logger.info(f"📬 Received message: {message.message_type.value} from {message.sender_id}")
        return None

    async def demo_coordinated_workflow(self):
        """Demo coordinated workflow with multiple sub-agents."""
        logger.info("\n=== Demo 4: Coordinated Multi-Agent Workflow ===")

        # Create a coordinated workflow
        workflow_id = await self.coordinator.create_coordinated_workflow(
            session_id="demo_session_004",
            topic="The Impact of Artificial Intelligence on Healthcare",
            description="Comprehensive research and analysis of AI applications in healthcare",
            workflow_type="standard_research",
            quality_requirements={
                "min_success_rate": 80,
                "min_quality_score": 75
            }
        )

        logger.info(f"🚀 Created coordinated workflow: {workflow_id[:8]}...")

        # Start the workflow
        await self.coordinator.start_workflow(workflow_id)

        # Monitor workflow progress
        max_wait_time = 120  # 2 minutes
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < max_wait_time:
            status = await self.coordinator.get_workflow_status(workflow_id)

            if status:
                logger.info(f"📊 Workflow status: {status['status']} - Progress: {status['progress_percentage']:.1f}%")

                if status['status'] in ['completed', 'failed']:
                    break

            await asyncio.sleep(5)

        # Get final results
        final_status = await self.coordinator.get_workflow_status(workflow_id)
        if final_status and final_status['status'] == 'completed':
            results = await self.coordinator.get_workflow_results(workflow_id)
            if results:
                logger.info(f"✅ Workflow completed successfully in {results['total_duration']:.2f}s")
                logger.info(f"📋 Task results: {len(results['task_results'])} tasks completed")
            else:
                logger.warning("⚠️ Workflow completed but no results available")
        else:
            logger.warning("⚠️ Workflow did not complete in the expected time")
            await self.coordinator.cancel_workflow(workflow_id)

    async def demo_performance_monitoring(self):
        """Demo performance monitoring capabilities."""
        logger.info("\n=== Demo 5: Performance Monitoring ===")

        # Get system performance
        system_perf = await self.performance_monitor.get_system_performance()
        logger.info(f"💻 System performance: CPU={system_perf.get('system_resources', {}).get('cpu_percent', 0):.1f}%, "
                   f"Memory={system_perf.get('system_resources', {}).get('memory_percent', 0):.1f}%")

        # Get factory status
        factory_status = self.factory.get_factory_status()
        logger.info(f"🏭 Factory status: {factory_status['active_instances']} active sub-agents")

        # Get isolation status
        isolation_status = self.isolation_manager.get_isolation_status()
        logger.info(f"🔒 Isolation status: {isolation_status['active_contexts']} active contexts")

        # Show monitoring status
        monitoring_status = self.performance_monitor.get_monitoring_status()
        logger.info(f"📈 Monitoring status: {monitoring_status['tracked_agents']} agents tracked")

    async def demo_specialized_configurations(self):
        """Demo specialized sub-agent configurations."""
        logger.info("\n=== Demo 6: Specialized Sub-Agent Configurations ===")

        # Show different agent configurations
        agent_types = [
            SubAgentType.RESEARCHER,
            SubAgentType.REPORT_WRITER,
            SubAgentType.EDITORIAL_REVIEWER,
            SubAgentType.QUALITY_ASSESSOR,
            SubAgentType.GAP_RESEARCHER
        ]

        for agent_type in agent_types:
            config = create_sub_agent_config(agent_type)
            logger.info(f"🤖 {agent_type.value.title()}:")
            logger.info(f"   - Tools: {len(config.capabilities.allowed_tools)} allowed")
            logger.info(f"   - Max turns: {config.capabilities.max_turns}")
            logger.info(f"   - Isolation: {config.isolation_level}")
            logger.info(f"   - Expertise: {', '.join(config.persona.expertise_areas[:3])}...")

        # Show configuration details for a specific agent
        researcher_config = create_sub_agent_config(SubAgentType.RESEARCHER)
        logger.info(f"\n🔍 Researcher Configuration Details:")
        logger.info(f"   - Name: {researcher_config.persona.name}")
        logger.info(f"   - Description: {researcher_config.persona.description}")
        logger.info(f"   - Quality Standards: {researcher_config.persona.quality_standards}")

    async def run_demo(self):
        """Run the complete demonstration."""
        try:
            await self.initialize()

            # Run all demos
            await self.demo_specialized_configurations()
            await self.demo_individual_sub_agents()
            await self.demo_context_isolation()
            await self.demo_communication_protocols()
            await self.demo_coordinated_workflow()
            await self.demo_performance_monitoring()

            logger.info("\n🎉 Sub-Agent Architecture Demo Completed Successfully!")

        except Exception as e:
            logger.error(f"❌ Demo failed with error: {e}")
            raise
        finally:
            await self.shutdown()


async def main():
    """Main demo entry point."""
    demo = SubAgentDemo()
    await demo.run_demo()


if __name__ == "__main__":
    print("🚀 Starting Sub-Agent Architecture Demo")
    print("=" * 60)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)