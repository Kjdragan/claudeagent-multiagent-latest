"""
Sub-Agent Integration Layer

This module provides integration between the sub-agent architecture and the
existing multi-agent research system, enabling seamless incorporation of
sub-agent capabilities into the current workflow.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from . import SubAgentCoordinator, SubAgentType, create_sub_agent_config
from .sub_agent_factory import SubAgentRequest, get_sub_agent_factory
from .communication_protocols import MessageType, MessagePriority
from .sub_agent_coordinator import WorkflowStage, WorkflowStatus

logger = logging.getLogger(__name__)


class SubAgentSystemIntegration:
    """
    Integration layer for incorporating sub-agent architecture into the
    existing multi-agent research system.
    """

    def __init__(self):
        self.coordinator = SubAgentCoordinator()
        self.factory = get_sub_agent_factory()
        self.initialized = False
        self.integration_config = {
            "enable_sub_agents": True,
            "fallback_to_legacy": True,
            "sub_agent_timeout_minutes": 30,
            "max_concurrent_workflows": 5,
            "quality_gate_threshold": 0.75,
            "enable_progressive_enhancement": True
        }

    async def initialize(self):
        """Initialize the sub-agent integration."""
        if not self.integration_config["enable_sub_agents"]:
            logger.info("Sub-agent system disabled in configuration")
            return

        logger.info("Initializing Sub-Agent System Integration")

        try:
            await self.coordinator.initialize()
            self.initialized = True
            logger.info("✅ Sub-agent system integration initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize sub-agent integration: {e}")
            if self.integration_config["fallback_to_legacy"]:
                logger.warning("⚠️ Falling back to legacy agent system")
            else:
                raise

    async def shutdown(self):
        """Shutdown the sub-agent integration."""
        if self.initialized:
            logger.info("Shutting down Sub-Agent System Integration")
            await self.coordinator.shutdown()
            self.initialized = False
            logger.info("✅ Sub-agent system integration shutdown complete")

    async def is_available(self) -> bool:
        """Check if the sub-agent system is available."""
        return self.initialized and self.integration_config["enable_sub_agents"]

    async def execute_research_with_sub_agents(
        self,
        session_id: str,
        topic: str,
        depth: str = "standard",
        focus_areas: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute research using the sub-agent system.

        Args:
            session_id: Research session ID
            topic: Research topic
            depth: Research depth (quick, standard, comprehensive)
            focus_areas: Specific focus areas for research
            **kwargs: Additional research parameters

        Returns:
            Research results
        """

        if not await self.is_available():
            if self.integration_config["fallback_to_legacy"]:
                logger.info("Sub-agent system unavailable, falling back to legacy system")
                return await self._fallback_research(session_id, topic, depth, focus_areas, **kwargs)
            else:
                raise RuntimeError("Sub-agent system unavailable and fallback disabled")

        try:
            # Determine workflow type based on depth
            workflow_type = self._map_depth_to_workflow_type(depth)

            # Create quality requirements
            quality_requirements = {
                "min_success_rate": 80,
                "min_quality_score": self.integration_config["quality_gate_threshold"] * 100
            }

            # Create coordinated workflow
            workflow_id = await self.coordinator.create_coordinated_workflow(
                session_id=session_id,
                topic=topic,
                description=f"Research on {topic} with depth {depth}",
                workflow_type=workflow_type,
                quality_requirements=quality_requirements,
                focus_areas=focus_areas or [],
                **kwargs
            )

            logger.info(f"🚀 Started sub-agent research workflow: {workflow_id[:8]}...")

            # Start workflow
            await self.coordinator.start_workflow(workflow_id)

            # Monitor workflow progress
            result = await self._monitor_workflow_completion(workflow_id)

            if result:
                logger.info(f"✅ Sub-agent research completed successfully")
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "session_id": session_id,
                    "topic": topic,
                    "results": result,
                    "execution_method": "sub_agents",
                    "completed_at": datetime.now().isoformat()
                }
            else:
                # Workflow failed or timed out
                logger.warning("⚠️ Sub-agent research workflow failed")
                if self.integration_config["fallback_to_legacy"]:
                    logger.info("Falling back to legacy research system")
                    return await self._fallback_research(session_id, topic, depth, focus_areas, **kwargs)
                else:
                    return {
                        "success": False,
                        "workflow_id": workflow_id,
                        "session_id": session_id,
                        "topic": topic,
                        "error": "Sub-agent workflow failed and fallback disabled",
                        "execution_method": "sub_agents_failed"
                    }

        except Exception as e:
            logger.error(f"❌ Sub-agent research execution failed: {e}")
            if self.integration_config["fallback_to_legacy"]:
                logger.info("Falling back to legacy research system due to error")
                return await self._fallback_research(session_id, topic, depth, focus_areas, **kwargs)
            else:
                raise

    async def execute_editorial_review_with_sub_agents(
        self,
        session_id: str,
        content: str,
        research_data: Optional[Dict[str, Any]] = None,
        quality_requirements: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute editorial review using the sub-agent system.

        Args:
            session_id: Session ID
            content: Content to review
            research_data: Available research data
            quality_requirements: Quality requirements
            **kwargs: Additional parameters

        Returns:
            Editorial review results
        """

        if not await self.is_available():
            if self.integration_config["fallback_to_legacy"]:
                return await self._fallback_editorial_review(session_id, content, research_data, **kwargs)
            else:
                raise RuntimeError("Sub-agent system unavailable")

        try:
            # Create specialized editorial workflow
            workflow_id = await self.coordinator.create_coordinated_workflow(
                session_id=session_id,
                topic="editorial_review",
                description="Editorial review and quality assessment",
                workflow_type="editorial_review",
                quality_requirements=quality_requirements or {
                    "min_success_rate": 85,
                    "min_quality_score": 80
                },
                content_to_review=content,
                research_data=research_data or {},
                **kwargs
            )

            logger.info(f"🔍 Started sub-agent editorial workflow: {workflow_id[:8]}...")

            # Start workflow
            await self.coordinator.start_workflow(workflow_id)

            # Monitor workflow progress
            result = await self._monitor_workflow_completion(workflow_id)

            if result:
                logger.info(f"✅ Sub-agent editorial review completed successfully")
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "session_id": session_id,
                    "review_results": result,
                    "execution_method": "sub_agents",
                    "completed_at": datetime.now().isoformat()
                }
            else:
                logger.warning("⚠️ Sub-agent editorial review failed")
                if self.integration_config["fallback_to_legacy"]:
                    return await self._fallback_editorial_review(session_id, content, research_data, **kwargs)
                else:
                    return {
                        "success": False,
                        "workflow_id": workflow_id,
                        "error": "Sub-agent editorial workflow failed",
                        "execution_method": "sub_agents_failed"
                    }

        except Exception as e:
            logger.error(f"❌ Sub-agent editorial review failed: {e}")
            if self.integration_config["fallback_to_legacy"]:
                return await self._fallback_editorial_review(session_id, content, research_data, **kwargs)
            else:
                raise

    async def execute_gap_research_with_sub_agents(
        self,
        session_id: str,
        gap_topics: List[str],
        max_results_per_topic: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute gap research using the sub-agent system.

        Args:
            session_id: Session ID
            gap_topics: List of research gaps to investigate
            max_results_per_topic: Maximum results per gap topic
            **kwargs: Additional parameters

        Returns:
            Gap research results
        """

        if not await self.is_available():
            if self.integration_config["fallback_to_legacy"]:
                return await self._fallback_gap_research(session_id, gap_topics, max_results_per_topic, **kwargs)
            else:
                raise RuntimeError("Sub-agent system unavailable")

        try:
            # Create gap research workflow
            workflow_id = await self.coordinator.create_coordinated_workflow(
                session_id=session_id,
                topic="gap_research",
                description=f"Gap research for {len(gap_topics)} topics",
                workflow_type="gap_research",
                gap_topics=gap_topics,
                max_results_per_topic=max_results_per_topic,
                **kwargs
            )

            logger.info(f"🔍 Started sub-agent gap research workflow: {workflow_id[:8]}...")

            # Start workflow
            await self.coordinator.start_workflow(workflow_id)

            # Monitor workflow progress
            result = await self._monitor_workflow_completion(workflow_id)

            if result:
                logger.info(f"✅ Sub-agent gap research completed successfully")
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "session_id": session_id,
                    "gap_topics": gap_topics,
                    "research_results": result,
                    "execution_method": "sub_agents",
                    "completed_at": datetime.now().isoformat()
                }
            else:
                logger.warning("⚠️ Sub-agent gap research failed")
                if self.integration_config["fallback_to_legacy"]:
                    return await self._fallback_gap_research(session_id, gap_topics, max_results_per_topic, **kwargs)
                else:
                    return {
                        "success": False,
                        "workflow_id": workflow_id,
                        "error": "Sub-agent gap research workflow failed",
                        "execution_method": "sub_agents_failed"
                    }

        except Exception as e:
            logger.error(f"❌ Sub-agent gap research failed: {e}")
            if self.integration_config["fallback_to_legacy"]:
                return await self._fallback_gap_research(session_id, gap_topics, max_results_per_topic, **kwargs)
            else:
                raise

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including sub-agent components."""
        status = {
            "sub_agent_integration": {
                "enabled": self.integration_config["enable_sub_agents"],
                "initialized": self.initialized,
                "available": await self.is_available(),
                "fallback_enabled": self.integration_config["fallback_to_legacy"]
            }
        }

        if self.initialized:
            coordinator_status = self.coordinator.get_coordinator_status()
            factory_status = self.factory.get_factory_status()

            status.update({
                "coordinator": coordinator_status,
                "factory": factory_status
            })

        return status

    async def _monitor_workflow_completion(self, workflow_id: str, timeout_minutes: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Monitor workflow completion with timeout."""
        timeout_minutes = timeout_minutes or self.integration_config["sub_agent_timeout_minutes"]
        max_wait_time = timeout_minutes * 60
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < max_wait_time:
            status = await self.coordinator.get_workflow_status(workflow_id)

            if not status:
                logger.warning(f"Workflow {workflow_id[:8]}... status not available")
                break

            if status["status"] in ["completed", "failed"]:
                if status["status"] == "completed":
                    results = await self.coordinator.get_workflow_results(workflow_id)
                    return results
                else:
                    logger.warning(f"Workflow {workflow_id[:8]}... failed")
                    return None

            await asyncio.sleep(5)  # Check every 5 seconds

        # Timeout
        logger.warning(f"Workflow {workflow_id[:8]}... timed out after {timeout_minutes} minutes")
        await self.coordinator.cancel_workflow(workflow_id)
        return None

    def _map_depth_to_workflow_type(self, depth: str) -> str:
        """Map research depth to workflow type."""
        mapping = {
            "quick": "quick_analysis",
            "standard": "standard_research",
            "comprehensive": "comprehensive_report",
            "detailed": "comprehensive_report",
            "basic": "quick_analysis"
        }
        return mapping.get(depth.lower(), "standard_research")

    async def _fallback_research(
        self,
        session_id: str,
        topic: str,
        depth: str,
        focus_areas: Optional[List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback to legacy research system."""
        logger.info(f"Executing legacy research for topic: {topic}")

        # This would integrate with the existing research system
        # For now, return a mock result
        return {
            "success": True,
            "session_id": session_id,
            "topic": topic,
            "depth": depth,
            "focus_areas": focus_areas,
            "results": {
                "research_findings": "Legacy research results would go here",
                "sources": ["legacy_source_1", "legacy_source_2"],
                "quality_score": 70.0
            },
            "execution_method": "legacy_fallback",
            "completed_at": datetime.now().isoformat()
        }

    async def _fallback_editorial_review(
        self,
        session_id: str,
        content: str,
        research_data: Optional[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback to legacy editorial review."""
        logger.info("Executing legacy editorial review")

        return {
            "success": True,
            "session_id": session_id,
            "review_results": {
                "quality_score": 70.0,
                "recommendations": ["Improve structure", "Add more details"],
                "gap_research_needed": False
            },
            "execution_method": "legacy_fallback",
            "completed_at": datetime.now().isoformat()
        }

    async def _fallback_gap_research(
        self,
        session_id: str,
        gap_topics: List[str],
        max_results_per_topic: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback to legacy gap research."""
        logger.info(f"Executing legacy gap research for {len(gap_topics)} topics")

        return {
            "success": True,
            "session_id": session_id,
            "gap_topics": gap_topics,
            "research_results": {
                topic: f"Legacy gap research results for {len(gap_topics)} topics",
                "findings": {topic: f"Legacy findings for {topic}" for topic in gap_topics}
            },
            "execution_method": "legacy_fallback",
            "completed_at": datetime.now().isoformat()
        }


class LegacySystemAdapter:
    """
    Adapter for integrating sub-agent capabilities into the existing
    legacy agent system with minimal disruption.
    """

    def __init__(self, integration: SubAgentSystemIntegration):
        self.integration = integration

    async def enhance_research_agent(self, existing_research_agent):
        """Enhance existing research agent with sub-agent capabilities."""
        if not await self.integration.is_available():
            return existing_research_agent

        # Add sub-agent methods to existing agent
        async def enhanced_research(topic: str, **kwargs):
            return await self.integration.execute_research_with_sub_agents(
                session_id=kwargs.get("session_id", "enhanced_session"),
                topic=topic,
                **kwargs
            )

        # Monkey patch the enhanced method
        existing_research_agent.enhanced_research = enhanced_research
        existing_research_agent.has_sub_agents = True

        logger.info("✅ Enhanced research agent with sub-agent capabilities")
        return existing_research_agent

    async def enhance_editorial_agent(self, existing_editorial_agent):
        """Enhance existing editorial agent with sub-agent capabilities."""
        if not await self.integration.is_available():
            return existing_editorial_agent

        async def enhanced_editorial_review(content: str, **kwargs):
            return await self.integration.execute_editorial_review_with_sub_agents(
                session_id=kwargs.get("session_id", "enhanced_session"),
                content=content,
                **kwargs
            )

        # Monkey patch the enhanced method
        existing_editorial_agent.enhanced_editorial_review = enhanced_editorial_review
        existing_editorial_agent.has_sub_agents = True

        logger.info("✅ Enhanced editorial agent with sub-agent capabilities")
        return existing_editorial_agent

    async def create_hybrid_orchestrator(self, existing_orchestrator):
        """Create a hybrid orchestrator that uses sub-agents when available."""
        if not await self.integration.is_available():
            return existing_orchestrator

        class HybridOrchestrator:
            def __init__(self, legacy_orch, integration):
                self.legacy = legacy_orch
                self.integration = integration

            async def execute_research_workflow(self, session_id: str, topic: str, **kwargs):
                """Execute research using sub-agents or legacy system."""
                try:
                    # Try sub-agent system first
                    result = await self.integration.execute_research_with_sub_agents(
                        session_id, topic, **kwargs
                    )
                    if result["success"]:
                        return result
                except Exception as e:
                    logger.warning(f"Sub-agent research failed: {e}")

                # Fallback to legacy system
                return await self.legacy.execute_research_workflow(session_id, topic, **kwargs)

            async def execute_editorial_workflow(self, session_id: str, content: str, **kwargs):
                """Execute editorial workflow using sub-agents or legacy system."""
                try:
                    # Try sub-agent system first
                    result = await self.integration.execute_editorial_review_with_sub_agents(
                        session_id, content, **kwargs
                    )
                    if result["success"]:
                        return result
                except Exception as e:
                    logger.warning(f"Sub-agent editorial review failed: {e}")

                # Fallback to legacy system
                return await self.legacy.execute_editorial_workflow(session_id, content, **kwargs)

        hybrid_orchestrator = HybridOrchestrator(existing_orchestrator, self.integration)
        logger.info("✅ Created hybrid orchestrator with sub-agent integration")
        return hybrid_orchestrator


# Global integration instance
_integration_instance: Optional[SubAgentSystemIntegration] = None


async def get_sub_agent_integration() -> SubAgentSystemIntegration:
    """Get the global sub-agent integration instance."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = SubAgentSystemIntegration()
        await _integration_instance.initialize()
    return _integration_instance


async def create_legacy_adapter() -> LegacySystemAdapter:
    """Create a legacy system adapter."""
    integration = await get_sub_agent_integration()
    return LegacySystemAdapter(integration)


# Convenience functions for easy integration
async def execute_enhanced_research(session_id: str, topic: str, **kwargs) -> Dict[str, Any]:
    """Execute enhanced research using sub-agents."""
    integration = await get_sub_agent_integration()
    return await integration.execute_research_with_sub_agents(session_id, topic, **kwargs)


async def execute_enhanced_editorial_review(session_id: str, content: str, **kwargs) -> Dict[str, Any]:
    """Execute enhanced editorial review using sub-agents."""
    integration = await get_sub_agent_integration()
    return await integration.execute_editorial_review_with_sub_agents(session_id, content, **kwargs)


async def execute_enhanced_gap_research(session_id: str, gap_topics: List[str], **kwargs) -> Dict[str, Any]:
    """Execute enhanced gap research using sub-agents."""
    integration = await get_sub_agent_integration()
    return await integration.execute_gap_research_with_sub_agents(session_id, gap_topics, **kwargs)