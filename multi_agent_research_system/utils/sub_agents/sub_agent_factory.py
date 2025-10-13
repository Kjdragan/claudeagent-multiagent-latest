"""
Sub-Agent Factory

This module provides factory patterns for creating, configuring, and managing
sub-agents with proper context isolation and specialized capabilities.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
import logging

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from .sub_agent_types import (
    SubAgentType, SubAgentConfiguration, create_sub_agent_config,
    SubAgentCapabilities, SubAgentPersona
)
from .context_isolation import ContextIsolationManager
from .communication_protocols import SubAgentCommunicationManager
from .performance_monitor import SubAgentPerformanceMonitor


logger = logging.getLogger(__name__)


@dataclass
class SubAgentInstance:
    """Represents an active sub-agent instance."""

    instance_id: str
    agent_type: SubAgentType
    configuration: SubAgentConfiguration
    client: ClaudeSDKClient
    created_at: datetime
    last_activity: datetime
    session_context: Dict[str, Any] = field(default_factory=dict)
    isolation_context: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active, idle, error, terminated

    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()

    def is_expired(self, max_idle_minutes: int = 30) -> bool:
        """Check if the sub-agent has expired due to inactivity."""
        idle_time = (datetime.now() - self.last_activity).total_seconds() / 60
        return idle_time > max_idle_minutes


@dataclass
class SubAgentRequest:
    """Request for creating a new sub-agent."""

    agent_type: SubAgentType
    task_description: str
    session_id: str
    parent_agent: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1-5, 1 being highest
    timeout_seconds: int = 300
    isolation_level: str = "moderate"
    custom_config: Optional[Dict[str, Any]] = None


@dataclass
class SubAgentResult:
    """Result from sub-agent execution."""

    instance_id: str
    agent_type: SubAgentType
    success: bool
    result_data: Dict[str, Any]
    execution_time: float
    quality_score: Optional[float] = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SubAgentFactory:
    """
    Factory for creating and managing sub-agents with context isolation,
    proper configuration, and performance monitoring.
    """

    def __init__(self):
        self.active_instances: Dict[str, SubAgentInstance] = {}
        self.isolation_manager = ContextIsolationManager()
        self.communication_manager = SubAgentCommunicationManager()
        self.performance_monitor = SubAgentPerformanceMonitor()
        self.factory_config = {
            "max_concurrent_agents": 10,
            "default_timeout": 300,
            "cleanup_interval": 60,  # seconds
            "max_idle_time": 1800,  # 30 minutes
            "enable_performance_tracking": True,
            "enable_context_isolation": True
        }
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self):
        """Initialize the sub-agent factory."""
        logger.info("Initializing Sub-Agent Factory")
        self._running = True

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Initialize managers
        await self.isolation_manager.initialize()
        await self.communication_manager.initialize()
        await self.performance_monitor.initialize()

    async def shutdown(self):
        """Shutdown the sub-agent factory."""
        logger.info("Shutting down Sub-Agent Factory")
        self._running = False

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cleanup all active instances
        await self.cleanup_all_instances()

        # Shutdown managers
        await self.isolation_manager.shutdown()
        await self.communication_manager.shutdown()
        await self.performance_monitor.shutdown()

    async def create_sub_agent(self, request: SubAgentRequest) -> SubAgentInstance:
        """
        Create a new sub-agent instance.

        Args:
            request: Sub-agent creation request

        Returns:
            Created sub-agent instance
        """

        # Check concurrent agent limit
        if len(self.active_instances) >= self.factory_config["max_concurrent_agents"]:
            await self._cleanup_expired_instances()
            if len(self.active_instances) >= self.factory_config["max_concurrent_agents"]:
                raise RuntimeError("Maximum concurrent sub-agents limit reached")

        # Generate instance ID
        instance_id = str(uuid.uuid4())

        # Create configuration
        config = create_sub_agent_config(
            request.agent_type,
            **(request.custom_config or {})
        )

        # Apply request-specific modifications
        if request.timeout_seconds:
            config.capabilities.timeout_seconds = request.timeout_seconds
        config.isolation_level = request.isolation_level

        # Create isolation context if enabled
        isolation_context = None
        if self.factory_config["enable_context_isolation"]:
            isolation_context = await self.isolation_manager.create_isolation_context(
                instance_id, request.agent_type, request.session_id
            )

        # Create Claude SDK client
        client = await self._create_claude_client(config, isolation_context)

        # Create sub-agent instance
        instance = SubAgentInstance(
            instance_id=instance_id,
            agent_type=request.agent_type,
            configuration=config,
            client=client,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            session_context=request.context_data,
            isolation_context=isolation_context
        )

        # Store instance
        self.active_instances[instance_id] = instance

        # Log creation
        logger.info(f"Created sub-agent {instance_id} of type {request.agent_type.value}")

        # Track performance
        if self.factory_config["enable_performance_tracking"]:
            await self.performance_monitor.track_agent_creation(instance)

        return instance

    async def execute_sub_agent_task(
        self,
        instance_id: str,
        task_prompt: str,
        context_data: Optional[Dict[str, Any]] = None
    ) -> SubAgentResult:
        """
        Execute a task using a specific sub-agent instance.

        Args:
            instance_id: ID of the sub-agent instance
            task_prompt: Task prompt for the sub-agent
            context_data: Additional context data

        Returns:
            Result from sub-agent execution
        """

        if instance_id not in self.active_instances:
            raise ValueError(f"Sub-agent instance {instance_id} not found")

        instance = self.active_instances[instance_id]
        start_time = datetime.now()

        try:
            # Update instance activity
            instance.update_activity()

            # Prepare task context
            task_context = {
                **instance.session_context,
                **(context_data or {}),
                "instance_id": instance_id,
                "agent_type": instance.agent_type.value,
                "execution_context": "sub_agent_task"
            }

            # Create enhanced prompt with context
            enhanced_prompt = self._create_enhanced_prompt(
                task_prompt, task_context, instance.configuration.persona
            )

            # Execute task
            logger.debug(f"Executing task on sub-agent {instance_id}")
            await instance.client.query(enhanced_prompt)

            # Collect response
            response_data = []
            async for message in instance.client.receive_response():
                response_data.append(message)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = SubAgentResult(
                instance_id=instance_id,
                agent_type=instance.agent_type,
                success=True,
                result_data={"messages": response_data, "context": task_context},
                execution_time=execution_time
            )

            # Update performance metrics
            if self.factory_config["enable_performance_tracking"]:
                result.performance_metrics = await self.performance_monitor.track_execution(
                    instance, execution_time, result.success
                )

            logger.info(f"Task completed on sub-agent {instance_id} in {execution_time:.2f}s")
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Task execution failed: {str(e)}"

            logger.error(f"Task failed on sub-agent {instance_id}: {error_message}")

            # Update instance status
            instance.status = "error"

            # Track error performance
            if self.factory_config["enable_performance_tracking"]:
                await self.performance_monitor.track_execution_error(instance, str(e))

            return SubAgentResult(
                instance_id=instance_id,
                agent_type=instance.agent_type,
                success=False,
                result_data={},
                execution_time=execution_time,
                error_message=error_message
            )

    async def create_and_execute(
        self,
        request: SubAgentRequest,
        task_prompt: str
    ) -> SubAgentResult:
        """
        Create a sub-agent and execute a task in one operation.

        Args:
            request: Sub-agent creation request
            task_prompt: Task prompt for execution

        Returns:
            Result from sub-agent execution
        """

        instance = await self.create_sub_agent(request)

        try:
            result = await self.execute_sub_agent_task(
                instance.instance_id,
                task_prompt,
                request.context_data
            )
            return result
        finally:
            # Cleanup instance after execution
            await self.cleanup_instance(instance.instance_id)

    async def get_instance(self, instance_id: str) -> Optional[SubAgentInstance]:
        """Get a sub-agent instance by ID."""
        return self.active_instances.get(instance_id)

    async def cleanup_instance(self, instance_id: str):
        """Cleanup a specific sub-agent instance."""
        if instance_id not in self.active_instances:
            return

        instance = self.active_instances[instance_id]

        try:
            # Disconnect client
            if instance.client:
                await instance.client.disconnect()

            # Cleanup isolation context
            if instance.isolation_context:
                await self.isolation_manager.cleanup_isolation_context(instance.isolation_context)

            # Remove from active instances
            del self.active_instances[instance_id]

            logger.info(f"Cleaned up sub-agent instance {instance_id}")

        except Exception as e:
            logger.error(f"Error cleaning up instance {instance_id}: {e}")

    async def cleanup_all_instances(self):
        """Cleanup all active sub-agent instances."""
        instance_ids = list(self.active_instances.keys())
        for instance_id in instance_ids:
            await self.cleanup_instance(instance_id)

    async def _cleanup_expired_instances(self):
        """Cleanup expired sub-agent instances."""
        expired_instances = [
            instance_id for instance_id, instance in self.active_instances.items()
            if instance.is_expired(self.factory_config["max_idle_time"])
        ]

        for instance_id in expired_instances:
            logger.info(f"Cleaning up expired instance {instance_id}")
            await self.cleanup_instance(instance_id)

    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.factory_config["cleanup_interval"])
                await self._cleanup_expired_instances()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _create_claude_client(
        self,
        config: SubAgentConfiguration,
        isolation_context: Optional[str]
    ) -> ClaudeSDKClient:
        """Create a Claude SDK client with the specified configuration."""

        # Modify options for isolation context
        options = config.claude_options

        if isolation_context:
            # Add isolation context to system prompt
            original_prompt = options.system_prompt or ""
            isolation_prompt = f"\n\nISOLATION CONTEXT: You are operating in isolated context {isolation_context}. "
            isolation_prompt += "Maintain strict context isolation and do not share information outside this context."
            options.system_prompt = original_prompt + isolation_prompt

        # Create client
        client = ClaudeSDKClient(options=options)
        await client.connect()

        return client

    def _create_enhanced_prompt(
        self,
        task_prompt: str,
        context_data: Dict[str, Any],
        persona: SubAgentPersona
    ) -> str:
        """Create an enhanced prompt with context and persona information."""

        enhanced_prompt = f"""You are {persona.name}, {persona.description}.

{persona.system_prompt}

CURRENT TASK:
{task_prompt}

CONTEXT INFORMATION:
"""

        # Add relevant context data
        for key, value in context_data.items():
            if key not in ["system_prompt", "task_prompt"]:
                enhanced_prompt += f"- {key}: {value}\n"

        enhanced_prompt += """

Please execute this task according to your specialized capabilities and standards.
Provide clear, actionable results and maintain professional communication throughout.
"""

        return enhanced_prompt

    def get_factory_status(self) -> Dict[str, Any]:
        """Get the current status of the sub-agent factory."""

        active_instances_by_type = {}
        for instance in self.active_instances.values():
            agent_type = instance.agent_type.value
            if agent_type not in active_instances_by_type:
                active_instances_by_type[agent_type] = 0
            active_instances_by_type[agent_type] += 1

        return {
            "running": self._running,
            "active_instances": len(self.active_instances),
            "max_concurrent": self.factory_config["max_concurrent_agents"],
            "instances_by_type": active_instances_by_type,
            "performance_tracking_enabled": self.factory_config["enable_performance_tracking"],
            "context_isolation_enabled": self.factory_config["enable_context_isolation"]
        }

    def get_instance_metrics(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific instance."""
        if instance_id not in self.active_instances:
            return None

        instance = self.active_instances[instance_id]

        return {
            "instance_id": instance_id,
            "agent_type": instance.agent_type.value,
            "status": instance.status,
            "created_at": instance.created_at.isoformat(),
            "last_activity": instance.last_activity.isoformat(),
            "uptime_seconds": (datetime.now() - instance.created_at).total_seconds(),
            "idle_time_seconds": (datetime.now() - instance.last_activity).total_seconds(),
            "performance_metrics": instance.performance_metrics,
            "isolation_context": instance.isolation_context
        }


# Global factory instance
_sub_agent_factory: Optional[SubAgentFactory] = None


def get_sub_agent_factory() -> SubAgentFactory:
    """Get the global sub-agent factory instance."""
    global _sub_agent_factory
    if _sub_agent_factory is None:
        _sub_agent_factory = SubAgentFactory()
    return _sub_agent_factory


async def initialize_sub_agent_factory():
    """Initialize the global sub-agent factory."""
    factory = get_sub_agent_factory()
    await factory.initialize()


async def shutdown_sub_agent_factory():
    """Shutdown the global sub-agent factory."""
    global _sub_agent_factory
    if _sub_agent_factory:
        await _sub_agent_factory.shutdown()
        _sub_agent_factory = None