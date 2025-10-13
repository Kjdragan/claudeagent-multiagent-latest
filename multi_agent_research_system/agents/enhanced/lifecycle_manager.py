"""Agent Lifecycle Management System

This module provides comprehensive lifecycle management for enhanced agents,
including initialization, health monitoring, graceful shutdown, and recovery mechanisms.

Key Features:
- Agent Lifecycle Management
- Health Monitoring and Check-ups
- Graceful Shutdown Procedures
- Recovery and Restart Mechanisms
- Resource Management
- State Persistence and Recovery
"""

import asyncio
import json
import logging
import signal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable

from .base_agent import EnhancedBaseAgent, AgentStatus
from .agent_factory import EnhancedAgentFactory


class LifecycleEvent(Enum):
    """Lifecycle events for agents."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    RECOVERING = "recovering"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


class HealthStatus(Enum):
    """Agent health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class LifecycleEventRecord:
    """Record of a lifecycle event."""
    agent_id: str
    event: LifecycleEvent
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    agent_id: str
    status: HealthStatus
    timestamp: datetime
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_count: int
    last_activity: Optional[datetime]
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AgentLifecycleState:
    """Complete lifecycle state for an agent."""
    agent_id: str
    agent_type: str
    current_status: AgentStatus
    health_status: HealthStatus
    created_at: datetime
    last_heartbeat: datetime
    last_health_check: Optional[datetime]
    total_restarts: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0
    last_error: Optional[datetime]
    last_restart: Optional[datetime]
    uptime_seconds: float = 0.0
    events: List[LifecycleEventRecord] = field(default_factory=list)
    health_history: List[HealthCheckResult] = field(default_factory=list)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    auto_restart_enabled: bool = True
    health_check_interval_seconds: int = 30
    heartbeat_timeout_seconds: int = 120


class AgentLifecycleManager:
    """Comprehensive lifecycle manager for enhanced agents."""

    def __init__(self, factory: EnhancedAgentFactory, state_dir: Optional[Path] = None):
        self.logger = logging.getLogger("lifecycle_manager")
        self.factory = factory
        self.state_dir = state_dir or Path("data/agent_lifecycle")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Agent tracking
        self.agent_states: Dict[str, AgentLifecycleState] = {}
        self.managed_agents: Dict[str, EnhancedBaseAgent] = {}
        self.agent_tasks: Dict[str, Set[asyncio.Task]] = {}

        # Lifecycle monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.heartbeat_monitor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

        # Configuration
        self.default_health_check_interval = 30
        self.default_heartbeat_timeout = 120
        self.max_concurrent_health_checks = 10
        self.health_check_timeout_seconds = 10

        # Event handlers
        self.lifecycle_event_handlers: Dict[LifecycleEvent, List[Callable]] = {}
        self.health_check_handlers: List[Callable[[HealthCheckResult], None]] = []

        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self.shutdown_handlers: List[Callable] = []

        # Statistics
        self.total_agents_created = 0
        self.total_agents_shutdown = 0
        self.total_recoveries_attempted = 0
        self.total_recoveries_successful = 0

        self.logger.info("Agent lifecycle manager initialized")

    async def start(self) -> None:
        """Start the lifecycle manager."""
        self.logger.info("Starting agent lifecycle manager")

        # Load existing state
        await self._load_persistent_state()

        # Start monitoring tasks
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.heartbeat_monitor_task = asyncio.create_task(self._heartbeat_monitor_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Register signal handlers
        self._register_signal_handlers()

        self.logger.info("Agent lifecycle manager started")

    async def stop(self) -> None:
        """Stop the lifecycle manager and shutdown all agents."""
        self.logger.info("Stopping agent lifecycle manager")

        # Signal shutdown
        self.shutdown_event.set()

        # Cancel monitoring tasks
        tasks_to_cancel = [
            self.health_check_task,
            self.heartbeat_monitor_task,
            self.cleanup_task
        ]

        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Shutdown all managed agents
        await self.shutdown_all_agents()

        # Call shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                await handler()
            except Exception as e:
                self.logger.error(f"Error in shutdown handler: {e}")

        # Save state
        await self._save_persistent_state()

        self.logger.info("Agent lifecycle manager stopped")

    async def register_agent(self, agent: EnhancedBaseAgent, auto_restart: bool = True,
                           health_check_interval: Optional[int] = None) -> bool:
        """Register an agent for lifecycle management."""
        try:
            agent_id = agent.agent_id

            if agent_id in self.managed_agents:
                self.logger.warning(f"Agent {agent_id} already registered")
                return False

            # Create lifecycle state
            state = AgentLifecycleState(
                agent_id=agent_id,
                agent_type=agent.agent_type,
                current_status=AgentStatus.INITIALIZING,
                health_status=HealthStatus.HEALTHY,
                created_at=datetime.now(),
                last_heartbeat=datetime.now(),
                auto_restart_enabled=auto_restart,
                health_check_interval_seconds=health_check_interval or self.default_health_check_interval,
                heartbeat_timeout_seconds=self.default_heartbeat_timeout
            )

            # Store agent and state
            self.managed_agents[agent_id] = agent
            self.agent_states[agent_id] = state
            self.agent_tasks[agent_id] = set()

            # Record lifecycle event
            await self._record_lifecycle_event(agent_id, LifecycleEvent.CREATED)

            # Start agent-specific monitoring
            await self._start_agent_monitoring(agent_id)

            self.total_agents_created += 1
            self.logger.info(f"Registered agent {agent_id} for lifecycle management")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False

    async def unregister_agent(self, agent_id: str, graceful_shutdown: bool = True) -> bool:
        """Unregister an agent from lifecycle management."""
        try:
            if agent_id not in self.managed_agents:
                self.logger.warning(f"Agent {agent_id} not registered")
                return False

            agent = self.managed_agents[agent_id]

            # Graceful shutdown
            if graceful_shutdown:
                await self.shutdown_agent(agent_id)
            else:
                await self._force_shutdown_agent(agent_id)

            # Cancel agent-specific tasks
            if agent_id in self.agent_tasks:
                for task in self.agent_tasks[agent_id]:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                del self.agent_tasks[agent_id]

            # Remove from tracking
            del self.managed_agents[agent_id]
            del self.agent_states[agent_id]

            # Record lifecycle event
            await self._record_lifecycle_event(agent_id, LifecycleEvent.TERMINATED)

            self.total_agents_shutdown += 1
            self.logger.info(f"Unregistered agent {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False

    async def shutdown_agent(self, agent_id: str, timeout_seconds: int = 30) -> bool:
        """Gracefully shutdown a specific agent."""
        try:
            if agent_id not in self.managed_agents:
                return False

            agent = self.managed_agents[agent_id]
            state = self.agent_states[agent_id]

            await self._record_lifecycle_event(agent_id, LifecycleEvent.SHUTTING_DOWN)

            # Initiate graceful shutdown
            shutdown_task = asyncio.create_task(agent.shutdown())

            try:
                await asyncio.wait_for(shutdown_task, timeout=timeout_seconds)
                state.current_status = AgentStatus.TERMINATED
                self.logger.info(f"Agent {agent_id} shutdown gracefully")
                return True
            except asyncio.TimeoutError:
                self.logger.warning(f"Agent {agent_id} shutdown timeout, forcing")
                await self._force_shutdown_agent(agent_id)
                return True

        except Exception as e:
            self.logger.error(f"Failed to shutdown agent {agent_id}: {e}")
            return False

    async def shutdown_all_agents(self) -> None:
        """Shutdown all managed agents."""
        self.logger.info("Shutting down all agents")

        # Get all agent IDs
        agent_ids = list(self.managed_agents.keys())

        # Shutdown agents concurrently with timeout
        shutdown_tasks = [
            self.shutdown_agent(agent_id, timeout_seconds=15)
            for agent_id in agent_ids
        ]

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        self.logger.info("All agents shutdown completed")

    async def restart_agent(self, agent_id: str) -> bool:
        """Restart an agent."""
        try:
            if agent_id not in self.agent_states:
                self.logger.error(f"Cannot restart unknown agent: {agent_id}")
                return False

            state = self.agent_states[agent_id]

            # Check if we've exceeded max restart attempts
            if state.recovery_attempts >= state.max_recovery_attempts:
                self.logger.error(f"Max restart attempts exceeded for agent {agent_id}")
                return False

            await self._record_lifecycle_event(agent_id, LifecycleEvent.RECOVERING)

            # Get original configuration
            original_agent = self.managed_agents[agent_id]
            config = original_agent.config

            # Shutdown current agent
            await self._force_shutdown_agent(agent_id)

            # Create new agent instance
            new_agent = await self.factory.create_agent_from_config(config)
            new_agent.agent_id = agent_id  # Preserve ID

            # Initialize new agent
            await new_agent.initialize(self.factory.registry)

            # Update tracking
            self.managed_agents[agent_id] = new_agent
            state.total_restarts += 1
            state.last_restart = datetime.now()
            state.recovery_attempts += 1
            state.current_status = AgentStatus.READY
            state.health_status = HealthStatus.HEALTHY

            # Reset error counters
            state.consecutive_errors = 0

            await self._record_lifecycle_event(agent_id, LifecycleEvent.READY)
            await self._start_agent_monitoring(agent_id)

            self.total_recoveries_successful += 1
            self.logger.info(f"Successfully restarted agent {agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to restart agent {agent_id}: {e}")
            state.recovery_attempts += 1
            return False

    async def _start_agent_monitoring(self, agent_id: str) -> None:
        """Start monitoring tasks for a specific agent."""
        # Heartbeat monitoring
        heartbeat_task = asyncio.create_task(self._monitor_agent_heartbeat(agent_id))
        self.agent_tasks[agent_id].add(heartbeat_task)

        # State update monitoring
        state_task = asyncio.create_task(self._monitor_agent_state(agent_id))
        self.agent_tasks[agent_id].add(state_task)

    async def _monitor_agent_heartbeat(self, agent_id: str) -> None:
        """Monitor agent heartbeat."""
        state = self.agent_states[agent_id]

        while not self.shutdown_event.is_set() and agent_id in self.managed_agents:
            try:
                await asyncio.sleep(state.heartbeat_timeout_seconds / 2)

                if agent_id not in self.managed_agents:
                    break

                # Check if heartbeat is recent
                time_since_heartbeat = datetime.now() - state.last_heartbeat
                if time_since_heartbeat.total_seconds() > state.heartbeat_timeout_seconds:
                    self.logger.warning(f"Agent {agent_id} heartbeat timeout")
                    await self._handle_agent_timeout(agent_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor for {agent_id}: {e}")

    async def _monitor_agent_state(self, agent_id: str) -> None:
        """Monitor agent state changes."""
        agent = self.managed_agents[agent_id]
        state = self.agent_states[agent_id]

        while not self.shutdown_event.is_set() and agent_id in self.managed_agents:
            try:
                await asyncio.sleep(5)

                if agent_id not in self.managed_agents:
                    break

                # Check for status changes
                current_status = agent.status
                if current_status != state.current_status:
                    old_status = state.current_status
                    state.current_status = current_status

                    # Record lifecycle event
                    event_map = {
                        AgentStatus.READY: LifecycleEvent.READY,
                        AgentStatus.BUSY: LifecycleEvent.BUSY,
                        AgentStatus.ERROR: LifecycleEvent.ERROR,
                        AgentStatus.SHUTTING_DOWN: LifecycleEvent.SHUTTING_DOWN,
                        AgentStatus.TERMINATED: LifecycleEvent.TERMINATED
                    }

                    if current_status in event_map:
                        await self._record_lifecycle_event(agent_id, event_map[current_status])

                    self.logger.info(f"Agent {agent_id} status changed: {old_status.value} -> {current_status.value}")

                # Update heartbeat
                if current_status not in [AgentStatus.SHUTTING_DOWN, AgentStatus.TERMINATED]:
                    state.last_heartbeat = datetime.now()

                # Update uptime
                if state.current_status not in [AgentStatus.SHUTTING_DOWN, AgentStatus.TERMINATED]:
                    state.uptime_seconds = (datetime.now() - state.created_at).total_seconds()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in state monitor for {agent_id}: {e}")

    async def _health_check_loop(self) -> None:
        """Main health check loop."""
        while not self.shutdown_event.is_set():
            try:
                # Get agents that need health checks
                agents_to_check = [
                    agent_id for agent_id, state in self.agent_states.items()
                    if (datetime.now() - state.last_health_check).total_seconds() >= state.health_check_interval_seconds
                ]

                if agents_to_check:
                    # Run health checks concurrently with limit
                    semaphore = asyncio.Semaphore(self.max_concurrent_health_checks)
                    health_check_tasks = [
                        self._perform_health_check(agent_id, semaphore)
                        for agent_id in agents_to_check
                    ]

                    if health_check_tasks:
                        await asyncio.gather(*health_check_tasks, return_exceptions=True)

                await asyncio.sleep(10)  # Check every 10 seconds for agents needing health checks

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)

    async def _perform_health_check(self, agent_id: str, semaphore: asyncio.Semaphore) -> Optional[HealthCheckResult]:
        """Perform health check on an agent."""
        async with semaphore:
            try:
                if agent_id not in self.managed_agents:
                    return None

                agent = self.managed_agents[agent_id]
                state = self.agent_states[agent_id]

                start_time = datetime.now()

                # Perform basic health checks
                health_status = HealthStatus.HEALTHY
                issues = []
                recommendations = []

                # Check agent status
                if agent.status == AgentStatus.ERROR:
                    health_status = HealthStatus.CRITICAL
                    issues.append("Agent in ERROR state")
                    recommendations.append("Consider restarting agent")

                elif agent.status == AgentStatus.SHUTTING_DOWN:
                    health_status = HealthStatus.UNHEALTHY
                    issues.append("Agent shutting down")

                # Check heartbeat
                time_since_heartbeat = datetime.now() - state.last_heartbeat
                if time_since_heartbeat.total_seconds() > state.heartbeat_timeout_seconds:
                    health_status = HealthStatus.CRITICAL
                    issues.append(f"Heartbeat timeout: {time_since_heartbeat.total_seconds():.0f}s")
                    recommendations.append("Agent may be unresponsive")

                # Check error rate
                if state.consecutive_errors > 5:
                    health_status = HealthStatus.UNHEALTHY
                    issues.append(f"High consecutive errors: {state.consecutive_errors}")
                    recommendations.append("Monitor agent closely")

                # Get resource usage
                memory_usage = self._get_agent_memory_usage(agent)
                cpu_usage = self._get_agent_cpu_usage(agent)

                if memory_usage > 500:  # 500MB threshold
                    if health_status == HealthStatus.HEALTHY:
                        health_status = HealthStatus.DEGRADED
                    issues.append(f"High memory usage: {memory_usage:.1f}MB")
                    recommendations.append("Monitor memory usage")

                # Calculate response time
                response_time = (datetime.now() - start_time).total_seconds() * 1000

                # Create health check result
                result = HealthCheckResult(
                    agent_id=agent_id,
                    status=health_status,
                    timestamp=datetime.now(),
                    response_time_ms=response_time,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_usage,
                    error_count=state.total_errors,
                    last_activity=state.last_heartbeat,
                    issues=issues,
                    recommendations=recommendations
                )

                # Update state
                state.last_health_check = datetime.now()
                state.health_history.append(result)

                # Keep only recent history (last 100 checks)
                if len(state.health_history) > 100:
                    state.health_history = state.health_history[-100:]

                # Update health status
                state.health_status = health_status

                # Call health check handlers
                for handler in self.health_check_handlers:
                    try:
                        await handler(result)
                    except Exception as e:
                        self.logger.error(f"Error in health check handler: {e}")

                # Handle unhealthy agents
                if health_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    await self._handle_unhealthy_agent(agent_id, result)

                return result

            except Exception as e:
                self.logger.error(f"Health check failed for agent {agent_id}: {e}")
                return None

    async def _handle_unhealthy_agent(self, agent_id: str, health_result: HealthCheckResult) -> None:
        """Handle unhealthy agent based on health status."""
        state = self.agent_states[agent_id]

        if health_result.status == HealthStatus.CRITICAL:
            # Critical - attempt recovery
            if state.auto_restart_enabled and state.recovery_attempts < state.max_recovery_attempts:
                self.logger.warning(f"Agent {agent_id} is critical, attempting restart")
                self.total_recoveries_attempted += 1
                await self.restart_agent(agent_id)
            else:
                self.logger.error(f"Agent {agent_id} is critical and auto-restart disabled or max attempts reached")

        elif health_result.status == HealthStatus.UNHEALTHY:
            # Unhealthy - log and monitor
            self.logger.warning(f"Agent {agent_id} is unhealthy: {health_result.issues}")

    async def _handle_agent_timeout(self, agent_id: str) -> None:
        """Handle agent heartbeat timeout."""
        state = self.agent_states[agent_id]
        state.consecutive_errors += 1
        state.last_error = datetime.now()

        await self._record_lifecycle_event(agent_id, LifecycleEvent.ERROR, {
            "reason": "heartbeat_timeout",
            "consecutive_errors": state.consecutive_errors
        })

        # Attempt recovery if enabled
        if state.auto_restart_enabled and state.consecutive_errors >= 3:
            await self.restart_agent(agent_id)

    async def _heartbeat_monitor_loop(self) -> None:
        """Monitor overall heartbeat status."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute

                # Log overall health status
                total_agents = len(self.agent_states)
                healthy_agents = sum(1 for state in self.agent_states.values() if state.health_status == HealthStatus.HEALTHY)
                unhealthy_agents = total_agents - healthy_agents

                if unhealthy_agents > 0:
                    self.logger.warning(f"Health status: {healthy_agents}/{total_agents} agents healthy")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Cleanup loop for maintenance tasks."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Cleanup old health history
                for state in self.agent_states.values():
                    if len(state.health_history) > 1000:
                        state.health_history = state.health_history[-1000:]

                # Cleanup old event records
                for state in self.agent_states.values():
                    if len(state.events) > 1000:
                        state.events = state.events[-1000:]

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    async def _force_shutdown_agent(self, agent_id: str) -> None:
        """Force shutdown of an agent."""
        try:
            if agent_id in self.managed_agents:
                agent = self.managed_agents[agent_id]
                # Force termination
                if hasattr(agent, 'client') and agent.client:
                    await agent.client.close()

                state = self.agent_states[agent_id]
                state.current_status = AgentStatus.TERMINATED

        except Exception as e:
            self.logger.error(f"Error force shutting down agent {agent_id}: {e}")

    def _get_agent_memory_usage(self, agent: EnhancedBaseAgent) -> float:
        """Get memory usage for an agent."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_agent_cpu_usage(self, agent: EnhancedBaseAgent) -> float:
        """Get CPU usage for an agent."""
        try:
            import psutil
            process = psutil.Process()
            return process.cpu_percent()
        except ImportError:
            return 0.0

    async def _record_lifecycle_event(self, agent_id: str, event: LifecycleEvent,
                                    details: Optional[Dict[str, Any]] = None) -> None:
        """Record a lifecycle event."""
        if agent_id in self.agent_states:
            state = self.agent_states[agent_id]

            event_record = LifecycleEventRecord(
                agent_id=agent_id,
                event=event,
                timestamp=datetime.now(),
                details=details or {}
            )

            state.events.append(event_record)

            # Keep only recent events
            if len(state.events) > 100:
                state.events = state.events[-100:]

            # Call event handlers
            if event in self.lifecycle_event_handlers:
                for handler in self.lifecycle_event_handlers[event]:
                    try:
                        await handler(event_record)
                    except Exception as e:
                        self.logger.error(f"Error in lifecycle event handler: {e}")

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def _load_persistent_state(self) -> None:
        """Load persistent state from disk."""
        state_file = self.state_dir / "lifecycle_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)

                # Load statistics
                self.total_agents_created = data.get("total_agents_created", 0)
                self.total_agents_shutdown = data.get("total_agents_shutdown", 0)
                self.total_recoveries_attempted = data.get("total_recoveries_attempted", 0)
                self.total_recoveries_successful = data.get("total_recoveries_successful", 0)

                self.logger.info("Loaded persistent lifecycle state")
            except Exception as e:
                self.logger.error(f"Failed to load persistent state: {e}")

    async def _save_persistent_state(self) -> None:
        """Save persistent state to disk."""
        state_file = self.state_dir / "lifecycle_state.json"
        try:
            data = {
                "total_agents_created": self.total_agents_created,
                "total_agents_shutdown": self.total_agents_shutdown,
                "total_recoveries_attempted": self.total_recoveries_attempted,
                "total_recoveries_successful": self.total_recoveries_successful,
                "saved_at": datetime.now().isoformat()
            }

            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info("Saved persistent lifecycle state")
        except Exception as e:
            self.logger.error(f"Failed to save persistent state: {e}")

    def add_lifecycle_event_handler(self, event: LifecycleEvent, handler: Callable) -> None:
        """Add a handler for lifecycle events."""
        if event not in self.lifecycle_event_handlers:
            self.lifecycle_event_handlers[event] = []
        self.lifecycle_event_handlers[event].append(handler)

    def add_health_check_handler(self, handler: Callable[[HealthCheckResult], None]) -> None:
        """Add a handler for health check results."""
        self.health_check_handlers.append(handler)

    def add_shutdown_handler(self, handler: Callable) -> None:
        """Add a handler for shutdown events."""
        self.shutdown_handlers.append(handler)

    def get_agent_status(self, agent_id: str) -> Optional[AgentLifecycleState]:
        """Get lifecycle status for an agent."""
        return self.agent_states.get(agent_id)

    def get_all_agent_statuses(self) -> Dict[str, AgentLifecycleState]:
        """Get lifecycle statuses for all agents."""
        return self.agent_states.copy()

    def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle statistics."""
        current_agents = len(self.agent_states)
        healthy_agents = sum(1 for state in self.agent_states.values() if state.health_status == HealthStatus.HEALTHY)
        unhealthy_agents = current_agents - healthy_agents

        total_uptime = sum(state.uptime_seconds for state in self.agent_states.values())
        avg_uptime = total_uptime / current_agents if current_agents > 0 else 0

        return {
            "total_agents_created": self.total_agents_created,
            "total_agents_shutdown": self.total_agents_shutdown,
            "currently_managed_agents": current_agents,
            "healthy_agents": healthy_agents,
            "unhealthy_agents": unhealthy_agents,
            "health_percentage": (healthy_agents / current_agents * 100) if current_agents > 0 else 0,
            "total_recoveries_attempted": self.total_recoveries_attempted,
            "total_recoveries_successful": self.total_recoveries_successful,
            "recovery_success_rate": (self.total_recoveries_successful / self.total_recoveries_attempted * 100) if self.total_recoveries_attempted > 0 else 0,
            "average_uptime_seconds": avg_uptime,
            "total_restarts": sum(state.total_restarts for state in self.agent_states.values()),
            "agents_with_errors": sum(1 for state in self.agent_states.values() if state.total_errors > 0)
        }


# Global lifecycle manager instance
_lifecycle_manager: Optional[AgentLifecycleManager] = None


def get_lifecycle_manager(factory: Optional[EnhancedAgentFactory] = None,
                        state_dir: Optional[Path] = None) -> AgentLifecycleManager:
    """Get or create the global lifecycle manager."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        if factory is None:
            raise ValueError("Factory must be provided for first-time initialization")
        _lifecycle_manager = AgentLifecycleManager(factory, state_dir)
    return _lifecycle_manager