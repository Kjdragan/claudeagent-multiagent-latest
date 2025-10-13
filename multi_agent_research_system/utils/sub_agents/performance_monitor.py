"""
Sub-Agent Performance Monitor

This module provides performance monitoring and optimization capabilities
for sub-agents, including execution tracking, resource usage monitoring,
and performance analytics.
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""

    metric_id: str
    metric_type: str
    agent_id: str
    agent_type: str
    session_id: str
    timestamp: datetime
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "metric_id": self.metric_id,
            "metric_type": self.metric_type,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "unit": self.unit,
            "metadata": self.metadata
        }


@dataclass
class AgentPerformanceProfile:
    """Performance profile for a specific agent."""

    agent_id: str
    agent_type: str
    created_at: datetime
    last_updated: datetime
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    average_quality_score: float = 0.0
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_metrics: deque = field(default_factory=lambda: deque(maxlen=100))

    def update_execution(self, execution_time: float, success: bool, quality_score: Optional[float] = None):
        """Update profile with new execution data."""
        self.total_executions += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.total_executions
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)

        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        if quality_score is not None:
            self.quality_scores.append(quality_score)
            # Keep only last 50 quality scores
            if len(self.quality_scores) > 50:
                self.quality_scores = self.quality_scores[-50:]
            self.average_quality_score = sum(self.quality_scores) / len(self.quality_scores)

        self.last_updated = datetime.now()

    def get_success_rate(self) -> float:
        """Get the success rate as a percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a performance summary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "total_executions": self.total_executions,
            "success_rate": self.get_success_rate(),
            "average_execution_time": self.average_execution_time,
            "min_execution_time": self.min_execution_time,
            "max_execution_time": self.max_execution_time,
            "average_quality_score": self.average_quality_score,
            "current_memory_mb": self.memory_usage_mb,
            "current_cpu_percent": self.cpu_usage_percent,
            "error_summary": dict(self.error_types),
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    memory_used_mb: float
    disk_usage_percent: float
    active_processes: int
    sub_agent_count: int


class SubAgentPerformanceMonitor:
    """
    Monitors and analyzes performance of sub-agents, providing insights
    and optimization recommendations.
    """

    def __init__(self):
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.metrics_history: List[PerformanceMetric] = []
        self.resource_snapshots: deque = deque(maxlen=1000)
        self.performance_alerts: List[Dict[str, Any]] = []
        self.monitoring_config = {
            "enable_resource_monitoring": True,
            "enable_quality_tracking": True,
            "enable_error_tracking": True,
            "resource_interval": 5,  # seconds
            "max_metrics_history": 10000,
            "performance_thresholds": {
                "execution_time_warning": 30.0,  # seconds
                "execution_time_critical": 60.0,  # seconds
                "memory_usage_warning": 512,  # MB
                "memory_usage_critical": 1024,  # MB
                "cpu_usage_warning": 70,  # percent
                "cpu_usage_critical": 90,  # percent
                "success_rate_warning": 80,  # percent
                "success_rate_critical": 60,  # percent
                "quality_score_warning": 60,  # percent
                "quality_score_critical": 40  # percent
            },
            "alerts_enabled": True,
            "persistent_storage": False,
            "storage_path": "performance_data"
        }
        self._monitoring_task: Optional[asyncio.Task] = None
        self._resource_monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._process = psutil.Process()

    async def initialize(self):
        """Initialize the performance monitor."""
        logger.info("Initializing Sub-Agent Performance Monitor")
        self._running = True

        # Start monitoring tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        # Start resource monitoring in separate thread
        if self.monitoring_config["enable_resource_monitoring"]:
            self._resource_monitor_thread = threading.Thread(
                target=self._resource_monitoring_loop,
                daemon=True
            )
            self._resource_monitor_thread.start()

        # Load existing data if persistent storage is enabled
        if self.monitoring_config["persistent_storage"]:
            await self._load_performance_data()

        logger.info("Performance monitor initialized")

    async def shutdown(self):
        """Shutdown the performance monitor."""
        logger.info("Shutting down Performance Monitor")
        self._running = False

        # Cancel monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Save performance data if persistent storage is enabled
        if self.monitoring_config["persistent_storage"]:
            await self._save_performance_data()

        logger.info("Performance monitor shutdown complete")

    async def track_agent_creation(self, agent_instance):
        """Track the creation of a new agent instance."""
        agent_id = getattr(agent_instance, 'instance_id', str(id(agent_instance)))
        agent_type = getattr(agent_instance, 'agent_type', 'unknown')

        if agent_id not in self.agent_profiles:
            profile = AgentPerformanceProfile(
                agent_id=agent_id,
                agent_type=agent_type.value if hasattr(agent_type, 'value') else str(agent_type),
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            self.agent_profiles[agent_id] = profile

            logger.debug(f"Started tracking performance for agent {agent_id} ({agent_type})")

    async def track_execution(
        self,
        agent_instance,
        execution_time: float,
        success: bool,
        quality_score: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        """Track execution performance for an agent."""
        agent_id = getattr(agent_instance, 'instance_id', str(id(agent_instance)))

        # Get or create profile
        if agent_id not in self.agent_profiles:
            await self.track_agent_creation(agent_instance)

        profile = self.agent_profiles[agent_id]

        # Update execution metrics
        profile.update_execution(execution_time, success, quality_score)

        # Track errors
        if not success and error_message:
            error_type = self._classify_error(error_message)
            profile.error_types[error_type] += 1

        # Create performance metric
        metric = PerformanceMetric(
            metric_id=f"{agent_id}_{int(time.time())}",
            metric_type="execution_time",
            agent_id=agent_id,
            agent_type=profile.agent_type,
            session_id=getattr(agent_instance, 'session_context', {}).get('session_id', 'unknown'),
            timestamp=datetime.now(),
            value=execution_time,
            unit="seconds",
            metadata={
                "success": success,
                "quality_score": quality_score,
                "error_message": error_message
            }
        )

        await self._add_metric(metric)

        # Check for performance alerts
        if self.monitoring_config["alerts_enabled"]:
            await self._check_performance_alerts(agent_id, execution_time, success, quality_score)

        logger.debug(f"Tracked execution for agent {agent_id}: {execution_time:.2f}s, success={success}")

    async def track_execution_error(self, agent_instance, error_message: str):
        """Track an execution error for an agent."""
        agent_id = getattr(agent_instance, 'instance_id', str(id(agent_instance)))

        if agent_id not in self.agent_profiles:
            await self.track_agent_creation(agent_instance)

        profile = self.agent_profiles[agent_id]
        error_type = self._classify_error(error_message)
        profile.error_types[error_type] += 1

        # Create error metric
        metric = PerformanceMetric(
            metric_id=f"{agent_id}_error_{int(time.time())}",
            metric_type="error",
            agent_id=agent_id,
            agent_type=profile.agent_type,
            session_id=getattr(agent_instance, 'session_context', {}).get('session_id', 'unknown'),
            timestamp=datetime.now(),
            value=1.0,
            unit="count",
            metadata={
                "error_type": error_type,
                "error_message": error_message
            }
        )

        await self._add_metric(metric)

    async def track_resource_usage(self, agent_id: str, memory_mb: float, cpu_percent: float):
        """Track resource usage for an agent."""
        if agent_id not in self.agent_profiles:
            return

        profile = self.agent_profiles[agent_id]
        profile.memory_usage_mb = memory_mb
        profile.cpu_usage_percent = cpu_percent

        # Create resource metrics
        memory_metric = PerformanceMetric(
            metric_id=f"{agent_id}_memory_{int(time.time())}",
            metric_type="memory_usage",
            agent_id=agent_id,
            agent_type=profile.agent_type,
            session_id="resource_monitoring",
            timestamp=datetime.now(),
            value=memory_mb,
            unit="MB"
        )

        cpu_metric = PerformanceMetric(
            metric_id=f"{agent_id}_cpu_{int(time.time())}",
            metric_type="cpu_usage",
            agent_id=agent_id,
            agent_type=profile.agent_type,
            session_id="resource_monitoring",
            timestamp=datetime.now(),
            value=cpu_percent,
            unit="percent"
        )

        await self._add_metric(memory_metric)
        await self._add_metric(cpu_metric)

    async def get_agent_performance(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get performance data for a specific agent."""
        if agent_id not in self.agent_profiles:
            return None

        return self.agent_profiles[agent_id].get_performance_summary()

    async def get_agent_type_performance(self, agent_type: str) -> Dict[str, Any]:
        """Get aggregated performance data for an agent type."""
        type_profiles = [
            profile for profile in self.agent_profiles.values()
            if profile.agent_type == agent_type
        ]

        if not type_profiles:
            return {"agent_type": agent_type, "agent_count": 0}

        total_executions = sum(p.total_executions for p in type_profiles)
        successful_executions = sum(p.successful_executions for p in type_profiles)
        avg_execution_time = sum(p.average_execution_time for p in type_profiles) / len(type_profiles)
        avg_quality_score = sum(p.average_quality_score for p in type_profiles if p.quality_scores) / len([p for p in type_profiles if p.quality_scores]) if any(p.quality_scores for p in type_profiles) else 0

        return {
            "agent_type": agent_type,
            "agent_count": len(type_profiles),
            "total_executions": total_executions,
            "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
            "average_execution_time": avg_execution_time,
            "average_quality_score": avg_quality_score,
            "average_memory_usage": sum(p.memory_usage_mb for p in type_profiles) / len(type_profiles),
            "average_cpu_usage": sum(p.cpu_usage_percent for p in type_profiles) / len(type_profiles)
        }

    async def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance data."""
        if not self.resource_snapshots:
            return {}

        latest_snapshot = self.resource_snapshots[-1]

        return {
            "timestamp": latest_snapshot.timestamp.isoformat(),
            "system_resources": {
                "cpu_percent": latest_snapshot.cpu_percent,
                "memory_percent": latest_snapshot.memory_percent,
                "memory_available_mb": latest_snapshot.memory_available_mb,
                "disk_usage_percent": latest_snapshot.disk_usage_percent,
                "active_processes": latest_snapshot.active_processes
            },
            "sub_agents": {
                "total_count": latest_snapshot.sub_agent_count,
                "active_count": len(self.agent_profiles)
            },
            "performance_summary": {
                "total_executions": sum(p.total_executions for p in self.agent_profiles.values()),
                "total_errors": sum(p.failed_executions for p in self.agent_profiles.values()),
                "average_success_rate": sum(p.get_success_rate() for p in self.agent_profiles.values()) / len(self.agent_profiles) if self.agent_profiles else 0
            },
            "recent_alerts": len([a for a in self.performance_alerts if (datetime.now() - a['timestamp']).total_seconds() < 3600])
        }

    async def get_performance_trends(self, agent_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends for an agent over a time period."""
        if agent_id not in self.agent_profiles:
            return {}

        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter metrics for this agent and time period
        agent_metrics = [
            metric for metric in self.metrics_history
            if metric.agent_id == agent_id and metric.timestamp > cutoff_time
        ]

        if not agent_metrics:
            return {"agent_id": agent_id, "time_period_hours": hours, "data_points": 0}

        # Group by metric type
        metrics_by_type = defaultdict(list)
        for metric in agent_metrics:
            metrics_by_type[metric.metric_type].append(metric)

        trends = {}
        for metric_type, metrics in metrics_by_type.items():
            if metric_type == "execution_time":
                values = [m.value for m in metrics]
                trends[metric_type] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "trend": self._calculate_trend(values)
                }
            elif metric_type == "quality_score":
                values = [m.value for m in metrics if m.metadata.get('quality_score') is not None]
                if values:
                    trends[metric_type] = {
                        "count": len(values),
                        "average": sum(values) / len(values),
                        "trend": self._calculate_trend(values)
                    }

        return {
            "agent_id": agent_id,
            "time_period_hours": hours,
            "data_points": len(agent_metrics),
            "trends": trends
        }

    async def _add_metric(self, metric: PerformanceMetric):
        """Add a performance metric to the history."""
        self.metrics_history.append(metric)

        # Limit history size
        if len(self.metrics_history) > self.monitoring_config["max_metrics_history"]:
            self.metrics_history = self.metrics_history[-self.monitoring_config["max_metrics_history"]:]

        # Add to agent profile's recent metrics
        if metric.agent_id in self.agent_profiles:
            self.agent_profiles[metric.agent_id].recent_metrics.append(metric)

    async def _check_performance_alerts(
        self,
        agent_id: str,
        execution_time: float,
        success: bool,
        quality_score: Optional[float]
    ):
        """Check for performance alerts and create them if needed."""
        thresholds = self.monitoring_config["performance_thresholds"]
        profile = self.agent_profiles[agent_id]

        alerts = []

        # Execution time alerts
        if execution_time > thresholds["execution_time_critical"]:
            alerts.append({
                "type": "execution_time_critical",
                "message": f"Critical execution time: {execution_time:.2f}s",
                "value": execution_time,
                "threshold": thresholds["execution_time_critical"]
            })
        elif execution_time > thresholds["execution_time_warning"]:
            alerts.append({
                "type": "execution_time_warning",
                "message": f"High execution time: {execution_time:.2f}s",
                "value": execution_time,
                "threshold": thresholds["execution_time_warning"]
            })

        # Success rate alerts
        success_rate = profile.get_success_rate()
        if success_rate < thresholds["success_rate_critical"] and profile.total_executions >= 10:
            alerts.append({
                "type": "success_rate_critical",
                "message": f"Critical success rate: {success_rate:.1f}%",
                "value": success_rate,
                "threshold": thresholds["success_rate_critical"]
            })
        elif success_rate < thresholds["success_rate_warning"] and profile.total_executions >= 10:
            alerts.append({
                "type": "success_rate_warning",
                "message": f"Low success rate: {success_rate:.1f}%",
                "value": success_rate,
                "threshold": thresholds["success_rate_warning"]
            })

        # Quality score alerts
        if quality_score is not None:
            if quality_score < thresholds["quality_score_critical"]:
                alerts.append({
                    "type": "quality_score_critical",
                    "message": f"Critical quality score: {quality_score:.1f}%",
                    "value": quality_score,
                    "threshold": thresholds["quality_score_critical"]
                })
            elif quality_score < thresholds["quality_score_warning"]:
                alerts.append({
                    "type": "quality_score_warning",
                    "message": f"Low quality score: {quality_score:.1f}%",
                    "value": quality_score,
                    "threshold": thresholds["quality_score_warning"]
                })

        # Add alerts
        for alert in alerts:
            alert_entry = {
                "timestamp": datetime.now(),
                "agent_id": agent_id,
                "agent_type": profile.agent_type,
                **alert
            }
            self.performance_alerts.append(alert_entry)

            # Keep only last 1000 alerts
            if len(self.performance_alerts) > 1000:
                self.performance_alerts = self.performance_alerts[-1000:]

            logger.warning(f"Performance alert for agent {agent_id}: {alert['message']}")

    def _classify_error(self, error_message: str) -> str:
        """Classify an error message into an error type."""
        error_message = error_message.lower()

        if "timeout" in error_message:
            return "timeout"
        elif "memory" in error_message or "out of memory" in error_message:
            return "memory"
        elif "network" in error_message or "connection" in error_message:
            return "network"
        elif "permission" in error_message or "access" in error_message:
            return "permission"
        elif "api" in error_message or "rate limit" in error_message:
            return "api"
        elif "validation" in error_message or "invalid" in error_message:
            return "validation"
        else:
            return "unknown"

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"

        # Simple linear regression to determine trend
        n = len(values)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        if n * sum_x2 - sum_x ** 2 == 0:
            return "stable"

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Update agent profiles with current resource usage
                for agent_id, profile in self.agent_profiles.items():
                    # This is a placeholder - in practice you'd get actual resource usage
                    # for each specific agent
                    pass

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def _resource_monitoring_loop(self):
        """Resource monitoring loop running in separate thread."""
        while self._running:
            try:
                # Get system resource information
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                processes = len(psutil.pids())

                snapshot = ResourceSnapshot(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available_mb=memory.available / 1024 / 1024,
                    memory_used_mb=memory.used / 1024 / 1024,
                    disk_usage_percent=disk.percent,
                    active_processes=processes,
                    sub_agent_count=len(self.agent_profiles)
                )

                self.resource_snapshots.append(snapshot)

                time.sleep(self.monitoring_config["resource_interval"])

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")

    async def _save_performance_data(self):
        """Save performance data to persistent storage."""
        if not self.monitoring_config["persistent_storage"]:
            return

        storage_path = Path(self.monitoring_config["storage_path"])
        storage_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save agent profiles
            profiles_data = {
                agent_id: {
                    "agent_id": profile.agent_id,
                    "agent_type": profile.agent_type,
                    "created_at": profile.created_at.isoformat(),
                    "last_updated": profile.last_updated.isoformat(),
                    "total_executions": profile.total_executions,
                    "successful_executions": profile.successful_executions,
                    "failed_executions": profile.failed_executions,
                    "total_execution_time": profile.total_execution_time,
                    "average_execution_time": profile.average_execution_time,
                    "min_execution_time": profile.min_execution_time,
                    "max_execution_time": profile.max_execution_time,
                    "memory_usage_mb": profile.memory_usage_mb,
                    "cpu_usage_percent": profile.cpu_usage_percent,
                    "quality_scores": profile.quality_scores,
                    "average_quality_score": profile.average_quality_score,
                    "error_types": dict(profile.error_types)
                }
                for agent_id, profile in self.agent_profiles.items()
            }

            with open(storage_path / "agent_profiles.json", 'w') as f:
                json.dump(profiles_data, f, indent=2)

            logger.info("Performance data saved to persistent storage")

        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")

    async def _load_performance_data(self):
        """Load performance data from persistent storage."""
        if not self.monitoring_config["persistent_storage"]:
            return

        storage_path = Path(self.monitoring_config["storage_path"])

        try:
            profiles_file = storage_path / "agent_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)

                for agent_id, profile_data in profiles_data.items():
                    profile = AgentPerformanceProfile(
                        agent_id=profile_data["agent_id"],
                        agent_type=profile_data["agent_type"],
                        created_at=datetime.fromisoformat(profile_data["created_at"]),
                        last_updated=datetime.fromisoformat(profile_data["last_updated"]),
                        total_executions=profile_data["total_executions"],
                        successful_executions=profile_data["successful_executions"],
                        failed_executions=profile_data["failed_executions"],
                        total_execution_time=profile_data["total_execution_time"],
                        average_execution_time=profile_data["average_execution_time"],
                        min_execution_time=profile_data["min_execution_time"],
                        max_execution_time=profile_data["max_execution_time"],
                        memory_usage_mb=profile_data["memory_usage_mb"],
                        cpu_usage_percent=profile_data["cpu_usage_percent"],
                        quality_scores=profile_data["quality_scores"],
                        average_quality_score=profile_data["average_quality_score"],
                        error_types=defaultdict(int, profile_data["error_types"])
                    )
                    self.agent_profiles[agent_id] = profile

                logger.info(f"Loaded performance data for {len(self.agent_profiles)} agents")

        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get the current status of the performance monitor."""
        return {
            "running": self._running,
            "tracked_agents": len(self.agent_profiles),
            "metrics_collected": len(self.metrics_history),
            "resource_snapshots": len(self.resource_snapshots),
            "active_alerts": len(self.performance_alerts),
            "monitoring_config": self.monitoring_config
        }