"""Agent Performance Monitoring and Optimization System

This module provides comprehensive performance monitoring, optimization,
and analytics for enhanced agents with real-time metrics and adaptive tuning.

Key Features:
- Real-time Performance Monitoring
- Adaptive Performance Optimization
- Resource Usage Tracking
- Performance Analytics and Reporting
- Bottleneck Detection and Resolution
- Auto-tuning and Optimization
"""

import asyncio
import json
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Union

from .base_agent import EnhancedBaseAgent, AgentPerformanceMetrics


class PerformanceLevel(Enum):
    """Performance level classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    MANUAL = "manual"


class ResourceType(Enum):
    """Types of resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    TOKENS = "tokens"
    REQUESTS = "requests"


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    open_files: int
    threads: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceEvent:
    """Performance-related event."""
    event_type: str
    timestamp: datetime
    agent_id: str
    severity: str  # info, warning, error, critical
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    resource_type: ResourceType
    warning_threshold: float
    critical_threshold: float
    measurement_window_seconds: int = 300
    consecutive_violations: int = 3
    auto_adjustment_enabled: bool = True


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    agent_id: str
    issue_type: str
    severity: str
    description: str
    recommended_actions: List[str]
    expected_improvement: str
    confidence_score: float
    estimated_impact: str


class PerformanceAnalyzer:
    """Advanced performance analysis engine."""

    def __init__(self):
        self.logger = logging.getLogger("performance_analyzer")
        self.analysis_window_minutes = 30
        self.baseline_calculators: Dict[str, Callable] = {}
        self.anomaly_detectors: List[Callable] = []

    def analyze_performance(self, agent_id: str,
                          metrics_history: List[AgentPerformanceMetrics],
                          resource_history: List[ResourceSnapshot]) -> Dict[str, Any]:
        """Comprehensive performance analysis."""
        try:
            analysis = {
                "agent_id": agent_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "performance_level": self._calculate_performance_level(metrics_history),
                "trends": self._analyze_trends(metrics_history, resource_history),
                "bottlenecks": self._identify_bottlenecks(metrics_history, resource_history),
                "anomalies": self._detect_anomalies(metrics_history, resource_history),
                "resource_efficiency": self._analyze_resource_efficiency(resource_history),
                "recommendations": []
            }

            # Generate optimization recommendations
            analysis["recommendations"] = self._generate_recommendations(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Performance analysis failed for {agent_id}: {e}")
            return {"error": str(e), "agent_id": agent_id}

    def _calculate_performance_level(self, metrics_history: List[AgentPerformanceMetrics]) -> PerformanceLevel:
        """Calculate overall performance level."""
        if not metrics_history:
            return PerformanceLevel.ACCEPTABLE

        recent_metrics = metrics_history[-10:]  # Last 10 measurements

        # Calculate weighted score
        success_rate = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.average_response_time for m in recent_metrics) / len(recent_metrics)
        avg_quality = sum(m.average_quality_score for m in recent_metrics) / len(recent_metrics)

        # Convert to performance score (0-100)
        performance_score = (
            success_rate * 40 +  # 40% weight
            max(0, 100 - avg_response_time) * 30 +  # 30% weight (inverse of response time)
            avg_quality * 30  # 30% weight
        )

        if performance_score >= 90:
            return PerformanceLevel.EXCELLENT
        elif performance_score >= 75:
            return PerformanceLevel.GOOD
        elif performance_score >= 60:
            return PerformanceLevel.ACCEPTABLE
        elif performance_score >= 40:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL

    def _analyze_trends(self, metrics_history: List[AgentPerformanceMetrics],
                       resource_history: List[ResourceSnapshot]) -> Dict[str, str]:
        """Analyze performance trends."""
        trends = {}

        if len(metrics_history) >= 5:
            # Success rate trend
            recent_success_rates = [m.success_rate for m in metrics_history[-5:]]
            success_trend = self._calculate_trend(recent_success_rates)
            trends["success_rate"] = success_trend

            # Response time trend
            recent_response_times = [m.average_response_time for m in metrics_history[-5:]]
            response_trend = self._calculate_trend(recent_response_times)
            trends["response_time"] = response_trend

        if len(resource_history) >= 5:
            # Memory trend
            recent_memory = [r.memory_mb for r in resource_history[-5:]]
            memory_trend = self._calculate_trend(recent_memory)
            trends["memory_usage"] = memory_trend

            # CPU trend
            recent_cpu = [r.cpu_percent for r in resource_history[-5:]]
            cpu_trend = self._calculate_trend(recent_cpu)
            trends["cpu_usage"] = cpu_trend

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction and strength."""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear regression
        n = len(values)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _identify_bottlenecks(self, metrics_history: List[AgentPerformanceMetrics],
                             resource_history: List[ResourceSnapshot]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        if not metrics_history:
            return bottlenecks

        recent_metrics = metrics_history[-10:]

        # Check response time bottleneck
        avg_response_time = sum(m.average_response_time for m in recent_metrics) / len(recent_metrics)
        if avg_response_time > 5.0:  # 5 seconds
            bottlenecks.append({
                "type": "response_time",
                "severity": "high" if avg_response_time > 10.0 else "medium",
                "value": avg_response_time,
                "description": f"High average response time: {avg_response_time:.2f}s"
            })

        # Check success rate bottleneck
        avg_success_rate = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
        if avg_success_rate < 0.9:  # 90% success rate
            bottlenecks.append({
                "type": "success_rate",
                "severity": "high" if avg_success_rate < 0.8 else "medium",
                "value": avg_success_rate,
                "description": f"Low success rate: {avg_success_rate:.1%}"
            })

        # Check resource bottlenecks
        if resource_history:
            recent_resources = resource_history[-10:]

            avg_memory = sum(r.memory_mb for r in recent_resources) / len(recent_resources)
            if avg_memory > 500:  # 500MB
                bottlenecks.append({
                    "type": "memory",
                    "severity": "high" if avg_memory > 1000 else "medium",
                    "value": avg_memory,
                    "description": f"High memory usage: {avg_memory:.1f}MB"
                })

            avg_cpu = sum(r.cpu_percent for r in recent_resources) / len(recent_resources)
            if avg_cpu > 80:  # 80% CPU
                bottlenecks.append({
                    "type": "cpu",
                    "severity": "high" if avg_cpu > 95 else "medium",
                    "value": avg_cpu,
                    "description": f"High CPU usage: {avg_cpu:.1f}%"
                })

        return bottlenecks

    def _detect_anomalies(self, metrics_history: List[AgentPerformanceMetrics],
                         resource_history: List[ResourceSnapshot]) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        anomalies = []

        if len(metrics_history) < 10:
            return anomalies

        # Simple anomaly detection using standard deviation
        recent_metrics = metrics_history[-20:]

        # Check for response time anomalies
        response_times = [m.average_response_time for m in recent_metrics]
        avg_response = sum(response_times) / len(response_times)
        std_response = (sum((x - avg_response) ** 2 for x in response_times) / len(response_times)) ** 0.5

        for i, metrics in enumerate(recent_metrics[-5:]):  # Check last 5
            if abs(metrics.average_response_time - avg_response) > 2 * std_response:
                anomalies.append({
                    "type": "response_time_anomaly",
                    "timestamp": metrics.last_activity.isoformat() if metrics.last_activity else None,
                    "value": metrics.average_response_time,
                    "expected_range": f"{avg_response - 2*std_response:.2f} - {avg_response + 2*std_response:.2f}",
                    "description": "Response time outside normal range"
                })

        return anomalies

    def _analyze_resource_efficiency(self, resource_history: List[ResourceSnapshot]) -> Dict[str, float]:
        """Analyze resource usage efficiency."""
        if not resource_history:
            return {}

        recent_resources = resource_history[-20:]

        efficiency_scores = {}

        # Memory efficiency (lower is better)
        avg_memory = sum(r.memory_mb for r in recent_resources) / len(recent_resources)
        memory_efficiency = max(0, 1 - avg_memory / 1000)  # Normalize to 1GB
        efficiency_scores["memory_efficiency"] = memory_efficiency

        # CPU efficiency (lower is better)
        avg_cpu = sum(r.cpu_percent for r in recent_resources) / len(recent_resources)
        cpu_efficiency = max(0, 1 - avg_cpu / 100)
        efficiency_scores["cpu_efficiency"] = cpu_efficiency

        # Overall efficiency
        efficiency_scores["overall_efficiency"] = (memory_efficiency + cpu_efficiency) / 2

        return efficiency_scores

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        agent_id = analysis["agent_id"]
        bottlenecks = analysis.get("bottlenecks", [])
        performance_level = analysis.get("performance_level")

        # Generate recommendations for each bottleneck
        for bottleneck in bottlenecks:
            recommendation = self._create_bottleneck_recommendation(agent_id, bottleneck)
            recommendations.append(recommendation)

        # Generate recommendations based on performance level
        if performance_level == PerformanceLevel.POOR:
            recommendations.append(OptimizationRecommendation(
                agent_id=agent_id,
                issue_type="overall_performance",
                severity="medium",
                description="Overall performance is poor",
                recommended_actions=[
                    "Increase timeout settings",
                    "Enable caching",
                    "Reduce concurrent operations"
                ],
                expected_improvement="20-30% performance improvement",
                confidence_score=0.7,
                estimated_impact="medium"
            ))
        elif performance_level == PerformanceLevel.CRITICAL:
            recommendations.append(OptimizationRecommendation(
                agent_id=agent_id,
                issue_type="critical_performance",
                severity="critical",
                description="Performance is critical - immediate action required",
                recommended_actions=[
                    "Reduce agent workload",
                    "Increase resource allocation",
                    "Consider agent restart"
                ],
                expected_improvement="Significant performance recovery",
                confidence_score=0.9,
                estimated_impact="high"
            ))

        return recommendations

    def _create_bottleneck_recommendation(self, agent_id: str,
                                        bottleneck: Dict[str, Any]) -> OptimizationRecommendation:
        """Create recommendation for a specific bottleneck."""
        bottleneck_type = bottleneck["type"]
        severity = bottleneck["severity"]

        if bottleneck_type == "response_time":
            return OptimizationRecommendation(
                agent_id=agent_id,
                issue_type="high_response_time",
                severity=severity,
                description=f"High response time detected: {bottleneck['value']:.2f}s",
                recommended_actions=[
                    "Optimize agent logic",
                    "Increase timeout settings",
                    "Enable response caching",
                    "Reduce concurrent operations"
                ],
                expected_improvement="30-50% faster response times",
                confidence_score=0.8,
                estimated_impact="high"
            )
        elif bottleneck_type == "memory":
            return OptimizationRecommendation(
                agent_id=agent_id,
                issue_type="high_memory_usage",
                severity=severity,
                description=f"High memory usage detected: {bottleneck['value']:.1f}MB",
                recommended_actions=[
                    "Implement memory cleanup",
                    "Reduce data retention",
                    "Optimize data structures",
                    "Enable memory compression"
                ],
                expected_improvement="40-60% memory reduction",
                confidence_score=0.7,
                estimated_impact="medium"
            )
        elif bottleneck_type == "cpu":
            return OptimizationRecommendation(
                agent_id=agent_id,
                issue_type="high_cpu_usage",
                severity=severity,
                description=f"High CPU usage detected: {bottleneck['value']:.1f}%",
                recommended_actions=[
                    "Optimize algorithmic complexity",
                    "Implement caching",
                    "Reduce polling frequency",
                    "Use async operations"
                ],
                expected_improvement="25-40% CPU reduction",
                confidence_score=0.75,
                estimated_impact="medium"
            )
        else:
            return OptimizationRecommendation(
                agent_id=agent_id,
                issue_type=bottleneck_type,
                severity=severity,
                description=bottleneck["description"],
                recommended_actions=["Investigate specific issue"],
                expected_improvement="Issue-dependent",
                confidence_score=0.5,
                estimated_impact="unknown"
            )


class PerformanceOptimizer:
    """Automated performance optimization engine."""

    def __init__(self):
        self.logger = logging.getLogger("performance_optimizer")
        self.optimization_strategies: Dict[str, Callable] = {}
        self.auto_optimization_enabled = True
        self.optimization_history: List[Dict[str, Any]] = []

    def register_optimization_strategy(self, issue_type: str, strategy: Callable) -> None:
        """Register an optimization strategy."""
        self.optimization_strategies[issue_type] = strategy

    async def apply_optimization(self, agent: EnhancedBaseAgent,
                               recommendation: OptimizationRecommendation) -> bool:
        """Apply optimization recommendation to an agent."""
        try:
            self.logger.info(f"Applying optimization to {agent.agent_id}: {recommendation.issue_type}")

            # Check if we have a specific strategy
            if recommendation.issue_type in self.optimization_strategies:
                strategy = self.optimization_strategies[recommendation.issue_type]
                result = await strategy(agent, recommendation)
            else:
                # Apply generic optimization
                result = await self._apply_generic_optimization(agent, recommendation)

            # Record optimization
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent.agent_id,
                "recommendation": recommendation.__dict__,
                "result": result,
                "success": result.get("success", False)
            }
            self.optimization_history.append(optimization_record)

            # Keep only recent history
            if len(self.optimization_history) > 1000:
                self.optimization_history = self.optimization_history[-1000:]

            return result.get("success", False)

        except Exception as e:
            self.logger.error(f"Failed to apply optimization to {agent.agent_id}: {e}")
            return False

    async def _apply_generic_optimization(self, agent: EnhancedBaseAgent,
                                        recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """Apply generic optimization based on recommendation."""
        success = False
        changes_made = []

        try:
            if recommendation.issue_type == "high_response_time":
                # Increase timeout
                if hasattr(agent, 'config'):
                    old_timeout = agent.config.timeout_seconds
                    agent.config.timeout_seconds = int(old_timeout * 1.5)
                    changes_made.append(f"Increased timeout from {old_timeout}s to {agent.config.timeout_seconds}s")
                    success = True

            elif recommendation.issue_type == "high_memory_usage":
                # Enable cleanup if available
                if hasattr(agent, 'enable_memory_cleanup'):
                    agent.enable_memory_cleanup()
                    changes_made.append("Enabled memory cleanup")
                    success = True

            elif recommendation.issue_type == "high_cpu_usage":
                # Reduce concurrent operations if available
                if hasattr(agent, 'config') and hasattr(agent.config, 'max_concurrent_operations'):
                    old_concurrent = agent.config.max_concurrent_operations
                    agent.config.max_concurrent_operations = max(1, old_concurrent // 2)
                    changes_made.append(f"Reduced concurrent operations from {old_concurrent} to {agent.config.max_concurrent_operations}")
                    success = True

            return {
                "success": success,
                "changes_made": changes_made,
                "optimization_type": "generic"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "optimization_type": "generic"
            }


class AgentPerformanceMonitor:
    """Comprehensive performance monitoring system for agents."""

    def __init__(self, persistence_dir: Optional[Path] = None):
        self.logger = logging.getLogger("agent_performance_monitor")
        self.persistence_dir = persistence_dir or Path("data/performance_monitoring")
        self.persistence_dir.mkdir(parents=True, exist_ok=True)

        # Monitoring components
        self.analyzer = PerformanceAnalyzer()
        self.optimizer = PerformanceOptimizer()
        self.monitored_agents: Dict[str, EnhancedBaseAgent] = {}
        self.agent_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.agent_resources: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Configuration
        self.monitoring_interval_seconds = 30
        self.resource_monitoring_interval_seconds = 10
        self.analysis_interval_minutes = 5
        self.optimization_enabled = True

        # Performance thresholds
        self.thresholds: Dict[ResourceType, PerformanceThreshold] = {
            ResourceType.MEMORY: PerformanceThreshold(
                resource_type=ResourceType.MEMORY,
                warning_threshold=70.0,  # 70%
                critical_threshold=90.0,  # 90%
                measurement_window_seconds=300
            ),
            ResourceType.CPU: PerformanceThreshold(
                resource_type=ResourceType.CPU,
                warning_threshold=70.0,  # 70%
                critical_threshold=90.0,  # 90%
                measurement_window_seconds=300
            )
        }

        # Monitoring state
        self.monitoring_task: Optional[asyncio.Task] = None
        self.resource_monitor_task: Optional[asyncio.Task] = None
        self.analysis_task: Optional[asyncio.Task] = None
        self.running = False

        # Performance events
        self.performance_events: deque = deque(maxlen=10000)
        self.event_handlers: List[Callable] = []

        self.logger.info("Agent performance monitor initialized")

    async def start(self) -> None:
        """Start performance monitoring."""
        if self.running:
            return

        self.running = True
        self.logger.info("Starting agent performance monitor")

        # Start monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.resource_monitor_task = asyncio.create_task(self._resource_monitoring_loop())
        self.analysis_task = asyncio.create_task(self._analysis_loop())

        # Load persisted data
        await self._load_persisted_data()

        self.logger.info("Agent performance monitor started")

    async def stop(self) -> None:
        """Stop performance monitoring."""
        if not self.running:
            return

        self.running = False
        self.logger.info("Stopping agent performance monitor")

        # Cancel monitoring tasks
        tasks = [self.monitoring_task, self.resource_monitor_task, self.analysis_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Save data
        await self._save_persisted_data()

        self.logger.info("Agent performance monitor stopped")

    def register_agent(self, agent: EnhancedBaseAgent) -> None:
        """Register an agent for performance monitoring."""
        self.monitored_agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent for monitoring: {agent.agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from monitoring."""
        if agent_id in self.monitored_agents:
            del self.monitored_agents[agent_id]
            self.logger.info(f"Unregistered agent from monitoring: {agent_id}")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                start_time = time.time()

                # Collect metrics from all agents
                for agent_id, agent in self.monitored_agents.items():
                    try:
                        metrics = self._collect_agent_metrics(agent)
                        self.agent_metrics[agent_id].append(metrics)
                    except Exception as e:
                        self.logger.error(f"Failed to collect metrics from {agent_id}: {e}")

                # Calculate and update monitoring frequency
                collection_time = time.time() - start_time
                sleep_time = max(1, self.monitoring_interval_seconds - collection_time)

                await asyncio.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval_seconds)

    async def _resource_monitoring_loop(self) -> None:
        """Resource monitoring loop."""
        while self.running:
            try:
                # Collect system-wide resources
                resource_snapshot = self._collect_resource_snapshot()

                # Assign to all agents (system-wide monitoring)
                for agent_id in self.monitored_agents.keys():
                    self.agent_resources[agent_id].append(resource_snapshot)

                # Check thresholds
                await self._check_resource_thresholds(resource_snapshot)

                await asyncio.sleep(self.resource_monitoring_interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(self.resource_monitoring_interval_seconds)

    async def _analysis_loop(self) -> None:
        """Performance analysis loop."""
        while self.running:
            try:
                # Analyze each agent
                for agent_id in self.monitored_agents.keys():
                    await self._analyze_agent_performance(agent_id)

                await asyncio.sleep(self.analysis_interval_minutes * 60)

            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(self.analysis_interval_minutes * 60)

    def _collect_agent_metrics(self, agent: EnhancedBaseAgent) -> AgentPerformanceMetrics:
        """Collect performance metrics from an agent."""
        try:
            # Get metrics from agent if available
            if hasattr(agent, 'get_performance_summary'):
                summary = agent.get_performance_summary()
                return AgentPerformanceMetrics(
                    agent_id=agent.agent_id,
                    session_id="",  # Would be filled with actual session ID
                    start_time=datetime.now(),
                    total_requests=summary.get("total_requests", 0),
                    successful_requests=summary.get("successful_requests", 0),
                    failed_requests=summary.get("failed_requests", 0),
                    average_response_time=summary.get("average_response_time", 0.0),
                    memory_usage_mb=summary.get("average_memory_usage_mb", 0.0),
                    cpu_usage_percent=0.0,  # Would be collected separately
                    error_count=summary.get("error_count", 0),
                    last_activity=datetime.now()
                )
            else:
                # Create basic metrics
                return AgentPerformanceMetrics(
                    agent_id=agent.agent_id,
                    session_id="",
                    start_time=datetime.now(),
                    total_requests=0,
                    successful_requests=0,
                    failed_requests=0,
                    average_response_time=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    error_count=0,
                    last_activity=datetime.now()
                )

        except Exception as e:
            self.logger.error(f"Failed to collect metrics from {agent.agent_id}: {e}")
            # Return empty metrics
            return AgentPerformanceMetrics(
                agent_id=agent.agent_id,
                session_id="",
                start_time=datetime.now(),
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                error_count=1,
                last_activity=datetime.now()
            )

    def _collect_resource_snapshot(self) -> ResourceSnapshot:
        """Collect system resource snapshot."""
        try:
            import psutil
            process = psutil.Process()

            # Get network I/O
            network_io = psutil.net_io_counters()
            disk_io = psutil.disk_io_counters()

            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=process.cpu_percent(),
                memory_mb=process.memory_info().rss / 1024 / 1024,
                memory_percent=process.memory_percent(),
                disk_io_read_mb=disk_io.read_bytes / 1024 / 1024 if disk_io else 0,
                disk_io_write_mb=disk_io.write_bytes / 1024 / 1024 if disk_io else 0,
                network_io_sent_mb=network_io.bytes_sent / 1024 / 1024 if network_io else 0,
                network_io_recv_mb=network_io.bytes_recv / 1024 / 1024 if network_io else 0,
                open_files=len(process.open_files()),
                threads=process.num_threads()
            )

        except ImportError:
            # psutil not available, return basic snapshot
            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_io_sent_mb=0.0,
                network_io_recv_mb=0.0,
                open_files=0,
                threads=0
            )

    async def _check_resource_thresholds(self, snapshot: ResourceSnapshot) -> None:
        """Check resource usage against thresholds."""
        for resource_type, threshold in self.thresholds.items():
            current_value = 0.0

            if resource_type == ResourceType.MEMORY:
                current_value = snapshot.memory_percent
            elif resource_type == ResourceType.CPU:
                current_value = snapshot.cpu_percent

            # Check critical threshold
            if current_value >= threshold.critical_threshold:
                await self._handle_threshold_violation(resource_type, current_value, "critical")
            # Check warning threshold
            elif current_value >= threshold.warning_threshold:
                await self._handle_threshold_violation(resource_type, current_value, "warning")

    async def _handle_threshold_violation(self, resource_type: ResourceType,
                                        current_value: float, severity: str) -> None:
        """Handle resource threshold violation."""
        event = PerformanceEvent(
            event_type="threshold_violation",
            timestamp=datetime.now(),
            agent_id="system",
            severity=severity,
            message=f"{resource_type.value} usage {severity}: {current_value:.1f}%",
            metrics={resource_type.value: current_value}
        )

        self.performance_events.append(event)

        # Call event handlers
        for handler in self.event_handlers:
            try:
                await handler(event)
            except Exception as e:
                self.logger.error(f"Error in performance event handler: {e}")

        self.logger.warning(f"Resource threshold violation: {resource_type.value} = {current_value:.1f}% ({severity})")

    async def _analyze_agent_performance(self, agent_id: str) -> None:
        """Analyze performance for a specific agent."""
        try:
            metrics_history = list(self.agent_metrics[agent_id])
            resource_history = list(self.agent_resources[agent_id])

            if not metrics_history:
                return

            # Perform analysis
            analysis = self.analyzer.analyze_performance(agent_id, metrics_history, resource_history)

            # Apply optimizations if enabled and recommendations exist
            if (self.optimization_enabled and
                analysis.get("recommendations") and
                agent_id in self.monitored_agents):

                agent = self.monitored_agents[agent_id]

                for recommendation in analysis["recommendations"]:
                    if recommendation.severity in ["high", "critical"]:
                        success = await self.optimizer.apply_optimization(agent, recommendation)
                        if success:
                            self.logger.info(f"Applied optimization to {agent_id}: {recommendation.issue_type}")
                        else:
                            self.logger.warning(f"Failed to apply optimization to {agent_id}: {recommendation.issue_type}")

        except Exception as e:
            self.logger.error(f"Failed to analyze performance for {agent_id}: {e}")

    def add_event_handler(self, handler: Callable) -> None:
        """Add performance event handler."""
        self.event_handlers.append(handler)

    def get_agent_performance_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get performance summary for an agent."""
        if agent_id not in self.monitored_agents:
            return {"error": "Agent not monitored"}

        metrics_history = list(self.agent_metrics[agent_id])
        resource_history = list(self.agent_resources[agent_id])

        if not metrics_history:
            return {"error": "No metrics available"}

        # Calculate summary statistics
        recent_metrics = metrics_history[-10:]

        summary = {
            "agent_id": agent_id,
            "monitoring_duration_minutes": len(metrics_history) * self.monitoring_interval_seconds / 60,
            "total_requests": sum(m.total_requests for m in recent_metrics),
            "average_success_rate": sum(m.success_rate for m in recent_metrics) / len(recent_metrics),
            "average_response_time": sum(m.average_response_time for m in recent_metrics) / len(recent_metrics),
            "average_memory_usage": sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
            "total_errors": sum(m.error_count for m in recent_metrics),
            "performance_level": self.analyzer._calculate_performance_level(recent_metrics).value,
            "recent_events": len([e for e in self.performance_events if e.agent_id == agent_id])
        }

        # Add resource summary if available
        if resource_history:
            recent_resources = resource_history[-10:]
            summary.update({
                "average_cpu_usage": sum(r.cpu_percent for r in recent_resources) / len(recent_resources),
                "peak_memory_usage": max(r.memory_mb for r in recent_resources),
                "peak_cpu_usage": max(r.cpu_percent for r in recent_resources)
            })

        return summary

    def get_system_performance_overview(self) -> Dict[str, Any]:
        """Get system-wide performance overview."""
        total_agents = len(self.monitored_agents)
        agent_summaries = {}

        for agent_id in self.monitored_agents.keys():
            agent_summaries[agent_id] = self.get_agent_performance_summary(agent_id)

        # Calculate system-wide metrics
        if agent_summaries:
            avg_success_rate = sum(s.get("average_success_rate", 0) for s in agent_summaries.values()) / total_agents
            avg_response_time = sum(s.get("average_response_time", 0) for s in agent_summaries.values()) / total_agents
            total_errors = sum(s.get("total_errors", 0) for s in agent_summaries.values())

            performance_levels = [s.get("performance_level", "unknown") for s in agent_summaries.values()]
            level_counts = {level: performance_levels.count(level) for level in set(performance_levels)}
        else:
            avg_success_rate = 0
            avg_response_time = 0
            total_errors = 0
            level_counts = {}

        return {
            "monitoring_active": self.running,
            "total_agents_monitored": total_agents,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "system_average_success_rate": avg_success_rate,
            "system_average_response_time": avg_response_time,
            "system_total_errors": total_errors,
            "performance_level_distribution": level_counts,
            "recent_performance_events": len([e for e in list(self.performance_events)[-100:]]),
            "agent_summaries": agent_summaries
        }

    async def _load_persisted_data(self) -> None:
        """Load persisted performance data."""
        try:
            # Load performance events
            events_file = self.persistence_dir / "performance_events.json"
            if events_file.exists():
                with open(events_file, 'r') as f:
                    events_data = json.load(f)
                    for event_data in events_data[-1000:]:  # Load last 1000
                        event = PerformanceEvent(**event_data)
                        self.performance_events.append(event)

            self.logger.info("Loaded persisted performance data")
        except Exception as e:
            self.logger.error(f"Failed to load persisted data: {e}")

    async def _save_persisted_data(self) -> None:
        """Save performance data to disk."""
        try:
            # Save performance events
            events_file = self.persistence_dir / "performance_events.json"
            events_data = []
            for event in list(self.performance_events)[-1000:]:  # Save last 1000
                events_data.append({
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "agent_id": event.agent_id,
                    "severity": event.severity,
                    "message": event.message,
                    "metrics": event.metrics,
                    "context": event.context
                })

            with open(events_file, 'w') as f:
                json.dump(events_data, f, indent=2)

            self.logger.info("Saved performance data")
        except Exception as e:
            self.logger.error(f"Failed to save performance data: {e}")


# Global performance monitor instance
_performance_monitor: Optional[AgentPerformanceMonitor] = None


def get_performance_monitor(persistence_dir: Optional[Path] = None) -> AgentPerformanceMonitor:
    """Get or create the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = AgentPerformanceMonitor(persistence_dir)
    return _performance_monitor