"""
Real-Time Monitoring Infrastructure for Comprehensive Hooks System

Phase 3.1.1: Real-Time Monitoring Infrastructure with centralized metrics collection

Provides advanced real-time monitoring capabilities including:
- Centralized metrics collection and aggregation
- Real-time performance monitoring and alerting
- Live dashboard data feeds
- System health monitoring with automatic detection
- Performance bottleneck identification
- Resource usage tracking and optimization
"""

import asyncio
import json
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path

# Import parent modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from agent_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

from .base_hooks import HookContext, HookResult, HookStatus


class MetricType(Enum):
    """Types of metrics for monitoring."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthLevel(Enum):
    """System health levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Individual metric value with metadata."""
    name: str
    value: Union[int, float, str]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_name: str
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    metric_type: MetricType = MetricType.GAUGE
    comparison: str = "greater_than"  # greater_than, less_than, equals
    window_seconds: int = 300  # 5-minute window


@dataclass
class Alert:
    """System alert definition."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealthSnapshot:
    """Complete system health snapshot."""
    timestamp: datetime
    overall_health: HealthLevel
    component_health: Dict[str, HealthLevel]
    active_sessions: int
    total_sessions: int
    error_rate: float
    average_response_time: float
    system_metrics: Dict[str, float]
    alerts: List[Alert]
    performance_metrics: Dict[str, Any]
    resource_usage: Dict[str, float]


class MetricsCollector:
    """
    Centralized metrics collection and aggregation system.

    Collects, aggregates, and processes metrics from all system components
    for real-time monitoring and analysis.
    """

    def __init__(self, retention_hours: int = 24, aggregation_intervals: List[int] = None):
        """
        Initialize metrics collector.

        Args:
            retention_hours: How long to retain metrics data
            aggregation_intervals: List of aggregation intervals in seconds
        """
        self.logger = get_logger("metrics_collector")
        self.retention_hours = retention_hours
        self.aggregation_intervals = aggregation_intervals or [60, 300, 900, 3600]  # 1min, 5min, 15min, 1hour

        # Metric storage
        self.raw_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, Dict[int, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))

        # Performance tracking
        self.metric_counts: Dict[str, int] = defaultdict(int)
        self.metric_sums: Dict[str, float] = defaultdict(float)
        self.last_aggregation: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Background tasks
        self.collection_tasks: List[asyncio.Task] = []
        self.aggregation_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

        self._running = False

    async def start(self):
        """Start metrics collection and aggregation."""
        if self._running:
            return

        self._running = True
        self.logger.info("Starting metrics collector", retention_hours=self.retention_hours)

        # Start background aggregation
        self.aggregation_task = asyncio.create_task(self._aggregate_metrics_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.info("Metrics collector started successfully")

    async def stop(self):
        """Stop metrics collection and cleanup."""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping metrics collector...")

        # Cancel background tasks
        tasks = [task for task in [self.aggregation_task, self.cleanup_task] if task]
        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Cancel collection tasks
        for task in self.collection_tasks:
            task.cancel()

        if self.collection_tasks:
            await asyncio.gather(*self.collection_tasks, return_exceptions=True)

        self.logger.info("Metrics collector stopped")

    def record_metric(self, metric: MetricValue):
        """
        Record a new metric value.

        Args:
            metric: Metric value to record
        """
        if not self._running:
            return

        # Store raw metric
        metric_key = self._get_metric_key(metric)
        self.raw_metrics[metric_key].append(metric)

        # Update counters for aggregation
        self.metric_counts[metric_key] += 1
        if isinstance(metric.value, (int, float)):
            self.metric_sums[metric_key] += metric.value

        self.logger.debug(f"Recorded metric: {metric.name} = {metric.value}")

    def get_metrics(self, metric_name: str, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None, labels: Optional[Dict[str, str]] = None) -> List[MetricValue]:
        """
        Retrieve metrics with optional filtering.

        Args:
            metric_name: Name of the metric to retrieve
            start_time: Optional start time filter
            end_time: Optional end time filter
            labels: Optional label filters

        Returns:
            List of metric values matching criteria
        """
        metric_key = f"{metric_name}_{json.dumps(labels or {}, sort_keys=True)}"

        if metric_key not in self.raw_metrics:
            return []

        metrics = list(self.raw_metrics[metric_key])

        # Apply time filters
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]

        return sorted(metrics, key=lambda m: m.timestamp)

    def get_aggregated_metrics(self, metric_name: str, interval_seconds: int,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get aggregated metrics for a specific interval.

        Args:
            metric_name: Name of the metric
            interval_seconds: Aggregation interval in seconds
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            List of aggregated metric data points
        """
        if interval_seconds not in self.aggregation_intervals:
            raise ValueError(f"Invalid aggregation interval: {interval_seconds}")

        metric_key = metric_name
        if metric_key not in self.aggregated_metrics[interval_seconds]:
            return []

        aggregated = list(self.aggregated_metrics[interval_seconds][metric_key])

        # Apply time filters
        if start_time:
            aggregated = [m for m in aggregated if m['timestamp'] >= start_time]
        if end_time:
            aggregated = [m for m in aggregated if m['timestamp'] <= end_time]

        return sorted(aggregated, key=lambda m: m['timestamp'])

    def _get_metric_key(self, metric: MetricValue) -> str:
        """Generate unique key for metric storage."""
        label_str = json.dumps(metric.labels, sort_keys=True)
        return f"{metric.name}_{label_str}"

    async def _aggregate_metrics_loop(self):
        """Background task to aggregate metrics continuously."""
        self.logger.info("Starting metrics aggregation loop")

        while self._running:
            try:
                await self._perform_aggregation()
                await asyncio.sleep(60)  # Aggregate every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics aggregation error: {e}")
                await asyncio.sleep(10)

    async def _perform_aggregation(self):
        """Perform metric aggregation for all intervals."""
        current_time = datetime.now()

        for interval in self.aggregation_intervals:
            for metric_key, metrics_deque in self.raw_metrics.items():
                if not metrics_deque:
                    continue

                # Get metrics in the aggregation window
                window_start = current_time - timedelta(seconds=interval)
                window_metrics = [m for m in metrics_deque if m.timestamp >= window_start]

                if not window_metrics:
                    continue

                # Calculate aggregates
                aggregated = self._calculate_aggregates(window_metrics, current_time)

                # Store aggregated data
                if metric_key not in self.aggregated_metrics[interval]:
                    self.aggregated_metrics[interval][metric_key] = deque(maxlen=1000)

                self.aggregated_metrics[interval][metric_key].append(aggregated)

    def _calculate_aggregates(self, metrics: List[MetricValue], timestamp: datetime) -> Dict[str, Any]:
        """Calculate aggregated statistics for metrics."""
        numeric_values = [m.value for m in metrics if isinstance(m.value, (int, float))]

        if not numeric_values:
            return {
                'timestamp': timestamp,
                'count': len(metrics),
                'values': list(m.value for m in metrics)
            }

        aggregates = {
            'timestamp': timestamp,
            'count': len(metrics),
            'sum': sum(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'avg': sum(numeric_values) / len(numeric_values),
            'values': numeric_values
        }

        # Calculate percentiles
        sorted_values = sorted(numeric_values)
        n = len(sorted_values)
        if n >= 4:
            aggregates['p50'] = sorted_values[n // 2]
            aggregates['p75'] = sorted_values[int(n * 0.75)]
            aggregates['p90'] = sorted_values[int(n * 0.9)]
            aggregates['p95'] = sorted_values[int(n * 0.95)]
            aggregates['p99'] = sorted_values[int(n * 0.99)]

        return aggregates

    async def _cleanup_loop(self):
        """Background task to cleanup old metrics data."""
        self.logger.info("Starting metrics cleanup loop")

        while self._running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

                # Cleanup raw metrics
                for metric_key, metrics_deque in self.raw_metrics.items():
                    while metrics_deque and metrics_deque[0].timestamp < cutoff_time:
                        metrics_deque.popleft()

                # Cleanup aggregated metrics
                for interval, metrics_dict in self.aggregated_metrics.items():
                    for metric_key, aggregated_deque in metrics_dict.items():
                        while aggregated_deque and aggregated_deque[0]['timestamp'] < cutoff_time:
                            aggregated_deque.popleft()

                await asyncio.sleep(3600)  # Cleanup every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(60)


class RealTimeMonitor:
    """
    Real-time monitoring system with alerting and health assessment.

    Provides comprehensive real-time monitoring including:
    - Performance monitoring and alerting
    - System health assessment
    - Bottleneck detection
    - Resource usage tracking
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize real-time monitor.

        Args:
            metrics_collector: Metrics collector instance
        """
        self.logger = get_logger("real_time_monitor")
        self.metrics_collector = metrics_collector

        # Alerting system
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Health monitoring
        self.component_health: Dict[str, HealthLevel] = {}
        self.last_health_assessment: Optional[datetime] = None

        # Performance tracking
        self.performance_baseline: Dict[str, float] = {}
        self.performance_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self._running = False

        # Initialize default thresholds
        self._initialize_default_thresholds()

    async def start(self):
        """Start real-time monitoring."""
        if self._running:
            return

        self._running = True
        self.logger.info("Starting real-time monitoring")

        # Start background monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())

        self.logger.info("Real-time monitoring started")

    async def stop(self):
        """Stop real-time monitoring."""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping real-time monitoring...")

        # Cancel background tasks
        tasks = [task for task in [self.monitoring_task, self.health_check_task] if task]
        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info("Real-time monitoring stopped")

    def add_threshold(self, threshold: PerformanceThreshold):
        """
        Add performance threshold for monitoring.

        Args:
            threshold: Performance threshold to add
        """
        self.thresholds[threshold.metric_name] = threshold
        self.logger.info(f"Added threshold for {threshold.metric_name}",
                        warning=threshold.warning_threshold,
                        error=threshold.error_threshold,
                        critical=threshold.critical_threshold)

    def remove_threshold(self, metric_name: str):
        """
        Remove performance threshold.

        Args:
            metric_name: Name of metric to remove threshold for
        """
        if metric_name in self.thresholds:
            del self.thresholds[metric_name]
            self.logger.info(f"Removed threshold for {metric_name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """
        Add callback for alert notifications.

        Args:
            callback: Callback function to call on alerts
        """
        self.alert_callbacks.append(callback)
        self.logger.info(f"Added alert callback: {callback.__name__}")

    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())

    def get_system_health(self) -> HealthLevel:
        """Get overall system health level."""
        if not self.component_health:
            return HealthLevel.GOOD

        # Map health levels to numeric values
        health_values = {
            HealthLevel.EXCELLENT: 5,
            HealthLevel.GOOD: 4,
            HealthLevel.DEGRADED: 3,
            HealthLevel.POOR: 2,
            HealthLevel.CRITICAL: 1
        }

        # Calculate average health
        values = [health_values.get(level, 3) for level in self.component_health.values()]
        avg_health = sum(values) / len(values)

        # Map back to health levels
        if avg_health >= 4.5:
            return HealthLevel.EXCELLENT
        elif avg_health >= 3.5:
            return HealthLevel.GOOD
        elif avg_health >= 2.5:
            return HealthLevel.DEGRADED
        elif avg_health >= 1.5:
            return HealthLevel.POOR
        else:
            return HealthLevel.CRITICAL

    def create_health_snapshot(self) -> SystemHealthSnapshot:
        """Create complete system health snapshot."""
        current_time = datetime.now()

        # Get recent metrics for performance analysis
        recent_metrics = self._get_recent_performance_metrics()

        # Calculate system metrics
        system_metrics = {
            'error_rate': self._calculate_error_rate(),
            'average_response_time': self._calculate_average_response_time(),
            'throughput': self._calculate_throughput(),
            'active_sessions': self._get_active_sessions(),
            'total_sessions': self._get_total_sessions()
        }

        # Get resource usage
        resource_usage = self._get_resource_usage()

        return SystemHealthSnapshot(
            timestamp=current_time,
            overall_health=self.get_system_health(),
            component_health=self.component_health.copy(),
            active_sessions=system_metrics['active_sessions'],
            total_sessions=system_metrics['total_sessions'],
            error_rate=system_metrics['error_rate'],
            average_response_time=system_metrics['average_response_time'],
            system_metrics=system_metrics,
            alerts=list(self.active_alerts.values()),
            performance_metrics=recent_metrics,
            resource_usage=resource_usage
        )

    async def _monitoring_loop(self):
        """Background monitoring loop for threshold checking."""
        self.logger.info("Starting threshold monitoring loop")

        while self._running:
            try:
                await self._check_thresholds()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)

    async def _health_check_loop(self):
        """Background health assessment loop."""
        self.logger.info("Starting health assessment loop")

        while self._running:
            try:
                await self._assess_system_health()
                await asyncio.sleep(60)  # Assess every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(30)

    async def _check_thresholds(self):
        """Check all configured thresholds and trigger alerts if needed."""
        current_time = datetime.now()

        for metric_name, threshold in self.thresholds.items():
            try:
                # Get recent metric values
                recent_metrics = self.metrics_collector.get_metrics(
                    metric_name,
                    start_time=current_time - timedelta(seconds=threshold.window_seconds)
                )

                if not recent_metrics:
                    continue

                # Get latest numeric value
                latest_value = None
                for metric in reversed(recent_metrics):
                    if isinstance(metric.value, (int, float)):
                        latest_value = metric.value
                        break

                if latest_value is None:
                    continue

                # Check thresholds
                alert_severity = None
                threshold_value = None

                if self._threshold_exceeded(latest_value, threshold.critical_threshold, threshold.comparison):
                    alert_severity = AlertSeverity.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif self._threshold_exceeded(latest_value, threshold.error_threshold, threshold.comparison):
                    alert_severity = AlertSeverity.ERROR
                    threshold_value = threshold.error_threshold
                elif self._threshold_exceeded(latest_value, threshold.warning_threshold, threshold.comparison):
                    alert_severity = AlertSeverity.WARNING
                    threshold_value = threshold.warning_threshold

                # Create or update alert
                alert_id = f"{metric_name}_{threshold.metric_type.value}"

                if alert_severity:
                    if alert_id not in self.active_alerts:
                        # New alert
                        alert = Alert(
                            id=alert_id,
                            severity=alert_severity,
                            title=f"{alert_severity.value.upper()}: {metric_name}",
                            description=f"Metric {metric_name} has exceeded {alert_severity.value} threshold",
                            metric_name=metric_name,
                            current_value=latest_value,
                            threshold=threshold_value,
                            timestamp=current_time
                        )

                        self.active_alerts[alert_id] = alert
                        self.alert_history.append(alert)

                        # Trigger callbacks
                        for callback in self.alert_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                self.logger.error(f"Alert callback error: {e}")

                        self.logger.warning(f"Alert triggered: {alert.title}",
                                          value=latest_value,
                                          threshold=threshold_value)
                else:
                    # Clear existing alert if resolved
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        alert.resolved = True
                        alert.resolved_at = current_time

                        del self.active_alerts[alert_id]

                        self.logger.info(f"Alert resolved: {alert.title}")

            except Exception as e:
                self.logger.error(f"Threshold check error for {metric_name}: {e}")

    async def _assess_system_health(self):
        """Assess overall system health based on metrics and alerts."""
        current_time = datetime.now()

        # Assess component health based on metrics
        component_health = {}

        # Hook execution health
        hook_metrics = self.metrics_collector.get_metrics('hook_execution_time',
                                                         start_time=current_time - timedelta(minutes=5))
        if hook_metrics:
            avg_execution_time = sum(m.value for m in hook_metrics if isinstance(m.value, (int, float))) / len(hook_metrics)
            if avg_execution_time > 5.0:
                component_health['hooks'] = HealthLevel.POOR
            elif avg_execution_time > 2.0:
                component_health['hooks'] = HealthLevel.DEGRADED
            elif avg_execution_time > 1.0:
                component_health['hooks'] = HealthLevel.GOOD
            else:
                component_health['hooks'] = HealthLevel.EXCELLENT

        # Error rate health
        error_metrics = self.metrics_collector.get_metrics('error_rate',
                                                         start_time=current_time - timedelta(minutes=5))
        if error_metrics:
            latest_error_rate = error_metrics[-1].value if error_metrics else 0
            if latest_error_rate > 0.1:  # 10%
                component_health['errors'] = HealthLevel.CRITICAL
            elif latest_error_rate > 0.05:  # 5%
                component_health['errors'] = HealthLevel.POOR
            elif latest_error_rate > 0.01:  # 1%
                component_health['errors'] = HealthLevel.DEGRADED
            else:
                component_health['errors'] = HealthLevel.GOOD

        # Update component health
        self.component_health = component_health
        self.last_health_assessment = current_time

        overall_health = self.get_system_health()
        self.logger.debug(f"System health assessment: {overall_health.value}",
                         components=component_health)

    def _threshold_exceeded(self, value: float, threshold: float, comparison: str) -> bool:
        """Check if threshold is exceeded based on comparison type."""
        if comparison == "greater_than":
            return value > threshold
        elif comparison == "less_than":
            return value < threshold
        elif comparison == "equals":
            return abs(value - threshold) < 0.001
        else:
            return False

    def _initialize_default_thresholds(self):
        """Initialize default performance thresholds."""
        default_thresholds = [
            PerformanceThreshold(
                metric_name="hook_execution_time",
                warning_threshold=2.0,
                error_threshold=5.0,
                critical_threshold=10.0,
                metric_type=MetricType.TIMER
            ),
            PerformanceThreshold(
                metric_name="error_rate",
                warning_threshold=0.01,
                error_threshold=0.05,
                critical_threshold=0.1,
                metric_type=MetricType.GAUGE
            ),
            PerformanceThreshold(
                metric_name="memory_usage",
                warning_threshold=0.8,
                error_threshold=0.9,
                critical_threshold=0.95,
                metric_type=MetricType.GAUGE
            ),
            PerformanceThreshold(
                metric_name="cpu_usage",
                warning_threshold=0.7,
                error_threshold=0.85,
                critical_threshold=0.95,
                metric_type=MetricType.GAUGE
            )
        ]

        for threshold in default_thresholds:
            self.add_threshold(threshold)

    def _get_recent_performance_metrics(self) -> Dict[str, Any]:
        """Get recent performance metrics for health snapshot."""
        current_time = datetime.now()
        start_time = current_time - timedelta(minutes=5)

        metrics = {}

        # Hook performance
        hook_time_metrics = self.metrics_collector.get_metrics('hook_execution_time', start_time=start_time)
        if hook_time_metrics:
            times = [m.value for m in hook_time_metrics if isinstance(m.value, (int, float))]
            metrics['hook_execution'] = {
                'count': len(times),
                'average': sum(times) / len(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0
            }

        # Agent performance
        agent_metrics = self.metrics_collector.get_metrics('agent_response_time', start_time=start_time)
        if agent_metrics:
            times = [m.value for m in agent_metrics if isinstance(m.value, (int, float))]
            metrics['agent_response'] = {
                'count': len(times),
                'average': sum(times) / len(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0
            }

        return metrics

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        current_time = datetime.now()
        start_time = current_time - timedelta(minutes=5)

        error_metrics = self.metrics_collector.get_metrics('error_count', start_time=start_time)
        total_metrics = self.metrics_collector.get_metrics('total_requests', start_time=start_time)

        error_count = sum(m.value for m in error_metrics if isinstance(m.value, (int, float)))
        total_count = sum(m.value for m in total_metrics if isinstance(m.value, (int, float)))

        return error_count / total_count if total_count > 0 else 0.0

    def _calculate_average_response_time(self) -> float:
        """Calculate average response time."""
        current_time = datetime.now()
        start_time = current_time - timedelta(minutes=5)

        response_metrics = self.metrics_collector.get_metrics('response_time', start_time=start_time)
        if not response_metrics:
            return 0.0

        times = [m.value for m in response_metrics if isinstance(m.value, (int, float))]
        return sum(times) / len(times) if times else 0.0

    def _calculate_throughput(self) -> float:
        """Calculate current throughput."""
        current_time = datetime.now()
        start_time = current_time - timedelta(minutes=1)

        throughput_metrics = self.metrics_collector.get_metrics('requests_per_second', start_time=start_time)
        if not throughput_metrics:
            return 0.0

        return sum(m.value for m in throughput_metrics if isinstance(m.value, (int, float)))

    def _get_active_sessions(self) -> int:
        """Get number of active sessions."""
        current_time = datetime.now()
        start_time = current_time - timedelta(minutes=5)

        session_metrics = self.metrics_collector.get_metrics('active_sessions', start_time=start_time)
        if not session_metrics:
            return 0

        return max(m.value for m in session_metrics if isinstance(m.value, int))

    def _get_total_sessions(self) -> int:
        """Get total number of sessions."""
        total_metrics = self.metrics_collector.get_metrics('total_sessions')
        if not total_metrics:
            return 0

        return max(m.value for m in total_metrics if isinstance(m.value, int))

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            import psutil

            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_io': sum(psutil.net_io_counters().bytes_sent, psutil.net_io_counters().bytes_recv)
            }
        except ImportError:
            return {}


# Factory function for creating real-time monitoring infrastructure
def create_real_time_monitoring(retention_hours: int = 24) -> tuple[MetricsCollector, RealTimeMonitor]:
    """
    Create and configure real-time monitoring infrastructure.

    Args:
        retention_hours: Hours to retain metrics data

    Returns:
        Tuple of (metrics_collector, real_time_monitor)
    """
    metrics_collector = MetricsCollector(retention_hours=retention_hours)
    real_time_monitor = RealTimeMonitor(metrics_collector)

    return metrics_collector, real_time_monitor