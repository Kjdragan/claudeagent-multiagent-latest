"""
Monitoring and Metrics Collection System

This module provides comprehensive monitoring and metrics collection capabilities
for the multi-agent research system, including real-time performance monitoring,
resource tracking, and alerting.

Key Features:
- Real-time performance monitoring
- Resource usage tracking
- Alert system for threshold breaches
- Metrics export to Prometheus and other formats
- Dashboard generation
- Health checks and system status

Based on Redesign Plan PLUS SDK Implementation (October 13, 2025)
"""

import time
import threading
import psutil
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
import json
import asyncio
from collections import defaultdict, deque

from .enhanced_logger import get_enhanced_logger, LogLevel, LogCategory, AgentEventType


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics collected."""
    COUNTER = "counter"      # Cumulative count
    GAUGE = "gauge"         # Current value
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"         # Timing measurements


@dataclass
class Metric:
    """A single metric data point."""

    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "unit": self.unit
        }


@dataclass
class Alert:
    """An alert generated from threshold monitoring."""

    name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    current_value: Union[int, float]
    threshold: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class MetricsCollector:
    """Collects and manages metrics for the system."""

    def __init__(self):
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

        # Resource monitoring
        self._process = psutil.Process()
        self._system_monitoring_enabled = True

        # Background monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._monitoring_interval = 5.0  # seconds

    def start_monitoring(self):
        """Start background monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(self._monitoring_interval)
            except Exception as e:
                logger = get_enhanced_logger("monitoring")
                logger.log_event(
                    LogLevel.ERROR,
                    LogCategory.SYSTEM,
                    AgentEventType.ERROR,
                    f"Monitoring error: {str(e)}",
                    error_details={"error_type": type(e).__name__, "error_message": str(e)}
                )

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        if not self._system_monitoring_enabled:
            return

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_gauge("system_cpu_percent", cpu_percent, {"unit": "percent"})

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_gauge("system_memory_percent", memory.percent, {"unit": "percent"})
            self.record_gauge("system_memory_used_gb", memory.used / (1024**3), {"unit": "GB"})
            self.record_gauge("system_memory_available_gb", memory.available / (1024**3), {"unit": "GB"})

            # Process-specific metrics
            process_memory = self._process.memory_info()
            self.record_gauge("process_memory_mb", process_memory.rss / (1024**2), {"unit": "MB"})
            self.record_gauge("process_cpu_percent", self._process.cpu_percent(), {"unit": "percent"})
            self.record_gauge("process_threads", self._process.num_threads(), {"unit": "count"})

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_gauge("disk_usage_percent", (disk.used / disk.total) * 100, {"unit": "percent"})
            self.record_gauge("disk_free_gb", disk.free / (1024**3), {"unit": "GB"})

        except Exception as e:
            logger = get_enhanced_logger("monitoring")
            logger.log_event(
                LogLevel.WARNING,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                f"Failed to collect system metrics: {str(e)}"
            )

    def record_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value

            metric = Metric(
                name=name,
                value=self._counters[key],
                metric_type=MetricType.COUNTER,
                timestamp=datetime.now(timezone.utc),
                labels=labels or {}
            )
            self._metrics[key].append(metric)

    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value

            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=datetime.now(timezone.utc),
                labels=labels or {}
            )
            self._metrics[key].append(metric)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)

            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                timestamp=datetime.now(timezone.utc),
                labels=labels or {}
            )
            self._metrics[key].append(metric)

    def record_timer(self, name: str, duration_ms: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._timers[key].append(duration_ms)

            metric = Metric(
                name=name,
                value=duration_ms,
                metric_type=MetricType.TIMER,
                timestamp=datetime.now(timezone.utc),
                labels=labels or {},
                unit="ms"
            )
            self._metrics[key].append(metric)

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Metric]:
        """Get the latest value for a metric."""
        key = self._make_key(name, labels)
        with self._lock:
            if key in self._metrics and self._metrics[key]:
                return self._metrics[key][-1]
        return None

    def get_metrics_history(self, name: str, labels: Optional[Dict[str, str]] = None,
                           since: Optional[datetime] = None) -> List[Metric]:
        """Get historical values for a metric."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._metrics:
                return []

            metrics = list(self._metrics[key])
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]

            return metrics

    def get_all_metrics(self) -> Dict[str, List[Metric]]:
        """Get all current metrics."""
        with self._lock:
            result = {}
            for key, metrics in self._metrics.items():
                if metrics:
                    result[key] = list(metrics)
            return result

    def calculate_statistics(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Calculate statistics for a metric."""
        metrics = self.get_metrics_history(name, labels)
        if not metrics:
            return {}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "sum": sum(values)
        }

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        with self._lock:
            # Export counters
            for key, value in self._counters.items():
                lines.append(f"# TYPE {key} counter")
                lines.append(f"{key} {value}")

            # Export gauges
            for key, value in self._gauges.items():
                lines.append(f"# TYPE {key} gauge")
                lines.append(f"{key} {value}")

            # Export histogram statistics
            for key, values in self._histograms.items():
                if values:
                    lines.append(f"# TYPE {key} histogram")
                    lines.append(f"{key}_count {len(values)}")
                    lines.append(f"{key}_sum {sum(values)}")
                    lines.append(f"{key}_bucket{{le=\"+Inf\"}} {len(values)}")

            # Export timer statistics
            for key, values in self._timers.items():
                if values:
                    lines.append(f"# TYPE {key} histogram")
                    lines.append(f"{key}_count {len(values)}")
                    lines.append(f"{key}_sum {sum(values)}")
                    lines.append(f"{key}_bucket{{le=\"+Inf\"}} {len(values)}")

        return "\n".join(lines)

    def clear_metrics(self, older_than: Optional[datetime] = None):
        """Clear old metrics."""
        cutoff_time = older_than or (datetime.now(timezone.utc) - timedelta(hours=24))

        with self._lock:
            for key in list(self._metrics.keys()):
                # Keep only recent metrics
                self._metrics[key] = deque(
                    [m for m in self._metrics[key] if m.timestamp >= cutoff_time],
                    maxlen=10000
                )

                # Remove empty metrics
                if not self._metrics[key]:
                    del self._metrics[key]


class AlertManager:
    """Manages alerts and threshold monitoring."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._alerts: Dict[str, Alert] = {}
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

        # Background alert checking
        self._alert_thread: Optional[threading.Thread] = None
        self._alert_active = False
        self._alert_interval = 10.0  # seconds

    def add_alert_rule(self, name: str, metric_name: str, threshold: Union[int, float],
                      comparison: str = "greater_than", severity: AlertSeverity = AlertSeverity.WARNING,
                      message_template: Optional[str] = None, labels: Optional[Dict[str, str]] = None):
        """Add an alert rule."""
        rule = {
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
            "message_template": message_template or f"{metric_name} is {{current_value}} (threshold: {threshold})",
            "labels": labels or {}
        }

        with self._lock:
            self._alert_rules[name] = rule

    def remove_alert_rule(self, name: str):
        """Remove an alert rule."""
        with self._lock:
            self._alert_rules.pop(name, None)
            # Also resolve any existing alerts for this rule
            if name in self._alerts:
                self._alerts[name].resolved = True
                self._alerts[name].resolved_at = datetime.now(timezone.utc)

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback for alert notifications."""
        self._alert_callbacks.append(callback)

    def start_alert_monitoring(self):
        """Start alert monitoring."""
        if self._alert_active:
            return

        self._alert_active = True
        self._alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self._alert_thread.start()

    def stop_alert_monitoring(self):
        """Stop alert monitoring."""
        self._alert_active = False
        if self._alert_thread:
            self._alert_thread.join(timeout=5.0)

    def _alert_loop(self):
        """Background alert checking loop."""
        while self._alert_active:
            try:
                self._check_alerts()
                time.sleep(self._alert_interval)
            except Exception as e:
                logger = get_enhanced_logger("alerts")
                logger.log_event(
                    LogLevel.ERROR,
                    LogCategory.SYSTEM,
                    AgentEventType.ERROR,
                    f"Alert monitoring error: {str(e)}",
                    error_details={"error_type": type(e).__name__, "error_message": str(e)}
                )

    def _check_alerts(self):
        """Check all alert rules and generate alerts if needed."""
        with self._lock:
            for rule_name, rule in self._alert_rules.items():
                try:
                    metric = self.metrics_collector.get_metric(rule["metric_name"], rule.get("labels"))
                    if metric is None:
                        continue

                    current_value = metric.value
                    threshold = rule["threshold"]
                    comparison = rule["comparison"]

                    # Check if alert should be triggered
                    should_alert = False
                    if comparison == "greater_than" and current_value > threshold:
                        should_alert = True
                    elif comparison == "less_than" and current_value < threshold:
                        should_alert = True
                    elif comparison == "equals" and current_value == threshold:
                        should_alert = True

                    # Create or resolve alert
                    if should_alert and (rule_name not in self._alerts or self._alerts[rule_name].resolved):
                        alert = Alert(
                            name=rule_name,
                            severity=rule["severity"],
                            message=rule["message_template"].format(current_value=current_value),
                            metric_name=rule["metric_name"],
                            current_value=current_value,
                            threshold=threshold,
                            timestamp=datetime.now(timezone.utc),
                            labels=rule["labels"]
                        )

                        self._alerts[rule_name] = alert
                        self._notify_alert(alert)

                    elif not should_alert and rule_name in self._alerts and not self._alerts[rule_name].resolved:
                        # Resolve alert
                        self._alerts[rule_name].resolved = True
                        self._alerts[rule_name].resolved_at = datetime.now(timezone.utc)
                        self._notify_alert(self._alerts[rule_name])

                except Exception as e:
                    logger = get_enhanced_logger("alerts")
                    logger.log_event(
                        LogLevel.ERROR,
                        LogCategory.SYSTEM,
                        AgentEventType.ERROR,
                        f"Error checking alert rule {rule_name}: {str(e)}"
                    )

    def _notify_alert(self, alert: Alert):
        """Notify all callbacks of an alert."""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger = get_enhanced_logger("alerts")
                logger.log_event(
                    LogLevel.ERROR,
                    LogCategory.SYSTEM,
                    AgentEventType.ERROR,
                    f"Error in alert callback: {str(e)}"
                )

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self._alerts.values() if not alert.resolved]

    def get_all_alerts(self, since: Optional[datetime] = None) -> List[Alert]:
        """Get all alerts, optionally filtered by time."""
        with self._lock:
            alerts = list(self._alerts.values())
            if since:
                alerts = [a for a in alerts if a.timestamp >= since]
            return alerts

    def resolve_alert(self, name: str):
        """Manually resolve an alert."""
        with self._lock:
            if name in self._alerts:
                self._alerts[name].resolved = True
                self._alerts[name].resolved_at = datetime.now(timezone.utc)
                self._notify_alert(self._alerts[name])


class HealthChecker:
    """System health checker."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._health_checks: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def add_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Add a health check function."""
        with self._lock:
            self._health_checks[name] = check_func

    def remove_health_check(self, name: str):
        """Remove a health check."""
        with self._lock:
            self._health_checks.pop(name, None)

    def check_health(self) -> Dict[str, Any]:
        """Run all health checks and return overall health status."""
        results = {}
        overall_healthy = True
        overall_status = "healthy"

        with self._lock:
            for name, check_func in self._health_checks.items():
                try:
                    result = check_func()
                    results[name] = result

                    if not result.get("healthy", True):
                        overall_healthy = False
                        if overall_status == "healthy":
                            overall_status = result.get("status", "unhealthy")

                except Exception as e:
                    results[name] = {
                        "healthy": False,
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    overall_healthy = False
                    overall_status = "error"

        return {
            "healthy": overall_healthy,
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": results
        }


# Default health check functions
def check_system_resources() -> Dict[str, Any]:
    """Check system resource health."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        healthy = True
        issues = []

        if cpu_percent > 90:
            healthy = False
            issues.append(f"High CPU usage: {cpu_percent}%")

        if memory.percent > 90:
            healthy = False
            issues.append(f"High memory usage: {memory.percent}%")

        if (disk.used / disk.total) > 0.9:
            healthy = False
            issues.append(f"High disk usage: {(disk.used / disk.total) * 100:.1f}%")

        return {
            "healthy": healthy,
            "status": "unhealthy" if issues else "healthy",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": (disk.used / disk.total) * 100,
            "issues": issues,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        return {
            "healthy": False,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


def check_process_health() -> Dict[str, Any]:
    """Check the health of the current process."""
    try:
        process = psutil.Process()
        return {
            "healthy": process.is_running(),
            "status": "running" if process.is_running() else "not_running",
            "pid": process.pid,
            "memory_mb": process.memory_info().rss / (1024**2),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        return {
            "healthy": False,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class MonitoringSystem:
    """Main monitoring system that coordinates all monitoring components."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_checker = HealthChecker(self.metrics_collector)

        # Setup default health checks
        self.health_checker.add_health_check("system_resources", check_system_resources)
        self.health_checker.add_health_check("process_health", check_process_health)

        # Setup default alert rules
        self._setup_default_alerts()

        # Logging
        self.logger = get_enhanced_logger("monitoring_system")

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # System alerts
        self.alert_manager.add_alert_rule(
            "high_cpu",
            "system_cpu_percent",
            85,
            "greater_than",
            AlertSeverity.WARNING,
            "High CPU usage: {current_value:.1f}%"
        )

        self.alert_manager.add_alert_rule(
            "high_memory",
            "system_memory_percent",
            85,
            "greater_than",
            AlertSeverity.WARNING,
            "High memory usage: {current_value:.1f}%"
        )

        self.alert_manager.add_alert_rule(
            "high_process_memory",
            "process_memory_mb",
            2048,  # 2GB
            "greater_than",
            AlertSeverity.WARNING,
            "High process memory usage: {current_value:.1f}MB"
        )

        # Error rate alerts
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            "error_rate",
            10,  # 10%
            "greater_than",
            AlertSeverity.ERROR,
            "High error rate: {current_value:.1f}%"
        )

    def start(self):
        """Start all monitoring components."""
        self.logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.SESSION_START,
            "Starting monitoring system"
        )

        self.metrics_collector.start_monitoring()
        self.alert_manager.start_alert_monitoring()

    def stop(self):
        """Stop all monitoring components."""
        self.logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.SESSION_END,
            "Stopping monitoring system"
        )

        self.metrics_collector.stop_monitoring()
        self.alert_manager.stop_alert_monitoring()

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "health": self.health_checker.check_health(),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "metrics_summary": self._get_metrics_summary(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics."""
        summary = {}

        # System metrics
        cpu_metric = self.metrics_collector.get_metric("system_cpu_percent")
        if cpu_metric:
            summary["cpu_percent"] = cpu_metric.value

        memory_metric = self.metrics_collector.get_metric("system_memory_percent")
        if memory_metric:
            summary["memory_percent"] = memory_metric.value

        # Process metrics
        process_memory = self.metrics_collector.get_metric("process_memory_mb")
        if process_memory:
            summary["process_memory_mb"] = process_memory.value

        # Performance metrics
        task_duration = self.metrics_collector.calculate_statistics("agent_task_duration")
        if task_duration:
            summary["avg_task_duration_ms"] = task_duration["avg"]

        return summary

    def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format."""
        if format.lower() == "prometheus":
            return self.metrics_collector.export_prometheus_format()
        elif format.lower() == "json":
            metrics = self.metrics_collector.get_all_metrics()
            return json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": {k: [m.to_dict() for m in v] for k, v in metrics.items()}
            }, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def record_agent_task(self, agent_type: str, task_name: str, duration_ms: float,
                         success: bool = True, quality_score: Optional[float] = None):
        """Record agent task metrics."""
        labels = {"agent_type": agent_type, "task_name": task_name}
        self.metrics_collector.record_timer("agent_task_duration", duration_ms, labels)
        self.metrics_collector.record_counter("agent_tasks_total", 1, labels)

        if success:
            self.metrics_collector.record_counter("agent_tasks_successful", 1, labels)
            if quality_score:
                self.metrics_collector.record_gauge("agent_task_quality", quality_score, labels)
        else:
            self.metrics_collector.record_counter("agent_tasks_failed", 1, labels)

    def record_tool_execution(self, tool_name: str, duration_ms: float, success: bool = True):
        """Record tool execution metrics."""
        labels = {"tool_name": tool_name}
        self.metrics_collector.record_timer("tool_execution_duration", duration_ms, labels)
        self.metrics_collector.record_counter("tool_executions_total", 1, labels)

        if success:
            self.metrics_collector.record_counter("tool_executions_successful", 1, labels)
        else:
            self.metrics_collector.record_counter("tool_executions_failed", 1, labels)


# Global monitoring system instance
_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system() -> MonitoringSystem:
    """Get the global monitoring system instance."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system


def start_monitoring():
    """Start the global monitoring system."""
    system = get_monitoring_system()
    system.start()


def stop_monitoring():
    """Stop the global monitoring system."""
    system = get_monitoring_system()
    system.stop()


def record_agent_task(agent_type: str, task_name: str, duration_ms: float,
                     success: bool = True, quality_score: Optional[float] = None):
    """Record agent task metrics in the global monitoring system."""
    system = get_monitoring_system()
    system.record_agent_task(agent_type, task_name, duration_ms, success, quality_score)


def record_tool_execution(tool_name: str, duration_ms: float, success: bool = True):
    """Record tool execution metrics in the global monitoring system."""
    system = get_monitoring_system()
    system.record_tool_execution(tool_name, duration_ms, success)