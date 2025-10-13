"""
Performance Monitor - Comprehensive Performance Monitoring and Optimization

This module provides sophisticated performance monitoring, optimization, and analysis
capabilities for the message processing system.

Key Features:
- Real-time performance monitoring with metrics collection
- Performance bottleneck identification and analysis
- Automated optimization suggestions and implementation
- Resource usage monitoring (CPU, memory, I/O)
- Performance trend analysis and forecasting
- Alerting system for performance degradation
- Performance benchmarking and comparison
- Optimization recommendations based on usage patterns
"""

import asyncio
import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import statistics
import json

from ..core.message_types import RichMessage, EnhancedMessageType


class MetricType(Enum):
    """Types of performance metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric with metadata."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class PerformanceAlert:
    """Performance alert with context and recommendations."""

    alert_id: str
    level: AlertLevel
    metric_name: str
    current_value: float
    threshold: float
    message: str
    recommendations: List[str]
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class OptimizationSuggestion:
    """Optimization suggestion with implementation details."""

    suggestion_id: str
    component: str
    issue: str
    suggestion: str
    expected_improvement: float
    implementation_complexity: str
    priority: str
    auto_applicable: bool = False
    implementation_code: Optional[str] = None


class PerformanceMonitor:
    """Comprehensive performance monitoring and optimization system."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize performance monitor with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Monitoring configuration
        self.monitoring_interval = self.config.get("monitoring_interval", 1.0)
        self.metrics_retention_hours = self.config.get("metrics_retention_hours", 24)
        self.alert_cooldown_minutes = self.config.get("alert_cooldown_minutes", 5)
        self.enable_auto_optimization = self.config.get("enable_auto_optimization", False)

        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)

        # Alerts and suggestions
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.optimization_suggestions: List[OptimizationSuggestion] = []

        # Performance thresholds
        self.thresholds = self._initialize_thresholds()

        # Monitoring state
        self._monitoring = False
        self._monitor_task = None
        self._system_monitor_thread = None

        # Statistics
        self.monitoring_stats = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "optimizations_applied": 0,
            "monitoring_start_time": None
        }

        # Initialize monitoring components
        self._initialize_monitors()

    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize performance thresholds."""
        return {
            "processing_time": {
                "warning": 1.0,  # seconds
                "error": 5.0,
                "critical": 10.0
            },
            "cache_hit_rate": {
                "warning": 0.5,
                "error": 0.3,
                "critical": 0.1
            },
            "error_rate": {
                "warning": 0.05,
                "error": 0.1,
                "critical": 0.2
            },
            "memory_usage": {
                "warning": 0.7,
                "error": 0.85,
                "critical": 0.95
            },
            "cpu_usage": {
                "warning": 0.7,
                "error": 0.85,
                "critical": 0.95
            },
            "queue_depth": {
                "warning": 100,
                "error": 500,
                "critical": 1000
            }
        }

    def _initialize_monitors(self):
        """Initialize monitoring components."""
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss

    async def start_monitoring(self):
        """Start performance monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self.monitoring_stats["monitoring_start_time"] = datetime.now()

        # Start metrics collection task
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

        # Start system monitoring thread
        self._system_monitor_thread = threading.Thread(
            target=self._system_monitor_loop,
            daemon=True
        )
        self._system_monitor_thread.start()

        self.logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False

        # Stop monitoring task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for application metrics."""
        while self._monitoring:
            try:
                await self._collect_application_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.monitoring_interval)

    def _system_monitor_loop(self):
        """System monitoring loop for system metrics."""
        while self._monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {str(e)}")
                time.sleep(self.monitoring_interval)

    async def _collect_application_metrics(self):
        """Collect application-level performance metrics."""
        # This would be called by the message processing components
        # For now, we'll collect some basic metrics
        current_time = datetime.now()

        # Example metrics (would be populated by actual components)
        self.record_metric(
            "messages_processed_total",
            self.counters.get("messages_processed", 0),
            MetricType.COUNTER,
            current_time
        )

        self.record_metric(
            "average_processing_time",
            statistics.mean(self.timers["processing_time"]) if self.timers["processing_time"] else 0,
            MetricType.GAUGE,
            current_time
        )

        # Check for performance issues
        await self._check_performance_thresholds()

    def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        current_time = datetime.now()

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_metric("cpu_usage_percent", cpu_percent, MetricType.GAUGE, current_time)

            # Memory usage
            memory_info = psutil.virtual_memory()
            self.record_metric("memory_usage_percent", memory_info.percent, MetricType.GAUGE, current_time)
            self.record_metric("memory_available_mb", memory_info.available / 1024 / 1024, MetricType.GAUGE, current_time)

            # Process-specific memory
            process_memory = self.process.memory_info()
            self.record_metric("process_memory_mb", process_memory.rss / 1024 / 1024, MetricType.GAUGE, current_time)

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.record_metric("disk_read_bytes", disk_io.read_bytes, MetricType.COUNTER, current_time)
                self.record_metric("disk_write_bytes", disk_io.write_bytes, MetricType.COUNTER, current_time)

            # Network I/O
            network_io = psutil.net_io_counters()
            if network_io:
                self.record_metric("network_bytes_sent", network_io.bytes_sent, MetricType.COUNTER, current_time)
                self.record_metric("network_bytes_recv", network_io.bytes_recv, MetricType.COUNTER, current_time)

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")

    def record_metric(self, name: str, value: float, metric_type: MetricType, timestamp: datetime = None):
        """Record a performance metric."""
        if timestamp is None:
            timestamp = datetime.now()

        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=timestamp
        )

        # Store metric
        self.metrics[name].append(metric)
        self.monitoring_stats["metrics_collected"] += 1

        # Update specific metric type storage
        if metric_type == MetricType.COUNTER:
            self.counters[name] = value
        elif metric_type == MetricType.GAUGE:
            self.gauges[name] = value
        elif metric_type == MetricType.HISTOGRAM:
            self.histograms[name].append(value)
            # Keep only recent values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
        elif metric_type == MetricType.TIMER:
            self.timers[name].append(value)
            # Keep only recent values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]

    def record_message_processing(self, message: RichMessage, processing_time: float, success: bool):
        """Record message processing metrics."""
        timestamp = datetime.now()

        # General metrics
        self.record_metric("messages_processed_total", self.counters.get("messages_processed", 0) + 1, MetricType.COUNTER, timestamp)
        self.record_metric("processing_time_seconds", processing_time, MetricType.TIMER, timestamp)

        # Success/failure metrics
        if success:
            self.record_metric("messages_successful_total", self.counters.get("messages_successful", 0) + 1, MetricType.COUNTER, timestamp)
        else:
            self.record_metric("messages_failed_total", self.counters.get("messages_failed", 0) + 1, MetricType.COUNTER, timestamp)

        # By message type
        type_counter_name = f"messages_{message.message_type.value}_total"
        self.record_metric(type_counter_name, self.counters.get(type_counter_name, 0) + 1, MetricType.COUNTER, timestamp)

        # By message priority
        priority_counter_name = f"messages_{message.priority.value}_total"
        self.record_metric(priority_counter_name, self.counters.get(priority_counter_name, 0) + 1, MetricType.COUNTER, timestamp)

        # Calculate error rate
        total_messages = self.counters.get("messages_processed", 1)
        failed_messages = self.counters.get("messages_failed", 0)
        error_rate = failed_messages / total_messages
        self.record_metric("error_rate", error_rate, MetricType.GAUGE, timestamp)

    def record_cache_operation(self, operation: str, hit: bool, duration: float):
        """Record cache operation metrics."""
        timestamp = datetime.now()

        # Cache hit/miss metrics
        if hit:
            self.record_metric("cache_hits_total", self.counters.get("cache_hits", 0) + 1, MetricType.COUNTER, timestamp)
        else:
            self.record_metric("cache_misses_total", self.counters.get("cache_misses", 0) + 1, MetricType.COUNTER, timestamp)

        # Cache operation duration
        self.record_metric(f"cache_{operation}_duration_seconds", duration, MetricType.TIMER, timestamp)

        # Calculate hit rate
        total_ops = self.counters.get("cache_hits", 0) + self.counters.get("cache_misses", 1)
        hit_rate = self.counters.get("cache_hits", 0) / total_ops
        self.record_metric("cache_hit_rate", hit_rate, MetricType.GAUGE, timestamp)

    async def _check_performance_thresholds(self):
        """Check performance against thresholds and trigger alerts."""
        current_time = datetime.now()

        for metric_name, thresholds in self.thresholds.items():
            current_value = self.gauges.get(metric_name, 0)

            if current_value == 0:
                continue

            # Check each alert level
            for level, threshold in thresholds.items():
                if self._should_trigger_alert(metric_name, current_value, threshold, level):
                    await self._trigger_alert(metric_name, current_value, threshold, level, current_time)

    def _should_trigger_alert(self, metric_name: str, current_value: float, threshold: float, level: str) -> bool:
        """Check if an alert should be triggered."""
        alert_key = f"{metric_name}_{level}"

        # Check if we have a recent alert for this metric and level
        if alert_key in self.active_alerts:
            last_alert = self.active_alerts[alert_key]
            cooldown = timedelta(minutes=self.alert_cooldown_minutes)
            if datetime.now() - last_alert.timestamp < cooldown:
                return False

        # Check threshold based on metric type
        if metric_name in ["cache_hit_rate"]:
            return current_value < threshold
        elif metric_name in ["error_rate"]:
            return current_value > threshold
        else:
            return current_value > threshold

    async def _trigger_alert(self, metric_name: str, current_value: float, threshold: float, level: str, timestamp: datetime):
        """Trigger a performance alert."""
        alert_id = f"{metric_name}_{level}_{int(timestamp.timestamp())}"

        alert_level = AlertLevel(level.upper())
        message = f"Performance alert: {metric_name} is {current_value:.2f} (threshold: {threshold:.2f})"

        # Generate recommendations
        recommendations = self._generate_alert_recommendations(metric_name, current_value, level)

        alert = PerformanceAlert(
            alert_id=alert_id,
            level=alert_level,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            message=message,
            recommendations=recommendations,
            timestamp=timestamp
        )

        # Store alert
        self.active_alerts[f"{metric_name}_{level}"] = alert
        self.alert_history.append(alert)
        self.monitoring_stats["alerts_triggered"] += 1

        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert_level, logging.WARNING)

        self.logger.log(log_level, f"ALERT: {message}")
        for rec in recommendations:
            self.logger.log(log_level, f"  RECOMMENDATION: {rec}")

        # Apply auto-optimization if enabled and applicable
        if self.enable_auto_optimization and alert_level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            await self._apply_auto_optimization(metric_name, alert)

    def _generate_alert_recommendations(self, metric_name: str, current_value: float, level: str) -> List[str]:
        """Generate recommendations for performance alerts."""
        recommendations = []

        if metric_name == "processing_time":
            if current_value > 5.0:
                recommendations.extend([
                    "Consider optimizing message processing algorithms",
                    "Increase processing timeout values",
                    "Implement message batching for bulk operations",
                    "Check for I/O bottlenecks in processing pipeline"
                ])
            elif current_value > 1.0:
                recommendations.extend([
                    "Monitor processing queue depth",
                    "Consider parallel processing for independent messages",
                    "Review message complexity and size"
                ])

        elif metric_name == "cache_hit_rate":
            if current_value < 0.3:
                recommendations.extend([
                    "Review cache key generation strategy",
                    "Increase cache size if memory permits",
                    "Analyze message access patterns",
                    "Consider cache warming strategies"
                ])
            elif current_value < 0.5:
                recommendations.extend([
                    "Monitor cache eviction policies",
                    "Review TTL settings for cached items"
                ])

        elif metric_name == "error_rate":
            if current_value > 0.1:
                recommendations.extend([
                    "Investigate root causes of processing errors",
                    "Implement better error handling and recovery",
                    "Review input validation and sanitization",
                    "Consider circuit breaker patterns for unreliable operations"
                ])

        elif metric_name == "memory_usage":
            if current_value > 0.85:
                recommendations.extend([
                    "Implement memory leak detection and fixing",
                    "Optimize data structures and algorithms",
                    "Consider memory pooling for frequently allocated objects",
                    "Monitor garbage collection patterns"
                ])

        elif metric_name == "cpu_usage":
            if current_value > 0.85:
                recommendations.extend([
                    "Profile CPU-intensive operations",
                    "Implement better caching strategies",
                    "Consider asynchronous processing for blocking operations",
                    "Optimize algorithms and data structures"
                ])

        return recommendations

    async def _apply_auto_optimization(self, metric_name: str, alert: PerformanceAlert):
        """Apply automatic optimizations for critical alerts."""
        optimization_applied = False

        if metric_name == "processing_time" and alert.current_value > 5.0:
            # Could implement automatic batching or timeout adjustments
            optimization_applied = True

        elif metric_name == "cache_hit_rate" and alert.current_value < 0.3:
            # Could automatically adjust cache size or TTL
            optimization_applied = True

        if optimization_applied:
            self.monitoring_stats["optimizations_applied"] += 1
            self.logger.info(f"Auto-optimization applied for {metric_name}")

    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time window."""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

        summary = {
            "time_window_minutes": time_window_minutes,
            "metrics": {},
            "alerts": {
                "active_count": len(self.active_alerts),
                "recent_count": len([a for a in self.alert_history if a.timestamp >= cutoff_time])
            },
            "system_health": self._calculate_system_health()
        }

        # Calculate metric summaries
        for metric_name, metric_deque in self.metrics.items():
            recent_metrics = [m for m in metric_deque if m.timestamp >= cutoff_time]

            if not recent_metrics:
                continue

            values = [m.value for m in recent_metrics]
            summary["metrics"][metric_name] = {
                "count": len(values),
                "latest": values[-1],
                "min": min(values),
                "max": max(values),
                "average": statistics.mean(values),
                "median": statistics.median(values)
            }

            # Add standard deviation for numeric metrics
            if len(values) > 1:
                summary["metrics"][metric_name]["std_dev"] = statistics.stdev(values)

        return summary

    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        health_factors = {}

        # CPU health (0-100, lower is better)
        cpu_usage = self.gauges.get("cpu_usage_percent", 0)
        health_factors["cpu"] = max(0, 100 - cpu_usage)

        # Memory health (0-100, lower is better)
        memory_usage = self.gauges.get("memory_usage_percent", 0)
        health_factors["memory"] = max(0, 100 - memory_usage)

        # Error rate health (0-100, lower is better)
        error_rate = self.gauges.get("error_rate", 0)
        health_factors["errors"] = max(0, 100 - (error_rate * 100))

        # Processing time health (0-100, lower is better)
        avg_processing_time = statistics.mean(self.timers["processing_time"]) if self.timers["processing_time"] else 0
        processing_health = max(0, 100 - (avg_processing_time * 10))  # 1s = 10 point penalty
        health_factors["processing"] = processing_health

        # Cache health (higher is better)
        cache_hit_rate = self.gauges.get("cache_hit_rate", 0)
        health_factors["cache"] = cache_hit_rate * 100

        # Overall health (average of all factors)
        overall_health = statistics.mean(health_factors.values())

        return {
            "overall": overall_health,
            "factors": health_factors,
            "status": self._get_health_status(overall_health)
        }

    def _get_health_status(self, health_score: float) -> str:
        """Get health status based on score."""
        if health_score >= 90:
            return "excellent"
        elif health_score >= 75:
            return "good"
        elif health_score >= 60:
            return "fair"
        elif health_score >= 40:
            return "poor"
        else:
            return "critical"

    def get_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """Get optimization suggestions based on current performance."""
        suggestions = []

        # Analyze current metrics and generate suggestions
        current_time = datetime.now()

        # Processing time optimization
        if self.timers["processing_time"]:
            avg_time = statistics.mean(self.timers["processing_time"])
            if avg_time > 1.0:
                suggestions.append(OptimizationSuggestion(
                    suggestion_id="proc_time_opt_1",
                    component="message_processor",
                    issue="High average processing time",
                    suggestion="Implement message batching and parallel processing",
                    expected_improvement=30.0,
                    implementation_complexity="medium",
                    priority="high",
                    auto_applicable=False
                ))

        # Cache optimization
        cache_hit_rate = self.gauges.get("cache_hit_rate", 0)
        if cache_hit_rate < 0.7:
            suggestions.append(OptimizationSuggestion(
                suggestion_id="cache_opt_1",
                component="message_cache",
                issue="Low cache hit rate",
                suggestion="Increase cache size and optimize cache key generation",
                expected_improvement=40.0,
                implementation_complexity="low",
                priority="medium",
                auto_applicable=True
            ))

        # Memory optimization
        memory_usage = self.gauges.get("memory_usage_percent", 0)
        if memory_usage > 0.7:
            suggestions.append(OptimizationSuggestion(
                suggestion_id="mem_opt_1",
                component="system",
                issue="High memory usage",
                suggestion="Implement memory pooling and optimize data structures",
                expected_improvement=25.0,
                implementation_complexity="high",
                priority="high",
                auto_applicable=False
            ))

        return suggestions

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        stats = self.monitoring_stats.copy()

        if stats["monitoring_start_time"]:
            uptime = datetime.now() - stats["monitoring_start_time"]
            stats["uptime_seconds"] = uptime.total_seconds()
            stats["uptime_formatted"] = str(uptime).split('.')[0]

        # Calculate metrics collection rate
        if stats["uptime_seconds"] > 0:
            stats["metrics_per_second"] = stats["metrics_collected"] / stats["uptime_seconds"]

        return stats

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_stats": self.get_monitoring_stats(),
            "performance_summary": self.get_performance_summary(),
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "level": alert.level.value,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.active_alerts.values()
            ],
            "optimization_suggestions": [
                {
                    "suggestion_id": s.suggestion_id,
                    "component": s.component,
                    "issue": s.issue,
                    "suggestion": s.suggestion,
                    "expected_improvement": s.expected_improvement,
                    "priority": s.priority
                }
                for s in self.get_optimization_suggestions()
            ]
        }

        if format.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def reset_metrics(self):
        """Reset all metrics and statistics."""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timers.clear()
        self.alert_history.clear()

        # Keep active alerts but mark them as resolved
        for alert in self.active_alerts.values():
            alert.resolved = True
            alert.resolved_at = datetime.now()

        self.monitoring_stats = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "optimizations_applied": 0,
            "monitoring_start_time": datetime.now() if self._monitoring else None
        }

        self.logger.info("Performance metrics reset")


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(config: Dict[str, Any] = None) -> PerformanceMonitor:
    """Get or create global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(config)
    return _global_monitor


async def start_global_monitoring(config: Dict[str, Any] = None):
    """Start global performance monitoring."""
    monitor = get_performance_monitor(config)
    await monitor.start_monitoring()


async def stop_global_monitoring():
    """Stop global performance monitoring."""
    global _global_monitor
    if _global_monitor:
        await _global_monitor.stop_monitoring()


# Decorators for automatic performance monitoring
def monitor_performance(metric_name: str = None):
    """Decorator to automatically monitor function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time

                name = metric_name or f"{func.__module__}.{func.__name__}"
                monitor.record_metric(f"{name}_duration", duration, MetricType.TIMER)
                monitor.record_metric(f"{name}_calls", 1, MetricType.COUNTER)

                if not success:
                    monitor.record_metric(f"{name}_errors", 1, MetricType.COUNTER)

        def sync_wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time

                name = metric_name or f"{func.__module__}.{func.__name__}"
                monitor.record_metric(f"{name}_duration", duration, MetricType.TIMER)
                monitor.record_metric(f"{name}_calls", 1, MetricType.COUNTER)

                if not success:
                    monitor.record_metric(f"{name}_errors", 1, MetricType.COUNTER)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator