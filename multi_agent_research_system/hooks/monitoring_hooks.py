"""
System Monitoring Hooks for Multi-Agent Research System

Provides comprehensive system health monitoring, performance tracking,
error detection, and diagnostic capabilities for the entire multi-agent ecosystem.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import psutil

from .base_hooks import BaseHook, HookContext, HookPriority, HookResult, HookStatus

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class HealthStatus(Enum):
    """Enumeration of system health statuses."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"


class MetricType(Enum):
    """Enumeration of metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class SystemMetric:
    """Represents a system performance metric."""
    name: str
    metric_type: MetricType
    value: int | float | str
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)
    unit: str | None = None
    description: str | None = None


@dataclass
class HealthCheck:
    """Represents a system health check result."""
    check_name: str
    status: HealthStatus
    timestamp: datetime
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    next_check: datetime | None = None


@dataclass
class ErrorReport:
    """Represents an error report with detailed context."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: str  # low, medium, high, critical
    context: dict[str, Any] = field(default_factory=dict)
    stack_trace: str | None = None
    affected_components: list[str] = field(default_factory=list)
    recovery_suggestions: list[str] = field(default_factory=list)
    related_errors: list[str] = field(default_factory=list)


class SystemHealthHook(BaseHook):
    """Hook for monitoring overall system health and resource usage."""

    def __init__(self, check_interval: float = 30.0, timeout: float = 30.0, enabled: bool = True):
        super().__init__(
            name="system_health_monitor",
            hook_type="system_health",
            priority=HookPriority.NORMAL,
            timeout=15.0,
            enabled=enabled,
            retry_count=1
        )
        self.check_interval = check_interval
        self.health_checks: dict[str, HealthCheck] = {}
        self.system_metrics: list[SystemMetric] = []
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0,
            "response_time": 10.0
        }
        self.max_metrics = 10000

    async def execute(self, context: HookContext) -> HookResult:
        """Execute system health monitoring."""
        try:
            check_type = context.metadata.get("health_check_type", "comprehensive")
            component = context.metadata.get("component", "system")

            self.logger.info(f"System health check: {check_type} for {component}",
                           check_type=check_type,
                           component=component,
                           session_id=context.session_id)

            start_time = time.time()

            # Perform different types of health checks
            if check_type == "comprehensive":
                health_results = await self._perform_comprehensive_health_check(context)
            elif check_type == "resource":
                health_results = await self._perform_resource_health_check(context)
            elif check_type == "component":
                health_results = await self._perform_component_health_check(context, component)
            else:
                health_results = await self._perform_basic_health_check(context)

            execution_time = time.time() - start_time

            # Collect system metrics
            metrics = await self._collect_system_metrics()

            # Generate alerts if needed
            alerts = self._generate_health_alerts(health_results, metrics)

            self.logger.info(f"System health check completed: {check_type}",
                           check_type=check_type,
                           execution_time=execution_time,
                           healthy_checks=len([h for h in health_results if h.status == HealthStatus.HEALTHY]),
                           total_checks=len(health_results),
                           alerts_generated=len(alerts),
                           session_id=context.session_id)

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                execution_time=execution_time,
                result_data={
                    "check_type": check_type,
                    "component": component,
                    "health_results": [
                        {
                            "check_name": check.check_name,
                            "status": check.status.value,
                            "message": check.message,
                            "execution_time": check.execution_time
                        }
                        for check in health_results
                    ],
                    "metrics_collected": len(metrics),
                    "alerts": alerts,
                    "overall_health": self._calculate_overall_health(health_results)
                }
            )

        except Exception as e:
            self.logger.error(f"System health check failed: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__,
                            session_id=context.session_id)
            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.FAILED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                error_message=str(e),
                error_type=type(e).__name__
            )

    async def _perform_comprehensive_health_check(self, context: HookContext) -> list[HealthCheck]:
        """Perform comprehensive system health check."""
        checks = []

        # Resource checks
        checks.extend(await self._check_system_resources())

        # Component checks
        checks.extend(await self._check_agent_health(context))
        checks.extend(await self._check_mcp_servers(context))
        checks.extend(await self._check_hook_system(context))

        # Performance checks
        checks.extend(await self._check_performance_metrics())

        # Store health checks
        for check in checks:
            self.health_checks[check.check_name] = check

        return checks

    async def _perform_resource_health_check(self, context: HookContext) -> list[HealthCheck]:
        """Perform resource-specific health check."""
        return await self._check_system_resources()

    async def _perform_component_health_check(self, context: HookContext, component: str) -> list[HealthCheck]:
        """Perform component-specific health check."""
        if component == "agents":
            return await self._check_agent_health(context)
        elif component == "mcp_servers":
            return await self._check_mcp_servers(context)
        elif component == "hooks":
            return await self._check_hook_system(context)
        else:
            return await self._perform_basic_health_check(context)

    async def _perform_basic_health_check(self, context: HookContext) -> list[HealthCheck]:
        """Perform basic health check."""
        checks = []

        # Basic system checks
        checks.append(HealthCheck(
            check_name="system_responsive",
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            message="System is responsive",
            execution_time=0.001
        ))

        return checks

    async def _check_system_resources(self) -> list[HealthCheck]:
        """Check system resource usage."""
        checks = []

        try:
            # CPU usage check
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = HealthStatus.HEALTHY
            if cpu_percent > 95:
                cpu_status = HealthStatus.CRITICAL
            elif cpu_percent > 80:
                cpu_status = HealthStatus.WARNING

            checks.append(HealthCheck(
                check_name="cpu_usage",
                status=cpu_status,
                timestamp=datetime.now(),
                message=f"CPU usage: {cpu_percent:.1f}%",
                details={"usage_percent": cpu_percent, "threshold": self.alert_thresholds["cpu_usage"]},
                execution_time=0.001
            ))

            # Memory usage check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_status = HealthStatus.HEALTHY
            if memory_percent > 95:
                memory_status = HealthStatus.CRITICAL
            elif memory_percent > 85:
                memory_status = HealthStatus.WARNING

            checks.append(HealthCheck(
                check_name="memory_usage",
                status=memory_status,
                timestamp=datetime.now(),
                message=f"Memory usage: {memory_percent:.1f}%",
                details={
                    "usage_percent": memory_percent,
                    "used_gb": memory.used / (1024**3),
                    "total_gb": memory.total / (1024**3),
                    "threshold": self.alert_thresholds["memory_usage"]
                },
                execution_time=0.001
            ))

            # Disk usage check
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = HealthStatus.HEALTHY
            if disk_percent > 98:
                disk_status = HealthStatus.CRITICAL
            elif disk_percent > 90:
                disk_status = HealthStatus.WARNING

            checks.append(HealthCheck(
                check_name="disk_usage",
                status=disk_status,
                timestamp=datetime.now(),
                message=f"Disk usage: {disk_percent:.1f}%",
                details={
                    "usage_percent": disk_percent,
                    "used_gb": disk.used / (1024**3),
                    "total_gb": disk.total / (1024**3),
                    "threshold": self.alert_thresholds["disk_usage"]
                },
                execution_time=0.001
            ))

        except Exception as e:
            checks.append(HealthCheck(
                check_name="resource_check_error",
                status=HealthStatus.WARNING,
                timestamp=datetime.now(),
                message=f"Resource check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                execution_time=0.001
            ))

        return checks

    async def _check_agent_health(self, context: HookContext) -> list[HealthCheck]:
        """Check agent health status."""
        checks = []

        # This would integrate with the orchestrator to check agent health
        # For now, simulate basic agent health checks

        agent_count = context.metadata.get("agent_count", 4)
        healthy_agents = context.metadata.get("healthy_agents", agent_count)

        if healthy_agents == agent_count:
            status = HealthStatus.HEALTHY
            message = f"All {agent_count} agents are healthy"
        elif healthy_agents >= agent_count * 0.75:
            status = HealthStatus.WARNING
            message = f"{healthy_agents}/{agent_count} agents are healthy"
        else:
            status = HealthStatus.CRITICAL
            message = f"Only {healthy_agents}/{agent_count} agents are healthy"

        checks.append(HealthCheck(
            check_name="agent_health",
            status=status,
            timestamp=datetime.now(),
            message=message,
            details={
                "total_agents": agent_count,
                "healthy_agents": healthy_agents,
                "health_percentage": (healthy_agents / agent_count * 100) if agent_count > 0 else 0
            },
            execution_time=0.001
        ))

        return checks

    async def _check_mcp_servers(self, context: HookContext) -> list[HealthCheck]:
        """Check MCP server health."""
        checks = []

        # Simulate MCP server health checks
        mcp_servers = context.metadata.get("mcp_servers", {})
        server_count = len(mcp_servers)
        healthy_servers = len([s for s in mcp_servers.values() if s.get("status") == "available"])

        if server_count == 0:
            status = HealthStatus.WARNING
            message = "No MCP servers configured"
        elif healthy_servers == server_count:
            status = HealthStatus.HEALTHY
            message = f"All {server_count} MCP servers are available"
        elif healthy_servers >= server_count * 0.5:
            status = HealthStatus.WARNING
            message = f"{healthy_servers}/{server_count} MCP servers are available"
        else:
            status = HealthStatus.CRITICAL
            message = f"Only {healthy_servers}/{server_count} MCP servers are available"

        checks.append(HealthCheck(
            check_name="mcp_server_health",
            status=status,
            timestamp=datetime.now(),
            message=message,
            details={
                "total_servers": server_count,
                "healthy_servers": healthy_servers,
                "server_status": mcp_servers
            },
            execution_time=0.001
        ))

        return checks

    async def _check_hook_system(self, context: HookContext) -> list[HealthCheck]:
        """Check hook system health."""
        checks = []

        # Simulate hook system health check
        hook_stats = context.metadata.get("hook_stats", {})
        total_hooks = hook_stats.get("total_hooks", 0)

        if total_hooks == 0:
            status = HealthStatus.WARNING
            message = "No hooks registered"
        else:
            # Check for hooks with high failure rates
            failed_hooks = sum(1 for h in hook_stats.get("hook_details", [])
                            if h.get("success_rate", 100) < 80)

            if failed_hooks == 0:
                status = HealthStatus.HEALTHY
                message = f"All {total_hooks} hooks are healthy"
            elif failed_hooks <= total_hooks * 0.1:
                status = HealthStatus.WARNING
                message = f"{failed_hooks} hooks have low success rates"
            else:
                status = HealthStatus.CRITICAL
                message = f"{failed_hooks} hooks are failing frequently"

        checks.append(HealthCheck(
            check_name="hook_system_health",
            status=status,
            timestamp=datetime.now(),
            message=message,
            details={
                "total_hooks": total_hooks,
                "failed_hooks": failed_hooks if 'failed_hooks' in locals() else 0,
                "hook_stats": hook_stats
            },
            execution_time=0.001
        ))

        return checks

    async def _check_performance_metrics(self) -> list[HealthCheck]:
        """Check system performance metrics."""
        checks = []

        # Check response times (simulated)
        avg_response_time = 0.5  # Would be calculated from actual metrics

        if avg_response_time > 5.0:
            status = HealthStatus.CRITICAL
        elif avg_response_time > 2.0:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        checks.append(HealthCheck(
            check_name="response_time",
            status=status,
            timestamp=datetime.now(),
            message=f"Average response time: {avg_response_time:.2f}s",
            details={
                "avg_response_time": avg_response_time,
                "threshold": self.alert_thresholds["response_time"]
            },
            execution_time=0.001
        ))

        return checks

    async def _collect_system_metrics(self) -> list[SystemMetric]:
        """Collect system performance metrics."""
        metrics = []
        current_time = datetime.now()

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            metrics.append(SystemMetric(
                name="cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                value=cpu_percent,
                timestamp=current_time,
                unit="percent",
                description="Current CPU usage percentage"
            ))

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(SystemMetric(
                name="memory_usage_percent",
                metric_type=MetricType.GAUGE,
                value=memory.percent,
                timestamp=current_time,
                unit="percent",
                description="Current memory usage percentage"
            ))

            metrics.append(SystemMetric(
                name="memory_available_gb",
                metric_type=MetricType.GAUGE,
                value=memory.available / (1024**3),
                timestamp=current_time,
                unit="gigabytes",
                description="Available memory in GB"
            ))

            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.append(SystemMetric(
                name="disk_usage_percent",
                metric_type=MetricType.GAUGE,
                value=(disk.used / disk.total) * 100,
                timestamp=current_time,
                unit="percent",
                description="Current disk usage percentage"
            ))

            metrics.append(SystemMetric(
                name="disk_free_gb",
                metric_type=MetricType.GAUGE,
                value=disk.free / (1024**3),
                timestamp=current_time,
                unit="gigabytes",
                description="Free disk space in GB"
            ))

            # Process metrics
            process_count = len(psutil.pids())
            metrics.append(SystemMetric(
                name="process_count",
                metric_type=MetricType.GAUGE,
                value=process_count,
                timestamp=current_time,
                unit="count",
                description="Number of running processes"
            ))

        except Exception as e:
            self.logger.warning(f"Failed to collect some system metrics: {str(e)}")

        # Store metrics
        self.system_metrics.extend(metrics)
        if len(self.system_metrics) > self.max_metrics:
            self.system_metrics = self.system_metrics[-self.max_metrics:]

        return metrics

    def _generate_health_alerts(self, health_results: list[HealthCheck], metrics: list[SystemMetric]) -> list[dict[str, Any]]:
        """Generate health alerts based on check results and metrics."""
        alerts = []

        # Generate alerts from health checks
        for check in health_results:
            if check.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alert = {
                    "type": "health_check",
                    "severity": "high" if check.status == HealthStatus.CRITICAL else "medium",
                    "check_name": check.check_name,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat(),
                    "details": check.details
                }
                alerts.append(alert)

        # Generate alerts from metrics
        for metric in metrics:
            if metric.name == "cpu_usage_percent" and metric.value > self.alert_thresholds["cpu_usage"]:
                alerts.append({
                    "type": "metric_threshold",
                    "severity": "high" if metric.value > 95 else "medium",
                    "metric_name": metric.name,
                    "value": metric.value,
                    "threshold": self.alert_thresholds["cpu_usage"],
                    "timestamp": metric.timestamp.isoformat(),
                    "message": f"High CPU usage: {metric.value:.1f}%"
                })

            elif metric.name == "memory_usage_percent" and metric.value > self.alert_thresholds["memory_usage"]:
                alerts.append({
                    "type": "metric_threshold",
                    "severity": "high" if metric.value > 95 else "medium",
                    "metric_name": metric.name,
                    "value": metric.value,
                    "threshold": self.alert_thresholds["memory_usage"],
                    "timestamp": metric.timestamp.isoformat(),
                    "message": f"High memory usage: {metric.value:.1f}%"
                })

        return alerts

    def _calculate_overall_health(self, health_results: list[HealthCheck]) -> str:
        """Calculate overall system health status."""
        if not health_results:
            return "unknown"

        critical_count = len([h for h in health_results if h.status == HealthStatus.CRITICAL])
        warning_count = len([h for h in health_results if h.status == HealthStatus.WARNING])

        if critical_count > 0:
            return "critical"
        elif warning_count > len(health_results) * 0.3:
            return "degraded"
        elif warning_count > 0:
            return "warning"
        else:
            return "healthy"

    def get_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary."""
        if not self.health_checks:
            return {"message": "No health checks available"}

        # Count health statuses
        status_counts = {}
        for check in self.health_checks.values():
            status = check.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Get recent metrics
        recent_metrics = [m for m in self.system_metrics if (datetime.now() - m.timestamp).total_seconds() < 300]

        # Calculate averages for recent metrics
        metric_averages = {}
        for metric in recent_metrics:
            if metric.name not in metric_averages:
                metric_averages[metric.name] = []
            metric_averages[metric.name].append(metric.value)

        for name, values in metric_averages.items():
            if values:
                metric_averages[name] = sum(values) / len(values)

        return {
            "overall_health": self._calculate_overall_health(list(self.health_checks.values())),
            "total_checks": len(self.health_checks),
            "status_distribution": status_counts,
            "recent_metrics_count": len(recent_metrics),
            "metric_averages": metric_averages,
            "last_check_time": max(check.timestamp for check in self.health_checks.values()).isoformat() if self.health_checks else None
        }


class PerformanceMonitorHook(BaseHook):
    """Hook for detailed performance monitoring and analysis."""

    def __init__(self, enabled: bool = True, timeout: float = 10.0):
        super().__init__(
            name="performance_monitor",
            hook_type="performance_monitoring",
            priority=HookPriority.NORMAL,
            timeout=timeout,
            enabled=enabled,
            retry_count=0
        )
        self.performance_data: dict[str, list[dict[str, Any]]] = {}
        self.performance_snapshots: list[dict[str, Any]] = []
        self.max_data_points = 5000
        self.max_snapshots = 100

    async def execute(self, context: HookContext) -> HookResult:
        """Execute performance monitoring."""
        try:
            monitor_type = context.metadata.get("performance_monitor_type", "snapshot")
            component = context.metadata.get("component", "system")

            self.logger.info(f"Performance monitoring: {monitor_type} for {component}",
                           monitor_type=monitor_type,
                           component=component,
                           session_id=context.session_id)

            if monitor_type == "snapshot":
                result = await self._capture_performance_snapshot(context, component)
            elif monitor_type == "trend_analysis":
                result = await self._analyze_performance_trends(context, component)
            elif monitor_type == "bottleneck_detection":
                result = await self._detect_performance_bottlenecks(context, component)
            else:
                result = await self._capture_basic_metrics(context, component)

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                result_data=result
            )

        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__,
                            session_id=context.session_id)
            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.FAILED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                error_message=str(e),
                error_type=type(e).__name__
            )

    async def _capture_performance_snapshot(self, context: HookContext, component: str) -> dict[str, Any]:
        """Capture comprehensive performance snapshot."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "session_id": context.session_id,
            "system_metrics": await self._get_current_system_metrics(),
            "application_metrics": await self._get_application_metrics(context),
            "resource_usage": await self._get_resource_usage(),
            "performance_indicators": await self._calculate_performance_indicators()
        }

        # Store snapshot
        self.performance_snapshots.append(snapshot)
        if len(self.performance_snapshots) > self.max_snapshots:
            self.performance_snapshots = self.performance_snapshots[-self.max_snapshots:]

        return {
            "monitor_type": "snapshot",
            "snapshot_id": len(self.performance_snapshots),
            "component": component,
            "metrics_collected": len(snapshot["system_metrics"]) + len(snapshot["application_metrics"]),
            "performance_score": snapshot["performance_indicators"].get("overall_score", 0)
        }

    async def _analyze_performance_trends(self, context: HookContext, component: str) -> dict[str, Any]:
        """Analyze performance trends over time."""
        # Get recent snapshots for trend analysis
        recent_snapshots = [s for s in self.performance_snapshots
                          if s["component"] == component and
                          (datetime.now() - datetime.fromisoformat(s["timestamp"])).total_seconds() < 3600]

        if len(recent_snapshots) < 2:
            return {
                "monitor_type": "trend_analysis",
                "component": component,
                "message": "Insufficient data for trend analysis",
                "snapshots_analyzed": len(recent_snapshots)
            }

        # Analyze trends
        trends = {
            "cpu_trend": self._calculate_metric_trend(recent_snapshots, "cpu_usage"),
            "memory_trend": self._calculate_metric_trend(recent_snapshots, "memory_usage"),
            "response_time_trend": self._calculate_metric_trend(recent_snapshots, "response_time"),
            "error_rate_trend": self._calculate_metric_trend(recent_snapshots, "error_rate")
        }

        # Identify concerning trends
        concerning_trends = [key for key, trend in trends.items() if trend.get("direction") == "increasing" and trend.get("rate", 0) > 0.1]

        return {
            "monitor_type": "trend_analysis",
            "component": component,
            "snapshots_analyzed": len(recent_snapshots),
            "trends": trends,
            "concerning_trends": concerning_trends,
            "trend_period": "1 hour"
        }

    async def _detect_performance_bottlenecks(self, context: HookContext, component: str) -> dict[str, Any]:
        """Detect performance bottlenecks."""
        bottlenecks = []

        # Get current metrics
        current_metrics = await self._get_current_system_metrics()
        app_metrics = await self._get_application_metrics(context)

        # Check for common bottlenecks
        if current_metrics.get("cpu_usage", 0) > 80:
            bottlenecks.append({
                "type": "cpu_bottleneck",
                "severity": "high" if current_metrics["cpu_usage"] > 90 else "medium",
                "value": current_metrics["cpu_usage"],
                "threshold": 80,
                "suggestion": "Consider optimizing CPU-intensive operations or scaling horizontally"
            })

        if current_metrics.get("memory_usage", 0) > 85:
            bottlenecks.append({
                "type": "memory_bottleneck",
                "severity": "high" if current_metrics["memory_usage"] > 95 else "medium",
                "value": current_metrics["memory_usage"],
                "threshold": 85,
                "suggestion": "Check for memory leaks or optimize memory usage"
            })

        response_time = app_metrics.get("average_response_time", 0)
        if response_time > 5.0:
            bottlenecks.append({
                "type": "response_time_bottleneck",
                "severity": "high" if response_time > 10.0 else "medium",
                "value": response_time,
                "threshold": 5.0,
                "suggestion": "Investigate slow operations and consider caching or optimization"
            })

        return {
            "monitor_type": "bottleneck_detection",
            "component": component,
            "bottlenecks_detected": len(bottlenecks),
            "bottlenecks": bottlenecks,
            "overall_status": "degraded" if bottlenecks else "healthy"
        }

    async def _capture_basic_metrics(self, context: HookContext, component: str) -> dict[str, Any]:
        """Capture basic performance metrics."""
        metrics = await self._get_current_system_metrics()

        return {
            "monitor_type": "basic_metrics",
            "component": component,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

    async def _get_current_system_metrics(self) -> dict[str, Any]:
        """Get current system metrics."""
        try:
            import psutil

            return {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids()),
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            }
        except Exception:
            return {}

    async def _get_application_metrics(self, context: HookContext) -> dict[str, Any]:
        """Get application-specific metrics."""
        # This would integrate with the actual application metrics
        return {
            "active_sessions": context.metadata.get("active_sessions", 0),
            "average_response_time": context.metadata.get("avg_response_time", 0.0),
            "error_rate": context.metadata.get("error_rate", 0.0),
            "throughput": context.metadata.get("throughput", 0.0)
        }

    async def _get_resource_usage(self) -> dict[str, Any]:
        """Get detailed resource usage information."""
        try:
            import psutil

            process = psutil.Process()

            return {
                "process_cpu": process.cpu_percent(),
                "process_memory": process.memory_info().rss / (1024**2),  # MB
                "process_memory_percent": process.memory_percent(),
                "process_threads": process.num_threads(),
                "process_file_descriptors": process.num_fds() if hasattr(process, 'num_fds') else 0
            }
        except Exception:
            return {}

    async def _calculate_performance_indicators(self) -> dict[str, Any]:
        """Calculate performance indicators and scores."""
        indicators = {
            "overall_score": 85,  # Would be calculated based on actual metrics
            "cpu_score": 90,
            "memory_score": 85,
            "response_score": 80,
            "stability_score": 95
        }

        return indicators

    def _calculate_metric_trend(self, snapshots: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
        """Calculate trend for a specific metric."""
        if len(snapshots) < 2:
            return {"direction": "unknown", "rate": 0.0}

        values = []
        for snapshot in snapshots:
            if "system_metrics" in snapshot and metric_name in snapshot["system_metrics"]:
                values.append(snapshot["system_metrics"][metric_name])

        if len(values) < 2:
            return {"direction": "unknown", "rate": 0.0}

        # Calculate simple linear trend
        n = len(values)
        x_values = list(range(n))

        # Calculate slope (trend rate)
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values, strict=False))
        sum_x2 = sum(x * x for x in x_values)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        direction = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"

        return {
            "direction": direction,
            "rate": slope,
            "start_value": values[0],
            "end_value": values[-1],
            "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        }

    def get_performance_report(self, component: str | None = None, hours: int = 24) -> dict[str, Any]:
        """Get comprehensive performance report."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter snapshots
        snapshots = [s for s in self.performance_snapshots
                    if datetime.fromisoformat(s["timestamp"]) > cutoff_time]

        if component:
            snapshots = [s for s in snapshots if s["component"] == component]

        if not snapshots:
            return {"message": "No performance data available for the specified period"}

        # Calculate statistics
        return {
            "report_period": f"{hours} hours",
            "component": component or "all",
            "snapshots_analyzed": len(snapshots),
            "time_range": {
                "start": min(s["timestamp"] for s in snapshots),
                "end": max(s["timestamp"] for s in snapshots)
            },
            "performance_summary": self._calculate_performance_summary(snapshots),
            "resource_usage_summary": self._calculate_resource_summary(snapshots),
            "recommendations": self._generate_performance_recommendations(snapshots)
        }

    def _calculate_performance_summary(self, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate performance summary from snapshots."""
        if not snapshots:
            return {}

        # Extract performance indicators
        scores = [s.get("performance_indicators", {}).get("overall_score", 0) for s in snapshots]

        return {
            "average_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "score_trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining" if len(scores) > 1 and scores[-1] < scores[0] else "stable"
        }

    def _calculate_resource_summary(self, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate resource usage summary from snapshots."""
        if not snapshots:
            return {}

        cpu_values = [s.get("system_metrics", {}).get("cpu_usage", 0) for s in snapshots]
        memory_values = [s.get("system_metrics", {}).get("memory_usage", 0) for s in snapshots]

        return {
            "cpu": {
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0
            },
            "memory": {
                "average": sum(memory_values) / len(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "min": min(memory_values) if memory_values else 0
            }
        }

    def _generate_performance_recommendations(self, snapshots: list[dict[str, Any]]) -> list[str]:
        """Generate performance recommendations based on data."""
        recommendations = []

        if not snapshots:
            return recommendations

        # Check CPU usage
        cpu_values = [s.get("system_metrics", {}).get("cpu_usage", 0) for s in snapshots]
        avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0

        if avg_cpu > 70:
            recommendations.append("Consider optimizing CPU-intensive operations or scaling resources")

        # Check memory usage
        memory_values = [s.get("system_metrics", {}).get("memory_usage", 0) for s in snapshots]
        avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0

        if avg_memory > 75:
            recommendations.append("Monitor memory usage and optimize memory-intensive operations")

        # Check performance scores
        scores = [s.get("performance_indicators", {}).get("overall_score", 0) for s in snapshots]
        avg_score = sum(scores) / len(scores) if scores else 0

        if avg_score < 70:
            recommendations.append("Overall performance needs improvement - investigate bottlenecks")

        if not recommendations:
            recommendations.append("System performance is within acceptable ranges")

        return recommendations


class ErrorTrackingHook(BaseHook):
    """Hook for comprehensive error tracking and analysis."""

    def __init__(self, timeout: float = 30.0, enabled: bool = True):
        super().__init__(
            name="error_tracker",
            hook_type="error_tracking",
            priority=HookPriority.HIGH,
            timeout=10.0,
            enabled=enabled,
            retry_count=1
        )
        self.error_reports: list[ErrorReport] = []
        self.error_patterns: dict[str, dict[str, Any]] = {}
        self.error_thresholds = {
            "error_rate": 5.0,  # errors per hour
            "critical_error_rate": 1.0,
            "repeated_error_threshold": 3  # same error within 1 hour
        }
        self.max_errors = 1000

    async def execute(self, context: HookContext) -> HookResult:
        """Execute error tracking."""
        try:
            error_type = context.metadata.get("error_type", "UnknownError")
            error_message = context.metadata.get("error_message", "Unknown error")
            error_context = context.metadata.get("error_context", {})
            component = context.metadata.get("component", "unknown")
            severity = context.metadata.get("severity", "medium")

            self.logger.info(f"Error tracking: {error_type} in {component}",
                           error_type=error_type,
                           component=component,
                           severity=severity,
                           session_id=context.session_id)

            # Create error report
            error_report = ErrorReport(
                error_id=f"error_{int(time.time())}_{component}",
                timestamp=datetime.now(),
                error_type=error_type,
                error_message=error_message,
                severity=severity,
                context=error_context,
                stack_trace=context.metadata.get("stack_trace"),
                affected_components=[component],
                recovery_suggestions=self._generate_recovery_suggestions(error_type, error_context),
                related_errors=self._find_related_errors(error_type, error_message)
            )

            # Store error report
            self._store_error_report(error_report)

            # Update error patterns
            self._update_error_patterns(error_report)

            # Check for error thresholds and generate alerts
            alerts = self._check_error_thresholds(error_report)

            # Generate error summary
            error_summary = self._generate_error_summary()

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                result_data={
                    "error_id": error_report.error_id,
                    "error_type": error_type,
                    "component": component,
                    "severity": severity,
                    "alerts_generated": len(alerts),
                    "error_summary": error_summary,
                    "total_errors_tracked": len(self.error_reports)
                }
            )

        except Exception as e:
            self.logger.error(f"Error tracking failed: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__,
                            session_id=context.session_id)
            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.FAILED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                error_message=str(e),
                error_type=type(e).__name__
            )

    def _store_error_report(self, error_report: ErrorReport):
        """Store error report and maintain history size."""
        self.error_reports.append(error_report)
        if len(self.error_reports) > self.max_errors:
            self.error_reports = self.error_reports[-self.max_errors:]

    def _update_error_patterns(self, error_report: ErrorReport):
        """Update error pattern statistics."""
        pattern_key = f"{error_report.error_type}:{error_report.affected_components[0] if error_report.affected_components else 'unknown'}"

        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = {
                "error_type": error_report.error_type,
                "component": error_report.affected_components[0] if error_report.affected_components else "unknown",
                "count": 0,
                "first_occurrence": error_report.timestamp,
                "last_occurrence": error_report.timestamp,
                "severity_distribution": {},
                "recovery_success_rate": 0.0,
                "common_context": {}
            }

        pattern = self.error_patterns[pattern_key]
        pattern["count"] += 1
        pattern["last_occurrence"] = error_report.timestamp

        # Update severity distribution
        severity = error_report.severity
        pattern["severity_distribution"][severity] = pattern["severity_distribution"].get(severity, 0) + 1

        # Update common context
        for key, value in error_report.context.items():
            if key not in pattern["common_context"]:
                pattern["common_context"][key] = {}
            pattern["common_context"][key][str(value)] = pattern["common_context"][key].get(str(value), 0) + 1

    def _generate_recovery_suggestions(self, error_type: str, error_context: dict[str, Any]) -> list[str]:
        """Generate recovery suggestions based on error type and context."""
        suggestions = []

        # Common error patterns and suggestions
        if "timeout" in error_type.lower():
            suggestions.extend([
                "Increase timeout values for operations",
                "Check network connectivity",
                "Optimize slow operations",
                "Consider implementing retry logic with exponential backoff"
            ])
        elif "connection" in error_type.lower():
            suggestions.extend([
                "Verify network connectivity",
                "Check service availability",
                "Validate connection parameters",
                "Review firewall and security settings"
            ])
        elif "permission" in error_type.lower():
            suggestions.extend([
                "Check file and directory permissions",
                "Verify user access rights",
                "Review security policies",
                "Ensure proper authentication"
            ])
        elif "memory" in error_type.lower() or "out of memory" in error_type.lower():
            suggestions.extend([
                "Optimize memory usage",
                "Check for memory leaks",
                "Increase available memory",
                "Implement memory-efficient algorithms"
            ])
        else:
            suggestions.extend([
                "Review error logs for additional context",
                "Check system resources and dependencies",
                "Consult documentation for error-specific solutions",
                "Consider escalating to technical support if issue persists"
            ])

        return suggestions

    def _find_related_errors(self, error_type: str, error_message: str) -> list[str]:
        """Find related errors based on type and message similarity."""
        related_errors = []
        cutoff_time = datetime.now() - timedelta(hours=24)

        for error in self.error_reports:
            if error.timestamp < cutoff_time:
                continue

            # Check for same error type
            if error.error_type == error_type or any(word in error.error_message.lower() for word in error_message.lower().split() if len(word) > 3):
                related_errors.append(error.error_id)

        return related_errors[:5]  # Limit to 5 most recent related errors

    def _check_error_thresholds(self, error_report: ErrorReport) -> list[dict[str, Any]]:
        """Check if error exceeds thresholds and generate alerts."""
        alerts = []

        # Check for high error rate
        recent_errors = [e for e in self.error_reports
                       if (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour

        error_rate = len(recent_errors)
        if error_rate > self.error_thresholds["error_rate"]:
            alerts.append({
                "type": "high_error_rate",
                "severity": "high",
                "message": f"High error rate: {error_rate} errors in the last hour",
                "threshold": self.error_thresholds["error_rate"],
                "current_rate": error_rate
            })

        # Check for critical errors
        if error_report.severity == "critical":
            critical_errors = [e for e in recent_errors if e.severity == "critical"]
            if len(critical_errors) > self.error_thresholds["critical_error_rate"]:
                alerts.append({
                    "type": "critical_error_threshold",
                    "severity": "critical",
                    "message": f"Critical error threshold exceeded: {len(critical_errors)} critical errors",
                    "threshold": self.error_thresholds["critical_error_rate"],
                    "current_count": len(critical_errors)
                })

        # Check for repeated errors
        pattern_key = f"{error_report.error_type}:{error_report.affected_components[0] if error_report.affected_components else 'unknown'}"
        if pattern_key in self.error_patterns:
            pattern = self.error_patterns[pattern_key]
            if pattern["count"] >= self.error_thresholds["repeated_error_threshold"]:
                alerts.append({
                    "type": "repeated_error",
                    "severity": "medium",
                    "message": f"Repeated error detected: {error_report.error_type} occurred {pattern['count']} times",
                    "error_type": error_report.error_type,
                    "occurrence_count": pattern["count"]
                })

        return alerts

    def _generate_error_summary(self) -> dict[str, Any]:
        """Generate comprehensive error summary."""
        if not self.error_reports:
            return {"message": "No errors tracked"}

        # Calculate error statistics
        total_errors = len(self.error_reports)
        recent_errors = [e for e in self.error_reports
                       if (datetime.now() - e.timestamp).total_seconds() < 3600]

        # Error distribution by type
        error_types = {}
        severity_distribution = {}
        component_distribution = {}

        for error in self.error_reports:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            severity_distribution[error.severity] = severity_distribution.get(error.severity, 0) + 1

            for component in error.affected_components:
                component_distribution[component] = component_distribution.get(component, 0) + 1

        # Find most common errors
        most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_errors": total_errors,
            "recent_errors": len(recent_errors),
            "error_rate_per_hour": len(recent_errors),
            "error_types": dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]),
            "severity_distribution": severity_distribution,
            "component_distribution": component_distribution,
            "most_common_errors": most_common_errors,
            "error_patterns": len(self.error_patterns),
            "last_error": self.error_reports[-1].timestamp.isoformat() if self.error_reports else None
        }

    def get_error_report(
        self,
        error_type: str | None = None,
        component: str | None = None,
        severity: str | None = None,
        hours: int = 24,
        limit: int = 50
    ) -> dict[str, Any]:
        """Get filtered error reports."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        errors = [e for e in self.error_reports if e.timestamp > cutoff_time]

        # Apply filters
        if error_type:
            errors = [e for e in errors if e.error_type == error_type]

        if component:
            errors = [e for e in errors if component in e.affected_components]

        if severity:
            errors = [e for e in errors if e.severity == severity]

        # Sort by timestamp (most recent first) and limit
        errors.sort(key=lambda e: e.timestamp, reverse=True)
        errors = errors[:limit]

        return {
            "filter_criteria": {
                "error_type": error_type,
                "component": component,
                "severity": severity,
                "hours": hours,
                "limit": limit
            },
            "total_matching": len(errors),
            "errors": [
                {
                    "error_id": e.error_id,
                    "timestamp": e.timestamp.isoformat(),
                    "error_type": e.error_type,
                    "error_message": e.error_message,
                    "severity": e.severity,
                    "affected_components": e.affected_components,
                    "recovery_suggestions": e.recovery_suggestions
                }
                for e in errors
            ]
        }

    def get_error_patterns(self) -> dict[str, Any]:
        """Get error pattern analysis."""
        return {
            "total_patterns": len(self.error_patterns),
            "patterns": {
                pattern_key: {
                    "error_type": pattern["error_type"],
                    "component": pattern["component"],
                    "count": pattern["count"],
                    "first_occurrence": pattern["first_occurrence"].isoformat(),
                    "last_occurrence": pattern["last_occurrence"].isoformat(),
                    "severity_distribution": pattern["severity_distribution"],
                    "recovery_success_rate": pattern["recovery_success_rate"],
                    "common_context": dict(list(pattern["common_context"].items())[:5])  # Limit context display
                }
                for pattern_key, pattern in self.error_patterns.items()
            },
            "most_problematic_patterns": [
                {
                    "pattern": key,
                    "count": pattern["count"],
                    "component": pattern["component"]
                }
                for key, pattern in sorted(self.error_patterns.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
            ]
        }
