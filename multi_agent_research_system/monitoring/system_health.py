"""
System Health Monitor for comprehensive health tracking.

This module provides system health monitoring, including service health checks,
resource monitoring, and automated health reporting.
"""

import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_logging import StructuredLogger


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check configuration and result."""
    name: str
    description: str
    check_function: Callable
    interval_seconds: int
    timeout_seconds: int
    last_check: datetime | None = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_result: dict[str, Any] | None = None
    consecutive_failures: int = 0
    enabled: bool = True


@dataclass
class HealthReport:
    """Comprehensive health report."""
    timestamp: datetime
    overall_status: HealthStatus
    system_uptime_seconds: float
    total_checks: int
    healthy_checks: int
    warning_checks: int
    critical_checks: int
    unknown_checks: int
    check_results: list[dict[str, Any]]
    system_resources: dict[str, Any]
    alerts: list[dict[str, Any]]
    recommendations: list[str]


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""

    def __init__(self,
                 session_id: str,
                 health_dir: str = "health",
                 check_interval: int = 60):
        """
        Initialize the system health monitor.

        Args:
            session_id: Session identifier
            health_dir: Directory to store health data
            check_interval: Seconds between health check cycles
        """
        self.session_id = session_id
        self.health_dir = Path(health_dir)
        self.health_dir.mkdir(parents=True, exist_ok=True)
        self.check_interval = check_interval

        # Initialize structured logger
        self.logger = StructuredLogger(
            name="system_health",
            log_dir=self.health_dir
        )

        # Health tracking
        self.health_checks: dict[str, HealthCheck] = {}
        self.start_time = datetime.now()
        self.health_history: list[HealthReport] = []
        self.alert_thresholds = {
            'consecutive_failures_warning': 3,
            'consecutive_failures_critical': 5,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0
        }

        # Monitoring task
        self.monitoring_task: asyncio.Task | None = None
        self.is_monitoring = False

        # Initialize default health checks
        self._initialize_default_health_checks()

        self.logger.info("SystemHealthMonitor initialized",
                        session_id=session_id,
                        check_interval=check_interval,
                        initial_checks=len(self.health_checks))

    def _initialize_default_health_checks(self) -> None:
        """Initialize default health checks for system monitoring."""

        # Memory health check
        self.add_health_check(
            name="memory_usage",
            description="System memory usage monitoring",
            check_function=self._check_memory_health,
            interval_seconds=30,
            timeout_seconds=10
        )

        # CPU health check
        self.add_health_check(
            name="cpu_usage",
            description="System CPU usage monitoring",
            check_function=self._check_cpu_health,
            interval_seconds=30,
            timeout_seconds=10
        )

        # Disk health check
        self.add_health_check(
            name="disk_usage",
            description="Disk space monitoring",
            check_function=self._check_disk_health,
            interval_seconds=60,
            timeout_seconds=15
        )

        # Process health check
        self.add_health_check(
            name="process_health",
            description="System process monitoring",
            check_function=self._check_process_health,
            interval_seconds=60,
            timeout_seconds=15
        )

        # Logging system health check
        self.add_health_check(
            name="logging_system",
            description="Logging system functionality",
            check_function=self._check_logging_health,
            interval_seconds=120,
            timeout_seconds=10
        )

    def add_health_check(self,
                        name: str,
                        description: str,
                        check_function: Callable,
                        interval_seconds: int = 60,
                        timeout_seconds: int = 30) -> None:
        """
        Add a new health check.

        Args:
            name: Unique name for the health check
            description: Description of what the check monitors
            check_function: Async function that performs the health check
            interval_seconds: How often to run this check
            timeout_seconds: Timeout for the check execution
        """
        health_check = HealthCheck(
            name=name,
            description=description,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds
        )

        self.health_checks[name] = health_check
        self.logger.info(f"Added health check: {name}",
                        check_name=name,
                        interval_seconds=interval_seconds,
                        timeout_seconds=timeout_seconds)

    def remove_health_check(self, name: str) -> None:
        """
        Remove a health check.

        Args:
            name: Name of the health check to remove
        """
        if name in self.health_checks:
            del self.health_checks[name]
            self.logger.info(f"Removed health check: {name}",
                            check_name=name)

    def enable_health_check(self, name: str) -> None:
        """Enable a health check."""
        if name in self.health_checks:
            self.health_checks[name].enabled = True
            self.logger.info(f"Enabled health check: {name}",
                            check_name=name)

    def disable_health_check(self, name: str) -> None:
        """Disable a health check."""
        if name in self.health_checks:
            self.health_checks[name].enabled = False
            self.logger.info(f"Disabled health check: {name}",
                            check_name=name)

    async def start_monitoring(self) -> None:
        """Start the health monitoring background task."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("System health monitoring started",
                        session_id=self.session_id)

    async def stop_monitoring(self) -> None:
        """Stop the health monitoring background task."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("System health monitoring stopped",
                        session_id=self.session_id)

    async def run_health_check(self, name: str) -> dict[str, Any]:
        """
        Run a specific health check manually.

        Args:
            name: Name of the health check to run

        Returns:
            Health check result
        """
        if name not in self.health_checks:
            return {
                'name': name,
                'status': HealthStatus.UNKNOWN.value,
                'error': f"Health check '{name}' not found",
                'timestamp': datetime.now().isoformat()
            }

        health_check = self.health_checks[name]
        return await self._execute_health_check(health_check)

    async def run_all_health_checks(self) -> dict[str, Any]:
        """
        Run all enabled health checks.

        Returns:
            Comprehensive health check results
        """
        results = {}
        tasks = []

        for name, health_check in self.health_checks.items():
            if health_check.enabled:
                tasks.append(self._execute_health_check(health_check))

        if tasks:
            check_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(check_results):
                if isinstance(result, Exception):
                    check_name = list(self.health_checks.keys())[i]
                    results[check_name] = {
                        'name': check_name,
                        'status': HealthStatus.CRITICAL.value,
                        'error': str(result),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    results[result['name']] = result

        return results

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs health checks periodically."""
        while self.is_monitoring:
            try:
                # Run all enabled health checks
                await self.run_all_health_checks()

                # Generate health report
                report = await self.generate_health_report()
                if report:
                    self.health_history.append(report)

                    # Keep only last 24 hours of reports
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    self.health_history = [
                        r for r in self.health_history
                        if r.timestamp > cutoff_time
                    ]

                    # Log health status
                    self._log_health_status(report)

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}",
                                session_id=self.session_id)
                await asyncio.sleep(self.check_interval)

    async def _execute_health_check(self, health_check: HealthCheck) -> dict[str, Any]:
        """Execute a single health check."""
        start_time = time.time()
        result = {
            'name': health_check.name,
            'description': health_check.description,
            'timestamp': datetime.now().isoformat(),
            'execution_time': 0.0,
            'status': HealthStatus.UNKNOWN.value,
            'details': {},
            'error': None
        }

        try:
            # Execute the health check with timeout
            check_result = await asyncio.wait_for(
                health_check.check_function(),
                timeout=health_check.timeout_seconds
            )

            result['details'] = check_result
            result['execution_time'] = time.time() - start_time

            # Determine health status
            if isinstance(check_result, dict):
                status = check_result.get('status', HealthStatus.UNKNOWN.value)
                if isinstance(status, str):
                    result['status'] = status
                else:
                    result['status'] = status.value if hasattr(status, 'value') else str(status)
            else:
                result['status'] = HealthStatus.HEALTHY.value
                result['details'] = {'result': check_result}

            # Update health check state
            health_check.last_check = datetime.now()
            health_check.last_status = HealthStatus(result['status'])
            health_check.last_result = check_result

            if result['status'] == HealthStatus.HEALTHY.value:
                health_check.consecutive_failures = 0
            else:
                health_check.consecutive_failures += 1

        except asyncio.TimeoutError:
            result['status'] = HealthStatus.CRITICAL.value
            result['error'] = f"Health check timed out after {health_check.timeout_seconds} seconds"
            result['execution_time'] = time.time() - start_time
            health_check.consecutive_failures += 1

        except Exception as e:
            result['status'] = HealthStatus.CRITICAL.value
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            health_check.consecutive_failures += 1

        return result

    async def _check_memory_health(self) -> dict[str, Any]:
        """Check system memory health."""
        import psutil

        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        status = HealthStatus.HEALTHY
        if memory_percent >= self.alert_thresholds['memory_critical']:
            status = HealthStatus.CRITICAL
        elif memory_percent >= self.alert_thresholds['memory_warning']:
            status = HealthStatus.WARNING

        return {
            'status': status.value,
            'memory_percent': memory_percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'thresholds': {
                'warning': self.alert_thresholds['memory_warning'],
                'critical': self.alert_thresholds['memory_critical']
            }
        }

    async def _check_cpu_health(self) -> dict[str, Any]:
        """Check system CPU health."""
        import psutil

        cpu_percent = psutil.cpu_percent(interval=1)
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]

        status = HealthStatus.HEALTHY
        if cpu_percent >= self.alert_thresholds['cpu_critical']:
            status = HealthStatus.CRITICAL
        elif cpu_percent >= self.alert_thresholds['cpu_warning']:
            status = HealthStatus.WARNING

        return {
            'status': status.value,
            'cpu_percent': cpu_percent,
            'load_average': load_avg,
            'cpu_count': psutil.cpu_count(),
            'thresholds': {
                'warning': self.alert_thresholds['cpu_warning'],
                'critical': self.alert_thresholds['cpu_critical']
            }
        }

    async def _check_disk_health(self) -> dict[str, Any]:
        """Check disk space health."""
        import psutil

        disk_usage = psutil.disk_usage('/')
        disk_percent = (disk_usage.used / disk_usage.total) * 100

        status = HealthStatus.HEALTHY
        if disk_percent >= self.alert_thresholds['disk_critical']:
            status = HealthStatus.CRITICAL
        elif disk_percent >= self.alert_thresholds['disk_warning']:
            status = HealthStatus.WARNING

        return {
            'status': status.value,
            'disk_percent': disk_percent,
            'disk_free_gb': disk_usage.free / (1024**3),
            'disk_used_gb': disk_usage.used / (1024**3),
            'disk_total_gb': disk_usage.total / (1024**3),
            'thresholds': {
                'warning': self.alert_thresholds['disk_warning'],
                'critical': self.alert_thresholds['disk_critical']
            }
        }

    async def _check_process_health(self) -> dict[str, Any]:
        """Check system process health."""
        import psutil

        processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
        total_processes = len(processes)

        # Count high resource usage processes
        high_cpu_processes = len([p for p in processes if p.info['cpu_percent'] and p.info['cpu_percent'] > 80])
        high_memory_processes = len([p for p in processes if p.info['memory_percent'] and p.info['memory_percent'] > 80])

        status = HealthStatus.HEALTHY
        if total_processes > 500 or high_cpu_processes > 10 or high_memory_processes > 10:
            status = HealthStatus.WARNING
        if total_processes > 1000 or high_cpu_processes > 20 or high_memory_processes > 20:
            status = HealthStatus.CRITICAL

        return {
            'status': status.value,
            'total_processes': total_processes,
            'high_cpu_processes': high_cpu_processes,
            'high_memory_processes': high_memory_processes,
            'top_cpu_processes': sorted(
                [p.info for p in processes if p.info['cpu_percent']],
                key=lambda x: x['cpu_percent'] or 0,
                reverse=True
            )[:5]
        }

    async def _check_logging_health(self) -> dict[str, Any]:
        """Check logging system health."""
        try:
            # Test logging functionality
            test_log_file = self.health_dir / "health_test.log"
            test_message = f"Health check test - {datetime.now().isoformat()}"

            # Write test log entry
            with open(test_log_file, 'w') as f:
                f.write(test_message + '\n')

            # Read it back
            with open(test_log_file) as f:
                content = f.read().strip()

            # Clean up
            test_log_file.unlink()

            success = content == test_message

            return {
                'status': HealthStatus.HEALTHY.value if success else HealthStatus.CRITICAL.value,
                'logging_functional': success,
                'log_directory_accessible': True,
                'test_message_written': success,
                'test_message_read': success
            }

        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'logging_functional': False,
                'log_directory_accessible': False,
                'error': str(e)
            }

    async def generate_health_report(self) -> HealthReport | None:
        """Generate a comprehensive health report."""
        if not self.health_checks:
            return None

        # Run all health checks
        check_results = await self.run_all_health_checks()

        # Count by status
        healthy_count = len([r for r in check_results.values() if r.get('status') == HealthStatus.HEALTHY.value])
        warning_count = len([r for r in check_results.values() if r.get('status') == HealthStatus.WARNING.value])
        critical_count = len([r for r in check_results.values() if r.get('status') == HealthStatus.CRITICAL.value])
        unknown_count = len([r for r in check_results.values() if r.get('status') == HealthStatus.UNKNOWN.value])

        # Determine overall status
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        elif healthy_count > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        # Get system resources
        system_resources = await self._get_system_resources()

        # Generate alerts
        alerts = self._generate_health_alerts(check_results)

        # Generate recommendations
        recommendations = self._generate_health_recommendations(check_results, alerts)

        report = HealthReport(
            timestamp=datetime.now(),
            overall_status=overall_status,
            system_uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
            total_checks=len(check_results),
            healthy_checks=healthy_count,
            warning_checks=warning_count,
            critical_checks=critical_count,
            unknown_checks=unknown_count,
            check_results=list(check_results.values()),
            system_resources=system_resources,
            alerts=alerts,
            recommendations=recommendations
        )

        return report

    async def _get_system_resources(self) -> dict[str, Any]:
        """Get current system resource information."""
        try:
            import psutil

            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
            }
        except Exception as e:
            return {'error': f"Failed to get system resources: {e}"}

    def _generate_health_alerts(self, check_results: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate health alerts based on check results."""
        alerts = []

        for result in check_results.values():
            status = result.get('status')
            if status in [HealthStatus.WARNING.value, HealthStatus.CRITICAL.value]:
                alert = {
                    'check_name': result['name'],
                    'severity': status,
                    'message': f"Health check '{result['name']}' reported {status}",
                    'timestamp': result['timestamp'],
                    'details': result.get('details', {}),
                    'consecutive_failures': self.health_checks.get(result['name'], HealthCheck()).consecutive_failures
                }
                alerts.append(alert)

        return alerts

    def _generate_health_recommendations(self,
                                       check_results: dict[str, Any],
                                       alerts: list[dict[str, Any]]) -> list[str]:
        """Generate health recommendations based on check results."""
        recommendations = []

        for result in check_results.values():
            if result.get('status') == HealthStatus.CRITICAL.value:
                if 'memory' in result['name'].lower():
                    recommendations.append("Consider freeing up memory or increasing system RAM")
                elif 'cpu' in result['name'].lower():
                    recommendations.append("High CPU usage detected - consider optimizing processes or adding CPU resources")
                elif 'disk' in result['name'].lower():
                    recommendations.append("Low disk space - consider cleaning up files or expanding storage")
                elif 'logging' in result['name'].lower():
                    recommendations.append("Logging system issues detected - check log directory permissions")

        # Add general recommendations
        if len(alerts) > 3:
            recommendations.append("Multiple health issues detected - consider system maintenance")

        if not recommendations:
            recommendations.append("System health is good - no immediate action required")

        return recommendations

    def _log_health_status(self, report: HealthReport) -> None:
        """Log health status summary."""
        status_emoji = {
            HealthStatus.HEALTHY: "🟢",
            HealthStatus.WARNING: "🟡",
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.UNKNOWN: "⚪"
        }

        emoji = status_emoji.get(report.overall_status, "⚪")

        log_message = f"Health Report - {report.overall_status.upper()}: {report.healthy_checks}/{report.total_checks} checks healthy"

        log_data = {
            'session_id': self.session_id,
            'overall_status': report.overall_status.value,
            'healthy_checks': report.healthy_checks,
            'warning_checks': report.warning_checks,
            'critical_checks': report.critical_checks,
            'uptime_seconds': report.system_uptime_seconds,
            'alerts_count': len(report.alerts)
        }

        if report.overall_status == HealthStatus.CRITICAL:
            self.logger.error(f"{emoji} {log_message}", **log_data)
        elif report.overall_status == HealthStatus.WARNING:
            self.logger.warning(f"{emoji} {log_message}", **log_data)
        else:
            self.logger.info(f"{emoji} {log_message}", **log_data)

    def get_health_summary(self) -> dict[str, Any]:
        """Get current health summary."""
        if not self.health_checks:
            return {
                'status': 'unknown',
                'message': 'No health checks configured'
            }

        # Get latest report or create current summary
        if self.health_history:
            latest_report = self.health_history[-1]
            return {
                'timestamp': latest_report.timestamp.isoformat(),
                'overall_status': latest_report.overall_status.value,
                'uptime_seconds': latest_report.system_uptime_seconds,
                'healthy_checks': latest_report.healthy_checks,
                'warning_checks': latest_report.warning_checks,
                'critical_checks': latest_report.critical_checks,
                'total_checks': latest_report.total_checks,
                'alerts_count': len(latest_report.alerts),
                'recommendations_count': len(latest_report.recommendations)
            }
        else:
            # Return current check status
            enabled_checks = [c for c in self.health_checks.values() if c.enabled]
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'unknown',
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'enabled_checks': len(enabled_checks),
                'total_checks': len(self.health_checks),
                'monitoring_active': self.is_monitoring
            }

    def export_health_report(self, file_path: str | None = None) -> str:
        """
        Export health data to a JSON file.

        Args:
            file_path: Optional custom file path

        Returns:
            Path to the exported file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self.health_dir / f"health_report_{self.session_id}_{timestamp}.json")

        export_data = {
            'session_id': self.session_id,
            'export_timestamp': datetime.now().isoformat(),
            'start_time': self.start_time.isoformat(),
            'health_summary': self.get_health_summary(),
            'health_checks': {name: asdict(check) for name, check in self.health_checks.items()},
            'health_history': [asdict(report) for report in self.health_history[-10:]],  # Last 10 reports
            'alert_thresholds': self.alert_thresholds
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Health report exported to: {file_path}",
                        session_id=self.session_id,
                        file_path=file_path)

        return file_path

    async def cleanup(self) -> None:
        """Clean up health monitoring resources."""
        await self.stop_monitoring()

        # Export final health report
        try:
            self.export_health_report()
        except Exception as e:
            self.logger.error(f"Error exporting final health report: {e}",
                            session_id=self.session_id)

        self.logger.info("SystemHealthMonitor cleanup completed",
                        session_id=self.session_id)
