"""
Performance Monitor for real-time performance tracking and analysis.

This module provides high-level performance monitoring capabilities that integrate
with the metrics collector to provide real-time insights into system performance.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .metrics_collector import MetricsCollector


@dataclass
class PerformanceThreshold:
    """Configuration for performance monitoring thresholds."""
    name: str
    warning_threshold: float
    critical_threshold: float
    unit: str
    description: str


@dataclass
class PerformanceAlert:
    """Alert generated when performance thresholds are exceeded."""
    alert_id: str
    timestamp: datetime
    alert_type: str  # 'warning', 'critical'
    metric_name: str
    current_value: float
    threshold_value: float
    agent_name: str | None
    tool_name: str | None
    workflow_id: str | None
    message: str
    metadata: dict[str, Any]


class PerformanceMonitor:
    """High-level performance monitoring for the multi-agent system."""

    def __init__(self,
                 metrics_collector: MetricsCollector,
                 alert_cooldown_minutes: int = 5):
        """
        Initialize the performance monitor.

        Args:
            metrics_collector: MetricsCollector instance for data source
            alert_cooldown_minutes: Minutes to wait between similar alerts
        """
        self.metrics_collector = metrics_collector
        self.alert_cooldown_minutes = alert_cooldown_minutes

        # Performance tracking
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.performance_history: list[dict[str, Any]] = []
        self.alerts: list[PerformanceAlert] = []
        self.alert_cooldowns: dict[str, datetime] = {}

        # Performance thresholds
        self.thresholds = {
            'tool_execution_time': PerformanceThreshold(
                name='tool_execution_time',
                warning_threshold=30.0,
                critical_threshold=60.0,
                unit='seconds',
                description='Tool execution time'
            ),
            'workflow_stage_duration': PerformanceThreshold(
                name='workflow_stage_duration',
                warning_threshold=300.0,
                critical_threshold=600.0,
                unit='seconds',
                description='Workflow stage duration'
            ),
            'agent_error_rate': PerformanceThreshold(
                name='agent_error_rate',
                warning_threshold=0.1,  # 10%
                critical_threshold=0.2,  # 20%
                unit='ratio',
                description='Agent error rate'
            ),
            'memory_usage': PerformanceThreshold(
                name='memory_usage',
                warning_threshold=80.0,
                critical_threshold=95.0,
                unit='percent',
                description='System memory usage'
            ),
            'cpu_usage': PerformanceThreshold(
                name='cpu_usage',
                warning_threshold=80.0,
                critical_threshold=95.0,
                unit='percent',
                description='System CPU usage'
            )
        }

        # Monitoring task
        self.monitoring_task: asyncio.Task | None = None
        self.is_monitoring = False

        self.metrics_collector.logger.info("PerformanceMonitor initialized",
                                         alert_cooldown_minutes=alert_cooldown_minutes,
                                         thresholds_count=len(self.thresholds))

    async def start_monitoring(self) -> None:
        """Start the performance monitoring background task."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.metrics_collector.logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the performance monitoring background task."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.metrics_collector.logger.info("Performance monitoring stopped")

    @asynccontextmanager
    async def monitor_tool_execution(self,
                                   tool_name: str,
                                   agent_name: str,
                                   input_size: int = 0) -> AsyncGenerator[dict[str, Any], None]:
        """
        Context manager to monitor tool execution performance.

        Args:
            tool_name: Name of the tool being executed
            agent_name: Name of the agent executing the tool
            input_size: Size of input data

        Yields:
            Dictionary containing execution context data
        """
        execution_id = f"{tool_name}_{agent_name}_{int(time.time() * 1000)}"
        start_time = time.time()

        context = {
            'execution_id': execution_id,
            'tool_name': tool_name,
            'agent_name': agent_name,
            'start_time': start_time,
            'input_size': input_size
        }

        try:
            yield context
            execution_time = time.time() - start_time

            # Record successful execution
            self.metrics_collector.record_tool_metric(
                tool_name=tool_name,
                agent_name=agent_name,
                execution_time=execution_time,
                success=True,
                input_size=input_size
            )

            # Check for performance alerts
            await self._check_tool_performance_alerts(
                tool_name, agent_name, execution_time, True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_type = type(e).__name__

            # Record failed execution
            self.metrics_collector.record_tool_metric(
                tool_name=tool_name,
                agent_name=agent_name,
                execution_time=execution_time,
                success=False,
                input_size=input_size,
                error_type=error_type
            )

            # Check for performance alerts
            await self._check_tool_performance_alerts(
                tool_name, agent_name, execution_time, False, error_type
            )

            raise

    @asynccontextmanager
    async def monitor_workflow_stage(self,
                                   workflow_id: str,
                                   stage_name: str,
                                   agents_involved: list[str]) -> AsyncGenerator[dict[str, Any], None]:
        """
        Context manager to monitor workflow stage performance.

        Args:
            workflow_id: Unique workflow identifier
            stage_name: Name of the workflow stage
            agents_involved: List of agents involved in this stage

        Yields:
            Dictionary containing stage context data
        """
        start_time = time.time()
        total_start_time = self._get_workflow_start_time(workflow_id) or start_time

        context = {
            'workflow_id': workflow_id,
            'stage_name': stage_name,
            'agents_involved': agents_involved,
            'start_time': start_time,
            'total_start_time': total_start_time
        }

        try:
            yield context
            stage_duration = time.time() - start_time
            total_duration = time.time() - total_start_time

            # Record successful stage completion
            self.metrics_collector.record_workflow_metric(
                workflow_id=workflow_id,
                stage_name=stage_name,
                stage_duration=stage_duration,
                total_duration=total_duration,
                success=True,
                agents_involved=agents_involved,
                tools_used=[]  # Tools would be tracked separately
            )

            # Check for performance alerts
            await self._check_workflow_performance_alerts(
                workflow_id, stage_name, stage_duration, True
            )

        except Exception:
            stage_duration = time.time() - start_time
            total_duration = time.time() - total_start_time

            # Record failed stage completion
            self.metrics_collector.record_workflow_metric(
                workflow_id=workflow_id,
                stage_name=stage_name,
                stage_duration=stage_duration,
                total_duration=total_duration,
                success=False,
                agents_involved=agents_involved,
                tools_used=[]
            )

            # Check for performance alerts
            await self._check_workflow_performance_alerts(
                workflow_id, stage_name, stage_duration, False
            )

            raise

    def record_agent_activity(self,
                            agent_name: str,
                            activity_type: str,
                            activity_name: str,
                            value: float,
                            unit: str = "count",
                            metadata: dict[str, Any] | None = None) -> None:
        """
        Record an agent activity for performance tracking.

        Args:
            agent_name: Name of the agent
            activity_type: Type of activity (performance, usage, error, resource)
            activity_name: Name of the activity
            value: Activity value
            unit: Unit of measurement
            metadata: Additional activity metadata
        """
        self.metrics_collector.record_agent_metric(
            agent_name=agent_name,
            metric_type=activity_type,
            metric_name=activity_name,
            value=value,
            unit=unit,
            metadata=metadata
        )

        # Check for performance alerts
        if activity_type == "error" and value > 0:
            asyncio.create_task(self._check_agent_error_alerts(
                agent_name, activity_name, value
            ))

    def start_session_tracking(self, session_id: str, session_data: dict[str, Any]) -> None:
        """
        Start tracking a new session.

        Args:
            session_id: Session identifier
            session_data: Initial session data
        """
        self.active_sessions[session_id] = {
            'session_id': session_id,
            'start_time': datetime.now(),
            'session_data': session_data,
            'activities': [],
            'performance_metrics': {}
        }

        self.metrics_collector.logger.info(f"Started tracking session: {session_id}",
                                         session_id=session_id)

    def end_session_tracking(self, session_id: str, summary_data: dict[str, Any]) -> dict[str, Any]:
        """
        End tracking for a session and generate performance summary.

        Args:
            session_id: Session identifier
            summary_data: Final session summary data

        Returns:
            Session performance summary
        """
        if session_id not in self.active_sessions:
            return {}

        session = self.active_sessions[session_id]
        end_time = datetime.now()
        duration = (end_time - session['start_time']).total_seconds()

        session_summary = {
            'session_id': session_id,
            'start_time': session['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_activities': len(session['activities']),
            'session_data': session['session_data'],
            'summary_data': summary_data,
            'performance_metrics': session['performance_metrics']
        }

        # Add to performance history
        self.performance_history.append(session_summary)

        # Remove from active sessions
        del self.active_sessions[session_id]

        self.metrics_collector.logger.info(f"Ended tracking session: {session_id}",
                                         session_id=session_id,
                                         duration_seconds=duration,
                                         total_activities=len(session['activities']))

        return session_summary

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs periodically."""
        while self.is_monitoring:
            try:
                await self._check_system_performance()
                await self._cleanup_old_data()
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.metrics_collector.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _check_system_performance(self) -> None:
        """Check system performance metrics and generate alerts."""
        system_summary = self.metrics_collector.get_system_summary()

        if not system_summary:
            return

        current = system_summary.get('current', {})
        if not current:
            return

        # Check CPU usage
        cpu_percent = current.get('cpu_percent', 0)
        await self._check_threshold_alert(
            'cpu_usage', cpu_percent, 'system', None, None
        )

        # Check memory usage
        memory_percent = current.get('memory_percent', 0)
        await self._check_threshold_alert(
            'memory_usage', memory_percent, 'system', None, None
        )

    async def _check_tool_performance_alerts(self,
                                           tool_name: str,
                                           agent_name: str,
                                           execution_time: float,
                                           success: bool,
                                           error_type: str | None = None) -> None:
        """Check for tool performance alerts."""
        if not success:
            # Generate alert for tool failure
            await self._generate_alert(
                alert_type='critical',
                metric_name='tool_failure',
                current_value=1.0,
                threshold_value=0.0,
                agent_name=agent_name,
                tool_name=tool_name,
                message=f"Tool execution failed: {tool_name} ({error_type})",
                metadata={'error_type': error_type, 'execution_time': execution_time}
            )
        else:
            # Check execution time threshold
            await self._check_threshold_alert(
                'tool_execution_time', execution_time, agent_name, tool_name, None
            )

    async def _check_workflow_performance_alerts(self,
                                                workflow_id: str,
                                                stage_name: str,
                                                stage_duration: float,
                                                success: bool) -> None:
        """Check for workflow performance alerts."""
        if not success:
            # Generate alert for workflow stage failure
            await self._generate_alert(
                alert_type='critical',
                metric_name='workflow_stage_failure',
                current_value=1.0,
                threshold_value=0.0,
                workflow_id=workflow_id,
                message=f"Workflow stage failed: {stage_name}",
                metadata={'stage_name': stage_name, 'stage_duration': stage_duration}
            )
        else:
            # Check stage duration threshold
            await self._check_threshold_alert(
                'workflow_stage_duration', stage_duration, None, None, workflow_id,
                metadata={'stage_name': stage_name}
            )

    async def _check_agent_error_alerts(self,
                                       agent_name: str,
                                       activity_name: str,
                                       error_count: float) -> None:
        """Check for agent error rate alerts."""
        # Get recent error metrics for this agent
        agent_summary = self.metrics_collector.get_agent_summary(agent_name)

        # Calculate error rate (this is simplified - in practice you'd want
        # to calculate this over a meaningful time window)
        total_metrics = agent_summary.get('total_metrics', 0)
        error_metrics = len([m for m in agent_summary.get('metrics_by_type', {}).get('error', [])])

        if total_metrics > 0:
            error_rate = error_metrics / total_metrics
            await self._check_threshold_alert(
                'agent_error_rate', error_rate, agent_name, None, None,
                metadata={'activity_name': activity_name, 'error_count': error_count}
            )

    async def _check_threshold_alert(self,
                                   threshold_name: str,
                                   current_value: float,
                                   agent_name: str | None,
                                   tool_name: str | None,
                                   workflow_id: str | None,
                                   metadata: dict[str, Any] | None = None) -> None:
        """Check if a value exceeds its threshold and generate alert if needed."""
        if threshold_name not in self.thresholds:
            return

        threshold = self.thresholds[threshold_name]
        alert_key = f"{threshold_name}_{agent_name}_{tool_name}_{workflow_id}"

        # Check if we're in cooldown period
        if (alert_key in self.alert_cooldowns and
            datetime.now() < self.alert_cooldowns[alert_key]):
            return

        alert_type = None
        threshold_value = None

        if current_value >= threshold.critical_threshold:
            alert_type = 'critical'
            threshold_value = threshold.critical_threshold
        elif current_value >= threshold.warning_threshold:
            alert_type = 'warning'
            threshold_value = threshold.warning_threshold

        if alert_type:
            await self._generate_alert(
                alert_type=alert_type,
                metric_name=threshold_name,
                current_value=current_value,
                threshold_value=threshold_value,
                agent_name=agent_name,
                tool_name=tool_name,
                workflow_id=workflow_id,
                message=f"{threshold.description} {alert_type}: {current_value:.2f} {threshold.unit} (threshold: {threshold_value:.2f} {threshold.unit})",
                metadata=metadata or {}
            )

            # Set cooldown
            self.alert_cooldowns[alert_key] = datetime.now() + timedelta(minutes=self.alert_cooldown_minutes)

    async def _generate_alert(self,
                            alert_type: str,
                            metric_name: str,
                            current_value: float,
                            threshold_value: float,
                            agent_name: str | None,
                            tool_name: str | None,
                            workflow_id: str | None,
                            message: str,
                            metadata: dict[str, Any] | None = None) -> None:
        """Generate and store a performance alert."""
        alert = PerformanceAlert(
            alert_id=f"{metric_name}_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            alert_type=alert_type,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            agent_name=agent_name,
            tool_name=tool_name,
            workflow_id=workflow_id,
            message=message,
            metadata=metadata or {}
        )

        self.alerts.append(alert)

        # Log the alert
        if alert_type == 'critical':
            self.metrics_collector.logger.error(f"Performance Alert: {message}",
                                              **{k: v for k, v in alert.__dict__.items() if k != 'message'})
        else:
            self.metrics_collector.logger.warning(f"Performance Alert: {message}",
                                                **{k: v for k, v in alert.__dict__.items() if k != 'message'})

    def _get_workflow_start_time(self, workflow_id: str) -> float | None:
        """Get the start time for a workflow from performance history."""
        for session in self.active_sessions.values():
            if workflow_id in str(session.get('session_data', {})):
                return session['start_time'].timestamp()
        return None

    async def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        cutoff_time = datetime.now() - timedelta(hours=24)

        # Clean up old alerts
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]

        # Clean up old cooldowns
        self.alert_cooldowns = {
            key: time for key, time in self.alert_cooldowns.items()
            if time > datetime.now()
        }

        # Clean up old performance history (keep last 100 entries)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        recent_time = datetime.now() - timedelta(hours=1)

        # Get recent alerts
        recent_alerts = [alert for alert in self.alerts if alert.timestamp > recent_time]

        # Get system metrics
        system_summary = self.metrics_collector.get_system_summary()

        # Get active sessions
        active_sessions_count = len(self.active_sessions)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': 'active' if self.is_monitoring else 'inactive',
            'active_sessions': active_sessions_count,
            'recent_alerts': {
                'total': len(recent_alerts),
                'critical': len([a for a in recent_alerts if a.alert_type == 'critical']),
                'warning': len([a for a in recent_alerts if a.alert_type == 'warning'])
            },
            'system_status': system_summary.get('current', {}),
            'performance_metrics': {
                'total_agent_metrics': len(self.metrics_collector.agent_metrics),
                'total_tool_metrics': len(self.metrics_collector.tool_metrics),
                'total_workflow_metrics': len(self.metrics_collector.workflow_metrics),
                'total_system_metrics': len(self.metrics_collector.system_metrics)
            }
        }

        return summary

    def get_alerts_summary(self,
                          alert_type: str | None = None,
                          hours_back: int = 24) -> dict[str, Any]:
        """
        Get summary of performance alerts.

        Args:
            alert_type: Filter by alert type ('warning', 'critical')
            hours_back: Hours to look back for alerts

        Returns:
            Dictionary containing alert summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        filtered_alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]

        if alert_type:
            filtered_alerts = [alert for alert in filtered_alerts if alert.alert_type == alert_type]

        # Group alerts by type
        alerts_by_type = {}
        alerts_by_metric = {}
        alerts_by_agent = {}

        for alert in filtered_alerts:
            # Group by type
            if alert.alert_type not in alerts_by_type:
                alerts_by_type[alert.alert_type] = []
            alerts_by_type[alert.alert_type].append(alert)

            # Group by metric
            if alert.metric_name not in alerts_by_metric:
                alerts_by_metric[alert.metric_name] = []
            alerts_by_metric[alert.metric_name].append(alert)

            # Group by agent
            if alert.agent_name:
                if alert.agent_name not in alerts_by_agent:
                    alerts_by_agent[alert.agent_name] = []
                alerts_by_agent[alert.agent_name].append(alert)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'hours_back': hours_back,
            'alert_type_filter': alert_type,
            'total_alerts': len(filtered_alerts),
            'alerts_by_type': {k: len(v) for k, v in alerts_by_type.items()},
            'alerts_by_metric': {k: len(v) for k, v in alerts_by_metric.items()},
            'alerts_by_agent': {k: len(v) for k, v in alerts_by_agent.items()},
            'most_recent_alert': filtered_alerts[-1].__dict__ if filtered_alerts else None
        }

        return summary

    async def cleanup(self) -> None:
        """Clean up performance monitoring resources."""
        await self.stop_monitoring()

        # Export final performance data
        try:
            summary = self.get_performance_summary()
            self.metrics_collector.logger.info("Final performance summary",
                                            **summary)
        except Exception as e:
            self.metrics_collector.logger.error(f"Error generating final performance summary: {e}")

        self.metrics_collector.logger.info("PerformanceMonitor cleanup completed")
