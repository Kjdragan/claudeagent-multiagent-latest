"""
Metrics Collector for real-time performance monitoring.

This module provides comprehensive metrics collection for the multi-agent system,
including agent performance, tool usage, system resources, and user interactions.
"""

import asyncio
import json
import os
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_logging import StructuredLogger


@dataclass
class AgentMetric:
    """Metric data for agent performance."""
    agent_name: str
    session_id: str
    timestamp: datetime
    metric_type: str  # 'performance', 'usage', 'error', 'resource'
    metric_name: str
    value: float
    unit: str
    metadata: dict[str, Any]


@dataclass
class SystemMetric:
    """Metric data for system resources."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io: dict[str, int]
    process_count: int
    active_agents: int


@dataclass
class ToolMetric:
    """Metric data for tool usage."""
    tool_name: str
    agent_name: str
    session_id: str
    timestamp: datetime
    execution_time: float
    success: bool
    input_size: int
    output_size: int
    error_type: str | None = None


@dataclass
class WorkflowMetric:
    """Metric data for workflow performance."""
    workflow_id: str
    session_id: str
    timestamp: datetime
    stage_name: str
    stage_duration: float
    total_duration: float
    success: bool
    agents_involved: list[str]
    tools_used: list[str]


class MetricsCollector:
    """Collects and manages performance metrics for the multi-agent system."""

    def __init__(self,
                 session_id: str,
                 metrics_dir: str = "metrics",
                 retention_hours: int = 24,
                 collection_interval: int = 30):
        """
        Initialize the metrics collector.

        Args:
            session_id: Session identifier for metric grouping
            metrics_dir: Directory to store metric data
            retention_hours: Hours to retain metrics data
            collection_interval: Seconds between system metric collections
        """
        self.session_id = session_id
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.retention_hours = retention_hours
        self.collection_interval = collection_interval

        # Initialize structured logger for metrics
        self.logger = StructuredLogger(
            name="metrics_collector",
            log_dir=self.metrics_dir
        )

        # Metric storage
        self.agent_metrics: deque[AgentMetric] = deque(maxlen=10000)
        self.system_metrics: deque[SystemMetric] = deque(maxlen=10000)
        self.tool_metrics: deque[ToolMetric] = deque(maxlen=10000)
        self.workflow_metrics: deque[WorkflowMetric] = deque(maxlen=10000)

        # Aggregated metrics for real-time monitoring
        self.agent_performance: dict[str, dict[str, Any]] = defaultdict(dict)
        self.tool_performance: dict[str, dict[str, Any]] = defaultdict(dict)
        self.workflow_performance: dict[str, dict[str, Any]] = defaultdict(dict)

        # System monitoring task
        self.system_monitoring_task: asyncio.Task | None = None
        self.is_monitoring = False

        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'tool_execution_warning': 30.0,
            'tool_execution_critical': 60.0,
            'workflow_stage_warning': 300.0,
            'workflow_stage_critical': 600.0
        }

        self.logger.info("MetricsCollector initialized",
                        session_id=session_id,
                        metrics_dir=str(self.metrics_dir),
                        retention_hours=retention_hours,
                        collection_interval=collection_interval)

    async def start_monitoring(self) -> None:
        """Start the metrics collection background task."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.system_monitoring_task = asyncio.create_task(self._collect_system_metrics())
        self.logger.info("Metrics monitoring started",
                        session_id=self.session_id,
                        collection_interval=self.collection_interval)

    async def stop_monitoring(self) -> None:
        """Stop the metrics collection background task."""
        self.is_monitoring = False
        if self.system_monitoring_task:
            self.system_monitoring_task.cancel()
            try:
                await self.system_monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Metrics monitoring stopped",
                        session_id=self.session_id)

    def record_agent_metric(self,
                           agent_name: str,
                           metric_type: str,
                           metric_name: str,
                           value: float,
                           unit: str = "count",
                           metadata: dict[str, Any] | None = None) -> None:
        """
        Record an agent performance metric.

        Args:
            agent_name: Name of the agent
            metric_type: Type of metric (performance, usage, error, resource)
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            metadata: Additional metric metadata
        """
        metric = AgentMetric(
            agent_name=agent_name,
            session_id=self.session_id,
            timestamp=datetime.now(),
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )

        self.agent_metrics.append(metric)
        self._update_agent_performance_aggregates(metric)

        # Check for performance alerts
        self._check_agent_metric_alerts(metric)

        self.logger.debug(f"Agent metric recorded: {metric_name}",
                         agent_name=agent_name,
                         metric_type=metric_type,
                         metric_name=metric_name,
                         value=value,
                         unit=unit)

    def record_tool_metric(self,
                          tool_name: str,
                          agent_name: str,
                          execution_time: float,
                          success: bool,
                          input_size: int = 0,
                          output_size: int = 0,
                          error_type: str | None = None) -> None:
        """
        Record a tool usage metric.

        Args:
            tool_name: Name of the tool used
            agent_name: Name of the agent using the tool
            execution_time: Time taken to execute the tool
            success: Whether the tool execution was successful
            input_size: Size of input data
            output_size: Size of output data
            error_type: Type of error if execution failed
        """
        metric = ToolMetric(
            tool_name=tool_name,
            agent_name=agent_name,
            session_id=self.session_id,
            timestamp=datetime.now(),
            execution_time=execution_time,
            success=success,
            input_size=input_size,
            output_size=output_size,
            error_type=error_type
        )

        self.tool_metrics.append(metric)
        self._update_tool_performance_aggregates(metric)

        # Check for performance alerts
        self._check_tool_metric_alerts(metric)

        self.logger.debug(f"Tool metric recorded: {tool_name}",
                         tool_name=tool_name,
                         agent_name=agent_name,
                         execution_time=execution_time,
                         success=success)

    def record_workflow_metric(self,
                              workflow_id: str,
                              stage_name: str,
                              stage_duration: float,
                              total_duration: float,
                              success: bool,
                              agents_involved: list[str],
                              tools_used: list[str]) -> None:
        """
        Record a workflow performance metric.

        Args:
            workflow_id: Unique identifier for the workflow
            stage_name: Name of the workflow stage
            stage_duration: Duration of this stage
            total_duration: Total workflow duration so far
            success: Whether the stage completed successfully
            agents_involved: List of agents involved in this stage
            tools_used: List of tools used in this stage
        """
        metric = WorkflowMetric(
            workflow_id=workflow_id,
            session_id=self.session_id,
            timestamp=datetime.now(),
            stage_name=stage_name,
            stage_duration=stage_duration,
            total_duration=total_duration,
            success=success,
            agents_involved=agents_involved,
            tools_used=tools_used
        )

        self.workflow_metrics.append(metric)
        self._update_workflow_performance_aggregates(metric)

        # Check for performance alerts
        self._check_workflow_metric_alerts(metric)

        self.logger.debug(f"Workflow metric recorded: {stage_name}",
                         workflow_id=workflow_id,
                         stage_name=stage_name,
                         stage_duration=stage_duration,
                         success=success)

    async def _collect_system_metrics(self) -> None:
        """Background task to collect system resource metrics."""
        while self.is_monitoring:
            try:
                # Get system resource information
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()

                # Count active processes (simplified - in real implementation,
                # you'd count agent-specific processes)
                process_count = len(psutil.pids())

                metric = SystemMetric(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available_mb=memory.available / (1024 * 1024),
                    disk_usage_percent=disk.percent,
                    network_io={
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv,
                        'packets_sent': network.packets_sent,
                        'packets_recv': network.packets_recv
                    },
                    process_count=process_count,
                    active_agents=len(self.agent_performance)
                )

                self.system_metrics.append(metric)

                # Check for system alerts
                self._check_system_metric_alerts(metric)

                # Clean up old metrics
                self._cleanup_old_metrics()

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}",
                                session_id=self.session_id)
                await asyncio.sleep(self.collection_interval)

    def _update_agent_performance_aggregates(self, metric: AgentMetric) -> None:
        """Update aggregated performance data for agents."""
        agent_key = f"{metric.agent_name}_{metric.metric_name}"

        if agent_key not in self.agent_performance:
            self.agent_performance[agent_key] = {
                'agent_name': metric.agent_name,
                'metric_name': metric.metric_name,
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0.0,
                'last_updated': None
            }

        perf = self.agent_performance[agent_key]
        perf['count'] += 1
        perf['sum'] += metric.value
        perf['min'] = min(perf['min'], metric.value)
        perf['max'] = max(perf['max'], metric.value)
        perf['avg'] = perf['sum'] / perf['count']
        perf['last_updated'] = metric.timestamp.isoformat()

    def _update_tool_performance_aggregates(self, metric: ToolMetric) -> None:
        """Update aggregated performance data for tools."""
        tool_key = f"{metric.tool_name}_{metric.agent_name}"

        if tool_key not in self.tool_performance:
            self.tool_performance[tool_key] = {
                'tool_name': metric.tool_name,
                'agent_name': metric.agent_name,
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'total_execution_time': 0.0,
                'avg_execution_time': 0.0,
                'min_execution_time': float('inf'),
                'max_execution_time': float('-inf'),
                'success_rate': 0.0,
                'last_used': None
            }

        perf = self.tool_performance[tool_key]
        perf['total_executions'] += 1
        perf['total_execution_time'] += metric.execution_time
        perf['min_execution_time'] = min(perf['min_execution_time'], metric.execution_time)
        perf['max_execution_time'] = max(perf['max_execution_time'], metric.execution_time)
        perf['avg_execution_time'] = perf['total_execution_time'] / perf['total_executions']

        if metric.success:
            perf['successful_executions'] += 1
        else:
            perf['failed_executions'] += 1

        perf['success_rate'] = perf['successful_executions'] / perf['total_executions']
        perf['last_used'] = metric.timestamp.isoformat()

    def _update_workflow_performance_aggregates(self, metric: WorkflowMetric) -> None:
        """Update aggregated performance data for workflows."""
        workflow_key = f"{metric.workflow_id}_{metric.stage_name}"

        if workflow_key not in self.workflow_performance:
            self.workflow_performance[workflow_key] = {
                'workflow_id': metric.workflow_id,
                'stage_name': metric.stage_name,
                'total_runs': 0,
                'successful_runs': 0,
                'failed_runs': 0,
                'total_stage_duration': 0.0,
                'avg_stage_duration': 0.0,
                'success_rate': 0.0,
                'agents_involved': set(metric.agents_involved),
                'tools_used': set(metric.tools_used),
                'last_run': None
            }

        perf = self.workflow_performance[workflow_key]
        perf['total_runs'] += 1
        perf['total_stage_duration'] += metric.stage_duration
        perf['avg_stage_duration'] = perf['total_stage_duration'] / perf['total_runs']
        perf['agents_involved'].update(metric.agents_involved)
        perf['tools_used'].update(metric.tools_used)

        if metric.success:
            perf['successful_runs'] += 1
        else:
            perf['failed_runs'] += 1

        perf['success_rate'] = perf['successful_runs'] / perf['total_runs']
        perf['last_run'] = metric.timestamp.isoformat()

    def _check_agent_metric_alerts(self, metric: AgentMetric) -> None:
        """Check for agent metric alerts and log them."""
        # Example alert checks - customize based on your requirements
        if metric.metric_type == "error" and metric.value > 0:
            self.logger.warning(f"Agent error detected: {metric.agent_name}",
                              agent_name=metric.agent_name,
                              metric_name=metric.metric_name,
                              error_count=metric.value,
                              severity="warning")

    def _check_tool_metric_alerts(self, metric: ToolMetric) -> None:
        """Check for tool metric alerts and log them."""
        if not metric.success:
            self.logger.warning(f"Tool execution failed: {metric.tool_name}",
                              tool_name=metric.tool_name,
                              agent_name=metric.agent_name,
                              error_type=metric.error_type,
                              execution_time=metric.execution_time,
                              severity="warning")

        if metric.execution_time > self.thresholds['tool_execution_critical']:
            self.logger.error(f"Tool execution time critical: {metric.tool_name}",
                            tool_name=metric.tool_name,
                            agent_name=metric.agent_name,
                            execution_time=metric.execution_time,
                            threshold=self.thresholds['tool_execution_critical'],
                            severity="critical")
        elif metric.execution_time > self.thresholds['tool_execution_warning']:
            self.logger.warning(f"Tool execution time slow: {metric.tool_name}",
                              tool_name=metric.tool_name,
                              agent_name=metric.agent_name,
                              execution_time=metric.execution_time,
                              threshold=self.thresholds['tool_execution_warning'],
                              severity="warning")

    def _check_workflow_metric_alerts(self, metric: WorkflowMetric) -> None:
        """Check for workflow metric alerts and log them."""
        if not metric.success:
            self.logger.warning(f"Workflow stage failed: {metric.stage_name}",
                              workflow_id=metric.workflow_id,
                              stage_name=metric.stage_name,
                              stage_duration=metric.stage_duration,
                              agents_involved=metric.agents_involved,
                              severity="warning")

        if metric.stage_duration > self.thresholds['workflow_stage_critical']:
            self.logger.error(f"Workflow stage duration critical: {metric.stage_name}",
                            workflow_id=metric.workflow_id,
                            stage_name=metric.stage_name,
                            stage_duration=metric.stage_duration,
                            threshold=self.thresholds['workflow_stage_critical'],
                            severity="critical")
        elif metric.stage_duration > self.thresholds['workflow_stage_warning']:
            self.logger.warning(f"Workflow stage duration slow: {metric.stage_name}",
                              workflow_id=metric.workflow_id,
                              stage_name=metric.stage_name,
                              stage_duration=metric.stage_duration,
                              threshold=self.thresholds['workflow_stage_warning'],
                              severity="warning")

    def _check_system_metric_alerts(self, metric: SystemMetric) -> None:
        """Check for system metric alerts and log them."""
        if metric.cpu_percent > self.thresholds['cpu_critical']:
            self.logger.error(f"CPU usage critical: {metric.cpu_percent:.1f}%",
                            cpu_percent=metric.cpu_percent,
                            threshold=self.thresholds['cpu_critical'],
                            severity="critical")
        elif metric.cpu_percent > self.thresholds['cpu_warning']:
            self.logger.warning(f"CPU usage high: {metric.cpu_percent:.1f}%",
                              cpu_percent=metric.cpu_percent,
                              threshold=self.thresholds['cpu_warning'],
                              severity="warning")

        if metric.memory_percent > self.thresholds['memory_critical']:
            self.logger.error(f"Memory usage critical: {metric.memory_percent:.1f}%",
                            memory_percent=metric.memory_percent,
                            threshold=self.thresholds['memory_critical'],
                            severity="critical")
        elif metric.memory_percent > self.thresholds['memory_warning']:
            self.logger.warning(f"Memory usage high: {metric.memory_percent:.1f}%",
                              memory_percent=metric.memory_percent,
                              threshold=self.thresholds['memory_warning'],
                              severity="warning")

    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than the retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

        # Clean up agent metrics
        self.agent_metrics = deque(
            (m for m in self.agent_metrics if m.timestamp > cutoff_time),
            maxlen=10000
        )

        # Clean up system metrics
        self.system_metrics = deque(
            (m for m in self.system_metrics if m.timestamp > cutoff_time),
            maxlen=10000
        )

        # Clean up tool metrics
        self.tool_metrics = deque(
            (m for m in self.tool_metrics if m.timestamp > cutoff_time),
            maxlen=10000
        )

        # Clean up workflow metrics
        self.workflow_metrics = deque(
            (m for m in self.workflow_metrics if m.timestamp > cutoff_time),
            maxlen=10000
        )

    def get_agent_summary(self, agent_name: str | None = None) -> dict[str, Any]:
        """
        Get performance summary for agents.

        Args:
            agent_name: Specific agent name, or None for all agents

        Returns:
            Dictionary containing agent performance summary
        """
        recent_time = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in self.agent_metrics
                         if m.timestamp > recent_time and
                         (agent_name is None or m.agent_name == agent_name)]

        summary = {
            'timestamp': datetime.now().isoformat(),
            'agent_name': agent_name,
            'total_metrics': len(recent_metrics),
            'metrics_by_type': defaultdict(list),
            'performance_aggregates': {}
        }

        # Group metrics by type and agent
        for metric in recent_metrics:
            summary['metrics_by_type'][metric.metric_type].append(metric)

        # Add performance aggregates
        for key, perf in self.agent_performance.items():
            if agent_name is None or perf['agent_name'] == agent_name:
                summary['performance_aggregates'][key] = perf

        return dict(summary)

    def get_tool_summary(self, tool_name: str | None = None) -> dict[str, Any]:
        """
        Get performance summary for tools.

        Args:
            tool_name: Specific tool name, or None for all tools

        Returns:
            Dictionary containing tool performance summary
        """
        recent_time = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in self.tool_metrics
                         if m.timestamp > recent_time and
                         (tool_name is None or m.tool_name == tool_name)]

        summary = {
            'timestamp': datetime.now().isoformat(),
            'tool_name': tool_name,
            'total_executions': len(recent_metrics),
            'successful_executions': len([m for m in recent_metrics if m.success]),
            'failed_executions': len([m for m in recent_metrics if not m.success]),
            'avg_execution_time': 0.0,
            'performance_aggregates': {}
        }

        if recent_metrics:
            summary['avg_execution_time'] = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)

        # Add performance aggregates
        for key, perf in self.tool_performance.items():
            if tool_name is None or perf['tool_name'] == tool_name:
                summary['performance_aggregates'][key] = perf

        return summary

    def get_system_summary(self) -> dict[str, Any]:
        """Get current system resource summary."""
        if not self.system_metrics:
            return {}

        latest_metric = self.system_metrics[-1]

        # Calculate averages over last hour
        recent_time = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in self.system_metrics if m.timestamp > recent_time]

        summary = {
            'timestamp': latest_metric.timestamp.isoformat(),
            'current': asdict(latest_metric),
            'averages': {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'process_count': 0
            },
            'peak': {
                'cpu_percent': 0.0,
                'memory_percent': 0.0
            }
        }

        if recent_metrics:
            summary['averages']['cpu_percent'] = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            summary['averages']['memory_percent'] = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            summary['averages']['process_count'] = sum(m.process_count for m in recent_metrics) / len(recent_metrics)

            summary['peak']['cpu_percent'] = max(m.cpu_percent for m in recent_metrics)
            summary['peak']['memory_percent'] = max(m.memory_percent for m in recent_metrics)

        return summary

    def export_metrics(self,
                      file_path: str | None = None,
                      metric_types: list[str] | None = None) -> str:
        """
        Export all metrics to a JSON file.

        Args:
            file_path: Optional custom file path
            metric_types: Types of metrics to export ('agent', 'system', 'tool', 'workflow')

        Returns:
            Path to the exported file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self.metrics_dir / f"metrics_export_{self.session_id}_{timestamp}.json")

        metric_types = metric_types or ['agent', 'system', 'tool', 'workflow']

        export_data = {
            'session_id': self.session_id,
            'export_timestamp': datetime.now().isoformat(),
            'metric_types': metric_types,
            'data': {}
        }

        if 'agent' in metric_types:
            export_data['data']['agent_metrics'] = [asdict(m) for m in self.agent_metrics]
            export_data['data']['agent_performance'] = self.agent_performance

        if 'system' in metric_types:
            export_data['data']['system_metrics'] = [asdict(m) for m in self.system_metrics]

        if 'tool' in metric_types:
            export_data['data']['tool_metrics'] = [asdict(m) for m in self.tool_metrics]
            export_data['data']['tool_performance'] = self.tool_performance

        if 'workflow' in metric_types:
            export_data['data']['workflow_metrics'] = [asdict(m) for m in self.workflow_metrics]
            export_data['data']['workflow_performance'] = self.workflow_performance

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Metrics exported to file: {file_path}",
                        session_id=self.session_id,
                        file_path=file_path,
                        metric_types=metric_types,
                        total_metrics=len(self.agent_metrics) + len(self.system_metrics) +
                                      len(self.tool_metrics) + len(self.workflow_metrics))

        return file_path

    async def cleanup(self) -> None:
        """Clean up resources and stop monitoring."""
        await self.stop_monitoring()

        # Export final metrics
        try:
            self.export_metrics()
        except Exception as e:
            self.logger.error(f"Error exporting final metrics: {e}",
                            session_id=self.session_id)

        self.logger.info("MetricsCollector cleanup completed",
                        session_id=self.session_id)
