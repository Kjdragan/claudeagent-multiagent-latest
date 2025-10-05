"""
Diagnostic Tools for enhanced error reporting and session reconstruction.

This module provides comprehensive diagnostic capabilities including error analysis,
session reconstruction, debug information collection, and troubleshooting tools.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .metrics_collector import MetricsCollector
from .performance_monitor import PerformanceMonitor
from .system_health import SystemHealthMonitor

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_logging import StructuredLogger


class DiagnosticTools:
    """Comprehensive diagnostic tools for the multi-agent system."""

    def __init__(self,
                 session_id: str,
                 metrics_collector: MetricsCollector,
                 performance_monitor: PerformanceMonitor,
                 health_monitor: SystemHealthMonitor,
                 diagnostics_dir: str = "diagnostics"):
        """
        Initialize diagnostic tools.

        Args:
            session_id: Session identifier
            metrics_collector: MetricsCollector instance
            performance_monitor: PerformanceMonitor instance
            health_monitor: SystemHealthMonitor instance
            diagnostics_dir: Directory to store diagnostic data
        """
        self.session_id = session_id
        self.metrics_collector = metrics_collector
        self.performance_monitor = performance_monitor
        self.health_monitor = health_monitor
        self.diagnostics_dir = Path(diagnostics_dir)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

        # Initialize structured logger
        self.logger = StructuredLogger(
            name="diagnostics",
            log_dir=self.diagnostics_dir
        )

        # Diagnostic data cache
        self._diagnostic_cache: dict[str, Any] = {}
        self._cache_timestamp: datetime | None = None
        self._cache_ttl_minutes = 5

        self.logger.info("DiagnosticTools initialized",
                        session_id=session_id,
                        diagnostics_dir=str(self.diagnostics_dir))

    async def generate_comprehensive_diagnostic_report(self) -> dict[str, Any]:
        """
        Generate a comprehensive diagnostic report for the current session.

        Returns:
            Complete diagnostic report including all system components
        """
        report_start_time = datetime.now()

        # Check cache first
        if (self._cache_timestamp and
            datetime.now() - self._cache_timestamp < timedelta(minutes=self._cache_ttl_minutes)):
            self.logger.debug("Returning cached diagnostic report")
            return self._diagnostic_cache

        self.logger.info("Generating comprehensive diagnostic report")

        report = {
            'session_info': await self._get_session_info(),
            'system_status': await self._get_system_status(),
            'agent_analysis': await self._analyze_agent_activities(),
            'tool_performance': await self._analyze_tool_performance(),
            'workflow_analysis': await self._analyze_workflow_performance(),
            'error_analysis': await self._analyze_errors(),
            'performance_analysis': await self._analyze_performance_trends(),
            'health_assessment': await self._get_health_assessment(),
            'resource_utilization': await self._analyze_resource_utilization(),
            'alerts_summary': await self._get_alerts_summary(),
            'recommendations': await self._generate_recommendations(),
            'debug_info': await self._collect_debug_info(),
            'report_metadata': {
                'generated_at': report_start_time.isoformat(),
                'generation_duration_seconds': (datetime.now() - report_start_time).total_seconds(),
                'session_id': self.session_id
            }
        }

        # Update cache
        self._diagnostic_cache = report
        self._cache_timestamp = datetime.now()

        self.logger.info(f"Comprehensive diagnostic report generated in {report['report_metadata']['generation_duration_seconds']:.2f}s")

        return report

    async def reconstruct_session(self,
                                 session_id: str | None = None,
                                 time_range_hours: int = 24) -> dict[str, Any]:
        """
        Reconstruct a session timeline with all activities and events.

        Args:
            session_id: Specific session ID to reconstruct, or None for current
            time_range_hours: Hours of data to include in reconstruction

        Returns:
            Detailed session reconstruction timeline
        """
        target_session_id = session_id or self.session_id
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)

        self.logger.info(f"Reconstructing session: {target_session_id}",
                        time_range_hours=time_range_hours)

        reconstruction = {
            'session_id': target_session_id,
            'reconstruction_period': {
                'start_time': cutoff_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'hours': time_range_hours
            },
            'timeline': [],
            'agent_activities': defaultdict(list),
            'tool_executions': [],
            'workflow_stages': [],
            'system_events': [],
            'errors': [],
            'performance_events': [],
            'summary': {}
        }

        # Reconstruct agent activities
        for metric in self.metrics_collector.agent_metrics:
            if (metric.session_id == target_session_id and
                metric.timestamp > cutoff_time):

                event = {
                    'timestamp': metric.timestamp.isoformat(),
                    'type': 'agent_activity',
                    'agent': metric.agent_name,
                    'activity_type': metric.metric_type,
                    'activity_name': metric.metric_name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'metadata': metric.metadata
                }

                reconstruction['timeline'].append(event)
                reconstruction['agent_activities'][metric.agent_name].append(event)

        # Reconstruct tool executions
        for metric in self.metrics_collector.tool_metrics:
            if (metric.session_id == target_session_id and
                metric.timestamp > cutoff_time):

                event = {
                    'timestamp': metric.timestamp.isoformat(),
                    'type': 'tool_execution',
                    'tool': metric.tool_name,
                    'agent': metric.agent_name,
                    'execution_time': metric.execution_time,
                    'success': metric.success,
                    'input_size': metric.input_size,
                    'output_size': metric.output_size,
                    'error_type': metric.error_type
                }

                reconstruction['timeline'].append(event)
                reconstruction['tool_executions'].append(event)

        # Reconstruct workflow stages
        for metric in self.metrics_collector.workflow_metrics:
            if (metric.session_id == target_session_id and
                metric.timestamp > cutoff_time):

                event = {
                    'timestamp': metric.timestamp.isoformat(),
                    'type': 'workflow_stage',
                    'workflow_id': metric.workflow_id,
                    'stage_name': metric.stage_name,
                    'stage_duration': metric.stage_duration,
                    'total_duration': metric.total_duration,
                    'success': metric.success,
                    'agents_involved': metric.agents_involved,
                    'tools_used': metric.tools_used
                }

                reconstruction['timeline'].append(event)
                reconstruction['workflow_stages'].append(event)

        # Add performance alerts
        for alert in self.performance_monitor.alerts:
            if alert.timestamp > cutoff_time:
                event = {
                    'timestamp': alert.timestamp.isoformat(),
                    'type': 'performance_alert',
                    'alert_type': alert.alert_type,
                    'metric_name': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'agent_name': alert.agent_name,
                    'tool_name': alert.tool_name,
                    'workflow_id': alert.workflow_id,
                    'message': alert.message
                }

                reconstruction['timeline'].append(event)
                reconstruction['performance_events'].append(event)

        # Sort timeline by timestamp
        reconstruction['timeline'].sort(key=lambda x: x['timestamp'])

        # Generate summary
        reconstruction['summary'] = {
            'total_events': len(reconstruction['timeline']),
            'agent_activity_count': len([e for e in reconstruction['timeline'] if e['type'] == 'agent_activity']),
            'tool_execution_count': len(reconstruction['tool_executions']),
            'workflow_stage_count': len(reconstruction['workflow_stages']),
            'performance_alert_count': len(reconstruction['performance_events']),
            'unique_agents': len(reconstruction['agent_activities']),
            'unique_tools': len(set(e['tool'] for e in reconstruction['tool_executions'])),
            'unique_workflows': len(set(e['workflow_id'] for e in reconstruction['workflow_stages']))
        }

        self.logger.info(f"Session reconstruction completed: {reconstruction['summary']['total_events']} events")

        return reconstruction

    async def analyze_error_patterns(self,
                                   time_range_hours: int = 24) -> dict[str, Any]:
        """
        Analyze error patterns and provide insights.

        Args:
            time_range_hours: Hours of data to analyze

        Returns:
            Error pattern analysis
        """
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)

        # Collect error-related data
        error_data = {
            'agent_errors': [],
            'tool_failures': [],
            'workflow_failures': [],
            'system_errors': [],
            'performance_alerts': []
        }

        # Agent errors
        for metric in self.metrics_collector.agent_metrics:
            if (metric.timestamp > cutoff_time and
                metric.metric_type == 'error' and
                metric.value > 0):
                error_data['agent_errors'].append({
                    'timestamp': metric.timestamp,
                    'agent': metric.agent_name,
                    'error_name': metric.metric_name,
                    'error_count': metric.value,
                    'metadata': metric.metadata
                })

        # Tool failures
        for metric in self.metrics_collector.tool_metrics:
            if (metric.timestamp > cutoff_time and
                not metric.success):
                error_data['tool_failures'].append({
                    'timestamp': metric.timestamp,
                    'tool': metric.tool_name,
                    'agent': metric.agent_name,
                    'error_type': metric.error_type,
                    'execution_time': metric.execution_time
                })

        # Workflow failures
        for metric in self.metrics_collector.workflow_metrics:
            if (metric.timestamp > cutoff_time and
                not metric.success):
                error_data['workflow_failures'].append({
                    'timestamp': metric.timestamp,
                    'workflow_id': metric.workflow_id,
                    'stage_name': metric.stage_name,
                    'agents_involved': metric.agents_involved,
                    'stage_duration': metric.stage_duration
                })

        # Performance alerts (critical ones)
        for alert in self.performance_monitor.alerts:
            if (alert.timestamp > cutoff_time and
                alert.alert_type == 'critical'):
                error_data['performance_alerts'].append({
                    'timestamp': alert.timestamp,
                    'metric_name': alert.metric_name,
                    'current_value': alert.current_value,
                    'agent_name': alert.agent_name,
                    'tool_name': alert.tool_name,
                    'message': alert.message
                })

        # Analyze patterns
        analysis = {
            'time_range_hours': time_range_hours,
            'total_errors': sum(len(errors) for errors in error_data.values()),
            'error_breakdown': {k: len(v) for k, v in error_data.items()},
            'patterns': await self._identify_error_patterns(error_data),
            'trends': await self._analyze_error_trends(error_data),
            'recommendations': await self._generate_error_recommendations(error_data),
            'raw_data': error_data
        }

        return analysis

    async def _get_session_info(self) -> dict[str, Any]:
        """Get session information."""
        return {
            'session_id': self.session_id,
            'start_time': self.performance_monitor.active_sessions.get(self.session_id, {}).get('start_time'),
            'duration_seconds': (datetime.now() - self.performance_monitor.active_sessions.get(self.session_id, {}).get('start_time', datetime.now())).total_seconds(),
            'active_monitoring': {
                'metrics_collector': self.metrics_collector.is_monitoring,
                'performance_monitor': self.performance_monitor.is_monitoring,
                'health_monitor': self.health_monitor.is_monitoring
            }
        }

    async def _get_system_status(self) -> dict[str, Any]:
        """Get current system status."""
        return self.metrics_collector.get_system_summary()

    async def _analyze_agent_activities(self) -> dict[str, Any]:
        """Analyze agent activities and performance."""
        agent_summary = self.metrics_collector.get_agent_summary()

        analysis = {
            'summary': agent_summary,
            'performance_trends': {},
            'activity_patterns': {},
            'agent_health': {}
        }

        # Analyze performance trends for each agent
        for agent_name in set(m.agent_name for m in self.metrics_collector.agent_metrics):
            agent_metrics = [m for m in self.metrics_collector.agent_metrics if m.agent_name == agent_name]

            # Calculate activity trends
            recent_time = datetime.now() - timedelta(hours=1)
            recent_activities = [m for m in agent_metrics if m.timestamp > recent_time]
            older_time = datetime.now() - timedelta(hours=2)
            older_activities = [m for m in agent_metrics if older_time < m.timestamp <= recent_time]

            trend = "stable"
            if len(recent_activities) > len(older_activities) * 1.2:
                trend = "increasing"
            elif len(recent_activities) < len(older_activities) * 0.8:
                trend = "decreasing"

            analysis['performance_trends'][agent_name] = {
                'recent_activities': len(recent_activities),
                'older_activities': len(older_activities),
                'trend': trend
            }

        return analysis

    async def _analyze_tool_performance(self) -> dict[str, Any]:
        """Analyze tool performance patterns."""
        tool_summary = self.metrics_collector.get_tool_summary()

        analysis = {
            'summary': tool_summary,
            'performance_analysis': {},
            'failure_patterns': {},
            'efficiency_metrics': {}
        }

        # Analyze individual tool performance
        for tool_name in set(m.tool_name for m in self.metrics_collector.tool_metrics):
            tool_metrics = [m for m in self.metrics_collector.tool_metrics if m.tool_name == tool_name]

            if tool_metrics:
                execution_times = [m.execution_time for m in tool_metrics]
                success_rate = sum(1 for m in tool_metrics if m.success) / len(tool_metrics)

                analysis['performance_analysis'][tool_name] = {
                    'total_executions': len(tool_metrics),
                    'success_rate': success_rate,
                    'avg_execution_time': sum(execution_times) / len(execution_times),
                    'min_execution_time': min(execution_times),
                    'max_execution_time': max(execution_times),
                    'recent_performance': success_rate  # Simplified - could be more sophisticated
                }

        return analysis

    async def _analyze_workflow_performance(self) -> dict[str, Any]:
        """Analyze workflow performance patterns."""
        analysis = {
            'workflow_summary': {},
            'stage_analysis': {},
            'bottlenecks': [],
            'efficiency_metrics': {}
        }

        # Analyze workflow stages
        for workflow_id in set(m.workflow_id for m in self.metrics_collector.workflow_metrics):
            workflow_metrics = [m for m in self.metrics_collector.workflow_metrics if m.workflow_id == workflow_id]

            stage_analysis = {}
            for metric in workflow_metrics:
                stage_name = metric.stage_name
                if stage_name not in stage_analysis:
                    stage_analysis[stage_name] = {
                        'executions': 0,
                        'successes': 0,
                        'total_duration': 0,
                        'avg_duration': 0
                    }

                stage_analysis[stage_name]['executions'] += 1
                stage_analysis[stage_name]['total_duration'] += metric.stage_duration
                if metric.success:
                    stage_analysis[stage_name]['successes'] += 1

            # Calculate averages and success rates
            for stage_name, data in stage_analysis.items():
                data['avg_duration'] = data['total_duration'] / data['executions']
                data['success_rate'] = data['successes'] / data['executions']

            analysis['stage_analysis'][workflow_id] = stage_analysis

        return analysis

    async def _analyze_errors(self) -> dict[str, Any]:
        """Analyze error patterns and occurrences."""
        return await self.analyze_error_patterns()

    async def _analyze_performance_trends(self) -> dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {
            'agent_performance_trends': {},
            'tool_performance_trends': {},
            'system_performance_trends': {},
            'overall_health_trend': 'stable'
        }

        # Analyze recent vs older performance
        recent_time = datetime.now() - timedelta(hours=1)
        older_time = datetime.now() - timedelta(hours=2)

        # Agent performance trends
        for agent_name in set(m.agent_name for m in self.metrics_collector.agent_metrics):
            recent_metrics = [m for m in self.metrics_collector.agent_metrics
                            if m.agent_name == agent_name and m.timestamp > recent_time]
            older_metrics = [m for m in self.metrics_collector.agent_metrics
                           if m.agent_name == agent_name and older_time < m.timestamp <= recent_time]

            if recent_metrics and older_metrics:
                recent_avg = sum(m.value for m in recent_metrics) / len(recent_metrics)
                older_avg = sum(m.value for m in older_metrics) / len(older_metrics)

                change_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0

                trends['agent_performance_trends'][agent_name] = {
                    'change_percent': change_percent,
                    'trend': 'improving' if change_percent < -10 else 'degrading' if change_percent > 10 else 'stable'
                }

        return trends

    async def _get_health_assessment(self) -> dict[str, Any]:
        """Get current health assessment."""
        return self.health_monitor.get_health_summary()

    async def _analyze_resource_utilization(self) -> dict[str, Any]:
        """Analyze resource utilization patterns."""
        system_summary = self.metrics_collector.get_system_summary()

        if not system_summary:
            return {'status': 'no_data'}

        current = system_summary.get('current', {})
        averages = system_summary.get('averages', {})
        peaks = system_summary.get('peak', {})

        return {
            'current_utilization': current,
            'average_utilization': averages,
            'peak_utilization': peaks,
            'efficiency_assessment': await self._assess_resource_efficiency(current, averages, peaks)
        }

    async def _assess_resource_efficiency(self,
                                        current: dict[str, Any],
                                        averages: dict[str, Any],
                                        peaks: dict[str, Any]) -> dict[str, str]:
        """Assess resource efficiency."""
        assessment = {}

        # CPU efficiency
        cpu_avg = averages.get('cpu_percent', 0)
        if cpu_avg < 30:
            assessment['cpu'] = 'underutilized'
        elif cpu_avg < 70:
            assessment['cpu'] = 'optimal'
        else:
            assessment['cpu'] = 'overutilized'

        # Memory efficiency
        memory_avg = averages.get('memory_percent', 0)
        if memory_avg < 50:
            assessment['memory'] = 'underutilized'
        elif memory_avg < 80:
            assessment['memory'] = 'optimal'
        else:
            assessment['memory'] = 'overutilized'

        return assessment

    async def _get_alerts_summary(self) -> dict[str, Any]:
        """Get alerts summary."""
        return self.performance_monitor.get_alerts_summary()

    async def _generate_recommendations(self) -> list[str]:
        """Generate system optimization recommendations."""
        recommendations = []

        # Analyze current state and generate recommendations
        system_summary = self.metrics_collector.get_system_summary()
        if system_summary and system_summary.get('current'):
            current = system_summary['current']

            if current.get('cpu_percent', 0) > 80:
                recommendations.append("High CPU usage detected - consider optimizing processes or adding CPU resources")

            if current.get('memory_percent', 0) > 80:
                recommendations.append("High memory usage detected - consider freeing up memory or increasing RAM")

        # Check for performance alerts
        alerts_summary = await self._get_alerts_summary()
        if alerts_summary.get('total_alerts', 0) > 5:
            recommendations.append("High number of performance alerts - system may need optimization")

        # Check error patterns
        error_analysis = await self.analyze_error_patterns()
        if error_analysis.get('total_errors', 0) > 10:
            recommendations.append("High error rate detected - investigate and fix underlying issues")

        if not recommendations:
            recommendations.append("System performance appears to be optimal")

        return recommendations

    async def _collect_debug_info(self) -> dict[str, Any]:
        """Collect debug information for troubleshooting."""
        import os
        import platform
        import sys

        debug_info = {
            'system_info': {
                'platform': platform.platform(),
                'python_version': sys.version,
                'working_directory': os.getcwd(),
                'environment_variables': {k: v for k, v in os.environ.items()
                                       if k.startswith(('PYTHON', 'PATH', 'HOME'))}
            },
            'process_info': {
                'process_id': os.getpid(),
                'parent_process_id': os.getppid()
            },
            'module_info': {
                'metrics_collector_available': self.metrics_collector is not None,
                'performance_monitor_available': self.performance_monitor is not None,
                'health_monitor_available': self.health_monitor is not None
            },
            'storage_info': {
                'log_directory': str(self.metrics_collector.metrics_dir),
                'health_directory': str(self.health_monitor.health_dir),
                'diagnostics_directory': str(self.diagnostics_dir)
            }
        }

        return debug_info

    async def _identify_error_patterns(self, error_data: dict[str, list]) -> dict[str, Any]:
        """Identify patterns in error data."""
        patterns = {
            'most_common_errors': {},
            'error_frequency_by_time': {},
            'error_correlations': []
        }

        # Most common errors
        for error_type, errors in error_data.items():
            if errors:
                patterns['most_common_errors'][error_type] = len(errors)

        return patterns

    async def _analyze_error_trends(self, error_data: dict[str, list]) -> dict[str, Any]:
        """Analyze error trends over time."""
        trends = {
            'error_rate_trend': 'stable',
            'hourly_distribution': {},
            'recent_spike_detected': False
        }

        # Simple trend analysis
        total_errors = sum(len(errors) for errors in error_data.values())
        if total_errors > 20:
            trends['recent_spike_detected'] = True
            trends['error_rate_trend'] = 'increasing'

        return trends

    async def _generate_error_recommendations(self, error_data: dict[str, list]) -> list[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []

        total_errors = sum(len(errors) for errors in error_data.values())
        if total_errors > 15:
            recommendations.append("High error rate detected - review system configuration and resource allocation")

        if len(error_data['tool_failures']) > 5:
            recommendations.append("Multiple tool failures detected - check tool configurations and dependencies")

        if len(error_data['performance_alerts']) > 3:
            recommendations.append("Performance alerts indicate system stress - consider scaling resources")

        return recommendations

    def export_diagnostic_report(self,
                               report: dict[str, Any],
                               file_path: str | None = None) -> str:
        """
        Export diagnostic report to file.

        Args:
            report: Diagnostic report to export
            file_path: Optional custom file path

        Returns:
            Path to exported file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self.diagnostics_dir / f"diagnostic_report_{self.session_id}_{timestamp}.json")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Diagnostic report exported to: {file_path}")

        return file_path

    async def cleanup(self) -> None:
        """Clean up diagnostic tools resources."""
        # Clear cache
        self._diagnostic_cache.clear()
        self._cache_timestamp = None

        self.logger.info("DiagnosticTools cleanup completed",
                        session_id=self.session_id)
