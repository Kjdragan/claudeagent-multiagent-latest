"""
Monitoring Integration Module

This module provides a unified interface for integrating all monitoring components
with the multi-agent research system orchestrator.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Import monitoring components (with graceful fallback for missing dependencies)
try:
    from .monitoring import (
        DiagnosticTools,
        MetricsCollector,
        PerformanceMonitor,
        SystemHealthMonitor,
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced monitoring components not available: {e}")
    MONITORING_AVAILABLE = False

from .agent_logging import create_agent_logger


class MonitoringIntegration:
    """
    Unified monitoring integration for the multi-agent research system.

    Provides a single interface for all monitoring functionality, with graceful
    degradation when advanced monitoring components are not available.
    """

    def __init__(self,
                 session_id: str | None = None,
                 monitoring_dir: str = "monitoring",
                 enable_advanced_monitoring: bool = True):
        """
        Initialize monitoring integration.

        Args:
            session_id: Optional session identifier
            monitoring_dir: Base directory for monitoring data
            enable_advanced_monitoring: Whether to enable advanced monitoring features
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.enable_advanced_monitoring = enable_advanced_monitoring

        # Initialize basic components (always available)
        self.agent_loggers: dict[str, Any] = {}
        self._initialize_agent_loggers()

        # Initialize advanced components (if available and enabled)
        self.metrics_collector: Any | None = None
        self.performance_monitor: Any | None = None
        self.health_monitor: Any | None = None
        self.diagnostics: Any | None = None

        if (self.enable_advanced_monitoring and
            MONITORING_AVAILABLE):
            self._initialize_advanced_monitoring()

        print("✅ Monitoring Integration initialized")
        print(f"   Session ID: {self.session_id}")
        print(f"   Monitoring Directory: {self.monitoring_dir}")
        print(f"   Advanced Monitoring: {'✅ Enabled' if self.metrics_collector else '❌ Disabled'}")

    def _initialize_agent_loggers(self) -> None:
        """Initialize agent-specific loggers."""
        agent_types = [
            "research_agent",
            "report_agent",
            "editor_agent",
            "ui_coordinator"
        ]

        for agent_type in agent_types:
            try:
                agent_logger = create_agent_logger(agent_type, self.session_id)
                self.agent_loggers[agent_type] = agent_logger
                print(f"   ✅ {agent_type} logger initialized")
            except Exception as e:
                print(f"   ⚠️  Failed to initialize {agent_type} logger: {e}")

    def _initialize_advanced_monitoring(self) -> None:
        """Initialize advanced monitoring components."""
        try:
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector(
                session_id=self.session_id,
                metrics_dir=str(self.monitoring_dir / "metrics")
            )

            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor(
                metrics_collector=self.metrics_collector
            )

            # Initialize system health monitor
            self.health_monitor = SystemHealthMonitor(
                session_id=self.session_id,
                health_dir=str(self.monitoring_dir / "health")
            )

            # Initialize diagnostic tools
            self.diagnostics = DiagnosticTools(
                session_id=self.session_id,
                metrics_collector=self.metrics_collector,
                performance_monitor=self.performance_monitor,
                health_monitor=self.health_monitor,
                diagnostics_dir=str(self.monitoring_dir / "diagnostics")
            )

            print("   ✅ Advanced monitoring components initialized")

        except Exception as e:
            print(f"   ⚠️  Failed to initialize advanced monitoring: {e}")
            self.enable_advanced_monitoring = False

    async def start_monitoring(self) -> None:
        """Start all monitoring components."""
        # Start advanced monitoring if available
        if self.enable_advanced_monitoring and self.metrics_collector:
            try:
                await self.metrics_collector.start_monitoring()
                await self.performance_monitor.start_monitoring()
                await self.health_monitor.start_monitoring()
                print("✅ Advanced monitoring started")
            except Exception as e:
                print(f"⚠️  Failed to start advanced monitoring: {e}")

    async def stop_monitoring(self) -> None:
        """Stop all monitoring components."""
        # Stop advanced monitoring if available
        if self.enable_advanced_monitoring:
            try:
                if self.metrics_collector:
                    await self.metrics_collector.stop_monitoring()
                if self.performance_monitor:
                    await self.performance_monitor.stop_monitoring()
                if self.health_monitor:
                    await self.health_monitor.stop_monitoring()
                if self.diagnostics:
                    await self.diagnostics.cleanup()
                print("✅ Advanced monitoring stopped")
            except Exception as e:
                print(f"⚠️  Failed to stop advanced monitoring: {e}")

    def get_agent_logger(self, agent_name: str) -> Any | None:
        """Get the logger for a specific agent."""
        return self.agent_loggers.get(agent_name)

    def log_agent_activity(self,
                          agent_name: str,
                          activity_type: str,
                          activity_name: str,
                          **kwargs) -> None:
        """
        Log an agent activity using the appropriate logger.

        Args:
            agent_name: Name of the agent
            activity_type: Type of activity
            activity_name: Name of the specific activity
            **kwargs: Additional activity data
        """
        logger = self.get_agent_logger(agent_name)
        if logger:
            logger.log_activity(
                agent_name=agent_name,
                activity_type=activity_type,
                stage=activity_name,
                **kwargs
            )

        # Also log to advanced monitoring if available
        if self.performance_monitor:
            try:
                self.performance_monitor.record_agent_activity(
                    agent_name=agent_name,
                    activity_type=activity_type,
                    activity_name=activity_name,
                    value=kwargs.get('value', 1.0),
                    unit=kwargs.get('unit', 'count'),
                    metadata=kwargs
                )
            except Exception as e:
                print(f"⚠️  Failed to log to performance monitor: {e}")

    async def get_monitoring_summary(self) -> dict[str, Any]:
        """Get a comprehensive monitoring summary."""
        summary = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'agent_logging': {
                'initialized_agents': list(self.agent_loggers.keys()),
                'total_agents': len(self.agent_loggers)
            },
            'advanced_monitoring': {
                'enabled': self.enable_advanced_monitoring,
                'available': MONITORING_AVAILABLE
            }
        }

        # Add advanced monitoring data if available
        if self.enable_advanced_monitoring and self.metrics_collector:
            try:
                summary['metrics_summary'] = self.metrics_collector.get_system_summary()
                summary['performance_summary'] = self.performance_monitor.get_performance_summary()
                summary['health_summary'] = self.health_monitor.get_health_summary()
            except Exception as e:
                summary['advanced_monitoring']['error'] = str(e)

        return summary

    async def generate_comprehensive_report(self) -> dict[str, Any]:
        """Generate a comprehensive monitoring report."""
        if not self.diagnostics:
            return {
                'error': 'Advanced monitoring not available',
                'session_id': self.session_id
            }

        try:
            return await self.diagnostics.generate_comprehensive_diagnostic_report()
        except Exception as e:
            return {
                'error': f'Failed to generate report: {e}',
                'session_id': self.session_id
            }

    async def export_all_data(self, export_dir: str | None = None) -> dict[str, str]:
        """
        Export all monitoring data.

        Args:
            export_dir: Optional custom export directory

        Returns:
            Dictionary mapping data types to export file paths
        """
        if not export_dir:
            export_dir = str(self.monitoring_dir / "exports" / f"export_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # Export agent logger data
        for agent_name, logger in self.agent_loggers.items():
            try:
                file_path = logger.export_session_data()
                exported_files[f'agent_{agent_name}'] = file_path
            except Exception as e:
                print(f"⚠️  Failed to export {agent_name} data: {e}")

        # Export advanced monitoring data
        if self.enable_advanced_monitoring and self.metrics_collector:
            try:
                metrics_file = self.metrics_collector.export_metrics()
                exported_files['metrics'] = metrics_file
            except Exception as e:
                print(f"⚠️  Failed to export metrics: {e}")

            try:
                health_file = self.health_monitor.export_health_report()
                exported_files['health'] = health_file
            except Exception as e:
                print(f"⚠️  Failed to export health report: {e}")

            try:
                diagnostic_report = await self.diagnostics.generate_comprehensive_diagnostic_report()
                diagnostics_file = self.diagnostics.export_diagnostic_report(diagnostic_report)
                exported_files['diagnostics'] = diagnostics_file
            except Exception as e:
                print(f"⚠️  Failed to export diagnostics: {e}")

        print(f"✅ Exported {len(exported_files)} monitoring files")
        return exported_files


# Global monitoring integration instance
_monitoring_integration: MonitoringIntegration | None = None


def initialize_monitoring(session_id: str | None = None,
                         monitoring_dir: str = "monitoring",
                         enable_advanced_monitoring: bool = True) -> MonitoringIntegration:
    """
    Initialize the global monitoring integration.

    Args:
        session_id: Optional session identifier
        monitoring_dir: Base directory for monitoring data
        enable_advanced_monitoring: Whether to enable advanced monitoring features

    Returns:
        MonitoringIntegration instance
    """
    global _monitoring_integration
    _monitoring_integration = MonitoringIntegration(
        session_id=session_id,
        monitoring_dir=monitoring_dir,
        enable_advanced_monitoring=enable_advanced_monitoring
    )
    return _monitoring_integration


def get_monitoring() -> MonitoringIntegration | None:
    """Get the global monitoring integration instance."""
    return _monitoring_integration


async def start_global_monitoring() -> None:
    """Start the global monitoring integration."""
    if _monitoring_integration:
        await _monitoring_integration.start_monitoring()


async def stop_global_monitoring() -> None:
    """Stop the global monitoring integration."""
    if _monitoring_integration:
        await _monitoring_integration.stop_monitoring()


def log_activity(agent_name: str,
                activity_type: str,
                activity_name: str,
                **kwargs) -> None:
    """
    Log an activity using the global monitoring integration.

    Args:
        agent_name: Name of the agent
        activity_type: Type of activity
        activity_name: Name of the specific activity
        **kwargs: Additional activity data
    """
    if _monitoring_integration:
        _monitoring_integration.log_agent_activity(
            agent_name=agent_name,
            activity_type=activity_type,
            activity_name=activity_name,
            **kwargs
        )


def get_agent_logger(agent_name: str) -> Any | None:
    """Get the logger for a specific agent from the global integration."""
    if _monitoring_integration:
        return _monitoring_integration.get_agent_logger(agent_name)
    return None


# Context manager for monitoring sessions
class MonitoringSession:
    """Context manager for monitoring sessions."""

    def __init__(self,
                 session_id: str | None = None,
                 monitoring_dir: str = "monitoring",
                 enable_advanced_monitoring: bool = True):
        self.session_id = session_id
        self.monitoring_dir = monitoring_dir
        self.enable_advanced_monitoring = enable_advanced_monitoring
        self.monitoring: MonitoringIntegration | None = None

    async def __aenter__(self) -> MonitoringIntegration:
        """Enter the monitoring session context."""
        self.monitoring = MonitoringIntegration(
            session_id=self.session_id,
            monitoring_dir=self.monitoring_dir,
            enable_advanced_monitoring=self.enable_advanced_monitoring
        )
        await self.monitoring.start_monitoring()
        return self.monitoring

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the monitoring session context."""
        if self.monitoring:
            await self.monitoring.stop_monitoring()


# Convenience function for creating monitoring sessions
async def with_monitoring(session_id: str | None = None,
                         monitoring_dir: str = "monitoring",
                         enable_advanced_monitoring: bool = True):
    """
    Create a monitoring session context manager.

    Args:
        session_id: Optional session identifier
        monitoring_dir: Base directory for monitoring data
        enable_advanced_monitoring: Whether to enable advanced monitoring features

    Returns:
        MonitoringSession context manager
    """
    return MonitoringSession(
        session_id=session_id,
        monitoring_dir=monitoring_dir,
        enable_advanced_monitoring=enable_advanced_monitoring
    )
