"""
Real-time Monitoring Dashboard for the multi-agent research system.

This module provides a web-based dashboard for real-time visualization of
system performance, agent activities, and workflow status.
"""

import time
from datetime import datetime, timedelta

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .metrics_collector import MetricsCollector
from .performance_monitor import PerformanceMonitor


class RealTimeDashboard:
    """Real-time web dashboard for monitoring the multi-agent system."""

    def __init__(self,
                 metrics_collector: MetricsCollector,
                 performance_monitor: PerformanceMonitor,
                 port: int = 8503,
                 refresh_interval: int = 5):
        """
        Initialize the real-time dashboard.

        Args:
            metrics_collector: MetricsCollector instance for data
            performance_monitor: PerformanceMonitor instance for alerts
            port: Port to run the dashboard on
            refresh_interval: Seconds between data refreshes
        """
        self.metrics_collector = metrics_collector
        self.performance_monitor = performance_monitor
        self.port = port
        self.refresh_interval = refresh_interval

        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit is required for the dashboard. Install with: pip install streamlit plotly")

        self.dashboard_running = False

        metrics_collector.logger.info("RealTimeDashboard initialized",
                                    port=port,
                                    refresh_interval=refresh_interval)

    def run_dashboard(self) -> None:
        """Run the Streamlit dashboard."""
        if self.dashboard_running:
            return

        self.dashboard_running = True

        # Configure Streamlit page
        st.set_page_config(
            page_title="Multi-Agent Research System Monitor",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Main dashboard app
        self._create_dashboard_app()

    def _create_dashboard_app(self) -> None:
        """Create the main Streamlit dashboard application."""
        st.title("🔬 Multi-Agent Research System Monitor")
        st.markdown("---")

        # Sidebar configuration
        self._create_sidebar()

        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", "🤖 Agents", "🔧 Tools", "🚨 Alerts"
        ])

        with tab1:
            self._create_overview_tab()

        with tab2:
            self._create_agents_tab()

        with tab3:
            self._create_tools_tab()

        with tab4:
            self._create_alerts_tab()

    def _create_sidebar(self) -> None:
        """Create the sidebar with controls and information."""
        st.sidebar.header("⚙️ Controls")

        # Auto-refresh control
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        if auto_refresh:
            st.sidebar.write(f"🔄 Refreshing every {self.refresh_interval}s")
            time.sleep(self.refresh_interval)
            st.rerun()

        # Session info
        st.sidebar.header("📋 Session Info")
        st.sidebar.write(f"**Session ID:** `{self.metrics_collector.session_id}`")
        st.sidebar.write(f"**Monitoring Status:** {'🟢 Active' if self.performance_monitor.is_monitoring else '🔴 Inactive'}")

        # Quick stats
        st.sidebar.header("📈 Quick Stats")
        system_summary = self.metrics_collector.get_system_summary()
        if system_summary:
            current = system_summary.get('current', {})
            st.sidebar.write(f"**CPU Usage:** {current.get('cpu_percent', 0):.1f}%")
            st.sidebar.write(f"**Memory Usage:** {current.get('memory_percent', 0):.1f}%")
            st.sidebar.write(f"**Active Agents:** {current.get('active_agents', 0)}")

        # Export controls
        st.sidebar.header("💾 Export Data")
        if st.sidebar.button("Export All Metrics"):
            try:
                file_path = self.metrics_collector.export_metrics()
                st.sidebar.success(f"Metrics exported to: `{file_path}`")
            except Exception as e:
                st.sidebar.error(f"Export failed: {e}")

    def _create_overview_tab(self) -> None:
        """Create the overview tab with system-wide metrics."""
        st.header("📊 System Overview")

        # Get performance summary
        perf_summary = self.performance_monitor.get_performance_summary()
        system_summary = self.metrics_collector.get_system_summary()

        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Active Sessions",
                value=perf_summary.get('active_sessions', 0),
                delta=None
            )

        with col2:
            recent_alerts = perf_summary.get('recent_alerts', {})
            st.metric(
                label="Recent Alerts",
                value=recent_alerts.get('total', 0),
                delta=f"⚠️ {recent_alerts.get('critical', 0)} critical"
            )

        with col3:
            metrics = perf_summary.get('performance_metrics', {})
            st.metric(
                label="Total Metrics",
                value=metrics.get('total_agent_metrics', 0) +
                       metrics.get('total_tool_metrics', 0) +
                       metrics.get('total_workflow_metrics', 0),
                delta=None
            )

        with col4:
            if system_summary and system_summary.get('current'):
                current = system_summary['current']
                st.metric(
                    label="System Health",
                    value="🟢 Healthy",
                    delta=f"CPU: {current.get('cpu_percent', 0):.1f}%",
                    delta_color="normal"
                )
            else:
                st.metric(
                    label="System Health",
                    value="🔴 Unknown",
                    delta="No data",
                    delta_color="inverse"
                )

        # System resource charts
        st.subheader("🖥️ System Resources")

        if system_summary and system_summary.get('current'):
            col1, col2 = st.columns(2)

            with col1:
                # CPU and Memory gauge chart
                current = system_summary['current']
                fig = make_subplots(
                    rows=1, cols=2,
                    specs=[[{"type": "indicator"}, {"type": "indicator"}]],
                    subplot_titles=("CPU Usage", "Memory Usage")
                )

                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=current.get('cpu_percent', 0),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "CPU %"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=current.get('memory_percent', 0),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Memory %"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ),
                    row=1, col=2
                )

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Agent activity summary
                agent_summary = self.metrics_collector.get_agent_summary()
                st.write("**Agent Activity Summary**")
                if agent_summary:
                    metrics_by_type = agent_summary.get('metrics_by_type', {})
                    for metric_type, metrics in metrics_by_type.items():
                        st.write(f"- **{metric_type.title()}:** {len(metrics)} activities")
                else:
                    st.write("No agent activity data available")

        # Recent activity timeline
        st.subheader("⏰ Recent Activity Timeline")

        # Get recent metrics for timeline
        recent_time = datetime.now() - timedelta(hours=1)
        recent_agent_metrics = [
            m for m in self.metrics_collector.agent_metrics
            if m.timestamp > recent_time
        ]

        if recent_agent_metrics:
            # Create timeline data
            timeline_data = []
            for metric in recent_agent_metrics[-20:]:  # Last 20 activities
                timeline_data.append({
                    'timestamp': metric.timestamp,
                    'agent': metric.agent_name,
                    'activity': f"{metric.metric_type}: {metric.metric_name}",
                    'value': metric.value
                })

            if timeline_data:
                df = {'timestamp': [d['timestamp'] for d in timeline_data],
                      'agent': [d['agent'] for d in timeline_data],
                      'activity': [d['activity'] for d in timeline_data],
                      'value': [d['value'] for d in timeline_data]}

                fig = px.scatter(
                    df,
                    x='timestamp',
                    y='agent',
                    color='agent',
                    size='value',
                    hover_data=['activity'],
                    title="Recent Agent Activities"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No recent activity data available")

    def _create_agents_tab(self) -> None:
        """Create the agents tab with agent-specific metrics."""
        st.header("🤖 Agent Performance")

        # Agent selection
        agent_names = set(m.agent_name for m in self.metrics_collector.agent_metrics)
        selected_agent = st.selectbox(
            "Select Agent",
            options=list(agent_names) if agent_names else ["No agents"],
            index=0 if agent_names else None
        )

        if selected_agent and selected_agent != "No agents":
            # Get agent summary
            agent_summary = self.metrics_collector.get_agent_summary(selected_agent)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"📊 {selected_agent} Metrics")
                if agent_summary:
                    st.write(f"**Total Activities:** {agent_summary.get('total_metrics', 0)}")

                    metrics_by_type = agent_summary.get('metrics_by_type', {})
                    for metric_type, metrics in metrics_by_type.items():
                        st.write(f"**{metric_type.title()} Activities:** {len(metrics)}")

                    # Performance aggregates
                    perf_aggregates = agent_summary.get('performance_aggregates', {})
                    if perf_aggregates:
                        st.write("**Performance Summary:**")
                        for key, perf in list(perf_aggregates.items())[:5]:  # Show first 5
                            st.write(f"- {perf['metric_name']}: avg {perf['avg']:.2f}")

            with col2:
                st.subheader("📈 Activity Chart")
                # Create activity chart
                recent_time = datetime.now() - timedelta(hours=2)
                recent_metrics = [
                    m for m in self.metrics_collector.agent_metrics
                    if m.agent_name == selected_agent and m.timestamp > recent_time
                ]

                if recent_metrics:
                    # Group by metric type for chart
                    chart_data = {'timestamp': [], 'metric_type': [], 'value': []}
                    for metric in recent_metrics:
                        chart_data['timestamp'].append(metric.timestamp)
                        chart_data['metric_type'].append(f"{metric.metric_type}: {metric.metric_name}")
                        chart_data['value'].append(metric.value)

                    fig = px.line(
                        chart_data,
                        x='timestamp',
                        y='value',
                        color='metric_type',
                        title=f"{selected_agent} Activity Timeline"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No recent activity data for this agent")

            # Recent activities table
            st.subheader("📋 Recent Activities")
            if recent_metrics:
                activity_data = []
                for metric in recent_metrics[-10:]:  # Last 10 activities
                    activity_data.append({
                        'Timestamp': metric.timestamp.strftime("%H:%M:%S"),
                        'Type': metric.metric_type,
                        'Activity': metric.metric_name,
                        'Value': f"{metric.value:.2f}",
                        'Unit': metric.unit
                    })

                st.dataframe(activity_data, use_container_width=True)
            else:
                st.write("No recent activities")

    def _create_tools_tab(self) -> None:
        """Create the tools tab with tool performance metrics."""
        st.header("🔧 Tool Performance")

        # Get tool summary
        tool_summary = self.metrics_collector.get_tool_summary()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Total Tool Executions",
                value=tool_summary.get('total_executions', 0),
                delta=None
            )

        with col2:
            success_rate = tool_summary.get('successful_executions', 0)
            total_exec = tool_summary.get('total_executions', 1)
            rate = (success_rate / total_exec * 100) if total_exec > 0 else 0
            st.metric(
                label="Success Rate",
                value=f"{rate:.1f}%",
                delta=f"{tool_summary.get('failed_executions', 0)} failed"
            )

        with col3:
            avg_time = tool_summary.get('avg_execution_time', 0)
            st.metric(
                label="Avg Execution Time",
                value=f"{avg_time:.2f}s",
                delta=None
            )

        # Tool performance chart
        st.subheader("📊 Tool Performance Comparison")

        performance_aggregates = tool_summary.get('performance_aggregates', {})
        if performance_aggregates:
            # Prepare data for chart
            tool_data = []
            for key, perf in performance_aggregates.items():
                tool_data.append({
                    'Tool': perf['tool_name'],
                    'Agent': perf['agent_name'],
                    'Executions': perf['total_executions'],
                    'Avg Time': perf['avg_execution_time'],
                    'Success Rate': perf['success_rate'] * 100
                })

            if tool_data:
                # Create comparison chart
                fig = px.scatter(
                    tool_data,
                    x='Avg Time',
                    y='Success Rate',
                    size='Executions',
                    color='Tool',
                    hover_data=['Agent'],
                    title="Tool Performance: Execution Time vs Success Rate"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Tool performance table
                st.subheader("📋 Tool Performance Details")
                st.dataframe(tool_data, use_container_width=True)
        else:
            st.write("No tool performance data available")

        # Recent tool executions
        st.subheader("⏰ Recent Tool Executions")

        recent_time = datetime.now() - timedelta(hours=1)
        recent_tool_metrics = [
            m for m in self.metrics_collector.tool_metrics
            if m.timestamp > recent_time
        ]

        if recent_tool_metrics:
            execution_data = []
            for metric in recent_tool_metrics[-15:]:  # Last 15 executions
                execution_data.append({
                    'Timestamp': metric.timestamp.strftime("%H:%M:%S"),
                    'Tool': metric.tool_name,
                    'Agent': metric.agent_name,
                    'Duration': f"{metric.execution_time:.2f}s",
                    'Success': "✅" if metric.success else "❌",
                    'Error': metric.error_type or ""
                })

            st.dataframe(execution_data, use_container_width=True)
        else:
            st.write("No recent tool executions")

    def _create_alerts_tab(self) -> None:
        """Create the alerts tab with performance alerts."""
        st.header("🚨 Performance Alerts")

        # Alert controls
        col1, col2 = st.columns(2)

        with col1:
            alert_type_filter = st.selectbox(
                "Filter by Alert Type",
                options=["All", "Critical", "Warning"],
                index=0
            )

        with col2:
            hours_back = st.slider(
                "Hours to Look Back",
                min_value=1,
                max_value=24,
                value=6,
                step=1
            )

        # Get alerts summary
        filter_type = None if alert_type_filter == "All" else alert_type_filter.lower()
        alerts_summary = self.performance_monitor.get_alerts_summary(
            alert_type=filter_type,
            hours_back=hours_back
        )

        # Alert summary cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Total Alerts",
                value=alerts_summary.get('total_alerts', 0),
                delta=None
            )

        with col2:
            alerts_by_type = alerts_summary.get('alerts_by_type', {})
            critical_count = alerts_by_type.get('critical', 0)
            st.metric(
                label="Critical Alerts",
                value=critical_count,
                delta="🚨" if critical_count > 0 else "✅",
                delta_color="inverse" if critical_count > 0 else "normal"
            )

        with col3:
            warning_count = alerts_by_type.get('warning', 0)
            st.metric(
                label="Warning Alerts",
                value=warning_count,
                delta="⚠️" if warning_count > 0 else "✅",
                delta_color="inverse" if warning_count > 0 else "normal"
            )

        # Alert charts
        if alerts_summary.get('total_alerts', 0) > 0:
            col1, col2 = st.columns(2)

            with col1:
                # Alerts by type
                alerts_by_type = alerts_summary.get('alerts_by_type', {})
                if alerts_by_type:
                    fig = px.pie(
                        values=list(alerts_by_type.values()),
                        names=list(alerts_by_type.keys()),
                        title="Alerts by Type"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Alerts by metric
                alerts_by_metric = alerts_summary.get('alerts_by_metric', {})
                if alerts_by_metric:
                    # Show top 5 metrics with most alerts
                    top_metrics = sorted(alerts_by_metric.items(), key=lambda x: x[1], reverse=True)[:5]
                    metrics_df = {'Metric': [m[0] for m in top_metrics],
                                'Alert Count': [m[1] for m in top_metrics]}

                    fig = px.bar(
                        metrics_df,
                        x='Alert Count',
                        y='Metric',
                        orientation='h',
                        title="Top Alert Sources"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

        # Recent alerts table
        st.subheader("📋 Recent Alerts")

        recent_time = datetime.now() - timedelta(hours=hours_back)
        filter_map = {"critical": "critical", "warning": "warning"}
        recent_alerts = [
            alert for alert in self.performance_monitor.alerts
            if alert.timestamp > recent_time and
            (filter_type is None or alert.alert_type == filter_type)
        ]

        if recent_alerts:
            alert_data = []
            for alert in recent_alerts[-20:]:  # Last 20 alerts
                alert_data.append({
                    'Time': alert.timestamp.strftime("%H:%M:%S"),
                    'Type': alert.alert_type.upper(),
                    'Metric': alert.metric_name,
                    'Agent': alert.agent_name or "System",
                    'Tool': alert.tool_name or "",
                    'Message': alert.message,
                    'Value': f"{alert.current_value:.2f}"
                })

            st.dataframe(alert_data, use_container_width=True)
        else:
            st.write("No alerts found matching the current filters")

        # Most recent alert details
        most_recent = alerts_summary.get('most_recent_alert')
        if most_recent:
            st.subheader("🔍 Most Recent Alert Details")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Time:** {most_recent['timestamp']}")
                st.write(f"**Type:** {most_recent['alert_type'].upper()}")
                st.write(f"**Metric:** {most_recent['metric_name']}")
                st.write(f"**Value:** {most_recent['current_value']}")

            with col2:
                st.write(f"**Agent:** {most_recent['agent_name'] or 'System'}")
                st.write(f"**Tool:** {most_recent['tool_name'] or 'N/A'}")
                st.write(f"**Threshold:** {most_recent['threshold_value']}")
                st.write(f"**Message:** {most_recent['message']}")

    def get_dashboard_url(self) -> str:
        """Get the URL where the dashboard is running."""
        return f"http://localhost:{self.port}"

    def stop_dashboard(self) -> None:
        """Stop the dashboard."""
        self.dashboard_running = False
        self.metrics_collector.logger.info("RealTimeDashboard stopped")


def create_dashboard(metrics_collector: MetricsCollector,
                    performance_monitor: PerformanceMonitor,
                    port: int = 8503) -> RealTimeDashboard:
    """
    Create and configure a real-time dashboard.

    Args:
        metrics_collector: MetricsCollector instance
        performance_monitor: PerformanceMonitor instance
        port: Port to run the dashboard on

    Returns:
        Configured RealTimeDashboard instance
    """
    dashboard = RealTimeDashboard(
        metrics_collector=metrics_collector,
        performance_monitor=performance_monitor,
        port=port
    )

    return dashboard


# Command line interface for running the dashboard
def main():
    """Main function to run the dashboard standalone."""
    import os
    import sys

    # Add parent directory to path for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

    # This would need proper initialization when run standalone
    print("Dashboard requires proper initialization with metrics collector and performance monitor.")
    print("Please run through the main application interface.")


if __name__ == "__main__":
    main()
