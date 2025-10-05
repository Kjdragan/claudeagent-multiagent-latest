"""
Report Generator for comprehensive automated reporting.

This module provides automated report generation capabilities for
performance analytics, compliance, and system insights.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .analytics_engine import AnalyticsEngine
from .audit_trail import AuditTrailManager
from .log_aggregator import LogAggregator


class ReportType(Enum):
    """Types of reports that can be generated."""
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_ANALYSIS = "weekly_analysis"
    MONTHLY_REPORT = "monthly_report"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    COMPLIANCE_REPORT = "compliance_report"
    SECURITY_REPORT = "security_report"
    INCIDENT_REPORT = "incident_report"
    EXECUTIVE_SUMMARY = "executive_summary"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    MARKDOWN = "markdown"


@dataclass
class ReportSection:
    """Individual report section."""
    title: str
    content: Any
    section_type: str  # 'text', 'chart', 'table', 'metrics', 'insights'
    priority: int = 1
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GeneratedReport:
    """Generated report structure."""
    report_id: str
    report_type: ReportType
    title: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    sections: list[ReportSection]
    metadata: dict[str, Any]
    file_paths: dict[str, str]  # Format -> file path mapping


class ReportGenerator:
    """Comprehensive automated report generation system."""

    def __init__(self,
                 session_id: str,
                 reports_dir: str = "reports"):
        """
        Initialize the report generator.

        Args:
            session_id: Session identifier
            reports_dir: Directory to store generated reports
        """
        self.session_id = session_id
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Report templates and configurations
        self.report_templates = self._initialize_templates()
        self.report_schedules = {}

        # Generated reports tracking
        self.generated_reports: list[GeneratedReport] = []

    def _initialize_templates(self) -> dict[ReportType, dict[str, Any]]:
        """Initialize report templates and configurations."""
        return {
            ReportType.DAILY_SUMMARY: {
                'title_template': 'Daily System Summary - {date}',
                'sections': [
                    'system_overview',
                    'performance_metrics',
                    'error_summary',
                    'usage_statistics',
                    'security_alerts'
                ],
                'retention_days': 30
            },
            ReportType.WEEKLY_ANALYSIS: {
                'title_template': 'Weekly Analysis Report - {week_start} to {week_end}',
                'sections': [
                    'executive_summary',
                    'performance_trends',
                    'usage_patterns',
                    'error_analysis',
                    'capacity_planning',
                    'recommendations'
                ],
                'retention_days': 90
            },
            ReportType.MONTHLY_REPORT: {
                'title_template': 'Monthly Performance Report - {month} {year}',
                'sections': [
                    'executive_summary',
                    'detailed_performance_analysis',
                    'compliance_status',
                    'security_assessment',
                    'cost_analysis',
                    'strategic_recommendations'
                ],
                'retention_days': 365
            },
            ReportType.PERFORMANCE_ANALYSIS: {
                'title_template': 'Performance Analysis Report - {date}',
                'sections': [
                    'performance_overview',
                    'response_time_analysis',
                    'throughput_metrics',
                    'error_rates',
                    'bottleneck_identification',
                    'optimization_recommendations'
                ],
                'retention_days': 180
            },
            ReportType.COMPLIANCE_REPORT: {
                'title_template': 'Compliance Report - {standard} - {period}',
                'sections': [
                    'compliance_overview',
                    'policy_adherence',
                    'violation_analysis',
                    'risk_assessment',
                    'remediation_status',
                    'audit_trail_summary'
                ],
                'retention_days': 2555  # 7 years
            },
            ReportType.SECURITY_REPORT: {
                'title_template': 'Security Analysis Report - {date}',
                'sections': [
                    'security_overview',
                    'threat_analysis',
                    'access_control_review',
                    'incident_summary',
                    'vulnerability_assessment',
                    'security_recommendations'
                ],
                'retention_days': 730  # 2 years
            },
            ReportType.INCIDENT_REPORT: {
                'title_template': 'Incident Report - {incident_id} - {date}',
                'sections': [
                    'incident_overview',
                    'timeline_analysis',
                    'impact_assessment',
                    'root_cause_analysis',
                    'lessons_learned',
                    'preventive_measures'
                ],
                'retention_days': 1825  # 5 years
            },
            ReportType.EXECUTIVE_SUMMARY: {
                'title_template': 'Executive Summary - {period}',
                'sections': [
                    'key_metrics',
                    'business_impact',
                    'risk_summary',
                    'strategic_initiatives',
                    'recommendations'
                ],
                'retention_days': 365
            }
        }

    async def generate_report(self,
                             report_type: ReportType,
                             period_start: datetime,
                             period_end: datetime,
                             log_aggregator: LogAggregator | None = None,
                             analytics_engine: AnalyticsEngine | None = None,
                             audit_trail: AuditTrailManager | None = None,
                             custom_config: dict[str, Any] | None = None,
                             formats: list[ReportFormat] = None) -> GeneratedReport:
        """
        Generate a comprehensive report.

        Args:
            report_type: Type of report to generate
            period_start: Start of reporting period
            period_end: End of reporting period
            log_aggregator: Log aggregator instance
            analytics_engine: Analytics engine instance
            audit_trail: Audit trail manager instance
            custom_config: Custom configuration options
            formats: Output formats to generate

        Returns:
            Generated report object
        """
        if formats is None:
            formats = [ReportFormat.JSON, ReportFormat.HTML]

        # Generate report ID
        report_id = f"{report_type.value}_{self.session_id}_{int(period_start.timestamp())}"

        # Get report template
        template = self.report_templates.get(report_type, {})
        custom_config = custom_config or {}

        # Generate report title
        title = self._generate_title(template, report_type, period_start, period_end)

        # Generate report sections
        sections = await self._generate_sections(
            report_type,
            template.get('sections', []),
            period_start,
            period_end,
            log_aggregator,
            analytics_engine,
            audit_trail,
            custom_config
        )

        # Create report metadata
        metadata = {
            'session_id': self.session_id,
            'report_type': report_type.value,
            'generation_duration_seconds': None,
            'data_sources': self._identify_data_sources(log_aggregator, analytics_engine, audit_trail),
            'template_used': template,
            'custom_config': custom_config
        }

        # Create report object
        report = GeneratedReport(
            report_id=report_id,
            report_type=report_type,
            title=title,
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            sections=sections,
            metadata=metadata,
            file_paths={}
        )

        # Generate report files in different formats
        for format_type in formats:
            file_path = await self._export_report(report, format_type)
            report.file_paths[format_type.value] = file_path

        # Track generated report
        self.generated_reports.append(report)

        return report

    async def _generate_sections(self,
                                report_type: ReportType,
                                section_configs: list[str],
                                period_start: datetime,
                                period_end: datetime,
                                log_aggregator: LogAggregator | None,
                                analytics_engine: AnalyticsEngine | None,
                                audit_trail: AuditTrailManager | None,
                                custom_config: dict[str, Any]) -> list[ReportSection]:
        """Generate report sections based on configuration."""
        sections = []

        for section_config in section_configs:
            section = await self._generate_section(
                section_config,
                period_start,
                period_end,
                log_aggregator,
                analytics_engine,
                audit_trail,
                custom_config
            )
            if section:
                sections.append(section)

        return sections

    async def _generate_section(self,
                               section_type: str,
                               period_start: datetime,
                               period_end: datetime,
                               log_aggregator: LogAggregator | None,
                               analytics_engine: AnalyticsEngine | None,
                               audit_trail: AuditTrailManager | None,
                               custom_config: dict[str, Any]) -> ReportSection | None:
        """Generate a specific report section."""
        section_generators = {
            'system_overview': self._generate_system_overview,
            'executive_summary': self._generate_executive_summary,
            'performance_metrics': self._generate_performance_metrics,
            'performance_trends': self._generate_performance_trends,
            'usage_statistics': self._generate_usage_statistics,
            'usage_patterns': self._generate_usage_patterns,
            'error_summary': self._generate_error_summary,
            'error_analysis': self._generate_error_analysis,
            'security_alerts': self._generate_security_alerts,
            'compliance_status': self._generate_compliance_status,
            'recommendations': self._generate_recommendations,
            'capacity_planning': self._generate_capacity_planning,
            'strategic_recommendations': self._generate_strategic_recommendations
        }

        generator = section_generators.get(section_type)
        if generator:
            return await generator(
                period_start, period_end,
                log_aggregator, analytics_engine, audit_trail, custom_config
            )

        return None

    async def _generate_system_overview(self,
                                       period_start: datetime,
                                       period_end: datetime,
                                       log_aggregator: LogAggregator | None,
                                       analytics_engine: AnalyticsEngine | None,
                                       audit_trail: AuditTrailManager | None,
                                       custom_config: dict[str, Any]) -> ReportSection:
        """Generate system overview section."""
        overview_data = {
            'period': {
                'start': period_start.isoformat(),
                'end': period_end.isoformat(),
                'duration_hours': (period_end - period_start).total_seconds() / 3600
            }
        }

        # Get log aggregator stats
        if log_aggregator:
            stats = log_aggregator.get_aggregation_stats()
            overview_data['logs'] = {
                'total_entries': stats['total_entries'],
                'entries_by_source': stats['entries_by_source'],
                'entries_by_level': stats['entries_by_level']
            }

        # Get analytics summary
        if analytics_engine:
            analytics_summary = analytics_engine.get_analytics_summary()
            overview_data['analytics'] = analytics_summary

        # Get audit summary
        if audit_trail:
            audit_summary = audit_trail.get_audit_summary()
            overview_data['audit'] = audit_summary

        return ReportSection(
            title="System Overview",
            content=overview_data,
            section_type="metrics",
            priority=1
        )

    async def _generate_executive_summary(self,
                                         period_start: datetime,
                                         period_end: datetime,
                                         log_aggregator: LogAggregator | None,
                                         analytics_engine: AnalyticsEngine | None,
                                         audit_trail: AuditTrailManager | None,
                                         custom_config: dict[str, Any]) -> ReportSection:
        """Generate executive summary section."""
        summary_data = {
            'key_points': [],
            'metrics_summary': {},
            'business_impact': 'neutral'
        }

        # Analyze performance trends
        if analytics_engine and log_aggregator:
            entries = log_aggregator.get_entries(
                start_time=period_start,
                end_time=period_end
            )

            if entries:
                analysis = await analytics_engine.analyze_logs(
                    entries,
                    ['performance', 'errors', 'trends']
                )

                # Extract key insights
                if 'insights' in analysis:
                    high_impact_insights = [
                        insight for insight in analysis['insights']
                        if insight.get('impact_level') == 'high'
                    ]
                    summary_data['key_points'].extend([
                        insight['description'] for insight in high_impact_insights[:3]
                    ])

                # Performance summary
                if 'performance' in analysis:
                    perf = analysis['performance']
                    summary_data['metrics_summary'] = {
                        'error_rate': perf.get('error_rate', {}).get('error_rate_percent', 0),
                        'avg_response_time': perf.get('response_times', {}).get('avg', 0),
                        'total_requests': perf.get('throughput', {}).get('total_requests', 0)
                    }

                    # Determine business impact
                    error_rate = summary_data['metrics_summary']['error_rate']
                    if error_rate > 10:
                        summary_data['business_impact'] = 'negative'
                    elif error_rate < 1:
                        summary_data['business_impact'] = 'positive'

        return ReportSection(
            title="Executive Summary",
            content=summary_data,
            section_type="insights",
            priority=1
        )

    async def _generate_performance_metrics(self,
                                           period_start: datetime,
                                           period_end: datetime,
                                           log_aggregator: LogAggregator | None,
                                           analytics_engine: AnalyticsEngine | None,
                                           audit_trail: AuditTrailManager | None,
                                           custom_config: dict[str, Any]) -> ReportSection:
        """Generate performance metrics section."""
        metrics_data = {
            'response_times': {},
            'throughput': {},
            'error_rates': {},
            'agent_performance': {}
        }

        if analytics_engine and log_aggregator:
            entries = log_aggregator.get_entries(
                start_time=period_start,
                end_time=period_end
            )

            if entries:
                analysis = await analytics_engine.analyze_logs(entries, ['performance'])
                if 'performance' in analysis:
                    metrics_data = analysis['performance']

        return ReportSection(
            title="Performance Metrics",
            content=metrics_data,
            section_type="metrics",
            priority=2
        )

    async def _generate_error_summary(self,
                                     period_start: datetime,
                                     period_end: datetime,
                                     log_aggregator: LogAggregator | None,
                                     analytics_engine: AnalyticsEngine | None,
                                     audit_trail: AuditTrailManager | None,
                                     custom_config: dict[str, Any]) -> ReportSection:
        """Generate error summary section."""
        error_data = {
            'total_errors': 0,
            'error_rate': 0,
            'errors_by_type': {},
            'errors_by_agent': {},
            'top_errors': []
        }

        if log_aggregator:
            error_entries = log_aggregator.get_entries(
                start_time=period_start,
                end_time=period_end,
                level_filter='ERROR'
            )

            error_data['total_errors'] = len(error_entries)

            # Get total entries for rate calculation
            total_entries = log_aggregator.get_entries(
                start_time=period_start,
                end_time=period_end
            )
            if total_entries:
                error_data['error_rate'] = (len(error_entries) / len(total_entries)) * 100

            # Categorize errors
            from collections import Counter
            error_data['errors_by_agent'] = dict(Counter(
                e.agent_name for e in error_entries if e.agent_name
            ).most_common(10))

            error_data['errors_by_type'] = dict(Counter(
                e.activity_type for e in error_entries if e.activity_type
            ).most_common(10))

            # Top error messages
            error_data['top_errors'] = [
                {'message': e.message, 'count': 1}
                for e in error_entries[:10]
            ]

        return ReportSection(
            title="Error Summary",
            content=error_data,
            section_type="table",
            priority=3
        )

    async def _generate_usage_statistics(self,
                                       period_start: datetime,
                                       period_end: datetime,
                                       log_aggregator: LogAggregator | None,
                                       analytics_engine: AnalyticsEngine | None,
                                       audit_trail: AuditTrailManager | None,
                                       custom_config: dict[str, Any]) -> ReportSection:
        """Generate usage statistics section."""
        usage_data = {
            'total_sessions': 0,
            'total_activities': 0,
            'activity_distribution': {},
            'peak_usage_times': {},
            'agent_utilization': {}
        }

        if log_aggregator:
            entries = log_aggregator.get_entries(
                start_time=period_start,
                end_time=period_end
            )

            usage_data['total_activities'] = len(entries)

            # Session statistics
            sessions = set(e.session_id for e in entries)
            usage_data['total_sessions'] = len(sessions)

            # Activity distribution
            from collections import Counter
            activity_types = Counter(e.activity_type for e in entries if e.activity_type)
            usage_data['activity_distribution'] = dict(activity_types.most_common(10))

            # Agent utilization
            agents = Counter(e.agent_name for e in entries if e.agent_name)
            usage_data['agent_utilization'] = dict(agents.most_common())

        return ReportSection(
            title="Usage Statistics",
            content=usage_data,
            section_type="chart",
            priority=2
        )

    async def _generate_security_alerts(self,
                                       period_start: datetime,
                                       period_end: datetime,
                                       log_aggregator: LogAggregator | None,
                                       analytics_engine: AnalyticsEngine | None,
                                       audit_trail: AuditTrailManager | None,
                                       custom_config: dict[str, Any]) -> ReportSection:
        """Generate security alerts section."""
        security_data = {
            'total_alerts': 0,
            'alert_types': {},
            'high_risk_activities': [],
            'compliance_status': 'compliant'
        }

        if audit_trail:
            # Get security events from audit trail
            security_events = audit_trail.security_events
            period_security_events = [
                event for event in security_events
                if period_start <= datetime.fromisoformat(event['timestamp']) <= period_end
            ]

            security_data['total_alerts'] = len(period_security_events)

            # Categorize security events
            from collections import Counter
            concerns = []
            for event in period_security_events:
                concerns.extend(event.get('concerns', []))

            security_data['alert_types'] = dict(Counter(concerns).most_common())

            # High risk activities
            if security_data['total_alerts'] > 0:
                security_data['compliance_status'] = 'attention_required'

        return ReportSection(
            title="Security Alerts",
            content=security_data,
            section_type="table",
            priority=4
        )

    async def _generate_recommendations(self,
                                       period_start: datetime,
                                       period_end: datetime,
                                       log_aggregator: LogAggregator | None,
                                       analytics_engine: AnalyticsEngine | None,
                                       audit_trail: AuditTrailManager | None,
                                       custom_config: dict[str, Any]) -> ReportSection:
        """Generate recommendations section."""
        recommendations_data = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'estimated_impact': {}
        }

        if analytics_engine and log_aggregator:
            entries = log_aggregator.get_entries(
                start_time=period_start,
                end_time=period_end
            )

            if entries:
                analysis = await analytics_engine.analyze_logs(entries)

                if 'insights' in analysis:
                    for insight in analysis['insights']:
                        priority = insight.get('impact_level', 'medium')
                        recommendation = {
                            'title': insight.get('title', 'Recommendation'),
                            'description': insight.get('description', ''),
                            'recommendations': insight.get('recommendations', [])
                        }

                        if priority == 'high':
                            recommendations_data['high_priority'].append(recommendation)
                        elif priority == 'medium':
                            recommendations_data['medium_priority'].append(recommendation)
                        else:
                            recommendations_data['low_priority'].append(recommendation)

        return ReportSection(
            title="Recommendations",
            content=recommendations_data,
            section_type="insights",
            priority=5
        )

    # Placeholder methods for other section types
    async def _generate_performance_trends(self, *args) -> ReportSection:
        """Generate performance trends section."""
        return ReportSection(
            title="Performance Trends",
            content={"message": "Performance trends analysis coming soon"},
            section_type="chart",
            priority=2
        )

    async def _generate_usage_patterns(self, *args) -> ReportSection:
        """Generate usage patterns section."""
        return ReportSection(
            title="Usage Patterns",
            content={"message": "Usage patterns analysis coming soon"},
            section_type="chart",
            priority=2
        )

    async def _generate_error_analysis(self, *args) -> ReportSection:
        """Generate error analysis section."""
        return ReportSection(
            title="Error Analysis",
            content={"message": "Detailed error analysis coming soon"},
            section_type="table",
            priority=3
        )

    async def _generate_compliance_status(self, *args) -> ReportSection:
        """Generate compliance status section."""
        return ReportSection(
            title="Compliance Status",
            content={"message": "Compliance status analysis coming soon"},
            section_type="metrics",
            priority=4
        )

    async def _generate_capacity_planning(self, *args) -> ReportSection:
        """Generate capacity planning section."""
        return ReportSection(
            title="Capacity Planning",
            content={"message": "Capacity planning analysis coming soon"},
            section_type="insights",
            priority=3
        )

    async def _generate_strategic_recommendations(self, *args) -> ReportSection:
        """Generate strategic recommendations section."""
        return ReportSection(
            title="Strategic Recommendations",
            content={"message": "Strategic recommendations coming soon"},
            section_type="insights",
            priority=5
        )

    def _generate_title(self,
                       template: dict[str, Any],
                       report_type: ReportType,
                       period_start: datetime,
                       period_end: datetime) -> str:
        """Generate report title from template."""
        title_template = template.get('title_template', f'{report_type.value.title()} Report')

        # Format template with date information
        try:
            title = title_template.format(
                date=period_start.strftime('%Y-%m-%d'),
                week_start=period_start.strftime('%Y-%m-%d'),
                week_end=period_end.strftime('%Y-%m-%d'),
                month=period_start.strftime('%B'),
                year=period_start.strftime('%Y'),
                period=f"{period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}"
            )
        except KeyError:
            title = title_template

        return title

    def _identify_data_sources(self,
                              log_aggregator: LogAggregator | None,
                              analytics_engine: AnalyticsEngine | None,
                              audit_trail: AuditTrailManager | None) -> list[str]:
        """Identify available data sources."""
        sources = []
        if log_aggregator:
            sources.append('log_aggregator')
        if analytics_engine:
            sources.append('analytics_engine')
        if audit_trail:
            sources.append('audit_trail')
        return sources

    async def _export_report(self, report: GeneratedReport, format_type: ReportFormat) -> str:
        """Export report to specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.report_id}_{timestamp}.{format_type.value}"
        file_path = self.reports_dir / filename

        if format_type == ReportFormat.JSON:
            await self._export_json(report, file_path)
        elif format_type == ReportFormat.HTML:
            await self._export_html(report, file_path)
        elif format_type == ReportFormat.MARKDOWN:
            await self._export_markdown(report, file_path)
        elif format_type == ReportFormat.CSV:
            await self._export_csv(report, file_path)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        return str(file_path)

    async def _export_json(self, report: GeneratedReport, file_path: Path) -> None:
        """Export report in JSON format."""
        export_data = {
            'report_id': report.report_id,
            'report_type': report.report_type.value,
            'title': report.title,
            'generated_at': report.generated_at.isoformat(),
            'period_start': report.period_start.isoformat(),
            'period_end': report.period_end.isoformat(),
            'metadata': report.metadata,
            'sections': []
        }

        for section in report.sections:
            section_data = {
                'title': section.title,
                'section_type': section.section_type,
                'priority': section.priority,
                'metadata': section.metadata,
                'content': section.content
            }
            export_data['sections'].append(section_data)

        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)

    async def _export_html(self, report: GeneratedReport, file_path: Path) -> None:
        """Export report in HTML format."""
        html_content = self._generate_html_report(report)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_html_report(self, report: GeneratedReport) -> str:
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .section h2 {{ color: #333; margin-top: 0; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f0f0f0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.title}</h1>
                <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Period:</strong> {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}</p>
            </div>
        """

        for section in report.sections:
            html += '<div class="section">'
            html += f'<h2>{section.title}</h2>'

            if section.section_type == 'metrics':
                html += self._render_metrics_html(section.content)
            elif section.section_type == 'table':
                html += self._render_table_html(section.content)
            else:
                html += f'<pre>{json.dumps(section.content, indent=2, default=str)}</pre>'

            html += '</div>'

        html += """
        </body>
        </html>
        """
        return html

    def _render_metrics_html(self, metrics_data: dict[str, Any]) -> str:
        """Render metrics data as HTML."""
        html = '<div class="metrics">'

        for key, value in metrics_data.items():
            if isinstance(value, (int, float)):
                html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
            elif isinstance(value, dict):
                html += f'<div class="metric"><strong>{key}:</strong><ul>'
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        html += f'<li>{sub_key}: {sub_value}</li>'
                html += '</ul></div>'

        html += '</div>'
        return html

    def _render_table_html(self, table_data: dict[str, Any]) -> str:
        """Render table data as HTML."""
        html = '<table>'

        # Simple table rendering
        if isinstance(table_data, dict):
            html += '<tr><th>Metric</th><th>Value</th></tr>'
            for key, value in table_data.items():
                if isinstance(value, (int, float, str)):
                    html += f'<tr><td>{key}</td><td>{value}</td></tr>'

        html += '</table>'
        return html

    async def _export_markdown(self, report: GeneratedReport, file_path: Path) -> None:
        """Export report in Markdown format."""
        markdown_content = self._generate_markdown_report(report)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

    def _generate_markdown_report(self, report: GeneratedReport) -> str:
        """Generate Markdown report content."""
        markdown = f"# {report.title}\n\n"
        markdown += f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        markdown += f"**Period:** {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}\n\n"

        for section in report.sections:
            markdown += f"## {section.title}\n\n"
            if isinstance(section.content, dict):
                for key, value in section.content.items():
                    markdown += f"**{key}:** {value}\n"
            markdown += "\n"

        return markdown

    async def _export_csv(self, report: GeneratedReport, file_path: Path) -> None:
        """Export report in CSV format (simplified)."""
        import csv

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Section', 'Key', 'Value'])

            for section in report.sections:
                if isinstance(section.content, dict):
                    for key, value in section.content.items():
                        if isinstance(value, (int, float, str)):
                            writer.writerow([section.title, key, value])

    def schedule_report(self,
                        report_type: ReportType,
                        schedule: str,  # Cron-like expression
                        recipients: list[str],
                        formats: list[ReportFormat] = None) -> str:
        """
        Schedule automated report generation.

        Args:
            report_type: Type of report to schedule
            schedule: Schedule expression
            recipients: Email recipients for the report
            formats: Output formats

        Returns:
            Schedule ID
        """
        import uuid
        schedule_id = str(uuid.uuid4())

        self.report_schedules[schedule_id] = {
            'report_type': report_type,
            'schedule': schedule,
            'recipients': recipients,
            'formats': formats or [ReportFormat.JSON, ReportFormat.HTML],
            'created_at': datetime.now(),
            'last_run': None,
            'next_run': None,
            'active': True
        }

        return schedule_id

    def get_report_summary(self) -> dict[str, Any]:
        """Get summary of generated reports."""
        return {
            'session_id': self.session_id,
            'total_reports': len(self.generated_reports),
            'reports_by_type': {
                report_type.value: len([r for r in self.generated_reports if r.report_type == report_type])
                for report_type in ReportType
            },
            'active_schedules': len([s for s in self.report_schedules.values() if s['active']]),
            'reports_directory': str(self.reports_dir)
        }
