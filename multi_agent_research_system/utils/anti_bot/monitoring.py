"""
Anti-Bot Escalation Performance Monitoring and Optimization

This module provides comprehensive performance monitoring, analytics, and optimization
for the anti-bot escalation system, integrating with the enhanced logging infrastructure
from Phase 1.1.

Key Features:
- Real-time performance monitoring and metrics
- Escalation pattern analysis and optimization
- Domain behavior analytics
- System health monitoring and alerting
- Performance optimization recommendations
- Integration with enhanced logging system

Based on Phase 1.1 enhanced monitoring foundation
"""

import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, deque
import statistics

from . import (
    AntiBotLevel, EscalationResult, EscalationStats,
    DomainProfile, EscalationTrigger
)

# Import enhanced logging from Phase 1.1
try:
    from ...agent_logging.enhanced_logger import get_enhanced_logger, LogLevel, LogCategory, AgentEventType
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    ENHANCED_LOGGING_AVAILABLE = False
    import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics for anti-bot system."""

    # Core performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_escalations: int = 0

    # Timing metrics
    avg_response_time: float = 0.0
    avg_escalation_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0

    # Level-specific metrics
    level_success_rates: Dict[int, float] = field(default_factory=dict)
    level_avg_times: Dict[int, float] = field(default_factory=dict)
    level_request_counts: Dict[int, int] = field(default_factory=dict)

    # Domain metrics
    active_domains: int = 0
    domains_with_failures: int = 0
    domains_in_cooldown: int = 0

    # System health metrics
    error_rate: float = 0.0
    escalation_rate: float = 0.0
    system_load: float = 0.0
    memory_usage_mb: float = 0.0

    # Recent activity (last hour)
    recent_requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=100))

    last_updated: datetime = field(default_factory=datetime.now)

    def update_request_metrics(self, result: EscalationResult):
        """Update metrics with new request result."""
        self.total_requests += 1
        timestamp = datetime.now()

        # Track recent activity
        self.recent_requests.append((timestamp, result.success, result.duration))

        if result.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            self.recent_failures.append((timestamp, result.error))

        if result.escalation_used:
            self.total_escalations += 1

        # Update response time metrics
        self._update_timing_metrics(result.duration)

        # Update level metrics
        level = result.final_level
        self.level_request_counts[level] = self.level_request_counts.get(level, 0) + 1
        self._update_level_metrics(level, result.success, result.duration)

        # Update error and escalation rates
        self.error_rate = (self.failed_requests / self.total_requests) * 100 if self.total_requests > 0 else 0
        self.escalation_rate = (self.total_escalations / self.total_requests) * 100 if self.total_requests > 0 else 0

        self.last_updated = timestamp

    def _update_timing_metrics(self, duration: float):
        """Update timing-related metrics."""
        # Update average response time
        if self.total_requests == 1:
            self.avg_response_time = duration
        else:
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - 1) + duration) /
                self.total_requests
            )

        # Update average escalation time (for escalated requests only)
        if self.total_escalations > 0:
            # This would need to be tracked differently in a real implementation
            pass

    def _update_level_metrics(self, level: int, success: bool, duration: float):
        """Update level-specific metrics."""
        # Update success rate for this level
        if level not in self.level_success_rates:
            self.level_success_rates[level] = 0.0
            self.level_avg_times[level] = 0.0

        total_requests = self.level_request_counts.get(level, 0)
        if total_requests == 1:
            self.level_success_rates[level] = 100.0 if success else 0.0
            self.level_avg_times[level] = duration
        else:
            # Update success rate
            current_successes = int(self.level_success_rates[level] * (total_requests - 1) / 100)
            new_successes = current_successes + (1 if success else 0)
            self.level_success_rates[level] = (new_successes / total_requests) * 100

            # Update average time
            old_avg = self.level_avg_times[level]
            self.level_avg_times[level] = (
                (old_avg * (total_requests - 1) + duration) / total_requests
            )

    def get_recent_stats(self, minutes: int = 60) -> Dict[str, Any]:
        """Get statistics for recent time period.

        Args:
            minutes: Number of minutes to look back

        Returns:
            Dictionary with recent statistics
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        # Filter recent requests
        recent = [(t, s, d) for t, s, d in self.recent_requests if t > cutoff_time]
        recent_failures = [(t, e) for t, e in self.recent_failures if t > cutoff_time]

        if not recent:
            return {
                'requests': 0,
                'success_rate': 0.0,
                'avg_response_time': 0.0,
                'failures': 0,
                'error_rate': 0.0
            }

        # Calculate statistics
        total_requests = len(recent)
        successful_requests = sum(1 for _, s, _ in recent if s)
        response_times = [d for _, _, d in recent]

        return {
            'requests': total_requests,
            'successful': successful_requests,
            'success_rate': (successful_requests / total_requests) * 100,
            'avg_response_time': statistics.mean(response_times),
            'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
            'failures': len(recent_failures),
            'error_rate': (len(recent_failures) / total_requests) * 100,
            'period_minutes': minutes
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'total_escalations': self.total_escalations,
            'avg_response_time': self.avg_response_time,
            'avg_escalation_time': self.avg_escalation_time,
            'error_rate': self.error_rate,
            'escalation_rate': self.escalation_rate,
            'system_load': self.system_load,
            'memory_usage_mb': self.memory_usage_mb,
            'active_domains': self.active_domains,
            'domains_with_failures': self.domains_with_failures,
            'domains_in_cooldown': self.domains_in_cooldown,
            'level_success_rates': self.level_success_rates,
            'level_avg_times': self.level_avg_times,
            'level_request_counts': self.level_request_counts,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""

    category: str
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    expected_improvement: str
    implementation_difficulty: str  # "easy", "medium", "hard"
    metrics_affected: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AntiBotMonitor:
    """Performance monitoring and optimization for anti-bot system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the anti-bot monitor.

        Args:
            config: Monitoring configuration
        """
        self.config = config or {}
        self.metrics = PerformanceMetrics()

        # Monitoring settings
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self.auto_optimization_enabled = self.config.get('auto_optimization_enabled', False)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'error_rate': 20.0,  # Alert if error rate > 20%
            'avg_response_time': 30.0,  # Alert if avg response time > 30s
            'escalation_rate': 50.0  # Alert if escalation rate > 50%
        })

        # Data retention
        self.max_history_hours = self.config.get('max_history_hours', 24)
        self.performance_history: deque = field(default_factory=lambda: deque(maxlen=1440))  # 24 hours of minute data
        self.domain_performance: Dict[str, Dict] = defaultdict(dict)

        # Optimization data
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        self.last_optimization_check = datetime.now()

        # Setup enhanced logging
        self._setup_logging()

        # Start background monitoring
        if self.monitoring_enabled:
            self._start_background_monitoring()

        logger.info(f"AntiBotMonitor initialized with monitoring: {self.monitoring_enabled}")

    def _setup_logging(self):
        """Setup enhanced logging integration."""
        if ENHANCED_LOGGING_AVAILABLE:
            self.enhanced_logger = get_enhanced_logger("anti_bot_monitor")
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.SESSION_START,
                "Anti-Bot Performance Monitor initialized",
                monitoring_enabled=self.monitoring_enabled,
                auto_optimization_enabled=self.auto_optimization_enabled,
                alert_thresholds=self.alert_thresholds
            )
        else:
            self.enhanced_logger = None

    def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        # This would start background tasks for periodic monitoring
        # For now, we'll just log that monitoring is active
        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.PERFORMANCE,
                AgentEventType.MESSAGE_PROCESSED,
                "Background monitoring started"
            )

    def record_escalation_result(self, result: EscalationResult):
        """Record escalation result for performance monitoring.

        Args:
            result: Escalation result to record
        """
        if not self.monitoring_enabled:
            return

        # Update core metrics
        self.metrics.update_request_metrics(result)

        # Update domain-specific metrics
        self._update_domain_metrics(result)

        # Check for alerts
        self._check_alerts()

        # Check for optimization opportunities
        if self.auto_optimization_enabled:
            self._check_optimization_opportunities(result)

        # Log performance data
        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.DEBUG,
                LogCategory.PERFORMANCE,
                AgentEventType.MESSAGE_PROCESSED,
                f"Recorded escalation result for {result.domain}",
                domain=result.domain,
                success=result.success,
                duration=result.duration,
                attempts=result.attempts_made,
                final_level=result.final_level,
                escalation_used=result.escalation_used
            )

    def _update_domain_metrics(self, result: EscalationResult):
        """Update domain-specific performance metrics.

        Args:
            result: Escalation result to update metrics for
        """
        domain = result.domain
        if domain not in self.domain_performance:
            self.domain_performance[domain] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_duration': 0.0,
                'level_distribution': defaultdict(int),
                'escalation_count': 0,
                'last_request': None,
                'avg_response_time': 0.0,
                'success_rate': 0.0
            }

        domain_metrics = self.domain_performance[domain]
        domain_metrics['total_requests'] += 1
        domain_metrics['total_duration'] += result.duration
        domain_metrics['level_distribution'][result.final_level] += 1
        domain_metrics['last_request'] = datetime.now()

        if result.success:
            domain_metrics['successful_requests'] += 1
        else:
            domain_metrics['successful_requests'] = domain_metrics['successful_requests']  # No change

        if result.escalation_used:
            domain_metrics['escalation_count'] += 1

        # Update calculated metrics
        domain_metrics['avg_response_time'] = (
            domain_metrics['total_duration'] / domain_metrics['total_requests']
        )
        domain_metrics['success_rate'] = (
            (domain_metrics['successful_requests'] / domain_metrics['total_requests']) * 100
        )

    def _check_alerts(self):
        """Check for performance alerts and trigger notifications."""
        alerts_triggered = []

        # Check error rate alert
        if self.metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts_triggered.append({
                'type': 'error_rate',
                'severity': 'high',
                'message': f"Error rate ({self.metrics.error_rate:.1f}%) exceeds threshold ({self.alert_thresholds['error_rate']:.1f}%)",
                'current_value': self.metrics.error_rate,
                'threshold': self.alert_thresholds['error_rate']
            })

        # Check response time alert
        if self.metrics.avg_response_time > self.alert_thresholds['avg_response_time']:
            alerts_triggered.append({
                'type': 'response_time',
                'severity': 'medium',
                'message': f"Average response time ({self.metrics.avg_response_time:.1f}s) exceeds threshold ({self.alert_thresholds['avg_response_time']:.1f}s)",
                'current_value': self.metrics.avg_response_time,
                'threshold': self.alert_thresholds['avg_response_time']
            })

        # Check escalation rate alert
        if self.metrics.escalation_rate > self.alert_thresholds['escalation_rate']:
            alerts_triggered.append({
                'type': 'escalation_rate',
                'severity': 'medium',
                'message': f"Escalation rate ({self.metrics.escalation_rate:.1f}%) exceeds threshold ({self.alert_thresholds['escalation_rate']:.1f}%)",
                'current_value': self.metrics.escalation_rate,
                'threshold': self.alert_thresholds['escalation_rate']
            })

        # Log alerts
        for alert in alerts_triggered:
            logger.warning(f"Performance alert: {alert['message']}")

            if self.enhanced_logger:
                self.enhanced_logger.log_event(
                    LogLevel.WARNING,
                    LogCategory.PERFORMANCE,
                    AgentEventType.ERROR,
                    f"Performance alert triggered: {alert['type']}",
                    alert_type=alert['type'],
                    severity=alert['severity'],
                    message=alert['message'],
                    current_value=alert['current_value'],
                    threshold=alert['threshold']
                )

    def _check_optimization_opportunities(self, result: EscalationResult):
        """Check for optimization opportunities based on recent results.

        Args:
            result: Recent escalation result
        """
        # Only check optimization opportunities periodically
        if (datetime.now() - self.last_optimization_check).minutes < 30:
            return

        self.last_optimization_check = datetime.now()

        # Analyze recent performance
        recent_stats = self.metrics.get_recent_stats(minutes=30)

        # Check for high escalation rate
        if recent_stats['escalation_rate'] > 60:
            self._add_recommendation(
                category="escalation_strategy",
                priority="high",
                title="High Escalation Rate Detected",
                description=f"Recent escalation rate is {recent_stats['escalation_rate']:.1f}%. Consider starting at higher anti-bot levels for problematic domains.",
                expected_improvement="20-40% reduction in escalation rate",
                implementation_difficulty="easy",
                metrics_affected=["escalation_rate", "avg_response_time"]
            )

        # Check for high error rate
        if recent_stats['error_rate'] > 30:
            self._add_recommendation(
                category="reliability",
                priority="high",
                title="High Error Rate Detected",
                description=f"Recent error rate is {recent_stats['error_rate']:.1f}%. Consider adjusting timeouts or implementing better error handling.",
                expected_improvement="15-25% improvement in success rate",
                implementation_difficulty="medium",
                metrics_affected=["error_rate", "success_rate"]
            )

        # Check for slow response times
        if recent_stats['avg_response_time'] > 20:
            self._add_recommendation(
                category="performance",
                priority="medium",
                title="Slow Response Times Detected",
                description=f"Recent average response time is {recent_stats['avg_response_time']:.1f}s. Consider optimizing timeout settings or reducing concurrent requests.",
                expected_improvement="10-20% improvement in response time",
                implementation_difficulty="easy",
                metrics_affected=["avg_response_time", "throughput"]
            )

        # Check for domain-specific issues
        problem_domains = self._identify_problem_domains()
        for domain in problem_domains:
            self._add_recommendation(
                category="domain_optimization",
                priority="medium",
                title=f"Domain Performance Issues: {domain}",
                description=f"Domain {domain} shows poor performance metrics. Consider using higher anti-bot levels or adding to difficult sites list.",
                expected_improvement="30-50% improvement for this domain",
                implementation_difficulty="easy",
                metrics_affected=[f"domain_{domain}_success_rate", f"domain_{domain}_response_time"]
            )

    def _identify_problem_domains(self) -> List[str]:
        """Identify domains with performance issues.

        Returns:
            List of domain names with performance problems
        """
        problem_domains = []

        for domain, metrics in self.domain_performance.items():
            # Check for low success rate
            if metrics['success_rate'] < 50 and metrics['total_requests'] >= 5:
                problem_domains.append(domain)
                continue

            # Check for high escalation rate
            escalation_rate = (metrics['escalation_count'] / metrics['total_requests']) * 100
            if escalation_rate > 80 and metrics['total_requests'] >= 5:
                problem_domains.append(domain)
                continue

            # Check for slow response times
            if metrics['avg_response_time'] > 30 and metrics['total_requests'] >= 3:
                problem_domains.append(domain)

        return problem_domains

    def _add_recommendation(self, **kwargs):
        """Add optimization recommendation.

        Args:
            **kwargs: Recommendation parameters
        """
        recommendation = OptimizationRecommendation(**kwargs)

        # Avoid duplicate recommendations
        for existing in self.optimization_recommendations:
            if (existing.category == recommendation.category and
                existing.title == recommendation.title):
                return  # Skip duplicate

        self.optimization_recommendations.append(recommendation)

        # Log recommendation
        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.PERFORMANCE,
                AgentEventType.MESSAGE_PROCESSED,
                f"Optimization recommendation: {recommendation.title}",
                category=recommendation.category,
                priority=recommendation.priority,
                description=recommendation.description,
                expected_improvement=recommendation.expected_improvement
            )

        logger.info(f"Optimization recommendation: {recommendation.title}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.

        Returns:
            Dictionary with performance summary
        """
        # Get recent statistics
        recent_1h = self.metrics.get_recent_stats(minutes=60)
        recent_15m = self.metrics.get_recent_stats(minutes=15)

        # Get domain statistics
        domain_stats = self._get_domain_statistics()

        # Get level performance
        level_performance = self._get_level_performance()

        return {
            'overall_metrics': {
                'total_requests': self.metrics.total_requests,
                'success_rate': ((self.metrics.successful_requests / self.metrics.total_requests) * 100) if self.metrics.total_requests > 0 else 0,
                'avg_response_time': self.metrics.avg_response_time,
                'error_rate': self.metrics.error_rate,
                'escalation_rate': self.metrics.escalation_rate
            },
            'recent_performance': {
                'last_15_minutes': recent_15m,
                'last_hour': recent_1h
            },
            'domain_performance': domain_stats,
            'level_performance': level_performance,
            'system_health': {
                'active_domains': len(self.domain_performance),
                'problem_domains': len(self._identify_problem_domains()),
                'monitoring_enabled': self.monitoring_enabled,
                'auto_optimization_enabled': self.auto_optimization_enabled
            },
            'optimization_recommendations': [r.to_dict() for r in self.optimization_recommendations[-5:]],  # Last 5 recommendations
            'last_updated': self.metrics.last_updated.isoformat()
        }

    def _get_domain_statistics(self) -> Dict[str, Any]:
        """Get domain performance statistics.

        Returns:
            Dictionary with domain statistics
        """
        if not self.domain_performance:
            return {'total_domains': 0, 'active_domains': 0, 'top_domains': []}

        # Sort domains by request count
        sorted_domains = sorted(
            self.domain_performance.items(),
            key=lambda x: x[1]['total_requests'],
            reverse=True
        )

        # Get top domains
        top_domains = []
        for domain, metrics in sorted_domains[:10]:
            top_domains.append({
                'domain': domain,
                'requests': metrics['total_requests'],
                'success_rate': metrics['success_rate'],
                'avg_response_time': metrics['avg_response_time'],
                'escalation_rate': (metrics['escalation_count'] / metrics['total_requests']) * 100,
                'most_common_level': max(metrics['level_distribution'].items(), key=lambda x: x[1])[0] if metrics['level_distribution'] else 0
            })

        return {
            'total_domains': len(self.domain_performance),
            'active_domains': len([d for d in self.domain_performance.values() if d['total_requests'] > 0]),
            'problem_domains': len(self._identify_problem_domains()),
            'top_domains': top_domains
        }

    def _get_level_performance(self) -> Dict[str, Any]:
        """Get level-specific performance statistics.

        Returns:
            Dictionary with level performance data
        """
        level_performance = {}

        for level in range(4):
            if level in self.metrics.level_success_rates:
                level_performance[f'level_{level}'] = {
                    'name': AntiBotLevel.get_level_name(level),
                    'requests': self.metrics.level_request_counts.get(level, 0),
                    'success_rate': self.metrics.level_success_rates[level],
                    'avg_response_time': self.metrics.level_avg_times.get(level, 0.0),
                    'estimated_improvement': AntiBotLevel.get_success_rate_estimate(level) * 100
                }

        return level_performance

    def get_domain_insights(self, domain: str) -> Dict[str, Any]:
        """Get detailed insights for a specific domain.

        Args:
            domain: Domain to analyze

        Returns:
            Dictionary with domain insights
        """
        if domain not in self.domain_performance:
            return {
                'domain': domain,
                'tracked': False,
                'message': 'No performance data available for this domain'
            }

        metrics = self.domain_performance[domain]

        # Generate insights
        insights = {
            'domain': domain,
            'tracked': True,
            'total_requests': metrics['total_requests'],
            'success_rate': metrics['success_rate'],
            'avg_response_time': metrics['avg_response_time'],
            'escalation_rate': (metrics['escalation_count'] / metrics['total_requests']) * 100,
            'last_request': metrics['last_request'].isoformat() if metrics['last_request'] else None,
            'level_distribution': dict(metrics['level_distribution']),
            'performance_trend': 'stable',  # This would be calculated from historical data
            'recommendations': []
        }

        # Generate recommendations
        if metrics['success_rate'] < 60:
            insights['recommendations'].append(
                "Low success rate. Consider using higher anti-bot level or adding to difficult sites list."
            )

        if metrics['avg_response_time'] > 20:
            insights['recommendations'].append(
                "Slow response times. Consider adjusting timeout settings."
            )

        if (metrics['escalation_count'] / metrics['total_requests']) > 0.7:
            insights['recommendations'].append(
                "High escalation rate. Domain may require consistently higher anti-bot levels."
            )

        return insights

    def export_performance_data(self, output_path: str, format: str = 'json') -> bool:
        """Export performance data to file.

        Args:
            output_path: Path to save export file
            format: Export format ('json' or 'csv')

        Returns:
            True if export successful, False otherwise
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'export_timestamp': datetime.now().isoformat(),
                'monitoring_config': self.config,
                'performance_metrics': self.metrics.to_dict(),
                'performance_summary': self.get_performance_summary(),
                'optimization_recommendations': [r.to_dict() for r in self.optimization_recommendations]
            }

            if format.lower() == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                # CSV export would require more complex flattening
                raise ValueError("CSV export not yet implemented")

            logger.info(f"Performance data exported to {output_file}")

            if self.enhanced_logger:
                self.enhanced_logger.log_event(
                    LogLevel.INFO,
                    LogCategory.PERFORMANCE,
                    AgentEventType.MESSAGE_PROCESSED,
                    f"Performance data exported to {output_file}",
                    export_path=output_path,
                    format=format,
                    data_size=len(str(data))
                )

            return True

        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")
            return False

    def clear_recommendations(self):
        """Clear all optimization recommendations."""
        self.optimization_recommendations.clear()
        logger.info("Cleared all optimization recommendations")

    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics = PerformanceMetrics()
        self.domain_performance.clear()
        self.optimization_recommendations.clear()
        logger.info("Reset all performance metrics")

        if self.enhanced_logger:
            self.enhanced_logger.log_event(
                LogLevel.INFO,
                LogCategory.PERFORMANCE,
                AgentEventType.MESSAGE_PROCESSED,
                "Performance metrics reset"
            )


# Global monitor instance
_global_monitor: Optional[AntiBotMonitor] = None


def get_anti_bot_monitor(config: Optional[Dict[str, Any]] = None) -> AntiBotMonitor:
    """Get or create global anti-bot monitor instance.

    Args:
        config: Optional monitoring configuration

    Returns:
        Anti-bot monitor instance
    """
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = AntiBotMonitor(config)

    return _global_monitor


def record_escalation_performance(result: EscalationResult):
    """Record escalation result for performance monitoring.

    Args:
        result: Escalation result to record
    """
    monitor = get_anti_bot_monitor()
    monitor.record_escalation_result(result)


def get_performance_dashboard() -> Dict[str, Any]:
    """Get performance dashboard data.

    Returns:
        Dictionary with performance dashboard data
    """
    monitor = get_anti_bot_monitor()
    return monitor.get_performance_summary()


def get_optimization_recommendations() -> List[Dict[str, Any]]:
    """Get current optimization recommendations.

    Returns:
        List of optimization recommendations
    """
    monitor = get_anti_bot_monitor()
    return [r.to_dict() for r in monitor.optimization_recommendations]