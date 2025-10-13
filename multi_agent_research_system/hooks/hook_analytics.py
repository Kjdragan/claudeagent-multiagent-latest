"""
Hook Analytics and Optimization System for Multi-Agent Research System

Phase 3.1.5: Hook Analytics and Optimization

Provides comprehensive analytics for hook performance monitoring, bottleneck detection,
and automated optimization recommendations. This system analyzes hook execution patterns,
identifies performance issues, and suggests optimizations to improve system efficiency.

Features:
- Real-time hook performance analytics
- Bottleneck detection and analysis
- Automated optimization recommendations
- Performance trend analysis
- Hook efficiency scoring
- Resource usage optimization
- Predictive performance modeling
"""

import asyncio
import json
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

# Import system components
try:
    from ..core.logging_config import get_logger
    from ..utils.message_processing.main import MessageProcessor, MessageType
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    MessageType = None
    MessageProcessor = None

from .comprehensive_hooks import HookCategory, HookExecutionResult


class OptimizationType(Enum):
    """Types of hook optimizations."""
    PRIORITY_ADJUSTMENT = "priority_adjustment"
    PARALLEL_EXECUTION = "parallel_execution"
    CACHING_IMPLEMENTATION = "caching_implementation"
    RESOURCE_ALLOCATION = "resource_allocation"
    FREQUENCY_TUNING = "frequency_tuning"
    CONDITIONAL_EXECUTION = "conditional_execution"
    BATCH_PROCESSING = "batch_processing"
    ASYNC_OPTIMIZATION = "async_optimization"


class PerformanceLevel(Enum):
    """Performance classification levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class HookPerformanceMetrics:
    """Comprehensive performance metrics for a hook."""
    hook_name: str
    hook_type: str
    category: HookCategory

    # Execution metrics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    median_execution_time: float = 0.0
    p95_execution_time: float = 0.0
    p99_execution_time: float = 0.0

    # Performance trends
    execution_times: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_performance: deque = field(default_factory=lambda: deque(maxlen=20))
    performance_trend: str = "stable"  # improving, degrading, stable

    # Resource usage
    average_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    cpu_usage: float = 0.0

    # Efficiency metrics
    success_rate: float = 0.0
    efficiency_score: float = 0.0
    performance_level: PerformanceLevel = PerformanceLevel.AVERAGE

    # Timestamps
    last_execution: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    metrics_updated: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation for a hook."""
    hook_name: str
    hook_type: str
    optimization_type: OptimizationType
    priority: int  # 1-10, 1 being highest
    description: str
    expected_improvement: str
    implementation_complexity: str  # low, medium, high
    current_impact: str
    recommended_action: str
    code_suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks."""
    bottleneck_type: str
    affected_hooks: List[str]
    severity: str  # low, medium, high, critical
    impact_description: str
    root_cause: str
    recommendations: List[OptimizationRecommendation]
    estimated_impact: str
    created_at: datetime = field(default_factory=datetime.now)


class HookAnalyticsEngine:
    """
    Advanced analytics engine for hook performance monitoring and optimization.

    Provides comprehensive analysis of hook execution patterns, identifies bottlenecks,
    and generates actionable optimization recommendations.
    """

    def __init__(self,
                 analysis_window_minutes: int = 60,
                 performance_history_size: int = 1000,
                 optimization_threshold: float = 0.1):
        """
        Initialize hook analytics engine.

        Args:
            analysis_window_minutes: Time window for performance analysis
            performance_history_size: Size of performance history to retain
            optimization_threshold: Threshold for triggering optimization recommendations
        """
        self.logger = get_logger("hook_analytics_engine")
        self.analysis_window = timedelta(minutes=analysis_window_minutes)
        self.performance_history_size = performance_history_size
        self.optimization_threshold = optimization_threshold

        # Performance data storage
        self.hook_metrics: Dict[str, HookPerformanceMetrics] = {}
        self.execution_history: deque = deque(maxlen=performance_history_size)
        self.performance_snapshots: deque = deque(maxlen=100)  # Store periodic snapshots

        # Analytics configuration
        self.enable_trend_analysis = True
        self.enable_bottleneck_detection = True
        self.enable_optimization_recommendations = True
        self.enable_predictive_analysis = True

        # Message processing for notifications
        self.message_processor: Optional[MessageProcessor] = None
        if MessageProcessor:
            try:
                self.message_processor = MessageProcessor()
            except Exception as e:
                self.logger.warning(f"Failed to initialize message processor: {e}")

        # Background analytics tasks
        self.analytics_task: Optional[asyncio.Task] = None
        self._running = False

        # Analytics results cache
        self._bottleneck_cache: Optional[BottleneckAnalysis] = None
        self._recommendations_cache: List[OptimizationRecommendation] = []
        self._last_analysis_time: Optional[datetime] = None

    async def start(self):
        """Start the analytics engine."""
        if self._running:
            return

        self._running = True
        self.logger.info("Starting hook analytics engine")

        # Start background analytics task
        self.analytics_task = asyncio.create_task(self._analytics_loop())

        self.logger.info("Hook analytics engine started")

    async def stop(self):
        """Stop the analytics engine."""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping hook analytics engine...")

        if self.analytics_task:
            self.analytics_task.cancel()
            try:
                await self.analytics_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Hook analytics engine stopped")

    def record_hook_execution(self, result: HookExecutionResult):
        """
        Record a hook execution result for analytics.

        Args:
            result: Hook execution result to record
        """
        hook_key = f"{result.hook_type}:{result.hook_name}"
        current_time = datetime.now()

        # Initialize metrics if not exists
        if hook_key not in self.hook_metrics:
            self.hook_metrics[hook_key] = HookPerformanceMetrics(
                hook_name=result.hook_name,
                hook_type=result.hook_type,
                category=result.hook_category
            )

        metrics = self.hook_metrics[hook_key]

        # Update basic metrics
        metrics.total_executions += 1
        if result.success:
            metrics.successful_executions += 1
        else:
            metrics.failed_executions += 1
            metrics.last_failure = current_time

        # Update execution time metrics
        execution_time = result.execution_time
        metrics.execution_times.append(execution_time)
        metrics.recent_performance.append(execution_time)

        # Update min/max
        metrics.min_execution_time = min(metrics.min_execution_time, execution_time)
        metrics.max_execution_time = max(metrics.max_execution_time, execution_time)

        # Update last execution
        metrics.last_execution = current_time

        # Store in execution history
        self.execution_history.append({
            'timestamp': current_time,
            'hook_key': hook_key,
            'execution_time': execution_time,
            'success': result.success,
            'result': result
        })

        # Trigger periodic analysis
        if len(self.execution_history) % 50 == 0:
            asyncio.create_task(self._perform_analysis())

    async def _analytics_loop(self):
        """Background analytics processing loop."""
        self.logger.info("Starting analytics processing loop")

        while self._running:
            try:
                await self._perform_analysis()
                await asyncio.sleep(300)  # Analyze every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Analytics loop error: {e}")
                await asyncio.sleep(60)

    async def _perform_analysis(self):
        """Perform comprehensive performance analysis."""
        current_time = datetime.now()

        # Update metrics calculations
        await self._update_performance_metrics()

        # Perform trend analysis
        if self.enable_trend_analysis:
            await self._analyze_performance_trends()

        # Detect bottlenecks
        if self.enable_bottleneck_detection:
            await self._detect_bottlenecks()

        # Generate optimization recommendations
        if self.enable_optimization_recommendations:
            await self._generate_optimization_recommendations()

        # Create performance snapshot
        snapshot = await self._create_performance_snapshot(current_time)
        self.performance_snapshots.append(snapshot)

        self._last_analysis_time = current_time
        self.logger.debug("Performance analysis completed")

    async def _update_performance_metrics(self):
        """Update calculated performance metrics for all hooks."""
        for hook_key, metrics in self.hook_metrics.items():
            if not metrics.execution_times:
                continue

            # Calculate statistical measures
            execution_times = list(metrics.execution_times)
            metrics.average_execution_time = statistics.mean(execution_times)
            metrics.median_execution_time = statistics.median(execution_times)

            # Calculate percentiles
            sorted_times = sorted(execution_times)
            n = len(sorted_times)
            if n >= 20:
                metrics.p95_execution_time = sorted_times[int(n * 0.95)]
                metrics.p99_execution_time = sorted_times[int(n * 0.99)]

            # Calculate success rate
            if metrics.total_executions > 0:
                metrics.success_rate = metrics.successful_executions / metrics.total_executions

            # Calculate efficiency score (0-100)
            # Factors: success rate, execution time consistency, recent performance
            time_consistency = 1.0 - (statistics.stdev(execution_times) / metrics.average_execution_time) if len(execution_times) > 1 else 1.0
            recent_avg = statistics.mean(list(metrics.recent_performance)) if metrics.recent_performance else metrics.average_execution_time
            recent_performance_score = max(0, 1.0 - (recent_avg / (metrics.average_execution_time * 1.5)))

            metrics.efficiency_score = (
                metrics.success_rate * 0.4 +
                min(1.0, time_consistency) * 0.3 +
                recent_performance_score * 0.3
            ) * 100

            # Determine performance level
            if metrics.efficiency_score >= 90:
                metrics.performance_level = PerformanceLevel.EXCELLENT
            elif metrics.efficiency_score >= 75:
                metrics.performance_level = PerformanceLevel.GOOD
            elif metrics.efficiency_score >= 60:
                metrics.performance_level = PerformanceLevel.AVERAGE
            elif metrics.efficiency_score >= 40:
                metrics.performance_level = PerformanceLevel.POOR
            else:
                metrics.performance_level = PerformanceLevel.CRITICAL

            metrics.metrics_updated = datetime.now()

    async def _analyze_performance_trends(self):
        """Analyze performance trends for hooks."""
        for hook_key, metrics in self.hook_metrics.items():
            if len(metrics.recent_performance) < 10:
                continue

            recent_times = list(metrics.recent_performance)
            older_times = list(metrics.execution_times)[-20:-10] if len(metrics.execution_times) >= 20 else []

            if not older_times:
                continue

            recent_avg = statistics.mean(recent_times)
            older_avg = statistics.mean(older_times)

            # Determine trend
            change_ratio = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

            if change_ratio < -0.1:  # 10% improvement
                metrics.performance_trend = "improving"
            elif change_ratio > 0.1:  # 10% degradation
                metrics.performance_trend = "degrading"
            else:
                metrics.performance_trend = "stable"

            # Log significant changes
            if abs(change_ratio) > 0.2:  # 20% change
                self.logger.info(f"Significant performance trend detected for {hook_key}: "
                               f"{metrics.performance_trend} ({change_ratio:+.1%})")

    async def _detect_bottlenecks(self):
        """Detect performance bottlenecks in hook execution."""
        bottlenecks = []

        # Find slow hooks
        for hook_key, metrics in self.hook_metrics.items():
            if metrics.average_execution_time > 5.0:  # 5 second threshold
                bottlenecks.append({
                    'type': 'slow_execution',
                    'hook_key': hook_key,
                    'severity': 'high' if metrics.average_execution_time > 10.0 else 'medium',
                    'impact': f"Average execution time: {metrics.average_execution_time:.2f}s"
                })

            # Find high failure rate hooks
            if metrics.success_rate < 0.9 and metrics.total_executions > 10:
                bottlenecks.append({
                    'type': 'high_failure_rate',
                    'hook_key': hook_key,
                    'severity': 'high' if metrics.success_rate < 0.7 else 'medium',
                    'impact': f"Success rate: {metrics.success_rate:.1%}"
                })

            # Find inconsistent performance
            if len(metrics.execution_times) > 20:
                cv = statistics.stdev(metrics.execution_times) / metrics.average_execution_time
                if cv > 0.5:  # High coefficient of variation
                    bottlenecks.append({
                        'type': 'inconsistent_performance',
                        'hook_key': hook_key,
                        'severity': 'medium',
                        'impact': f"Performance variation: {cv:.2f}"
                    })

        # Group bottlenecks by type and create analysis
        if bottlenecks:
            bottleneck_groups = defaultdict(list)
            for bottleneck in bottlenecks:
                bottleneck_groups[bottleneck['type']].append(bottleneck)

            for bottleneck_type, group in bottleneck_groups.items():
                analysis = await self._create_bottleneck_analysis(bottleneck_type, group)
                self._bottleneck_cache = analysis

                # Send notification if severe
                if analysis.severity in ['high', 'critical']:
                    await self._send_bottleneck_notification(analysis)

    async def _create_bottleneck_analysis(self, bottleneck_type: str, bottlenecks: List[Dict]) -> BottleneckAnalysis:
        """Create bottleneck analysis from detected issues."""
        affected_hooks = [b['hook_key'] for b in bottlenecks]
        max_severity = max(b['severity'] for b in bottlenecks)

        # Determine root cause and recommendations
        root_cause, recommendations = await self._analyze_bottleneck_root_cause(bottleneck_type, bottlenecks)

        # Estimate impact
        total_impact = len(bottlenecks)
        impact_level = "critical" if total_impact > 5 else "high" if total_impact > 3 else "medium"

        return BottleneckAnalysis(
            bottleneck_type=bottleneck_type,
            affected_hooks=affected_hooks,
            severity=max_severity,
            impact_description=f"{total_impact} hooks affected",
            root_cause=root_cause,
            recommendations=recommendations,
            estimated_impact=impact_level
        )

    async def _analyze_bottleneck_root_cause(self, bottleneck_type: str, bottlenecks: List[Dict]) -> Tuple[str, List[OptimizationRecommendation]]:
        """Analyze root cause of bottlenecks and generate recommendations."""
        recommendations = []

        if bottleneck_type == "slow_execution":
            root_cause = "Hooks taking excessive time to execute"
            recommendations.extend([
                OptimizationRecommendation(
                    hook_name=bottleneck['hook_key'].split(':')[1],
                    hook_type=bottleneck['hook_key'].split(':')[0],
                    optimization_type=OptimizationType.ASYNC_OPTIMIZATION,
                    priority=2,
                    description="Optimize async execution patterns",
                    expected_improvement="20-40% faster execution",
                    implementation_complexity="medium",
                    current_impact=bottleneck['impact'],
                    recommended_action="Review and optimize async/await patterns, add proper concurrency"
                )
                for bottleneck in bottlenecks
            ])

        elif bottleneck_type == "high_failure_rate":
            root_cause = "Hooks experiencing frequent failures"
            recommendations.extend([
                OptimizationRecommendation(
                    hook_name=bottleneck['hook_key'].split(':')[1],
                    hook_type=bottleneck['hook_key'].split(':')[0],
                    optimization_type=OptimizationType.ERROR_HANDLING,
                    priority=1,
                    description="Improve error handling and resilience",
                    expected_improvement="50-80% better success rate",
                    implementation_complexity="low",
                    current_impact=bottleneck['impact'],
                    recommended_action="Add comprehensive error handling, retry logic, and fallback mechanisms"
                )
                for bottleneck in bottlenecks
            ])

        elif bottleneck_type == "inconsistent_performance":
            root_cause = "Hooks showing inconsistent execution times"
            recommendations.extend([
                OptimizationRecommendation(
                    hook_name=bottleneck['hook_key'].split(':')[1],
                    hook_type=bottleneck['hook_key'].split(':')[0],
                    optimization_type=OptimizationType.CACHING_IMPLEMENTATION,
                    priority=3,
                    description="Implement caching for consistent performance",
                    expected_improvement="30-60% more consistent execution",
                    implementation_complexity="medium",
                    current_impact=bottleneck['impact'],
                    recommended_action="Add result caching, connection pooling, or resource pre-allocation"
                )
                for bottleneck in bottlenecks
            ])

        return root_cause, recommendations

    async def _generate_optimization_recommendations(self):
        """Generate optimization recommendations for all hooks."""
        recommendations = []

        for hook_key, metrics in self.hook_metrics.items():
            hook_recommendations = await self._analyze_hook_for_optimization(hook_key, metrics)
            recommendations.extend(hook_recommendations)

        # Sort by priority and impact
        recommendations.sort(key=lambda r: (r.priority, -len(r.affected_hooks) if hasattr(r, 'affected_hooks') else 0))

        self._recommendations_cache = recommendations[:10]  # Keep top 10 recommendations

        # Send high-priority recommendations
        high_priority = [r for r in recommendations if r.priority <= 3]
        if high_priority:
            await self._send_optimization_recommendations(high_priority)

    async def _analyze_hook_for_optimization(self, hook_key: str, metrics: HookPerformanceMetrics) -> List[OptimizationRecommendation]:
        """Analyze a specific hook for optimization opportunities."""
        recommendations = []
        hook_name, hook_type = hook_key.split(':', 1)

        # Check for slow execution
        if metrics.average_execution_time > 2.0:
            recommendations.append(OptimizationRecommendation(
                hook_name=hook_name,
                hook_type=hook_type,
                optimization_type=OptimizationType.PARALLEL_EXECUTION,
                priority=3,
                description="Enable parallel execution for faster processing",
                expected_improvement=f"Reduce execution time from {metrics.average_execution_time:.2f}s to ~{metrics.average_execution_time * 0.6:.2f}s",
                implementation_complexity="low",
                current_impact=f"Current execution time: {metrics.average_execution_time:.2f}s",
                recommended_action="Enable parallel execution in hook configuration"
            ))

        # Check for inconsistent performance
        if len(metrics.execution_times) > 10:
            cv = statistics.stdev(metrics.execution_times) / metrics.average_execution_time
            if cv > 0.3:
                recommendations.append(OptimizationRecommendation(
                    hook_name=hook_name,
                    hook_type=hook_type,
                    optimization_type=OptimizationType.CACHING_IMPLEMENTATION,
                    priority=4,
                    description="Implement caching to improve consistency",
                    expected_improvement="Reduce performance variation by 50-70%",
                    implementation_complexity="medium",
                    current_impact=f"Performance variation: {cv:.2f}",
                    recommended_action="Add caching for expensive operations or results"
                ))

        # Check for degrading performance
        if metrics.performance_trend == "degrading":
            recommendations.append(OptimizationRecommendation(
                hook_name=hook_name,
                hook_type=hook_type,
                optimization_type=OptimizationType.RESOURCE_ALLOCATION,
                priority=2,
                description="Address performance degradation",
                expected_improvement="Restore original performance levels",
                implementation_complexity="medium",
                current_impact="Performance degrading over time",
                recommended_action="Investigate resource leaks, memory issues, or external service degradation"
            ))

        # Check for low success rate
        if metrics.success_rate < 0.95 and metrics.total_executions > 20:
            recommendations.append(OptimizationRecommendation(
                hook_name=hook_name,
                hook_type=hook_type,
                optimization_type=OptimizationType.CONDITIONAL_EXECUTION,
                priority=1,
                description="Improve reliability with conditional execution",
                expected_improvement="Increase success rate to 95%+",
                implementation_complexity="low",
                current_impact=f"Success rate: {metrics.success_rate:.1%}",
                recommended_action="Add pre-execution checks and conditional logic"
            ))

        return recommendations

    async def _create_performance_snapshot(self, timestamp: datetime) -> Dict[str, Any]:
        """Create a comprehensive performance snapshot."""
        total_hooks = len(self.hook_metrics)
        performance_levels = defaultdict(int)
        category_performance = defaultdict(list)

        for metrics in self.hook_metrics.values():
            performance_levels[metrics.performance_level.value] += 1
            category_performance[metrics.category.value].append(metrics.efficiency_score)

        # Calculate category averages
        category_averages = {
            category: (sum(scores) / len(scores)) if scores else 0
            for category, scores in category_performance.items()
        }

        return {
            'timestamp': timestamp,
            'total_hooks': total_hooks,
            'performance_distribution': dict(performance_levels),
            'category_performance': category_averages,
            'total_executions': sum(m.total_executions for m in self.hook_metrics.values()),
            'average_efficiency': sum(m.efficiency_score for m in self.hook_metrics.values()) / total_hooks if total_hooks > 0 else 0,
            'active_bottlenecks': 1 if self._bottleneck_cache and self._bottleneck_cache.severity in ['high', 'critical'] else 0,
            'pending_recommendations': len(self._recommendations_cache)
        }

    async def _send_bottleneck_notification(self, analysis: BottleneckAnalysis):
        """Send notification about detected bottlenecks."""
        if not self.message_processor:
            return

        try:
            await self.message_processor.process_message(
                MessageType.WARNING,
                f"⚠️ Performance bottleneck detected: {analysis.bottleneck_type}",
                metadata={
                    'event_type': 'bottleneck_detected',
                    'severity': analysis.severity,
                    'affected_hooks': analysis.affected_hooks,
                    'root_cause': analysis.root_cause,
                    'estimated_impact': analysis.estimated_impact,
                    'recommendations_count': len(analysis.recommendations)
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to send bottleneck notification: {e}")

    async def _send_optimization_recommendations(self, recommendations: List[OptimizationRecommendation]):
        """Send optimization recommendations notification."""
        if not self.message_processor:
            return

        try:
            await self.message_processor.process_message(
                MessageType.INFO,
                f"💡 {len(recommendations)} optimization recommendations available",
                metadata={
                    'event_type': 'optimization_recommendations',
                    'recommendations': [
                        {
                            'hook': r.hook_name,
                            'type': r.optimization_type.value,
                            'priority': r.priority,
                            'description': r.description,
                            'expected_improvement': r.expected_improvement
                        }
                        for r in recommendations
                    ]
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to send optimization recommendations: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.hook_metrics:
            return {"message": "No performance data available"}

        # Calculate overall statistics
        total_executions = sum(m.total_executions for m in self.hook_metrics.values())
        total_successes = sum(m.successful_executions for m in self.hook_metrics.values())
        overall_success_rate = total_successes / total_executions if total_executions > 0 else 0

        average_efficiency = sum(m.efficiency_score for m in self.hook_metrics.values()) / len(self.hook_metrics)

        # Performance distribution
        performance_distribution = defaultdict(int)
        for metrics in self.hook_metrics.values():
            performance_distribution[metrics.performance_level.value] += 1

        # Category breakdown
        category_breakdown = defaultdict(lambda: {'count': 0, 'avg_efficiency': 0, 'avg_time': 0})
        for metrics in self.hook_metrics.values():
            category = metrics.category.value
            category_breakdown[category]['count'] += 1
            category_breakdown[category]['avg_efficiency'] += metrics.efficiency_score
            category_breakdown[category]['avg_time'] += metrics.average_execution_time

        # Calculate category averages
        for category, data in category_breakdown.items():
            if data['count'] > 0:
                data['avg_efficiency'] /= data['count']
                data['avg_time'] /= data['count']

        return {
            'summary_period': f"{self.analysis_window.total_seconds() / 60:.0f} minutes",
            'total_hooks': len(self.hook_metrics),
            'total_executions': total_executions,
            'overall_success_rate': overall_success_rate,
            'average_efficiency': average_efficiency,
            'performance_distribution': dict(performance_distribution),
            'category_breakdown': dict(category_breakdown),
            'current_bottlenecks': asdict(self._bottleneck_cache) if self._bottleneck_cache else None,
            'top_recommendations': [
                {
                    'hook': r.hook_name,
                    'type': r.optimization_type.value,
                    'priority': r.priority,
                    'description': r.description
                }
                for r in self._recommendations_cache[:5]
            ],
            'last_analysis': self._last_analysis_time.isoformat() if self._last_analysis_time else None
        }

    def get_hook_details(self, hook_name: str, hook_type: str) -> Optional[Dict[str, Any]]:
        """Get detailed performance information for a specific hook."""
        hook_key = f"{hook_type}:{hook_name}"

        if hook_key not in self.hook_metrics:
            return None

        metrics = self.hook_metrics[hook_key]

        # Get recent execution history
        recent_executions = [
            {
                'timestamp': exec_data['timestamp'].isoformat(),
                'execution_time': exec_data['execution_time'],
                'success': exec_data['success']
            }
            for exec_data in list(self.execution_history)[-20:]
            if exec_data['hook_key'] == hook_key
        ]

        return {
            'hook_name': hook_name,
            'hook_type': hook_type,
            'category': metrics.category.value,
            'performance_metrics': asdict(metrics),
            'recent_executions': recent_executions,
            'recommendations': [
                asdict(r) for r in self._recommendations_cache
                if r.hook_name == hook_name and r.hook_type == hook_type
            ]
        }

    def export_analytics_data(self, format: str = "json") -> str:
        """Export analytics data for external analysis."""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'analysis_window_minutes': int(self.analysis_window.total_seconds() / 60),
            'hook_metrics': {key: asdict(metrics) for key, metrics in self.hook_metrics.items()},
            'performance_snapshots': list(self.performance_snapshots),
            'bottleneck_analysis': asdict(self._bottleneck_cache) if self._bottleneck_cache else None,
            'optimization_recommendations': [asdict(r) for r in self._recommendations_cache],
            'summary': self.get_performance_summary()
        }

        if format.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Factory function for creating analytics engine
def create_hook_analytics_engine(analysis_window_minutes: int = 60,
                               performance_history_size: int = 1000) -> HookAnalyticsEngine:
    """
    Create and configure a hook analytics engine.

    Args:
        analysis_window_minutes: Time window for performance analysis
        performance_history_size: Size of performance history to retain

    Returns:
        Configured HookAnalyticsEngine instance
    """
    return HookAnalyticsEngine(
        analysis_window_minutes=analysis_window_minutes,
        performance_history_size=performance_history_size
    )