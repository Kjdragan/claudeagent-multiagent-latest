"""
Enhanced Quality Assurance Framework for Multi-Agent Research System.

This module provides a comprehensive quality assurance system that integrates progressive
enhancement, quality gates, continuous monitoring, and intelligent workflow optimization
to ensure consistently high-quality research outputs.

Phase 3.4 Implementation: Build Quality Assurance Framework with progressive enhancement
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .quality_framework import QualityAssessment, QualityFramework, QualityLevel
from .quality_gates import QualityGateManager, GateDecision, GateResult
from .progressive_enhancement import ProgressiveEnhancementPipeline


class QualityAssuranceMode(Enum):
    """Quality assurance operational modes."""
    STRICT = "strict"  # Enforce high quality standards
    BALANCED = "balanced"  # Balance quality with efficiency
    ADAPTIVE = "adaptive"  # Adapt quality requirements based on context
    CONTINUOUS = "continuous"  # Continuous quality improvement


class QualityMetricType(Enum):
    """Types of quality metrics tracked."""
    ASSESSMENT_SCORE = "assessment_score"
    ENHANCEMENT_SUCCESS = "enhancement_success"
    GATE_COMPLIANCE = "gate_compliance"
    WORKFLOW_EFFICIENCY = "workflow_efficiency"
    USER_SATISFACTION = "user_satisfaction"
    SYSTEM_PERFORMANCE = "system_performance"


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics tracking."""
    timestamp: datetime
    session_id: str
    stage: str
    metric_type: QualityMetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    threshold_met: bool = True
    improvement_trend: float = 0.0


@dataclass
class QualityAssuranceConfig:
    """Configuration for quality assurance system."""
    mode: QualityAssuranceMode = QualityAssuranceMode.BALANCED
    target_quality_score: int = 85
    minimum_acceptable_score: int = 70
    max_enhancement_cycles: int = 3
    enable_continuous_monitoring: bool = True
    enable_adaptive_thresholds: bool = True
    quality_degradation_threshold: float = 0.1  # 10% degradation triggers alert
    improvement_threshold: float = 0.05  # 5% improvement triggers success
    performance_tracking_enabled: bool = True
    auto_enhancement_enabled: bool = True


@dataclass
class QualityAssuranceReport:
    """Comprehensive quality assurance report."""
    session_id: str
    timestamp: datetime
    overall_quality_score: int
    quality_level: QualityLevel
    enhancement_cycles_completed: int
    total_improvement: int
    gate_compliance_rate: float
    metrics: List[QualityMetrics] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    issues_identified: List[str] = field(default_factory=list)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    trend_analysis: Dict[str, float] = field(default_factory=dict)


class QualityAssuranceFramework:
    """
    Enhanced Quality Assurance Framework for progressive content optimization.

    This framework integrates quality assessment, progressive enhancement, quality gates,
    and continuous monitoring to ensure consistently high-quality research outputs with
    intelligent workflow optimization and adaptive quality standards.
    """

    def __init__(self, config: Optional[QualityAssuranceConfig] = None):
        """
        Initialize the Enhanced Quality Assurance Framework.

        Args:
            config: Configuration for quality assurance behavior
        """
        self.config = config or QualityAssuranceConfig()
        self.logger = logging.getLogger(__name__)

        # Core quality components
        self.quality_framework = QualityFramework()
        self.quality_gate_manager = QualityGateManager(self.logger)
        self.progressive_enhancement_pipeline = ProgressiveEnhancementPipeline(self.quality_framework)

        # Quality tracking and monitoring
        self.quality_metrics: List[QualityMetrics] = []
        self.session_reports: Dict[str, QualityAssuranceReport] = {}
        self.trend_data: Dict[str, List[float]] = {}

        # Performance tracking
        self.performance_baseline = {}
        self.improvement_targets = {}

        self.logger.info(f"Enhanced Quality Assurance Framework initialized in {self.config.mode.value} mode")

    async def assess_and_enhance_content(
        self,
        content: str,
        session_id: str,
        stage: str,
        context: Optional[Dict[str, Any]] = None,
        target_quality: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive quality assessment and progressive enhancement.

        Args:
            content: Content to assess and enhance
            session_id: Session identifier for tracking
            stage: Current workflow stage
            context: Additional context for assessment
            target_quality: Optional target quality score

        Returns:
            Comprehensive quality assurance results
        """
        self.logger.info(f"Starting comprehensive quality assurance for session {session_id}, stage {stage}")

        start_time = datetime.now()
        target_quality = target_quality or self.config.target_quality_score

        try:
            # Initial quality assessment
            initial_assessment = await self.quality_framework.assess_quality(content, context)

            # Record initial metrics
            await self._record_quality_metrics(
                session_id, stage, QualityMetricType.ASSESSMENT_SCORE,
                initial_assessment.overall_score, {
                    "quality_level": initial_assessment.quality_level.value,
                    "criteria_scores": {name: result.score for name, result in initial_assessment.criteria_results.items()}
                }
            )

            # Check if enhancement is needed
            if initial_assessment.overall_score >= target_quality:
                self.logger.info(f"Content already meets quality target: {initial_assessment.overall_score} >= {target_quality}")

                return {
                    "success": True,
                    "enhanced_content": content,
                    "initial_assessment": initial_assessment,
                    "final_assessment": initial_assessment,
                    "enhancement_applied": False,
                    "total_improvement": 0,
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "quality_target_met": True,
                    "recommendations": initial_assessment.actionable_recommendations,
                    "metrics": await self._get_session_metrics(session_id)
                }

            # Apply progressive enhancement
            enhancement_result = await self.progressive_enhancement_pipeline.enhance_content(
                content, context, target_quality, self.config.max_enhancement_cycles
            )

            # Record enhancement metrics
            await self._record_quality_metrics(
                session_id, stage, QualityMetricType.ENHANCEMENT_SUCCESS,
                enhancement_result["total_improvement"], {
                    "stages_applied": enhancement_result["stages_applied"],
                    "enhancement_log": enhancement_result["enhancement_log"],
                    "target_met": enhancement_result["target_met"]
                }
            )

            # Final quality assessment
            final_assessment = enhancement_result["final_assessment"]

            # Record final metrics
            await self._record_quality_metrics(
                session_id, stage, QualityMetricType.ASSESSMENT_SCORE,
                final_assessment.overall_score, {
                    "quality_level": final_assessment.quality_level.value,
                    "final_assessment": True,
                    "total_improvement": enhancement_result["total_improvement"]
                }
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            # Generate quality assurance report
            qa_report = await self._generate_quality_assurance_report(
                session_id, stage, initial_assessment, final_assessment, enhancement_result, processing_time
            )

            self.session_reports[f"{session_id}_{stage}"] = qa_report

            self.logger.info(f"Quality assurance completed for session {session_id}")
            self.logger.info(f"Quality improvement: {initial_assessment.overall_score} → {final_assessment.overall_score} (+{enhancement_result['total_improvement']})")

            return {
                "success": True,
                "enhanced_content": enhancement_result["enhanced_content"],
                "initial_assessment": initial_assessment,
                "final_assessment": final_assessment,
                "enhancement_applied": enhancement_result["total_improvement"] > 0,
                "total_improvement": enhancement_result["total_improvement"],
                "processing_time": processing_time,
                "quality_target_met": final_assessment.overall_score >= target_quality,
                "enhancement_details": enhancement_result,
                "quality_assurance_report": qa_report,
                "recommendations": self._generate_integrated_recommendations(initial_assessment, final_assessment, enhancement_result),
                "metrics": await self._get_session_metrics(session_id)
            }

        except Exception as e:
            self.logger.error(f"Quality assurance failed for session {session_id}: {e}")

            # Record failure metrics
            await self._record_quality_metrics(
                session_id, stage, QualityMetricType.ASSESSMENT_SCORE,
                0, {"error": str(e), "failed_at": datetime.now().isoformat()}
            )

            return {
                "success": False,
                "error": str(e),
                "enhanced_content": content,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "quality_target_met": False,
                "metrics": await self._get_session_metrics(session_id)
            }

    async def evaluate_quality_gates(
        self,
        stage: str,
        assessment: QualityAssessment,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> GateResult:
        """
        Evaluate quality gates with enhanced decision logic.

        Args:
            stage: Current workflow stage
            assessment: Quality assessment results
            session_id: Session identifier
            context: Additional context for gate evaluation

        Returns:
            Enhanced gate evaluation result
        """
        from .workflow_state import WorkflowSession, WorkflowStage

        # Create a mock session for gate evaluation (in real implementation, this would be provided)
        mock_session = WorkflowSession(
            session_id=session_id,
            topic="research_topic",
            user_requirements={},
            stage=WorkflowStage.RESEARCH,  # Default, would be actual stage
            status="in_progress",
            start_time=datetime.now()
        )

        # Evaluate quality gate
        gate_result = self.quality_gate_manager.evaluate_quality_gate(
            WorkflowStage(stage.upper()), assessment, mock_session, context
        )

        # Record gate compliance metrics
        await self._record_quality_metrics(
            session_id, stage, QualityMetricType.GATE_COMPLIANCE,
            1.0 if gate_result.decision == GateDecision.PROCEED else 0.0, {
                "gate_decision": gate_result.decision.value,
                "confidence": gate_result.confidence,
                "reasoning": gate_result.reasoning,
                "triggered_criteria": gate_result.triggered_criteria
            }
        )

        # Apply adaptive threshold adjustment if enabled
        if self.config.enable_adaptive_thresholds:
            await self._adjust_adaptive_thresholds(stage, gate_result)

        return gate_result

    async def monitor_continuous_quality(
        self,
        session_id: str,
        sampling_interval: timedelta = timedelta(minutes=5)
    ) -> Dict[str, Any]:
        """
        Monitor quality continuously throughout the workflow.

        Args:
            session_id: Session to monitor
            sampling_interval: Interval between quality checks

        Returns:
            Continuous monitoring results
        """
        if not self.config.enable_continuous_monitoring:
            return {"enabled": False, "reason": "Continuous monitoring disabled in configuration"}

        self.logger.info(f"Starting continuous quality monitoring for session {session_id}")

        monitoring_results = {
            "session_id": session_id,
            "monitoring_start": datetime.now().isoformat(),
            "sampling_interval": sampling_interval.total_seconds(),
            "quality_trends": {},
            "alerts": [],
            "performance_metrics": {},
            "compliance_status": {}
        }

        try:
            # Get historical metrics for the session
            session_metrics = await self._get_session_metrics(session_id)

            # Analyze quality trends
            quality_trends = await self._analyze_quality_trends(session_metrics)
            monitoring_results["quality_trends"] = quality_trends

            # Check for quality alerts
            alerts = await self._check_quality_alerts(session_id, session_metrics)
            monitoring_results["alerts"] = alerts

            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(session_id)
            monitoring_results["performance_metrics"] = performance_metrics

            # Check compliance status
            compliance_status = await self._check_compliance_status(session_id)
            monitoring_results["compliance_status"] = compliance_status

            # Store trend data
            self.trend_data[session_id] = session_metrics

            self.logger.info(f"Continuous quality monitoring completed for session {session_id}")

        except Exception as e:
            self.logger.error(f"Continuous quality monitoring failed for session {session_id}: {e}")
            monitoring_results["error"] = str(e)

        return monitoring_results

    async def optimize_quality_workflow(
        self,
        session_id: str,
        performance_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize quality workflow based on performance data and trends.

        Args:
            session_id: Session to optimize
            performance_data: Optional performance data for optimization

        Returns:
            Workflow optimization recommendations
        """
        self.logger.info(f"Starting quality workflow optimization for session {session_id}")

        optimization_results = {
            "session_id": session_id,
            "optimization_timestamp": datetime.now().isoformat(),
            "recommendations": [],
            "adjustments": {},
            "expected_improvements": {},
            "implementation_priority": []
        }

        try:
            # Analyze current performance
            current_performance = performance_data or await self._calculate_performance_metrics(session_id)

            # Identify bottlenecks and improvement opportunities
            bottlenecks = await self._identify_quality_bottlenecks(session_id, current_performance)

            # Generate optimization recommendations
            recommendations = await self._generate_optimization_recommendations(bottlenecks, current_performance)
            optimization_results["recommendations"] = recommendations

            # Calculate expected improvements
            expected_improvements = await self._calculate_expected_improvements(recommendations)
            optimization_results["expected_improvements"] = expected_improvements

            # Prioritize recommendations
            prioritized_recommendations = await self._prioritize_optimization_recommendations(recommendations)
            optimization_results["implementation_priority"] = prioritized_recommendations

            # Suggest configuration adjustments
            config_adjustments = await self._suggest_configuration_adjustments(bottlenecks)
            optimization_results["adjustments"] = config_adjustments

            self.logger.info(f"Quality workflow optimization completed for session {session_id}")
            self.logger.info(f"Generated {len(recommendations)} optimization recommendations")

        except Exception as e:
            self.logger.error(f"Quality workflow optimization failed for session {session_id}: {e}")
            optimization_results["error"] = str(e)

        return optimization_results

    async def generate_comprehensive_quality_report(
        self,
        session_id: str,
        include_trends: bool = True,
        include_recommendations: bool = True
    ) -> QualityAssuranceReport:
        """
        Generate comprehensive quality assurance report.

        Args:
            session_id: Session to report on
            include_trends: Whether to include trend analysis
            include_recommendations: Whether to include recommendations

        Returns:
            Comprehensive quality assurance report
        """
        self.logger.info(f"Generating comprehensive quality report for session {session_id}")

        # Get all session metrics
        session_metrics = await self._get_session_metrics(session_id)

        if not session_metrics:
            # Create default report if no metrics available
            return QualityAssuranceReport(
                session_id=session_id,
                timestamp=datetime.now(),
                overall_quality_score=0,
                quality_level=QualityLevel.POOR,
                enhancement_cycles_completed=0,
                total_improvement=0,
                gate_compliance_rate=0.0,
                issues_identified=["No quality metrics available for this session"]
            )

        # Calculate aggregate metrics
        overall_quality_score = await self._calculate_overall_quality_score(session_metrics)
        quality_level = self._determine_quality_level(overall_quality_score)
        enhancement_cycles = await self._count_enhancement_cycles(session_metrics)
        total_improvement = await self._calculate_total_improvement(session_metrics)
        gate_compliance_rate = await self._calculate_gate_compliance_rate(session_metrics)

        # Create report
        report = QualityAssuranceReport(
            session_id=session_id,
            timestamp=datetime.now(),
            overall_quality_score=overall_quality_score,
            quality_level=quality_level,
            enhancement_cycles_completed=enhancement_cycles,
            total_improvement=total_improvement,
            gate_compliance_rate=gate_compliance_rate,
            metrics=session_metrics
        )

        # Add trend analysis if requested
        if include_trends:
            report.trend_analysis = await self._analyze_quality_trends(session_metrics)

        # Add recommendations if requested
        if include_recommendations:
            report.recommendations = await self._generate_session_recommendations(session_id, session_metrics)

        # Add performance summary
        report.performance_summary = await self._calculate_performance_metrics(session_id)

        # Add compliance status
        report.compliance_status = await self._check_compliance_status(session_id)

        # Identify issues
        report.issues_identified = await self._identify_quality_issues(session_metrics)

        self.logger.info(f"Comprehensive quality report generated for session {session_id}")

        return report

    # Private helper methods

    async def _record_quality_metrics(
        self,
        session_id: str,
        stage: str,
        metric_type: QualityMetricType,
        value: float,
        metadata: Dict[str, Any]
    ) -> None:
        """Record quality metrics for tracking and analysis."""

        metric = QualityMetrics(
            timestamp=datetime.now(),
            session_id=session_id,
            stage=stage,
            metric_type=metric_type,
            value=value,
            metadata=metadata,
            threshold_met=await self._check_threshold_met(metric_type, value),
            improvement_trend=await self._calculate_improvement_trend(session_id, metric_type, value)
        )

        self.quality_metrics.append(metric)

        # Keep only recent metrics (last 1000 per session)
        session_metrics = [m for m in self.quality_metrics if m.session_id == session_id]
        if len(session_metrics) > 1000:
            self.quality_metrics = [m for m in self.quality_metrics if m.session_id != session_id] + session_metrics[-1000:]

        self.logger.debug(f"Recorded {metric_type.value} metric for session {session_id}: {value}")

    async def _get_session_metrics(self, session_id: str) -> List[QualityMetrics]:
        """Get all metrics for a specific session."""
        return [m for m in self.quality_metrics if m.session_id == session_id]

    async def _check_threshold_met(self, metric_type: QualityMetricType, value: float) -> bool:
        """Check if a metric value meets its threshold."""

        thresholds = {
            QualityMetricType.ASSESSMENT_SCORE: self.config.minimum_acceptable_score,
            QualityMetricType.ENHANCEMENT_SUCCESS: 0.0,  # Any improvement is good
            QualityMetricType.GATE_COMPLIANCE: 0.8,  # 80% compliance rate
            QualityMetricType.WORKFLOW_EFFICIENCY: 0.7,  # 70% efficiency
            QualityMetricType.USER_SATISFACTION: 0.8,  # 80% satisfaction
            QualityMetricType.SYSTEM_PERFORMANCE: 0.7  # 70% performance
        }

        threshold = thresholds.get(metric_type, 0.0)
        return value >= threshold

    async def _calculate_improvement_trend(self, session_id: str, metric_type: QualityMetricType, current_value: float) -> float:
        """Calculate improvement trend for a metric."""

        session_metrics = await self._get_session_metrics(session_id)
        same_type_metrics = [m for m in session_metrics if m.metric_type == metric_type]

        if len(same_type_metrics) < 2:
            return 0.0

        # Calculate trend based on last few values
        recent_values = [m.value for m in same_type_metrics[-5:]]
        if len(recent_values) < 2:
            return 0.0

        # Simple trend calculation (current vs. average of previous)
        previous_avg = sum(recent_values[:-1]) / len(recent_values[:-1])
        trend = (current_value - previous_avg) / max(previous_avg, 0.1)  # Avoid division by zero

        return round(trend, 3)

    async def _generate_quality_assurance_report(
        self,
        session_id: str,
        stage: str,
        initial_assessment: QualityAssessment,
        final_assessment: QualityAssessment,
        enhancement_result: Dict[str, Any],
        processing_time: float
    ) -> QualityAssuranceReport:
        """Generate comprehensive quality assurance report."""

        # Get session metrics
        session_metrics = await self._get_session_metrics(session_id)

        # Calculate compliance status
        compliance_status = {
            "quality_target_met": final_assessment.overall_score >= self.config.target_quality_score,
            "minimum_threshold_met": final_assessment.overall_score >= self.config.minimum_acceptable_score,
            "enhancement_successful": enhancement_result["total_improvement"] > 0,
            "processing_efficient": processing_time < 300  # 5 minutes
        }

        # Generate recommendations
        recommendations = self._generate_integrated_recommendations(
            initial_assessment, final_assessment, enhancement_result
        )

        # Identify issues
        issues = []
        if final_assessment.overall_score < self.config.minimum_acceptable_score:
            issues.append("Content fails to meet minimum quality standards")

        if enhancement_result["total_improvement"] == 0 and final_assessment.overall_score < self.config.target_quality_score:
            issues.append("No quality improvement achieved during enhancement")

        if processing_time > 300:
            issues.append("Processing time exceeded efficiency threshold")

        return QualityAssuranceReport(
            session_id=session_id,
            timestamp=datetime.now(),
            overall_quality_score=final_assessment.overall_score,
            quality_level=final_assessment.quality_level,
            enhancement_cycles_completed=enhancement_result["stages_applied"],
            total_improvement=enhancement_result["total_improvement"],
            gate_compliance_rate=1.0,  # Would be calculated from actual gate results
            metrics=session_metrics,
            recommendations=recommendations,
            issues_identified=issues,
            performance_summary={"processing_time": processing_time},
            compliance_status=compliance_status,
            trend_analysis=await self._analyze_quality_trends(session_metrics)
        )

    def _generate_integrated_recommendations(
        self,
        initial_assessment: QualityAssessment,
        final_assessment: QualityAssessment,
        enhancement_result: Dict[str, Any]
    ) -> List[str]:
        """Generate integrated recommendations based on assessment and enhancement results."""

        recommendations = []

        # Add recommendations from final assessment
        recommendations.extend(final_assessment.actionable_recommendations)

        # Add enhancement-specific recommendations
        if enhancement_result["total_improvement"] == 0:
            recommendations.append("Consider manual review and enhancement as automatic enhancement was ineffective")

        if final_assessment.overall_score < self.config.target_quality_score:
            recommendations.append(f"Content still {self.config.target_quality_score - final_assessment.overall_score} points below target - consider additional research or content expansion")

        # Add criteria-specific recommendations
        for criterion_name, result in final_assessment.criteria_results.items():
            if result.score < 70:
                recommendations.append(f"Focus on improving {criterion_name.replace('_', ' ')} - current score: {result.score}/100")

        return list(set(recommendations))  # Remove duplicates

    async def _adjust_adaptive_thresholds(self, stage: str, gate_result: GateResult) -> None:
        """Adjust quality thresholds adaptively based on performance."""

        # This would implement adaptive threshold logic
        # For now, it's a placeholder for future enhancement
        pass

    async def _analyze_quality_trends(self, metrics: List[QualityMetrics]) -> Dict[str, float]:
        """Analyze quality trends from metrics data."""

        if not metrics:
            return {}

        trends = {}

        # Group by metric type
        by_type = {}
        for metric in metrics:
            if metric.metric_type not in by_type:
                by_type[metric.metric_type] = []
            by_type[metric.metric_type].append(metric.value)

        # Calculate trends for each metric type
        for metric_type, values in by_type.items():
            if len(values) >= 2:
                # Simple linear trend calculation
                trend = (values[-1] - values[0]) / len(values)
                trends[metric_type.value] = trend

        return trends

    async def _check_quality_alerts(self, session_id: str, metrics: List[QualityMetrics]) -> List[Dict[str, Any]]:
        """Check for quality alerts based on metrics."""

        alerts = []

        # Check for quality degradation
        assessment_metrics = [m for m in metrics if m.metric_type == QualityMetricType.ASSESSMENT_SCORE]
        if len(assessment_metrics) >= 2:
            recent_avg = sum(m.value for m in assessment_metrics[-3:]) / min(3, len(assessment_metrics[-3:]))
            earlier_avg = sum(m.value for m in assessment_metrics[:3]) / min(3, len(assessment_metrics[:3]))

            if earlier_avg > 0 and (earlier_avg - recent_avg) / earlier_avg > self.config.quality_degradation_threshold:
                alerts.append({
                    "type": "quality_degradation",
                    "severity": "high",
                    "message": f"Quality degraded by {((earlier_avg - recent_avg) / earlier_avg * 100):.1f}%",
                    "session_id": session_id
                })

        # Check for enhancement failures
        enhancement_metrics = [m for m in metrics if m.metric_type == QualityMetricType.ENHANCEMENT_SUCCESS]
        failed_enhancements = [m for m in enhancement_metrics if m.value <= 0]

        if len(failed_enhancements) > 2:
            alerts.append({
                "type": "enhancement_failures",
                "severity": "medium",
                "message": f"Multiple enhancement failures detected: {len(failed_enhancements)}",
                "session_id": session_id
            })

        return alerts

    async def _calculate_performance_metrics(self, session_id: str) -> Dict[str, Any]:
        """Calculate performance metrics for a session."""

        session_metrics = await self._get_session_metrics(session_id)

        if not session_metrics:
            return {}

        # Calculate various performance metrics
        performance = {
            "total_assessments": len([m for m in session_metrics if m.metric_type == QualityMetricType.ASSESSMENT_SCORE]),
            "average_quality_score": 0.0,
            "enhancement_success_rate": 0.0,
            "gate_compliance_rate": 0.0,
            "processing_efficiency": 0.0
        }

        # Calculate average quality score
        assessment_scores = [m.value for m in session_metrics if m.metric_type == QualityMetricType.ASSESSMENT_SCORE]
        if assessment_scores:
            performance["average_quality_score"] = sum(assessment_scores) / len(assessment_scores)

        # Calculate enhancement success rate
        enhancement_scores = [m.value for m in session_metrics if m.metric_type == QualityMetricType.ENHANCEMENT_SUCCESS]
        if enhancement_scores:
            successful_enhancements = len([s for s in enhancement_scores if s > 0])
            performance["enhancement_success_rate"] = successful_enhancements / len(enhancement_scores)

        # Calculate gate compliance rate
        gate_scores = [m.value for m in session_metrics if m.metric_type == QualityMetricType.GATE_COMPLIANCE]
        if gate_scores:
            performance["gate_compliance_rate"] = sum(gate_scores) / len(gate_scores)

        return performance

    async def _check_compliance_status(self, session_id: str) -> Dict[str, bool]:
        """Check compliance status for various quality standards."""

        session_metrics = await self._get_session_metrics(session_id)

        compliance = {
            "minimum_quality_met": True,
            "target_quality_met": True,
            "enhancement_working": True,
            "gates_passing": True
        }

        # Check minimum quality
        assessment_scores = [m.value for m in session_metrics if m.metric_type == QualityMetricType.ASSESSMENT_SCORE]
        if assessment_scores and assessment_scores[-1] < self.config.minimum_acceptable_score:
            compliance["minimum_quality_met"] = False

        # Check target quality
        if assessment_scores and assessment_scores[-1] < self.config.target_quality_score:
            compliance["target_quality_met"] = False

        # Check enhancement effectiveness
        enhancement_scores = [m.value for m in session_metrics if m.metric_type == QualityMetricType.ENHANCEMENT_SUCCESS]
        if enhancement_scores and all(s <= 0 for s in enhancement_scores[-3:]):
            compliance["enhancement_working"] = False

        # Check gate compliance
        gate_scores = [m.value for m in session_metrics if m.metric_type == QualityMetricType.GATE_COMPLIANCE]
        if gate_scores and gate_scores[-1] < 0.8:
            compliance["gates_passing"] = False

        return compliance

    async def _identify_quality_bottlenecks(self, session_id: str, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify quality bottlenecks based on performance data."""

        bottlenecks = []

        # Check for low quality scores
        if performance_data.get("average_quality_score", 0) < self.config.minimum_acceptable_score:
            bottlenecks.append({
                "type": "low_quality_score",
                "severity": "high",
                "description": "Average quality score below minimum threshold",
                "impact": "Content fails to meet quality standards"
            })

        # Check for enhancement failures
        if performance_data.get("enhancement_success_rate", 0) < 0.5:
            bottlenecks.append({
                "type": "enhancement_ineffective",
                "severity": "medium",
                "description": "Enhancement success rate below 50%",
                "impact": "Automatic quality improvement is not working effectively"
            })

        # Check for gate compliance issues
        if performance_data.get("gate_compliance_rate", 0) < 0.8:
            bottlenecks.append({
                "type": "gate_compliance_issues",
                "severity": "medium",
                "description": "Gate compliance rate below 80%",
                "impact": "Workflow progression may be blocked by quality gates"
            })

        return bottlenecks

    async def _generate_optimization_recommendations(self, bottlenecks: List[Dict[str, Any]], performance_data: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on identified bottlenecks."""

        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck["type"] == "low_quality_score":
                recommendations.append("Increase research depth and source diversity")
                recommendations.append("Enhance content structure and organization")
                recommendations.append("Improve analytical depth and examples")

            elif bottleneck["type"] == "enhancement_ineffective":
                recommendations.append("Review and enhance progressive enhancement algorithms")
                recommendations.append("Consider custom enhancement stages for specific content types")
                recommendations.append("Increase enhancement cycle limits or adjust target thresholds")

            elif bottleneck["type"] == "gate_compliance_issues":
                recommendations.append("Review and adjust quality gate thresholds")
                recommendations.append("Improve content quality before gate evaluation")
                recommendations.append("Consider alternative workflow paths for problematic stages")

        return recommendations

    async def _calculate_expected_improvements(self, recommendations: List[str]) -> Dict[str, float]:
        """Calculate expected improvements from optimization recommendations."""

        # Simple estimation based on recommendation types
        expected_improvements = {
            "quality_score_improvement": len(recommendations) * 5.0,  # 5 points per recommendation
            "efficiency_improvement": len(recommendations) * 0.1,   # 10% per recommendation
            "compliance_improvement": len(recommendations) * 0.15    # 15% per recommendation
        }

        return expected_improvements

    async def _prioritize_optimization_recommendations(self, recommendations: List[str]) -> List[Dict[str, Any]]:
        """Prioritize optimization recommendations based on expected impact."""

        prioritized = []

        for i, rec in enumerate(recommendations):
            # Simple priority calculation based on recommendation content
            priority = "medium"
            impact_score = 5.0

            if "increase" in rec.lower() or "improve" in rec.lower():
                priority = "high"
                impact_score = 8.0
            elif "review" in rec.lower() or "consider" in rec.lower():
                priority = "low"
                impact_score = 3.0

            prioritized.append({
                "recommendation": rec,
                "priority": priority,
                "impact_score": impact_score,
                "implementation_order": i + 1
            })

        # Sort by impact score (descending)
        prioritized.sort(key=lambda x: x["impact_score"], reverse=True)

        return prioritized

    async def _suggest_configuration_adjustments(self, bottlenecks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest configuration adjustments to address bottlenecks."""

        adjustments = {}

        for bottleneck in bottlenecks:
            if bottleneck["type"] == "low_quality_score":
                adjustments["target_quality_score"] = max(self.config.target_quality_score - 5, 70)
                adjustments["max_enhancement_cycles"] = self.config.max_enhancement_cycles + 1

            elif bottleneck["type"] == "enhancement_ineffective":
                adjustments["auto_enhancement_enabled"] = False
                adjustments["enable_adaptive_thresholds"] = True

            elif bottleneck["type"] == "gate_compliance_issues":
                adjustments["minimum_acceptable_score"] = max(self.config.minimum_acceptable_score - 5, 60)

        return adjustments

    async def _count_enhancement_cycles(self, metrics: List[QualityMetrics]) -> int:
        """Count enhancement cycles from metrics."""
        return len([m for m in metrics if m.metric_type == QualityMetricType.ENHANCEMENT_SUCCESS])

    async def _calculate_total_improvement(self, metrics: List[QualityMetrics]) -> int:
        """Calculate total quality improvement from metrics."""
        enhancement_metrics = [m for m in metrics if m.metric_type == QualityMetricType.ENHANCEMENT_SUCCESS]
        return int(sum(m.value for m in enhancement_metrics))

    async def _calculate_gate_compliance_rate(self, metrics: List[QualityMetrics]) -> float:
        """Calculate gate compliance rate from metrics."""
        gate_metrics = [m for m in metrics if m.metric_type == QualityMetricType.GATE_COMPLIANCE]
        if not gate_metrics:
            return 0.0
        return sum(m.value for m in gate_metrics) / len(gate_metrics)

    def _determine_quality_level(self, score: int) -> QualityLevel:
        """Determine quality level from score."""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 80:
            return QualityLevel.GOOD
        elif score >= 70:
            return QualityLevel.ACCEPTABLE
        elif score >= 60:
            return QualityLevel.NEEDS_IMPROVEMENT
        else:
            return QualityLevel.POOR

    async def _calculate_overall_quality_score(self, metrics: List[QualityMetrics]) -> int:
        """Calculate overall quality score from metrics."""
        assessment_metrics = [m for m in metrics if m.metric_type == QualityMetricType.ASSESSMENT_SCORE]
        if not assessment_metrics:
            return 0
        return int(sum(m.value for m in assessment_metrics) / len(assessment_metrics))

    async def _generate_session_recommendations(self, session_id: str, metrics: List[QualityMetrics]) -> List[str]:
        """Generate recommendations for a session based on metrics."""

        recommendations = []

        # Analyze assessment scores
        assessment_metrics = [m for m in metrics if m.metric_type == QualityMetricType.ASSESSMENT_SCORE]
        if assessment_metrics:
            latest_score = assessment_metrics[-1].value
            if latest_score < self.config.target_quality_score:
                recommendations.append(f"Content is {self.config.target_quality_score - latest_score} points below target - consider additional enhancement")

        # Analyze enhancement effectiveness
        enhancement_metrics = [m for m in metrics if m.metric_type == QualityMetricType.ENHANCEMENT_SUCCESS]
        if enhancement_metrics:
            avg_improvement = sum(m.value for m in enhancement_metrics) / len(enhancement_metrics)
            if avg_improvement < 5:
                recommendations.append("Enhancement effectiveness is low - consider manual review and content improvement")

        # Analyze gate compliance
        gate_metrics = [m for m in metrics if m.metric_type == QualityMetricType.GATE_COMPLIANCE]
        if gate_metrics:
            compliance_rate = sum(m.value for m in gate_metrics) / len(gate_metrics)
            if compliance_rate < 0.8:
                recommendations.append("Gate compliance rate is below 80% - review quality standards and thresholds")

        return recommendations

    async def _identify_quality_issues(self, metrics: List[QualityMetrics]) -> List[str]:
        """Identify quality issues from metrics."""

        issues = []

        # Check for consistently low scores
        assessment_metrics = [m for m in metrics if m.metric_type == QualityMetricType.ASSESSMENT_SCORE]
        if assessment_metrics and all(m.value < self.config.minimum_acceptable_score for m in assessment_metrics[-5:]):
            issues.append("Consistently low quality scores detected")

        # Check for failed enhancements
        enhancement_metrics = [m for m in metrics if m.metric_type == QualityMetricType.ENHANCEMENT_SUCCESS]
        if enhancement_metrics and all(m.value <= 0 for m in enhancement_metrics[-3:]):
            issues.append("Multiple consecutive enhancement failures")

        # Check for gate failures
        gate_metrics = [m for m in metrics if m.metric_type == QualityMetricType.GATE_COMPLIANCE]
        if gate_metrics and any(m.value < 0.5 for m in gate_metrics[-3:]):
            issues.append("Recent quality gate failures detected")

        return issues


# Convenience function for quick quality assurance
async def ensure_content_quality(
    content: str,
    session_id: str,
    stage: str,
    context: Optional[Dict[str, Any]] = None,
    target_quality: int = 85
) -> Dict[str, Any]:
    """
    Quick quality assurance function for content enhancement.

    Args:
        content: Content to assess and enhance
        session_id: Session identifier
        stage: Current workflow stage
        context: Additional context for assessment
        target_quality: Target quality score

    Returns:
        Quality assurance results
    """
    qa_framework = QualityAssuranceFramework()
    return await qa_framework.assess_and_enhance_content(
        content, session_id, stage, context, target_quality
    )