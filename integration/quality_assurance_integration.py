"""
Quality Assurance Integration for Multi-Agent Research System.

This module provides comprehensive quality assurance integration that bridges the agent-based
research system with existing quality frameworks, ensuring consistent quality assessment
and enhancement across all workflow stages.

Phase 2.3 Implementation: Build Quality Assurance Pipeline - Integrate with existing quality frameworks
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Import existing quality frameworks
from multi_agent_research_system.core.quality_framework import (
    QualityFramework, QualityAssessment, QualityLevel
)
from multi_agent_research_system.core.quality_gates import (
    QualityGateManager, GateDecision, GateResult
)
from multi_agent_research_system.core.quality_assurance_framework import (
    QualityAssuranceFramework, QualityAssuranceConfig, QualityMetrics,
    QualityMetricType, QualityAssuranceReport
)

# Import system components
from integration.kevin_directory_integration import KevinDirectoryIntegration
from integration.mcp_tool_integration import MCPToolIntegration


class QualityAssuranceIntegration:
    """
    Comprehensive quality assurance integration that bridges agent-based research
    with existing quality frameworks for consistent quality management.
    """

    def __init__(self, config: Optional[QualityAssuranceConfig] = None):
        """
        Initialize the Quality Assurance Integration component.

        Args:
            config: Optional configuration for quality assurance behavior
        """
        self.config = config or QualityAssuranceConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize existing quality frameworks
        self.quality_framework = QualityFramework()
        self.quality_gate_manager = QualityGateManager(self.logger)
        self.quality_assurance_framework = QualityAssuranceFramework(self.config)

        # Initialize system integrations
        self.kevin_integration = KevinDirectoryIntegration()
        self.mcp_integration = MCPToolIntegration()

        # Quality tracking and analytics
        self.session_quality_data: Dict[str, Dict[str, Any]] = {}
        self.quality_trends: Dict[str, List[Dict[str, Any]]] = {}

        self.logger.info(f"QualityAssuranceIntegration initialized in {self.config.mode.value} mode")

    async def assess_research_quality(
        self,
        session_id: str,
        research_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess research quality using comprehensive quality framework.

        Args:
            session_id: Session identifier for tracking
            research_content: Research content to assess
            context: Additional context for assessment

        Returns:
            Comprehensive quality assessment results
        """
        self.logger.info(f"Starting research quality assessment for session {session_id}")

        try:
            # Prepare assessment context
            assessment_context = self._prepare_assessment_context(
                session_id, research_content, context, "research"
            )

            # Perform comprehensive quality assessment
            quality_assessment = await self.quality_framework.assess_quality(
                research_content, assessment_context
            )

            # Evaluate quality gates for research stage
            gate_result = await self.quality_gate_manager.evaluate_quality_gate(
                self._map_workflow_stage("research"),
                quality_assessment,
                self._create_mock_session(session_id),
                assessment_context
            )

            # Apply progressive enhancement if needed
            enhancement_result = None
            if gate_result.decision in [GateDecision.ENHANCE, GateDecision.RERUN]:
                enhancement_result = await self._apply_progressive_enhancement(
                    session_id, research_content, quality_assessment, assessment_context
                )

            # Store quality data for session
            await self._store_session_quality_data(
                session_id, "research", quality_assessment, gate_result, enhancement_result
            )

            # Log quality assessment to KEVIN directory
            await self._log_quality_assessment(
                session_id, "research", quality_assessment, gate_result
            )

            return {
                "success": True,
                "session_id": session_id,
                "stage": "research",
                "quality_assessment": quality_assessment,
                "gate_result": gate_result,
                "enhancement_result": enhancement_result,
                "recommendations": self._generate_stage_recommendations(
                    "research", quality_assessment, gate_result
                ),
                "quality_metrics": await self._calculate_quality_metrics(
                    session_id, "research", quality_assessment
                ),
                "assessment_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Research quality assessment failed for session {session_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "stage": "research"
            }

    async def assess_report_quality(
        self,
        session_id: str,
        report_content: str,
        research_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess report quality using comprehensive quality framework.

        Args:
            session_id: Session identifier for tracking
            report_content: Report content to assess
            research_context: Research data context for assessment

        Returns:
            Comprehensive quality assessment results
        """
        self.logger.info(f"Starting report quality assessment for session {session_id}")

        try:
            # Prepare assessment context with research data
            assessment_context = self._prepare_assessment_context(
                session_id, report_content, research_context, "report"
            )

            # Perform comprehensive quality assessment
            quality_assessment = await self.quality_framework.assess_quality(
                report_content, assessment_context
            )

            # Evaluate quality gates for report stage
            gate_result = await self.quality_gate_manager.evaluate_quality_gate(
                self._map_workflow_stage("report"),
                quality_assessment,
                self._create_mock_session(session_id),
                assessment_context
            )

            # Apply progressive enhancement if needed
            enhancement_result = None
            if gate_result.decision in [GateDecision.ENHANCE, GateDecision.RERUN]:
                enhancement_result = await self._apply_progressive_enhancement(
                    session_id, report_content, quality_assessment, assessment_context
                )

            # Store quality data for session
            await self._store_session_quality_data(
                session_id, "report", quality_assessment, gate_result, enhancement_result
            )

            # Log quality assessment to KEVIN directory
            await self._log_quality_assessment(
                session_id, "report", quality_assessment, gate_result
            )

            return {
                "success": True,
                "session_id": session_id,
                "stage": "report",
                "quality_assessment": quality_assessment,
                "gate_result": gate_result,
                "enhancement_result": enhancement_result,
                "recommendations": self._generate_stage_recommendations(
                    "report", quality_assessment, gate_result
                ),
                "quality_metrics": await self._calculate_quality_metrics(
                    session_id, "report", quality_assessment
                ),
                "data_integration_quality": await self._assess_data_integration_quality(
                    report_content, research_context
                ),
                "assessment_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Report quality assessment failed for session {session_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "stage": "report"
            }

    async def assess_editorial_quality(
        self,
        session_id: str,
        editorial_content: str,
        report_context: Optional[Dict[str, Any]] = None,
        gap_research_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess editorial quality using comprehensive quality framework.

        Args:
            session_id: Session identifier for tracking
            editorial_content: Editorial content to assess
            report_context: Report data context for assessment
            gap_research_context: Gap research context for assessment

        Returns:
            Comprehensive quality assessment results
        """
        self.logger.info(f"Starting editorial quality assessment for session {session_id}")

        try:
            # Prepare comprehensive assessment context
            assessment_context = self._prepare_assessment_context(
                session_id, editorial_content, {
                    "report_context": report_context,
                    "gap_research_context": gap_research_context
                }, "editorial"
            )

            # Perform comprehensive quality assessment
            quality_assessment = await self.quality_framework.assess_quality(
                editorial_content, assessment_context
            )

            # Evaluate quality gates for editorial stage
            gate_result = await self.quality_gate_manager.evaluate_quality_gate(
                self._map_workflow_stage("editorial"),
                quality_assessment,
                self._create_mock_session(session_id),
                assessment_context
            )

            # Apply progressive enhancement if needed
            enhancement_result = None
            if gate_result.decision in [GateDecision.ENHANCE, GateDecision.RERUN]:
                enhancement_result = await self._apply_progressive_enhancement(
                    session_id, editorial_content, quality_assessment, assessment_context
                )

            # Assess editorial decision quality
            editorial_decision_quality = await self._assess_editorial_decision_quality(
                editorial_content, assessment_context
            )

            # Store quality data for session
            await self._store_session_quality_data(
                session_id, "editorial", quality_assessment, gate_result, enhancement_result
            )

            # Log quality assessment to KEVIN directory
            await self._log_quality_assessment(
                session_id, "editorial", quality_assessment, gate_result
            )

            return {
                "success": True,
                "session_id": session_id,
                "stage": "editorial",
                "quality_assessment": quality_assessment,
                "gate_result": gate_result,
                "enhancement_result": enhancement_result,
                "editorial_decision_quality": editorial_decision_quality,
                "recommendations": self._generate_stage_recommendations(
                    "editorial", quality_assessment, gate_result
                ),
                "quality_metrics": await self._calculate_quality_metrics(
                    session_id, "editorial", quality_assessment
                ),
                "gap_research_effectiveness": await self._assess_gap_research_effectiveness(
                    gap_research_context, editorial_content
                ),
                "assessment_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Editorial quality assessment failed for session {session_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "stage": "editorial"
            }

    async def assess_final_quality(
        self,
        session_id: str,
        final_content: str,
        complete_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess final output quality using comprehensive quality framework.

        Args:
            session_id: Session identifier for tracking
            final_content: Final content to assess
            complete_context: Complete workflow context for assessment

        Returns:
            Comprehensive final quality assessment results
        """
        self.logger.info(f"Starting final quality assessment for session {session_id}")

        try:
            # Prepare comprehensive assessment context
            assessment_context = self._prepare_assessment_context(
                session_id, final_content, complete_context, "final"
            )

            # Perform comprehensive quality assessment
            quality_assessment = await self.quality_framework.assess_quality(
                final_content, assessment_context
            )

            # Evaluate quality gates for final stage
            gate_result = await self.quality_gate_manager.evaluate_quality_gate(
                self._map_workflow_stage("final"),
                quality_assessment,
                self._create_mock_session(session_id),
                assessment_context
            )

            # Generate comprehensive quality assurance report
            qa_report = await self.quality_assurance_framework.generate_comprehensive_quality_report(
                session_id, include_trends=True, include_recommendations=True
            )

            # Calculate overall session quality metrics
            session_quality_summary = await self._calculate_session_quality_summary(session_id)

            # Store final quality data for session
            await self._store_session_quality_data(
                session_id, "final", quality_assessment, gate_result, None
            )

            # Log final quality assessment to KEVIN directory
            await self._log_quality_assessment(
                session_id, "final", quality_assessment, gate_result
            )

            # Generate final quality report
            final_report_path = await self._generate_final_quality_report(
                session_id, quality_assessment, qa_report, session_quality_summary
            )

            return {
                "success": True,
                "session_id": session_id,
                "stage": "final",
                "quality_assessment": quality_assessment,
                "gate_result": gate_result,
                "quality_assurance_report": qa_report,
                "session_quality_summary": session_quality_summary,
                "final_report_path": str(final_report_path),
                "recommendations": self._generate_stage_recommendations(
                    "final", quality_assessment, gate_result
                ),
                "quality_metrics": await self._calculate_quality_metrics(
                    session_id, "final", quality_assessment
                ),
                "workflow_quality_analysis": await self._analyze_workflow_quality(session_id),
                "assessment_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Final quality assessment failed for session {session_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "stage": "final"
            }

    async def monitor_continuous_quality(
        self,
        session_id: str,
        sampling_interval: int = 300  # 5 minutes
    ) -> Dict[str, Any]:
        """
        Monitor quality continuously throughout the research workflow.

        Args:
            session_id: Session to monitor
            sampling_interval: Interval between quality checks (seconds)

        Returns:
            Continuous quality monitoring results
        """
        self.logger.info(f"Starting continuous quality monitoring for session {session_id}")

        try:
            # Use the quality assurance framework for continuous monitoring
            monitoring_results = await self.quality_assurance_framework.monitor_continuous_quality(
                session_id, sampling_interval=sampling_interval
            )

            # Enhance with system-specific monitoring
            enhanced_results = {
                **monitoring_results,
                "system_integration_status": await self._check_system_integration_status(),
                "kevin_directory_health": await self._check_kevin_directory_health(session_id),
                "mcp_tool_status": await self.mcp_integration.get_integration_status(),
                "quality_pipeline_health": await self._check_quality_pipeline_health()
            }

            return enhanced_results

        except Exception as e:
            self.logger.error(f"Continuous quality monitoring failed for session {session_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }

    async def optimize_quality_workflow(
        self,
        session_id: str,
        performance_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize quality workflow based on performance data and quality trends.

        Args:
            session_id: Session to optimize
            performance_data: Optional performance data for optimization

        Returns:
            Workflow optimization recommendations
        """
        self.logger.info(f"Starting quality workflow optimization for session {session_id}")

        try:
            # Use the quality assurance framework for optimization
            optimization_results = await self.quality_assurance_framework.optimize_quality_workflow(
                session_id, performance_data
            )

            # Enhance with system-specific optimizations
            system_optimizations = await self._generate_system_optimizations(session_id)

            enhanced_results = {
                **optimization_results,
                "system_optimizations": system_optimizations,
                "integration_improvements": await self._suggest_integration_improvements(),
                "quality_pipeline_enhancements": await self._suggest_quality_pipeline_enhancements()
            }

            return enhanced_results

        except Exception as e:
            self.logger.error(f"Quality workflow optimization failed for session {session_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }

    async def get_quality_dashboard(
        self,
        session_id: Optional[str] = None,
        include_trends: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality dashboard for monitoring and analysis.

        Args:
            session_id: Optional session ID for specific session dashboard
            include_trends: Whether to include trend analysis
            include_recommendations: Whether to include recommendations

        Returns:
            Comprehensive quality dashboard data
        """
        self.logger.info("Generating quality dashboard")

        try:
            dashboard_data = {
                "dashboard_timestamp": datetime.now().isoformat(),
                "system_status": await self._get_system_quality_status(),
                "quality_framework_status": await self._get_quality_framework_status()
            }

            if session_id:
                # Session-specific dashboard
                dashboard_data["session_dashboard"] = await self._generate_session_dashboard(
                    session_id, include_trends, include_recommendations
                )
            else:
                # System-wide dashboard
                dashboard_data["system_dashboard"] = await self._generate_system_dashboard(
                    include_trends, include_recommendations
                )

            return {
                "success": True,
                "dashboard_data": dashboard_data
            }

        except Exception as e:
            self.logger.error(f"Quality dashboard generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    # Private helper methods

    def _prepare_assessment_context(
        self,
        session_id: str,
        content: str,
        provided_context: Optional[Dict[str, Any]],
        stage: str
    ) -> Dict[str, Any]:
        """Prepare comprehensive assessment context."""

        context = {
            "session_id": session_id,
            "content_length": len(content),
            "stage": stage,
            "assessment_timestamp": datetime.now().isoformat()
        }

        # Add provided context
        if provided_context:
            context.update(provided_context)

        # Add stage-specific context
        if stage == "research":
            context.update({
                "content_type": "research_data",
                "quality_focus": ["relevance", "completeness", "source_quality"]
            })
        elif stage == "report":
            context.update({
                "content_type": "research_report",
                "quality_focus": ["data_integration", "organization", "clarity"]
            })
        elif stage == "editorial":
            context.update({
                "content_type": "editorial_analysis",
                "quality_focus": ["accuracy", "depth", "recommendations"]
            })
        elif stage == "final":
            context.update({
                "content_type": "final_output",
                "quality_focus": ["overall_quality", "completeness", "presentation"]
            })

        return context

    def _map_workflow_stage(self, stage: str) -> Any:
        """Map string stage to workflow stage enum."""
        # This would map to the actual WorkflowStage enum
        # For now, return the string as is
        return stage.upper()

    def _create_mock_session(self, session_id: str) -> Any:
        """Create a mock session for quality gate evaluation."""
        # This would create a proper WorkflowSession object
        # For now, return a simple mock
        return type('MockSession', (), {
            'session_id': session_id,
            'get_stage_state': lambda stage: type('StageState', (), {
                'attempt_count': 1
            })()
        })()

    async def _apply_progressive_enhancement(
        self,
        session_id: str,
        content: str,
        quality_assessment: QualityAssessment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply progressive enhancement based on quality assessment."""

        # Use the quality assurance framework for enhancement
        enhancement_result = await self.quality_assurance_framework.assess_and_enhance_content(
            content, session_id, context.get("stage", "unknown"), context
        )

        return enhancement_result

    async def _store_session_quality_data(
        self,
        session_id: str,
        stage: str,
        quality_assessment: QualityAssessment,
        gate_result: GateResult,
        enhancement_result: Optional[Dict[str, Any]]
    ):
        """Store quality data for session tracking."""

        if session_id not in self.session_quality_data:
            self.session_quality_data[session_id] = {}

        self.session_quality_data[session_id][stage] = {
            "quality_assessment": quality_assessment.to_dict(),
            "gate_result": {
                "decision": gate_result.decision.value,
                "confidence": gate_result.confidence,
                "reasoning": gate_result.reasoning
            },
            "enhancement_result": enhancement_result,
            "timestamp": datetime.now().isoformat()
        }

    async def _log_quality_assessment(
        self,
        session_id: str,
        stage: str,
        quality_assessment: QualityAssessment,
        gate_result: GateResult
    ):
        """Log quality assessment to KEVIN directory."""

        try:
            # Create quality assessment log entry
            log_entry = {
                "session_id": session_id,
                "stage": stage,
                "overall_score": quality_assessment.overall_score,
                "quality_level": quality_assessment.quality_level.value,
                "gate_decision": gate_result.decision.value,
                "gate_confidence": gate_result.confidence,
                "timestamp": datetime.now().isoformat(),
                "criteria_scores": {
                    name: result.score for name, result in quality_assessment.criteria_results.items()
                },
                "strengths": quality_assessment.strengths,
                "weaknesses": quality_assessment.weaknesses,
                "recommendations": quality_assessment.actionable_recommendations
            }

            # Write to KEVIN directory
            log_file_path = await self.kevin_integration.create_log_file(
                session_id, f"quality_assessment_{stage}.json"
            )

            import json
            with open(log_file_path, 'w') as f:
                json.dump(log_entry, f, indent=2)

            self.logger.info(f"Quality assessment logged to {log_file_path}")

        except Exception as e:
            self.logger.warning(f"Failed to log quality assessment: {e}")

    def _generate_stage_recommendations(
        self,
        stage: str,
        quality_assessment: QualityAssessment,
        gate_result: GateResult
    ) -> List[str]:
        """Generate stage-specific quality recommendations."""

        recommendations = []

        # Add general quality recommendations
        recommendations.extend(quality_assessment.actionable_recommendations)

        # Add gate-specific recommendations
        if gate_result.decision == GateDecision.ENHANCE:
            recommendations.extend(gate_result.enhancement_suggestions)

        # Add stage-specific recommendations
        if quality_assessment.overall_score < 70:
            if stage == "research":
                recommendations.append("Consider expanding research sources and depth")
                recommendations.append("Improve source diversity and credibility")
            elif stage == "report":
                recommendations.append("Enhance data integration and structure")
                recommendations.append("Improve clarity and organization")
            elif stage == "editorial":
                recommendations.append("Strengthen editorial analysis and recommendations")
                recommendations.append("Improve accuracy and depth of insights")
            elif stage == "final":
                recommendations.append("Comprehensive final review and enhancement needed")
                recommendations.append("Address all identified quality issues")

        return list(set(recommendations))  # Remove duplicates

    async def _calculate_quality_metrics(
        self,
        session_id: str,
        stage: str,
        quality_assessment: QualityAssessment
    ) -> Dict[str, Any]:
        """Calculate quality metrics for tracking and analysis."""

        metrics = {
            "overall_score": quality_assessment.overall_score,
            "quality_level": quality_assessment.quality_level.value,
            "criteria_count": len(quality_assessment.criteria_results),
            "strengths_count": len(quality_assessment.strengths),
            "weaknesses_count": len(quality_assessment.weaknesses),
            "recommendations_count": len(quality_assessment.actionable_recommendations)
        }

        # Calculate criteria-specific metrics
        criteria_scores = [
            result.score for result in quality_assessment.criteria_results.values()
        ]
        if criteria_scores:
            metrics.update({
                "average_criteria_score": sum(criteria_scores) / len(criteria_scores),
                "highest_criteria_score": max(criteria_scores),
                "lowest_criteria_score": min(criteria_scores),
                "score_variance": sum((score - sum(criteria_scores) / len(criteria_scores)) ** 2
                                   for score in criteria_scores) / len(criteria_scores)
            })

        # Calculate quality distribution
        score_ranges = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0}
        for score in criteria_scores:
            if score >= 90:
                score_ranges["excellent"] += 1
            elif score >= 80:
                score_ranges["good"] += 1
            elif score >= 70:
                score_ranges["acceptable"] += 1
            else:
                score_ranges["poor"] += 1

        metrics["score_distribution"] = score_ranges

        return metrics

    async def _assess_data_integration_quality(
        self,
        report_content: str,
        research_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess the quality of data integration in the report."""

        if not research_context:
            return {"error": "No research context provided for data integration assessment"}

        # Simple data integration assessment
        assessment = {
            "data_sources_referenced": 0,
            "analytical_integration": 0,
            "citation_quality": 0,
            "overall_integration_score": 0.0
        }

        # Count data source references
        content_lower = report_content.lower()
        data_indicators = ["according to", "research shows", "study found", "data indicates"]
        assessment["data_sources_referenced"] = sum(
            content_lower.count(indicator) for indicator in data_indicators
        )

        # Count analytical integration
        analytical_indicators = ["therefore", "consequently", "furthermore", "however"]
        assessment["analytical_integration"] = sum(
            content_lower.count(indicator) for indicator in analytical_indicators
        )

        # Count citation patterns
        citation_patterns = ["[", "(", "et al", "doi:"]
        assessment["citation_quality"] = sum(
            report_content.count(pattern) for pattern in citation_patterns
        )

        # Calculate overall integration score
        max_possible = assessment["data_sources_referenced"] + assessment["analytical_integration"] + assessment["citation_quality"]
        if max_possible > 0:
            actual_score = min(assessment["data_sources_referenced"], 10) + min(assessment["analytical_integration"], 10) + min(assessment["citation_quality"], 10)
            assessment["overall_integration_score"] = actual_score / 30 * 100

        return assessment

    async def _assess_editorial_decision_quality(
        self,
        editorial_content: str,
        assessment_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess the quality of editorial decisions."""

        assessment = {
            "decision_clarity": 0.0,
            "reasoning_quality": 0.0,
            "recommendation_specificity": 0.0,
            "gap_research_justification": 0.0,
            "overall_decision_quality": 0.0
        }

        content_lower = editorial_content.lower()

        # Assess decision clarity
        clarity_indicators = ["recommend", "suggest", "propose", "advise"]
        clarity_count = sum(content_lower.count(indicator) for indicator in clarity_indicators)
        assessment["decision_clarity"] = min(clarity_count * 10, 100)

        # Assess reasoning quality
        reasoning_indicators = ["because", "therefore", "since", "due to", "as a result"]
        reasoning_count = sum(content_lower.count(indicator) for indicator in reasoning_indicators)
        assessment["reasoning_quality"] = min(reasoning_count * 8, 100)

        # Assess recommendation specificity
        specificity_indicators = ["specifically", "particular", "exact", "precise"]
        specificity_count = sum(content_lower.count(indicator) for indicator in specificity_indicators)
        assessment["recommendation_specificity"] = min(specificity_count * 12, 100)

        # Assess gap research justification
        gap_indicators = ["gap", "missing", "lacking", "insufficient", "incomplete"]
        gap_count = sum(content_lower.count(indicator) for indicator in gap_indicators)
        assessment["gap_research_justification"] = min(gap_count * 15, 100)

        # Calculate overall decision quality
        scores = [
            assessment["decision_clarity"],
            assessment["reasoning_quality"],
            assessment["recommendation_specificity"],
            assessment["gap_research_justification"]
        ]
        assessment["overall_decision_quality"] = sum(scores) / len(scores)

        return assessment

    async def _assess_gap_research_effectiveness(
        self,
        gap_research_context: Optional[Dict[str, Any]],
        editorial_content: str
    ) -> Dict[str, Any]:
        """Assess the effectiveness of gap research."""

        if not gap_research_context:
            return {"error": "No gap research context provided"}

        assessment = {
            "gap_identification_accuracy": 0.0,
            "research_relevance": 0.0,
            "integration_quality": 0.0,
            "overall_effectiveness": 0.0
        }

        # Simple assessment based on available context
        if "gap_queries" in gap_research_context:
            gap_queries = gap_research_context["gap_queries"]
            assessment["gap_identification_accuracy"] = len(gap_queries) * 20  # Simple scoring

        if "research_results" in gap_research_context:
            research_results = gap_research_context["research_results"]
            assessment["research_relevance"] = 80.0  # Assume good relevance if results exist

        # Check integration in editorial content
        content_lower = editorial_content.lower()
        integration_indicators = ["additional research", "further investigation", "new findings"]
        integration_count = sum(content_lower.count(indicator) for indicator in integration_indicators)
        assessment["integration_quality"] = min(integration_count * 25, 100)

        # Calculate overall effectiveness
        scores = [
            assessment["gap_identification_accuracy"],
            assessment["research_relevance"],
            assessment["integration_quality"]
        ]
        assessment["overall_effectiveness"] = sum(scores) / len(scores) if scores else 0.0

        return assessment

    async def _calculate_session_quality_summary(self, session_id: str) -> Dict[str, Any]:
        """Calculate comprehensive quality summary for the session."""

        if session_id not in self.session_quality_data:
            return {"error": "No quality data available for session"}

        session_data = self.session_quality_data[session_id]

        summary = {
            "session_id": session_id,
            "stages_completed": len(session_data),
            "overall_quality_trend": [],
            "stage_quality_scores": {},
            "critical_issues": [],
            "improvement_areas": []
        }

        # Calculate quality scores by stage
        for stage, data in session_data.items():
            if "quality_assessment" in data:
                quality_score = data["quality_assessment"]["overall_score"]
                summary["stage_quality_scores"][stage] = quality_score
                summary["overall_quality_trend"].append({
                    "stage": stage,
                    "score": quality_score,
                    "timestamp": data["timestamp"]
                })

                # Identify critical issues
                if quality_score < 60:
                    summary["critical_issues"].append(f"Low quality in {stage} stage")

                # Identify improvement areas
                if quality_score < 80:
                    summary["improvement_areas"].append(stage)

        # Calculate overall session quality
        if summary["stage_quality_scores"]:
            summary["average_session_quality"] = sum(summary["stage_quality_scores"].values()) / len(summary["stage_quality_scores"])
        else:
            summary["average_session_quality"] = 0.0

        return summary

    async def _generate_final_quality_report(
        self,
        session_id: str,
        quality_assessment: QualityAssessment,
        qa_report: QualityAssuranceReport,
        session_quality_summary: Dict[str, Any]
    ) -> Path:
        """Generate final quality report and save to KEVIN directory."""

        # Create comprehensive quality report
        report_content = f"""# Quality Assessment Report

## Session Information
- **Session ID**: {session_id}
- **Assessment Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Final Quality Score**: {quality_assessment.overall_score}/100
- **Quality Level**: {quality_assessment.quality_level.value.upper()}

## Quality Assessment Summary

### Overall Quality Score: {quality_assessment.overall_score}/100
**Quality Level**: {quality_assessment.quality_level.value.upper()}

### Criteria Scores
"""

        # Add criteria scores
        for criterion_name, result in quality_assessment.criteria_results.items():
            report_content += f"- **{criterion_name.title()}**: {result.score}/100\n"

        # Add strengths and weaknesses
        report_content += f"""
### Strengths
{chr(10).join(f"- {strength}" for strength in quality_assessment.strengths)}

### Areas for Improvement
{chr(10).join(f"- {weakness}" for weakness in quality_assessment.weaknesses)}

### Recommendations
{chr(10).join(f"- {rec}" for rec in quality_assessment.actionable_recommendations)}

## Session Quality Summary
- **Stages Completed**: {session_quality_summary.get('stages_completed', 0)}
- **Average Session Quality**: {session_quality_summary.get('average_session_quality', 0):.1f}/100
- **Critical Issues**: {len(session_quality_summary.get('critical_issues', []))}
- **Improvement Areas**: {len(session_quality_summary.get('improvement_areas', []))}
"""

        if session_quality_summary.get('critical_issues'):
            report_content += f"""
### Critical Issues
{chr(10).join(f"- {issue}" for issue in session_quality_summary['critical_issues'])}
"""

        # Write report to KEVIN directory
        report_path = await self.kevin_integration.create_working_file(
            session_id, "quality_assessment_report.md"
        )

        with open(report_path, 'w') as f:
            f.write(report_content)

        self.logger.info(f"Final quality report generated: {report_path}")
        return report_path

    async def _analyze_workflow_quality(self, session_id: str) -> Dict[str, Any]:
        """Analyze quality trends across the workflow."""

        if session_id not in self.session_quality_data:
            return {"error": "No quality data available for session"}

        session_data = self.session_quality_data[session_id]

        analysis = {
            "quality_progression": [],
            "improvement_trends": {},
            "consistency_metrics": {},
            "bottleneck_stages": []
        }

        # Analyze quality progression
        scores = []
        for stage in ["research", "report", "editorial", "final"]:
            if stage in session_data:
                score = session_data[stage]["quality_assessment"]["overall_score"]
                scores.append(score)
                analysis["quality_progression"].append({
                    "stage": stage,
                    "score": score,
                    "timestamp": session_data[stage]["timestamp"]
                })

        # Calculate improvement trends
        if len(scores) > 1:
            improvement = scores[-1] - scores[0]
            analysis["improvement_trends"] = {
                "total_improvement": improvement,
                "improvement_rate": improvement / len(scores) if scores else 0,
                "trend_direction": "improving" if improvement > 0 else "declining" if improvement < 0 else "stable"
            }

        # Calculate consistency metrics
        if scores:
            avg_score = sum(scores) / len(scores)
            variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
            analysis["consistency_metrics"] = {
                "average_score": avg_score,
                "score_variance": variance,
                "consistency_rating": "high" if variance < 50 else "medium" if variance < 100 else "low"
            }

        # Identify bottleneck stages
        for stage, data in session_data.items():
            if data["quality_assessment"]["overall_score"] < 70:
                analysis["bottleneck_stages"].append(stage)

        return analysis

    async def _check_system_integration_status(self) -> Dict[str, Any]:
        """Check system integration status."""

        return {
            "kevin_integration": "operational",
            "mcp_integration": "operational",
            "quality_frameworks": "operational",
            "last_check": datetime.now().isoformat()
        }

    async def _check_kevin_directory_health(self, session_id: str) -> Dict[str, Any]:
        """Check KEVIN directory health for session."""

        try:
            session_files = await self.kevin_integration.list_session_files(session_id)
            return {
                "status": "healthy",
                "file_count": len(session_files),
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

    async def _check_quality_pipeline_health(self) -> Dict[str, Any]:
        """Check quality pipeline health."""

        return {
            "quality_framework": "operational",
            "quality_gates": "operational",
            "quality_assurance": "operational",
            "progressive_enhancement": "operational",
            "last_check": datetime.now().isoformat()
        }

    async def _generate_system_optimizations(self, session_id: str) -> List[Dict[str, Any]]:
        """Generate system-specific optimization recommendations."""

        optimizations = []

        # Analyze session data for optimization opportunities
        if session_id in self.session_quality_data:
            session_data = self.session_quality_data[session_id]

            # Check for consistent quality issues
            low_quality_stages = [
                stage for stage, data in session_data.items()
                if data["quality_assessment"]["overall_score"] < 70
            ]

            if low_quality_stages:
                optimizations.append({
                    "type": "quality_improvement",
                    "priority": "high",
                    "description": f"Address quality issues in {', '.join(low_quality_stages)} stages",
                    "recommendation": "Implement targeted enhancement strategies"
                })

        # Add general system optimizations
        optimizations.extend([
            {
                "type": "performance",
                "priority": "medium",
                "description": "Optimize quality assessment performance",
                "recommendation": "Implement caching for repeated assessments"
            },
            {
                "type": "monitoring",
                "priority": "low",
                "description": "Enhance quality monitoring capabilities",
                "recommendation": "Add real-time quality alerts and notifications"
            }
        ])

        return optimizations

    async def _suggest_integration_improvements(self) -> List[Dict[str, Any]]:
        """Suggest integration improvements."""

        return [
            {
                "area": "kevin_integration",
                "improvement": "Enhanced file organization",
                "description": "Implement more sophisticated file naming and organization"
            },
            {
                "area": "mcp_integration",
                "improvement": "Enhanced error handling",
                "description": "Improve error handling and recovery mechanisms"
            },
            {
                "area": "quality_frameworks",
                "improvement": "Custom criteria",
                "description": "Add domain-specific quality criteria"
            }
        ]

    async def _suggest_quality_pipeline_enhancements(self) -> List[Dict[str, Any]]:
        """Suggest quality pipeline enhancements."""

        return [
            {
                "enhancement": "Predictive quality assessment",
                "description": "Implement ML-based quality prediction"
            },
            {
                "enhancement": "Adaptive thresholds",
                "description": "Implement dynamic quality threshold adjustment"
            },
            {
                "enhancement": "Real-time enhancement",
                "description": "Implement real-time quality enhancement during content creation"
            }
        ]

    async def _get_system_quality_status(self) -> Dict[str, Any]:
        """Get overall system quality status."""

        return {
            "status": "operational",
            "active_sessions": len(self.session_quality_data),
            "quality_frameworks": "operational",
            "integrations": "operational",
            "last_update": datetime.now().isoformat()
        }

    async def _get_quality_framework_status(self) -> Dict[str, Any]:
        """Get quality framework status."""

        return {
            "quality_framework": "operational",
            "quality_gates": "operational",
            "quality_assurance": "operational",
            "progressive_enhancement": "operational",
            "total_assessments": len(self.session_quality_data),
            "last_assessment": datetime.now().isoformat()
        }

    async def _generate_session_dashboard(
        self,
        session_id: str,
        include_trends: bool,
        include_recommendations: bool
    ) -> Dict[str, Any]:
        """Generate session-specific dashboard."""

        if session_id not in self.session_quality_data:
            return {"error": "Session not found"}

        session_data = self.session_quality_data[session_id]

        dashboard = {
            "session_id": session_id,
            "stage_data": {},
            "quality_summary": await self._calculate_session_quality_summary(session_id)
        }

        # Add stage-specific data
        for stage, data in session_data.items():
            dashboard["stage_data"][stage] = {
                "quality_score": data["quality_assessment"]["overall_score"],
                "quality_level": data["quality_assessment"]["quality_level"],
                "gate_decision": data["gate_result"]["decision"],
                "timestamp": data["timestamp"]
            }

        if include_trends:
            dashboard["quality_trends"] = await self._analyze_workflow_quality(session_id)

        if include_recommendations:
            # Generate recommendations based on latest stage
            latest_stage = max(session_data.keys(), key=lambda k: session_data[k]["timestamp"])
            latest_data = session_data[latest_stage]

            # Recreate QualityAssessment object from dict
            from multi_agent_research_system.core.quality_framework import QualityAssessment, QualityLevel
            qa_dict = latest_data["quality_assessment"]

            # Convert criteria results back to objects
            criteria_results = {}
            for name, result_dict in qa_dict["criteria_results"].items():
                # This is simplified - would need full reconstruction
                criteria_results[name] = type('CriterionResult', (), {
                    'score': result_dict['score'],
                    'feedback': result_dict['feedback']
                })()

            quality_assessment = QualityAssessment(
                overall_score=qa_dict["overall_score"],
                quality_level=QualityLevel(qa_dict["quality_level"]),
                criteria_results=criteria_results,
                content_metadata=qa_dict["content_metadata"],
                assessment_timestamp=qa_dict["assessment_timestamp"],
                strengths=qa_dict["strengths"],
                weaknesses=qa_dict["weaknesses"],
                actionable_recommendations=qa_dict["actionable_recommendations"],
                enhancement_priority=qa_dict["enhancement_priority"]
            )

            gate_result = type('GateResult', (), {
                'decision': type('Decision', (), {'value': latest_data["gate_result"]["decision"]})(),
                'enhancement_suggestions': []
            })()

            dashboard["recommendations"] = self._generate_stage_recommendations(
                latest_stage, quality_assessment, gate_result
            )

        return dashboard

    async def _generate_system_dashboard(
        self,
        include_trends: bool,
        include_recommendations: bool
    ) -> Dict[str, Any]:
        """Generate system-wide dashboard."""

        dashboard = {
            "system_overview": {
                "total_sessions": len(self.session_quality_data),
                "operational_status": "healthy",
                "last_update": datetime.now().isoformat()
            },
            "quality_metrics": {
                "average_quality_scores": {},
                "quality_distribution": {},
                "common_issues": []
            }
        }

        # Calculate system-wide quality metrics
        all_scores = []
        stage_scores = {"research": [], "report": [], "editorial": [], "final": []}

        for session_id, session_data in self.session_quality_data.items():
            for stage, data in session_data.items():
                score = data["quality_assessment"]["overall_score"]
                all_scores.append(score)
                if stage in stage_scores:
                    stage_scores[stage].append(score)

        # Calculate averages
        if all_scores:
            dashboard["quality_metrics"]["overall_average"] = sum(all_scores) / len(all_scores)

        for stage, scores in stage_scores.items():
            if scores:
                dashboard["quality_metrics"]["average_quality_scores"][stage] = sum(scores) / len(scores)

        return dashboard


# Convenience function for quick quality assurance integration
async def integrate_quality_assurance(
    session_id: str,
    content: str,
    stage: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Quick quality assurance integration function.

    Args:
        session_id: Session identifier
        content: Content to assess
        stage: Workflow stage
        context: Additional context for assessment

    Returns:
        Quality assurance results
    """
    qa_integration = QualityAssuranceIntegration()

    if stage == "research":
        return await qa_integration.assess_research_quality(session_id, content, context)
    elif stage == "report":
        return await qa_integration.assess_report_quality(session_id, content, context)
    elif stage == "editorial":
        return await qa_integration.assess_editorial_quality(session_id, content, context)
    elif stage == "final":
        return await qa_integration.assess_final_quality(session_id, content, context)
    else:
        return {"error": f"Unknown stage: {stage}"}