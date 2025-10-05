"""
Quality Gate Management for Multi-Agent Research System

Provides intelligent quality-based workflow progression with configurable thresholds,
adaptive criteria, and sophisticated decision-making capabilities.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .quality_framework import QualityAssessment, QualityFramework
from .workflow_state import WorkflowSession, WorkflowStage


class GateDecision(Enum):
    """Quality gate decisions."""
    PROCEED = "proceed"
    ENHANCE = "enhance"
    RERUN = "rerun"
    ESCALATE = "escalate"
    SKIP = "skip"


class QualityThreshold(Enum):
    """Quality threshold levels."""
    MINIMAL = 60  # Minimum acceptable quality
    STANDARD = 75  # Standard quality expectation
    HIGH = 85  # High quality standard
    EXCELLENT = 95  # Excellent quality standard


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""
    stage: WorkflowStage
    threshold: QualityThreshold
    required_criteria: list[str] = field(default_factory=list)
    optional_criteria: list[str] = field(default_factory=list)
    enhancement_triggers: dict[str, int] = field(default_factory=dict)
    max_enhancement_attempts: int = 3
    allow_skip: bool = False
    skip_conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    """Result of quality gate evaluation."""
    decision: GateDecision
    confidence: float  # 0.0 to 1.0
    reasoning: str
    assessment: QualityAssessment
    triggered_criteria: list[str] = field(default_factory=list)
    enhancement_suggestions: list[str] = field(default_factory=list)
    fallback_available: bool = False
    next_stage: WorkflowStage | None = None
    estimated_effort: str | None = None


class QualityGateManager:
    """Manages quality gate decisions and workflow progression."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.quality_framework = QualityFramework()
        self.gate_configs: dict[WorkflowStage, QualityGateConfig] = {}
        self.decision_history: list[dict[str, Any]] = []
        self._initialize_default_gates()

    def _initialize_default_gates(self):
        """Initialize default quality gate configurations."""

        # Research stage gate - requires good content gathering
        self.gate_configs[WorkflowStage.RESEARCH] = QualityGateConfig(
            stage=WorkflowStage.RESEARCH,
            threshold=QualityThreshold.STANDARD,
            required_criteria=["relevance", "completeness"],
            optional_criteria=["depth"],
            enhancement_triggers={
                "relevance": 70,
                "completeness": 65,
                "depth": 60
            },
            max_enhancement_attempts=2,
            allow_skip=False
        )

        # Report generation gate - requires comprehensive coverage
        self.gate_configs[WorkflowStage.REPORT_GENERATION] = QualityGateConfig(
            stage=WorkflowStage.REPORT_GENERATION,
            threshold=QualityThreshold.STANDARD,
            required_criteria=["completeness", "organization", "clarity"],
            optional_criteria=["relevance", "depth"],
            enhancement_triggers={
                "completeness": 70,
                "organization": 75,
                "clarity": 70,
                "depth": 60
            },
            max_enhancement_attempts=3,
            allow_skip=False
        )

        # Editorial review gate - high standards for editorial quality
        self.gate_configs[WorkflowStage.EDITORIAL_REVIEW] = QualityGateConfig(
            stage=WorkflowStage.EDITORIAL_REVIEW,
            threshold=QualityThreshold.HIGH,
            required_criteria=["clarity", "organization", "accuracy"],
            optional_criteria=["relevance", "completeness", "depth"],
            enhancement_triggers={
                "clarity": 80,
                "organization": 80,
                "accuracy": 85,
                "completeness": 75
            },
            max_enhancement_attempts=2,
            allow_skip=True,
            skip_conditions={
                "decoupled_available": True,
                "minimal_quality": 70
            }
        )

        # Decoupled editorial review gate - very high standards
        self.gate_configs[WorkflowStage.DECOUPLED_EDITORIAL_REVIEW] = QualityGateConfig(
            stage=WorkflowStage.DECOUPLED_EDITORIAL_REVIEW,
            threshold=QualityThreshold.HIGH,
            required_criteria=["clarity", "organization", "accuracy"],
            optional_criteria=["relevance", "completeness", "depth"],
            enhancement_triggers={
                "clarity": 80,
                "organization": 80,
                "accuracy": 85,
                "completeness": 75,
                "depth": 70
            },
            max_enhancement_attempts=3,
            allow_skip=False
        )

        # Quality assessment gate - comprehensive quality check
        self.gate_configs[WorkflowStage.QUALITY_ASSESSMENT] = QualityGateConfig(
            stage=WorkflowStage.QUALITY_ASSESSMENT,
            threshold=QualityThreshold.STANDARD,
            required_criteria=["relevance", "completeness", "accuracy"],
            optional_criteria=["clarity", "organization", "depth"],
            enhancement_triggers={
                "relevance": 75,
                "completeness": 70,
                "accuracy": 80,
                "organization": 70,
                "clarity": 70
            },
            max_enhancement_attempts=2,
            allow_skip=True,
            skip_conditions={
                "overall_score": 80,
                "critical_criteria_met": True
            }
        )

        # Progressive enhancement gate - flexible standards
        self.gate_configs[WorkflowStage.PROGRESSIVE_ENHANCEMENT] = QualityGateConfig(
            stage=WorkflowStage.PROGRESSIVE_ENHANCEMENT,
            threshold=QualityThreshold.MINIMAL,
            required_criteria=["relevance", "accuracy"],
            optional_criteria=["completeness", "clarity", "organization", "depth"],
            enhancement_triggers={
                "relevance": 65,
                "accuracy": 70,
                "completeness": 60
            },
            max_enhancement_attempts=4,
            allow_skip=True,
            skip_conditions={
                "enhancement_exhausted": True,
                "minimal_quality_met": True
            }
        )

        # Final output gate - highest standards
        self.gate_configs[WorkflowStage.FINAL_OUTPUT] = QualityGateConfig(
            stage=WorkflowStage.FINAL_OUTPUT,
            threshold=QualityThreshold.EXCELLENT,
            required_criteria=["relevance", "completeness", "accuracy", "clarity", "organization"],
            optional_criteria=["depth"],
            enhancement_triggers={
                "relevance": 90,
                "completeness": 85,
                "accuracy": 90,
                "clarity": 85,
                "organization": 85,
                "depth": 80
            },
            max_enhancement_attempts=2,
            allow_skip=False
        )

    def evaluate_quality_gate(
        self,
        stage: WorkflowStage,
        assessment: QualityAssessment,
        session: WorkflowSession,
        context: dict[str, Any] | None = None
    ) -> GateResult:
        """Evaluate quality gate for a specific stage."""

        self.logger.info(f"Evaluating quality gate for stage: {stage.value}")

        # Get gate configuration
        gate_config = self.gate_configs.get(stage)
        if not gate_config:
            self.logger.warning(f"No gate configuration for stage: {stage.value}")
            return GateResult(
                decision=GateDecision.PROCEED,
                confidence=0.8,
                reasoning="No quality gate configured - proceeding by default",
                assessment=assessment
            )

        # Evaluate against thresholds
        threshold_value = gate_config.threshold.value
        overall_score = assessment.overall_score

        # Check required criteria
        failed_required = []
        passed_required = []
        for criterion_name in gate_config.required_criteria:
            criterion_score = assessment.get_criterion_score(criterion_name)
            if criterion_score < threshold_value:
                failed_required.append(criterion_name)
            else:
                passed_required.append(criterion_name)

        # Check enhancement triggers
        enhancement_needed = []
        for criterion_name, trigger_score in gate_config.enhancement_triggers.items():
            criterion_score = assessment.get_criterion_score(criterion_name)
            if criterion_score < trigger_score:
                enhancement_needed.append(criterion_name)

        # Make decision
        decision, confidence, reasoning = self._make_gate_decision(
            stage, assessment, gate_config,
            overall_score, failed_required,
            enhancement_needed, session, context
        )

        # Generate enhancement suggestions if needed
        enhancement_suggestions = []
        if decision in [GateDecision.ENHANCE, GateDecision.RERUN]:
            enhancement_suggestions = self._generate_enhancement_suggestions(
                assessment, enhancement_needed, gate_config
            )

        # Determine next stage
        next_stage = self._determine_next_stage(stage, decision, session)

        # Check fallback availability
        fallback_available = self._check_fallback_availability(stage, session, context)

        # Estimate effort
        estimated_effort = self._estimate_enhancement_effort(
            decision, enhancement_needed, gate_config
        )

        result = GateResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            assessment=assessment,
            triggered_criteria=failed_required + enhancement_needed,
            enhancement_suggestions=enhancement_suggestions,
            fallback_available=fallback_available,
            next_stage=next_stage,
            estimated_effort=estimated_effort
        )

        # Record decision
        self._record_gate_decision(stage, result, session)

        self.logger.info(f"Quality gate decision: {decision.value} (confidence: {confidence:.2f})")
        return result

    def _make_gate_decision(
        self,
        stage: WorkflowStage,
        assessment: QualityAssessment,
        gate_config: QualityGateConfig,
        overall_score: int,
        failed_required: list[str],
        enhancement_needed: list[str],
        session: WorkflowSession,
        context: dict[str, Any] | None
    ) -> tuple[GateDecision, float, str]:
        """Make quality gate decision based on assessment."""

        threshold_value = gate_config.threshold.value

        # Critical failures - must rerun
        if failed_required:
            return GateDecision.RERUN, 0.9, f"Required criteria failed: {', '.join(failed_required)}"

        # High quality - proceed
        if overall_score >= threshold_value + 10:
            return GateDecision.PROCEED, 0.95, f"Excellent quality score: {overall_score}"

        # Acceptable quality - proceed
        if overall_score >= threshold_value:
            confidence = 0.8 + (overall_score - threshold_value) / 50
            return GateDecision.PROCEED, confidence, f"Acceptable quality score: {overall_score}"

        # Below threshold but close - enhance
        if overall_score >= threshold_value - 15:
            stage_state = session.get_stage_state(stage)
            if stage_state.attempt_count < gate_config.max_enhancement_attempts:
                return GateDecision.ENHANCE, 0.8, f"Quality below threshold, enhancement recommended: {overall_score}"
            else:
                # Enhancement attempts exhausted
                if gate_config.allow_skip and self._check_skip_conditions(gate_config, assessment, session, context):
                    return GateDecision.SKIP, 0.6, f"Enhancement attempts exhausted, skipping allowed: {overall_score}"
                else:
                    return GateDecision.ESCALATE, 0.9, f"Enhancement attempts exhausted, escalation required: {overall_score}"

        # Significantly below threshold - escalate or rerun
        if overall_score >= threshold_value - 30:
            return GateDecision.ESCALATE, 0.85, f"Quality significantly below threshold: {overall_score}"

        # Very poor quality - rerun
        return GateDecision.RERUN, 0.95, f"Very poor quality, rerun required: {overall_score}"

    def _generate_enhancement_suggestions(
        self,
        assessment: QualityAssessment,
        enhancement_needed: list[str],
        gate_config: QualityGateConfig
    ) -> list[str]:
        """Generate specific enhancement suggestions."""

        suggestions = []

        for criterion_name in enhancement_needed:
            criterion_result = assessment.criteria_results.get(criterion_name)
            if criterion_result and criterion_result.feedback:
                suggestions.extend(criterion_result.recommendations)

            # Add general suggestions based on criterion type
            if criterion_name == "relevance":
                suggestions.append("Focus research more closely on the core topic requirements")
                suggestions.append("Remove peripheral or tangential content")
            elif criterion_name == "completeness":
                suggestions.append("Add missing information and fill content gaps")
                suggestions.append("Expand coverage of underdeveloped sections")
            elif criterion_name == "accuracy":
                suggestions.append("Verify facts, figures, and sources")
                suggestions.append("Correct any factual errors or inconsistencies")
            elif criterion_name == "clarity":
                suggestions.append("Improve sentence structure and readability")
                suggestions.append("Add explanations for complex concepts")
            elif criterion_name == "organization":
                suggestions.append("Restructure content for better flow")
                suggestions.append("Improve section transitions and logical progression")
            elif criterion_name == "depth":
                suggestions.append("Add more detailed analysis and examples")
                suggestions.append("Expand on key concepts and implications")

        return list(set(suggestions))  # Remove duplicates

    def _determine_next_stage(
        self,
        current_stage: WorkflowStage,
        decision: GateDecision,
        session: WorkflowSession
    ) -> WorkflowStage | None:
        """Determine the next stage based on gate decision."""

        if decision == GateDecision.PROCEED:
            return session._get_next_stage(current_stage)
        elif decision == GateDecision.ENHANCE:
            # Stay in current stage for enhancement
            return current_stage
        elif decision == GateDecision.RERUN:
            # Rerun current stage
            return current_stage
        elif decision == GateDecision.ESCALATE:
            # Escalate to higher authority or alternative approach
            return self._get_escalation_stage(current_stage)
        elif decision == GateDecision.SKIP:
            # Skip to next stage
            return session._get_next_stage(current_stage)

        return None

    def _get_escalation_stage(self, current_stage: WorkflowStage) -> WorkflowStage:
        """Get escalation stage for a given stage."""

        escalation_map = {
            WorkflowStage.RESEARCH: WorkflowStage.RESEARCH,  # Rerun with different approach
            WorkflowStage.REPORT_GENERATION: WorkflowStage.EDITORIAL_REVIEW,  # Skip to editorial
            WorkflowStage.EDITORIAL_REVIEW: WorkflowStage.DECOUPLED_EDITORIAL_REVIEW,  # Use decoupled
            WorkflowStage.DECOUPLED_EDITORIAL_REVIEW: WorkflowStage.PROGRESSIVE_ENHANCEMENT,  # Try enhancement
            WorkflowStage.QUALITY_ASSESSMENT: WorkflowStage.PROGRESSIVE_ENHANCEMENT,
            WorkflowStage.PROGRESSIVE_ENHANCEMENT: WorkflowStage.FINAL_OUTPUT,  # Force completion
            WorkflowStage.FINAL_OUTPUT: WorkflowStage.FINAL_OUTPUT  # Must complete
        }

        return escalation_map.get(current_stage, current_stage)

    def _check_fallback_availability(
        self,
        stage: WorkflowStage,
        session: WorkflowSession,
        context: dict[str, Any] | None
    ) -> bool:
        """Check if fallback options are available."""

        # Check for decoupled editorial fallback
        if stage == WorkflowStage.EDITORIAL_REVIEW:
            return context.get("decoupled_available", False) if context else False

        # Check for alternative research approaches
        if stage == WorkflowStage.RESEARCH:
            return session.get_stage_state(stage).attempt_count < 3

        # Check for progressive enhancement availability
        if stage in [WorkflowStage.DECOUPLED_EDITORIAL_REVIEW, WorkflowStage.QUALITY_ASSESSMENT]:
            return True

        return False

    def _check_skip_conditions(
        self,
        gate_config: QualityGateConfig,
        assessment: QualityAssessment,
        session: WorkflowSession,
        context: dict[str, Any] | None
    ) -> bool:
        """Check if skip conditions are met."""

        if not gate_config.allow_skip:
            return False

        skip_conditions = gate_config.skip_conditions

        # Check minimal quality condition
        if "minimal_quality" in skip_conditions:
            if assessment.overall_score < skip_conditions["minimal_quality"]:
                return False

        # Check decoupled availability
        if "decoupled_available" in skip_conditions:
            if not context or not context.get("decoupled_available", False):
                return False

        # Check overall score condition
        if "overall_score" in skip_conditions:
            if assessment.overall_score < skip_conditions["overall_score"]:
                return False

        # Check critical criteria condition
        if "critical_criteria_met" in skip_conditions:
            critical_criteria = ["relevance", "accuracy"]  # Define critical criteria
            for criterion in critical_criteria:
                if assessment.get_criterion_score(criterion) < 70:
                    return False

        # Check enhancement exhaustion
        if "enhancement_exhausted" in skip_conditions:
            stage = gate_config.stage
            stage_state = session.get_stage_state(stage)
            if stage_state.attempt_count < gate_config.max_enhancement_attempts:
                return False

        return True

    def _estimate_enhancement_effort(
        self,
        decision: GateDecision,
        enhancement_needed: list[str],
        gate_config: QualityGateConfig
    ) -> str | None:
        """Estimate the effort required for enhancement."""

        if decision not in [GateDecision.ENHANCE, GateDecision.RERUN]:
            return None

        # Base effort on number of criteria needing improvement
        criteria_count = len(enhancement_needed)

        if criteria_count <= 1:
            return "Low"
        elif criteria_count <= 3:
            return "Medium"
        elif criteria_count <= 5:
            return "High"
        else:
            return "Very High"

    def _record_gate_decision(
        self,
        stage: WorkflowStage,
        result: GateResult,
        session: WorkflowSession
    ):
        """Record quality gate decision for analysis."""

        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session.session_id,
            "stage": stage.value,
            "decision": result.decision.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "overall_score": result.assessment.overall_score,
            "triggered_criteria": result.triggered_criteria,
            "enhancement_suggestions": result.enhancement_suggestions,
            "estimated_effort": result.estimated_effort
        }

        self.decision_history.append(decision_record)

        # Limit history size
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]

    def get_gate_statistics(self) -> dict[str, Any]:
        """Get statistics on quality gate decisions."""

        if not self.decision_history:
            return {"total_decisions": 0}

        stats = {
            "total_decisions": len(self.decision_history),
            "decisions_by_type": {},
            "decisions_by_stage": {},
            "average_confidence": 0.0,
            "average_scores": {},
            "recent_decisions": self.decision_history[-10:]
        }

        total_confidence = 0.0
        scores_by_stage = {}

        for record in self.decision_history:
            # Count by decision type
            decision = record["decision"]
            stats["decisions_by_type"][decision] = stats["decisions_by_type"].get(decision, 0) + 1

            # Count by stage
            stage = record["stage"]
            stats["decisions_by_stage"][stage] = stats["decisions_by_stage"].get(stage, 0) + 1

            # Accumulate confidence
            total_confidence += record["confidence"]

            # Accumulate scores by stage
            if stage not in scores_by_stage:
                scores_by_stage[stage] = []
            scores_by_stage[stage].append(record["overall_score"])

        stats["average_confidence"] = total_confidence / len(self.decision_history)

        # Calculate average scores by stage
        for stage, scores in scores_by_stage.items():
            stats["average_scores"][stage] = sum(scores) / len(scores)

        return stats

    def update_gate_config(
        self,
        stage: WorkflowStage,
        config: QualityGateConfig
    ):
        """Update quality gate configuration."""

        self.gate_configs[stage] = config
        self.logger.info(f"Updated gate configuration for stage: {stage.value}")

    def reset_history(self):
        """Reset decision history."""
        self.decision_history = []
        self.logger.info("Reset quality gate decision history")
