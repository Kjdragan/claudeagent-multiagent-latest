"""
Editorial Decision Engine

Phase 1.3.2: Create content cleaning pipeline with quality validation (continued)

This module provides automated editorial decisions based on confidence scores
and quality assessments. Integrates with the content cleaning pipeline to provide
intelligent recommendations for content acceptance, enhancement, or rejection.

Key Features:
- Automated editorial decisions based on confidence thresholds
- Gap research trigger logic
- Content enhancement recommendations
- Quality-based routing decisions
- Integration with multi-agent workflow
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .fast_confidence_scorer import ConfidenceSignals
from .content_cleaning_pipeline import CleaningResult

logger = logging.getLogger(__name__)


class EditorialDecision(Enum):
    """Editorial decision types."""

    ACCEPT_CONTENT = "ACCEPT_CONTENT"
    ENHANCE_CONTENT = "ENHANCE_CONTENT"
    GAP_RESEARCH = "GAP_RESEARCH"
    REJECT_CONTENT = "REJECT_CONTENT"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class PriorityLevel(Enum):
    """Priority levels for content processing."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class EditorialAction:
    """Recommended editorial action."""

    decision: EditorialDecision
    priority: PriorityLevel
    reasoning: str
    confidence: float
    estimated_effort: str  # "low", "medium", "high"
    suggested_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGate:
    """Quality gate configuration for editorial decisions."""

    name: str
    threshold: float
    weight: float
    description: str
    failure_action: EditorialDecision


class EditorialDecisionEngine:
    """
    Automated editorial decision engine based on confidence scoring.

    Provides intelligent recommendations for content processing based on
    quality assessments and confidence signals.
    """

    def __init__(self, custom_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the editorial decision engine.

        Args:
            custom_thresholds: Custom threshold overrides
        """
        # Quality thresholds - UPDATED FOR HIGH THRESHOLDS per repair2.md
        self.thresholds = {
            'gap_research_trigger': 0.9,   # INCREASED: Trigger gap research below this (was 0.7)
            'acceptable_quality': 0.6,     # Minimum acceptable quality
            'good_quality': 0.8,           # Good quality threshold
            'excellent_quality': 0.9,      # Excellent quality threshold
            'critical_failure': 0.3        # Critical failure threshold
        }

        # Apply custom thresholds if provided
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

        # Quality gates for different content aspects
        self.quality_gates = [
            QualityGate(
                name="overall_confidence",
                threshold=self.thresholds['acceptable_quality'],
                weight=0.3,
                description="Overall content confidence and quality",
                failure_action=EditorialDecision.GAP_RESEARCH
            ),
            QualityGate(
                name="content_length",
                threshold=0.5,
                weight=0.1,
                description="Adequate content length and depth",
                failure_action=EditorialDecision.GAP_RESEARCH
            ),
            QualityGate(
                name="domain_authority",
                threshold=0.6,
                weight=0.2,
                description="Source authority and credibility",
                failure_action=EditorialDecision.GAP_RESEARCH
            ),
            QualityGate(
                name="relevance",
                threshold=0.7,
                weight=0.25,
                description="Content relevance to search query",
                failure_action=EditorialDecision.GAP_RESEARCH
            ),
            QualityGate(
                name="cleanliness",
                threshold=0.6,
                weight=0.15,
                description="Content cleanliness and structure",
                failure_action=EditorialDecision.ENHANCE_CONTENT
            )
        ]

        logger.info("EditorialDecisionEngine initialized with confidence-based decisions")

    def evaluate_content(self, confidence_signals: ConfidenceSignals) -> EditorialAction:
        """
        Evaluate content and recommend editorial action.

        Args:
            confidence_signals: Confidence assessment from FastConfidenceScorer

        Returns:
            EditorialAction with recommendation and reasoning
        """
        logger.debug(f"Evaluating content with overall confidence: {confidence_signals.overall_confidence:.3f}")

        # Check for critical failures first
        if confidence_signals.overall_confidence < self.thresholds['critical_failure']:
            return self._create_critical_failure_action(confidence_signals)

        # Evaluate quality gates
        failed_gates = self._evaluate_quality_gates(confidence_signals)

        # Determine primary decision based on overall confidence
        if confidence_signals.overall_confidence >= self.thresholds['excellent_quality']:
            return self._create_accept_action(confidence_signals, failed_gates)
        elif confidence_signals.overall_confidence >= self.thresholds['good_quality']:
            return self._create_enhance_action(confidence_signals, failed_gates)
        elif confidence_signals.overall_confidence >= self.thresholds['gap_research_trigger']:
            return self._create_gap_research_action(confidence_signals, failed_gates)
        else:
            return self._create_reject_action(confidence_signals, failed_gates)

    def evaluate_cleaning_result(self, cleaning_result: CleaningResult) -> EditorialAction:
        """
        Evaluate content cleaning result and recommend next steps.

        Args:
            cleaning_result: Result from ContentCleaningPipeline

        Returns:
            EditorialAction for processed content
        """
        if not cleaning_result.confidence_signals:
            # No confidence assessment available
            return EditorialAction(
                decision=EditorialDecision.MANUAL_REVIEW,
                priority=PriorityLevel.MEDIUM,
                reasoning="No quality assessment available - requires manual review",
                confidence=0.0,
                estimated_effort="medium",
                suggested_actions=["Perform manual quality assessment", "Consider re-running confidence scoring"]
            )

        # Consider cleaning improvement
        base_action = self.evaluate_content(cleaning_result.confidence_signals)

        # Adjust based on cleaning performance
        if cleaning_result.cleaning_performed:
            if cleaning_result.quality_improvement > 0.1:
                # Good improvement - may upgrade decision
                if base_action.decision == EditorialDecision.GAP_RESEARCH:
                    base_action.decision = EditorialDecision.ENHANCE_CONTENT
                    base_action.reasoning += " (upgraded due to good cleaning improvement)"
            elif cleaning_result.quality_improvement < 0:
                # Negative improvement - may downgrade decision
                if base_action.decision == EditorialDecision.ACCEPT_CONTENT:
                    base_action.decision = EditorialDecision.ENHANCE_CONTENT
                    base_action.reasoning += " (downgraded due to negative cleaning impact)"

        return base_action

    def evaluate_content_batch(
        self,
        confidence_signals_list: List[ConfidenceSignals]
    ) -> List[EditorialAction]:
        """
        Evaluate multiple content pieces and recommend actions.

        Args:
            confidence_signals_list: List of confidence assessments

        Returns:
            List of EditorialAction recommendations
        """
        actions = []

        for i, signals in enumerate(confidence_signals_list):
            try:
                action = self.evaluate_content(signals)
                action.metadata['batch_index'] = i
                actions.append(action)
            except Exception as e:
                logger.error(f"Error evaluating content {i}: {e}")
                # Create fallback action
                fallback_action = EditorialAction(
                    decision=EditorialDecision.MANUAL_REVIEW,
                    priority=PriorityLevel.MEDIUM,
                    reasoning=f"Evaluation error: {str(e)}",
                    confidence=0.0,
                    estimated_effort="medium",
                    metadata={'batch_index': i, 'error': str(e)}
                )
                actions.append(fallback_action)

        return actions

    def get_batch_recommendations(self, actions: List[EditorialAction]) -> Dict[str, Any]:
        """
        Get batch-level recommendations and statistics.

        Args:
            actions: List of EditorialAction recommendations

        Returns:
            Dictionary with batch recommendations
        """
        # Count decisions
        decision_counts = {}
        for action in actions:
            decision = action.decision.value
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        # Calculate priority distribution
        priority_counts = {}
        for action in actions:
            priority = action.priority.name
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        # Generate batch recommendations
        total_items = len(actions)
        recommendations = []

        if decision_counts.get('GAP_RESEARCH', 0) > total_items * 0.5:
            recommendations.append("High proportion of gap research needed - consider expanding search scope")

        if decision_counts.get('REJECT_CONTENT', 0) > total_items * 0.3:
            recommendations.append("High rejection rate - review search strategy and source quality")

        if decision_counts.get('ACCEPT_CONTENT', 0) > total_items * 0.7:
            recommendations.append("Good overall quality - minimal enhancement needed")

        # Priority recommendations
        high_priority_count = priority_counts.get('HIGH', 0) + priority_counts.get('CRITICAL', 0)
        if high_priority_count > total_items * 0.3:
            recommendations.append("Many high-priority items - consider additional processing resources")

        return {
            'total_items': total_items,
            'decision_distribution': decision_counts,
            'priority_distribution': priority_counts,
            'recommendations': recommendations,
            'processing_priority': self._calculate_batch_priority(actions)
        }

    def _evaluate_quality_gates(self, confidence_signals: ConfidenceSignals) -> List[QualityGate]:
        """
        Evaluate quality gates and return failed ones.

        Args:
            confidence_signals: Confidence assessment

        Returns:
            List of failed quality gates
        """
        failed_gates = []

        for gate in self.quality_gates:
            # Get the corresponding score from confidence signals
            score = getattr(confidence_signals, gate.name, 0.0)

            if score < gate.threshold:
                failed_gates.append(gate)
                logger.debug(f"Quality gate failed: {gate.name} (score: {score:.3f}, threshold: {gate.threshold})")

        return failed_gates

    def _create_critical_failure_action(self, signals: ConfidenceSignals) -> EditorialAction:
        """Create action for critically failing content."""
        return EditorialAction(
            decision=EditorialDecision.REJECT_CONTENT,
            priority=PriorityLevel.HIGH,
            reasoning=f"Critical quality failure (confidence: {signals.overall_confidence:.3f})",
            confidence=signals.overall_confidence,
            estimated_effort="high",
            suggested_actions=[
                "Reject this source entirely",
                "Look for alternative, higher-quality sources",
                "Review search query for better targeting"
            ],
            metadata={
                'critical_failure': True,
                'failed_gates': [gate.name for gate in self._evaluate_quality_gates(signals)]
            }
        )

    def _create_accept_action(self, signals: ConfidenceSignals, failed_gates: List[QualityGate]) -> EditorialAction:
        """Create action for acceptable content."""
        suggestions = []
        priority = PriorityLevel.LOW

        if failed_gates:
            suggestions.append("Address minor quality issues in final report")
            priority = PriorityLevel.MEDIUM

        return EditorialAction(
            decision=EditorialDecision.ACCEPT_CONTENT,
            priority=priority,
            reasoning=f"Good quality content (confidence: {signals.overall_confidence:.3f})",
            confidence=signals.overall_confidence,
            estimated_effort="low",
            suggested_actions=suggestions,
            metadata={
                'quality_level': 'good',
                'failed_gates': [gate.name for gate in failed_gates]
            }
        )

    def _create_enhance_action(self, signals: ConfidenceSignals, failed_gates: List[QualityGate]) -> EditorialAction:
        """Create action for content that needs enhancement."""
        suggestions = []

        # Generate specific suggestions based on failed gates
        for gate in failed_gates:
            if gate.name == "cleanliness":
                suggestions.append("Enhance content structure and readability")
            elif gate.name == "content_length":
                suggestions.append("Find additional content for better coverage")
            elif gate.name == "relevance":
                suggestions.append("Improve content relevance through targeted research")
            elif gate.name == "domain_authority":
                suggestions.append("Supplement with higher-authority sources")

        return EditorialAction(
            decision=EditorialDecision.ENHANCE_CONTENT,
            priority=PriorityLevel.MEDIUM,
            reasoning=f"Content acceptable but needs enhancement (confidence: {signals.overall_confidence:.3f})",
            confidence=signals.overall_confidence,
            estimated_effort="medium",
            suggested_actions=suggestions,
            metadata={
                'quality_level': 'acceptable',
                'failed_gates': [gate.name for gate in failed_gates]
            }
        )

    def _create_gap_research_action(self, signals: ConfidenceSignals, failed_gates: List[QualityGate]) -> EditorialAction:
        """Create action for content that needs gap research."""
        suggestions = [
            "Conduct gap research to address information gaps",
            "Look for more comprehensive sources"
        ]

        # Generate specific gap research suggestions
        if any(gate.name == "relevance" for gate in failed_gates):
            suggestions.append("Focus on finding more relevant sources")

        if any(gate.name == "content_length" for gate in failed_gates):
            suggestions.append("Seek sources with better content depth")

        if any(gate.name == "domain_authority" for gate in failed_gates):
            suggestions.append("Prioritize authoritative sources in gap research")

        return EditorialAction(
            decision=EditorialDecision.GAP_RESEARCH,
            priority=PriorityLevel.HIGH,
            reasoning=f"Content below threshold - gap research needed (confidence: {signals.overall_confidence:.3f})",
            confidence=signals.overall_confidence,
            estimated_effort="high",
            suggested_actions=suggestions,
            metadata={
                'quality_level': 'below_threshold',
                'failed_gates': [gate.name for gate in failed_gates],
                'gap_research_priority': 'high'
            }
        )

    def _create_reject_action(self, signals: ConfidenceSignals, failed_gates: List[QualityGate]) -> EditorialAction:
        """Create action for rejected content."""
        return EditorialAction(
            decision=EditorialDecision.REJECT_CONTENT,
            priority=PriorityLevel.HIGH,
            reasoning=f"Content quality too low (confidence: {signals.overall_confidence:.3f})",
            confidence=signals.overall_confidence,
            estimated_effort="high",
            suggested_actions=[
                "Reject this source",
                "Conduct fresh research with different search terms",
                "Focus on higher-quality source domains"
            ],
            metadata={
                'quality_level': 'poor',
                'failed_gates': [gate.name for gate in failed_gates],
                'rejection_reason': 'low_quality'
            }
        )

    def _calculate_batch_priority(self, actions: List[EditorialAction]) -> str:
        """Calculate overall batch processing priority."""
        high_priority_count = sum(1 for a in actions if a.priority in [PriorityLevel.HIGH, PriorityLevel.CRITICAL])
        total_count = len(actions)

        if high_priority_count / total_count > 0.5:
            return "HIGH"
        elif high_priority_count / total_count > 0.2:
            return "MEDIUM"
        else:
            return "LOW"

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update decision thresholds.

        Args:
            new_thresholds: New threshold values
        """
        for key, value in new_thresholds.items():
            if key in self.thresholds:
                if 0.0 <= value <= 1.0:
                    self.thresholds[key] = value
                    logger.info(f"Updated threshold {key}: {value}")
                else:
                    logger.warning(f"Invalid threshold value for {key}: {value}")
            else:
                logger.warning(f"Unknown threshold: {key}")

        # Update quality gates with new thresholds
        self._update_quality_gates()

    def _update_quality_gates(self):
        """Update quality gates with current thresholds."""
        for gate in self.quality_gates:
            if gate.name == "overall_confidence":
                gate.threshold = self.thresholds['acceptable_quality']

    def get_threshold_info(self) -> Dict[str, Any]:
        """
        Get current threshold configuration.

        Returns:
            Dictionary with threshold information
        """
        return {
            'current_thresholds': self.thresholds.copy(),
            'quality_gates': [
                {
                    'name': gate.name,
                    'threshold': gate.threshold,
                    'weight': gate.weight,
                    'description': gate.description
                }
                for gate in self.quality_gates
            ]
        }