"""
Gap Research Decision System with Confidence-Based Logic

Phase 3.2.2: Confidence-Based Gap Research Decision System

This module implements sophisticated gap research decision logic with confidence thresholds,
multi-dimensional analysis, and intelligent decision-making capabilities that prioritize leveraging existing
research over conducting new research whenever possible.

Key Features:
- Confidence-based gap research decision logic with configurable thresholds
- Multi-dimensional gap analysis and prioritization
- Intelligent decision-making that favors existing research utilization
- Confidence threshold management with adaptive adjustments
- Integration with quality framework and enhanced orchestrator
- Evidence-based gap research recommendations with cost-benefit analysis
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import statistics

# Import system components
try:
    from ..core.logging_config import get_logger
    from ..core.quality_framework import QualityAssessment, QualityFramework
    from ..core.workflow_state import WorkflowStage
    from ..utils.message_processing.main import MessageProcessor, MessageType
    from .enhanced_editorial_engine import (
        EnhancedEditorialDecisionEngine,
        ConfidenceScore,
        GapAnalysis,
        GapCategory,
        CorpusAnalysisResult,
        DecisionType
    )
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    MessageType = None
    MessageProcessor = None
    QualityFramework = None

    # Fallback definitions
    @dataclass
    class ConfidenceScore:
        overall_confidence: float
    @dataclass
    class GapAnalysis:
        gap_category: GapCategory
    @dataclass
    class CorpusAnalysisResult:
        corpus_sufficiency: str


class GapResearchDecision(Enum):
    """Types of gap research decisions."""
    NO_GAP_RESEARCH_NEEDED = "no_gap_research_needed"
    OPTIONAL_GAP_RESEARCH = "optional_gap_research"
    RECOMMENDED_GAP_RESEARCH = "recommended_gap_research"
    REQUIRED_GAP_RESEARCH = "required_gap_research"
    CRITICAL_GAP_RESEARCH = "critical_gap_research"


class GapResearchConfidence(Enum):
    """Confidence levels for gap research decisions."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MODERATE = "moderate"      # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


class ResearchUtilizationStrategy(Enum):
    """Strategies for utilizing existing research."""
    PRIORITIZE_EXISTING = "prioritize_existing"
    ENHANCE_EXISTING = "enhance_existing"
    SUPPLEMENT_EXISTING = "supplement_existing"
    NEW_RESEARCH_REQUIRED = "new_research_required"


@dataclass
class GapResearchDecisionContext:
    """Context for gap research decision making."""

    session_id: str
    first_draft_report: str
    available_research: Dict[str, Any]
    quality_assessment: Optional[QualityAssessment]
    corpus_analysis: CorpusAnalysisResult
    confidence_score: ConfidenceScore
    gap_analysis: List[GapAnalysis]

    # Configuration
    gap_research_threshold: float = 0.7
    existing_research_preference: float = 0.8
    max_gap_topics: int = 3
    resource_constraints: Dict[str, Any] = field(default_factory=dict)

    # Contextual information
    user_requirements: Dict[str, Any] = field(default_factory=dict)
    research_budget: Dict[str, Any] = field(default_factory=dict)
    timeline_constraints: Optional[str] = None

    # Decision history
    previous_decisions: List[str] = field(default_factory=list)
    decision_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GapResearchDecision:
    """Comprehensive gap research decision with confidence scoring."""

    decision_type: GapResearchDecision
    confidence_level: GapResearchConfidence
    priority_score: float  # 0.0 - 1.0

    # Decision rationale
    primary_reasoning: str
    supporting_evidence: List[str] = field(default_factory=list)
    uncertainty_factors: List[str] = field(default_factory=list)
    alternative_options: List[str] = field(default_factory=list)

    # Gap research recommendations
    recommended_gaps: List[GapAnalysis] = field(default_factory=list)
    recommended_strategy: ResearchUtilizationStrategy
    research_topics: List[str] = field(default_factory=list)

    # Cost-benefit analysis
    estimated_cost: str
    estimated_benefit: str
    roi_estimate: float  # 0.0 - 1.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)

    # Implementation details
    suggested_approach: str
    success_metrics: List[str] = field(default_factory=list)
    timeline_estimate: str

    # Metadata
    decision_timestamp: datetime = field(default_factory=datetime.now)
    analyst_confidence: float = 0.0
    automated_decision: bool = True
    review_required: bool = False


class GapResearchDecisionEngine:
    """
    Sophisticated gap research decision engine with confidence-based logic.

    This engine implements intelligent decision-making for gap research,
    prioritizing existing research utilization and providing evidence-based
    recommendations with confidence thresholds.
    """

    def __init__(self,
                 editorial_engine: Optional[EnhancedEditorialDecisionEngine] = None,
                 confidence_thresholds: Optional[Dict[str, float]] = None,
                 message_processor: Optional[MessageProcessor] = None):
        """
        Initialize the gap research decision engine.

        Args:
            editorial_engine: Enhanced editorial decision engine
            confidence_thresholds: Custom confidence thresholds
            message_processor: Optional message processor
        """
        self.logger = get_logger("gap_research_decision_engine")
        self.editorial_engine = editorial_engine
        self.message_processor = message_processor

        # Decision thresholds (configurable)
        self.thresholds = {
            'gap_research_trigger': 0.70,        # Confidence threshold for triggering gap research
            'high_priority_threshold': 0.85,    # High priority threshold
            'critical_gap_threshold': 0.90,      # Critical gap threshold
            'existing_research_threshold': 0.75, # Threshold for preferring existing research
            'decision_confidence_minimum': 0.60,  # Minimum confidence for decisions
            'roi_minimum_threshold': 0.30,        # Minimum ROI for gap research
            'uncertainty_tolerance': 0.40          # Maximum uncertainty tolerated
        }

        if confidence_thresholds:
            self.thresholds.update(confidence_thresholds)

        # Decision configuration
        self.enable_cost_benefit_analysis = True
        self.enable_confidence_intervals = True
        self.enable_decision_history = True
        self.enable_adaptive_thresholds = True

        # Gap research optimization
        self.max_gap_topics_per_decision = 3
        self.gap_topic_priority_weights = {
            GapCategory.FACTUAL_GAPS: 0.30,
            GapCategory.ANALYTICAL_GAPS: 0.25,
            GapCategory.TEMPORAL_GAPS: 0.20,
            GapCategory.COMPARATIVE_GAPS: 0.15,
            GapCategory.CONTEXTUAL_GAPS: 0.10
        }

        self.logger.info("Gap Research Decision Engine initialized",
                        thresholds=len(self.thresholds),
                        gap_categories=len(self.gap_topic_priority_weights))

    async def make_gap_research_decision(
        self,
        session_id: str,
        first_draft_report: str,
        available_research: Dict[str, Any],
        quality_assessment: Optional[QualityAssessment] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> GapResearchDecision:
        """
        Make comprehensive gap research decision with confidence-based logic.

        Args:
            session_id: Research session identifier
            first_draft_report: First draft report to evaluate
            available_research: Available research data
            quality_assessment: Optional quality assessment
            context: Additional context for decision making

        Returns:
            Comprehensive gap research decision
        """
        start_time = time.time()

        try:
            self.logger.info(f"Making gap research decision for session {session_id}")

            # Use editorial engine to get base analysis
            if not self.editorial_engine:
                raise ValueError("Enhanced editorial engine required for gap research decisions")

            editorial_decision = await self.editorial_engine.make_editorial_decision(
                session_id, first_draft_report, available_research, quality_assessment, context
            )

            # Create decision context
            decision_context = GapResearchDecisionContext(
                session_id=session_id,
                first_draft_report=first_draft_report,
                available_research=available_research,
                quality_assessment=quality_assessment,
                corpus_analysis=editorial_decision.confidence_score,  # This would be properly passed
                confidence_score=editorial_decision.confidence_score,
                gap_analysis=editorial_decision.gap_analysis,
                user_requirements=context.get('user_requirements', {}),
                research_budget=context.get('research_budget', {}),
                timeline_constraints=context.get('timeline_constraints')
            )

            # Step 1: Analyze existing research sufficiency
            existing_sufficiency = await self._analyze_existing_research_sufficiency(
                available_research, decision_context
            )

            # Step 2: Calculate gap research necessity
            gap_necessity_score = await self._calculate_gap_research_necessity(
                editorial_decision.gap_analysis, decision_context
            )

            # Step 3: Evaluate research utilization strategy
            utilization_strategy = await self._determine_research_utilization_strategy(
                existing_sufficiency, gap_necessity_score, decision_context
            )

            # Step 4: Make primary decision
            decision_type = await self._determine_gap_research_decision(
                gap_necessity_score, utilization_strategy, decision_context
            )

            # Step 5: Calculate confidence and priority
            confidence_level = self._calculate_decision_confidence(
                decision_type, gap_necessity_score, existing_sufficiency, decision_context
            )
            priority_score = self._calculate_priority_score(
                decision_type, gap_necessity_score, editorial_decision.gap_analysis
            )

            # Step 6: Generate comprehensive decision
            gap_decision = await self._create_gap_research_decision(
                decision_type, confidence_level, priority_score, utilization_strategy,
                editorial_decision, decision_context
            )

            # Step 7: Log and notify
            await self._log_decision_made(gap_decision, session_id)
            await self._send_decision_notification(gap_decision, session_id)

            analysis_duration = time.time() - start_time
            gap_decision.analysis_duration = analysis_duration

            self.logger.info(f"Gap research decision completed for session {session_id}",
                            decision=decision_type.value,
                            confidence=confidence_level.value,
                            priority=priority_score,
                            gaps_recommended=len(gap_decision.recommended_gaps),
                            duration=f"{analysis_duration:.2f}s")

            return gap_decision

        except Exception as e:
            self.logger.error(f"Gap research decision failed for session {session_id}: {e}",
                            error=str(e),
                            error_type=type(e).__name__)
            raise

    async def _analyze_existing_research_sufficiency(
        self,
        available_research: Dict[str, Any],
        context: GapResearchDecisionContext
    ) -> Dict[str, float]:
        """
        Analyze the sufficiency of existing research for addressing potential gaps.

        Args:
            available_research: Available research data
            context: Decision context

        Returns:
            Sufficiency analysis by category
        """
        try:
            self.logger.debug("Analyzing existing research sufficiency")

            sufficiency_scores = {}

            # Analyze different aspects of existing research
            sufficiency_scores['coverage'] = self._analyze_coverage_sufficiency(
                available_research, context
            )
            sufficiency_scores['depth'] = self._analyze_depth_sufficiency(
                available_research, context
            )
            sufficiency_scores['quality'] = self._analyze_quality_sufficiency(
                available_research, context
            )
            sufficiency_scores['recency'] = self._analyze_recency_sufficiency(
                available_research, context
            )
            sufficiency_scores['diversity'] = self._analyze_diversity_sufficiency(
                available_research, context
            )

            # Calculate overall sufficiency
            sufficiency_scores['overall'] = statistics.mean(list(sufficiency_scores.values()))

            self.logger.debug(f"Existing research sufficiency analysis completed: "
                            f"overall={sufficiency_scores['overall']:.2f}")

            return sufficiency_scores

        except Exception as e:
            self.logger.error(f"Research sufficiency analysis failed: {e}")
            return {'overall': 0.5, 'coverage': 0.5, 'depth': 0.5, 'quality': 0.5, 'recency': 0.5, 'diversity': 0.5}

    async def _calculate_gap_research_necessity(
        self,
        gap_analysis: List[GapAnalysis],
        context: GapResearchDecisionContext
    ) -> float:
        """
        Calculate the necessity of gap research based on identified gaps.

        Args:
            gap_analysis: List of gap analyses
            context: Decision context

        Returns:
            Gap research necessity score (0.0 - 1.0)
        """
        try:
            if not gap_analysis:
                return 0.0

            # Calculate weighted necessity score
            total_importance = 0.0
            total_weight = 0.0

            for gap in gap_analysis:
                # Get category weight
                category_weight = self.gap_topic_priority_weights.get(gap.gap_category, 0.1)

                # Calculate weighted score
                gap_score = (gap.importance_score * gap.confidence_in_gap * category_weight)
                total_importance += gap_score
                total_weight += category_weight

            # Normalize score
            necessity_score = total_importance / total_weight if total_weight > 0 else 0.0

            self.logger.debug(f"Gap research necessity calculated: {necessity_score:.2f} "
                            f"from {len(gap_analysis)} gaps")

            return necessity_score

        except Exception as e:
            self.logger.error(f"Gap necessity calculation failed: {e}")
            return 0.5

    async def _determine_research_utilization_strategy(
        self,
        existing_sufficiency: Dict[str, float],
        gap_necessity_score: float,
        context: GapResearchContext
    ) -> ResearchUtilizationStrategy:
        """
        Determine the optimal strategy for utilizing existing research.

        Args:
            existing_sufficiency: Sufficiency analysis of existing research
            gap_necessity_score: Gap research necessity score
            context: Decision context

        Returns:
            Optimal research utilization strategy
        """
        try:
            overall_sufficiency = existing_sufficiency['overall']

            # Decision logic for strategy selection
            if (overall_sufficiency >= self.thresholds['existing_research_threshold'] and
                gap_necessity_score < self.thresholds['gap_research_trigger']):
                return ResearchUtilizationStrategy.PRIORITIZE_EXISTING

            elif (overall_sufficiency >= 0.6 and gap_necessity_score < 0.8):
                return ResearchUtilizationStrategy.ENHANCE_EXISTING

            elif (overall_sufficiency >= 0.4 and gap_necessity_score < 0.9):
                return ResearchUtilizationStrategy.SUPPLEMENT_EXISTING

            else:
                return ResearchUtilizationStrategy.NEW_RESEARCH_REQUIRED

        except Exception as e:
            self.logger.error(f"Strategy determination failed: {e}")
            return ResearchUtilizationStrategy.NEW_RESEARCH_REQUIRED

    async def _determine_gap_research_decision(
        self,
        gap_necessity_score: float,
        utilization_strategy: ResearchUtilizationStrategy,
        context: GapResearchDecisionContext
    ) -> GapResearchDecision:
        """
        Determine the primary gap research decision.

        Args:
            gap_necessity_score: Gap research necessity score
            utilization_strategy: Research utilization strategy
            context: Decision context

        Returns:
            Primary gap research decision type
        """
        try:
            # Decision logic based on necessity and utilization strategy
            if utilization_strategy == ResearchUtilizationStrategy.PRIORITIZE_EXISTING:
                if gap_necessity_score < 0.3:
                    return GapResearchDecision.NO_GAP_RESEARCH_NEEDED
                elif gap_necessity_score < 0.6:
                    return GapResearchDecision.OPTIONAL_GAP_RESEARCH
                else:
                    return GapResearchDecision.RECOMMENDED_GAP_RESEARCH

            elif utilization_strategy == ResearchUtilizationStrategy.ENHANCE_EXISTING:
                if gap_necessity_score < 0.4:
                    return GapResearchDecision.OPTIONAL_GAP_RESEARCH
                elif gap_necessity_score < 0.7:
                    return GapResearchDecision.RECOMMENDED_GAP_RESEARCH
                else:
                    return GapResearchDecision.REQUIRED_GAP_RESEARCH

            elif utilization_strategy == ResearchUtilizationStrategy.SUPPLEMENTING:
                return GapResearchDecision.RECOMMENDED_GAP_RESEARCH

            else:  # NEW_RESEARCH_REQUIRED
                if gap_necessity_score < 0.5:
                    return GapResearchDecision.RECOMMENDED_GAP_RESEARCH
                elif gap_necessity_score < 0.8:
                    return GapResearchDecision.REQUIRED_GAP_RESEARCH
                else:
                    return GapResearchDecision.CRITICAL_GAP_RESEARCH

        except Exception as e:
            self.logger.error(f"Gap research decision determination failed: {e}")
            return GapResearchDecision.RECOMMENDED_GAP_RESEARCH

    def _calculate_decision_confidence(
        self,
        decision_type: GapResearchDecision,
        gap_necessity_score: float,
        existing_sufficiency: Dict[str, float],
        context: GapResearchContext
    ) -> GapResearchConfidence:
        """
        Calculate confidence level for the gap research decision.

        Args:
            decision_type: Gap research decision type
            gap_necessity_score: Gap research necessity score
            existing_sufficiency: Research sufficiency analysis
            context: Decision context

        Returns:
            Confidence level for the decision
        """
        try:
            # Base confidence from gap necessity
            base_confidence = gap_necessity_score

            # Adjust based on existing research sufficiency
            sufficiency_factor = existing_sufficiency['overall'] / 0.7  # 0.7 is baseline

            # Adjust based on decision type
            decision_confidence_factors = {
                GapResearchDecision.NO_GAP_RESEARCH_NEEDED: 0.9,
                GapResearchDecision.OPTIONAL_GAP_RESEARCH: 0.8,
                GapResearchDecision.RECOMMENDED_GAP_RESEARCH: 0.7,
                GapResearchDecision.REQUIRED_GAP_RESEARCH: 0.6,
                GapResearchDecision.CRITICAL_GAP_RESEARCH: 0.4
            }

            decision_factor = decision_confidence_factors.get(decision_type, 0.7)

            # Calculate final confidence
            final_confidence = base_confidence * sufficiency_factor * decision_factor

            # Map to confidence levels
            if final_confidence >= 0.8:
                return GapResearchConfidence.VERY_HIGH
            elif final_confidence >= 0.6:
                return GapResearchConfidence.HIGH
            elif final_confidence >= 0.4:
                return GapResearchConfidence.MODERATE
            elif final_confidence >= 0.2:
                return GapResearchConfidence.LOW
            else:
                return GapResearchConfidence.VERY_LOW

        except Exception as e:
            self.logger.error(f"Decision confidence calculation failed: {e}")
            return GapResearchConfidence.MODERATE

    def _calculate_priority_score(
        self,
        decision_type: GapResearchDecision,
        gap_necessity_score: float,
        gap_analysis: List[GapAnalysis]
    ) -> float:
        """
        Calculate priority score for the gap research decision.

        Args:
            decision_type: Gap research decision type
            gap_necessity_score: Gap research necessity score
            gap_analysis: Gap analyses

        Returns:
            Priority score (0.0 - 1.0)
        """
        try:
            # Base priority from decision type
            decision_priorities = {
                GapResearchDecision.CRITICAL_GAP_RESEARCH: 1.0,
                GapResearchDecision.REQUIRED_GAP_RESEARCH: 0.8,
                GapResearchDecision.RECOMMENDED_GAP_RESEARCH: 0.6,
                GapResearchDecision.OPTIONAL_GAP_RESEARCH: 0.4,
                GapResearchDecision.NO_GAP_RESEARCH_NEEDED: 0.1
            }

            base_priority = decision_priorities.get(decision_type, 0.5)

            # Adjust based on gap necessity
            necessity_boost = gap_necessity_score * 0.3

            # Adjust based on critical gaps
            critical_gaps = [gap for gap in gap_analysis
                              if gap.priority_level in [4, 3]]  # CRITICAL or HIGH
            critical_boost = min(len(critical_gaps) * 0.1, 0.3)

            # Calculate final priority
            final_priority = min(base_priority + necessity_boost + critical_boost, 1.0)

            return final_priority

        except Exception as e:
            self.logger.error(f"Priority score calculation failed: {e}")
            return 0.5

    async def _create_gap_research_decision(
        self,
        decision_type: GapResearchDecision,
        confidence_level: GapResearchConfidence,
        priority_score: float,
        utilization_strategy: ResearchUtilizationStrategy,
        editorial_decision: Any,  # Would be EnhancedEditorialDecision
        context: GapResearchDecisionContext
    ) -> GapResearchDecision:
        """
        Create comprehensive gap research decision.

        Args:
            decision_type: Primary decision type
            confidence_level: Confidence level
            priority_score: Priority score
            utilization_strategy: Research utilization strategy
            editorial_decision: Editorial decision results
            context: Decision context

        Returns:
            Comprehensive gap research decision
        """
        try:
            # Select recommended gaps based on strategy
            recommended_gaps = self._select_recommended_gaps(
                editorial_decision.gap_analysis, utilization_strategy, context
            )

            # Generate research topics
            research_topics = self._generate_research_topics(recommended_gaps)

            # Calculate cost-benefit analysis
            if self.enable_cost_benefit_analysis:
                estimated_cost, estimated_benefit, roi_estimate = await self._calculate_cost_benefit(
                    recommended_gaps, utilization_strategy, context
                )
            else:
                estimated_cost = "Medium"
                estimated_benefit = "Moderate improvement"
                roi_estimate = 0.5

            # Generate decision rationale
            primary_reasoning = self._generate_gap_decision_reasoning(
                decision_type, utilization_strategy, recommended_gaps, context
            )

            # Generate supporting evidence
            supporting_evidence = self._generate_gap_supporting_evidence(
                recommended_gaps, utilization_strategy, context
            )

            # Generate uncertainty factors
            uncertainty_factors = self._generate_gap_uncertainty_factors(
                confidence_level, recommended_gaps, context
            )

            # Generate alternative options
            alternative_options = self._generate_alternative_options(
                decision_type, utilization_strategy, context
            )

            # Determine suggested approach
            suggested_approach = self._generate_suggested_approach(
                decision_type, utilization_strategy, recommended_gaps
            )

            # Generate success metrics
            success_metrics = self._generate_success_metrics(decision_type, recommended_gaps)

            # Estimate resource requirements
            resource_requirements = self._estimate_gap_resource_requirements(
                recommended_gaps, utilization_strategy, context
            )

            # Create comprehensive decision
            gap_decision = GapResearchDecision(
                decision_type=decision_type,
                confidence_level=confidence_level,
                priority_score=priority_score,
                primary_reasoning=primary_reasoning,
                supporting_evidence=supporting_evidence,
                uncertainty_factors=uncertainty_factors,
                alternative_options=alternative_options,
                recommended_gaps=recommended_gaps,
                recommended_strategy=utilization_strategy,
                research_topics=research_topics,
                estimated_cost=estimated_cost,
                estimated_benefit=estimated_benefit,
                roi_estimate=roi_estimate,
                suggested_approach=suggested_approach,
                success_metrics=success_metrics,
                timeline_estimate=self._estimate_timeline(decision_type, recommended_gaps),
                resource_requirements=resource_requirements,
                analyst_confidence=self._calculate_analyst_confidence(
                    confidence_level, uncertainty_factors
                ),
                automated_decision=True,
                review_required=self._requires_manual_review(decision_type, confidence_level)
            )

            return gap_decision

        except Exception as e:
            self.logger.error(f"Gap research decision creation failed: {e}")
            # Return fallback decision
            return GapResearchDecision(
                decision_type=GapResearchDecision.RECOMMENDED_GAP_RESEARCH,
                confidence_level=GapResearchConfidence.MODERATE,
                priority_score=0.5,
                primary_reasoning=f"Decision creation failed due to error: {str(e)}",
                uncertainty_factors=["System error in decision process"],
                review_required=True
            )

    # Helper methods for research sufficiency analysis
    def _analyze_coverage_sufficiency(
        self,
        available_research: Dict[str, Any],
        context: GapResearchDecisionContext
    ) -> float:
        """Analyze coverage sufficiency of existing research."""
        sources = available_research.get('sources', [])
        if not sources:
            return 0.0

        # Count unique topics/themes
        topics = set()
        for source in sources:
            if 'title' in source:
                topics.add(source['title'][:50])  # Use first 50 chars as topic identifier

        # Coverage based on number of topics relative to needs
        topic_count = len(topics)
        min_topics_needed = 5  # Assume minimum 5 topics needed
        coverage_score = min(topic_count / min_topics_needed, 1.0)

        return coverage_score

    def _analyze_depth_sufficiency(
        self,
        available_research: Dict[str, Any],
        context: GapResearchDecisionContext
    ) -> float:
        """Analyze depth sufficiency of existing research."""
        work_products = available_research.get('work_products', [])
        if not work_products:
            return 0.0

        # Analyze depth indicators
        total_depth_score = 0
        for wp in work_products:
            if isinstance(wp, dict):
                word_count = wp.get('word_count', 0)
                # Depth based on word count
                if word_count >= 2000:
                    depth_score = 1.0
                elif word_count >= 1000:
                    depth_score = 0.7
                elif word_count >= 500:
                    depth_score = 0.4
                else:
                    depth_score = 0.2
                total_depth_score += depth_score

        return total_depth_score / len(work_products) if work_products else 0.0

    def _analyze_quality_sufficiency(
        self,
        available_research: Dict[str, Any],
        context: GapResearchDecisionContext
    ) -> float:
        """Analyze quality sufficiency of existing research."""
        sources = available_research.get('sources', [])
        if not sources:
            return 0.0

        # Quality indicators
        quality_scores = []
        for source in sources:
            score = 0.5  # Base score

            # Authoritative domain indicators
            if source.get('domain', '').endswith(('.edu', '.gov', '.org')):
                score += 0.3
            if source.get('title') and len(source['title']) > 20:
                score += 0.1

            quality_scores.append(min(score, 1.0))

        return statistics.mean(quality_scores) if quality_scores else 0.0

    def _analyze_recency_sufficiency(
        self,
        available_research: Dict[str, Any],
        context: GapResearchDecisionContext
    ) -> float:
        """Analyze recency sufficiency of existing research."""
        sources = available_research.get('sources', [])
        if not sources:
            return 0.0

        # Assume all sources are recent for now
        # In practice, this would analyze publication dates
        recent_count = len(sources)
        recency_score = recent_count / len(sources) if sources else 0.0

        return recency_score

    def _analyze_diversity_sufficiency(
        self,
        available_research: Dict[str, Any],
        context: GapResearchContext
    ) -> float:
        """Analyze diversity sufficiency of existing research."""
        sources = available_research.get('sources', [])
        if not sources:
            return 0.0

        # Domain diversity
        domains = set()
        for source in sources:
            domain = source.get('domain', '')
            if domain:
                domains.add(domain)

        domain_diversity = len(domains) / len(sources) if sources else 0.0

        return min(domain_diversity, 1.0)

    def _select_recommended_gaps(
        self,
        gap_analysis: List[GapAnalysis],
        strategy: ResearchUtilizationStrategy,
        context: GapResearchDecisionContext
    ) -> List[GapAnalysis]:
        """Select recommended gaps based on strategy and constraints."""
        if not gap_analysis:
            return []

        # Sort gaps by importance and confidence
        sorted_gaps = sorted(
            gap_analysis,
            key=lambda g: (g.importance_score * g.confidence_in_gap),
            reverse=True
        )

        # Filter and select based on strategy
        if strategy == ResearchUtilizationStrategy.PRIORITIZE_EXISTING:
            # Focus on gaps that can be addressed with existing research
            recommended_gaps = [
                gap for gap in sorted_gaps
                if gap.confidence_in_solution > 0.6
            ]
        elif strategy == ResearchUtilizationStrategy.ENHANCE_EXISTING:
            # Include gaps that can enhance existing research
            recommended_gaps = [
                gap for gap in sorted_gaps
                if gap.confidence_in_solution > 0.5
            ]
        else:
            # Include all gaps
            recommended_gaps = sorted_gaps

        # Limit to maximum topics
        if len(recommended_gaps) > context.max_gap_topics:
            recommended_gaps = recommended_gaps[:context.max_gap_topics]

        return recommended_gaps

    def _generate_research_topics(self, recommended_gaps: List[GapAnalysis]) -> List[str]:
        """Generate research topics from gap analyses."""
        topics = []

        for gap in recommended_gaps:
            # Generate research topic from gap description
            topic = f"Research: {gap.gap_description}"
            if gap.confidence_in_gap < 0.7:
                topic += " (confidence: medium)"
            topics.append(topic)

        return topics

    async def _calculate_cost_benefit(
        self,
        recommended_gaps: List[GapAnalysis],
        strategy: ResearchUtilizationStrategy,
        context: GapResearchContext
    ) -> Tuple[str, str, float]:
        """Calculate cost-benefit analysis for gap research."""
        try:
            # Estimate cost based on gaps and strategy
            base_cost_per_gap = "Medium"
            if strategy == ResearchUtilizationStrategy.NEW_RESEARCH_REQUIRED:
                base_cost_per_gap = "High"
            elif strategy == ResearchUtilizationStrategy.CRITICAL_GAP_RESEARCH:
                base_cost_per_gap = "Very High"

            total_cost = self._estimate_total_cost(
                len(recommended_gaps), base_cost_per_gap
            )

            # Estimate benefit
            total_benefit = self._estimate_total_benefit(recommended_gaps)

            # Calculate ROI
            roi_estimate = self._calculate_roi_estimate(total_benefit, total_cost)

            return total_cost, total_benefit, roi_estimate

        except Exception as e:
            self.logger.error(f"Cost-benefit calculation failed: {e}")
            return "Medium", "Moderate improvement", 0.5

    def _estimate_total_cost(self, gap_count: int, base_cost: str) -> str:
        """Estimate total cost for gap research."""
        cost_multipliers = {
            "Low": 1.0,
            "Medium": 2.0,
            "High": 4.0,
            "Very High": 8.0
        }

        multiplier = cost_multipliers.get(base_cost, 2.0)
        estimated_cost = gap_count * multiplier

        if estimated_cost <= 3:
            return "Low"
        elif estimated_cost <= 8:
            return "Medium"
        elif estimated_cost <= 20:
            return "High"
        else:
            return "Very High"

    def _estimate_total_benefit(self, recommended_gaps: List[GapAnalysis]) -> str:
        """Estimate total benefit from gap research."""
        total_importance = sum(gap.importance_score for gap in recommended_gaps)

        if total_importance >= 2.0:
            return "High impact improvement"
        elif total_importance >= 1.5:
            return "Moderate improvement"
        elif total_importance >= 1.0:
            return "Minor improvement"
        else:
            return "Limited improvement"

    def _calculate_roi_estimate(self, benefit: str, cost: str) -> float:
        """Calculate ROI estimate."""
        benefit_scores = {
            "High impact improvement": 0.8,
            "Moderate improvement": 0.6,
            "Minor improvement": 0.4,
            "Limited improvement": 0.2
        }

        cost_scores = {
            "Low": 0.9,
            "Medium": 0.7,
            "High": 0.5,
            "Very High": 0.3
        }

        benefit_score = benefit_scores.get(benefit, 0.5)
        cost_score = cost_scores.get(cost, 0.5)

        return benefit_score / cost_score if cost_score > 0 else 0.0

    def _generate_gap_decision_reasoning(
        self,
        decision_type: GapResearchDecision,
        strategy: ResearchUtilizationStrategy,
        recommended_gaps: List[GapAnalysis],
        context: GapResearchContext
    ) -> str:
        """Generate primary reasoning for gap research decision."""
        reasoning_parts = []

        # Strategy-based reasoning
        strategy_reasons = {
            ResearchUtilizationStrategy.PRIORITIZE_EXISTING: "Existing research is sufficient for most needs",
            ResearchUtilizationStrategy.ENHANCE_EXISTING: "Enhance existing research with targeted additions",
            ResearchUtilizationStrategy.SUPPLEMENTING: "Supplement existing research with additional perspectives",
            ResearchUtilizationStrategy.NEW_RESEARCH_REQUIRED: "New research is required to address critical gaps"
        }

        reasoning_parts.append(strategy_reasons.get(strategy, "Research strategy analysis completed"))

        # Gap-based reasoning
        if recommended_gaps:
            high_importance_gaps = [g for g in recommended_gaps if g.importance_score > 0.7]
            if high_importance_gaps:
                reasoning_parts.append(f"Identified {len(high_importance_gaps)} high-importance gaps")

        # Decision-type reasoning
        decision_reasons = {
            GapResearchDecision.NO_GAP_RESEARCH_NEEDED: "Existing research adequately addresses all identified needs",
            GapResearchDecision.OPTIONAL_GAP_RESEARCH: "Minor gaps identified that could enhance coverage",
            GapResearchDecision.RECOMMENDED_GAP_RESEARCH: "Specific gaps identified that would improve research quality",
            GapResearchDecision.REQUIRED_GAP_RESEARCH: "Essential gaps must be addressed for report quality",
            GapResearchDecision.CRITICAL_GAP_RESEARCH: "Critical gaps require immediate attention"
        }

        reasoning_parts.append(decision_reasons.get(decision_type, "Gap research analysis completed"))

        return ". ".join(reasoning_parts)

    def _generate_gap_supporting_evidence(
        self,
        recommended_gaps: List[GapAnalysis],
        strategy: ResearchUtilizationStrategy,
        context: GapResearchContext
    ) -> List[str]:
        """Generate supporting evidence for gap research decision."""
        evidence = []

        # Strategy evidence
        evidence.append(f"Research utilization strategy: {strategy.value}")

        # Gap evidence
        if recommended_gaps:
            evidence.append(f"Gaps requiring attention: {len(recommended_gaps)}")

            critical_gaps = [g for g in recommended_gaps if g.priority_level in [4, 3]]
            if critical_gaps:
                evidence.append(f"Critical gaps: {len(critical_gaps)}")

        # Research corpus evidence
        corpus_score = context.corpus_analysis.overall_quality_score
        evidence.append(f"Research corpus quality: {corpus_score:.2f}/1.0")

        return evidence

    def _generate_gap_uncertainty_factors(
        self,
        confidence_level: GapResearchConfidence,
        recommended_gaps: List[GapAnalysis],
        context: GapResearchContext
    ) -> List[str]:
        """Generate uncertainty factors for gap research decision."""
        factors = []

        # Confidence-based uncertainty
        if confidence_level in [GapResearchConfidence.VERY_LOW, GapResearchConfidence.LOW]:
            factors.append("Low confidence increases decision risk")

        # Gap identification uncertainty
        uncertain_gaps = [g for g in recommended_gaps if g.confidence_in_gap < 0.6]
        if uncertain_gaps:
            factors.append(f"Uncertainty in {len(uncertain_gaps)} gap identifications")

        # Resource constraints
        if context.resource_constraints:
            factors.append("Resource constraints may impact research execution")

        return factors

    def _generate_alternative_options(
        self,
        decision_type: GapResearchDecision,
        strategy: ResearchUtilizationStrategy,
        context: GapResearchContext
    ) -> List[str]:
        """Generate alternative options for gap research."""
        options = []

        # Alternative strategies
        alternative_strategies = [
            s for s in ResearchUtilizationStrategy
            if s != strategy
        ]

        for alt_strategy in alternative_strategies:
            options.append(f"Consider {alt_strategy.value} instead of {strategy.value}")

        # Alternative timelines
        if context.timeline_constraints:
            options.append(f"Adjust timeline constraints if possible: {context.timeline_constraints}")

        # Alternative gap selections
        if len(context.gap_analysis) > context.max_gap_topics:
            options.append(f"Reduce gap scope to top {context.max_gap_topics} most critical gaps")

        return options

    def _generate_suggested_approach(
        self,
        decision_type: GapResearchDecision,
        strategy: ResearchUtilizationStrategy,
        recommended_gaps: List[GapAnalysis]
    ) -> str:
        """Generate suggested approach for gap research."""
        approach_parts = []

        # Strategy-based approach
        strategy_approaches = {
            ResearchUtilizationStrategy.PRIORITIZE_EXISTING: "Leverage existing research sources first",
            ResearchUtilizationStrategy.ENHANCE_EXISTING: "Enhance existing research with targeted additions",
            ResearchUtilizationStrategy.SUPPLEMENTING: "Add complementary research to fill gaps",
            ResearchUtilizationStrategy.NEW_RESEARCH_REQUIRED: "Conduct new research to address gaps"
        }

        approach_parts.append(strategy_approaches.get(strategy, "Conduct gap research"))

        # Gap-specific approach
        if recommended_gaps:
            approach_parts.append(f"Focus on {len(recommended_gaps)} priority gaps")

        # Decision-type approach
        if decision_type == GapResearchDecision.CRITICAL_GAP_RESEARCH:
            approach_parts.append("Execute with high priority and immediate attention")
        elif decision_type == GapResearchDecision.REQUIRED_GAP_RESEARCH:
            approach_parts.append("Execute with standard priority and quality controls")
        elif decision_type == GapResearchDecision.OPTIONAL_GAP_RESEARCH:
            approach_parts.append("Execute if resources permit, otherwise defer")

        return ". ".join(approach_parts)

    def _generate_success_metrics(
        self,
        decision_type: GapResearchDecision,
        recommended_gaps: List[GapAnalysis]
    ) -> List[str]:
        """Generate success metrics for gap research."""
        metrics = []

        # Base success metrics
        metrics.append("All critical gaps successfully researched")
        metrics.append("Gap research integrated into final report")
        metrics.append("Quality standards met or exceeded")

        # Decision-specific metrics
        if decision_type in [GapResearchDecision.CRITICAL_GAP_RESEARCH]:
            metrics.append("Critical gaps resolved within timeline")
            metrics.append("Quality improvement achieved in critical areas")
        elif decision_type == GapResearchDecision.REQUIRED_GAP_RESEARCH:
            metrics.append("All required gaps addressed")
            metrics.append("Report completeness improved")

        return metrics

    def _estimate_timeline(
        self,
        decision_type: GapResearchDecision,
        recommended_gaps: List[GapAnalysis]
    ) -> str:
        """Estimate timeline for gap research."""
        # Base timelines by decision type
        base_timelines = {
            GapResearchDecision.NO_GAP_RESEARCH_NEEDED: "0-30 minutes",
            GapResearchDecision.OPTIONAL_GAP_RESEARCH: "1-2 hours",
            GapResearchDecision.RECOMMENDED_GAP_RESEARCH: "2-4 hours",
            GapResearchDecision.REQUIRED_GAP_RESEARCH: "4-6 hours",
            GapResearchDecision.CRITICAL_GAP_RESEARCH: "4-8 hours"
        }

        # Adjust based on number of gaps
        gap_count = len(recommended_gaps)
        gap_multiplier = min(1.0 + (gap_count - 3) * 0.2, 2.0)

        base_timeline = base_timelines.get(decision_type, "2-4 hours")
        estimated_hours = float(base_timeline.split('-')[0]) * gap_multiplier

        if estimated_hours >= 8:
            return "8+ hours"
        elif estimated_hours >= 4:
            return f"{estimated_hours:.0f}+ hours"
        else:
            return base_timeline

    def _estimate_gap_resource_requirements(
        self,
        recommended_gaps: List[GapAnalysis],
        strategy: ResearchUtilizationStrategy,
        context: GapResearchContext
    ) -> Dict[str, Any]:
        """Estimate resource requirements for gap research."""
        requirements = {}

        # Base requirements
        requirements['research_time'] = self._estimate_timeline(
            GapResearchDecision.RECOMMENDED_GAP_RESEARCH, recommended_gaps
        )

        # Gap-specific requirements
        requirements['gap_topics'] = len(recommended_gaps)
        requirements['critical_gaps'] = [
            gap for gap in recommended_gaps
            if gap.priority_level == LegacyPriorityLevel.CRITICAL
        ]

        # Strategy-based requirements
        if strategy in [ResearchUtilizationStrategy.NEW_RESEARCH_REQUIRED]:
            requirements['new_research_budget'] = "allocated"
            requirements['additional_sources'] = min(10, len(recommended_gaps) * 2)
        else:
            requirements['existing_research_utilization'] = "high"

        return requirements

    def _calculate_analyst_confidence(
        self,
        confidence_level: GapResearchConfidence,
        uncertainty_factors: List[str]
    ) -> float:
        """Calculate analyst confidence in the decision."""
        base_confidence = {
            GapResearchConfidence.VERY_HIGH: 0.95,
            GapResearchConfidence.HIGH: 0.85,
            GapResearchConfidence.MODERATE: 0.75,
            GapResearchConfidence.LOW: 0.65,
            GapResearchConfidence.VERY_LOW: 0.55
        }

        confidence = base_confidence.get(confidence_level, 0.75)

        # Reduce confidence based on uncertainty factors
        uncertainty_penalty = len(uncertainty_factors) * 0.05
        final_confidence = max(0.3, confidence - uncertainty_penalty)

        return final_confidence

    def _requires_manual_review(
        self,
        decision_type: GapResearchDecision,
        confidence_level: GapResearchConfidence
    ) -> bool:
        """Determine if manual review is required."""
        # Require manual review for critical decisions with low confidence
        if (decision_type == GapResearchDecision.CRITICAL_GAP_RESEARCH and
            confidence_level in [GapResearchConfidence.LOW, GapResearchConfidence.VERY_LOW]):
            return True

        # Require manual review for very low confidence decisions
        if confidence_level == GapResearchConfidence.VERY_LOW:
            return True

        return False

    async def _log_decision_made(self, decision: GapResearchDecision, session_id: str):
        """Log the gap research decision made."""
        self.logger.info(f"Gap research decision made for session {session_id}",
                        decision=decision.decision_type.value,
                        confidence=decision.confidence_level.value,
                        priority=f"{decision.priority_score:.2f}",
                        gaps=len(decision.recommended_gaps),
                        strategy=decision.recommended_strategy.value)

    async def _send_decision_notification(
        self,
        decision: GapResearchDecision,
        session_id: str
    ):
        """Send notification about the gap research decision."""
        if not self.message_processor:
            return

        try:
            message_type = MessageType.SUCCESS if decision.decision_type in [
                GapResearchDecision.NO_GAP_RESEARCH_NEEDED
            ] else MessageType.WARNING

            await self.message_processor.process_message(
                message_type,
                f"🔍 Gap Research Decision: {decision.decision_type.value.replace('_', ' ').title()}",
                metadata={
                    'session_id': session_id,
                    'decision_type': decision.decision_type.value,
                    'confidence_level': decision.confidence_level.value,
                    'priority_score': decision.priority_score,
                    'recommended_strategy': decision.recommended_strategy.value,
                    'gaps_count': len(decision.recommended_gaps),
                    'estimated_cost': decision.estimated_cost,
                    'estimated_timeline': decision.timeline_estimate,
                    'roi_estimate': decision.roi_estimate
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to send gap research decision notification: {e}")


# Factory function for creating gap research decision engine
def create_gap_research_decision_engine(
    editorial_engine: Optional[EnhancedEditorialDecisionEngine] = None,
    confidence_thresholds: Optional[Dict[str, float]] = None,
    message_processor: Optional[MessageProcessor] = None
) -> GapResearchDecisionEngine:
    """
    Create and configure a gap research decision engine.

    Args:
        editorial_engine: Enhanced editorial decision engine
        confidence_thresholds: Custom confidence thresholds
        message_processor: Optional message processor

    Returns:
        Configured GapResearchDecisionEngine instance
    """
    return GapResearchDecisionEngine(
        editorial_engine=editorial_engine,
        confidence_thresholds=confidence_thresholds,
        message_processor=message_processor
    )