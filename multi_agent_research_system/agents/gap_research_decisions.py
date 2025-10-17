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
    class GapCategory(Enum):
        FACTUAL_GAPS = "factual_gaps"
        TEMPORAL_GAPS = "temporal_gaps"
        COMPARATIVE_GAPS = "comparative_gaps"
        ANALYTICAL_GAPS = "analytical_gaps"
        CONTEXTUAL_GAPS = "contextual_gaps"
        METHODOLOGICAL_GAPS = "methodological_gaps"
        EXPERT_OPINION_GAPS = "expert_opinion_gaps"
        DATA_GAPS = "data_gaps"

    class EnhancedEditorialDecisionEngine:
        def __init__(self, *args, **kwargs):
            pass

    class GapDecisionConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    @dataclass
    class ResourceRequirements:
        estimated_scrapes_needed: int
        estimated_queries_needed: int
        time_requirement: timedelta
        budget_requirement: float
        complexity_level: int

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

    # Gap research recommendations
    recommended_strategy: ResearchUtilizationStrategy

    # Cost-benefit analysis
    estimated_cost: str
    estimated_benefit: str
    roi_estimate: float  # 0.0 - 1.0

    # Implementation details
    suggested_approach: str
    timeline_estimate: str

    # Fields with defaults
    supporting_evidence: List[str] = field(default_factory=list)
    uncertainty_factors: List[str] = field(default_factory=list)
    alternative_options: List[str] = field(default_factory=list)
    recommended_gaps: List[GapAnalysis] = field(default_factory=list)
    research_topics: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    success_metrics: List[str] = field(default_factory=list)

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

        # Decision thresholds (configurable) - UPDATED FOR HIGH THRESHOLDS per repair2.md
        self.thresholds = {
            'gap_research_trigger': 0.90,        # INCREASED: Confidence threshold for triggering gap research (was 0.70)
            'high_priority_threshold': 0.95,    # INCREASED: High priority threshold (was 0.85)
            'critical_gap_threshold': 0.97,      # INCREASED: Critical gap threshold (was 0.90)
            'existing_research_threshold': 0.80, # INCREASED: Threshold for preferring existing research (was 0.75)
            'decision_confidence_minimum': 0.75,  # INCREASED: Minimum confidence for decisions (was 0.60)
            'roi_minimum_threshold': 0.50,        # INCREASED: Minimum ROI for gap research (was 0.30)
            'uncertainty_tolerance': 0.20          # DECREASED: Maximum uncertainty tolerated (was 0.40)
        }


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