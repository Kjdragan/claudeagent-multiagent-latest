#!/usr/bin/env python3
"""
Hook-Enhanced Report Validation System

This module provides comprehensive hook-based validation for data integration and quality,
implementing real-time quality scoring, template detection, and compliance tracking
using the Claude Agent SDK hooks.

Key Features:
- Real-time quality scoring and validation
- Template response detection and prevention
- Hook-based compliance monitoring
- Data integration assessment
- Research pipeline adherence tracking
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..core.quality_framework import QualityFramework, QualityAssessment
from ..config.sdk_config import get_sdk_config

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check with detailed feedback."""

    is_valid: bool
    score: float
    confidence: float
    feedback: str
    issues: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TemplateDetectionResult:
    """Result of template response analysis."""

    is_template: bool
    template_score: float  # 0-100, higher = more template-like
    template_patterns: List[str]
    generic_phrases: List[str]
    missing_data_indicators: List[str]
    content_specificity: float  # 0-100, higher = more specific

    def __post_init__(self):
        if self.template_patterns is None:
            self.template_patterns = []
        if self.generic_phrases is None:
            self.generic_phrases = []
        if self.missing_data_indicators is None:
            self.missing_data_indicators = []


@dataclass
class DataIntegrationResult:
    """Result of data integration assessment."""

    integration_score: float  # 0-100
    source_count: int
    citations_found: int
    data_points_mentioned: int
    specific_references: List[str]
    missing_sources: List[str]
    data_quality_indicators: Dict[str, float]

    def __post_init__(self):
        if self.specific_references is None:
            self.specific_references = []
        if self.missing_sources is None:
            self.missing_sources = []
        if self.data_quality_indicators is None:
            self.data_quality_indicators = {}


class ReportValidationSystem:
    """
    Hook-enhanced validation system for reports and research outputs.

    Provides comprehensive validation using Claude Agent SDK hooks for:
    - Template response detection and prevention
    - Data integration quality assessment
    - Research pipeline compliance tracking
    - Real-time quality monitoring
    """

    def __init__(self):
        self.quality_framework = QualityFramework()
        self.sdk_config = get_sdk_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Template detection patterns
        self.template_patterns = [
            r"(This\s+(?:report|analysis|study|overview)\s+(?:provides|offers|presents|gives)\s+(?:a\s+)?(?:comprehensive|detailed|thorough|general|broad)\s+(?:overview|summary|analysis|examination|look\s+at))",
            r"(In\s+(?:conclusion|summary|to\s+conclude|to\s+summarize))",
            r"(As\s+(?:mentioned\s+above|noted\s+previously|discussed\s+earlier|we\s+can\s+see))",
            r"(It\s+is\s+(?:important\s+to\s+note|worth\s+noting|clear\s+that|evident\s+that))",
            r"(This\s+(?:research|analysis|topic)\s+(?:requires|needs|demands)\s+(?:further\s+(?:research|study|investigation|analysis)))",
            r"(The\s+(?:findings|results|data|evidence)\s+(?:suggest|indicate|show|demonstrate))",
            r"(\d+\.\s+(?:Introduction|Background|Overview|Summary|Conclusion|Methodology|Results|Discussion))",
            r"(Overall,\s+(?:this\s+)?(?:report|analysis|study)\s+(?:provides|offers|gives))",
        ]

        # Generic phrases indicating template responses
        self.generic_phrases = [
            "further research is needed",
            "more comprehensive analysis",
            "broader context",
            "additional data",
            "beyond the scope",
            "general overview",
            "comprehensive analysis",
            "detailed examination",
            "thorough review",
            "in conclusion",
            "to summarize",
            "as noted above",
            "it is worth noting",
            "this report provides",
            "this analysis offers",
            "the findings suggest",
        ]

        # Indicators of missing specific data
        self.missing_data_patterns = [
            r"\[\s*\d+\s+source[s]?\s*\]",
            r"\[\s*citation\s+needed\s*\]",
            r"\[\s*source\s*\]",
            r"\[\s*ref\s*\]",
            r"\[Source:\s*\d+\]",
            r"(?:source|citation|reference):\s*\[\s*\]",
            r"(?:according|based\s+on)\s+(?:sources|reports|studies|data)\s+(?:but|however)\s+(?:no\s+specific|without\s+specific)",
        ]

    def detect_template_response(self, content: str) -> TemplateDetectionResult:
        """
        Analyze content for template response patterns.

        Args:
            content: Content to analyze for template patterns

        Returns:
            TemplateDetectionResult with detailed analysis
        """
        content_lower = content.lower()

        # Check for template patterns
        template_matches = []
        for pattern in self.template_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            template_matches.extend(matches)

        # Check for generic phrases
        generic_matches = []
        for phrase in self.generic_phrases:
            if phrase in content_lower:
                generic_matches.append(phrase)

        # Check for missing data indicators
        missing_data_matches = []
        for pattern in self.missing_data_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            missing_data_matches.extend(matches)

        # Calculate content specificity
        # Higher specificity = more concrete details, fewer generic statements
        specificity_score = self._calculate_content_specificity(content)

        # Calculate template score (0-100, higher = more template-like)
        template_score = min(100, (
            len(template_matches) * 15 +
            len(generic_matches) * 10 +
            len(missing_data_matches) * 20 +
            (100 - specificity_score) * 0.3
        ))

        # Determine if it's a template response
        is_template = template_score > 60  # Threshold for template detection

        return TemplateDetectionResult(
            is_template=is_template,
            template_score=template_score,
            template_patterns=template_matches,
            generic_phrases=generic_matches,
            missing_data_indicators=missing_data_matches,
            content_specificity=specificity_score
        )

    def assess_data_integration(self, content: str, expected_sources: int = 0) -> DataIntegrationResult:
        """
        Assess how well the content integrates research data.

        Args:
            content: Content to analyze for data integration
            expected_sources: Expected number of sources to be referenced

        Returns:
            DataIntegrationResult with detailed integration analysis
        """
        # Extract source references
        source_references = self._extract_source_references(content)

        # Count citations
        citations_found = len(source_references)

        # Look for specific data points
        data_points = self._extract_data_points(content)
        data_points_mentioned = len(data_points)

        # Find specific references to sources
        specific_references = self._find_specific_references(content)

        # Identify missing sources (if expected_sources is provided)
        missing_sources = []
        if expected_sources > 0:
            missing_sources_count = max(0, expected_sources - citations_found)
            missing_sources = [f"Missing source {i+1}" for i in range(missing_sources_count)]

        # Calculate data quality indicators
        data_quality_indicators = {
            "source_diversity": self._calculate_source_diversity(source_references),
            "data_richness": min(100, data_points_mentioned * 10),
            "citation_accuracy": self._assess_citation_accuracy(content, source_references),
            "contextual_integration": self._assess_contextual_integration(content, data_points),
        }

        # Calculate overall integration score
        integration_score = self._calculate_integration_score(
            citations_found, expected_sources, data_points_mentioned,
            data_quality_indicators, specific_references
        )

        return DataIntegrationResult(
            integration_score=integration_score,
            source_count=citations_found,
            citations_found=citations_found,
            data_points_mentioned=data_points_mentioned,
            specific_references=specific_references,
            missing_sources=missing_sources,
            data_quality_indicators=data_quality_indicators
        )

    async def validate_report_quality(self, content: str, context: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive report quality validation using hooks.

        Args:
            content: Report content to validate
            context: Context information (topic, sources, session_id, etc.)

        Returns:
            ValidationResult with comprehensive quality analysis
        """
        issues = []
        recommendations = []
        metrics = {}

        # Template detection
        template_result = self.detect_template_response(content)
        metrics["template_score"] = template_result.template_score
        metrics["content_specificity"] = template_result.content_specificity

        if template_result.is_template:
            issues.append("Content appears to be a template response")
            recommendations.append("Replace generic phrases with specific research findings")
            recommendations.append("Include actual data points and source citations")

        # Data integration assessment
        expected_sources = context.get("expected_sources", 0)
        integration_result = self.assess_data_integration(content, expected_sources)
        metrics.update(integration_result.data_quality_indicators)
        metrics["source_count"] = integration_result.source_count
        metrics["integration_score"] = integration_result.integration_score

        if integration_result.integration_score < 40:
            issues.append("Poor data integration quality")
            recommendations.append("Add more specific source references")
            recommendations.append("Include concrete data points and statistics")

        if integration_result.missing_sources:
            issues.append(f"Missing references to {len(integration_result.missing_sources)} expected sources")
            recommendations.append("Ensure all expected sources are properly cited")

        # Quality framework assessment
        quality_assessment = await self.quality_framework.assess_quality(content, context)
        metrics["quality_score"] = quality_assessment.overall_score

        if quality_assessment.overall_score < 70:
            issues.append("Quality assessment score below threshold")
            recommendations.extend(quality_assessment.actionable_recommendations)

        # Calculate overall validation score
        validation_score = self._calculate_validation_score(
            template_result, integration_result, quality_assessment
        )

        # Determine if content is valid
        is_valid = (
            not template_result.is_template and
            integration_result.integration_score >= 40 and
            quality_assessment.overall_score >= 70
        )

        # Generate feedback
        feedback = self._generate_validation_feedback(
            template_result, integration_result, quality_assessment, is_valid
        )

        return ValidationResult(
            is_valid=is_valid,
            score=validation_score,
            confidence=self._calculate_confidence(metrics),
            feedback=feedback,
            issues=issues,
            recommendations=recommendations,
            metrics=metrics
        )

    def validate_research_pipeline_compliance(self, session_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate research pipeline compliance using hook tracking.

        Args:
            session_data: Session metadata and tracking data

        Returns:
            ValidationResult with compliance analysis
        """
        issues = []
        recommendations = []
        metrics = {}

        # Check if research was actually performed
        research_performed = session_data.get("research_performed", False)
        metrics["research_performed"] = research_performed

        if not research_performed:
            issues.append("No research pipeline execution detected")
            recommendations.append("Execute actual research pipeline before report generation")

        # Check source count accuracy
        expected_sources = session_data.get("expected_sources", 0)
        actual_sources = session_data.get("actual_sources", 0)
        metrics["source_count_accuracy"] = min(100, (actual_sources / expected_sources * 100) if expected_sources > 0 else 0)

        if actual_sources < expected_sources * 0.8:  # Allow 20% tolerance
            issues.append(f"Source count mismatch: expected {expected_sources}, got {actual_sources}")
            recommendations.append("Ensure accurate source count reporting")

        # Check data integration hooks execution
        hooks_executed = session_data.get("hooks_executed", [])
        required_hooks = ["validate_research_data_usage", "enforce_citation_requirements"]
        missing_hooks = [hook for hook in required_hooks if hook not in hooks_executed]

        metrics["hook_compliance_rate"] = len([h for h in required_hooks if h in hooks_executed]) / len(required_hooks) * 100

        if missing_hooks:
            issues.append(f"Missing required hook executions: {missing_hooks}")
            recommendations.append("Ensure all required hooks are executed during report generation")

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(metrics)
        is_valid = compliance_score >= 80

        feedback = self._generate_compliance_feedback(compliance_score, issues, is_valid)

        return ValidationResult(
            is_valid=is_valid,
            score=compliance_score,
            confidence=0.9,  # High confidence in compliance checks
            feedback=feedback,
            issues=issues,
            recommendations=recommendations,
            metrics=metrics
        )

    # Helper methods

    def _calculate_content_specificity(self, content: str) -> float:
        """Calculate content specificity score (0-100)."""
        # Count specific indicators
        specific_indicators = [
            r"\d+%|\d+\.\d+%",  # Percentages
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?",  # Money amounts
            r"\d{4}",  # Years
            r"\b(?:million|billion|trillion)\b",  # Large numbers
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s*[A-Z]{2,}",  # Organizations with acronyms
            r"https?://[^\s]+",  # URLs
            r"[A-Z][a-z]+\s+[A-Z][a-z]+,\s*[A-Z]{2}",  # City, State format
        ]

        specific_count = sum(len(re.findall(pattern, content)) for pattern in specific_indicators)

        # Calculate specificity based on content length and specific indicators
        word_count = len(content.split())
        specificity = min(100, (specific_count / max(word_count / 100, 1)) * 50)

        # Boost score for longer, detailed content
        if word_count > 500:
            specificity = min(100, specificity + 20)
        elif word_count > 200:
            specificity = min(100, specificity + 10)

        return specificity

    def _extract_source_references(self, content: str) -> List[str]:
        """Extract source references from content."""
        # Pattern to match various citation formats
        citation_patterns = [
            r"\[(\d+)\]",
            r"\[Source:\s*(\d+)\]",
            r"\(source\s*\d+\)",
            r"\[([A-Za-z]+\s+et\s+al\.?,?\s*\d{4})\]",
            r"\(([A-Za-z]+\s+et\s+al\.?,?\s*\d{4})\)",
            r"\[([A-Za-z]+\s*\d{4})\]",
            r"\(([A-Za-z]+\s*\d{4})\)",
        ]

        references = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references.extend(matches)

        return list(set(references))  # Remove duplicates

    def _extract_data_points(self, content: str) -> List[str]:
        """Extract specific data points from content."""
        data_patterns = [
            r"\d+(?:\.\d+)?%",  # Percentages
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?",  # Money
            r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion)",  # Large numbers
            r"\b\d{4}\b",  # Years
            r"\d+(?:\.\d+)?\s*(?:times|fold|increase|decrease|growth|decline)",  # Changes
            r"\d+(?:\.\d+)?\s*(?:years|months|days|hours)",  # Time periods
        ]

        data_points = []
        for pattern in data_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            data_points.extend(matches)

        return data_points

    def _find_specific_references(self, content: str) -> List[str]:
        """Find specific references to sources, organizations, or studies."""
        # Look for specific reference patterns
        reference_patterns = [
            r"[A-Z][a-z]+\s+[A-Z][a-z]+\s+University",
            r"[A-Z][a-z]+\s+[A-Z][a-z]+\s+Institute",
            r"[A-Z][a-z]+\s+[A-Z][a-z]+\s+Center",
            r"[A-Z][a-z]+\s+[A-Z][a-z]+\s+Organization",
            r"[A-Z][a-z]+\s+[A-Z][a-z]+\s+Agency",
            r"[A-Z][a-z]+\s+[A-Z][a-z]+\s+Department",
            r"according\s+to\s+[A-Z][a-z]+\s+[A-Z][a-z]+",
            r"[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)",
        ]

        references = []
        for pattern in reference_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references.extend(matches)

        return list(set(references))

    def _calculate_source_diversity(self, sources: List[str]) -> float:
        """Calculate source diversity score."""
        if not sources:
            return 0

        # Simple diversity calculation based on unique source types
        unique_sources = len(set(sources))
        diversity_score = min(100, (unique_sources / len(sources)) * 100)

        return diversity_score

    def _assess_citation_accuracy(self, content: str, citations: List[str]) -> float:
        """Assess citation accuracy and consistency."""
        if not citations:
            return 0

        # Check for consistent citation format
        citation_formats = [
            r"\[\d+\]",
            r"\[Source:\s*\d+\]",
            r"\(source\s*\d+\)",
        ]

        format_consistency = 0
        for pattern in citation_formats:
            if all(re.match(pattern, cit) for cit in citations):
                format_consistency = 100
                break

        return format_consistency

    def _assess_contextual_integration(self, content: str, data_points: List[str]) -> float:
        """Assess how well data points are integrated into context."""
        if not data_points:
            return 0

        # Look for contextual integration patterns
        integration_patterns = [
            r"\d+(?:\.\d+)?%[^.]*\b(?:increase|decrease|growth|decline|change)\b",
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?[^.]*\b(?:cost|value|worth|budget|revenue)\b",
            r"\d{4}[^.]*\b(?:study|report|research|analysis|published)\b",
        ]

        integrated_points = 0
        for pattern in integration_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            integrated_points += len(matches)

        integration_score = min(100, (integrated_points / len(data_points)) * 100)
        return integration_score

    def _calculate_integration_score(self, citations: int, expected_sources: int,
                                   data_points: int, quality_indicators: Dict[str, float],
                                   specific_refs: List[str]) -> float:
        """Calculate overall data integration score."""
        score_components = []

        # Source coverage
        if expected_sources > 0:
            source_coverage = min(100, (citations / expected_sources) * 100)
            score_components.append(source_coverage)

        # Data richness
        score_components.append(quality_indicators.get("data_richness", 0))

        # Citation accuracy
        score_components.append(quality_indicators.get("citation_accuracy", 0))

        # Contextual integration
        score_components.append(quality_indicators.get("contextual_integration", 0))

        # Specific references bonus
        specific_ref_bonus = min(20, len(specific_refs) * 5)
        score_components.append(specific_ref_bonus)

        # Average the scores
        return sum(score_components) / len(score_components) if score_components else 0

    def _calculate_validation_score(self, template_result: TemplateDetectionResult,
                                  integration_result: DataIntegrationResult,
                                  quality_assessment: QualityAssessment) -> float:
        """Calculate overall validation score."""
        # Penalty for template responses
        template_penalty = template_result.template_score if template_result.is_template else 0

        # Weight the different components
        scores = [
            max(0, 100 - template_penalty),  # Template penalty (inverted)
            integration_result.integration_score,  # Data integration
            quality_assessment.overall_score,  # Quality assessment
        ]

        weights = [0.3, 0.4, 0.3]  # Emphasize data integration

        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        return weighted_score

    def _calculate_confidence(self, metrics: Dict[str, Any]) -> float:
        """Calculate confidence in validation results."""
        # Base confidence
        confidence = 0.8

        # Adjust based on data availability
        if metrics.get("source_count", 0) > 0:
            confidence += 0.1
        if metrics.get("integration_score", 0) > 70:
            confidence += 0.1

        return min(1.0, confidence)

    def _generate_validation_feedback(self, template_result: TemplateDetectionResult,
                                    integration_result: DataIntegrationResult,
                                    quality_assessment: QualityAssessment,
                                    is_valid: bool) -> str:
        """Generate comprehensive validation feedback."""
        if is_valid:
            return "✅ Content validation passed. Report demonstrates good data integration and quality."

        feedback_parts = []

        if template_result.is_template:
            feedback_parts.append("❌ Template response detected. Content contains generic patterns.")

        if integration_result.integration_score < 70:
            feedback_parts.append("❌ Poor data integration. Insufficient source references and data points.")

        if quality_assessment.overall_score < 70:
            feedback_parts.append(f"❌ Low quality score ({quality_assessment.overall_score}/100).")

        return " | ".join(feedback_parts) if feedback_parts else "Content validation failed."

    def _calculate_compliance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate research pipeline compliance score."""
        score_components = [
            metrics.get("research_performed", 0) * 100,  # Binary: 0 or 100
            metrics.get("source_count_accuracy", 0),  # Accuracy percentage
            metrics.get("hook_compliance_rate", 0),  # Hook compliance percentage
        ]

        return sum(score_components) / len(score_components) if score_components else 0

    def _generate_compliance_feedback(self, compliance_score: float, issues: List[str], is_valid: bool) -> str:
        """Generate compliance feedback."""
        if is_valid:
            return "✅ Research pipeline compliance verified. All required steps and hooks executed properly."

        return f"❌ Compliance issues detected. Score: {compliance_score:.1f}/100. Issues: {'; '.join(issues)}"


# Hook implementations for Claude Agent SDK

def validate_research_data_usage(session_id: str, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook implementation: Validate research data usage before report generation.

    Args:
        session_id: Session identifier
        content: Content to validate
        context: Context information

    Returns:
        Hook result with validation status
    """
    validator = ReportValidationSystem()

    # Check if research corpus exists
    corpus_path = Path("KEVIN") / "sessions" / session_id / "research_corpus.json"
    corpus_exists = corpus_path.exists()

    if not corpus_exists:
        return {
            "success": False,
            "message": "No research corpus found. Execute research pipeline first.",
            "hook_name": "validate_research_data_usage",
            "required_action": "build_research_corpus"
        }

    # Validate data integration
    integration_result = validator.assess_data_integration(content)

    if integration_result.integration_score < 50:
        return {
            "success": False,
            "message": f"Insufficient data integration (score: {integration_result.integration_score}/100)",
            "hook_name": "validate_research_data_usage",
            "integration_score": integration_result.integration_score,
            "required_action": "integrate_research_data"
        }

    return {
        "success": True,
        "message": "Research data validation passed",
        "hook_name": "validate_research_data_usage",
        "integration_score": integration_result.integration_score
    }


def enforce_citation_requirements(session_id: str, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook implementation: Enforce proper citation requirements.

    Args:
        session_id: Session identifier
        content: Content to validate
        context: Context information

    Returns:
        Hook result with citation validation
    """
    validator = ReportValidationSystem()

    # Extract source references
    source_refs = validator._extract_source_references(content)
    expected_sources = context.get("expected_sources", 1)

    if len(source_refs) < expected_sources * 0.8:  # Allow 20% tolerance
        return {
            "success": False,
            "message": f"Insufficient citations. Expected {expected_sources}, found {len(source_refs)}",
            "hook_name": "enforce_citation_requirements",
            "citations_found": len(source_refs),
            "citations_expected": expected_sources,
            "required_action": "add_proper_citations"
        }

    return {
        "success": True,
        "message": "Citation requirements satisfied",
        "hook_name": "enforce_citation_requirements",
        "citations_found": len(source_refs)
    }


def validate_report_quality_standards(session_id: str, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hook implementation: Validate outgoing report quality standards.

    Args:
        session_id: Session identifier
        content: Content to validate
        context: Context information

    Returns:
        Hook result with quality validation
    """
    validator = ReportValidationSystem()

    # Template detection
    template_result = validator.detect_template_response(content)

    if template_result.is_template:
        return {
            "success": False,
            "message": f"Template response detected (score: {template_result.template_score}/100)",
            "hook_name": "validate_report_quality_standards",
            "template_score": template_result.template_score,
            "template_patterns": template_result.template_patterns,
            "required_action": "replace_template_with_data_driven_content"
        }

    # Comprehensive quality validation
    validation_result = validator.validate_report_quality(content, context)

    if not validation_result.is_valid:
        return {
            "success": False,
            "message": f"Report quality standards not met (score: {validation_result.score}/100)",
            "hook_name": "validate_report_quality_standards",
            "validation_score": validation_result.score,
            "issues": validation_result.issues,
            "required_action": "improve_report_quality"
        }

    return {
        "success": True,
        "message": "Report quality standards validated",
        "hook_name": "validate_report_quality_standards",
        "validation_score": validation_result.score,
        "quality_metrics": validation_result.metrics
    }