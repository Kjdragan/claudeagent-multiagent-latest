"""
Message Quality Analyzer - Comprehensive Quality Assessment and Scoring

This module provides sophisticated message quality analysis with multi-dimensional
assessment, detailed feedback, and actionable improvement recommendations.

Key Features:
- Multi-dimensional quality assessment (content, structure, clarity, relevance)
- Context-aware quality scoring with adaptive thresholds
- Detailed quality feedback with specific improvement recommendations
- Quality trend analysis and historical tracking
- Comparative quality assessment across message types
- Actionable quality enhancement suggestions
"""

import asyncio
import re
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.message_types import RichMessage, EnhancedMessageType, MessagePriority


class QualityDimension(Enum):
    """Quality assessment dimensions."""

    CONTENT = "content"
    STRUCTURE = "structure"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    ACCESSIBILITY = "accessibility"


@dataclass
class QualityMetric:
    """Individual quality metric with detailed assessment."""

    dimension: QualityDimension
    score: float
    weight: float
    feedback: str
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment result."""

    overall_score: float
    quality_level: str
    metrics: Dict[QualityDimension, QualityMetric]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    improvement_potential: float
    processing_time: float
    assessment_metadata: Dict[str, Any] = field(default_factory=dict)


class MessageQualityAnalyzer:
    """Advanced message quality analyzer with multi-dimensional assessment."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize quality analyzer with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Quality dimension weights (configurable)
        self.dimension_weights = {
            QualityDimension.CONTENT: self.config.get("weights", {}).get("content", 0.25),
            QualityDimension.STRUCTURE: self.config.get("weights", {}).get("structure", 0.15),
            QualityDimension.CLARITY: self.config.get("weights", {}).get("clarity", 0.20),
            QualityDimension.RELEVANCE: self.config.get("weights", {}).get("relevance", 0.15),
            QualityDimension.COMPLETENESS: self.config.get("weights", {}).get("completeness", 0.10),
            QualityDimension.ACCURACY: self.config.get("weights", {}).get("accuracy", 0.10),
            QualityDimension.CONSISTENCY: self.config.get("weights", {}).get("consistency", 0.05),
        }

        # Quality thresholds
        self.quality_thresholds = {
            "excellent": self.config.get("thresholds", {}).get("excellent", 0.9),
            "good": self.config.get("thresholds", {}).get("good", 0.75),
            "fair": self.config.get("thresholds", {}).get("fair", 0.6),
            "poor": self.config.get("thresholds", {}).get("poor", 0.0),
        }

        # Assessment statistics
        self.assessment_stats = {
            "total_assessments": 0,
            "average_score": 0.0,
            "assessments_by_type": {},
            "assessments_by_level": {},
            "processing_time": 0.0
        }

        # Quality assessment functions
        self.assessors = {
            QualityDimension.CONTENT: self._assess_content_quality,
            QualityDimension.STRUCTURE: self._assess_structure_quality,
            QualityDimension.CLARITY: self._assess_clarity_quality,
            QualityDimension.RELEVANCE: self._assess_relevance_quality,
            QualityDimension.COMPLETENESS: self._assess_completeness_quality,
            QualityDimension.ACCURACY: self._assess_accuracy_quality,
            QualityDimension.CONSISTENCY: self._assess_consistency_quality,
            QualityDimension.ACCESSIBILITY: self._assess_accessibility_quality,
        }

    async def assess_message(self, message: RichMessage) -> Dict[str, Any]:
        """Perform comprehensive quality assessment of a message."""
        start_time = datetime.now()

        try:
            # Assess each quality dimension
            metrics = {}
            total_weighted_score = 0.0
            total_weight = 0.0

            for dimension, assessor_func in self.assessors.items():
                try:
                    metric = await assessor_func(message)
                    metrics[dimension] = metric

                    # Calculate weighted contribution
                    weight = self.dimension_weights.get(dimension, 0.1)
                    total_weighted_score += metric.score * weight
                    total_weight += weight

                except Exception as e:
                    self.logger.warning(f"Failed to assess {dimension.value}: {str(e)}")
                    # Create default metric for failed assessment
                    metrics[dimension] = QualityMetric(
                        dimension=dimension,
                        score=0.5,
                        weight=self.dimension_weights.get(dimension, 0.1),
                        feedback=f"Assessment failed: {str(e)}",
                        issues=[f"Unable to assess {dimension.value}"]
                    )

            # Calculate overall score
            overall_score = total_weighted_score / max(total_weight, 0.1)

            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)

            # Generate strengths and weaknesses
            strengths, weaknesses = self._analyze_strengths_weaknesses(metrics)

            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, weaknesses)

            # Calculate improvement potential
            improvement_potential = self._calculate_improvement_potential(metrics)

            # Create assessment result
            processing_time = (datetime.now() - start_time).total_seconds()
            assessment = QualityAssessment(
                overall_score=overall_score,
                quality_level=quality_level,
                metrics=metrics,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                improvement_potential=improvement_potential,
                processing_time=processing_time,
                assessment_metadata={
                    "message_type": message.message_type.value,
                    "message_length": len(message.content),
                    "assessment_timestamp": datetime.now().isoformat(),
                    "dimensions_assessed": list(metrics.keys())
                }
            )

            # Update statistics
            self._update_assessment_stats(message, assessment)

            self.logger.debug(f"Assessed message {message.id}: {quality_level} quality ({overall_score:.2f})")

            return {
                "overall_quality": overall_score,
                "quality_level": quality_level,
                "dimensions": {dim.value: self._metric_to_dict(metric) for dim, metric in metrics.items()},
                "strengths": strengths,
                "weaknesses": weaknesses,
                "recommendations": recommendations,
                "improvement_potential": improvement_potential,
                "processing_time": processing_time
            }

        except Exception as e:
            self.logger.error(f"Failed to assess message {message.id}: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                "overall_quality": 0.5,
                "quality_level": "unknown",
                "error": str(e),
                "processing_time": processing_time
            }

    async def _assess_content_quality(self, message: RichMessage) -> QualityMetric:
        """Assess content quality dimension."""
        content = message.content
        score = 0.0
        issues = []
        recommendations = []
        evidence = {}

        # Length assessment
        word_count = len(content.split())
        char_count = len(content)

        if word_count < 10:
            issues.append("Content is too brief")
            recommendations.append("Expand content with more detail and context")
            score += 0.2
        elif word_count < 30:
            issues.append("Content could be more detailed")
            recommendations.append("Consider adding more examples or explanations")
            score += 0.5
        elif word_count > 500:
            issues.append("Content is quite long")
            recommendations.append("Consider breaking into smaller sections")
            score += 0.8
        else:
            score += 1.0

        evidence["word_count"] = word_count
        evidence["char_count"] = char_count

        # Substance assessment
        substantive_indicators = [
            "analysis", "research", "findings", "results", "conclusion",
            "implementation", "strategy", "recommendation", "evaluation"
        ]

        substantive_score = sum(1 for indicator in substantive_indicators if indicator in content.lower())
        substance_ratio = substantive_score / max(word_count / 20, 1)  # Normalize by content length

        if substance_ratio > 0.3:
            score += 1.0
        elif substance_ratio > 0.1:
            score += 0.7
        else:
            issues.append("Content lacks substantive elements")
            recommendations.append("Include more analytical or research-based content")
            score += 0.3

        evidence["substantive_indicators"] = substantive_score

        # Originality assessment (simple heuristic)
        unique_phrases = set()
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                unique_phrases.add(sentence.strip()[:20])  # First 20 chars as phrase signature

        originality_ratio = len(unique_phrases) / max(len(sentences), 1)
        score += originality_ratio

        evidence["unique_phrases"] = len(unique_phrases)
        evidence["total_sentences"] = len(sentences)

        # Normalize score
        final_score = min(1.0, score / 4.0)

        feedback = self._generate_content_feedback(final_score, issues, evidence)

        return QualityMetric(
            dimension=QualityDimension.CONTENT,
            score=final_score,
            weight=self.dimension_weights[QualityDimension.CONTENT],
            feedback=feedback,
            issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    async def _assess_structure_quality(self, message: RichMessage) -> QualityMetric:
        """Assess structure quality dimension."""
        content = message.content
        score = 0.0
        issues = []
        recommendations = []
        evidence = {}

        # Heading structure
        heading_matches = list(re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE))
        heading_levels = [len(match.group(1)) for match in heading_matches]

        if heading_levels:
            # Check for proper heading hierarchy
            hierarchy_issues = 0
            for i in range(1, len(heading_levels)):
                if heading_levels[i] > heading_levels[i-1] + 1:
                    hierarchy_issues += 1

            if hierarchy_issues == 0:
                score += 1.0
            elif hierarchy_issues <= len(heading_levels) * 0.2:
                score += 0.7
                issues.append("Some heading levels skip hierarchy")
                recommendations.append("Ensure heading levels follow proper sequence (h1 → h2 → h3)")
            else:
                score += 0.3
                issues.append("Heading hierarchy is inconsistent")
                recommendations.append("Restructure headings to follow proper hierarchy")
        else:
            score += 0.5  # Neutral score for content without headings

        evidence["heading_count"] = len(heading_levels)
        evidence["heading_hierarchy_issues"] = hierarchy_issues if heading_levels else 0

        # List structure
        list_patterns = [
            r'^(\s*)[-*+]\s+(.+)$',  # Bulleted lists
            r'^(\s*)\d+\.\s+(.+)$',  # Numbered lists
        ]

        list_count = 0
        for pattern in list_patterns:
            list_count += len(re.findall(pattern, content, re.MULTILINE))

        if list_count > 0:
            score += 0.8
        evidence["list_count"] = list_count

        # Paragraph structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1)

        if 20 <= avg_paragraph_length <= 100:
            score += 1.0
        elif avg_paragraph_length < 20:
            issues.append("Paragraphs are too short")
            recommendations.append("Combine short paragraphs or add more detail")
            score += 0.6
        elif avg_paragraph_length > 100:
            issues.append("Paragraphs are too long")
            recommendations.append("Break long paragraphs into smaller ones")
            score += 0.6
        else:
            score += 0.8

        evidence["paragraph_count"] = len(paragraphs)
        evidence["avg_paragraph_length"] = avg_paragraph_length

        # Code blocks
        code_blocks = content.count('```')
        if code_blocks > 0 and code_blocks % 2 == 0:
            score += 0.5
        elif code_blocks > 0:
            issues.append("Unclosed code blocks detected")
            recommendations.append("Ensure all code blocks are properly closed")
            score -= 0.2

        evidence["code_blocks"] = code_blocks

        # Normalize score
        final_score = min(1.0, max(0.0, score / 3.5))

        feedback = self._generate_structure_feedback(final_score, issues, evidence)

        return QualityMetric(
            dimension=QualityDimension.STRUCTURE,
            score=final_score,
            weight=self.dimension_weights[QualityDimension.STRUCTURE],
            feedback=feedback,
            issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    async def _assess_clarity_quality(self, message: RichMessage) -> QualityMetric:
        """Assess clarity quality dimension."""
        content = message.content
        score = 0.0
        issues = []
        recommendations = []
        evidence = {}

        # Sentence length analysis
        sentences = re.split(r'[.!?]+', content)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

        if sentence_lengths:
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
            long_sentences = [l for l in sentence_lengths if l > 25]

            if avg_sentence_length <= 15:
                score += 1.0
            elif avg_sentence_length <= 20:
                score += 0.8
            elif avg_sentence_length <= 25:
                score += 0.6
                issues.append("Sentences tend to be long")
                recommendations.append("Break long sentences into shorter ones")
            else:
                score += 0.3
                issues.append("Sentences are too long and complex")
                recommendations.append("Significantly reduce sentence length for better readability")

            evidence["avg_sentence_length"] = avg_sentence_length
            evidence["long_sentence_count"] = len(long_sentences)
        else:
            score += 0.5
            evidence["sentence_count"] = 0

        # Vocabulary complexity
        words = content.split()
        complex_words = [w for w in words if len(w) > 8]

        if words:
            complexity_ratio = len(complex_words) / len(words)
            if complexity_ratio <= 0.1:
                score += 1.0
            elif complexity_ratio <= 0.2:
                score += 0.8
            elif complexity_ratio <= 0.3:
                score += 0.6
            else:
                score += 0.4
                issues.append("High density of complex words")
                recommendations.append("Consider using simpler terminology where possible")

            evidence["complex_word_ratio"] = complexity_ratio

        # Passive voice detection
        passive_patterns = [
            r'\b(is|are|was|were|be|been|being)\s+\w+\s+by\b',
            r'\b(was|were)\s+\w+ed\b'
        ]

        passive_count = sum(len(re.findall(pattern, content)) for pattern in passive_patterns)
        sentences_with_passive = passive_count / max(len(sentences), 1)

        if sentences_with_passive <= 0.2:
            score += 1.0
        elif sentences_with_passive <= 0.4:
            score += 0.8
        else:
            score += 0.5
            issues.append("High use of passive voice")
            recommendations.append("Consider using more active voice for clearer communication")

        evidence["passive_voice_ratio"] = sentences_with_passive

        # Jargon and technical terms
        technical_indicators = ['implementation', 'architecture', 'methodology', 'framework', 'paradigm']
        technical_count = sum(1 for indicator in technical_indicators if indicator in content.lower())

        if technical_count <= 2:
            score += 1.0
        elif technical_count <= 4:
            score += 0.8
        else:
            score += 0.6
            if len(words) < 100:  # Only flag for shorter content
                issues.append("High density of technical terms for content length")
                recommendations.append("Consider explaining technical terms or reducing jargon")

        evidence["technical_term_count"] = technical_count

        # Normalize score
        final_score = min(1.0, score / 4.0)

        feedback = self._generate_clarity_feedback(final_score, issues, evidence)

        return QualityMetric(
            dimension=QualityDimension.CLARITY,
            score=final_score,
            weight=self.dimension_weights[QualityDimension.CLARITY],
            feedback=feedback,
            issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    async def _assess_relevance_quality(self, message: RichMessage) -> QualityMetric:
        """Assess relevance quality dimension."""
        content = message.content
        score = 0.8  # Start with good default relevance
        issues = []
        recommendations = []
        evidence = {}

        # Context-based relevance assessment
        context_indicators = {
            EnhancedMessageType.RESEARCH_QUERY: ["research", "query", "search", "investigation"],
            EnhancedMessageType.RESEARCH_RESULT: ["results", "findings", "data", "analysis"],
            EnhancedMessageType.ANALYSIS_RESULT: ["analysis", "evaluation", "assessment", "conclusions"],
            EnhancedMessageType.QUALITY_ASSESSMENT: ["quality", "assessment", "evaluation", "metrics"],
            EnhancedMessageType.RECOMMENDATION: ["recommend", "suggest", "advise", "propose"],
        }

        expected_keywords = context_indicators.get(message.message_type, [])
        keyword_matches = sum(1 for keyword in expected_keywords if keyword in content.lower())

        if expected_keywords:
            relevance_ratio = keyword_matches / len(expected_keywords)
            score = relevance_ratio

            if relevance_ratio < 0.3:
                issues.append("Content may not be relevant to message type")
                recommendations.append("Include more content relevant to the message type context")
        else:
            # For generic message types, assess general relevance
            score = 0.8

        evidence["expected_keyword_matches"] = keyword_matches
        evidence["expected_keywords_total"] = len(expected_keywords)

        # Topic consistency assessment
        if message.metadata.session_id:
            # In a real implementation, this would compare with other messages in the session
            # For now, we'll use a simple heuristic
            score += 0.1  # Small bonus for having session context

        # Audience appropriateness
        content_length = len(content.split())
        if message.message_type in [EnhancedMessageType.SYSTEM_INFO, EnhancedMessageType.SYSTEM_ERROR]:
            if content_length > 100:
                issues.append("System messages should be concise")
                recommendations.append("Keep system messages brief and to the point")
                score -= 0.2

        evidence["content_length"] = content_length

        # Final score adjustment
        final_score = min(1.0, max(0.0, score))

        feedback = self._generate_relevance_feedback(final_score, issues, evidence)

        return QualityMetric(
            dimension=QualityDimension.RELEVANCE,
            score=final_score,
            weight=self.dimension_weights[QualityDimension.RELEVANCE],
            feedback=feedback,
            issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    async def _assess_completeness_quality(self, message: RichMessage) -> QualityMetric:
        """Assess completeness quality dimension."""
        content = message.content
        score = 0.0
        issues = []
        recommendations = []
        evidence = {}

        # Content completeness based on message type
        completeness_requirements = {
            EnhancedMessageType.RESEARCH_RESULT: {
                "indicators": ["findings", "methodology", "conclusions", "data"],
                "min_count": 2,
                "description": "research results"
            },
            EnhancedMessageType.ANALYSIS_RESULT: {
                "indicators": ["analysis", "results", "implications", "recommendations"],
                "min_count": 2,
                "description": "analysis results"
            },
            EnhancedMessageType.QUALITY_ASSESSMENT: {
                "indicators": ["quality", "score", "assessment", "metrics", "evaluation"],
                "min_count": 2,
                "description": "quality assessment"
            },
            EnhancedMessageType.RECOMMENDATION: {
                "indicators": ["recommend", "suggest", "propose", "advise"],
                "min_count": 1,
                "description": "recommendations"
            },
        }

        requirements = completeness_requirements.get(message.message_type)
        if requirements:
            found_indicators = sum(1 for indicator in requirements["indicators"] if indicator in content.lower())
            completeness_ratio = found_indicators / len(requirements["indicators"])

            if found_indicators >= requirements["min_count"]:
                score = 0.8 + (completeness_ratio * 0.2)  # 0.8-1.0
            else:
                score = completeness_ratio * 0.7  # 0.0-0.7
                issues.append(f"Missing key elements for {requirements['description']}")
                recommendations.append(f"Include more {requirements['description']} elements")

            evidence["required_indicators_found"] = found_indicators
            evidence["required_indicators_total"] = len(requirements["indicators"])
        else:
            # For generic content, assess basic completeness
            if len(content.split()) < 5:
                score = 0.3
                issues.append("Content is incomplete")
                recommendations.append("Add more detail to make content complete")
            elif len(content.split()) < 15:
                score = 0.6
                issues.append("Content could be more complete")
                recommendations.append("Expand content with additional information")
            else:
                score = 0.8

        # Structural completeness
        has_introduction = bool(re.search(r'^(.{10,})$', content, re.MULTILINE))
        has_conclusion = any(word in content.lower() for word in ["conclusion", "summary", "in summary", "therefore"])

        if has_introduction and has_conclusion:
            score += 0.1
        elif not has_introduction and len(content.split()) > 50:
            issues.append("Missing clear introduction")
            recommendations.append("Add an introduction to provide context")
        elif not has_conclusion and len(content.split()) > 50:
            issues.append("Missing clear conclusion")
            recommendations.append("Add a conclusion to summarize key points")

        evidence["has_introduction"] = has_introduction
        evidence["has_conclusion"] = has_conclusion

        # Normalize score
        final_score = min(1.0, score)

        feedback = self._generate_completeness_feedback(final_score, issues, evidence)

        return QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            score=final_score,
            weight=self.dimension_weights[QualityDimension.COMPLETENESS],
            feedback=feedback,
            issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    async def _assess_accuracy_quality(self, message: RichMessage) -> QualityMetric:
        """Assess accuracy quality dimension."""
        content = message.content
        score = 0.8  # Start with good default (can't fully verify accuracy automatically)
        issues = []
        recommendations = []
        evidence = {}

        # Check for potential accuracy issues
        accuracy_warnings = [
            (r'\b(approximately|about|around|roughly)\s+\d+%?\b', "estimated_values"),
            (r'\b(always|never|all|none|every)\b', "absolute_claims"),
            (r'\b(probably|maybe|perhaps|possibly)\b', "uncertainty_indicators"),
        ]

        warning_count = 0
        for pattern, warning_type in accuracy_warnings:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            if matches > 0:
                warning_count += matches
                evidence[warning_type] = matches

        if warning_count > len(content.split()) / 20:  # More than 5% of content has warnings
            score -= 0.2
            issues.append("Content contains multiple uncertainty indicators")
            recommendations.append("Consider verifying claims and reducing uncertainty language")

        # Check for data and evidence
        evidence_indicators = ["data shows", "research indicates", "according to", "evidence suggests"]
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in content.lower())

        if evidence_count > 0:
            score += 0.1
        else:
            if message.message_type in [EnhancedMessageType.RESEARCH_RESULT, EnhancedMessageType.ANALYSIS_RESULT]:
                score -= 0.1
                issues.append("Lacks supporting evidence or data references")
                recommendations.append("Include data sources or evidence to support claims")

        evidence["evidence_indicators"] = evidence_count

        # Check for consistency
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', content)
        if len(numbers) > 1:
            # Simple consistency check (would be more sophisticated in practice)
            score += 0.1  # Bonus for having multiple data points
        evidence["numeric_values"] = len(numbers)

        # Final score adjustment
        final_score = min(1.0, max(0.0, score))

        feedback = self._generate_accuracy_feedback(final_score, issues, evidence)

        return QualityMetric(
            dimension=QualityDimension.ACCURACY,
            score=final_score,
            weight=self.dimension_weights[QualityDimension.ACCURACY],
            feedback=feedback,
            issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    async def _assess_consistency_quality(self, message: RichMessage) -> QualityMetric:
        """Assess consistency quality dimension."""
        content = message.content
        score = 0.8  # Start with good default
        issues = []
        recommendations = []
        evidence = {}

        # Terminology consistency
        words = content.lower().split()
        word_variations = {
            "analyze": ["analysis", "analyzing", "analysed"],
            "implement": ["implementation", "implementing", "implemented"],
            "optimize": ["optimization", "optimizing", "optimized"],
        }

        consistency_score = 1.0
        for base_term, variations in word_variations.items():
            term_count = sum(1 for word in words if base_term in word)
            variation_count = sum(1 for word in words if any(var in word for var in variations))
            total_count = term_count + variation_count

            if total_count > 1:
                consistency_ratio = term_count / total_count
                consistency_score *= consistency_ratio

        score = consistency_score * 0.8 + 0.2  # Keep baseline

        if consistency_score < 0.7:
            issues.append("Inconsistent terminology usage")
            recommendations.append("Use consistent terminology throughout the content")

        evidence["terminology_consistency"] = consistency_score

        # Formatting consistency
        heading_styles = set()
        for match in re.finditer(r'^(#{1,6})\s+', content, re.MULTILINE):
            heading_styles.add(match.group(1))

        if len(heading_styles) > 1:
            score -= 0.1
            issues.append("Inconsistent heading formatting")
            recommendations.append("Use consistent heading formatting")

        evidence["heading_style_variations"] = len(heading_styles)

        # Tense consistency
        past_tense_indicators = ["was", "were", "had", "did", "been", "implemented", "completed"]
        present_tense_indicators = ["is", "are", "has", "have", "does", "do", "implement", "complete"]

        past_count = sum(1 for indicator in past_tense_indicators if indicator in content.lower())
        present_count = sum(1 for indicator in present_tense_indicators if indicator in content.lower())

        if past_count > 0 and present_count > 0:
            tense_ratio = min(past_count, present_count) / max(past_count, present_count)
            if tense_ratio < 0.3:  # Highly inconsistent
                score -= 0.1
                issues.append("Inconsistent tense usage")
                recommendations.append("Use consistent tense throughout the content")

        evidence["past_tense_count"] = past_count
        evidence["present_tense_count"] = present_count

        # Final score adjustment
        final_score = min(1.0, max(0.0, score))

        feedback = self._generate_consistency_feedback(final_score, issues, evidence)

        return QualityMetric(
            dimension=QualityDimension.CONSISTENCY,
            score=final_score,
            weight=self.dimension_weights[QualityDimension.CONSISTENCY],
            feedback=feedback,
            issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    async def _assess_accessibility_quality(self, message: RichMessage) -> QualityMetric:
        """Assess accessibility quality dimension."""
        content = message.content
        score = 0.0
        issues = []
        recommendations = []
        evidence = {}

        # Heading structure for screen readers
        has_headings = bool(re.search(r'^#{1,6}\s+', content, re.MULTILINE))
        if has_headings:
            score += 0.2
        else:
            issues.append("No headings found")
            recommendations.append("Add headings to improve navigation")

        evidence["has_headings"] = has_headings

        # Link accessibility
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        if links:
            descriptive_links = sum(1 for text, url in links if len(text.strip()) > 3 and text.lower() not in ["here", "link", "click"])
            link_ratio = descriptive_links / len(links)
            score += link_ratio * 0.2

            if link_ratio < 0.7:
                issues.append("Some links lack descriptive text")
                recommendations.append("Use descriptive link text instead of 'click here'")

            evidence["descriptive_link_ratio"] = link_ratio
        else:
            score += 0.2  # No links to assess

        # Image accessibility
        images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
        if images:
            images_with_alt = sum(1 for alt, url in images if alt.strip())
            alt_ratio = images_with_alt / len(images)
            score += alt_ratio * 0.2

            if alt_ratio < 0.8:
                issues.append("Some images lack alt text")
                recommendations.append("Add descriptive alt text for all images")

            evidence["alt_text_ratio"] = alt_ratio
        else:
            score += 0.2  # No images to assess

        # List accessibility
        has_lists = bool(re.search(r'^(\s*[-*+]|\d+\.)\s+', content, re.MULTILINE))
        if has_lists:
            score += 0.2
        evidence["has_lists"] = has_lists

        # Content structure
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.2
        evidence["paragraph_count"] = len(paragraphs)

        # Normalize score
        final_score = min(1.0, score)

        feedback = self._generate_accessibility_feedback(final_score, issues, evidence)

        return QualityMetric(
            dimension=QualityDimension.ACCESSIBILITY,
            score=final_score,
            weight=self.dimension_weights.get(QualityDimension.ACCESSIBILITY, 0.05),
            feedback=feedback,
            issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    # Helper methods for feedback generation
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        if score >= self.quality_thresholds["excellent"]:
            return "excellent"
        elif score >= self.quality_thresholds["good"]:
            return "good"
        elif score >= self.quality_thresholds["fair"]:
            return "fair"
        else:
            return "poor"

    def _analyze_strengths_weaknesses(self, metrics: Dict[QualityDimension, QualityMetric]) -> Tuple[List[str], List[str]]:
        """Analyze strengths and weaknesses from quality metrics."""
        strengths = []
        weaknesses = []

        for dimension, metric in metrics.items():
            if metric.score >= 0.8:
                strengths.append(f"Excellent {dimension.value} quality")
            elif metric.score >= 0.6:
                strengths.append(f"Good {dimension.value} quality")
            elif metric.score < 0.4:
                weaknesses.append(f"Poor {dimension.value} quality")
            else:
                weaknesses.append(f"Fair {dimension.value} quality - needs improvement")

        return strengths, weaknesses

    def _generate_recommendations(self, metrics: Dict[QualityDimension, QualityMetric], weaknesses: List[str]) -> List[str]:
        """Generate actionable recommendations based on quality assessment."""
        recommendations = []

        # Collect recommendations from metrics with low scores
        low_scoring_metrics = [dim for dim, metric in metrics.items() if metric.score < 0.6]

        for dimension in low_scoring_metrics:
            metric = metrics[dimension]
            recommendations.extend(metric.recommendations)

        # Remove duplicates and prioritize
        unique_recommendations = list(set(recommendations))

        # Sort by priority (based on dimension weight)
        prioritized_recommendations = sorted(
            unique_recommendations,
            key=lambda rec: self.dimension_weights.get(low_scoring_metrics[0], 0.1),
            reverse=True
        ) if low_scoring_metrics else unique_recommendations

        return prioritized_recommendations[:5]  # Return top 5 recommendations

    def _calculate_improvement_potential(self, metrics: Dict[QualityDimension, QualityMetric]) -> float:
        """Calculate potential for improvement."""
        current_scores = [metric.score for metric in metrics.values()]
        current_average = sum(current_scores) / len(current_scores)
        improvement_potential = 1.0 - current_average
        return improvement_potential

    # Feedback generation methods
    def _generate_content_feedback(self, score: float, issues: List[str], evidence: Dict[str, Any]) -> str:
        """Generate content quality feedback."""
        if score >= 0.8:
            return "Content is substantive and well-developed with appropriate length and depth."
        elif score >= 0.6:
            return "Content is generally good but could benefit from more detail or substance."
        else:
            return "Content needs significant improvement in length, substance, or detail."

    def _generate_structure_feedback(self, score: float, issues: List[str], evidence: Dict[str, Any]) -> str:
        """Generate structure quality feedback."""
        if score >= 0.8:
            return "Content is well-structured with proper headings, lists, and paragraph organization."
        elif score >= 0.6:
            return "Structure is generally good but could benefit from better organization."
        else:
            return "Content structure needs improvement in organization, headings, or formatting."

    def _generate_clarity_feedback(self, score: float, issues: List[str], evidence: Dict[str, Any]) -> str:
        """Generate clarity quality feedback."""
        if score >= 0.8:
            return "Content is clear and easy to understand with appropriate sentence structure and vocabulary."
        elif score >= 0.6:
            return "Content is generally clear but could benefit from improved readability."
        else:
            return "Content clarity needs improvement through better sentence structure and simpler language."

    def _generate_relevance_feedback(self, score: float, issues: List[str], evidence: Dict[str, Any]) -> str:
        """Generate relevance quality feedback."""
        if score >= 0.8:
            return "Content is highly relevant to the context and message type."
        elif score >= 0.6:
            return "Content is generally relevant but could be more focused."
        else:
            return "Content relevance needs improvement to better match the expected context."

    def _generate_completeness_feedback(self, score: float, issues: List[str], evidence: Dict[str, Any]) -> str:
        """Generate completeness quality feedback."""
        if score >= 0.8:
            return "Content is complete with all necessary elements and proper structure."
        elif score >= 0.6:
            return "Content is mostly complete but may need some additional elements."
        else:
            return "Content completeness needs improvement with missing key elements."

    def _generate_accuracy_feedback(self, score: float, issues: List[str], evidence: Dict[str, Any]) -> str:
        """Generate accuracy quality feedback."""
        if score >= 0.8:
            return "Content appears accurate with appropriate evidence and supporting data."
        elif score >= 0.6:
            return "Content seems generally accurate but could benefit from more supporting evidence."
        else:
            return "Content accuracy needs verification and additional supporting evidence."

    def _generate_consistency_feedback(self, score: float, issues: List[str], evidence: Dict[str, Any]) -> str:
        """Generate consistency quality feedback."""
        if score >= 0.8:
            return "Content is consistent in terminology, formatting, and style."
        elif score >= 0.6:
            return "Content is generally consistent with minor inconsistencies."
        else:
            return "Content consistency needs improvement in terminology or formatting."

    def _generate_accessibility_feedback(self, score: float, issues: List[str], evidence: Dict[str, Any]) -> str:
        """Generate accessibility quality feedback."""
        if score >= 0.8:
            return "Content is well-structured for accessibility with proper headings and descriptions."
        elif score >= 0.6:
            return "Content has some accessibility features but could be improved."
        else:
            return "Content accessibility needs significant improvement."

    def _metric_to_dict(self, metric: QualityMetric) -> Dict[str, Any]:
        """Convert quality metric to dictionary."""
        return {
            "score": metric.score,
            "weight": metric.weight,
            "feedback": metric.feedback,
            "issues": metric.issues,
            "recommendations": metric.recommendations,
            "evidence": metric.evidence
        }

    def _update_assessment_stats(self, message: RichMessage, assessment: QualityAssessment):
        """Update assessment statistics."""
        self.assessment_stats["total_assessments"] += 1
        self.assessment_stats["processing_time"] += assessment.processing_time

        # Update average score
        current_avg = self.assessment_stats["average_score"]
        count = self.assessment_stats["total_assessments"]
        self.assessment_stats["average_score"] = ((current_avg * (count - 1)) + assessment.overall_score) / count

        # Update by type
        msg_type = message.message_type.value
        if msg_type not in self.assessment_stats["assessments_by_type"]:
            self.assessment_stats["assessments_by_type"][msg_type] = {"count": 0, "avg_score": 0.0}

        type_stats = self.assessment_stats["assessments_by_type"][msg_type]
        type_stats["count"] += 1
        type_stats["avg_score"] = ((type_stats["avg_score"] * (type_stats["count"] - 1)) + assessment.overall_score) / type_stats["count"]

        # Update by level
        level = assessment.quality_level
        if level not in self.assessment_stats["assessments_by_level"]:
            self.assessment_stats["assessments_by_level"][level] = 0
        self.assessment_stats["assessments_by_level"][level] += 1

    def get_assessment_stats(self) -> Dict[str, Any]:
        """Get comprehensive assessment statistics."""
        stats = self.assessment_stats.copy()

        # Calculate additional metrics
        if stats["total_assessments"] > 0:
            stats["average_processing_time"] = stats["processing_time"] / stats["total_assessments"]
        else:
            stats["average_processing_time"] = 0.0

        return stats

    def reset_stats(self):
        """Reset assessment statistics."""
        self.assessment_stats = {
            "total_assessments": 0,
            "average_score": 0.0,
            "assessments_by_type": {},
            "assessments_by_level": {},
            "processing_time": 0.0
        }