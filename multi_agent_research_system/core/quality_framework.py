"""
Comprehensive Quality Assessment Framework for Multi-Agent Research System.

This module provides a unified quality assessment system that can evaluate content
across multiple dimensions, providing detailed feedback and recommendations for improvement.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


@dataclass
class CriterionResult:
    """Result for a single quality criterion evaluation."""
    name: str
    score: int  # 0-100
    weight: float
    feedback: str
    specific_issues: list[str]
    recommendations: list[str]
    evidence: dict[str, Any]


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment results."""
    overall_score: int
    quality_level: QualityLevel
    criteria_results: dict[str, CriterionResult]
    content_metadata: dict[str, Any]
    assessment_timestamp: str
    strengths: list[str]
    weaknesses: list[str]
    actionable_recommendations: list[str]
    enhancement_priority: list[tuple[str, int]]  # (criterion_name, priority_score)
    recommendations: list[str]  # MISSING FIELD - ADDING TO FIX ATTRIBUTE ERROR

    def get_criterion_score(self, criterion_name: str) -> int:
        """Get the score for a specific criterion."""
        if criterion_name in self.criteria_results:
            return self.criteria_results[criterion_name].score
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Convert assessment to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "quality_level": self.quality_level.value,
            "criteria_results": {
                name: {
                    "score": result.score,
                    "weight": result.weight,
                    "feedback": result.feedback,
                    "specific_issues": result.specific_issues,
                    "recommendations": result.recommendations,
                    "evidence": result.evidence
                }
                for name, result in self.criteria_results.items()
            },
            "content_metadata": self.content_metadata,
            "assessment_timestamp": self.assessment_timestamp,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "actionable_recommendations": self.actionable_recommendations,
            "enhancement_priority": self.enhancement_priority,
            "recommendations": self.recommendations  # ADD MISSING FIELD
        }


class BaseQualityCriterion:
    """Base class for quality assessment criteria."""

    def __init__(self, name: str, weight: float = 0.1, description: str = ""):
        self.name = name
        self.weight = weight
        self.description = description

    async def evaluate(self, content: str, context: dict[str, Any] | None = None) -> CriterionResult:
        """
        Evaluate content against this criterion.

        Args:
            content: Content to evaluate
            context: Additional context for evaluation

        Returns:
            CriterionResult with detailed evaluation
        """
        raise NotImplementedError("Subclasses must implement evaluate method")


class RelevanceCriterion(BaseQualityCriterion):
    """Evaluates content relevance to the intended topic."""

    def __init__(self, weight: float = 0.25):
        super().__init__("relevance", weight, "Assesses content relevance to the topic")

    async def evaluate(self, content: str, context: dict[str, Any] | None = None) -> CriterionResult:
        """Evaluate content relevance."""
        score = 75  # Base score
        issues = []
        recommendations = []
        evidence = {}

        # Extract topic from context
        topic = context.get("topic", "") if context else ""
        topic_words = set(topic.lower().split()) if topic else set()

        # Check for topic consistency
        content_lower = content.lower()
        topic_mentions = sum(1 for word in topic_words if word in content_lower and len(word) > 3)

        if topic_words:
            relevance_ratio = topic_mentions / len(topic_words) if topic_words else 0
            evidence["topic_words_found"] = topic_mentions
            evidence["topic_words_total"] = len(topic_words)
            evidence["relevance_ratio"] = relevance_ratio

            if relevance_ratio >= 0.7:
                score += 20
            elif relevance_ratio >= 0.5:
                score += 10
            elif relevance_ratio < 0.3:
                score -= 20
                issues.append(f"Low topic relevance: only {topic_mentions}/{len(topic_words)} topic terms found")
                recommendations.append("Increase focus on the main topic throughout the content")

        # Check for consistent thematic focus
        sentences = content.split('. ')
        substantive_sentences = [s for s in sentences if len(s.strip()) > 20]

        if substantive_sentences:
            # Simple consistency check based on sentence patterns
            consistent_sentences = 0
            for sentence in substantive_sentences[:10]:  # Check first 10 sentences
                if any(word in sentence.lower() for word in topic_words):
                    consistent_sentences += 1

            consistency_ratio = consistent_sentences / min(len(substantive_sentences), 10)
            evidence["thematic_consistency"] = consistency_ratio

            if consistency_ratio >= 0.6:
                score += 10
            elif consistency_ratio < 0.3:
                score -= 15
                issues.append("Inconsistent thematic focus detected")
                recommendations.append("Maintain consistent focus on the main topic")

        # Check for off-topic content indicators
        off_topic_patterns = [
            r'unrelated to the topic',
            r'completely different subject',
            r'changing the subject',
            r'talking about something else'
        ]

        off_topic_count = sum(len(re.findall(pattern, content_lower)) for pattern in off_topic_patterns)
        if off_topic_count > 0:
            score -= min(10, off_topic_count * 2)
            issues.append(f"Found {off_topic_count} indicators of off-topic content")

        score = max(0, min(100, score))

        feedback = self._generate_feedback(score, relevance_ratio if topic_words else 0, issues)

        return CriterionResult(
            name=self.name,
            score=score,
            weight=self.weight,
            feedback=feedback,
            specific_issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    def _generate_feedback(self, score: int, relevance_ratio: float, issues: list[str]) -> str:
        """Generate detailed feedback for relevance evaluation."""
        if score >= 85:
            return "Excellent relevance with strong topic focus throughout the content."
        elif score >= 70:
            return "Good relevance with consistent thematic focus."
        elif score >= 60:
            return "Acceptable relevance but could benefit from stronger topic consistency."
        elif score >= 50:
            return "Needs improvement - topic focus is inconsistent or weak."
        else:
            return "Poor relevance - content appears to be off-topic or lacks focus."


class CompletenessCriterion(BaseQualityCriterion):
    """Evaluates content completeness and structural coverage."""

    def __init__(self, weight: float = 0.20):
        super().__init__("completeness", weight, "Assesses content completeness and structure")

    async def evaluate(self, content: str, context: dict[str, Any] | None = None) -> CriterionResult:
        """Evaluate content completeness."""
        score = 70  # Base score
        issues = []
        recommendations = []
        evidence = {}

        # Check for structural elements
        structural_elements = {
            "introduction": any(term in content.lower() for term in [
                "introduction", "overview", "background", "summary", "executive summary"
            ]),
            "main_content": len([line for line in content.split('\n') if len(line.strip()) > 50]) > 5,
            "conclusion": any(term in content.lower() for term in [
                "conclusion", "summary", "final", "in conclusion", "wrap up"
            ]),
            "headings": bool(re.search(r'^#+\s+', content, re.MULTILINE)),
            "paragraphs": content.count('\n\n') >= 2
        }

        evidence["structural_elements"] = structural_elements

        # Score based on structural completeness
        completeness_score = sum(structural_elements.values()) * 6
        score += completeness_score

        # Check content depth indicators
        word_count = len(content.split())
        evidence["word_count"] = word_count

        if word_count >= 800:
            score += 15
        elif word_count >= 500:
            score += 10
        elif word_count >= 300:
            score += 5
        elif word_count < 200:
            score -= 15
            issues.append("Content is too short for comprehensive coverage")
            recommendations.append("Expand content with more detailed analysis and examples")

        # Check for analytical elements
        analytical_indicators = [
            "analysis", "because", "therefore", "however", "consequently",
            "furthermore", "moreover", "in contrast", "similarly", "for example"
        ]

        analytical_count = sum(content.lower().count(indicator) for indicator in analytical_indicators)
        evidence["analytical_indicators"] = analytical_count

        if analytical_count >= 10:
            score += 10
        elif analytical_count >= 5:
            score += 5
        elif analytical_count < 3:
            score -= 10
            issues.append("Limited analytical content - content may lack depth")
            recommendations.append("Add more analytical language and reasoning")

        # Check for supporting evidence
        evidence_patterns = [
            r'\d+%',  # Statistics
            r'according to',  # Citations
            r'research shows',  # Research references
            r'example[s]?',  # Examples
            r'data',  # Data references
        ]

        evidence_count = sum(len(re.findall(pattern, content.lower())) for pattern in evidence_patterns)
        evidence["supporting_evidence"] = evidence_count

        if evidence_count >= 5:
            score += 10
        elif evidence_count >= 2:
            score += 5
        elif evidence_count == 0:
            score -= 5
            issues.append("No supporting evidence or examples found")
            recommendations.append("Add supporting evidence, examples, or data")

        # Identify missing elements
        missing_elements = [elem for elem, present in structural_elements.items() if not present]
        if missing_elements:
            issues.extend([f"Missing {elem}" for elem in missing_elements])
            recommendations.extend([f"Add {elem} to improve structure" for elem in missing_elements])

        score = max(0, min(100, score))

        feedback = self._generate_feedback(score, word_count, missing_elements)

        return CriterionResult(
            name=self.name,
            score=score,
            weight=self.weight,
            feedback=feedback,
            specific_issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    def _generate_feedback(self, score: int, word_count: int, missing_elements: list[str]) -> str:
        """Generate detailed feedback for completeness evaluation."""
        if score >= 85:
            return "Comprehensive content with excellent structure and depth."
        elif score >= 70:
            return "Good completeness with adequate structure and coverage."
        elif score >= 60:
            return "Acceptable completeness but could benefit from additional depth or structure."
        elif score >= 50:
            return "Needs improvement - content lacks some structural elements or depth."
        else:
            return f"Poor completeness - missing {', '.join(missing_elements)} and insufficient depth."


class AccuracyCriterion(BaseQualityCriterion):
    """Evaluates content accuracy and factual reliability."""

    def __init__(self, weight: float = 0.20):
        super().__init__("accuracy", weight, "Assesses content accuracy and factual reliability")

    async def evaluate(self, content: str, context: dict[str, Any] | None = None) -> CriterionResult:
        """Evaluate content accuracy."""
        score = 80  # Base score assuming reasonable accuracy
        issues = []
        recommendations = []
        evidence = {}

        # Note: This is a simplified accuracy check
        # In practice, this would require external fact-checking capabilities

        # Check for citation patterns (indicating verified information)
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, Year) format
            r'\[\d+\]',  # [1] citation format
            r'according to[^.]*',  # "according to" references
            r'research[^.]*shows',  # Research references
            r'stud(y|ies)[^.]*found',  # Study references
        ]

        citation_count = sum(len(re.findall(pattern, content.lower())) for pattern in citation_patterns)
        evidence["citations_found"] = citation_count

        if citation_count >= 5:
            score += 15
        elif citation_count >= 2:
            score += 8
        elif citation_count == 0:
            score -= 10
            issues.append("No citations or references found - information may be unsubstantiated")
            recommendations.append("Add citations or references to support claims")

        # Check for speculative language (potential accuracy concerns)
        speculative_patterns = [
            r'\b(might|could|perhaps|maybe|possibly)\b',
            r'\b(apparently|seems|appears)\b',
            r'\b(presumably|arguably|theoretically)\b'
        ]

        speculative_count = sum(len(re.findall(pattern, content.lower())) for pattern in speculative_patterns)
        evidence["speculative_language"] = speculative_count

        # Too much speculative language may indicate uncertainty
        if speculative_count > 10:
            score -= 10
            issues.append("High amount of speculative language may reduce content reliability")
            recommendations.append("Replace some speculative language with more definitive statements where possible")

        # Check for contradictory statements
        sentences = content.split('. ')
        contradictions = 0

        # Simple contradiction detection (basic implementation)
        for i, sentence in enumerate(sentences):
            if 'however' in sentence.lower() and i > 0:
                # Look for potential contradictions with previous sentences
                contradictions += 1

        evidence["potential_contradictions"] = contradictions
        if contradictions > 2:
            score -= 5
            issues.append("Potential contradictory statements detected")
            recommendations.append("Review content for consistency and resolve contradictions")

        # Check for outdated information indicators
        outdated_patterns = [
            r'recently',  # May become outdated
            r'current',   # May become outdated
            r'latest',    # May become outdated
            r'\d{4}s',    # Decade references that may be dated
        ]

        outdated_count = sum(len(re.findall(pattern, content.lower())) for pattern in outdated_patterns)
        evidence["temporal_references"] = outdated_count

        score = max(0, min(100, score))

        feedback = self._generate_feedback(score, citation_count, speculative_count)

        return CriterionResult(
            name=self.name,
            score=score,
            weight=self.weight,
            feedback=feedback,
            specific_issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    def _generate_feedback(self, score: int, citations: int, speculative: int) -> str:
        """Generate detailed feedback for accuracy evaluation."""
        if score >= 90:
            return "Excellent accuracy with strong citation support and minimal speculation."
        elif score >= 80:
            return "Good accuracy with adequate supporting evidence."
        elif score >= 70:
            return "Acceptable accuracy but could benefit from more citations or less speculation."
        elif score >= 60:
            return "Needs improvement - limited supporting evidence or excessive speculation."
        else:
            return "Poor accuracy - lacks sufficient evidence and contains too much speculation."


class ClarityCriterion(BaseQualityCriterion):
    """Evaluates content clarity and readability."""

    def __init__(self, weight: float = 0.15):
        super().__init__("clarity", weight, "Assesses content clarity and readability")

    async def evaluate(self, content: str, context: dict[str, Any] | None = None) -> CriterionResult:
        """Evaluate content clarity."""
        score = 75  # Base score
        issues = []
        recommendations = []
        evidence = {}

        # Calculate readability metrics
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        words = content.split()

        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            evidence["avg_sentence_length"] = avg_sentence_length

            # Optimal sentence length is 15-20 words
            if 15 <= avg_sentence_length <= 25:
                score += 15
            elif 10 <= avg_sentence_length <= 30:
                score += 5
            elif avg_sentence_length > 35:
                score -= 15
                issues.append("Sentences are too long - reduces readability")
                recommendations.append("Break up long sentences for better clarity")
            elif avg_sentence_length < 10:
                score -= 10
                issues.append("Sentences are too short - may feel choppy")
                recommendations.append("Combine short sentences for better flow")

        # Check paragraph structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if paragraphs:
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            evidence["avg_paragraph_length"] = avg_paragraph_length

            # Optimal paragraph length is 50-150 words
            if 50 <= avg_paragraph_length <= 150:
                score += 10
            elif avg_paragraph_length > 200:
                score -= 10
                issues.append("Paragraphs are too long - may be difficult to read")
                recommendations.append("Break up long paragraphs for easier reading")
            elif avg_paragraph_length < 30:
                score -= 5
                issues.append("Paragraphs are very short - may lack substance")
                recommendations.append("Combine related short paragraphs")

        # Check for clear structure
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        evidence["heading_count"] = len(headings)

        if len(headings) >= 3:
            score += 10
        elif len(headings) >= 1:
            score += 5
        else:
            score -= 10
            issues.append("No headings found - content structure is unclear")
            recommendations.append("Add headings to organize content and improve readability")

        # Check for clarity issues
        clarity_issues = 0

        # Check for passive voice (simple pattern)
        passive_patterns = [r'is \w+ed', r'are \w+ed', r'was \w+ed', r'were \w+ed']
        passive_count = sum(len(re.findall(pattern, content.lower())) for pattern in passive_patterns)
        evidence["passive_voice_count"] = passive_count

        if passive_count > len(sentences) * 0.3:  # More than 30% passive
            clarity_issues += 1
            issues.append("High use of passive voice may reduce clarity")
            recommendations.append("Consider using more active voice for clearer communication")

        # Check for jargon and complex terms
        jargon_patterns = [
            r'\b\w{10,}\b',  # Long words
            r'[A-Z]{2,}',    # Acronyms
        ]

        jargon_count = sum(len(re.findall(pattern, content)) for pattern in jargon_patterns)
        evidence["jargon_count"] = jargon_count

        if jargon_count > len(words) * 0.1:  # More than 10% jargon
            clarity_issues += 1
            issues.append("High amount of jargon or complex terminology")
            recommendations.append("Consider defining technical terms or using simpler language")

        # Check for sentence variety
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            length_variance = max(sentence_lengths) - min(sentence_lengths)
            evidence["sentence_length_variance"] = length_variance

            if length_variance < 5:
                score -= 5
                issues.append("Limited sentence variety - may seem monotonous")
                recommendations.append("Vary sentence length for better flow")

        score = max(0, min(100, score))

        feedback = self._generate_feedback(score, avg_sentence_length if sentences else 0, clarity_issues)

        return CriterionResult(
            name=self.name,
            score=score,
            weight=self.weight,
            feedback=feedback,
            specific_issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    def _generate_feedback(self, score: int, avg_sentence_length: float, clarity_issues: int) -> str:
        """Generate detailed feedback for clarity evaluation."""
        if score >= 85:
            return "Excellent clarity with well-structured sentences and paragraphs."
        elif score >= 75:
            return "Good clarity with readable structure and flow."
        elif score >= 65:
            return "Acceptable clarity but could benefit from structural improvements."
        elif score >= 55:
            return "Needs improvement - some clarity issues affect readability."
        else:
            return f"Poor clarity - multiple issues ({clarity_issues}) need to be addressed."


class DepthCriterion(BaseQualityCriterion):
    """Evaluates content depth and analytical quality."""

    def __init__(self, weight: float = 0.10):
        super().__init__("depth", weight, "Assesses content depth and analytical quality")

    async def evaluate(self, content: str, context: dict[str, Any] | None = None) -> CriterionResult:
        """Evaluate content depth."""
        score = 70  # Base score
        issues = []
        recommendations = []
        evidence = {}

        # Check for analytical depth
        analytical_terms = [
            "analysis", "analyze", "examine", "evaluate", "assess", "consider",
            "because", "therefore", "however", "consequently", "furthermore",
            "moreover", "in contrast", "similarly", "specifically", "particularly"
        ]

        analytical_count = sum(content.lower().count(term) for term in analytical_terms)
        evidence["analytical_terms"] = analytical_count

        if analytical_count >= 15:
            score += 20
        elif analytical_count >= 10:
            score += 15
        elif analytical_count >= 5:
            score += 10
        elif analytical_count < 3:
            score -= 15
            issues.append("Limited analytical language - content may be superficial")
            recommendations.append("Add more analytical terms and deeper reasoning")

        # Check for examples and illustrations
        example_patterns = [
            r'for example',
            r'for instance',
            r'such as',
            r'including',
            r'like',
            r'e\.g\.'
        ]

        example_count = sum(len(re.findall(pattern, content.lower())) for pattern in example_patterns)
        evidence["examples_count"] = example_count

        if example_count >= 5:
            score += 10
        elif example_count >= 2:
            score += 5
        elif example_count == 0:
            score -= 10
            issues.append("No examples or illustrations found")
            recommendations.append("Add examples to illustrate key points")

        # Check for detailed explanations
        detailed_sentences = 0
        sentences = content.split('. ')

        for sentence in sentences:
            word_count = len(sentence.split())
            if word_count > 25:  # Longer sentences often indicate detail
                detailed_sentences += 1

        detail_ratio = detailed_sentences / len(sentences) if sentences else 0
        evidence["detailed_sentence_ratio"] = detail_ratio

        if detail_ratio >= 0.3:
            score += 10
        elif detail_ratio >= 0.2:
            score += 5
        elif detail_ratio < 0.1:
            score -= 10
            issues.append("Lack of detailed explanations")
            recommendations.append("Provide more detailed explanations for complex points")

        # Check for multi-faceted analysis
        perspective_indicators = [
            "from one perspective", "alternatively", "on the other hand",
            "another view", "different approach", "various factors"
        ]

        perspective_count = sum(content.lower().count(indicator) for indicator in perspective_indicators)
        evidence["perspective_indicators"] = perspective_count

        if perspective_count >= 3:
            score += 10
        elif perspective_count >= 1:
            score += 5

        # Check for cause-effect reasoning
        cause_effect_patterns = [
            r'because',
            r'therefore',
            r'consequently',
            r'as a result',
            r'leads to',
            r'results in'
        ]

        cause_effect_count = sum(len(re.findall(pattern, content.lower())) for pattern in cause_effect_patterns)
        evidence["cause_effect_indicators"] = cause_effect_count

        if cause_effect_count >= 5:
            score += 10
        elif cause_effect_count >= 2:
            score += 5
        elif cause_effect_count == 0:
            score -= 5
            issues.append("Limited cause-effect reasoning")
            recommendations.append("Include more cause-effect relationships")

        score = max(0, min(100, score))

        feedback = self._generate_feedback(score, analytical_count, example_count)

        return CriterionResult(
            name=self.name,
            score=score,
            weight=self.weight,
            feedback=feedback,
            specific_issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    def _generate_feedback(self, score: int, analytical_count: int, example_count: int) -> str:
        """Generate detailed feedback for depth evaluation."""
        if score >= 85:
            return "Excellent depth with strong analytical content and detailed explanations."
        elif score >= 75:
            return "Good depth with adequate analysis and examples."
        elif score >= 65:
            return "Acceptable depth but could benefit from more detailed analysis."
        elif score >= 55:
            return "Needs improvement - analysis is somewhat superficial."
        else:
            return "Poor depth - lacks analytical content and detailed explanations."


class OrganizationCriterion(BaseQualityCriterion):
    """Evaluates content organization and structure."""

    def __init__(self, weight: float = 0.10):
        super().__init__("organization", weight, "Assesses content organization and structure")

    async def evaluate(self, content: str, context: dict[str, Any] | None = None) -> CriterionResult:
        """Evaluate content organization."""
        score = 75  # Base score
        issues = []
        recommendations = []
        evidence = {}

        # Check heading structure
        headings = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
        heading_levels = [len(h[0]) for h in headings]
        evidence["headings"] = {
            "total": len(headings),
            "levels": heading_levels,
            "h1_count": heading_levels.count(1),
            "h2_count": heading_levels.count(2),
            "h3_count": heading_levels.count(3)
        }

        # Check for logical heading hierarchy
        if heading_levels:
            # Should have at least one H1
            if 1 in heading_levels:
                score += 10
            else:
                score -= 10
                issues.append("No main title (H1) found")
                recommendations.append("Add a main title using # heading")

            # Should have H2 headings for structure
            if 2 in heading_levels:
                score += 10
            else:
                score -= 5
                issues.append("No section headings (H2) found")
                recommendations.append("Add section headings using ##")

            # Check heading level progression (shouldn't skip levels)
            for i in range(1, len(heading_levels)):
                if heading_levels[i] - heading_levels[i-1] > 1:
                    score -= 5
                    issues.append("Heading levels skip (e.g., H1 to H3)")
                    recommendations.append("Use proper heading hierarchy without skipping levels")
                    break

        # Check for logical flow indicators
        flow_indicators = [
            "first", "second", "third", "finally", "next", "then",
            "in addition", "furthermore", "moreover", "also",
            "however", "in contrast", "on the other hand",
            "therefore", "consequently", "as a result",
            "in conclusion", "to summarize", "in summary"
        ]

        flow_count = sum(content.lower().count(indicator) for indicator in flow_indicators)
        evidence["flow_indicators"] = flow_count

        if flow_count >= 8:
            score += 15
        elif flow_count >= 4:
            score += 10
        elif flow_count >= 2:
            score += 5
        elif flow_count == 0:
            score -= 10
            issues.append("No logical flow indicators found")
            recommendations.append("Add transition words to improve flow")

        # Check paragraph structure
        paragraphs = content.split('\n\n')
        evidence["paragraph_count"] = len(paragraphs)

        # Check for consistent paragraph length
        if len(paragraphs) > 1:
            paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]
            if paragraph_lengths:
                avg_length = sum(paragraph_lengths) / len(paragraph_lengths)
                evidence["avg_paragraph_length"] = avg_length

                # Check for very short or very long paragraphs
                very_short = sum(1 for length in paragraph_lengths if length < 20)
                very_long = sum(1 for length in paragraph_lengths if length > 200)

                evidence["very_short_paragraphs"] = very_short
                evidence["very_long_paragraphs"] = very_long

                if very_short > len(paragraphs) * 0.3:
                    score -= 10
                    issues.append("Many very short paragraphs")
                    recommendations.append("Combine related short paragraphs")

                if very_long > len(paragraphs) * 0.2:
                    score -= 10
                    issues.append("Some paragraphs are too long")
                    recommendations.append("Break up long paragraphs for better readability")

        # Check for introduction and conclusion
        content_lower = content.lower()
        has_intro = any(term in content_lower for term in [
            "introduction", "overview", "background", "summary"
        ])
        has_conclusion = any(term in content_lower for term in [
            "conclusion", "summary", "final", "in conclusion"
        ])

        evidence["has_introduction"] = has_intro
        evidence["has_conclusion"] = has_conclusion

        if has_intro and has_conclusion:
            score += 10
        elif has_intro or has_conclusion:
            score += 5
        else:
            score -= 15
            issues.append("Missing clear introduction or conclusion")
            recommendations.append("Add introduction and conclusion for better structure")

        # Check for logical grouping
        # Simple heuristic: check if related content is grouped together
        sections = re.split(r'^##\s+', content, flags=re.MULTILINE)[1:]  # Split by H2 headings
        if len(sections) >= 2:
            evidence["sections_found"] = len(sections)
            score += 5  # Bonus for having multiple sections

        score = max(0, min(100, score))

        feedback = self._generate_feedback(score, len(headings), flow_count)

        return CriterionResult(
            name=self.name,
            score=score,
            weight=self.weight,
            feedback=feedback,
            specific_issues=issues,
            recommendations=recommendations,
            evidence=evidence
        )

    def _generate_feedback(self, score: int, heading_count: int, flow_count: int) -> str:
        """Generate detailed feedback for organization evaluation."""
        if score >= 85:
            return "Excellent organization with clear structure and logical flow."
        elif score >= 75:
            return "Good organization with adequate structure and flow indicators."
        elif score >= 65:
            return "Acceptable organization but could benefit from better structure."
        elif score >= 55:
            return "Needs improvement - organization issues affect readability."
        else:
            return "Poor organization - lacks clear structure and logical flow."


class QualityFramework:
    """
    Comprehensive quality assessment framework for multi-agent research system.

    Provides unified quality assessment across multiple dimensions with detailed
    feedback and actionable recommendations.
    """

    def __init__(self, custom_criteria: list[BaseQualityCriterion] | None = None):
        """
        Initialize the quality framework.

        Args:
            custom_criteria: Optional custom quality criteria
        """
        self.logger = logging.getLogger(__name__)

        # Default quality criteria
        self.criteria = {
            "relevance": RelevanceCriterion(),
            "completeness": CompletenessCriterion(),
            "accuracy": AccuracyCriterion(),
            "clarity": ClarityCriterion(),
            "depth": DepthCriterion(),
            "organization": OrganizationCriterion()
        }

        # Add custom criteria if provided
        if custom_criteria:
            for criterion in custom_criteria:
                self.criteria[criterion.name] = criterion

        self.logger.info(f"QualityFramework initialized with {len(self.criteria)} criteria")

    async def assess_quality(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        criteria_weights: dict[str, float] | None = None
    ) -> QualityAssessment:
        """
        Assess content quality across all criteria.

        Args:
            content: Content to assess
            context: Additional context for assessment
            criteria_weights: Optional custom weights for criteria

        Returns:
            Comprehensive QualityAssessment
        """
        self.logger.info(f"Starting quality assessment for {len(content)} characters of content")

        assessment_start = datetime.now()

        # Apply custom weights if provided
        if criteria_weights:
            for criterion_name, weight in criteria_weights.items():
                if criterion_name in self.criteria:
                    self.criteria[criterion_name].weight = weight

        criteria_results = {}
        strengths = []
        weaknesses = []
        all_recommendations = []

        # Evaluate each criterion
        for criterion_name, criterion in self.criteria.items():
            try:
                result = await criterion.evaluate(content, context)
                criteria_results[criterion_name] = result

                # Collect strengths and weaknesses
                if result.score >= 80:
                    strengths.append(f"{criterion_name.title()}: {result.feedback}")
                elif result.score < 60:
                    weaknesses.append(f"{criterion_name.title()}: {result.feedback}")

                # Collect recommendations
                all_recommendations.extend(result.recommendations)

                self.logger.debug(f"Criterion {criterion_name}: {result.score}/100")

            except Exception as e:
                self.logger.error(f"Error evaluating criterion {criterion_name}: {e}")
                # Create a default failed result
                criteria_results[criterion_name] = CriterionResult(
                    name=criterion_name,
                    score=0,
                    weight=criterion.weight,
                    feedback=f"Evaluation failed: {str(e)}",
                    specific_issues=[f"Evaluation error: {str(e)}"],
                    recommendations=["Retry evaluation"],
                    evidence={"error": str(e)}
                )

        # Calculate weighted overall score
        overall_score = 0
        total_weight = 0

        for criterion_name, result in criteria_results.items():
            weight = result.weight
            overall_score += result.score * weight
            total_weight += weight

        if total_weight > 0:
            overall_score = overall_score / total_weight
        else:
            overall_score = 0

        overall_score = int(overall_score)

        # Determine quality level
        quality_level = self.determine_quality_level(overall_score)

        # Remove duplicate recommendations and prioritize
        unique_recommendations = list(set(all_recommendations))
        prioritized_recommendations = self.prioritize_recommendations(
            unique_recommendations, criteria_results
        )

        # Create enhancement priority list
        enhancement_priority = self.create_enhancement_priority(criteria_results)

        # Generate content metadata
        content_metadata = self.generate_content_metadata(content, context)

        assessment_duration = (datetime.now() - assessment_start).total_seconds()

        self.logger.info(f"Quality assessment completed in {assessment_duration:.2f}s: {overall_score}/100 ({quality_level.value})")

        return QualityAssessment(
            overall_score=overall_score,
            quality_level=quality_level,
            criteria_results=criteria_results,
            content_metadata=content_metadata,
            assessment_timestamp=datetime.now().isoformat(),
            strengths=strengths,
            weaknesses=weaknesses,
            actionable_recommendations=prioritized_recommendations,
            enhancement_priority=enhancement_priority,
            recommendations=prioritized_recommendations  # Add missing field
        )

    def determine_quality_level(self, score: int) -> QualityLevel:
        """Determine quality level based on score."""
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

    def prioritize_recommendations(
        self,
        recommendations: list[str],
        criteria_results: dict[str, CriterionResult]
    ) -> list[str]:
        """Prioritize recommendations based on impact and difficulty."""
        # Simple prioritization based on which criteria need the most improvement
        priority_scores = {}

        for criterion_name, result in criteria_results.items():
            if result.score < 70:  # Only consider criteria that need improvement
                priority_scores[criterion_name] = 100 - result.score

        # Sort recommendations by priority
        prioritized = []
        for rec in recommendations:
            # Assign priority based on related criterion scores
            priority = 50  # Default priority

            for criterion_name, priority_score in priority_scores.items():
                if any(keyword in rec.lower() for keyword in [criterion_name, criterion_name.replace("_", " ")]):
                    priority = max(priority, priority_score)

            prioritized.append((rec, priority))

        # Sort by priority (descending) and extract recommendations
        prioritized.sort(key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in prioritized]

    def create_enhancement_priority(
        self,
        criteria_results: dict[str, CriterionResult]
    ) -> list[tuple[str, int]]:
        """Create prioritized list of criteria for enhancement."""
        priorities = []

        for criterion_name, result in criteria_results.items():
            if result.score < 85:  # Only include criteria that can be improved
                priority_score = 100 - result.score
                priorities.append((criterion_name, priority_score))

        # Sort by priority (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities

    def generate_content_metadata(
        self,
        content: str,
        context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Generate metadata about the assessed content."""
        return {
            "content_length": len(content),
            "word_count": len(content.split()),
            "sentence_count": len([s for s in content.split('.') if s.strip()]),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            "has_context": context is not None,
            "assessment_version": "1.0",
            "assessment_timestamp": datetime.now().isoformat()
        }

    def get_criterion_summary(self, assessment: QualityAssessment) -> dict[str, Any]:
        """Get a summary of criterion performance."""
        summary = {}

        for criterion_name, result in assessment.criteria_results.items():
            summary[criterion_name] = {
                "score": result.score,
                "weight": result.weight,
                "weighted_score": result.score * result.weight,
                "feedback": result.feedback,
                "priority": "high" if result.score < 60 else "medium" if result.score < 80 else "low"
            }

        return summary

    def compare_assessments(
        self,
        original_assessment: QualityAssessment,
        improved_assessment: QualityAssessment
    ) -> dict[str, Any]:
        """Compare two quality assessments to show improvements."""
        comparison = {
            "overall_improvement": improved_assessment.overall_score - original_assessment.overall_score,
            "quality_level_change": {
                "from": original_assessment.quality_level.value,
                "to": improved_assessment.quality_level.value
            },
            "criterion_improvements": {},
            "new_strengths": [],
            "resolved_weaknesses": [],
            "new_weaknesses": []
        }

        # Compare criterion scores
        for criterion_name in original_assessment.criteria_results:
            if criterion_name in improved_assessment.criteria_results:
                original_score = original_assessment.criteria_results[criterion_name].score
                improved_score = improved_assessment.criteria_results[criterion_name].score
                improvement = improved_score - original_score

                comparison["criterion_improvements"][criterion_name] = {
                    "original_score": original_score,
                    "improved_score": improved_score,
                    "improvement": improvement
                }

        # Identify new strengths and resolved weaknesses
        original_weaknesses = set(original_assessment.weaknesses)
        improved_weaknesses = set(improved_assessment.weaknesses)

        comparison["resolved_weaknesses"] = list(original_weaknesses - improved_weaknesses)
        comparison["new_weaknesses"] = list(improved_weaknesses - original_weaknesses)

        return comparison


# Convenience function for quick quality assessment
async def assess_content_quality(
    content: str,
    context: dict[str, Any] | None = None,
    custom_criteria: list[BaseQualityCriterion] | None = None
) -> QualityAssessment:
    """
    Quick quality assessment function.

    Args:
        content: Content to assess
        context: Additional context for assessment
        custom_criteria: Optional custom criteria

    Returns:
        QualityAssessment results
    """
    framework = QualityFramework(custom_criteria)
    return await framework.assess_quality(content, context)
