"""
Content Quality Judge System

Implements AI-powered content quality assessment with judge scoring and feedback loops
as specified in the technical documentation.

Features:
- Judge assessment scoring (0-100)
- Content quality criteria evaluation
- Feedback generation for cleaning optimization
- Performance tracking and analytics
- Structured quality assessment reports
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Pydantic AI imports
try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.openai import OpenAIModel
    PYDAI_AVAILABLE = True
except ImportError:
    logging.warning("Pydantic AI not available - using fallback quality assessment")
    PYDAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class QualityCriterion(Enum):
    """Content quality assessment criteria."""
    RELEVANCE = "relevance"           # Relevance to search query
    COMPLETENESS = "completeness"     # Information completeness
    ACCURACY = "accuracy"            # Factual accuracy indicators
    CLARITY = "clarity"              # Readability and clarity
    DEPTH = "depth"                 # Depth of information
    ORGANIZATION = "organization"     # Structure and organization
    SOURCE_CREDIBILITY = "source_credibility"  # Source authority


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment result."""
    overall_score: int  # 0-100
    quality_level: str
    criteria_scores: dict[QualityCriterion, int]
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    confidence: float  # 0.0-1.0
    processing_time: float
    model_used: str


@dataclass
class QualityJudgmentContext:
    """Context for quality judgment operations."""
    content: str
    original_query: str
    query_terms: list[str]
    source_url: str
    source_domain: str
    content_length: int
    session_id: str
    min_acceptable_score: int = 60


class ContentQualityJudge:
    """
    AI-powered content quality judge with comprehensive assessment capabilities.

    Features:
    - Multi-criteria quality assessment
    - Judge scoring with confidence levels
    - Detailed feedback generation
    - Performance tracking
    - Feedback loops for cleaning optimization
    """

    def __init__(self, model_name: str = "gpt-5-nano", api_key: str | None = None):
        """
        Initialize the content quality judge.

        Args:
            model_name: OpenAI model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            logger.warning("No OpenAI API key found - quality assessment will be limited")
            self.agent = None
        elif PYDAI_AVAILABLE:
            try:
                # Initialize Pydantic AI agent with OpenAI model
                model = OpenAIModel(self.model_name, api_key=self.api_key)
                self.agent = Agent(
                    model,
                    system_prompt=self._get_judge_system_prompt(),
                    deps_type=QualityJudgmentContext
                )
                logger.info(f"Content quality judge initialized with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize quality judge agent: {e}")
                self.agent = None
        else:
            self.agent = None

        # Performance metrics
        self.stats = {
            'total_assessments': 0,
            'avg_quality_score': 0.0,
            'avg_processing_time': 0.0,
            'quality_distribution': {},
            'criteria_averages': {}
        }

    async def assess_content_quality(
        self,
        context: QualityJudgmentContext
    ) -> QualityAssessment:
        """
        Assess content quality using AI-powered judgment.

        Args:
            context: Quality judgment context with content and metadata

        Returns:
            QualityAssessment with comprehensive quality evaluation
        """
        start_time = datetime.now()

        try:
            # Pre-assessment checks
            if not self._should_assess_content(context):
                return self._create_minimal_assessment(context, "Content failed pre-assessment checks")

            # Use AI assessment if available
            if self.agent:
                result = await self._ai_assess_quality(context)
            else:
                # Fallback to rule-based assessment
                result = await self._rule_based_assess_quality(context)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time

            # Update statistics
            self._update_stats(result)

            logger.debug(f"Quality assessment completed: {result.overall_score}/100 "
                        f"({result.quality_level}) in {processing_time:.2f}s")

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Quality assessment failed for {context.source_url}: {e}")

            return QualityAssessment(
                overall_score=0,
                quality_level="Unusable",
                criteria_scores={},
                strengths=[],
                weaknesses=[f"Assessment failed: {str(e)}"],
                recommendations=[],
                confidence=0.0,
                processing_time=processing_time,
                model_used="failed"
            )

    async def assess_multiple_contents(
        self,
        contexts: list[QualityJudgmentContext],
        max_concurrent: int | None = None
    ) -> list[QualityAssessment]:
        """
        Assess multiple contents concurrently.

        Args:
            contexts: List of quality judgment contexts
            max_concurrent: Maximum concurrent assessments

        Returns:
            List of QualityAssessment results
        """
        if not contexts:
            return []

        logger.info(f"Starting batch quality assessment: {len(contexts)} items, "
                   f"max_concurrent={max_concurrent if max_concurrent and max_concurrent > 0 else 'unbounded'}")

        # Create semaphore to limit concurrent operations (optional)
        semaphore = (
            asyncio.Semaphore(max_concurrent)
            if max_concurrent and max_concurrent > 0
            else None
        )

        async def assess_with_semaphore(ctx: QualityJudgmentContext) -> QualityAssessment:
            if semaphore is None:
                return await self.assess_content_quality(ctx)

            async with semaphore:
                return await self.assess_content_quality(ctx)

        # Execute assessments concurrently
        start_time = datetime.now()
        tasks = [assess_with_semaphore(context) for context in contexts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                context = contexts[i]
                final_results.append(QualityAssessment(
                    overall_score=0,
                    quality_level="Unusable",
                    criteria_scores={},
                    strengths=[],
                    weaknesses=[f"Processing failed: {str(result)}"],
                    recommendations=[],
                    confidence=0.0,
                    processing_time=0.0,
                    model_used="failed"
                ))
            else:
                final_results.append(result)

        # Log batch summary
        total_time = (datetime.now() - start_time).total_seconds()
        avg_quality = sum(r.overall_score for r in final_results) / len(final_results)
        acceptable = sum(1 for r in final_results if r.overall_score >= contexts[0].min_acceptable_score)

        logger.info(f"Batch assessment completed: {acceptable}/{len(final_results)} acceptable "
                   f"(avg quality: {avg_quality:.1f}, total time: {total_time:.1f}s)")

        return final_results

    async def _ai_assess_quality(self, context: QualityJudgmentContext) -> QualityAssessment:
        """Use AI to assess content quality via Pydantic AI agent."""
        try:
            # Prepare the assessment prompt
            assessment_prompt = self._create_assessment_prompt(context)

            # Run the AI agent
            result = await self.agent.run(assessment_prompt, deps=context)

            # Parse the structured result
            assessment_data = result.data

            # Convert criteria scores to enum keys
            criteria_scores = {}
            for criterion_name, score in assessment_data.get('criteria_scores', {}).items():
                try:
                    criterion = QualityCriterion(criterion_name)
                    criteria_scores[criterion] = int(score)
                except ValueError:
                    logger.warning(f"Unknown quality criterion: {criterion_name}")

            return QualityAssessment(
                overall_score=assessment_data.get('overall_score', 50),
                quality_level=assessment_data.get('quality_level', 'Acceptable'),
                criteria_scores=criteria_scores,
                strengths=assessment_data.get('strengths', []),
                weaknesses=assessment_data.get('weaknesses', []),
                recommendations=assessment_data.get('recommendations', []),
                confidence=assessment_data.get('confidence', 0.7),
                processing_time=0.0,  # Will be set by caller
                model_used=self.model_name
            )

        except Exception as e:
            logger.error(f"AI quality assessment failed: {e}")
            # Fallback to rule-based assessment
            return await self._rule_based_assess_quality(context)

    async def _rule_based_assess_quality(self, context: QualityJudgmentContext) -> QualityAssessment:
        """Fallback rule-based quality assessment."""
        try:
            # Assess each criterion
            criteria_scores = {
                QualityCriterion.RELEVANCE: self._assess_relevance(context),
                QualityCriterion.COMPLETENESS: self._assess_completeness(context),
                QualityCriterion.CLARITY: self._assess_clarity(context),
                QualityCriterion.DEPTH: self._assess_depth(context),
                QualityCriterion.ORGANIZATION: self._assess_organization(context)
            }

            # Calculate overall score
            overall_score = sum(criteria_scores.values()) // len(criteria_scores)

            # Determine quality level
            quality_level = self._get_quality_level(overall_score)

            # Generate strengths and weaknesses
            strengths, weaknesses = self._generate_feedback(criteria_scores)

            # Generate recommendations
            recommendations = self._generate_recommendations(criteria_scores, context)

            return QualityAssessment(
                overall_score=overall_score,
                quality_level=quality_level,
                criteria_scores=criteria_scores,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                confidence=0.6,  # Lower confidence for rule-based
                processing_time=0.0,  # Will be set by caller
                model_used="rule_based"
            )

        except Exception as e:
            logger.error(f"Rule-based assessment failed: {e}")
            return self._create_minimal_assessment(context, f"Rule-based assessment failed: {e}")

    def _should_assess_content(self, context: QualityJudgmentContext) -> bool:
        """Check if content should be assessed based on basic criteria."""
        # Length checks
        if context.content_length < 100:
            return False

        if context.content_length > 100000:  # Too long for effective assessment
            return False

        # Basic content checks
        if not context.content.strip():
            return False

        return True

    def _assess_relevance(self, context: QualityJudgmentContext) -> int:
        """Assess content relevance to search query."""
        if not context.query_terms:
            return 50

        content_lower = context.content.lower()
        matches = 0

        for term in context.query_terms:
            if term.lower() in content_lower:
                matches += 1

        relevance_score = (matches / len(context.query_terms)) * 100

        # Bonus for phrase matches
        query_lower = context.original_query.lower()
        if query_lower in content_lower:
            relevance_score = min(100, relevance_score + 20)

        return int(relevance_score)

    def _assess_completeness(self, context: QualityJudgmentContext) -> int:
        """Assess information completeness."""
        content = context.content

        # Basic completeness indicators
        word_count = len(content.split())

        if word_count < 100:
            return 20
        elif word_count < 300:
            return 40
        elif word_count < 800:
            return 70
        elif word_count < 2000:
            return 85
        else:
            return 95

    def _assess_clarity(self, context: QualityJudgmentContext) -> int:
        """Assess content clarity and readability."""
        content = context.content

        # Average sentence length
        sentences = content.split('.')
        if not sentences:
            return 30

        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Optimal sentence length is 15-20 words
        if 10 <= avg_sentence_length <= 25:
            clarity_score = 80
        elif 5 <= avg_sentence_length <= 35:
            clarity_score = 60
        else:
            clarity_score = 40

        # Check for proper capitalization
        if content and content[0].isupper():
            clarity_score += 10

        # Check punctuation
        punctuation_ratio = content.count('.') + content.count('!') + content.count('?')
        if punctuation_ratio > 0:
            clarity_score += 10

        return min(100, clarity_score)

    def _assess_depth(self, context: QualityJudgmentContext) -> int:
        """Assess depth of information."""
        content = context.content.lower()

        # Depth indicators
        depth_indicators = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'analysis', 'research', 'study', 'evidence', 'data', 'statistics',
            'example', 'instance', 'specifically', 'particularly', 'notably'
        ]

        indicator_count = sum(1 for indicator in depth_indicators if indicator in content)

        # Base depth score
        if indicator_count >= 5:
            depth_score = 80
        elif indicator_count >= 3:
            depth_score = 60
        elif indicator_count >= 1:
            depth_score = 40
        else:
            depth_score = 20

        # Bonus for longer content (indicating more depth)
        word_count = len(context.content.split())
        if word_count > 1000:
            depth_score += 20
        elif word_count > 500:
            depth_score += 10

        return min(100, depth_score)

    def _assess_organization(self, context: QualityJudgmentContext) -> int:
        """Assess content structure and organization."""
        content = context.content

        # Organization indicators
        has_headings = bool(re.search(r'^#+\s', content, re.MULTILINE))
        has_paragraphs = content.count('\n\n') >= 2
        has_lists = bool(re.search(r'^\s*[-*+]\s', content, re.MULTILINE))

        organization_score = 50  # Base score

        if has_headings:
            organization_score += 25
        if has_paragraphs:
            organization_score += 15
        if has_lists:
            organization_score += 10

        return min(100, organization_score)

    def _generate_feedback(
        self,
        criteria_scores: dict[QualityCriterion, int]
    ) -> tuple[list[str], list[str]]:
        """Generate strengths and weaknesses based on criteria scores."""
        strengths = []
        weaknesses = []

        for criterion, score in criteria_scores.items():
            if score >= 75:
                strengths.append(self._get_criterion_strength(criterion, score))
            elif score < 50:
                weaknesses.append(self._get_criterion_weakness(criterion, score))

        return strengths, weaknesses

    def _generate_recommendations(
        self,
        criteria_scores: dict[QualityCriterion, int],
        context: QualityJudgmentContext
    ) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        for criterion, score in criteria_scores.items():
            if score < 60:
                recommendations.append(self._get_criterion_recommendation(criterion, score))

        return recommendations[:5]  # Limit to 5 recommendations

    def _get_criterion_strength(self, criterion: QualityCriterion, score: int) -> str:
        """Get strength description for a criterion."""
        strength_map = {
            QualityCriterion.RELEVANCE: f"Highly relevant to search query ({score}/100)",
            QualityCriterion.COMPLETENESS: f"Comprehensive information coverage ({score}/100)",
            QualityCriterion.CLARITY: f"Clear and readable content ({score}/100)",
            QualityCriterion.DEPTH: f"In-depth analysis and details ({score}/100)",
            QualityCriterion.ORGANIZATION: f"Well-structured and organized ({score}/100)"
        }
        return strength_map.get(criterion, f"Strong in {criterion.value} ({score}/100)")

    def _get_criterion_weakness(self, criterion: QualityCriterion, score: int) -> str:
        """Get weakness description for a criterion."""
        weakness_map = {
            QualityCriterion.RELEVANCE: f"Limited relevance to search query ({score}/100)",
            QualityCriterion.COMPLETENESS: f"Incomplete information coverage ({score}/100)",
            QualityCriterion.CLARITY: f"Unclear or difficult to read ({score}/100)",
            QualityCriterion.DEPTH: f"Lacks depth and detail ({score}/100)",
            QualityCriterion.ORGANIZATION: f"Poorly structured content ({score}/100)"
        }
        return weakness_map.get(criterion, f"Weak in {criterion.value} ({score}/100)")

    def _get_criterion_recommendation(self, criterion: QualityCriterion, score: int) -> str:
        """Get improvement recommendation for a criterion."""
        recommendation_map = {
            QualityCriterion.RELEVANCE: "Increase focus on search query topics and keywords",
            QualityCriterion.COMPLETENESS: "Add more comprehensive information and details",
            QualityCriterion.CLARITY: "Improve sentence structure and readability",
            QualityCriterion.DEPTH: "Include more analysis, examples, and supporting evidence",
            QualityCriterion.ORGANIZATION: "Improve content structure with headings and paragraphs"
        }
        return recommendation_map.get(criterion, f"Improve {criterion.value}")

    def _get_quality_level(self, score: int) -> str:
        """Convert quality score to quality level."""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Acceptable"
        elif score >= 40:
            return "Poor"
        else:
            return "Unusable"

    def _create_minimal_assessment(
        self,
        context: QualityJudgmentContext,
        reason: str
    ) -> QualityAssessment:
        """Create a minimal assessment for content that fails processing."""
        return QualityAssessment(
            overall_score=0,
            quality_level="Unusable",
            criteria_scores={},
            strengths=[],
            weaknesses=[reason],
            recommendations=[],
            confidence=0.0,
            processing_time=0.0,
            model_used="none"
        )

    def _get_judge_system_prompt(self) -> str:
        """Get the system prompt for the AI quality judge."""
        return """You are an expert content quality judge with deep expertise in evaluating information quality, accuracy, and relevance. Your task is to comprehensively assess content quality across multiple criteria.

Given content and a search query, you must evaluate:

**Quality Criteria (score 0-100 each):**
1. **Relevance**: How well the content matches the search query and addresses the user's information needs
2. **Completeness**: How thoroughly the topic is covered, including important aspects and details
3. **Accuracy**: Indicators of factual accuracy, reliable information, and proper sourcing
4. **Clarity**: Readability, clear expression, logical flow, and proper language use
5. **Depth**: Level of detail, analysis, examples, and supporting evidence provided
6. **Organization**: Structure, coherence, logical progression, and proper formatting
7. **Source Credibility**: Authority and trustworthiness indicators of the source

**Assessment Guidelines:**
- Be objective and consistent in your evaluation
- Consider the content's purpose and intended audience
- Provide specific, actionable feedback
- Consider both strengths and areas for improvement
- Assign confidence level based on assessment certainty

**Scoring Standards:**
- 90-100: Excellent quality across all criteria
- 80-89: High quality with minor areas for improvement
- 70-79: Good quality, meets basic requirements well
- 60-69: Acceptable quality, has some noticeable issues
- 40-59: Poor quality, significant improvements needed
- 0-39: Unusable, major quality issues

Return your assessment in this JSON format:
{
    "overall_score": 85,
    "quality_level": "Good",
    "criteria_scores": {
        "relevance": 90,
        "completeness": 80,
        "accuracy": 85,
        "clarity": 80,
        "depth": 75,
        "organization": 85,
        "source_credibility": 80
    },
    "strengths": [
        "Highly relevant to search query",
        "Well-structured and organized",
        "Clear and readable language"
    ],
    "weaknesses": [
        "Could use more supporting examples",
        "Some sections lack depth"
    ],
    "recommendations": [
        "Add specific examples to illustrate key points",
        "Include more supporting evidence and data",
        "Expand analysis in weaker sections"
    ],
    "confidence": 0.8
}

Focus on providing fair, constructive, and detailed quality assessments."""

    def _create_assessment_prompt(self, context: QualityJudgmentContext) -> str:
        """Create the assessment prompt for the AI."""
        return f"""Please assess the quality of this content for the search query: "{context.original_query}"

**Content Information:**
- Source URL: {context.source_url}
- Source Domain: {context.source_domain}
- Content Length: {context.content_length} characters
- Query Terms: {', '.join(context.query_terms)}

**Content to Assess:**
{context.content[:12000]}  # Limit content to avoid token limits

Please evaluate this content comprehensively using the quality criteria outlined in the instructions. Return your assessment in valid JSON format."""

    def _update_stats(self, assessment: QualityAssessment):
        """Update assessment statistics."""
        self.stats['total_assessments'] += 1

        # Update average quality score
        total = self.stats['total_assessments']
        current_avg = self.stats['avg_quality_score']
        self.stats['avg_quality_score'] = (current_avg * (total - 1) + assessment.overall_score) / total

        # Update quality distribution
        level = assessment.quality_level
        if level not in self.stats['quality_distribution']:
            self.stats['quality_distribution'][level] = 0
        self.stats['quality_distribution'][level] += 1

        # Update criteria averages
        for criterion, score in assessment.criteria_scores.items():
            if criterion.value not in self.stats['criteria_averages']:
                self.stats['criteria_averages'][criterion.value] = []
            self.stats['criteria_averages'][criterion.value].append(score)

    def get_stats(self) -> dict[str, Any]:
        """Get assessment statistics."""
        # Calculate criteria averages
        criteria_averages = {}
        for criterion, scores in self.stats['criteria_averages'].items():
            if scores:
                criteria_averages[criterion] = sum(scores) / len(scores)

        return {
            **self.stats,
            'agent_available': self.agent is not None,
            'model_used': self.model_name,
            'criteria_averages': criteria_averages
        }


# Global quality judge instance
_global_quality_judge: ContentQualityJudge | None = None


def get_content_quality_judge() -> ContentQualityJudge:
    """Get or create global content quality judge."""
    global _global_quality_judge
    if _global_quality_judge is None:
        _global_quality_judge = ContentQualityJudge()
    return _global_quality_judge
