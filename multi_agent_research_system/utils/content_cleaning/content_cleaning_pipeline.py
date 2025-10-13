"""
Content Cleaning Pipeline with Quality Validation

Phase 1.3.2: Create content cleaning pipeline with quality validation

This module provides a comprehensive content cleaning pipeline that integrates
with the FastConfidenceScorer to provide quality validation and enhancement
decisions for web-crawled content.

Key Features:
- Integration with FastConfidenceScorer for quality assessment
- Multi-stage content cleaning (basic → AI-enhanced)
- Quality validation and enhancement decisions
- Performance optimization with skip logic
- Comprehensive error handling and fallbacks
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime

from .fast_confidence_scorer import FastConfidenceScorer, ConfidenceSignals

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    """Result of content cleaning pipeline with quality validation."""

    original_content: str
    cleaned_content: str
    url: str
    search_query: Optional[str] = None

    # Quality assessment
    confidence_signals: Optional[ConfidenceSignals] = None
    cleaning_performed: bool = False
    quality_improvement: float = 0.0

    # Processing metadata
    processing_time_ms: float = 0.0
    cleaning_stage: str = "none"  # none, basic, ai_enhanced, quality_validated
    error_message: Optional[str] = None

    # Recommendations
    editorial_recommendation: str = "UNKNOWN"
    enhancement_suggestions: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Configuration for content cleaning pipeline."""

    # Cleaning thresholds
    cleanliness_threshold: float = 0.7  # Skip cleaning if content is already clean
    minimum_quality_threshold: float = 0.6  # Minimum acceptable quality
    enhancement_threshold: float = 0.8  # Enhance content below this threshold

    # Processing options
    enable_ai_cleaning: bool = True
    enable_quality_validation: bool = True
    enable_performance_optimization: bool = True

    # Content limits
    max_content_length_for_ai: int = 50000  # Maximum content length for AI cleaning
    min_content_length_for_cleaning: int = 500  # Minimum content length for cleaning

    # Quality settings
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        'relevance': 0.3,
        'completeness': 0.25,
        'clarity': 0.2,
        'authority': 0.15,
        'freshness': 0.1
    })

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.cleanliness_threshold <= 1.0:
            raise ValueError("cleanliness_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.minimum_quality_threshold <= 1.0:
            raise ValueError("minimum_quality_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.enhancement_threshold <= 1.0:
            raise ValueError("enhancement_threshold must be between 0.0 and 1.0")


class ContentCleaningPipeline:
    """
    Comprehensive content cleaning pipeline with quality validation.

    Integrates FastConfidenceScorer for intelligent cleaning decisions and
    provides multi-stage cleaning with quality assessment.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the content cleaning pipeline.

        Args:
            config: Pipeline configuration options
        """
        self.config = config or PipelineConfig()
        self.confidence_scorer = FastConfidenceScorer(
            cache_enabled=True,
            cache_size=1000
        )

        logger.info("ContentCleaningPipeline initialized with quality validation")

    async def clean_content(
        self,
        content: str,
        url: str,
        search_query: Optional[str] = None,
        extraction_metadata: Optional[Dict[str, Any]] = None
    ) -> CleaningResult:
        """
        Clean content with quality validation and confidence scoring.

        Args:
            content: Raw content to clean
            url: Source URL for context
            search_query: Original search query for relevance
            extraction_metadata: Metadata from extraction process

        Returns:
            CleaningResult with cleaned content and quality assessment
        """
        import time
        start_time = time.time()

        logger.debug(f"Starting content cleaning pipeline for {url}")

        # Initialize result
        result = CleaningResult(
            original_content=content,
            cleaned_content=content,
            url=url,
            search_query=search_query
        )

        try:
            # Stage 1: Content validation and initial assessment
            if not self._validate_content_for_cleaning(content):
                result.cleaning_stage = "rejected"
                result.error_message = "Content validation failed"
                logger.warning(f"Content validation failed for {url}")
                return result

            # Stage 2: Confidence assessment
            if self.config.enable_quality_validation:
                result.confidence_signals = await self.confidence_scorer.assess_content_confidence(
                    content=content,
                    url=url,
                    search_query=search_query,
                    extraction_metadata=extraction_metadata
                )

                # Stage 3: Editorial recommendation
                result.editorial_recommendation = self.confidence_scorer.get_editorial_recommendation(
                    result.confidence_signals
                )

            # Stage 4: Cleaning decision and execution
            await self._execute_cleaning_strategy(result)

            # Stage 5: Post-cleaning quality validation
            if result.cleaning_performed and self.config.enable_quality_validation:
                await self._validate_cleaning_quality(result)

            # Stage 6: Generate enhancement suggestions
            if result.confidence_signals:
                result.enhancement_suggestions = self._generate_enhancement_suggestions(
                    result.confidence_signals
                )

        except Exception as e:
            logger.error(f"Error in content cleaning pipeline for {url}: {e}")
            result.error_message = str(e)
            result.cleaned_content = content  # Fallback to original

        # Calculate processing time
        result.processing_time_ms = (time.time() - start_time) * 1000

        logger.debug(f"Content cleaning completed for {url}: "
                    f"stage={result.cleaning_stage}, "
                    f"quality={result.confidence_signals.overall_confidence if result.confidence_signals else 'N/A'}, "
                    f"time={result.processing_time_ms:.1f}ms")

        return result

    def _validate_content_for_cleaning(self, content: str) -> bool:
        """
        Validate content meets minimum requirements for cleaning.

        Args:
            content: Content to validate

        Returns:
            True if content is valid for cleaning
        """
        # Check minimum content length
        if len(content.strip()) < self.config.min_content_length_for_cleaning:
            logger.debug(f"Content too short for cleaning: {len(content)} chars")
            return False

        # Check for meaningful content (not just navigation/HTML)
        if self._is_mostly_navigation(content):
            logger.debug("Content appears to be mostly navigation, skipping cleaning")
            return False

        return True

    def _is_mostly_navigation(self, content: str) -> bool:
        """
        Simple heuristic to detect if content is mostly navigation elements.

        Args:
            content: Content to analyze

        Returns:
            True if content appears to be mostly navigation
        """
        nav_indicators = [
            'menu', 'navigation', 'nav', 'sidebar', 'header', 'footer',
            'login', 'signup', 'search', 'cart', 'account', 'profile'
        ]

        nav_count = sum(content.lower().count(indicator) for indicator in nav_indicators)
        content_words = len(content.split())

        # If navigation indicators are more than 10% of content, consider it mostly navigation
        return content_words > 0 and (nav_count / content_words) > 0.1

    async def _execute_cleaning_strategy(self, result: CleaningResult) -> None:
        """
        Execute cleaning strategy based on confidence assessment.

        Args:
            result: CleaningResult to update
        """
        if not result.confidence_signals:
            # No quality validation, perform basic cleaning
            await self._perform_basic_cleaning(result)
            return

        # Check if cleaning is needed based on cleanliness score
        cleanliness = result.confidence_signals.cleanliness_score

        if cleanliness >= self.config.cleanliness_threshold:
            logger.debug(f"Content already clean ({cleanliness:.2f}), skipping cleaning")
            result.cleaning_stage = "skip_clean"
            result.cleaned_content = result.original_content
            return

        # Determine cleaning strategy based on quality
        overall_quality = result.confidence_signals.overall_confidence

        if overall_quality >= self.config.enhancement_threshold:
            # Good quality, basic cleaning only
            await self._perform_basic_cleaning(result)
        else:
            # Lower quality, perform AI-enhanced cleaning
            if self.config.enable_ai_cleaning:
                await self._perform_ai_enhanced_cleaning(result)
            else:
                await self._perform_basic_cleaning(result)

    async def _perform_basic_cleaning(self, result: CleaningResult) -> None:
        """
        Perform basic content cleaning using simple heuristics.

        Args:
            result: CleaningResult to update
        """
        logger.debug(f"Performing basic cleaning for {result.url}")

        # Import existing basic cleaning functions
        try:
            from multi_agent_research_system.utils.content_cleaning import clean_content_with_gpt5_nano

            # Use existing AI cleaning function as basic cleaning
            cleaned = await clean_content_with_gpt5_nano(
                content=result.original_content,
                url=result.url,
                search_query=result.search_query
            )

            result.cleaned_content = cleaned
            result.cleaning_performed = True
            result.cleaning_stage = "basic"

        except Exception as e:
            logger.warning(f"Basic cleaning failed for {result.url}: {e}")
            # Fallback to simple heuristic cleaning
            result.cleaned_content = self._simple_heuristic_cleaning(result.original_content)
            result.cleaning_performed = True
            result.cleaning_stage = "basic_fallback"

    async def _perform_ai_enhanced_cleaning(self, result: CleaningResult) -> None:
        """
        Perform AI-enhanced content cleaning for lower quality content.

        Args:
            result: CleaningResult to update
        """
        logger.debug(f"Performing AI-enhanced cleaning for {result.url}")

        try:
            # Use the existing AI cleaning function with optimization
            from multi_agent_research_system.utils.content_cleaning import clean_content_with_judge_optimization

            cleaned, metadata = await clean_content_with_judge_optimization(
                content=result.original_content,
                url=result.url,
                search_query=result.search_query,
                cleanliness_threshold=self.config.cleanliness_threshold,
                skip_judge=False
            )

            result.cleaned_content = cleaned
            result.cleaning_performed = True
            result.cleaning_stage = "ai_enhanced"

            # Store cleaning metadata
            if metadata:
                result.error_message = f"Cleaning metadata: {metadata.get('optimization_used', 'unknown')}"

        except Exception as e:
            logger.warning(f"AI-enhanced cleaning failed for {result.url}: {e}")
            # Fallback to basic cleaning
            await self._perform_basic_cleaning(result)

    def _simple_heuristic_cleaning(self, content: str) -> str:
        """
        Simple heuristic cleaning as fallback.

        Args:
            content: Content to clean

        Returns:
            Cleaned content
        """
        lines = content.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip common navigation and UI elements
            skip_patterns = [
                'menu', 'navigation', 'login', 'signup', 'search', 'cart',
                'cookie', 'privacy', 'terms', 'subscribe', 'follow us',
                'share this', 'related', 'trending', 'popular'
            ]

            if any(pattern in line.lower() for pattern in skip_patterns):
                continue

            # Skip very short lines (likely UI elements)
            if len(line) < 10:
                continue

            # Skip lines with mostly special characters (likely formatting)
            if sum(c.isalnum() for c in line) / len(line) < 0.5:
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    async def _validate_cleaning_quality(self, result: CleaningResult) -> None:
        """
        Validate the quality of cleaned content.

        Args:
            result: CleaningResult to update
        """
        if not result.confidence_signals or not result.cleaning_performed:
            return

        # Calculate quality improvement
        original_quality = result.confidence_signals.overall_confidence

        # Quick assessment of cleaned content
        try:
            cleaned_signals = await self.confidence_scorer.assess_content_confidence(
                content=result.cleaned_content,
                url=result.url,
                search_query=result.search_query
            )

            cleaned_quality = cleaned_signals.overall_confidence
            result.quality_improvement = cleaned_quality - original_quality

            # Update cleaning stage based on quality
            if result.quality_improvement > 0.1:
                result.cleaning_stage = "quality_validated"
            elif result.quality_improvement > 0.0:
                result.cleaning_stage = "basic_improved"
            else:
                result.cleaning_stage = "no_improvement"

            logger.debug(f"Quality validation: {original_quality:.3f} → {cleaned_quality:.3f} "
                        f"(improvement: {result.quality_improvement:.3f})")

        except Exception as e:
            logger.warning(f"Quality validation failed for {result.url}: {e}")

    def _generate_enhancement_suggestions(self, signals: ConfidenceSignals) -> List[str]:
        """
        Generate enhancement suggestions based on confidence signals.

        Args:
            signals: ConfidenceSignals from assessment

        Returns:
            List of enhancement suggestions
        """
        suggestions = []

        # Analyze component scores and generate specific suggestions
        if signals.content_length_score < 0.7:
            suggestions.append("Consider finding more comprehensive sources with better content depth")

        if signals.structure_score < 0.6:
            suggestions.append("Content lacks proper structure - look for sources with better organization")

        if signals.relevance_score < 0.7:
            suggestions.append("Content relevance is low - consider more targeted search queries")

        if signals.domain_authority_score < 0.6:
            suggestions.append("Source authority is low - prioritize academic, government, or established news sources")

        if signals.freshness_score < 0.6:
            suggestions.append("Content appears outdated - look for more recent sources")

        if signals.extraction_confidence < 0.7:
            suggestions.append("Content extraction had issues - consider alternative sources or extraction methods")

        if signals.llm_assessment < 0.7:
            suggestions.append("Overall content quality is low - gap research recommended for better coverage")

        # Overall quality recommendations
        if signals.overall_confidence < 0.6:
            suggestions.append("Overall quality is below acceptable threshold - gap research strongly recommended")
        elif signals.overall_confidence < 0.8:
            suggestions.append("Content quality is acceptable but could be enhanced with additional research")

        return suggestions

    async def clean_content_batch(
        self,
        content_list: List[Tuple[str, str]],
        search_query: Optional[str] = None,
        extraction_metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[CleaningResult]:
        """
        Clean multiple content pieces in parallel.

        Args:
            content_list: List of (content, url) tuples
            search_query: Common search query for all content
            extraction_metadata_list: List of extraction metadata dictionaries

        Returns:
            List of CleaningResult objects
        """
        logger.info(f"Starting batch content cleaning for {len(content_list)} items")

        # Prepare metadata for each item
        metadata_list = extraction_metadata_list or [None] * len(content_list)

        # Create cleaning tasks
        tasks = [
            self.clean_content(
                content=content,
                url=url,
                search_query=search_query,
                extraction_metadata=metadata
            )
            for (content, url), metadata in zip(content_list, metadata_list)
        ]

        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and ensure we return CleaningResult objects
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch cleaning failed for item {i}: {result}")
                # Create a fallback result
                content, url = content_list[i]
                fallback_result = CleaningResult(
                    original_content=content,
                    cleaned_content=content,
                    url=url,
                    search_query=search_query,
                    error_message=str(result),
                    cleaning_stage="failed"
                )
                final_results.append(fallback_result)
            else:
                final_results.append(result)

        # Calculate batch statistics
        successful_cleanings = sum(1 for r in final_results if r.cleaning_performed and not r.error_message)
        logger.info(f"Batch cleaning completed: {successful_cleanings}/{len(final_results)} successful")

        return final_results

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline performance statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        cache_stats = self.confidence_scorer.get_cache_stats()

        return {
            'config': {
                'cleanliness_threshold': self.config.cleanliness_threshold,
                'minimum_quality_threshold': self.config.minimum_quality_threshold,
                'enable_ai_cleaning': self.config.enable_ai_cleaning,
                'enable_quality_validation': self.config.enable_quality_validation
            },
            'cache_stats': cache_stats
        }

    def clear_caches(self):
        """Clear all pipeline caches."""
        self.confidence_scorer.clear_cache()
        logger.info("Content cleaning pipeline caches cleared")