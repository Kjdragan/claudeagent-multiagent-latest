"""
Fast Confidence Scorer with GPT-5-Nano Integration

Phase 1.3.1: Implement FastConfidenceScorer with GPT-5-nano integration

This module provides fast confidence assessment using simple heuristics combined with
GPT-5-nano LLM calls for quality scoring. Designed for speed and simplicity
without complex rate limiting assumptions.

Technical Specifications:
- Simple weighted scoring (content length, structure, relevance, domain authority, freshness, extraction confidence, cleanliness)
- GPT-5-Nano Integration: Fast LLM calls with 50 tokens max, temperature 0.1
- Thresholds: Gap research trigger at 0.7, acceptable quality at 0.6, good quality at 0.8
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse
import time

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceSignals:
    """Data class for confidence scoring signals."""

    # Content-based signals
    content_length_score: float = 0.0  # Optimal 500-5000 words
    structure_score: float = 0.0      # Headings, paragraphs, lists, links
    relevance_score: float = 0.0      # Content relevance to query
    cleanliness_score: float = 0.0    # Content cleanliness assessment

    # Source-based signals
    domain_authority_score: float = 0.0  # edu/gov/org = 0.9, news = 0.8, etc.
    freshness_score: float = 0.0        # Content freshness assessment
    extraction_confidence: float = 0.0  # Confidence in extraction quality

    # Overall scores
    overall_confidence: float = 0.0
    llm_assessment: float = 0.0       # GPT-5-nano quality assessment

    # Metadata
    assessment_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: float = 0.0
    cache_hit: bool = False


class FastConfidenceScorer:
    """
    Fast confidence scorer using simple heuristics + GPT-5-nano integration.

    Designed for speed and reliability with minimal complexity.
    """

    def __init__(self, cache_enabled: bool = True, cache_size: int = 1000):
        """
        Initialize the FastConfidenceScorer.

        Args:
            cache_enabled: Enable LRU caching for repeated assessments
            cache_size: Maximum number of cached assessments
        """
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        self._cache = {} if cache_enabled else None
        self._cache_access_order = [] if cache_enabled else None

        # Scoring weights (simple, balanced approach)
        self.weights = {
            'content_length': 0.15,
            'structure': 0.15,
            'relevance': 0.20,
            'cleanliness': 0.15,
            'domain_authority': 0.15,
            'freshness': 0.10,
            'extraction_confidence': 0.10
        }

        # Thresholds as specified in requirements
        self.thresholds = {
            'gap_research_trigger': 0.7,   # Trigger gap research below this
            'acceptable_quality': 0.6,     # Minimum acceptable quality
            'good_quality': 0.8,           # Good quality threshold
            'excellent_quality': 0.9       # Excellent quality threshold
        }

        logger.info("FastConfidenceScorer initialized with cache_enabled=%s", cache_enabled)

    async def assess_content_confidence(
        self,
        content: str,
        url: str,
        search_query: Optional[str] = None,
        extraction_metadata: Optional[Dict[str, Any]] = None
    ) -> ConfidenceSignals:
        """
        Assess content confidence using fast heuristics + GPT-5-nano.

        Args:
            content: Content to assess
            url: Source URL for context
            search_query: Original search query for relevance scoring
            extraction_metadata: Metadata from extraction process

        Returns:
            ConfidenceSignals object with detailed scoring
        """
        start_time = time.time()

        # Check cache first
        cache_key = self._generate_cache_key(content, url, search_query)
        if self.cache_enabled and cache_key in self._cache:
            cached_result = self._cache[cache_key]
            cached_result.cache_hit = True
            cached_result.assessment_timestamp = datetime.now().isoformat()
            self._update_cache_access_order(cache_key)
            logger.debug(f"Cache hit for confidence assessment: {url}")
            return cached_result

        logger.debug(f"Assessing confidence for content from {url} (length: {len(content)})")

        # Initialize signals
        signals = ConfidenceSignals()

        # 1. Content length scoring (optimal 500-5000 words)
        signals.content_length_score = self._score_content_length(content)

        # 2. Structure assessment
        signals.structure_score = self._score_content_structure(content)

        # 3. Relevance scoring
        signals.relevance_score = self._score_content_relevance(content, search_query)

        # 4. Cleanliness assessment
        signals.cleanliness_score = await self._assess_content_cleanliness(content, url)

        # 5. Domain authority scoring
        signals.domain_authority_score = self._score_domain_authority(url)

        # 6. Freshness assessment
        signals.freshness_score = self._assess_content_freshness(content)

        # 7. Extraction confidence
        signals.extraction_confidence = self._score_extraction_confidence(
            content, extraction_metadata
        )

        # 8. GPT-5-nano quality assessment
        signals.llm_assessment = await self._get_gpt5_nano_assessment(content, url, search_query)

        # 9. Calculate overall confidence
        signals.overall_confidence = self._calculate_overall_confidence(signals)

        # 10. Record processing time
        signals.processing_time_ms = (time.time() - start_time) * 1000
        signals.assessment_timestamp = datetime.now().isoformat()

        # Cache result
        if self.cache_enabled:
            self._add_to_cache(cache_key, signals)

        logger.debug(f"Confidence assessment completed: {signals.overall_confidence:.3f} in {signals.processing_time_ms:.1f}ms")

        return signals

    def _generate_cache_key(self, content: str, url: str, search_query: Optional[str]) -> str:
        """Generate cache key for assessment."""
        import hashlib
        content_hash = hashlib.md5(content[:1000].encode()).hexdigest()[:8]
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        query_hash = hashlib.md5((search_query or "").encode()).hexdigest()[:8]
        return f"{content_hash}_{url_hash}_{query_hash}"

    def _update_cache_access_order(self, cache_key: str):
        """Update LRU cache access order."""
        if cache_key in self._cache_access_order:
            self._cache_access_order.remove(cache_key)
        self._cache_access_order.append(cache_key)

    def _add_to_cache(self, cache_key: str, signals: ConfidenceSignals):
        """Add result to cache with LRU eviction."""
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = self._cache_access_order.pop(0)
            del self._cache[oldest_key]

        self._cache[cache_key] = signals
        self._cache_access_order.append(cache_key)

    def _score_content_length(self, content: str) -> float:
        """
        Score content based on optimal length (500-5000 words).

        Simple approach:
        - Too short (< 200 words): 0.3
        - Good range (500-5000 words): 1.0
        - Too long (> 10000 words): 0.7
        - Linear interpolation between ranges
        """
        word_count = len(content.split())

        if word_count < 200:
            return 0.3
        elif 200 <= word_count < 500:
            # Linear from 0.3 to 1.0
            return 0.3 + (word_count - 200) * (0.7 / 300)
        elif 500 <= word_count <= 5000:
            return 1.0
        elif 5000 < word_count <= 10000:
            # Linear from 1.0 to 0.7
            return 1.0 - (word_count - 5000) * (0.3 / 5000)
        else:
            return 0.7

    def _score_content_structure(self, content: str) -> float:
        """
        Score content structure (headings, paragraphs, lists, links).

        Simple heuristic scoring based on structural elements.
        """
        score = 0.0

        # Check for headings (markdown style)
        headings = len(re.findall(r'^#{1,6}\s', content, re.MULTILINE))
        if headings >= 3:
            score += 0.3
        elif headings >= 1:
            score += 0.15

        # Check for paragraphs
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        if paragraphs >= 5:
            score += 0.3
        elif paragraphs >= 2:
            score += 0.15

        # Check for lists
        lists = len(re.findall(r'^[\-\*\+]\s', content, re.MULTILINE))
        if lists >= 3:
            score += 0.2
        elif lists >= 1:
            score += 0.1

        # Check for links
        links = len(re.findall(r'https?://[^\s]+', content))
        if links >= 3:
            score += 0.2
        elif links >= 1:
            score += 0.1

        return min(score, 1.0)

    def _score_content_relevance(self, content: str, search_query: Optional[str]) -> float:
        """
        Score content relevance to search query.

        Simple keyword-based relevance scoring.
        """
        if not search_query:
            return 0.5  # Neutral score if no query

        # Extract keywords from query (simple approach)
        query_words = set(search_query.lower().split())
        content_lower = content.lower()

        # Count query word occurrences
        matches = 0
        for word in query_words:
            if len(word) > 2:  # Ignore very short words
                matches += content_lower.count(word)

        # Normalize by content length
        content_words = len(content.split())
        if content_words == 0:
            return 0.0

        relevance_density = matches / content_words

        # Convert to 0-1 scale (simple approach)
        if relevance_density > 0.05:
            return 1.0
        elif relevance_density > 0.02:
            return 0.8
        elif relevance_density > 0.01:
            return 0.6
        elif relevance_density > 0.005:
            return 0.4
        else:
            return 0.2

    async def _assess_content_cleanliness(self, content: str, url: str) -> float:
        """
        Assess content cleanliness using existing assessment function.

        Uses the existing assess_content_cleanliness function from content_cleaning module.
        """
        try:
            # Import the existing cleanliness assessment
            from multi_agent_research_system.utils.content_cleaning import assess_content_cleanliness

            is_clean, cleanliness_score = await assess_content_cleanliness(content, url, threshold=0.5)
            return cleanliness_score

        except Exception as e:
            logger.warning(f"Cleanliness assessment failed for {url}: {e}")
            # Fallback: simple heuristic based on content characteristics
            return self._simple_cleanliness_heuristic(content)

    def _simple_cleanliness_heuristic(self, content: str) -> float:
        """
        Simple heuristic for cleanliness assessment when LLM assessment fails.
        """
        score = 0.5  # Base score

        # Penalize common navigation elements
        nav_indicators = ['menu', 'navigation', 'sidebar', 'footer', 'header']
        nav_count = sum(content.lower().count(indicator) for indicator in nav_indicators)
        if nav_count > 10:
            score -= 0.2
        elif nav_count > 5:
            score -= 0.1

        # Reward good content indicators
        content_indicators = ['article', 'content', 'main', 'section']
        content_count = sum(content.lower().count(indicator) for indicator in content_indicators)
        if content_count > 3:
            score += 0.2
        elif content_count > 1:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _score_domain_authority(self, url: str) -> float:
        """
        Score domain authority based on URL patterns.

        Simple approach:
        - edu/gov/org = 0.9
        - news sites = 0.8
        - established domains = 0.7
        - others = 0.5
        """
        try:
            domain = urlparse(url).netloc.lower()

            # High authority domains
            if any(tld in domain for tld in ['.edu', '.gov', '.org']):
                return 0.9

            # News domains
            if any(news in domain for news in ['news', 'reuters', 'bbc', 'cnn', 'ap']):
                return 0.8

            # Established domains
            if any(established in domain for established in ['wikipedia', 'nature', 'science', 'techcrunch']):
                return 0.8

            # Commercial domains
            if '.com' in domain:
                return 0.6

            # Other domains
            return 0.5

        except Exception:
            return 0.5  # Default for malformed URLs

    def _assess_content_freshness(self, content: str) -> float:
        """
        Assess content freshness based on date patterns in content.

        Simple approach: look for recent dates in content.
        """
        current_year = datetime.now().year
        recent_years = [current_year, current_year - 1, current_year - 2]

        # Look for year patterns
        year_pattern = r'\b(20\d{2})\b'
        found_years = re.findall(year_pattern, content)

        if not found_years:
            return 0.5  # No dates found

        # Check for recent years
        recent_count = sum(1 for year in found_years if int(year) in recent_years)
        total_count = len(found_years)

        if total_count == 0:
            return 0.5

        recent_ratio = recent_count / total_count

        if recent_ratio >= 0.8:
            return 1.0
        elif recent_ratio >= 0.5:
            return 0.8
        elif recent_ratio >= 0.3:
            return 0.6
        else:
            return 0.4

    def _score_extraction_confidence(
        self,
        content: str,
        extraction_metadata: Optional[Dict[str, Any]]
    ) -> float:
        """
        Score extraction confidence based on metadata and content characteristics.

        Simple approach based on extraction success indicators.
        """
        if not extraction_metadata:
            return self._simple_extraction_confidence(content)

        score = 0.5  # Base score

        # Check extraction success indicators
        if extraction_metadata.get('success', False):
            score += 0.3

        if extraction_metadata.get('status_code', 0) == 200:
            score += 0.2

        # Check content characteristics
        if len(content.strip()) > 1000:
            score += 0.1

        if len(content.strip()) > 5000:
            score += 0.1

        return min(score, 1.0)

    def _simple_extraction_confidence(self, content: str) -> float:
        """
        Simple extraction confidence based on content alone.
        """
        score = 0.5

        # Length-based confidence
        if len(content) > 2000:
            score += 0.3
        elif len(content) > 500:
            score += 0.2

        # Structure-based confidence
        if '\n\n' in content:  # Has paragraphs
            score += 0.1

        if 'http' in content:  # Has links
            score += 0.1

        return min(score, 1.0)

    async def _get_gpt5_nano_assessment(
        self,
        content: str,
        url: str,
        search_query: Optional[str]
    ) -> float:
        """
        Get GPT-5-nano quality assessment (fast LLM call, 50 tokens max, temperature 0.1).

        Simple prompt for quick quality assessment.
        """
        try:
            from pydantic_ai import Agent

            # Fast quality assessment agent
            agent = Agent(
                model="openai:gpt-5-nano",
                system_prompt="You are a content quality assessor. Rate content quality from 0.0 (poor) to 1.0 (excellent). Consider: relevance, clarity, completeness, and authority. Respond with ONLY a number between 0.0 and 1.0."
            )

            # Prepare content sample (first 1000 chars for speed)
            content_sample = content[:1000]
            query_context = f" Query: {search_query}" if search_query else ""

            prompt = f"""Rate this content quality (0.0-1.0):
URL: {url}{query_context}
Content sample: {content_sample}...

Quality score:"""

            # Fast LLM call
            result = await agent.run(prompt, temperature=0.1, max_tokens=50)

            # Extract numeric score
            response = str(result.data or result.output or result).strip()

            # Extract number from response
            score_match = re.search(r'0\.\d+|1\.0|0\.0', response)
            if score_match:
                score = float(score_match.group())
                return max(0.0, min(1.0, score))
            else:
                logger.warning(f"Could not extract numeric score from GPT-5-nano: {response}")
                return 0.5

        except Exception as e:
            logger.warning(f"GPT-5-nano assessment failed: {e}")
            return 0.5  # Fallback score

    def _calculate_overall_confidence(self, signals: ConfidenceSignals) -> float:
        """
        Calculate overall confidence score using weighted combination.

        Simple weighted approach as specified in technical requirements.
        """
        weighted_score = (
            signals.content_length_score * self.weights['content_length'] +
            signals.structure_score * self.weights['structure'] +
            signals.relevance_score * self.weights['relevance'] +
            signals.cleanliness_score * self.weights['cleanliness'] +
            signals.domain_authority_score * self.weights['domain_authority'] +
            signals.freshness_score * self.weights['freshness'] +
            signals.extraction_confidence * self.weights['extraction_confidence']
        )

        # Blend with LLM assessment (weighted average)
        overall_confidence = (weighted_score * 0.7) + (signals.llm_assessment * 0.3)

        return max(0.0, min(1.0, overall_confidence))

    def get_editorial_recommendation(self, confidence_signals: ConfidenceSignals) -> str:
        """
        Get editorial recommendation based on confidence scores.

        Returns recommendation string based on threshold criteria.
        """
        confidence = confidence_signals.overall_confidence

        if confidence >= self.thresholds['good_quality']:
            return "ACCEPT_CONTENT"  # Good quality, accept as-is
        elif confidence >= self.thresholds['acceptable_quality']:
            return "ENHANCE_CONTENT"  # Acceptable quality, enhance if possible
        elif confidence >= self.thresholds['gap_research_trigger']:
            return "GAP_RESEARCH"  # Below trigger threshold, need gap research
        else:
            return "REJECT_CONTENT"  # Poor quality, reject

    def get_detailed_assessment(self, confidence_signals: ConfidenceSignals) -> Dict[str, Any]:
        """
        Get detailed assessment with explanations and recommendations.
        """
        recommendation = self.get_editorial_recommendation(confidence_signals)

        assessment = {
            'overall_confidence': confidence_signals.overall_confidence,
            'recommendation': recommendation,
            'thresholds_met': {
                'acceptable_quality': confidence_signals.overall_confidence >= self.thresholds['acceptable_quality'],
                'good_quality': confidence_signals.overall_confidence >= self.thresholds['good_quality'],
                'gap_research_needed': confidence_signals.overall_confidence < self.thresholds['gap_research_trigger']
            },
            'component_scores': {
                'content_length': confidence_signals.content_length_score,
                'structure': confidence_signals.structure_score,
                'relevance': confidence_signals.relevance_score,
                'cleanliness': confidence_signals.cleanliness_score,
                'domain_authority': confidence_signals.domain_authority_score,
                'freshness': confidence_signals.freshness_score,
                'extraction_confidence': confidence_signals.extraction_confidence,
                'llm_assessment': confidence_signals.llm_assessment
            },
            'processing_info': {
                'processing_time_ms': confidence_signals.processing_time_ms,
                'cache_hit': confidence_signals.cache_hit,
                'assessment_timestamp': confidence_signals.assessment_timestamp
            }
        }

        return assessment

    def clear_cache(self):
        """Clear the assessment cache."""
        if self.cache_enabled:
            self._cache.clear()
            self._cache_access_order.clear()
            logger.info("Confidence assessment cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_enabled:
            return {'cache_enabled': False}

        return {
            'cache_enabled': True,
            'cache_size': len(self._cache),
            'max_cache_size': self.cache_size,
            'cache_utilization': len(self._cache) / self.cache_size
        }