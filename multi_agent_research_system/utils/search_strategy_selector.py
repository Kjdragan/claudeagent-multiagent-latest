"""
Search Strategy Auto-Selection System

Implements intelligent search strategy selection based on query analysis,
timing, and topic characteristics from the technical documentation.

Features:
- Google Search vs SERP News API routing
- Time-based and topic-based decision logic
- Query analysis for optimal search strategy
- Performance tracking and optimization
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategy options."""
    GOOGLE_SEARCH = "search"
    SERP_NEWS = "news"
    HYBRID = "hybrid"  # For complex queries needing both


@dataclass
class StrategyAnalysis:
    """Analysis result for search strategy selection."""
    recommended_strategy: SearchStrategy
    confidence: float
    reasoning: List[str]
    time_factor: float
    topic_factor: float
    query_factor: float


class SearchStrategySelector:
    """
    Intelligent search strategy selector based on query characteristics and timing.

    Algorithm:
    1. Analyze query for news indicators (time-sensitive topics)
    2. Check recency factors (breaking news, recent events)
    3. Evaluate topic characteristics (ongoing vs historical)
    4. Apply confidence scoring and reasoning
    5. Recommend optimal search strategy
    """

    def __init__(self):
        """Initialize the search strategy selector."""
        # News indicators that suggest news search
        self.news_keywords = {
            # Time-sensitive terms
            'latest', 'recent', 'breaking', 'new', 'update', 'current', 'today',
            'yesterday', 'this week', 'this month', '2024', '2025',

            # News events
            'election', 'elections', 'vote', 'campaign', 'poll', 'debate',
            'war', 'conflict', 'attack', 'strike', 'protest', 'riot',
            'market', 'stock', 'economy', 'inflation', 'recession',
            'court', 'ruling', 'verdict', 'legal', 'lawsuit', 'trial',
            'government', 'shutdown', 'policy', 'bill', 'law', 'regulation',
            'weather', 'storm', 'hurricane', 'earthquake', 'flood', 'fire',
            'sports', 'game', 'match', 'tournament', 'championship', 'score',
            'technology', 'launch', 'release', 'update', 'announcement',
            'health', 'outbreak', 'pandemic', 'vaccine', 'disease', 'covid',

            # Organizations and people
            'president', 'congress', 'senate', 'parliament', 'un', 'nato',
            'apple', 'google', 'microsoft', 'amazon', 'meta', 'tesla',
            'federal reserve', 'fed', 'supreme court', 'cdc', 'who'
        }

        # Research/academic indicators that suggest general search
        self.research_keywords = {
            'research', 'study', 'analysis', 'review', 'history', 'development',
            'evolution', 'theory', 'framework', 'methodology', 'approach',
            'comparison', 'overview', 'introduction', 'background', 'basics',
            'tutorial', 'guide', 'how to', 'best practices', 'principles'
        }

        # High-authority domains that work well with general search
        self.authority_domains = {
            'edu', 'gov', 'mil', 'org', 'academic', 'university', 'institute',
            'research', 'journal', 'publication', 'study', 'paper'
        }

        # Recency weight decay factors
        self.recency_weights = {
            'breaking': 1.0,
            'latest': 0.9,
            'recent': 0.8,
            'current': 0.7,
            'new': 0.6,
            'today': 0.9,
            'yesterday': 0.7,
            'this week': 0.6,
            'this month': 0.5
        }

    def select_search_strategy(
        self,
        query: str,
        current_time: Optional[datetime] = None,
        context: Optional[Dict] = None
    ) -> StrategyAnalysis:
        """
        Select the optimal search strategy for the given query.

        Args:
            query: Search query string
            current_time: Current time for temporal analysis
            context: Additional context (session info, previous searches, etc.)

        Returns:
            StrategyAnalysis with recommendation and reasoning
        """
        if current_time is None:
            current_time = datetime.now()

        # Analyze query characteristics
        time_factor = self._analyze_time_sensitivity(query, current_time)
        topic_factor = self._analyze_topic_characteristics(query)
        query_factor = self._analyze_query_structure(query)

        # Calculate overall confidence for news search
        news_confidence = (time_factor * 0.4 + topic_factor * 0.4 + query_factor * 0.2)

        # Determine strategy and reasoning
        reasoning = []

        if news_confidence >= 0.7:
            strategy = SearchStrategy.SERP_NEWS
            confidence = news_confidence
            reasoning.append("High news relevance detected")
        elif news_confidence <= 0.3:
            strategy = SearchStrategy.GOOGLE_SEARCH
            confidence = 1.0 - news_confidence
            reasoning.append("Research/academic query detected")
        else:
            # Mixed signals - use hybrid approach
            strategy = SearchStrategy.HYBRID
            confidence = 0.5
            reasoning.append("Mixed query characteristics - hybrid approach recommended")

        # Add detailed reasoning
        if time_factor > 0.6:
            reasoning.append(f"Time-sensitive query (factor: {time_factor:.2f})")
        if topic_factor > 0.6:
            reasoning.append(f"News-oriented topic (factor: {topic_factor:.2f})")
        if query_factor > 0.6:
            reasoning.append(f"News-specific language (factor: {query_factor:.2f})")

        logger.info(f"Search strategy selected: {strategy.value} (confidence: {confidence:.2f}) "
                   f"for query: '{query[:50]}...'")

        return StrategyAnalysis(
            recommended_strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            time_factor=time_factor,
            topic_factor=topic_factor,
            query_factor=query_factor
        )

    def _analyze_time_sensitivity(self, query: str, current_time: datetime) -> float:
        """
        Analyze time sensitivity of the query.

        Returns:
            Float between 0.0 (not time-sensitive) and 1.0 (highly time-sensitive)
        """
        query_lower = query.lower()
        time_score = 0.0

        # Check for explicit time indicators
        for keyword, weight in self.recency_weights.items():
            if keyword in query_lower:
                time_score = max(time_score, weight)

        # Check for years (especially current/recent years)
        current_year = current_time.year
        year_pattern = r'\b(20[2-9][0-9])\b'
        years_in_query = re.findall(year_pattern, query)

        for year in years_in_query:
            year_int = int(year)
            if year_int == current_year:
                time_score = max(time_score, 0.8)
            elif year_int == current_year - 1:
                time_score = max(time_score, 0.6)
            elif year_int >= current_year - 2:
                time_score = max(time_score, 0.4)

        # Check for months/seasons indicating recency
        recent_months = ['january', 'february', 'march', 'april', 'may', 'june']
        if any(month in query_lower for month in recent_months):
            time_score = max(time_score, 0.5)

        return min(time_score, 1.0)

    def _analyze_topic_characteristics(self, query: str) -> float:
        """
        Analyze topic characteristics for news vs research orientation.

        Returns:
            Float between 0.0 (research-oriented) and 1.0 (news-oriented)
        """
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)

        # Count news and research keywords
        news_keyword_count = sum(1 for word in words if word in self.news_keywords)
        research_keyword_count = sum(1 for word in words if word in self.research_keywords)

        # Calculate topic score
        total_keywords = news_keyword_count + research_keyword_count
        if total_keywords == 0:
            return 0.5  # Neutral

        news_ratio = news_keyword_count / total_keywords
        return news_ratio

    def _analyze_query_structure(self, query: str) -> float:
        """
        Analyze query structure for news vs research indicators.

        Returns:
            Float between 0.0 (research structure) and 1.0 (news structure)
        """
        query_lower = query.lower()

        # News question patterns
        news_patterns = [
            r'\bwhat\b.*\bhappened\b',
            r'\bwhat\b.*\bis\b.*\bgoing\b',
            r'\bwhen\b.*\bdid\b',
            r'\bwho\b.*\bsaid\b',
            r'\bhow\b.*\bmany\b',
            r'\bwhere\b.*\bis\b'
        ]

        # Research question patterns
        research_patterns = [
            r'\bwhat\b.*\bis\b',
            r'\bhow\b.*\bdoes\b',
            r'\bwhy\b.*\bis\b',
            r'\bcan\b.*\bexplain\b',
            r'\bdefine\b',
            r'\bcompare\b'
        ]

        news_pattern_score = sum(1 for pattern in news_patterns if re.search(pattern, query_lower))
        research_pattern_score = sum(1 for pattern in research_patterns if re.search(pattern, query_lower))

        # Check for entity mentions (people, organizations, places)
        entity_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Person names
            r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b',  # Proper nouns
        ]
        entity_count = sum(len(re.findall(pattern, query)) for pattern in entity_patterns)

        # Entity mentions suggest news context
        entity_score = min(entity_count * 0.2, 0.8)

        # Combine scores
        total_indicators = news_pattern_score + research_pattern_score
        if total_indicators == 0:
            return entity_score

        news_ratio = (news_pattern_score + entity_score) / (total_indicators + entity_score)
        return min(news_ratio, 1.0)

    def get_hybrid_strategy_config(
        self,
        query: str,
        analysis: StrategyAnalysis
    ) -> Dict[str, Any]:
        """
        Get configuration for hybrid search strategy.

        Args:
            query: Original search query
            analysis: Strategy analysis result

        Returns:
            Configuration dictionary for hybrid search execution
        """
        # Determine split based on confidence factors
        if analysis.time_factor > 0.7:
            # Heavy on news, light on general search
            news_weight = 0.7
            search_weight = 0.3
        elif analysis.topic_factor > 0.7:
            # Heavy on news
            news_weight = 0.6
            search_weight = 0.4
        else:
            # Balanced approach
            news_weight = 0.5
            search_weight = 0.5

        return {
            'strategy': 'hybrid',
            'news_weight': news_weight,
            'search_weight': search_weight,
            'news_query': self._optimize_query_for_news(query),
            'search_query': self._optimize_query_for_search(query),
            'total_results': 15,  # Default, can be overridden
            'news_results': int(15 * news_weight),
            'search_results': int(15 * search_weight)
        }

    def _optimize_query_for_news(self, query: str) -> str:
        """Optimize query for news search."""
        # Add news-related terms if not present
        query_lower = query.lower()
        news_terms = ['latest', 'recent', 'breaking', 'current']

        has_news_term = any(term in query_lower for term in news_terms)

        if not has_news_term and len(query.split()) <= 5:
            # Add a news term for better news results
            return f"latest {query}"

        return query

    def _optimize_query_for_search(self, query: str) -> str:
        """Optimize query for general search."""
        # Remove news-specific terms that might reduce general search quality
        query_lower = query.lower()
        news_terms_to_remove = ['breaking', 'latest', 'current', 'today', 'recent']

        optimized_query = query
        for term in news_terms_to_remove:
            # Remove term only if it's at the beginning or standalone
            if optimized_query.lower().startswith(term + ' '):
                optimized_query = optimized_query[len(term) + 1:]
            elif f' {term} ' in f' {optimized_query.lower()} ':
                optimized_query = optimized_query.lower().replace(f' {term} ', ' ')

        return optimized_query.strip()

    def analyze_search_performance(
        self,
        strategy: SearchStrategy,
        success_rate: float,
        result_quality: float,
        processing_time: float
    ) -> Dict[str, Any]:
        """
        Analyze performance of a search strategy for optimization.

        Args:
            strategy: The search strategy used
            success_rate: Success rate (0.0-1.0)
            result_quality: Quality of results (0.0-1.0)
            processing_time: Processing time in seconds

        Returns:
            Performance analysis and recommendations
        """
        performance_score = (success_rate * 0.4 + result_quality * 0.4 +
                           min(1.0, 10.0 / processing_time) * 0.2)

        recommendations = []

        if success_rate < 0.7:
            recommendations.append(f"Consider switching from {strategy.value} - low success rate")

        if result_quality < 0.6:
            recommendations.append(f"Query optimization needed for {strategy.value}")

        if processing_time > 30:
            recommendations.append(f"Consider faster alternatives to {strategy.value}")

        return {
            'strategy': strategy.value,
            'performance_score': performance_score,
            'success_rate': success_rate,
            'result_quality': result_quality,
            'processing_time': processing_time,
            'recommendations': recommendations,
            'overall_rating': self._get_performance_rating(performance_score)
        }

    def _get_performance_rating(self, score: float) -> str:
        """Get performance rating based on score."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Acceptable"
        elif score >= 0.6:
            return "Needs Improvement"
        else:
            return "Poor"


# Global strategy selector instance
_global_strategy_selector: Optional[SearchStrategySelector] = None


def get_search_strategy_selector() -> SearchStrategySelector:
    """Get or create global search strategy selector."""
    global _global_strategy_selector
    if _global_strategy_selector is None:
        _global_strategy_selector = SearchStrategySelector()
    return _global_strategy_selector