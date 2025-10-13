"""
Intelligent Ranking Algorithm for Enhanced URL Selection

This module implements a sophisticated multi-factor ranking algorithm that combines
search results from multiple streams into a unified master list with intelligent
scoring and diversity considerations.

Key Features:
- Multi-factor ranking algorithm combining multiple signals
- Search priority weighting (primary vs orthogonal streams)
- Query match assessment using existing relevance scoring
- Domain authority bonuses from existing system
- Domain diversity penalties for comprehensive coverage
- Configurable ranking parameters and weights
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from urllib.parse import urlparse

from .multi_stream_search import MultiSearchResults, SearchPriority
from .search_types import SearchResult
from .enhanced_relevance_scorer import calculate_domain_authority_boost

logger = logging.getLogger(__name__)


@dataclass
class RankedSearchResult:
    """Enhanced search result with ranking information."""
    search_result: SearchResult
    original_priority: SearchPriority
    original_position: int
    composite_score: float
    ranking_factors: Dict[str, float]
    domain_penalty: float = 0.0


@dataclass
class RankingConfig:
    """Configuration for ranking algorithm parameters."""
    # Priority weights
    primary_weight: float = 1.0
    orthogonal_weight: float = 0.7

    # Score component weights
    position_weight: float = 0.4
    relevance_weight: float = 0.3
    authority_weight: float = 0.2
    diversity_weight: float = 0.1

    # Domain diversity settings
    max_same_domain: int = 3
    domain_penalty_per_extra: float = 0.15

    # Position scoring settings
    position_decay_rate: float = 0.05
    position_decay_threshold: int = 3


class IntelligentRanker:
    """
    Intelligent ranking system that combines multiple search streams into
    a unified master list using sophisticated multi-factor scoring.

    The ranking algorithm considers:
    1. Search priority (primary vs orthogonal queries)
    2. SERP position scores with gentle decay
    3. Query relevance scores (from existing system)
    4. Domain authority bonuses (from existing system)
    5. Domain diversity penalties
    """

    def __init__(self, config: RankingConfig = None):
        """
        Initialize the intelligent ranker.

        Args:
            config: RankingConfig with custom parameters
        """
        self.config = config or RankingConfig()
        logger.info("Intelligent ranker initialized with config: "
                   f"primary_weight={self.config.primary_weight}, "
                   f"orthogonal_weight={self.config.orthogonal_weight}")

    def create_master_ranked_list(
        self,
        multi_search_results: MultiSearchResults,
        target_count: int = 50
    ) -> List[RankedSearchResult]:
        """
        Create a unified master ranked list from multi-search results.

        Args:
            multi_search_results: Results from multi-stream search
            target_count: Desired number of URLs in final list

        Returns:
            List of ranked search results
        """
        try:
            logger.info(f"Creating master ranked list with target_count={target_count}")

            # Step 1: Collect all results from all streams
            all_results = self._collect_all_results(multi_search_results)

            if not all_results:
                logger.warning("No search results to rank")
                return []

            logger.info(f"Collected {len(all_results)} total results from {len(multi_search_results.stream_results)} streams")

            # Step 2: Calculate initial scores without diversity penalties
            scored_results = self._calculate_initial_scores(all_results, multi_search_results)

            # Step 3: Apply domain diversity penalties
            diversified_results = self._apply_domain_diversity(scored_results)

            # Step 4: Sort by final composite scores
            final_results = sorted(diversified_results, key=lambda x: x.composite_score, reverse=True)

            # Step 5: Limit to target count
            final_results = final_results[:target_count]

            logger.info(f"Created master ranked list with {len(final_results)} results")
            self._log_ranking_summary(final_results)

            return final_results

        except Exception as e:
            logger.error(f"Failed to create master ranked list: {e}")
            return []

    def _collect_all_results(self, multi_search_results: MultiSearchResults) -> List[Tuple[SearchResult, SearchPriority, int]]:
        """
        Collect all search results from all streams with their metadata.

        Args:
            multi_search_results: Multi-search results

        Returns:
            List of (result, priority, original_position) tuples
        """
        all_results = []

        for priority, stream_result in multi_search_results.stream_results.items():
            if not stream_result.success:
                logger.warning(f"Skipping failed stream {priority.value}: {stream_result.error_message}")
                continue

            logger.debug(f"Processing {len(stream_result.results)} results from {priority.value} stream")

            for i, result in enumerate(stream_result.results):
                # Store original position from this stream
                original_position = i + 1
                all_results.append((result, priority, original_position))

        return all_results

    def _calculate_initial_scores(
        self,
        all_results: List[Tuple[SearchResult, SearchPriority, int]],
        multi_search_results: MultiSearchResults
    ) -> List[RankedSearchResult]:
        """
        Calculate initial composite scores for all results.

        Args:
            all_results: List of (result, priority, position) tuples
            multi_search_results: Original multi-search results for context

        Returns:
            List of RankedSearchResult objects
        """
        scored_results = []

        for search_result, priority, original_position in all_results:
            try:
                # Calculate individual score components
                position_score = self._calculate_position_score(original_position, priority)
                relevance_score = float(search_result.relevance_score)
                authority_score = self._calculate_authority_score(search_result.link)

                # Apply priority weight
                priority_weight = self._get_priority_weight(priority)

                # Calculate composite score
                composite_score = (
                    position_score * self.config.position_weight +
                    relevance_score * self.config.relevance_weight +
                    authority_score * self.config.authority_weight
                ) * priority_weight

                # Create ranked result
                ranked_result = RankedSearchResult(
                    search_result=search_result,
                    original_priority=priority,
                    original_position=original_position,
                    composite_score=composite_score,
                    ranking_factors={
                        "position_score": position_score,
                        "relevance_score": relevance_score,
                        "authority_score": authority_score,
                        "priority_weight": priority_weight,
                        "raw_composite": composite_score / priority_weight
                    }
                )

                scored_results.append(ranked_result)

            except Exception as e:
                logger.warning(f"Failed to score result {search_result.link}: {e}")
                continue

        logger.debug(f"Calculated initial scores for {len(scored_results)} results")
        return scored_results

    def _apply_domain_diversity(self, scored_results: List[RankedSearchResult]) -> List[RankedSearchResult]:
        """
        Apply domain diversity penalties to encourage variety in sources.

        Args:
            scored_results: List of already scored results

        Returns:
            List of results with diversity penalties applied
        """
        domain_counts = {}
        domain_penalties = {}

        # First pass: count domains
        for result in scored_results:
            domain = self._extract_domain(result.search_result.link)
            if domain not in domain_counts:
                domain_counts[domain] = 0
            domain_counts[domain] += 1

        # Second pass: calculate penalties
        for domain, count in domain_counts.items():
            if count > self.config.max_same_domain:
                excess_count = count - self.config.max_same_domain
                penalty = excess_count * self.config.domain_penalty_per_extra
                domain_penalties[domain] = penalty
                logger.debug(f"Domain {domain} has {count} results, applying penalty of {penalty:.3f}")

        # Third pass: apply penalties
        for result in scored_results:
            domain = self._extract_domain(result.search_result.link)
            penalty = domain_penalties.get(domain, 0.0)
            result.domain_penalty = penalty
            result.composite_score -= penalty
            result.ranking_factors["domain_penalty"] = penalty

        logger.debug(f"Applied domain diversity penalties to {len(scored_results)} results")
        return scored_results

    def _calculate_position_score(self, position: int, priority: SearchPriority) -> float:
        """
        Calculate position score with gentle decay logic.

        Args:
            position: Original position in search results (1-based)
            priority: Search priority stream

        Returns:
            Position score between 0.0 and 1.0
        """
        # Gentle decay for multi-query collation
        if position <= self.config.position_decay_threshold:
            # Top positions: no decay
            return 1.0
        else:
            # Calculate decay groups
            decay_groups = (position - self.config.position_decay_threshold - 1) // 3
            score = 1.0 - (decay_groups * self.config.position_decay_rate)
            return max(0.05, score)  # Minimum score to prevent complete exclusion

    def _calculate_authority_score(self, url: str) -> float:
        """
        Calculate domain authority score using existing system.

        Args:
            url: URL to analyze

        Returns:
            Authority score between 0.0 and 1.0
        """
        try:
            return calculate_domain_authority_boost(url)
        except Exception as e:
            logger.debug(f"Failed to calculate authority score for {url}: {e}")
            return 0.0

    def _get_priority_weight(self, priority: SearchPriority) -> float:
        """
        Get priority weight based on search stream type.

        Args:
            priority: Search priority type

        Returns:
            Priority weight multiplier
        """
        if priority == SearchPriority.PRIMARY:
            return self.config.primary_weight
        else:
            return self.config.orthogonal_weight

    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL for diversity analysis.

        Args:
            url: URL to extract domain from

        Returns:
            Domain string
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www prefix for consistent domain matching
            if domain.startswith('www.'):
                domain = domain[4:]

            return domain
        except Exception as e:
            logger.debug(f"Failed to extract domain from {url}: {e}")
            return "unknown"

    def _log_ranking_summary(self, final_results: List[RankedSearchResult]):
        """Log summary statistics of the final ranking."""
        if not final_results:
            return

        # Priority distribution
        priority_counts = {}
        for result in final_results:
            priority = result.original_priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        # Score statistics
        scores = [result.composite_score for result in final_results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)

        # Domain diversity
        domains = set()
        for result in final_results:
            domains.add(self._extract_domain(result.search_result.link))

        logger.info(f"Ranking summary - Total: {len(final_results)}, "
                   f"Avg score: {avg_score:.3f}, "
                   f"Score range: {min_score:.3f}-{max_score:.3f}, "
                   f"Unique domains: {len(domains)}")
        logger.info(f"Priority distribution: {priority_counts}")

    def get_ranking_statistics(self, ranked_results: List[RankedSearchResult]) -> Dict[str, any]:
        """
        Get comprehensive statistics about the ranking results.

        Args:
            ranked_results: List of ranked search results

        Returns:
            Dictionary with ranking statistics
        """
        if not ranked_results:
            return {"error": "No results to analyze"}

        # Basic statistics
        scores = [result.composite_score for result in ranked_results]
        priorities = [result.original_priority.value for result in ranked_results]
        domains = [self._extract_domain(result.search_result.link) for result in ranked_results]

        # Calculate statistics
        stats = {
            "total_results": len(ranked_results),
            "score_statistics": {
                "average": sum(scores) / len(scores),
                "maximum": max(scores),
                "minimum": min(scores),
                "median": sorted(scores)[len(scores) // 2]
            },
            "priority_distribution": {},
            "domain_statistics": {
                "unique_domains": len(set(domains)),
                "most_common_domain": max(set(domains), key=domains.count) if domains else None,
                "domain_diversity_ratio": len(set(domains)) / len(domains) if domains else 0
            },
            "ranking_factors": {
                "position_scores": [result.ranking_factors.get("position_score", 0) for result in ranked_results],
                "relevance_scores": [result.ranking_factors.get("relevance_score", 0) for result in ranked_results],
                "authority_scores": [result.ranking_factors.get("authority_score", 0) for result in ranked_results],
                "domain_penalties": [result.domain_penalty for result in ranked_results]
            }
        }

        # Priority distribution
        for priority in set(priorities):
            stats["priority_distribution"][priority] = priorities.count(priority)

        return stats


# Global ranker instance for reuse
_intelligent_ranker = None

def get_intelligent_ranker(config: RankingConfig = None) -> IntelligentRanker:
    """
    Get or create an intelligent ranker instance.

    Args:
        config: Custom ranking configuration

    Returns:
        IntelligentRanker instance
    """
    global _intelligent_ranker

    if _intelligent_ranker is None:
        _intelligent_ranker = IntelligentRanker(config)

    return _intelligent_ranker


def create_master_ranked_list(
    multi_search_results: MultiSearchResults,
    target_count: int = 50,
    config: RankingConfig = None
) -> List[RankedSearchResult]:
    """
    Convenience function to create master ranked list.

    Args:
        multi_search_results: Results from multi-stream search
        target_count: Desired number of URLs
        config: Custom ranking configuration

    Returns:
        List of ranked search results
    """
    ranker = get_intelligent_ranker(config)
    return ranker.create_master_ranked_list(multi_search_results, target_count)