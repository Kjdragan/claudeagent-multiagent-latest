"""
Enhanced URL Selector - Main Entry Point for Intelligent URL Selection

This module provides the main interface for the enhanced URL selection system,
replacing the simple relevance-threshold approach with sophisticated multi-query
optimization and intelligent ranking.

Key Features:
- Drop-in replacement for existing select_urls_for_crawling() function
- Complete workflow from user query to ranked URL list
- Integration with existing scraping/cleaning pipeline
- Comprehensive error handling and fallback mechanisms
- Performance monitoring and statistics tracking
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .query_enhancer import enhance_user_query, EnhancedQueries
from .multi_stream_search import execute_multi_stream_search, MultiSearchResults
from .intelligent_ranker import create_master_ranked_list, RankedSearchResult, RankingConfig

logger = logging.getLogger(__name__)


@dataclass
class URLSelectionResult:
    """Result from enhanced URL selection process."""
    urls: List[str]
    selection_metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class EnhancedURLSelector:
    """
    Main class for enhanced URL selection using intelligent multi-query optimization
    and sophisticated ranking algorithms.

    This class orchestrates the complete workflow:
    1. Query enhancement with GPT-5 Nano
    2. Multi-stream parallel search execution
    3. Intelligent ranking and master list creation
    4. Final URL list generation for scraping pipeline
    """

    def __init__(self, ranking_config: RankingConfig = None):
        """
        Initialize the enhanced URL selector.

        Args:
            ranking_config: Custom ranking configuration
        """
        self.ranking_config = ranking_config
        logger.info("Enhanced URL selector initialized")

    async def select_urls_for_crawling(
        self,
        query: str,
        session_id: str = "default",
        target_count: int = 50,
        search_type: str = "search",
        result_distribution: Optional[Dict[str, int]] = None
    ) -> URLSelectionResult:
        """
        Enhanced URL selection using intelligent multi-query optimization.

        This is the main entry point that replaces the simple threshold-based
        URL selection with a sophisticated multi-factor approach.

        Args:
            query: Original user research query
            session_id: Session identifier for tracking
            target_count: Desired number of URLs in final list
            search_type: Type of search (search or news)
            result_distribution: Custom result distribution per search stream

        Returns:
            URLSelectionResult with ranked URLs and metadata
        """
        start_time = time.time()

        try:
            logger.info(f"Starting enhanced URL selection for session {session_id}: '{query}'")

            # Step 1: Query Enhancement
            logger.info("Step 1: Enhancing user query with GPT-5 Nano")
            enhanced_queries = await enhance_user_query(query, session_id)
            query_enhancement_time = time.time() - start_time

            # Step 2: Multi-Stream Search Execution
            logger.info("Step 2: Executing multi-stream parallel searches")
            multi_search_results = await execute_multi_stream_search(
                enhanced_queries=enhanced_queries,
                session_id=session_id,
                search_type=search_type,
                result_distribution=result_distribution
            )
            search_execution_time = time.time() - start_time - query_enhancement_time

            # Step 3: Intelligent Ranking
            logger.info("Step 3: Creating master ranked list with intelligent scoring")
            ranked_results = create_master_ranked_list(
                multi_search_results=multi_search_results,
                target_count=target_count,
                config=self.ranking_config
            )
            ranking_time = time.time() - start_time - query_enhancement_time - search_execution_time

            # Step 4: Extract Final URLs
            final_urls = [result.search_result.link for result in ranked_results]

            # Calculate performance metrics
            total_execution_time = time.time() - start_time

            # Create comprehensive result
            selection_result = URLSelectionResult(
                urls=final_urls,
                selection_metadata={
                    "original_query": query,
                    "enhanced_queries": {
                        "primary": enhanced_queries.primary_query,
                        "orthogonal_1": enhanced_queries.orthogonal_query_1,
                        "orthogonal_2": enhanced_queries.orthogonal_query_2
                    },
                    "target_count": target_count,
                    "actual_count": len(final_urls),
                    "search_type": search_type,
                    "result_distribution": result_distribution,
                    "multi_search_results": multi_search_results.metadata,
                    "ranking_config": {
                        "primary_weight": self.ranking_config.primary_weight if self.ranking_config else 1.0,
                        "orthogonal_weight": self.ranking_config.orthogonal_weight if self.ranking_config else 0.7
                    }
                },
                performance_metrics={
                    "total_execution_time": total_execution_time,
                    "query_enhancement_time": query_enhancement_time,
                    "search_execution_time": search_execution_time,
                    "ranking_time": ranking_time,
                    "urls_per_second": len(final_urls) / total_execution_time if total_execution_time > 0 else 0,
                    "successful_streams": multi_search_results.successful_streams,
                    "failed_streams": multi_search_results.failed_streams,
                    "total_search_results": multi_search_results.total_results
                },
                success=True
            )

            logger.info(f"Enhanced URL selection completed for session {session_id}: "
                       f"{len(final_urls)} URLs selected in {total_execution_time:.2f}s")

            return selection_result

        except Exception as e:
            total_execution_time = time.time() - start_time
            error_message = f"Enhanced URL selection failed: {str(e)}"
            logger.error(f"{error_message} (after {total_execution_time:.2f}s)")

            # Return failure result
            return URLSelectionResult(
                urls=[],
                selection_metadata={
                    "original_query": query,
                    "session_id": session_id,
                    "target_count": target_count
                },
                performance_metrics={
                    "total_execution_time": total_execution_time,
                    "error_occurred": True
                },
                success=False,
                error_message=error_message
            )

    async def select_urls_with_fallback(
        self,
        query: str,
        session_id: str = "default",
        target_count: int = 50,
        search_type: str = "search"
    ) -> URLSelectionResult:
        """
        Enhanced URL selection with fallback to traditional method.

        Args:
            query: Original user research query
            session_id: Session identifier for tracking
            target_count: Desired number of URLs
            search_type: Type of search

        Returns:
            URLSelectionResult with enhanced or fallback URLs
        """
        try:
            # Try enhanced selection first
            result = await self.select_urls_for_crawling(
                query=query,
                session_id=session_id,
                target_count=target_count,
                search_type=search_type
            )

            if result.success and len(result.urls) >= min(10, target_count // 2):
                logger.info(f"Enhanced selection successful for session {session_id}")
                return result
            else:
                logger.warning(f"Enhanced selection insufficient for session {session_id}, using fallback")
                return await self._fallback_selection(query, session_id, target_count, search_type)

        except Exception as e:
            logger.error(f"Enhanced selection failed for session {session_id}, using fallback: {e}")
            return await self._fallback_selection(query, session_id, target_count, search_type)

    async def _fallback_selection(
        self,
        query: str,
        session_id: str,
        target_count: int,
        search_type: str
    ) -> URLSelectionResult:
        """
        Fallback selection using traditional approach.

        Args:
            query: Original query
            session_id: Session identifier
            target_count: Target number of URLs
            search_type: Type of search

        Returns:
            URLSelectionResult with fallback URLs
        """
        try:
            logger.info(f"Using fallback selection for session {session_id}")
            start_time = time.time()

            # Use existing SERP search functionality as fallback
            from .z_search_crawl_utils import execute_serper_search

            # Execute single search
            search_results = await execute_serper_search(
                query=query,
                search_type=search_type,
                num_results=min(target_count * 2, 100)  # Get more to allow for filtering
            )

            # Use existing selection logic
            fallback_urls = select_urls_for_crawling(
                search_results=search_results,
                limit=target_count,
                min_relevance=0.3,
                session_id=session_id
            )

            execution_time = time.time() - start_time

            return URLSelectionResult(
                urls=fallback_urls,
                selection_metadata={
                    "original_query": query,
                    "fallback_used": True,
                    "target_count": target_count,
                    "actual_count": len(fallback_urls)
                },
                performance_metrics={
                    "total_execution_time": execution_time,
                    "fallback_mode": True
                },
                success=True
            )

        except Exception as e:
            logger.error(f"Fallback selection also failed for session {session_id}: {e}")
            return URLSelectionResult(
                urls=[],
                selection_metadata={"original_query": query, "session_id": session_id},
                performance_metrics={"total_execution_time": 0, "complete_failure": True},
                success=False,
                error_message=f"Both enhanced and fallback selection failed: {str(e)}"
            )


# Global selector instance for reuse
_enhanced_selector = None

def get_enhanced_url_selector(ranking_config: RankingConfig = None) -> EnhancedURLSelector:
    """
    Get or create an enhanced URL selector instance.

    Args:
        ranking_config: Custom ranking configuration

    Returns:
        EnhancedURLSelector instance
    """
    global _enhanced_selector

    if _enhanced_selector is None:
        _enhanced_selector = EnhancedURLSelector(ranking_config)

    return _enhanced_selector


async def enhanced_select_urls_for_crawling(
    query: str,
    session_id: str = "default",
    target_count: int = 50,
    search_type: str = "search",
    result_distribution: Optional[Dict[str, int]] = None,
    use_fallback: bool = True
) -> List[str]:
    """
    Enhanced URL selection function - drop-in replacement for select_urls_for_crawling().

    This function provides the same interface as the original select_urls_for_crawling
    but uses the sophisticated multi-query optimization and intelligent ranking.

    Args:
        query: Original user research query
        session_id: Session identifier for tracking
        target_count: Desired number of URLs (default: 50)
        search_type: Type of search (search or news)
        result_distribution: Custom result distribution per stream
        use_fallback: Whether to use fallback on failure

    Returns:
        List of URLs ready for scraping pipeline
    """
    selector = get_enhanced_url_selector()

    if use_fallback:
        result = await selector.select_urls_with_fallback(
            query=query,
            session_id=session_id,
            target_count=target_count,
            search_type=search_type
        )
    else:
        result = await selector.select_urls_for_crawling(
            query=query,
            session_id=session_id,
            target_count=target_count,
            search_type=search_type,
            result_distribution=result_distribution
        )

    if result.success:
        logger.info(f"Enhanced URL selection successful: {len(result.urls)} URLs selected")
        return result.urls
    else:
        logger.error(f"Enhanced URL selection failed: {result.error_message}")
        return []


# Performance monitoring functions
def get_selection_statistics(result: URLSelectionResult) -> Dict[str, Any]:
    """
    Get comprehensive statistics from URL selection result.

    Args:
        result: URLSelectionResult to analyze

    Returns:
        Dictionary with selection statistics
    """
    if not result.success:
        return {
            "success": False,
            "error_message": result.error_message,
            "performance_metrics": result.performance_metrics
        }

    stats = {
        "success": True,
        "url_count": len(result.urls),
        "selection_metadata": result.selection_metadata,
        "performance_metrics": result.performance_metrics,
        "efficiency_metrics": {
            "urls_per_second": result.performance_metrics.get("urls_per_second", 0),
            "success_rate": 1.0,
            "stream_success_rate": (
                result.performance_metrics.get("successful_streams", 0) /
                (result.performance_metrics.get("successful_streams", 0) + result.performance_metrics.get("failed_streams", 0))
                if (result.performance_metrics.get("successful_streams", 0) + result.performance_metrics.get("failed_streams", 0)) > 0 else 0
            )
        }
    }

    return stats