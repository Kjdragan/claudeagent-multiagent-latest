"""
Multi-Stream Search Executor for Enhanced URL Selection

This module executes parallel SERP API searches for multiple queries,
coordinating the execution and handling errors gracefully.

Key Features:
- Parallel execution of multiple search queries
- Error handling and fallback mechanisms
- Integration with existing SERP search infrastructure
- Configurable search parameters and result distribution
- Comprehensive logging and monitoring
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

from .query_enhancer import EnhancedQueries
from .z_search_crawl_utils import execute_serper_search
from .search_types import SearchResult

logger = logging.getLogger(__name__)


class SearchPriority(Enum):
    """Search priority levels for different query types."""
    PRIMARY = "primary"
    ORTHOGONAL_1 = "orthogonal_1"
    ORTHOGONAL_2 = "orthogonal_2"


@dataclass
class SearchRequest:
    """Configuration for a single search request."""
    query: str
    priority: SearchPriority
    num_results: int
    search_type: str = "search"
    weight: float = 1.0


@dataclass
class SearchStreamResult:
    """Results from a single search stream."""
    priority: SearchPriority
    query: str
    results: List[SearchResult]
    num_requested: int
    num_received: int
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class MultiSearchResults:
    """Combined results from all search streams."""
    stream_results: Dict[SearchPriority, SearchStreamResult]
    total_results: int
    successful_streams: int
    failed_streams: int
    total_execution_time: float
    metadata: Dict[str, Any]


class MultiStreamSearchExecutor:
    """
    Executes multiple parallel searches and coordinates the results.

    This class manages the execution of multiple search queries in parallel,
    handling errors, timing, and result collection efficiently.
    """

    def __init__(self, max_concurrent_searches: int = 3):
        """
        Initialize the multi-stream search executor.

        Args:
            max_concurrent_searches: Maximum number of concurrent search operations
        """
        self.max_concurrent_searches = max_concurrent_searches
        logger.info(f"Multi-stream search executor initialized with max_concurrent={max_concurrent_searches}")

    async def execute_multi_search(
        self,
        enhanced_queries: EnhancedQueries,
        session_id: str = "default",
        search_type: str = "search",
        result_distribution: Optional[Dict[str, int]] = None
    ) -> MultiSearchResults:
        """
        Execute parallel searches for enhanced queries.

        Args:
            enhanced_queries: EnhancedQueries object with three optimized queries
            session_id: Session identifier for tracking
            search_type: Type of search (search or news)
            result_distribution: Custom distribution of results per stream

        Returns:
            MultiSearchResults object with combined search results
        """
        import time
        start_time = time.time()

        try:
            logger.info(f"Starting multi-stream search for session {session_id}")

            # Default result distribution: 30 primary, 10 orthogonal each
            if result_distribution is None:
                result_distribution = {
                    "primary": 30,
                    "orthogonal_1": 10,
                    "orthogonal_2": 10
                }

            # Create search requests
            search_requests = self._create_search_requests(
                enhanced_queries, search_type, result_distribution
            )

            # Execute searches in parallel
            stream_results = await self._execute_parallel_searches(
                search_requests, session_id
            )

            # Calculate execution metrics
            total_execution_time = time.time() - start_time
            successful_streams = sum(1 for result in stream_results.values() if result.success)
            failed_streams = len(stream_results) - successful_streams
            total_results = sum(len(result.results) for result in stream_results.values())

            # Create combined results object
            multi_results = MultiSearchResults(
                stream_results=stream_results,
                total_results=total_results,
                successful_streams=successful_streams,
                failed_streams=failed_streams,
                total_execution_time=total_execution_time,
                metadata={
                    "session_id": session_id,
                    "search_type": search_type,
                    "result_distribution": result_distribution,
                    "enhanced_queries": enhanced_queries,
                    "max_concurrent_searches": self.max_concurrent_searches
                }
            )

            logger.info(f"Multi-stream search completed for session {session_id}: "
                       f"{total_results} total results, {successful_streams}/{len(stream_results)} streams successful "
                       f"in {total_execution_time:.2f}s")

            return multi_results

        except Exception as e:
            logger.error(f"Multi-stream search failed for session {session_id}: {e}")
            # Return empty results on failure
            return MultiSearchResults(
                stream_results={},
                total_results=0,
                successful_streams=0,
                failed_streams=3,
                total_execution_time=time.time() - start_time,
                metadata={
                    "session_id": session_id,
                    "error": str(e),
                    "enhanced_queries": enhanced_queries
                }
            )

    def _create_search_requests(
        self,
        enhanced_queries: EnhancedQueries,
        search_type: str,
        result_distribution: Dict[str, int]
    ) -> List[SearchRequest]:
        """
        Create search request objects from enhanced queries.

        Args:
            enhanced_queries: EnhancedQueries object
            search_type: Type of search
            result_distribution: Distribution of results per stream

        Returns:
            List of SearchRequest objects
        """
        requests = []

        # Primary search request (highest priority)
        requests.append(SearchRequest(
            query=enhanced_queries.primary_query,
            priority=SearchPriority.PRIMARY,
            num_results=result_distribution.get("primary", 30),
            search_type=search_type,
            weight=1.0
        ))

        # Orthogonal search request 1 (medium priority)
        requests.append(SearchRequest(
            query=enhanced_queries.orthogonal_query_1,
            priority=SearchPriority.ORTHOGONAL_1,
            num_results=result_distribution.get("orthogonal_1", 10),
            search_type=search_type,
            weight=0.7
        ))

        # Orthogonal search request 2 (medium priority)
        requests.append(SearchRequest(
            query=enhanced_queries.orthogonal_query_2,
            priority=SearchPriority.ORTHOGONAL_2,
            num_results=result_distribution.get("orthogonal_2", 10),
            search_type=search_type,
            weight=0.7
        ))

        logger.debug(f"Created {len(requests)} search requests")
        return requests

    async def _execute_parallel_searches(
        self,
        search_requests: List[SearchRequest],
        session_id: str
    ) -> Dict[SearchPriority, SearchStreamResult]:
        """
        Execute search requests in parallel.

        Args:
            search_requests: List of search requests
            session_id: Session identifier

        Returns:
            Dictionary mapping priority to search results
        """
        # Create semaphore to limit concurrent searches
        semaphore = asyncio.Semaphore(self.max_concurrent_searches)

        # Create search tasks
        search_tasks = [
            self._execute_single_search(request, semaphore, session_id)
            for request in search_requests
        ]

        # Wait for all searches to complete
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results into dictionary
        stream_results = {}
        for i, result in enumerate(search_results):
            priority = search_requests[i].priority
            if isinstance(result, Exception):
                logger.error(f"Search failed for {priority.value}: {result}")
                stream_results[priority] = SearchStreamResult(
                    priority=priority,
                    query=search_requests[i].query,
                    results=[],
                    num_requested=search_requests[i].num_results,
                    num_received=0,
                    success=False,
                    error_message=str(result)
                )
            else:
                stream_results[priority] = result

        return stream_results

    async def _execute_single_search(
        self,
        search_request: SearchRequest,
        semaphore: asyncio.Semaphore,
        session_id: str
    ) -> SearchStreamResult:
        """
        Execute a single search request with error handling.

        Args:
            search_request: SearchRequest to execute
            semaphore: Semaphore for concurrency control
            session_id: Session identifier

        Returns:
            SearchStreamResult with search results
        """
        import time
        start_time = time.time()

        async with semaphore:
            try:
                logger.debug(f"Executing {search_request.priority.value} search: '{search_request.query}'")

                # Execute the search using existing SERP functionality
                search_results = await execute_serper_search(
                    query=search_request.query,
                    search_type=search_request.search_type,
                    num_results=search_request.num_results
                )

                execution_time = time.time() - start_time

                # Create successful result
                stream_result = SearchStreamResult(
                    priority=search_request.priority,
                    query=search_request.query,
                    results=search_results,
                    num_requested=search_request.num_results,
                    num_received=len(search_results),
                    success=True,
                    execution_time=execution_time
                )

                logger.debug(f"{search_request.priority.value} search completed: "
                           f"{len(search_results)} results in {execution_time:.2f}s")

                return stream_result

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{search_request.priority.value} search failed after {execution_time:.2f}s: {e}")

                # Create failed result
                return SearchStreamResult(
                    priority=search_request.priority,
                    query=search_request.query,
                    results=[],
                    num_requested=search_request.num_results,
                    num_received=0,
                    success=False,
                    error_message=str(e),
                    execution_time=execution_time
                )

    def get_search_statistics(self, results: MultiSearchResults) -> Dict[str, Any]:
        """
        Get comprehensive statistics from multi-search results.

        Args:
            results: MultiSearchResults object

        Returns:
            Dictionary with search statistics
        """
        stats = {
            "total_results": results.total_results,
            "successful_streams": results.successful_streams,
            "failed_streams": results.failed_streams,
            "total_execution_time": results.total_execution_time,
            "streams": {}
        }

        for priority, stream_result in results.stream_results.items():
            stats["streams"][priority.value] = {
                "query": stream_result.query,
                "num_requested": stream_result.num_requested,
                "num_received": stream_result.num_received,
                "success": stream_result.success,
                "execution_time": stream_result.execution_time,
                "error_message": stream_result.error_message
            }

        return stats


# Global executor instance for reuse
_multi_search_executor = None

def get_multi_stream_executor(max_concurrent_searches: int = 3) -> MultiStreamSearchExecutor:
    """
    Get or create a multi-stream search executor instance.

    Args:
        max_concurrent_searches: Maximum concurrent searches

    Returns:
        MultiStreamSearchExecutor instance
    """
    global _multi_search_executor

    if _multi_search_executor is None:
        _multi_search_executor = MultiStreamSearchExecutor(max_concurrent_searches)

    return _multi_search_executor


async def execute_multi_stream_search(
    enhanced_queries: EnhancedQueries,
    session_id: str = "default",
    search_type: str = "search",
    result_distribution: Optional[Dict[str, int]] = None
) -> MultiSearchResults:
    """
    Convenience function to execute multi-stream search.

    Args:
        enhanced_queries: EnhancedQueries object
        session_id: Session identifier
        search_type: Type of search
        result_distribution: Custom result distribution

    Returns:
        MultiSearchResults object
    """
    executor = get_multi_stream_executor()
    return await executor.execute_multi_search(
        enhanced_queries=enhanced_queries,
        session_id=session_id,
        search_type=search_type,
        result_distribution=result_distribution
    )