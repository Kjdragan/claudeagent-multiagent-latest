"""
Streaming Scrape-Clean Pipeline

Implements the streaming parallel processing architecture from asynceval1.md.
Eliminates the sequential bottleneck by starting content cleaning immediately
after each URL is scraped, rather than waiting for all scraping to complete.

Performance Improvement:
- Current: ~109s (45s scraping + 64s cleaning sequentially)
- Streaming: ~65-75s (30-40% faster through parallel overlap)

Key Features:
- Immediate cleaning after scrape completion (no waiting)
- Content length filtering (500-150,000 chars)
- Semaphore-based concurrency control
- Comprehensive error isolation
- Detailed performance metrics
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    """Result from streaming scrape+clean pipeline."""

    url: str
    scrape_success: bool
    clean_success: bool
    cleaned_content: str
    quality_score: int
    scrape_time: float
    clean_time: float
    total_time: float
    processing_stage: str
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class StreamingScrapeCleanPipeline:
    """
    Streaming parallel scrape+clean pipeline.

    This pipeline processes URLs with immediate cleaning after scraping,
    eliminating the sequential bottleneck between stages.
    """

    def __init__(self, max_concurrent_scrapes: int | None = None, max_concurrent_cleans: int | None = None):
        """
        Initialize streaming pipeline.

        Args:
            max_concurrent_scrapes: Maximum concurrent scraping operations (None for unbounded)
            max_concurrent_cleans: Maximum concurrent cleaning operations (None for unbounded)
        """
        self.scrape_semaphore = (
            asyncio.Semaphore(max_concurrent_scrapes)
            if max_concurrent_scrapes and max_concurrent_scrapes > 0
            else None
        )
        self.clean_semaphore = (
            asyncio.Semaphore(max_concurrent_cleans)
            if max_concurrent_cleans and max_concurrent_cleans > 0
            else None
        )

        self.scrape_concurrency = (
            max_concurrent_scrapes if max_concurrent_scrapes and max_concurrent_scrapes > 0 else None
        )
        self.clean_concurrency = (
            max_concurrent_cleans if max_concurrent_cleans and max_concurrent_cleans > 0 else None
        )

        # Statistics tracking
        self.stats = {
            "total_urls": 0,
            "successful_scrapes": 0,
            "successful_cleans": 0,
            "filtered_out": 0,
            "scrape_failures": 0,
            "clean_failures": 0,
            "total_scrape_time": 0.0,
            "total_clean_time": 0.0,
            "overlap_time": 0.0
        }

    async def process_urls_streaming(
        self,
        urls: list[str],
        search_query: str,
        session_id: str,
        initial_level: int = 1,
        max_level: int = 3,
    ) -> list[StreamingResult]:
        """
        Process URLs with streaming scrape→clean pipeline.

        Each URL flows through: scrape → filter → clean (immediately)
        No waiting for batch completion between stages.

        Args:
            urls: List of URLs to process
            search_query: Search query for relevance scoring
            session_id: Session identifier
            initial_level: Initial anti-bot level
            max_level: Maximum anti-bot escalation level

        Returns:
            List of StreamingResult objects
        """
        if not urls:
            return []

        self.stats["total_urls"] = len(urls)
        logger.info(f"🚀 Starting streaming pipeline for {len(urls)} URLs")
        logger.info(
            "Scrape concurrency: %s, Clean concurrency: %s",
            self.scrape_concurrency if self.scrape_concurrency else "unbounded",
            self.clean_concurrency if self.clean_concurrency else "unbounded",
        )

        pipeline_start = time.time()

        async def process_single_url(url: str) -> StreamingResult:
            """Process a single URL through the complete pipeline."""
            url_start = time.time()
            scrape_start = url_start

            try:
                # STEP 1: Scrape with anti-bot escalation
                from .anti_bot_escalation import get_escalation_manager

                escalation_manager = get_escalation_manager()
                if self.scrape_semaphore is None:
                    scrape_result = await escalation_manager.crawl_with_escalation(
                        url=url,
                        initial_level=initial_level,
                        max_level=max_level,
                        use_content_filter=False,
                        session_id=session_id
                    )
                else:
                    async with self.scrape_semaphore:
                        scrape_result = await escalation_manager.crawl_with_escalation(
                            url=url,
                            initial_level=initial_level,
                            max_level=max_level,
                            use_content_filter=False,
                            session_id=session_id
                        )

                scrape_time = time.time() - scrape_start
                self.stats["total_scrape_time"] += scrape_time

                # Check scrape success
                if not scrape_result.success:
                    self.stats["scrape_failures"] += 1
                    return StreamingResult(
                        url=url,
                        scrape_success=False,
                        clean_success=False,
                        cleaned_content="",
                        quality_score=0,
                        scrape_time=scrape_time,
                        clean_time=0.0,
                        total_time=time.time() - url_start,
                        processing_stage="scrape_failed",
                        error_message=scrape_result.error
                    )

                self.stats["successful_scrapes"] += 1

                # STEP 2: Content filtering (Phase 2 requirement)
                filter_reason = self._should_clean(scrape_result)
                if filter_reason:
                    self.stats["filtered_out"] += 1
                    return StreamingResult(
                        url=url,
                        scrape_success=True,
                        clean_success=False,
                        cleaned_content="",
                        quality_score=0,
                        scrape_time=scrape_time,
                        clean_time=0.0,
                        total_time=time.time() - url_start,
                        processing_stage="filtered_out",
                        error_message=filter_reason
                    )

                # STEP 3: IMMEDIATE cleaning (streaming!)
                clean_start = time.time()
                from ..agents.content_cleaner_agent import (
                    ContentCleaningContext,
                    get_content_cleaner
                )

                content_cleaner = get_content_cleaner()
                context = self._create_cleaning_context(url, search_query, session_id)

                if self.clean_semaphore is None:
                    clean_result = await content_cleaner.clean_content(
                        scrape_result.content, context
                    )
                else:
                    async with self.clean_semaphore:
                        clean_result = await content_cleaner.clean_content(
                            scrape_result.content, context
                        )

                clean_time = time.time() - clean_start
                self.stats["total_clean_time"] += clean_time

                if clean_result.quality_score >= context.min_quality_threshold:
                    self.stats["successful_cleans"] += 1
                    return StreamingResult(
                        url=url,
                        scrape_success=True,
                        clean_success=True,
                        cleaned_content=clean_result.cleaned_content,
                        quality_score=clean_result.quality_score,
                        scrape_time=scrape_time,
                        clean_time=clean_time,
                        total_time=time.time() - url_start,
                        processing_stage="completed",
                        metadata={
                            "quality_level": clean_result.quality_level.value,
                            "relevance_score": clean_result.relevance_score,
                            "word_count": clean_result.word_count
                        }
                    )
                else:
                    self.stats["clean_failures"] += 1
                    return StreamingResult(
                        url=url,
                        scrape_success=True,
                        clean_success=False,
                        cleaned_content="",
                        quality_score=clean_result.quality_score,
                        scrape_time=scrape_time,
                        clean_time=clean_time,
                        total_time=time.time() - url_start,
                        processing_stage="quality_too_low",
                        error_message=f"Quality score {clean_result.quality_score} below threshold {context.min_quality_threshold}"
                    )

            except Exception as e:
                logger.error(f"Pipeline error for {url}: {e}")
                return StreamingResult(
                    url=url,
                    scrape_success=False,
                    clean_success=False,
                    cleaned_content="",
                    quality_score=0,
                    scrape_time=time.time() - scrape_start if scrape_start else 0.0,
                    clean_time=0.0,
                    total_time=time.time() - url_start,
                    processing_stage="error",
                    error_message=str(e)
                )

        # Launch all URLs concurrently with streaming processing
        tasks = [process_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions from gather
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception processing {urls[i]}: {result}")
                final_results.append(StreamingResult(
                    url=urls[i],
                    scrape_success=False,
                    clean_success=False,
                    cleaned_content="",
                    quality_score=0,
                    scrape_time=0.0,
                    clean_time=0.0,
                    total_time=0.0,
                    processing_stage="exception",
                    error_message=str(result)
                ))
            else:
                final_results.append(result)

        pipeline_duration = time.time() - pipeline_start

        # Calculate overlap time (performance gain indicator)
        self._calculate_overlap_time(pipeline_duration)

        logger.info(f"✅ Streaming pipeline completed in {pipeline_duration:.2f}s")
        self._log_statistics()

        return final_results

    def _should_clean(self, scrape_result) -> Optional[str]:
        """
        Pre-filter content based on length (Phase 2 requirement).

        Args:
            scrape_result: Result from scraping

        Returns:
            None if should clean, otherwise reason string for filtering out
        """
        char_count = scrape_result.char_count

        # Too short (minimum 500 characters)
        if char_count < 500:
            logger.info(f"⏭️  Skipping cleaning: content too short ({char_count} chars)")
            return f"Content too short ({char_count} chars, minimum: 500)"

        # Too long (maximum 150,000 characters per requirement)
        if char_count > 150000:
            logger.warning(f"⏭️  Skipping cleaning: content too long ({char_count} chars, max: 150,000)")
            return f"Content too long ({char_count} chars, maximum: 150,000)"

        return None  # Should clean

    def _create_cleaning_context(self, url: str, search_query: str, session_id: str):
        """Create ContentCleaningContext for a URL."""
        from ..agents.content_cleaner_agent import ContentCleaningContext

        domain = urlparse(url).netloc.lower()
        query_terms = search_query.split()

        return ContentCleaningContext(
            search_query=search_query,
            query_terms=query_terms,
            url=url,
            source_domain=domain,
            session_id=session_id,
            min_quality_threshold=50,
            max_content_length=50000
        )

    def _calculate_overlap_time(self, pipeline_duration: float):
        """
        Calculate time overlap between scraping and cleaning.

        This is the performance gain from streaming processing.
        """
        # Sequential would be: total_scrape_time + total_clean_time
        sequential_time = self.stats["total_scrape_time"] + self.stats["total_clean_time"]

        # Overlap is the difference between sequential and actual (streaming)
        self.stats["overlap_time"] = max(0, sequential_time - pipeline_duration)

        # Save for reporting
        self.stats["pipeline_duration"] = pipeline_duration
        self.stats["sequential_would_be"] = sequential_time

    def _log_statistics(self):
        """Log pipeline statistics."""
        logger.info("=" * 60)
        logger.info("Streaming Pipeline Statistics")
        logger.info("=" * 60)
        logger.info(f"Total URLs processed: {self.stats['total_urls']}")
        logger.info(f"Successful scrapes: {self.stats['successful_scrapes']}")
        logger.info(f"Successful cleans: {self.stats['successful_cleans']}")
        logger.info(f"Filtered out: {self.stats['filtered_out']}")
        logger.info(f"Scrape failures: {self.stats['scrape_failures']}")
        logger.info(f"Clean failures: {self.stats['clean_failures']}")
        logger.info("-" * 60)
        logger.info(f"Total scrape time: {self.stats['total_scrape_time']:.2f}s")
        logger.info(f"Total clean time: {self.stats['total_clean_time']:.2f}s")
        logger.info(f"Pipeline duration: {self.stats.get('pipeline_duration', 0):.2f}s")
        logger.info(f"Sequential would be: {self.stats.get('sequential_would_be', 0):.2f}s")
        logger.info(f"Time saved (overlap): {self.stats['overlap_time']:.2f}s")

        if self.stats.get('sequential_would_be', 0) > 0:
            improvement = (self.stats['overlap_time'] / self.stats['sequential_would_be']) * 100
            logger.info(f"Performance improvement: {improvement:.1f}%")

        logger.info("=" * 60)

    def get_statistics(self) -> dict:
        """Get pipeline statistics for reporting."""
        return self.stats.copy()
