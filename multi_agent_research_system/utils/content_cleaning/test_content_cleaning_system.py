"""
Test Suite for Content Cleaning System - Phase 1.3

Comprehensive tests for the GPT-5-Nano Content Cleaning Module with confidence scoring.
Tests all components: FastConfidenceScorer, ContentCleaningPipeline, EditorialDecisionEngine, and CachingOptimizer.
"""

import asyncio
import logging
import pytest
import time
from typing import List, Dict, Any

# Import the modules we're testing
from .fast_confidence_scorer import FastConfidenceScorer, ConfidenceSignals
from .content_cleaning_pipeline import ContentCleaningPipeline, CleaningResult, PipelineConfig
from .editorial_decision_engine import EditorialDecisionEngine, EditorialDecision
from .caching_optimizer import CachingOptimizer, LRUCache

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestFastConfidenceScorer:
    """Test cases for FastConfidenceScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a FastConfidenceScorer instance for testing."""
        return FastConfidenceScorer(cache_enabled=True, cache_size=100)

    @pytest.fixture
    def sample_content(self):
        """Sample content for testing."""
        return """
        # Artificial Intelligence in Healthcare: Recent Developments

        Artificial intelligence (AI) is revolutionizing healthcare in 2024 with breakthrough applications in diagnostics, drug discovery, and personalized medicine. Major hospitals and research institutions are reporting significant improvements in patient outcomes through AI-powered systems.

        ## Key Developments in 2024

        **Diagnostic AI Systems**
        - Google's DeepMind has achieved 95% accuracy in detecting diabetic retinopathy
        - Stanford Medical Center implemented AI for early cancer detection
        - FDA approved 15 new AI diagnostic tools this year

        **Drug Discovery Acceleration**
        - AI-powered drug discovery reduced development time by 40%
        - 50 new drug candidates identified using machine learning
        - Pharmaceutical companies investing $12B in AI research

        The integration of AI in healthcare represents one of the most significant technological advances in modern medicine, with projected market growth to $120B by 2030.

        Sources: Medical journals, hospital reports, and industry analysis from 2024.
        """

    @pytest.fixture
    def sample_url(self):
        """Sample URL for testing."""
        return "https://stanford.edu/healthcare-ai-2024"

    @pytest.fixture
    def sample_query(self):
        """Sample search query for testing."""
        return "artificial intelligence healthcare developments 2024"

    @pytest.mark.asyncio
    async def test_confidence_assessment_basic(self, scorer, sample_content, sample_url, sample_query):
        """Test basic confidence assessment functionality."""
        signals = await scorer.assess_content_confidence(
            content=sample_content,
            url=sample_url,
            search_query=sample_query
        )

        # Verify all signals are populated
        assert isinstance(signals, ConfidenceSignals)
        assert signals.overall_confidence > 0.0
        assert signals.overall_confidence <= 1.0
        assert signals.processing_time_ms > 0

        # Verify component scores are in valid range
        assert 0.0 <= signals.content_length_score <= 1.0
        assert 0.0 <= signals.structure_score <= 1.0
        assert 0.0 <= signals.relevance_score <= 1.0
        assert 0.0 <= signals.cleanliness_score <= 1.0
        assert 0.0 <= signals.domain_authority_score <= 1.0
        assert 0.0 <= signals.freshness_score <= 1.0
        assert 0.0 <= signals.extraction_confidence <= 1.0
        assert 0.0 <= signals.llm_assessment <= 1.0

        logger.info(f"Confidence assessment completed: {signals.overall_confidence:.3f}")

    @pytest.mark.asyncio
    async def test_cache_functionality(self, scorer, sample_content, sample_url, sample_query):
        """Test caching functionality."""
        # First assessment (cache miss)
        start_time = time.time()
        signals1 = await scorer.assess_content_confidence(
            content=sample_content,
            url=sample_url,
            search_query=sample_query
        )
        first_duration = time.time() - start_time

        # Second assessment (cache hit)
        start_time = time.time()
        signals2 = await scorer.assess_content_confidence(
            content=sample_content,
            url=sample_url,
            search_query=sample_query
        )
        second_duration = time.time() - start_time

        # Verify cache hit
        assert signals2.cache_hit is True
        assert signals1.overall_confidence == signals2.overall_confidence
        assert second_duration < first_duration  # Cache should be faster

        logger.info(f"Cache test: first={first_duration:.3f}s, second={second_duration:.3f}s")

    def test_content_length_scoring(self, scorer):
        """Test content length scoring logic."""
        # Test short content
        short_content = "Very short content."
        score = scorer._score_content_length(short_content)
        assert score < 0.5

        # Test optimal content
        optimal_content = " ".join(["word"] * 1000)  # ~1000 words
        score = scorer._score_content_length(optimal_content)
        assert score == 1.0

        # Test very long content
        long_content = " ".join(["word"] * 15000)  # ~15000 words
        score = scorer._score_content_length(long_content)
        assert score < 1.0
        assert score > 0.5

    def test_domain_authority_scoring(self, scorer):
        """Test domain authority scoring."""
        # High authority domains
        assert scorer._score_domain_authority("https://mit.edu/research") == 0.9
        assert scorer._score_domain_authority("https://cdc.gov/health") == 0.9
        assert scorer._score_domain_authority("https://who.int/updates") == 0.9

        # News domains
        assert scorer._score_domain_authority("https://reuters.com/article") == 0.8
        assert scorer._score_domain_authority("https://bbc.com/news") == 0.8

        # Commercial domains
        assert scorer._score_domain_authority("https://example.com/page") == 0.6

    def test_editorial_recommendations(self, scorer):
        """Test editorial recommendation logic."""
        # Create test signals for different quality levels
        high_quality = ConfidenceSignals(overall_confidence=0.85)
        acceptable_quality = ConfidenceSignals(overall_confidence=0.7)
        low_quality = ConfidenceSignals(overall_confidence=0.5)
        poor_quality = ConfidenceSignals(overall_confidence=0.3)

        # Test recommendations
        assert scorer.get_editorial_recommendation(high_quality) == "ACCEPT_CONTENT"
        assert scorer.get_editorial_recommendation(acceptable_quality) == "ENHANCE_CONTENT"
        assert scorer.get_editorial_recommendation(low_quality) == "GAP_RESEARCH"
        assert scorer.get_editorial_recommendation(poor_quality) == "REJECT_CONTENT"

    def test_threshold_validation(self, scorer):
        """Test threshold validation and edge cases."""
        # Test that scores are properly clamped
        test_signals = ConfidenceSignals(
            content_length_score=1.5,  # Above 1.0
            structure_score=-0.5,      # Below 0.0
            overall_confidence=0.75
        )

        # Should not crash with invalid scores
        detailed = scorer.get_detailed_assessment(test_signals)
        assert detailed is not None
        assert 'overall_confidence' in detailed


class TestContentCleaningPipeline:
    """Test cases for ContentCleaningPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a ContentCleaningPipeline instance for testing."""
        config = PipelineConfig(
            cleanliness_threshold=0.7,
            minimum_quality_threshold=0.6,
            enable_ai_cleaning=True,
            enable_quality_validation=True
        )
        return ContentCleaningPipeline(config)

    @pytest.fixture
    def sample_dirty_content(self):
        """Sample dirty content for testing."""
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Healthcare AI News</title></head>
        <body>
        <nav>Menu | Home | About | Contact | Subscribe</nav>
        <header>Healthcare AI News - Latest Updates</header>

        <main>
        <article>
        <h1>AI Breakthrough in Cancer Detection</h1>
        <p>Published: March 15, 2024</p>

        <p>Researchers at MIT have developed a new artificial intelligence system that can detect cancer with 95% accuracy, representing a major breakthrough in medical diagnostics.</p>

        <p>The AI system, trained on millions of medical images, outperforms human radiologists in identifying early-stage cancers, potentially saving thousands of lives through early detection.</p>

        <p>"This is a game-changer for cancer diagnosis," said Dr. Sarah Chen, lead researcher at MIT's Computer Science and Artificial Intelligence Laboratory.</p>

        <p>The technology is currently being tested in three major hospitals and could be widely available by 2025.</p>
        </article>
        </main>

        <footer>© 2024 Healthcare News | Privacy Policy | Terms of Service | Follow us on Twitter | Newsletter Signup</footer>
        <div class="ads">Advertisement | Subscribe now for more updates</div>
        </body>
        </html>
        """

    @pytest.mark.asyncio
    async def test_basic_cleaning(self, pipeline, sample_dirty_content):
        """Test basic content cleaning functionality."""
        result = await pipeline.clean_content(
            content=sample_dirty_content,
            url="https://healthcare-news.com/ai-breakthrough",
            search_query="AI cancer detection breakthrough"
        )

        # Verify result structure
        assert isinstance(result, CleaningResult)
        assert result.original_content == sample_dirty_content
        assert result.url == "https://healthcare-news.com/ai-breakthrough"
        assert result.search_query == "AI cancer detection breakthrough"
        assert result.processing_time_ms > 0

        # Verify cleaning occurred
        assert len(result.cleaned_content) > 0
        assert result.cleaned_content != result.original_content  # Should be cleaned

        logger.info(f"Basic cleaning completed: stage={result.cleaning_stage}, time={result.processing_time_ms:.1f}ms")

    @pytest.mark.asyncio
    async def test_content_validation(self, pipeline):
        """Test content validation logic."""
        # Test content that's too short
        short_content = "Too short"
        result = await pipeline.clean_content(
            content=short_content,
            url="https://example.com/short"
        )

        assert result.cleaning_stage == "rejected"
        assert result.error_message is not None

        # Test navigation-heavy content
        nav_content = "menu navigation login search cart menu navigation"
        result = await pipeline.clean_content(
            content=nav_content,
            url="https://example.com/nav"
        )

        assert result.cleaning_stage == "rejected"

    @pytest.mark.asyncio
    async def test_batch_cleaning(self, pipeline, sample_dirty_content):
        """Test batch content cleaning."""
        content_list = [
            (sample_dirty_content, "https://example1.com"),
            (sample_dirty_content, "https://example2.com"),
            (sample_dirty_content, "https://example3.com")
        ]

        results = await pipeline.clean_content_batch(
            content_list=content_list,
            search_query="AI healthcare"
        )

        assert len(results) == 3
        assert all(isinstance(r, CleaningResult) for r in results)
        assert all(r.processing_time_ms > 0 for r in results)

        logger.info(f"Batch cleaning completed: {len(results)} items processed")


class TestEditorialDecisionEngine:
    """Test cases for EditorialDecisionEngine."""

    @pytest.fixture
    def engine(self):
        """Create an EditorialDecisionEngine instance for testing."""
        return EditorialDecisionEngine()

    def test_high_quality_content(self, engine):
        """Test decision for high quality content."""
        signals = ConfidenceSignals(
            overall_confidence=0.9,
            content_length_score=0.9,
            structure_score=0.8,
            relevance_score=0.9,
            cleanliness_score=0.8,
            domain_authority_score=0.9,
            freshness_score=0.9,
            extraction_confidence=0.9,
            llm_assessment=0.85
        )

        action = engine.evaluate_content(signals)

        assert action.decision == EditorialDecision.ACCEPT_CONTENT
        assert action.priority.value == "LOW"
        assert "good quality" in action.reasoning.lower()

    def test_medium_quality_content(self, engine):
        """Test decision for medium quality content."""
        signals = ConfidenceSignals(
            overall_confidence=0.75,
            content_length_score=0.7,
            structure_score=0.6,
            relevance_score=0.8,
            cleanliness_score=0.5,  # Lower cleanliness
            domain_authority_score=0.7,
            freshness_score=0.8,
            extraction_confidence=0.7,
            llm_assessment=0.7
        )

        action = engine.evaluate_content(signals)

        assert action.decision == EditorialDecision.ENHANCE_CONTENT
        assert action.priority.value == "MEDIUM"

    def test_low_quality_content(self, engine):
        """Test decision for low quality content."""
        signals = ConfidenceSignals(
            overall_confidence=0.5,
            content_length_score=0.4,
            structure_score=0.3,
            relevance_score=0.5,
            cleanliness_score=0.4,
            domain_authority_score=0.4,
            freshness_score=0.6,
            extraction_confidence=0.5,
            llm_assessment=0.5
        )

        action = engine.evaluate_content(signals)

        assert action.decision == EditorialDecision.GAP_RESEARCH
        assert action.priority.value == "HIGH"

    def test_critical_failure_content(self, engine):
        """Test decision for critically failing content."""
        signals = ConfidenceSignals(
            overall_confidence=0.2,
            content_length_score=0.1,
            structure_score=0.2,
            relevance_score=0.3,
            cleanliness_score=0.2,
            domain_authority_score=0.3,
            freshness_score=0.4,
            extraction_confidence=0.2,
            llm_assessment=0.2
        )

        action = engine.evaluate_content(signals)

        assert action.decision == EditorialDecision.REJECT_CONTENT
        assert action.priority.value == "HIGH"

    def test_batch_evaluation(self, engine):
        """Test batch content evaluation."""
        signals_list = [
            ConfidenceSignals(overall_confidence=0.9),  # High quality
            ConfidenceSignals(overall_confidence=0.5),  # Low quality
            ConfidenceSignals(overall_confidence=0.75), # Medium quality
            ConfidenceSignals(overall_confidence=0.2),  # Critical failure
        ]

        actions = engine.evaluate_content_batch(signals_list)
        recommendations = engine.get_batch_recommendations(actions)

        assert len(actions) == 4
        assert len(recommendations['decision_distribution']) > 0
        assert 'processing_priority' in recommendations

        # Verify decisions
        decisions = [action.decision for action in actions]
        assert EditorialDecision.ACCEPT_CONTENT in decisions
        assert EditorialDecision.GAP_RESEARCH in decisions
        assert EditorialDecision.REJECT_CONTENT in decisions

    def test_threshold_updates(self, engine):
        """Test threshold updates."""
        original_threshold = engine.thresholds['good_quality']

        # Update threshold
        engine.update_thresholds({'good_quality': 0.85})

        assert engine.thresholds['good_quality'] == 0.85
        assert engine.thresholds['good_quality'] != original_threshold

        # Test invalid threshold
        original_acceptable = engine.thresholds['acceptable_quality']
        engine.update_thresholds({'acceptable_quality': 1.5})  # Invalid

        # Should not change
        assert engine.thresholds['acceptable_quality'] == original_acceptable


class TestCachingOptimizer:
    """Test cases for CachingOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create a CachingOptimizer instance for testing."""
        return CachingOptimizer(
            enable_lru_cache=True,
            enable_similarity_cache=True,
            lru_cache_size=100,
            similarity_cache_size=50
        )

    @pytest.mark.asyncio
    async def test_lru_cache_functionality(self, optimizer):
        """Test LRU cache functionality."""
        cache = LRUCache(max_size=5, default_ttl=1)  # 1 second TTL for testing

        # Test basic put/get
        assert cache.put("key1", "value1") is True
        assert cache.get("key1") == "value1"

        # Test cache miss
        assert cache.get("nonexistent") is None

        # Test cache expiration
        cache.put("expire_key", "expire_value", ttl=0.1)
        assert cache.get("expire_key") == "expire_value"
        await asyncio.sleep(0.2)
        assert cache.get("expire_key") is None  # Should be expired

        # Test LRU eviction
        for i in range(6):  # More than max_size
            cache.put(f"key{i}", f"value{i}")

        # Oldest keys should be evicted
        assert cache.get("key0") is None
        assert cache.get("key5") is not None

        # Test stats
        stats = cache.get_stats()
        assert stats.cache_size <= 5
        assert stats.total_requests > 0

    def test_similarity_cache_functionality(self, optimizer):
        """Test similarity cache functionality."""
        similarity_cache = optimizer.similarity_cache

        if not similarity_cache:
            pytest.skip("Similarity cache not enabled")

        from .fast_confidence_scorer import ConfidenceSignals

        # Test caching and retrieval
        content1 = "Artificial intelligence in healthcare is transforming patient care."
        content2 = "AI in healthcare is revolutionizing how patients are treated."  # Similar
        content3 = "Machine learning algorithms are optimizing supply chains."  # Different

        signals = ConfidenceSignals(overall_confidence=0.8)

        # Cache content1
        similarity_cache.put(content1, signals)

        # Find similar content (content2 should match)
        found_signals = similarity_cache.get_similar(content2)
        assert found_signals is not None
        assert found_signals.overall_confidence == 0.8

        # Different content should not match
        found_signals = similarity_cache.get_similar(content3)
        assert found_signals is None

    @pytest.mark.asyncio
    async def test_optimizer_integration(self, optimizer):
        """Test caching optimizer integration."""
        await optimizer.start()

        try:
            from .fast_confidence_scorer import ConfidenceSignals

            # Test caching workflow
            content = "AI in healthcare is making significant progress in 2024."
            url = "https://example.com/ai-healthcare"
            query = "artificial intelligence healthcare developments"

            signals = ConfidenceSignals(overall_confidence=0.75)

            # Cache signals
            optimizer.cache_confidence_signals(content, url, query, signals)

            # Retrieve cached signals
            cached_signals = optimizer.get_cached_confidence_signals(content, url, query)
            assert cached_signals is not None
            assert cached_signals.overall_confidence == 0.75

            # Test performance stats
            stats = optimizer.get_performance_stats()
            assert 'lru_cache' in stats
            assert 'similarity_cache' in stats
            assert stats['lru_cache_enabled'] is True

        finally:
            await optimizer.stop()

    @pytest.mark.asyncio
    async def test_cache_cleanup(self, optimizer):
        """Test cache cleanup functionality."""
        await optimizer.start()

        try:
            # Force some cache activity
            from .fast_confidence_scorer import ConfidenceSignals

            for i in range(10):
                content = f"Test content {i}"
                url = f"https://example.com/test{i}"
                signals = ConfidenceSignals(overall_confidence=0.5 + i * 0.05)
                optimizer.cache_confidence_signals(content, url, None, signals)

            # Trigger cleanup
            await optimizer._perform_cleanup()

            # Verify cleanup didn't break anything
            stats = optimizer.get_performance_stats()
            assert stats is not None

        finally:
            await optimizer.stop()


# Integration tests
class TestContentCleaningIntegration:
    """Integration tests for the complete content cleaning system."""

    @pytest.fixture
    def full_system(self):
        """Create a complete content cleaning system."""
        config = PipelineConfig(
            cleanliness_threshold=0.7,
            minimum_quality_threshold=0.6,
            enable_ai_cleaning=True,
            enable_quality_validation=True
        )

        pipeline = ContentCleaningPipeline(config)
        engine = EditorialDecisionEngine()
        optimizer = CachingOptimizer(
            enable_lru_cache=True,
            enable_similarity_cache=True
        )

        return {
            'pipeline': pipeline,
            'engine': engine,
            'optimizer': optimizer
        }

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, full_system):
        """Test complete end-to-end workflow."""
        pipeline = full_system['pipeline']
        engine = full_system['engine']
        optimizer = full_system['optimizer']

        await optimizer.start()

        try:
            # Sample content
            content = """
            # AI in Healthcare: Latest Developments

            Artificial intelligence continues to transform healthcare in 2024 with new applications in diagnostics and treatment. Major hospitals report improved patient outcomes through AI-powered systems.

            Recent breakthroughs include:
            - 95% accuracy in diabetic retinopathy detection
            - 40% reduction in drug discovery time
            - FDA approval of 15 new AI diagnostic tools

            Source: Medical Journal 2024
            """

            url = "https://medical-journal.edu/ai-healthcare-2024"
            query = "artificial intelligence healthcare 2024 developments"

            # Step 1: Clean content
            cleaning_result = await pipeline.clean_content(
                content=content,
                url=url,
                search_query=query
            )

            assert cleaning_result.confidence_signals is not None
            assert cleaning_result.cleaned_content is not None

            # Step 2: Get editorial recommendation
            editorial_action = engine.evaluate_cleaning_result(cleaning_result)
            assert editorial_action.decision is not None
            assert editorial_action.reasoning is not None

            # Step 3: Cache the results
            if cleaning_result.confidence_signals:
                optimizer.cache_confidence_signals(
                    content, url, query, cleaning_result.confidence_signals
                )

                # Step 4: Verify cache retrieval
                cached_signals = optimizer.get_cached_confidence_signals(content, url, query)
                assert cached_signals is not None
                assert cached_signals.overall_confidence == cleaning_result.confidence_signals.overall_confidence

            logger.info(f"End-to-end workflow completed: "
                       f"cleaning_stage={cleaning_result.cleaning_stage}, "
                       f"editorial_decision={editorial_action.decision.value}")

        finally:
            await optimizer.stop()

    def test_configuration_compatibility(self):
        """Test that all components work together with different configurations."""
        # Test with minimal configuration
        minimal_config = PipelineConfig(
            enable_ai_cleaning=False,
            enable_quality_validation=False,
            enable_performance_optimization=False
        )
        pipeline = ContentCleaningPipeline(minimal_config)
        assert pipeline.config.enable_ai_cleaning is False

        # Test with custom thresholds
        custom_engine = EditorialDecisionEngine({
            'good_quality': 0.85,
            'acceptable_quality': 0.65
        })
        assert custom_engine.thresholds['good_quality'] == 0.85

        # Test cache-only optimizer
        cache_only = CachingOptimizer(
            enable_lru_cache=True,
            enable_similarity_cache=False
        )
        assert cache_only.enable_lru_cache is True
        assert cache_only.enable_similarity_cache is False


# Performance tests
class TestContentCleaningPerformance:
    """Performance tests for the content cleaning system."""

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self):
        """Test batch processing performance."""
        config = PipelineConfig(
            enable_performance_optimization=True,
            enable_quality_validation=True
        )
        pipeline = ContentCleaningPipeline(config)

        # Generate test content
        content_list = []
        for i in range(20):  # 20 items
            content = f"""
            # Article {i}

            This is test article number {i} about artificial intelligence in healthcare.
            It contains multiple paragraphs to simulate real content.

            Key points:
            - Point 1 for article {i}
            - Point 2 for article {i}
            - Point 3 for article {i}

            Conclusion for article {i}.
            """
            content_list.append((content, f"https://example.com/article{i}"))

        # Measure batch processing time
        start_time = time.time()
        results = await pipeline.clean_content_batch(
            content_list=content_list,
            search_query="artificial intelligence healthcare"
        )
        processing_time = time.time() - start_time

        # Verify performance
        assert len(results) == 20
        assert processing_time < 30.0  # Should complete within 30 seconds
        assert all(r.processing_time_ms > 0 for r in results)

        avg_time_per_item = processing_time / 20
        logger.info(f"Batch processing: {len(results)} items in {processing_time:.2f}s "
                   f"({avg_time_per_item:.3f}s per item)")

    def test_cache_performance(self):
        """Test cache performance characteristics."""
        cache = LRUCache(max_size=1000, default_ttl=3600)

        # Measure cache performance
        test_data = {f"key{i}": f"value{i}" for i in range(100)}

        # Warm up cache
        start_time = time.time()
        for key, value in test_data.items():
            cache.put(key, value)
        put_time = time.time() - start_time

        # Test cache hits
        start_time = time.time()
        for key in test_data.keys():
            value = cache.get(key)
            assert value is not None
        hit_time = time.time() - start_time

        # Test cache misses
        start_time = time.time()
        for i in range(100, 200):
            value = cache.get(f"nonexistent{i}")
            assert value is None
        miss_time = time.time() - start_time

        # Performance assertions
        stats = cache.get_stats()
        assert stats.hit_rate > 0.95  # Should have high hit rate
        assert stats.avg_access_time_ms < 1.0  # Should be very fast

        logger.info(f"Cache performance: put={put_time:.3f}s, "
                   f"hit={hit_time:.3f}s, miss={miss_time:.3f}s, "
                   f"hit_rate={stats.hit_rate:.3f}")


# Test runner
def run_content_cleaning_tests():
    """Run all content cleaning tests."""
    import sys
    import pytest

    # Configure test arguments
    args = [
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-x"  # Stop on first failure
    ]

    # Run tests
    result = pytest.main(args)
    return result


if __name__ == "__main__":
    print("Running Content Cleaning System Tests - Phase 1.3")
    print("=" * 60)

    # Run the test suite
    result = run_content_cleaning_tests()

    if result == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {result}")
        sys.exit(result)