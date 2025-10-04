"""
Content Cleaner Agent with GPT-5-nano Integration

Implements AI-powered content cleaning with search query filtering and quality assessment
as specified in the technical documentation.

Features:
- GPT-5-nano powered content cleaning via Pydantic AI
- Search query relevance filtering
- Content quality scoring (0-100)
- Structured output with clean content and metadata
- Performance optimization for batch processing
"""

import asyncio
import logging
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Pydantic AI imports
try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.openai import OpenAIModel
    PYDAI_AVAILABLE = True
except ImportError:
    logging.warning("Pydantic AI not available - using fallback content cleaning")
    PYDAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class ContentQuality(Enum):
    """Content quality levels."""
    EXCELLENT = "excellent"      # 90-100
    GOOD = "good"               # 70-89
    ACCEPTABLE = "acceptable"   # 50-69
    POOR = "poor"              # 30-49
    UNUSABLE = "unusable"       # 0-29


@dataclass
class CleanedContent:
    """Structured result from content cleaning."""
    original_content: str
    cleaned_content: str
    quality_score: int  # 0-100
    quality_level: ContentQuality
    relevance_score: float  # 0.0-1.0
    word_count: int
    char_count: int
    key_points: List[str]
    topics_detected: List[str]
    cleaning_notes: List[str]
    processing_time: float
    model_used: str


@dataclass
class ContentCleaningContext:
    """Context for content cleaning operations."""
    search_query: str
    query_terms: List[str]
    url: str
    source_domain: str
    session_id: str
    min_quality_threshold: int = 50
    max_content_length: int = 50000


class ContentCleanerAgent:
    """
    AI-powered content cleaner using GPT-5-nano via Pydantic AI.

    Features:
    - Intelligent content cleaning with search query filtering
    - Quality assessment and scoring
    - Key point extraction
    - Topic detection
    - Performance optimization
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize the content cleaner agent.

        Args:
            model_name: OpenAI model to use (fallback to gpt-4o-mini if GPT-5-nano unavailable)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            logger.warning("No OpenAI API key found - content cleaning will be limited")
            self.agent = None
        elif PYDAI_AVAILABLE:
            try:
                # Initialize Pydantic AI agent with OpenAI model
                model = OpenAIModel(self.model_name, api_key=self.api_key)
                self.agent = Agent(
                    model,
                    system_prompt=self._get_system_prompt(),
                    deps_type=ContentCleaningContext
                )
                logger.info(f"Content cleaner agent initialized with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Pydantic AI agent: {e}")
                self.agent = None
        else:
            self.agent = None

        # Performance metrics
        self.stats = {
            'total_cleaned': 0,
            'avg_quality_score': 0.0,
            'avg_processing_time': 0.0,
            'quality_distribution': {}
        }

    async def clean_content(
        self,
        raw_content: str,
        context: ContentCleaningContext
    ) -> CleanedContent:
        """
        Clean raw content using AI-powered processing.

        Args:
            raw_content: Raw content from web crawling
            context: Cleaning context with search query and metadata

        Returns:
            CleanedContent with structured cleaning results
        """
        start_time = datetime.now()

        try:
            # Pre-processing and basic filtering
            if not self._should_process_content(raw_content, context):
                return self._create_unusable_result(raw_content, context, "Content failed pre-filtering")

            # Use AI cleaning if available
            if self.agent:
                result = await self._ai_clean_content(raw_content, context)
            else:
                # Fallback to rule-based cleaning
                result = await self._rule_based_clean_content(raw_content, context)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time

            # Update statistics
            self._update_stats(result)

            logger.debug(f"Content cleaned: {result.quality_score}/100 "
                        f"({result.quality_level.value}) in {processing_time:.2f}s")

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Content cleaning failed for {context.url}: {e}")

            return CleanedContent(
                original_content=raw_content,
                cleaned_content=raw_content,
                quality_score=0,
                quality_level=ContentQuality.UNUSABLE,
                relevance_score=0.0,
                word_count=len(raw_content.split()),
                char_count=len(raw_content),
                key_points=[],
                topics_detected=[],
                cleaning_notes=[f"Cleaning failed: {str(e)}"],
                processing_time=processing_time,
                model_used="failed"
            )

    async def clean_multiple_contents(
        self,
        contents: List[Tuple[str, ContentCleaningContext]],
        max_concurrent: int = 5
    ) -> List[CleanedContent]:
        """
        Clean multiple contents concurrently.

        Args:
            contents: List of (raw_content, context) tuples
            max_concurrent: Maximum concurrent cleaning operations

        Returns:
            List of CleanedContent results
        """
        if not contents:
            return []

        logger.info(f"Starting batch content cleaning: {len(contents)} items, "
                   f"max_concurrent={max_concurrent}")

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)

        async def clean_with_semaphore(content: Tuple[str, ContentCleaningContext]) -> CleanedContent:
            async with semaphore:
                raw_content, context = content
                return await self.clean_content(raw_content, context)

        # Execute cleaning concurrently
        start_time = datetime.now()
        tasks = [clean_with_semaphore(item) for item in contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raw_content, context = contents[i]
                final_results.append(CleanedContent(
                    original_content=raw_content,
                    cleaned_content=raw_content,
                    quality_score=0,
                    quality_level=ContentQuality.UNUSABLE,
                    relevance_score=0.0,
                    word_count=len(raw_content.split()),
                    char_count=len(raw_content),
                    key_points=[],
                    topics_detected=[],
                    cleaning_notes=[f"Processing failed: {str(result)}"],
                    processing_time=0.0,
                    model_used="failed"
                ))
            else:
                final_results.append(result)

        # Log batch summary
        total_time = (datetime.now() - start_time).total_seconds()
        avg_quality = sum(r.quality_score for r in final_results) / len(final_results)
        successful = sum(1 for r in final_results if r.quality_score >= context.min_quality_threshold)

        logger.info(f"Batch cleaning completed: {successful}/{len(final_results)} passed quality threshold "
                   f"(avg quality: {avg_quality:.1f}, total time: {total_time:.1f}s)")

        return final_results

    async def _ai_clean_content(
        self,
        raw_content: str,
        context: ContentCleaningContext
    ) -> CleanedContent:
        """Use AI to clean content via Pydantic AI agent."""
        try:
            # Prepare the cleaning prompt
            cleaning_prompt = self._create_cleaning_prompt(raw_content, context)

            # Run the AI agent
            result = await self.agent.run(cleaning_prompt, deps=context)

            # Parse the structured result
            cleaned_data = result.data

            return CleanedContent(
                original_content=raw_content,
                cleaned_content=cleaned_data.get('cleaned_content', raw_content),
                quality_score=cleaned_data.get('quality_score', 50),
                quality_level=self._get_quality_level(cleaned_data.get('quality_score', 50)),
                relevance_score=cleaned_data.get('relevance_score', 0.5),
                word_count=len(cleaned_data.get('cleaned_content', '').split()),
                char_count=len(cleaned_data.get('cleaned_content', '')),
                key_points=cleaned_data.get('key_points', []),
                topics_detected=cleaned_data.get('topics_detected', []),
                cleaning_notes=cleaned_data.get('cleaning_notes', []),
                processing_time=0.0,  # Will be set by caller
                model_used=self.model_name
            )

        except Exception as e:
            logger.error(f"AI content cleaning failed: {e}")
            # Fallback to rule-based cleaning
            return await self._rule_based_clean_content(raw_content, context)

    async def _rule_based_clean_content(
        self,
        raw_content: str,
        context: ContentCleaningContext
    ) -> CleanedContent:
        """Fallback rule-based content cleaning."""
        try:
            # Apply basic cleaning rules
            cleaned = self._apply_basic_cleaning(raw_content)

            # Calculate relevance to search query
            relevance_score = self._calculate_relevance_score(cleaned, context.search_query, context.query_terms)

            # Estimate quality based on content characteristics
            quality_score = self._estimate_quality_score(cleaned, relevance_score)

            # Extract key points (basic approach)
            key_points = self._extract_key_points(cleaned)

            # Detect topics (basic approach)
            topics = self._detect_topics(cleaned)

            return CleanedContent(
                original_content=raw_content,
                cleaned_content=cleaned,
                quality_score=quality_score,
                quality_level=self._get_quality_level(quality_score),
                relevance_score=relevance_score,
                word_count=len(cleaned.split()),
                char_count=len(cleaned),
                key_points=key_points,
                topics_detected=topics,
                cleaning_notes=["Rule-based cleaning (AI unavailable)"],
                processing_time=0.0,  # Will be set by caller
                model_used="rule_based"
            )

        except Exception as e:
            logger.error(f"Rule-based cleaning failed: {e}")
            return self._create_unusable_result(raw_content, context, f"Rule-based cleaning failed: {e}")

    def _should_process_content(self, content: str, context: ContentCleaningContext) -> bool:
        """Check if content should be processed based on basic criteria."""
        # Length checks
        if len(content.strip()) < 200:
            return False

        if len(content) > context.max_content_length:
            return False

        # Basic spam/adult content checks
        spam_indicators = ['click here', 'buy now', 'limited time', 'act now', 'free trial']
        spam_count = sum(1 for indicator in spam_indicators if indicator.lower() in content.lower())

        if spam_count > 5:  # Too many spam indicators
            return False

        return True

    def _apply_basic_cleaning(self, content: str) -> str:
        """Apply basic content cleaning rules."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)

        # Remove common navigation elements
        nav_patterns = [
            r'Skip to main content',
            r'Menu\s*Navigation',
            r'Search\s*Search',
            r'Close\s*Search',
            r'Cookie\s*Notice',
            r'Privacy\s*Policy',
            r'Terms\s*of\s*Use'
        ]

        for pattern in nav_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        # Remove repeated characters
        content = re.sub(r'(.)\1{3,}', r'\1\1', content)

        # Clean up punctuation spacing
        content = re.sub(r'\s+([,.!?;:])', r'\1', content)

        # Remove empty lines and trim
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return '\n'.join(lines)

    def _calculate_relevance_score(self, content: str, query: str, query_terms: List[str]) -> float:
        """Calculate content relevance to search query."""
        if not query_terms:
            return 0.5  # Default relevance

        content_lower = content.lower()
        query_lower = query.lower()

        # Count term matches
        term_matches = sum(1 for term in query_terms if term.lower() in content_lower)
        term_score = term_matches / len(query_terms)

        # Check for phrase matches
        phrase_matches = 0
        for i in range(len(query_terms)):
            for j in range(i + 1, len(query_terms) + 1):
                phrase = ' '.join(query_terms[i:j]).lower()
                if phrase in content_lower:
                    phrase_matches += 1

        phrase_score = min(1.0, phrase_matches / max(1, len(query_terms) - 1))

        # Combine scores
        relevance = (term_score * 0.6 + phrase_score * 0.4)
        return min(1.0, relevance)

    def _estimate_quality_score(self, content: str, relevance_score: float) -> int:
        """Estimate content quality based on various factors."""
        score = 50  # Base score

        # Length factor (optimal 500-5000 words)
        word_count = len(content.split())
        if 500 <= word_count <= 5000:
            score += 20
        elif 200 <= word_count < 500:
            score += 10
        elif word_count > 5000:
            score += 5

        # Relevance factor
        score += int(relevance_score * 25)

        # Sentence structure (basic)
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        if 10 <= avg_sentence_length <= 25:
            score += 10

        # Capitalization and punctuation
        if content[0].isupper() if content else False:
            score += 5

        # Penalize excessive repetition
        words = content.lower().split()
        unique_words = len(set(words))
        repetition_ratio = unique_words / max(1, len(words))
        score += int(repetition_ratio * 10)

        return min(100, max(0, score))

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points using basic heuristics."""
        sentences = content.split('.')
        key_points = []

        # Look for sentences with indicators
        indicator_patterns = [
            r'\b(important|key|main|primary|significant|notable)\b',
            r'\b(conclusion|finding|result|discovery)\b',
            r'\b(therefore|however|furthermore|moreover)\b'
        ]

        for sentence in sentences[:20]:  # Check first 20 sentences
            sentence = sentence.strip()
            if len(sentence) > 50 and len(sentence) < 300:
                for pattern in indicator_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        key_points.append(sentence[:200] + "..." if len(sentence) > 200 else sentence)
                        break

            if len(key_points) >= 5:  # Limit to 5 key points
                break

        return key_points[:5]

    def _detect_topics(self, content: str) -> List[str]:
        """Detect topics using basic keyword matching."""
        # Simple topic detection based on common domains
        topic_keywords = {
            'technology': ['ai', 'machine learning', 'software', 'computer', 'digital', 'data'],
            'health': ['health', 'medical', 'disease', 'treatment', 'patient', 'hospital'],
            'business': ['business', 'company', 'market', 'economy', 'financial', 'investment'],
            'science': ['research', 'study', 'scientific', 'experiment', 'analysis', 'discovery'],
            'politics': ['government', 'policy', 'political', 'election', 'congress', 'president'],
            'education': ['education', 'school', 'university', 'student', 'learning', 'academic']
        }

        content_lower = content.lower()
        detected_topics = []

        for topic, keywords in topic_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
            if keyword_count >= 2:  # Need at least 2 keywords
                detected_topics.append(topic)

        return detected_topics[:3]  # Limit to 3 topics

    def _get_quality_level(self, score: int) -> ContentQuality:
        """Convert quality score to quality level."""
        if score >= 90:
            return ContentQuality.EXCELLENT
        elif score >= 70:
            return ContentQuality.GOOD
        elif score >= 50:
            return ContentQuality.ACCEPTABLE
        elif score >= 30:
            return ContentQuality.POOR
        else:
            return ContentQuality.UNUSABLE

    def _create_unusable_result(
        self,
        raw_content: str,
        context: ContentCleaningContext,
        reason: str
    ) -> CleanedContent:
        """Create an unusable result for content that fails processing."""
        return CleanedContent(
            original_content=raw_content,
            cleaned_content=raw_content,
            quality_score=0,
            quality_level=ContentQuality.UNUSABLE,
            relevance_score=0.0,
            word_count=len(raw_content.split()),
            char_count=len(raw_content),
            key_points=[],
            topics_detected=[],
            cleaning_notes=[reason],
            processing_time=0.0,
            model_used="none"
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI content cleaner."""
        return """You are an expert content cleaner and analyzer. Your task is to clean and evaluate web content for relevance and quality.

Given raw web content and a search query, you must:

1. **Clean the content** by:
   - Removing navigation elements, ads, and irrelevant sections
   - Fixing formatting issues and excessive whitespace
   - Preserving the core information and structure
   - Ensuring readability and coherence

2. **Evaluate quality** (0-100 scale) based on:
   - Relevance to the search query
   - Content depth and substance
   - Readability and organization
   - Source credibility indicators
   - Information accuracy and completeness

3. **Assess relevance** (0.0-1.0 scale) by:
   - Matching content to search query terms
   - Identifying key concepts and topics
   - Evaluating information density

4. **Extract key points** (3-5 main insights)

5. **Identify topics** (main subjects covered)

Return your analysis in this JSON format:
{
    "cleaned_content": "The cleaned and formatted content",
    "quality_score": 85,
    "relevance_score": 0.78,
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "topics_detected": ["topic1", "topic2"],
    "cleaning_notes": ["Note about cleaning process"]
}

Focus on preserving valuable information while removing noise and improving readability."""

    def _create_cleaning_prompt(self, content: str, context: ContentCleaningContext) -> str:
        """Create the cleaning prompt for the AI."""
        return f"""Please clean and evaluate this web content for the search query: "{context.search_query}"

URL: {context.url}
Source Domain: {context.source_domain}
Query Terms: {', '.join(context.query_terms)}

Raw Content:
{content[:15000]}  # Limit content to avoid token limits

Please analyze and clean this content according to the instructions. Return your response in valid JSON format."""

    def _update_stats(self, result: CleanedContent):
        """Update cleaning statistics."""
        self.stats['total_cleaned'] += 1

        # Update average quality score
        total = self.stats['total_cleaned']
        current_avg = self.stats['avg_quality_score']
        self.stats['avg_quality_score'] = (current_avg * (total - 1) + result.quality_score) / total

        # Update quality distribution
        level = result.quality_level.value
        if level not in self.stats['quality_distribution']:
            self.stats['quality_distribution'][level] = 0
        self.stats['quality_distribution'][level] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics."""
        return {
            **self.stats,
            'agent_available': self.agent is not None,
            'model_used': self.model_name
        }


# Global content cleaner instance
_global_content_cleaner: Optional[ContentCleanerAgent] = None


def get_content_cleaner() -> ContentCleanerAgent:
    """Get or create global content cleaner agent."""
    global _global_content_cleaner
    if _global_content_cleaner is None:
        _global_content_cleaner = ContentCleanerAgent()
    return _global_content_cleaner