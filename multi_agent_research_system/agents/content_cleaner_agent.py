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
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use existing environment

# Pydantic AI imports
try:
    from pydantic import BaseModel
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIModel
    PYDAI_AVAILABLE = True
except ImportError:
    logging.warning("Pydantic AI not available - using fallback content cleaning")
    PYDAI_AVAILABLE = False
    BaseModel = None

# Performance timer imports
try:
    from ..utils.performance_timers import async_timed
except ImportError:
    # Fallback no-op decorator if performance_timers not available
    def async_timed(metadata=None):
        def decorator(func):
            return func
        return decorator

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
    key_points: list[str]
    topics_detected: list[str]
    cleaning_notes: list[str]
    processing_time: float
    model_used: str
    salient_points: str = ""  # 300-word bullet summary of key facts/themes


@dataclass
class ContentCleaningContext:
    """Context for content cleaning operations."""
    search_query: str
    query_terms: list[str]
    url: str
    source_domain: str
    session_id: str
    min_quality_threshold: int = 50
    max_content_length: int = 50000


# Pydantic model for structured AI output
if PYDAI_AVAILABLE and BaseModel:
    class CleanedContentOutput(BaseModel):
        """Structured output from AI content cleaning."""
        cleaned_content: str
        quality_score: int  # 0-100
        relevance_score: float  # 0.0-1.0
        key_points: list[str]
        topics_detected: list[str]
        cleaning_notes: list[str]
        salient_points: str  # 300-word bullet summary of key facts/themes
else:
    CleanedContentOutput = None


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

    def __init__(self, model_name: str = "gpt-5-nano", api_key: str | None = None):
        """
        Initialize the content cleaner agent.

        Args:
            model_name: OpenAI model to use (gpt-5-nano)
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
                # Note: Pydantic AI gets API key from OPENAI_API_KEY environment variable
                model = OpenAIModel(self.model_name)
                self.agent = Agent(
                    model,
                    output_type=CleanedContentOutput,
                    system_prompt=self._get_system_prompt(),
                    deps_type=ContentCleaningContext
                )
                logger.info(f"Content cleaner agent initialized with model: {model_name} (structured output: CleanedContentOutput)")
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
            'quality_distribution': {},
            'llm_attempts': 0,
            'llm_successes': 0,
            'llm_timeouts': 0,
            'llm_exceptions': 0,
            'llm_quality_rejections': 0,
            'llm_noise_rejections': 0,
            'llm_length_rejections': 0
        }

    def _calculate_adaptive_timeout(self, content_size_bytes: int) -> float:
        """
        Calculate adaptive timeout based on content size.
        
        Small articles clean quickly (2 min), large articles need more time (5 min).
        
        Args:
            content_size_bytes: Size of raw scraped content in bytes
            
        Returns:
            Timeout in seconds (120-300)
        """
        size_kb = content_size_bytes / 1024
        
        if size_kb < 50:
            # Small article: 2 minutes
            return 120.0
        elif size_kb < 200:
            # Medium article: 3 minutes  
            return 180.0
        elif size_kb < 500:
            # Large article: 4 minutes
            return 240.0
        else:
            # Very large: 5 minutes (likely garbage but give it a shot)
            return 300.0

    @async_timed(metadata={"category": "cleaning", "stage": "ai_processing"})
    async def clean_content(
        self,
        raw_content: str,
        context: ContentCleaningContext
    ) -> CleanedContent | None:
        """
        Binary LLM-only content cleaning with NO fallback.
        
        Either succeeds with high-quality LLM cleaning, or returns None.
        NO fallback to rule-based cleaning - binary success/fail only.

        Args:
            raw_content: Raw content from web crawling
            context: Cleaning context with search query and metadata

        Returns:
            CleanedContent if successful, None if failed (timeout, error, or low quality)
        """
        start_time = datetime.now()
        self.stats['llm_attempts'] += 1

        try:
            # Pre-processing and basic filtering
            if not self._should_process_content(raw_content, context):
                logger.error(f"🚫 PRE-FILTER REJECT: {context.url} - Content failed basic checks")
                return None

            # Require AI agent - no fallback
            if not self.agent:
                logger.error(f"🚫 NO LLM AGENT: {context.url} - Cannot clean without LLM")
                return None

            # Binary LLM-only cleaning
            result = await self._ai_clean_content_binary(raw_content, context)
            
            if result:
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                result.processing_time = processing_time
                
                # Update statistics
                self._update_stats(result)
                self.stats['llm_successes'] += 1
                
                logger.info(f"✅ LLM SUCCESS: {context.url} quality={result.quality_score} "
                           f"words={result.word_count} time={processing_time:.1f}s")
            
            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['llm_exceptions'] += 1
            logger.error(f"🚫 LLM EXCEPTION: {context.url} - {str(e)} - REJECTING (no fallback)")
            return None

    async def clean_multiple_contents(
        self,
        contents: list[tuple[str, ContentCleaningContext]],
        max_concurrent: int | None = None
    ) -> list[CleanedContent]:
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
                   f"max_concurrent={max_concurrent if max_concurrent and max_concurrent > 0 else 'unbounded'}")

        # Create semaphore to limit concurrent operations (optional)
        semaphore = (
            asyncio.Semaphore(max_concurrent)
            if max_concurrent and max_concurrent > 0
            else None
        )

        async def clean_with_semaphore(content: tuple[str, ContentCleaningContext]) -> CleanedContent:
            raw_content, context = content
            if semaphore is None:
                return await self.clean_content(raw_content, context)

            async with semaphore:
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
                    model_used="failed",
                    salient_points=""
                ))
            else:
                final_results.append(result)

        # Log batch summary
        total_time = (datetime.now() - start_time).total_seconds()
        avg_quality = sum(r.quality_score for r in final_results) / len(final_results)
        # Use first context's threshold (all should have same threshold)
        min_threshold = contents[0][1].min_quality_threshold if contents else 50
        successful = sum(1 for r in final_results if r.quality_score >= min_threshold)

        logger.info(f"Batch cleaning completed: {successful}/{len(final_results)} passed quality threshold "
                   f"(avg quality: {avg_quality:.1f}, total time: {total_time:.1f}s)")

        return final_results

    async def _ai_clean_content_binary(
        self,
        raw_content: str,
        context: ContentCleaningContext
    ) -> CleanedContent | None:
        """
        Binary LLM-only cleaning with NO fallback and strict validation.
        
        Returns CleanedContent if successful, None if failed for any reason.
        """
        try:
            # Calculate adaptive timeout based on content size
            content_size = len(raw_content.encode('utf-8'))
            adaptive_timeout = self._calculate_adaptive_timeout(content_size)
            
            logger.info(f"🤖 LLM CLEANING: {context.url} ({content_size/1024:.1f}KB, timeout={adaptive_timeout:.0f}s)")
            
            # Prepare the cleaning prompt
            cleaning_prompt = self._create_cleaning_prompt(raw_content, context)

            # Run the AI agent with adaptive timeout
            try:
                result = await asyncio.wait_for(
                    self.agent.run(cleaning_prompt, deps=context),
                    timeout=adaptive_timeout
                )
            except asyncio.TimeoutError:
                self.stats['llm_timeouts'] += 1
                logger.error(f"🚫 LLM TIMEOUT: {context.url} after {adaptive_timeout:.0f}s - REJECTING (no fallback)")
                return None

            # Parse the structured result from Pydantic model
            cleaned_data = result.output

            # STRICT VALIDATION: Quality score must be >= 70
            if cleaned_data.quality_score < 70:
                self.stats['llm_quality_rejections'] += 1
                logger.error(f"🚫 LOW QUALITY: {context.url} score={cleaned_data.quality_score} - REJECTING")
                return None
            
            # STRICT VALIDATION: Check for noise indicators
            noise_indicators = [
                'sign in', 'subscribe', 'menu', 'navigation', 'accept cookies',
                'facebook', 'twitter', 'instagram', 'linkedin', 'cookie consent',
                'privacy policy', 'newsletter'
            ]
            noise_count = sum(1 for n in noise_indicators if n in cleaned_data.cleaned_content.lower())
            
            if noise_count > 5:
                self.stats['llm_noise_rejections'] += 1
                logger.error(f"🚫 TOO MUCH NOISE: {context.url} noise_indicators={noise_count} - REJECTING")
                return None
            
            # STRICT VALIDATION: Minimum word count
            word_count = len(cleaned_data.cleaned_content.split())
            if word_count < 100:
                self.stats['llm_length_rejections'] += 1
                logger.error(f"🚫 TOO SHORT: {context.url} words={word_count} - REJECTING")
                return None

            # SUCCESS - All validations passed
            return CleanedContent(
                original_content=raw_content,
                cleaned_content=cleaned_data.cleaned_content,
                quality_score=cleaned_data.quality_score,
                quality_level=self._get_quality_level(cleaned_data.quality_score),
                relevance_score=cleaned_data.relevance_score,
                word_count=word_count,
                char_count=len(cleaned_data.cleaned_content),
                key_points=cleaned_data.key_points,
                topics_detected=cleaned_data.topics_detected,
                cleaning_notes=cleaned_data.cleaning_notes,
                processing_time=0.0,  # Will be set by caller
                model_used=self.model_name,
                salient_points=cleaned_data.salient_points
            )

        except Exception as e:
            self.stats['llm_exceptions'] += 1
            logger.error(f"🚫 LLM EXCEPTION: {context.url} - {str(e)} - REJECTING (no fallback)")
            return None

    async def _rule_based_clean_content(
        self,
        raw_content: str,
        context: ContentCleaningContext
    ) -> CleanedContent:
        """Fallback rule-based content cleaning."""
        try:
            # Apply enhanced modern web content cleaning
            from multi_agent_research_system.utils.modern_content_cleaner import (
                ModernWebContentCleaner,
            )

            modern_cleaner = ModernWebContentCleaner(logger)
            cleaned_text = modern_cleaner.clean_article_content(
                raw_content,
                context.search_query
            )

            # Calculate relevance to search query
            relevance_score = self._calculate_relevance_score(
                cleaned_text, context.search_query, context.query_terms
            )

            # Estimate quality based on content characteristics
            quality_score = self._estimate_quality_score(cleaned_text, relevance_score)

            # Extract key points (basic approach)
            key_points = self._extract_key_points(cleaned_text)

            # Detect topics (basic approach)
            topics = self._detect_topics(cleaned_text)

            # Include cleaning metadata
            cleaning_notes = [
                "Enhanced rule-based cleaning with modern web patterns",
                f"Original length: {len(raw_content)} chars",
                f"Cleaned length: {len(cleaned_text)} chars",
                f"Estimated quality score: {quality_score}/100"
            ]

            return CleanedContent(
                original_content=raw_content,
                cleaned_content=cleaned_text,
                quality_score=quality_score,
                quality_level=self._get_quality_level(quality_score),
                relevance_score=relevance_score,
                word_count=len(cleaned_text.split()),
                char_count=len(cleaned_text),
                key_points=key_points,
                topics_detected=topics,
                cleaning_notes=cleaning_notes,
                processing_time=0.0,  # Will be set by caller
                model_used="enhanced_rule_based_modern",
                salient_points=""  # Rule-based cleaning doesn't generate salient points
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

    def _calculate_relevance_score(self, content: str, query: str, query_terms: list[str]) -> float:
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

    def _extract_key_points(self, content: str) -> list[str]:
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

    def _detect_topics(self, content: str) -> list[str]:
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
            model_used="none",
            salient_points=""
        )

    def _get_system_prompt(self) -> str:
        """Get the enhanced system prompt for the AI content cleaner."""
        return """You are an expert content cleaner specializing in modern news websites. Your task is to remove ALL noise and preserve ONLY valuable article content.

REMOVE THESE SPECIFIC ELEMENTS:

MODERN WEBSITE NAVIGATION:
- Header navigation bars: "Home", "News", "World", "Politics", "Tech", "Sports", etc.
- Breadcrumb navigation: "Home > World > Latin America > Venezuela"
- Category menus and dropdown navigation
- Site search bars and search forms
- Language selection dropdowns
- Mobile navigation menus

COOKIE & PRIVACY BANNERS:
- "Accept Cookies", "Privacy Policy", "Terms of Use" banners
- GDPR consent forms and privacy notices
- Cookie preference centers and settings
- "We use cookies" notices
- Privacy policy links and disclaimers

SOCIAL MEDIA & SHARING:
- Facebook, Twitter/X, Instagram, TikTok, YouTube, Reddit links
- Share buttons, social sharing widgets
- Follow buttons, social media icons
- Newsletter signup forms
- "Connect with us" sections

SITE UTILITIES:
- "About Us", "Contact Us", "Careers", "Advertise"
- Copyright notices, legal disclaimers
- Site maps, help sections, FAQs
- Mobile app download links
- Accessibility tools

CONTENT RECOMMENDATIONS:
- "Related Articles", "More from Author", "Recommended for You"
- Trending topics sections, "Most Popular"
- Author bios and headshots
- Opinion pieces from other authors
- Sponsored content labels

# Note: Images, videos, and multimedia are handled at scraping level via Crawl4AI text_mode=True
# Any remaining multimedia references should be minimal but can be removed if found

ADVERTISING & PROMOTIONAL:
- Ad placeholders, "Advertisement" labels
- Sponsored content, "Paid Promotion"
- Donation requests, "Support Our Journalism"
- Subscription prompts, paywall messages
- Affiliate links and promotional content

FOOTER BOILERPLATE:
- Repeated navigation links
- Social media link lists
- Legal disclaimers repeated multiple times
- Site edition links (US, International, etc.)
- Copyright notices

PRESERVE ONLY THESE ELEMENTS:
- Main article headline and subheadings
- Article body paragraphs and text content
- Direct quotes from sources/experts
- Key data points, statistics, numbers
- Context that directly relates to the search query
- Attributed statements and source references

CLEANING CRITICAL REQUIREMENTS:
✅ Remove ALL cookie consent banners and privacy notices
✅ Remove ALL navigation menus and breadcrumbs
✅ Remove ALL social media sharing widgets and links
✅ Remove ALL newsletter signup forms and prompts
✅ Remove ALL "support our journalism" donation requests
✅ Remove ALL "trending now" and "recommended" content blocks
✅ Remove ALL author bios and bylines unless they contain relevant information
✅ Remove ALL repeated footer content and legal disclaimers
✅ Remove ALL multimedia references (images, videos, galleries)
✅ Remove ALL advertising and promotional content
✅ Remove ALL site utility links and menus

If an article is primarily navigation, ads, or boilerplate with minimal relevant content, return a low quality score and minimal cleaned content.

Be AGGRESSIVE in removing noise. When in doubt, remove it. Better to have less content that's clean than more content that's noisy.

Return your analysis in this JSON format:
{
    "cleaned_content": "The cleaned and formatted content with all noise removed",
    "quality_score": 85,
    "relevance_score": 0.78,
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "topics_detected": ["topic1", "topic2"],
    "cleaning_notes": ["Note about cleaning process"],
    "salient_points": "• Specific fact/statistic with numbers\n• Key theme or argument\n• Notable quote or expert opinion\n• Unique insight from this article"
}

QUALITY EVALUATION CRITERIA:
- Relevance to search query (25 points)
- Content depth and substance after cleaning (25 points)
- Readability and organization (20 points)
- Source credibility indicators (15 points)
- Information accuracy and completeness (15 points)

RELEVANCE ASSESSMENT:
- Direct term matching (40%)
- Conceptual relevance (30%)
- Information density (20%)
- Context alignment (10%)"""

    def _create_cleaning_prompt(self, content: str, context: ContentCleaningContext) -> str:
        """Create the enhanced cleaning prompt for the AI."""
        return f"""Please clean and evaluate this web content for the search query: "{context.search_query}"

URL: {context.url}
Source Domain: {context.source_domain}
Query Terms: {', '.join(context.query_terms)}

ENHANCED CLEANING INSTRUCTIONS: Apply aggressive content cleaning for modern news websites. Remove ALL noise elements while preserving only high-value article content.

Raw Content (first 15,000 chars):
{content[:15000]}

CRITICAL CLEANING CHECKLIST:
✅ Remove ALL cookie consent banners and privacy notices
✅ Remove ALL navigation menus and breadcrumbs (Home > World > Venezuela)
✅ Remove ALL social media widgets and sharing links
✅ Remove ALL newsletter signup forms and prompts
✅ Remove ALL donation requests ("Support Our Journalism")
✅ Remove ALL "trending now" and "recommended" content
✅ Remove ALL author bios unless they contain relevant information
✅ Remove ALL repeated footer content and legal disclaimers
✅ Remove ALL multimedia references (images, videos, galleries)
✅ Remove ALL advertising and promotional content
✅ Remove ALL site utility links (About Us, Contact Us, Careers)

MODERN WEBSITE ELEMENTS TO REMOVE:
- Language selection dropdowns
- Mobile navigation menus
- GDPR compliance forms
- Cookie preference centers
- Social media icons and follow buttons
- "Connect with us" sections
- Site edition links (US, International)
- Accessibility tools
- Mobile app download links

PRESERVE ONLY:
- Main article headline and subheadings
- Article body paragraphs and text content
- Direct quotes from sources/experts
- Key data points, statistics, numbers
- Context directly relating to "{context.search_query}"
- Attributed statements and source references

QUALITY STANDARDS:
- Content must be highly relevant to search query
- Remove any content that is primarily navigation or boilerplate
- Maintain logical flow and proper paragraph structure
- Ensure readability and organization

SALIENT POINTS REQUIREMENTS (CRITICAL):
Generate a ~300-word summary in bullet format that captures SPECIFIC, INTERESTING information:
• Focus on concrete facts, statistics, dates, numbers
• Main themes and arguments (not generic summary)
• Notable quotes or expert opinions
• Unique insights specific to this article
• DO NOT write generic summaries - be specific and factual
• Each bullet should contain actionable research information

Example good salient_points:
"• Trump brokered ceasefire with 20 living hostages released on October 13, 2025\n• Exchange involved 250 Palestinian prisoners serving life sentences plus 1,718 other detainees\n• 20-point peace plan signed at Egypt summit with leaders from 20+ countries\n• Key challenge: Hamas missed deadline for returning deceased hostages' remains\n• Israel controls 53% of Gaza territory after partial withdrawal"

Example bad salient_points (too generic):
"• Article discusses recent developments\n• Peace deal was announced\n• Many people were involved\n• There are ongoing challenges"

Please analyze and clean this content according to the detailed instructions. Return your response in valid JSON format with ALL fields including salient_points."""

    def get_binary_cleaning_stats(self) -> dict:
        """
        Get comprehensive binary LLM cleaning statistics.
        
        Returns:
            Dictionary with detailed cleaning metrics
        """
        attempts = self.stats['llm_attempts']
        successes = self.stats['llm_successes']
        
        if attempts == 0:
            return {"message": "No LLM cleaning attempts yet"}
        
        success_rate = (successes / attempts) * 100
        
        return {
            "llm_attempts": attempts,
            "llm_successes": successes,
            "llm_timeouts": self.stats['llm_timeouts'],
            "llm_exceptions": self.stats['llm_exceptions'],
            "llm_quality_rejections": self.stats['llm_quality_rejections'],
            "llm_noise_rejections": self.stats['llm_noise_rejections'],
            "llm_length_rejections": self.stats['llm_length_rejections'],
            "success_rate_percent": round(success_rate, 1),
            "total_rejections": attempts - successes,
            "rejection_breakdown": {
                "timeouts": self.stats['llm_timeouts'],
                "exceptions": self.stats['llm_exceptions'],
                "low_quality": self.stats['llm_quality_rejections'],
                "too_noisy": self.stats['llm_noise_rejections'],
                "too_short": self.stats['llm_length_rejections']
            }
        }
    
    def log_binary_cleaning_summary(self):
        """Log a summary of binary LLM cleaning performance."""
        stats = self.get_binary_cleaning_stats()
        
        if "message" in stats:
            logger.info(stats["message"])
            return
        
        logger.info("=" * 60)
        logger.info("🤖 BINARY LLM CLEANING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Attempts: {stats['llm_attempts']}")
        logger.info(f"✅ Successes: {stats['llm_successes']} ({stats['success_rate_percent']}%)")
        logger.info(f"🚫 Rejections: {stats['total_rejections']}")
        logger.info("")
        logger.info("Rejection Breakdown:")
        logger.info(f"  ⏱️  Timeouts: {stats['llm_timeouts']}")
        logger.info(f"  ❌ Exceptions: {stats['llm_exceptions']}")
        logger.info(f"  📊 Low Quality: {stats['llm_quality_rejections']}")
        logger.info(f"  🔊 Too Noisy: {stats['llm_noise_rejections']}")
        logger.info(f"  📏 Too Short: {stats['llm_length_rejections']}")
        logger.info("=" * 60)
    
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

    def get_stats(self) -> dict[str, Any]:
        """Get cleaning statistics."""
        return {
            **self.stats,
            'agent_available': self.agent is not None,
            'model_used': self.model_name
        }


# Global content cleaner instance
_global_content_cleaner: ContentCleanerAgent | None = None


def get_content_cleaner() -> ContentCleanerAgent:
    """Get or create global content cleaner agent."""
    global _global_content_cleaner
    if _global_content_cleaner is None:
        _global_content_cleaner = ContentCleanerAgent()
    return _global_content_cleaner
