"""
Query Enhancement Module for Intelligent URL Selection

This module uses GPT-5 Nano to transform user queries into optimized search queries
with orthogonal exploration for comprehensive research coverage.

Key Features:
- GPT-5 Nano integration for fast, cost-effective query optimization
- Primary query enhancement for maximum search relevance
- Two orthogonal queries for diverse perspective exploration
- Fallback mechanisms for reliability
- Structured query generation with validation
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQueries:
    """Container for enhanced query results."""
    primary_query: str
    orthogonal_query_1: str
    orthogonal_query_2: str
    enhancement_metadata: dict


class QueryEnhancer:
    """
    Intelligent query enhancement system using GPT-5 Nano.

    Transforms user queries into three optimized search queries:
    1. Primary: Enhanced for maximum search relevance
    2. Orthogonal 1: Complementary perspective exploration
    3. Orthogonal 2: Alternative angle for comprehensive coverage
    """

    def __init__(self, model: str = "gpt-5-nano"):
        """
        Initialize the query enhancer.

        Args:
            model: OpenAI model to use (default: gpt-5-nano for speed and cost efficiency)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")

        self.model = model
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client with API key."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not found in environment variables")
                raise RuntimeError("OPENAI_API_KEY required for query enhancement")

            self.client = AsyncOpenAI(api_key=api_key)
            logger.info(f"Query enhancer initialized with model: {self.model}")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    async def enhance_query(self, original_query: str, session_id: str = "default") -> EnhancedQueries:
        """
        Enhance a user query into three optimized search queries.

        Args:
            original_query: The user's original research query
            session_id: Session identifier for tracking

        Returns:
            EnhancedQueries object with three optimized queries and metadata
        """
        try:
            logger.info(f"Enhancing query for session {session_id}: '{original_query}'")

            # Generate enhanced queries in parallel for efficiency
            primary_task = asyncio.create_task(self._generate_primary_query(original_query))
            orthogonal_1_task = asyncio.create_task(self._generate_orthogonal_query_1(original_query))
            orthogonal_2_task = asyncio.create_task(self._generate_orthogonal_query_2(original_query))

            # Wait for all queries to complete
            primary, orthogonal_1, orthogonal_2 = await asyncio.gather(
                primary_task, orthogonal_1_task, orthogonal_2_task,
                return_exceptions=True
            )

            # Handle exceptions and provide fallbacks
            if isinstance(primary, Exception):
                logger.warning(f"Primary query enhancement failed: {primary}")
                primary = original_query  # Fallback to original

            if isinstance(orthogonal_1, Exception):
                logger.warning(f"Orthogonal query 1 enhancement failed: {orthogonal_1}")
                orthogonal_1 = self._create_fallback_orthogonal_query(original_query, 1)

            if isinstance(orthogonal_2, Exception):
                logger.warning(f"Orthogonal query 2 enhancement failed: {orthogonal_2}")
                orthogonal_2 = self._create_fallback_orthogonal_query(original_query, 2)

            # Validate and enhance queries
            primary = self._validate_and_clean_query(primary, original_query, "primary")
            orthogonal_1 = self._validate_and_clean_query(orthogonal_1, original_query, "orthogonal_1")
            orthogonal_2 = self._validate_and_clean_query(orthogonal_2, original_query, "orthogonal_2")

            enhanced_queries = EnhancedQueries(
                primary_query=primary,
                orthogonal_query_1=orthogonal_1,
                orthogonal_query_2=orthogonal_2,
                enhancement_metadata={
                    "original_query": original_query,
                    "session_id": session_id,
                    "model_used": self.model,
                    "enhancement_timestamp": asyncio.get_event_loop().time()
                }
            )

            logger.info(f"Query enhancement completed for session {session_id}")
            logger.debug(f"Primary: '{primary}'")
            logger.debug(f"Orthogonal 1: '{orthogonal_1}'")
            logger.debug(f"Orthogonal 2: '{orthogonal_2}'")

            return enhanced_queries

        except Exception as e:
            logger.error(f"Query enhancement failed for session {session_id}: {e}")
            # Return fallback queries on complete failure
            return self._create_fallback_queries(original_query, session_id)

    async def _generate_primary_query(self, original_query: str) -> str:
        """Generate optimized primary search query."""
        prompt = f"""
You are an expert at crafting search queries for research. Transform the following user query into an optimized search query that will return the most relevant and authoritative results from Google.

USER QUERY: {original_query}

REQUIREMENTS:
1. Create a simple, clear search query that captures the main intent
2. Use natural language that people actually search for (4-8 words max)
3. Include ONLY 2-3 key terms from the original query
4. CRITICAL: Keep query under 80 characters (much shorter is better)
5. AVOID complex operators, quotes, OR statements, or site: restrictions
6. Focus on what actually returns good search results
7. Make it readable and natural, like a real search query

Return ONLY the optimized search query, no additional text or explanation.
"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert search query optimizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    async def _generate_orthogonal_query_1(self, original_query: str) -> str:
        """Generate complementary orthogonal query."""
        prompt = f"""
Generate a complementary search query that explores related concepts and different angles of this research topic.

ORIGINAL TOPIC: {original_query}

REQUIREMENTS:
1. Focus on applications and impacts related to the topic
2. Use very simple, natural search language (3-8 words max)
3. Include ONLY 2-3 key terms that complement the original query
4. CRITICAL: Keep query under 100 characters (much shorter is better)
5. AVOID complex sentences, commas, or lists
6. Make it a simple search phrase someone would actually type
7. Focus on practical applications or recent trends

Return ONLY the complementary search query, no additional text or explanation.
"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at finding diverse perspectives for research topics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    async def _generate_orthogonal_query_2(self, original_query: str) -> str:
        """Generate alternative orthogonal query."""
        prompt = f"""
Create an alternative search query that approaches this topic from a different perspective.

ORIGINAL TOPIC: {original_query}

REQUIREMENTS:
1. Focus on challenges, solutions, or future outlooks related to the topic
2. Use different but related terminology (3-7 words max)
3. Keep it very simple and natural, like something someone would search for
4. CRITICAL: Keep query under 80 characters (shorter is much better)
5. AVOID complex sentences, commas, or lists
6. Consider human impact, challenges, or future developments
7. Make it a simple search phrase people actually type

Return ONLY the alternative search query, no additional text or explanation.
"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at finding diverse approaches to research topics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.8
        )

        return response.choices[0].message.content.strip()

    def _validate_and_clean_query(self, query: str, original_query: str, query_type: str) -> str:
        """
        Validate and clean the generated query.

        Args:
            query: Generated query to validate
            original_query: Original user query for context
            query_type: Type of query for validation context

        Returns:
            Cleaned and validated query
        """
        if not query or len(query.strip()) < 3:
            logger.warning(f"Generated {query_type} query too short, using fallback")
            return self._create_fallback_orthogonal_query(original_query, query_type)

        # Clean up common issues
        query = query.strip()

        # Remove any explanatory text (LLMs sometimes add explanations)
        if any(phrase in query.lower() for phrase in ['here is', 'this query', 'the search', 'optimized query:']):
            # Try to extract just the query part
            lines = query.split('\n')
            for line in lines:
                if len(line.strip()) > 5 and not any(word in line.lower() for word in ['here is', 'this query', 'the search']):
                    query = line.strip()
                    break

        # Ensure it's not too long for SERP API (256 char limit, using 200 as safe target)
        if len(query) > 200:
            logger.warning(f"Generated {query_type} query too long ({len(query)} chars), truncating")
            query = query[:200].rsplit(' ', 1)[0]  # Truncate at word boundary

        return query

    def _create_fallback_orthogonal_query(self, original_query: str, query_number: int) -> str:
        """Create a simple fallback orthogonal query."""
        query_terms = original_query.lower().split()

        if query_number == 1:
            # Add context/applications terms
            additions = ["applications", "implementation", "examples", "use cases", "case studies"]
            for addition in additions:
                if addition not in original_query.lower():
                    return f"{original_query} {addition}"
            return original_query

        elif query_number == 2:
            # Add alternative perspective terms
            additions = ["challenges", "limitations", "benefits", "advantages", "disadvantages"]
            for addition in additions:
                if addition not in original_query.lower():
                    return f"{original_query} {addition}"
            return original_query

        return original_query

    def _create_fallback_queries(self, original_query: str, session_id: str) -> EnhancedQueries:
        """Create fallback queries when all enhancement fails."""
        logger.warning(f"Using fallback queries for session {session_id}")

        return EnhancedQueries(
            primary_query=original_query,
            orthogonal_query_1=self._create_fallback_orthogonal_query(original_query, 1),
            orthogonal_query_2=self._create_fallback_orthogonal_query(original_query, 2),
            enhancement_metadata={
                "original_query": original_query,
                "session_id": session_id,
                "model_used": "fallback",
                "enhancement_timestamp": asyncio.get_event_loop().time(),
                "fallback_reason": "complete_enhancement_failure"
            }
        )


# Global instance for reuse
_query_enhancer_instance = None

def get_query_enhancer(model: str = "gpt-5-nano") -> QueryEnhancer:
    """
    Get or create a query enhancer instance.

    Args:
        model: OpenAI model to use

    Returns:
        QueryEnhancer instance
    """
    global _query_enhancer_instance

    if _query_enhancer_instance is None:
        _query_enhancer_instance = QueryEnhancer(model=model)

    return _query_enhancer_instance


async def enhance_user_query(original_query: str, session_id: str = "default") -> EnhancedQueries:
    """
    Convenience function to enhance a user query.

    Args:
        original_query: The user's original research query
        session_id: Session identifier for tracking

    Returns:
        EnhancedQueries object with three optimized queries
    """
    enhancer = get_query_enhancer()
    return await enhancer.enhance_query(original_query, session_id)