"""
Fix for Content Cleaning Errors: 'str' object has no attribute 'get'

This script provides robust result extraction from pydantic_ai agents.
"""

import logging
from typing import Any, Union

logger = logging.getLogger(__name__)

def safe_extract_content_from_result(result: Any) -> str:
    """
    Safely extract content from pydantic_ai result objects.

    This function handles various result object structures and prevents
    'str' object has no attribute 'get' errors.

    Args:
        result: The result object from pydantic_ai agent

    Returns:
        Extracted content as string
    """
    try:
        # Handle None result
        if result is None:
            logger.warning("Received None result from pydantic_ai agent")
            return ""

        # Direct string result
        if isinstance(result, str):
            return result.strip()

        # Dictionary-like result
        if isinstance(result, dict):
            # Try common keys
            for key in ['data', 'content', 'text', 'output', 'response', 'message']:
                if key in result:
                    value = result[key]
                    if isinstance(value, str):
                        return value.strip()
                    elif hasattr(value, '__str__'):
                        return str(value).strip()

            # If no known keys found, convert entire dict to string
            logger.warning(f"Unknown dict structure in pydantic_ai result: {list(result.keys())}")
            return str(result)

        # Object with attributes
        if hasattr(result, '__dict__'):
            # Try common attributes
            for attr in ['data', 'content', 'text', 'output', 'response', 'message']:
                if hasattr(result, attr):
                    value = getattr(result, attr)
                    if isinstance(value, str):
                        return value.strip()
                    elif hasattr(value, '__str__'):
                        return str(value).strip()

            # Try to convert object to string
            try:
                return str(result).strip()
            except Exception as e:
                logger.error(f"Failed to convert pydantic_ai result to string: {e}")
                return ""

        # Fallback: try to convert to string
        try:
            return str(result).strip()
        except Exception as e:
            logger.error(f"Failed to extract content from pydantic_ai result: {e}")
            return ""

    except Exception as e:
        logger.error(f"Error in safe_extract_content_from_result: {e}")
        return ""

def safe_extract_score_from_result(result: Any) -> float:
    """
    Safely extract numeric score from pydantic_ai result.

    Args:
        result: The result object from pydantic_ai agent

    Returns:
        Numeric score as float
    """
    try:
        content = safe_extract_content_from_result(result)

        # Try to extract float from content
        import re
        # Look for patterns like "0.7", "0.75", "1.0", etc.
        score_patterns = [
            r'(\d+\.?\d*)',  # Numbers like 0.7, 1.0, etc.
            r'(\d+)/(\d+)',  # Fractions like 7/10
        ]

        for pattern in score_patterns:
            matches = re.findall(pattern, content)
            if matches:
                if pattern == r'(\d+)/(\d+)':
                    # Handle fractions
                    numerator, denominator = float(matches[0][0]), float(matches[0][1])
                    if denominator > 0:
                        return numerator / denominator
                else:
                    # Handle decimal numbers
                    score = float(matches[0])
                    if 0 <= score <= 1:
                        return score
                    elif score > 1:
                        return score / 10  # Normalize scores > 1

        # If no numeric pattern found, return default
        logger.warning(f"Could not extract numeric score from: {content[:100]}...")
        return 0.5  # Default middle score

    except Exception as e:
        logger.error(f"Error extracting score from pydantic_ai result: {e}")
        return 0.5

def apply_fix_to_assess_content_cleanliness():
    """
    Instructions for fixing the assess_content_cleanliness function.
    """

    fix_code = '''
    # Replace lines 66-84 in content_cleaning.py with this:

    # Extract the response using safe extraction
    response = safe_extract_content_from_result(result)

    try:
        score = float(response)
        score = max(0.0, min(1.0, score))  # Clamp to 0.0-1.0
    except ValueError:
        # Fallback: try to extract numeric score from text
        score = safe_extract_score_from_result(result)

    is_clean = score >= threshold
    '''

    return fix_code

def apply_fix_to_clean_content_with_gpt5_nano():
    """
    Instructions for fixing the clean_content_with_gpt5_nano function.
    """

    fix_code = '''
    # Replace lines 170-176 in content_cleaning.py with this:

    # Extract the actual content using safe extraction
    cleaned_content = safe_extract_content_from_result(result)
    '''

    return fix_code

# Updated functions to replace in content_cleaning.py
async def fixed_assess_content_cleanliness(content: str, url: str, threshold: float = 0.7) -> tuple[bool, float]:
    """
    Fixed version of assess_content_cleanliness with robust result extraction.
    """
    try:
        from pydantic_ai import Agent

        judge_agent = Agent(
            model="openai:gpt-5-nano",
            system_prompt="""You are a content cleanliness judge. Assess if web content is clean enough to use as-is.

EVALUATE FOR:
✅ CLEAN (high score):
- Main content clearly separated from navigation
- Minimal ads, popups, or promotional content
- Technical documentation, articles, or news content
- Good content-to-noise ratio
- Readable structure and formatting

❌ DIRTY (low score):
- Heavy navigation, menus, sidebars
- Many ads, subscription prompts, or popups
- Poor content-to-noise ratio
- Fragmented or poorly structured content
- Social media feeds or comment sections mixed in

RESPOND WITH ONLY A SCORE: 0.0 (very dirty) to 1.0 (very clean)
Consider 0.7+ as "clean enough" for most use cases."""
        )

        assessment_prompt = f"""Assess cleanliness of this content from {url}:

Content length: {len(content)} characters
Sample (first 2000 chars):
{content[:2000]}

Score this content's cleanliness (0.0-1.0):"""

        result = await judge_agent.run(assessment_prompt)

        # Use safe extraction to prevent errors
        response = safe_extract_content_from_result(result)

        try:
            score = float(response)
            score = max(0.0, min(1.0, score))  # Clamp to 0.0-1.0
        except ValueError:
            # Fallback: try to extract numeric score from text
            score = safe_extract_score_from_result(result)

        is_clean = score >= threshold

        logger.info(f"Cleanliness assessment for {url}: {score:.2f} ({'clean enough' if is_clean else 'needs cleaning'})")

        return is_clean, score

    except Exception as e:
        logger.error(f"Error in cleanliness assessment: {e}")
        # Conservative fallback: assume needs cleaning
        return False, 0.0

async def fixed_clean_content_with_gpt5_nano(content: str, url: str, search_query: str = None) -> str:
    """
    Fixed version of clean_content_with_gpt5_nano with robust result extraction.
    """
    try:
        # Skip LLM cleaning if content is too short
        if len(content.strip()) < 500:
            return content

        # Create enhanced cleaning prompt with search query context
        query_context = f"**Search Query Context**: {search_query}\n" if search_query else ""

        cleaning_prompt = f"""You are an expert content extractor specializing in removing web clutter and filtering for relevance. Clean this scraped content by preserving ONLY the main article content that is directly relevant to the search query.

{query_context}**Source URL**: {url}

**CRITICAL: REMOVE ALL UNRELATED ARTICLES**
If this page contains multiple articles or stories, extract ONLY the article most relevant to the search query above. Remove any unrelated news stories, breaking news sections, or "other articles" that don't match the search topic.

**REMOVE COMPLETELY:**
1. **Navigation**: Menus, breadcrumbs, category links, site navigation
2. **Social Media**: Follow buttons, sharing widgets, social platform links
3. **Video/Media Controls**: Player interfaces, timers, modal dialogs, captions settings
4. **Advertisement**: "AD Loading", promotional banners, subscription prompts
5. **Author Clutter**: Detailed bios, contact info, "Writers Page" links (keep only name)
6. **Site Branding**: Logos, taglines, "Skip to content", newsletter signups
7. **Related Content**: "You might also like", trending stories, suggested articles
8. **Translation/Accessibility**: Language dropdowns, AI translation disclaimers
9. **Comments/User Content**: User comments, review sections
10. **Legal/Privacy**: Cookie notices, privacy policy links, terms of service
11. **UNRELATED ARTICLES**: Any complete articles or news stories not related to the search query

**PRESERVE ONLY:**
1. Main article headline (most relevant to search query)
2. Publication date and source name
3. Article body content directly related to the search query
4. Key facts, quotes, and data points from the main story
5. Essential context that helps understand the main article

**OUTPUT FORMAT:**
Return clean markdown with:
- Clear article title
- Clean paragraph structure
- Essential quotes and facts
- No HTML artifacts, navigation elements, or unrelated content

**Raw Content to Clean**:
{content}

**Clean Article Content (relevant to search query only):**"""

        # Use gpt-5-nano for fast content cleaning
        from pydantic_ai import Agent

        cleaning_agent = Agent(
            model="openai:gpt-5-nano",
            system_prompt="You are an expert content extractor that removes navigation and irrelevant elements while preserving main article content."
        )

        # Execute cleaning with no output length restrictions
        result = await cleaning_agent.run(cleaning_prompt)

        # Use safe extraction to prevent errors
        cleaned_content = safe_extract_content_from_result(result)

        # Validate cleaned content isn't too short (indicating over-cleaning)
        if len(cleaned_content.strip()) < 200:
            logger.warning(f"LLM cleaning resulted in very short content for {url}, using original")
            return content

        logger.info(f"Successfully cleaned content for {url}: {len(content)} -> {len(cleaned_content)} chars")
        return cleaned_content.strip()

    except Exception as e:
        logger.error(f"Error in LLM content cleaning for {url}: {e}")
        # Return original content if cleaning fails
        return content

if __name__ == "__main__":
    print("Content Cleaning Error Fix")
    print("=" * 50)
    print("This fix addresses:")
    print("1. Safe extraction from pydantic_ai result objects")
    print("2. Robust numeric score extraction")
    print("3. Prevention of 'str' object has no attribute 'get' errors")
    print("4. Graceful fallback for unknown result structures")
    print("\nReplace the functions in content_cleaning.py with the fixed versions")