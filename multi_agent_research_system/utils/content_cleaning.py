"""
Content cleaning utilities for web crawling results.

This module provides fast content cleaning using gpt-5-nano to remove
navigation, ads, and irrelevant content while preserving main article text.
Includes optimization with cleanliness assessment to skip unnecessary cleaning.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def assess_content_cleanliness(content: str, url: str, threshold: float = 0.7) -> tuple[bool, float]:
    """
    Quickly assess if content is clean enough to use without full cleaning.

    Uses a fast GPT-5-nano judge to evaluate content cleanliness and determine
    if expensive full cleaning can be skipped.

    Args:
        content: Raw scraped content to assess
        url: Source URL for context
        threshold: Cleanliness threshold (0.0-1.0, default 0.7)

    Returns:
        Tuple of (is_clean_enough: bool, cleanliness_score: float)
    """
    try:
        from pydantic_ai import Agent

        # Fast cleanliness judge agent
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

        # Quick assessment prompt
        assessment_prompt = f"""Assess cleanliness of this content from {url}:

Content length: {len(content)} characters
Sample (first 2000 chars):
{content[:2000]}

Score this content's cleanliness (0.0-1.0):"""

        result = await judge_agent.run(assessment_prompt)

        # Extract numeric score using robust result extraction
        if hasattr(result, 'data'):
            response = str(result.data).strip()
        elif hasattr(result, 'output'):
            response = str(result.output).strip()
        else:
            response = str(result).strip()

        try:
            score = float(response)
            score = max(0.0, min(1.0, score))  # Clamp to 0.0-1.0
        except ValueError:
            # Fallback: conservative approach if we can't parse
            logger.warning(f"Could not parse cleanliness score: {response}")
            score = 0.5

        is_clean = score >= threshold

        logger.info(f"Cleanliness assessment for {url}: {score:.2f} ({'clean enough' if is_clean else 'needs cleaning'})")

        return is_clean, score

    except Exception as e:
        logger.error(f"Error in cleanliness assessment: {e}")
        # Conservative fallback: assume needs cleaning
        return False, 0.0


async def clean_content_with_gpt5_nano(content: str, url: str, search_query: str = None) -> str:
    """
    Use gpt-5-nano to intelligently clean extracted content with search query context.

    This function removes navigation, menus, advertisements, footers,
    subscription prompts, unrelated articles, and other irrelevant elements
    while preserving only the main article content relevant to the search query.

    Args:
        content: Raw content to clean
        url: Source URL for context
        search_query: Original search query for relevance filtering

    Returns:
        Cleaned article content relevant to the search query
    """
    try:
        # Skip LLM cleaning if content is too short
        if len(content.strip()) < 500:
            return content

        # Create enhanced cleaning prompt with search query context
        query_context = f"**Search Query Context**: {search_query}\n" if search_query else ""

        cleaning_prompt = f"""You are an expert content extractor specializing in removing web clutter while preserving valuable article content. Clean this scraped content by removing navigation, ads, and page furniture while preserving the full article content including context and analysis.

{query_context}**Source URL**: {url}

**REMOVE COMPLETELY (Web Page Furniture):**
1. **Navigation**: Menus, breadcrumbs, category links, site navigation, header/footer
2. **Social Media**: Follow buttons, sharing widgets, social platform links, social media feeds
3. **Video/Media Controls**: Player interfaces, timers, modal dialogs, captions settings, video embeds
4. **Advertisement**: "AD Loading", promotional banners, subscription prompts, sponsored content
5. **Author Clutter**: Detailed bios, contact info, "Writers Page" links (keep author name and credentials)
6. **Site Branding**: Logos, taglines, "Skip to content", newsletter signups, site notices
7. **Related Articles Section**: "You might also like", trending stories, suggested articles sidebar
8. **Interactive Elements**: Language dropdowns, AI translation disclaimers, accessibility controls
9. **Comments/User Content**: User comments, review sections, user-generated content
10. **Legal/Privacy**: Cookie notices, privacy policy links, terms of service, disclaimers
11. **Technical HTML**: Scripts, stylesheets, meta tags, HTML artifacts

**PRESERVE FULL ARTICLE CONTENT:**
1. **Main Article**: Complete article text, including all paragraphs and sections
2. **Headline & Subheadings**: All article headings and subheadings
3. **Context & Background**: Relevant background information, historical context
4. **Analysis & Insights**: Expert analysis, commentary, strategic assessments
5. **Data & Facts**: Statistics, figures, data points, timelines
6. **Quotes & Sources**: Direct quotes, source attributions, references
7. **Publication Info**: Publication date, author name, source attribution
8. **Related Context**: Brief mentions of related events that provide understanding

**IMPORTANT CONSERVATION GUIDELINES:**
- **Preserve 80-90% of original article content**
- **Remove only web page furniture, not substantive content**
- **Keep analysis, context, and background information**
- **Maintain article's full narrative and analytical flow**
- **Don't cut content just because it's not directly about the search query**
- **Preserve valuable context that helps understanding**

**OUTPUT FORMAT:**
Return clean markdown with:
- Complete article title and all subheadings
- Full article body with paragraph structure
- All analysis, context, and background information preserved
- Essential quotes, facts, and data points
- Publication date and source information
- No HTML artifacts, navigation, or promotional content

**Raw Content to Clean**:
{content}

**Clean Article Content (preserving full article with context):**"""

        # Use gpt-5-nano for fast content cleaning
        from pydantic_ai import Agent

        cleaning_agent = Agent(
            model="openai:gpt-5-nano",
            system_prompt="You are an expert content extractor that removes navigation and irrelevant elements while preserving main article content."
        )

        # Execute cleaning with no output length restrictions
        result = await cleaning_agent.run(cleaning_prompt)

        # Extract the actual content from the pydantic_ai result
        if hasattr(result, 'data'):
            cleaned_content = result.data
        elif hasattr(result, 'output'):
            cleaned_content = result.output
        else:
            cleaned_content = str(result)

        # Validate cleaned content quality
        original_length = len(content)
        cleaned_length = len(cleaned_content.strip())
        compression_ratio = cleaned_length / original_length if original_length > 0 else 1.0

        # Check for over-aggressive cleaning (too much content removed)
        if cleaned_length < 200:
            logger.warning(f"LLM cleaning resulted in very short content for {url} ({cleaned_length} chars), using original")
            return content

        # Check for excessive compression (more than 80% removed)
        if compression_ratio < 0.2 and original_length > 1000:
            logger.warning(f"LLM cleaning was too aggressive for {url}: {original_length} -> {cleaned_length} chars ({compression_ratio:.1%}), using original")
            return content

        # Check for suspiciously low compression on long content (suggests good cleaning)
        if original_length > 5000 and compression_ratio > 0.95:
            logger.info(f"LLM cleaning had minimal effect for {url} ({compression_ratio:.1%} preserved), content was already clean")
        elif compression_ratio < 0.5:
            logger.info(f"LLM cleaning significantly reduced content for {url}: {original_length} -> {cleaned_length} chars ({compression_ratio:.1%})")
        else:
            logger.info(f"LLM cleaning moderately reduced content for {url}: {original_length} -> {cleaned_length} chars ({compression_ratio:.1%})")

        return cleaned_content.strip()

    except Exception as e:
        logger.error(f"Error in LLM content cleaning for {url}: {e}")
        # Return original content if cleaning fails
        return content


async def clean_content_with_judge_optimization(
    content: str,
    url: str,
    search_query: str = None,
    cleanliness_threshold: float = 0.7,
    skip_judge: bool = False
) -> tuple[str, dict]:
    """
    Optimized content cleaning with judge assessment to skip unnecessary cleaning.

    First uses a fast GPT-5-nano judge to assess content cleanliness.
    Only performs expensive full cleaning if content is deemed dirty enough.

    Args:
        content: Raw scraped content to clean
        url: Source URL for context
        search_query: Optional search query for relevance filtering
        cleanliness_threshold: Threshold for skipping cleaning (0.0-1.0, default 0.7)
        skip_judge: If True, skip judge and always do full cleaning

    Returns:
        Tuple of (cleaned_content: str, metadata: dict)
        metadata includes: judge_score, cleaning_performed, processing_time
    """
    import time
    start_time = time.time()

    try:
        # Skip judge assessment if requested
        if skip_judge:
            logger.info(f"Skipping judge assessment for {url}, performing full cleaning")
            cleaned_content = await clean_content_with_gpt5_nano(content, url, search_query)

            metadata = {
                "judge_score": None,
                "cleaning_performed": True,
                "processing_time": time.time() - start_time,
                "optimization_used": False
            }

            return cleaned_content, metadata

        # Step 1: Quick cleanliness assessment
        logger.info(f"Assessing content cleanliness for {url}")
        is_clean, judge_score = await assess_content_cleanliness(content, url, cleanliness_threshold)

        if is_clean:
            # Content is clean enough - return original content
            logger.info(f"Content clean enough ({judge_score:.2f} >= {cleanliness_threshold}), skipping full cleaning")

            metadata = {
                "judge_score": judge_score,
                "cleaning_performed": False,
                "processing_time": time.time() - start_time,
                "optimization_used": True,
                "latency_saved": "~35-40 seconds"
            }

            return content, metadata
        else:
            # Content needs cleaning - proceed with full cleaning
            logger.info(f"Content needs cleaning ({judge_score:.2f} < {cleanliness_threshold}), performing full cleaning")
            cleaned_content = await clean_content_with_gpt5_nano(content, url, search_query)

            metadata = {
                "judge_score": judge_score,
                "cleaning_performed": True,
                "processing_time": time.time() - start_time,
                "optimization_used": True
            }

            return cleaned_content, metadata

    except Exception as e:
        logger.error(f"Error in optimized content cleaning for {url}: {e}")
        # Fallback to original content
        metadata = {
            "judge_score": None,
            "cleaning_performed": False,
            "processing_time": time.time() - start_time,
            "optimization_used": False,
            "error": str(e)
        }

        return content, metadata


async def clean_content_batch(content_urls: list[tuple[str, str]], search_query: str = None) -> list[str]:
    """
    Clean multiple content pieces in parallel using gpt-5-nano with search query context.

    Args:
        content_urls: List of (content, url) tuples to clean
        search_query: Original search query for relevance filtering

    Returns:
        List of cleaned content strings relevant to the search query
    """
    try:
        logger.info(f"Starting parallel cleaning of {len(content_urls)} content pieces")

        # Create cleaning tasks for parallel execution with search query context
        cleaning_tasks = [
            clean_content_with_gpt5_nano(content, url, search_query)
            for content, url in content_urls
        ]

        # Wait for all cleaning to complete
        cleaned_results = await asyncio.gather(*cleaning_tasks, return_exceptions=True)

        # Handle any exceptions in results
        final_results = []
        for i, result in enumerate(cleaned_results):
            if isinstance(result, Exception):
                logger.error(f"Cleaning failed for item {i}: {result}")
                # Use original content if cleaning failed
                final_results.append(content_urls[i][0])
            else:
                final_results.append(result)

        logger.info(f"Parallel cleaning completed: {len(final_results)} items processed")
        return final_results

    except Exception as e:
        logger.error(f"Error in batch content cleaning: {e}")
        # Return original content for all items if batch cleaning fails
        return [content for content, url in content_urls]


def format_cleaned_results(cleaned_contents: list[str], urls: list[str], titles: list[str] = None) -> str:
    """
    Format cleaned content results into a structured output.

    Args:
        cleaned_contents: List of cleaned content strings
        urls: List of source URLs
        titles: Optional list of article titles

    Returns:
        Formatted results string
    """
    if not cleaned_contents:
        return "No content was successfully cleaned."

    result_parts = [
        f"# Crawl Results ({len(cleaned_contents)} articles)",
        f"**Total Articles**: {len(cleaned_contents)}",
        ""
    ]

    for i, (content, url) in enumerate(zip(cleaned_contents, urls, strict=False), 1):
        title = titles[i-1] if titles and len(titles) >= i else f"Article {i}"

        result_parts.extend([
            f"## {i}. {title}",
            f"**URL**: {url}",
            "",
            content[:5000] + ("..." if len(content) > 5000 else ""),  # Limit content size
            "",
            "---",
            ""
        ])

    return "\n".join(result_parts)


async def clean_technical_content_with_gpt5_nano(
    content: str,
    url: str,
    search_query: str = None,
    session_id: str = "default"
) -> str:
    """
    Enhanced content cleaning specifically for technical documentation.

    Preserves code examples, installation commands, and technical accuracy
    while removing navigation and irrelevant content.

    Args:
        content: Raw content to clean
        url: Source URL for context
        search_query: Original search query for relevance filtering
        session_id: Session identifier

    Returns:
        Cleaned technical content with preserved code examples
    """
    try:
        # Skip LLM cleaning if content is too short
        if len(content.strip()) < 500:
            return content

        # Create technical content cleaning prompt
        query_context = f"**Search Query Context**: {search_query}\n" if search_query else ""

        technical_cleaning_prompt = f"""You are an expert technical content extractor specializing in preserving code examples and installation instructions. Clean this scraped technical documentation while preserving ALL technical accuracy.

{query_context}**Source URL**: {url}

**CRITICAL: PRESERVE TECHNICAL CONTENT INTEGRITY**
Maintain EXACT syntax for:
- **Installation Commands**: 'pip install package-name', 'npm install package', 'go get package', etc.
- **Import Statements**: 'from package import module', 'import module', 'require package' etc.
- **Code Examples**: ALL code blocks with exact syntax, spacing, and structure
- **API Calls**: Function names, parameters, return values
- **Configuration**: YAML, JSON, XML, and other config formats
- **File Paths**: Exact file paths and directory structures
- **Version Numbers**: Software versions, API versions, dependency versions

**REMOVE COMPLETELY:**
1. **Navigation**: Menus, breadcrumbs, category links, site navigation
2. **Social Media**: Follow buttons, sharing widgets, social platform links
3. **Video/Media Controls**: Player interfaces, timers, modal dialogs
4. **Advertisement**: Promotional banners, subscription prompts
5. **Author Clutter**: Detailed bios, contact info (keep only name)
6. **Site Branding**: Logos, taglines, newsletter signups
7. **Related Content**: "You might also like", suggested articles
8. **Comments/User Content**: User comments, review sections
9. **Legal/Privacy**: Cookie notices, privacy policy links

**PRESERVE TECHNICAL STRUCTURE:**
1. **Code Blocks**: ALL fenced code blocks with exact syntax (```python, ```bash, etc.)
2. **Installation Instructions**: Complete command-line instructions
3. **API Documentation**: Function signatures, parameter descriptions, return types
4. **Configuration Examples**: All configuration file formats
5. **Error Messages**: Technical error messages and solutions
6. **Version Information**: Software version requirements and compatibility

**OUTPUT FORMAT:**
Return clean markdown with:
- Preserved code blocks with exact syntax
- Complete installation commands
- Technical accuracy maintained
- Clear structure and hierarchy
- No HTML artifacts or navigation elements

**Raw Technical Content to Clean:**
{content}

**Clean Technical Content (with preserved code examples):**"""

        # Use gpt-5-nano for fast technical content cleaning
        from pydantic_ai import Agent

        technical_cleaning_agent = Agent(
            model="openai:gpt-5-nano",
            system_prompt="You are a technical content extractor that preserves code examples, installation commands, and API documentation with exact syntax."
        )

        # Execute technical cleaning
        result = await technical_cleaning_agent.run(technical_cleaning_prompt)

        # Extract the actual content from the pydantic_ai result
        if hasattr(result, 'data'):
            cleaned_content = result.data
        elif hasattr(result, 'output'):
            cleaned_content = result.output
        else:
            cleaned_content = str(result)

        # Validate technical content isn't too short (indicating over-cleaning)
        if len(cleaned_content.strip()) < 200:
            logger.warning(f"Technical cleaning resulted in very short content for {url}, using original")
            return content

        # Validate that code examples are preserved
        if '```' in content and '```' not in cleaned_content:
            logger.warning(f"Code blocks were removed during cleaning for {url}, using original")
            return content

        # Validate that common installation commands are preserved
        common_commands = ['pip install', 'npm install', 'go get', 'cargo add', 'apt-get', 'brew install']
        original_has_commands = any(cmd in content for cmd in common_commands)
        cleaned_has_commands = any(cmd in cleaned_content for cmd in common_commands)

        if original_has_commands and not cleaned_has_commands:
            logger.warning(f"Installation commands were corrupted during cleaning for {url}, using original")
            return content

        logger.info(f"Successfully cleaned technical content for {url}: {len(content)} -> {len(cleaned_content)} chars")
        return cleaned_content.strip()

    except Exception as e:
        logger.error(f"Error in technical content cleaning for {url}: {e}")
        # Return original content if cleaning fails
        return content


# Export commonly used functions
__all__ = [
    'clean_content_with_gpt5_nano',
    'clean_content_batch',
    'format_cleaned_results',
    'clean_technical_content_with_gpt5_nano'
]
