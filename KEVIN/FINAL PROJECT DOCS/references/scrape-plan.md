# Advanced Scraping Integration Plan - Claude Agent SDK Approach

**Date**: October 2, 2025
**Branch**: `dev`
**Architecture**: Claude Agent SDK with @tool decorators (MCP-compatible)
**Status**: Planning Phase - Ready for Implementation

---

## Executive Summary

This plan integrates the production-ready Crawl4AI + AI cleaning infrastructure from z-playground1 into the multi-agent research system **following proper Claude Agent SDK patterns**. The integration will be implemented as **SDK-compatible @tool functions** that agents can invoke, maintaining the MCP architecture while upgrading from 30% to 70-100% content extraction success rates.

**Key Insight**: The current system already uses `@tool` decorators (e.g., `serp_search_tool.py`). We will follow this exact pattern to create new advanced scraping tools that replace the basic HTTP+regex extraction.

---

## Current Architecture Analysis

### How Claude Agent SDK Tools Work

**Pattern**: Tools are defined using `@tool` decorator from `claude_agent_sdk`:

```python
# Example: multi_agent_research_system/tools/serp_search_tool.py
from claude_agent_sdk import tool

@tool(
    "serp_search",
    "Description of tool functionality",
    {
        "query": str,
        "session_id": str,
        # ... parameter schema
    }
)
async def serp_search(args):
    """Tool implementation."""
    # Extract parameters
    query = args.get("query")

    # Execute functionality
    result = await serp_search_and_extract(...)

    # Return SDK-compatible response
    return {"content": [{"type": "text", "text": result}]}
```

**Agent Integration**: Agents declare tools in `config/agents.py`:

```python
AgentDefinition(
    description="Research Agent",
    prompt="...",
    tools=["mcp__research_tools__serp_search", "Read", "Write"],  # Tool names
    model="sonnet"
)
```

**Current Limitation**: The `serp_search` tool internally calls `serp_search_utils.py` which uses **basic HTTP+regex** extraction (30% success rate).

---

## Implementation Strategy - SDK Tool Approach

### Phase 1: Utility Layer Enhancement ⭐ FOUNDATION
**Risk**: Low | **Effort**: 2-3 hours | **Impact**: Critical

**Objective**: Add advanced scraping utilities that tools can call internally.

#### Step 1.1: Install Dependencies

Add to `multi_agent_research_system/requirements.txt`:
```txt
# Advanced Scraping Dependencies
crawl4ai==0.7.4
playwright>=1.55.0
pydantic-ai>=1.0.2
cssselect>=1.3.0
beautifulsoup4>=4.12.3
```

**Commands**:
```bash
cd /home/kjdragan/lrepos/claude-agent-sdk-python/multi_agent_research_system
pip install -r requirements.txt
playwright install chromium  # Install browser for automation
```

#### Step 1.2: Copy Advanced Utilities

**Copy files** from z-playground1 to multi_agent_research_system/utils/:

1. **`crawl4ai_utils.py`** (1,026 lines)
   - Source: `/home/kjdragan/lrepos/z-playground1/utils/crawl4ai_utils.py`
   - Contains: `SimpleCrawler`, `scrape_and_clean_single_url_direct()`, `crawl_multiple_urls_with_cleaning()`
   - Provides: Multi-stage extraction with 70-100% success rate

2. **`content_cleaning.py`** (487 lines)
   - Source: `/home/kjdragan/lrepos/z-playground1/utils/content_cleaning.py`
   - Contains: `clean_content_with_gpt5_nano()`, `assess_content_cleanliness()`, `clean_technical_content_with_gpt5_nano()`
   - Provides: AI-powered content cleaning with judge optimization

**Handle Logfire Dependency**:
- Both files have no-op fallback for logfire (lines 27-44 in crawl4ai_utils.py)
- No changes needed - fallback will activate automatically
- Optional: Add `logfire[fastapi]>=4.4.0` if you want Logfire tracing

#### Step 1.3: Verify Environment Variables

Ensure `.env` file contains:
```bash
# Required
ANTHROPIC_API_KEY=your_key_here
SERP_API_KEY=your_serper_key_here

# For AI content cleaning
OPENAI_API_KEY=your_openai_key_here

# Optional
ANTHROPIC_BASE_URL=https://api.anthropic.com
```

**Success Criteria**:
- ✅ All dependencies installed
- ✅ Playwright browsers configured
- ✅ Utility files copied and imports resolve
- ✅ API keys configured

---

### Phase 2: Create Advanced Scraping Tools ⭐ CORE IMPLEMENTATION
**Risk**: Low | **Effort**: 3-4 hours | **Impact**: Transformative

**Objective**: Create new SDK tools using `@tool` decorator that leverage advanced scraping utilities.

#### Step 2.1: Create Advanced URL Scraping Tool

**New file**: `multi_agent_research_system/tools/advanced_scraping_tool.py`

```python
"""Advanced web scraping tool using Crawl4AI and AI content cleaning.

This tool provides high-quality content extraction with:
- Browser automation for JavaScript-heavy sites
- Multi-stage extraction with fallback strategies
- AI-powered content cleaning (GPT-5-nano)
- Judge optimization for speed (saves 35-40s per URL)
- Technical content preservation
"""

import asyncio
from pathlib import Path
from typing import Any, Dict

from claude_agent_sdk import tool

try:
    from ..utils.crawl4ai_utils import (
        scrape_and_clean_single_url_direct,
        crawl_multiple_urls_with_cleaning
    )
    from ..utils.content_cleaning import clean_content_with_judge_optimization
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.crawl4ai_utils import (
        scrape_and_clean_single_url_direct,
        crawl_multiple_urls_with_cleaning
    )
    from utils.content_cleaning import clean_content_with_judge_optimization


@tool(
    "advanced_scrape_url",
    "Advanced web scraping with Crawl4AI browser automation, AI content cleaning, and technical content preservation. Handles JavaScript sites, applies judge optimization for speed, and achieves 70-100% success rates. Returns clean article content with navigation/ads removed.",
    {
        "url": str,
        "session_id": str,
        "search_query": str,
        "preserve_technical": bool
    }
)
async def advanced_scrape_url(args):
    """
    Advanced single URL scraping with multi-stage extraction and AI cleaning.

    Features:
    - Stage 1: Fast CSS selector extraction
    - Stage 2: Robust fallback extraction (universal compatibility)
    - Stage 3: Judge assessment and AI cleaning
    - Technical content preservation (code blocks, installation commands)
    - 30K-58K character content extraction (vs 2K limit in basic scraping)
    """
    url = args.get("url")
    session_id = args.get("session_id", "default")
    search_query = args.get("search_query", None)
    preserve_technical = args.get("preserve_technical", True)

    try:
        # Execute multi-stage scraping with AI cleaning
        result = await scrape_and_clean_single_url_direct(
            url=url,
            session_id=session_id,
            search_query=search_query,
            extraction_mode="article",
            include_metadata=True,
            preserve_technical_content=preserve_technical
        )

        if result['success']:
            # Format response with metadata
            response_text = f"""# Scraped Content from {url}

**Status**: ✅ Success
**Content Length**: {result['char_count']} characters ({result['word_count']} words)
**Processing Stage**: {result.get('stage', 'unknown')}
**Duration**: {result['duration']:.2f}s

---

## Cleaned Content

{result['cleaned_content']}

---

**Metadata**:
- Extraction Mode: {result['extraction_mode']}
- Technical Content Preserved: {preserve_technical}
"""

            # Add cleaning metadata if available
            if result.get('metadata') and result['metadata'].get('cleaning_optimization'):
                cleaning_meta = result['metadata']['cleaning_optimization']
                judge_score = cleaning_meta.get('judge_score', 'N/A')
                latency_saved = cleaning_meta.get('latency_saved', 'N/A')
                response_text += f"\n- Judge Score: {judge_score}"
                response_text += f"\n- Latency Saved: {latency_saved}"

            return {"content": [{"type": "text", "text": response_text}]}
        else:
            # Extraction failed
            error_msg = f"""# Scraping Failed for {url}

**Status**: ❌ Failed
**Error**: {result['error_message']}
**Duration**: {result['duration']:.2f}s

The multi-stage extraction system attempted to scrape this URL but was unsuccessful.
"""
            return {"content": [{"type": "text", "text": error_msg}], "is_error": True}

    except Exception as e:
        error_msg = f"Advanced scraping failed for {url}: {str(e)}"

        # Check for common issues
        if "playwright" in str(e).lower():
            error_msg += "\n\n⚠️ **Playwright not installed**\nRun: playwright install chromium"

        if "OPENAI_API_KEY" in str(e):
            error_msg += "\n\n⚠️ **OPENAI_API_KEY not found**\nAdd OPENAI_API_KEY to .env for AI content cleaning."

        return {"content": [{"type": "text", "text": error_msg}], "is_error": True}


@tool(
    "advanced_scrape_multiple_urls",
    "Advanced parallel scraping of multiple URLs with Crawl4AI + AI cleaning. Processes URLs concurrently, applies search query filtering to remove unrelated content, and returns cleaned article content. Achieves 70-100% success rates.",
    {
        "urls": list,
        "session_id": str,
        "search_query": str,
        "max_concurrent": int
    }
)
async def advanced_scrape_multiple_urls(args):
    """
    Advanced parallel URL scraping with immediate AI cleaning.

    Features:
    - Parallel crawl+clean processing (reduced latency)
    - Search query context filtering (removes unrelated articles)
    - Progressive anti-bot detection (4-level system)
    - Batch processing with configurable concurrency
    """
    urls = args.get("urls", [])
    session_id = args.get("session_id", "default")
    search_query = args.get("search_query", "")
    max_concurrent = args.get("max_concurrent", 5)

    if not urls:
        return {"content": [{"type": "text", "text": "❌ No URLs provided for scraping."}], "is_error": True}

    try:
        # Execute parallel crawl+clean
        results = await crawl_multiple_urls_with_cleaning(
            urls=urls,
            session_id=session_id,
            search_query=search_query,
            max_concurrent=max_concurrent,
            extraction_mode="article",
            include_metadata=True
        )

        # Format response
        successful_count = sum(1 for r in results if r['success'])
        total_content_length = sum(r.get('char_count', 0) for r in results if r['success'])

        response_text = f"""# Parallel URL Scraping Results

**Total URLs**: {len(urls)}
**Successful**: {successful_count}/{len(urls)} ({successful_count/len(urls)*100:.1f}%)
**Total Content**: {total_content_length} characters
**Search Query Context**: {search_query}

---

"""

        # Add individual results
        for i, result in enumerate(results, 1):
            url = result['url']
            if result['success']:
                response_text += f"""## ✅ {i}. Success: {url}

**Content Length**: {result['char_count']} characters ({result['word_count']} words)
**Duration**: {result['duration']:.2f}s

### Cleaned Content

{result['cleaned_content'][:1000]}{"..." if len(result.get('cleaned_content', '')) > 1000 else ""}

---

"""
            else:
                response_text += f"""## ❌ {i}. Failed: {url}

**Error**: {result['error_message']}
**Duration**: {result['duration']:.2f}s

---

"""

        response_text += f"""
## Processing Summary

- **Parallel Processing**: {max_concurrent} concurrent operations
- **Search Query Filtering**: Applied to remove unrelated content
- **Average Success Rate**: {successful_count/len(urls)*100:.1f}%
- **Average Content Length**: {total_content_length/successful_count if successful_count > 0 else 0:.0f} characters per URL
"""

        return {"content": [{"type": "text", "text": response_text}]}

    except Exception as e:
        error_msg = f"Parallel scraping failed: {str(e)}"
        return {"content": [{"type": "text", "text": error_msg}], "is_error": True}


# Export tools
__all__ = ['advanced_scrape_url', 'advanced_scrape_multiple_urls']
```

#### Step 2.2: Update Existing SERP Search Tool

**Modify**: `multi_agent_research_system/tools/serp_search_tool.py`

**Change**: Update imports and replace internal extraction with advanced scraping:

```python
# At top of file, add:
try:
    from ..utils.crawl4ai_utils import scrape_and_clean_single_url_direct
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.crawl4ai_utils import scrape_and_clean_single_url_direct

# Keep existing @tool decorator and serp_search function
# Only modify the call to serp_search_and_extract to use advanced extraction internally
```

**Modify**: `multi_agent_research_system/utils/serp_search_utils.py`

**In function** `simple_content_extraction()` (lines 268-311), replace with:

```python
async def advanced_content_extraction(url: str, session_id: str, search_query: str = None) -> str:
    """
    Advanced content extraction using Crawl4AI + AI cleaning.

    Replaces simple HTTP+regex with multi-stage browser automation.
    """
    try:
        result = await scrape_and_clean_single_url_direct(
            url=url,
            session_id=session_id,
            search_query=search_query,
            extraction_mode="article",
            include_metadata=False,
            preserve_technical_content=True
        )

        if result['success']:
            return result['cleaned_content']
        else:
            logger.warning(f"Advanced extraction failed for {url}: {result['error_message']}")
            return ""

    except Exception as e:
        logger.error(f"Advanced content extraction error for {url}: {e}")
        return ""
```

**Update call sites** in `serp_search_and_extract()` function (around line 497):

```python
# OLD:
tasks = [simple_content_extraction(url) for url in urls_to_extract]

# NEW:
tasks = [advanced_content_extraction(url, session_id, query) for url in urls_to_extract]
```

**Remove** the 2,000 character truncation limit (lines 300-302).

**Success Criteria**:
- ✅ New `advanced_scraping_tool.py` created with 2 SDK tools
- ✅ `serp_search_utils.py` updated to use advanced extraction
- ✅ All imports resolve correctly
- ✅ No syntax errors in tool definitions

---

### Phase 3: Agent Integration ⭐ ENABLE AGENTS
**Risk**: Low | **Effort**: 1-2 hours | **Impact**: High

**Objective**: Make new tools available to research agents.

#### Step 3.1: Register Tools with Agents

**Modify**: `multi_agent_research_system/config/agents.py`

**Update** `get_research_agent_definition()` to include new tools:

```python
def get_research_agent_definition() -> AgentDefinition:
    return AgentDefinition(
        description="Expert Research Agent...",
        prompt="""You are a Research Agent...

Available Tools:
- mcp__research_tools__serp_search: HIGH-PERFORMANCE Google search (now with advanced scraping)
- mcp__research_tools__advanced_scrape_url: Direct URL scraping with Crawl4AI + AI cleaning
- mcp__research_tools__advanced_scrape_multiple_urls: Parallel URL scraping
- mcp__research_tools__save_research_findings: Save research data
- Read/Write/Edit: File operations

ENHANCED SCRAPING CAPABILITIES:
- Use advanced_scrape_url for single URLs requiring deep content extraction
- Use advanced_scrape_multiple_urls for batch URL processing
- Automatic AI cleaning removes navigation, ads, and unrelated content
- Technical content (code blocks, commands) automatically preserved
- 70-100% success rates vs 30% with basic scraping
""",
        tools=[
            "mcp__research_tools__serp_search",
            "mcp__research_tools__advanced_scrape_url",  # NEW
            "mcp__research_tools__advanced_scrape_multiple_urls",  # NEW
            "mcp__research_tools__save_research_findings",
            "mcp__research_tools__capture_search_results",
            "Read", "Write", "Edit", "Bash"
        ],
        model="sonnet"
    )
```

#### Step 3.2: Import Tools in Orchestrator

**Modify**: `multi_agent_research_system/core/orchestrator.py`

**Add imports** (around lines 115-120):

```python
# Import advanced scraping tools
try:
    from ..tools.advanced_scraping_tool import advanced_scrape_url, advanced_scrape_multiple_urls
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from tools.advanced_scraping_tool import advanced_scrape_url, advanced_scrape_multiple_urls
```

**Register tools** in orchestrator initialization (if needed by SDK pattern).

**Success Criteria**:
- ✅ Tools registered in agent definitions
- ✅ Tools imported in orchestrator
- ✅ Agents can discover and invoke new tools
- ✅ No import errors when starting orchestrator

---

### Phase 4: Testing & Validation ⭐ QUALITY GATE
**Risk**: Low | **Effort**: 2-3 hours | **Impact**: Critical

**Objective**: Verify advanced scraping works end-to-end.

#### Step 4.1: Unit Test - Direct Tool Invocation

**Create**: `multi_agent_research_system/test_advanced_scraping_tools.py`

```python
"""Test advanced scraping tools directly."""

import asyncio
from pathlib import Path

async def test_single_url_scraping():
    """Test advanced_scrape_url tool."""
    from tools.advanced_scraping_tool import advanced_scrape_url

    # Test URL
    args = {
        "url": "https://docs.anthropic.com/en/docs/agents",
        "session_id": "test-scraping",
        "search_query": "Claude Agent SDK",
        "preserve_technical": True
    }

    result = await advanced_scrape_url(args)

    print("=" * 80)
    print("SINGLE URL SCRAPING TEST")
    print("=" * 80)
    print(result['content'][0]['text'])
    print("=" * 80)

    # Verify success
    assert "Success" in result['content'][0]['text'], "Scraping should succeed"
    assert len(result['content'][0]['text']) > 1000, "Should extract substantial content"

    print("\n✅ Single URL scraping test PASSED")


async def test_multiple_url_scraping():
    """Test advanced_scrape_multiple_urls tool."""
    from tools.advanced_scraping_tool import advanced_scrape_multiple_urls

    # Test URLs
    args = {
        "urls": [
            "https://docs.anthropic.com/en/docs/agents",
            "https://www.anthropic.com/news",
            "https://www.anthropic.com/research"
        ],
        "session_id": "test-parallel-scraping",
        "search_query": "Anthropic AI research",
        "max_concurrent": 3
    }

    result = await advanced_scrape_multiple_urls(args)

    print("=" * 80)
    print("PARALLEL URL SCRAPING TEST")
    print("=" * 80)
    print(result['content'][0]['text'])
    print("=" * 80)

    # Verify success
    assert "Successful:" in result['content'][0]['text'], "Should report success count"

    print("\n✅ Parallel URL scraping test PASSED")


async def test_serp_search_with_advanced_extraction():
    """Test SERP search with advanced extraction backend."""
    from utils.serp_search_utils import serp_search_and_extract

    kevin_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN"

    result = await serp_search_and_extract(
        query="Claude Agent SDK documentation",
        search_type="search",
        num_results=5,
        auto_crawl_top=3,
        crawl_threshold=0.3,
        session_id="test-serp-advanced",
        kevin_dir=kevin_dir
    )

    print("=" * 80)
    print("SERP SEARCH WITH ADVANCED EXTRACTION TEST")
    print("=" * 80)
    print(f"Result length: {len(result)} characters")
    print(f"First 1000 characters:\n{result[:1000]}")
    print("=" * 80)

    # Verify improvement
    assert len(result) > 10000, "Should extract much more content than old 2K limit"
    assert "EXTRACTED CONTENT" in result, "Should include extracted article content"

    print("\n✅ SERP search with advanced extraction test PASSED")


async def run_all_tests():
    """Run all scraping tests."""
    print("\n" + "=" * 80)
    print("ADVANCED SCRAPING INTEGRATION TESTS")
    print("=" * 80 + "\n")

    try:
        await test_single_url_scraping()
        print()

        await test_multiple_url_scraping()
        print()

        await test_serp_search_with_advanced_extraction()
        print()

        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
```

**Run tests**:
```bash
cd /home/kjdragan/lrepos/claude-agent-sdk-python/multi_agent_research_system
python test_advanced_scraping_tools.py
```

#### Step 4.2: Integration Test - Full Multi-Agent Workflow

**Create**: `multi_agent_research_system/test_full_workflow.py`

```python
"""Test full multi-agent research workflow with advanced scraping."""

import asyncio
from core.orchestrator import ResearchOrchestrator

async def test_research_with_advanced_scraping():
    """Test complete research workflow."""
    orchestrator = ResearchOrchestrator(debug_mode=True)

    # Test topic
    topic = "Claude Agent SDK multi-agent patterns"

    print(f"\n🚀 Starting research on: {topic}")
    print("=" * 80)

    # Execute research workflow
    result = await orchestrator.conduct_research(topic)

    print("=" * 80)
    print(f"✅ Research completed")
    print(f"Session ID: {result.get('session_id')}")
    print(f"Final report length: {len(result.get('final_report', ''))} characters")
    print("=" * 80)

    # Verify advanced scraping was used
    assert len(result.get('final_report', '')) > 5000, "Report should be comprehensive"

    print("\n✅ Full workflow test PASSED")


if __name__ == "__main__":
    asyncio.run(test_research_with_advanced_scraping())
```

#### Step 4.3: Comparison Analysis

**Create**: `KEVIN/PROJECT_DOCUMENTATION/scraping_before_after_comparison.md`

Document:
- Before: Success rates, content length, quality
- After: Measured improvements
- Sample URLs tested
- Performance metrics

**Success Criteria**:
- ✅ All unit tests pass
- ✅ Integration test completes successfully
- ✅ Success rates improved (30% → 70-90%)
- ✅ Content length increased (500-1,500 → 10,000-30,000 chars)
- ✅ Content quality verified (no navigation/ads)
- ✅ Work products generated correctly

---

### Phase 5: Optimization & Documentation ⭐ POLISH
**Risk**: Low | **Effort**: 2-3 hours | **Impact**: High

**Objective**: Enable performance optimizations and document system.

#### Step 5.1: Enable Judge Optimization (Optional but Recommended)

The judge optimization system automatically assesses content cleanliness and skips expensive GPT-5-nano cleaning if content is already clean (0.7+ score), saving 35-40 seconds per URL.

**Already integrated** in `scrape_and_clean_single_url_direct()` via:
```python
clean_content_with_judge_optimization(
    content=raw_content,
    url=url,
    search_query=search_query,
    cleanliness_threshold=0.7
)
```

**To monitor** savings, add logging in `advanced_scraping_tool.py`:

```python
if result.get('metadata', {}).get('cleaning_optimization'):
    cleaning_meta = result['metadata']['cleaning_optimization']
    logger.info(f"Judge optimization - Score: {cleaning_meta.get('judge_score')}, "
                f"Latency saved: {cleaning_meta.get('latency_saved', 'N/A')}")
```

#### Step 5.2: Create Usage Documentation

**Create**: `KEVIN/PROJECT_DOCUMENTATION/advanced_scraping_usage_guide.md`

```markdown
# Advanced Scraping Usage Guide

## Available Tools

### 1. advanced_scrape_url
Single URL scraping with Crawl4AI + AI cleaning.

**When to use**:
- Deep content extraction from single URLs
- JavaScript-heavy sites
- Technical documentation (preserves code blocks)

**Example**:
```
Use mcp__research_tools__advanced_scrape_url with:
- url: "https://docs.anthropic.com/en/docs/agents"
- session_id: "my-research-session"
- search_query: "Claude Agent SDK"
- preserve_technical: true
```

### 2. advanced_scrape_multiple_urls
Parallel URL scraping with concurrent processing.

**When to use**:
- Batch processing of multiple URLs
- Need for speed (parallel execution)
- Search query filtering across multiple sources

**Example**:
```
Use mcp__research_tools__advanced_scrape_multiple_urls with:
- urls: ["https://url1.com", "https://url2.com", "https://url3.com"]
- session_id: "batch-scraping"
- search_query: "AI research"
- max_concurrent: 5
```

### 3. serp_search (Enhanced)
SERP API search with advanced content extraction.

**What changed**:
- Backend now uses Crawl4AI instead of basic HTTP+regex
- 70-100% success rates (vs 30% before)
- 30K-58K character extraction (vs 2K limit before)
- AI cleaning removes navigation/ads

**Usage** (unchanged):
```
Use mcp__research_tools__serp_search with:
- query: "research topic"
- num_results: 15
- auto_crawl_top: 8
- crawl_threshold: 0.3
```

## Performance Characteristics

| Feature | Basic (Old) | Advanced (New) |
|---------|-------------|----------------|
| Success Rate | ~30% | 70-100% |
| Content Length | 500-1,500 chars | 10,000-30,000+ chars |
| JavaScript Sites | ❌ Fails | ✅ Works |
| Anti-Bot Detection | ❌ Blocked | ✅ Handled |
| Content Quality | Poor (clutter) | High (clean) |
| Processing Time | 3-5s | 8-12s (3-8s with judge opt) |

## Best Practices

1. **Use search_query context** - Helps AI cleaning filter unrelated content
2. **Enable preserve_technical** - For documentation/code-heavy content
3. **Set appropriate max_concurrent** - Balance speed vs server load (5-10 recommended)
4. **Check session work products** - Review KEVIN/work_products/ for saved results
```

#### Step 5.3: Update System Documentation

**Modify**: `multi_agent_research_system/SYSTEM_SUMMARY.md`

Add section:
```markdown
## Advanced Scraping System (v2.0)

The system now uses **production-grade Crawl4AI + AI cleaning** for content extraction:

- **Multi-stage extraction**: CSS selector → Robust fallback → Judge assessment
- **Browser automation**: Handles JavaScript-heavy sites with Playwright
- **AI content cleaning**: GPT-5-nano removes navigation, ads, unrelated content
- **Judge optimization**: Saves 35-40s per URL by assessing cleanliness first
- **Technical preservation**: Code blocks, installation commands protected from corruption
- **Progressive anti-bot**: 4-level system adapts to site protection

**Success Rates**: 70-100% (vs 30% with basic HTTP+regex)
**Content Length**: 10K-30K+ characters (vs 2K limit before)
**Quality**: Clean article content with search query relevance filtering
```

**Success Criteria**:
- ✅ Judge optimization enabled and monitored
- ✅ Usage documentation created
- ✅ System summary updated
- ✅ README reflects new capabilities

---

## File Structure After Implementation

```
multi_agent_research_system/
├── tools/
│   ├── serp_search_tool.py              # ✅ Existing (backend upgraded)
│   └── advanced_scraping_tool.py        # ⭐ NEW - Advanced scraping tools
│
├── utils/
│   ├── serp_search_utils.py             # ✅ Modified (uses advanced extraction)
│   ├── crawl4ai_utils.py                # ⭐ NEW - Crawl4AI infrastructure
│   └── content_cleaning.py              # ⭐ NEW - AI cleaning + judge opt
│
├── config/
│   └── agents.py                        # ✅ Modified (tools registered)
│
├── core/
│   └── orchestrator.py                  # ✅ Modified (imports new tools)
│
├── tests/
│   ├── test_advanced_scraping_tools.py  # ⭐ NEW - Unit tests
│   └── test_full_workflow.py            # ⭐ NEW - Integration test
│
├── requirements.txt                      # ✅ Modified (new dependencies)
└── .env                                  # ✅ Verify (API keys present)
```

---

## Rollback Strategy

**If issues occur during implementation:**

### Option 1: Branch Rollback
```bash
# Restore from backup
git checkout dev-backup
git checkout -b dev-restored
```

### Option 2: Selective Revert
```bash
# Revert specific commits
git log --oneline  # Find commit hashes
git revert <commit-hash>
```

### Option 3: Keep Fallback Function
In `serp_search_utils.py`, keep old function as backup:
```python
async def simple_content_extraction_fallback(url: str, timeout: int = 30) -> str:
    """Original HTTP+regex extraction (backup)."""
    # Original implementation
    ...

async def advanced_content_extraction(url: str, session_id: str, search_query: str = None) -> str:
    """New Crawl4AI extraction."""
    try:
        result = await scrape_and_clean_single_url_direct(...)
        ...
    except Exception as e:
        logger.warning(f"Advanced extraction failed, using fallback: {e}")
        return await simple_content_extraction_fallback(url)
```

---

## Expected Results

### Before (Current System)
```
Test URL: https://docs.anthropic.com/en/docs/agents
- ❌ Success: ~30% (often fails)
- ❌ Content: 500-1,500 chars (2K max)
- ❌ Quality: Navigation menus, "Skip to content", ads, social widgets
- ❌ JavaScript: Fails to render dynamic content
- ❌ Bot detection: Gets blocked frequently
```

### After (Z-Playground1 Integration)
```
Test URL: https://docs.anthropic.com/en/docs/agents
- ✅ Success: 90-100% (multi-stage fallback)
- ✅ Content: 15,000-30,000 chars (full article)
- ✅ Quality: Clean article only, navigation/ads removed
- ✅ JavaScript: Renders with Playwright browser
- ✅ Bot detection: Handled by 4-level anti-bot system
- ✅ Technical: Code blocks and commands preserved
- ✅ Speed: 8-12s total (or 3-8s with judge optimization)
```

### Metrics to Track

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Success Rate | 70-90% | Test 20 random URLs, count successes |
| Content Length | 10K-30K chars | Average extracted content per URL |
| Content Quality | No navigation/ads | Manual review of 10 samples |
| Processing Time | 8-12s per URL | Log duration from test runs |
| Judge Optimization Hit Rate | 30-40% | Log "cleaning skipped" events |
| JavaScript Site Success | 80%+ | Test 10 JS-heavy sites |

---

## Timeline Estimate

| Phase | Duration | Can Pause? | Priority |
|-------|----------|------------|----------|
| Phase 1: Foundation | 2-3 hours | ✅ Yes | Critical |
| Phase 2: Core Implementation | 3-4 hours | ❌ No* | Critical |
| Phase 3: Agent Integration | 1-2 hours | ✅ Yes | Critical |
| Phase 4: Testing | 2-3 hours | ✅ Yes | Critical |
| Phase 5: Optimization | 2-3 hours | ✅ Yes | Optional |

**Total**: 10-15 hours over 2-3 days

*Phase 2 should be completed in one session once started to avoid partial integration state.

**Recommended Session Breaks**:
1. After Phase 1: Dependencies and utilities ready
2. After Phase 2: Tools created and utilities updated
3. After Phase 3: Agents integrated
4. After Phase 4: Testing complete
5. After Phase 5: System optimized and documented

---

## Success Criteria Checklist

### Phase 1 ✅
- [ ] crawl4ai, playwright, pydantic-ai installed
- [ ] Playwright browsers installed (`playwright install`)
- [ ] crawl4ai_utils.py and content_cleaning.py copied
- [ ] All imports resolve without errors
- [ ] OPENAI_API_KEY and SERP_API_KEY in .env

### Phase 2 ✅
- [ ] advanced_scraping_tool.py created with 2 @tool functions
- [ ] serp_search_utils.py updated with advanced_content_extraction()
- [ ] 2,000 character limit removed
- [ ] No syntax errors, all imports resolve

### Phase 3 ✅
- [ ] Tools registered in config/agents.py
- [ ] Tools imported in orchestrator.py
- [ ] Research agent can discover new tools
- [ ] No import errors when starting orchestrator

### Phase 4 ✅
- [ ] Unit tests pass (test_advanced_scraping_tools.py)
- [ ] Integration test passes (test_full_workflow.py)
- [ ] Success rate 70%+ on 20 test URLs
- [ ] Average content length 10K+ characters
- [ ] Content quality verified (no navigation/ads)
- [ ] Work products generated correctly

### Phase 5 ✅
- [ ] Judge optimization enabled and monitored
- [ ] Usage guide created
- [ ] System documentation updated
- [ ] Performance metrics collected

---

## Next Steps

1. **Approve this plan** - Review and confirm approach
2. **Set up todo tracking** - Use TodoWrite to track progress
3. **Begin Phase 1** - Install dependencies and copy utilities
4. **Execute phases sequentially** - Follow plan step-by-step
5. **Document deviations** - Note any changes or issues
6. **Create comparison report** - Document before/after improvements

---

## Questions & Considerations

### Q: Why not use MCP servers for Crawl4AI?
**A**: MCP servers add complexity and latency. The `@tool` decorator pattern is simpler, keeps utilities co-located with code, and provides better performance for compute-heavy operations like browser automation.

### Q: What if Playwright installation fails?
**A**: Fallback to basic extraction temporarily. Playwright requires system dependencies. On Linux: `sudo apt-get install -y libgbm-dev`. On Mac: Usually works out of box.

### Q: Can we run without OpenAI API key?
**A**: Yes, but AI cleaning won't work. Content will be extracted with Crawl4AI but not cleaned. Judge optimization will be skipped. Success rate will be ~70% instead of 90%+.

### Q: How does this interact with existing SERP search tool?
**A**: `serp_search` tool keeps same interface. We only change its internal implementation. Agents use it the same way. Backward compatible.

### Q: What about Logfire tracing?
**A**: Optional. Both utils have no-op fallback. If you want tracing, add `logfire[fastapi]>=4.4.0` to requirements.

---

## Architectural Decision Record

**Decision**: Integrate z-playground1 scraping as Claude Agent SDK `@tool` functions

**Context**: Current basic HTTP+regex extraction has 30% success rate and 2K content limit

**Alternatives Considered**:
1. ❌ MCP server for Crawl4AI - Too complex, adds latency
2. ❌ Replace entire orchestrator - Too risky, loses existing work
3. ✅ SDK tool pattern - Clean, maintainable, backward compatible

**Rationale**:
- Follows existing patterns (`serp_search_tool.py`)
- Minimal changes to agents and orchestrator
- Utilities remain co-located with code
- Easy to test and rollback
- Maintains MCP compatibility

**Consequences**:
- Positive: 70-100% success rates, 10K-30K char content, clean output
- Positive: Backward compatible, agents work unchanged
- Positive: Easy to test and validate
- Negative: Additional dependencies (crawl4ai, playwright)
- Negative: Slightly longer processing time (8-12s vs 3-5s)

**Status**: Approved for implementation

---

**End of Plan Document**
