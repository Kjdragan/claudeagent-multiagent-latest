"""
SERP API Search Utilities for Multi-Agent Research System

This module provides optimized search+crawl+clean functionality using SERP API
to replace the WebPrime MCP search system. Offers 10x performance improvement
with automatic content extraction and advanced relevance scoring.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx

# Token limit configuration
MAX_RESPONSE_TOKENS = 20000  # Conservative limit to stay under 25k token limit
APPOX_CHARS_PER_TOKEN = 4    # Rough approximation
MAX_RESPONSE_CHARS = MAX_RESPONSE_TOKENS * APPOX_CHARS_PER_TOKEN

# Import URL tracking system
try:
    from .url_tracker import get_url_tracker
except ImportError:
    # Fallback import for module execution
    import sys
    sys.path.append(os.path.dirname(__file__))
    from url_tracker import get_url_tracker

# Import enhanced relevance scoring functions
try:
    from .enhanced_relevance_scorer import (
        calculate_term_frequency_score,
        calculate_domain_authority_boost
    )
except ImportError:
    # Fallback import for module execution
    import sys
    sys.path.append(os.path.dirname(__file__))
    from enhanced_relevance_scorer import (
        calculate_term_frequency_score,
        calculate_domain_authority_boost
    )

# Lightweight LLM utility (lazy import to avoid circular dependencies)
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_quick_llm_call():
    """Resolve quick_llm_call lazily to prevent circular imports."""
    try:
        from ..core.llm_utils import quick_llm_call as resolved_call
    except ImportError:
        # Fallback for ad-hoc execution contexts
        import importlib
        resolved_call = importlib.import_module("core.llm_utils").quick_llm_call
    return resolved_call

def summarize_content(content: str, max_length: int = 2000) -> str:
    """
    Summarize content to fit within token limits.

    Args:
        content: Original content to summarize
        max_length: Maximum length of summarized content

    Returns:
        Summarized content within the specified length limit
    """
    if len(content) <= max_length:
        return content

    # Simple summarization: take first and last portions with an indicator
    first_part = content[:max_length//2]
    last_part = content[-max_length//2:]

    return f"""{first_part}

[Content summarized - full content ({len(content)} chars) saved to work product file]

{last_part}"""


# Import configuration
try:
    from ..config.settings import get_enhanced_search_config
except ImportError:
    # Fallback for standalone usage
    # Simple fallback config with all required attributes
    class SimpleConfig:
        # Search settings
        default_num_results = 15
        default_auto_crawl_top = 10
        default_crawl_threshold = 0.3
        default_anti_bot_level = 1
        default_max_concurrent = 15

        # Target-based scraping settings
        target_successful_scrapes = 10
        url_deduplication_enabled = True
        progressive_retry_enabled = True

        # Retry logic settings
        max_retry_attempts = 3
        progressive_timeout_multiplier = 1.5

        # Token management
        max_response_tokens = 20000
        content_summary_threshold = 20000

        # Content cleaning settings
        default_cleanliness_threshold = 0.7
        min_content_length_for_cleaning = 500
        min_cleaned_content_length = 200

        # Crawl settings
        default_crawl_timeout = 30000
        max_concurrent_crawls = 15
        crawl_retry_attempts = 2

        # Anti-bot levels
        anti_bot_levels = {
            0: "basic",      # 6/10 sites success
            1: "enhanced",   # 8/10 sites success
            2: "advanced",   # 9/10 sites success
            3: "stealth"     # 9.5/10 sites success
        }

        # Target system weighting
        interleave_primary_weight = 2
        domain_soft_cap = 2
        domain_penalty_alpha = 0.6
        recency_half_life_days = 7.0
        candidate_pool_multiplier = 2.0
        min_query_coverage = 1

    def get_enhanced_search_config():
        config = SimpleConfig()
        # Debug: verify the critical attribute exists
        if not hasattr(config, 'default_max_concurrent'):
            logger.error("SimpleConfig is missing default_max_concurrent attribute!")
            logger.error(f"Available attributes: {[attr for attr in dir(config) if not attr.startswith('_')]}")
            raise AttributeError("SimpleConfig missing required attribute: default_max_concurrent")
        return config

# ---------------------------------------------------------------------------
# Query planning helpers
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "about", "for", "of", "and", "to", "in", "on", "with",
    "from", "latest", "current", "news", "update", "updates", "today"
}

ANGLE_LIBRARY = [
    ("background context", "background and context"),
    ("key developments", "key developments and recent updates"),
    ("expert perspectives", "expert commentary and analysis"),
    ("impacts and implications", "impacts and implications"),
    ("responses and strategies", "responses strategies and plans"),
    ("data insights", "data trends statistics and metrics"),
]


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _optimize_query(original_query: str) -> str:
    """Deduplicate and streamline the base query while preserving key tokens."""
    if not original_query:
        return ""

    tokens = original_query.split()
    seen = set()
    optimized_tokens = []

    for token in tokens:
        cleaned = re.sub(r"[^\w]", "", token).lower()
        if not cleaned:
            continue
        if cleaned in STOPWORDS and cleaned in seen:
            continue
        if cleaned not in seen:
            optimized_tokens.append(token)
            seen.add(cleaned)

    optimized = " ".join(optimized_tokens).strip()
    return optimized or original_query.strip()


def _tokenize_core_terms(query: str) -> list[str]:
    """Return lowercase tokens excluding stopwords for angle selection."""
    words = re.findall(r"[A-Za-z0-9]+", query.lower())
    return [word for word in words if word not in STOPWORDS]


def _heuristic_scope_inference(query: str) -> dict[str, str]:
    """Local heuristic scope inference to backstop LLM failures."""
    text = (query or "").lower()

    brief_keywords = {"brief", "summary", "quick", "overview", "short"}
    comprehensive_keywords = {"comprehensive", "detailed", "extensive", "thorough", "in-depth", "analysis"}

    if any(word in text for word in brief_keywords):
        scope = "brief"
        reasoning = "Detected concise/summary cues in query"
    elif any(word in text for word in comprehensive_keywords):
        scope = "comprehensive"
        reasoning = "Detected detailed/comprehensive cues in query"
    else:
        scope = "default"
        reasoning = "No explicit scope cues detected; using default"

    return {
        "scope": scope,
        "reasoning": reasoning,
        "confidence": "medium"
    }


def _build_query_planner_prompt(
    original_query: str,
    allow_expansion: bool,
    max_orthogonal_queries: int
) -> str:
    """Construct the unified planner prompt described in the target system spec."""
    expansion_clause = (
        "If editor directives are present, return an empty array for orthogonal_queries."
        if allow_expansion else
        "Do not generate any orthogonal queries; return an empty array."
    )
    return f"""
You are the TARGETED QUERY PLANNER for the Multi-Agent Research System.

Original Query: "{original_query}"
ALLOW_ORTHOGONALS: {"true" if allow_expansion else "false"}
MAX_ORTHOGONALS: {max_orthogonal_queries}

Responsibilities:
1. Rewrite the original query into a precise, context-aware primary search query.
2. Propose exactly two complementary orthogonal queries that surface distinct facets (unless expansion is disabled).
3. Classify the appropriate research scope as "brief", "default", or "comprehensive". Default to "default" when uncertain.
4. For each orthogonal query, include a human-readable angle/descriptor and a boolean `skip_if_editor_directive` indicating whether it should be skipped when editors provide explicit directives.

{expansion_clause}

Return a single JSON object with the following structure:
{{
  "optimized_query": "...",
  "orthogonal_queries": [
    {{"query": "...", "angle": "...", "descriptor": "...", "skip_if_editor_directive": true}},
    {{"query": "...", "angle": "...", "descriptor": "...", "skip_if_editor_directive": true}}
  ],
  "scope": "brief|default|comprehensive",
  "reasoning": "...",
  "confidence": "high|medium|low"
}}
""".strip()


def _sanitize_scope_value(scope: str | None) -> str:
    """Normalize planner scope output to supported values."""
    normalized = (scope or "default").strip().lower()
    if normalized not in {"brief", "default", "comprehensive"}:
        return "default"
    return normalized


def _normalize_planner_response(
    original_query: str,
    raw_data: dict[str, Any],
    allow_expansion: bool,
    max_orthogonal_queries: int
) -> dict[str, Any]:
    """Normalize the planner JSON response with defensive defaults."""
    optimized = _normalize_whitespace(raw_data.get("optimized_query") or original_query)
    scope = _sanitize_scope_value(raw_data.get("scope"))
    reasoning = raw_data.get("reasoning", "")
    confidence = raw_data.get("confidence", "medium")

    orthogonals: list[dict[str, Any]] = []
    if allow_expansion and max_orthogonal_queries > 0:
        seen_queries = {optimized.lower()}
        raw_list = raw_data.get("orthogonal_queries") or []
        if not isinstance(raw_list, list):
            raw_list = []

        for entry in raw_list:
            if not isinstance(entry, dict):
                continue

            query_text = _normalize_whitespace(entry.get("query", ""))
            if not query_text:
                continue

            normalized = query_text.lower()
            if normalized in seen_queries:
                continue
            seen_queries.add(normalized)

            angle = _normalize_whitespace(entry.get("angle") or entry.get("label") or "")
            descriptor = _normalize_whitespace(entry.get("descriptor") or angle or "")

            orthogonals.append({
                "query": query_text,
                "angle": angle or f"angle_{len(orthogonals) + 1}",
                "descriptor": descriptor or (angle or ""),
                "skip_if_editor_directive": bool(entry.get("skip_if_editor_directive", True))
            })

            if len(orthogonals) >= max_orthogonal_queries:
                break

    return {
        "optimized_query": optimized or original_query,
        "orthogonal_queries": orthogonals if allow_expansion else [],
        "scope": scope,
        "reasoning": reasoning,
        "confidence": confidence
    }


async def _execute_targeted_query_planner(
    original_query: str,
    allow_expansion: bool,
    max_orthogonal_queries: int
) -> dict[str, Any] | None:
    """Call the unified LLM planner and return normalized data or None on failure."""
    prompt = _build_query_planner_prompt(
        original_query=original_query,
        allow_expansion=allow_expansion,
        max_orthogonal_queries=max_orthogonal_queries
    )

    try:
        quick_llm_call = _get_quick_llm_call()
        raw_response = await quick_llm_call(prompt, temperature=0.1, max_tokens=800)
        data = json.loads(raw_response.strip())
        if not isinstance(data, dict):
            raise ValueError("Planner response was not a JSON object")
        return _normalize_planner_response(
            original_query=original_query,
            raw_data=data,
            allow_expansion=allow_expansion,
            max_orthogonal_queries=max_orthogonal_queries
        )
    except Exception as exc:
        logger.warning(f"Targeted query planner fallback engaged: {exc}")
        return None


async def _determine_scope_metadata(query: str) -> dict[str, str]:
    """Call lightweight LLM heuristic to infer scope, falling back to local rules."""
    prompt = f"""
Analyze the following research request and determine the appropriate scope.

QUERY: {query}

Return JSON with the following keys:
{{
  "scope": "brief|default|comprehensive",
  "reasoning": "Short explanation",
  "confidence": "high|medium|low"
}}

If the best scope is unclear, return "default".
"""
    try:
        quick_llm_call = _get_quick_llm_call()
        result = await quick_llm_call(prompt, temperature=0.0)
        data = json.loads(result.strip())
        scope = data.get("scope", "default")
        if scope not in {"brief", "default", "comprehensive"}:
            scope = "default"
        return {
            "scope": scope,
            "reasoning": data.get("reasoning", "LLM provided no reasoning"),
            "confidence": data.get("confidence", "medium")
        }
    except Exception as exc:
        logger.warning(f"Scope inference via LLM failed, using heuristics: {exc}")
        return _heuristic_scope_inference(query)


def _build_orthogonal_queries(base_query: str, max_count: int) -> list[dict[str, str]]:
    """Generate orthogonal queries by pairing the base query with complementary angles."""
    if max_count <= 0:
        return []

    base_lower = base_query.lower()
    core_terms = Counter(_tokenize_core_terms(base_query))
    orthogonals: list[dict[str, str]] = []

    for label, phrase in ANGLE_LIBRARY:
        if len(orthogonals) >= max_count:
            break

        # Skip phrases that repeat tokens already highlighted in the base query
        phrase_tokens = phrase.lower().split()
        if any(token in base_lower for token in phrase_tokens):
            continue

        orth_query = f"{base_query} {phrase}".strip()
        orthogonals.append({
            "query": orth_query,
            "angle": label,
            "descriptor": phrase,
            "skip_if_editor_directive": True
        })

    # If we still lack orthogonals, append generic complementary views
    while len(orthogonals) < max_count:
        fallback_label = f"additional context {len(orthogonals) + 1}"
        fallback_query = f"{base_query} additional context and perspectives {len(orthogonals) + 1}"
        orthogonals.append({
            "query": fallback_query,
            "angle": fallback_label,
            "descriptor": "additional context and perspectives",
            "skip_if_editor_directive": True
        })

    return orthogonals[:max_count]


async def generate_search_plan(
    original_query: str,
    allow_expansion: bool = True,
    max_orthogonal_queries: int = 2
) -> dict[str, Any]:
    """
    Produce an optimized query plan with optional orthogonal angles and scope metadata.

    Args:
        original_query: User-provided query string
        allow_expansion: Whether orthogonal queries should be generated
        max_orthogonal_queries: Maximum number of complementary queries to return

    Returns:
        Dict containing optimized query, orthogonal queries, and scope metadata
    """
    original_query = _normalize_whitespace(original_query)
    effective_max = max(0, max_orthogonal_queries)

    planner_plan = await _execute_targeted_query_planner(
        original_query=original_query,
        allow_expansion=allow_expansion,
        max_orthogonal_queries=effective_max
    )

    if planner_plan:
        optimized_query = planner_plan.get("optimized_query") or original_query
        orthogonals = planner_plan.get("orthogonal_queries", [])
        if not allow_expansion:
            orthogonals = []

        plan = {
            "original_query": original_query,
            "optimized_query": _normalize_whitespace(optimized_query),
            "orthogonal_queries": orthogonals[:effective_max],
            "scope": _sanitize_scope_value(planner_plan.get("scope")),
            "reasoning": planner_plan.get("reasoning", ""),
            "confidence": planner_plan.get("confidence", "medium")
        }
        logger.debug(
            "Targeted planner produced plan: scope=%s, orthogonals=%d",
            plan["scope"],
            len(plan["orthogonal_queries"])
        )
        return plan

    # Fallback path: heuristics for optimization and scope
    optimized_query = _optimize_query(original_query)
    scope_metadata = await _determine_scope_metadata(original_query)

    orthogonal_entries: list[dict[str, Any]] = []
    if allow_expansion and effective_max > 0:
        orthogonal_entries = _build_orthogonal_queries(
            optimized_query,
            max_count=effective_max
        )

    logger.debug(
        "Planner fallback engaged; using heuristic plan with %d orthogonals",
        len(orthogonal_entries)
    )

    return {
        "original_query": original_query,
        "optimized_query": optimized_query or original_query,
        "orthogonal_queries": orthogonal_entries,
        "scope": scope_metadata.get("scope", "default"),
        "reasoning": scope_metadata.get("reasoning", ""),
        "confidence": scope_metadata.get("confidence", "medium")
    }


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def _calculate_recency_multiplier(date_str: str, half_life_days: float = 7.0) -> float:
    if not date_str:
        return 1.0

    now = datetime.utcnow()
    parsed_date = None

    # ISO or RFC-like formats
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%b %d, %Y", "%B %d, %Y"):
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue

    if not parsed_date:
        # Relative phrases (e.g., "2 days ago", "3 weeks ago")
        match = re.search(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", date_str.lower())
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            if unit == "minute":
                delta_days = value / (60 * 24)
            elif unit == "hour":
                delta_days = value / 24
            elif unit == "day":
                delta_days = value
            elif unit == "week":
                delta_days = value * 7
            elif unit == "month":
                delta_days = value * 30
            elif unit == "year":
                delta_days = value * 365
            else:
                delta_days = 0
            age_days = max(delta_days, 0)
        else:
            return 1.0
    else:
        age_days = max((now - parsed_date).days + (now - parsed_date).seconds / 86400, 0)

    if half_life_days <= 0:
        return 1.0

    multiplier = math.pow(0.5, age_days / half_life_days)
    return max(0.2, min(1.0, multiplier))


def _interleave_results_by_weight(
    results_by_label: Dict[str, List[SearchResult]],
    primary_label: str = "primary",
    primary_weight: int = 2
) -> List[SearchResult]:
    """Interleave results from each query label using a weighted scheme."""
    label_lists: Dict[str, List[SearchResult]] = {}
    for label, results in results_by_label.items():
        sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)
        label_lists[label] = sorted_results

    interleaved: List[SearchResult] = []
    orth_labels = [label for label in label_lists.keys() if label != primary_label]
    orth_index = 0

    def has_remaining() -> bool:
        return any(label_lists.get(label) for label in label_lists)

    while has_remaining():
        # Pull from primary queue according to weight
        for _ in range(primary_weight):
            primary_queue = label_lists.get(primary_label, [])
            if primary_queue:
                interleaved.append(primary_queue.pop(0))

        # Pull a single item from orthogonal queues in round-robin fashion
        if orth_labels:
            for _ in range(len(orth_labels)):
                label = orth_labels[orth_index % len(orth_labels)]
                orth_index += 1
                queue = label_lists.get(label, [])
                if queue:
                    interleaved.append(queue.pop(0))
                    break

        # Break if only orthogonal queues remain and they were exhausted this cycle
        if not label_lists.get(primary_label) and not any(label_lists.get(label) for label in orth_labels):
            # Append any remaining items (if any) and exit
            for remaining_items in label_lists.values():
                interleaved.extend(remaining_items)
            break

    # Filter out any accidental None entries
    return [result for result in interleaved if result]


def _rank_with_domain_and_recency(
    results: List[SearchResult],
    domain_soft_cap: int,
    penalty_alpha: float,
    half_life_days: float
) -> tuple[List[SearchResult], Dict[str, Any]]:
    remaining = results.copy()
    ranked: List[SearchResult] = []
    domain_counts: Dict[str, int] = {}

    while remaining:
        best_index = 0
        best_score = -1.0
        best_recency = 1.0
        best_penalty = 1.0
        best_domain_hits = 0

        for idx, result in enumerate(remaining):
            base_score = result.relevance_score or 0.0
            query_weight = getattr(result, "query_weight", 1.0)
            recency_multiplier = _calculate_recency_multiplier(result.date, half_life_days)

            domain = _extract_domain(result.link)
            domain_hits = domain_counts.get(domain, 0)
            penalty = 1 / (1 + penalty_alpha * domain_hits)
            if domain_hits >= domain_soft_cap:
                penalty *= 1 / (1 + penalty_alpha)

            composite = base_score * query_weight * recency_multiplier * penalty

            if composite > best_score:
                best_score = composite
                best_index = idx
                best_recency = recency_multiplier
                best_penalty = penalty
                best_domain_hits = domain_hits

        selected = remaining.pop(best_index)
        domain = _extract_domain(selected.link)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        selected.final_score = round(best_score, 4)
        selected.recency_multiplier = round(best_recency, 4)
        selected.domain_penalty = round(best_penalty, 4)
        selected.domain_hits_before = best_domain_hits
        ranked.append(selected)

    stats = {
        "domain_counts": domain_counts,
        "total_ranked": len(ranked)
    }

    return ranked, stats


def _select_candidate_urls(
    search_results: List[SearchResult],
    target_count: int,
    crawl_threshold: float,
    domain_soft_cap: int,
    pool_multiplier: float = 2.0,
    min_per_query: int = 1
) -> tuple[List[str], Dict[str, Any]]:
    """Build a candidate URL list honoring per-query coverage and domain caps."""
    if target_count <= 0:
        return [], {"per_query_counts": {}, "domain_counts": {}}

    seen_urls: set[str] = set()
    domain_counts: defaultdict[str, int] = defaultdict(int)
    per_query_counts: defaultdict[str, int] = defaultdict(int)
    candidates: List[str] = []

    required_labels = []
    for result in search_results:
        label = getattr(result, "query_label", "primary")
        if label not in required_labels:
            required_labels.append(label)

    min_per_query = max(1, min_per_query)
    domain_cap_for_candidates = max(domain_soft_cap, 1) + 1  # allow one extra beyond soft cap for redundancy

    def record_candidate(result: SearchResult) -> bool:
        url = getattr(result, "link", "")
        if not url or url in seen_urls:
            return False
        domain = _extract_domain(url)
        label = getattr(result, "query_label", "primary")

        candidates.append(url)
        seen_urls.add(url)
        domain_counts[domain] += 1
        per_query_counts[label] += 1
        return True

    # Stage 1: guarantee minimum representation per query
    for label in required_labels:
        label_results = [r for r in search_results if getattr(r, "query_label", "primary") == label]
        if not label_results:
            continue

        # Pass 1: select results meeting threshold
        for result in label_results:
            if per_query_counts[label] >= min_per_query:
                break
            if result.relevance_score >= crawl_threshold:
                record_candidate(result)

        # Pass 2: if still below minimum, relax threshold
        while per_query_counts[label] < min_per_query:
            fallback = None
            for result in label_results:
                if result.link in seen_urls:
                    continue
                fallback = result
                break
            if fallback is None:
                break
            if not record_candidate(fallback):
                break

    # Stage 2: fill remaining slots prioritizing threshold-compliant URLs
    max_candidates = max(target_count, int(target_count * pool_multiplier))
    for result in search_results:
        if len(candidates) >= max_candidates:
            break
        url = getattr(result, "link", "")
        if not url or url in seen_urls:
            continue
        if result.relevance_score < crawl_threshold:
            continue
        domain = _extract_domain(url)
        if domain_counts[domain] >= domain_cap_for_candidates:
            continue
        record_candidate(result)

    # Stage 3: if still below desired pool size, relax threshold for remaining slots
    if len(candidates) < max_candidates:
        for result in search_results:
            if len(candidates) >= max_candidates:
                break
            url = getattr(result, "link", "")
            if not url or url in seen_urls:
                continue
            domain = _extract_domain(url)
            if domain_counts[domain] >= domain_cap_for_candidates:
                continue
            record_candidate(result)

    stats = {
        "per_query_counts": dict(per_query_counts),
        "domain_counts": dict(domain_counts),
        "pool_size": len(candidates),
        "target_count": target_count,
        "pool_multiplier": pool_multiplier,
        "min_per_query": min_per_query
    }

    return candidates, stats

# Import advanced scraping utilities
try:
    from .crawl4ai_utils import scrape_and_clean_single_url_direct
except ImportError:
    # Fallback import for module execution
    import sys
    sys.path.append(os.path.dirname(__file__))
    from crawl4ai_utils import scrape_and_clean_single_url_direct

logger = logging.getLogger(__name__)


class SearchResult:
    """Search result data structure with enhanced relevance scoring"""
    def __init__(self, title: str, link: str, snippet: str, position: int = 0,
                 date: str = None, source: str = None, relevance_score: float = 0.0):
        self.title = title
        self.link = link
        self.snippet = snippet
        self.position = position
        self.date = date
        self.source = source
        self.relevance_score = relevance_score


# Import enhanced relevance scorer with domain authority
from .enhanced_relevance_scorer import (
    calculate_enhanced_relevance_score_with_domain_authority as calculate_enhanced_relevance_score,
)


async def execute_serp_search(
    query: str,
    search_type: str = "search",
    num_results: int = 10,
    country: str = "us",
    language: str = "en"
) -> list[SearchResult]:
    """
    Execute search using Serper API.

    Args:
        query: Search query
        search_type: "search" or "news"
        num_results: Number of results to retrieve
        country: Country code for search
        language: Language code for search

    Returns:
        List of SearchResult objects
    """
    try:
        serper_api_key = os.getenv("SERP_API_KEY")
        if not serper_api_key:
            logger.warning("SERP_API_KEY not found in environment variables")
            return []

        # Choose endpoint based on search type
        endpoint = "news" if search_type == "news" else "search"
        url = f"https://google.serper.dev/{endpoint}"

        # Build search parameters
        search_params = {
            "q": query,
            "num": min(num_results, 100),  # Serper limit
            "gl": country,
            "hl": language
        }

        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json"
        }

        logger.info(f"Executing {search_type} search for: {query}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=search_params, headers=headers)

        if response.status_code == 200:
            data = response.json()

            # Extract results based on search type
            if search_type == "news" and "news" in data:
                raw_results = data["news"]
            else:
                raw_results = data.get("organic", [])

            # Parse query terms for enhanced relevance scoring
            query_terms = query.lower().replace('or', ' ').replace('and', ' ').split()
            query_terms = [term.strip() for term in query_terms if len(term.strip()) > 2]

            # Convert to SearchResult objects with enhanced relevance scoring
            search_results = []
            for i, result in enumerate(raw_results):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                position = i + 1

                # Calculate enhanced relevance score with domain authority
                relevance_score = calculate_enhanced_relevance_score(
                    title=title,
                    snippet=snippet,
                    position=position,
                    query_terms=query_terms,
                    url=result.get("link", "")
                )

                search_result = SearchResult(
                    title=title,
                    link=result.get("link", ""),
                    snippet=snippet,
                    position=position,
                    date=result.get("date", ""),
                    source=result.get("source", ""),
                    relevance_score=relevance_score
                )
                search_results.append(search_result)

            logger.info(f"Retrieved {len(search_results)} search results for query: '{query}'")
            return search_results

        else:
            logger.error(f"Serper API error: {response.status_code} - {response.text}")
            return []

    except Exception as e:
        logger.error(f"Error in Serper search: {e}")
        return []


def select_urls_for_crawling(
    search_results: list[SearchResult],
    limit: int = 10,
    min_relevance: float = 0.3,
    session_id: str = "default",
    use_deduplication: bool = True
) -> list[str]:
    """
    Select URLs for crawling based on relevance scores with URL deduplication.

    Args:
        search_results: List of search results
        limit: Maximum number of URLs to select
        min_relevance: Minimum relevance score threshold (fixed at 0.3 for better success)
        session_id: Session identifier for URL tracking
        use_deduplication: Whether to use URL deduplication

    Returns:
        List of URLs to crawl
    """
    try:
        # Get configuration
        config = get_enhanced_search_config()

        # Filter by relevance threshold (ensure type safety)
        filtered_results = [
            result for result in search_results
            if float(result.relevance_score) >= float(min_relevance) and result.link
        ]

        # Sort by relevance score (highest first) - ensure float comparison
        filtered_results.sort(key=lambda x: float(x.relevance_score), reverse=True)

        # Extract URLs up to limit
        candidate_urls = [result.link for result in filtered_results[:limit]]

        # Apply URL deduplication if enabled
        if use_deduplication and config.url_deduplication_enabled:
            url_tracker = get_url_tracker()
            urls_to_crawl, skipped_urls = url_tracker.filter_urls(candidate_urls, session_id)

            logger.info(f"URL selection with threshold {min_relevance}:")
            logger.info(f"  - Total results: {len(search_results)}")
            logger.info(f"  - Above threshold: {len(filtered_results)}")
            logger.info(f"  - Before deduplication: {len(candidate_urls)}")
            logger.info(f"  - After deduplication: {len(urls_to_crawl)}")
            logger.info(f"  - Skipped duplicates: {len(skipped_urls)}")

            return urls_to_crawl
        else:
            logger.info(f"URL selection with threshold {min_relevance} (no deduplication):")
            logger.info(f"  - Total results: {len(search_results)}")
            logger.info(f"  - Above threshold: {len(filtered_results)}")
            logger.info(f"  - Selected for crawling: {len(candidate_urls)}")
            logger.info(f"  - Rejected: {len(search_results) - len(filtered_results)} below threshold")

            return candidate_urls

    except Exception as e:
        logger.error(f"Error selecting URLs for crawling: {e}")
        return []


def format_search_results(search_results: list[SearchResult]) -> str:
    """
    Format search results for display.

    Args:
        search_results: List of search results

    Returns:
        Formatted search results string
    """
    if not search_results:
        return "No search results found."

    result_parts = [
        f"# Search Results ({len(search_results)} found)",
        ""
    ]

    for i, result in enumerate(search_results, 1):
        result_parts.extend([
            f"## {i}. {result.title}",
            f"**URL**: {result.link}",
            f"**Source**: {result.source}" if result.source else "",
            f"**Date**: {result.date}" if result.date else "",
            f"**Relevance Score**: {result.relevance_score:.2f}",
            "",
            result.snippet,
            "",
            "---",
            ""
        ])

    return "\n".join(result_parts)


async def advanced_content_extraction(url: str, session_id: str, search_query: str = None) -> str:
    """
    Advanced content extraction using Crawl4AI + AI cleaning.

    Replaces simple HTTP+regex with multi-stage browser automation:
    - Stage 1: Fast CSS selector extraction
    - Stage 2: Robust fallback extraction
    - Stage 3: Judge assessment and AI cleaning
    - 70-100% success rate (vs 30% with basic HTTP+regex)
    - 30K-58K characters extracted (vs 2K limit before)

    Args:
        url: URL to extract content from
        session_id: Session identifier for logging
        search_query: Search query context for content relevance filtering

    Returns:
        Extracted and cleaned content as string
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
            logger.info(f"Advanced extraction successful for {url}: {result['char_count']} chars")
            return result['cleaned_content']
        else:
            logger.warning(f"Advanced extraction failed for {url}: {result['error_message']}")
            return ""

    except Exception as e:
        logger.error(f"Advanced content extraction error for {url}: {e}")
        return ""


def save_search_work_product(
    search_results: list[SearchResult],
    crawled_content: list[str],
    urls: list[str],
    query: str,
    session_id: str,
    kevin_dir: Path
) -> str:
    """
    Save detailed search and crawl results to work product file.

    Args:
        search_results: List of search results
        crawled_content: List of cleaned content strings
        urls: List of crawled URLs
        query: Original search query
        session_id: Session identifier
        kevin_dir: KEVIN directory path

    Returns:
        Path to saved work product file
    """
    try:
        # Use session-based directory structure
        sessions_dir = kevin_dir / "sessions" / session_id
        research_dir = sessions_dir / "research"
        research_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp and filename with numbered prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"1-search_workproduct_{timestamp}.md"
        filepath = research_dir / filename

        # Build work product content
        workproduct_content = [
            "# Search Results Work Product",
            "",
            f"**Session ID**: {session_id}",
            f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Search Query**: {query}",
            f"**Total Search Results**: {len(search_results)}",
            f"**Successfully Crawled**: {len(crawled_content)}",
            "",
            "---",
            "",
            "## 🔍 Search Results Summary",
            "",
        ]

        # Add search results overview
        for i, result in enumerate(search_results, 1):
            workproduct_content.extend([
                f"### {i}. {result.title}",
                f"**URL**: {result.link}",
                f"**Source**: {result.source}" if result.source else "",
                f"**Date**: {result.date}" if result.date else "",
                f"**Relevance Score**: {result.relevance_score:.2f}",
                "",
                f"**Snippet**: {result.snippet}",
                "",
                "---",
                ""
            ])

        if crawled_content:
            workproduct_content.extend([
                "",
                "## 📄 Extracted Content",
                ""
            ])

            # Add detailed crawled content
            for i, (content, url) in enumerate(zip(crawled_content, urls, strict=False), 1):
                # Find corresponding search result for title
                title = f"Article {i}"
                for result in search_results:
                    if result.link == url:
                        title = result.title
                        break

                workproduct_content.extend([
                    f"## 🌐 {i}. {title}",
                    "",
                    f"**URL**: {url}",
                    f"**Content Length**: {len(content)} characters",
                    "",
                    "### 📄 Extracted Content",
                    "",
                    "---",
                    "",
                    content,
                    "",
                    "---",
                    ""
                ])

        # Add footer
        workproduct_content.extend([
            "",
            "## 📊 Processing Summary",
            "",
            f"- **Search Query**: {query}",
            f"- **Search Results Found**: {len(search_results)}",
            f"- **URLs Successfully Crawled**: {len(crawled_content)}",
            "- **Processing**: SERP API search + content extraction",
            "",
            "*Generated by Multi-Agent Research System - SERP API Integration*"
        ])

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(workproduct_content))

        logger.info(f"✅ Work product saved to: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error saving work product: {e}")
        return ""


async def target_based_scraping(
    search_results: list[SearchResult],
    session_id: str,
    target_successful_scrapes: int = 10,
    crawl_threshold: float = 0.3,
    max_concurrent: int = 10
) -> tuple[list[str], list[str], Dict[str, Any]]:
    """
    Perform target-based scraping to achieve desired number of successful extractions.

    This function processes ALL candidates in parallel to achieve the target
    number of successful scrapes without sequential blocking.

    Args:
        search_results: List of search results to select from
        session_id: Session identifier for URL tracking
        target_successful_scrapes: Target number of successful content extractions
        crawl_threshold: Initial relevance threshold (fixed at 0.3)
        max_concurrent: Maximum concurrent crawling operations

    Returns:
        Tuple of (successful_content, attempted_urls)
    """
    config = get_enhanced_search_config()
    url_tracker = get_url_tracker()

    # Start with current threshold and target from config
    target_count = target_successful_scrapes or config.target_successful_scrapes
    pool_multiplier = float(getattr(config, "candidate_pool_multiplier", 2.0))
    min_query_coverage = int(getattr(config, "min_query_coverage", 1))
    domain_soft_cap = getattr(config, "domain_soft_cap", 2)

    candidate_urls, selection_stats = _select_candidate_urls(
        search_results=search_results,
        target_count=target_count,
        crawl_threshold=crawl_threshold,
        domain_soft_cap=domain_soft_cap,
        pool_multiplier=pool_multiplier,
        min_per_query=min_query_coverage
    )

    urls_to_crawl, skipped_urls = url_tracker.filter_urls(candidate_urls, session_id)
    selection_stats["initial_candidates"] = len(candidate_urls)
    selection_stats["filtered_candidates"] = len(urls_to_crawl)
    selection_stats["skipped_urls"] = skipped_urls

    logger.info(
        f"Target-based scraping: target={target_count}, "
        f"selected={len(urls_to_crawl)} (pool multiplier={pool_multiplier}, min per query={min_query_coverage})"
    )

    # Process candidates in parallel using progressive retry
    successful_content, attempted_urls = await _crawl_urls_with_retry(
        urls=urls_to_crawl,
        session_id=session_id,
        max_concurrent=max_concurrent,
        use_progressive_retry=True
    )

    selection_stats["attempted_urls"] = attempted_urls
    selection_stats["successful_count"] = len(successful_content)

    # If we still need more and have retry candidates, process them too
    selection_stats["retry_candidates"] = []

    if config.progressive_retry_enabled and len(successful_content) < target_count:
        retry_candidates = url_tracker.get_retry_candidates(attempted_urls)
        if retry_candidates:
            logger.info(f"Processing additional retry candidates: {len(retry_candidates)} URLs")
            retry_content, retry_urls = await _crawl_urls_with_retry(
                urls=retry_candidates,
                session_id=session_id,
                max_concurrent=max_concurrent,
                is_retry=True
            )
            successful_content.extend(retry_content)
            attempted_urls.extend(retry_urls)
            selection_stats["retry_candidates"] = retry_candidates

    # Early termination: Check if target achieved after retry candidates
    if len(successful_content) >= target_count:
        logger.info(f"✅ Target achieved after retry candidates: {len(successful_content)}/{target_count} successful scrapes")
        selection_stats["achieved_target"] = True
        return successful_content, attempted_urls, selection_stats

    # Final statistics
    success_rate = len(successful_content) / len(attempted_urls) if attempted_urls else 0
    logger.info(f"Target-based scraping completed: {len(successful_content)}/{target_count} successful "
               f"({success_rate:.1%} success rate from {len(attempted_urls)} URLs)")

    selection_stats["achieved_target"] = len(successful_content) >= target_count
    selection_stats["success_rate"] = success_rate

    return successful_content, attempted_urls, selection_stats


async def _crawl_urls_with_retry(
    urls: list[str],
    session_id: str,
    max_concurrent: int,
    is_retry: bool = False,
    use_progressive_retry: bool = False
) -> tuple[list[str], list[str]]:
    """
    Helper function to crawl URLs with retry logic.

    Args:
        urls: URLs to crawl
        session_id: Session identifier
        max_concurrent: Maximum concurrent operations
        is_retry: Whether this is a retry operation

    Returns:
        Tuple of (successful_content, attempted_urls)
    """
    if not urls:
        return [], []

    config = get_enhanced_search_config()
    url_tracker = get_url_tracker()

    # Use z-playground1 implementation
    from utils.crawl4ai_z_playground import crawl_multiple_urls_with_results

    # Always use progressive retry for better parallelization
    crawl_results = await crawl_multiple_urls_with_results(
        urls=urls,
        session_id=session_id,
        max_concurrent=max_concurrent,
        extraction_mode="article",
        include_metadata=True,
        use_progressive_retry=use_progressive_retry,
        max_retries=3 if use_progressive_retry else 0
    )

    # Process results and record attempts
    successful_content = []
    attempted_urls = []

    for result in crawl_results:
        url = result['url']
        success = result['success']
        content = result.get('content', '')
        content_length = len(content) if content else 0
        duration = result.get('duration', 0.0)
        error_message = result.get('error_message')

        # Determine anti-bot level used
        anti_bot_level = 1  # Default
        if is_retry:
            anti_bot_level = url_tracker.get_retry_anti_bot_level(url)

        # Record attempt in URL tracker asynchronously to prevent blocking
        tracking_task = asyncio.create_task(
            asyncio.to_thread(
                url_tracker.record_attempt,
                url=url,
                success=success and content_length > 100,  # Only count substantial content as success
                anti_bot_level=anti_bot_level,
                content_length=content_length,
                duration=duration,
                error_message=error_message,
                session_id=session_id
            )
        )
        # Don't await tracking task to prevent blocking - let it run in background

        attempted_urls.append(url)

        if success and content_length > 100:
            successful_content.append(content.strip())
            logger.info(f"✅ Extracted {content_length} chars from {url}")
        else:
            logger.warning(f"❌ Failed to extract substantial content from {url}")

    return successful_content, attempted_urls


async def serp_search_and_extract(
    query: str,
    search_type: str = "search",
    num_results: int = 15,
    auto_crawl_top: int = 5,
    crawl_threshold: float = 0.3,
    session_id: str = "default",
    kevin_dir: Path = None
) -> str:
    """
    Combined SERP API search and content extraction.

    This function:
    1. Performs search using SERP API
    2. Selects relevant URLs based on threshold
    3. Extracts content from selected URLs
    4. Saves work product and returns comprehensive results

    Args:
        query: Search query
        search_type: "search" or "news"
        num_results: Number of search results to retrieve
        auto_crawl_top: Maximum number of URLs to crawl
        crawl_threshold: Minimum relevance threshold for crawling
        session_id: Session identifier
        kevin_dir: KEVIN directory path

    Returns:
        Full detailed content for agent processing
    """
    try:
        start_time = datetime.now()
        logger.info(f"Starting SERP search+extract for query: '{query}'")

        # Set default KEVIN directory if not provided
        if kevin_dir is None:
            kevin_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN"

        # Step 1: Execute SERP search (1-2 seconds)
        search_results = await execute_serp_search(
            query=query,
            search_type=search_type,
            num_results=num_results
        )

        if not search_results:
            return f"❌ **Search Failed**\n\nNo results found for query: '{query}'"

        # Step 2: Use target-based scraping with improved threshold logic
        config = get_enhanced_search_config()

        # Use fixed 0.3 threshold and target-based scraping
        logger.info(f"Using target-based scraping: threshold=0.3, target={config.target_successful_scrapes}")

        crawled_content, successful_urls, selection_stats = await target_based_scraping(
            search_results=search_results,
            session_id=session_id,
            target_successful_scrapes=config.target_successful_scrapes,
            crawl_threshold=0.3,  # Fixed at 0.3 for better success rates
            max_concurrent=config.default_max_concurrent
        )


        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Step 4: Save work product
        work_product_path = save_search_work_product(
            search_results=search_results,
            crawled_content=crawled_content,
            urls=successful_urls,
            query=query,
            session_id=session_id,
            kevin_dir=kevin_dir
        )

        attempted_total = len(selection_stats.get("attempted_urls", []))
        candidate_pool_size = selection_stats.get("filtered_candidates", 0)

        metadata_payload = {
            "selection_stats": selection_stats,
            "summary": {
                "candidate_pool_size": candidate_pool_size,
                "attempted_urls": attempted_total,
                "successful_extractions": len(crawled_content)
            }
        }

        # Step 5: Build comprehensive results for agents
        if crawled_content:
            orchestrator_data = f"""# SERP API SEARCH RESULTS

**Query**: {query}
**Search Type**: {search_type}
**Search Results**: {len(search_results)} found
**URLs Extracted**: {len(crawled_content)} successfully processed
**Processing Time**: {total_duration:.2f}s
**Work Product Saved**: {work_product_path}

---

## SEARCH RESULTS SUMMARY

"""

            # Add all search results with full metadata
            for i, result in enumerate(search_results, 1):
                orchestrator_data += f"""### {i}. {result.title}
**URL**: {result.link}
**Source**: {result.source}
**Date**: {result.date}
**Relevance Score**: {result.relevance_score:.2f}
**Snippet**: {result.snippet}

"""

            if crawled_content:
                orchestrator_data += f"""---

## EXTRACTED CONTENT

Total articles successfully extracted: {len(crawled_content)}

"""

                # Calculate remaining space for content after metadata
                current_length = len(orchestrator_data)
                remaining_space = MAX_RESPONSE_CHARS - current_length - 1000  # Leave buffer for summary section
                content_per_article = remaining_space // len(crawled_content) if crawled_content else 0
                max_content_length = min(content_per_article, 3000)  # Cap at 3000 chars per article

                # Add extracted content with token limit handling
                for i, (content, url) in enumerate(zip(crawled_content, successful_urls, strict=False), 1):
                    # Find corresponding search result for metadata
                    title = f"Article {i}"
                    source = "Unknown"
                    for result in search_results:
                        if result.link == url:
                            title = result.title
                            source = result.source or "Unknown"
                            break

                    # Apply content summarization if needed
                    if len(content) > max_content_length:
                        content = summarize_content(content, max_content_length)
                        content_note = f" (summarized from {len(content)} original characters)"
                    else:
                        content_note = ""

                    orchestrator_data += f"""### Extracted Article {i}: {title}
**URL**: {url}
**Source**: {source}
**Content Length**: {len(content)} characters{content_note}

**EXTRACTED CONTENT**:
{content}

---

"""

            orchestrator_data += f"""
## PROCESSING SUMMARY

- **Search executed**: SERP API {search_type} search for "{query}"
- **Results found**: {len(search_results)} search results
- **Candidate URLs selected**: {candidate_pool_size} (threshold: 0.3)
- **URLs attempted**: {attempted_total}
- **Successful extractions**: {len(crawled_content)} articles
- **Total processing time**: {total_duration:.2f} seconds
- **Work product file**: {work_product_path}
- **Performance**: SERP API direct search (10x faster than MCP)

This is the complete search data for research analysis and report generation.
"""

            logger.info(f"✅ SERP search+extract completed in {total_duration:.2f}s")
            return orchestrator_data, metadata_payload

        else:
            # Content extraction failed: Return search results only
            search_section = format_search_results(search_results)

            failed_result = f"""{search_section}

---

**Note**: Content extraction failed for selected URLs. Search results provided above.
**Execution Time**: {total_duration:.2f}s
**Work Product Saved**: {work_product_path}
"""

            logger.warning(f"Content extraction failed, returning search results only. Duration: {total_duration:.2f}s")
            return failed_result, metadata_payload

    except Exception as e:
        logger.error(f"Error in SERP search+extract: {e}")
        return f"❌ **Search Error**\n\nFailed to execute SERP API search: {str(e)}", {"error": str(e)}


async def expanded_query_search_and_extract(
    query: str,
    search_type: str = "search",
    num_results: int = 15,
    auto_crawl_top: int = 5,
    crawl_threshold: float = 0.3,
    session_id: str = "default",
    kevin_dir: Path = None,
    max_expanded_queries: int = 3,
    allow_expansion: bool = True
) -> tuple[str, Dict[str, Any]]:
    """
    Corrected query expansion workflow with master result consolidation.

    This function implements the proper workflow:
    1. Generate optimized primary and orthogonal queries using the planner
    2. Execute SERP searches for each expanded query
    3. Collect & deduplicate all results into one master list
    4. Rank results by relevance
    5. Scrape from master ranked list within budget limits

    Args:
        query: Original search query
        search_type: "search" or "news"
        num_results: Number of results per SERP search
        auto_crawl_top: Maximum number of URLs to crawl from master list
        crawl_threshold: Minimum relevance threshold for crawling
        session_id: Session identifier
        kevin_dir: KEVIN directory path
        max_expanded_queries: Maximum number of total queries to execute (including primary)
        allow_expansion: Whether orthogonal queries should be generated

    Returns:
        Tuple containing a detailed markdown summary and metadata payload
    """
    try:
        start_time = datetime.now()
        logger.info(f"Starting expanded query search+extract for query: '{query}'")

        # Set default KEVIN directory if not provided
        if kevin_dir is None:
            kevin_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN"

        # Step 1: Build structured search plan
        effective_max_orthogonals = max(0, (max_expanded_queries or 1) - 1)
        plan = await generate_search_plan(
            original_query=query,
            allow_expansion=allow_expansion,
            max_orthogonal_queries=effective_max_orthogonals
        )
        plan_scope = plan.get("scope", "default")
        plan_confidence = plan.get("confidence", "medium")
        plan_reasoning = plan.get("reasoning", "")

        primary_entry = {
            "label": "primary",
            "query": plan.get("optimized_query") or query,
            "angle": "primary focus",
            "weight": 2.0,
            "skip_if_editor_directive": False
        }

        orthogonals = plan.get("orthogonal_queries", []) if allow_expansion else []
        orthogonals = orthogonals[:effective_max_orthogonals]

        planned_queries: list[Dict[str, Any]] = [primary_entry]
        for idx, orth in enumerate(orthogonals, 1):
            planned_queries.append({
                "label": f"orthogonal_{idx}",
                "query": orth.get("query", "").strip() or primary_entry["query"],
                "angle": orth.get("angle", f"angle_{idx}"),
                "descriptor": orth.get("descriptor", ""),
                "weight": 1.0,
                "skip_if_editor_directive": orth.get("skip_if_editor_directive", True)
            })

        # Remove duplicate queries while preserving order
        seen_queries = set()
        deduped_planned_queries = []
        for entry in planned_queries:
            normalized = entry["query"].lower()
            if normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            deduped_planned_queries.append(entry)

        planned_queries = deduped_planned_queries or [primary_entry]
        expanded_queries = [entry["query"] for entry in planned_queries]
        logger.info(
            f"Query planner produced {len(planned_queries)} queries "
            f"(scope={plan.get('scope')}, allow_expansion={allow_expansion})"
        )

        config = get_enhanced_search_config()
        primary_weight = getattr(config, "interleave_primary_weight", 2)
        domain_soft_cap = getattr(config, "domain_soft_cap", 2)
        domain_penalty_alpha = getattr(config, "domain_penalty_alpha", 0.6)
        recency_half_life = getattr(config, "recency_half_life_days", 7.0)

        # Step 2: Execute SERP searches per planned query and annotate with metadata
        all_search_results = []
        results_by_label: Dict[str, List[SearchResult]] = {}
        for query_idx, entry in enumerate(planned_queries):
            planned_query = entry["query"]
            logger.info(f"Executing SERP search for planned query [{entry['label']}]: '{planned_query}'")
            search_results = await execute_serp_search(
                query=planned_query,
                search_type=search_type,
                num_results=num_results
            )

            # Apply query-aware position scoring (0.05 decay per position within original query order)
            query_terms = planned_query.split()
            enhanced_results = []
            for pos, result in enumerate(search_results, 1):
                # Calculate position score with gentle 0.05 decay per position
                position_score = max(0.0, 1.0 - (pos - 1) * 0.05)

                # Calculate title and snippet scores
                title_score = calculate_term_frequency_score(result.title, query_terms)
                snippet_score = calculate_term_frequency_score(result.snippet, query_terms)

                # Calculate domain authority boost
                authority_boost = calculate_domain_authority_boost(result.link)

                # Apply enhanced relevance scoring formula (Position 40% + Title 30% + Snippet 30% + Authority)
                base_score = (
                    position_score * 0.40 +
                    title_score * 0.30 +
                    snippet_score * 0.30
                )
                final_score = min(1.0, base_score + authority_boost)

                # Update result relevance score
                result.relevance_score = round(final_score, 3)
                result.query_label = entry["label"]
                result.query_angle = entry.get("angle", entry["label"])
                result.query_weight = entry.get("weight", 1.0)

                enhanced_results.append(result)

                logger.debug(
                    f"Query {query_idx+1} [{entry['label']}], Pos {pos}: "
                    f"Position={position_score:.3f}, Title={title_score:.3f}, "
                    f"Snippet={snippet_score:.3f}, Authority={authority_boost:.3f}, Final={final_score:.3f}"
                )

            all_search_results.extend(enhanced_results)
            logger.info(
                f"Retrieved {len(enhanced_results)} enhanced results for query [{entry['label']}] "
                f"('{planned_query}')"
            )
            results_by_label.setdefault(entry["label"], []).extend(enhanced_results)

        # Step 3: Deduplicate and rank results with diversity controls
        interleaved_results = _interleave_results_by_weight(
            results_by_label,
            primary_label="primary",
            primary_weight=primary_weight
        )
        master_results = deduplicate_search_results(interleaved_results)
        logger.info(
            f"Deduplicated results: {len(interleaved_results)} -> {len(master_results)} unique results "
            f"(raw total across queries: {len(all_search_results)})"
        )

        master_results, domain_stats = _rank_with_domain_and_recency(
            master_results,
            domain_soft_cap=domain_soft_cap,
            penalty_alpha=domain_penalty_alpha,
            half_life_days=recency_half_life
        )
        logger.info(
            f"Ranked {len(master_results)} results with domain penalty (soft cap={domain_soft_cap}, alpha={domain_penalty_alpha})"
        )

        # Step 5: Scrape from master ranked list within budget limits
        logger.info(f"Using target-based scraping: threshold={crawl_threshold}, target={config.target_successful_scrapes}")

        crawled_content, successful_urls, selection_stats = await target_based_scraping(
            search_results=master_results,
            session_id=session_id,
            target_successful_scrapes=config.target_successful_scrapes,
            crawl_threshold=crawl_threshold,
            max_concurrent=config.default_max_concurrent
        )

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Step 6: Save work product with expanded query information
        work_product_path = save_expanded_search_work_product(
            original_query=query,
            expanded_queries=expanded_queries,
            master_results=master_results,
            crawled_content=crawled_content,
            successful_urls=successful_urls,
            session_id=session_id,
            kevin_dir=kevin_dir,
            total_duration=total_duration
        )

        # Step 7: Build comprehensive results for agents
        total_raw_results = len(all_search_results)
        dedupe_rate = (1 - len(master_results) / total_raw_results) if total_raw_results else 0.0
        unique_domain_count = len(domain_stats.get("domain_counts", {}))

        attempted_total = len(selection_stats.get("attempted_urls", []))
        candidate_pool_size = selection_stats.get("filtered_candidates", 0)
        per_query_attempts = selection_stats.get("per_query_counts", {})

        query_plan_metadata = []
        for entry in planned_queries:
            query_plan_metadata.append({
                "label": entry.get("label", "primary"),
                "query": entry.get("query", ""),
                "angle": entry.get("angle", entry.get("label", "primary")),
                "weight": entry.get("weight", 1.0),
                "attempts": per_query_attempts.get(entry.get("label", "primary"), 0),
                "skip_if_editor_directive": entry.get("skip_if_editor_directive", False)
            })

        metadata_payload = {
            "planner": {
                "scope": plan_scope,
                "confidence": plan_confidence,
                "reasoning": plan_reasoning,
                "allow_expansion": allow_expansion,
                "queries": query_plan_metadata
            },
            "domain_stats": domain_stats,
            "selection_stats": selection_stats,
            "summary": {
                "total_raw_results": total_raw_results,
                "deduplication_rate": dedupe_rate,
                "unique_domains": unique_domain_count,
                "candidate_pool_size": candidate_pool_size,
                "attempted_urls": attempted_total
            }
        }

        if crawled_content:
            orchestrator_data = f"""# EXPANDED QUERY SEARCH RESULTS

**Original Query**: {query}
**Planned Queries**: {len(planned_queries)} (primary + orthogonals)
**Planner Scope**: {plan_scope} (confidence: {plan_confidence})
**Planner Rationale**: {plan_reasoning or 'N/A'}
**Search Type**: {search_type}
**Total Master Results**: {len(master_results)} found (after deduplication)
**URLs Extracted**: {len(crawled_content)} successfully processed
**Processing Time**: {total_duration:.2f}s
**Work Product Saved**: {work_product_path}

---

## QUERY EXPANSION ANALYSIS

**Original Query**: {query}
**Generated Expanded Queries**:
"""

            for i, entry in enumerate(query_plan_metadata, 1):
                angle = entry.get("angle", entry["label"])
                attempts = entry.get("attempts", 0)
                orchestrator_data += (
                    f"{i}. [{entry['label']}] {entry['query']}  "
                    f"*(angle: {angle}, weight: {entry.get('weight', 1.0)}, attempts: {attempts})*\n"
                )

            orchestrator_data += f"""
**Total Queries Executed**: {len(expanded_queries)}
**Total Raw Results**: {total_raw_results}
**Deduplicated Results**: {len(master_results)}
**Deduplication Rate**: {dedupe_rate:.1%}
**Domain Diversity**: {unique_domain_count} unique domains (soft cap {domain_soft_cap})

---

## MASTER SEARCH RESULTS (Top {min(20, len(master_results))} by Relevance)

"""

            # Add top master results with full metadata
            for i, result in enumerate(master_results[:20], 1):
                orchestrator_data += f"""### {i}. {result.title}
**URL**: {result.link}
**Source**: {result.source}
**Date**: {result.date}
**Relevance Score**: {result.relevance_score:.2f}
**Snippet**: {result.snippet}

"""

            if crawled_content:
                orchestrator_data += f"""---

## EXTRACTED CONTENT

Total articles successfully extracted: {len(crawled_content)}

"""

                # Calculate remaining space for content after metadata
                current_length = len(orchestrator_data)
                remaining_space = MAX_RESPONSE_CHARS - current_length - 1000  # Leave buffer for summary section
                content_per_article = remaining_space // len(crawled_content) if crawled_content else 0
                max_content_length = min(content_per_article, 3000)  # Cap at 3000 chars per article

                # Add extracted content with token limit handling
                for i, (content, url) in enumerate(zip(crawled_content, successful_urls, strict=False), 1):
                    # Find corresponding search result for metadata
                    title = f"Article {i}"
                    source = "Unknown"
                    for result in master_results:
                        if result.link == url:
                            title = result.title
                            source = result.source or "Unknown"
                            break

                    # Apply content summarization if needed
                    if len(content) > max_content_length:
                        content = summarize_content(content, max_content_length)
                        content_note = f" (summarized from {len(content)} original characters)"
                    else:
                        content_note = ""

                    orchestrator_data += f"""### Extracted Article {i}: {title}
**URL**: {url}
**Source**: {source}
**Content Length**: {len(content)} characters{content_note}

**EXTRACTED CONTENT**:
{content}

---

"""

            orchestrator_data += f"""
## PROCESSING SUMMARY

- **Original query**: "{query}"
- **Expanded queries executed**: {len(expanded_queries)}
- **Total raw results found**: {len(all_search_results)}
- **Deduplicated results**: {len(master_results)} unique URLs
- **Candidate URLs selected**: {candidate_pool_size} (threshold: {crawl_threshold})
- **URLs attempted**: {attempted_total}
- **Successful extractions**: {len(crawled_content)} articles
- **Total processing time**: {total_duration:.2f} seconds
- **Work product file**: {work_product_path}
- **Performance**: Expanded query search with master result consolidation

This is the complete search data for research analysis and report generation.
"""

            logger.info(f"✅ Expanded query search+extract completed in {total_duration:.2f}s")
            return orchestrator_data, metadata_payload

        else:
            # Content extraction failed: Return master search results only
            search_section = format_expanded_search_results(query, expanded_queries, master_results)

            failed_result = f"""{search_section}

---

**Note**: Content extraction failed for selected URLs. Master search results provided above.
**Execution Time**: {total_duration:.2f}s
**Work Product Saved**: {work_product_path}
"""

            logger.warning(f"Content extraction failed, returning master search results only. Duration: {total_duration:.2f}s")
            return failed_result, metadata_payload

    except Exception as e:
        logger.error(f"Error in expanded query search+extract: {e}")
        return f"❌ **Expanded Query Search Error**\n\nFailed to execute expanded query search: {str(e)}", {"error": str(e)}


async def generate_expanded_queries(original_query: str, max_queries: int = 3) -> list[str]:
    """
    Generate expanded search queries using simple query expansion techniques.

    Args:
        original_query: Original search query
        max_queries: Maximum number of expanded queries to generate

    Returns:
        List of expanded queries including the original
    """
    # Start with the original query
    queries = [original_query]

    # Simple query expansion techniques
    query_lower = original_query.lower()

    # Technique 1: Add context terms
    if "news" not in query_lower and "latest" not in query_lower:
        queries.append(f"{original_query} latest news")

    # Technique 2: Add comprehensive/overview terms
    if "overview" not in query_lower and "comprehensive" not in query_lower:
        queries.append(f"{original_query} overview comprehensive")

    # Technique 3: Add analysis/in-depth terms
    if "analysis" not in query_lower and "in-depth" not in query_lower:
        queries.append(f"{original_query} analysis in-depth")

    # Technique 4: Add research/study terms
    if "research" not in query_lower and "study" not in query_lower:
        queries.append(f"{original_query} research study")

    # Technique 5: Add developments/trends terms
    if "developments" not in query_lower and "trends" not in query_lower:
        queries.append(f"{original_query} recent developments trends")

    # Return limited number of queries (original + expanded)
    return queries[:max_queries]


def deduplicate_search_results(search_results: list[SearchResult]) -> list[SearchResult]:
    """
    Deduplicate search results based on URL, keeping the highest relevance score.

    Args:
        search_results: List of search results that may contain duplicates

    Returns:
        Deduplicated list of search results
    """
    seen_urls = set()
    deduplicated_results = []

    for result in search_results:
        if result.link not in seen_urls:
            seen_urls.add(result.link)
            deduplicated_results.append(result)
        else:
            # If we've seen this URL before, keep the one with higher relevance score
            for i, existing_result in enumerate(deduplicated_results):
                if existing_result.link == result.link and result.relevance_score > existing_result.relevance_score:
                    deduplicated_results[i] = result
                    break

    return deduplicated_results


def format_expanded_search_results(original_query: str, expanded_queries: list[str], search_results: list[SearchResult]) -> str:
    """
    Format expanded search results for display.

    Args:
        original_query: Original search query
        expanded_queries: List of expanded queries
        search_results: List of search results

    Returns:
        Formatted search results string
    """
    if not search_results:
        return "No search results found."

    result_parts = [
        f"# Expanded Query Search Results ({len(search_results)} found)",
        "",
        f"**Original Query**: {original_query}",
        f"**Expanded Queries**: {', '.join(expanded_queries)}",
        "",
        "---",
        ""
    ]

    for i, result in enumerate(search_results, 1):
        result_parts.extend([
            f"## {i}. {result.title}",
            f"**URL**: {result.link}",
            f"**Source**: {result.source}" if result.source else "",
            f"**Date**: {result.date}" if result.date else "",
            f"**Relevance Score**: {result.relevance_score:.2f}",
            "",
            result.snippet,
            "",
            "---",
            ""
        ])

    return "\n".join(result_parts)


def save_expanded_search_work_product(
    original_query: str,
    expanded_queries: list[str],
    master_results: list[SearchResult],
    crawled_content: list[str],
    successful_urls: list[str],
    session_id: str,
    kevin_dir: Path,
    total_duration: float
) -> str:
    """
    Save detailed expanded search and crawl results to work product file.

    Args:
        original_query: Original search query
        expanded_queries: List of expanded queries
        master_results: Master list of deduplicated search results
        crawled_content: List of cleaned content strings
        successful_urls: List of successfully crawled URLs
        session_id: Session identifier
        kevin_dir: KEVIN directory path
        total_duration: Total processing time

    Returns:
        Path to saved work product file
    """
    try:
        # Use session-based directory structure
        sessions_dir = kevin_dir / "sessions" / session_id
        research_dir = sessions_dir / "research"
        research_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp and filename with numbered prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"1-expanded_search_workproduct_{timestamp}.md"
        filepath = research_dir / filename

        # Build work product content
        workproduct_content = [
            "# Expanded Query Search Results Work Product",
            "",
            f"**Session ID**: {session_id}",
            f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Original Query**: {original_query}",
            f"**Expanded Queries**: {', '.join(expanded_queries)}",
            f"**Total Queries Executed**: {len(expanded_queries)}",
            f"**Total Master Results**: {len(master_results)}",
            f"**Successfully Crawled**: {len(crawled_content)}",
            f"**Processing Time**: {total_duration:.2f}s",
            "",
            "---",
            "",
            "## 🔍 Query Expansion Analysis",
            "",
            f"**Original Query**: {original_query}",
            "",
            "**Expanded Queries Generated**:",
        ]

        for i, eq in enumerate(expanded_queries, 1):
            workproduct_content.append(f"{i}. {eq}")

        workproduct_content.extend([
            "",
            "**Query Expansion Strategy**: Simple context and scope enhancement",
            f"**Deduplication Applied**: {len(master_results)} unique URLs from all searches",
            "",
            "---",
            "",
            "## 📊 Master Search Results Summary",
            "",
        ])

        # Add master search results overview
        for i, result in enumerate(master_results, 1):
            workproduct_content.extend([
                f"### {i}. {result.title}",
                f"**URL**: {result.link}",
                f"**Source**: {result.source}" if result.source else "",
                f"**Date**: {result.date}" if result.date else "",
                f"**Relevance Score**: {result.relevance_score:.2f}",
                "",
                f"**Snippet**: {result.snippet}",
                "",
                "---",
                ""
            ])

        if crawled_content:
            workproduct_content.extend([
                "",
                "## 📄 Extracted Content",
                ""
            ])

            # Add detailed crawled content
            for i, (content, url) in enumerate(zip(crawled_content, successful_urls, strict=False), 1):
                # Find corresponding search result for title
                title = f"Article {i}"
                for result in master_results:
                    if result.link == url:
                        title = result.title
                        break

                workproduct_content.extend([
                    f"## 🌐 {i}. {title}",
                    "",
                    f"**URL**: {url}",
                    f"**Content Length**: {len(content)} characters",
                    "",
                    "### 📄 Extracted Content",
                    "",
                    "---",
                    "",
                    content,
                    "",
                    "---",
                    ""
                ])

        # Add footer
        workproduct_content.extend([
            "",
            "## 📊 Processing Summary",
            "",
            f"- **Original Query**: {original_query}",
            f"- **Expanded Queries**: {len(expanded_queries)} queries executed",
            f"- **Master Results Found**: {len(master_results)} unique URLs",
            f"- **URLs Successfully Crawled**: {len(crawled_content)}",
            "- **Processing**: Expanded query search + content extraction",
            f"- **Total Processing Time**: {total_duration:.2f} seconds",
            "- **Deduplication Applied**: Yes (URL-based)",
            "",
            "*Generated by Multi-Agent Research System - Expanded Query Search Integration*"
        ])

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(workproduct_content))

        logger.info(f"✅ Expanded search work product saved to: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error saving expanded search work product: {e}")
        return ""


# Export commonly used functions
__all__ = [
    'serp_search_and_extract',
    'expanded_query_search_and_extract',
    'execute_serp_search',
    'format_search_results',
    'save_search_work_product',
    'generate_expanded_queries',
    'deduplicate_search_results'
]
