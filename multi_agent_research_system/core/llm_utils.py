"""
LLM utility functions for quick scope determination and lightweight planning tasks.
"""

import json
import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)


def _infer_scope_from_text(text: str) -> Tuple[str, str]:
    """Heuristic scope inference used by the mock planner."""
    lowered = (text or "").lower()
    brief_keywords = {"brief", "summary", "quick", "short", "overview"}
    comprehensive_keywords = {"comprehensive", "detailed", "extensive", "thorough", "in-depth", "analysis"}

    for keyword in brief_keywords:
        if keyword in lowered:
            return "brief", f"Detected keyword '{keyword}' implying a brief scope"

    for keyword in comprehensive_keywords:
        if keyword in lowered:
            return "comprehensive", f"Detected keyword '{keyword}' implying a comprehensive scope"

    return "default", "No explicit scope cues detected; defaulting to balanced coverage"


def _heuristic_optimize_query(query: str) -> str:
    """Simple token de-duplication to mimic query optimization."""
    tokens = (query or "").split()
    seen = set()
    optimized_tokens: list[str] = []

    for token in tokens:
        cleaned = re.sub(r"[^\w]", "", token).lower()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        optimized_tokens.append(token)

    optimized = " ".join(optimized_tokens).strip()
    return optimized or (query or "").strip()


def _mock_orthogonal_queries(base_query: str, max_count: int) -> list[dict]:
    """Generate deterministic orthogonal queries for testing without real LLM calls."""
    if max_count <= 0:
        return []

    seeds = [
        ("impacts and implications", "impacts and implications"),
        ("expert perspectives", "expert commentary and analysis"),
        ("responses and strategies", "responses strategies and plans"),
        ("data insights", "data trends statistics and metrics"),
        ("background context", "background and context"),
    ]

    orthogonals: list[dict] = []
    seen = {base_query.lower()}

    for angle, descriptor in seeds:
        candidate = f"{base_query} {descriptor}".strip()
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)

        orthogonals.append({
            "query": candidate,
            "angle": angle,
            "descriptor": descriptor,
            "skip_if_editor_directive": True
        })

        if len(orthogonals) >= max_count:
            break

    return orthogonals


async def quick_llm_call(prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
    """
    Make a quick LLM call for lightweight decisions like scope determination.

    The current implementation provides a deterministic mock so tests can run
    without live LLM connectivity while we wire up the production agent client.
    """
    try:
        prompt_lower = prompt.lower()

        # Unified planner branch
        if "targeted query planner" in prompt_lower and '"optimized_query"' in prompt_lower:
            original_match = re.search(r'Original Query:\s*"(.+?)"', prompt, re.IGNORECASE | re.DOTALL)
            original_query = (original_match.group(1).strip() if original_match else prompt).strip()

            allow_match = re.search(r"allow_orthogonals:\s*(true|false|1|0)", prompt_lower)
            allow_expansion = True
            if allow_match:
                allow_expansion = allow_match.group(1) in {"true", "1"}

            max_match = re.search(r"max_orthogonals:\s*(\d+)", prompt, re.IGNORECASE)
            max_orthogonals = int(max_match.group(1)) if max_match else 2

            optimized_query = _heuristic_optimize_query(original_query)
            scope, scope_reason = _infer_scope_from_text(original_query)

            orthogonal_queries = []
            if allow_expansion and max_orthogonals > 0:
                orthogonal_queries = _mock_orthogonal_queries(optimized_query, max_orthogonals)

            planner_response = {
                "optimized_query": optimized_query,
                "orthogonal_queries": orthogonal_queries,
                "scope": scope,
                "reasoning": scope_reason,
                "confidence": "medium"
            }

            logger.info(f"Mock planner produced plan for query '{original_query}' (scope={scope})")
            return json.dumps(planner_response)

        # Lightweight scope inference branch (legacy behaviour)
        scope, reasoning = _infer_scope_from_text(prompt_lower)
        special_req = ""

        focus_match = re.search(r"focus on ([^.]+)", prompt_lower)
        if focus_match:
            special_req = f"Special focus: {focus_match.group(1).strip()}"
        else:
            emphasize_match = re.search(r"emphasize ([^.]+)", prompt_lower)
            if emphasize_match:
                special_req = f"Special focus: {emphasize_match.group(1).strip()}"

        response = {
            "scope": scope,
            "reasoning": reasoning,
            "special_requirements": special_req,
            "confidence": "high"
        }

        logger.info(f"Quick LLM scope inference result: {response}")
        return json.dumps(response)

    except Exception as exc:
        logger.error(f"Quick LLM call failed: {exc}")
        fallback = {
            "scope": "default",
            "reasoning": "LLM call failed, using default",
            "special_requirements": "",
            "confidence": "low"
        }
        return json.dumps(fallback)
