from collections import Counter, defaultdict
from typing import Dict, List
from urllib.parse import urlparse

import pytest

from multi_agent_research_system.utils.serp_search_utils import (
    SearchResult,
    generate_search_plan,
)


def weighted_candidate_selection(
    query_results: Dict[str, List[SearchResult]],
    *,
    primary_label: str = "primary",
    primary_weight: int = 2,
    domain_penalty_alpha: float = 0.6,
    domain_cap: int = 2,
    min_effective_score: float = 0.25,
    max_candidates: int = 6
) -> List[dict]:
    """
    Prototype helper that mirrors the proposed target system.

    Returns a list of dicts containing the selected results with metadata
    that can be asserted in tests. This function intentionally lives inside
    the test module so it can evolve in parallel with the production logic.
    """
    queues: Dict[str, List[SearchResult]] = {
        label: list(results) for label, results in query_results.items()
    }
    selection: List[dict] = []
    domain_counts: defaultdict[str, int] = defaultdict(int)

    orthogonal_labels = [label for label in queues.keys() if label != primary_label]
    primary_taken_since_reset = 0
    orth_index = 0

    def pull_candidate(label: str) -> dict | None:
        queue = queues[label]
        while queue:
            result = queue.pop(0)
            domain = urlparse(result.link).netloc.lower()
            if not domain:
                domain = result.link.lower()

            # Enforce soft cap: skip once domain quota is reached
            if domain_counts[domain] >= domain_cap:
                continue

            effective_score = result.relevance_score / (1 + domain_penalty_alpha * domain_counts[domain])
            if effective_score < min_effective_score:
                continue

            domain_counts[domain] += 1
            return {
                "query_label": label,
                "domain": domain,
                "url": result.link,
                "effective_score": round(effective_score, 3),
                "original_score": result.relevance_score
            }
        return None

    while len(selection) < max_candidates:
        primary_available = bool(queues.get(primary_label))
        orth_available = any(queues[label] for label in orthogonal_labels)
        candidate = None

        if primary_available and (primary_taken_since_reset < primary_weight or not orth_available):
            candidate = pull_candidate(primary_label)
            if candidate:
                primary_taken_since_reset += 1
        elif orth_available:
            attempts = 0
            while attempts < len(orthogonal_labels) and not candidate:
                label = orthogonal_labels[orth_index % len(orthogonal_labels)]
                orth_index += 1
                candidate = pull_candidate(label)
                attempts += 1
            if candidate:
                primary_taken_since_reset = 0
        else:
            break

        if not candidate:
            if not primary_available and not orth_available:
                break
            continue

        selection.append(candidate)

    return selection


def test_weighted_candidate_selection_balances_focus_and_diversity():
    primary_results = [
        SearchResult("Primary 1", "https://domainA.com/article/1", "snippet", relevance_score=0.92),
        SearchResult("Primary 2", "https://domainA.com/article/2", "snippet", relevance_score=0.88),
        SearchResult("Primary 3", "https://domainB.com/article/1", "snippet", relevance_score=0.83),
        SearchResult("Primary 4", "https://domainC.com/article/1", "snippet", relevance_score=0.75),
    ]

    orthogonal_one = [
        SearchResult("Orthogonal 1", "https://domainD.com/report/1", "snippet", relevance_score=0.86),
        SearchResult("Orthogonal 2", "https://domainA.com/related/3", "snippet", relevance_score=0.65),
    ]

    orthogonal_two = [
        SearchResult("Orthogonal Alt 1", "https://domainE.com/analysis/1", "snippet", relevance_score=0.81),
        SearchResult("Orthogonal Alt 2", "https://domainF.com/analysis/2", "snippet", relevance_score=0.74),
    ]

    selected = weighted_candidate_selection(
        {
            "primary": primary_results,
            "angle_one": orthogonal_one,
            "angle_two": orthogonal_two,
        },
        max_candidates=6
    )

    labels = [item["query_label"] for item in selected]
    domains = Counter(item["domain"] for item in selected)

    # Primary focus should dominate with a 2:1 ratio versus orthogonals.
    assert labels.count("primary") == 4
    assert labels.count("angle_one") == 1
    assert labels.count("angle_two") == 1
    assert labels[:2] == ["primary", "primary"]

    # Domain cap should prevent more than two hits from the same domain.
    assert domains["domaina.com"] == 2

    # Each orthogonal angle should surface at least once for diversity.
    assert "angle_one" in labels and "angle_two" in labels

    # The selection should respect the requested maximum count.
    assert len(selected) == 6


@pytest.mark.asyncio
async def test_generate_search_plan_includes_skip_flags():
    plan = await generate_search_plan(
        "solar energy policy analysis",
        allow_expansion=True,
        max_orthogonal_queries=2
    )

    assert plan["optimized_query"]
    assert plan["scope"] in {"brief", "default", "comprehensive"}
    assert len(plan["orthogonal_queries"]) <= 2

    for entry in plan["orthogonal_queries"]:
        assert "skip_if_editor_directive" in entry
        assert isinstance(entry["skip_if_editor_directive"], bool)
        assert entry["query"]
        assert entry["angle"]
