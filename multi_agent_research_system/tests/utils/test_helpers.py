"""Test helper functions for the multi-agent research system."""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def create_mock_session(
    session_id: str | None = None,
    topic: str = "Test Research Topic",
    status: str = "in_progress",
    current_stage: str = "research"
) -> dict[str, Any]:
    """Create mock session data for testing."""
    if session_id is None:
        session_id = str(uuid.uuid4())

    return {
        "session_id": session_id,
        "topic": topic,
        "user_requirements": {
            "depth": "Standard Research",
            "audience": "General",
            "format": "Standard Report",
            "requirements": "Test requirements",
            "timeline": "Within 24 hours"
        },
        "status": status,
        "created_at": datetime.now().isoformat(),
        "current_stage": current_stage,
        "workflow_history": [
            {
                "stage": "research",
                "completed_at": datetime.now().isoformat(),
                "results_count": 5
            }
        ],
        "final_report": None
    }


def validate_research_output(research_data: dict[str, Any]) -> bool:
    """Validate that research output contains expected fields."""
    required_fields = ["topic", "findings", "sources_used"]

    for field in required_fields:
        if field not in research_data:
            return False

    # Validate findings structure
    findings = research_data.get("findings", [])
    for finding in findings:
        if not isinstance(finding, dict):
            return False
        if "fact" not in finding:
            return False

    # Validate sources structure
    sources = research_data.get("sources_used", [])
    for source in sources:
        if not isinstance(source, dict):
            return False
        if "title" not in source or "url" not in source:
            return False

    return True


def compare_tool_results(result1: dict[str, Any], result2: dict[str, Any]) -> float:
    """Compare two tool results and return similarity score (0-1)."""
    if not result1 or not result2:
        return 0.0

    # Compare content
    content1 = result1.get("content", [])
    content2 = result2.get("content", [])

    if not content1 or not content2:
        return 0.0

    # Simple text comparison for now
    text1 = " ".join([c.get("text", "") for c in content1 if isinstance(c, dict)])
    text2 = " ".join([c.get("text", "") for c in content2 if isinstance(c, dict)])

    if not text1 or not text2:
        return 0.0

    # Calculate simple word overlap
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


async def async_test_wrapper(test_func, *args, **kwargs):
    """Wrapper for async test functions with timeout."""
    timeout = kwargs.pop("timeout", 30.0)

    try:
        return await asyncio.wait_for(test_func(*args, **kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Test timed out after {timeout} seconds")


def create_test_session_files(session_dir: Path, session_data: dict[str, Any]) -> None:
    """Create test session files for testing file operations."""
    # Create session state file
    session_file = session_dir / "session_state.json"
    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=2, default=str)

    # Create sample research results
    research_file = session_dir / "research_results.json"
    sample_research = {
        "topic": session_data["topic"],
        "findings": [
            {
                "fact": "Sample finding 1",
                "sources": ["source1.com", "source2.com"],
                "confidence": "high",
                "context": "Additional context for finding 1"
            },
            {
                "fact": "Sample finding 2",
                "sources": ["source3.com"],
                "confidence": "medium",
                "context": "Additional context for finding 2"
            }
        ],
        "sources_used": [
            {
                "title": "Sample Source 1",
                "url": "https://source1.com",
                "type": "academic",
                "reliability": "high",
                "date": "2024-01-01"
            }
        ]
    }

    with open(research_file, 'w') as f:
        json.dump(sample_research, f, indent=2, default=str)

    # Create sample report
    report_file = session_dir / "report_v1_20240101_120000.md"
    report_content = f"""# {session_data['topic']} Report

## Executive Summary
This is a sample executive summary for {session_data['topic']}.

## Key Findings
- Sample finding 1: This is an important finding from our research
- Sample finding 2: Another key discovery that deserves attention

## Analysis
The analysis section would contain detailed examination of the findings...

## Conclusions
Based on the research conducted, we can conclude that...

## Sources
1. Sample Source 1 - https://source1.com
"""

    with open(report_file, 'w') as f:
        f.write(report_content)


def validate_file_structure(base_dir: Path, expected_files: list[str]) -> bool:
    """Validate that expected files exist in the directory structure."""
    for file_path in expected_files:
        full_path = base_dir / file_path
        if not full_path.exists():
            return False
        if full_path.stat().st_size == 0:
            return False
    return True


def count_words_in_text(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from text using simple regex."""
    import re
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def validate_markdown_structure(content: str) -> dict[str, bool]:
    """Validate markdown structure and return analysis."""
    lines = content.split('\n')

    has_title = any(line.strip().startswith('# ') for line in lines)
    has_sections = any(line.strip().startswith('## ') for line in lines)
    has_list_items = any(line.strip().startswith(('- ', '* ', '+ ')) for line in lines)

    return {
        "has_title": has_title,
        "has_sections": has_sections,
        "has_list_items": has_list_items,
        "is_valid_markdown": has_title  # Basic validation
    }


def create_sample_user_request() -> dict[str, Any]:
    """Create a sample user request for testing."""
    return {
        "topic": "The Future of Electric Vehicles",
        "depth": "Comprehensive Analysis",
        "audience": "Technical",
        "format": "Technical Documentation",
        "requirements": "Include current market trends, technology developments, and environmental impact",
        "timeline": "Within 1 week"
    }


def measure_research_quality(research_results: dict[str, Any]) -> dict[str, Any]:
    """Measure quality metrics for research results."""
    findings = research_results.get("findings", [])
    sources = research_results.get("sources_used", [])

    return {
        "total_findings": len(findings),
        "total_sources": len(sources),
        "avg_confidence": calculate_avg_confidence(findings),
        "source_diversity": calculate_source_diversity(sources),
        "coverage_score": calculate_coverage_score(findings)
    }


def calculate_avg_confidence(findings: list[dict[str, Any]]) -> float:
    """Calculate average confidence score from findings."""
    if not findings:
        return 0.0

    confidence_scores = []
    for finding in findings:
        confidence = finding.get("confidence", "medium").lower()
        if confidence == "high":
            confidence_scores.append(3)
        elif confidence == "medium":
            confidence_scores.append(2)
        elif confidence == "low":
            confidence_scores.append(1)
        else:
            confidence_scores.append(2)  # Default to medium

    return sum(confidence_scores) / len(confidence_scores)


def calculate_source_diversity(sources: list[dict[str, Any]]) -> float:
    """Calculate source diversity score."""
    if not sources:
        return 0.0

    source_types = set()
    domains = set()

    for source in sources:
        source_type = source.get("type", "unknown")
        url = source.get("url", "")

        source_types.add(source_type)

        # Extract domain from URL
        if url:
            try:
                domain = url.split('/')[2]
                domains.add(domain)
            except IndexError:
                pass

    # Diversity score based on variety of source types and domains
    type_score = len(source_types) / 5  # Assuming max 5 types
    domain_score = min(len(domains) / 10, 1.0)  # Normalize to max 1.0

    return (type_score + domain_score) / 2


def calculate_coverage_score(findings: list[dict[str, Any]]) -> float:
    """Calculate coverage score based on finding completeness."""
    if not findings:
        return 0.0

    complete_findings = 0
    for finding in findings:
        required_fields = ["fact", "sources"]
        if all(field in finding for field in required_fields):
            complete_findings += 1

    return complete_findings / len(findings)
