"""Custom assertions for test validation."""

from pathlib import Path
from typing import Any


def assert_valid_session_data(session_data: dict[str, Any]):
    """Assert that session data contains all required fields."""
    required_fields = [
        "session_id", "topic", "user_requirements", "status",
        "created_at", "current_stage", "workflow_history"
    ]

    for field in required_fields:
        assert field in session_data, f"Missing required field: {field}"

    # Validate session ID format (UUID-like)
    session_id = session_data["session_id"]
    assert isinstance(session_id, str), "Session ID must be a string"
    assert len(session_id) > 10, "Session ID seems too short"

    # Validate topic
    topic = session_data["topic"]
    assert isinstance(topic, str), "Topic must be a string"
    assert len(topic) > 0, "Topic cannot be empty"

    # Validate user requirements
    requirements = session_data["user_requirements"]
    assert isinstance(requirements, dict), "User requirements must be a dict"
    required_requirement_fields = ["depth", "audience", "format"]
    for field in required_requirement_fields:
        assert field in requirements, f"Missing requirement field: {field}"

    # Validate status
    status = session_data["status"]
    valid_statuses = ["initialized", "researching", "generating_report", "editorial_review", "revising", "completed", "error"]
    assert status in valid_statuses, f"Invalid status: {status}"

    # Validate workflow history
    history = session_data["workflow_history"]
    assert isinstance(history, list), "Workflow history must be a list"


def assert_valid_research_results(research_data: dict[str, Any]):
    """Assert that research results have proper structure and quality."""
    required_fields = ["topic", "findings", "sources_used"]
    for field in required_fields:
        assert field in research_data, f"Missing research field: {field}"

    # Validate findings
    findings = research_data["findings"]
    assert isinstance(findings, list), "Findings must be a list"
    assert len(findings) > 0, "At least one finding is required"

    for i, finding in enumerate(findings):
        assert isinstance(finding, dict), f"Finding {i} must be a dict"
        assert "fact" in finding, f"Finding {i} missing 'fact' field"
        assert isinstance(finding["fact"], str), f"Finding {i} fact must be a string"
        assert len(finding["fact"]) > 10, f"Finding {i} fact seems too short"

        # Validate sources if present
        if "sources" in finding:
            sources = finding["sources"]
            assert isinstance(sources, list), f"Finding {i} sources must be a list"

    # Validate sources
    sources = research_data["sources_used"]
    assert isinstance(sources, list), "Sources must be a list"
    assert len(sources) > 0, "At least one source is required"

    for i, source in enumerate(sources):
        assert isinstance(source, dict), f"Source {i} must be a dict"
        required_source_fields = ["title", "url"]
        for field in required_source_fields:
            assert field in source, f"Source {i} missing {field}"

        # Validate URL format
        url = source["url"]
        assert isinstance(url, str), f"Source {i} URL must be a string"
        assert url.startswith(("http://", "https://")), f"Source {i} has invalid URL format"


def assert_valid_report_structure(report_data: dict[str, Any]):
    """Assert that report data has proper structure."""
    required_fields = ["title", "sections"]
    for field in required_fields:
        assert field in report_data, f"Missing report field: {field}"

    # Validate title
    title = report_data["title"]
    assert isinstance(title, str), "Report title must be a string"
    assert len(title) > 0, "Report title cannot be empty"

    # Validate sections
    sections = report_data["sections"]
    assert isinstance(sections, dict), "Report sections must be a dict"
    assert len(sections) > 0, "Report must have at least one section"

    # Check for common sections
    common_sections = ["summary", "findings", "conclusions"]
    has_common_section = any(section in sections for section in common_sections)
    assert has_common_section, "Report should include common sections like summary or findings"


def assert_valid_tool_result(tool_result: dict[str, Any], expected_tool: str):
    """Assert that tool result has proper structure."""
    assert "content" in tool_result, "Tool result missing 'content' field"

    content = tool_result["content"]
    assert isinstance(content, list), "Tool content must be a list"
    assert len(content) > 0, "Tool content cannot be empty"

    # Validate content items
    for i, item in enumerate(content):
        assert isinstance(item, dict), f"Content item {i} must be a dict"
        assert "type" in item, f"Content item {i} missing 'type' field"
        assert "text" in item, f"Content item {i} missing 'text' field"

    # Tool-specific validation
    if "research" in expected_tool.lower():
        assert "research_data" in tool_result, "Research tool missing research_data"
        assert_valid_research_results(tool_result["research_data"])

    elif "report" in expected_tool.lower():
        assert "report_data" in tool_result, "Report tool missing report_data"
        assert_valid_report_structure(tool_result["report_data"])


def assert_file_exists_and_not_empty(file_path: Path):
    """Assert that a file exists and is not empty."""
    assert file_path.exists(), f"File does not exist: {file_path}"
    assert file_path.stat().st_size > 0, f"File is empty: {file_path}"


def assert_markdown_content_valid(content: str):
    """Assert that markdown content has basic structure."""
    assert isinstance(content, str), "Content must be a string"
    assert len(content) > 100, "Content seems too short for a report"

    lines = content.split('\n')
    # Should have a title (starting with #)
    has_title = any(line.strip().startswith('# ') for line in lines)
    assert has_title, "Markdown content should have a title"

    # Should have multiple sections
    section_headers = [line for line in lines if line.strip().startswith('## ')]
    assert len(section_headers) >= 2, "Report should have multiple sections"


def assert_workflow_progression(workflow_history: list[dict[str, Any]]):
    """Assert that workflow has proper progression."""
    assert isinstance(workflow_history, list), "Workflow history must be a list"
    assert len(workflow_history) > 0, "Workflow history cannot be empty"

    expected_stages = ["research", "report_generation", "editorial_review"]
    actual_stages = [entry.get("stage") for entry in workflow_history]

    # Check that stages progress in expected order
    for expected_stage in expected_stages:
        assert expected_stage in actual_stages, f"Missing expected stage: {expected_stage}"

    # Validate each history entry
    for i, entry in enumerate(workflow_history):
        assert isinstance(entry, dict), f"History entry {i} must be a dict"
        assert "stage" in entry, f"History entry {i} missing stage"
        assert "completed_at" in entry, f"History entry {i} missing completed_at"


def assert_quality_metrics(metrics: dict[str, Any], min_thresholds: dict[str, float]):
    """Assert that quality metrics meet minimum thresholds."""
    for metric, threshold in min_thresholds.items():
        assert metric in metrics, f"Missing quality metric: {metric}"
        value = metrics[metric]
        assert isinstance(value, (int, float)), f"Metric {metric} must be numeric"
        assert value >= threshold, f"Metric {metric} ({value}) below threshold ({threshold})"


def assert_no_duplicate_sources(sources: list[dict[str, Any]]):
    """Assert that there are no duplicate sources."""
    urls = [source.get("url", "").lower() for source in sources]
    unique_urls = set(urls)
    assert len(urls) == len(unique_urls), "Found duplicate source URLs"


def assert_sources_are_credible(sources: list[dict[str, Any]]):
    """Assert that sources meet basic credibility criteria."""
    for i, source in enumerate(sources):
        # Should have title and URL
        assert source.get("title"), f"Source {i} missing title"
        assert source.get("url"), f"Source {i} missing URL"

        # URL should be from a credible domain
        url = source["url"].lower()
        suspicious_patterns = [
            "example.com", "test.com", "fake.com", "spam."
        ]
        for pattern in suspicious_patterns:
            assert pattern not in url, f"Source {i} URL looks suspicious: {url}"


def assert_findings_are_supported(findings: list[dict[str, Any]]):
    """Assert that findings have supporting sources."""
    for i, finding in enumerate(findings):
        if "sources" in finding:
            sources = finding["sources"]
            assert isinstance(sources, list), f"Finding {i} sources must be a list"
            assert len(sources) > 0, f"Finding {i} should have supporting sources"


def assert_session_directory_structure(session_dir: Path):
    """Assert that session directory has proper structure."""
    assert session_dir.exists(), f"Session directory does not exist: {session_dir}"
    assert session_dir.is_dir(), f"Session path is not a directory: {session_dir}"

    # Check for expected files
    expected_files = ["session_state.json"]
    for file_name in expected_files:
        file_path = session_dir / file_name
        assert_file_exists_and_not_empty(file_path)


def assert_time_reasonable(start_time: str, end_time: str, max_duration_minutes: int = 30):
    """Assert that task completion time is reasonable."""
    from datetime import datetime

    try:
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
    except ValueError as e:
        assert False, f"Invalid datetime format: {e}"

    duration = end_dt - start_dt
    duration_minutes = duration.total_seconds() / 60

    assert duration_minutes <= max_duration_minutes, f"Task took too long: {duration_minutes:.1f} minutes"
    assert duration_minutes > 0, "End time should be after start time"
