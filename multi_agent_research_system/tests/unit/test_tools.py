"""Unit tests for individual tool functions."""

import os
import sys

import pytest

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.research_tools import (
    analyze_sources,
    conduct_research,
    generate_report,
    identify_research_gaps,
    manage_session,
    review_report,
    revise_report,
    save_report,
)


class TestResearchTools:
    """Test individual research tool functions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conduct_research_tool(self):
        """Test conduct_research tool function."""
        args = {
            "topic": "Artificial Intelligence in Healthcare",
            "depth": "medium",
            "focus_areas": ["diagnostics", "treatment"],
            "max_sources": 5
        }

        result = await conduct_research(args)

        # Test result structure
        assert "content" in result
        assert "research_data" in result

        # Test content
        content = result["content"]
        assert isinstance(content, list)
        assert len(content) > 0

        content_item = content[0]
        assert content_item["type"] == "text"
        assert "research completed" in content_item["text"].lower()

        # Test research data
        research_data = result["research_data"]
        assert "topic" in research_data
        assert "findings" in research_data
        assert "sources_used" in research_data

        assert research_data["topic"] == args["topic"]
        assert isinstance(research_data["findings"], list)
        assert isinstance(research_data["sources_used"], list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conduct_research_tool_minimal_args(self):
        """Test conduct_research with minimal arguments."""
        args = {"topic": "Test Topic"}

        result = await conduct_research(args)

        assert "content" in result
        assert "research_data" in result
        assert result["research_data"]["topic"] == "Test Topic"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analyze_sources_tool(self):
        """Test analyze_sources tool function."""
        sources = [
            {
                "title": "Academic Study on AI",
                "url": "https://academic.edu/study",
                "type": "academic",
                "reliability": "high"
            },
            {
                "title": "News Article About AI",
                "url": "https://news.com/ai-article",
                "type": "news",
                "reliability": "medium"
            }
        ]

        args = {"sources": sources}
        result = await analyze_sources(args)

        # Test result structure
        assert "content" in result
        assert "analysis_result" in result

        # Test content
        content = result["content"][0]
        assert "source analysis completed" in content["text"].lower()

        # Test analysis result
        analysis = result["analysis_result"]
        assert "sources_analyzed" in analysis
        assert "reliability_scores" in analysis
        assert "overall_assessment" in analysis

        assert analysis["sources_analyzed"] == 2
        assert isinstance(analysis["reliability_scores"], dict)
        assert isinstance(analysis["overall_assessment"], dict)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_report_tool(self):
        """Test generate_report tool function."""
        research_data = {
            "topic": "Remote Work Productivity",
            "findings": [
                {
                    "fact": "Remote work increases productivity for many employees",
                    "sources": ["study1.com", "study2.com"],
                    "confidence": "high"
                }
            ],
            "sources_used": [
                {
                    "title": "Productivity Study",
                    "url": "https://study1.com",
                    "type": "academic"
                }
            ]
        }

        args = {
            "research_data": research_data,
            "format": "markdown",
            "audience": "business",
            "sections": ["summary", "findings", "conclusions"]
        }

        result = await generate_report(args)

        # Test result structure
        assert "content" in result
        assert "report_data" in result

        # Test content
        content = result["content"][0]
        assert "report generated" in content["text"].lower()

        # Test report data
        report_data = result["report_data"]
        assert "title" in report_data
        assert "sections" in report_data
        assert "executive_summary" in report_data
        assert "key_findings" in report_data

        assert "Remote Work Productivity" in report_data["title"]
        assert isinstance(report_data["sections"], dict)
        assert len(report_data["sections"]) == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_revise_report_tool(self):
        """Test revise_report tool function."""
        current_report = {
            "title": "Original Report",
            "sections": {
                "summary": "Original summary",
                "findings": "Original findings"
            },
            "version": 1
        }

        feedback = [
            "Add more recent sources",
            "Improve clarity in conclusions"
        ]

        args = {
            "current_report": current_report,
            "feedback": feedback
        }

        result = await revise_report(args)

        # Test result structure
        assert "content" in result
        assert "revised_report" in result

        # Test content
        content = result["content"][0]
        assert "revised" in content["text"].lower()

        # Test revised report
        revised_report = result["revised_report"]
        assert revised_report["version"] == 2
        assert "revisions_made" in revised_report
        assert revised_report["revisions_made"] == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_review_report_tool(self):
        """Test review_report tool function."""
        report = {
            "title": "Report for Review",
            "sections": {
                "summary": "This is the summary section",
                "findings": "These are the key findings",
                "conclusions": "These are the conclusions"
            }
        }

        args = {
            "report": report,
            "review_criteria": ["accuracy", "clarity", "completeness"]
        }

        result = await review_report(args)

        # Test result structure
        assert "content" in result
        assert "review_result" in result

        # Test content
        content = result["content"][0]
        assert "review completed" in content["text"].lower()
        assert "overall score" in content["text"].lower()

        # Test review result
        review_result = result["review_result"]
        assert "overall_score" in review_result
        assert "criteria_scores" in review_result
        assert "strengths" in review_result
        assert "improvement_areas" in review_result
        assert "feedback" in review_result
        assert "recommendation" in review_result

        assert isinstance(review_result["overall_score"], (int, float))
        assert isinstance(review_result["criteria_scores"], dict)
        assert isinstance(review_result["strengths"], list)
        assert isinstance(review_result["improvement_areas"], list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_identify_research_gaps_tool(self):
        """Test identify_research_gaps tool function."""
        report = {
            "title": "Incomplete Report",
            "sections": {
                "summary": "Summary with limited data",
                "findings": "Some findings but could be more comprehensive"
            }
        }

        args = {
            "report": report,
            "required_completeness": "comprehensive"
        }

        result = await identify_research_gaps(args)

        # Test result structure
        assert "content" in result
        assert "research_gaps" in result

        # Test content
        content = result["content"][0]
        assert "research gaps" in content["text"].lower()

        # Test research gaps
        gaps = result["research_gaps"]
        assert isinstance(gaps, list)
        assert len(gaps) > 0

        for gap in gaps:
            assert "gap" in gap
            assert "section" in gap
            assert "priority" in gap

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manage_session_tool(self):
        """Test manage_session tool function."""
        session_id = "test-session-123"
        action = "update"
        data = {"status": "in_progress", "current_stage": "research"}

        args = {
            "session_id": session_id,
            "action": action,
            "data": data
        }

        result = await manage_session(args)

        # Test result structure
        assert "content" in result
        assert "session_result" in result

        # Test content
        content = result["content"][0]
        assert session_id in content["text"]
        assert action in content["text"]

        # Test session result
        session_result = result["session_result"]
        assert session_result["session_id"] == session_id
        assert session_result["status"] == "updated"
        assert "path" in session_result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_save_report_tool(self):
        """Test save_report tool function."""
        report = {
            "title": "Test Report",
            "executive_summary": "This is a test report",
            "sections": {
                "summary": "Summary content",
                "findings": "Findings content"
            }
        }

        args = {
            "report": report,
            "session_id": "test-session-456",
            "format": "markdown",
            "version": 1
        }

        result = await save_report(args)

        # Test result structure
        assert "content" in result
        assert "save_result" in result

        # Test content
        content = result["content"][0]
        assert "saved to:" in content["text"].lower()

        # Test save result
        save_result = result["save_result"]
        assert "filepath" in save_result
        assert save_result["version"] == 1
        assert save_result["format"] == "markdown"
        assert "size" in save_result

        # Verify file was actually created
        from pathlib import Path
        filepath = Path(save_result["filepath"])
        assert filepath.exists()
        assert filepath.stat().st_size > 0

        # Clean up test file
        if filepath.exists():
            filepath.unlink()

    @pytest.mark.unit
    def test_tool_decorators(self):
        """Test that all tools have proper decorators."""
        tools_to_test = [
            conduct_research,
            analyze_sources,
            generate_report,
            revise_report,
            review_report,
            identify_research_gaps,
            manage_session,
            save_report
        ]

        for tool_func in tools_to_test:
            # Test that tools have the required metadata
            assert hasattr(tool_func, '_tool_name'), f"{tool_func.__name__} missing tool name"
            assert hasattr(tool_func, '_tool_description'), f"{tool_func.__name__} missing tool description"
            assert hasattr(tool_func, '_tool_params'), f"{tool_func.__name__} missing tool params"

            # Test tool name format
            tool_name = tool_func._tool_name
            assert isinstance(tool_name, str)
            assert len(tool_name) > 0

            # Test tool description
            description = tool_func._tool_description
            assert isinstance(description, str)
            assert len(description) > 10

            # Test tool params
            params = tool_func._tool_params
            assert isinstance(params, dict)
            assert len(params) > 0

    @pytest.mark.unit
    def test_convert_to_markdown_function(self):
        """Test the convert_to_markdown utility function."""
        from core.research_tools import convert_to_markdown

        report = {
            "title": "Test Report",
            "executive_summary": "This is the executive summary",
            "key_findings": [
                {
                    "fact": "Important finding 1",
                    "context": "Context for finding 1",
                    "sources": ["source1.com", "source2.com"]
                },
                {
                    "fact": "Important finding 2",
                    "sources": ["source3.com"]
                }
            ],
            "sections": {
                "introduction": "This is the introduction",
                "methodology": "This is the methodology"
            },
            "conclusions": [
                "Main conclusion 1",
                "Main conclusion 2"
            ]
        }

        markdown = convert_to_markdown(report)

        # Test markdown structure
        assert "# Test Report" in markdown
        assert "## Executive Summary" in markdown
        assert "## Key Findings" in markdown
        assert "## Introduction" in markdown
        assert "## Conclusions" in markdown

        # Test content inclusion
        assert "This is the executive summary" in markdown
        assert "Important finding 1" in markdown
        assert "Main conclusion 1" in markdown

        # Test source inclusion
        assert "source1.com" in markdown

        # Test markdown formatting
        assert markdown.startswith("# ")
        assert "### " in markdown  # Subheadings for findings
        assert "- " in markdown  # List items for conclusions
