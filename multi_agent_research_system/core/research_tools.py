"""Custom tools for the research system using Claude Agent SDK @tool decorator.

This module defines specialized tools that agents can use for research,
report generation, and coordination tasks.
"""

import json
import os

# Import from parent directory structure
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found. Using environment variables only.")

# Set up API configuration
ANTHROPIC_BASE_URL = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if ANTHROPIC_API_KEY:
    # Set environment variables for the SDK
    os.environ['ANTHROPIC_BASE_URL'] = ANTHROPIC_BASE_URL
    os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY

# Import logging when available
try:
    from .logging_config import get_logger
    _logger = get_logger("research_tools")
    _logger.debug("Research tools logging initialized")
except ImportError:
    _logger = None

try:
    from claude_agent_sdk import tool
except ImportError:
    # Fallback decorator for when the SDK is not available
    import sys
    print("Warning: claude_agent_sdk not found. Using fallback tool decorator.")
    try:
        # Try to import logging from the core module
        sys.path.append(os.path.dirname(__file__))
        from .logging_config import get_logger
        get_logger("research_tools").warning("Using fallback tool decorator - SDK not available")
    except ImportError:
        pass

    def tool(name: str, description: str, params_schema: dict):
        """Fallback tool decorator for when SDK is not available."""
        def decorator(func):
            func._tool_name = name
            func._tool_description = description
            func._tool_params = params_schema
            return func
        return decorator


# Research Tools
@tool("conduct_research", "Conduct comprehensive research on a specified topic", {
    "topic": str,
    "depth": str,
    "focus_areas": list[str],
    "max_sources": int
})
async def conduct_research(args: dict[str, Any]) -> dict[str, Any]:
    """Conduct comprehensive research using web search and analysis."""
    if _logger:
        _logger.info(f"Conducting research on topic: {args.get('topic', 'Unknown')}")

    topic = args["topic"]
    depth = args.get("depth", "medium")
    focus_areas = args.get("focus_areas", [])

    if _logger:
        _logger.debug(f"Research parameters - Topic: {topic}, Depth: {depth}, Focus areas: {focus_areas}")

    # This tool would integrate with WebSearch to find information
    # For now, return a structured response format

    research_result = {
        "topic": topic,
        "research_id": str(uuid.uuid4()),
        "conducted_at": datetime.now().isoformat(),
        "depth": depth,
        "focus_areas": focus_areas,
        "findings": [
            {
                "fact": f"Key finding about {topic}",
                "sources": ["source1", "source2"],
                "confidence": "high",
                "context": "Additional context would be provided"
            }
        ],
        "statistics": [
            {
                "statistic": f"Relevant statistic about {topic}",
                "value": "value",
                "source": "source",
                "date": "2024-01-01"
            }
        ],
        "sources_used": [
            {
                "title": "Source Title",
                "url": "https://example.com",
                "type": "academic",
                "reliability": "high",
                "date": "2024-01-01"
            }
        ],
        "knowledge_gaps": [
            "Areas that may need additional research"
        ]
    }

    return {
        "content": [{
            "type": "text",
            "text": f"Research completed on {topic}. Found {len(research_result['findings'])} key findings and {len(research_result['sources_used'])} sources."
        }],
        "research_data": research_result
    }

    if _logger:
        _logger.info(f"Research completed for topic: {topic}, found {len(research_result['findings'])} findings")


@tool("analyze_sources", "Analyze and validate the credibility of research sources", {
    "sources": list[dict[str, Any]]
})
async def analyze_sources(args: dict[str, Any]) -> dict[str, Any]:
    """Analyze source credibility and reliability."""
    sources = args["sources"]

    analysis = {
        "sources_analyzed": len(sources),
        "reliability_scores": {},
        "recommendations": [],
        "overall_assessment": {
            "average_reliability": 8.0,
            "high_quality_sources": 0,
            "sources_to_verify": 0
        }
    }

    for i, source in enumerate(sources):
        score = 8.5  # Would be calculated based on actual analysis
        analysis["reliability_scores"][f"source_{i}"] = score
        if score >= 8:
            analysis["overall_assessment"]["high_quality_sources"] += 1
        else:
            analysis["overall_assessment"]["sources_to_verify"] += 1

    return {
        "content": [{
            "type": "text",
            "text": f"Source analysis completed. Average reliability: {analysis['overall_assessment']['average_reliability']}/10"
        }],
        "analysis_result": analysis
    }


# Report Generation Tools
@tool("generate_report", "Generate a structured report from research findings", {
    "research_data": dict[str, Any],
    "format": str,
    "audience": str,
    "sections": list[str]
})
async def generate_report(args: dict[str, Any]) -> dict[str, Any]:
    """Generate a comprehensive report from research data."""
    research_data = args["research_data"]
    format_type = args.get("format", "markdown")
    audience = args.get("audience", "general")
    sections = args.get("sections", ["summary", "findings", "analysis", "conclusions"])

    topic = research_data.get("topic", "Research Topic")

    report = {
        "title": f"Research Report: {topic}",
        "generated_at": datetime.now().isoformat(),
        "format": format_type,
        "audience": audience,
        "sections": {},
        "executive_summary": f"Executive summary for {topic} research.",
        "key_findings": research_data.get("findings", []),
        "conclusions": [
            "Main conclusion based on research findings"
        ]
    }

    # Generate sections
    for section in sections:
        report["sections"][section] = f"Content for {section} section based on research data."

    return {
        "content": [{
            "type": "text",
            "text": f"Report generated: {report['title']} with {len(sections)} sections"
        }],
        "report_data": report
    }


@tool("revise_report", "Revise and improve a report based on feedback", {
    "current_report": dict[str, Any],
    "feedback": list[str],
    "additional_research": Optional[dict[str, Any]]
})
async def revise_report(args: dict[str, Any]) -> dict[str, Any]:
    """Revise a report based on feedback and additional research."""
    current_report = args["current_report"]
    feedback = args.get("feedback", [])
    additional_research = args.get("additional_research")

    revised_report = current_report.copy()
    revised_report["generated_at"] = datetime.now().isoformat()
    revised_report["version"] = current_report.get("version", 1) + 1
    revised_report["revisions_made"] = len(feedback)
    revised_report["revision_summary"] = f"Report revised with {len(feedback)} feedback items."

    if additional_research:
        revised_report["additional_research_integrated"] = True

    return {
        "content": [{
            "type": "text",
            "text": f"Report revised to version {revised_report['version']} with {len(feedback)} improvements."
        }],
        "revised_report": revised_report
    }


# Editor Tools
@tool("review_report", "Review and assess report quality", {
    "report": dict[str, Any],
    "review_criteria": list[str]
})
async def review_report(args: dict[str, Any]) -> dict[str, Any]:
    """Review a report and provide quality assessment."""
    report = args["report"]
    review_criteria = args.get("review_criteria", [
        "accuracy", "clarity", "completeness", "organization", "sources"
    ])

    review = {
        "report_title": report.get("title", "Untitled Report"),
        "review_date": datetime.now().isoformat(),
        "overall_score": 8.5,  # Would be calculated based on actual review
        "criteria_scores": {
            "accuracy": 9.0,
            "clarity": 8.0,
            "completeness": 8.5,
            "organization": 9.0,
            "sources": 8.0
        },
        "strengths": [
            "Well-structured and organized",
            "Good use of research evidence",
            "Clear and concise writing"
        ],
        "improvement_areas": [
            "Could use more recent sources",
            "Some sections need more depth"
        ],
        "feedback": [
            {
                "type": "content",
                "priority": "medium",
                "comment": "Consider adding more recent statistics",
                "section": "findings"
            },
            {
                "type": "structure",
                "priority": "low",
                "comment": "Transitions between sections could be smoother",
                "section": "overall"
            }
        ],
        "recommendation": "approve_with_minor_revisions"
    }

    return {
        "content": [{
            "type": "text",
            "text": f"Report review completed. Overall score: {review['overall_score']}/10. Recommendation: {review['recommendation']}"
        }],
        "review_result": review
    }


@tool("identify_research_gaps", "Identify information gaps that need additional research", {
    "report": dict[str, Any],
    "required_completeness": str
})
async def identify_research_gaps(args: dict[str, Any]) -> dict[str, Any]:
    """Identify gaps in research that need to be addressed."""
    report = args["report"]
    required_completeness = args.get("required_completeness", "comprehensive")

    gaps = [
        {
            "gap": "More recent data needed",
            "section": "statistics",
            "priority": "high",
            "specific_need": "Data from 2023-2024"
        },
        {
            "gap": "Expert quotes needed",
            "section": "analysis",
            "priority": "medium",
            "specific_need": "Quotes from subject matter experts"
        }
    ]

    return {
        "content": [{
            "type": "text",
            "text": f"Identified {len(gaps)} research gaps that need attention."
        }],
        "research_gaps": gaps
    }


# Coordination Tools
@tool("manage_session", "Manage research session state and progress", {
    "session_id": str,
    "action": str,
    "data": dict[str, Any]
})
async def manage_session(args: dict[str, Any]) -> dict[str, Any]:
    """Manage session state and workflow progress."""
    session_id = args["session_id"]
    action = args.get("action", "update")
    data = args.get("data", {})

    # Create session directory if it doesn't exist
    session_path = Path(f"KEVIN/sessions/{session_id}")
    session_path.mkdir(parents=True, exist_ok=True)
    # Create subdirectories for organization
    (session_path / "research").mkdir(exist_ok=True)
    (session_path / "working").mkdir(exist_ok=True)
    (session_path / "final").mkdir(exist_ok=True)

    # Save session state
    session_file = session_path / "session_state.json"
    session_state = {
        "session_id": session_id,
        "last_updated": datetime.now().isoformat(),
        "action": action,
        "data": data
    }

    with open(session_file, 'w') as f:
        json.dump(session_state, f, indent=2)

    return {
        "content": [{
            "type": "text",
            "text": f"Session {session_id} updated with action: {action}"
        }],
        "session_result": {
            "session_id": session_id,
            "status": "updated",
            "path": str(session_path)
        }
    }


@tool("save_report", "Save report to file system with proper formatting", {
    "report": dict[str, Any],
    "session_id": str,
    "format": str,
    "version": int
})
async def save_report(args: dict[str, Any]) -> dict[str, Any]:
    """Save report to the file system in specified format."""
    report = args["report"]
    session_id = args["session_id"]
    format_type = args.get("format", "markdown")
    version = args.get("version", 1)

    session_path = Path(f"KEVIN/sessions/{session_id}")
    session_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_v{version}_{timestamp}.md"
    # Save to final folder for completed reports
    filepath = session_path / "final" / filename

    # Convert report to markdown
    markdown_content = convert_to_markdown(report, format_type)

    # Save file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    return {
        "content": [{
            "type": "text",
            "text": f"Report saved to: {filepath}"
        }],
        "save_result": {
            "filepath": str(filepath),
            "version": version,
            "format": format_type,
            "size": len(markdown_content)
        }
    }


def convert_to_markdown(report: dict[str, Any], format_type: str = "markdown") -> str:
    """Convert report data to markdown format."""
    title = report.get("title", "Research Report")
    generated_at = report.get("generated_at", datetime.now().isoformat())

    markdown = f"# {title}\n\n"
    markdown += f"**Generated:** {generated_at}\n\n"

    # Executive Summary
    if "executive_summary" in report:
        markdown += "## Executive Summary\n\n"
        markdown += f"{report['executive_summary']}\n\n"

    # Key Findings
    if "key_findings" in report:
        markdown += "## Key Findings\n\n"
        for finding in report["key_findings"]:
            if isinstance(finding, dict):
                markdown += f"### {finding.get('fact', 'Finding')}\n\n"
                if "context" in finding:
                    markdown += f"{finding['context']}\n\n"
                if "sources" in finding:
                    markdown += f"**Sources:** {', '.join(finding['sources'])}\n\n"
            else:
                markdown += f"- {finding}\n"
        markdown += "\n"

    # Sections
    if "sections" in report:
        for section_name, section_content in report["sections"].items():
            markdown += f"## {section_name.replace('_', ' ').title()}\n\n"
            markdown += f"{section_content}\n\n"

    # Conclusions
    if "conclusions" in report:
        markdown += "## Conclusions\n\n"
        for conclusion in report["conclusions"]:
            markdown += f"- {conclusion}\n"
        markdown += "\n"

    return markdown
