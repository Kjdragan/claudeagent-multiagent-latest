"""
Critique-Specific MCP Tools for Editorial Agent

This module provides specialized tools for editorial review and critique generation,
following Claude Agent SDK best practices for tool definition and MCP server creation.

Tools:
- review_report: Analyze report structure and quality
- analyze_content_quality: Assess specific quality dimensions
- identify_research_gaps: Detect missing information
- generate_critique: Create structured critique output
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from claude_agent_sdk import tool, create_sdk_mcp_server

logger = logging.getLogger(__name__)


# ============================================================================
# Tool 1: Review Report
# ============================================================================

@tool(
    name="review_report",
    description="Analyze a research report's structure, completeness, and quality. Returns structured assessment of report components.",
    input_schema={
        "type": "object",
        "properties": {
            "report_path": {
                "type": "string",
                "description": "Path to the report file to review"
            },
            "session_id": {
                "type": "string", 
                "description": "Session ID for context"
            }
        },
        "required": ["report_path", "session_id"]
    }
)
async def review_report_tool(args: dict[str, Any]) -> dict[str, Any]:
    """
    Analyzes a report file and returns structured assessment.
    
    Returns:
        dict with structure: {
            "has_executive_summary": bool,
            "has_key_findings": bool,
            "has_sources": bool,
            "section_count": int,
            "word_count": int,
            "source_count": int,
            "structure_score": float  # 0-1
        }
    """
    try:
        report_path = Path(args["report_path"])
        session_id = args["session_id"]
        
        if not report_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Error: Report file not found at {report_path}"
                }],
                "is_error": True
            }
        
        # Read report content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Analyze structure
        analysis = {
            "has_executive_summary": "executive summary" in content.lower(),
            "has_key_findings": "key findings" in content.lower() or "findings" in content.lower(),
            "has_sources": "sources" in content.lower() or "references" in content.lower(),
            "section_count": content.count('\n## ') + content.count('\n# '),
            "word_count": len(content.split()),
            "source_count": content.count('http://') + content.count('https://'),
            "has_conclusion": "conclusion" in content.lower(),
            "has_introduction": "introduction" in content.lower()
        }
        
        # Calculate structure score
        structure_elements = [
            analysis["has_executive_summary"],
            analysis["has_key_findings"],
            analysis["has_sources"],
            analysis["has_conclusion"],
            analysis["section_count"] >= 3
        ]
        analysis["structure_score"] = sum(structure_elements) / len(structure_elements)
        
        result_text = f"""## Report Structure Analysis

**Session**: {session_id}
**Report**: {report_path.name}

### Structure Assessment
- **Executive Summary**: {"✅ Present" if analysis["has_executive_summary"] else "❌ Missing"}
- **Key Findings**: {"✅ Present" if analysis["has_key_findings"] else "❌ Missing"}
- **Sources**: {"✅ Present" if analysis["has_sources"] else "❌ Missing"}
- **Conclusion**: {"✅ Present" if analysis["has_conclusion"] else "❌ Missing"}

### Metrics
- **Sections**: {analysis["section_count"]}
- **Word Count**: {analysis["word_count"]:,}
- **Source Count**: {analysis["source_count"]}
- **Structure Score**: {analysis["structure_score"]:.2f}/1.00

### Quality Indicators
- Structure completeness: {analysis["structure_score"] * 100:.0f}%
- Content length: {"Adequate" if analysis["word_count"] > 500 else "Too short"}
- Source citations: {"Good" if analysis["source_count"] >= 5 else "Insufficient"}
"""
        
        return {
            "content": [{
                "type": "text",
                "text": result_text
            }],
            "metadata": analysis
        }
        
    except Exception as e:
        logger.error(f"Error in review_report: {e}", exc_info=True)
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Error reviewing report: {str(e)}"
            }],
            "is_error": True
        }


# ============================================================================
# Tool 2: Analyze Content Quality
# ============================================================================

@tool(
    name="analyze_content_quality",
    description="Assess specific quality dimensions of report content including clarity, depth, accuracy indicators, and coherence.",
    input_schema={
        "type": "object",
        "properties": {
            "report_path": {
                "type": "string",
                "description": "Path to the report file to analyze"
            },
            "dimensions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Quality dimensions to assess: clarity, depth, accuracy, coherence, sourcing",
                "default": ["clarity", "depth", "accuracy", "coherence", "sourcing"]
            }
        },
        "required": ["report_path"]
    }
)
async def analyze_content_quality_tool(args: dict[str, Any]) -> dict[str, Any]:
    """
    Analyzes content quality across multiple dimensions.
    
    Returns quality scores and specific observations for each dimension.
    """
    try:
        report_path = Path(args["report_path"])
        dimensions = args.get("dimensions", ["clarity", "depth", "accuracy", "coherence", "sourcing"])
        
        if not report_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Error: Report file not found at {report_path}"
                }],
                "is_error": True
            }
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Quality assessment heuristics
        scores = {}
        observations = {}
        
        # Clarity assessment
        if "clarity" in dimensions:
            avg_sentence_length = len(content.split()) / max(content.count('.'), 1)
            has_headings = content.count('\n#') > 0
            has_bullets = content.count('\n-') > 0
            
            clarity_score = 0.0
            if avg_sentence_length < 30:
                clarity_score += 0.4
            if has_headings:
                clarity_score += 0.3
            if has_bullets:
                clarity_score += 0.3
            
            scores["clarity"] = min(clarity_score, 1.0)
            observations["clarity"] = [
                f"Average sentence length: {avg_sentence_length:.1f} words",
                "Clear structure with headings" if has_headings else "Missing structural headings",
                "Uses bullet points for clarity" if has_bullets else "Could benefit from bullet points"
            ]
        
        # Depth assessment
        if "depth" in dimensions:
            word_count = len(content.split())
            section_count = content.count('\n##')
            has_analysis = "analysis" in content.lower()
            has_implications = "implications" in content.lower() or "impact" in content.lower()
            
            depth_score = 0.0
            if word_count > 1000:
                depth_score += 0.3
            if section_count >= 5:
                depth_score += 0.3
            if has_analysis:
                depth_score += 0.2
            if has_implications:
                depth_score += 0.2
            
            scores["depth"] = min(depth_score, 1.0)
            observations["depth"] = [
                f"Content length: {word_count:,} words",
                f"Sections: {section_count}",
                "Includes analysis section" if has_analysis else "Missing explicit analysis",
                "Discusses implications" if has_implications else "Could explore implications further"
            ]
        
        # Accuracy indicators
        if "accuracy" in dimensions:
            source_count = content.count('http://') + content.count('https://')
            has_dates = content.count('202') > 0  # Recent dates
            has_specifics = content.count('%') > 0 or bool(any(char.isdigit() for char in content))
            
            accuracy_score = 0.0
            if source_count >= 5:
                accuracy_score += 0.4
            if has_dates:
                accuracy_score += 0.3
            if has_specifics:
                accuracy_score += 0.3
            
            scores["accuracy"] = min(accuracy_score, 1.0)
            observations["accuracy"] = [
                f"Sources cited: {source_count}",
                "Includes recent dates" if has_dates else "Missing temporal context",
                "Contains specific data/statistics" if has_specifics else "Could use more specific data"
            ]
        
        # Coherence assessment
        if "coherence" in dimensions:
            has_intro = "introduction" in content.lower()
            has_conclusion = "conclusion" in content.lower()
            has_transitions = any(word in content.lower() for word in ["however", "therefore", "additionally", "furthermore"])
            
            coherence_score = 0.0
            if has_intro:
                coherence_score += 0.3
            if has_conclusion:
                coherence_score += 0.3
            if has_transitions:
                coherence_score += 0.4
            
            scores["coherence"] = min(coherence_score, 1.0)
            observations["coherence"] = [
                "Has introduction" if has_intro else "Missing introduction",
                "Has conclusion" if has_conclusion else "Missing conclusion",
                "Uses transition words" if has_transitions else "Could improve flow with transitions"
            ]
        
        # Sourcing quality
        if "sourcing" in dimensions:
            source_count = content.count('http://') + content.count('https://')
            diverse_domains = len(set([
                line.split('//')[1].split('/')[0] 
                for line in content.split('\n') 
                if 'http' in line and '//' in line
            ]))
            
            sourcing_score = 0.0
            if source_count >= 5:
                sourcing_score += 0.5
            if diverse_domains >= 3:
                sourcing_score += 0.5
            
            scores["sourcing"] = min(sourcing_score, 1.0)
            observations["sourcing"] = [
                f"Total sources: {source_count}",
                f"Unique domains: {diverse_domains}",
                "Good source diversity" if diverse_domains >= 3 else "Limited source diversity"
            ]
        
        # Generate report
        result_lines = ["## Content Quality Analysis\n"]
        for dimension in dimensions:
            score = scores.get(dimension, 0.0)
            obs = observations.get(dimension, [])
            result_lines.append(f"### {dimension.capitalize()}: {score:.2f}/1.00")
            for observation in obs:
                result_lines.append(f"- {observation}")
            result_lines.append("")
        
        # Overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        result_lines.append(f"### Overall Quality Score: {overall_score:.2f}/1.00")
        
        return {
            "content": [{
                "type": "text",
                "text": "\n".join(result_lines)
            }],
            "metadata": {
                "scores": scores,
                "overall_score": overall_score,
                "observations": observations
            }
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_content_quality: {e}", exc_info=True)
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Error analyzing content quality: {str(e)}"
            }],
            "is_error": True
        }


# ============================================================================
# Tool 3: Identify Research Gaps
# ============================================================================

@tool(
    name="identify_research_gaps",
    description="Detect missing information, unexplored angles, and research gaps in a report based on topic analysis.",
    input_schema={
        "type": "object",
        "properties": {
            "report_path": {
                "type": "string",
                "description": "Path to the report file to analyze"
            },
            "topic": {
                "type": "string",
                "description": "The main topic of the research for context"
            }
        },
        "required": ["report_path", "topic"]
    }
)
async def identify_research_gaps_tool(args: dict[str, Any]) -> dict[str, Any]:
    """
    Identifies information gaps and missing perspectives in a report.
    
    Returns structured list of gaps with priority levels.
    """
    try:
        report_path = Path(args["report_path"])
        topic = args["topic"]
        
        if not report_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Error: Report file not found at {report_path}"
                }],
                "is_error": True
            }
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()
        
        # Define common gap patterns based on topic
        gaps = []
        
        # Temporal gaps
        if "recent" not in content and "2025" not in content and "2024" not in content:
            gaps.append({
                "type": "temporal",
                "priority": "high",
                "gap": "Missing recent developments or current status",
                "suggestion": f"Add recent news or updates about {topic} from 2024-2025"
            })
        
        # Statistical/data gaps
        if "%" not in content and not any(char.isdigit() for char in content):
            gaps.append({
                "type": "statistical",
                "priority": "high",
                "gap": "No quantitative data or statistics provided",
                "suggestion": f"Include specific numbers, percentages, or metrics related to {topic}"
            })
        
        # Source diversity gaps
        source_count = content.count('http://') + content.count('https://')
        if source_count < 5:
            gaps.append({
                "type": "sourcing",
                "priority": "medium",
                "gap": f"Insufficient source citations (only {source_count} found)",
                "suggestion": "Add more diverse sources to support claims"
            })
        
        # Perspective gaps
        if "impact" not in content and "effect" not in content:
            gaps.append({
                "type": "perspective",
                "priority": "medium",
                "gap": "Missing impact or consequences analysis",
                "suggestion": f"Explore the impacts and implications of {topic}"
            })
        
        # Context gaps
        if "background" not in content and "history" not in content:
            gaps.append({
                "type": "context",
                "priority": "low",
                "gap": "Limited historical context or background",
                "suggestion": f"Provide background information to contextualize {topic}"
            })
        
        # Expert opinion gaps
        if "expert" not in content and "analyst" not in content and "researcher" not in content:
            gaps.append({
                "type": "expert_opinion",
                "priority": "medium",
                "gap": "No expert opinions or analysis cited",
                "suggestion": "Include perspectives from experts or analysts in the field"
            })
        
        # Generate report
        result_lines = ["## Research Gaps Identified\n"]
        result_lines.append(f"**Topic**: {topic}")
        result_lines.append(f"**Gaps Found**: {len(gaps)}\n")
        
        # Group by priority
        for priority in ["high", "medium", "low"]:
            priority_gaps = [g for g in gaps if g["priority"] == priority]
            if priority_gaps:
                result_lines.append(f"### {priority.upper()} Priority Gaps\n")
                for i, gap in enumerate(priority_gaps, 1):
                    result_lines.append(f"**Gap {i}**: {gap['gap']}")
                    result_lines.append(f"- **Type**: {gap['type']}")
                    result_lines.append(f"- **Recommendation**: {gap['suggestion']}")
                    result_lines.append("")
        
        if not gaps:
            result_lines.append("✅ No major gaps identified. Report appears comprehensive.")
        
        return {
            "content": [{
                "type": "text",
                "text": "\n".join(result_lines)
            }],
            "metadata": {
                "gaps": gaps,
                "gap_count": len(gaps),
                "high_priority_count": len([g for g in gaps if g["priority"] == "high"])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in identify_research_gaps: {e}", exc_info=True)
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Error identifying research gaps: {str(e)}"
            }],
            "is_error": True
        }


# ============================================================================
# Tool 4: Generate Critique
# ============================================================================

@tool(
    name="generate_critique",
    description="Generate a structured editorial critique with assessment, issues, gaps, and recommendations.",
    input_schema={
        "type": "object",
        "properties": {
            "report_path": {
                "type": "string",
                "description": "Path to the report file being critiqued"
            },
            "structure_analysis": {
                "type": "object",
                "description": "Results from review_report tool"
            },
            "quality_analysis": {
                "type": "object",
                "description": "Results from analyze_content_quality tool"
            },
            "gaps_analysis": {
                "type": "object",
                "description": "Results from identify_research_gaps tool"
            },
            "session_id": {
                "type": "string",
                "description": "Session ID for context"
            }
        },
        "required": ["report_path", "session_id"]
    }
)
async def generate_critique_tool(args: dict[str, Any]) -> dict[str, Any]:
    """
    Generates a comprehensive structured critique combining all analysis results.
    
    This is the final tool that compiles all editorial analysis into a critique document.
    """
    try:
        report_path = Path(args["report_path"])
        session_id = args["session_id"]
        structure_data = args.get("structure_analysis", {})
        quality_data = args.get("quality_analysis", {})
        gaps_data = args.get("gaps_analysis", {})
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build critique document
        critique_lines = [
            "# Editorial Critique",
            f"**Session ID**: {session_id}",
            f"**Report**: {report_path.name}",
            f"**Critique Date**: {datetime.now().strftime('%B %d, %Y')}",
            f"**Analysis Timestamp**: {timestamp}\n",
            "---\n",
            "## Quality Assessment\n"
        ]
        
        # Add structure assessment
        if structure_data:
            structure_score = structure_data.get("structure_score", 0)
            critique_lines.append(f"### Structure: {structure_score:.2f}/1.00")
            critique_lines.append(f"- Sections: {structure_data.get('section_count', 0)}")
            critique_lines.append(f"- Word count: {structure_data.get('word_count', 0):,}")
            critique_lines.append(f"- Sources: {structure_data.get('source_count', 0)}\n")
        
        # Add quality scores
        if quality_data:
            scores = quality_data.get("scores", {})
            overall = quality_data.get("overall_score", 0)
            critique_lines.append(f"### Overall Quality: {overall:.2f}/1.00\n")
            for dimension, score in scores.items():
                critique_lines.append(f"- **{dimension.capitalize()}**: {score:.2f}/1.00")
            critique_lines.append("")
        
        # Add identified issues
        critique_lines.append("## Identified Issues\n")
        issue_count = 0
        
        # Issues from structure
        if structure_data:
            if not structure_data.get("has_executive_summary"):
                issue_count += 1
                critique_lines.append(f"{issue_count}. **Missing Executive Summary**: Report lacks a concise executive summary at the beginning.")
            if not structure_data.get("has_conclusion"):
                issue_count += 1
                critique_lines.append(f"{issue_count}. **Missing Conclusion**: Report needs a conclusion section to synthesize findings.")
            if structure_data.get("source_count", 0) < 5:
                issue_count += 1
                critique_lines.append(f"{issue_count}. **Insufficient Citations**: Only {structure_data.get('source_count', 0)} sources cited. Recommend minimum 5 sources.")
        
        # Issues from quality analysis
        if quality_data:
            obs = quality_data.get("observations", {})
            for dimension, observations in obs.items():
                for observation in observations:
                    if "missing" in observation.lower() or "could" in observation.lower():
                        issue_count += 1
                        critique_lines.append(f"{issue_count}. **{dimension.capitalize()} Issue**: {observation}")
        
        if issue_count == 0:
            critique_lines.append("No major structural or quality issues identified.\n")
        else:
            critique_lines.append("")
        
        # Add information gaps
        critique_lines.append("## Information Gaps\n")
        if gaps_data and gaps_data.get("gaps"):
            gaps = gaps_data["gaps"]
            high_priority = [g for g in gaps if g["priority"] == "high"]
            medium_priority = [g for g in gaps if g["priority"] == "medium"]
            
            if high_priority:
                critique_lines.append("### HIGH PRIORITY\n")
                for gap in high_priority:
                    critique_lines.append(f"**{gap['type'].replace('_', ' ').title()}**: {gap['gap']}")
                    critique_lines.append(f"- Recommendation: {gap['suggestion']}\n")
            
            if medium_priority:
                critique_lines.append("### MEDIUM PRIORITY\n")
                for gap in medium_priority:
                    critique_lines.append(f"**{gap['type'].replace('_', ' ').title()}**: {gap['gap']}")
                    critique_lines.append(f"- Recommendation: {gap['suggestion']}\n")
        else:
            critique_lines.append("No significant information gaps identified.\n")
        
        # Add recommendations
        critique_lines.append("## Recommendations\n")
        critique_lines.append("### Immediate Actions\n")
        
        rec_count = 0
        if structure_data and structure_data.get("structure_score", 0) < 0.7:
            rec_count += 1
            critique_lines.append(f"{rec_count}. Improve document structure by adding missing sections (executive summary, conclusion)")
        
        if quality_data and quality_data.get("overall_score", 0) < 0.6:
            rec_count += 1
            critique_lines.append(f"{rec_count}. Enhance content quality focusing on low-scoring dimensions")
        
        if gaps_data and gaps_data.get("high_priority_count", 0) > 0:
            rec_count += 1
            critique_lines.append(f"{rec_count}. Conduct gap research to address {gaps_data['high_priority_count']} high-priority information gaps")
        
        if rec_count == 0:
            critique_lines.append("Report meets quality standards. Minor refinements optional.")
        
        critique_lines.append("\n### Enhancement Opportunities\n")
        critique_lines.append("- Consider adding visual elements (charts, tables) if applicable")
        critique_lines.append("- Enhance readability with better section transitions")
        critique_lines.append("- Strengthen conclusions with actionable insights\n")
        
        critique_lines.append("---\n")
        critique_lines.append("*This critique was generated using automated analysis tools. ")
        critique_lines.append("Human editorial review recommended for final enhancement decisions.*")
        
        critique_text = "\n".join(critique_lines)
        
        return {
            "content": [{
                "type": "text",
                "text": critique_text
            }],
            "metadata": {
                "critique_generated": True,
                "timestamp": timestamp,
                "issue_count": issue_count,
                "gap_count": gaps_data.get("gap_count", 0) if gaps_data else 0,
                "overall_quality_score": quality_data.get("overall_score", 0) if quality_data else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in generate_critique: {e}", exc_info=True)
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Error generating critique: {str(e)}"
            }],
            "is_error": True
        }


# ============================================================================
# MCP Server Creation
# ============================================================================

def create_critique_mcp_server():
    """
    Create MCP server for critique tools following Claude Agent SDK patterns.
    
    This server provides specialized editorial review and critique tools
    that replace report generation tools in the editorial agent.
    """
    return create_sdk_mcp_server(
        name="critique",
        version="1.0.0",
        tools=[
            review_report_tool,
            analyze_content_quality_tool,
            identify_research_gaps_tool,
            generate_critique_tool
        ]
    )


# Create the server instance
try:
    critique_server = create_critique_mcp_server()
    logger.info("✅ Critique MCP server created successfully")
except Exception as e:
    logger.error(f"❌ Failed to create critique MCP server: {e}")
    critique_server = None
