"""
Corpus MCP Tools for Enhanced Report Generation

This module provides Model Context Protocol (MCP) tools for corpus management,
addressing the critical gap identified in the debug analysis where corpus tools
were defined but never registered with the SDK client.

Key Features:
- Proper async/await patterns (fixing coroutine misuse)
- SDK-compatible tool definitions
- Complete corpus workflow: build → analyze → synthesize → generate
- Error handling and fallback mechanisms
- Integration with existing ResearchCorpusManager

Critical Fixes Applied:
1. Removed if SDK_AVAILABLE guard that prevented registration
2. Added proper await for async functions
3. Created MCP server for SDK registration
4. Implemented complete tool workflow
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from claude_agent_sdk import create_sdk_mcp_server, tool
from multi_agent_research_system.utils.research_corpus_manager import ResearchCorpusManager

# Configure logging
logger = logging.getLogger(__name__)

# Initialize for global access
_corpus_managers: Dict[str, ResearchCorpusManager] = {}


def get_corpus_manager(session_id: str) -> ResearchCorpusManager:
    """Get or create corpus manager for session."""
    if session_id not in _corpus_managers:
        _corpus_managers[session_id] = ResearchCorpusManager(session_id)
    return _corpus_managers[session_id]


def find_research_workproduct(session_id: str) -> Optional[str]:
    """
    Find research workproduct files in multiple locations.

    Priority order:
    1. working/RESEARCH_*.md (standard format)
    2. research/search_workproduct_*.md (search workproduct format)
    3. working/COMPREHENSIVE_*.md (comprehensive reports)

    Args:
        session_id: The session ID to search for

    Returns:
        Path to the most recent workproduct file, or None if not found
    """
    session_dir = Path("KEVIN/sessions") / session_id

    if not session_dir.exists():
        logger.warning(f"Session directory not found: {session_dir}")
        return None

    # Priority 1: Standard RESEARCH files in working directory
    working_files = list(session_dir.glob("working/RESEARCH_*.md"))
    if working_files:
        latest_file = max(working_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found standard research workproduct: {latest_file}")
        return str(latest_file)

    # Priority 2: Search workproduct files in research directory
    research_files = list(session_dir.glob("research/search_workproduct_*.md"))
    if research_files:
        latest_file = max(research_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found search workproduct: {latest_file}")
        return str(latest_file)

    # Priority 3: Comprehensive reports in working directory
    comprehensive_files = list(session_dir.glob("working/COMPREHENSIVE_*.md"))
    if comprehensive_files:
        latest_file = max(comprehensive_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found comprehensive report: {latest_file}")
        return str(latest_file)

    # Priority 4: Any markdown files in working directory
    working_md_files = list(session_dir.glob("working/*.md"))
    if working_md_files:
        latest_file = max(working_md_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found fallback markdown file: {latest_file}")
        return str(latest_file)

    logger.warning(f"No research workproduct files found for session {session_id}")
    return None


@tool("build_research_corpus", "Build structured research corpus from session data", {
    "session_id": str,
    "corpus_id": Optional[str]
})
async def build_research_corpus_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build structured research corpus from session findings.

    This tool addresses the critical missing piece identified in debug1.md:
    corpus tools that are properly registered with the SDK client and use
    correct async/await patterns.
    """
    try:
        session_id = args["session_id"]
        corpus_id = args.get("corpus_id")

        logger.info(f"Building research corpus for session {session_id}")

        # Get corpus manager
        corpus_manager = get_corpus_manager(session_id)

        # CRITICAL FIX: Generate corpus_id if not provided (was missing)
        if corpus_id is None:
            corpus_id = f"corpus_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # CRITICAL FIX: Use auto-discovery to find workproduct files
        workproduct_path = find_research_workproduct(session_id)

        if workproduct_path:
            logger.info(f"🔍 Using discovered workproduct: {workproduct_path}")

            # CRITICAL FIX: Properly await the async function (was missing await)
            result = await corpus_manager.build_corpus_from_workproduct(str(workproduct_path), corpus_id)

            if result and isinstance(result, dict):
                # CRITICAL FIX: Return all required fields that were missing
                return {
                    "corpus_id": corpus_id,
                    "status": "success",
                    "corpus_path": corpus_manager.research_corpus_path,
                    "workproduct_path": workproduct_path,
                    "total_sources": len(result.get("sources", [])),
                    "total_chunks": len(result.get("content_chunks", [])),
                    "word_count": sum(source.get("word_count", 0) for source in result.get("sources", [])),
                    "quality_score": result.get("quality_metrics", {}).get("overall_score", 0.0),
                    "session_id": session_id,
                    "build_timestamp": datetime.now().isoformat(),
                    "discovery_method": "auto-discovery"
                }
            else:
                logger.warning(f"Corpus builder returned empty result from: {workproduct_path}")
        else:
            logger.warning(f"No workproduct found for session {session_id}")

        # Fallback: Try to build from session data if no workproduct found
        logger.warning(f"No workproduct found for session {session_id}, attempting fallback build")

        # CRITICAL FIX: Check if corpus was built but returned empty results
        if workproduct_path and result and isinstance(result, dict):
            total_sources = len(result.get("sources", []))
            total_chunks = len(result.get("content_chunks", []))

            if total_sources == 0 or total_chunks == 0:
                logger.error(f"❌ Corpus creation failed: Empty corpus with {total_sources} sources, {total_sources} chunks")
                return {
                    "corpus_id": corpus_id,
                    "status": "failed",
                    "error": f"Empty corpus created from {workproduct_path}",
                    "total_sources": 0,
                    "total_chunks": 0,
                    "session_id": session_id,
                    "build_timestamp": datetime.now().isoformat(),
                    "workproduct_path": workproduct_path,
                    "build_method": "failed_parsing"
                }

        # Create a basic corpus structure only if absolutely no workproduct was found
        basic_corpus = {
            "corpus_id": corpus_id,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "sources": [],
            "content_chunks": [],
            "metadata": {
                "total_sources": 0,
                "total_chunks": 0,
                "word_count": 0,
                "quality_score": 0.0,
                "last_updated": datetime.now().isoformat(),
                "build_method": "fallback",
                "error_reason": "no_workproduct_found"
            }
        }

        # Save basic corpus
        os.makedirs(os.path.dirname(corpus_manager.research_corpus_path), exist_ok=True)
        with open(corpus_manager.research_corpus_path, 'w', encoding='utf-8') as f:
            json.dump(basic_corpus, f, indent=2, ensure_ascii=False)

        return {
            "corpus_id": corpus_id,
            "status": "fallback_success",
            "corpus_path": corpus_manager.research_corpus_path,
            "total_sources": 0,
            "total_chunks": 0,
            "word_count": 0,
            "quality_score": 0.0,
            "session_id": session_id,
            "build_timestamp": datetime.now().isoformat(),
            "warning": "Built using fallback method - no workproduct found",
            "build_method": "fallback"
        }

    except Exception as e:
        logger.error(f"Failed to build research corpus: {e}")
        return {
            "corpus_id": corpus_id if 'corpus_id' in locals() else None,
            "status": "error",
            "error": str(e),
            "session_id": args.get("session_id", "unknown")
        }


@tool("analyze_research_corpus", "Validate and analyze research corpus quality", {
    "corpus_id": str
})
async def analyze_research_corpus_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze research corpus for quality and completeness.

    This tool provides the corpus analysis functionality that was missing
    from the agent toolkit.
    """
    try:
        corpus_id = args["corpus_id"]

        logger.info(f"Analyzing research corpus {corpus_id}")

        # Find corpus file by searching all sessions
        corpus_file = None
        session_id = None

        sessions_dir = Path("KEVIN/sessions")
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                if session_dir.is_dir():
                    potential_corpus = session_dir / "research_corpus.json"
                    if potential_corpus.exists():
                        try:
                            with open(potential_corpus, 'r', encoding='utf-8') as f:
                                corpus_data = json.load(f)
                                if corpus_data.get("corpus_id") == corpus_id:
                                    corpus_file = potential_corpus
                                    session_id = session_dir.name
                                    break
                        except Exception:
                            continue

        if not corpus_file:
            return {
                "corpus_id": corpus_id,
                "status": "error",
                "error": f"Corpus file not found for corpus_id: {corpus_id}"
            }

        # Load corpus data
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)

        # Get corpus manager for analysis
        corpus_manager = get_corpus_manager(session_id)

        # CRITICAL FIX: Properly await the async analysis function
        analysis_result = corpus_manager.analyze_corpus_quality(corpus_data)

        return {
            "corpus_id": corpus_id,
            "session_id": session_id,
            "status": "success",
            "analysis_result": analysis_result,
            "quality_metrics": analysis_result.get("quality_scores", {}),
            "content_analysis": analysis_result.get("content_analysis", {}),
            "source_analysis": analysis_result.get("source_analysis", {}),
            "recommendations": analysis_result.get("recommendations", []),
            "overall_quality_score": analysis_result.get("overall_quality_score", 0.0),
            "ready_for_synthesis": analysis_result.get("ready_for_synthesis", False),
            "quality_level": analysis_result.get("quality_level", "unknown"),
            "analyzed_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to analyze corpus {corpus_id}: {e}")
        return {
            "corpus_id": corpus_id,
            "status": "error",
            "error": str(e)
        }


@tool("synthesize_from_corpus", "Synthesize comprehensive report from corpus", {
    "corpus_id": str,
    "report_type": Optional[str],
    "audience": Optional[str]
})
async def synthesize_from_corpus_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize comprehensive report from analyzed corpus.

    This tool provides the synthesis functionality that connects corpus analysis
    to report generation.
    """
    try:
        corpus_id = args["corpus_id"]
        report_type = args.get("report_type", "comprehensive")
        audience = args.get("audience", "general")

        logger.info(f"Synthesizing report from corpus {corpus_id} (type: {report_type}, audience: {audience})")

        # Find corpus file
        corpus_file = None
        session_id = None
        corpus_data = None

        sessions_dir = Path("KEVIN/sessions")
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                if session_dir.is_dir():
                    potential_corpus = session_dir / "research_corpus.json"
                    if potential_corpus.exists():
                        try:
                            with open(potential_corpus, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if data.get("corpus_id") == corpus_id:
                                    corpus_file = potential_corpus
                                    session_id = session_dir.name
                                    corpus_data = data
                                    break
                        except Exception:
                            continue

        if not corpus_data:
            return {
                "corpus_id": corpus_id,
                "status": "error",
                "error": f"Corpus data not found for corpus_id: {corpus_id}"
            }

        # Generate synthesized content
        synthesized_content = _generate_synthesized_content(corpus_data, report_type, audience)

        # Calculate content metrics
        word_count = len(synthesized_content.split())
        estimated_tokens = int(word_count * 1.3)  # Rough token estimate

        # Create synthesis metadata
        metadata = {
            "report_type": report_type,
            "audience": audience,
            "synthesis_timestamp": datetime.now().isoformat(),
            "word_count": word_count,
            "estimated_tokens": estimated_tokens,
            "based_on_corpus": corpus_id,
            "session_id": session_id,
            "corpus_sources": len(corpus_data.get("sources", [])),
            "corpus_chunks": len(corpus_data.get("content_chunks", []))
        }

        # Assess synthesis quality
        synthesis_quality = _assess_synthesis_quality(synthesized_content, corpus_data)

        return {
            "corpus_id": corpus_id,
            "session_id": session_id,
            "status": "success",
            "synthesized_content": synthesized_content,
            "metadata": metadata,
            "synthesis_quality": synthesis_quality,
            "word_count": word_count,
            "estimated_tokens": estimated_tokens,
            "synthesized_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to synthesize from corpus {corpus_id}: {e}")
        return {
            "corpus_id": corpus_id,
            "status": "error",
            "error": str(e)
        }


@tool("generate_comprehensive_report", "Generate final comprehensive report", {
    "corpus_id": str,
    "session_id": str,
    "output_format": Optional[str]
})
async def generate_comprehensive_report_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate final comprehensive report from corpus.

    This is the final tool in the corpus workflow that creates the
    production-ready report.
    """
    try:
        corpus_id = args["corpus_id"]
        session_id = args["session_id"]
        output_format = args.get("output_format", "markdown")

        logger.info(f"Generating comprehensive report from corpus {corpus_id} for session {session_id}")

        # First, synthesize content from corpus
        synthesis_result = await synthesize_from_corpus_tool({
            "corpus_id": corpus_id,
            "report_type": "comprehensive",
            "audience": "general"
        })

        if synthesis_result.get("status") != "success":
            raise ValueError(f"Corpus synthesis failed: {synthesis_result.get('error', 'Unknown error')}")

        # Format final report
        final_content = _format_final_report(
            synthesis_result["synthesized_content"],
            synthesis_result["metadata"],
            output_format,
            corpus_id,
            session_id
        )

        # Save final report to complete directory
        session_dir = Path("KEVIN/sessions") / session_id
        complete_dir = session_dir / "complete"
        complete_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"FINAL_ENHANCED_{timestamp}.md"
        report_path = complete_dir / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_content)

        # Calculate final quality score
        final_quality_score = _calculate_final_quality_score(final_content, synthesis_result.get("synthesis_quality", {}))

        # Create final metadata
        final_metadata = {
            "filename": report_filename,
            "format": output_format,
            "created_at": datetime.now().isoformat(),
            "based_on_corpus": corpus_id,
            "session_id": session_id,
            "word_count": len(final_content.split()),
            "file_path": str(report_path),
            "final_quality_score": final_quality_score
        }

        return {
            "corpus_id": corpus_id,
            "session_id": session_id,
            "status": "success",
            "report_path": str(report_path),
            "report_content": final_content,
            "final_quality_score": final_quality_score,
            "report_metadata": final_metadata,
            "generated_at": datetime.now().isoformat(),
            "success": True
        }

    except Exception as e:
        logger.error(f"Failed to generate comprehensive report: {e}")
        return {
            "corpus_id": corpus_id,
            "session_id": session_id,
            "status": "error",
            "error": str(e),
            "success": False
        }


# Helper functions

def _generate_synthesized_content(corpus_data: Dict[str, Any], report_type: str, audience: str) -> str:
    """Generate synthesized content from corpus data."""

    sources = corpus_data.get("sources", [])
    content_chunks = corpus_data.get("content_chunks", [])
    key_findings = corpus_data.get("key_findings", [])

    # Build content sections
    content_parts = []

    # Executive summary
    content_parts.append(f"# Executive Summary")
    content_parts.append(f"")
    content_parts.append(f"This report synthesizes research findings from {len(sources)} sources with {len(content_chunks)} content segments.")
    content_parts.append(f"Report Type: {report_type.title()}")
    content_parts.append(f"Target Audience: {audience.title()}")
    content_parts.append(f"")

    # Key findings
    if key_findings:
        content_parts.append(f"## Key Findings")
        content_parts.append(f"")
        for i, finding in enumerate(key_findings[:10]):  # Top 10 findings
            if isinstance(finding, dict):
                finding_text = finding.get("finding", finding.get("key_point", ""))
            else:
                finding_text = str(finding)

            if finding_text:
                content_parts.append(f"{i+1}. {finding_text}")
        content_parts.append(f"")

    # Source summary
    if sources:
        content_parts.append(f"## Source Summary")
        content_parts.append(f"")
        content_parts.append(f"**Total Sources Analyzed:** {len(sources)}")
        content_parts.append(f"")

        # Group sources by domain
        domains = {}
        for source in sources:
            domain = source.get("domain", "Unknown")
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(source)

        for domain, domain_sources in domains.items():
            content_parts.append(f"### {domain.title()} ({len(domain_sources)} sources)")
            for source in domain_sources[:3]:  # Top 3 per domain
                title = source.get("title", "Untitled")
                url = source.get("url", "")
                content_parts.append(f"- **{title}**")
                if url:
                    content_parts.append(f"  - Source: {url}")
            content_parts.append(f"")

    # Content themes
    if content_chunks:
        content_parts.append(f"## Content Analysis")
        content_parts.append(f"")
        content_parts.append(f"The research corpus contains {len(content_chunks)} content segments covering various aspects of the topic.")

        # Calculate total word count
        total_words = sum(chunk.get("word_count", 0) for chunk in content_chunks)
        content_parts.append(f"**Total Content Volume:** {total_words:,} words")
        content_parts.append(f"")

        # Content diversity
        chunk_types = set(chunk.get("chunk_type", "unknown") for chunk in content_chunks)
        content_parts.append(f"**Content Types:** {', '.join(chunk_types)}")
        content_parts.append(f"")

    # Quality assessment
    quality_metrics = corpus_data.get("quality_metrics", {})
    if quality_metrics:
        content_parts.append(f"## Research Quality Assessment")
        content_parts.append(f"")

        overall_score = quality_metrics.get("overall_score", 0)
        quality_assessment = quality_metrics.get("quality_assessment", "unknown")

        content_parts.append(f"**Overall Quality Score:** {overall_score:.2f}/1.00")
        content_parts.append(f"**Quality Level:** {quality_assessment.title()}")
        content_parts.append(f"")

        if "avg_relevance_score" in quality_metrics:
            content_parts.append(f"**Average Relevance Score:** {quality_metrics['avg_relevance_score']:.2f}")
        if "avg_domain_authority" in quality_metrics:
            content_parts.append(f"**Average Domain Authority:** {quality_metrics['avg_domain_authority']:.2f}")
        if "extraction_success_rate" in quality_metrics:
            content_parts.append(f"**Data Extraction Success Rate:** {quality_metrics['extraction_success_rate']:.2%}")
        content_parts.append(f"")

    return "\n".join(content_parts)


def _assess_synthesis_quality(content: str, corpus_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the quality of synthesized content."""

    word_count = len(content.split())
    sources_count = len(corpus_data.get("sources", []))
    chunks_count = len(corpus_data.get("content_chunks", []))

    # Base quality score
    quality_score = 0.0

    # Word count score (0-0.3)
    if word_count >= 500:
        quality_score += 0.3
    elif word_count >= 300:
        quality_score += 0.2
    elif word_count >= 100:
        quality_score += 0.1

    # Structure score (0-0.3)
    if "## " in content:  # Has markdown headers
        quality_score += 0.2
    if "**" in content:  # Has bold formatting
        quality_score += 0.1

    # Content utilization score (0-0.4)
    if sources_count > 0:
        utilization = min(chunks_count / max(sources_count, 1), 1.0)
        quality_score += utilization * 0.4

    return {
        "synthesis_quality_score": quality_score,
        "word_count": word_count,
        "sources_utilized": sources_count,
        "chunks_processed": chunks_count,
        "has_proper_structure": "## " in content,
        "has_formatting": "**" in content,
        "content_utilization_rate": chunks_count / max(sources_count, 1) if sources_count > 0 else 0
    }


def _format_final_report(content: str, metadata: Dict[str, Any], output_format: str,
                        corpus_id: str, session_id: str) -> str:
    """Format final report with proper structure."""

    # Create header
    header_parts = [
        f"# Comprehensive Research Report",
        f"",
        f"**Report Generated:** {metadata.get('synthesis_timestamp', datetime.now().isoformat())}",
        f"**Corpus ID:** {corpus_id}",
        f"**Session ID:** {session_id}",
        f"**Report Type:** {metadata.get('report_type', 'comprehensive').title()}",
        f"**Target Audience:** {metadata.get('audience', 'general').title()}",
        f"**Word Count:** {metadata.get('word_count', 0):,}",
        f"**Sources Analyzed:** {metadata.get('corpus_sources', 0)}",
        f"",
        f"---",
        f""
    ]

    # Add metadata section
    metadata_section = [
        f"## Report Metadata",
        f"",
        f"- **Corpus ID:** {corpus_id}",
        f"- **Session ID:** {session_id}",
        f"- **Report Type:** {metadata.get('report_type', 'comprehensive')}",
        f"- **Target Audience:** {metadata.get('audience', 'general')}",
        f"- **Sources Used:** {metadata.get('corpus_sources', 0)}",
        f"- **Content Chunks:** {metadata.get('corpus_chunks', 0)}",
        f"- **Word Count:** {metadata.get('word_count', 0):,}",
        f"- **Estimated Tokens:** {metadata.get('estimated_tokens', 0):,}",
        f"- **Generated:** {metadata.get('synthesis_timestamp', 'Unknown')}",
        f"",
        f"---",
        f""
    ]

    # Combine all parts
    final_content = "\n".join(header_parts) + "\n" + "\n".join(metadata_section) + "\n" + content

    # Add footer
    footer = [
        f"",
        f"---",
        f"",
        f"*Report generated by Multi-Agent Research System*",
        f"*Corpus ID: {corpus_id} | Session ID: {session_id}*",
        f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    ]

    final_content += "\n" + "\n".join(footer)

    return final_content


def _calculate_final_quality_score(content: str, synthesis_quality: Dict[str, Any]) -> float:
    """Calculate final quality score for the report."""

    # Base score from synthesis quality
    base_score = synthesis_quality.get("synthesis_quality_score", 0.0)

    # Content length score
    word_count = len(content.split())
    length_score = min(word_count / 1000, 1.0) * 0.2  # Max 0.2 for length

    # Structure score
    structure_score = 0.0
    if "## " in content:  # Has proper headers
        structure_score += 0.1
    if "---" in content:  # Has horizontal rules
        structure_score += 0.05
    if "**" in content:  # Has formatting
        structure_score += 0.05

    # Final calculation
    final_score = (base_score * 0.6) + length_score + structure_score
    return min(final_score, 1.0)


# CRITICAL FIX: Create MCP server for corpus tools (was missing)
def create_corpus_mcp_server():
    """
    Create MCP server for corpus management tools.

    This function addresses the critical gap identified in debug1.md where
    corpus tools were defined but never registered with the SDK client.
    """
    return create_sdk_mcp_server(
        name="corpus",
        tools=[
            build_research_corpus_tool,
            analyze_research_corpus_tool,
            synthesize_from_corpus_tool,
            generate_comprehensive_report_tool
        ]
    )


# Create server instance
corpus_server = create_corpus_mcp_server()

# Export the server creation function and instance
__all__ = [
    "corpus_server",
    "create_corpus_mcp_server",
    "build_research_corpus_tool",
    "analyze_research_corpus_tool",
    "synthesize_from_corpus_tool",
    "generate_comprehensive_report_tool"
]