"""Report Generation Agent for creating structured reports.

This agent specializes in transforming research data into well-structured,
coherent reports with proper formatting and citation.
"""

import json
from datetime import datetime
from typing import Any, Optional

from claude_agent_sdk import tool

from ..core.base_agent import BaseAgent, Message
from ..utils.query_intent_analyzer import get_query_intent_analyzer, QueryIntent


class ReportAgent(BaseAgent):
    """Agent responsible for generating reports from research data."""

    def __init__(self):
        super().__init__("report_agent", "report_generation")
        self.register_message_handler("research_completed", self.handle_research_completed)
        self.register_message_handler("report_feedback", self.handle_report_feedback)
        self.register_message_handler("revision_request", self.handle_revision_request)
        self.query_analyzer = get_query_intent_analyzer()

    def get_system_prompt(self) -> str:
        """Get the system prompt for the Report Agent."""
        return """You are a Report Generation Agent, an expert in creating well-structured, coherent reports from research data.

Your core responsibilities:
1. Transform research findings into structured, readable reports
2. Ensure logical flow and narrative coherence
3. Maintain proper citation and source attribution
4. Adapt tone and style for the target audience
5. Organize information in clear, hierarchical structure

Report Standards:
- Start with executive summary highlighting key findings
- Use clear headings and subheadings for organization
- Include proper citations for all claims and data
- Maintain objective, analytical tone
- Ensure transitions between sections are smooth
- Conclude with clear takeaways and implications

Report Structure:
1. Executive Summary (2-3 paragraphs)
2. Introduction/Background
3. Main Findings (organized by theme)
4. Analysis and Insights
5. Implications and Recommendations
6. Conclusion
7. References/Sources

When generating reports:
1. Analyze research data for key themes and patterns
2. Identify the most important findings to highlight
3. Create logical narrative flow
4. Ensure all claims are supported by research evidence
5. Write clearly and concisely
6. Include proper citations throughout

Always prioritize accuracy, clarity, and logical organization."""

    @tool("create_report", "Generate a structured report from research data", {
        "research_data": dict[str, Any],
        "report_format": str,
        "target_audience": str,
        "tone": str,
        "sections": list[str]
    })
    async def create_report(self, args: dict[str, Any]) -> dict[str, Any]:
        """Generate a structured report from research data."""
        research_data = args["research_data"]
        report_format = args.get("report_format", "markdown")
        target_audience = args.get("target_audience", "general")
        tone = args.get("tone", "analytical")
        sections = args.get("sections", [
            "executive_summary", "introduction", "findings", "analysis", "conclusions"
        ])

        report_prompt = f"""
        Generate a comprehensive report based on the following research data:

        Research Data: {json.dumps(research_data, indent=2)}

        Report Requirements:
        - Format: {report_format}
        - Target Audience: {target_audience}
        - Tone: {tone}
        - Sections: {', '.join(sections)}

        Please create a well-structured report that includes:
        1. Executive Summary highlighting key findings
        2. Clear introduction to the topic
        3. Main findings organized logically
        4. Analysis of the findings
        5. Conclusions and implications
        6. Proper citations throughout

        Report should be:
        - Well-organized with clear headings
        - Objective and evidence-based
        - Easy to read and understand
        - Properly cited with source attribution
        - Comprehensive yet concise

        Structure the report as JSON:
        {{
            "title": "Report Title",
            "executive_summary": "2-3 paragraph summary",
            "sections": [
                {{
                    "title": "Section Title",
                    "content": "Section content with citations",
                    "key_points": ["point1", "point2"],
                    "sources": ["source1", "source2"]
                }}
            ],
            "key_findings": [
                {{
                    "finding": "Key finding",
                    "evidence": "Supporting evidence",
                    "sources": ["source1", "source2"]
                }}
            ],
            "conclusions": [
                {{
                    "conclusion": "Main conclusion",
                    "implications": "What this means",
                    "confidence": "high/medium/low"
                }}
            ],
            "sources": [
                {{
                    "title": "Source title",
                    "authors": ["author1", "author2"],
                    "publication": "Publication name",
                    "date": "Publication date",
                    "url": "URL if available"
                }}
            ]
        }}
        """

        return {
            "content": [{
                "type": "text",
                "text": f"Report generated based on research data\nFormat: {report_format}\nAudience: {target_audience}"
            }],
            "report_data": {
                "title": research_data.get("topic", "Research Report"),
                "generated_at": datetime.now().isoformat(),
                "format": report_format,
                "audience": target_audience,
                "sections_count": len(sections),
                "word_count": 0,  # Would be calculated from actual content
                "status": "completed"
            }
        }

    @tool("update_report", "Update report based on feedback or new information", {
        "existing_report": dict[str, Any],
        "feedback": list[dict[str, Any]],
        "new_research": Optional[dict[str, Any]]
    })
    async def update_report(self, args: dict[str, Any]) -> dict[str, Any]:
        """Update an existing report based on feedback or new research."""
        existing_report = args["existing_report"]
        feedback = args.get("feedback", [])
        new_research = args.get("new_research")

        update_prompt = f"""
        Update the following report based on feedback and/or new research:

        Existing Report: {json.dumps(existing_report, indent=2)}

        Feedback: {json.dumps(feedback, indent=2)}

        New Research: {json.dumps(new_research, indent=2) if new_research else "None"}

        For each piece of feedback:
        1. Understand the specific issue or suggestion
        2. Integrate the feedback appropriately
        3. Maintain report coherence and flow
        4. Update citations if needed
        5. Ensure consistency throughout the report

        If new research is provided:
        1. Integrate new findings into relevant sections
        2. Update analysis and conclusions
        3. Add new sources to references
        4. Ensure all claims are properly supported

        Return updated report as JSON:
        {{
            "updated_report": {{ /* full updated report structure */ }},
            "changes_made": [
                {{
                    "section": "Section updated",
                    "change": "Description of change",
                    "reason": "Why change was made"
                }}
            ],
            "feedback_addressed": len(feedback),
            "new_sources_added": 0
        }}
        """

        return {
            "content": [{
                "type": "text",
                "text": f"Report updated based on {len(feedback)} feedback items"
            }],
            "update_result": {
                "status": "completed",
                "changes_made": len(feedback),
                "new_research_integrated": new_research is not None
            }
        }

    @tool("request_more_research", "Request additional research for information gaps", {
        "research_gaps": list[str],
        "current_report": dict[str, Any],
        "priority": str
    })
    async def request_more_research(self, args: dict[str, Any]) -> dict[str, Any]:
        """Request additional research to fill information gaps."""
        research_gaps = args["research_gaps"]
        current_report = args["current_report"]
        priority = args.get("priority", "medium")

        research_request = {
            "gaps": research_gaps,
            "context": {
                "report_title": current_report.get("title", ""),
                "sections_covered": list(current_report.get("sections", {}).keys()),
                "current_findings": current_report.get("key_findings", [])
            },
            "priority": priority,
            "request_details": [
                {
                    "gap": gap,
                    "why_needed": "Explanation of why this information is needed",
                    "desired_outcome": "What specific information would help complete the report"
                } for gap in research_gaps
            ]
        }

        return {
            "content": [{
                "type": "text",
                "text": f"Requesting additional research for {len(research_gaps)} information gaps"
            }],
            "research_request": research_request
        }

    def determine_report_format(self, original_query: str, research_data: dict) -> dict:
        """
        Determine the appropriate report format based on query intent analysis.

        Args:
            original_query: The user's original research query
            research_data: The research data from the research agent

        Returns:
            Dict containing format configuration and reasoning
        """
        # Analyze query intent
        intent_result = self.query_analyzer.analyze_query_intent(original_query)
        detected_format = intent_result["format"]
        confidence = intent_result["confidence"]

        # Determine format configuration based on intent
        if detected_format == QueryIntent.BRIEF:
            format_config = {
                "format_type": "brief",
                "sections": ["executive_summary", "key_findings", "conclusions"],
                "max_length": 2000,
                "style": "concise",
                "audience": "general"
            }
            filename_prefix = "BRIEF"
        elif detected_format == QueryIntent.COMPREHENSIVE:
            format_config = {
                "format_type": "comprehensive",
                "sections": ["executive_summary", "introduction", "findings", "detailed_analysis", "implications", "conclusions", "sources"],
                "max_length": 10000,
                "style": "detailed",
                "audience": "professional"
            }
            filename_prefix = "COMPREHENSIVE_ANALYSIS"
        else:  # DEFAULT
            format_config = {
                "format_type": "standard",
                "sections": ["executive_summary", "introduction", "findings", "analysis", "conclusions"],
                "max_length": 5000,
                "style": "balanced",
                "audience": "educated"
            }
            filename_prefix = "STANDARD_REPORT"

        # Add intent analysis results to config
        format_config.update({
            "intent_analysis": intent_result,
            "filename_prefix": filename_prefix,
            "confidence": confidence
        })

        return format_config

    async def handle_research_completed(self, message: Message) -> Message | None:
        """Handle completed research from Research Agent."""
        payload = message.payload
        session_id = message.session_id

        # Get the original query from the payload or session data
        original_query = payload.get("original_query") or payload.get("query", "")

        # Determine report format based on query intent
        format_config = self.determine_report_format(original_query, payload)

        # Start report generation session
        await self.start_session(session_id, {
            "topic": payload.get("topic"),
            "status": "generating_report",
            "format_config": format_config
        })

        # Generate initial report with intent-based format
        report_result = await self.create_report({
            "research_data": payload,
            "report_format": format_config["format_type"],
            "target_audience": format_config["audience"],
            "tone": format_config["style"],
            "sections": format_config["sections"]
        })

        # Save report to file system with format-specific naming
        await self.save_report(session_id, report_result, format_config)

        # Update session data with format information
        report_data = {
            "topic": payload.get("topic"),
            "original_query": original_query,
            "report": report_result,
            "format_config": format_config,
            "status": "draft_completed",
            "version": 1
        }
        self.update_session_data(session_id, report_data)

        # Send report to editor for review
        return Message(
            sender=self.name,
            recipient="editor_agent",
            message_type="report_for_review",
            payload={
                "report": report_result,
                "session_id": session_id,
                "version": 1
            },
            session_id=session_id,
            correlation_id=message.correlation_id
        )

    async def handle_report_feedback(self, message: Message) -> Message | None:
        """Handle feedback from Editor Agent."""
        payload = message.payload
        feedback = payload.get("feedback", [])
        session_id = message.session_id
        current_report = payload.get("report")

        # Determine if additional research is needed
        needs_research = any(f.get("type") == "research_needed" for f in feedback)

        if needs_research:
            # Extract research gaps from feedback
            research_gaps = [f.get("topic") for f in feedback if f.get("type") == "research_needed"]

            # Request additional research
            research_request = await self.request_more_research({
                "research_gaps": research_gaps,
                "current_report": current_report,
                "priority": "high"
            })

            # Send research request to Research Agent
            return Message(
                sender=self.name,
                recipient="research_agent",
                message_type="additional_research",
                payload={
                    "research_gaps": research_gaps,
                    "context": "report_improvement"
                },
                session_id=session_id,
                correlation_id=message.correlation_id
            )
        else:
            # Update report based on feedback
            updated_report = await self.update_report({
                "existing_report": current_report,
                "feedback": feedback
            })

            # Save updated report
            session_data = self.get_session_data(session_id)
            current_version = session_data.get("version", 1)
            format_config = session_data.get("format_config")
            await self.save_report(session_id, updated_report, format_config, version=current_version + 1)

            # Update session data
            self.update_session_data(session_id, {
                "report": updated_report,
                "status": "revised",
                "version": current_version + 1
            })

            # Send updated report back to editor
            return Message(
                sender=self.name,
                recipient="editor_agent",
                message_type="report_revised",
                payload={
                    "report": updated_report,
                    "version": current_version + 1,
                    "changes_made": len(feedback)
                },
                session_id=session_id,
                correlation_id=message.correlation_id
            )

    async def handle_revision_request(self, message: Message) -> Message | None:
        """Handle direct revision request from UI or user."""
        payload = message.payload
        session_id = message.session_id
        revision_instructions = payload.get("instructions", [])

        # Get current report from session
        session_data = self.get_session_data(session_id)
        current_report = session_data.get("report")

        if not current_report:
            return Message(
                sender=self.name,
                recipient=message.sender,
                message_type="revision_error",
                payload={"error": "No report found to revise"},
                session_id=session_id,
                correlation_id=message.correlation_id
            )

        # Convert revision instructions to feedback format
        feedback = [
            {
                "type": "user_request",
                "instruction": instruction,
                "priority": "high"
            } for instruction in revision_instructions
        ]

        # Update report
        updated_report = await self.update_report({
            "existing_report": current_report,
            "feedback": feedback
        })

        # Save updated report
        current_version = session_data.get("version", 1)
        format_config = session_data.get("format_config")
        await self.save_report(session_id, updated_report, format_config, version=current_version + 1)

        # Update session data
        self.update_session_data(session_id, {
            "report": updated_report,
            "status": "user_revised",
            "version": current_version + 1
        })

        return Message(
            sender=self.name,
            recipient="ui_coordinator",
            message_type="report_updated",
            payload={
                "report": updated_report,
                "version": current_version + 1,
                "instructions_applied": len(revision_instructions)
            },
            session_id=session_id,
            correlation_id=message.correlation_id
        )

    async def save_report(self, session_id: str, report_data: dict[str, Any], format_config: dict[str, Any] = None, version: int = 1):
        """Save report to file system with format-specific naming."""
        try:
            import os
            from datetime import datetime

            # Create session directories
            session_dir = f"KEVIN/sessions/{session_id}"
            working_dir = os.path.join(session_dir, "working")
            final_dir = os.path.join(session_dir, "final")

            os.makedirs(working_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)

            # Generate filename based on format configuration
            if format_config:
                filename_prefix = format_config.get("filename_prefix", "REPORT")
                format_type = format_config.get("format_type", "standard")
            else:
                filename_prefix = "REPORT"
                format_type = "standard"

            # Create filename with timestamp and format info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic = report_data.get("report_data", {}).get("title", "research").replace(" ", "_")[:30]

            # Save as working draft first
            filename = f"{filename_prefix}_{format_type}_{topic}_{timestamp}_DRAFT.md"
            working_file = os.path.join(working_dir, filename)

            # Convert report data to markdown format
            markdown_content = self.convert_to_markdown(report_data, format_config)

            with open(working_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"Report saved to: {working_file}")
            print(f"Format: {format_type} (confidence: {format_config.get('confidence', 'N/A')})")

            # Log intent analysis results
            if format_config and "intent_analysis" in format_config:
                intent = format_config["intent_analysis"]
                print(f"Query intent: {intent['format']} - {intent['reasoning']}")

        except Exception as e:
            print(f"Error saving report: {e}")

    async def save_final_report(self, session_id: str, report_data: dict[str, Any], format_config: dict[str, Any] = None, version: int = 1):
        """Save final report to both working and final directories with proper naming."""
        try:
            import os
            from datetime import datetime

            # Create session directories
            session_dir = f"KEVIN/sessions/{session_id}"
            working_dir = os.path.join(session_dir, "working")
            final_dir = os.path.join(session_dir, "final")

            os.makedirs(working_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)

            # Generate filename based on format configuration
            if format_config:
                filename_prefix = format_config.get("filename_prefix", "REPORT")
                format_type = format_config.get("format_type", "standard")
            else:
                filename_prefix = "REPORT"
                format_type = "standard"

            # Create filename with timestamp and format info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic = report_data.get("report_data", {}).get("title", "research").replace(" ", "_")[:30]

            # Convert report data to markdown format
            markdown_content = self.convert_to_markdown(report_data, format_config)

            # Save final version to /final/ directory
            final_filename = f"{filename_prefix}_{format_type}_{topic}_{timestamp}_FINAL.md"
            final_file = os.path.join(final_dir, final_filename)

            with open(final_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"✅ Final report saved to: {final_file}")
            print(f"📁 Format: {format_type} (confidence: {format_config.get('confidence', 'N/A')})")

            # Also save to working directory for backup
            working_filename = f"{filename_prefix}_{format_type}_{topic}_{timestamp}_WORKING.md"
            working_file = os.path.join(working_dir, working_filename)

            with open(working_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"📝 Working copy saved to: {working_file}")

            # Log intent analysis results
            if format_config and "intent_analysis" in format_config:
                intent = format_config["intent_analysis"]
                print(f"🎯 Query intent: {intent['format']} - {intent['reasoning']}")

            # Create a summary file in the final directory
            await self._create_report_summary(session_id, final_dir, final_filename, format_config, report_data)

            return {
                "final_file_path": final_file,
                "working_file_path": working_file,
                "format_type": format_type,
                "filename": final_filename
            }

        except Exception as e:
            print(f"❌ Error saving final report: {e}")
            return None

    async def _create_report_summary(self, session_id: str, final_dir: str, filename: str, format_config: dict[str, Any], report_data: dict[str, Any]):
        """Create a summary file with report metadata and information."""
        try:
            import os
            import json
            from datetime import datetime

            summary_file = os.path.join(final_dir, f"REPORT_SUMMARY_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

            summary = {
                "session_id": session_id,
                "report_filename": filename,
                "generated_at": datetime.now().isoformat(),
                "format_config": format_config,
                "report_metadata": {
                    "title": report_data.get("report_data", {}).get("title", "Unknown"),
                    "format": format_config.get("format_type", "standard"),
                    "confidence": format_config.get("confidence", 0.0),
                    "sections": format_config.get("sections", []),
                    "audience": format_config.get("audience", "general"),
                    "style": format_config.get("style", "balanced")
                },
                "file_organization": {
                    "final_reports_directory": final_dir,
                    "this_file": summary_file,
                    "report_file": filename
                }
            }

            # Add intent analysis if available
            if format_config and "intent_analysis" in format_config:
                summary["intent_analysis"] = format_config["intent_analysis"]

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"📋 Report summary saved to: {summary_file}")

        except Exception as e:
            print(f"⚠️ Warning: Could not create report summary: {e}")

    def convert_to_markdown(self, report_data: dict[str, Any], format_config: dict[str, Any] = None) -> str:
        """Convert report data to markdown format with format-specific styling."""
        # Get format information
        if format_config:
            format_type = format_config.get("format_type", "standard")
            style = format_config.get("style", "balanced")
            audience = format_config.get("audience", "educated")
            sections = format_config.get("sections", ["executive_summary", "introduction", "findings", "analysis", "conclusions"])
        else:
            format_type = "standard"
            style = "balanced"
            audience = "educated"
            sections = ["executive_summary", "introduction", "findings", "analysis", "conclusions"]

        title = report_data.get("report_data", {}).get("title", "Research Report")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Start building markdown with format-specific header
        markdown = f"# {title}\n\n"

        # Add format information
        if format_type != "standard":
            markdown += f"**Report Format:** {format_type.title()}\n"
            markdown += f"**Target Audience:** {audience.title()}\n"
            markdown += f"**Style:** {style.title()}\n"

        markdown += f"**Generated:** {timestamp}\n\n"

        # Add intent analysis information if available
        if format_config and "intent_analysis" in format_config:
            intent = format_config["intent_analysis"]
            markdown += f"**Query Intent Analysis:** {intent['format'].title()} Format\n"
            markdown += f"**Confidence:** {intent['confidence']:.1%}\n"
            markdown += f"**Reasoning:** {intent['reasoning']}\n\n"

        # Add sections based on format configuration
        if "executive_summary" in sections:
            markdown += "## Executive Summary\n\n"
            if format_type == "brief":
                markdown += "*Concise overview highlighting key findings and main takeaways.*\n\n"
            elif format_type == "comprehensive":
                markdown += "*Detailed summary covering all major findings, methodology, and implications.*\n\n"
            else:
                markdown += "*Balanced overview of the research findings and their significance.*\n\n"
            markdown += "[Executive summary content would be generated here based on research data]\n\n"

        if "introduction" in sections:
            markdown += "## Introduction\n\n"
            if format_type == "brief":
                markdown += "*Brief context and background information.*\n\n"
            elif format_type == "comprehensive":
                markdown += "*Comprehensive background, context, and research scope.*\n\n"
            else:
                markdown += "*Background and context for the research topic.*\n\n"
            markdown += "[Introduction content would be generated here]\n\n"

        if "findings" in sections or "key_findings" in sections:
            if format_type == "brief":
                markdown += "## Key Findings\n\n"
                markdown += "*Essential findings in bullet point format.*\n\n"
            elif format_type == "comprehensive":
                markdown += "## Detailed Findings\n\n"
                markdown += "*Comprehensive presentation of all research findings with supporting evidence.*\n\n"
            else:
                markdown += "## Main Findings\n\n"
                markdown += "*Key findings from the research analysis.*\n\n"
            markdown += "[Findings content would be formatted here]\n\n"

        if "detailed_analysis" in sections:
            markdown += "## Detailed Analysis\n\n"
            markdown += "*In-depth analysis and interpretation of findings.*\n\n"
            markdown += "[Detailed analysis content would be formatted here]\n\n"

        if "analysis" in sections:
            markdown += "## Analysis\n\n"
            if format_type == "brief":
                markdown += "*Brief analysis of the key findings.*\n\n"
            elif format_type == "comprehensive":
                markdown += "*Comprehensive analysis with multiple perspectives and implications.*\n\n"
            else:
                markdown += "*Analysis of the research findings.*\n\n"
            markdown += "[Analysis content would be formatted here]\n\n"

        if "implications" in sections:
            markdown += "## Implications\n\n"
            markdown += "*Implications and significance of the findings.*\n\n"
            markdown += "[Implications content would be formatted here]\n\n"

        if "conclusions" in sections:
            markdown += "## Conclusions\n\n"
            if format_type == "brief":
                markdown += "*Brief conclusion with main takeaways.*\n\n"
            elif format_type == "comprehensive":
                markdown += "*Comprehensive conclusions with recommendations and future research directions.*\n\n"
            else:
                markdown += "*Conclusions and final thoughts.*\n\n"
            markdown += "[Conclusions would be formatted here]\n\n"

        if "sources" in sections:
            markdown += "## Sources\n\n"
            markdown += "*Comprehensive list of sources and references.*\n\n"
            markdown += "[Sources would be listed here]\n\n"

        return markdown

    def get_tools(self) -> list:
        """Get the list of tools for this agent."""
        return [self.create_report, self.update_report, self.request_more_research]
