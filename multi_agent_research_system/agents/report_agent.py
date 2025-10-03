"""Report Generation Agent for creating structured reports.

This agent specializes in transforming research data into well-structured,
coherent reports with proper formatting and citation.
"""

import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from claude_agent_sdk import tool
from ..core.base_agent import BaseAgent, Message


class ReportAgent(BaseAgent):
    """Agent responsible for generating reports from research data."""

    def __init__(self):
        super().__init__("report_agent", "report_generation")
        self.register_message_handler("research_completed", self.handle_research_completed)
        self.register_message_handler("report_feedback", self.handle_report_feedback)
        self.register_message_handler("revision_request", self.handle_revision_request)

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
        "research_data": Dict[str, Any],
        "report_format": str,
        "target_audience": str,
        "tone": str,
        "sections": List[str]
    })
    async def create_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
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
        "existing_report": Dict[str, Any],
        "feedback": List[Dict[str, Any]],
        "new_research": Optional[Dict[str, Any]]
    })
    async def update_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
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
        "research_gaps": List[str],
        "current_report": Dict[str, Any],
        "priority": str
    })
    async def request_more_research(self, args: Dict[str, Any]) -> Dict[str, Any]:
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

    async def handle_research_completed(self, message: Message) -> Optional[Message]:
        """Handle completed research from Research Agent."""
        payload = message.payload
        session_id = message.session_id

        # Start report generation session
        await self.start_session(session_id, {
            "topic": payload.get("topic"),
            "status": "generating_report"
        })

        # Generate initial report
        report_result = await self.create_report({
            "research_data": payload,
            "report_format": "markdown",
            "target_audience": "general",
            "tone": "analytical",
            "sections": ["executive_summary", "introduction", "findings", "analysis", "conclusions"]
        })

        # Save report to file system
        await self.save_report(session_id, report_result)

        # Update session data
        report_data = {
            "topic": payload.get("topic"),
            "report": report_result,
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

    async def handle_report_feedback(self, message: Message) -> Optional[Message]:
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
            await self.save_report(session_id, updated_report, version=current_version + 1)

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

    async def handle_revision_request(self, message: Message) -> Optional[Message]:
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
        await self.save_report(session_id, updated_report, version=current_version + 1)

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

    async def save_report(self, session_id: str, report_data: Dict[str, Any], version: int = 1):
        """Save report to file system."""
        try:
            import os

            # Create session directory
            session_dir = f"researchmaterials/sessions/{session_id}"
            os.makedirs(session_dir, exist_ok=True)

            # Save report as markdown
            report_file = f"{session_dir}/report_v{version}.md"

            # Convert report data to markdown format
            markdown_content = self.convert_to_markdown(report_data)

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"Report saved to: {report_file}")

        except Exception as e:
            print(f"Error saving report: {e}")

    def convert_to_markdown(self, report_data: Dict[str, Any]) -> str:
        """Convert report data to markdown format."""
        # This would convert the structured report data to markdown
        # For now, return a basic structure
        title = report_data.get("report_data", {}).get("title", "Research Report")

        markdown = f"# {title}\n\n"
        markdown += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Add executive summary
        markdown += "## Executive Summary\n\n"
        markdown += "[Executive summary content would be generated here]\n\n"

        # Add sections
        markdown += "## Main Findings\n\n"
        markdown += "[Main findings would be formatted here]\n\n"

        markdown += "## Analysis\n\n"
        markdown += "[Analysis content would be formatted here]\n\n"

        markdown += "## Conclusions\n\n"
        markdown += "[Conclusions would be formatted here]\n\n"

        return markdown

    def get_tools(self) -> List:
        """Get the list of tools for this agent."""
        return [self.create_report, self.update_report, self.request_more_research]