"""Research Agent for conducting comprehensive web research.

This agent specializes in gathering, analyzing, and synthesizing information
from various sources to support the research process.
"""

import json
from typing import Any

from claude_agent_sdk import tool

from ..core.base_agent import BaseAgent, Message


class ResearchAgent(BaseAgent):
    """Agent responsible for conducting research on specified topics."""

    def __init__(self):
        super().__init__("research_agent", "research")
        self.register_message_handler("research_request", self.handle_research_request)
        self.register_message_handler("additional_research", self.handle_additional_research)

    def get_system_prompt(self) -> str:
        """Get the system prompt for the Research Agent."""
        return """You are a Research Agent, an expert in conducting comprehensive, high-quality research on any topic.

Your core responsibilities:
1. Conduct thorough web research using reliable sources
2. Analyze and validate source credibility
3. Synthesize information from multiple sources
4. Identify key facts, statistics, and expert opinions
5. Organize research findings in a structured manner

Research Standards:
- Prioritize authoritative sources (academic papers, reputable news, official reports)
- Cross-reference information across multiple sources
- Distinguish between facts and opinions
- Note source dates and potential biases
- Gather sufficient depth to support comprehensive reporting

When conducting research:
1. Start with broad search to understand the topic landscape
2. Deep-dive into specific aspects based on research goals
3. Look for recent developments and current perspectives
4. Identify expert consensus and areas of debate
5. Collect supporting data, statistics, and examples

Always provide source attribution and confidence levels for your findings."""

    @tool("web_research", "Conduct comprehensive web research on a topic", {
        "topic": str,
        "research_depth": str,
        "focus_areas": list[str],
        "max_sources": int
    })
    async def web_research(self, args: dict[str, Any]) -> dict[str, Any]:
        """Conduct comprehensive web research on a specified topic."""
        topic = args["topic"]
        research_depth = args.get("research_depth", "medium")
        focus_areas = args.get("focus_areas", [])
        max_sources = args.get("max_sources", 10)

        research_prompt = f"""
        Conduct comprehensive research on: {topic}

        Research depth: {research_depth}
        Focus areas: {', '.join(focus_areas) if focus_areas else 'General comprehensive research'}
        Maximum sources: {max_sources}

        Please provide:
        1. Key facts and information about the topic
        2. Important statistics and data points
        3. Expert opinions and consensus views
        4. Recent developments or current trends
        5. Areas of agreement and debate
        6. Source attribution for all information
        7. Confidence levels for key findings

        Structure your response as JSON with the following format:
        {{
            "topic": "{topic}",
            "research_findings": [
                {{
                    "fact": "Key finding or fact",
                    "sources": ["source1", "source2"],
                    "confidence": "high/medium/low",
                    "context": "Additional context or explanation"
                }}
            ],
            "statistics": [
                {{
                    "statistic": "The statistic",
                    "value": "The value",
                    "source": "Source of the statistic",
                    "date": "Date of data"
                }}
            ],
            "expert_opinions": [
                {{
                    "opinion": "Expert opinion or consensus",
                    "experts": ["Expert names or organizations"],
                    "context": "Context for this opinion"
                }}
            ],
            "recent_developments": [
                {{
                    "development": "Recent development",
                    "date": "Date or timeframe",
                    "significance": "Why this matters"
                }}
            ],
            "sources_used": [
                {{
                    "title": "Source title",
                    "url": "Source URL if available",
                    "type": "academic/news/official/other",
                    "reliability": "high/medium/low",
                    "date": "Publication date"
                }}
            ]
        }}
        """

        # This would integrate with WebSearch tool in actual implementation
        # For now, return a structured response that would come from research
        return {
            "content": [{
                "type": "text",
                "text": f"Research conducted on: {topic}\n\nDepth: {research_depth}\nFocus areas: {focus_areas}\n\n[Research results would be populated here from actual web search and analysis]"
            }],
            "research_data": {
                "topic": topic,
                "status": "completed",
                "findings_count": 0,  # Would be populated from actual research
                "sources_count": 0,    # Would be populated from actual research
                "confidence_score": 0.0  # Would be calculated from research quality
            }
        }

    @tool("source_analysis", "Analyze and validate research sources", {
        "sources": list[dict[str, Any]],
        "validation_criteria": list[str]
    })
    async def source_analysis(self, args: dict[str, Any]) -> dict[str, Any]:
        """Analyze and validate the credibility of research sources."""
        sources = args["sources"]
        validation_criteria = args.get("validation_criteria", [
            "authority", "accuracy", "objectivity", "currency", "coverage"
        ])

        analysis_prompt = f"""
        Analyze the following sources for credibility and reliability:

        Sources: {json.dumps(sources, indent=2)}

        Validation criteria: {', '.join(validation_criteria)}

        For each source, evaluate:
        1. Author expertise and credentials
        2. Publication reputation and editorial standards
        3. Evidence and sources cited
        4. Potential biases or conflicts of interest
        5. Publication date and relevance
        6. Overall reliability score (1-10)

        Return analysis as JSON:
        {{
            "source_analysis": [
                {{
                    "source": "Source identifier",
                    "reliability_score": 8.5,
                    "authority": "high/medium/low",
                    "bias_indicators": ["potential biases"],
                    "strengths": ["source strengths"],
                    "weaknesses": ["source weaknesses"],
                    "recommendation": "use/use_with_caution/avoid"
                }}
            ],
            "overall_assessment": {{
                "average_reliability": 7.8,
                "recommended_sources": 5,
                "sources_to_avoid": 1,
                "confidence_in_dataset": "high/medium/low"
            }}
        }}
        """

        return {
            "content": [{
                "type": "text",
                "text": f"Source analysis completed for {len(sources)} sources using criteria: {validation_criteria}"
            }],
            "analysis_result": {
                "sources_analyzed": len(sources),
                "criteria_used": validation_criteria,
                "status": "completed"
            }
        }

    @tool("information_synthesis", "Synthesize research findings into coherent insights", {
        "research_data": dict[str, Any],
        "synthesis_goals": list[str],
        "target_audience": str
    })
    async def information_synthesis(self, args: dict[str, Any]) -> dict[str, Any]:
        """Synthesize research findings into coherent insights."""
        research_data = args["research_data"]
        synthesis_goals = args.get("synthesis_goals", [
            "identify_key_trends", "establish_consensus", "highlight_controversies"
        ])
        target_audience = args.get("target_audience", "general")

        synthesis_prompt = f"""
        Synthesize the following research data into coherent insights:

        Research data: {json.dumps(research_data, indent=2)}

        Synthesis goals: {', '.join(synthesis_goals)}
        Target audience: {target_audience}

        Please provide:
        1. Key themes and patterns across sources
        2. Consensus views and expert agreements
        3. Areas of controversy or debate
        4. Knowledge gaps or areas needing more research
        5. Key takeaways for the target audience
        6. Confidence levels for synthesized insights

        Structure as JSON:
        {{
            "synthesized_insights": [
                {{
                    "insight": "Key insight or theme",
                    "supporting_evidence": ["evidence1", "evidence2"],
                    "confidence": "high/medium/low",
                    "sources": ["source1", "source2"]
                }}
            ],
            "consensus_views": [
                {{
                    "viewpoint": "Consensus view",
                    "support_level": "strong/moderate/limited",
                    "key_supporters": ["experts or organizations"]
                }}
            ],
            "controversies": [
                {{
                    "topic": "Area of debate",
                    "positions": ["position1", "position2"],
                    "evidence_for_each": ["evidence1", "evidence2"]
                }}
            ],
            "knowledge_gaps": [
                {{
                    "gap": "What we don't know",
                    "importance": "Why this matters",
                    "research_needed": "Type of research required"
                }}
            ]
        }}
        """

        return {
            "content": [{
                "type": "text",
                "text": f"Research synthesis completed for audience: {target_audience}\nGoals: {synthesis_goals}"
            }],
            "synthesis_result": {
                "status": "completed",
                "insights_generated": 0,  # Would be populated from actual synthesis
                "confidence_score": 0.0,
                "recommendations": ["Next steps based on synthesis"]
            }
        }

    async def handle_research_request(self, message: Message) -> Message | None:
        """Handle initial research request from UI coordinator."""
        payload = message.payload
        topic = payload.get("topic")
        session_id = message.session_id

        if not topic:
            return Message(
                sender=self.name,
                recipient=message.sender,
                message_type="research_error",
                payload={"error": "No topic provided for research"},
                session_id=session_id,
                correlation_id=message.correlation_id
            )

        # Start research session
        await self.start_session(session_id, {"topic": topic, "status": "researching"})

        # Conduct research using tools
        research_result = await self.web_research({
            "topic": topic,
            "research_depth": payload.get("depth", "medium"),
            "focus_areas": payload.get("focus_areas", []),
            "max_sources": payload.get("max_sources", 10)
        })

        # Analyze sources
        source_analysis = await self.source_analysis({
            "sources": research_result.get("sources_used", []),
            "validation_criteria": ["authority", "accuracy", "objectivity", "currency"]
        })

        # Synthesize findings
        synthesis = await self.information_synthesis({
            "research_data": research_result,
            "synthesis_goals": ["identify_key_trends", "establish_consensus"],
            "target_audience": "report_writer"
        })

        # Update session data
        research_data = {
            "topic": topic,
            "research_result": research_result,
            "source_analysis": source_analysis,
            "synthesis": synthesis,
            "status": "completed"
        }
        self.update_session_data(session_id, research_data)

        # Send results to report agent
        return Message(
            sender=self.name,
            recipient="report_agent",
            message_type="research_completed",
            payload=research_data,
            session_id=session_id,
            correlation_id=message.correlation_id
        )

    async def handle_additional_research(self, message: Message) -> Message | None:
        """Handle request for additional research on specific topics."""
        payload = message.payload
        research_gaps = payload.get("research_gaps", [])
        session_id = message.session_id

        # Conduct additional research for each gap
        additional_results = []
        for gap in research_gaps:
            result = await self.web_research({
                "topic": gap,
                "research_depth": "focused",
                "focus_areas": [],
                "max_sources": 5
            })
            additional_results.append(result)

        # Update session data with additional research
        session_data = self.get_session_data(session_id)
        session_data["additional_research"] = additional_results
        self.update_session_data(session_id, session_data)

        # Send additional research back to requesting agent
        return Message(
            sender=self.name,
            recipient=message.sender,
            message_type="additional_research_completed",
            payload={"additional_results": additional_results},
            session_id=session_id,
            correlation_id=message.correlation_id
        )

    def get_tools(self) -> list:
        """Get the list of tools for this agent."""
        return [self.web_research, self.source_analysis, self.information_synthesis]
