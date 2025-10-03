"""Mock SDK implementations for development testing."""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any


class MockClaudeSDKClient:
    """Mock Claude SDK client for development testing."""

    def __init__(self, options: dict[str, Any] | None = None):
        self.options = options or {}
        self.agent_name = "mock_agent"
        self.conversation_history = []
        self.responses = []
        self.connected = False

    async def connect(self):
        """Mock connection to Claude."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.connected = True

    async def disconnect(self):
        """Mock disconnection."""
        await asyncio.sleep(0.05)
        self.connected = False

    async def query(self, prompt: str, **kwargs) -> str:
        """Mock query with simulated response."""
        await asyncio.sleep(0.5)  # Simulate processing time

        # Generate mock response based on agent type and prompt
        response = self._generate_mock_response(prompt)
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    async def receive_response(self) -> AsyncGenerator[dict[str, Any], None]:
        """Mock response stream."""
        await asyncio.sleep(0.2)

        # Generate mock response data
        response_data = {
            "id": str(uuid.uuid4()),
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": self._generate_mock_response("current context")
                }
            ],
            "model": "claude-3-sonnet-20240229",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 150
            }
        }

        yield response_data

    def _generate_mock_response(self, prompt: str) -> str:
        """Generate contextual mock responses."""
        prompt_lower = prompt.lower()

        if "research" in prompt_lower:
            return """I'll conduct comprehensive research on this topic using web search and analysis tools.

Research Plan:
1. Search for recent academic studies and reports
2. Analyze credible sources and extract key information
3. Identify statistics and expert opinions
4. Organize findings for report generation

I'll focus on finding reliable sources and gathering comprehensive information to support the research objectives."""

        elif "report" in prompt_lower or "generate" in prompt_lower:
            return """I'll create a well-structured report based on the research findings.

Report Structure:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Conclusions and Recommendations

The report will be properly formatted with clear sections, professional tone, and appropriate citations to ensure credibility and readability."""

        elif "review" in prompt_lower or "edit" in prompt_lower:
            return """I'll review the report for quality, accuracy, and completeness.

Review Criteria:
- Content accuracy and source reliability
- Clarity and organization
- Completeness of coverage
- Professional presentation

I'll provide specific feedback and recommendations for improvement to ensure the report meets professional standards."""

        else:
            return """I understand the task and will use the appropriate tools to complete it effectively. I'll ensure high-quality results that meet the specified requirements."""

    def add_response(self, response: dict[str, Any]):
        """Add a predefined response for testing."""
        self.responses.append(response)


class MockAgentDefinition:
    """Mock AgentDefinition for development testing."""

    def __init__(
        self,
        description: str = "Mock Agent Description",
        prompt: str = "You are a helpful AI assistant.",
        tools: list[str] = None,
        model: str = "sonnet"
    ):
        self.description = description
        self.prompt = prompt
        self.tools = tools or []
        self.model = model


class MockMCPTool:
    """Mock MCP tool for development testing."""

    def __init__(self, name: str, description: str, params_schema: dict[str, Any]):
        self.name = name
        self.description = description
        self.params_schema = params_schema
        self.call_count = 0
        self.calls = []

    async def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        """Mock tool execution."""
        self.call_count += 1
        self.calls.append(args)

        # Generate mock result based on tool type
        return self._generate_mock_result(args)

    def _generate_mock_result(self, args: dict[str, Any]) -> dict[str, Any]:
        """Generate mock results based on tool type."""
        tool_name = self.name

        if "conduct_research" in tool_name:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Research completed on {args.get('topic', 'unknown topic')}. Found 5 key findings and 8 sources."
                }],
                "research_data": {
                    "topic": args.get("topic", "Unknown Topic"),
                    "findings": [
                        {
                            "fact": f"Key finding about {args.get('topic', 'the topic')}",
                            "sources": ["source1.com", "source2.com"],
                            "confidence": "high",
                            "context": "Additional context would be provided"
                        }
                    ],
                    "sources_used": [
                        {
                            "title": "Sample Source Title",
                            "url": "https://example.com",
                            "type": "academic",
                            "reliability": "high",
                            "date": "2024-01-01"
                        }
                    ]
                }
            }

        elif "generate_report" in tool_name:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Report generated: Research Report on {args.get('research_data', {}).get('topic', 'Unknown Topic')}"
                }],
                "report_data": {
                    "title": f"Research Report: {args.get('research_data', {}).get('topic', 'Unknown Topic')}",
                    "sections": {
                        "summary": "Executive summary of research findings",
                        "findings": "Key findings from the research",
                        "conclusions": "Conclusions and recommendations"
                    }
                }
            }

        elif "review_report" in tool_name:
            return {
                "content": [{
                    "type": "text",
                    "text": "Report review completed. Overall score: 8.5/10. Recommendation: approve_with_minor_revisions"
                }],
                "review_result": {
                    "overall_score": 8.5,
                    "strengths": ["Well-structured and organized", "Good use of research evidence"],
                    "improvement_areas": ["Could use more recent sources"],
                    "recommendation": "approve_with_minor_revisions"
                }
            }

        else:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Tool {tool_name} executed successfully"
                }]
            }


def create_mock_orchestrator() -> "MockResearchOrchestrator":
    """Create a mock orchestrator for testing."""
    return MockResearchOrchestrator()


class MockResearchOrchestrator:
    """Mock ResearchOrchestrator for development testing."""

    def __init__(self):
        self.agent_clients = {}
        self.active_sessions = {}
        self.agent_definitions = {
            "research_agent": MockAgentDefinition("Research Agent", "Research specialist", ["WebSearch"]),
            "report_agent": MockAgentDefinition("Report Agent", "Report specialist", ["Read", "Write"]),
            "editor_agent": MockAgentDefinition("Editor Agent", "Editor specialist", ["Read", "Edit"]),
            "ui_coordinator": MockAgentDefinition("UI Coordinator", "Coordinator", ["Read"])
        }
        self.initialized = False

    async def initialize(self):
        """Mock initialization."""
        await asyncio.sleep(0.1)

        # Create mock clients for each agent
        for agent_name in self.agent_definitions.keys():
            client = MockClaudeSDKClient()
            client.agent_name = agent_name
            await client.connect()
            self.agent_clients[agent_name] = client

        self.initialized = True

    async def start_research_session(self, topic: str, user_requirements: dict[str, Any]) -> str:
        """Mock starting a research session."""
        if not self.initialized:
            await self.initialize()

        session_id = str(uuid.uuid4())

        self.active_sessions[session_id] = {
            "session_id": session_id,
            "topic": topic,
            "user_requirements": user_requirements,
            "status": "researching",
            "created_at": "2024-01-01T12:00:00",
            "current_stage": "research",
            "workflow_history": []
        }

        # Simulate workflow progression
        asyncio.create_task(self._mock_workflow_progression(session_id))

        return session_id

    async def _mock_workflow_progression(self, session_id: str):
        """Simulate the research workflow progression."""
        stages = [
            ("research", "researching", "Conducting research"),
            ("report_generation", "generating_report", "Generating report"),
            ("editorial_review", "editorial_review", "Reviewing report"),
            ("completion", "completed", "Research workflow completed")
        ]

        for stage_name, status, message in stages:
            await asyncio.sleep(1.0)  # Simulate work

            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = status
                self.active_sessions[session_id]["current_stage"] = stage_name
                self.active_sessions[session_id]["workflow_history"].append({
                    "stage": stage_name,
                    "completed_at": "2024-01-01T12:00:00",
                    "results_count": 1
                })

        if session_id in self.active_sessions:
            self.active_sessions[session_id]["completed_at"] = "2024-01-01T12:04:00"

    async def get_session_status(self, session_id: str) -> dict[str, Any]:
        """Mock getting session status."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session_data = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "status": session_data.get("status"),
            "current_stage": session_data.get("current_stage"),
            "topic": session_data.get("topic"),
            "created_at": session_data.get("created_at"),
            "workflow_history": session_data.get("workflow_history", [])
        }

    async def cleanup(self):
        """Mock cleanup."""
        for client in self.agent_clients.values():
            await client.disconnect()
        self.agent_clients.clear()
        self.active_sessions.clear()
        self.initialized = False
