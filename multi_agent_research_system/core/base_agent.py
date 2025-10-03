"""Base agent class for the multi-agent research system.

This module provides the foundation for all agents in the research system,
leveraging Claude Agent SDK patterns for communication and tool management.
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
)


class Message:
    """Message structure for inter-agent communication."""

    def __init__(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        payload: dict[str, Any],
        session_id: str,
        correlation_id: str | None = None
    ):
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type
        self.payload = payload
        self.session_id = session_id
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "payload": self.payload,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        msg = cls(
            sender=data["sender"],
            recipient=data["recipient"],
            message_type=data["message_type"],
            payload=data["payload"],
            session_id=data["session_id"],
            correlation_id=data.get("correlation_id")
        )
        msg.timestamp = data["timestamp"]
        return msg


class BaseAgent(ABC):
    """Base class for all agents in the research system."""

    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.message_handlers: dict[str, callable] = {}
        self.client: ClaudeSDKClient | None = None
        self.active_sessions: dict[str, dict[str, Any]] = {}

    def register_message_handler(self, message_type: str, handler: callable):
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler

    async def initialize(self):
        """Initialize the agent with Claude SDK client."""
        options = ClaudeAgentOptions(
            allowed_tools=["Read", "Write", "Edit", "Bash", "WebSearch", "WebFetch"],
            system_prompt=self.get_system_prompt()
        )

        self.client = ClaudeSDKClient(options=options)
        await self.client.connect()

    async def send_message(self, message: Message) -> Message | None:
        """Send a message to another agent through the message router."""
        if not self.client:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        try:
            # Use the message router tool to send the message
            prompt = f"Send message to {message.recipient}: {json.dumps(message.to_dict())}"
            await self.client.query(prompt)

            # For now, return None - in a real implementation, this would wait for a response
            return None
        except Exception as e:
            print(f"Error sending message: {e}")
            return None

    async def handle_message(self, message: Message) -> Message | None:
        """Handle an incoming message."""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                return await handler(message)
            except Exception as e:
                print(f"Error handling message {message.message_type}: {e}")
                return None
        else:
            print(f"No handler registered for message type: {message.message_type}")
            return None

    async def start_session(self, session_id: str, initial_data: dict[str, Any] = None):
        """Start a new research session."""
        self.active_sessions[session_id] = {
            "started_at": datetime.now().isoformat(),
            "status": "active",
            "data": initial_data or {}
        }

    async def end_session(self, session_id: str):
        """End a research session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "completed"
            self.active_sessions[session_id]["ended_at"] = datetime.now().isoformat()

    def get_session_data(self, session_id: str) -> dict[str, Any]:
        """Get data for a specific session."""
        return self.active_sessions.get(session_id, {}).get("data", {})

    def update_session_data(self, session_id: str, data: dict[str, Any]):
        """Update data for a specific session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["data"].update(data)

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @abstractmethod
    def get_tools(self) -> list:
        """Get the list of tools for this agent."""
        pass

    async def cleanup(self):
        """Cleanup resources."""
        if self.client:
            await self.client.disconnect()


class AgentRegistry:
    """Registry for managing all agents in the system."""

    def __init__(self):
        self.agents: dict[str, BaseAgent] = {}
        self.message_router = MessageRouter()

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the registry."""
        self.agents[agent.name] = agent
        self.message_router.register_agent(agent)

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get an agent by name."""
        return self.agents.get(name)

    async def initialize_all_agents(self):
        """Initialize all registered agents."""
        for agent in self.agents.values():
            await agent.initialize()

    async def cleanup_all_agents(self):
        """Cleanup all registered agents."""
        for agent in self.agents.values():
            await agent.cleanup()


class MessageRouter:
    """Router for handling inter-agent communication."""

    def __init__(self):
        self.agents: dict[str, BaseAgent] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the router."""
        self.agents[agent.name] = agent

    async def route_message(self, message: Message):
        """Route a message to the appropriate agent."""
        recipient_agent = self.agents.get(message.recipient)
        if recipient_agent:
            await recipient_agent.handle_message(message)
        else:
            print(f"Unknown recipient: {message.recipient}")

    async def start_processing(self):
        """Start processing messages in the background."""
        self.running = True
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self.route_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing message: {e}")

    def stop_processing(self):
        """Stop processing messages."""
        self.running = False

    async def send_message(self, message: Message):
        """Queue a message for processing."""
        await self.message_queue.put(message)
