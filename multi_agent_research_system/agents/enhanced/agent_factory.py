"""Agent Factory - Comprehensive Agent Creation and Management System

This module provides a sophisticated factory pattern for creating and managing
enhanced agents with comprehensive configuration, lifecycle management,
and performance monitoring.

Key Features:
- Factory Pattern Implementation
- Configuration Management
- Agent Registry and Discovery
- Lifecycle Management
- Dynamic Agent Creation
- Template-Based Agent Generation
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .base_agent import EnhancedBaseAgent, AgentConfiguration, AgentPriority, AgentStatus
from ..research_agent import ResearchAgent
from ..report_agent import ReportAgent
from ..decoupled_editorial_agent import DecoupledEditorialAgent


class AgentType(Enum):
    """Supported agent types."""
    RESEARCH = "research"
    REPORT = "report"
    EDITORIAL = "editorial"
    CONTENT_CLEANER = "content_cleaner"
    QUALITY_JUDGE = "quality_judge"
    GAP_RESEARCH = "gap_research"
    DATA_INTEGRATION = "data_integration"
    CONTENT_ENHANCEMENT = "content_enhancement"
    STYLE_OPTIMIZATION = "style_optimization"
    QUALITY_VALIDATION = "quality_validation"
    CUSTOM = "custom"


@dataclass
class AgentTemplate:
    """Template for agent creation."""
    name: str
    agent_type: AgentType
    description: str
    default_config: AgentConfiguration
    required_tools: List[str] = field(default_factory=list)
    optional_tools: List[str] = field(default_factory=list)
    system_prompt_template: Optional[str] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentCreationRequest:
    """Request for agent creation."""
    agent_type: AgentType
    agent_id: Optional[str] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    template_name: Optional[str] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    auto_initialize: bool = True
    dependencies: List[str] = field(default_factory=dict)


@dataclass
class AgentRegistry:
    """Registry for active agents and discovery."""
    agents: Dict[str, EnhancedBaseAgent] = field(default_factory=dict)
    agent_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    agent_templates: Dict[str, AgentTemplate] = field(default_factory=dict)
    agent_types: Dict[AgentType, Type[EnhancedBaseAgent]] = field(default_factory=dict)
    message_routing: Dict[str, str] = field(default_factory=dict)  # message_type -> agent_id

    def __post_init__(self):
        """Initialize registry with default agent types."""
        self._register_default_agent_types()
        self._register_default_templates()

    def _register_default_agent_types(self) -> None:
        """Register default agent types."""
        # Note: These would be the enhanced versions of agents
        # For now, using placeholder mappings
        self.agent_types = {
            AgentType.RESEARCH: ResearchAgent,
            AgentType.REPORT: ReportAgent,
            AgentType.EDITORIAL: type("EnhancedEditorialAgent", (DecoupledEditorialAgent,), {}),
            # Additional agent types would be registered here
        }

    def _register_default_templates(self) -> None:
        """Register default agent templates."""
        # Research Agent Template
        self.agent_templates["research_default"] = AgentTemplate(
            name="research_default",
            agent_type=AgentType.RESEARCH,
            description="Default research agent for web search and information gathering",
            default_config=AgentConfiguration(
                agent_type="research",
                agent_id="",  # Will be set during creation
                max_turns=30,
                allowed_tools=["web_search", "source_analysis", "information_synthesis"],
                priority=AgentPriority.HIGH,
                timeout_seconds=300,
                quality_threshold=0.7
            ),
            required_tools=["web_search"],
            optional_tools=["source_analysis", "information_synthesis"],
            dependencies=[]
        )

        # Report Agent Template
        self.agent_templates["report_default"] = AgentTemplate(
            name="report_default",
            agent_type=AgentType.REPORT,
            description="Default report generation agent",
            default_config=AgentConfiguration(
                agent_type="report",
                agent_id="",
                max_turns=25,
                allowed_tools=["create_report", "update_report", "format_content"],
                priority=AgentPriority.NORMAL,
                timeout_seconds=240,
                quality_threshold=0.75
            ),
            required_tools=["create_report"],
            optional_tools=["update_report", "format_content"],
            dependencies=["research"]
        )

        # Editorial Agent Template
        self.agent_templates["editorial_default"] = AgentTemplate(
            name="editorial_default",
            agent_type=AgentType.EDITORIAL,
            description="Default editorial enhancement agent",
            default_config=AgentConfiguration(
                agent_type="editorial",
                agent_id="",
                max_turns=20,
                allowed_tools=["enhance_content", "coordinate_gap_research", "apply_progressive_enhancement"],
                priority=AgentPriority.HIGH,
                timeout_seconds=180,
                quality_threshold=0.8
            ),
            required_tools=["enhance_content"],
            optional_tools=["coordinate_gap_research", "apply_progressive_enhancement"],
            dependencies=["research", "report"]
        )

        # Gap Research Agent Template
        self.agent_templates["gap_research_default"] = AgentTemplate(
            name="gap_research_default",
            agent_type=AgentType.GAP_RESEARCH,
            description="Specialized agent for targeted gap research",
            default_config=AgentConfiguration(
                agent_type="gap_research",
                agent_id="",
                max_turns=15,
                allowed_tools=["targeted_search", "gap_analysis", "research_integration"],
                priority=AgentPriority.HIGH,
                timeout_seconds=200,
                quality_threshold=0.7
            ),
            required_tools=["targeted_search"],
            optional_tools=["gap_analysis", "research_integration"],
            dependencies=["editorial"]
        )


class EnhancedAgentFactory:
    """Enhanced factory for creating and managing agents with comprehensive lifecycle support."""

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger("agent_factory")
        self.registry = AgentRegistry()
        self.config_path = config_path or Path("config/agent_factory_config.json")
        self.creation_lock = asyncio.Lock()
        self._load_factory_config()

    def _load_factory_config(self) -> None:
        """Load factory configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self._apply_factory_config(config)
                    self.logger.info("Loaded factory configuration")
            else:
                self.logger.info("No factory config found, using defaults")
        except Exception as e:
            self.logger.error(f"Failed to load factory config: {e}")

    def _apply_factory_config(self, config: Dict[str, Any]) -> None:
        """Apply factory configuration."""
        # Load custom templates
        if "templates" in config:
            for template_data in config["templates"]:
                template = self._create_template_from_config(template_data)
                self.registry.agent_templates[template.name] = template

        # Load custom agent types
        if "agent_types" in config:
            for type_config in config["agent_types"]:
                self._register_custom_agent_type(type_config)

    def _create_template_from_config(self, template_data: Dict[str, Any]) -> AgentTemplate:
        """Create template from configuration data."""
        return AgentTemplate(
            name=template_data["name"],
            agent_type=AgentType(template_data["agent_type"]),
            description=template_data["description"],
            default_config=AgentConfiguration(**template_data["default_config"]),
            required_tools=template_data.get("required_tools", []),
            optional_tools=template_data.get("optional_tools", []),
            system_prompt_template=template_data.get("system_prompt_template"),
            custom_parameters=template_data.get("custom_parameters", {}),
            dependencies=template_data.get("dependencies", [])
        )

    def _register_custom_agent_type(self, type_config: Dict[str, Any]) -> None:
        """Register custom agent type from configuration."""
        # This would dynamically load custom agent classes
        agent_type = AgentType(type_config["name"])
        # Implementation would load the class from module_path
        self.logger.info(f"Registered custom agent type: {agent_type}")

    async def create_agent(self, request: AgentCreationRequest) -> EnhancedBaseAgent:
        """Create an enhanced agent based on the request."""
        async with self.creation_lock:
            try:
                self.logger.info(f"Creating agent of type {request.agent_type}")

                # Get agent template
                template = self._get_agent_template(request)

                # Create agent configuration
                config = self._create_agent_config(template, request)

                # Create agent instance
                agent = self._instantiate_agent(config, request)

                # Initialize agent if requested
                if request.auto_initialize:
                    await agent.initialize(self.registry)

                # Register agent
                await self._register_agent(agent, request)

                self.logger.info(f"Successfully created agent {agent.agent_id}")
                return agent

            except Exception as e:
                self.logger.error(f"Failed to create agent: {e}")
                raise

    def _get_agent_template(self, request: AgentCreationRequest) -> AgentTemplate:
        """Get agent template for the request."""
        template_name = request.template_name or f"{request.agent_type.value}_default"

        if template_name not in self.registry.agent_templates:
            raise ValueError(f"Template '{template_name}' not found")

        return self.registry.agent_templates[template_name]

    def _create_agent_config(self, template: AgentTemplate, request: AgentCreationRequest) -> AgentConfiguration:
        """Create agent configuration from template and request."""
        # Start with template default config
        config_data = template.default_config.__dict__.copy()

        # Override with request config
        config_data.update(request.config_overrides)

        # Set agent ID
        agent_id = request.agent_id or self._generate_agent_id(template.agent_type)
        config_data["agent_id"] = agent_id

        # Apply template-specific configurations
        if template.required_tools:
            config_data.setdefault("allowed_tools", []).extend(template.required_tools)

        if template.optional_tools:
            config_data.setdefault("allowed_tools", []).extend(template.optional_tools)

        # Remove duplicates
        if "allowed_tools" in config_data:
            config_data["allowed_tools"] = list(set(config_data["allowed_tools"]))

        return AgentConfiguration(**config_data)

    def _instantiate_agent(self, config: AgentConfiguration, request: AgentCreationRequest) -> EnhancedBaseAgent:
        """Instantiate agent based on type and configuration."""
        agent_type_class = self.registry.agent_types.get(AgentType(config.agent_type))

        if not agent_type_class:
            raise ValueError(f"Unknown agent type: {config.agent_type}")

        # For existing agents that don't inherit from EnhancedBaseAgent yet,
        # we need to create a wrapper or enhanced version
        if hasattr(agent_type_class, '__bases__') and EnhancedBaseAgent not in agent_type_class.__bases__:
            # Create enhanced wrapper
            return self._create_enhanced_wrapper(agent_type_class, config)
        else:
            # Direct instantiation
            return agent_type_class(config)

    def _create_enhanced_wrapper(self, original_class: Type, config: AgentConfiguration) -> EnhancedBaseAgent:
        """Create enhanced wrapper for existing agent classes."""
        class EnhancedWrapper(EnhancedBaseAgent):
            def __init__(self, config: AgentConfiguration):
                super().__init__(config)
                self.wrapped_agent = original_class()

            def get_system_prompt(self) -> str:
                if hasattr(self.wrapped_agent, 'get_system_prompt'):
                    return self.wrapped_agent.get_system_prompt()
                return f"Enhanced {self.agent_type} agent"

            def get_default_tools(self) -> List[str]:
                return self.config.allowed_tools

        return EnhancedWrapper(config)

    def _generate_agent_id(self, agent_type: AgentType) -> str:
        """Generate unique agent ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{agent_type.value}_{timestamp}_{uuid.uuid4().hex[:8]}"

    async def _register_agent(self, agent: EnhancedBaseAgent, request: AgentCreationRequest) -> None:
        """Register agent in the registry."""
        # Store in registry
        self.registry.agents[agent.agent_id] = agent

        # Store metadata
        self.registry.agent_metadata[agent.agent_id] = {
            "created_at": datetime.now().isoformat(),
            "agent_type": agent.agent_type,
            "template_used": request.template_name,
            "config_overrides": request.config_overrides,
            "dependencies": request.dependencies,
            "status": agent.status.value
        }

        # Register message routing
        await self._register_message_routing(agent)

    async def _register_message_routing(self, agent: EnhancedBaseAgent) -> None:
        """Register message routing for the agent."""
        # Register common message types
        common_types = [
            f"{agent.agent_type}_request",
            f"{agent.agent_type}_query",
            "ping",
            "status",
            "health_check"
        ]

        for message_type in common_types:
            self.registry.message_routing[message_type] = agent.agent_id

    async def create_agent_workflow(self, workflow_config: Dict[str, Any]) -> List[EnhancedBaseAgent]:
        """Create a complete workflow of related agents."""
        agents = []

        for agent_config in workflow_config.get("agents", []):
            request = AgentCreationRequest(
                agent_type=AgentType(agent_config["type"]),
                agent_id=agent_config.get("agent_id"),
                config_overrides=agent_config.get("config_overrides", {}),
                template_name=agent_config.get("template"),
                auto_initialize=False  # Initialize all at once after creation
            )

            agent = await self.create_agent(request)
            agents.append(agent)

        # Initialize all agents
        for agent in agents:
            await agent.initialize(self.registry)

        return agents

    async def route_message(self, message) -> bool:
        """Route message to appropriate agent."""
        recipient_agent_id = self.registry.message_routing.get(message.message_type)

        if not recipient_agent_id:
            # Try direct recipient
            recipient_agent_id = message.recipient

        if recipient_agent_id in self.registry.agents:
            agent = self.registry.agents[recipient_agent_id]
            await agent.receive_message(message)
            return True
        else:
            self.logger.warning(f"No agent found for message type: {message.message_type}")
            return False

    def get_agent(self, agent_id: str) -> Optional[EnhancedBaseAgent]:
        """Get agent by ID."""
        return self.registry.agents.get(agent_id)

    def get_agents_by_type(self, agent_type: AgentType) -> List[EnhancedBaseAgent]:
        """Get all agents of a specific type."""
        return [
            agent for agent in self.registry.agents.values()
            if agent.agent_type == agent_type.value
        ]

    def get_agent_metadata(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent metadata."""
        return self.registry.agent_metadata.get(agent_id)

    async def shutdown_agent(self, agent_id: str) -> bool:
        """Shutdown a specific agent."""
        if agent_id in self.registry.agents:
            agent = self.registry.agents[agent_id]
            await agent.shutdown()

            # Remove from registry
            del self.registry.agents[agent_id]
            del self.registry.agent_metadata[agent_id]

            # Clean up message routing
            self.registry.message_routing = {
                k: v for k, v in self.registry.message_routing.items()
                if v != agent_id
            }

            self.logger.info(f"Shutdown agent {agent_id}")
            return True
        else:
            self.logger.warning(f"Agent {agent_id} not found")
            return False

    async def shutdown_all_agents(self) -> None:
        """Shutdown all agents."""
        agent_ids = list(self.registry.agents.keys())
        for agent_id in agent_ids:
            await self.shutdown_agent(agent_id)

    def get_factory_status(self) -> Dict[str, Any]:
        """Get comprehensive factory status."""
        return {
            "total_agents": len(self.registry.agents),
            "agent_types": {
                agent_type.value: len(self.get_agents_by_type(agent_type))
                for agent_type in AgentType
            },
            "active_agents": len([
                agent for agent in self.registry.agents.values()
                if agent.status == AgentStatus.READY
            ]),
            "available_templates": list(self.registry.agent_templates.keys()),
            "message_routes": len(self.registry.message_routing),
            "config_path": str(self.config_path)
        }

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates."""
        return [
            {
                "name": template.name,
                "agent_type": template.agent_type.value,
                "description": template.description,
                "required_tools": template.required_tools,
                "optional_tools": template.optional_tools,
                "dependencies": template.dependencies
            }
            for template in self.registry.agent_templates.values()
        ]

    def register_custom_template(self, template: AgentTemplate) -> None:
        """Register a custom agent template."""
        self.registry.agent_templates[template.name] = template
        self.logger.info(f"Registered custom template: {template.name}")

    def register_custom_agent_type(self, agent_type: AgentType, agent_class: Type[EnhancedBaseAgent]) -> None:
        """Register a custom agent type."""
        self.registry.agent_types[agent_type] = agent_class
        self.logger.info(f"Registered custom agent type: {agent_type}")


# Global factory instance
_agent_factory: Optional[EnhancedAgentFactory] = None


def get_agent_factory(config_path: Optional[Path] = None) -> EnhancedAgentFactory:
    """Get or create the global agent factory instance."""
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = EnhancedAgentFactory(config_path)
    return _agent_factory


async def create_agent_from_config(config: Dict[str, Any]) -> EnhancedBaseAgent:
    """Convenience function to create agent from configuration."""
    factory = get_agent_factory()
    request = AgentCreationRequest(
        agent_type=AgentType(config["agent_type"]),
        agent_id=config.get("agent_id"),
        config_overrides=config.get("config_overrides", {}),
        template_name=config.get("template"),
        auto_initialize=config.get("auto_initialize", True)
    )
    return await factory.create_agent(request)


# Import for UUID generation
import uuid