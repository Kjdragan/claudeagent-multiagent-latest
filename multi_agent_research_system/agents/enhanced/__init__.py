"""Enhanced Agents Module - Comprehensive Agent System with Claude SDK Integration

This module provides enhanced agent capabilities with comprehensive Claude Agent SDK
integration, advanced configuration management, performance monitoring, and
sophisticated communication protocols.

Key Components:
- EnhancedBaseAgent: Advanced base class with comprehensive SDK integration
- AgentFactory: Factory pattern for consistent agent creation and management
- SDKConfigManager: Comprehensive SDK configuration management
- LifecycleManager: Agent lifecycle management with health monitoring
- CommunicationManager: Rich message processing and communication protocols
- PerformanceMonitor: Real-time performance monitoring and optimization

Features:
- Comprehensive Claude Agent SDK integration
- Advanced configuration management with presets
- Agent lifecycle management with health monitoring
- Rich message processing with delivery guarantees
- Real-time performance monitoring and optimization
- Comprehensive testing and validation
"""

# Import core components
from .base_agent import (
    EnhancedBaseAgent,
    AgentConfiguration,
    AgentPerformanceMetrics,
    RichMessage,
    AgentStatus,
    AgentPriority
)

from .agent_factory import (
    EnhancedAgentFactory,
    AgentCreationRequest,
    AgentType,
    AgentTemplate,
    get_agent_factory
)

from .sdk_config import (
    ComprehensiveSDKConfig,
    SDKConfigManager,
    get_sdk_config_manager,
    get_agent_sdk_config
)

from .lifecycle_manager import (
    AgentLifecycleManager,
    get_lifecycle_manager
)

from .communication import (
    AgentCommunicationManager,
    RichMessageProcessor,
    get_communication_manager,
    shutdown_all_communication_managers
)

from .performance_monitor import (
    AgentPerformanceMonitor,
    get_performance_monitor
)

# Version and metadata
__version__ = "2.0.0"
__author__ = "Multi-Agent Research System Team"
__description__ = "Enhanced Agents with Comprehensive Claude SDK Integration"


# Convenience functions for common operations

async def create_enhanced_agent(agent_type: AgentType,
                              agent_id: Optional[str] = None,
                              config_overrides: Optional[dict] = None,
                              auto_initialize: bool = True) -> EnhancedBaseAgent:
    """Create an enhanced agent with sensible defaults."""
    factory = get_agent_factory()
    await factory.start()

    request = AgentCreationRequest(
        agent_type=agent_type,
        agent_id=agent_id,
        config_overrides=config_overrides or {},
        auto_initialize=auto_initialize
    )

    return await factory.create_agent(request)


async def create_agent_workflow(workflow_config: dict) -> list[EnhancedBaseAgent]:
    """Create a complete agent workflow."""
    factory = get_agent_factory()
    await factory.start()

    return await factory.create_agent_workflow(workflow_config)


def get_agent_configuration(agent_type: str,
                           preset: Optional[str] = None,
                           overrides: Optional[dict] = None) -> ComprehensiveSDKConfig:
    """Get configuration for an agent type."""
    config_manager = get_sdk_config_manager()
    return config_manager.get_config_for_agent(agent_type, preset, overrides)


# Module-level initialization
def initialize_enhanced_agents(config_dir: Optional[str] = None,
                             persistence_dir: Optional[str] = None) -> None:
    """Initialize enhanced agents system with optional configuration."""
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    logger.info("Initializing enhanced agents system")

    # Initialize configuration directory
    if config_dir:
        config_path = Path(config_dir)
    else:
        config_path = Path("config/enhanced_agents")

    # Initialize persistence directory
    if persistence_dir:
        persistence_path = Path(persistence_dir)
    else:
        persistence_path = Path("data/enhanced_agents")

    # Create directories if they don't exist
    config_path.mkdir(parents=True, exist_ok=True)
    persistence_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Enhanced agents system initialized - Config: {config_path}, Persistence: {persistence_path}")


# Export public API
__all__ = [
    # Core classes
    "EnhancedBaseAgent",
    "AgentConfiguration",
    "AgentPerformanceMetrics",
    "RichMessage",
    "AgentStatus",
    "AgentPriority",

    # Factory and creation
    "EnhancedAgentFactory",
    "AgentCreationRequest",
    "AgentType",
    "AgentTemplate",
    "get_agent_factory",
    "create_enhanced_agent",
    "create_agent_workflow",

    # Configuration
    "ComprehensiveSDKConfig",
    "SDKConfigManager",
    "get_sdk_config_manager",
    "get_agent_sdk_config",
    "get_agent_configuration",

    # Lifecycle management
    "AgentLifecycleManager",
    "get_lifecycle_manager",

    # Communication
    "AgentCommunicationManager",
    "RichMessageProcessor",
    "get_communication_manager",
    "shutdown_all_communication_managers",

    # Performance monitoring
    "AgentPerformanceMonitor",
    "get_performance_monitor",

    # System initialization
    "initialize_enhanced_agents",

    # Metadata
    "__version__",
    "__author__",
    "__description__"
]