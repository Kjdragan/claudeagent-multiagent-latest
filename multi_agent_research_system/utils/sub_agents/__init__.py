"""
Sub-Agent Architecture for Multi-Agent Research System

This module provides a comprehensive sub-agent architecture that enables
specialized agents with context isolation, proper tool assignment, and
coordinated workflows using Claude Agent SDK patterns.

Key Features:
- Context isolation between sub-agents
- Specialized tool assignment and configuration
- Coordinated workflows with proper handoffs
- Performance monitoring and optimization
- Error recovery and resilience
"""

from .sub_agent_factory import SubAgentFactory
from .sub_agent_coordinator import SubAgentCoordinator
from .sub_agent_types import SubAgentType, create_sub_agent_config
from .context_isolation import ContextIsolationManager
from .communication_protocols import SubAgentCommunicationManager
from .performance_monitor import SubAgentPerformanceMonitor

__all__ = [
    'SubAgentFactory',
    'SubAgentCoordinator',
    'SubAgentType',
    'create_sub_agent_config',
    'ContextIsolationManager',
    'SubAgentCommunicationManager',
    'SubAgentPerformanceMonitor'
]

__version__ = "2.0.0"
__author__ = "Multi-Agent Research System Team"