"""Multi-Agent Research System

A comprehensive research workflow system using the Claude Agent SDK.
This system coordinates multiple specialized agents to conduct research,
generate reports, and provide editorial review.
"""

__version__ = "1.0.0"
__author__ = "Multi-Agent Research System"

from .core.orchestrator import ResearchOrchestrator
from .config.agents import get_all_agent_definitions

__all__ = ["ResearchOrchestrator", "get_all_agent_definitions"]