"""Configuration modules for the multi-agent research system."""

from .agents import (
    create_agent_config_file,
    get_all_agent_definitions,
    get_editor_agent_definition,
    get_report_agent_definition,
    get_research_agent_definition,
    get_ui_coordinator_definition,
)

__all__ = [
    "get_research_agent_definition", "get_report_agent_definition",
    "get_editor_agent_definition", "get_ui_coordinator_definition",
    "get_all_agent_definitions", "create_agent_config_file"
]
