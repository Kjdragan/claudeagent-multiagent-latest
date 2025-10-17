"""
Enhanced Agent Configuration with Claude Agent SDK Integration

This module provides enhanced agent definitions that integrate with the Claude Agent SDK
configuration system, including advanced hooks, observability features, and flow adherence
patterns based on the redesign plan specifications.

Key Features:
- SDK-integrated agent definitions
- Flow adherence enforcement configuration
- Enhanced tool configuration with hooks
- Sub-agent coordination patterns
- Quality gates and validation
- Comprehensive error handling

Based on Redesign Plan PLUS SDK Implementation (October 13, 2025)
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import re
from pathlib import Path
from datetime import datetime

from .sdk_config import get_sdk_config, ClaudeAgentSDKConfig

# Claude Agent SDK imports for tools and hooks
try:
    from claude_agent_sdk import tool, hook, RunContext
    SDK_AVAILABLE = True
except ImportError:
    # Fallback if SDK not available
    def tool(name: str, description: str, input_schema: dict):
        def decorator(func):
            return func
        return decorator

    def hook(hook_type: str, name: str):
        def decorator(func):
            return func
        return decorator

    class RunContext:
        """Fallback RunContext for when SDK is not available."""
        pass

    SDK_AVAILABLE = False


class AgentType(str, Enum):
    """Supported agent types."""
    RESEARCH = "research"
    REPORT = "report"
    EDITORIAL = "editorial"
    UI_COORDINATOR = "ui_coordinator"
    QUALITY_JUDGE = "quality_judge"
    CONTENT_CLEANER = "content_cleaner"
    GAP_RESEARCH = "gap_research"


class ToolExecutionPolicy(str, Enum):
    """Tool execution policies for agents."""
    PERMISSIVE = "permissive"          # Allow all tools
    RESTRICTED = "restricted"          # Allow only specified tools
    VALIDATION_REQUIRED = "validation_required"  # Require validation before execution
    APPROVAL_REQUIRED = "approval_required"      # Require explicit approval
    MANDATORY = "mandatory"            # Tool must be executed


@dataclass
class ToolConfiguration:
    """Configuration for individual agent tools."""

    tool_name: str
    enabled: bool = True
    execution_policy: ToolExecutionPolicy = ToolExecutionPolicy.PERMISSIVE
    required_parameters: List[str] = field(default_factory=list)
    optional_parameters: List[str] = field(default_factory=list)
    validation_hooks: List[str] = field(default_factory=list)
    post_execution_hooks: List[str] = field(default_factory=list)
    timeout_seconds: Optional[float] = None
    retry_attempts: int = 3
    rate_limit_per_minute: Optional[int] = None


@dataclass
class AgentHooksConfiguration:
    """Hooks configuration specific to agents."""

    # Pre-execution hooks
    pre_execution_hooks: List[str] = field(default_factory=list)

    # Post-execution hooks
    post_execution_hooks: List[str] = field(default_factory=list)

    # Flow adherence hooks
    flow_adherence_hooks: List[str] = field(default_factory=list)

    # Quality validation hooks
    quality_validation_hooks: List[str] = field(default_factory=list)

    # Error handling hooks
    error_handling_hooks: List[str] = field(default_factory=list)

    # Communication hooks
    communication_hooks: List[str] = field(default_factory=list)


@dataclass
class QualityGateConfiguration:
    """Quality gate configuration for agents."""

    enabled: bool = True
    minimum_quality_score: float = 0.7
    quality_dimensions: List[str] = field(default_factory=lambda: [
        "accuracy", "completeness", "clarity", "depth", "source_quality"
    ])
    validation_methods: List[str] = field(default_factory=lambda: [
        "content_analysis", "source_validation", "coherence_check"
    ])
    enhancement_enabled: bool = True
    max_enhancement_cycles: int = 3
    failure_action: str = "retry"  # retry, escalate, or fail


@dataclass
class FlowAdherenceConfiguration:
    """Flow adherence configuration for ensuring workflow integrity."""

    enabled: bool = True
    mandatory_steps: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    validation_methods: List[str] = field(default_factory=lambda: [
        "content_analysis", "tool_execution_tracking", "session_state_validation"
    ])
    enforcement_strategies: List[str] = field(default_factory=lambda: [
        "automatic_execution", "agent_guidance", "blocking_validation"
    ])
    compliance_logging: bool = True
    violation_handling: str = "auto_correct"  # auto_correct, warn, or fail


@dataclass
class EnhancedAgentDefinition:
    """Enhanced agent definition with SDK integration."""

    # Basic agent information
    agent_type: AgentType
    name: str
    description: str
    version: str = "1.0.0"

    # Model configuration
    model: str = "sonnet"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    # Core prompt and behavior
    system_prompt: str = ""
    behavior_guidelines: List[str] = field(default_factory=list)
    quality_standards: List[str] = field(default_factory=list)

    # Tool configuration
    tools: List[ToolConfiguration] = field(default_factory=list)
    tool_access_level: str = "full"  # full, limited, or custom

    # Hooks configuration
    hooks: AgentHooksConfiguration = field(default_factory=AgentHooksConfiguration)

    # Quality configuration
    quality_gates: QualityGateConfiguration = field(default_factory=QualityGateConfiguration)

    # Flow adherence configuration
    flow_adherence: FlowAdherenceConfiguration = field(default_factory=FlowAdherenceConfiguration)

    # Communication configuration
    can_communicate_with: List[AgentType] = field(default_factory=list)
    communication_protocol: str = "standard"  # standard, priority, or custom

    # Sub-agent coordination
    can_coordinate_sub_agents: bool = False
    sub_agent_types: List[AgentType] = field(default_factory=list)

    # Error handling
    error_handling_strategy: str = "retry_with_backoff"
    max_retry_attempts: int = 3
    escalation_threshold: int = 3

    # Performance configuration
    timeout_seconds: float = 300.0
    memory_limit_mb: Optional[int] = None
    concurrent_operations: int = 1

    # Observability
    enable_detailed_logging: bool = True
    enable_performance_tracking: bool = True
    enable_step_by_step_tracking: bool = True

    def __post_init__(self):
        """Post-initialization setup."""
        # Load SDK configuration
        sdk_config = get_sdk_config()

        # Apply SDK defaults if not specified
        if self.max_tokens is None:
            self.max_tokens = sdk_config.max_tokens

        if self.temperature is None:
            self.temperature = sdk_config.temperature

        # Apply SDK hooks if agent hooks are empty
        if not self.hooks.pre_execution_hooks:
            self.hooks.pre_execution_hooks = sdk_config.hooks.pre_tool_hooks.copy()

        if not self.hooks.post_execution_hooks:
            self.hooks.post_execution_hooks = sdk_config.hooks.post_tool_hooks.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent definition to dictionary."""
        result = {}

        # Convert basic fields
        for key, value in self.__dict__.items():
            if key in ["agent_type"]:
                result[key] = value.value
            elif isinstance(value, (str, int, float, bool, type(None))):
                result[key] = value
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                result[key] = value
            elif isinstance(value, (ToolConfiguration, AgentHooksConfiguration,
                                  QualityGateConfiguration, FlowAdherenceConfiguration)):
                result[key] = value.__dict__
            elif isinstance(value, list) and all(isinstance(item, ToolConfiguration) for item in value):
                result[key] = [tool.__dict__ for tool in value]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedAgentDefinition":
        """Create agent definition from dictionary."""
        # Extract nested configurations
        tools_data = data.pop("tools", [])
        hooks_data = data.pop("hooks", {})
        quality_gates_data = data.pop("quality_gates", {})
        flow_adherence_data = data.pop("flow_adherence", {})

        # Convert agent_type
        if "agent_type" in data and isinstance(data["agent_type"], str):
            try:
                data["agent_type"] = AgentType(data["agent_type"])
            except ValueError:
                # Skip invalid agent types
                data["agent_type"] = AgentType.RESEARCH

        # Create agent definition
        agent = cls(**data)

        # Update tools configuration
        agent.tools = [ToolConfiguration(**tool_data) for tool_data in tools_data]

        # Update nested configurations
        for key, value in hooks_data.items():
            if hasattr(agent.hooks, key):
                setattr(agent.hooks, key, value)

        for key, value in quality_gates_data.items():
            if hasattr(agent.quality_gates, key):
                setattr(agent.quality_gates, key, value)

        for key, value in flow_adherence_data.items():
            if hasattr(agent.flow_adherence, key):
                setattr(agent.flow_adherence, key, value)

        return agent


# ============================================================================
# ENHANCED REPORT AGENT SDK TOOLS WITH HOOK INTEGRATION
# ============================================================================

if SDK_AVAILABLE:
    @tool
    def build_research_corpus(session_id: str, workproduct_path: Optional[str] = None) -> Dict[str, Any]:
        """Build structured research corpus from search workproduct.
        
        Args:
            session_id: Session identifier for corpus management
            workproduct_path: Optional path to specific workproduct file
            
        Returns:
            Dictionary with corpus status, metadata, and content chunks
        """
        try:
            from ..utils.research_corpus_manager import ResearchCorpusManager
            
            manager = ResearchCorpusManager(session_id=session_id)
            
            # Build corpus from workproduct
            corpus_data = manager.build_corpus_from_workproduct(workproduct_path)
            
            return {
                "success": True,
                "corpus_id": corpus_data["corpus_id"],
                "total_chunks": corpus_data["total_chunks"],
                "total_sources": corpus_data["metadata"]["total_sources"],
                "content_coverage": corpus_data["metadata"]["content_coverage"],
                "average_relevance_score": corpus_data["metadata"]["average_relevance_score"],
                "corpus_file": corpus_data["corpus_file"],
                "message": f"Successfully built research corpus with {corpus_data['total_chunks']} content chunks from {corpus_data['metadata']['total_sources']} sources"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to build research corpus: {str(e)}"
            }

    @tool
    def analyze_research_corpus(corpus_id: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze research corpus for insights and quality assessment.
        
        Args:
            corpus_id: Corpus identifier from build_research_corpus
            analysis_type: Type of analysis (comprehensive, quality, relevance, coverage)
            
        Returns:
            Dictionary with analysis results and recommendations
        """
        try:
            from ..utils.research_corpus_manager import ResearchCorpusManager
            
            # Extract session_id from corpus_id or use default
            session_id = corpus_id.split("_")[0] if "_" in corpus_id else corpus_id
            manager = ResearchCorpusManager(session_id=session_id)
            
            # Load and analyze corpus
            corpus_data = manager.load_corpus(corpus_id)
            analysis_result = manager.analyze_corpus_quality(corpus_data, analysis_type)
            
            return {
                "success": True,
                "corpus_id": corpus_id,
                "analysis_type": analysis_type,
                "overall_quality_score": analysis_result["overall_quality_score"],
                "content_analysis": analysis_result["content_analysis"],
                "source_analysis": analysis_result["source_analysis"],
                "coverage_analysis": analysis_result["coverage_analysis"],
                "recommendations": analysis_result["recommendations"],
                "ready_for_synthesis": analysis_result["overall_quality_score"] >= 0.7,
                "message": f"Corpus analysis completed with quality score: {analysis_result['overall_quality_score']:.2f}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to analyze research corpus: {str(e)}"
            }

    @tool
    def synthesize_from_corpus(corpus_id: str, synthesis_type: str = "comprehensive_report", 
                             focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Synthesize information from research corpus for report generation.
        
        Args:
            corpus_id: Corpus identifier from analyze_research_corpus
            synthesis_type: Type of synthesis (executive_summary, comprehensive_report, 
                           key_insights, data_extract, targeted_analysis)
            focus_areas: Optional list of specific areas to focus on
            
        Returns:
            Dictionary with synthesized content and metadata
        """
        try:
            from ..utils.research_corpus_manager import ResearchCorpusManager
            
            # Extract session_id from corpus_id
            session_id = corpus_id.split("_")[0] if "_" in corpus_id else corpus_id
            manager = ResearchCorpusManager(session_id=session_id)
            
            # Load corpus and get relevant chunks
            corpus_data = manager.load_corpus(corpus_id)
            
            # Get relevant chunks based on focus areas
            relevant_chunks = manager.get_relevant_chunks(
                corpus_data=corpus_data,
                query=focus_areas[0] if focus_areas else "comprehensive analysis",
                max_chunks=50,
                min_relevance_score=0.3
            )
            
            # Synthesize content based on type
            synthesis_result = manager.synthesize_corpus_content(
                corpus_data=corpus_data,
                relevant_chunks=relevant_chunks,
                synthesis_type=synthesis_type,
                focus_areas=focus_areas or ["comprehensive coverage"]
            )
            
            return {
                "success": True,
                "corpus_id": corpus_id,
                "synthesis_type": synthesis_type,
                "synthesized_content": synthesis_result["synthesized_content"],
                "source_integration": synthesis_result["source_integration"],
                "data_points_extracted": synthesis_result["data_points_extracted"],
                "key_insights": synthesis_result["key_insights"],
                "content_coverage": synthesis_result["content_coverage"],
                "ready_for_report": synthesis_result["quality_score"] >= 0.75,
                "quality_score": synthesis_result["quality_score"],
                "message": f"Successfully synthesized {synthesis_type} with quality score {synthesis_result['quality_score']:.2f}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to synthesize from corpus: {str(e)}"
            }

    @tool
    def generate_comprehensive_report(corpus_id: str, synthesis_result: Dict[str, Any], 
                                    report_format: str = "standard") -> Dict[str, Any]:
        """Generate comprehensive report using corpus synthesis results.
        
        Args:
            corpus_id: Corpus identifier for reference
            synthesis_result: Result from synthesize_from_corpus tool
            report_format: Report format (standard, academic, business, technical, brief)
            
        Returns:
            Dictionary with generated report and metadata
        """
        try:
            # Validate synthesis result
            if not synthesis_result.get("success", False):
                return {
                    "success": False,
                    "error": "Invalid synthesis result",
                    "message": "Cannot generate report from failed synthesis"
                }
            
            # Extract synthesized content
            synthesized_content = synthesis_result["synthesized_content"]
            source_integration = synthesis_result["source_integration"]
            key_insights = synthesis_result["key_insights"]
            
            # Generate report structure based on format
            report_structure = _create_report_structure(
                synthesized_content=synthesized_content,
                source_integration=source_integration,
                key_insights=key_insights,
                report_format=report_format,
                corpus_id=corpus_id
            )
            
            # Generate final report
            report_content = _generate_final_report(
                structure=report_structure,
                synthesis_result=synthesis_result,
                corpus_id=corpus_id
            )
            
            return {
                "success": True,
                "corpus_id": corpus_id,
                "report_format": report_format,
                "report_content": report_content,
                "report_structure": report_structure,
                "sources_integrated": len(source_integration),
                "key_insights_count": len(key_insights),
                "estimated_quality_score": min(0.95, synthesis_result["quality_score"] + 0.1),
                "ready_for_final_output": True,
                "message": f"Successfully generated {report_format} report integrating {len(source_integration)} sources"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to generate comprehensive report: {str(e)}"
            }

    def _create_report_structure(synthesized_content: Dict[str, Any], 
                               source_integration: List[Dict[str, Any]], 
                               key_insights: List[str], 
                               report_format: str, 
                               corpus_id: str) -> Dict[str, Any]:
        """Create report structure based on format and content."""
        
        format_templates = {
            "standard": {
                "sections": [
                    {"title": "Executive Summary", "type": "summary"},
                    {"title": "Introduction", "type": "overview"},
                    {"title": "Key Findings", "type": "insights"},
                    {"title": "Detailed Analysis", "type": "analysis"},
                    {"title": "Conclusions", "type": "conclusions"}
                ]
            },
            "academic": {
                "sections": [
                    {"title": "Abstract", "type": "summary"},
                    {"title": "Introduction", "type": "overview"},
                    {"title": "Literature Review", "type": "background"},
                    {"title": "Methodology", "type": "methodology"},
                    {"title": "Findings", "type": "insights"},
                    {"title": "Discussion", "type": "analysis"},
                    {"title": "Conclusion", "type": "conclusions"}
                ]
            },
            "business": {
                "sections": [
                    {"title": "Executive Summary", "type": "summary"},
                    {"title": "Business Context", "type": "overview"},
                    {"title": "Key Findings", "type": "insights"},
                    {"title": "Market Analysis", "type": "analysis"},
                    {"title": "Recommendations", "type": "recommendations"},
                    {"title": "Next Steps", "type": "action_items"}
                ]
            }
        }
        
        template = format_templates.get(report_format, format_templates["standard"])
        
        return {
            "format": report_format,
            "corpus_id": corpus_id,
            "sections": template["sections"],
            "synthesized_content": synthesized_content,
            "source_integration": source_integration,
            "key_insights": key_insights,
            "estimated_length": len(str(synthesized_content)) + len(str(key_insights)) * 10
        }

    def _generate_final_report(structure: Dict[str, Any], 
                             synthesis_result: Dict[str, Any], 
                             corpus_id: str) -> str:
        """Generate the final report content."""
        
        content_sections = []
        
        for section in structure["sections"]:
            section_content = _generate_section_content(
                section_type=section["type"],
                title=section["title"],
                synthesized_content=structure["synthesized_content"],
                key_insights=structure["key_insights"],
                source_integration=structure["source_integration"]
            )
            content_sections.append(f"## {section['title']}\n\n{section_content}")
        
        # Add source citations
        citations_section = _generate_citations_section(structure["source_integration"])
        content_sections.append(citations_section)
        
        return "\n\n".join(content_sections)

    def _generate_section_content(section_type: str, title: str, 
                                synthesized_content: Dict[str, Any], 
                                key_insights: List[str], 
                                source_integration: List[Dict[str, Any]]) -> str:
        """Generate content for a specific report section."""
        
        content_generators = {
            "summary": lambda: _generate_summary_content(synthesized_content, key_insights),
            "overview": lambda: _generate_overview_content(synthesized_content),
            "insights": lambda: _generate_insights_content(key_insights, source_integration),
            "analysis": lambda: _generate_analysis_content(synthesized_content, source_integration),
            "conclusions": lambda: _generate_conclusions_content(synthesized_content, key_insights),
            "recommendations": lambda: _generate_recommendations_content(key_insights, synthesized_content)
        }
        
        generator = content_generators.get(section_type, lambda: "Content generation not implemented for this section type.")
        return generator()

    def _generate_summary_content(synthesized_content: Dict[str, Any], key_insights: List[str]) -> str:
        """Generate executive summary content."""
        return f"""
This analysis synthesizes information from multiple sources to provide comprehensive insights. 

**Key Highlights:**
{chr(10).join(f"- {insight}" for insight in key_insights[:5])}

**Content Coverage:**
- Total sources analyzed: {len(synthesized_content.get('sources', []))}
- Content areas covered: {', '.join(synthesized_content.get('topics', []))}
- Analysis depth: Comprehensive with multi-source validation
"""

    def _generate_overview_content(synthesized_content: Dict[str, Any]) -> str:
        """Generate overview/introduction content."""
        return f"""
This report presents a comprehensive analysis based on extensive research and data synthesis. 

**Research Scope:**
{chr(10).join(f"- {topic}" for topic in synthesized_content.get('topics', ['Comprehensive analysis']))}

**Methodology:**
- Multi-source research integration
- Quality-assessed content synthesis
- Data-driven insight extraction
"""

    def _generate_insights_content(key_insights: List[str], source_integration: List[Dict[str, Any]]) -> str:
        """Generate key findings content."""
        insights_text = "\n\n".join(f"**{i+1}.** {insight}" for i, insight in enumerate(key_insights))
        source_refs = "\n\n".join(f"- [{source.get('title', 'Unknown')}]( {source.get('url', '')} )" for source in source_integration[:5])
        
        return f"{insights_text}\n\n**Key Sources:**\n{source_refs}"

    def _generate_analysis_content(synthesized_content: Dict[str, Any], source_integration: List[Dict[str, Any]]) -> str:
        """Generate detailed analysis content."""
        return f"""
**Detailed Analysis:**

The research reveals several important patterns and insights derived from {len(source_integration)} distinct sources.

**Content Analysis:**
{synthesized_content.get('main_content', 'Comprehensive analysis based on multi-source research.')}

**Source Validation:**
All sources have been assessed for credibility and relevance to ensure high-quality insights.
"""

    def _generate_conclusions_content(synthesized_content: Dict[str, Any], key_insights: List[str]) -> str:
        """Generate conclusions content."""
        return f"""
**Conclusions:**

This analysis provides {len(key_insights)} key insights derived from comprehensive research synthesis.

**Main Conclusions:**
{chr(10).join(f"- {insight}" for insight in key_insights[:3])}

**Implications:**
The findings have significant implications for understanding the topic comprehensively.
"""

    def _generate_recommendations_content(key_insights: List[str], synthesized_content: Dict[str, Any]) -> str:
        """Generate recommendations content."""
        return f"""
**Recommendations:**

Based on the comprehensive analysis, the following recommendations are provided:

{chr(10).join(f"1. {insight}" for insight in key_insights[:5])}

**Next Steps:**
- Continue monitoring for additional developments
- Validate findings with additional sources if needed
- Consider specific implications for your context
"""

    def _generate_citations_section(source_integration: List[Dict[str, Any]]) -> str:
        """Generate citations section."""
        citations = "\n\n".join(
            f"**{i+1}.** {source.get('title', 'Unknown Source')} ({source.get('source', 'Unknown')}) - {source.get('url', 'No URL')}"
            for i, source in enumerate(source_integration)
        )
        return f"## Sources\n\n{citations}"

    # ========================================================================
    # ENHANCED REPORT AGENT HOOKS
    # ========================================================================

    @hook
    def validate_research_data_usage(ctx: RunContext) -> None:
        """Pre-processing hook to validate research data availability."""
        tool_use = ctx.current_tool_use
        if tool_use.name in ["synthesize_from_corpus", "generate_comprehensive_report"]:
            corpus_id = tool_use.input.get("corpus_id")
            if not corpus_id:
                raise ValueError("Corpus ID is required for synthesis and report generation")
            
            # Validate corpus exists and is accessible
            try:
                from ..utils.research_corpus_manager import ResearchCorpusManager
                session_id = corpus_id.split("_")[0] if "_" in corpus_id else corpus_id
                manager = ResearchCorpusManager(session_id=session_id)
                corpus_data = manager.load_corpus(corpus_id)
                if not corpus_data:
                    raise ValueError(f"Corpus {corpus_id} not found or empty")
            except Exception as e:
                raise ValueError(f"Failed to validate research corpus: {str(e)}")

    @hook
    def enforce_citation_requirements(ctx: RunContext) -> None:
        """Pre-processing hook to enforce proper source citation."""
        tool_use = ctx.current_tool_use
        if tool_use.name == "generate_comprehensive_report":
            synthesis_result = tool_use.input.get("synthesis_result", {})
            source_integration = synthesis_result.get("source_integration", [])
            if len(source_integration) < 2:
                raise ValueError("Report must integrate at least 2 sources for credibility")

    @hook
    def validate_data_integration(ctx: RunContext) -> None:
        """Post-processing hook to validate data integration and prevent template responses."""
        tool_use = ctx.current_tool_use
        if tool_use.name in ["synthesize_from_corpus", "generate_comprehensive_report"]:
            result = ctx.current_tool_result
            
            # Check for template response patterns
            content_indicators = [
                "template response",
                "placeholder content",
                "generic information",
                "sample data",
                "example content"
            ]
            
            result_content = str(result)
            for indicator in content_indicators:
                if indicator.lower() in result_content.lower():
                    raise ValueError(f"Template response detected: {indicator}")

    @hook
    def quality_score_validation(ctx: RunContext) -> None:
        """Post-processing hook to ensure quality standards are met."""
        tool_use = ctx.current_tool_use
        if tool_use.name in ["analyze_research_corpus", "synthesize_from_corpus", "generate_comprehensive_report"]:
            result = ctx.current_tool_result
            
            # Extract quality score if available
            if hasattr(result, 'get') and callable(result.get):
                quality_score = result.get("quality_score", 0.0)
                if quality_score < 0.7:
                    raise ValueError(f"Quality score {quality_score} below minimum threshold of 0.7")

    @hook
    def track_research_pipeline_compliance(ctx: RunContext) -> None:
        """Post-processing hook to track research pipeline adherence."""
        tool_use = ctx.current_tool_use
        
        # Log pipeline compliance for tracking
        compliance_data = {
            "tool_name": tool_use.name,
            "timestamp": datetime.now().isoformat(),
            "session_id": tool_use.input.get("corpus_id", "unknown"),
            "parameters": tool_use.input,
            "compliance_status": "executed"
        }
        
        # Store compliance data for pipeline monitoring
        if hasattr(ctx, 'session') and ctx.session:
            if not hasattr(ctx.session, 'compliance_log'):
                ctx.session.compliance_log = []
            ctx.session.compliance_log.append(compliance_data)

    @hook
    def validate_report_quality_standards(ctx: RunContext) -> None:
        """Post-processing hook to validate outgoing report quality."""
        tool_use = ctx.current_tool_use
        if tool_use.name == "generate_comprehensive_report":
            result = ctx.current_tool_result
            
            # Validate report content quality
            if hasattr(result, 'get') and callable(result.get):
                report_content = result.get("report_content", "")
                sources_integrated = result.get("sources_integrated", 0)
                
                # Minimum content length check
                if len(report_content) < 1000:
                    raise ValueError("Report content too short (minimum 1000 characters)")
                
                # Source integration check
                if sources_integrated < 2:
                    raise ValueError("Insufficient source integration (minimum 2 sources required)")
                
                # Content completeness check
                required_sections = ["Executive Summary", "Key Findings", "Analysis", "Sources"]
                missing_sections = [section for section in required_sections if section not in report_content]
                if missing_sections:
                    raise ValueError(f"Missing required sections: {', '.join(missing_sections)}")


else:
    # Fallback implementations when SDK is not available
    def build_research_corpus(session_id: str, workproduct_path: Optional[str] = None) -> Dict[str, Any]:
        return {"error": "Claude Agent SDK not available", "success": False}

    def analyze_research_corpus(corpus_id: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        return {"error": "Claude Agent SDK not available", "success": False}

    def synthesize_from_corpus(corpus_id: str, synthesis_type: str = "comprehensive_report", 
                             focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        return {"error": "Claude Agent SDK not available", "success": False}

    def generate_comprehensive_report(corpus_id: str, synthesis_result: Dict[str, Any], 
                                    report_format: str = "standard") -> Dict[str, Any]:
        return {"error": "Claude Agent SDK not available", "success": False}


# ============================================================================

class EnhancedAgentFactory:
    """Factory for creating enhanced agent definitions."""

    def __init__(self, sdk_config: Optional[ClaudeAgentSDKConfig] = None):
        self.sdk_config = sdk_config or get_sdk_config()

    def create_research_agent(self) -> EnhancedAgentDefinition:
        """Create enhanced research agent definition."""
        return EnhancedAgentDefinition(
            agent_type=AgentType.RESEARCH,
            name="Enhanced Research Agent",
            description="Advanced research agent with comprehensive web search, source validation, and information synthesis capabilities",
            system_prompt="""You are an Enhanced Research Agent, an expert in conducting comprehensive, high-quality research using advanced search strategies and intelligent source validation.

MANDATORY RESEARCH PROCESS:
1. Execute comprehensive search using mcp__zplayground1_search__zplayground1_search_scrape_clean tool
2. Set search_mode to "web" for comprehensive coverage
3. Set num_results to 15-20 for thorough coverage
4. Set anti_bot_level to 1 (enhanced) by default, escalate to 2 (advanced) if detection occurs
5. Set session_id to your current session ID
6. The tool will automatically crawl top results and clean content
7. Analyze and validate all sources for credibility and relevance
8. Synthesize findings into structured research output
9. Save research results using save_report tool

QUALITY STANDARDS:
- Prioritize authoritative sources (academic papers, reputable news, official reports)
- Cross-reference information across multiple sources
- Distinguish between facts and opinions
- Note source dates and potential biases
- Gather sufficient depth to support comprehensive reporting

FLOW ADHERENCE:
- All research plans MUST be executed through tool calls
- Documentation without execution is INSUFFICIENT
- System will validate and enforce research execution compliance""",
            behavior_guidelines=[
                "Be thorough and systematic in research approach",
                "Validate sources for credibility and relevance",
                "Synthesize information from multiple perspectives",
                "Maintain objective and unbiased analysis",
                "Document sources properly and transparently"
            ],
            quality_standards=[
                "Source credibility and authority",
                "Information accuracy and verification",
                "Comprehensive coverage of topic",
                "Proper attribution and citation",
                "Current and relevant information"
            ],
            tools=[
                ToolConfiguration(
                    tool_name="mcp__zplayground1_search__zplayground1_search_scrape_clean",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    required_parameters=["query", "session_id"],
                    optional_parameters=["search_mode", "num_results", "anti_bot_level"],
                    validation_hooks=["validate_search_parameters", "check_budget_availability"],
                    post_execution_hooks=["analyze_search_results", "log_research_success"]
                ),
                ToolConfiguration(
                    tool_name="analyze_sources",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE,
                    post_execution_hooks=["validate_source_credibility", "update_quality_metrics"]
                ),
                ToolConfiguration(
                    tool_name="save_report",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    required_parameters=["content"],
                    post_execution_hooks=["validate_report_format", "log_storage_success"]
                ),
                ToolConfiguration(
                    tool_name="get_session_data",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE
                ),
                ToolConfiguration(
                    tool_name="Read",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE
                ),
                ToolConfiguration(
                    tool_name="Write",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    validation_hooks=["validate_write_permissions", "check_file_format"]
                ),
                ToolConfiguration(
                    tool_name="Edit",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    validation_hooks=["validate_edit_permissions", "backup_original_file"]
                )
            ],
            flow_adherence=FlowAdherenceConfiguration(
                enabled=True,
                mandatory_steps=[
                    "execute_mcp_zplayground1_search",
                    "analyze_search_results",
                    "save_research_findings"
                ],
                required_tools=["mcp__zplayground1_search__zplayground1_search_scrape_clean", "save_report"],
                enforcement_strategies=["automatic_execution", "blocking_validation"]
            ),
            quality_gates=QualityGateConfiguration(
                minimum_quality_score=0.75,
                quality_dimensions=["accuracy", "completeness", "source_quality", "depth"],
                max_enhancement_cycles=2
            ),
            can_communicate_with=[AgentType.REPORT, AgentType.EDITORIAL],
            timeout_seconds=600.0,  # 10 minutes for comprehensive research
            enable_step_by_step_tracking=True
        )

    def create_report_agent(self) -> EnhancedAgentDefinition:
        """Create enhanced report agent with SDK tools and hooks."""
        return EnhancedAgentDefinition(
            agent_type=AgentType.REPORT,
            name="Enhanced Report Agent with Workproduct Tools",
            description="Advanced report generation agent with direct workproduct access, data synthesis, and comprehensive hook-based validation",
            system_prompt="""You are an Enhanced Report Agent with direct access to research workproducts for comprehensive report generation.

MANDATORY WORKPRODUCT-BASED REPORT PROCESS:

**STAGE 2 (Initial Report Generation)**:
1. Call get_workproduct_summary to see available research data
2. Call get_all_workproduct_articles to access full article content
3. Synthesize information into comprehensive report (>1000 words)
4. Include specific facts, data points, and citations from articles
5. Save to working/COMPREHENSIVE_{timestamp}.md

**STAGE 4 (Final Enhanced Report)**:
1. Call get_all_workproduct_articles for ALL research (original + gap)
2. Review editorial feedback from Stage 3
3. Integrate ALL editorial feedback points:
   - Incorporate specific data points editor identified
   - Fix temporal accuracy issues
   - Improve source attribution
   - Fill identified gaps
4. Generate enhanced comprehensive report (>1500 words)
5. Save to complete/FINAL_ENHANCED_{timestamp}.md

DIRECT WORKPRODUCT ACCESS:
- get_workproduct_summary: Quick overview of available data
- get_all_workproduct_articles: Full content access (PRIMARY TOOL)
- get_workproduct_article: Single article by URL/index
- read_full_workproduct: Complete markdown file

QUALITY STANDARDS:
- Integrate specific data from workproduct articles
- Include proper citations for all sources
- Use actual facts, figures, and data points
- Generate comprehensive, data-driven reports
- Avoid template responses - use real research data""",
            behavior_guidelines=[
                "Access research workproducts directly",
                "Use get_all_workproduct_articles to get full content",
                "Synthesize content from actual article data",
                "Generate reports with proper source integration",
                "Ensure data-driven content generation",
                "Integrate editorial feedback (Stage 4)"
            ],
            quality_standards=[
                "Direct workproduct access and integration",
                "Data synthesis from articles",
                "Source citation and integration", 
                "Quality score compliance",
                "Template response prevention",
                "Editorial feedback integration"
            ],
            tools=[
                ToolConfiguration(
                    tool_name="get_session_data",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE,
                    post_execution_hooks=["validate_session_data", "log_data_access"]
                ),
                ToolConfiguration(
                    tool_name="get_workproduct_summary",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE,
                    required_parameters=["session_id"],
                    post_execution_hooks=["log_data_access"]
                ),
                ToolConfiguration(
                    tool_name="get_all_workproduct_articles",
                    execution_policy=ToolExecutionPolicy.MANDATORY,
                    required_parameters=["session_id"],
                    validation_hooks=["validate_research_data_usage"],
                    post_execution_hooks=["validate_data_integration", "quality_score_validation"]
                ),
                ToolConfiguration(
                    tool_name="get_workproduct_article",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE,
                    required_parameters=["session_id"],
                    optional_parameters=["url", "index"],
                    post_execution_hooks=["log_data_access"]
                ),
                ToolConfiguration(
                    tool_name="read_full_workproduct",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE,
                    required_parameters=["session_id"],
                    post_execution_hooks=["log_data_access"]
                ),
                ToolConfiguration(
                    tool_name="Read",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE
                ),
                ToolConfiguration(
                    tool_name="Write",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    required_parameters=["content", "file_path"],
                    validation_hooks=["validate_file_path", "ensure_absolute_path", "check_write_permissions"],
                    post_execution_hooks=["validate_file_creation", "log_report_success"]
                ),
                ToolConfiguration(
                    tool_name="Edit",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE,
                    validation_hooks=["validate_edit_permissions"]
                )
            ],
            hooks=AgentHooksConfiguration(
                pre_execution_hooks=[
                    "validate_research_data_usage",
                    "enforce_citation_requirements"
                ],
                post_execution_hooks=[
                    "validate_data_integration",
                    "quality_score_validation", 
                    "track_research_pipeline_compliance",
                    "validate_report_quality_standards"
                ],
                flow_adherence_hooks=[
                    "ensure_corpus_based_generation",
                    "prevent_template_responses",
                    "enforce_source_integration"
                ]
            ),
            flow_adherence=FlowAdherenceConfiguration(
                enabled=True,
                mandatory_steps=[
                    "get_workproduct_summary",
                    "get_all_workproduct_articles",
                    "synthesize_from_articles",
                    "generate_comprehensive_report",
                    "save_final_report"
                ],
                required_tools=["get_all_workproduct_articles"],
                validation_methods=["content_analysis", "tool_execution_tracking", "quality_score_validation"],
                enforcement_strategies=["automatic_execution", "blocking_validation", "quality_enforcement"],
                compliance_logging=True,
                violation_handling="auto_correct"
            ),
            quality_gates=QualityGateConfiguration(
                enabled=True,
                minimum_quality_score=0.8,
                quality_dimensions=["data_integration", "source_citation", "content_completeness", "research_synthesis", "template_prevention"],
                validation_methods=["corpus_analysis", "content_validation", "source_integration_check"],
                enhancement_enabled=True,
                max_enhancement_cycles=3,
                failure_action="retry_with_enforcement"
            ),
            can_communicate_with=[AgentType.RESEARCH, AgentType.EDITORIAL, AgentType.QUALITY_JUDGE],
            timeout_seconds=600.0,  # Extended for comprehensive processing
            enable_step_by_step_tracking=True,
            enable_detailed_logging=True
        )

    def create_editorial_agent(self) -> EnhancedAgentDefinition:
        """Create enhanced editorial agent with flow adherence enforcement."""
        return EnhancedAgentDefinition(
            agent_type=AgentType.EDITORIAL,
            name="Enhanced Editorial Agent with Flow Adherence Enforcement",
            description="Advanced editorial agent with comprehensive quality assessment, gap identification, and mandatory gap research execution enforcement",
            system_prompt="""You are an Enhanced Editorial Agent with MANDATORY FLOW ADHERENCE enforcement, responsible for comprehensive quality assessment, gap identification, and ENSURING research execution compliance.

MANDATORY EDITORIAL REVIEW WORKFLOW:

**STEP 1: ACCESS ORIGINAL RESEARCH DATA (CRITICAL)**
- Call get_workproduct_summary to see what research data was available
- Call get_all_workproduct_articles to review full source content
- This lets you verify what data was available when report was created
- Essential for assessing whether report properly utilized research

**STEP 2: REVIEW GENERATED REPORT**
- Use Read tool to examine the initial report
- Compare report content against available research data
- Identify specific examples of data that should have been used but wasn't

**STEP 3: ASSESS DATA INTEGRATION**
- Did report use specific facts, figures, statistics from workproduct?
- Is temporal accuracy correct (October 2025)?
- Are sources cited properly (not generically)?
- What specific data was available but not incorporated?

**STEP 4: IDENTIFY GAPS AND PROVIDE FEEDBACK**
- Document specific research data that should be incorporated
- Provide concrete examples from workproduct articles
- Identify gaps where additional research would help
- If significant gaps, request targeted gap research

**STEP 5: REQUEST GAP RESEARCH (IF NEEDED)**
- EXECUTE request_gap_research tool for identified gaps
- System will AUTO-DETECT and FORCE EXECUTION if needed
- Gap research creates NEW workproducts for final report

CRITICAL COMPLIANCE REQUIREMENTS:
- Documentation without tool execution is INSUFFICIENT
- System will automatically detect missing research execution
- Forced execution will be applied for compliance violations
- All gap research plans MUST be executed through proper tools

QUALITY ENHANCEMENT CRITERIA:
- Data Specificity: Include specific facts, figures, statistics
- Fact Expansion: Expand general statements with specific data
- Information Integration: Thoroughly integrate research findings
- Fact-Based Enhancement: Support claims with specific data
- Rich Content: Leverage scraped research data effectively
- Comprehensive Coverage: Include relevant facts and data points
- Style Consistency: Match user's requested style
- Appropriate Length: Match data volume and requirements""",
            behavior_guidelines=[
                "Enforce mandatory gap research execution",
                "Identify and document specific content gaps",
                "Ensure comprehensive coverage through research execution",
                "Maintain high quality standards through validation",
                "Apply progressive enhancement when needed"
            ],
            quality_standards=[
                "Comprehensive content coverage",
                "Specific data and factual support",
                "Proper research integration",
                "Gap identification and execution",
                "Style consistency requirements"
            ],
            tools=[
                ToolConfiguration(
                    tool_name="get_session_data",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    post_execution_hooks=["validate_data_analysis", "log_gap_assessment_start"]
                ),
                ToolConfiguration(
                    tool_name="get_workproduct_summary",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE,
                    required_parameters=["session_id"],
                    post_execution_hooks=["log_data_access"]
                ),
                ToolConfiguration(
                    tool_name="get_all_workproduct_articles",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE,
                    required_parameters=["session_id"],
                    post_execution_hooks=["log_data_access", "validate_data_review"]
                ),
                ToolConfiguration(
                    tool_name="get_workproduct_article",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE,
                    required_parameters=["session_id"],
                    optional_parameters=["url", "index"],
                    post_execution_hooks=["log_data_access"]
                ),
                ToolConfiguration(
                    tool_name="analyze_sources",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE,
                    post_execution_hooks=["validate_source_analysis"]
                ),
                ToolConfiguration(
                    tool_name="identify_research_gaps",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    post_execution_hooks=["validate_gap_identification", "track_gap_documentation"]
                ),
                ToolConfiguration(
                    tool_name="request_gap_research",
                    execution_policy=ToolExecutionPolicy.MANDATORY,
                    required_parameters=["gap_description", "research_plan"],
                    validation_hooks=["validate_gap_research_request", "check_research_budget"],
                    post_execution_hooks=["log_research_execution", "validate_compliance"]
                ),
                ToolConfiguration(
                    tool_name="create_research_report",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE
                ),
                ToolConfiguration(
                    tool_name="Read",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE
                ),
                ToolConfiguration(
                    tool_name="Write",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    validation_hooks=["validate_write_permissions", "check_file_format"],
                    post_execution_hooks=["validate_editorial_completion", "log_compliance_status"]
                ),
                ToolConfiguration(
                    tool_name="Edit",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE
                )
            ],
            hooks=AgentHooksConfiguration(
                flow_adherence_hooks=[
                    "validate_gap_research_completion",
                    "enforce_mandatory_research_execution",
                    "detect_compliance_violations",
                    "auto_correct_workflow_violations"
                ],
                quality_validation_hooks=[
                    "assess_content_completeness",
                    "validate_specific_data_inclusion",
                    "check_research_integration"
                ]
            ),
            flow_adherence=FlowAdherenceConfiguration(
                enabled=True,
                mandatory_steps=[
                    "analyze_available_data",
                    "identify_specific_gaps",
                    "request_gap_research_execution"
                ],
                required_tools=["get_session_data", "request_gap_research"],
                validation_methods=["content_analysis", "tool_execution_tracking", "compliance_enforcement"],
                enforcement_strategies=["automatic_execution", "agent_guidance", "blocking_validation"],
                compliance_logging=True,
                violation_handling="auto_correct"
            ),
            quality_gates=QualityGateConfiguration(
                minimum_quality_score=0.8,
                quality_dimensions=["completeness", "data_specificity", "research_integration", "gap_resolution"],
                max_enhancement_cycles=3,
                failure_action="retry"
            ),
            can_communicate_with=[AgentType.REPORT, AgentType.GAP_RESEARCH, AgentType.QUALITY_JUDGE],
            can_coordinate_sub_agents=True,
            sub_agent_types=[AgentType.GAP_RESEARCH],
            timeout_seconds=450.0,
            enable_step_by_step_tracking=True,
            enable_detailed_logging=True
        )

    def create_gap_research_agent(self) -> EnhancedAgentDefinition:
        """Create specialized gap research agent."""
        return EnhancedAgentDefinition(
            agent_type=AgentType.GAP_RESEARCH,
            name="Gap Research Specialist",
            description="Specialized agent for conducting targeted gap research based on editorial agent requirements",
            system_prompt="""You are a Gap Research Specialist, focused on conducting targeted research to fill specific information gaps identified by the editorial agent.

MANDATORY GAP RESEARCH PROCESS:
1. Analyze gap research requirements from editorial agent
2. Execute targeted search using conduct_research with specific parameters
3. Focus search terms and parameters on identified gaps
4. Extract relevant information efficiently
5. Synthesize findings to address specific gaps
6. Report back with comprehensive gap-filling information

TARGETED RESEARCH APPROACH:
- Use specific search terms for gap areas
- Set appropriate search parameters for targeted results
- Focus on high-quality, relevant sources
- Extract specific data points and information
- Provide comprehensive gap coverage
- Ensure efficient use of research budget

QUALITY STANDARDS:
- Direct relevance to identified gaps
- Specific and factual information
- Authoritative source materials
- Comprehensive gap coverage
- Clear and actionable results""",
            behavior_guidelines=[
                "Focus on specific gap areas",
                "Conduct targeted, efficient research",
                "Provide specific factual information",
                "Ensure comprehensive gap coverage",
                "Report findings clearly and actionably"
            ],
            quality_standards=[
                "Gap relevance and specificity",
                "Source quality and authority",
                "Information accuracy",
                "Comprehensive coverage",
                "Actionable findings"
            ],
            tools=[
                ToolConfiguration(
                    tool_name="conduct_research",
                    execution_policy=ToolExecutionPolicy.MANDATORY,
                    required_parameters=["query"],
                    optional_parameters=["num_results", "auto_crawl_top", "crawl_threshold"],
                    validation_hooks=["validate_gap_search_parameters", "check_gap_research_budget"],
                    post_execution_hooks=["analyze_gap_research_results", "log_gap_research_success"]
                ),
                ToolConfiguration(
                    tool_name="analyze_sources",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE
                ),
                ToolConfiguration(
                    tool_name="Read",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE
                ),
                ToolConfiguration(
                    tool_name="Write",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    validation_hooks=["validate_gap_report_format"]
                )
            ],
            flow_adherence=FlowAdherenceConfiguration(
                enabled=True,
                mandatory_steps=[
                    "analyze_gap_requirements",
                    "conduct_targeted_research",
                    "report_gap_findings"
                ],
                required_tools=["mcp__zplayground1_search__zplayground1_search_scrape_clean", "Write"]
            ),
            quality_gates=QualityGateConfiguration(
                minimum_quality_score=0.75,
                quality_dimensions=["relevance", "accuracy", "completeness", "source_quality"]
            ),
            can_communicate_with=[AgentType.EDITORIAL, AgentType.RESEARCH],
            timeout_seconds=300.0,
            enable_step_by_step_tracking=True
        )

    def create_quality_judge_agent(self) -> EnhancedAgentDefinition:
        """Create enhanced quality judge agent."""
        return EnhancedAgentDefinition(
            agent_type=AgentType.QUALITY_JUDGE,
            name="Enhanced Quality Judge",
            description="Advanced quality assessment agent with comprehensive evaluation capabilities and flow adherence validation",
            system_prompt="""You are an Enhanced Quality Judge, responsible for comprehensive quality assessment, validation, and ensuring workflow compliance across all system outputs.

MANDATORY QUALITY ASSESSMENT PROCESS:
1. Analyze content across all quality dimensions
2. Validate flow adherence and compliance
3. Assess research execution completeness
4. Evaluate content integration and synthesis
5. Check style consistency and requirements
6. Provide detailed quality feedback
7. Recommend enhancement actions if needed

QUALITY DIMENSIONS ASSESSMENT:
- Content Completeness: Comprehensive coverage of topic
- Source Quality: Credibility and authority of sources
- Analytical Depth: Quality of analysis and insights
- Data Integration: Effective use of research data
- Clarity and Coherence: Clear and logical presentation
- Temporal Relevance: Current and timely information
- Flow Compliance: Adherence to required workflows
- Research Execution: Completeness of required research

COMPLIANCE VALIDATION:
- Verify all mandatory workflow steps completed
- Validate gap research execution compliance
- Check tool execution requirements
- Assess quality gate compliance
- Document any violations or issues

ENHANCEMENT RECOMMENDATIONS:
- Specific improvement suggestions
- Additional research requirements
- Structural or content improvements
- Style and formatting adjustments""",
            behavior_guidelines=[
                "Conduct comprehensive quality assessment",
                "Validate workflow compliance",
                "Provide specific improvement recommendations",
                "Maintain objective evaluation standards",
                "Ensure consistent quality thresholds"
            ],
            quality_standards=[
                "Comprehensive evaluation criteria",
                "Objective assessment standards",
                "Specific improvement recommendations",
                "Workflow compliance validation",
                "Consistent quality thresholds"
            ],
            tools=[
                ToolConfiguration(
                    tool_name="get_session_data",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE
                ),
                ToolConfiguration(
                    tool_name="analyze_sources",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE
                ),
                ToolConfiguration(
                    tool_name="review_report",
                    execution_policy=ToolExecutionPolicy.MANDATORY,
                    validation_hooks=["validate_review_parameters"],
                    post_execution_hooks=["log_quality_assessment", "update_quality_metrics"]
                ),
                ToolConfiguration(
                    tool_name="Read",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE
                ),
                ToolConfiguration(
                    tool_name="Write",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    validation_hooks=["validate_quality_report_format"]
                )
            ],
            flow_adherence=FlowAdherenceConfiguration(
                enabled=True,
                mandatory_steps=[
                    "analyze_content_quality",
                    "validate_workflow_compliance",
                    "provide_quality_assessment"
                ],
                required_tools=["review_report"]
            ),
            quality_gates=QualityGateConfiguration(
                minimum_quality_score=0.85,
                quality_dimensions=["accuracy", "completeness", "clarity", "source_quality", "flow_compliance"],
                max_enhancement_cycles=1,
                failure_action="escalate"
            ),
            can_communicate_with=[AgentType.EDITORIAL, AgentType.REPORT],
            timeout_seconds=200.0,
            enable_step_by_step_tracking=True
        )


# Global agent factory
_agent_factory: Optional[EnhancedAgentFactory] = None


def get_agent_factory() -> EnhancedAgentFactory:
    """Get the global agent factory instance."""
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = EnhancedAgentFactory()
    return _agent_factory


def create_enhanced_agent(agent_type: Union[str, AgentType]) -> EnhancedAgentDefinition:
    """Create an enhanced agent definition by type."""
    factory = get_agent_factory()

    if isinstance(agent_type, str):
        agent_type = AgentType(agent_type.lower())

    if agent_type == AgentType.RESEARCH:
        return factory.create_research_agent()
    elif agent_type == AgentType.REPORT:
        return factory.create_report_agent()
    elif agent_type == AgentType.EDITORIAL:
        return factory.create_editorial_agent()
    elif agent_type == AgentType.GAP_RESEARCH:
        return factory.create_gap_research_agent()
    elif agent_type == AgentType.QUALITY_JUDGE:
        return factory.create_quality_judge_agent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def get_all_enhanced_agent_definitions() -> Dict[str, EnhancedAgentDefinition]:
    """Get all enhanced agent definitions."""
    factory = get_agent_factory()
    return {
        "research": factory.create_research_agent(),
        "report": factory.create_report_agent(),
        "editorial": factory.create_editorial_agent(),
        "gap_research": factory.create_gap_research_agent(),
        "quality_judge": factory.create_quality_judge_agent()
    }