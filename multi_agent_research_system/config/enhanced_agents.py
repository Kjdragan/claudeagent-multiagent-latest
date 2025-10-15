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

from .sdk_config import get_sdk_config, ClaudeAgentSDKConfig


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
1. Execute comprehensive search using conduct_research tool
2. Set num_results to 15-20 for thorough coverage
3. Set auto_crawl_top to 10-12 for detailed content extraction
4. Set crawl_threshold to 0.3 for relevant content filtering
5. **Set anti_bot_level to 1 (enhanced) by default, escalate to 2 (advanced) if detection occurs**
6. Analyze and validate all sources for credibility and relevance
7. Synthesize findings into structured research output
8. Save research results using save_report tool

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
                    tool_name="conduct_research",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    required_parameters=["query"],
                    optional_parameters=["num_results", "auto_crawl_top", "crawl_threshold"],
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
                    "execute_conduct_research",
                    "analyze_search_results",
                    "save_research_findings"
                ],
                required_tools=["conduct_research", "save_report"],
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
        """Create enhanced report agent definition."""
        return EnhancedAgentDefinition(
            agent_type=AgentType.REPORT,
            name="Enhanced Report Agent",
            description="Advanced report generation agent with structured formatting, content synthesis, and professional presentation capabilities",
            system_prompt="""You are an Enhanced Report Agent, an expert in transforming research findings into well-structured, comprehensive, and professional reports.

MANDATORY REPORT PROCESS:
1. Access research data using get_session_data tool
2. Analyze and synthesize research findings
3. Create structured report using create_research_report tool
4. Set appropriate report format and style based on requirements
5. Include executive summary and detailed analysis
6. Ensure proper citations and source attribution
7. Save final report using Write tool with absolute paths
8. Validate report completeness and quality

REPORT STRUCTURE REQUIREMENTS:
- Clear executive summary with key findings
- Detailed analysis with supporting evidence
- Proper section organization and flow
- Citations and source references
- Conclusion with actionable insights
- Minimum 1000 words for comprehensive coverage

QUALITY STANDARDS:
- Professional writing style and tone
- Logical structure and clear organization
- Comprehensive coverage of research findings
- Proper integration of multiple sources
- Actionable insights and recommendations""",
            behavior_guidelines=[
                "Create clear, well-structured reports",
                "Synthesize information from multiple sources",
                "Maintain professional tone and style",
                "Ensure logical flow and organization",
                "Include actionable insights"
            ],
            quality_standards=[
                "Professional writing quality",
                "Logical structure and organization",
                "Comprehensive content coverage",
                "Proper source integration",
                "Actionable recommendations"
            ],
            tools=[
                ToolConfiguration(
                    tool_name="get_session_data",
                    execution_policy=ToolExecutionPolicy.PERMISSIVE,
                    post_execution_hooks=["validate_session_data", "log_data_access"]
                ),
                ToolConfiguration(
                    tool_name="create_research_report",
                    execution_policy=ToolExecutionPolicy.VALIDATION_REQUIRED,
                    required_parameters=["research_data", "report_format"],
                    post_execution_hooks=["validate_report_structure", "check_content_quality"]
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
            flow_adherence=FlowAdherenceConfiguration(
                enabled=True,
                mandatory_steps=[
                    "access_research_data",
                    "create_structured_report",
                    "save_final_report"
                ],
                required_tools=["get_session_data", "create_research_report", "Write"],
                enforcement_strategies=["automatic_execution", "blocking_validation"]
            ),
            quality_gates=QualityGateConfiguration(
                minimum_quality_score=0.8,
                quality_dimensions=["clarity", "completeness", "structure", "professional_tone"],
                max_enhancement_cycles=2
            ),
            can_communicate_with=[AgentType.RESEARCH, AgentType.EDITORIAL, AgentType.QUALITY_JUDGE],
            timeout_seconds=300.0,
            enable_step_by_step_tracking=True
        )

    def create_editorial_agent(self) -> EnhancedAgentDefinition:
        """Create enhanced editorial agent with flow adherence enforcement."""
        return EnhancedAgentDefinition(
            agent_type=AgentType.EDITORIAL,
            name="Enhanced Editorial Agent with Flow Adherence Enforcement",
            description="Advanced editorial agent with comprehensive quality assessment, gap identification, and mandatory gap research execution enforcement",
            system_prompt="""You are an Enhanced Editorial Agent with MANDATORY FLOW ADHERENCE enforcement, responsible for comprehensive quality assessment, gap identification, and ENSURING research execution compliance.

MANDATORY THREE-STEP WORKFLOW:
STEP 1: ANALYZE AVAILABLE DATA
- Execute get_session_data to access all available information
- Thoroughly analyze research data and existing reports
- Identify strengths, weaknesses, and content gaps

STEP 2: IDENTIFY SPECIFIC GAPS
- Document specific information gaps with precise details
- Prioritize gaps by importance and impact
- Create detailed research plan for gap filling

STEP 3: REQUEST GAP RESEARCH (MANDATORY)
- EXECUTE request_gap_research tool for ALL identified gaps
- NEVER document gaps without executing research requests
- System will AUTO-DETECT and FORCE EXECUTION of unrequested gap research

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
                required_tools=["conduct_research", "Write"]
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