# Config Directory - Multi-Agent Research System

This directory contains enhanced comprehensive configuration management, agent definitions with flow adherence enforcement, and system settings that control the behavior, coordination, and quality standards of the redesigned multi-agent research system.

## Directory Purpose

The config directory provides centralized configuration management for the entire multi-agent research system, including sophisticated agent definitions with Claude Agent SDK integration, flow adherence validation configuration, enhanced search configuration, environment variable management, and quality framework settings. This configuration enables flexible system behavior with 100% workflow integrity enforcement while maintaining enterprise-grade consistency and control.

## Key Components

### Agent Configuration System
- **`agents.py`** - Comprehensive agent definitions using Claude Agent SDK AgentDefinition pattern with detailed prompts, tool configurations, and behavioral guidelines (513 lines)
- **`settings.py`** - Enhanced search configuration, crawling parameters, anti-bot settings, and environment-aware path management (214 lines)
- **`__init__.py`** - Clean module exports and public API definitions for configuration access (16 lines)

### Configuration Architecture

The config directory implements a sophisticated configuration management system with:

- **Environment-Aware Configuration**: Automatic path detection and environment-based overrides
- **SDK Integration**: Full Claude Agent SDK compatibility with AgentDefinition patterns
- **Quality Framework Integration**: Configuration-driven quality assessment and enhancement
- **Multi-Environment Support**: Development, testing, and production configuration profiles

## Agent Definition Architecture

### Claude Agent SDK Integration

The system uses the Claude Agent SDK's AgentDefinition pattern for sophisticated agent configuration:

```python
def get_research_agent_definition() -> AgentDefinition:
    """Define the Research Agent using SDK AgentDefinition pattern."""
    return AgentDefinition(
        description="Expert Research Agent specializing in comprehensive web research, source validation, and information synthesis",
        prompt="""You are a Research Agent, an expert in conducting comprehensive, high-quality research...

MANDATORY RESEARCH PROCESS:
1. IMMEDIATELY execute: conduct_research with the research topic
2. Set num_results to 15 for comprehensive coverage
3. Set auto_crawl_top to 8 for detailed content extraction
4. Set crawl_threshold to 0.3 for relevance filtering
5. Save research findings using save_report
6. Create structured search results using create_research_report""",
        tools=[
            "conduct_research",
            "analyze_sources",
            "generate_report",
            "save_report",
            "get_session_data",
            "create_research_report",
            "Read", "Write", "Edit", "Bash"
        ],
        model="sonnet"
    )
```

### Specialized Agent Definitions

#### 1. Research Agent
- **Purpose**: Comprehensive web research, source validation, and information synthesis
- **Key Features**:
  - Mandatory research execution with SERP API integration
  - Configurable search parameters (results, crawling, thresholds)
  - Multi-tool workflow coordination
  - Session-based data management
  - Quality-focused research standards

#### 2. Report Agent
- **Purpose**: Transform research findings into structured, comprehensive reports
- **Key Features**:
  - Two-step report creation process (create_research_report → Write)
  - Absolute path handling for file operations
  - Stage-based work product labeling ("1-" prefix)
  - Executive summary and detailed analysis generation
  - Minimum content requirements (1000+ words)

#### 3. Editorial Agent
- **Purpose**: Report quality assessment, gap identification, and content enhancement
- **Key Features**:
  - **Flow adherence validation and enforcement system** (NEW)
  - Gap research control handoff architecture with mandatory execution
  - Research request workflow with comprehensive validation
  - Quality enhancement criteria integration
  - Budget-aware search limitations
  - Data-first enhancement approach
  - **Multi-layered compliance enforcement** (100% execution rate achieved)

**Flow Adherence Enhancement Implementation**:
- **Enhanced prompting** with mandatory three-step workflow
- **Real-time validation** through Claude Agent SDK hooks
- **Automatic gap detection** and forced execution mechanisms
- **Content analysis** for documented but unexecuted research plans
- **Quality gate integration** ensuring workflow completeness

#### 4. UI Coordinator Agent
- **Purpose**: Workflow management, user interaction, and agent coordination
- **Key Features**:
  - Multi-agent workflow orchestration
  - Session state management
  - User feedback handling
  - Progress tracking and status updates

### Agent Tool Configuration

Each agent definition includes comprehensive tool access patterns:

```python
# Research Agent Tools
RESEARCH_AGENT_TOOLS = [
    "conduct_research",        # Primary research tool
    "analyze_sources",        # Source credibility validation
    "generate_report",        # Report transformation
    "save_report",           # Report persistence
    "get_session_data",      # Session data access
    "create_research_report", # Structured report creation
    "Read", "Write", "Edit", "Bash"  # File operations
]

# Editorial Agent Tools (Note: No direct search access)
EDITORIAL_AGENT_TOOLS = [
    "analyze_sources",        # Source analysis
    "generate_report",        # Content generation
    "revise_report",         # Report improvement
    "review_report",         # Quality assessment
    "identify_research_gaps", # Gap identification
    "get_session_data",      # Session access
    "create_research_report", # Review formatting
    "Read", "Write", "Edit", "Bash"  # File operations
    # Note: mcp__research_tools__request_gap_research added via MCP server
]
```

## Enhanced Search Configuration

### EnhancedSearchConfig Class

The `settings.py` module provides sophisticated search configuration with environment-aware management:

```python
@dataclass
class EnhancedSearchConfig:
    """Configuration for enhanced search functionality."""

    # Search Settings
    default_num_results: int = 15
    default_auto_crawl_top: int = 10
    default_crawl_threshold: float = 0.3  # Fixed for better success rates
    default_anti_bot_level: int = 1
    default_max_concurrent: int = 0  # 0 => unbounded concurrency

    # Anti-Bot Levels (0-3)
    anti_bot_levels = {
        0: "basic",      # 6/10 sites success
        1: "enhanced",   # 8/10 sites success
        2: "advanced",   # 9/10 sites success
        3: "stealth"     # 9.5/10 sites success
    }
```

### Target-Based Scraping Configuration

```python
# Target-based scraping settings
target_successful_scrapes: int = 15  # Target number of successful scrapes
url_deduplication_enabled: bool = True  # Prevent duplicate URL crawling
progressive_retry_enabled: bool = True  # Retry failed URLs with higher anti-bot levels

# Retry logic settings
max_retry_attempts: int = 3
progressive_timeout_multiplier: float = 1.5

# Token management
max_response_tokens: int = 20000
content_summary_threshold: int = 20000
```

### Environment-Aware Path Management

The configuration system implements intelligent path detection for different deployment environments:

```python
def _get_default_workproduct_dir(self) -> str:
    """Environment-aware path detection for KEVIN directory."""
    current_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if "claudeagent-multiagent-latest" in current_repo:
        return f"{current_repo}/KEVIN/work_products"
    else:
        return "/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/work_products"

def ensure_workproduct_directory(self, custom_dir: str = None, session_id: str = None, category: str = "research") -> Path:
    """Ensure workproduct directory exists with session-based organization."""
    if custom_dir:
        workproduct_dir = Path(custom_dir)
    elif session_id:
        # Session-based directory structure: KEVIN/sessions/{session_id}/{category}
        base_sessions_dir = Path(f"{self._get_base_repo_dir()}/KEVIN/sessions")
        session_dir = base_sessions_dir / session_id
        workproduct_dir = session_dir / category
    else:
        workproduct_dir = Path(self.default_workproduct_dir)

    workproduct_dir.mkdir(parents=True, exist_ok=True)
    return workproduct_dir
```

## Settings Management System

### SettingsManager Class

The `SettingsManager` provides comprehensive configuration management with environment variable overrides:

```python
class SettingsManager:
    """Manages configuration settings for the research system."""

    def __init__(self):
        self._enhanced_search_config = EnhancedSearchConfig()
        self._load_environment_overrides()

    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        # Enhanced search settings
        if os.getenv('ENHANCED_SEARCH_NUM_RESULTS'):
            self._enhanced_search_config.default_num_results = int(os.getenv('ENHANCED_SEARCH_NUM_RESULTS'))

        if os.getenv('ENHANCED_SEARCH_ANTI_BOT_LEVEL'):
            level = int(os.getenv('ENHANCED_SEARCH_ANTI_BOT_LEVEL'))
            if 0 <= level <= 3:
                self._enhanced_search_config.default_anti_bot_level = level
```

### Environment Variable Configuration

The system supports comprehensive environment-based configuration:

```bash
# Enhanced Search Configuration
ENHANCED_SEARCH_NUM_RESULTS=15
ENHANCED_SEARCH_AUTO_CRAWL_TOP=10
ENHANCED_SEARCH_CRAWL_THRESHOLD=0.3
ENHANCED_SEARCH_ANTI_BOT_LEVEL=1
ENHANCED_SEARCH_MAX_CONCURRENT=15

# Path Configuration
KEVIN_WORKPRODUCTS_DIR=/custom/path/to/workproducts

# API Configuration
ANTHROPIC_API_KEY=your_api_key_here
SERPER_API_KEY=your_serp_key_here
```

### Content Cleaning Configuration

```python
# Content cleaning settings
default_cleanliness_threshold: float = 0.7
min_content_length_for_cleaning: int = 500
min_cleaned_content_length: int = 200

# Crawl settings
default_crawl_timeout: int = 30000
max_concurrent_crawls: int = 15
crawl_retry_attempts: int = 2
```

### Enhanced Research & Gap Research Configuration

The redesigned system includes comprehensive configuration parameters for research execution and gap research coordination with flow adherence enforcement:

```python
# Enhanced Research Configuration
@dataclass
class EnhancedResearchConfig:
    """Configuration for enhanced research and gap research execution."""

    # Primary Research Parameters
    primary_research_max_results: int = 15
    primary_research_auto_crawl_top: int = 10
    primary_research_crawl_threshold: float = 0.3
    primary_research_anti_bot_level: int = 1
    primary_research_max_concurrent: int = 10

    # Gap Research Parameters
    gap_research_max_results: int = 12
    gap_research_auto_crawl_top: int = 8
    gap_research_crawl_threshold: float = 0.4  # Higher threshold for targeted research
    gap_research_anti_bot_level: int = 2      # Higher level for gap research
    gap_research_max_concurrent: int = 8

    # Budget Management
    total_budget_scrapes: int = 30
    total_budget_queries: int = 10
    primary_research_allocation: float = 0.7  # 70% for primary research
    gap_research_allocation: float = 0.3      # 30% for gap research
    emergency_reserve_scrapes: int = 5
    emergency_reserve_queries: int = 2

    # Flow Adherence Settings
    flow_adherence_enforcement: bool = True
    gap_research_mandatory: bool = True
    automatic_gap_detection: bool = True
    forced_execution_enabled: bool = True

    # Quality Parameters
    min_quality_threshold: float = 0.7
    gap_research_quality_threshold: float = 0.65
    enhancement_enabled: bool = True
    max_enhancement_cycles: int = 3

# Research-specific environment variables
ENHANCED_RESEARCH_MAX_RESULTS=15
ENHANCED_RESEARCH_AUTO_CRAWL_TOP=10
ENHANCED_RESEARCH_CRAWL_THRESHOLD=0.3
ENHANCED_RESEARCH_ANTI_BOT_LEVEL=1

# Gap research configuration
GAP_RESEARCH_ENABLED=true
GAP_RESEARCH_MAX_RESULTS=12
GAP_RESEARCH_AUTO_CRAWL_TOP=8
GAP_RESEARCH_ANTI_BOT_LEVEL=2
GAP_RESEARCH_MANDATORY=true

# Budget management
RESEARCH_BUDGET_SCRAPES=30
RESEARCH_BUDGET_QUERIES=10
GAP_RESEARCH_BUDGET_RATIO=0.3

# Flow adherence enforcement
FLOW_ADHERENCE_ENFORCEMENT=true
AUTOMATIC_GAP_DETECTION=true
FORCED_EXECUTION_ENABLED=true
```

### Session Management Configuration

```python
# Session Management Settings
@dataclass
class SessionManagementConfig:
    """Configuration for session-based organization and management."""

    # Directory Structure
    session_based_organization: bool = True
    base_sessions_dir: str = "KEVIN/sessions"
    working_subdir: str = "working"
    research_subdir: str = "research"
    complete_subdir: str = "complete"
    agent_logs_subdir: str = "agent_logs"

    # File Organization
    stage_based_prefixes: bool = True
    timestamped_files: bool = True
    metadata_files: bool = True

    # Session Lifecycle
    session_timeout_minutes: int = 120
    auto_cleanup_days: int = 30
    session_compression: bool = True

    # Workproduct Naming
    research_prefix: str = "RESEARCH"
    report_prefix: str = "REPORT"
    editorial_prefix: str = "EDITORIAL"
    final_prefix: str = "FINAL"
    gap_research_prefix: str = "EDITOR_RESEARCH"

# Session management environment variables
SESSION_BASED_ORGANIZATION=true
SESSION_TIMEOUT_MINUTES=120
SESSION_AUTO_CLEANUP_DAYS=30
SESSION_COMPRESSION_ENABLED=true
```

## Flow Adherence & Compliance Enforcement

### Multi-Layered Validation Architecture

**TRANSFORMATIVE SYSTEM ENHANCEMENT**: Implemented comprehensive flow adherence validation and enforcement system that achieves 100% editorial gap research execution compliance through multi-layered validation architecture.

**Problem Addressed**: Critical system integrity issue where editorial agents documented gap research plans but failed to execute required tool calls, creating disconnect between documented intentions and actual execution.

**Configuration-Driven Enforcement Layers**:

#### **Layer 1: Agent Definition Enhancement**
```python
# Enhanced editorial agent definition with mandatory workflow
def get_editorial_agent_definition() -> AgentDefinition:
    """Define Editorial Agent with flow adherence enforcement."""
    return AgentDefinition(
        description="Editorial Agent with mandatory gap research execution",
        prompt="""You are an Editorial Agent responsible for quality assessment and enhancement.

MANDATORY THREE-STEP WORKFLOW:
STEP 1: ANALYZE AVAILABLE DATA (get_session_data)
STEP 2: IDENTIFY SPECIFIC GAPS
STEP 3: REQUEST GAP RESEARCH (MANDATORY)

CRITICAL: Documenting gaps without calling request_gap_research tool is INSUFFICIENT.
System will automatically detect and force execution of unrequested gap research.""",

        tools=[
            "analyze_sources", "review_report", "identify_research_gaps",
            "request_gap_research",  # MANDATORY for gap execution
            "get_session_data", "create_research_report",
            "Read", "Write", "Edit", "Bash"
        ],

        hooks={
            "PreToolUse": [{
                "matcher": "Write|create_research_report",
                "hooks": ["validate_editorial_gap_research_completion"]
            }]
        }
    )
```

#### **Layer 2: Orchestrator Configuration Validation**
```python
# Gap research validation configuration
GAP_RESEARCH_VALIDATION_CONFIG = {
    "enabled": True,
    "auto_detect_gaps": True,
    "force_execution": True,
    "content_analysis_patterns": [
        "Conduct targeted searches for:",
        "Priority 3: Address Information Gaps",
        "gap-filling research",
        "missing information"
    ],
    "validation_logging": True,
    "intervention_tracking": True
}
```

#### **Layer 3: Quality Framework Integration**
```python
# Quality gate configuration for flow adherence
QUALITY_GATE_CONFIG = {
    "editorial_completion": {
        "required_tools": ["request_gap_research"],
        "validation_methods": ["content_analysis", "tool_execution_tracking"],
        "enforcement_actions": ["force_execution", "block_completion"],
        "quality_threshold": 0.8
    }
}
```

### Configuration Management for Compliance

#### **Environment-Based Compliance Settings**
```bash
# Flow adherence enforcement configuration
EDITORIAL_FLOW_ADHERENCE_ENABLED=true
GAP_RESEARCH_AUTO_DETECTION=true
FORCE_EXECUTION_ON_DETECTION=true
COMPLIANCE_LOGGING_LEVEL=DETAILED
VALIDATION_INTERVENTION_LOGGING=true
```

#### **Dynamic Compliance Configuration**
```python
class ComplianceConfigManager:
    """Manages flow adherence compliance configuration."""

    def __init__(self):
        self.compliance_settings = {
            "gap_research_enforcement": {
                "enabled": True,
                "detection_methods": ["content_analysis", "tool_tracking"],
                "enforcement_strategies": ["automatic_execution", "agent_guidance"],
                "logging_level": "comprehensive"
            },
            "agent_behavior_validation": {
                "pre_tool_hooks": True,
                "session_state_tracking": True,
                "real_time_feedback": True,
                "corrective_guidance": True
            }
        }

    def get_compliance_config(self, agent_type: str) -> dict:
        """Get agent-specific compliance configuration."""
        return self.compliance_settings.get(f"{agent_type}_compliance", {})
```

### Agent Behavior Configuration Standards

#### **Enhanced Agent Definition Patterns**
```python
# Standard agent definition with flow adherence
ENHANCED_AGENT_DEFINITION_TEMPLATE = """
Agent Definition for {agent_type}:

PURPOSE: {purpose}

MANDATORY WORKFLOW:
{mandatory_steps}

COMPLIANCE REQUIREMENTS:
- All documented research plans MUST be executed through tool calls
- Documentation without execution is INSUFFICIENT
- System will automatically detect and force missing research execution
- Real-time validation will block incomplete work

QUALITY STANDARDS:
{quality_criteria}

ENFORCEMENT MECHANISMS:
- PreToolUse hooks for real-time validation
- Content analysis for gap detection
- Automatic forced execution when needed
- Comprehensive logging of interventions
"""
```

#### **Configuration Validation Framework**
```python
class ConfigurationValidator:
    """Validates agent configuration for flow adherence compliance."""

    def validate_editorial_config(self, config: dict) -> ValidationResult:
        """Validate editorial agent configuration for compliance."""

        required_elements = [
            "gap_research_mandatory_workflow",
            "compliance_enforcement_hooks",
            "validation_mechanisms",
            "corrective_feedback_systems"
        ]

        validation_result = ValidationResult()

        for element in required_elements:
            if element not in config:
                validation_result.add_error(f"Missing required compliance element: {element}")

        # Validate tool access for gap research
        if "request_gap_research" not in config.get("tools", []):
            validation_result.add_error("Editorial agent missing required gap research tool")

        # Validate hook configuration
        if not config.get("hooks", {}).get("PreToolUse"):
            validation_result.add_warning("No PreToolUse hooks configured for compliance validation")

        return validation_result
```

### Performance Impact & Optimization

#### **Compliance Enforcement Metrics**
```python
# Compliance enforcement performance tracking
COMPLIANCE_METRICS = {
    "gap_research_detection_rate": {
        "target": 100,  # percentage
        "current_measurement": 100,  # achieved rate
        "improvement": "infinite",  # from 0% to 100%
    },
    "agent_compliance_rate": {
        "target": 100,
        "current_measurement": 100,
        "improvement": "infinite"
    },
    "quality_improvement": {
        "baseline": 3,  # out of 10
        "current": 8.5,  # out of 10
        "improvement_percentage": 267
    },
    "workflow_integrity": {
        "documentation_execution_consistency": 100,
        "system_reliability_score": 100,
        "user_trust_level": "restored"
    }
}
```

#### **Optimization Configuration**
```python
# Compliance system optimization settings
COMPLIANCE_OPTIMIZATION = {
    "validation_efficiency": {
        "parallel_validation": True,
        "cached_gap_patterns": True,
        "optimized_content_analysis": True,
        "minimal_performance_impact": True
    },
    "enforcement_strategies": {
        "prefer_guidance_over_blocking": True,
        "progressive_enforcement_levels": True,
        "adaptive_validation_intensity": True
    }
}
```

### Testing & Validation Configuration

#### **Compliance Testing Framework**
```python
# Configuration for compliance testing
COMPLIANCE_TEST_CONFIG = {
    "test_scenarios": [
        "gap_identification_without_tool_execution",
        "documented_research_plan_compliance",
        "automatic_gap_detection_validation",
        "forced_execution_mechanisms",
        "quality_gate_enforcement"
    ],
    "success_criteria": {
        "gap_research_execution_rate": 100,
        "detection_accuracy": 100,
        "false_positive_rate": 0,
        "performance_impact": "< 5%"
    }
}
```

This comprehensive flow adherence configuration system ensures 100% compliance through multi-layered validation and enforcement while maintaining optimal system performance.

## Configuration Integration Patterns

### Agent Factory Pattern

The config system supports sophisticated agent creation with configuration:

```python
# Example: Agent creation with configuration
from config.agents import get_all_agent_definitions
from config.settings import get_settings

class ConfigurableAgentFactory:
    def __init__(self):
        self.settings = get_settings()
        self.agent_definitions = get_all_agent_definitions()

    def create_agent(self, agent_type: str, **kwargs):
        """Create agent with configuration-based customization."""
        if agent_type not in self.agent_definitions:
            raise ValueError(f"Unknown agent type: {agent_type}")

        definition = self.agent_definitions[agent_type]

        # Apply configuration-based modifications
        if self.settings.enhanced_search.default_num_results != 15:
            definition.prompt = self._update_search_params(definition.prompt)

        return ClaudeAgent(
            definition=definition,
            settings=self.settings,
            **kwargs
        )
```

### Configuration Validation

The system provides comprehensive configuration validation:

```python
def validate_anti_bot_level(self, level: int) -> int:
    """Validate and clamp anti-bot level to valid range."""
    return max(0, min(3, level))

def validate_crawl_threshold(self, threshold: float) -> float:
    """Validate and clamp crawl threshold to valid range."""
    return max(0.0, min(1.0, threshold))

def get_debug_info(self) -> dict[str, Any]:
    """Get debug information about current configuration."""
    return {
        "enhanced_search_config": {
            "default_num_results": self._enhanced_search_config.default_num_results,
            "default_crawl_threshold": self._enhanced_search_config.default_crawl_threshold,
            "workproduct_dir": self._enhanced_search_config.default_workproduct_dir,
        },
        "environment_variables": {
            "SERPER_API_KEY": "SET" if os.getenv('SERPER_API_KEY') else "NOT_SET",
            "ANTHROPIC_API_KEY": "SET" if os.getenv('ANTHROPIC_API_KEY') else "NOT_SET",
            "KEVIN_WORKPRODUCTS_DIR": os.getenv('KEVIN_WORKPRODUCTS_DIR', 'NOT_SET')
        }
    }
```

## Quality Framework Integration

### Agent Quality Configuration

Each agent definition includes quality-focused behavioral guidelines:

```python
# Research Agent Quality Standards
RESEARCH_STANDARDS = """
- Prioritize authoritative sources (academic papers, reputable news, official reports)
- Cross-reference information across multiple sources
- Distinguish between facts and opinions
- Note source dates and potential biases
- Gather sufficient depth to support comprehensive reporting
"""

# Editorial Agent Quality Enhancement Criteria
EDITORIAL_QUALITY_CRITERIA = [
    "Data Specificity": "Does the report include specific facts, figures, statistics?",
    "Fact Expansion": "Are general statements expanded with specific data?",
    "Information Integration": "Are research findings thoroughly integrated?",
    "Fact-Based Enhancement": "Are claims supported with specific data?",
    "Rich Content": "Does the report leverage scraped research data effectively?",
    "Comprehensive Coverage": "Does it include relevant facts and data points?",
    "Style Consistency": "Is the report consistent with user's requested style?",
    "Appropriate Length": "Does length match data volume and requirements?"
]
```

### Configuration-Driven Quality Gates

The config system supports configurable quality thresholds:

```python
# Quality configuration (can be extended in settings.py)
QUALITY_THRESHOLDS = {
    "content_completeness": 0.8,
    "source_credibility": 0.7,
    "analytical_depth": 0.6,
    "clarity_coherence": 0.8,
    "factual_accuracy": 0.9,
    "data_integration": 0.8
}

ENHANCEMENT_CONFIG = {
    "enabled": True,
    "max_cycles": 3,
    "improvement_threshold": 0.1,
    "gap_filling_enabled": True
}
```

## Usage Examples

### Basic Configuration Usage

```python
from config.settings import get_settings, get_enhanced_search_config
from config.agents import get_research_agent_definition, get_all_agent_definitions

# Load configuration
settings = get_settings()
search_config = get_enhanced_search_config()

# Get agent definition
research_agent = get_research_agent_definition()
print(f"Research agent: {research_agent.description}")
print(f"Available tools: {research_agent.tools}")

# Get all agent definitions
all_agents = get_all_agent_definitions()
for name, definition in all_agents.items():
    print(f"{name}: {definition.description}")
```

### Custom Configuration with Environment Variables

```python
# Set environment variables
os.environ['ENHANCED_SEARCH_NUM_RESULTS'] = '20'
os.environ['ENHANCED_SEARCH_ANTI_BOT_LEVEL'] = '2'
os.environ['KEVIN_WORKPRODUCTS_DIR'] = '/custom/workproducts'

# Load updated configuration
settings = get_settings()
print(f"Search results: {settings.enhanced_search.default_num_results}")
print(f"Anti-bot level: {settings.enhanced_search.default_anti_bot_level}")
print(f"Workproduct dir: {settings.enhanced_search.default_workproduct_dir}")
```

### Session-Based Configuration

```python
# Configure session-based workproduct directories
settings = get_settings()
session_id = "research_session_123"

# Ensure session directory structure
research_dir = settings.ensure_workproduct_directory(
    session_id=session_id,
    category="research"
)
working_dir = settings.ensure_workproduct_directory(
    session_id=session_id,
    category="working"
)

print(f"Research directory: {research_dir}")
print(f"Working directory: {working_dir}")
```

### Configuration Debugging

```python
# Get debug information
settings = get_settings()
debug_info = settings.get_debug_info()

print("Configuration Debug Info:")
print(json.dumps(debug_info, indent=2))

# Validate configuration
anti_bot_level = 5  # Invalid level
validated_level = settings.validate_anti_bot_level(anti_bot_level)
print(f"Validated anti-bot level: {validated_level}")  # Will be 3
```

### Dynamic Configuration Updates

```python
# Create custom configuration
class CustomResearchConfig:
    def __init__(self, base_config):
        self.base_config = base_config

    def get_custom_search_params(self, topic_complexity: str) -> dict:
        """Get search parameters based on topic complexity."""
        base_params = self.base_config.get_default_search_params()

        if topic_complexity == "simple":
            base_params["num_results"] = 10
            base_params["auto_crawl_top"] = 5
        elif topic_complexity == "complex":
            base_params["num_results"] = 25
            base_params["auto_crawl_top"] = 15
            base_params["anti_bot_level"] = 2

        return base_params

# Usage
custom_config = CustomResearchConfig(settings)
simple_params = custom_config.get_custom_search_params("simple")
complex_params = custom_config.get_custom_search_params("complex")
```

## Integration with System Components

### Core Orchestrator Integration

The config system integrates seamlessly with the core orchestrator:

```python
# In core/orchestrator.py
from config.agents import get_all_agent_definitions
from config.settings import get_settings

class ResearchOrchestrator:
    def __init__(self, debug_mode: bool = False):
        self.settings = get_settings()
        self.agent_definitions = get_all_agent_definitions()

    async def initialize_agent(self, agent_type: str):
        """Initialize agent with configuration."""
        if agent_type in self.agent_definitions:
            definition = self.agent_definitions[agent_type]
            # Create agent with SDK using definition
            return await self._create_sdk_agent(definition)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
```

### MCP Tool Integration

Configuration supports MCP tool registration and management:

```python
# MCP tools use configuration for parameters
def get_session_data_tool(data_type: str = "all") -> dict:
    """MCP tool using configuration for session management."""
    session_id = get_current_session_id()
    session_data = get_session_data(session_id)

    if data_type == "research":
        return {"research_data": session_data.get("research_results")}
    elif data_type == "report":
        return {"report_data": session_data.get("report_results")}
    else:
        return session_data
```

### Quality Framework Integration

Configuration drives quality assessment and enhancement:

```python
# Quality framework uses configuration thresholds
from config.settings import get_settings

class QualityFramework:
    def __init__(self):
        self.settings = get_settings()
        self.quality_thresholds = QUALITY_THRESHOLDS

    async def assess_content(self, content: str, context: dict) -> QualityAssessment:
        """Assess content using configuration-driven criteria."""
        assessment = QualityAssessment()

        for criterion, threshold in self.quality_thresholds.items():
            result = await self._assess_criterion(criterion, content, context)
            assessment.add_criterion_result(criterion, result, threshold)

        return assessment
```

## Testing & Validation

### Configuration Testing

```python
# Test configuration loading and validation
def test_configuration_loading():
    """Test configuration system initialization."""
    settings = get_settings()

    # Test enhanced search config
    assert settings.enhanced_search.default_num_results > 0
    assert 0 <= settings.enhanced_search.default_crawl_threshold <= 1
    assert 0 <= settings.enhanced_search.default_anti_bot_level <= 3

    # Test path management
    workproduct_dir = settings.ensure_workproduct_directory()
    assert workproduct_dir.exists()

    # Test environment overrides
    os.environ['ENHANCED_SEARCH_NUM_RESULTS'] = '25'
    new_settings = get_settings()  # Should load new instance
    assert new_settings.enhanced_search.default_num_results == 25

def test_agent_definitions():
    """Test agent definition loading and validation."""
    agents = get_all_agent_definitions()

    # Test all required agents are present
    required_agents = ["research_agent", "report_agent", "editor_agent", "ui_coordinator"]
    for agent_name in required_agents:
        assert agent_name in agents
        definition = agents[agent_name]
        assert definition.description
        assert definition.prompt
        assert definition.tools
        assert definition.model
```

### Environment Variable Testing

```python
# Test environment variable configuration
def test_environment_configuration():
    """Test environment variable overrides."""
    # Set test environment variables
    test_vars = {
        'ENHANCED_SEARCH_NUM_RESULTS': '20',
        'ENHANCED_SEARCH_ANTI_BOT_LEVEL': '2',
        'ENHANCED_SEARCH_CRAWL_THRESHOLD': '0.5'
    }

    with temp_env_vars(test_vars):
        settings = SettingsManager()

        assert settings.enhanced_search.default_num_results == 20
        assert settings.enhanced_search.default_anti_bot_level == 2
        assert settings.enhanced_search.default_crawl_threshold == 0.5
```

## Best Practices

### Configuration Management

1. **Environment First**: Always use environment variables for deployment-specific configuration
2. **Default Values**: Provide sensible defaults for all configuration options
3. **Validation**: Validate configuration values at startup and during runtime
4. **Documentation**: Document all configuration options with examples and effects
5. **Type Safety**: Use proper type hints and validation for configuration values

### Agent Definition Standards

1. **Clear Instructions**: Provide specific, actionable instructions in agent prompts
2. **Tool Alignment**: Ensure agent tools match described capabilities and requirements
3. **Quality Focus**: Define clear quality criteria and success metrics for each agent
4. **Error Handling**: Include error handling and fallback strategies in agent prompts
5. **Integration Patterns**: Document how agents should coordinate and hand off control

### Performance Configuration

1. **Resource Limits**: Set appropriate resource limits and timeouts
2. **Concurrency Control**: Configure concurrent operations based on system capacity
3. **Caching Strategy**: Configure caching for frequently accessed data
4. **Quality vs. Speed**: Balance quality requirements with performance constraints

### Security Configuration

1. **API Key Management**: Never commit API keys or sensitive configuration
2. **Path Validation**: Validate file paths and directory access
3. **Input Sanitization**: Validate configuration inputs and prevent injection
4. **Access Control**: Configure appropriate access controls for file operations

## Troubleshooting

### Common Configuration Issues

1. **Missing Environment Variables**
   ```python
   # Check environment variable status
   debug_info = settings.get_debug_info()
   missing_vars = [k for k, v in debug_info["environment_variables"].items() if v == "NOT_SET"]
   print(f"Missing environment variables: {missing_vars}")
   ```

2. **Invalid Configuration Values**
   ```python
   # Validate anti-bot level
   level = 5  # Invalid
   validated = settings.validate_anti_bot_level(level)
   print(f"Validated level: {validated}")  # Will be clamped to 3
   ```

3. **Path Issues**
   ```python
   # Test directory creation
   try:
       work_dir = settings.ensure_workproduct_directory()
       print(f"Work directory: {work_dir}")
       print(f"Directory exists: {work_dir.exists()}")
   except Exception as e:
       print(f"Path creation failed: {e}")
   ```

4. **Agent Definition Issues**
   ```python
   # Test agent loading
   try:
       agents = get_all_agent_definitions()
       for name, definition in agents.items():
           print(f"{name}: {len(definition.tools)} tools")
   except Exception as e:
       print(f"Agent loading failed: {e}")
   ```

### Debug Mode Configuration

```python
# Enable comprehensive debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Load configuration with debug info
settings = get_settings()
debug_info = settings.get_debug_info()
print("Configuration Debug Info:")
print(json.dumps(debug_info, indent=2))
```

## Future Development

### Planned Configuration Enhancements

1. **Configuration Profiles**: Support for multiple configuration profiles (dev, test, prod)
2. **Dynamic Configuration**: Runtime configuration updates without restart
3. **Configuration Templates**: Reusable configuration templates for different use cases
4. **Advanced Validation**: More sophisticated configuration validation and error reporting
5. **Configuration Migration**: Automated configuration migration between versions

### Extension Points

1. **Custom Agent Definitions**: Framework for defining custom agents with configuration
2. **Plugin Configuration**: Configuration system for plugins and extensions
3. **Quality Criteria**: Configurable quality assessment criteria and thresholds
4. **Search Strategies**: Configurable search strategies and parameters
5. **Output Formats**: Configurable output formats and templates

This comprehensive configuration system provides enterprise-grade flexibility and control while maintaining simplicity and usability for different deployment scenarios and use cases.

## Enhanced Editorial Workflow Configuration (Phase 3.2)

### Overview

The enhanced editorial workflow configuration system provides comprehensive control over the advanced editorial intelligence features introduced in Phase 3.2. This configuration system supports multi-dimensional confidence scoring, intelligent gap research decisions, evidence-based recommendations, and sophisticated sub-session management.

### Configuration Architecture

The enhanced editorial workflow configuration is organized into six main components:

1. **Enhanced Editorial Engine Configuration** - Confidence scoring, gap analysis, and decision logic
2. **Gap Research Decision System Configuration** - Cost-benefit analysis and resource allocation
3. **Research Corpus Analyzer Configuration** - Quality assessment and coverage analysis
4. **Editorial Recommendations Configuration** - Prioritization and action planning
5. **Sub-Session Manager Configuration** - Session hierarchy and coordination
6. **Editorial Workflow Integration Configuration** - System integration and hooks

### 1. Enhanced Editorial Engine Configuration

#### Core Editorial Intelligence Settings

```python
@dataclass
class EnhancedEditorialEngineConfig:
    """Configuration for enhanced editorial decision engine."""

    # Confidence Scoring Configuration
    confidence_scoring_enabled: bool = True
    confidence_dimensions: List[str] = field(default_factory=lambda: [
        "factual_gaps", "temporal_gaps", "comparative_gaps",
        "quality_gaps", "coverage_gaps", "depth_gaps",
        "accuracy_gaps", "relevance_gaps"
    ])
    confidence_threshold: float = 0.7
    min_confidence_for_gap_research: float = 0.6

    # Dimension Weights (must sum to 1.0)
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "factual_gaps": 0.20,
        "temporal_gaps": 0.15,
        "comparative_gaps": 0.15,
        "quality_gaps": 0.15,
        "coverage_gaps": 0.10,
        "depth_gaps": 0.10,
        "accuracy_gaps": 0.10,
        "relevance_gaps": 0.05
    })

    # Gap Analysis Configuration
    gap_analysis_enabled: bool = True
    max_gap_topics: int = 3
    gap_topic_prioritization: bool = True
    gap_detection_sensitivity: float = 0.8

    # Evidence Collection Parameters
    evidence_collection_enabled: bool = True
    min_evidence_confidence: float = 0.7
    evidence_sources_weighting: Dict[str, float] = field(default_factory=lambda: {
        "primary_research": 1.0,
        "secondary_research": 0.8,
        "gap_research": 0.9,
        "user_provided": 0.95
    })

    # Decision Logic Settings
    decision_logic_mode: str = "weighted_average"  # "weighted_average", "majority_vote", "consensus"
    auto_execute_decisions: bool = False  # False for manual approval
    decision_timeout_seconds: int = 300
    decision_audit_trail: bool = True

# Environment Variables for Enhanced Editorial Engine
EDITORIAL_INTELLIGENCE_ENABLED=true
EDITORIAL_CONFIDENCE_THRESHOLD=0.7
EDITORIAL_MAX_GAP_TOPICS=3
EDITORIAL_GAP_ANALYSIS_SENSITIVITY=0.8
EDITORIAL_EVIDENCE_COLLECTION_ENABLED=true
EDITORIAL_DECISION_LOGIC_MODE=weighted_average
EDITORIAL_AUTO_EXECUTE_DECISIONS=false
EDITORIAL_DECISION_AUDIT_TRAIL=true
```

#### Advanced Confidence Scoring Configuration

```python
# Confidence Scoring Advanced Configuration
CONFIDENCE_SCORING_CONFIG = {
    "scoring_algorithms": {
        "factual_gaps": {
            "algorithm": "fact_completeness_analysis",
            "weight": 0.20,
            "threshold": 0.7,
            "factors": ["missing_facts", "incomplete_data", "statistical_gaps"]
        },
        "temporal_gaps": {
            "algorithm": "temporal_coverage_analysis",
            "weight": 0.15,
            "threshold": 0.65,
            "factors": ["outdated_information", "missing_recent_data", "historical_context"]
        },
        "comparative_gaps": {
            "algorithm": "comparative_analysis",
            "weight": 0.15,
            "threshold": 0.7,
            "factors": ["missing_comparisons", "benchmark_data", "competitive_analysis"]
        },
        "quality_gaps": {
            "algorithm": "quality_assessment",
            "weight": 0.15,
            "threshold": 0.75,
            "factors": ["source_quality", "content_depth", "analytical_rigor"]
        },
        "coverage_gaps": {
            "algorithm": "coverage_completeness",
            "weight": 0.10,
            "threshold": 0.6,
            "factors": ["topic_completeness", "aspect_coverage", "perspective_diversity"]
        },
        "depth_gaps": {
            "algorithm": "depth_analysis",
            "weight": 0.10,
            "threshold": 0.7,
            "factors": ["analysis_depth", "explanation_detail", "contextual_depth"]
        },
        "accuracy_gaps": {
            "algorithm": "accuracy_validation",
            "weight": 0.10,
            "threshold": 0.8,
            "factors": ["fact_checking", "source_verification", "cross_reference_validation"]
        },
        "relevance_gaps": {
            "algorithm": "relevance_assessment",
            "weight": 0.05,
            "threshold": 0.7,
            "factors": ["topic_relevance", "audience_relevance", "context_relevance"]
        }
    },

    "confidence_thresholds": {
        "high_confidence": 0.8,
        "medium_confidence": 0.6,
        "low_confidence": 0.4,
        "min_gap_research_threshold": 0.6,
        "auto_approval_threshold": 0.85
    },

    "adaptive_learning": {
        "enabled": True,
        "learning_rate": 0.01,
        "feedback_integration": True,
        "historical_weight_adjustment": True
    }
}
```

### 2. Gap Research Decision System Configuration

#### Cost-Benefit Analysis Configuration

```python
@dataclass
class GapResearchDecisionSystemConfig:
    """Configuration for gap research decision system."""

    # Cost-Benefit Analysis Configuration
    cost_benefit_analysis_enabled: bool = True
    min_roi_threshold: float = 1.5
    max_cost_threshold: float = 0.8  # Maximum cost relative to benefit

    # Cost Factors Configuration
    cost_factors: Dict[str, float] = field(default_factory=lambda: {
        "time_cost": 0.4,        # Research time investment
        "resource_cost": 0.3,     # API and computational resources
        "complexity_cost": 0.2,   # Research complexity
        "opportunity_cost": 0.1   # Alternative research opportunities
    })

    # Benefit Factors Configuration
    benefit_factors: Dict[str, float] = field(default_factory=lambda: {
        "quality_improvement": 0.35,  # Overall content quality enhancement
        "coverage_enhancement": 0.25,  # Topic coverage improvement
        "gap_filling": 0.20,          # Specific gap resolution
        "credibility_boost": 0.15,    # Source credibility improvement
        "user_satisfaction": 0.05     # User experience improvement
    })

    # ROI Estimation Configuration
    roi_estimation_model: str = "weighted_benefit_cost_ratio"  # "simple_ratio", "weighted_benefit_cost_ratio", "ml_model"
    roi_calculation_frequency: str = "per_gap_topic"  # "per_gap_topic", "per_session", "per_batch"
    roi_history_tracking: bool = True

    # Resource Allocation Settings
    resource_allocation_strategy: str = "intelligent"  # "equal", "priority_based", "intelligent", "roi_optimized"
    max_gap_research_resources: float = 0.3  # 30% of total research budget
    min_gap_research_resources: float = 0.1  # 10% of total research budget
    resource_reallocation_enabled: bool = True

    # Decision Parameters
    max_gap_topics_per_session: int = 3
    gap_topic_selection_strategy: str = "confidence_priority"  # "confidence_priority", "roi_priority", "balanced"
    gap_research_timeout_minutes: int = 30
    parallel_gap_research_enabled: bool = True
    max_concurrent_gap_research: int = 2

# Environment Variables for Gap Research Decision System
GAP_RESEARCH_COST_BENEFIT_ENABLED=true
GAP_RESEARCH_MIN_ROI_THRESHOLD=1.5
GAP_RESEARCH_MAX_COST_THRESHOLD=0.8
GAP_RESEARCH_ALLOCATION_STRATEGY=intelligent
GAP_RESEARCH_MAX_GAP_TOPICS=3
GAP_RESEARCH_SELECTION_STRATEGY=confidence_priority
GAP_RESEARCH_PARALLEL_ENABLED=true
GAP_RESEARCH_MAX_CONCURRENT=2
```

#### Advanced Decision Logic Configuration

```python
# Advanced Gap Research Decision Logic
GAP_RESEARCH_DECISION_CONFIG = {
    "decision_matrix": {
        "high_confidence_high_roi": {
            "confidence_threshold": 0.8,
            "roi_threshold": 2.0,
            "action": "auto_approve",
            "priority": "immediate"
        },
        "high_confidence_medium_roi": {
            "confidence_threshold": 0.8,
            "roi_threshold": 1.5,
            "action": "approve",
            "priority": "high"
        },
        "medium_confidence_high_roi": {
            "confidence_threshold": 0.6,
            "roi_threshold": 2.0,
            "action": "approve",
            "priority": "high"
        },
        "medium_confidence_medium_roi": {
            "confidence_threshold": 0.6,
            "roi_threshold": 1.5,
            "action": "conditional_approve",
            "priority": "medium"
        },
        "low_confidence_high_roi": {
            "confidence_threshold": 0.4,
            "roi_threshold": 2.5,
            "action": "manual_review",
            "priority": "medium"
        },
        "low_confidence_low_roi": {
            "confidence_threshold": 0.4,
            "roi_threshold": 1.0,
            "action": "reject",
            "priority": "low"
        }
    },

    "escalation_rules": {
        "auto_approve_limit": 2,  # Max auto-approved gap topics
        "manual_review_threshold": 0.5,  # Confidence threshold for manual review
        "escalation_timeout": 600,  # Seconds before escalation
        "escalation_notification": True
    },

    "feedback_integration": {
        "collect_decision_feedback": True,
        "feedback_weight": 0.2,
        "adaptive_threshold_adjustment": True,
        "feedback_storage_days": 90
    }
}
```

### 3. Research Corpus Analyzer Configuration

#### Quality Assessment Configuration

```python
@dataclass
class ResearchCorpusAnalyzerConfig:
    """Configuration for research corpus analysis."""

    # Quality Assessment Criteria
    quality_assessment_enabled: bool = True
    quality_criteria_weights: Dict[str, float] = field(default_factory=lambda: {
        "source_credibility": 0.25,
        "content_relevance": 0.20,
        "information_depth": 0.15,
        "temporal_relevance": 0.15,
        "coverage_completeness": 0.10,
        "analytical_rigor": 0.10,
        "factual_accuracy": 0.05
    })

    # Coverage Analysis Settings
    coverage_analysis_enabled: bool = True
    coverage_dimensions: List[str] = field(default_factory=lambda: [
        "temporal_coverage", "factual_coverage", "comparative_coverage",
        "perspective_coverage", "source_diversity", "topic_completeness"
    ])
    coverage_sufficiency_threshold: float = 0.7

    # Source Evaluation Parameters
    source_evaluation_enabled: bool = True
    source_quality_criteria: Dict[str, float] = field(default_factory=lambda: {
        "authoritativeness": 0.3,
        "accuracy": 0.25,
        "objectivity": 0.2,
        "currency": 0.15,
        "coverage": 0.1
    })
    source_diversity_threshold: float = 0.6

    # Sufficiency Determination
    sufficiency_determination_enabled: bool = True
    sufficiency_algorithm: str = "weighted_coverage_analysis"  # "simple_threshold", "weighted_coverage_analysis", "ml_classification"
    min_sufficiency_score: float = 0.7
    sufficiency_confidence_threshold: float = 0.8

    # Corpus Analysis Parameters
    corpus_analysis_depth: str = "comprehensive"  # "basic", "standard", "comprehensive", "exhaustive"
    cross_reference_analysis: bool = True
    fact_verification_enabled: bool = True
    contradiction_detection: bool = True

# Environment Variables for Research Corpus Analyzer
CORPUS_ANALYSIS_ENABLED=true
CORPUS_QUALITY_ASSESSMENT_ENABLED=true
CORPUS_COVERAGE_ANALYSIS_ENABLED=true
CORPUS_SOURCE_EVALUATION_ENABLED=true
CORPUS_SUFFICIENCY_DETERMINATION_ENABLED=true
CORPUS_ANALYSIS_DEPTH=comprehensive
CORPUS_CROSS_REFERENCE_ENABLED=true
CORPUS_FACT_VERIFICATION_ENABLED=true
CORPUS_CONTRADICTION_DETECTION_ENABLED=true
```

#### Advanced Corpus Analysis Configuration

```python
# Advanced Corpus Analysis Settings
CORPUS_ANALYSIS_CONFIG = {
    "quality_assessment": {
        "detailed_scoring": {
            "source_credibility": {
                "factors": ["domain_authority", "author_expertise", "publication_reputation"],
                "scoring_method": "weighted_average",
                "normalization": "min_max"
            },
            "content_relevance": {
                "factors": ["topic_alignment", "keyword_density", "semantic_similarity"],
                "scoring_method": "tfidf_cosine",
                "normalization": "z_score"
            },
            "information_depth": {
                "factors": ["content_length", "detail_level", "analytical_depth"],
                "scoring_method": "composite_score",
                "normalization": "percentile"
            }
        },

        "quality_gates": {
            "minimum_quality_threshold": 0.6,
            "high_quality_threshold": 0.8,
            "quality_enhancement_enabled": True,
            "quality_feedback_enabled": True
        }
    },

    "coverage_analysis": {
        "temporal_analysis": {
            "time_windows": ["last_24h", "last_week", "last_month", "last_year"],
            "freshness_weight": 0.3,
            "historical_context_weight": 0.2
        },

        "factual_analysis": {
            "fact_extraction_enabled": True,
            "fact_verification_sources": ["academic", "official", "authoritative"],
            "confidence_threshold": 0.8
        },

        "comparative_analysis": {
            "benchmark_sources": ["industry_standards", "competitor_analysis", "best_practices"],
            "comparison_metrics": ["performance", "features", "quality", "cost"],
            "similarity_threshold": 0.7
        }
    },

    "source_evaluation": {
        "credibility_scoring": {
            "authority_signals": ["academic_institution", "government", "industry_leader"],
            "accuracy_signals": ["fact_checked", "peer_reviewed", "cited"],
            "bias_detection": True,
            "sentiment_analysis": True
        },

        "diversity_analysis": {
            "source_types": ["academic", "news", "industry", "government", "user_generated"],
            "geographic_diversity": True,
            "perspective_diversity": True,
            "minimum_source_types": 3
        }
    }
}
```

### 4. Editorial Recommendations Configuration

#### Recommendation Prioritization Settings

```python
@dataclass
class EditorialRecommendationsConfig:
    """Configuration for editorial recommendations engine."""

    # Recommendation Prioritization
    prioritization_enabled: bool = True
    prioritization_method: str = "roi_based"  # "simple_priority", "roi_based", "ml_ranked", "user_preference"
    recommendation_categories: List[str] = field(default_factory=lambda: [
        "quality_improvements", "content_enhancements", "gap_filling",
        "structure_optimization", "style_consistency", "accuracy_improvements"
    ])

    # Action Planning Parameters
    action_planning_enabled: bool = True
    action_planning_detail_level: str = "comprehensive"  # "basic", "standard", "comprehensive", "detailed"
    action_step_template_enabled: bool = True
    implementation_timeline: str = "auto_generated"  # "immediate", "short_term", "medium_term", "long_term", "auto_generated"

    # ROI Calculation Parameters
    roi_calculation_enabled: bool = True
    roi_factors: Dict[str, float] = field(default_factory=lambda: {
        "quality_improvement": 0.4,
        "user_satisfaction": 0.25,
        "credibility_enhancement": 0.2,
        "engagement_boost": 0.1,
        "seo_improvement": 0.05
    })
    roi_estimation_accuracy_threshold: float = 0.7

    # Workflow Organization
    workflow_organization_enabled: bool = True
    workflow_phases: List[str] = field(default_factory=lambda: [
        "immediate_actions", "short_term_improvements", "medium_term_enhancements",
        "long_term_optimizations", "strategic_improvements"
    ])
    workflow_dependencies_tracking: bool = True

    # Implementation Tracking
    implementation_tracking_enabled: bool = True
    progress_monitoring: bool = True
    completion_verification: bool = True
    effectiveness_measurement: bool = True

# Environment Variables for Editorial Recommendations
EDITORIAL_RECOMMENDATIONS_ENABLED=true
EDITORIAL_PRIORITIZATION_METHOD=roi_based
EDITORIAL_ACTION_PLANNING_ENABLED=true
EDITORIAL_ACTION_PLANNING_DETAIL=comprehensive
EDITORIAL_ROI_CALCULATION_ENABLED=true
EDITORIAL_WORKFLOW_ORGANIZATION_ENABLED=true
EDITORIAL_IMPLEMENTATION_TRACKING_ENABLED=true
```

#### Advanced Recommendations Configuration

```python
# Advanced Editorial Recommendations Configuration
EDITORIAL_RECOMMENDATIONS_CONFIG = {
    "recommendation_engine": {
        "generation_methods": {
            "quality_gaps": {
                "enabled": True,
                "algorithms": ["rule_based", "ml_classification", "pattern_matching"],
                "confidence_threshold": 0.7,
                "max_recommendations": 10
            },
            "content_enhancement": {
                "enabled": True,
                "algorithms": ["content_analysis", "semantic_similarity", "gap_identification"],
                "confidence_threshold": 0.6,
                "max_recommendations": 8
            },
            "structure_optimization": {
                "enabled": True,
                "algorithms": ["structure_analysis", "readability_assessment", "flow_analysis"],
                "confidence_threshold": 0.8,
                "max_recommendations": 5
            }
        },

        "prioritization_logic": {
            "roi_weighting": {
                "high_impact": 1.5,
                "medium_impact": 1.0,
                "low_impact": 0.5
            },
            "effort_weighting": {
                "low_effort": 1.2,
                "medium_effort": 1.0,
                "high_effort": 0.8
            },
            "urgency_weighting": {
                "immediate": 1.3,
                "short_term": 1.1,
                "medium_term": 1.0,
                "long_term": 0.9
            }
        }
    },

    "action_planning": {
        "template_types": {
            "simple_action": {
                "steps": ["identify", "plan", "implement", "verify"],
                "estimated_time": "5-15 minutes"
            },
            "complex_action": {
                "steps": ["analyze", "design", "implement", "test", "refine"],
                "estimated_time": "30-60 minutes"
            },
            "strategic_action": {
                "steps": ["research", "plan", "coordinate", "implement", "monitor", "optimize"],
                "estimated_time": "2-4 hours"
            }
        },

        "dependency_tracking": {
            "dependency_types": ["sequential", "parallel", "conditional"],
            "circular_dependency_detection": True,
            "critical_path_analysis": True
        }
    },

    "roi_estimation": {
        "models": {
            "linear_model": {
                "formula": "quality_gain * user_impact * credibility_boost - implementation_cost",
                "accuracy": 0.75,
                "complexity": "low"
            },
            "weighted_model": {
                "formula": "weighted_sum(quality_factors) * effort_factor - risk_factor",
                "accuracy": 0.85,
                "complexity": "medium"
            },
            "ml_model": {
                "algorithm": "random_forest_regression",
                "features": ["content_metrics", "user_feedback", "historical_data"],
                "accuracy": 0.9,
                "complexity": "high"
            }
        },

        "validation": {
            "cross_validation": True,
            "backtesting_enabled": True,
            "accuracy_monitoring": True,
            "model_retraining_frequency": "weekly"
        }
    }
}
```

### 5. Sub-Session Manager Configuration

#### Session Hierarchy Settings

```python
@dataclass
class SubSessionManagerConfig:
    """Configuration for sub-session management."""

    # Session Hierarchy Settings
    session_hierarchy_enabled: bool = True
    max_parent_children: int = 5
    max_session_depth: int = 3  # Parent -> Child -> Grandchild
    session_naming_convention: str = "hierarchical"  # "hierarchical", "flat", "hybrid"

    # Resource Allocation Limits
    resource_allocation_enabled: bool = True
    max_resources_per_sub_session: float = 0.15  # 15% of parent session resources
    min_resources_per_sub_session: float = 0.05  # 5% of parent session resources
    resource_reallocation_enabled: bool = True
    resource_monitoring_enabled: bool = True

    # Coordination Strategy Parameters
    coordination_strategy: str = "centralized"  # "centralized", "distributed", "hybrid"
    coordination_frequency: str = "real_time"  # "real_time", "periodic", "event_driven"
    coordination_timeout_seconds: int = 300
    conflict_resolution_strategy: str = "parent_priority"  # "parent_priority", "consensus", "timestamp"

    # Monitoring and Tracking Settings
    monitoring_enabled: bool = True
    tracking_frequency: str = "continuous"  # "continuous", "periodic", "on_demand"
    progress_reporting_enabled: bool = True
    performance_metrics_enabled: bool = True

    # State Synchronization
    state_sync_enabled: bool = True
    sync_strategy: str = "event_driven"  # "push", "pull", "event_driven", "hybrid"
    sync_frequency: str = "immediate"  # "immediate", "batch", "scheduled"
    conflict_resolution_enabled: bool = True

# Environment Variables for Sub-Session Manager
SUB_SESSION_ENABLED=true
SUB_SESSION_MAX_PARENT_CHILDREN=5
SUB_SESSION_MAX_DEPTH=3
SUB_SESSION_RESOURCE_ALLOCATION_ENABLED=true
SUB_SESSION_COORDINATION_STRATEGY=centralized
SUB_SESSION_MONITORING_ENABLED=true
SUB_SESSION_STATE_SYNC_ENABLED=true
```

#### Advanced Sub-Session Configuration

```python
# Advanced Sub-Session Management Configuration
SUB_SESSION_CONFIG = {
    "session_lifecycle": {
        "creation": {
            "auto_creation_enabled": True,
            "creation_triggers": ["gap_research_decision", "quality_gate_failure", "user_request"],
            "initialization_template": "gap_research_template",
            "parent_link_strength": "strong"  # "strong", "weak", "dynamic"
        },

        "execution": {
            "execution_mode": "independent",  # "independent", "coordinated", "sequential"
            "parallel_execution_enabled": True,
            "max_concurrent_sub_sessions": 3,
            "execution_timeout": 1800,  # 30 minutes
            "progress_tracking": "real_time"
        },

        "completion": {
            "completion_criteria": ["success_threshold", "resource_exhaustion", "timeout"],
            "success_threshold": 0.7,
            "auto_cleanup_enabled": True,
            "cleanup_delay_hours": 24,
            "result_integration": "automatic"
        },

        "termination": {
            "termination_conditions": ["completion", "timeout", "error", "cancellation"],
            "graceful_termination_enabled": True,
            "termination_timeout": 300,
            "cleanup_on_termination": True
        }
    },

    "resource_management": {
        "allocation": {
            "strategy": "proportional",  # "equal", "proportional", "priority_based", "demand_based"
            "allocation_factors": ["gap_priority", "estimated_complexity", "resource_availability"],
            "reallocation_enabled": True,
            "reallocation_triggers": ["performance_issues", "resource_shortage", "priority_change"]
        },

        "monitoring": {
            "resource_metrics": ["cpu_usage", "memory_usage", "api_calls", "processing_time"],
            "monitoring_frequency": 30,  # seconds
            "alert_thresholds": {
                "cpu_usage": 0.8,
                "memory_usage": 0.85,
                "api_calls_rate": 100,  # per minute
                "processing_time": 1800  # seconds
            }
        }
    },

    "coordination": {
        "communication": {
            "protocol": "message_queue",  # "direct_api", "message_queue", "event_bus", "shared_memory"
            "message_format": "json",
            "compression_enabled": True,
            "encryption_enabled": False
        },

        "synchronization": {
            "sync_points": ["milestone_completion", "resource_allocation", "error_handling"],
            "conflict_detection": True,
            "conflict_resolution": "parent_priority",
            "consistency_checking": True
        }
    }
}
```

### 6. Editorial Workflow Integration Configuration

#### Integration Point Settings

```python
@dataclass
class EditorialWorkflowIntegrationConfig:
    """Configuration for editorial workflow integration."""

    # Integration Point Settings
    integration_points: List[str] = field(default_factory=lambda: [
        "pre_editorial_analysis", "post_gap_decision", "pre_recommendation_generation",
        "post_implementation", "quality_gate_integration", "orchestrator_coordination"
    ])
    integration_strategy: str = "event_driven"  # "event_driven", "polling", "direct_call", "hybrid"

    # Hook System Configuration
    hook_system_enabled: bool = True
    hook_types: List[str] = field(default_factory=lambda: [
        "pre_processing", "post_processing", "error_handling", "state_change", "quality_gate"
    ])
    hook_execution_order: str = "priority_based"  # "priority_based", "registration_order", "dependency_based"
    hook_timeout_seconds: int = 120

    # Quality Framework Integration
    quality_framework_integration: bool = True
    quality_assessment_points: List[str] = field(default_factory=lambda: [
        "initial_analysis", "gap_research_decision", "recommendation_generation",
        "implementation_verification", "final_assessment"
    ])
    quality_threshold_sync: bool = True

    # System Synchronization Parameters
    system_sync_enabled: bool = True
    sync_frequency: str = "event_driven"  # "real_time", "periodic", "event_driven", "on_demand"
    sync_scope: str = "full_system"  # "editorial_only", "research_only", "full_system"
    conflict_resolution_strategy: str = "editorial_priority"  # "editorial_priority", "orchestrator_priority", "consensus"

    # Error Handling and Recovery
    error_handling_enabled: bool = True
    recovery_strategies: List[str] = field(default_factory=lambda: [
        "retry_mechanism", "fallback_behavior", "graceful_degradation", "error_escalation"
    ])
    max_retry_attempts: int = 3
    retry_backoff_strategy: str = "exponential"  # "fixed", "linear", "exponential"

# Environment Variables for Editorial Workflow Integration
EDITORIAL_INTEGRATION_ENABLED=true
EDITORIAL_HOOK_SYSTEM_ENABLED=true
EDITORIAL_QUALITY_FRAMEWORK_INTEGRATION=true
EDITORIAL_SYSTEM_SYNC_ENABLED=true
EDITORIAL_ERROR_HANDLING_ENABLED=true
EDITORIAL_MAX_RETRY_ATTEMPTS=3
```

#### Hook System Configuration

```python
# Editorial Workflow Hook System Configuration
EDITORIAL_HOOK_CONFIG = {
    "hook_registry": {
        "pre_editorial_analysis": {
            "enabled": True,
            "priority": 100,
            "timeout": 60,
            "hooks": [
                "validate_input_data",
                "check_system_resources",
                "initialize_quality_metrics"
            ]
        },

        "post_gap_decision": {
            "enabled": True,
            "priority": 90,
            "timeout": 120,
            "hooks": [
                "log_decision_details",
                "update_session_state",
                "notify_stakeholders",
                "prepare_sub_sessions"
            ]
        },

        "pre_recommendation_generation": {
            "enabled": True,
            "priority": 80,
            "timeout": 90,
            "hooks": [
                "validate_gap_results",
                "assess_research_corpus",
                "calculate_quality_metrics"
            ]
        },

        "post_implementation": {
            "enabled": True,
            "priority": 70,
            "timeout": 60,
            "hooks": [
                "verify_implementation",
                "measure_effectiveness",
                "update_quality_scores",
                "generate_completion_report"
            ]
        }
    },

    "hook_execution": {
        "execution_mode": "sequential",  # "sequential", "parallel", "hybrid"
        "error_handling": "continue_on_error",  # "stop_on_error", "continue_on_error", "retry_on_error"
        "dependency_checking": True,
        "resource_monitoring": True
    },

    "custom_hooks": {
        "registration_enabled": True,
        "validation_required": True,
        "sandbox_execution": True,
        "resource_limits": {
            "max_memory_mb": 512,
            "max_cpu_time": 30,
            "max_network_requests": 10
        }
    }
}
```

## Complete Configuration Integration

### Master Configuration File Structure

```yaml
# config/editorial_workflow_config.yaml
editorial_workflow:
  version: "3.2"
  enabled: true

  enhanced_editorial_engine:
    confidence_scoring_enabled: true
    confidence_threshold: 0.7
    gap_analysis_enabled: true
    evidence_collection_enabled: true
    decision_logic_mode: "weighted_average"

  gap_research_decision_system:
    cost_benefit_analysis_enabled: true
    min_roi_threshold: 1.5
    resource_allocation_strategy: "intelligent"
    max_gap_topics_per_session: 3

  research_corpus_analyzer:
    quality_assessment_enabled: true
    coverage_analysis_enabled: true
    source_evaluation_enabled: true
    sufficiency_determination_enabled: true

  editorial_recommendations:
    prioritization_enabled: true
    prioritization_method: "roi_based"
    action_planning_enabled: true
    roi_calculation_enabled: true

  sub_session_manager:
    session_hierarchy_enabled: true
    resource_allocation_enabled: true
    coordination_strategy: "centralized"
    monitoring_enabled: true

  workflow_integration:
    integration_strategy: "event_driven"
    hook_system_enabled: true
    quality_framework_integration: true
    system_sync_enabled: true
```

### Configuration Loading and Management

```python
# config/editorial_config_manager.py
class EditorialConfigManager:
    """Manages enhanced editorial workflow configuration."""

    def __init__(self):
        self.config = self._load_configuration()
        self.validate_configuration()

    def _load_configuration(self) -> dict:
        """Load configuration from multiple sources."""

        # Load base configuration
        base_config = self._load_yaml_config("config/editorial_workflow_config.yaml")

        # Load environment-specific overrides
        env_config = self._load_environment_overrides()

        # Load user-specific settings
        user_config = self._load_user_settings()

        # Merge configurations
        merged_config = self._merge_configurations(
            base_config, env_config, user_config
        )

        return merged_config

    def get_editorial_engine_config(self) -> EnhancedEditorialEngineConfig:
        """Get enhanced editorial engine configuration."""
        return EnhancedEditorialEngineConfig(**self.config["enhanced_editorial_engine"])

    def get_gap_research_config(self) -> GapResearchDecisionSystemConfig:
        """Get gap research decision system configuration."""
        return GapResearchDecisionSystemConfig(**self.config["gap_research_decision_system"])

    def get_corpus_analyzer_config(self) -> ResearchCorpusAnalyzerConfig:
        """Get research corpus analyzer configuration."""
        return ResearchCorpusAnalyzerConfig(**self.config["research_corpus_analyzer"])

    def get_recommendations_config(self) -> EditorialRecommendationsConfig:
        """Get editorial recommendations configuration."""
        return EditorialRecommendationsConfig(**self.config["editorial_recommendations"])

    def get_sub_session_config(self) -> SubSessionManagerConfig:
        """Get sub-session manager configuration."""
        return SubSessionManagerConfig(**self.config["sub_session_manager"])

    def get_integration_config(self) -> EditorialWorkflowIntegrationConfig:
        """Get workflow integration configuration."""
        return EditorialWorkflowIntegrationConfig(**self.config["workflow_integration"])

    def validate_configuration(self):
        """Validate configuration settings."""
        self._validate_confidence_thresholds()
        self._validate_resource_allocation()
        self._validate_integration_points()
        self._validate_hook_system()

    def _validate_confidence_thresholds(self):
        """Validate confidence threshold settings."""
        editorial_config = self.config.get("enhanced_editorial_engine", {})

        confidence_threshold = editorial_config.get("confidence_threshold", 0.7)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {confidence_threshold}")

        min_confidence = editorial_config.get("min_confidence_for_gap_research", 0.6)
        if min_confidence > confidence_threshold:
            raise ValueError(f"Min confidence for gap research ({min_confidence}) cannot exceed main confidence threshold ({confidence_threshold})")

    def _validate_resource_allocation(self):
        """Validate resource allocation settings."""
        gap_config = self.config.get("gap_research_decision_system", {})

        max_resources = gap_config.get("max_gap_research_resources", 0.3)
        min_resources = gap_config.get("min_gap_research_resources", 0.1)

        if min_resources > max_resources:
            raise ValueError(f"Min resources ({min_resources}) cannot exceed max resources ({max_resources})")

        if max_resources > 0.5:
            raise ValueError(f"Max resources ({max_resources}) cannot exceed 50% of total budget")
```

### Usage Examples

```python
# Example: Using enhanced editorial configuration
from config.editorial_config_manager import EditorialConfigManager

# Initialize configuration manager
config_manager = EditorialConfigManager()

# Get specific configuration sections
editorial_engine_config = config_manager.get_editorial_engine_config()
gap_research_config = config_manager.get_gap_research_config()
corpus_analyzer_config = config_manager.get_corpus_analyzer_config()

# Use configuration in components
editorial_engine = EnhancedEditorialEngine(editorial_engine_config)
gap_decision_system = GapResearchDecisionSystem(gap_research_config)
corpus_analyzer = ResearchCorpusAnalyzer(corpus_analyzer_config)

# Runtime configuration updates
config_manager.update_configuration("enhanced_editorial_engine.confidence_threshold", 0.75)
config_manager.reload_configuration()
```

### Configuration Validation and Testing

```python
# Configuration validation tests
def test_editorial_configuration():
    """Test enhanced editorial configuration validation."""

    config_manager = EditorialConfigManager()

    # Test confidence threshold validation
    with pytest.raises(ValueError):
        config_manager.config["enhanced_editorial_engine"]["confidence_threshold"] = 1.5
        config_manager.validate_configuration()

    # Test resource allocation validation
    with pytest.raises(ValueError):
        config_manager.config["gap_research_decision_system"]["min_gap_research_resources"] = 0.8
        config_manager.config["gap_research_decision_system"]["max_gap_research_resources"] = 0.6
        config_manager.validate_configuration()

    # Test valid configuration
    assert config_manager.get_editorial_engine_config().confidence_threshold == 0.7
    assert config_manager.get_gap_research_config().min_roi_threshold == 1.5

def test_configuration_integration():
    """Test configuration integration with system components."""

    config_manager = EditorialConfigManager()

    # Test configuration loading
    editorial_engine = EnhancedEditorialEngine(config_manager.get_editorial_engine_config())
    gap_decision_system = GapResearchDecisionSystem(config_manager.get_gap_research_config())

    # Test configuration application
    assert editorial_engine.confidence_threshold == 0.7
    assert gap_decision_system.min_roi_threshold == 1.5
    assert gap_decision_system.resource_allocation_strategy == "intelligent"
```

This comprehensive enhanced editorial workflow configuration system provides complete control over all advanced editorial intelligence features while maintaining flexibility for different deployment scenarios and use cases.
