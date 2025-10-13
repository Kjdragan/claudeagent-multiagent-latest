# Multi-Agent Research System - Complete Developer Guide

**System Version**: 2.0 Enhanced Architecture
**Last Updated**: October 13, 2025
**Status**: Production-Ready with Complete System Redesign

---

## Executive Overview

The Multi-Agent Research System is a sophisticated AI-powered platform that delivers comprehensive, high-quality research outputs through coordinated multi-agent workflows. This enhanced system represents a complete architectural redesign featuring a two-module scraping system, intelligent editorial processes, and seamless Claude Agent SDK integration.

**Key System Capabilities:**
- **Enhanced Workflow Pipeline**: Target URLs → Initial Research → First Draft → Editorial Review → Final Report
- **Two-Module Scraping System**: Advanced anti-bot escalation with AI-powered content cleaning
- **Intelligent Editorial Process**: Priority on leveraging existing research with confidence-based gap research
- **Standardized File Management**: Clear naming conventions and organized session structure
- **Claude Agent SDK Integration**: Full MCP compliance with intelligent session management

---

## System Architecture Overview

### Enhanced Workflow Pipeline

```
Target URL Generation → Initial Research → First Draft Report → Editorial Review
                                                              ↓
                                              [Gap Research Decision with Confidence Score]
                                                              ↓
                                          Editor Recommendations → Final Report
```

### Core System Components

1. **New Scraping & Cleaning System**: Two-module architecture with early termination
2. **Enhanced Editorial Agent**: Gap research as informed fallback, not automatic priority
3. **Unified Tool Interface**: Single entry point replacing multiple legacy tools
4. **Standardized File Management**: Consistent naming and organization patterns
5. **Sub-Session Management**: Gap research as child sessions of main research

### Directory Structure

```
multi_agent_research_system/
├── core/           # Orchestration, quality management, error recovery
├── agents/         # Specialized AI agents (research, report, editorial, quality)
├── tools/          # High-level research tools and search interfaces
├── utils/          # Web crawling, content processing, anti-bot detection
├── config/         # Agent definitions and system configuration
├── mcp_tools/      # Claude SDK integration with intelligent token management
├── scraping/       # NEW: Two-module scraping system
├── agent_logging/  # Comprehensive monitoring and debugging infrastructure
└── KEVIN/          # Session data storage and output organization
    └── sessions/
        └── {session_id}/
            ├── working/       # Agent work files
            ├── research/      # Research work products
            │   └── sub_sessions/  # Gap research sub-sessions
            └── logs/          # Progress and operation logs
```

---

## 1. New Two-Module Scraping System

### Architecture Overview

The new scraping system replaces all legacy scraping/cleaning components with a unified, efficient approach:

```python
@tool("comprehensive_research", "Unified research with gap support", {
    "query_type": str,              # "initial" | "editorial_gap"
    "queries": {
        "original": str,
        "reformulated": str,
        "orthogonal_1": str,
        "orthogonal_2": str
    },
    "target_success_count": int,    # 10 for initial, 3 for editorial gaps
    "session_id": str,
    "workproduct_prefix": str       # "INITIAL_SEARCH" | "EDITOR-GAP-X"
})
async def comprehensive_research_tool(args):
    """Main research tool replacing all legacy scraping tools"""
```

### Key Features

#### Progressive Anti-Bot Escalation
- **Level 1**: Basic headers and rate limiting
- **Level 2**: Enhanced headers with browser fingerprinting
- **Level 3**: Advanced techniques with proxy rotation
- **Level 4**: Stealth mode with full browser simulation

#### AI-Powered Content Cleaning
- GPT-5-nano integration for intelligent content extraction
- Usefulness judgment with quality scoring
- Early termination when targets are met
- Real-time progress tracking and logging

#### Success Tracking with Early Termination
```python
class SuccessTracker:
    def __init__(self, target_count, total_urls):
        self.target_count = target_count
        self.processed_urls = 0
        self.final_successes = 0
        self.completion_reached = False

    async def record_success(self, url, success_details):
        self.final_successes += 1

        # Real-time progress logging
        progress_message = (
            f"[SUCCESS] ✓ ({self.final_successes}/{self.target_count}) "
            f"Processed: {url}\n"
            f"  - Scrape: ✓ Clean: ✓ Useful: ✓ "
            f"({success_details['source_query']})"
        )

        # Check for early termination
        if self.final_successes >= self.target_count:
            self.completion_reached = True
            await self.handle_target_reached()
```

### Configuration-Driven Parameters

```yaml
# Initial Research Configuration
initial_research:
  target_success_count: 10
  max_total_urls: 20
  max_concurrent_scrapes: 40
  max_concurrent_cleans: 20
  workproduct_prefix: "INITIAL_SEARCH"
  query_expansion: true

# Editorial Gap Research Configuration
editorial_gap_research:
  target_success_count: 3
  max_total_urls: 8
  max_concurrent_scrapes: 20
  max_concurrent_cleans: 10
  workproduct_prefix: "EDITOR-GAP"
  query_expansion: false
  use_query_as_is: true
```

---

## 2. Enhanced Editorial Process

### Priority-Based Research Strategy

The enhanced editorial agent prioritizes leveraging existing research before conducting gap research:

```python
class EditorialAgent:
    def __init__(self):
        self.gap_research_threshold = 0.7  # Configurable confidence threshold
        self.max_gap_topics = 2

    async def review_first_draft_report(self, session_id: str, first_draft_report: str):
        """Comprehensive editorial review with intelligent gap research decisions"""

        # Step 1: Analyze existing research corpus
        existing_research = await self.analyze_existing_research(session_id)

        # Step 2: Assess report quality and integration
        quality_assessment = await self.assess_report_quality(
            first_draft_report, existing_research
        )

        # Step 3: Determine if gap research is needed
        gap_research_decision = await self.assess_gap_research_necessity(
            quality_assessment, existing_research
        )

        # Step 4: Execute based on decision
        if gap_research_decision["should_do_gap_research"]:
            gap_results = await self.execute_gap_research(
                session_id, gap_research_decision["gap_queries"]
            )
            return await self.create_editorial_recommendations(
                session_id, first_draft_report, existing_research, gap_results
            )
        else:
            return await self.create_editorial_recommendations(
                session_id, first_draft_report, existing_research
            )
```

### Confidence-Based Decision Making

```python
async def assess_gap_research_necessity(self, quality_assessment, existing_research):
    """Binary decision with confidence scoring"""

    # Analyze if existing research can address gaps
    existing_sufficiency = await self.assess_existing_sufficiency(
        quality_assessment, existing_research
    )

    # Calculate confidence scores for different gap areas
    gap_areas = [
        {
            "area": "factual_gaps",
            "confidence": self.calculate_factual_gap_confidence(quality_assessment),
            "existing_coverage": existing_sufficiency["factual_coverage"]
        },
        {
            "area": "temporal_gaps",
            "confidence": self.calculate_temporal_gap_confidence(quality_assessment),
            "existing_coverage": existing_sufficiency["temporal_coverage"]
        },
        {
            "area": "comparative_gaps",
            "confidence": self.calculate_comparative_gap_confidence(quality_assessment),
            "existing_coverage": existing_sufficiency["comparative_coverage"]
        }
    ]

    # Determine if gap research is needed
    high_confidence_gaps = [
        gap for gap in gap_areas
        if gap["confidence"] >= self.gap_research_threshold and
           gap["existing_coverage"] < 0.6
    ]

    should_do_gap_research = len(high_confidence_gaps) > 0

    return {
        "should_do_gap_research": should_do_gap_research,
        "confidence_scores": {gap["area"]: gap["confidence"] for gap in gap_areas},
        "existing_sufficiency": existing_sufficiency,
        "gap_queries": [gap["query"] for gap in sorted(high_confidence_gaps,
                                                key=lambda x: x["confidence"],
                                                reverse=True)[:self.max_gap_topics]],
        "recommendation": (
            "Gap research recommended" if should_do_gap_research
            else "Existing research sufficient"
        )
    }
```

### Sub-Session Management

Gap research is conducted as sub-sessions linked to the main research session:

```python
class SubSessionManager:
    def __init__(self):
        self.sub_sessions = {}  # parent_session_id -> [sub_session_ids]

    async def link_sub_session_to_parent(self, sub_session_id, parent_session_id, gap_query):
        """Create link between gap research and main session"""

        if parent_session_id not in self.sub_sessions:
            self.sub_sessions[parent_session_id] = []

        self.sub_sessions[parent_session_id].append({
            "sub_session_id": sub_session_id,
            "gap_query": gap_query,
            "created_at": datetime.now(),
            "status": "active"
        })

        # Store parent reference in sub-session
        await self.update_sub_session_metadata(sub_session_id, {
            "parent_session_id": parent_session_id,
            "gap_query": gap_query,
            "gap_type": "editorial_research"
        })
```

---

## 3. Standardized File Management System

### Directory Structure

```
KEVIN/sessions/{session_id}/
├── working/                           # Agent work files
│   ├── INITIAL_RESEARCH_DRAFT.md      # First draft report
│   ├── EDITORIAL_REVIEW.md            # Editorial analysis and recommendations
│   ├── EDITORIAL_RECOMMENDATIONS.md   # Final editor recommendations
│   └── FINAL_REPORT.md                # Final improved report
├── research/                          # Research work products
│   ├── INITIAL_SEARCH_WORKPRODUCT.md  # Initial comprehensive research
│   ├── sub_sessions/                  # Gap research sub-sessions
│   │   ├── gap_1/
│   │   │   └── EDITOR-GAP-1_WORKPRODUCT.md
│   │   └── gap_2/
│   │       └── EDITOR-GAP-2_WORKPRODUCT.md
│   └── session_state.json             # Session metadata and links
└── logs/                             # Progress and operation logs
    ├── progress.log
    ├── editorial_decisions.log
    └── gap_research.log
```

### File Naming Conventions

```python
class FileManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.base_path = Path(f"KEVIN/sessions/{session_id}")

    def create_working_filename(self, stage: str, description: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{stage}_{description}_{timestamp}.md"

    def create_research_workproduct_name(self, prefix: str, query_type: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_WORKPRODUCT_{timestamp}.md"

    def get_file_paths(self):
        return {
            "initial_draft": self.base_path / "working" / self.create_working_filename(
                "INITIAL_RESEARCH", "DRAFT"
            ),
            "editorial_review": self.base_path / "working" / self.create_working_filename(
                "EDITORIAL", "REVIEW"
            ),
            "editorial_recommendations": self.base_path / "working" / self.create_working_filename(
                "EDITORIAL", "RECOMMENDATIONS"
            ),
            "final_report": self.base_path / "working" / self.create_working_filename(
                "FINAL", "REPORT"
            ),
            "initial_research": self.base_path / "research" / self.create_research_workproduct_name(
                "INITIAL_SEARCH", "comprehensive"
            ),
            "progress_log": self.base_path / "logs" / "progress.log"
        }
```

### Session State Management

```python
class SessionStateManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state_file = Path(f"KEVIN/sessions/{session_id}/research/session_state.json")

    async def initialize_session(self, initial_query: str):
        session_state = {
            "session_id": self.session_id,
            "initial_query": initial_query,
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
            "stages": {
                "target_generation": {"status": "pending"},
                "initial_research": {"status": "pending"},
                "first_draft": {"status": "pending"},
                "editorial_review": {"status": "pending"},
                "gap_research": {"status": "pending"},
                "final_report": {"status": "pending"}
            },
            "sub_sessions": [],
            "file_mappings": {},
            "research_metrics": {
                "total_urls_processed": 0,
                "successful_scrapes": 0,
                "successful_cleans": 0,
                "useful_content_count": 0
            }
        }

        await self.save_session_state(session_state)
        return session_state
```

---

## 4. Claude Agent SDK Integration

### Tool Interface Design

The system provides seamless integration with the Claude Agent SDK through well-defined MCP tools:

```python
from claude_agent_sdk import tool

@tool("comprehensive_research", "Unified research with gap support", {
    "query_type": str,
    "queries": dict,
    "target_success_count": int,
    "session_id": str,
    "workproduct_prefix": str
})
async def comprehensive_research_tool(args):
    """Main research tool for the multi-agent system"""

    # Implementation as described in Section 1
    pass

@tool("get_session_data", "Retrieve session research data", {
    "session_id": str,
    "data_type": str,  # "research" | "report" | "editorial" | "all"
    "include_sub_sessions": bool
})
async def get_session_data_tool(args):
    """Retrieve session data for agent analysis"""

    session_manager = SubSessionManager()
    integrated_context = await session_manager.get_integrated_research_context(
        args["session_id"]
    )

    return integrated_context
```

### Session Management Approaches

```python
class ClaudeSDKSessionManager:
    def __init__(self):
        self.active_sessions: dict[str, ClaudeSDKClient] = {}
        self.session_config = ClaudeAgentOptions(
            max_turns=50,
            continue_conversation=True,
            include_partial_messages=True,
            enable_hooks=True
        )

    async def create_session(self, session_id: str, agent_type: str) -> ClaudeSDKClient:
        """Create SDK session for specific agent type"""

        # Configure agent-specific tools
        if agent_type == "research":
            tools = ["comprehensive_research", "get_session_data"]
        elif agent_type == "editorial":
            tools = ["comprehensive_research", "get_session_data", "assess_content_quality"]
        else:
            tools = ["get_session_data"]

        # Create MCP server with appropriate tools
        mcp_server = create_sdk_mcp_server(tools)

        options = ClaudeAgentOptions(
            mcp_servers={"research": mcp_server},
            allowed_tools=tools,
            **self.session_config.__dict__
        )

        client = ClaudeSDKClient(options)
        self.active_sessions[session_id] = client

        return client

    async def coordinate_agent_handoff(self, session_id: str,
                                     from_agent: str, to_agent: str,
                                     handoff_data: dict):
        """Coordinate control handoff between agents"""

        # Clean up current session
        if session_id in self.active_sessions:
            await self.active_sessions[session_id].close()
            del self.active_sessions[session_id]

        # Create new session for target agent
        new_client = await self.create_session(session_id, to_agent)

        # Prepare handoff context
        handoff_prompt = self._create_handoff_prompt(
            from_agent, to_agent, handoff_data
        )

        return await new_client.query(handoff_prompt)
```

### Context Management Strategies

```python
class ContextManager:
    def __init__(self):
        self.context_cache: dict[str, dict] = {}
        self.max_context_size = 100000  # characters

    async def prepare_agent_context(self, session_id: str, agent_type: str) -> dict:
        """Prepare context for specific agent type"""

        if session_id in self.context_cache:
            cached_context = self.context_cache[session_id]
            if cached_context["agent_type"] == agent_type:
                return cached_context["data"]

        # Build fresh context
        session_manager = SubSessionManager()
        integrated_context = await session_manager.get_integrated_research_context(
            session_id
        )

        # Format for specific agent type
        if agent_type == "editorial":
            formatted_context = await session_manager.format_editorial_context(
                integrated_context
            )
        elif agent_type == "report":
            formatted_context = await session_manager.format_report_context(
                integrated_context
            )
        else:
            formatted_context = integrated_context

        # Cache context
        self.context_cache[session_id] = {
            "agent_type": agent_type,
            "data": formatted_context,
            "created_at": datetime.now()
        }

        return formatted_context

    def _create_handoff_prompt(self, from_agent: str, to_agent: str,
                              handoff_data: dict) -> str:
        """Create context-rich handoff prompt"""

        return f"""
        You are taking over from {from_agent} as the {to_agent} agent.

        Previous work completed:
        {handoff_data.get('previous_results', 'No previous results available')}

        Your specific task:
        {handoff_data.get('task_description', 'Complete the research workflow')}

        Available research data:
        {handoff_data.get('research_context', 'No research context available')}

        Please proceed with your specific responsibilities in the research workflow.
        """
```

---

## 5. Enhanced Orchestrator & Workflow

### Complete Research Workflow

```python
class EnhancedOrchestrator:
    def __init__(self):
        self.research_tool = comprehensive_research_tool
        self.editorial_agent = EditorialAgent()
        self.report_agent = ReportAgent()
        self.file_manager = FileManager()
        self.session_manager = SessionStateManager()
        self.sdk_manager = ClaudeSDKSessionManager()

    async def execute_complete_workflow(self, initial_query: str):
        """Execute complete research workflow with enhanced editorial process"""

        # Initialize session
        session_id = self.generate_session_id()
        await self.session_manager.initialize_session(initial_query)

        try:
            # Stage 1: Initial Research
            await self.session_manager.update_stage_status("initial_research", "running")
            initial_research = await self.execute_initial_research(session_id, initial_query)
            await self.session_manager.update_stage_status("initial_research", "completed",
                                                         {"workproduct_path": initial_research["workproduct_path"]})

            # Stage 2: First Draft Report
            await self.session_manager.update_stage_status("first_draft", "running")
            first_draft = await self.generate_first_draft_report(session_id, initial_research)
            await self.session_manager.update_stage_status("first_draft", "completed",
                                                         {"report_path": first_draft["report_path"]})

            # Stage 3: Editorial Review
            await self.session_manager.update_stage_status("editorial_review", "running")
            editorial_review = await self.editorial_agent.review_first_draft_report(
                session_id, first_draft["content"]
            )
            await self.session_manager.update_stage_status("editorial_review", "completed",
                                                         {"review_path": editorial_review["review_path"]})

            # Stage 4: Gap Research (if needed)
            if editorial_review.get("gap_research_completed"):
                await self.session_manager.update_stage_status("gap_research", "completed",
                                                             {"gap_count": len(editorial_review["gap_results"])})
            else:
                await self.session_manager.update_stage_status("gap_research", "skipped",
                                                             {"reason": "existing_research_sufficient"})

            # Stage 5: Final Report
            await self.session_manager.update_stage_status("final_report", "running")
            final_report = await self.generate_final_report(
                session_id, first_draft, editorial_review
            )
            await self.session_manager.update_stage_status("final_report", "completed",
                                                         {"final_path": final_report["report_path"]})

            return {
                "session_id": session_id,
                "status": "completed",
                "final_report_path": final_report["report_path"],
                "editorial_recommendations": editorial_review,
                "research_summary": await self.create_research_summary(session_id)
            }

        except Exception as e:
            await self.session_manager.update_stage_status("error", "failed", {"error": str(e)})
            raise

    async def execute_initial_research(self, session_id: str, initial_query: str):
        """Execute initial comprehensive research"""

        # Generate targeted URLs (existing system)
        targeted_urls = await self.generate_targeted_urls(initial_query)

        # Execute research with new scraping system
        research_result = await self.research_tool({
            "query_type": "initial",
            "queries": {
                "original": initial_query,
                "reformulated": await self.reformulate_query(initial_query),
                "orthogonal_1": await self.generate_orthogonal_query_1(initial_query),
                "orthogonal_2": await self.generate_orthogonal_query_2(initial_query)
            },
            "target_success_count": 10,
            "session_id": session_id,
            "workproduct_prefix": "INITIAL_SEARCH"
        })

        return research_result
```

### Quality Management Integration

```python
class QualityGatedWorkflow:
    def __init__(self):
        self.quality_framework = QualityFramework()
        self.quality_gate_manager = QualityGateManager()

    async def execute_with_quality_gates(self, stage: str, content: str,
                                       context: dict) -> dict:
        """Execute stage with comprehensive quality management"""

        # Quality assessment
        assessment = await self.quality_framework.assess_content(content, context)

        # Quality gate evaluation
        gate_result = await self.quality_gate_manager.evaluate_stage_output(
            stage, {"content": content, "context": context, "assessment": assessment}
        )

        if gate_result.decision == GateDecision.PROCEED:
            return {"success": True, "content": content, "assessment": assessment}
        elif gate_result.decision == GateDecision.ENHANCE:
            # Apply progressive enhancement
            enhanced_content = await self.apply_progressive_enhancement(
                content, assessment, context
            )
            return {"success": True, "content": enhanced_content, "assessment": assessment}
        else:
            # Quality too low, require rerun
            return {"success": False, "reason": "Quality threshold not met", "assessment": assessment}
```

---

## 6. Configuration Management

### Master Configuration File

```yaml
# multi_agent_research_system/config/system_config.yaml
system:
  version: "2.0"
  session_timeout_hours: 24
  max_concurrent_sessions: 10

research:
  initial:
    target_success_count: 10
    max_total_urls: 20
    max_concurrent_scrapes: 40
    max_concurrent_cleans: 20
    workproduct_prefix: "INITIAL_SEARCH"
    query_expansion: true

  editorial_gap:
    target_success_count: 3
    max_total_urls: 8
    max_concurrent_scrapes: 20
    max_concurrent_cleans: 10
    workproduct_prefix: "EDITOR-GAP"
    query_expansion: false
    use_query_as_is: true
    max_gap_topics: 2

editorial:
  gap_research_confidence_threshold: 0.7
  prioritize_existing_research: true
  comprehensive_analysis_required: true

file_management:
  base_directory: "KEVIN/sessions"
  working_subdirectory: "working"
  research_subdirectory: "research"
  logs_subdirectory: "logs"

logging:
  real_time_progress: true
  detailed_errors: false
  editorial_decisions: true
  gap_research_tracking: true

claude_sdk:
  max_turns: 50
  continue_conversation: true
  include_partial_messages: true
  enable_hooks: true
```

### Environment Setup

```bash
# Required API keys
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export SERP_API_KEY="your-serp-key"

# Optional configuration
export LOGFIRE_TOKEN="your-logfire-token"  # For enhanced monitoring
export RESEARCH_QUALITY_THRESHOLD="0.8"    # Default quality threshold
export MAX_SEARCH_RESULTS="10"              # Default search result limit
export DEBUG_MODE="false"                   # Enable debug logging
```

### Dynamic Configuration Loading

```python
class ConfigurationManager:
    def __init__(self):
        self.config = {}
        self.load_configuration()

    def load_configuration(self):
        """Load configuration from multiple sources"""

        # Load base configuration
        config_file = Path("multi_agent_research_system/config/system_config.yaml")
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)

        # Override with environment variables
        self.config.update({
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "serp_api_key": os.getenv("SERP_API_KEY"),
            "debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true",
            "quality_threshold": float(os.getenv("RESEARCH_QUALITY_THRESHOLD", "0.8"))
        })

        # Validate required configuration
        self.validate_configuration()

    def get_research_config(self, query_type: str) -> dict:
        """Get research configuration for specific query type"""
        return self.config.get("research", {}).get(query_type, {})

    def get_editorial_config(self) -> dict:
        """Get editorial configuration"""
        return self.config.get("editorial", {})
```

---

## 7. Development Guidelines

### System Design Principles

1. **Quality-First Architecture**: Every component includes quality assessment and enhancement
2. **Resilience & Recovery**: Comprehensive error handling and recovery mechanisms
3. **Scalability**: Async-first design with resource management
4. **Observability**: Extensive logging and monitoring capabilities
5. **Clean Separation**: Clear boundaries between components with well-defined interfaces

### Working with the New System

#### Adding New Agents

```python
from multi_agent_research_system.agents.base_agent import BaseAgent
from multi_agent_research_system.core.quality_framework import QualityAssessment

class CustomAgent(BaseAgent):
    agent_type = "custom"

    async def process_task(self, task_data):
        """Process task with quality management"""

        # Execute core functionality
        result = await self.execute_core_logic(task_data)

        # Quality assessment
        assessment = await self.assess_output_quality(result)

        # Return with quality metadata
        return {
            "result": result,
            "quality_assessment": assessment,
            "agent_type": self.agent_type
        }

    async def execute_core_logic(self, task_data):
        """Implement agent-specific logic"""
        # Your custom implementation here
        pass

    async def assess_output_quality(self, result):
        """Assess quality of agent output"""
        # Use quality framework for consistent assessment
        quality_framework = QualityFramework()
        return await quality_framework.assess_content(
            result, {"agent_type": self.agent_type}
        )
```

#### Adding New Tools

```python
from claude_agent_sdk import tool

@tool("custom_tool", "Custom tool description", {
    "param1": str,
    "param2": int,
    "session_id": str
})
async def custom_tool(args):
    """Custom tool implementation with MCP compliance"""

    session_id = args["session_id"]

    # Validate session
    session_manager = SessionStateManager()
    if not await session_manager.validate_session(session_id):
        return {"error": "Invalid session ID"}

    # Execute tool logic
    result = await execute_custom_logic(args)

    # Log tool usage
    logger = AgentLogger("custom_tool")
    await logger.log_info("Tool executed", {
        "session_id": session_id,
        "parameters": args,
        "result_summary": summarize_result(result)
    })

    return {
        "content": [{"type": "text", "text": str(result)}],
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    }
```

### Testing Approaches

#### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_comprehensive_research_tool():
    """Test the comprehensive research tool"""

    # Mock dependencies
    with patch('multi_agent_research_system.scraping.scraping_engine.ScrapingEngine') as mock_engine:
        mock_engine.return_value.process_urls = AsyncMock(return_value={
            "success_count": 10,
            "results": [{"url": "test.com", "content": "test content"}]
        })

        # Test execution
        result = await comprehensive_research_tool({
            "query_type": "initial",
            "queries": {
                "original": "test query",
                "reformulated": "reformulated test",
                "orthogonal_1": "orthogonal test 1",
                "orthogonal_2": "orthogonal test 2"
            },
            "target_success_count": 10,
            "session_id": "test_session",
            "workproduct_prefix": "TEST_SEARCH"
        })

        # Assertions
        assert result["status"] == "completed"
        assert result["success_count"] == 10
        assert "workproduct_path" in result

@pytest.mark.asyncio
async def test_editorial_gap_decision():
    """Test editorial gap research decision making"""

    editorial_agent = EditorialAgent()

    # Test scenario where existing research is sufficient
    sufficient_research = {
        "factual_coverage": 0.8,
        "temporal_coverage": 0.7,
        "comparative_coverage": 0.75
    }

    decision = await editorial_agent.assess_gap_research_necessity(
        quality_assessment={"overall_score": 85},
        existing_research=sufficient_research
    )

    assert decision["should_do_gap_research"] is False
    assert decision["recommendation"] == "Existing research sufficient"
```

#### Integration Testing

```python
@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete research workflow"""

    orchestrator = EnhancedOrchestrator()

    # Execute complete workflow
    result = await orchestrator.execute_complete_workflow(
        "artificial intelligence in healthcare"
    )

    # Verify workflow completion
    assert result["status"] == "completed"
    assert "final_report_path" in result
    assert "editorial_recommendations" in result

    # Verify file creation
    final_report_path = Path(result["final_report_path"])
    assert final_report_path.exists()
    assert final_report_path.stat().st_size > 1000  # Reasonable file size
```

### Common Patterns and Best Practices

#### Error Handling

```python
class ResearchSystemError(Exception):
    """Base exception for research system errors"""
    pass

class ConfigurationError(ResearchSystemError):
    """Configuration-related errors"""
    pass

class ScrapingError(ResearchSystemError):
    """Scraping-related errors"""
    pass

async def handle_system_error(error: Exception, context: dict) -> dict:
    """Handle system errors with appropriate recovery strategies"""

    logger = AgentLogger("error_handler")

    if isinstance(error, ScrapingError):
        logger.error("Scraping error occurred", {
            "error": str(error),
            "context": context,
            "recovery_strategy": "retry_with_alternative_source"
        })

        # Implement recovery logic
        return await retry_with_alternative_source(context)

    elif isinstance(error, ConfigurationError):
        logger.error("Configuration error", {
            "error": str(error),
            "context": context
        })

        # Configuration errors are typically fatal
        return {"success": False, "error": "Configuration error"}

    else:
        logger.error("Unexpected error", {
            "error": str(error),
            "context": context,
            "error_type": type(error).__name__
        })

        return {"success": False, "error": "Unexpected system error"}
```

#### Performance Optimization

```python
class PerformanceOptimizer:
    def __init__(self):
        self.cache = {}
        self.performance_metrics = {}

    async def cached_research_execution(self, query_hash: str, research_func, *args):
        """Cache research results to avoid duplicate work"""

        if query_hash in self.cache:
            cached_result = self.cache[query_hash]
            if self._is_cache_valid(cached_result):
                return cached_result["result"]

        # Execute research
        result = await research_func(*args)

        # Cache result
        self.cache[query_hash] = {
            "result": result,
            "timestamp": datetime.now(),
            "ttl": timedelta(hours=1)
        }

        return result

    def _is_cache_valid(self, cached_result: dict) -> bool:
        """Check if cached result is still valid"""
        return datetime.now() - cached_result["timestamp"] < cached_result["ttl"]

    async def monitor_performance(self, operation: str, duration: float):
        """Monitor operation performance"""

        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []

        self.performance_metrics[operation].append({
            "duration": duration,
            "timestamp": datetime.now()
        })

        # Alert on performance degradation
        if len(self.performance_metrics[operation]) > 10:
            avg_duration = sum(m["duration"] for m in self.performance_metrics[-10:]) / 10
            if avg_duration > self.get_performance_threshold(operation):
                await self.alert_performance_degradation(operation, avg_duration)
```

---

## 8. Quick Start Guide

### Running a Complete Research Workflow

#### Basic Usage

```bash
# Run research with default settings
python run_research.py "latest developments in quantum computing"

# Run with specific parameters
python run_research.py "climate change impacts" \
  --depth "Comprehensive Analysis" \
  --audience "Academic" \
  --quality-threshold 0.8

# Run with debug mode
python run_research.py "artificial intelligence trends" --debug
```

#### Programmatic Usage

```python
import asyncio
from multi_agent_research_system.core.orchestrator import EnhancedOrchestrator

async def run_research_example():
    # Initialize orchestrator
    orchestrator = EnhancedOrchestrator()

    # Execute complete workflow
    result = await orchestrator.execute_complete_workflow(
        "sustainable energy technologies"
    )

    print(f"Research completed: {result['status']}")
    print(f"Final report: {result['final_report_path']}")
    print(f"Quality assessment: {result['research_summary']['quality_score']}")

# Run the example
asyncio.run(run_research_example())
```

### Testing Individual Components

#### Test the Scraping System

```python
from multi_agent_research_system.scraping.scraping_engine import ScrapingEngine

async def test_scraping_system():
    engine = ScrapingEngine()

    result = await engine.process_urls(
        queries={"original": "renewable energy"},
        session_id="test_session",
        success_tracker=SuccessTracker(target_count=5, total_urls=10),
        config=engine.get_config("initial")
    )

    print(f"Scraping completed: {result['success_count']} successful results")
    print(f"Workproduct: {result['workproduct_path']}")

asyncio.run(test_scraping_system())
```

#### Test the Editorial Agent

```python
from multi_agent_research_system.agents.enhanced_editorial_agent import EditorialAgent

async def test_editorial_agent():
    editorial_agent = EditorialAgent()

    # Simulate first draft report
    first_draft = """
    # Renewable Energy Report

    This report covers recent developments in renewable energy.
    Solar and wind power are the main focus.

    ## Solar Energy
    Solar power has seen significant improvements in efficiency.

    ## Wind Energy
    Wind turbines are becoming more efficient.
    """

    result = await editorial_agent.review_first_draft_report(
        session_id="test_session",
        first_draft_report=first_draft
    )

    print(f"Gap research needed: {result['gap_research_decision']['should_do_gap_research']}")
    print(f"Recommendations: {len(result['recommendations']} items")

asyncio.run(test_editorial_agent())
```

### Debugging Issues

#### Enable Comprehensive Logging

```python
import logging
from multi_agent_research_system.core.logging_config import setup_logging

# Setup debug logging
setup_logging(level=logging.DEBUG,
             log_file="debug.log",
             console_output=True)

# Your code here
```

#### Monitor Session Progress

```python
from multi_agent_research_system.core.workflow_state import WorkflowStateManager

async def monitor_session(session_id: str):
    state_manager = WorkflowStateManager()

    while True:
        session = await state_manager.get_session(session_id)
        status = await state_manager.get_session_status(session_id)

        print(f"Current stage: {status['current_stage']}")
        print(f"Progress: {status['progress_percentage']}%")
        print(f"Quality score: {status.get('quality_assessment', {}).get('overall_score', 'N/A')}")

        if status['status'] in ['completed', 'error']:
            break

        await asyncio.sleep(10)  # Check every 10 seconds

# Usage
asyncio.run(monitor_session("your_session_id"))
```

#### Common Debugging Scenarios

```python
# Debug scraping issues
async def debug_scraping(session_id: str):
    session_manager = SessionStateManager()
    session_state = await session_manager.load_session_state()

    print(f"Total URLs processed: {session_state['research_metrics']['total_urls_processed']}")
    print(f"Successful scrapes: {session_state['research_metrics']['successful_scrapes']}")
    print(f"Successful cleans: {session_state['research_metrics']['successful_cleans']}")

    # Check progress log
    progress_log = Path(f"KEVIN/sessions/{session_id}/logs/progress.log")
    if progress_log.exists():
        with open(progress_log, 'r') as f:
            print("Recent progress:")
            print(''.join(f.readlines()[-20:]))  # Last 20 lines

# Debug editorial decisions
async def debug_editorial(session_id: str):
    editorial_log = Path(f"KEVIN/sessions/{session_id}/logs/editorial_decisions.log")
    if editorial_log.exists():
        with open(editorial_log, 'r') as f:
            print("Editorial decisions:")
            print(f.read())
```

### Extending Functionality

#### Add Custom Quality Criteria

```python
from multi_agent_research_system.core.quality_framework import QualityCriterion, CriterionResult

class CustomQualityCriterion(QualityCriterion):
    """Custom quality criterion for specific domain requirements"""

    def __init__(self):
        self.name = "domain_specific_quality"
        self.weight = 0.15

    async def evaluate(self, content: str, context: dict) -> CriterionResult:
        """Evaluate content against custom criteria"""

        # Your custom evaluation logic here
        score = self.calculate_custom_score(content, context)
        issues = self.identify_custom_issues(content, context)
        recommendations = self.generate_custom_recommendations(issues)

        return CriterionResult(
            name=self.name,
            score=score,
            weight=self.weight,
            feedback=self._generate_feedback(score, issues),
            specific_issues=issues,
            recommendations=recommendations,
            evidence={"custom_metrics": self.extract_custom_metrics(content)}
        )

    def calculate_custom_score(self, content: str, context: dict) -> float:
        # Implement your scoring logic
        pass

# Register the custom criterion
quality_framework = QualityFramework()
quality_framework.add_criterion(CustomQualityCriterion())
```

#### Add Custom Research Sources

```python
from multi_agent_research_system.scraping.scraping_engine import ScrapingEngine

class CustomScrapingEngine(ScrapingEngine):
    """Custom scraping engine with additional sources"""

    async def generate_targeted_urls(self, query: str, query_type: str) -> list[str]:
        """Generate URLs including custom sources"""

        # Get standard URLs
        standard_urls = await super().generate_targeted_urls(query, query_type)

        # Add custom sources
        custom_urls = await self.get_custom_source_urls(query)

        return standard_urls + custom_urls

    async def get_custom_source_urls(self, query: str) -> list[str]:
        """Get URLs from custom sources"""

        # Implement custom source logic
        # Examples: academic databases, internal repositories, specialized APIs
        custom_sources = [
            f"https://custom-academic.edu/search?q={query}",
            f"https://internal-repo.company.com/search?query={query}",
            f"https://specialized-api.org/data?search={query}"
        ]

        return custom_sources
```

---

## Migration from Legacy System

### What Was Removed

1. **Legacy Scraping Components**:
   - `enhanced_search_scrape_clean.py`
   - `zplayground1_search.py`
   - `advanced_scraping_tool.py`
   - Multiple content cleaning modules

2. **Legacy Configuration Files**:
   - `settings_broken.py`
   - Broken or obsolete configuration modules

3. **Inefficient Workflows**:
   - Automatic gap research without confidence assessment
   - Multiple competing scraping tools
   - Inconsistent file naming and organization

### What Was Enhanced

1. **Unified Research Tool**: Single comprehensive tool replacing multiple legacy tools
2. **Intelligent Editorial Process**: Confidence-based gap research decisions
3. **Standardized File Management**: Consistent naming and organization
4. **Enhanced Quality Management**: Comprehensive quality assessment and enhancement
5. **Better Claude SDK Integration**: Improved session management and agent coordination

### Migration Steps

1. **Update Dependencies**: Ensure new scraping system dependencies are installed
2. **Update Configuration**: Migrate to new configuration system
3. **Update Tool Usage**: Replace legacy tool calls with `comprehensive_research_tool`
4. **Update File Paths**: Use new standardized file management system
5. **Update Agent Logic**: Adapt to new editorial process and quality management

### Example Migration

**Before (Legacy)**:
```python
# Old approach with multiple tools
research_result = await enhanced_search_scrape_clean_tool({
    "query": query,
    "num_results": 20,
    "auto_crawl": True
})

# Separate gap research
gap_result = await zplayground1_search_tool({
    "query": gap_query,
    "search_mode": "news"
})
```

**After (Enhanced)**:
```python
# New unified approach
research_result = await comprehensive_research_tool({
    "query_type": "initial",
    "queries": {
        "original": query,
        "reformulated": reformulated_query,
        "orthogonal_1": orthogonal_1,
        "orthogonal_2": orthogonal_2
    },
    "target_success_count": 10,
    "session_id": session_id,
    "workproduct_prefix": "INITIAL_SEARCH"
})

# Gap research through editorial agent decision making
# (handled automatically by the enhanced editorial process)
```

---

## Performance & Optimization

### Performance Targets

- **Initial Research Success Rate**: ≥ 60% (6+ successful results from 10 target)
- **Gap Research Success Rate**: ≥ 70% (2+ successful results from 3 target)
- **Processing Time**: ≤ 5 minutes for initial research, ≤ 2 minutes for gap research
- **File Organization Consistency**: 100% standardized naming and structure
- **Editorial Decision Accuracy**: ≥ 80% appropriate gap research decisions

### Optimization Strategies

1. **Concurrent Processing**: Multiple agents work in parallel where possible
2. **Intelligent Caching**: Cache frequently accessed data and results
3. **Resource Management**: Monitor and manage system resources
4. **Quality vs. Speed**: Configurable trade-offs between quality and performance

### Monitoring Performance

```python
from multi_agent_research_system.monitoring.performance_monitor import PerformanceMonitor

# Initialize performance monitoring
monitor = PerformanceMonitor()

# Track operation performance
async def tracked_research_execution(orchestrator, query):
    start_time = time.time()

    result = await orchestrator.execute_complete_workflow(query)

    duration = time.time() - start_time
    monitor.track_operation("complete_workflow", duration, {
        "success": result["status"] == "completed",
        "quality_score": result["research_summary"]["quality_score"]
    })

    return result

# Get performance summary
summary = monitor.get_performance_summary()
print(f"Average workflow duration: {summary['operations']['complete_workflow']['average_duration']:.2f}s")
print(f"Success rate: {summary['operations']['complete_workflow']['success_rate']:.2%}")
```

---

## Conclusion

The enhanced Multi-Agent Research System represents a complete architectural redesign that delivers:

1. **Clean Architecture**: Removal of legacy components with modern, efficient replacement
2. **Intelligent Editorial Process**: Gap research as informed fallback, not automatic priority
3. **Enhanced User Experience**: Clear file organization and consistent naming
4. **Improved Performance**: Early termination and confidence-based decision making
5. **Better Integration**: Seamless Claude Agent SDK usage with proper session management

The system prioritizes leveraging existing research investments while providing targeted gap research when genuinely needed. All components are designed to work together cohesively with clear data contracts and standardized interfaces.

**System Status**: ✅ Production-Ready with Enhanced Architecture
**Implementation Status**: ✅ Complete System Redesign
**Migration Approach**: ✅ Clean break from legacy with comprehensive migration guide

---

## Appendix: Common Reference Materials

### Configuration Reference

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `target_success_count` | Number of successful results to target | 10 (initial), 3 (gap) | 1-20 |
| `max_total_urls` | Maximum URLs to process | 20 (initial), 8 (gap) | 5-50 |
| `gap_research_confidence_threshold` | Confidence threshold for gap research | 0.7 | 0.5-0.9 |
| `quality_threshold` | Minimum quality score for progression | 0.75 | 0.5-0.95 |

### File Naming Patterns

| File Type | Pattern | Example |
|-----------|---------|---------|
| Initial Draft | `INITIAL_RESEARCH_DRAFT_YYYYMMDD_HHMMSS.md` | `INITIAL_RESEARCH_DRAFT_20251013_143022.md` |
| Editorial Review | `EDITORIAL_REVIEW_YYYYMMDD_HHMMSS.md` | `EDITORIAL_REVIEW_20251013_150315.md` |
| Final Report | `FINAL_REPORT_YYYYMMDD_HHMMSS.md` | `FINAL_REPORT_20251013_154522.md` |
| Research Workproduct | `{PREFIX}_WORKPRODUCT_YYYYMMDD_HHMMSS.md` | `INITIAL_SEARCH_WORKPRODUCT_20251013_142205.md` |
| Gap Research | `EDITOR-GAP-{N}_WORKPRODUCT_YYYYMMDD_HHMMSS.md` | `EDITOR-GAP-1_WORKPRODUCT_20251013_151230.md` |

### Agent Handoff Patterns

```
Research Agent → Report Agent → Editorial Agent → [Gap Research Decision] → Final Report
                                     ↓
                                [Gap Research Sub-sessions]
                                     ↓
                                Enhanced Editorial Review
```

### Error Recovery Hierarchy

1. **Retry with Backoff**: Temporary issues (network timeouts, rate limits)
2. **Fallback Function**: Alternative approaches (different search sources, simplified logic)
3. **Minimal Execution**: Core functionality only (reduced scope, basic features)
4. **Skip Stage**: Non-critical failures (optional enhancements, nice-to-have features)
5. **Abort Workflow**: Critical failures (authentication issues, system errors)

---

**For support and questions, refer to the agent_logging directory for comprehensive system logs and monitoring tools.**