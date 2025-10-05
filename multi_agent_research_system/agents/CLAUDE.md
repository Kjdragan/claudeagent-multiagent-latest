# Agents Directory - Multi-Agent Research System

This directory contains specialized AI agent implementations that work together to perform comprehensive research tasks.

## Directory Purpose

The agents directory provides specialized AI agents, each with distinct responsibilities and capabilities, that collaborate through the multi-agent research system to deliver high-quality research outputs. Each agent is designed to excel at specific aspects of the research workflow.

## Key Components

### Core Research Agents
- **`research_agent.py`** - Expert research agent for web research, source validation, and information synthesis
- **`report_agent.py`** - Report generation agent for content structuring and formatting
- **`decoupled_editorial_agent.py`** - Editorial review agent for content enhancement and quality improvement
- **`content_cleaner_agent.py`** - Content processing agent for cleaning and standardization
- **`content_quality_judge.py`** - Quality assessment agent for content evaluation and scoring

## Agent Capabilities

### Research Agent (`research_agent.py`)
**Purpose**: Conduct comprehensive web research and source discovery

**Core Responsibilities**:
- Execute web searches using multiple search strategies
- Validate source credibility and authority
- Extract and synthesize information from diverse sources
- Identify key facts, statistics, and expert opinions
- Organize research findings in structured formats

**Key Features**:
- Multi-source research coordination
- Intelligent search strategy selection
- Source credibility assessment
- Content quality filtering
- Research data standardization

### Report Agent (`report_agent.py`)
**Purpose**: Generate well-structured, comprehensive research reports

**Core Responsibilities**:
- Synthesize research findings into coherent reports
- Structure content logically with appropriate headings
- Ensure consistency and clarity throughout reports
- Adapt report style to target audience requirements
- Integrate citations and references appropriately

**Key Features**:
- Multiple report format support (academic, business, technical)
- Audience-aware content adaptation
- Logical content organization
- Consistency checking and improvement
- Style and tone adjustment

### Editorial Agent (`decoupled_editorial_agent.py`)
**Purpose**: Enhance report quality through editorial review and gap analysis

**Core Responsibilities**:
- Analyze reports for quality and completeness
- Identify information gaps and areas for improvement
- Enhance content with specific facts and data from research
- Ensure proper integration of research findings
- Conduct targeted additional research when needed

**Key Features**:
- Quality-focused enhancement approach
- Research data integration
- Gap identification and filling
- Style and length optimization
- Fact verification and enhancement

### Content Cleaner Agent (`content_cleaner_agent.py`)
**Purpose**: Process and clean raw content for quality and consistency

**Core Responsibilities**:
- Clean and standardize content from various sources
- Remove formatting inconsistencies and artifacts
- Extract relevant information from raw text
- Standardize content structure and presentation
- Filter out low-quality or irrelevant content

**Key Features**:
- Multi-format content processing
- Intelligent content extraction
- Quality-based filtering
- Standardization and normalization
- Artifact removal and cleanup

### Content Quality Judge (`content_quality_judge.py`)
**Purpose**: Evaluate content quality and provide improvement recommendations

**Core Responsibilities**:
- Assess content quality across multiple dimensions
- Provide numerical quality scores and assessments
- Identify specific areas needing improvement
- Recommend enhancement strategies
- Validate content against quality standards

**Key Features**:
- Multi-dimensional quality assessment
- Numerical scoring systems
- Detailed improvement recommendations
- Quality threshold validation
- Benchmark comparisons

## Agent Workflow Integration

### Research Pipeline
```
User Query → Research Agent → Content Cleaner → Quality Judge → Report Agent → Editorial Agent → Final Output
```

### Agent Interactions
1. **Research Agent** discovers and processes source material
2. **Content Cleaner** standardizes and cleans the research data
3. **Quality Judge** evaluates content quality and identifies issues
4. **Report Agent** creates structured reports from research findings
5. **Editorial Agent** enhances reports with additional research and quality improvements

### Quality Loop
```
Content Creation → Quality Assessment → Enhancement → Re-assessment → Final Output
```

## Development Guidelines

### Agent Design Patterns
```python
# Standard agent implementation pattern
class BaseAgent:
    def __init__(self, config: dict):
        self.config = config
        self.tools = self._load_tools()
        self.quality_criteria = self._load_quality_criteria()

    async def execute(self, input_data: dict) -> dict:
        try:
            # Pre-processing
            processed_input = await self._preprocess(input_data)

            # Core processing
            results = await self._process(processed_input)

            # Post-processing and quality validation
            validated_results = await self._validate_quality(results)

            return validated_results

        except Exception as e:
            return await self._handle_error(e, input_data)
```

### Agent Configuration
```python
# Example: Agent configuration
AGENT_CONFIG = {
    "research_agent": {
        "max_sources": 20,
        "search_depth": "comprehensive",
        "quality_threshold": 0.7,
        "retry_attempts": 3
    },
    "report_agent": {
        "default_format": "standard_report",
        "audience_adaptation": True,
        "citation_style": "informal",
        "max_length": 50000
    },
    "editorial_agent": {
        "enhancement_focus": "data_integration",
        "gap_filling_enabled": True,
        "quality_improvement": True,
        "style_consistency": True
    }
}
```

### Quality Standards
```python
# Example: Quality criteria for agents
QUALITY_CRITERIA = {
    "completeness": {
        "threshold": 0.8,
        "factors": ["information_coverage", "source_diversity", "topic_depth"]
    },
    "accuracy": {
        "threshold": 0.9,
        "factors": ["factual_correctness", "source_credibility", "citation_quality"]
    },
    "clarity": {
        "threshold": 0.8,
        "factors": ["readability", "organization", "coherence"]
    }
}
```

## Testing & Debugging

### Agent Testing Strategies
1. **Unit Testing**: Test individual agent functions in isolation
2. **Integration Testing**: Test agent interactions and workflow integration
3. **Quality Testing**: Verify agent outputs meet quality standards
4. **Performance Testing**: Ensure agents perform within acceptable time limits

### Debugging Agent Behavior
1. **Verbose Logging**: Enable detailed logging for agent operations
2. **Step-by-Step Tracing**: Monitor agent decision-making processes
3. **Output Inspection**: Examine intermediate and final outputs
4. **Quality Metrics**: Track quality scores and improvement over time

### Common Agent Issues
- **Poor Quality Output**: Adjust prompts and quality criteria
- **Slow Performance**: Optimize algorithms and reduce unnecessary processing
- **Integration Failures**: Verify agent communication and data exchange
- **Tool Usage Issues**: Ensure agents have access to required tools

## Usage Examples

### Research Agent Usage
```python
from agents.research_agent import ResearchAgent

research_agent = ResearchAgent(config={
    "max_sources": 15,
    "search_strategy": "adaptive",
    "quality_threshold": 0.7
})

results = await research_agent.execute({
    "query": "artificial intelligence in healthcare",
    "depth": "comprehensive",
    "requirements": {
        "academic_sources": True,
        "recent_data": True,
        "expert_opinions": True
    }
})

print(f"Found {len(results['sources'])} sources")
print(f"Research quality score: {results['quality_score']}")
```

### Report Agent Usage
```python
from agents.report_agent import ReportAgent

report_agent = ReportAgent(config={
    "format": "academic_paper",
    "audience": "technical",
    "length": "detailed"
})

report = await report_agent.execute({
    "research_data": research_results,
    "topic": "AI in Healthcare",
    "requirements": {
        "include_abstract": True,
        "citation_style": "APA",
        "sections": ["introduction", "analysis", "conclusion"]
    }
})

print(f"Report generated: {len(report['content'])} characters")
print(f"Report quality: {report['quality_assessment']}")
```

### Editorial Agent Usage
```python
from agents.decoupled_editorial_agent import DecoupledEditorialAgent

editorial_agent = DecoupledEditorialAgent(config={
    "enhancement_focus": "data_integration",
    "gap_filling": True,
    "quality_improvement": True
})

enhanced_report = await editorial_agent.execute({
    "original_report": report,
    "research_data": research_results,
    "quality_requirements": {
        "min_quality_score": 0.8,
        "required_sections": ["executive_summary", "detailed_analysis"],
        "data_integration": True
    }
})

print(f"Enhancements made: {enhanced_report['enhancement_count']}")
print(f"New quality score: {enhanced_report['quality_score']}")
```

### Quality Judge Usage
```python
from agents.content_quality_judge import ContentQualityJudge

quality_judge = ContentQualityJudge()

assessment = await quality_judge.execute({
    "content": report_content,
    "content_type": "research_report",
    "requirements": {
        "min_quality": 0.7,
        "critical_factors": ["accuracy", "completeness", "clarity"]
    }
})

print(f"Overall quality: {assessment['overall_score']}")
print(f"Critical issues: {assessment['critical_issues']}")
print(f"Recommendations: {assessment['recommendations']}")
```

## Performance Considerations

### Agent Optimization
1. **Async Operations**: Use async/await patterns for concurrent processing
2. **Resource Management**: Monitor and manage memory and CPU usage
3. **Caching**: Cache frequently accessed data and computations
4. **Batch Processing**: Process multiple items together when possible

### Quality vs. Speed Trade-offs
- **High Quality Mode**: More thorough processing, longer execution time
- **Balanced Mode**: Good quality with reasonable performance
- **Fast Mode**: Quick processing with basic quality assurance

### Scaling Recommendations
- Implement agent pooling for concurrent execution
- Use distributed processing for large-scale operations
- Monitor agent performance and optimize bottlenecks
- Implement graceful degradation under high load

## Integration Patterns

### Agent Communication
```python
# Example: Agent communication pattern
class AgentOrchestrator:
    def __init__(self):
        self.agents = {
            "research": ResearchAgent(),
            "report": ReportAgent(),
            "editorial": EditorialAgent()
        }

    async def process_request(self, request: dict):
        # Research phase
        research_results = await self.agents["research"].execute(request)

        # Report generation phase
        report_request = {
            "research_data": research_results,
            "topic": request["topic"],
            "requirements": request.get("report_requirements", {})
        }
        report = await self.agents["report"].execute(report_request)

        # Editorial enhancement phase
        editorial_request = {
            "original_report": report,
            "research_data": research_results,
            "quality_requirements": request.get("quality_requirements", {})
        }
        final_report = await self.agents["editorial"].execute(editorial_request)

        return final_report
```

### Quality Feedback Loop
```python
# Example: Quality feedback integration
class QualityAwareAgent(BaseAgent):
    async def execute_with_quality_check(self, input_data: dict):
        result = await self.execute(input_data)

        # Quality assessment
        quality_assessment = await self.assess_quality(result)

        # Quality improvement loop
        while quality_assessment['overall_score'] < self.min_quality_threshold:
            result = await self.improve_quality(result, quality_assessment)
            quality_assessment = await self.assess_quality(result)

        return result
```

### Error Recovery in Agents
```python
# Example: Agent error recovery
class ResilientAgent(BaseAgent):
    async def execute_with_recovery(self, input_data: dict):
        for attempt in range(self.max_retry_attempts):
            try:
                return await self.execute(input_data)
            except Exception as e:
                if attempt == self.max_retry_attempts - 1:
                    # Final attempt with fallback strategy
                    return await self.fallback_execution(input_data)
                else:
                    # Wait and retry with modified approach
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    input_data = self.modify_request_for_retry(input_data, e)
```