"""
Agent-Specific Logger for Multi-Agent Research System

Provides specialized logging capabilities for different agent types,
including research, report, editor, and UI coordinator agents.
"""

from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path
import json

from .structured_logger import StructuredLogger, get_logger


class AgentLogger:
    """Specialized logger for agent-specific activities and events."""

    def __init__(self, agent_name: str, log_dir: Optional[Path] = None):
        """Initialize agent logger."""
        self.agent_name = agent_name
        self.logger = get_logger(f"agent.{agent_name}", log_dir=log_dir)

    def log_agent_initialization(
        self,
        model: str,
        tools: list,
        capabilities: Dict[str, Any],
        **metadata
    ) -> None:
        """Log agent initialization event."""
        self.logger.info(
            f"Agent initialized: {self.agent_name}",
            event_type="agent_initialization",
            agent_name=self.agent_name,
            model=model,
            tools_count=len(tools),
            tools_available=tools,
            capabilities=capabilities,
            **metadata
        )

    def log_health_check(
        self,
        status: str,
        response_time: float,
        capabilities_tested: list,
        issues_found: list,
        **metadata
    ) -> None:
        """Log agent health check event."""
        self.logger.info(
            f"Health check completed: {self.agent_name}",
            event_type="health_check",
            agent_name=self.agent_name,
            status=status,
            response_time_seconds=response_time,
            capabilities_tested=capabilities_tested,
            issues_found=issues_found,
            issues_count=len(issues_found),
            **metadata
        )

    def log_query_start(
        self,
        query_type: str,
        session_id: str,
        prompt_length: int,
        **metadata
    ) -> None:
        """Log query start event."""
        self.logger.info(
            f"Query started: {self.agent_name} - {query_type}",
            event_type="query_start",
            agent_name=self.agent_name,
            query_type=query_type,
            session_id=session_id,
            prompt_length_characters=prompt_length,
            **metadata
        )

    def log_query_complete(
        self,
        query_type: str,
        session_id: str,
        response_time: float,
        response_length: int,
        tools_executed: list,
        **metadata
    ) -> None:
        """Log query completion event."""
        self.logger.info(
            f"Query completed: {self.agent_name} - {query_type}",
            event_type="query_complete",
            agent_name=self.agent_name,
            query_type=query_type,
            session_id=session_id,
            response_time_seconds=response_time,
            response_length_characters=response_length,
            tools_executed=tools_executed,
            tools_count=len(tools_executed),
            **metadata
        )

    def log_tool_use_attempt(
        self,
        tool_name: str,
        tool_use_id: str,
        session_id: str,
        input_data: Dict[str, Any],
        **metadata
    ) -> None:
        """Log tool use attempt."""
        self.logger.info(
            f"Tool use attempt: {self.agent_name} - {tool_name}",
            event_type="tool_use_attempt",
            agent_name=self.agent_name,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            session_id=session_id,
            input_data=input_data,
            **metadata
        )

    def log_tool_use_result(
        self,
        tool_name: str,
        tool_use_id: str,
        session_id: str,
        success: bool,
        execution_time: float,
        result_size: int,
        **metadata
    ) -> None:
        """Log tool use result."""
        level = "info" if success else "error"
        getattr(self.logger, level)(
            f"Tool use result: {self.agent_name} - {tool_name} ({'success' if success else 'failed'})",
            event_type="tool_use_result",
            agent_name=self.agent_name,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            session_id=session_id,
            success=success,
            execution_time_seconds=execution_time,
            result_size_bytes=result_size,
            **metadata
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        session_id: str,
        context: Dict[str, Any],
        **metadata
    ) -> None:
        """Log agent error."""
        self.logger.error(
            f"Agent error: {self.agent_name} - {error_type}",
            event_type="agent_error",
            agent_name=self.agent_name,
            error_type=error_type,
            error_message=error_message,
            session_id=session_id,
            context=context,
            **metadata
        )

    def log_session_handoff(
        self,
        from_agent: str,
        to_agent: str,
        session_id: str,
        handoff_reason: str,
        **metadata
    ) -> None:
        """Log agent-to-agent session handoff."""
        self.logger.info(
            f"Session handoff: {from_agent} -> {to_agent}",
            event_type="session_handoff",
            from_agent=from_agent,
            to_agent=to_agent,
            session_id=session_id,
            handoff_reason=handoff_reason,
            **metadata
        )

    def log_performance_metrics(
        self,
        session_id: str,
        metrics: Dict[str, Any],
        **metadata
    ) -> None:
        """Log agent performance metrics."""
        self.logger.info(
            f"Performance metrics: {self.agent_name}",
            event_type="performance_metrics",
            agent_name=self.agent_name,
            session_id=session_id,
            **metrics,
            **metadata
        )


class ResearchAgentLogger(AgentLogger):
    """Specialized logger for research agent activities."""

    def __init__(self, log_dir: Optional[Path] = None):
        super().__init__("research_agent", log_dir)

    def log_search_start(
        self,
        query: str,
        search_type: str,
        num_results: int,
        session_id: str,
        **metadata
    ) -> None:
        """Log search start event."""
        self.logger.info(
            f"Search started: {query}",
            event_type="search_start",
            agent_name=self.agent_name,
            query=query,
            search_type=search_type,
        num_results_expected=num_results,
            session_id=session_id,
            **metadata
        )

    def log_search_complete(
        self,
        query: str,
        search_type: str,
        results_found: int,
        urls_processed: int,
        execution_time: float,
        session_id: str,
        **metadata
    ) -> None:
        """Log search completion event."""
        self.logger.info(
            f"Search completed: {query} ({results_found} results)",
            event_type="search_complete",
            agent_name=self.agent_name,
            query=query,
            search_type=search_type,
            results_found=results_found,
            urls_processed=urls_processed,
            execution_time_seconds=execution_time,
            session_id=session_id,
            **metadata
        )

    def log_content_extraction(
        self,
        url: str,
        content_length: int,
        extraction_success: bool,
        session_id: str,
        **metadata
    ) -> None:
        """Log content extraction event."""
        level = "info" if extraction_success else "warning"
        getattr(self.logger, level)(
            f"Content extraction: {url} ({'success' if extraction_success else 'failed'})",
            event_type="content_extraction",
            agent_name=self.agent_name,
            url=url,
            content_length_bytes=content_length,
            extraction_success=extraction_success,
            session_id=session_id,
            **metadata
        )

    def log_source_analysis(
        self,
        sources_analyzed: int,
        relevance_scores: Dict[str, float],
        session_id: str,
        **metadata
    ) -> None:
        """Log source analysis event."""
        self.logger.info(
            f"Source analysis completed: {sources_analyzed} sources",
            event_type="source_analysis",
            agent_name=self.agent_name,
            sources_analyzed=sources_analyzed,
            average_relevance_score=sum(relevance_scores.values()) / len(relevance_scores) if relevance_scores else 0,
            relevance_scores=relevance_scores,
            session_id=session_id,
            **metadata
        )

    def log_research_synthesis(
        self,
        findings_count: int,
        synthesis_time: float,
        session_id: str,
        **metadata
    ) -> None:
        """Log research synthesis event."""
        self.logger.info(
            f"Research synthesis completed: {findings_count} findings",
            event_type="research_synthesis",
            agent_name=self.agent_name,
            findings_count=findings_count,
            synthesis_time_seconds=synthesis_time,
            session_id=session_id,
            **metadata
        )


class ReportAgentLogger(AgentLogger):
    """Specialized logger for report agent activities."""

    def __init__(self, log_dir: Optional[Path] = None):
        super().__init__("report_agent", log_dir)

    def log_report_generation_start(
        self,
        topic: str,
        report_format: str,
        session_id: str,
        **metadata
    ) -> None:
        """Log report generation start."""
        self.logger.info(
            f"Report generation started: {topic}",
            event_type="report_generation_start",
            agent_name=self.agent_name,
            topic=topic,
            report_format=report_format,
            session_id=session_id,
            **metadata
        )

    def log_content_analysis(
        self,
        sources_reviewed: int,
        key_findings: int,
        analysis_time: float,
        session_id: str,
        **metadata
    ) -> None:
        """Log content analysis event."""
        self.logger.info(
            f"Content analysis completed: {sources_reviewed} sources, {key_findings} findings",
            event_type="content_analysis",
            agent_name=self.agent_name,
            sources_reviewed=sources_reviewed,
            key_findings=key_findings,
            analysis_time_seconds=analysis_time,
            session_id=session_id,
            **metadata
        )

    def log_report_structure(
        self,
        sections: list,
        word_count: int,
        session_id: str,
        **metadata
    ) -> None:
        """Log report structure creation."""
        self.logger.info(
            f"Report structure created: {len(sections)} sections, {word_count} words",
            event_type="report_structure",
            agent_name=self.agent_name,
            sections_count=len(sections),
            sections=sections,
            word_count=word_count,
            session_id=session_id,
            **metadata
        )

    def log_citation_integration(
        self,
        citations_added: int,
        sources_cited: int,
        session_id: str,
        **metadata
    ) -> None:
        """Log citation integration event."""
        self.logger.info(
            f"Citation integration completed: {citations_added} citations from {sources_cited} sources",
            event_type="citation_integration",
            agent_name=self.agent_name,
            citations_added=citations_added,
            sources_cited=sources_cited,
            session_id=session_id,
            **metadata
        )


class EditorAgentLogger(AgentLogger):
    """Specialized logger for editor agent activities."""

    def __init__(self, log_dir: Optional[Path] = None):
        super().__init__("editor_agent", log_dir)

    def log_editorial_review_start(
        self,
        report_type: str,
        review_focus: str,
        session_id: str,
        **metadata
    ) -> None:
        """Log editorial review start."""
        self.logger.info(
            f"Editorial review started: {report_type} - {review_focus}",
            event_type="editorial_review_start",
            agent_name=self.agent_name,
            report_type=report_type,
            review_focus=review_focus,
            session_id=session_id,
            **metadata
        )

    def log_quality_assessment(
        self,
        clarity_score: float,
        accuracy_score: float,
        completeness_score: float,
        issues_found: list,
        session_id: str,
        **metadata
    ) -> None:
        """Log quality assessment event."""
        self.logger.info(
            f"Quality assessment completed: clarity={clarity_score}, accuracy={accuracy_score}, completeness={completeness_score}",
            event_type="quality_assessment",
            agent_name=self.agent_name,
            clarity_score=clarity_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            overall_score=(clarity_score + accuracy_score + completeness_score) / 3,
            issues_found=issues_found,
            issues_count=len(issues_found),
            session_id=session_id,
            **metadata
        )

    def log_content_revision(
        self,
        revisions_made: int,
        sections_edited: int,
        improvement_areas: list,
        session_id: str,
        **metadata
    ) -> None:
        """Log content revision event."""
        self.logger.info(
            f"Content revision completed: {revisions_made} revisions across {sections_edited} sections",
            event_type="content_revision",
            agent_name=self.agent_name,
            revisions_made=revisions_made,
            sections_edited=sections_edited,
            improvement_areas=improvement_areas,
            session_id=session_id,
            **metadata
        )

    def log_final_approval(
        self,
        approval_status: str,
        final_checks: list,
        session_id: str,
        **metadata
    ) -> None:
        """Log final approval event."""
        self.logger.info(
            f"Final approval: {approval_status}",
            event_type="final_approval",
            agent_name=self.agent_name,
            approval_status=approval_status,
            final_checks=final_checks,
            session_id=session_id,
            **metadata
        )


class UICoordinatorLogger(AgentLogger):
    """Specialized logger for UI coordinator activities."""

    def __init__(self, log_dir: Optional[Path] = None):
        super().__init__("ui_coordinator", log_dir)

    def log_workflow_orchestration(
        self,
        workflow_type: str,
        participants: list,
        estimated_duration: float,
        session_id: str,
        **metadata
    ) -> None:
        """Log workflow orchestration event."""
        self.logger.info(
            f"Workflow orchestration started: {workflow_type} with {len(participants)} participants",
            event_type="workflow_orchestration",
            agent_name=self.agent_name,
            workflow_type=workflow_type,
            participants=participants,
            participants_count=len(participants),
            estimated_duration_seconds=estimated_duration,
            session_id=session_id,
            **metadata
        )

    def log_agent_coordination(
        self,
        coordination_event: str,
        involved_agents: list,
        decision_made: str,
        session_id: str,
        **metadata
    ) -> None:
        """Log agent coordination event."""
        self.logger.info(
            f"Agent coordination: {coordination_event} - {decision_made}",
            event_type="agent_coordination",
            agent_name=self.agent_name,
            coordination_event=coordination_event,
            involved_agents=involved_agents,
            decision_made=decision_made,
            session_id=session_id,
            **metadata
        )

    def log_progress_monitoring(
        self,
        current_stage: str,
        progress_percentage: float,
        milestones_reached: list,
        session_id: str,
        **metadata
    ) -> None:
        """Log progress monitoring event."""
        self.logger.info(
            f"Progress monitoring: {current_stage} - {progress_percentage}% complete",
            event_type="progress_monitoring",
            agent_name=self.agent_name,
            current_stage=current_stage,
            progress_percentage=progress_percentage,
            milestones_reached=milestones_reached,
            milestones_count=len(milestones_reached),
            session_id=session_id,
            **metadata
        )

    def log_user_interaction(
        self,
        interaction_type: str,
        user_input: str,
        system_response: str,
        session_id: str,
        **metadata
    ) -> None:
        """Log user interaction event."""
        self.logger.info(
            f"User interaction: {interaction_type}",
            event_type="user_interaction",
            agent_name=self.agent_name,
            interaction_type=interaction_type,
            user_input=user_input,
            system_response=system_response,
            session_id=session_id,
            **metadata
        )