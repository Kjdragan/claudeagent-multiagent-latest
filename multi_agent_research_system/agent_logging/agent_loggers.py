"""
Agent-Specific Logging Module for Multi-Agent Research System

This module provides specialized logging classes for each agent type,
enhancing visibility into agent-specific activities and decisions.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .structured_logger import StructuredLogger
from .base import AgentLogger


class ResearchAgentLogger(AgentLogger):
    """Specialized logger for Research Agent activities including search execution, source validation, and data collection."""

    def __init__(self, session_id: Optional[str] = None, base_log_dir: str = "logs"):
        super().__init__("research_agent", session_id, base_log_dir)
        self.search_history: List[Dict[str, Any]] = []
        self.source_quality_log: List[Dict[str, Any]] = []
        self.research_metrics: Dict[str, Any] = {
            "total_searches": 0,
            "successful_searches": 0,
            "total_sources_found": 0,
            "high_quality_sources": 0,
            "average_relevance_score": 0.0,
            "research_session_duration": 0.0,
            "topics_researched": []
        }

    def log_search_initiation(self,
                           query: str,
                           search_params: Dict[str, Any],
                           topic: str,
                           estimated_results: int) -> None:
        """Log the initiation of a research search."""
        search_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        search_entry = {
            "search_id": search_id,
            "timestamp": timestamp,
            "query": query,
            "topic": topic,
            "search_params": search_params,
            "estimated_results": estimated_results,
            "status": "initiated"
        }

        self.search_history.append(search_entry)

        # Update metrics
        self.research_metrics["total_searches"] += 1
        if topic not in self.research_metrics["topics_researched"]:
            self.research_metrics["topics_researched"].append(topic)

        self.structured_logger.info("Research search initiated",
                                   event_type="search_initiation",
                                   search_id=search_id,
                                   query=query,
                                   topic=topic,
                                   search_params=search_params,
                                   estimated_results=estimated_results,
                                   agent_name="research_agent")

    def log_search_results(self,
                         search_id: str,
                         results_count: int,
                         top_results: List[Dict[str, Any]],
                         relevance_scores: List[float],
                         search_duration: float) -> None:
        """Log the results of a research search."""
        timestamp = datetime.now().isoformat()

        # Find and update the search entry
        search_entry = next((s for s in self.search_history if s["search_id"] == search_id), None)
        if search_entry:
            search_entry.update({
                "timestamp_completed": timestamp,
                "results_count": results_count,
                "top_results": top_results[:5],  # Store top 5 for reference
                "average_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
                "search_duration": search_duration,
                "status": "completed"
            })

            # Update metrics
            self.research_metrics["successful_searches"] += 1
            self.research_metrics["total_sources_found"] += results_count

            if relevance_scores:
                avg_score = sum(relevance_scores) / len(relevance_scores)
                self.research_metrics["average_relevance_score"] = (
                    (self.research_metrics["average_relevance_score"] * (self.research_metrics["successful_searches"] - 1) + avg_score)
                    / self.research_metrics["successful_searches"]
                )

                # Count high-quality sources (score > 0.7)
                high_quality_count = sum(1 for score in relevance_scores if score > 0.7)
                self.research_metrics["high_quality_sources"] += high_quality_count

        self.structured_logger.info("Research search completed",
                                   event_type="search_completion",
                                   search_id=search_id,
                                   results_count=results_count,
                                   search_duration=search_duration,
                                   average_relevance=sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
                                   high_quality_sources=sum(1 for score in relevance_scores if score > 0.7) if relevance_scores else 0,
                                   agent_name="research_agent")

    def log_source_analysis(self,
                          source_url: str,
                          source_title: str,
                          credibility_score: float,
                          content_relevance: float,
                          content_type: str,
                          extraction_method: str) -> None:
        """Log detailed analysis of a research source."""
        timestamp = datetime.now().isoformat()

        source_entry = {
            "timestamp": timestamp,
            "source_url": source_url,
            "source_title": source_title,
            "credibility_score": credibility_score,
            "content_relevance": content_relevance,
            "content_type": content_type,
            "extraction_method": extraction_method,
            "overall_quality": credibility_score * content_relevance
        }

        self.source_quality_log.append(source_entry)

        self.structured_logger.info("Source analysis completed",
                                   event_type="source_analysis",
                                   source_url=source_url,
                                   source_title=source_title,
                                   credibility_score=credibility_score,
                                   content_relevance=content_relevance,
                                   overall_quality=source_entry["overall_quality"],
                                   content_type=content_type,
                                   agent_name="research_agent")

    def log_research_synthesis(self,
                            topic: str,
                            sources_used: int,
                            key_findings: List[str],
                            confidence_level: float,
                            synthesis_duration: float) -> None:
        """Log the synthesis of research findings."""
        timestamp = datetime.now().isoformat()

        self.structured_logger.info("Research synthesis completed",
                                   event_type="research_synthesis",
                                   topic=topic,
                                   sources_used=sources_used,
                                   key_findings_count=len(key_findings),
                                   key_findings=key_findings[:5],  # Store top 5 findings
                                   confidence_level=confidence_level,
                                   synthesis_duration=synthesis_duration,
                                   agent_name="research_agent")

    def log_data_extraction(self,
                          source_url: str,
                          extraction_method: str,
                          data_points_extracted: int,
                          extraction_success: bool,
                          extraction_time: float) -> None:
        """Log data extraction from sources."""
        self.structured_logger.info("Data extraction completed",
                                   event_type="data_extraction",
                                   source_url=source_url,
                                   extraction_method=extraction_method,
                                   data_points_extracted=data_points_extracted,
                                   extraction_success=extraction_success,
                                   extraction_time=extraction_time,
                                   agent_name="research_agent")

    def get_research_summary(self) -> Dict[str, Any]:
        """Get a summary of research activities."""
        return {
            "session_id": self.session_id,
            "research_metrics": self.research_metrics,
            "search_history_count": len(self.search_history),
            "source_analysis_count": len(self.source_quality_log),
            "last_search_time": self.search_history[-1]["timestamp"] if self.search_history else None,
            "average_source_quality": (
                sum(s["overall_quality"] for s in self.source_quality_log) / len(self.source_quality_log)
                if self.source_quality_log else 0.0
            )
        }


class ReportAgentLogger(AgentLogger):
    """Specialized logger for Report Agent activities including content generation, structuring, and synthesis."""

    def __init__(self, session_id: Optional[str] = None, base_log_dir: str = "logs"):
        super().__init__("report_agent", session_id, base_log_dir)
        self.content_generation_log: List[Dict[str, Any]] = []
        self.section_progress: Dict[str, Any] = {}
        self.report_metrics: Dict[str, Any] = {
            "total_sections_generated": 0,
            "total_words_generated": 0,
            "average_section_generation_time": 0.0,
            "research_sources_incorporated": 0,
            "sections_completed": [],
            "content_quality_indicators": {
                "coherence_score": 0.0,
                "depth_score": 0.0,
                "citation_coverage": 0.0
            }
        }

    def log_section_generation_start(self,
                                   section_name: str,
                                   section_type: str,
                                   research_sources_count: int,
                                   target_word_count: int) -> None:
        """Log the start of section generation."""
        section_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        self.section_progress[section_name] = {
            "section_id": section_id,
            "start_time": timestamp,
            "section_type": section_type,
            "research_sources_count": research_sources_count,
            "target_word_count": target_word_count,
            "status": "initiated"
        }

        self.structured_logger.info("Report section generation started",
                                   event_type="section_generation_start",
                                   section_id=section_id,
                                   section_name=section_name,
                                   section_type=section_type,
                                   research_sources_count=research_sources_count,
                                   target_word_count=target_word_count,
                                   agent_name="report_agent")

    def log_section_generation_complete(self,
                                      section_name: str,
                                      actual_word_count: int,
                                      generation_time: float,
                                      sources_cited: int,
                                      coherence_score: float,
                                      quality_metrics: Dict[str, float]) -> None:
        """Log the completion of section generation."""
        timestamp = datetime.now().isoformat()

        if section_name in self.section_progress:
            self.section_progress[section_name].update({
                "completion_time": timestamp,
                "actual_word_count": actual_word_count,
                "generation_time": generation_time,
                "sources_cited": sources_cited,
                "coherence_score": coherence_score,
                "quality_metrics": quality_metrics,
                "status": "completed"
            })

            # Update metrics
            self.report_metrics["total_sections_generated"] += 1
            self.report_metrics["total_words_generated"] += actual_word_count
            self.report_metrics["research_sources_incorporated"] += sources_cited
            self.report_metrics["sections_completed"].append(section_name)

            # Update average generation time
            if self.report_metrics["total_sections_generated"] > 0:
                total_time = self.report_metrics["average_section_generation_time"] * (self.report_metrics["total_sections_generated"] - 1)
                self.report_metrics["average_section_generation_time"] = (total_time + generation_time) / self.report_metrics["total_sections_generated"]

            # Update quality indicators
            for indicator, score in quality_metrics.items():
                if indicator in self.report_metrics["content_quality_indicators"]:
                    current_score = self.report_metrics["content_quality_indicators"][indicator]
                    sections_count = self.report_metrics["total_sections_generated"]
                    self.report_metrics["content_quality_indicators"][indicator] = (
                        (current_score * (sections_count - 1) + score) / sections_count
                    )

        generation_entry = {
            "timestamp": timestamp,
            "section_name": section_name,
            "actual_word_count": actual_word_count,
            "generation_time": generation_time,
            "sources_cited": sources_cited,
            "coherence_score": coherence_score,
            "quality_metrics": quality_metrics
        }

        self.content_generation_log.append(generation_entry)

        self.structured_logger.info("Report section generation completed",
                                   event_type="section_generation_complete",
                                   section_name=section_name,
                                   actual_word_count=actual_word_count,
                                   generation_time=generation_time,
                                   sources_cited=sources_cited,
                                   coherence_score=coherence_score,
                                   quality_metrics=quality_metrics,
                                   agent_name="report_agent")

    def log_content_synthesis(self,
                            topic: str,
                            research_findings_count: int,
                            synthesis_approach: str,
                            synthesis_time: float,
                            synthesis_quality: float) -> None:
        """Log the synthesis of research findings into report content."""
        self.structured_logger.info("Content synthesis completed",
                                   event_type="content_synthesis",
                                   topic=topic,
                                   research_findings_count=research_findings_count,
                                   synthesis_approach=synthesis_approach,
                                   synthesis_time=synthesis_time,
                                   synthesis_quality=synthesis_quality,
                                   agent_name="report_agent")

    def log_citation_processing(self,
                              citations_found: int,
                              citations_verified: int,
                              citation_format: str,
                              processing_time: float) -> None:
        """Log citation processing and verification."""
        self.structured_logger.info("Citation processing completed",
                                   event_type="citation_processing",
                                   citations_found=citations_found,
                                   citations_verified=citations_verified,
                                   citation_format=citation_format,
                                   verification_rate=citations_verified / citations_found if citations_found > 0 else 0.0,
                                   processing_time=processing_time,
                                   agent_name="report_agent")

    def log_report_structure_finalization(self,
                                        total_sections: int,
                                        total_words: int,
                                        structure_type: str,
                                        formatting_time: float) -> None:
        """Log the finalization of report structure."""
        self.structured_logger.info("Report structure finalized",
                                   event_type="report_structure_finalization",
                                   total_sections=total_sections,
                                   total_words=total_words,
                                   structure_type=structure_type,
                                   formatting_time=formatting_time,
                                   agent_name="report_agent")

    def get_report_summary(self) -> Dict[str, Any]:
        """Get a summary of report generation activities."""
        return {
            "session_id": self.session_id,
            "report_metrics": self.report_metrics,
            "sections_progress": self.section_progress,
            "content_generation_count": len(self.content_generation_log),
            "average_quality_score": (
                sum(entry["coherence_score"] for entry in self.content_generation_log) / len(self.content_generation_log)
                if self.content_generation_log else 0.0
            ),
            "estimated_reading_time": self.report_metrics["total_words_generated"] / 200  # Average reading speed
        }


class EditorAgentLogger(AgentLogger):
    """Specialized logger for Editor Agent activities including review, quality assessment, and feedback generation."""

    def __init__(self, session_id: Optional[str] = None, base_log_dir: str = "logs"):
        super().__init__("editor_agent", session_id, base_log_dir)
        self.review_history: List[Dict[str, Any]] = []
        self.quality_assessments: Dict[str, Any] = {}
        self.editor_metrics: Dict[str, Any] = {
            "total_reviews_completed": 0,
            "issues_identified": 0,
            "suggestions_provided": 0,
            "fact_checks_performed": 0,
            "average_review_time": 0.0,
            "review_categories": {
                "content": 0,
                "structure": 0,
                "style": 0,
                "sources": 0,
                "analysis": 0
            },
            "quality_scores_trend": []
        }

    def log_review_initiation(self,
                            document_title: str,
                            document_type: str,
                            word_count: int,
                            review_focus_areas: List[str]) -> None:
        """Log the initiation of an editorial review."""
        review_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        review_entry = {
            "review_id": review_id,
            "timestamp": timestamp,
            "document_title": document_title,
            "document_type": document_type,
            "word_count": word_count,
            "review_focus_areas": review_focus_areas,
            "status": "initiated"
        }

        self.review_history.append(review_entry)

        self.structured_logger.info("Editorial review initiated",
                                   event_type="review_initiation",
                                   review_id=review_id,
                                   document_title=document_title,
                                   document_type=document_type,
                                   word_count=word_count,
                                   review_focus_areas=review_focus_areas,
                                   agent_name="editor_agent")

    def log_quality_assessment(self,
                             review_id: str,
                             assessment_category: str,
                             score: float,
                             max_score: float,
                             issues_found: List[str],
                             strengths_identified: List[str]) -> None:
        """Log detailed quality assessment."""
        timestamp = datetime.now().isoformat()

        assessment_entry = {
            "timestamp": timestamp,
            "review_id": review_id,
            "assessment_category": assessment_category,
            "score": score,
            "max_score": max_score,
            "normalized_score": score / max_score if max_score > 0 else 0.0,
            "issues_found": issues_found,
            "strengths_identified": strengths_identified,
            "total_issues": len(issues_found),
            "total_strengths": len(strengths_identified)
        }

        if review_id not in self.quality_assessments:
            self.quality_assessments[review_id] = {}

        self.quality_assessments[review_id][assessment_category] = assessment_entry

        # Update metrics
        self.editor_metrics["issues_identified"] += len(issues_found)
        if assessment_category in self.editor_metrics["review_categories"]:
            self.editor_metrics["review_categories"][assessment_category] += 1

        self.structured_logger.info("Quality assessment completed",
                                   event_type="quality_assessment",
                                   review_id=review_id,
                                   assessment_category=assessment_category,
                                   score=score,
                                   max_score=max_score,
                                   normalized_score=assessment_entry["normalized_score"],
                                   issues_count=len(issues_found),
                                   strengths_count=len(strengths_identified),
                                   agent_name="editor_agent")

    def log_fact_checking(self,
                         review_id: str,
                         claims_verified: int,
                         claims_confirmed: int,
                         claims_corrected: int,
                         additional_sources_found: int,
                         fact_checking_time: float) -> None:
        """Log fact-checking activities."""
        timestamp = datetime.now().isoformat()

        fact_check_entry = {
            "timestamp": timestamp,
            "review_id": review_id,
            "claims_verified": claims_verified,
            "claims_confirmed": claims_confirmed,
            "claims_corrected": claims_corrected,
            "additional_sources_found": additional_sources_found,
            "accuracy_rate": claims_confirmed / claims_verified if claims_verified > 0 else 0.0,
            "fact_checking_time": fact_checking_time
        }

        if review_id not in self.quality_assessments:
            self.quality_assessments[review_id] = {}

        self.quality_assessments[review_id]["fact_checking"] = fact_check_entry

        # Update metrics
        self.editor_metrics["fact_checks_performed"] += claims_verified

        self.structured_logger.info("Fact checking completed",
                                   event_type="fact_checking",
                                   review_id=review_id,
                                   claims_verified=claims_verified,
                                   claims_confirmed=claims_confirmed,
                                   claims_corrected=claims_corrected,
                                   accuracy_rate=fact_check_entry["accuracy_rate"],
                                   fact_checking_time=fact_checking_time,
                                   agent_name="editor_agent")

    def log_feedback_generation(self,
                              review_id: str,
                              feedback_type: str,
                              suggestions_count: int,
                              examples_provided: int,
                              feedback_tone: str,
                              generation_time: float) -> None:
        """Log the generation of editorial feedback."""
        timestamp = datetime.now().isoformat()

        feedback_entry = {
            "timestamp": timestamp,
            "review_id": review_id,
            "feedback_type": feedback_type,
            "suggestions_count": suggestions_count,
            "examples_provided": examples_provided,
            "feedback_tone": feedback_tone,
            "generation_time": generation_time
        }

        if review_id not in self.quality_assessments:
            self.quality_assessments[review_id] = {}

        self.quality_assessments[review_id]["feedback_generation"] = feedback_entry

        # Update metrics
        self.editor_metrics["suggestions_provided"] += suggestions_count

        self.structured_logger.info("Feedback generation completed",
                                   event_type="feedback_generation",
                                   review_id=review_id,
                                   feedback_type=feedback_type,
                                   suggestions_count=suggestions_count,
                                   examples_provided=examples_provided,
                                   feedback_tone=feedback_tone,
                                   generation_time=generation_time,
                                   agent_name="editor_agent")

    def log_review_completion(self,
                            review_id: str,
                            overall_quality_score: float,
                            revision_recommendations: List[str],
                            review_duration: float) -> None:
        """Log the completion of the editorial review."""
        timestamp = datetime.now().isoformat()

        # Find and update the review entry
        review_entry = next((r for r in self.review_history if r["review_id"] == review_id), None)
        if review_entry:
            review_entry.update({
                "completion_time": timestamp,
                "overall_quality_score": overall_quality_score,
                "revision_recommendations": revision_recommendations,
                "review_duration": review_duration,
                "status": "completed"
            })

        # Update metrics
        self.editor_metrics["total_reviews_completed"] += 1
        self.editor_metrics["quality_scores_trend"].append({
            "timestamp": timestamp,
            "quality_score": overall_quality_score
        })

        if self.editor_metrics["total_reviews_completed"] > 0:
            total_time = self.editor_metrics["average_review_time"] * (self.editor_metrics["total_reviews_completed"] - 1)
            self.editor_metrics["average_review_time"] = (total_time + review_duration) / self.editor_metrics["total_reviews_completed"]

        self.structured_logger.info("Editorial review completed",
                                   event_type="review_completion",
                                   review_id=review_id,
                                   overall_quality_score=overall_quality_score,
                                   revision_recommendations_count=len(revision_recommendations),
                                   review_duration=review_duration,
                                   agent_name="editor_agent")

    def get_editor_summary(self) -> Dict[str, Any]:
        """Get a summary of editorial activities."""
        recent_scores = self.editor_metrics["quality_scores_trend"][-10:]  # Last 10 reviews
        average_recent_score = sum(s["quality_score"] for s in recent_scores) / len(recent_scores) if recent_scores else 0.0

        return {
            "session_id": self.session_id,
            "editor_metrics": self.editor_metrics,
            "reviews_completed": len([r for r in self.review_history if r["status"] == "completed"]),
            "reviews_in_progress": len([r for r in self.review_history if r["status"] == "initiated"]),
            "average_quality_score": average_recent_score,
            "most_active_review_category": max(self.editor_metrics["review_categories"].items(), key=lambda x: x[1])[0] if self.editor_metrics["review_categories"] else None
        }


class UICoordinatorLogger(AgentLogger):
    """Specialized logger for UI Coordinator activities including workflow management and user interactions."""

    def __init__(self, session_id: Optional[str] = None, base_log_dir: str = "logs"):
        super().__init__("ui_coordinator", session_id, base_log_dir)
        self.workflow_events: List[Dict[str, Any]] = []
        self.user_interactions: List[Dict[str, Any]] = []
        self.agent_handoffs: List[Dict[str, Any]] = []
        self.coordinator_metrics: Dict[str, Any] = {
            "total_workflows_managed": 0,
            "user_requests_handled": 0,
            "agent_handoffs_coordinated": 0,
            "average_workflow_duration": 0.0,
            "user_satisfaction_indicators": {
                "revision_requests": 0,
                "clarification_requests": 0,
                "positive_feedback_count": 0
            },
            "workflow_stages_completed": {
                "research_initiation": 0,
                "research_completion": 0,
                "report_generation": 0,
                "editorial_review": 0,
                "final_delivery": 0
            }
        }

    def log_workflow_initiation(self,
                              workflow_id: str,
                              user_request: str,
                              workflow_type: str,
                              estimated_stages: List[str],
                              priority_level: str) -> None:
        """Log the initiation of a research workflow."""
        timestamp = datetime.now().isoformat()

        workflow_entry = {
            "workflow_id": workflow_id,
            "timestamp": timestamp,
            "user_request": user_request,
            "workflow_type": workflow_type,
            "estimated_stages": estimated_stages,
            "priority_level": priority_level,
            "status": "initiated"
        }

        self.workflow_events.append(workflow_entry)

        # Update metrics
        self.coordinator_metrics["total_workflows_managed"] += 1

        self.structured_logger.info("Workflow initiated",
                                   event_type="workflow_initiation",
                                   workflow_id=workflow_id,
                                   user_request=user_request,
                                   workflow_type=workflow_type,
                                   estimated_stages=estimated_stages,
                                   priority_level=priority_level,
                                   agent_name="ui_coordinator")

    def log_stage_completion(self,
                           workflow_id: str,
                           stage_name: str,
                           stage_duration: float,
                           success_status: bool,
                           output_summary: str) -> None:
        """Log the completion of a workflow stage."""
        timestamp = datetime.now().isoformat()

        stage_entry = {
            "timestamp": timestamp,
            "workflow_id": workflow_id,
            "stage_name": stage_name,
            "stage_duration": stage_duration,
            "success_status": success_status,
            "output_summary": output_summary
        }

        self.workflow_events.append(stage_entry)

        # Update metrics
        if stage_name in self.coordinator_metrics["workflow_stages_completed"]:
            self.coordinator_metrics["workflow_stages_completed"][stage_name] += 1

        self.structured_logger.info("Workflow stage completed",
                                   event_type="stage_completion",
                                   workflow_id=workflow_id,
                                   stage_name=stage_name,
                                   stage_duration=stage_duration,
                                   success_status=success_status,
                                   output_summary=output_summary,
                                   agent_name="ui_coordinator")

    def log_agent_handoff(self,
                        workflow_id: str,
                        from_agent: str,
                        to_agent: str,
                        handoff_reason: str,
                        context_transmitted: Dict[str, Any]) -> None:
        """Log the coordination of agent handoffs."""
        timestamp = datetime.now().isoformat()

        handoff_entry = {
            "timestamp": timestamp,
            "workflow_id": workflow_id,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "handoff_reason": handoff_reason,
            "context_transmitted": context_transmitted,
            "handoff_id": str(uuid.uuid4())
        }

        self.agent_handoffs.append(handoff_entry)

        # Update metrics
        self.coordinator_metrics["agent_handoffs_coordinated"] += 1

        self.structured_logger.info("Agent handoff coordinated",
                                   event_type="agent_handoff",
                                   workflow_id=workflow_id,
                                   from_agent=from_agent,
                                   to_agent=to_agent,
                                   handoff_reason=handoff_reason,
                                   context_items_count=len(context_transmitted),
                                   agent_name="ui_coordinator")

    def log_user_interaction(self,
                           interaction_type: str,
                           user_message: str,
                           system_response: str,
                           satisfaction_indicator: Optional[str],
                           response_time: float) -> None:
        """Log user interactions and satisfaction indicators."""
        timestamp = datetime.now().isoformat()

        interaction_entry = {
            "timestamp": timestamp,
            "interaction_type": interaction_type,
            "user_message": user_message,
            "system_response": system_response,
            "satisfaction_indicator": satisfaction_indicator,
            "response_time": response_time
        }

        self.user_interactions.append(interaction_entry)

        # Update metrics
        self.coordinator_metrics["user_requests_handled"] += 1

        if satisfaction_indicator:
            if satisfaction_indicator in ["revision_request", "clarification_needed"]:
                satisfaction_category = "revision_requests" if satisfaction_indicator == "revision_request" else "clarification_requests"
                self.coordinator_metrics["user_satisfaction_indicators"][satisfaction_category] += 1
            elif satisfaction_indicator == "positive_feedback":
                self.coordinator_metrics["user_satisfaction_indicators"]["positive_feedback_count"] += 1

        self.structured_logger.info("User interaction completed",
                                   event_type="user_interaction",
                                   interaction_type=interaction_type,
                                   user_message_length=len(user_message),
                                   system_response_length=len(system_response),
                                   satisfaction_indicator=satisfaction_indicator,
                                   response_time=response_time,
                                   agent_name="ui_coordinator")

    def log_workflow_completion(self,
                              workflow_id: str,
                              final_deliverables: List[str],
                              total_duration: float,
                              user_satisfaction: Optional[float],
                              issues_encountered: List[str]) -> None:
        """Log the completion of a workflow."""
        timestamp = datetime.now().isoformat()

        # Find and update the workflow entry
        workflow_entry = next((w for w in self.workflow_events if w["workflow_id"] == workflow_id), None)
        if workflow_entry:
            workflow_entry.update({
                "completion_time": timestamp,
                "final_deliverables": final_deliverables,
                "total_duration": total_duration,
                "user_satisfaction": user_satisfaction,
                "issues_encountered": issues_encountered,
                "status": "completed"
            })

        # Update metrics
        if self.coordinator_metrics["total_workflows_managed"] > 0:
            total_time = self.coordinator_metrics["average_workflow_duration"] * (self.coordinator_metrics["total_workflows_managed"] - 1)
            self.coordinator_metrics["average_workflow_duration"] = (total_time + total_duration) / self.coordinator_metrics["total_workflows_managed"]

        self.structured_logger.info("Workflow completed",
                                   event_type="workflow_completion",
                                   workflow_id=workflow_id,
                                   final_deliverables_count=len(final_deliverables),
                                   total_duration=total_duration,
                                   user_satisfaction=user_satisfaction,
                                   issues_encountered_count=len(issues_encountered),
                                   agent_name="ui_coordinator")

    def log_progress_monitoring(
        self,
        current_stage: str,
        progress_percentage: float,
        milestones_reached: list,
        session_id: str,
        **metadata
    ) -> None:
        """Log workflow progress monitoring data."""
        self.structured_logger.info(
            f"Progress monitoring: {current_stage} - {progress_percentage}% complete",
            event_type="progress_monitoring",
            agent_name="ui_coordinator",
            current_stage=current_stage,
            progress_percentage=progress_percentage,
            milestones_reached=milestones_reached,
            milestones_count=len(milestones_reached),
            session_id=session_id,
            **metadata
        )

    def log_coordination_decision(self,
                                decision_context: str,
                                decision_made: str,
                                decision_rationale: str,
                                impact_assessment: str) -> None:
        """Log coordination decisions made by the UI coordinator."""
        self.structured_logger.info("Coordination decision made",
                                   event_type="coordination_decision",
                                   decision_context=decision_context,
                                   decision_made=decision_made,
                                   decision_rationale=decision_rationale,
                                   impact_assessment=impact_assessment,
                                   agent_name="ui_coordinator")

    def get_coordinator_summary(self) -> Dict[str, Any]:
        """Get a summary of coordination activities."""
        # Workflow events and workflow entries are different - workflow entries have status
        workflow_entries = [w for w in self.workflow_events if "workflow_id" in w and "status" in w]
        active_workflows = len([w for w in workflow_entries if w["status"] == "initiated"])
        completed_workflows = len([w for w in workflow_entries if w["status"] == "completed"])

        satisfaction_rate = 0.0
        total_interactions = len(self.user_interactions)
        if total_interactions > 0:
            positive_interactions = self.coordinator_metrics["user_satisfaction_indicators"]["positive_feedback_count"]
            satisfaction_rate = (positive_interactions / total_interactions) * 100

        return {
            "session_id": self.session_id,
            "coordinator_metrics": self.coordinator_metrics,
            "workflow_status": {
                "active_workflows": active_workflows,
                "completed_workflows": completed_workflows,
                "total_workflows": active_workflows + completed_workflows
            },
            "user_satisfaction_rate": satisfaction_rate,
            "total_user_interactions": total_interactions,
            "agent_handoffs_coordinated": len(self.agent_handoffs),
            "most_completed_stage": max(self.coordinator_metrics["workflow_stages_completed"].items(), key=lambda x: x[1])[0] if self.coordinator_metrics["workflow_stages_completed"] else None
        }


# Factory function to create appropriate agent logger
def create_agent_logger(agent_type: str, session_id: Optional[str] = None, base_log_dir: str = "logs") -> AgentLogger:
    """Create the appropriate agent logger based on agent type."""

    agent_logger_classes = {
        "research_agent": ResearchAgentLogger,
        "report_agent": ReportAgentLogger,
        "editor_agent": EditorAgentLogger,
        "ui_coordinator": UICoordinatorLogger
    }

    logger_class = agent_logger_classes.get(agent_type, AgentLogger)
    return logger_class(session_id, base_log_dir)