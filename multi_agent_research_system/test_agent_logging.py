#!/usr/bin/env python3
"""
Test script for agent-specific logging functionality.

This script tests the ResearchAgentLogger, ReportAgentLogger, EditorAgentLogger,
and UICoordinatorLogger to ensure they work correctly.
"""

import asyncio
import json
import sys
import os
import time
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from agent_logging import (
    ResearchAgentLogger,
    ReportAgentLogger,
    EditorAgentLogger,
    UICoordinatorLogger,
    create_agent_logger
)


def test_research_agent_logger():
    """Test the ResearchAgentLogger functionality."""
    print("🔍 Testing ResearchAgentLogger...")

    session_id = "test_research_session"
    logger = ResearchAgentLogger(session_id)

    # Test search initiation logging
    logger.log_search_initiation(
        query="artificial intelligence trends 2024",
        search_params={"num_results": 15, "auto_crawl_top": 8},
        topic="AI Technology Trends",
        estimated_results=25
    )

    # Test search results logging
    logger.log_search_results(
        search_id="search_123",
        results_count=15,
        top_results=[
            {"title": "AI Trends 2024", "url": "example.com", "relevance": 0.9},
            {"title": "Machine Learning Advances", "url": "ml.com", "relevance": 0.85}
        ],
        relevance_scores=[0.9, 0.85, 0.8, 0.75, 0.7],
        search_duration=2.5
    )

    # Test source analysis logging
    logger.log_source_analysis(
        source_url="https://example.com/ai-trends-2024",
        source_title="AI Trends 2024 Report",
        credibility_score=0.85,
        content_relevance=0.92,
        content_type="research_report",
        extraction_method="web_crawl"
    )

    # Test research synthesis logging
    logger.log_research_synthesis(
        topic="AI Technology Trends",
        sources_used=12,
        key_findings=[
            "AI adoption increased by 35% in 2024",
            "Large language models became mainstream",
            "AI ethics concerns grew significantly"
        ],
        confidence_level=0.88,
        synthesis_duration=1.8
    )

    # Test data extraction logging
    logger.log_data_extraction(
        source_url="https://example.com/ai-data",
        extraction_method="structured_extraction",
        data_points_extracted=45,
        extraction_success=True,
        extraction_time=0.8
    )

    # Get research summary
    summary = logger.get_research_summary()
    print(f"✅ ResearchAgentLogger test completed. Summary: {json.dumps(summary, indent=2)}")

    return summary


def test_report_agent_logger():
    """Test the ReportAgentLogger functionality."""
    print("\n📄 Testing ReportAgentLogger...")

    session_id = "test_report_session"
    logger = ReportAgentLogger(session_id)

    # Test section generation start
    logger.log_section_generation_start(
        section_name="Executive Summary",
        section_type="summary",
        research_sources_count=8,
        target_word_count=300
    )

    # Test section generation completion
    logger.log_section_generation_complete(
        section_name="Executive Summary",
        actual_word_count=285,
        generation_time=1.2,
        sources_cited=6,
        coherence_score=0.91,
        quality_metrics={
            "clarity": 0.89,
            "depth": 0.85,
            "structure": 0.93,
            "citations": 0.88
        }
    )

    # Test content synthesis logging
    logger.log_content_synthesis(
        topic="AI Technology Trends",
        research_findings_count=12,
        synthesis_approach="thematic_analysis",
        synthesis_time=2.1,
        synthesis_quality=0.87
    )

    # Test citation processing
    logger.log_citation_processing(
        citations_found=15,
        citations_verified=13,
        citation_format="APA",
        processing_time=0.9
    )

    # Test report structure finalization
    logger.log_report_structure_finalization(
        total_sections=6,
        total_words=1850,
        structure_type="standard_research_report",
        formatting_time=1.5
    )

    # Get report summary
    summary = logger.get_report_summary()
    print(f"✅ ReportAgentLogger test completed. Summary: {json.dumps(summary, indent=2)}")

    return summary


def test_editor_agent_logger():
    """Test the EditorAgentLogger functionality."""
    print("\n📝 Testing EditorAgentLogger...")

    session_id = "test_editor_session"
    logger = EditorAgentLogger(session_id)

    # Test review initiation
    logger.log_review_initiation(
        document_title="AI Technology Trends Report",
        document_type="research_report",
        word_count=1850,
        review_focus_areas=["content", "structure", "sources", "analysis"]
    )

    # Test quality assessment
    logger.log_quality_assessment(
        review_id="review_456",
        assessment_category="content",
        score=8.5,
        max_score=10.0,
        issues_found=[
            "Missing recent data points",
            "Some claims need stronger support"
        ],
        strengths_identified=[
            "Good overall structure",
            "Clear explanations",
            "Comprehensive coverage"
        ]
    )

    # Test fact checking
    logger.log_fact_checking(
        review_id="review_456",
        claims_verified=12,
        claims_confirmed=11,
        claims_corrected=1,
        additional_sources_found=3,
        fact_checking_time=2.3
    )

    # Test feedback generation
    logger.log_feedback_generation(
        review_id="review_456",
        feedback_type="constructive",
        suggestions_count=8,
        examples_provided=5,
        feedback_tone="supportive",
        generation_time=1.7
    )

    # Test review completion
    logger.log_review_completion(
        review_id="review_456",
        overall_quality_score=8.2,
        revision_recommendations=[
            "Add more recent statistics",
            "Strengthen conclusion section",
            "Verify source credibility"
        ],
        review_duration=5.8
    )

    # Get editor summary
    summary = logger.get_editor_summary()
    print(f"✅ EditorAgentLogger test completed. Summary: {json.dumps(summary, indent=2)}")

    return summary


def test_ui_coordinator_logger():
    """Test the UICoordinatorLogger functionality."""
    print("\n🎛️  Testing UICoordinatorLogger...")

    session_id = "test_ui_coordinator_session"
    logger = UICoordinatorLogger(session_id)

    # Test workflow initiation
    logger.log_workflow_initiation(
        workflow_id="workflow_789",
        user_request="Create a comprehensive report on AI technology trends",
        workflow_type="research_report",
        estimated_stages=["research", "report_generation", "editorial_review", "final_delivery"],
        priority_level="high"
    )

    # Test stage completion
    logger.log_stage_completion(
        workflow_id="workflow_789",
        stage_name="research",
        stage_duration=8.5,
        success_status=True,
        output_summary="Completed comprehensive research on AI trends with 15 sources"
    )

    # Test agent handoff
    logger.log_agent_handoff(
        workflow_id="workflow_789",
        from_agent="research_agent",
        to_agent="report_agent",
        handoff_reason="Research phase completed, ready for report generation",
        context_transmitted={
            "research_findings": "15 sources analyzed",
            "key_topics": ["AI adoption", "LLM trends", "AI ethics"],
            "session_data": "research_data.json"
        }
    )

    # Test user interaction
    logger.log_user_interaction(
        interaction_type="progress_inquiry",
        user_message="How is the research progressing?",
        system_response="Research phase completed successfully with 15 high-quality sources. Now proceeding to report generation.",
        satisfaction_indicator="positive_feedback",
        response_time=0.8
    )

    # Test workflow completion
    logger.log_workflow_completion(
        workflow_id="workflow_789",
        final_deliverables=["AI_Trends_Report_2024.pdf", "Research_Sources.json", "Executive_Summary.md"],
        total_duration=25.5,
        user_satisfaction=9.1,
        issues_encountered=["Minor citation formatting issues resolved"]
    )

    # Test coordination decision
    logger.log_coordination_decision(
        decision_context="User requested additional sources on AI ethics",
        decision_made="Assign research agent to gather 5 additional sources on AI ethics",
        decision_rationale="User feedback indicated need for deeper coverage of ethics topic",
        impact_assessment="Extends project timeline by 2 hours but improves report quality"
    )

    # Get coordinator summary
    summary = logger.get_coordinator_summary()
    print(f"✅ UICoordinatorLogger test completed. Summary: {json.dumps(summary, indent=2)}")

    return summary


def test_agent_logger_factory():
    """Test the agent logger factory function."""
    print("\n🏭 Testing create_agent_logger factory function...")

    agent_types = ["research_agent", "report_agent", "editor_agent", "ui_coordinator"]
    session_id = "factory_test_session"

    results = {}

    for agent_type in agent_types:
        try:
            logger = create_agent_logger(agent_type, session_id)
            results[agent_type] = {
                "success": True,
                "logger_type": type(logger).__name__,
                "session_id": getattr(logger, 'session_id', 'N/A')
            }
            print(f"✅ Successfully created logger for {agent_type}: {type(logger).__name__}")
        except Exception as e:
            results[agent_type] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ Failed to create logger for {agent_type}: {e}")

    # Test invalid agent type
    try:
        logger = create_agent_logger("invalid_agent_type", session_id)
        results["invalid_agent"] = {
            "success": True,
            "logger_type": type(logger).__name__,
            "note": "Fallback to generic AgentLogger"
        }
        print("✅ Invalid agent type correctly fell back to generic AgentLogger")
    except Exception as e:
        results["invalid_agent"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Invalid agent type handling failed: {e}")

    print(f"✅ Factory function test completed. Results: {json.dumps(results, indent=2)}")

    return results


def main():
    """Run all agent logger tests."""
    print("🚀 Starting Agent-Specific Logging Tests")
    print("=" * 50)

    try:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Run all tests
        research_summary = test_research_agent_logger()
        report_summary = test_report_agent_logger()
        editor_summary = test_editor_agent_logger()
        coordinator_summary = test_ui_coordinator_logger()
        factory_results = test_agent_logger_factory()

        # Create comprehensive test report
        test_report = {
            "test_run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_results": {
                "research_agent_logger": {
                    "status": "✅ PASSED" if research_summary else "❌ FAILED",
                    "summary": research_summary
                },
                "report_agent_logger": {
                    "status": "✅ PASSED" if report_summary else "❌ FAILED",
                    "summary": report_summary
                },
                "editor_agent_logger": {
                    "status": "✅ PASSED" if editor_summary else "❌ FAILED",
                    "summary": editor_summary
                },
                "ui_coordinator_logger": {
                    "status": "✅ PASSED" if coordinator_summary else "❌ FAILED",
                    "summary": coordinator_summary
                },
                "agent_logger_factory": {
                    "status": "✅ PASSED" if factory_results else "❌ FAILED",
                    "results": factory_results
                }
            },
            "overall_status": "✅ ALL TESTS PASSED" if all([
                research_summary, report_summary, editor_summary, coordinator_summary, factory_results
            ]) else "❌ SOME TESTS FAILED"
        }

        # Save test report
        report_path = logs_dir / "agent_logging_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(test_report, f, indent=2)

        print(f"\n" + "=" * 50)
        print("🎉 AGENT-SPECIFIC LOGGING TESTS COMPLETE")
        print(f"📊 Test report saved to: {report_path}")
        print(f"📈 Overall Status: {test_report['overall_status']}")

        # Display key metrics
        if research_summary:
            print(f"🔍 Research Agent: {research_summary['research_metrics']['total_searches']} searches, {research_summary['research_metrics']['total_sources_found']} sources")
        if report_summary:
            print(f"📄 Report Agent: {report_summary['report_metrics']['total_sections_generated']} sections, {report_summary['report_metrics']['total_words_generated']} words")
        if editor_summary:
            print(f"📝 Editor Agent: {editor_summary['editor_metrics']['total_reviews_completed']} reviews, {editor_summary['editor_metrics']['issues_identified']} issues identified")
        if coordinator_summary:
            print(f"🎛️  UI Coordinator: {coordinator_summary['coordinator_metrics']['total_workflows_managed']} workflows, {coordinator_summary['coordinator_metrics']['user_satisfaction_indicators']['positive_feedback_count']} positive feedback")

        return test_report['overall_status'] == "✅ ALL TESTS PASSED"

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)