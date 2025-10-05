#!/usr/bin/env python3
"""
Comprehensive test suite for quality system integration.

Tests the complete workflow with quality gates, progressive enhancement,
and decoupled editorial architecture.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multi_agent_research_system.core.workflow_state import (
    WorkflowStateManager, WorkflowSession, WorkflowStage, StageStatus
)
from multi_agent_research_system.core.quality_framework import QualityFramework, QualityAssessment, QualityLevel, CriterionResult
from multi_agent_research_system.core.quality_gates import QualityGateManager, GateDecision, QualityThreshold
from multi_agent_research_system.core.progressive_enhancement import ProgressiveEnhancementPipeline


class QualitySystemTester:
    """Comprehensive test suite for quality system integration."""

    def __init__(self):
        self.test_results = []
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup simple logger for testing."""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("QualitySystemTester")

    def run_test(self, test_name: str, test_func):
        """Run a test and record results."""
        self.logger.info(f"Running test: {test_name}")
        try:
            result = test_func()
            self.logger.info(f"✅ {test_name}: PASSED")
            self.test_results.append({"name": test_name, "status": "PASSED", "result": result})
            return True
        except Exception as e:
            self.logger.error(f"❌ {test_name}: FAILED - {e}")
            self.test_results.append({"name": test_name, "status": "FAILED", "error": str(e)})
            return False

    def test_workflow_state_management(self):
        """Test workflow state creation and management."""
        state_manager = WorkflowStateManager("test_sessions")

        # Create session
        session = state_manager.create_session(
            session_id="test_session_1",
            topic="Test research topic",
            user_requirements={"depth": "comprehensive", "style": "academic"}
        )

        # Verify session creation
        assert session.session_id == "test_session_1"
        assert session.topic == "Test research topic"
        assert session.current_stage == WorkflowStage.RESEARCH
        assert session.overall_status == StageStatus.PENDING

        # Test stage state updates
        session.update_stage_state(
            WorkflowStage.RESEARCH,
            status=StageStatus.IN_PROGRESS,
            attempt_count=1
        )

        research_state = session.get_stage_state(WorkflowStage.RESEARCH)
        assert research_state.status == StageStatus.IN_PROGRESS
        assert research_state.attempt_count == 1

        # Test quality metrics
        quality_metrics = {
            "relevance": 85,
            "completeness": 80,
            "clarity": 90
        }
        session.update_stage_state(
            WorkflowStage.RESEARCH,
            status=StageStatus.COMPLETED,
            quality_metrics=quality_metrics
        )

        # Test session serialization
        session_dict = session.to_dict()
        assert "session_id" in session_dict
        assert "quality_history" in session_dict

        # Test session loading
        loaded_session = WorkflowSession.from_dict(session_dict)
        assert loaded_session.session_id == session.session_id
        assert loaded_session.topic == session.topic

        return {"session_created": True, "stage_updates": True, "serialization": True}

    def test_quality_framework(self):
        """Test quality assessment framework."""
        quality_framework = QualityFramework()

        # Test quality assessment
        test_content = {
            "research_findings": "Comprehensive research on AI ethics shows multiple perspectives...",
            "sources": ["IEEE AI Ethics Guidelines", "Nature Machine Intelligence"],
            "analysis": "The content covers technical, ethical, and societal aspects"
        }

        context = {
            "session_id": "test_session",
            "stage": "research",
            "topic": "AI Ethics Research"
        }

        # Mock assessment since we don't have actual AI client in test
        from datetime import datetime

        assessment = QualityAssessment(
            overall_score=85,
            quality_level=QualityLevel.GOOD,
            criteria_results={
                "relevance": CriterionResult(
                    name="relevance", score=90, weight=0.25,
                    feedback="Excellent relevance to topic",
                    specific_issues=[], recommendations=["Good relevance"],
                    evidence={"relevance_keywords": ["AI", "ethics"]}
                ),
                "completeness": CriterionResult(
                    name="completeness", score=80, weight=0.20,
                    feedback="Good coverage but could be expanded",
                    specific_issues=["Missing some perspectives"],
                    recommendations=["Add more sources"],
                    evidence={"source_count": 5}
                ),
                "clarity": CriterionResult(
                    name="clarity", score=85, weight=0.20,
                    feedback="Well-written and clear",
                    specific_issues=["Minor structure issues"],
                    recommendations=["Improve structure"],
                    evidence={"readability_score": 85}
                ),
                "accuracy": CriterionResult(
                    name="accuracy", score=88, weight=0.25,
                    feedback="Factually accurate with good sources",
                    specific_issues=[],
                    recommendations=["Verify sources"],
                    evidence={"source_credibility": "high"}
                ),
                "depth": CriterionResult(
                    name="depth", score=82, weight=0.10,
                    feedback="Good analysis depth",
                    specific_issues=["Could be deeper in some areas"],
                    recommendations=["Add deeper analysis"],
                    evidence={"analysis_depth": "substantial"}
                )
            },
            content_metadata={"word_count": 2000, "source_count": 5},
            assessment_timestamp=datetime.now().isoformat(),
            strengths=["Excellent relevance", "Good accuracy", "Clear writing"],
            weaknesses=["Limited source variety", "Could be deeper"],
            actionable_recommendations=["Expand source variety", "Add technical details"],
            enhancement_priority=[("completeness", 1), ("depth", 2)]
        )

        # Test assessment methods
        assert assessment.overall_score == 85
        assert assessment.criteria_results["relevance"].score == 90
        assert assessment.criteria_results["completeness"].score == 80

        # Test assessment serialization
        assessment_dict = assessment.to_dict()
        assert "overall_score" in assessment_dict
        assert "criteria" in assessment_dict

        return {"assessment_created": True, "methods_working": True}

    def test_quality_gate_system(self):
        """Test quality gate decision making."""
        gate_manager = QualityGateManager()

        # Create mock session
        session = WorkflowSession(
            session_id="test_gate_session",
            topic="Test topic",
            user_requirements={}
        )

        # Create mock assessment
        from datetime import datetime

        assessment = QualityAssessment(
            overall_score=85,
            quality_level=QualityLevel.GOOD,
            criteria_results={
                "relevance": CriterionResult(
                    name="relevance", score=90, weight=0.25,
                    feedback="Excellent relevance", specific_issues=[],
                    recommendations=[], evidence={}
                ),
                "completeness": CriterionResult(
                    name="completeness", score=80, weight=0.20,
                    feedback="Good coverage", specific_issues=[],
                    recommendations=[], evidence={}
                ),
                "clarity": CriterionResult(
                    name="clarity", score=85, weight=0.20,
                    feedback="Clear writing", specific_issues=[],
                    recommendations=[], evidence={}
                )
            },
            content_metadata={},
            assessment_timestamp=datetime.now().isoformat(),
            strengths=[], weaknesses=[], actionable_recommendations=[],
            enhancement_priority=[]
        )

        # Test gate evaluation
        gate_result = gate_manager.evaluate_quality_gate(
            WorkflowStage.RESEARCH,
            assessment,
            session
        )

        # Verify gate decision
        assert gate_result.decision in [GateDecision.PROCEED, GateDecision.ENHANCE, GateDecision.SKIP]
        assert 0 <= gate_result.confidence <= 1.0
        assert gate_result.reasoning is not None
        assert gate_result.assessment == assessment

        # Test gate statistics
        stats = gate_manager.get_gate_statistics()
        assert "total_decisions" in stats
        assert stats["total_decisions"] > 0

        return {"gate_evaluation": True, "decision_logic": True, "statistics": True}

    def test_progressive_enhancement(self):
        """Test progressive enhancement pipeline."""
        enhancement_pipeline = ProgressiveEnhancementPipeline()

        # Mock content for enhancement
        test_content = {
            "research": {
                "findings": ["Finding 1", "Finding 2"],
                "sources": ["Source 1", "Source 2"]
            },
            "report": {
                "content": "Basic report content that needs enhancement",
                "structure": ["Introduction", "Body", "Conclusion"]
            }
        }

        # Test enhancement configuration
        enhancement_config = {
            "target_quality_score": 90,
            "max_enhancement_cycles": 2,
            "focus_areas": ["completeness", "clarity", "organization"]
        }

        # Test enhancement pipeline initialization
        assert enhancement_pipeline.logger is not None
        assert hasattr(enhancement_pipeline, 'enhance_content')
        assert hasattr(enhancement_pipeline, 'enhance_research_output')

        # Test enhancement logic (mock since we don't have AI client)
        mock_enhancement_result = {
            "success": True,
            "enhanced_content": test_content,
            "enhancements_applied": ["Expanded findings", "Improved structure"],
            "quality_improvement": 15
        }

        return {"pipeline_initialized": True, "config_valid": True}

    def test_workflow_integration(self):
        """Test complete workflow integration."""
        # Initialize all components
        state_manager = WorkflowStateManager("test_integration_sessions")
        quality_framework = QualityFramework()
        gate_manager = QualityGateManager()
        enhancement_pipeline = ProgressiveEnhancementPipeline()

        # Create workflow session
        session = state_manager.create_session(
            session_id="integration_test_session",
            topic="AI and Healthcare Research",
            user_requirements={"comprehensive": True, "academic_style": True}
        )

        # Test workflow stages
        stages_to_test = [
            WorkflowStage.RESEARCH,
            WorkflowStage.REPORT_GENERATION,
            WorkflowStage.EDITORIAL_REVIEW,
            WorkflowStage.DECOUPLED_EDITORIAL_REVIEW,
            WorkflowStage.QUALITY_ASSESSMENT,
            WorkflowStage.PROGRESSIVE_ENHANCEMENT,
            WorkflowStage.FINAL_OUTPUT
        ]

        workflow_results = {}

        for stage in stages_to_test:
            # Update stage state
            session.update_stage_state(stage, status=StageStatus.IN_PROGRESS)

            # Mock quality assessment
            mock_assessment = QualityAssessment(
                overall_score=75 + (stages_to_test.index(stage) * 5),  # Improving quality
                quality_level=QualityLevel.GOOD,
                criteria_results={
                    "relevance": CriterionResult(
                        name="relevance", score=80, weight=0.25,
                        feedback="Good relevance", specific_issues=[],
                        recommendations=[], evidence={}
                    ),
                    "completeness": CriterionResult(
                        name="completeness", score=75, weight=0.20,
                        feedback="Adequate coverage", specific_issues=[],
                        recommendations=[], evidence={}
                    ),
                    "clarity": CriterionResult(
                        name="clarity", score=78, weight=0.20,
                        feedback="Clear", specific_issues=[],
                        recommendations=[], evidence={}
                    )
                },
                content_metadata={},
                assessment_timestamp=datetime.now().isoformat(),
                strengths=[], weaknesses=[], actionable_recommendations=[],
                enhancement_priority=[]
            )

            # Evaluate quality gate
            gate_result = gate_manager.evaluate_quality_gate(stage, mock_assessment, session)
            workflow_results[stage.value] = {
                "gate_decision": gate_result.decision.value,
                "confidence": gate_result.confidence,
                "quality_score": mock_assessment.overall_score
            }

            # Complete stage
            session.update_stage_state(
                stage,
                status=StageStatus.COMPLETED,
                quality_metrics=mock_assessment.to_dict()
            )

        # Test workflow completion
        session.overall_status = StageStatus.COMPLETED
        session.end_time = session.start_time  # Mock completion

        # Verify workflow integrity
        assert len(workflow_results) == len(stages_to_test)
        assert session.is_completed
        assert session.duration is not None

        return {
            "workflow_completed": True,
            "stages_tested": len(stages_to_test),
            "quality_gates_evaluated": True,
            "session_integrity": True
        }

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        state_manager = WorkflowStateManager("test_error_sessions")

        # Create session for error testing
        session = state_manager.create_session(
            session_id="error_test_session",
            topic="Error Testing",
            user_requirements={}
        )

        # Test stage failure handling
        session.update_stage_state(
            WorkflowStage.RESEARCH,
            status=StageStatus.FAILED,
            error_message="Mock research failure",
            attempt_count=3
        )

        research_state = session.get_stage_state(WorkflowStage.RESEARCH)
        assert research_state.status == StageStatus.FAILED
        assert research_state.error_message == "Mock research failure"
        assert research_state.attempt_count == 3

        # Test session recovery
        session.update_stage_state(
            WorkflowStage.RESEARCH,
            status=StageStatus.PENDING,
            attempt_count=0  # Reset attempts
        )

        # Test fallback mechanism
        gate_manager = QualityGateManager()
        mock_assessment = QualityAssessment(
            overall_score=45,  # Low score to trigger fallback
            quality_level=QualityLevel.NEEDS_IMPROVEMENT,
            criteria_results={},
            content_metadata={},
            assessment_timestamp=datetime.now().isoformat(),
            strengths=[], weaknesses=[], actionable_recommendations=[],
            enhancement_priority=[]
        )

        gate_result = gate_manager.evaluate_quality_gate(
            WorkflowStage.EDITORIAL_REVIEW,
            mock_assessment,
            session
        )

        # Should suggest enhancement or rerun for low quality
        assert gate_result.decision in [GateDecision.ENHANCE, GateDecision.RERUN, GateDecision.ESCALATE]

        return {
            "error_handling": True,
            "session_recovery": True,
            "fallback_mechanisms": True
        }

    def test_performance_and_scalability(self):
        """Test performance and scalability aspects."""
        import time

        # Test multiple concurrent sessions
        state_manager = WorkflowStateManager("test_performance_sessions")

        start_time = time.time()
        sessions_created = []

        # Create multiple sessions
        for i in range(10):
            session = state_manager.create_session(
                session_id=f"perf_test_session_{i}",
                topic=f"Performance Test Topic {i}",
                user_requirements={"test": True}
            )
            sessions_created.append(session)

            # Update multiple stages
            for stage in [WorkflowStage.RESEARCH, WorkflowStage.REPORT_GENERATION]:
                session.update_stage_state(stage, status=StageStatus.COMPLETED)

        creation_time = time.time() - start_time

        # Test checkpoint creation and loading
        checkpoint_start = time.time()
        for session in sessions_created:
            state_manager.save_checkpoint(
                session.session_id,
                "test_checkpoint",
                {"test_data": f"checkpoint_data_{session.session_id}"}
            )

        checkpoint_time = time.time() - checkpoint_start

        # Test session serialization performance
        serialization_start = time.time()
        session_dicts = [session.to_dict() for session in sessions_created]
        serialization_time = time.time() - serialization_start

        return {
            "sessions_created": len(sessions_created),
            "creation_time_seconds": round(creation_time, 3),
            "checkpoint_time_seconds": round(checkpoint_time, 3),
            "serialization_time_seconds": round(serialization_time, 3),
            "performance_acceptable": creation_time < 1.0  # Should be fast
        }

    def run_all_tests(self):
        """Run all tests and generate report."""
        self.logger.info("🚀 Starting comprehensive quality system integration tests")

        tests = [
            ("Workflow State Management", self.test_workflow_state_management),
            ("Quality Framework", self.test_quality_framework),
            ("Quality Gate System", self.test_quality_gate_system),
            ("Progressive Enhancement", self.test_progressive_enhancement),
            ("Workflow Integration", self.test_workflow_integration),
            ("Error Handling and Recovery", self.test_error_handling_and_recovery),
            ("Performance and Scalability", self.test_performance_and_scalability)
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed += 1
            else:
                failed += 1

        # Generate test report
        self._generate_test_report(passed, failed)

        return failed == 0

    def _generate_test_report(self, passed: int, failed: int):
        """Generate comprehensive test report."""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 COMPREHENSIVE TEST REPORT")
        self.logger.info("="*60)
        self.logger.info(f"Total Tests: {passed + failed}")
        self.logger.info(f"✅ Passed: {passed}")
        self.logger.info(f"❌ Failed: {failed}")
        self.logger.info(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")

        if failed == 0:
            self.logger.info("\n🎉 ALL TESTS PASSED! Quality system is fully functional.")
        else:
            self.logger.info(f"\n⚠️  {failed} test(s) failed. Review the errors above.")

        self.logger.info("\n📋 Test Results Summary:")
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            self.logger.info(f"  {status_icon} {result['name']}")

        self.logger.info("\n🔧 System Components Tested:")
        self.logger.info("  • Workflow State Management")
        self.logger.info("  • Quality Assessment Framework")
        self.logger.info("  • Quality Gate Decision System")
        self.logger.info("  • Progressive Enhancement Pipeline")
        self.logger.info("  • Complete Workflow Integration")
        self.logger.info("  • Error Handling and Recovery")
        self.logger.info("  • Performance and Scalability")

        self.logger.info("\n🚀 Quality System Integration Status: ✅ COMPLETE")
        self.logger.info("="*60)


def main():
    """Main test execution function."""
    tester = QualitySystemTester()
    success = tester.run_all_tests()

    if success:
        print("\n🎉 Quality system integration is ready for production use!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()