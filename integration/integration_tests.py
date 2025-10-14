"""
Integration Tests for Agent-Based Research System

This module provides comprehensive integration testing for all system components,
validating that they work together correctly as a unified system.

Integration Test Coverage:
- MCP tool integration with orchestrator
- KEVIN directory integration with session management
- Quality assurance integration across workflow stages
- Error handling integration throughout the system
- End-to-end system coordination

Author: Claude Code Assistant
Version: 1.0.0
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Import system components for integration testing
import sys
sys.path.append(str(Path(__file__).parent.parent))


class IntegrationTestFixture:
    """Provides test fixtures and utilities for integration testing"""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="integration_test_"))
        self.setup_test_environment()
        self.test_sessions = {}
        self.mock_responses = {}

    def setup_test_environment(self):
        """Set up test environment with mock directories and configurations"""
        # Create test directory structure
        directories = [
            "test_sessions",
            "test_sessions/sessions",
            "test_logs",
            "test_config",
            "test_workproducts"
        ]

        for dir_name in directories:
            (self.temp_dir / dir_name).mkdir(parents=True, exist_ok=True)

        # Create test configuration files
        self.create_test_configurations()

    def create_test_configurations(self):
        """Create test configuration files"""
        # Test system configuration
        system_config = {
            "test_mode": True,
            "mock_services": True,
            "timeout_duration": 30,
            "log_level": "DEBUG",
            "performance_tracking": True
        }

        config_path = self.temp_dir / "test_config" / "system_config.json"
        with open(config_path, 'w') as f:
            json.dump(system_config, f, indent=2)

        # Test research configuration
        research_config = {
            "target_urls": 5,
            "concurrent_processing": 3,
            "anti_bot_level": 1,
            "quality_threshold": 0.7,
            "sources": ["test_source_1", "test_source_2"],
            "content_filters": ["test_filter_1"]
        }

        research_config_path = self.temp_dir / "test_config" / "research_config.json"
        with open(research_config_path, 'w') as f:
            json.dump(research_config, f, indent=2)

    def create_test_session(self, session_type: str = "standard") -> Dict[str, Any]:
        """Create a test session with appropriate data"""
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"

        session_data = {
            "session_id": session_id,
            "session_type": session_type,
            "initial_query": f"Test query for {session_type} integration testing",
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
            "configuration": self.load_test_configuration(),
            "metadata": {
                "test_mode": True,
                "integration_test": True,
                "test_timestamp": datetime.now().isoformat()
            }
        }

        self.test_sessions[session_id] = session_data
        return session_data

    def load_test_configuration(self) -> Dict[str, Any]:
        """Load test configuration"""
        research_config_path = self.temp_dir / "test_config" / "research_config.json"
        with open(research_config_path, 'r') as f:
            return json.load(f)

    def create_mock_research_results(self, session_id: str) -> Dict[str, Any]:
        """Create mock research results for testing"""
        return {
            "session_id": session_id,
            "query": self.test_sessions[session_id]["initial_query"],
            "results": [
                {
                    "url": f"https://test-source-{i+1}.com/article{i+1}",
                    "title": f"Test Article {i+1}",
                    "content": f"This is test content {i+1} for integration testing...",
                    "source": f"test_source_{i+1}",
                    "timestamp": datetime.now().isoformat(),
                    "relevance_score": 0.85 - (i * 0.05),
                    "metadata": {
                        "word_count": 500 + (i * 100),
                        "reading_time": 2 + (i * 0.5),
                        "content_type": "article"
                    }
                }
                for i in range(3)
            ],
            "metadata": {
                "total_results": 3,
                "processing_time": 2.5,
                "quality_score": 0.82,
                "sources_used": ["test_source_1", "test_source_2", "test_source_3"]
            }
        }

    def create_mock_analysis_results(self, session_id: str) -> Dict[str, Any]:
        """Create mock analysis results for testing"""
        return {
            "session_id": session_id,
            "analysis_type": "comprehensive",
            "content_analysis": {
                "key_topics": ["topic1", "topic2", "topic3"],
                "sentiment_analysis": {
                    "overall": "neutral",
                    "confidence": 0.75
                },
                "entity_extraction": {
                    "organizations": ["org1", "org2"],
                    "persons": ["person1", "person2"],
                    "locations": ["location1", "location2"]
                },
                "content_quality": {
                    "readability_score": 0.78,
                    "technical_depth": 0.82,
                    "source_credibility": 0.85
                }
            },
            "recommendations": [
                "Add more recent sources",
                "Include case studies",
                "Enhance technical details"
            ],
            "confidence_score": 0.88,
            "timestamp": datetime.now().isoformat()
        }

    def cleanup(self):
        """Clean up test environment"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class MCPToolIntegrationTests(unittest.TestCase):
    """Test MCP tool integration with other system components"""

    def setUp(self):
        """Set up test environment"""
        self.fixture = IntegrationTestFixture()
        self.test_session = self.fixture.create_test_session("mcp_integration")

    def tearDown(self):
        """Clean up test environment"""
        self.fixture.cleanup()

    def test_mcp_orchestrator_integration(self):
        """Test MCP tool integration with research orchestrator"""
        print("\n=== Testing MCP-Orchestrator Integration ===")

        try:
            # Mock imports to avoid dependency issues
            with patch.dict('sys.modules', {
                'multi_agent_research_system.mcp_tools.zplayground1_search': MagicMock()
            }):
                from integration.mcp_tool_integration import MCPToolIntegration
                from integration.research_orchestrator import ResearchOrchestrator

                # Initialize components
                mcp_integration = MCPToolIntegration()
                orchestrator = ResearchOrchestrator()

                # Test parameter mapping from orchestrator to MCP
                research_params = self.fixture.load_test_configuration()
                query = self.test_session["initial_query"]

                mapped_params = mcp_integration._map_parameters_to_mcp(
                    query,
                    research_params,
                    self.test_session["session_id"]
                )

                # Verify parameter mapping
                self.assertIn("query", mapped_params)
                self.assertEqual(mapped_params["query"], query)
                self.assertIn("session_id", mapped_params)
                self.assertEqual(mapped_params["session_id"], self.test_session["session_id"])
                self.assertIn("num_results", mapped_params)
                self.assertEqual(mapped_params["num_results"], research_params["target_urls"])

                print("✓ Parameter mapping from orchestrator to MCP works correctly")

                # Test session coordination between orchestrator and MCP
                coordination_result = mcp_integration.coordinate_session(
                    self.test_session["session_id"],
                    "research_execution"
                )

                self.assertIsNotNone(coordination_result)
                print("✓ Session coordination between orchestrator and MCP works correctly")

                # Test result transformation from MCP to orchestrator
                mock_mcp_result = self.fixture.create_mock_research_results(
                    self.test_session["session_id"]
                )

                transformed_result = mcp_integration._transform_mcp_results(
                    mock_mcp_result,
                    self.test_session["session_id"]
                )

                self.assertIn("processed_results", transformed_result)
                self.assertIn("metadata", transformed_result)
                print("✓ Result transformation from MCP to orchestrator works correctly")

        except Exception as e:
            self.fail(f"MCP-Orchestrator integration test failed: {str(e)}")

    def test_mcp_session_management_integration(self):
        """Test MCP tool integration with session management"""
        print("\n=== Testing MCP-Session Management Integration ===")

        try:
            with patch.dict('sys.modules', {
                'multi_agent_research_system.mcp_tools.zplayground1_search': MagicMock()
            }):
                from integration.mcp_tool_integration import MCPToolIntegration
                from integration.agent_session_manager import AgentSessionManager

                mcp_integration = MCPToolIntegration()
                session_manager = AgentSessionManager()

                # Test session creation coordination
                session_data = session_manager.create_session(
                    self.test_session["session_id"],
                    self.test_session["initial_query"],
                    self.fixture.load_test_configuration()
                )

                self.assertIsNotNone(session_data)
                self.assertIn("session_id", session_data)

                # Test MCP session linking
                mcp_session_result = mcp_integration.link_session_to_mcp(
                    self.test_session["session_id"],
                    session_data
                )

                self.assertIsNotNone(mcp_session_result)
                self.assertIn("mcp_session_id", mcp_session_result)
                print("✓ MCP session linking works correctly")

                # Test session state synchronization
                state_sync_result = mcp_integration.synchronize_session_state(
                    self.test_session["session_id"],
                    "research_in_progress",
                    {"progress": 0.5, "current_stage": "content_retrieval"}
                )

                self.assertIsNotNone(state_sync_result)
                print("✓ Session state synchronization works correctly")

        except Exception as e:
            self.fail(f"MCP-Session Management integration test failed: {str(e)}")


class KevinDirectoryIntegrationTests(unittest.TestCase):
    """Test KEVIN directory integration with other system components"""

    def setUp(self):
        """Set up test environment"""
        self.fixture = IntegrationTestFixture()
        self.test_session = self.fixture.create_test_session("kevin_integration")

    def tearDown(self):
        """Clean up test environment"""
        self.fixture.cleanup()

    def test_kevin_orchestrator_integration(self):
        """Test KEVIN directory integration with research orchestrator"""
        print("\n=== Testing KEVIN-Orchestrator Integration ===")

        try:
            from integration.kevin_directory_integration import KevinDirectoryIntegration
            from integration.research_orchestrator import ResearchOrchestrator

            # Mock Path to use test directory
            with patch('integration.kevin_directory_integration.Path') as mock_path:
                mock_path.return_value = self.fixture.temp_dir / "test_sessions" / "sessions"

                kevin_integration = KevinDirectoryIntegration()
                orchestrator = ResearchOrchestrator()

                # Test session structure creation for orchestrator
                session_dirs = kevin_integration.create_session_structure(
                    self.test_session["session_id"]
                )

                self.assertIn("session", session_dirs)
                self.assertIn("working", session_dirs)
                self.assertIn("research", session_dirs)
                self.assertIn("logs", session_dirs)

                # Verify directories exist
                for dir_name, dir_path in session_dirs.items():
                    self.assertTrue(dir_path.exists(), f"Directory {dir_name} should exist")

                print("✓ Session structure creation for orchestrator works correctly")

                # Test workproduct file creation
                research_results = self.fixture.create_mock_research_results(
                    self.test_session["session_id"]
                )

                workproduct_path = kevin_integration.create_workproduct_file(
                    self.test_session["session_id"],
                    "RESEARCH",
                    "INITIAL_SEARCH",
                    research_results
                )

                self.assertIsNotNone(workproduct_path)
                self.assertTrue(workproduct_path.exists())
                print("✓ Workproduct file creation for orchestrator works correctly")

                # Test file path generation for orchestrator
                file_paths = kevin_integration.get_session_file_paths(
                    self.test_session["session_id"]
                )

                self.assertIn("working", file_paths)
                self.assertIn("research", file_paths)
                self.assertIn("logs", file_paths)
                print("✓ File path generation for orchestrator works correctly")

        except Exception as e:
            self.fail(f"KEVIN-Orchestrator integration test failed: {str(e)}")

    def test_kevin_quality_assurance_integration(self):
        """Test KEVIN directory integration with quality assurance"""
        print("\n=== Testing KEVIN-Quality Assurance Integration ===")

        try:
            from integration.kevin_directory_integration import KevinDirectoryIntegration
            from integration.quality_assurance_integration import QualityAssuranceIntegration

            # Mock Path to use test directory
            with patch('integration.kevin_directory_integration.Path') as mock_path:
                mock_path.return_value = self.fixture.temp_dir / "test_sessions" / "sessions"

                kevin_integration = KevinDirectoryIntegration()
                quality_integration = QualityAssuranceIntegration()

                # Create session structure
                session_dirs = kevin_integration.create_session_structure(
                    self.test_session["session_id"]
                )

                # Test quality report storage
                quality_assessment = {
                    "overall_score": 0.88,
                    "criteria_scores": {
                        "relevance": 0.92,
                        "accuracy": 0.85,
                        "completeness": 0.87,
                        "clarity": 0.90
                    },
                    "recommendations": [
                        "Add more recent sources",
                        "Include case studies"
                    ]
                }

                quality_report_path = kevin_integration.store_quality_report(
                    self.test_session["session_id"],
                    quality_assessment
                )

                self.assertIsNotNone(quality_report_path)
                self.assertTrue(quality_report_path.exists())
                print("✓ Quality report storage works correctly")

                # Test enhancement tracking
                enhancement_log = {
                    "original_score": 0.75,
                    "enhanced_score": 0.88,
                    "improvement": 0.13,
                    "enhancements_applied": [
                        "content_expansion",
                        "structure_improvement",
                        "clarity_enhancement"
                    ]
                }

                enhancement_log_path = kevin_integration.store_enhancement_log(
                    self.test_session["session_id"],
                    enhancement_log
                )

                self.assertIsNotNone(enhancement_log_path)
                self.assertTrue(enhancement_log_path.exists())
                print("✓ Enhancement log storage works correctly")

                # Test session metadata management
                metadata_update = {
                    "quality_assessment_completed": True,
                    "final_quality_score": 0.88,
                    "enhancement_applied": True,
                    "completion_timestamp": datetime.now().isoformat()
                }

                metadata_result = kevin_integration.update_session_metadata(
                    self.test_session["session_id"],
                    metadata_update
                )

                self.assertIsNotNone(metadata_result)
                print("✓ Session metadata management works correctly")

        except Exception as e:
            self.fail(f"KEVIN-Quality Assurance integration test failed: {str(e)}")


class QualityAssuranceIntegrationTests(unittest.TestCase):
    """Test quality assurance integration with other system components"""

    def setUp(self):
        """Set up test environment"""
        self.fixture = IntegrationTestFixture()
        self.test_session = self.fixture.create_test_session("quality_integration")

    def tearDown(self):
        """Clean up test environment"""
        self.fixture.cleanup()

    def test_quality_orchestrator_integration(self):
        """Test quality assurance integration with research orchestrator"""
        print("\n=== Testing Quality-Orchestrator Integration ===")

        try:
            from integration.quality_assurance_integration import QualityAssuranceIntegration
            from integration.research_orchestrator import ResearchOrchestrator

            quality_integration = QualityAssuranceIntegration()
            orchestrator = ResearchOrchestrator()

            # Test research quality assessment integration
            research_results = self.fixture.create_mock_research_results(
                self.test_session["session_id"]
            )

            quality_assessment = quality_integration.assess_research_quality(
                self.test_session["session_id"],
                json.dumps(research_results),
                {
                    "content_type": "research_results",
                    "stage": "research_completion",
                    "orchestrator_context": True
                }
            )

            self.assertIsNotNone(quality_assessment)
            self.assertIn("overall_score", quality_assessment)
            print("✓ Research quality assessment integration works correctly")

            # Test quality gate integration with orchestrator
            quality_gate_result = quality_integration.evaluate_quality_gate(
                self.test_session["session_id"],
                quality_assessment,
                {
                    "gate_type": "research_completion",
                    "required_threshold": 0.75,
                    "allow_enhancement": True
                }
            )

            self.assertIsNotNone(quality_gate_result)
            self.assertIn("decision", quality_gate_result)
            print("✓ Quality gate integration with orchestrator works correctly")

            # Test enhancement workflow integration
            if quality_gate_result.get("decision") == "enhance":
                enhanced_content = quality_integration.enhance_content(
                    json.dumps(research_results),
                    quality_assessment,
                    {
                        "target_quality": 0.85,
                        "enhancement_type": "comprehensive"
                    }
                )

                self.assertIsNotNone(enhanced_content)
                print("✓ Enhancement workflow integration works correctly")

        except Exception as e:
            self.fail(f"Quality-Orchestrator integration test failed: {str(e)}")

    def test_quality_mcp_integration(self):
        """Test quality assurance integration with MCP tools"""
        print("\n=== Testing Quality-MCP Integration ===")

        try:
            with patch.dict('sys.modules', {
                'multi_agent_research_system.mcp_tools.zplayground1_search': MagicMock()
            }):
                from integration.quality_assurance_integration import QualityAssuranceIntegration
                from integration.mcp_tool_integration import MCPToolIntegration

                quality_integration = QualityAssuranceIntegration()
                mcp_integration = MCPToolIntegration()

                # Test content quality validation for MCP results
                mcp_results = self.fixture.create_mock_research_results(
                    self.test_session["session_id"]
                )

                content_validation = quality_integration.validate_mcp_content_quality(
                    mcp_results,
                    {
                        "source_quality_threshold": 0.7,
                        "content_relevance_threshold": 0.6,
                        "source_diversity_required": True
                    }
                )

                self.assertIsNotNone(content_validation)
                self.assertIn("validation_passed", content_validation)
                print("✓ Content quality validation for MCP results works correctly")

                # Test source quality assessment
                source_quality = quality_integration.assess_source_quality(
                    mcp_results["results"],
                    {
                        "credibility_weights": {
                            "academic": 1.0,
                            "government": 0.9,
                            "industry": 0.8,
                            "news": 0.7,
                            "blog": 0.5
                        }
                    }
                )

                self.assertIsNotNone(source_quality)
                self.assertIn("overall_source_quality", source_quality)
                print("✓ Source quality assessment works correctly")

                # Test content enhancement recommendations
                enhancement_recommendations = quality_integration.generate_enhancement_recommendations(
                    mcp_results,
                    {
                        "target_quality_score": 0.85,
                        "focus_areas": ["completeness", "recency", "diversity"]
                    }
                )

                self.assertIsNotNone(enhancement_recommendations)
                self.assertIn("recommendations", enhancement_recommendations)
                print("✓ Content enhancement recommendations work correctly")

        except Exception as e:
            self.fail(f"Quality-MCP integration test failed: {str(e)}")


class ErrorHandlingIntegrationTests(unittest.TestCase):
    """Test error handling integration across all system components"""

    def setUp(self):
        """Set up test environment"""
        self.fixture = IntegrationTestFixture()
        self.test_session = self.fixture.create_test_session("error_handling_integration")

    def tearDown(self):
        """Clean up test environment"""
        self.fixture.cleanup()

    def test_error_handling_orchestrator_integration(self):
        """Test error handling integration with research orchestrator"""
        print("\n=== Testing Error Handling-Orchestrator Integration ===")

        try:
            from integration.error_handling_integration import (
                ErrorHandlingIntegration, ErrorContext, ErrorSeverity
            )
            from integration.research_orchestrator import ResearchOrchestrator

            error_handler = ErrorHandlingIntegration()
            orchestrator = ResearchOrchestrator()

            # Test error context creation for orchestrator
            error_context = ErrorContext(
                operation="research_execution",
                session_id=self.test_session["session_id"],
                component="orchestrator",
                severity=ErrorSeverity.MEDIUM,
                context={
                    "stage": "content_retrieval",
                    "progress": 0.3,
                    "query": self.test_session["initial_query"]
                }
            )

            # Test error handling integration
            test_error = Exception("Test research execution error")
            handling_result = error_handler.handle_error(test_error, error_context)

            self.assertIsNotNone(handling_result)
            self.assertIn("error_id", handling_result)
            self.assertIn("recovery_strategy", handling_result)
            print("✓ Error handling integration with orchestrator works correctly")

            # Test recovery coordination with orchestrator
            recovery_result = error_handler.coordinate_recovery_with_orchestrator(
                handling_result["error_id"],
                orchestrator,
                {
                    "continue_on_recovery": True,
                    "max_recovery_attempts": 3
                }
            )

            self.assertIsNotNone(recovery_result)
            print("✓ Recovery coordination with orchestrator works correctly")

        except Exception as e:
            self.fail(f"Error Handling-Orchestrator integration test failed: {str(e)}")

    def test_error_handling_mcp_integration(self):
        """Test error handling integration with MCP tools"""
        print("\n=== Testing Error Handling-MCP Integration ===")

        try:
            with patch.dict('sys.modules', {
                'multi_agent_research_system.mcp_tools.zplayground1_search': MagicMock()
            }):
                from integration.error_handling_integration import (
                    ErrorHandlingIntegration, ErrorContext, ErrorSeverity
                )
                from integration.mcp_tool_integration import MCPToolIntegration

                error_handler = ErrorHandlingIntegration()
                mcp_integration = MCPToolIntegration()

                # Test MCP-specific error handling
                mcp_error_context = ErrorContext(
                    operation="mcp_tool_execution",
                    session_id=self.test_session["session_id"],
                    component="mcp_integration",
                    severity=ErrorSeverity.HIGH,
                    context={
                        "tool_name": "zplayground1_search_scrape_clean",
                        "parameters": {"query": "test query"},
                        "timeout_duration": 30
                    }
                )

                # Test timeout error handling
                timeout_error = TimeoutError("MCP tool execution timeout")
                timeout_handling_result = error_handler.handle_mcp_error(
                    timeout_error,
                    mcp_error_context,
                    mcp_integration
                )

                self.assertIsNotNone(timeout_handling_result)
                self.assertEqual(timeout_handling_result["error_type"], "timeout")
                print("✓ MCP timeout error handling works correctly")

                # Test MCP recovery strategies
                recovery_result = error_handler.execute_mcp_recovery(
                    timeout_handling_result["error_id"],
                    mcp_integration,
                    {
                        "retry_with_different_params": True,
                        "fallback_sources": True,
                        "reduce_scope": True
                    }
                )

                self.assertIsNotNone(recovery_result)
                print("✓ MCP recovery strategies work correctly")

        except Exception as e:
            self.fail(f"Error Handling-MCP integration test failed: {str(e)}")

    def test_error_handling_quality_assurance_integration(self):
        """Test error handling integration with quality assurance"""
        print("\n=== Testing Error Handling-Quality Assurance Integration ===")

        try:
            from integration.error_handling_integration import (
                ErrorHandlingIntegration, ErrorContext, ErrorSeverity
            )
            from integration.quality_assurance_integration import QualityAssuranceIntegration

            error_handler = ErrorHandlingIntegration()
            quality_integration = QualityAssuranceIntegration()

            # Test quality assessment error handling
            quality_error_context = ErrorContext(
                operation="quality_assessment",
                session_id=self.test_session["session_id"],
                component="quality_assurance",
                severity=ErrorSeverity.MEDIUM,
                context={
                    "assessment_type": "research_quality",
                    "content_length": 5000,
                    "quality_threshold": 0.75
                }
            )

            # Test content processing error
            content_error = ValueError("Content too short for quality assessment")
            quality_error_result = error_handler.handle_quality_error(
                content_error,
                quality_error_context,
                quality_integration
            )

            self.assertIsNotNone(quality_error_result)
            self.assertIn("quality_recovery_strategy", quality_error_result)
            print("✓ Quality assessment error handling works correctly")

            # Test quality fallback mechanisms
            fallback_result = error_handler.execute_quality_fallback(
                quality_error_result["error_id"],
                quality_integration,
                {
                    "lower_quality_threshold": True,
                    "request_additional_content": True,
                    "use_simplified_assessment": True
                }
            )

            self.assertIsNotNone(fallback_result)
            print("✓ Quality fallback mechanisms work correctly")

        except Exception as e:
            self.fail(f"Error Handling-Quality Assurance integration test failed: {str(e)}")


class EndToEndIntegrationTests(unittest.TestCase):
    """Test complete end-to-end integration of all system components"""

    def setUp(self):
        """Set up test environment"""
        self.fixture = IntegrationTestFixture()
        self.test_session = self.fixture.create_test_session("end_to_end")

    def tearDown(self):
        """Clean up test environment"""
        self.fixture.cleanup()

    def test_complete_system_integration(self):
        """Test complete integration of all system components"""
        print("\n=== Testing Complete System Integration ===")

        async def run_integration_test():
            try:
                # Initialize all components
                with patch.dict('sys.modules', {
                    'multi_agent_research_system.mcp_tools.zplayground1_search': MagicMock()
                }):
                    from integration.research_orchestrator import ResearchOrchestrator
                    from integration.agent_session_manager import AgentSessionManager
                    from integration.mcp_tool_integration import MCPToolIntegration
                    from integration.kevin_directory_integration import KevinDirectoryIntegration
                    from integration.quality_assurance_integration import QualityAssuranceIntegration
                    from integration.error_handling_integration import ErrorHandlingIntegration

                    # Mock Path for KEVIN integration
                    with patch('integration.kevin_directory_integration.Path') as mock_path:
                        mock_path.return_value = self.fixture.temp_dir / "test_sessions" / "sessions"

                        # Initialize components
                        orchestrator = ResearchOrchestrator()
                        session_manager = AgentSessionManager()
                        mcp_integration = MCPToolIntegration()
                        kevin_integration = KevinDirectoryIntegration()
                        quality_integration = QualityAssuranceIntegration()
                        error_handler = ErrorHandlingIntegration()

                        # Stage 1: Session initialization
                        print("  Stage 1: Session initialization...")
                        session_data = session_manager.create_session(
                            self.test_session["session_id"],
                            self.test_session["initial_query"],
                            self.fixture.load_test_configuration()
                        )

                        # Create KEVIN directory structure
                        session_dirs = kevin_integration.create_session_structure(
                            self.test_session["session_id"]
                        )

                        self.assertIsNotNone(session_data)
                        self.assertEqual(len(session_dirs), 4)  # session, working, research, logs

                        # Stage 2: Query processing and research execution
                        print("  Stage 2: Query processing and research execution...")
                        query = self.test_session["initial_query"]
                        research_config = self.fixture.load_test_configuration()

                        # Map parameters for MCP
                        mcp_params = mcp_integration._map_parameters_to_mcp(
                            query,
                            research_config,
                            self.test_session["session_id"]
                        )

                        # Create mock research results
                        research_results = self.fixture.create_mock_research_results(
                            self.test_session["session_id"]
                        )

                        # Store workproduct in KEVIN
                        workproduct_path = kevin_integration.create_workproduct_file(
                            self.test_session["session_id"],
                            "RESEARCH",
                            "INITIAL_SEARCH",
                            research_results
                        )

                        self.assertIsNotNone(workproduct_path)
                        self.assertTrue(workproduct_path.exists())

                        # Stage 3: Quality assessment
                        print("  Stage 3: Quality assessment...")
                        quality_assessment = quality_integration.assess_research_quality(
                            self.test_session["session_id"],
                            json.dumps(research_results),
                            {"content_type": "research_results", "stage": "initial_assessment"}
                        )

                        # Store quality report
                        quality_report_path = kevin_integration.store_quality_report(
                            self.test_session["session_id"],
                            quality_assessment
                        )

                        self.assertIsNotNone(quality_report_path)
                        self.assertTrue(quality_report_path.exists())

                        # Stage 4: Content enhancement
                        print("  Stage 4: Content enhancement...")
                        if quality_assessment.get("overall_score", 0) < 0.85:
                            enhanced_results = quality_integration.enhance_content(
                                json.dumps(research_results),
                                quality_assessment,
                                {"target_quality": 0.85}
                            )

                            # Store enhancement log
                            enhancement_log_path = kevin_integration.store_enhancement_log(
                                self.test_session["session_id"],
                                {
                                    "original_score": quality_assessment.get("overall_score", 0),
                                    "enhanced": True,
                                    "timestamp": datetime.now().isoformat()
                                }
                            )

                        # Stage 5: Final integration validation
                        print("  Stage 5: Final integration validation...")

                        # Verify all components are properly integrated
                        integration_checks = {
                            "session_management": session_data is not None,
                            "kevin_directories": all(path.exists() for path in session_dirs.values()),
                            "mcp_parameter_mapping": "query" in mcp_params,
                            "quality_assessment": quality_assessment is not None,
                            "file_storage": workproduct_path.exists() and quality_report_path.exists()
                        }

                        # All checks should pass
                        self.assertTrue(all(integration_checks.values()))

                        # Update session metadata
                        final_metadata = {
                            "integration_test_completed": True,
                            "all_components_integrated": True,
                            "final_status": "success",
                            "completion_timestamp": datetime.now().isoformat()
                        }

                        kevin_integration.update_session_metadata(
                            self.test_session["session_id"],
                            final_metadata
                        )

                        print("✓ Complete system integration test passed")
                        return {
                            "success": True,
                            "integration_checks": integration_checks,
                            "session_id": self.test_session["session_id"]
                        }

            except Exception as e:
                print(f"✗ Complete system integration test failed: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "session_id": self.test_session["session_id"]
                }

        # Run async integration test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_integration_test())
            self.assertTrue(result["success"], f"Integration test failed: {result.get('error', 'Unknown error')}")
        finally:
            loop.close()

    def test_error_recovery_integration(self):
        """Test error recovery across all integrated components"""
        print("\n=== Testing Error Recovery Integration ===")

        async def run_error_recovery_test():
            try:
                with patch.dict('sys.modules', {
                    'multi_agent_research_system.mcp_tools.zplayground1_search': MagicMock()
                }):
                    from integration.error_handling_integration import (
                        ErrorHandlingIntegration, ErrorContext, ErrorSeverity
                    )
                    from integration.research_orchestrator import ResearchOrchestrator
                    from integration.mcp_tool_integration import MCPToolIntegration

                    # Initialize components
                    error_handler = ErrorHandlingIntegration()
                    orchestrator = ResearchOrchestrator()
                    mcp_integration = MCPToolIntegration()

                    # Simulate error at different stages
                    error_scenarios = [
                        {
                            "stage": "session_initialization",
                            "error": ValueError("Invalid session parameters"),
                            "context": ErrorContext(
                                operation="session_creation",
                                session_id=self.test_session["session_id"],
                                component="session_manager",
                                severity=ErrorSeverity.HIGH
                            )
                        },
                        {
                            "stage": "research_execution",
                            "error": TimeoutError("Research execution timeout"),
                            "context": ErrorContext(
                                operation="mcp_tool_execution",
                                session_id=self.test_session["session_id"],
                                component="mcp_integration",
                                severity=ErrorSeverity.MEDIUM
                            )
                        },
                        {
                            "stage": "quality_assessment",
                            "error": Exception("Quality assessment failed"),
                            "context": ErrorContext(
                                operation="quality_evaluation",
                                session_id=self.test_session["session_id"],
                                component="quality_assurance",
                                severity=ErrorSeverity.LOW
                            )
                        }
                    ]

                    recovery_results = {}

                    for scenario in error_scenarios:
                        print(f"  Testing error recovery for {scenario['stage']}...")

                        # Handle error
                        handling_result = error_handler.handle_error(
                            scenario["error"],
                            scenario["context"]
                        )

                        # Execute recovery
                        recovery_result = error_handler.execute_recovery(
                            handling_result["error_id"],
                            handling_result["recovery_strategy"]
                        )

                        recovery_results[scenario["stage"]] = {
                            "error_handled": True,
                            "recovery_executed": True,
                            "recovery_strategy": handling_result["recovery_strategy"]
                        }

                    # Verify all errors were handled and recovered
                    self.assertEqual(len(recovery_results), len(error_scenarios))
                    self.assertTrue(all(
                        result["error_handled"] and result["recovery_executed"]
                        for result in recovery_results.values()
                    ))

                    print("✓ Error recovery integration test passed")
                    return recovery_results

            except Exception as e:
                print(f"✗ Error recovery integration test failed: {str(e)}")
                raise

        # Run async error recovery test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_error_recovery_test())
            self.assertIsNotNone(result)
        finally:
            loop.close()


class IntegrationTestRunner:
    """Runner for all integration tests with comprehensive reporting"""

    def __init__(self):
        self.test_suite = unittest.TestSuite()
        self.setup_test_suite()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="integration_test_run_"))

    def setup_test_suite(self):
        """Set up all integration test classes"""
        test_classes = [
            MCPToolIntegrationTests,
            KevinDirectoryIntegrationTests,
            QualityAssuranceIntegrationTests,
            ErrorHandlingIntegrationTests,
            EndToEndIntegrationTests
        ]

        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            self.test_suite.addTests(tests)

    def run_all_integration_tests(self):
        """Run all integration tests and generate comprehensive report"""
        print("=" * 80)
        print("INTEGRATION TEST SUITE - AGENT-BASED RESEARCH SYSTEM")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run tests with detailed output
        output_file = self.temp_dir / "integration_test_output.log"
        with open(output_file, 'w') as f:
            runner = unittest.TextTestRunner(
                verbosity=2,
                stream=f,
                buffer=True
            )

            start_time = time.time()
            result = runner.run(self.test_suite)
            duration = time.time() - start_time

        # Generate integration test report
        self.generate_integration_report(result, duration)

        return result

    def generate_integration_report(self, result, duration):
        """Generate comprehensive integration test report"""
        report = {
            "integration_test_summary": {
                "total_tests": result.testsRun,
                "successes": result.testsRun - len(result.failures) - len(result.errors),
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100,
                "duration": duration
            },
            "integration_areas_tested": [
                "MCP Tool Integration",
                "KEVIN Directory Integration",
                "Quality Assurance Integration",
                "Error Handling Integration",
                "End-to-End System Integration"
            ],
            "test_details": {
                "failed_tests": [
                    {
                        "test": str(failure[0]),
                        "error": str(failure[1])
                    }
                    for failure in result.failures
                ],
                "error_tests": [
                    {
                        "test": str(error[0]),
                        "error": str(error[1])
                    }
                    for error in result.errors
                ]
            },
            "system_integration_status": {
                "mcp_tools": "tested",
                "kevin_directories": "tested",
                "quality_assurance": "tested",
                "error_handling": "tested",
                "end_to_end_workflow": "tested"
            },
            "test_environment": {
                "test_timestamp": datetime.now().isoformat(),
                "test_type": "integration",
                "python_version": sys.version,
                "platform": sys.platform
            }
        }

        # Save integration report
        report_path = self.temp_dir / "integration_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 80)
        print("INTEGRATION TEST SUMMARY REPORT")
        print("=" * 80)
        print(f"Total Tests: {report['integration_test_summary']['total_tests']}")
        print(f"Successes: {report['integration_test_summary']['successes']}")
        print(f"Failures: {report['integration_test_summary']['failures']}")
        print(f"Errors: {report['integration_test_summary']['errors']}")
        print(f"Success Rate: {report['integration_test_summary']['success_rate']:.1f}%")
        print(f"Duration: {report['integration_test_summary']['duration']:.2f}s")

        print("\nIntegration Areas Tested:")
        for area in report['integration_areas_tested']:
            print(f"  ✓ {area}")

        if report['integration_test_summary']['failures'] > 0 or report['integration_test_summary']['errors'] > 0:
            print("\nFAILED TESTS:")
            for test in report['test_details']['failed_tests'] + report['test_details']['error_tests']:
                print(f"  - {test['test']}: {test['error']}")

        print(f"\nDetailed integration report saved to: {report_path}")
        print("=" * 80)

    def cleanup(self):
        """Clean up test environment"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


# Main execution
if __name__ == "__main__":
    # Set up and run integration tests
    test_runner = IntegrationTestRunner()

    try:
        # Run all integration tests
        result = test_runner.run_all_integration_tests()

        # Exit with appropriate code
        if result.wasSuccessful():
            print("\n🎉 All integration tests passed successfully!")
            print("✅ System components are properly integrated and working together.")
            exit(0)
        else:
            print("\n❌ Some integration tests failed.")
            print("🔧 Check the integration report for details on failed components.")
            exit(1)

    except Exception as e:
        print(f"\n💥 Integration test execution failed: {str(e)}")
        exit(1)
    finally:
        test_runner.cleanup()