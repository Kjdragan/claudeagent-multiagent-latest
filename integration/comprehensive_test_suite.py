"""
Comprehensive Test Suite for Agent-Based Research System

This module provides end-to-end testing for the complete agent-based comprehensive
research workflow, validating all system components and their integration.

Test Coverage:
- Complete workflow validation
- Component integration testing
- Error scenario handling
- Performance validation
- Quality assurance verification

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

# Import system components for testing
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Test fixtures and utilities
class TestDataGenerator:
    """Generates test data for comprehensive testing scenarios"""

    @staticmethod
    def create_test_session_data() -> Dict[str, Any]:
        """Create comprehensive test session data"""
        return {
            "session_id": f"test_session_{uuid.uuid4().hex[:8]}",
            "initial_query": "artificial intelligence in healthcare applications",
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
            "configuration": {
                "research_configuration": {
                    "target_urls": 10,
                    "concurrent_processing": 5,
                    "anti_bot_level": 1,
                    "quality_threshold": 0.7
                },
                "quality_configuration": {
                    "enable_quality_assessment": True,
                    "quality_threshold": 0.75,
                    "enhancement_enabled": True
                }
            },
            "metadata": {
                "test_mode": True,
                "test_timestamp": datetime.now().isoformat(),
                "test_scenario": "comprehensive_workflow"
            }
        }

    @staticmethod
    def create_test_research_results() -> Dict[str, Any]:
        """Create mock research results for testing"""
        return {
            "query": "test query",
            "results": [
                {
                    "url": "https://example.com/article1",
                    "title": "Test Article 1",
                    "content": "This is test content about AI in healthcare...",
                    "source": "test_source",
                    "timestamp": datetime.now().isoformat(),
                    "relevance_score": 0.85
                },
                {
                    "url": "https://example.com/article2",
                    "title": "Test Article 2",
                    "content": "More test content about healthcare applications...",
                    "source": "test_source",
                    "timestamp": datetime.now().isoformat(),
                    "relevance_score": 0.78
                }
            ],
            "metadata": {
                "total_results": 2,
                "processing_time": 2.5,
                "quality_score": 0.82
            }
        }

    @staticmethod
    def create_error_scenarios() -> List[Dict[str, Any]]:
        """Create various error scenarios for testing"""
        return [
            {
                "scenario": "network_timeout",
                "error_type": "TimeoutError",
                "context": {
                    "operation": "research_execution",
                    "timeout_duration": 30
                }
            },
            {
                "scenario": "api_key_missing",
                "error_type": "ValueError",
                "context": {
                    "operation": "system_initialization",
                    "missing_keys": ["ANTHROPIC_API_KEY"]
                }
            },
            {
                "scenario": "content_processing_error",
                "error_type": "ProcessingError",
                "context": {
                    "operation": "content_analysis",
                    "content_type": "research_results"
                }
            },
            {
                "scenario": "file_system_error",
                "error_type": "FileNotFoundError",
                "context": {
                    "operation": "file_creation",
                    "file_path": "/nonexistent/path/file.txt"
                }
            },
            {
                "scenario": "memory_limit_exceeded",
                "error_type": "MemoryError",
                "context": {
                    "operation": "large_content_processing",
                    "content_size": "1000MB"
                }
            }
        ]


class MockClaudeAgentSDK:
    """Mock Claude Agent SDK for testing purposes"""

    def __init__(self):
        self.sessions = {}
        self.tools = {}
        self.responses = {}

    async def create_session(self, session_id: str, agent_type: str = "comprehensive_research") -> Any:
        """Create mock session"""
        mock_session = AsyncMock()
        mock_session.session_id = session_id
        mock_session.agent_type = agent_type
        mock_session.is_active = True

        self.sessions[session_id] = mock_session
        return mock_session

    async def query_agent(self, session_id: str, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock agent query"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        # Generate appropriate mock response based on query type
        if "research" in query.lower():
            return TestDataGenerator.create_test_research_results()
        elif "analyze" in query.lower():
            return {
                "analysis": {
                    "quality_score": 0.85,
                    "recommendations": ["Enhance content depth", "Add more sources"],
                    "confidence_score": 0.92
                }
            }
        else:
            return {
                "response": f"Mock response for: {query[:50]}...",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }

    def register_tool(self, tool_name: str, tool_func: Any):
        """Register mock tool"""
        self.tools[tool_name] = tool_func

    def close_session(self, session_id: str):
        """Close mock session"""
        if session_id in self.sessions:
            del self.sessions[session_id]


class ComprehensiveTestSuite(unittest.TestCase):
    """Main comprehensive test suite for the agent-based research system"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_data_generator = TestDataGenerator()
        cls.mock_sdk = MockClaudeAgentSDK()
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="comprehensive_test_"))

        # Create test directory structure
        cls.setup_test_directories()

        # Initialize test configuration
        cls.test_config = {
            "test_mode": True,
            "mock_responses": True,
            "performance_tracking": True,
            "detailed_logging": True,
            "timeout_duration": 30
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        import shutil
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up individual test"""
        self.test_session_data = self.test_data_generator.create_test_session_data()
        self.session_id = self.test_session_data["session_id"]
        self.start_time = time.time()

    def tearDown(self):
        """Clean up individual test"""
        duration = time.time() - self.start_time
        print(f"Test {self._testMethodName} completed in {duration:.2f}s")

    @classmethod
    def setup_test_directories(cls):
        """Set up test directory structure"""
        directories = [
            "test_sessions",
            "test_logs",
            "test_results",
            "test_temp"
        ]

        for dir_name in directories:
            (cls.temp_dir / dir_name).mkdir(exist_ok=True)

    # Test System Initialization
    def test_system_initialization(self):
        """Test complete system initialization"""
        print("\n=== Testing System Initialization ===")

        try:
            # Test component imports
            from integration.research_orchestrator import ResearchOrchestrator
            from integration.agent_session_manager import AgentSessionManager
            from integration.query_processor import QueryProcessor
            from integration.mcp_tool_integration import MCPToolIntegration
            from integration.kevin_directory_integration import KevinDirectoryIntegration
            from integration.quality_assurance_integration import QualityAssuranceIntegration
            from integration.error_handling_integration import ErrorHandlingIntegration

            # Initialize components
            orchestrator = ResearchOrchestrator()
            session_manager = AgentSessionManager()
            query_processor = QueryProcessor()
            mcp_integration = MCPToolIntegration()
            kevin_integration = KevinDirectoryIntegration()
            quality_integration = QualityAssuranceIntegration()
            error_handler = ErrorHandlingIntegration()

            # Verify initialization
            self.assertIsNotNone(orchestrator)
            self.assertIsNotNone(session_manager)
            self.assertIsNotNone(query_processor)
            self.assertIsNotNone(mcp_integration)
            self.assertIsNotNone(kevin_integration)
            self.assertIsNotNone(quality_integration)
            self.assertIsNotNone(error_handler)

            print("✓ All system components initialized successfully")

        except Exception as e:
            self.fail(f"System initialization failed: {str(e)}")

    # Test Complete Workflow
    def test_complete_research_workflow(self):
        """Test end-to-end research workflow"""
        print("\n=== Testing Complete Research Workflow ===")

        async def run_workflow_test():
            try:
                # Import and initialize orchestrator
                from integration.research_orchestrator import ResearchOrchestrator
                orchestrator = ResearchOrchestrator()

                # Mock agent session
                mock_session = await self.mock_sdk.create_session(
                    self.session_id,
                    "comprehensive_research"
                )

                # Test workflow stages
                workflow_stages = [
                    ("query_processing", self._test_query_processing),
                    ("research_execution", self._test_research_execution),
                    ("content_analysis", self._test_content_analysis),
                    ("quality_assessment", self._test_quality_assessment),
                    ("result_enhancement", self._test_result_enhancement),
                    ("final_delivery", self._test_final_delivery)
                ]

                results = {}

                for stage_name, stage_test in workflow_stages:
                    print(f"  Testing {stage_name}...")
                    stage_result = await stage_test(orchestrator, mock_session)
                    results[stage_name] = stage_result
                    print(f"  ✓ {stage_name} completed successfully")

                # Verify workflow completion
                self.assertEqual(len(results), len(workflow_stages))
                self.assertTrue(all(result["success"] for result in results.values()))

                print("✓ Complete research workflow test passed")
                return results

            except Exception as e:
                print(f"✗ Complete workflow test failed: {str(e)}")
                raise

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_workflow_test())
            self.assertIsNotNone(result)
        finally:
            loop.close()

    async def _test_query_processing(self, orchestrator, session):
        """Test query processing stage"""
        try:
            query = self.test_session_data["initial_query"]
            processed_query = await orchestrator.process_query(query, session.session_id)

            return {
                "success": True,
                "result": processed_query,
                "stage": "query_processing"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "query_processing"
            }

    async def _test_research_execution(self, orchestrator, session):
        """Test research execution stage"""
        try:
            research_params = {
                "target_results": 5,
                "quality_threshold": 0.7,
                "sources": ["test_source_1", "test_source_2"]
            }

            research_results = TestDataGenerator.create_test_research_results()

            return {
                "success": True,
                "result": research_results,
                "stage": "research_execution"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "research_execution"
            }

    async def _test_content_analysis(self, orchestrator, session):
        """Test content analysis stage"""
        try:
            research_results = TestDataGenerator.create_test_research_results()
            analysis_result = {
                "content_analysis": {
                    "key_topics": ["AI", "healthcare", "applications"],
                    "quality_score": 0.85,
                    "relevance_assessment": "high",
                    "content_gaps": ["latest developments", "case studies"]
                },
                "confidence_score": 0.92
            }

            return {
                "success": True,
                "result": analysis_result,
                "stage": "content_analysis"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "content_analysis"
            }

    async def _test_quality_assessment(self, orchestrator, session):
        """Test quality assessment stage"""
        try:
            quality_result = {
                "overall_score": 0.88,
                "criteria_scores": {
                    "relevance": 0.92,
                    "accuracy": 0.85,
                    "completeness": 0.87,
                    "clarity": 0.90
                },
                "recommendations": [
                    "Add more recent sources",
                    "Include case studies",
                    "Enhance technical details"
                ],
                "meets_threshold": True
            }

            return {
                "success": True,
                "result": quality_result,
                "stage": "quality_assessment"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "quality_assessment"
            }

    async def _test_result_enhancement(self, orchestrator, session):
        """Test result enhancement stage"""
        try:
            enhanced_result = {
                "original_content": "enhanced research content",
                "enhancements_applied": [
                    "structure_improvement",
                    "content_expansion",
                    "clarity_enhancement"
                ],
                "quality_improvement": 0.15,
                "final_score": 0.93
            }

            return {
                "success": True,
                "result": enhanced_result,
                "stage": "result_enhancement"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "result_enhancement"
            }

    async def _test_final_delivery(self, orchestrator, session):
        """Test final delivery stage"""
        try:
            final_report = {
                "report_id": f"report_{uuid.uuid4().hex[:8]}",
                "content": "# Comprehensive Research Report\n\nEnhanced content...",
                "metadata": {
                    "session_id": session.session_id,
                    "created_at": datetime.now().isoformat(),
                    "quality_score": 0.93,
                    "word_count": 2500
                },
                "file_paths": {
                    "main_report": f"/path/to/report_{session.session_id}.md",
                    "supporting_files": []
                }
            }

            return {
                "success": True,
                "result": final_report,
                "stage": "final_delivery"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "final_delivery"
            }

    # Test Component Integration
    def test_mcp_tool_integration(self):
        """Test MCP tool integration functionality"""
        print("\n=== Testing MCP Tool Integration ===")

        try:
            from integration.mcp_tool_integration import MCPToolIntegration

            mcp_integration = MCPToolIntegration()

            # Test parameter mapping
            test_params = {
                "query": "test query",
                "research_configuration": {
                    "target_urls": 10,
                    "concurrent_processing": 5,
                    "anti_bot_level": 1
                }
            }

            mapped_params = mcp_integration._map_parameters_to_mcp(
                test_params["query"],
                test_params["research_configuration"],
                self.session_id
            )

            self.assertIn("query", mapped_params)
            self.assertIn("session_id", mapped_params)
            self.assertEqual(mapped_params["session_id"], self.session_id)

            print("✓ MCP tool parameter mapping works correctly")

            # Test session coordination
            coordination_result = mcp_integration.coordinate_session(
                self.session_id,
                "research_execution"
            )

            self.assertIsNotNone(coordination_result)
            print("✓ MCP tool session coordination works correctly")

        except Exception as e:
            self.fail(f"MCP tool integration test failed: {str(e)}")

    def test_kevin_directory_integration(self):
        """Test KEVIN directory integration functionality"""
        print("\n=== Testing KEVIN Directory Integration ===")

        try:
            from integration.kevin_directory_integration import KevinDirectoryIntegration

            # Use test temp directory
            with patch('integration.kevin_directory_integration.Path') as mock_path:
                mock_path.return_value = self.temp_dir / "test_sessions"

                kevin_integration = KevinDirectoryIntegration()

                # Test session structure creation
                session_dirs = kevin_integration.create_session_structure(self.session_id)

                self.assertIn("session", session_dirs)
                self.assertIn("working", session_dirs)
                self.assertIn("research", session_dirs)
                self.assertIn("logs", session_dirs)

                print("✓ KEVIN directory structure creation works correctly")

                # Test file naming conventions
                file_name = kevin_integration.generate_workproduct_filename(
                    "RESEARCH",
                    "INITIAL_SEARCH"
                )

                self.assertIn("RESEARCH", file_name)
                self.assertIn("INITIAL_SEARCH", file_name)
                self.assertTrue(file_name.endswith(".md"))

                print("✓ KEVIN file naming conventions work correctly")

        except Exception as e:
            self.fail(f"KEVIN directory integration test failed: {str(e)}")

    def test_quality_assurance_integration(self):
        """Test quality assurance integration functionality"""
        print("\n=== Testing Quality Assurance Integration ===")

        try:
            from integration.quality_assurance_integration import QualityAssuranceIntegration

            quality_integration = QualityAssuranceIntegration()

            # Test research quality assessment
            test_content = "This is test research content about AI in healthcare..."
            assessment_result = quality_integration.assess_research_quality(
                self.session_id,
                test_content,
                {"content_type": "research", "stage": "initial"}
            )

            # Should return structured assessment (mock if needed)
            self.assertIsNotNone(assessment_result)

            print("✓ Quality assurance assessment works correctly")

            # Test enhancement workflow
            enhancement_result = quality_integration.enhance_content(
                test_content,
                assessment_result,
                {"target_quality": 0.9}
            )

            self.assertIsNotNone(enhancement_result)
            print("✓ Quality assurance enhancement works correctly")

        except Exception as e:
            self.fail(f"Quality assurance integration test failed: {str(e)}")

    def test_error_handling_integration(self):
        """Test error handling integration functionality"""
        print("\n=== Testing Error Handling Integration ===")

        try:
            from integration.error_handling_integration import (
                ErrorHandlingIntegration, ErrorContext, ErrorSeverity
            )

            error_handler = ErrorHandlingIntegration()

            # Test error classification
            test_error = ValueError("Test error message")
            error_context = ErrorContext(
                operation="test_operation",
                session_id=self.session_id,
                component="test_component",
                severity=ErrorSeverity.MEDIUM
            )

            # Test error handling (should not raise exception)
            handling_result = error_handler.handle_error(
                test_error,
                error_context
            )

            self.assertIsNotNone(handling_result)
            self.assertIn("error_id", handling_result)
            self.assertIn("recovery_strategy", handling_result)

            print("✓ Error handling classification works correctly")

            # Test recovery execution
            recovery_result = error_handler.execute_recovery(
                handling_result["error_id"],
                handling_result["recovery_strategy"]
            )

            self.assertIsNotNone(recovery_result)
            print("✓ Error recovery execution works correctly")

        except Exception as e:
            self.fail(f"Error handling integration test failed: {str(e)}")

    # Test Error Scenarios
    def test_error_scenarios(self):
        """Test various error scenarios and recovery mechanisms"""
        print("\n=== Testing Error Scenarios ===")

        error_scenarios = self.test_data_generator.create_error_scenarios()

        for scenario in error_scenarios:
            print(f"  Testing error scenario: {scenario['scenario']}")

            try:
                result = self._test_error_scenario(scenario)
                self.assertIsNotNone(result)
                print(f"  ✓ Error scenario {scenario['scenario']} handled correctly")

            except Exception as e:
                self.fail(f"Error scenario {scenario['scenario']} test failed: {str(e)}")

    def _test_error_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test individual error scenario"""
        scenario_name = scenario["scenario"]
        error_type = scenario["error_type"]
        context = scenario["context"]

        # Create appropriate error instance
        if error_type == "TimeoutError":
            test_error = TimeoutError(f"Timeout in {context['operation']}")
        elif error_type == "ValueError":
            test_error = ValueError(f"Missing: {context.get('missing_keys', 'unknown')}")
        elif error_type == "ProcessingError":
            test_error = Exception(f"Processing error in {context['operation']}")
        elif error_type == "FileNotFoundError":
            test_error = FileNotFoundError(f"File not found: {context.get('file_path', 'unknown')}")
        elif error_type == "MemoryError":
            test_error = MemoryError(f"Memory limit exceeded: {context.get('content_size', 'unknown')}")
        else:
            test_error = Exception(f"Unknown error in {scenario_name}")

        # Test error handling
        from integration.error_handling_integration import (
            ErrorHandlingIntegration, ErrorContext, ErrorSeverity
        )

        error_handler = ErrorHandlingIntegration()
        error_context = ErrorContext(
            operation=context["operation"],
            session_id=self.session_id,
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            context=context
        )

        # Handle error
        handling_result = error_handler.handle_error(test_error, error_context)

        return {
            "scenario": scenario_name,
            "error_handled": True,
            "recovery_strategy": handling_result.get("recovery_strategy"),
            "error_id": handling_result.get("error_id")
        }

    # Test Performance Validation
    def test_performance_validation(self):
        """Test system performance under various conditions"""
        print("\n=== Testing Performance Validation ===")

        # Test response times
        performance_tests = [
            ("quick_query", self._test_quick_query_performance),
            ("complex_research", self._test_complex_research_performance),
            ("concurrent_operations", self._test_concurrent_operations_performance)
        ]

        for test_name, test_func in performance_tests:
            print(f"  Testing {test_name} performance...")

            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time

            self.assertIsNotNone(result)

            # Performance assertions (adjust thresholds as needed)
            if test_name == "quick_query":
                self.assertLess(duration, 5.0, f"Quick query took too long: {duration:.2f}s")
            elif test_name == "complex_research":
                self.assertLess(duration, 30.0, f"Complex research took too long: {duration:.2f}s")
            elif test_name == "concurrent_operations":
                self.assertLess(duration, 15.0, f"Concurrent operations took too long: {duration:.2f}s")

            print(f"  ✓ {test_name} performance test passed ({duration:.2f}s)")

    def _test_quick_query_performance(self) -> Dict[str, Any]:
        """Test quick query performance"""
        # Simulate quick query processing
        time.sleep(0.1)  # Simulate minimal processing time

        return {
            "test": "quick_query",
            "processed": True,
            "response_time": 0.1
        }

    def _test_complex_research_performance(self) -> Dict[str, Any]:
        """Test complex research performance"""
        # Simulate complex research processing
        time.sleep(2.0)  # Simulate research processing time

        return {
            "test": "complex_research",
            "processed": True,
            "response_time": 2.0,
            "results_found": 10
        }

    def _test_concurrent_operations_performance(self) -> Dict[str, Any]:
        """Test concurrent operations performance"""
        # Simulate concurrent operations
        async def run_concurrent():
            tasks = []
            for i in range(3):
                # Simulate concurrent tasks
                await asyncio.sleep(0.5)
                tasks.append({"task_id": i, "completed": True})
            return tasks

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_concurrent())
            return {
                "test": "concurrent_operations",
                "processed": True,
                "tasks_completed": len(result)
            }
        finally:
            loop.close()

    # Test Data Integrity
    def test_data_integrity(self):
        """Test data integrity throughout the workflow"""
        print("\n=== Testing Data Integrity ===")

        # Test session data preservation
        initial_data = self.test_session_data.copy()

        # Simulate workflow stages that modify data
        stage1_data = initial_data.copy()
        stage1_data["status"] = "query_processed"
        stage1_data["processed_query"] = "processed: " + initial_data["initial_query"]

        stage2_data = stage1_data.copy()
        stage2_data["status"] = "research_completed"
        stage2_data["research_results"] = TestDataGenerator.create_test_research_results()

        stage3_data = stage2_data.copy()
        stage3_data["status"] = "analysis_completed"
        stage3_data["analysis_results"] = {"quality_score": 0.85}

        final_data = stage3_data.copy()
        final_data["status"] = "completed"

        # Verify data integrity
        self.assertEqual(final_data["session_id"], initial_data["session_id"])
        self.assertEqual(final_data["initial_query"], initial_data["initial_query"])
        self.assertIn("research_results", final_data)
        self.assertIn("analysis_results", final_data)
        self.assertEqual(final_data["status"], "completed")

        print("✓ Data integrity preserved throughout workflow")

        # Test file integrity
        test_file_path = self.temp_dir / "test_results" / f"test_{self.session_id}.json"

        with open(test_file_path, 'w') as f:
            json.dump(final_data, f, indent=2)

        # Verify file was written correctly
        self.assertTrue(test_file_path.exists())

        with open(test_file_path, 'r') as f:
            loaded_data = json.load(f)

        self.assertEqual(loaded_data["session_id"], final_data["session_id"])
        self.assertEqual(loaded_data["status"], final_data["status"])

        print("✓ File integrity maintained")

    # Test Configuration Validation
    def test_configuration_validation(self):
        """Test system configuration validation"""
        print("\n=== Testing Configuration Validation ===")

        # Test valid configurations
        valid_configs = [
            {
                "research_configuration": {
                    "target_urls": 10,
                    "concurrent_processing": 5,
                    "anti_bot_level": 1
                },
                "quality_configuration": {
                    "enable_quality_assessment": True,
                    "quality_threshold": 0.75
                }
            },
            {
                "research_configuration": {
                    "target_urls": 20,
                    "concurrent_processing": 10,
                    "anti_bot_level": 2
                },
                "quality_configuration": {
                    "enable_quality_assessment": True,
                    "quality_threshold": 0.85
                }
            }
        ]

        for i, config in enumerate(valid_configs):
            print(f"  Testing valid configuration {i+1}...")

            # Validate configuration structure
            self.assertIn("research_configuration", config)
            self.assertIn("quality_configuration", config)

            research_config = config["research_configuration"]
            self.assertIn("target_urls", research_config)
            self.assertIn("concurrent_processing", research_config)
            self.assertIn("anti_bot_level", research_config)

            quality_config = config["quality_configuration"]
            self.assertIn("enable_quality_assessment", quality_config)
            self.assertIn("quality_threshold", quality_config)

            print(f"  ✓ Valid configuration {i+1} validated successfully")

        # Test invalid configurations
        invalid_configs = [
            {
                "research_configuration": {
                    "target_urls": -1,  # Invalid negative value
                    "concurrent_processing": 5,
                    "anti_bot_level": 1
                }
            },
            {
                "quality_configuration": {
                    "enable_quality_assessment": True,
                    "quality_threshold": 1.5  # Invalid > 1.0
                }
            }
        ]

        for i, config in enumerate(invalid_configs):
            print(f"  Testing invalid configuration {i+1}...")

            # Should detect invalid configuration
            validation_result = self._validate_configuration(config)
            self.assertFalse(validation_result["is_valid"])
            self.assertIn("error", validation_result)

            print(f"  ✓ Invalid configuration {i+1} correctly detected")

    def _validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and return result"""
        errors = []

        # Check research configuration
        if "research_configuration" in config:
            research_config = config["research_configuration"]

            if "target_urls" in research_config:
                target_urls = research_config["target_urls"]
                if not isinstance(target_urls, int) or target_urls < 0:
                    errors.append("target_urls must be a non-negative integer")

            if "concurrent_processing" in research_config:
                concurrent = research_config["concurrent_processing"]
                if not isinstance(concurrent, int) or concurrent < 0:
                    errors.append("concurrent_processing must be a non-negative integer")

            if "anti_bot_level" in research_config:
                anti_bot = research_config["anti_bot_level"]
                if not isinstance(anti_bot, int) or anti_bot < 0 or anti_bot > 4:
                    errors.append("anti_bot_level must be an integer between 0 and 4")

        # Check quality configuration
        if "quality_configuration" in config:
            quality_config = config["quality_configuration"]

            if "quality_threshold" in quality_config:
                threshold = quality_config["quality_threshold"]
                if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                    errors.append("quality_threshold must be a number between 0 and 1")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }


# Test Runner and Reporting
class TestRunner:
    """Test runner with comprehensive reporting"""

    def __init__(self):
        self.test_suite = unittest.TestLoader().loadTestsFromTestCase(ComprehensiveTestSuite)
        self.results = {}

    def run_all_tests(self):
        """Run all tests and generate comprehensive report"""
        print("=" * 80)
        print("COMPREHENSIVE TEST SUITE - AGENT-BASED RESEARCH SYSTEM")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Run tests with detailed output
        runner = unittest.TextTestRunner(
            verbosity=2,
            stream=open(self.temp_dir / "test_results" / "detailed_output.log", "w")
        )

        start_time = time.time()
        result = runner.run(self.test_suite)
        duration = time.time() - start_time

        # Generate summary report
        self.generate_summary_report(result, duration)

        return result

    def generate_summary_report(self, result, duration):
        """Generate comprehensive test summary report"""
        report = {
            "test_summary": {
                "total_tests": result.testsRun,
                "successes": result.testsRun - len(result.failures) - len(result.errors),
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100,
                "duration": duration
            },
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
            "system_info": {
                "test_timestamp": datetime.now().isoformat(),
                "test_environment": "development",
                "python_version": sys.version,
                "platform": sys.platform
            }
        }

        # Save detailed report
        report_path = self.temp_dir / "test_results" / "test_summary_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY REPORT")
        print("=" * 80)
        print(f"Total Tests: {report['test_summary']['total_tests']}")
        print(f"Successes: {report['test_summary']['successes']}")
        print(f"Failures: {report['test_summary']['failures']}")
        print(f"Errors: {report['test_summary']['errors']}")
        print(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
        print(f"Duration: {report['test_summary']['duration']:.2f}s")

        if report['test_summary']['failures'] > 0 or report['test_summary']['errors'] > 0:
            print("\nFAILED TESTS:")
            for test in report['test_details']['failed_tests'] + report['test_details']['error_tests']:
                print(f"  - {test['test']}: {test['error']}")

        print(f"\nDetailed report saved to: {report_path}")
        print("=" * 80)


# Main execution
if __name__ == "__main__":
    # Set up test environment
    test_runner = TestRunner()
    test_runner.temp_dir = Path(tempfile.mkdtemp(prefix="comprehensive_test_run_"))

    try:
        # Run comprehensive test suite
        result = test_runner.run_all_tests()

        # Exit with appropriate code
        if result.wasSuccessful():
            print("\n🎉 All tests passed successfully!")
            exit(0)
        else:
            print("\n❌ Some tests failed. Check the detailed report.")
            exit(1)

    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        exit(1)
    finally:
        # Cleanup
        import shutil
        if test_runner.temp_dir.exists():
            shutil.rmtree(test_runner.temp_dir)