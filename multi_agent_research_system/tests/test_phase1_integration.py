#!/usr/bin/env python3
"""
Phase 1: Enhanced Editorial Workflow Component Integration Testing

Tests that all enhanced editorial workflow components can be imported, instantiated,
and have proper methods and configurations without requiring external API dependencies.
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import traceback

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set mock environment variables to bypass configuration validation
os.environ['ANTHROPIC_API_KEY'] = 'mock-key-for-testing'
os.environ['OPENAI_API_KEY'] = 'mock-key-for-testing'
os.environ['SERPER_API_KEY'] = 'mock-key-for-testing'
os.environ['RESEARCH_ENVIRONMENT'] = 'testing'

def print_section(title):
    print(f"\n{'='*60}")
    print(f"Testing: {title}")
    print('='*60)

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def print_warning(message):
    print(f"⚠️  {message}")

async def test_imports():
    """Test that all enhanced editorial workflow components can be imported."""
    print_section("Component Import Testing")

    import_tests = [
        {
            "name": "Enhanced Editorial Decision Engine",
            "module": "multi_agent_research_system.agents.enhanced_editorial_engine",
            "class": "EnhancedEditorialDecisionEngine"
        },
        {
            "name": "Gap Research Decision Engine",
            "module": "multi_agent_research_system.agents.gap_research_decisions",
            "class": "GapResearchDecisionEngine"
        },
        {
            "name": "Research Corpus Analyzer",
            "module": "multi_agent_research_system.agents.research_corpus_analyzer",
            "class": "ResearchCorpusAnalyzer"
        },
        {
            "name": "Editorial Recommendations Engine",
            "module": "multi_agent_research_system.agents.editorial_recommendations",
            "class": "EnhancedRecommendationEngine"
        },
        {
            "name": "Sub-Session Manager",
            "module": "multi_agent_research_system.agents.sub_session_manager",
            "class": "SubSessionManager"
        },
        {
            "name": "Editorial Workflow Integration",
            "module": "multi_agent_research_system.agents.editorial_workflow_integration",
            "class": "EditorialWorkflowIntegrator"
        }
    ]

    successful_imports = 0
    total_imports = len(import_tests)

    for test in import_tests:
        try:
            print_info(f"Importing {test['name']}...")
            module = __import__(test['module'], fromlist=[test['class']])
            cls = getattr(module, test['class'])

            # Verify class is a proper class
            if not isinstance(cls, type):
                print_error(f"{test['name']}: {test['class']} is not a class")
                continue

            print_success(f"{test['name']} imported successfully")
            successful_imports += 1

        except Exception as e:
            print_error(f"{test['name']} import failed: {str(e)}")
            if "ImportError" in str(type(e)):
                print_warning("This might be due to missing dependencies - continue testing")

    print_info(f"Import success rate: {successful_imports}/{total_imports} ({successful_imports/total_imports*100:.1f}%)")
    return successful_imports == total_imports

async def test_component_instantiation():
    """Test that components can be instantiated without API dependencies."""
    print_section("Component Instantiation Testing")

    try:
        # Import all components
        from multi_agent_research_system.agents.enhanced_editorial_engine import EnhancedEditorialDecisionEngine
        from multi_agent_research_system.agents.gap_research_decisions import GapResearchDecisionEngine
        from multi_agent_research_system.agents.research_corpus_analyzer import ResearchCorpusAnalyzer
        from multi_agent_research_system.agents.editorial_recommendations import EnhancedRecommendationEngine
        from multi_agent_research_system.agents.sub_session_manager import SubSessionManager
        from multi_agent_research_system.agents.editorial_workflow_integration import EditorialWorkflowIntegrator

        # Test component instantiation with minimal configuration
        instantiation_tests = [
            {
                "name": "Enhanced Editorial Decision Engine",
                "component": EnhancedEditorialDecisionEngine,
                "config": {"confidence_threshold": 0.7, "max_gap_topics": 2}
            },
            {
                "name": "Gap Research Decision Engine",
                "component": GapResearchDecisionEngine,
                "config": {"cost_benefit_threshold": 1.5, "max_gap_topics": 2}
            },
            {
                "name": "Research Corpus Analyzer",
                "component": ResearchCorpusAnalyzer,
                "config": {"quality_threshold": 0.75, "analysis_depth": "comprehensive"}
            },
            {
                "name": "Editorial Recommendations Engine",
                "component": EnhancedRecommendationEngine,
                "config": {"max_recommendations": 10, "prioritization_strategy": "impact_based"}
            },
            {
                "name": "Sub-Session Manager",
                "component": SubSessionManager,
                "config": {"max_concurrent_sub_sessions": 5, "session_timeout": 3600}
            },
            {
                "name": "Editorial Workflow Integrator",
                "component": EditorialWorkflowIntegrator,
                "config": {"orchestrator_integration": True, "hook_integration": True}
            }
        ]

        successful_instantiations = 0

        for test in instantiation_tests:
            try:
                print_info(f"Instantiating {test['name']}...")

                # Handle different constructor patterns
                if "Manager" in test['name'] or "Integrator" in test['name']:
                    # Some components might expect different config patterns
                    if test['name'] == "Sub-Session Manager":
                        component = test['component']()
                    else:
                        component = test['component']()
                else:
                    component = test['component'](test['config'])

                print_success(f"{test['name']} instantiated successfully")
                successful_instantiations += 1

                # Basic component verification
                if hasattr(component, '__dict__'):
                    print_info(f"  - Component has {len(component.__dict__)} attributes")

            except Exception as e:
                print_error(f"{test['name']} instantiation failed: {str(e)}")
                if "missing" in str(e).lower() or "required" in str(e).lower():
                    print_warning("This might be due to missing required configuration")

        print_info(f"Instantiation success rate: {successful_instantiations}/{len(instantiation_tests)} ({successful_instantiations/len(instantiation_tests)*100:.1f}%)")
        return successful_instantiations == len(instantiation_tests)

    except ImportError as e:
        print_error(f"Component import failed during instantiation test: {str(e)}")
        return False

async def test_component_methods():
    """Test that components have expected methods."""
    print_section("Component Methods Testing")

    try:
        # Import components
        from multi_agent_research_system.agents.enhanced_editorial_engine import EnhancedEditorialDecisionEngine
        from multi_agent_research_system.agents.gap_research_decisions import GapResearchDecisionEngine
        from multi_agent_research_system.agents.research_corpus_analyzer import ResearchCorpusAnalyzer
        from multi_agent_research_system.agents.editorial_recommendations import EnhancedRecommendationEngine
        from multi_agent_research_system.agents.sub_session_manager import SubSessionManager
        from multi_agent_research_system.agents.editorial_workflow_integration import EditorialWorkflowIntegrator

        # Expected methods for each component
        method_tests = [
            {
                "name": "Enhanced Editorial Decision Engine",
                "component": EnhancedEditorialDecisionEngine,
                "config": {"confidence_threshold": 0.7},
                "expected_methods": [
                    "analyze_editorial_decision",
                    "calculate_confidence_scores",
                    "assess_gap_research_necessity"
                ]
            },
            {
                "name": "Gap Research Decision Engine",
                "component": GapResearchDecisionEngine,
                "config": {"cost_benefit_threshold": 1.5},
                "expected_methods": [
                    "analyze_gap_research_decisions",
                    "prioritize_gaps",
                    "calculate_cost_benefit_ratio"
                ]
            },
            {
                "name": "Research Corpus Analyzer",
                "component": ResearchCorpusAnalyzer,
                "config": {"quality_threshold": 0.75},
                "expected_methods": [
                    "analyze_research_corpus",
                    "assess_quality_dimensions",
                    "determine_sufficiency"
                ]
            },
            {
                "name": "Editorial Recommendations Engine",
                "component": EnhancedRecommendationEngine,
                "config": {"max_recommendations": 10},
                "expected_methods": [
                    "generate_editorial_recommendations",
                    "prioritize_recommendations",
                    "create_implementation_plan"
                ]
            },
            {
                "name": "Sub-Session Manager",
                "component": SubSessionManager,
                "config": {},
                "expected_methods": [
                    "create_sub_session",
                    "coordinate_gap_research",
                    "integrate_sub_session_results"
                ]
            },
            {
                "name": "Editorial Workflow Integrator",
                "component": EditorialWorkflowIntegrator,
                "config": {},
                "expected_methods": [
                    "initialize_editorial_workflow",
                    "coordinate_editorial_execution",
                    "integrate_with_orchestrator"
                ]
            }
        ]

        successful_method_tests = 0

        for test in method_tests:
            try:
                print_info(f"Testing methods for {test['name']}...")

                # Instantiate component
                if "Manager" in test['name'] or "Integrator" in test['name']:
                    component = test['component']()
                else:
                    component = test['component'](test['config'])

                # Check for expected methods
                found_methods = 0
                for method_name in test['expected_methods']:
                    if hasattr(component, method_name):
                        method = getattr(component, method_name)
                        if callable(method):
                            found_methods += 1
                        else:
                            print_warning(f"  - {method_name} exists but is not callable")
                    else:
                        print_warning(f"  - {method_name} not found")

                method_ratio = found_methods / len(test['expected_methods'])
                if method_ratio >= 0.8:  # 80% of expected methods found
                    print_success(f"{test['name']}: {found_methods}/{len(test['expected_methods'])} methods found")
                    successful_method_tests += 1
                else:
                    print_error(f"{test['name']}: Only {found_methods}/{len(test['expected_methods'])} methods found")

            except Exception as e:
                print_error(f"{test['name']} method testing failed: {str(e)}")

        print_info(f"Method testing success rate: {successful_method_tests}/{len(method_tests)} ({successful_method_tests/len(method_tests)*100:.1f}%)")
        return successful_method_tests == len(method_tests)

    except ImportError as e:
        print_error(f"Component import failed during method testing: {str(e)}")
        return False

async def test_data_structures():
    """Test that data structures can be created."""
    print_section("Data Structure Testing")

    try:
        # Test data structure imports and creation
        from multi_agent_research_system.agents.enhanced_editorial_engine import (
            ConfidenceScore, GapAnalysis, CorpusAnalysisResult, EnhancedEditorialDecision as EditorialDecision
        )
        from multi_agent_research_system.agents.gap_research_decisions import (
            GapResearchDecision, ResourceRequirements
        )
        from multi_agent_research_system.agents.research_corpus_analyzer import (
            ResearchCorpusAnalysis, CoverageAnalysis
        )
        from multi_agent_research_system.agents.editorial_recommendations import (
            EditorialRecommendation, EditorialRecommendationsPlan
        )
        from multi_agent_research_system.agents.sub_session_manager import (
            SubSession, SessionHierarchy
        )

        # Test data structure creation
        structure_tests = [
            {
                "name": "ConfidenceScore",
                "structure": ConfidenceScore,
                "data": {
                    "overall_confidence": 0.8,
                    "research_quality_confidence": 0.75,
                    "content_completeness_confidence": 0.85,
                    "source_credibility_confidence": 0.9,
                    "analytical_depth_confidence": 0.7,
                    "temporal_relevance_confidence": 0.8
                }
            },
            {
                "name": "GapAnalysis",
                "structure": GapAnalysis,
                "data": {
                    "gap_category": "factual_gaps",
                    "gap_description": "Missing recent statistics",
                    "importance_score": 0.8,
                    "urgency_score": 0.7,
                    "feasibility_score": 0.9,
                    "confidence_in_gap": 0.85,
                    "confidence_in_solution": 0.8,
                    "expected_research_success": 0.9,
                    "recommended_action": "Conduct targeted research",
                    "priority_level": "HIGH",
                    "estimated_research_effort": "Medium"
                }
            },
            {
                "name": "ResourceRequirements",
                "structure": ResourceRequirements,
                "data": {
                    "estimated_scrapes_needed": 5,
                    "estimated_queries_needed": 3,
                    "time_requirement": timedelta(minutes=30),
                    "budget_requirement": 50.0,
                    "complexity_level": 3
                }
            }
        ]

        successful_structures = 0

        for test in structure_tests:
            try:
                print_info(f"Testing {test['name']} creation...")

                # Handle enums and special types
                if test['name'] == 'GapAnalysis':
                    # Import required enums
                    from multi_agent_research_system.agents.enhanced_editorial_engine import GapCategory, LegacyPriorityLevel
                    data = test['data'].copy()
                    data['gap_category'] = GapCategory.FACTUAL_GAPS
                    data['priority_level'] = LegacyPriorityLevel.HIGH
                    instance = test['structure'](**data)
                else:
                    instance = test['structure'](**test['data'])

                print_success(f"{test['name']} created successfully")
                successful_structures += 1

                # Verify structure attributes
                if hasattr(instance, '__dict__'):
                    print_info(f"  - Structure has {len(instance.__dict__)} attributes")

            except Exception as e:
                print_error(f"{test['name']} creation failed: {str(e)}")
                # Print more details for debugging
                print_warning(f"  Data provided: {test['data']}")

        print_info(f"Data structure success rate: {successful_structures}/{len(structure_tests)} ({successful_structures/len(structure_tests)*100:.1f}%)")
        return successful_structures == len(structure_tests)

    except ImportError as e:
        print_error(f"Data structure import failed: {str(e)}")
        return False

async def test_configurations():
    """Test that component configurations are valid."""
    print_section("Configuration Testing")

    try:
        # Test configuration validation
        from multi_agent_research_system.agents.enhanced_editorial_engine import EnhancedEditorialEngineConfig
        from multi_agent_research_system.agents.gap_research_decisions import GapDecisionConfig
        from multi_agent_research_system.agents.research_corpus_analyzer import CorpusAnalyzerConfig
        from multi_agent_research_system.agents.editorial_recommendations import RecommendationsConfig
        from multi_agent_research_system.agents.sub_session_manager import SubSessionManagerConfig
        from multi_agent_research_system.agents.editorial_workflow_integration import IntegrationConfig

        # Test configuration creation and validation
        config_tests = [
            {
                "name": "Enhanced Editorial Engine Config",
                "config": EnhancedEditorialEngineConfig,
                "data": {
                    "confidence_threshold": 0.7,
                    "max_gap_topics": 2,
                    "quality_threshold": 0.75
                }
            },
            {
                "name": "Gap Decision Config",
                "config": GapDecisionConfig,
                "data": {
                    "cost_benefit_threshold": 1.5,
                    "max_gap_topics": 2,
                    "confidence_threshold": 0.7
                }
            },
            {
                "name": "Corpus Analyzer Config",
                "config": CorpusAnalyzerConfig,
                "data": {
                    "quality_threshold": 0.75,
                    "analysis_depth": "comprehensive",
                    "sufficiency_threshold": 0.8
                }
            }
        ]

        successful_configs = 0

        for test in config_tests:
            try:
                print_info(f"Testing {test['name']}...")

                config = test['config'](**test['data'])

                print_success(f"{test['name']} created successfully")
                successful_configs += 1

                # Validate configuration values
                if hasattr(config, '__dict__'):
                    for attr, value in config.__dict__.items():
                        if value is not None:
                            print_info(f"  - {attr}: {value}")

            except Exception as e:
                print_error(f"{test['name']} configuration failed: {str(e)}")

        print_info(f"Configuration success rate: {successful_configs}/{len(config_tests)} ({successful_configs/len(config_tests)*100:.1f}%)")
        return successful_configs == len(config_tests)

    except ImportError as e:
        print_error(f"Configuration import failed: {str(e)}")
        return False

async def main():
    """Run all Phase 1 component integration tests."""
    print_section("Enhanced Editorial Workflow - Phase 1 Component Integration Testing")
    print_info("Testing enhanced editorial workflow components without external API dependencies")
    print_info(f"Started at: {datetime.now().isoformat()}")

    # Run all test phases
    test_results = []

    # Test 1: Import components
    test_results.append(("Component Imports", await test_imports()))

    # Test 2: Component instantiation
    test_results.append(("Component Instantiation", await test_component_instantiation()))

    # Test 3: Component methods
    test_results.append(("Component Methods", await test_component_methods()))

    # Test 4: Data structures
    test_results.append(("Data Structures", await test_data_structures()))

    # Test 5: Configurations
    test_results.append(("Configurations", await test_configurations()))

    # Generate summary report
    print_section("Phase 1 Testing Summary")

    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    success_rate = passed_tests / total_tests
    print(f"\nOverall Success Rate: {passed_tests}/{total_tests} ({success_rate*100:.1f}%)")

    if success_rate >= 0.8:
        print_success("Phase 1 Component Integration Testing: SUCCESS")
        print_info("Enhanced editorial workflow components are ready for Phase 2 testing")
        return True
    else:
        print_error("Phase 1 Component Integration Testing: NEEDS ATTENTION")
        print_warning("Some components have issues that should be resolved before Phase 2")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)