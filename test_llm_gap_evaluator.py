#!/usr/bin/env python3
"""
Test script for LLM Gap Research Evaluator

This script tests the LLM-based gap research evaluation with various scenarios
to validate it works correctly and makes proper decisions.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multi_agent_research_system.utils.llm_gap_research_evaluator import (
    LLMGapResearchEvaluator,
    GapResearchEvaluation,
    create_gap_research_evaluator,
    evaluate_gap_research_need
)
from multi_agent_research_system.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_llm_gap_evaluator():
    """Test the LLM gap research evaluator with various scenarios."""

    print("🧪 Testing LLM Gap Research Evaluator")
    print("=" * 50)

    # Get settings
    settings = get_settings()

    # Create test workproducts
    test_scenarios = [
        {
            "name": "Sufficient Research - Good Coverage",
            "content": """# Enhanced Search+Crawl+Clean Workproduct

**Session ID**: test-session-001
**Export Date**: 2025-10-16T15:30:00Z
**Agent**: Enhanced Search+Crawl Tool
**Search Query**: artificial intelligence trends 2024
**Total Search Results**: 20
**Successfully Crawled**: 15

## 🔍 Search Results Summary

### 1. The Rise of Generative AI in 2024
**URL**: https://www.technologyreview.com/2024/01/generative-ai-trends/
**Source**: MIT Technology Review
**Date**: 2025-01-15
**Relevance Score**: 0.89

**Snippet**: Generative AI has transformed the technology landscape in 2024, with major advances in language models, image generation, and multimodal systems. The article covers key trends including GPT-5 development, open-source alternatives, and enterprise adoption.

### 2. AI Regulation and Policy Updates
**URL**: https://www.techcrunch.com/2024/02/ai-regulation-2024/
**Source**: TechCrunch
**Date**: 2025-02-10
**Relevance Score**: 0.85

**Snippet**: 2024 has seen significant developments in AI regulation worldwide. The EU AI Act implementation, US executive orders on AI safety, and China's AI governance framework are shaping the industry's future.

### 3. Enterprise AI Adoption Patterns
**URL**: https://hbr.org/2024/03/enterprise-ai-adoption-trends/
**Source**: Harvard Business Review
**Date**: 2025-03-20
**Relevance Score**: 0.91

**Snippet**: Enterprise adoption of AI has accelerated in 2024, with 75% of Fortune 500 companies implementing AI solutions. Key trends include focus on ROI, ethical AI frameworks, and custom model development.

---
""",
            "expected_decision": "SUFFICIENT"
        },
        {
            "name": "Insufficient Research - Major Gaps",
            "content": """# Enhanced Search+Crawl+Clean Workproduct

**Session ID**: test-session-002
**Export Date**: 2025-10-16T15:35:00Z
**Agent**: Enhanced Search+Crawl Tool
**Search Query**: quantum computing applications healthcare
**Total Search Results**: 5
**Successfully Crawled**: 2

## 🔍 Search Results Summary

### 1. Basic Quantum Computing Overview
**URL**: https://www.example.com/quantum-computing-basics/
**Source**: Generic Tech Blog
**Date**: 2023-12-01
**Relevance Score**: 0.45

**Snippet**: Quantum computing is a new technology that uses quantum bits instead of classical bits. It has potential applications in various fields.

### 2. Healthcare Technology News
**URL**: https://www.healthnews.com/tech-updates/
**Source**: Health News
**Date**: 2023-11-15
**Relevance Score**: 0.32

**Snippet**: Technology continues to impact healthcare with new developments in electronic health records and telemedicine.

---
""",
            "expected_decision": "MORE_RESEARCH_NEEDED"
        }
    ]

    # Test different strictness levels
    strictness_levels = ["lenient", "standard", "strict"]

    for strictness in strictness_levels:
        print(f"\n🔧 Testing with '{strictness}' strictness")
        print("-" * 40)

        # Create evaluator
        evaluator = create_gap_research_evaluator(
            model="gpt-5-nano",
            strictness=strictness
        )

        for scenario in test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            print(f"Expected: {scenario['expected_decision']}")

            try:
                # Create temporary workproduct file
                workproduct_path = f"temp_test_workproduct_{scenario['name'].replace(' ', '_').lower()}.md"
                with open(workproduct_path, 'w', encoding='utf-8') as f:
                    f.write(scenario['content'])

                # Evaluate
                result = await evaluator.evaluate_search_workproduct(
                    session_id=f"test-session-{strictness}",
                    workproduct_path=workproduct_path
                )

                print(f"Decision: {result.decision}")
                print(f"Reasoning: {result.reasoning}")
                print(f"Confidence: {result.confidence:.2f}")
                if result.suggested_queries:
                    print(f"Suggested Queries: {', '.join(result.suggested_queries)}")

                # Check if decision matches expectation (for standard strictness)
                if strictness == "standard":
                    if result.decision == scenario['expected_decision']:
                        print("✅ Decision matches expectation")
                    else:
                        print("❌ Decision doesn't match expectation")

                # Clean up
                os.remove(workproduct_path)

            except Exception as e:
                print(f"❌ Error testing scenario: {e}")

        print("\n" + "=" * 50)

async def test_error_handling():
    """Test error handling and edge cases."""

    print("\n🧪 Testing Error Handling and Edge Cases")
    print("=" * 50)

    evaluator = create_gap_research_evaluator()

    # Test 1: Non-existent file
    print("\n📋 Test 1: Non-existent file")
    try:
        result = await evaluator.evaluate_search_workproduct(
            session_id="test-error-1",
            workproduct_path="non_existent_file.md"
        )
        print(f"Decision: {result.decision} (should be SUFFICIENT)")
        print(f"Reasoning: {result.reasoning}")
        print("✅ Handled missing file gracefully")
    except Exception as e:
        print(f"❌ Error handling non-existent file: {e}")

    # Test 2: Empty file
    print("\n📋 Test 2: Empty file")
    empty_file = "temp_empty_test.md"
    with open(empty_file, 'w', encoding='utf-8') as f:
        f.write("")

    try:
        result = await evaluator.evaluate_search_workproduct(
            session_id="test-error-2",
            workproduct_path=empty_file
        )
        print(f"Decision: {result.decision}")
        print(f"Reasoning: {result.reasoning}")
        print("✅ Handled empty file gracefully")
    except Exception as e:
        print(f"❌ Error handling empty file: {e}")
    finally:
        os.remove(empty_file)

    # Test 3: Invalid session (no workproduct found)
    print("\n📋 Test 3: Invalid session")
    try:
        result = await evaluator.evaluate_search_workproduct(
            session_id="non-existent-session-12345"
        )
        print(f"Decision: {result.decision} (should be SUFFICIENT)")
        print(f"Reasoning: {result.reasoning}")
        print("✅ Handled invalid session gracefully")
    except Exception as e:
        print(f"❌ Error handling invalid session: {e}")

async def test_configuration():
    """Test configuration settings."""

    print("\n🧪 Testing Configuration Settings")
    print("=" * 50)

    settings = get_settings()

    # Test default settings
    print(f"\n📋 Default Configuration:")
    print(f"LLM Gap Research Enabled: {settings.llm_gap_research_enabled}")
    print(f"LLM Gap Research Model: {settings.llm_gap_research_model}")
    print(f"LLM Gap Research Strictness: {settings.llm_gap_research_strictness}")
    print(f"LLM Gap Research Timeout: {settings.llm_gap_research_timeout}")
    print(f"LLM Gap Research Max Tokens: {settings.llm_gap_research_max_tokens}")

    # Test creating evaluator with settings
    evaluator = LLMGapResearchEvaluator(
        model=settings.llm_gap_research_model,
        strictness=settings.llm_gap_research_strictness
    )

    print(f"✅ Evaluator created with settings")
    print(f"Model: {evaluator.model}")
    print(f"Strictness: {evaluator.strictness}")

async def main():
    """Main test function."""

    print("🚀 Starting LLM Gap Research Evaluator Tests")
    print("=" * 60)

    try:
        # Test configuration
        await test_configuration()

        # Test LLM evaluator
        await test_llm_gap_evaluator()

        # Test error handling
        await test_error_handling()

        print("\n🎉 All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        logger.exception("Test suite failed")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)