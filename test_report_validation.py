#!/usr/bin/env python3
"""
Test the new Report Validation System

This script tests the hook-enhanced quality validation system to ensure
template detection, data integration assessment, and quality validation work correctly.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the multi-agent system to Python path
sys.path.append(str(Path(__file__).parent))

def test_template_detection():
    """Test template detection functionality."""
    print("\n=== Testing Template Detection ===")

    try:
        from multi_agent_research_system.utils.report_validation import ReportValidationSystem

        validator = ReportValidationSystem()

        # Test template content
        template_content = """
        This report provides a comprehensive overview of artificial intelligence in healthcare.
        In conclusion, AI presents many opportunities for the healthcare industry.
        Further research is needed to understand the full implications.
        This analysis offers a detailed examination of current trends.
        It is worth noting that AI technologies are evolving rapidly.
        """

        result = validator.detect_template_response(template_content)

        print(f"Template Detection Score: {result.template_score}/100")
        print(f"Is Template: {result.is_template}")
        print(f"Template Patterns Found: {len(result.template_patterns)}")
        print(f"Generic Phrases: {len(result.generic_phrases)}")
        print(f"Content Specificity: {result.content_specificity}/100")

        # Should detect as template
        if result.is_template and result.template_score > 60:
            print("✅ Template detection working correctly")
            return True
        else:
            print("❌ Template detection failed")
            return False

    except Exception as e:
        print(f"❌ Template detection test failed: {e}")
        return False

def test_data_integration_assessment():
    """Test data integration assessment."""
    print("\n=== Testing Data Integration Assessment ===")

    try:
        from multi_agent_research_system.utils.report_validation import ReportValidationSystem

        validator = ReportValidationSystem()

        # Test content with good data integration
        good_content = """
        According to McKinsey's 2023 report, AI adoption in healthcare has increased by 45% since 2020.
        The World Health Organization estimates that AI could save $150 billion annually in healthcare costs.
        Research from Johns Hopkins University shows that diagnostic accuracy improved by 35% using AI systems.
        A study published in Nature Medicine (2022) demonstrated that AI reduced drug discovery time by 40%.
        The global healthcare AI market is projected to reach $187.7 billion by 2030, according to Grand View Research.
        """

        result = validator.assess_data_integration(good_content, expected_sources=5)

        print(f"Integration Score: {result.integration_score}/100")
        print(f"Source Count: {result.source_count}")
        print(f"Data Points: {result.data_points_mentioned}")
        print(f"Specific References: {len(result.specific_references)}")
        print(f"Data Quality Indicators: {result.data_quality_indicators}")

        # Should have good integration score
        if result.integration_score > 70:
            print("✅ Data integration assessment working correctly")
            return True
        else:
            print("❌ Data integration assessment failed")
            return False

    except Exception as e:
        print(f"❌ Data integration test failed: {e}")
        return False

def test_comprehensive_validation():
    """Test comprehensive report validation."""
    print("\n=== Testing Comprehensive Validation ===")

    try:
        from multi_agent_research_system.utils.report_validation import ReportValidationSystem

        validator = ReportValidationSystem()

        # Test high-quality content
        good_content = """
        ## AI in Healthcare: Market Analysis and Trends

        ### Current Market Status
        The global healthcare AI market reached $15.1 billion in 2022, representing a 47.2% increase from 2021, according to MarketsandMarkets research [1]. This growth is driven by increasing healthcare costs and the need for improved diagnostic accuracy.

        ### Key Statistics
        - Diagnostic accuracy improvement: 35-40% with AI implementation (Johns Hopkins, 2023)
        - Cost savings potential: $150 billion annually by 2030 (WHO estimates)
        - Market projection: $187.7 billion by 2030 (Grand View Research)
        - Adoption rate: 45% increase since 2020 (McKinsey & Company)

        ### Regional Analysis
        North America currently leads with 52% market share, followed by Europe at 28%. The Asia-Pacific region is expected to grow at the fastest rate (CAGR of 48.3%) through 2030 [2].

        ### Leading Companies
        Key players include IBM Watson Health, Google DeepMind, and Microsoft Healthcare Bot, collectively holding 35% of the market share [3].

        Sources:
        [1] MarketsandMarkets, "Healthcare AI Market Report", 2023
        [2] Deloitte, "Global Healthcare AI Analysis", Q2 2023
        [3] Fortune Business Insights, "AI in Healthcare Competitive Landscape", 2023
        """

        context = {
            "topic": "AI in Healthcare",
            "expected_sources": 3,
            "session_id": "test-validation"
        }

        # Run async test
        import asyncio
        result = asyncio.run(validator.validate_report_quality(good_content, context))

        print(f"Validation Result: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
        print(f"Overall Score: {result.score}/100")
        print(f"Confidence: {result.confidence}")
        print(f"Issues: {len(result.issues)}")
        print(f"Recommendations: {len(result.recommendations)}")
        print(f"Key Metrics: {result.metrics}")

        if result.is_valid and result.score > 70:
            print("✅ Comprehensive validation working correctly")
            return True
        else:
            print("❌ Comprehensive validation failed")
            print(f"Feedback: {result.feedback}")
            return False

    except Exception as e:
        print(f"❌ Comprehensive validation test failed: {e}")
        return False

def test_hook_implementations():
    """Test the hook implementations."""
    print("\n=== Testing Hook Implementations ===")

    try:
        from multi_agent_research_system.utils.report_validation import (
            validate_research_data_usage,
            enforce_citation_requirements,
            validate_report_quality_standards
        )

        # Test citation requirements hook
        content_with_citations = """
        According to research [1], AI adoption has increased significantly.
        Studies from McKinsey [2] show positive trends.
        WHO reports [3] confirm these findings.
        """

        context = {"expected_sources": 3}

        result = enforce_citation_requirements("test-session", content_with_citations, context)

        print(f"Citation Hook Result: {'✅ PASS' if result['success'] else '❌ FAIL'}")
        print(f"Citations Found: {result.get('citations_found', 0)}")

        # Test quality standards hook
        quality_result = validate_report_quality_standards("test-session", content_with_citations, context)

        print(f"Quality Hook Result: {'✅ PASS' if quality_result['success'] else '❌ FAIL'}")
        print(f"Validation Score: {quality_result.get('validation_score', 0)}")

        if result['success'] and quality_result['success']:
            print("✅ Hook implementations working correctly")
            return True
        else:
            print("❌ Hook implementations failed")
            return False

    except Exception as e:
        print(f"❌ Hook implementation test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 Report Validation System Test Suite")
    print("=" * 50)

    # Run all tests
    test_results = []

    test_results.append(test_template_detection())
    test_results.append(test_data_integration_assessment())
    test_results.append(test_comprehensive_validation())
    test_results.append(test_hook_implementations())

    # Generate summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)

    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("The hook-enhanced validation system is working correctly and ready to:")
        print("  - Detect and prevent template responses")
        print("  - Assess data integration quality")
        print("  - Validate report standards through hooks")
        print("  - Provide real-time quality monitoring")
        print("  - Track research pipeline compliance")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        print("Some validation components may need attention.")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)