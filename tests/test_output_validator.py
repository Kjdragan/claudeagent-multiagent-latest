"""
Unit tests for output_validator.py

Tests the OutputValidator class and its three validation methods:
- validate_editorial_output
- validate_report_output  
- validate_final_output
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multi_agent_research_system.core.output_validator import (
    OutputValidator,
    ValidationResult,
    get_output_validator,
)


def test_singleton_pattern():
    """Test that get_output_validator returns singleton instance."""
    validator1 = get_output_validator()
    validator2 = get_output_validator()
    assert validator1 is validator2
    print("✅ Test passed: Singleton pattern works")


def test_valid_critique():
    """Test validation of a proper critique document."""
    validator = OutputValidator()

    content = """
# Editorial Critique
**Session ID**: test_001
**Report**: test_report.md
**Critique Date**: 2025-10-16

## Quality Assessment
### Structure: 0.85/1.00
- Sections: 5
- Word count: 2,500
- Sources: 8

### Overall Quality: 0.72/1.00
- Clarity: 0.80/1.00
- Depth: 0.70/1.00
- Accuracy: 0.68/1.00
- Coherence: 0.75/1.00
- Sourcing: 0.65/1.00

## Identified Issues
1. **Missing Executive Summary**: The report lacks a concise opening summary.
   Example: Report starts directly with "Background" section without overview.
2. **Insufficient Citations**: Only 3 sources cited throughout 2,500 word report.
   Example: Claims about casualty figures on page 2 have no source attribution.

## Information Gaps
### HIGH PRIORITY
**Statistical Gap**: Missing recent casualty statistics
- Recommendation: Add search for "Ukraine war casualties October 2025 figures"

### MEDIUM PRIORITY
**Expert Opinion Gap**: No expert analysis included
- Recommendation: Include expert perspectives

## Recommendations
### Immediate Actions
1. Add executive summary at beginning
2. Include more source citations throughout

### Enhancement Opportunities
- Strengthen analysis sections
- Add more recent data
"""

    result = validator.validate_editorial_output(content, "test_001")

    assert result.is_valid == True, f"Expected valid, got invalid. Issues: {result.issues}"
    assert result.output_type == "critique"
    assert result.score >= 0.75, f"Expected score >= 0.75, got {result.score}"
    assert len(result.issues) == 0

    print(f"✅ Test passed: Valid critique recognized (score: {result.score:.2f})")


def test_invalid_critique_report_content():
    """Test that a report is rejected when validated as critique."""
    validator = OutputValidator()

    content = """
# Russia-Ukraine War Analysis

## Executive Summary
The ongoing conflict between Russia and Ukraine continues to evolve with
significant diplomatic and military developments.

## Key Findings
1. Diplomatic efforts have intensified in recent weeks
2. Military operations continue on multiple fronts
3. Humanitarian concerns remain critical

## Conclusion
The situation requires continued international attention and diplomatic engagement.
"""

    result = validator.validate_editorial_output(content, "test_001")

    assert result.is_valid == False, "Expected invalid (report content), got valid"
    assert result.output_type == "report"
    assert "report sections" in str(result.issues).lower()
    assert result.score < 0.6

    print(f"✅ Test passed: Report content rejected as critique (score: {result.score:.2f})")


def test_valid_report():
    """Test validation of a proper report document."""
    validator = OutputValidator()

    content = """
# Research Report: AI Developments 2025

## Executive Summary
This report analyzes recent developments in artificial intelligence,
focusing on transformer architectures and large language models.

## Introduction
Artificial intelligence has seen rapid advancement in 2025, with
particular breakthroughs in natural language processing and computer vision.

## Key Findings
1. **Transformer Models**: New architectures show 40% improvement in efficiency
2. **Training Methods**: Novel techniques reduce training time by 60%
3. **Applications**: Widespread adoption across healthcare and education

## Analysis
The developments indicate a shift toward more efficient and accessible
AI systems. Multiple sources confirm these trends.

Sources:
- https://example.com/ai-research
- https://example.com/tech-analysis
- Nature Machine Intelligence Journal

## Conclusion
The field of AI continues to advance rapidly with significant implications
for multiple sectors.
"""

    result = validator.validate_report_output(content, "test_001")

    assert result.is_valid == True, f"Expected valid report, got invalid. Issues: {result.issues}"
    assert result.output_type == "report"
    assert result.score >= 0.75

    print(f"✅ Test passed: Valid report recognized (score: {result.score:.2f})")


def test_invalid_report_critique_content():
    """Test that a critique is rejected when validated as report."""
    validator = OutputValidator()

    content = """
# Editorial Critique

## Quality Assessment
### Structure: 0.70/1.00

## Identified Issues
1. **Missing Data**: The report lacks specific statistics

## Recommendations
1. Add more data sources
"""

    result = validator.validate_report_output(content, "test_001")

    assert result.is_valid == False, "Expected invalid (critique content), got valid"
    assert "critique sections" in str(result.issues).lower() or result.score < 0.5

    print(f"✅ Test passed: Critique content rejected as report (score: {result.score:.2f})")


def test_json_metadata_detection():
    """Test detection of JSON metadata corruption in final output."""
    validator = OutputValidator()

    content = """{
    "session_id": "test_001",
    "topic": "AI developments",
    "user_requirements": {
        "depth": "comprehensive",
        "format": "report"
    },
    "created_at": "2025-10-16T23:30:00",
    "status": "completed",
    "research_complete": true
}"""

    result = validator.validate_final_output(content, "test_001")

    assert result.is_valid == False, "Expected invalid (JSON metadata), got valid"
    assert result.output_type == "json_metadata"
    assert "json session metadata" in str(result.issues).lower()

    print(f"✅ Test passed: JSON metadata detected (score: {result.score:.2f})")


def test_valid_final_output():
    """Test validation of proper final narrative output."""
    validator = OutputValidator()

    content = """
# Comprehensive Analysis of AI Developments in 2025

## Overview

The field of artificial intelligence has experienced remarkable growth
throughout 2025, with breakthrough developments in multiple domains.
This comprehensive analysis examines the key trends and their implications.

## Major Developments

### Transformer Architecture Innovations

Recent advancements in transformer models have led to significant improvements
in both efficiency and capability. Research teams at major institutions have
demonstrated 40% gains in computational efficiency while maintaining accuracy.

These improvements stem from novel attention mechanisms that better capture
long-range dependencies in data. Multiple peer-reviewed studies confirm these
findings across diverse benchmarks.

### Training Methodology Advances

New training techniques have dramatically reduced the time and resources
required for model development. Methods such as progressive distillation
and curriculum learning have shown 60% reductions in training time.

Industry adoption of these techniques has accelerated, with major tech
companies reporting successful deployments in production systems.

## Implications and Future Directions

The confluence of architectural and methodological improvements suggests
a transformative period ahead for AI applications. Healthcare and education
sectors are already seeing measurable benefits from these advances.

## Conclusion

The developments of 2025 mark a significant milestone in making AI systems
more efficient, accessible, and effective across diverse applications.
Continued research and responsible deployment will be essential to
realizing the full potential of these technologies.

---

**Sources:**
- MIT AI Lab Research Report 2025
- Nature Machine Intelligence, Vol. 7, 2025  
- Stanford HAI Annual Report
- Industry analysis from TechCrunch and ArXiv
"""

    result = validator.validate_final_output(content, "test_001")

    assert result.is_valid == True, f"Expected valid, got invalid. Issues: {result.issues}"
    assert result.output_type in ["report", "narrative"]
    assert result.score >= 0.75

    print(f"✅ Test passed: Valid final output recognized (score: {result.score:.2f})")


def test_minimal_content_rejection():
    """Test that minimal content is rejected."""
    validator = OutputValidator()

    content = """
# Short Report

This is too brief.
"""

    result = validator.validate_final_output(content, "test_001")

    assert result.is_valid == False, "Expected invalid (too brief), got valid"
    assert "minimum word count" in str(result.issues).lower() or result.score < 0.5

    print(f"✅ Test passed: Minimal content rejected (score: {result.score:.2f})")


def test_validation_result_boolean():
    """Test that ValidationResult can be used in boolean context."""
    result_valid = ValidationResult(
        is_valid=True,
        score=0.90,
        issues=[],
        output_type="critique"
    )

    result_invalid = ValidationResult(
        is_valid=False,
        score=0.40,
        issues=["Test issue"],
        output_type="unknown"
    )

    assert bool(result_valid) == True
    assert bool(result_invalid) == False

    # Test usage in conditionals
    if result_valid:
        validation_passed = True
    else:
        validation_passed = False

    assert validation_passed == True

    print("✅ Test passed: ValidationResult boolean evaluation works")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("OUTPUT VALIDATOR UNIT TESTS")
    print("="*60 + "\n")

    tests = [
        ("Singleton Pattern", test_singleton_pattern),
        ("Valid Critique", test_valid_critique),
        ("Invalid Critique (Report Content)", test_invalid_critique_report_content),
        ("Valid Report", test_valid_report),
        ("Invalid Report (Critique Content)", test_invalid_report_critique_content),
        ("JSON Metadata Detection", test_json_metadata_detection),
        ("Valid Final Output", test_valid_final_output),
        ("Minimal Content Rejection", test_minimal_content_rejection),
        ("ValidationResult Boolean", test_validation_result_boolean),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ Test failed: {test_name}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ Test error: {test_name}")
            print(f"   Exception: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
