"""
Standalone unit tests for output_validator.py
Tests without full package dependencies
"""

import os
import sys

# Add parent directory to path to import the module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multi_agent_research_system', 'core'))

# Direct import of output_validator module
import output_validator

OutputValidator = output_validator.OutputValidator
ValidationResult = output_validator.ValidationResult
get_output_validator = output_validator.get_output_validator


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

## Identified Issues
1. **Missing Executive Summary**: The report lacks a concise opening summary.
   Example: Report starts directly with "Background" section without overview.
2. **Insufficient Citations**: Only 3 sources cited throughout 2,500 word report.

## Information Gaps
### HIGH PRIORITY
**Statistical Gap**: Missing recent casualty statistics
- Recommendation: Add search for "Ukraine war casualties October 2025 figures"

## Recommendations
### Immediate Actions
1. Add executive summary at beginning
2. Include more source citations throughout
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

## Conclusion
The situation requires continued international attention.
"""

    result = validator.validate_editorial_output(content, "test_001")

    assert result.is_valid == False, "Expected invalid (report content), got valid"
    assert result.output_type == "report"
    assert "report sections" in str(result.issues).lower()
    assert result.score < 0.6

    print(f"✅ Test passed: Report content rejected as critique (score: {result.score:.2f})")


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
"""

    result = validator.validate_final_output(content, "test_001")

    assert result.is_valid == True, f"Expected valid, got invalid. Issues: {result.issues}"
    assert result.output_type in ["report", "narrative"]
    assert result.score >= 0.75

    print(f"✅ Test passed: Valid final output recognized (score: {result.score:.2f})")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("OUTPUT VALIDATOR UNIT TESTS (Standalone)")
    print("="*60 + "\n")

    tests = [
        ("Singleton Pattern", test_singleton_pattern),
        ("Valid Critique", test_valid_critique),
        ("Invalid Critique (Report Content)", test_invalid_critique_report_content),
        ("JSON Metadata Detection", test_json_metadata_detection),
        ("Valid Final Output", test_valid_final_output),
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
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
