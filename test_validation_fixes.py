#!/usr/bin/env python3
"""
Test script to verify validation and finalization fixes.

Run this to confirm:
1. Execution tracker works
2. Validation uses tracker
3. Finalization creates /final/ deliverables
4. No false negative validations

Usage:
    python test_validation_fixes.py
"""

import sys
from pathlib import Path

# Add system path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

def test_execution_tracker():
    """Test 1: Verify execution tracker is available and functional."""
    print("\n" + "="*80)
    print("TEST 1: Execution Tracker")
    print("="*80)
    
    try:
        from multi_agent_research_system.core.tool_execution_tracker import get_tool_execution_tracker
        tracker = get_tool_execution_tracker()
        
        # Test tracking a tool
        tracker.track_tool_start(
            tool_name="test_tool",
            tool_use_id="test_001",
            session_id="test_session",
            input_data={"test": "data"}
        )
        
        # Test tracking completion
        tracker.track_tool_completion(
            tool_use_id="test_001",
            result_data={"result": "success"},
            success=True
        )
        
        # Test validation
        successful = tracker.get_successful_tools("test_session")
        assert "test_tool" in successful, "Tool should be in successful list"
        
        # Test validation method
        validation = tracker.validate_required_tools(["test_tool"], "test_session")
        assert validation["valid"], "Validation should pass"
        assert len(validation["missing_tools"]) == 0, "Should have no missing tools"
        
        # Test summary
        summary = tracker.get_session_tool_summary("test_session")
        assert summary["successful"] == 1, "Should have 1 successful execution"
        
        print("✅ Execution tracker works correctly")
        print(f"   - Tool tracked: test_tool")
        print(f"   - Validation: {validation}")
        print(f"   - Summary: {summary}")
        return True
        
    except Exception as e:
        print(f"❌ Execution tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_integration():
    """Test 2: Verify orchestrator validation uses tracker."""
    print("\n" + "="*80)
    print("TEST 2: Validation Integration")
    print("="*80)
    
    try:
        from multi_agent_research_system.core.tool_execution_tracker import get_tool_execution_tracker
        
        # Simulate report validation scenario
        tracker = get_tool_execution_tracker()
        test_session = "validation_test"
        
        # Track some tools
        tools = [
            ("mcp__workproduct__get_all_workproduct_articles", "wp_001"),
            ("mcp__research_tools__create_research_report", "report_001")
        ]
        
        for tool_name, tool_id in tools:
            tracker.track_tool_start(
                tool_name=tool_name,
                tool_use_id=tool_id,
                session_id=test_session,
                input_data={}
            )
            tracker.track_tool_completion(
                tool_use_id=tool_id,
                result_data={},
                success=True
            )
        
        # Test validation (should work with substring matching)
        required_tools = ["workproduct", "create_research_report"]
        validation = tracker.validate_required_tools(required_tools, test_session, match_substring=True)
        
        assert validation["valid"], "Validation should pass with substring matching"
        assert len(validation["missing_tools"]) == 0, "Should find all tools despite mcp__ prefix"
        
        print("✅ Validation works with tool prefixes")
        print(f"   - Tools tracked: {[t[0] for t in tools]}")
        print(f"   - Required: {required_tools}")
        print(f"   - Validation result: {validation['valid']}")
        print(f"   - Found: {list(validation['found_tools'].keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Validation integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_finalization_logic():
    """Test 3: Verify finalization stage logic (without running full workflow)."""
    print("\n" + "="*80)
    print("TEST 3: Finalization Logic")
    print("="*80)
    
    try:
        from multi_agent_research_system.core.orchestrator import ResearchOrchestrator
        
        # Just verify the methods exist
        orchestrator = ResearchOrchestrator()
        
        assert hasattr(orchestrator, '_execute_finalization_stage'), \
            "Orchestrator should have _execute_finalization_stage method"
        
        assert hasattr(orchestrator, '_extract_executive_summary'), \
            "Orchestrator should have _extract_executive_summary method"
        
        # Test executive summary extraction
        test_content = """
# Report Title

## Executive Summary

This is a test executive summary.
It spans multiple lines.

## Introduction

This should not be included.
"""
        summary = orchestrator._extract_executive_summary(test_content)
        assert "test executive summary" in summary.lower(), "Should extract executive summary"
        assert "introduction" not in summary.lower(), "Should not include other sections"
        
        print("✅ Finalization logic is implemented")
        print(f"   - _execute_finalization_stage: exists")
        print(f"   - _extract_executive_summary: works")
        print(f"   - Summary extraction test: passed")
        return True
        
    except Exception as e:
        print(f"❌ Finalization logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_import_paths():
    """Test 4: Verify all modules can be imported."""
    print("\n" + "="*80)
    print("TEST 4: Module Imports")
    print("="*80)
    
    modules = [
        "multi_agent_research_system.core.tool_execution_tracker",
        "multi_agent_research_system.core.orchestrator",
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("VALIDATION & FINALIZATION FIXES - TEST SUITE")
    print("="*80)
    print("\nThis test suite verifies the fixes from doc #54")
    print("Tests will NOT run a full workflow (that requires API keys)")
    print("Instead, we test the core logic and integration points")
    
    results = {
        "Import Paths": test_import_paths(),
        "Execution Tracker": test_execution_tracker(),
        "Validation Integration": test_validation_integration(),
        "Finalization Logic": test_finalization_logic(),
    }
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nNext steps:")
        print("1. Review doc #54 for implementation details")
        print("2. Run a full workflow test: uv run run_research.py \"test topic\"")
        print("3. Check /final/ directory for deliverables")
        print("4. Verify no 'workflow incomplete' warnings in logs")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the errors above and check:")
        print("1. All files were saved correctly")
        print("2. Import paths are correct")
        print("3. No syntax errors in modified files")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
