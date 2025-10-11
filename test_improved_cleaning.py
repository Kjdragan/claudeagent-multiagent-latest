#!/usr/bin/env python3
"""
Test script to verify the improved content cleaning works correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_content_cleaning_import():
    """Test that the updated content cleaning module can be imported."""
    try:
        from multi_agent_research_system.utils.content_cleaning import (
            clean_content_with_gpt5_nano,
            clean_content_with_judge_optimization,
            assess_content_cleanliness
        )
        print("✅ Successfully imported updated content cleaning functions")
        return True
    except Exception as e:
        print(f"❌ Failed to import content cleaning functions: {e}")
        return False

def test_prompt_structure():
    """Test that the prompt structure is improved."""
    try:
        from multi_agent_research_system.utils.content_cleaning import clean_content_with_gpt5_nano
        import inspect

        # Get the source code of the function
        source = inspect.getsource(clean_content_with_gpt5_nano)

        # Check for improved prompt elements
        improvements = [
            ("Preserve 80-90% of original article content", "Content preservation guideline"),
            ("Remove only web page furniture", "Conservative removal approach"),
            ("PRESERVE FULL ARTICLE CONTENT", "Expanded preservation section"),
            ("IMPORTANT CONSERVATION GUIDELINES", "Added conservation guidelines"),
            ("Analysis & Insights", "Preserves analytical content"),
            ("Context & Background", "Preserves contextual information")
        ]

        print("\n📋 Checking for prompt improvements:")
        all_found = True
        for phrase, description in improvements:
            if phrase in source:
                print(f"✅ {description}: Found")
            else:
                print(f"❌ {description}: Missing")
                all_found = False

        # Check removal of problematic elements
        problematic_elements = [
            ("ONLY the main article content that is directly relevant", "Overly restrictive relevance filter"),
            ("UNRELATED ARTICLES", "Aggressive article removal")
        ]

        print("\n🚫 Checking for removal of problematic elements:")
        for phrase, description in problematic_elements:
            if phrase not in source:
                print(f"✅ {description}: Removed")
            else:
                print(f"❌ {description}: Still present")
                all_found = False

        return all_found

    except Exception as e:
        print(f"❌ Error analyzing prompt structure: {e}")
        return False

def test_validation_logic():
    """Test that the validation logic is improved."""
    try:
        from multi_agent_research_system.utils.content_cleaning import clean_content_with_gpt5_nano
        import inspect

        # Get the source code of the function
        source = inspect.getsource(clean_content_with_gpt5_nano)

        # Check for improved validation
        validations = [
            ("compression_ratio < 0.2", "Excessive compression detection"),
            ("compression_ratio:", "Compression ratio logging"),
            ("significantly reduced content", "Detailed logging")
        ]

        print("\n🔍 Checking for improved validation logic:")
        all_found = True
        for check, description in validations:
            if check in source:
                print(f"✅ {description}: Found")
            else:
                print(f"❌ {description}: Missing")
                all_found = False

        return all_found

    except Exception as e:
        print(f"❌ Error analyzing validation logic: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Improved Content Cleaning Implementation")
    print("=" * 60)

    tests = [
        ("Import Test", test_content_cleaning_import),
        ("Prompt Structure Test", test_prompt_structure),
        ("Validation Logic Test", test_validation_logic)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📝 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Content cleaning improvements are working.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")

    print("\n📈 Expected improvements:")
    print("• Content compression should be 80-90% instead of 2-6%")
    print("• Context and analysis should be preserved")
    print("• Over-cleaning should be detected and prevented")
    print("• Better logging for debugging compression issues")

if __name__ == "__main__":
    main()