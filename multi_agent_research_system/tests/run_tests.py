#!/usr/bin/env python3
"""
Test runner script for the multi-agent research system.

This script provides different testing modes:
- unit: Run unit tests only (fast, no external dependencies)
- integration: Run integration tests (component interaction)
- functional: Run functional tests with real API calls (requires credentials)
- all: Run all tests (default)
- quick: Run only fast unit and integration tests
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the result."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED (return code: {result.returncode})")

        return result.returncode == 0

    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def check_dependencies():
    """Check if testing dependencies are available."""
    print("🔍 Checking test dependencies...")

    required_packages = [
        "pytest",
        "pytest-asyncio",
        "pytest-mock"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - MISSING")

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("✅ All dependencies available")
    return True


def check_api_credentials():
    """Check if API credentials are available for functional tests."""
    print("🔑 Checking API credentials...")

    # Check for Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print("  ✅ ANTHROPIC_API_KEY found")
        return True
    else:
        print("  ❌ ANTHROPIC_API_KEY not found")
        print("  Set with: export ANTHROPIC_API_KEY='your-api-key'")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test runner for multi-agent research system")
    parser.add_argument(
        "mode",
        choices=["unit", "integration", "functional", "all", "quick"],
        default="all",
        help="Testing mode (default: all)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--skip-functional",
        action="store_true",
        help="Skip functional tests even in 'all' mode"
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML test report"
    )

    args = parser.parse_args()

    print("🚀 Multi-Agent Research System Test Runner")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Determine which tests to run
    tests_to_run = []

    if args.mode == "unit":
        tests_to_run = [("unit", "Unit Tests", "unit/", "-m unit")]
    elif args.mode == "integration":
        tests_to_run = [("integration", "Integration Tests", "integration/", "-m integration")]
    elif args.mode == "functional":
        if not check_api_credentials():
            print("❌ API credentials required for functional tests")
            sys.exit(1)
        tests_to_run = [("functional", "Functional Tests", "functional/", "-m functional -m slow")]
    elif args.mode == "quick":
        tests_to_run = [
            ("unit", "Unit Tests", "unit/", "-m unit"),
            ("integration", "Integration Tests", "integration/", "-m integration")
        ]
    elif args.mode == "all":
        tests_to_run = [
            ("unit", "Unit Tests", "unit/", "-m unit"),
            ("integration", "Integration Tests", "integration/", "-m integration")
        ]

        if not args.skip_functional:
            if check_api_credentials():
                tests_to_run.append(("functional", "Functional Tests", "functional/", "-m functional -m slow"))
            else:
                print("⚠️  Skipping functional tests (no API credentials)")

    # Prepare pytest command base
    pytest_base = ["python3", "-m", "pytest", "-v"]

    if args.verbose:
        pytest_base.extend(["-v", "-s"])

    if args.html_report:
        pytest_base.extend(["--html=test_report.html", "--self-contained-html"])

    # Run tests
    results = {}
    all_passed = True

    for test_type, description, path, markers in tests_to_run:
        cmd = pytest_base + [path] + markers.split()
        success = run_command(cmd, description)
        results[test_type] = success
        if not success:
            all_passed = False

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    for test_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_type.upper():15} - {status}")

    print(f"{'='*60}")

    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The multi-agent research system is ready for use!")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Please review the failures and fix the issues")

    # Additional information
    print("\n📋 Test Categories:")
    print("  🔬 Unit Tests: Test individual components in isolation")
    print("  🔗 Integration Tests: Test component interactions")
    print("  ⚡ Functional Tests: Test real API calls and workflows")

    if not args.skip_functional and not check_api_credentials():
        print("\n💡 To run functional tests:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        print("  python run_tests.py functional")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
