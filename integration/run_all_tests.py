#!/usr/bin/env python3
"""
Comprehensive Test Runner for Agent-Based Research System

This script executes all test suites and provides comprehensive reporting
and analysis of test results.

Usage:
    python integration/run_all_tests.py [options]

Options:
    --verbose, -v          Enable verbose output
    --parallel, -p         Run tests in parallel
    --coverage, -c         Generate coverage report
    --performance-only     Run only performance tests
    --integration-only     Run only integration tests
    --error-only           Run only error scenario tests
    --comprehensive-only   Run only comprehensive tests
    --output-dir DIR       Specify output directory for reports
    --timeout SECONDS      Set test timeout (default: 300)
    --debug                Enable debug mode
    --help, -h             Show this help message

Author: Claude Code Assistant
Version: 1.0.0
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import threading
import queue
import unittest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestResult:
    """Container for test execution results"""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = None
        self.end_time = None
        self.duration = 0.0
        self.success = False
        self.exit_code = 1
        self.output = ""
        self.error_output = ""
        self.exception = None
        self.metrics = {}

    def start(self):
        """Mark test start"""
        self.start_time = time.time()

    def finish(self, success: bool, exit_code: int = 0, output: str = "", error_output: str = "", exception: Optional[Exception] = None):
        """Mark test completion"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time if self.start_time else 0
        self.success = success
        self.exit_code = exit_code
        self.output = output
        self.error_output = error_output
        self.exception = exception

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "test_name": self.test_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "success": self.success,
            "exit_code": self.exit_code,
            "output_length": len(self.output),
            "error_output_length": len(self.error_output),
            "has_exception": self.exception is not None,
            "metrics": self.metrics
        }


class TestSuite:
    """Represents a test suite with execution capabilities"""

    def __init__(self, name: str, script_path: str, description: str = ""):
        self.name = name
        self.script_path = script_path
        self.description = description
        self.result = TestResult(name)

    async def execute(self, timeout: int = 300, verbose: bool = False, debug: bool = False) -> TestResult:
        """Execute the test suite"""
        self.result.start()

        try:
            # Prepare command
            cmd = [sys.executable, str(self.script_path)]

            if verbose:
                cmd.append("-v")

            if debug:
                cmd.append("--debug")

            # Execute test
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.script_path.parent)
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                success = process.returncode == 0
                output = stdout.decode('utf-8', errors='replace')
                error_output = stderr.decode('utf-8', errors='replace')

                self.result.finish(success, process.returncode, output, error_output)

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                self.result.finish(False, -1, "", "Test execution timed out")

        except Exception as e:
            self.result.finish(False, -1, "", str(e), e)

        return self.result


class TestRunner:
    """Main test runner for all test suites"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        self.test_suites = self._initialize_test_suites()
        self.results = []
        self.start_time = None
        self.end_time = None

    def _initialize_test_suites(self) -> List[TestSuite]:
        """Initialize all test suites"""
        integration_dir = Path(__file__).parent

        return [
            TestSuite(
                "Comprehensive Tests",
                integration_dir / "comprehensive_test_suite.py",
                "End-to-end workflow validation and system integration testing"
            ),
            TestSuite(
                "Integration Tests",
                integration_dir / "integration_tests.py",
                "Component integration and coordination testing"
            ),
            TestSuite(
                "Performance Tests",
                integration_dir / "performance_tests.py",
                "System performance under various loads and conditions"
            ),
            TestSuite(
                "Error Scenario Tests",
                integration_dir / "error_scenario_tests.py",
                "Error handling and recovery mechanism testing"
            )
        ]

    async def run_all_tests(
        self,
        timeout: int = 300,
        parallel: bool = False,
        verbose: bool = False,
        debug: bool = False,
        filter_suites: List[str] = None
    ) -> Dict[str, Any]:
        """Run all test suites"""
        self.start_time = time.time()

        # Filter test suites if specified
        suites_to_run = self.test_suites
        if filter_suites:
            suites_to_run = [
                suite for suite in self.test_suites
                if suite.name.lower() in [f.lower() for f in filter_suites]
            ]

        print(f"🚀 Starting Agent-Based Research System Test Suite")
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔧 Parallel execution: {parallel}")
        print(f"⏱️  Timeout: {timeout} seconds per suite")
        print(f"📊 Suites to run: {len(suites_to_run)}")
        print()

        if parallel:
            await self._run_tests_parallel(suites_to_run, timeout, verbose, debug)
        else:
            await self._run_tests_sequential(suites_to_run, timeout, verbose, debug)

        self.end_time = time.time()

        # Generate comprehensive report
        report = self._generate_comprehensive_report()

        # Save individual test results
        self._save_individual_results()

        return report

    async def _run_tests_sequential(
        self,
        suites: List[TestSuite],
        timeout: int,
        verbose: bool,
        debug: bool
    ):
        """Run tests sequentially"""
        for suite in suites:
            print(f"🔄 Running {suite.name}...")
            print(f"   {suite.description}")

            result = await suite.execute(timeout, verbose, debug)
            self.results.append(result)

            status = "✅ PASSED" if result.success else "❌ FAILED"
            duration_str = f"{result.duration:.2f}s"
            print(f"   {status} ({duration_str})")

            if not result.success:
                print(f"   Error: {result.error_output[:200]}...")

            print()

    async def _run_tests_parallel(
        self,
        suites: List[TestSuite],
        timeout: int,
        verbose: bool,
        debug: bool
    ):
        """Run tests in parallel"""
        print("🔄 Running tests in parallel...")

        # Create tasks for all suites
        tasks = [
            suite.execute(timeout, verbose, debug)
            for suite in suites
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exception
                error_result = TestResult(suites[i].name)
                error_result.finish(False, -1, "", str(result), result)
                self.results.append(error_result)
                print(f"   ❌ {suites[i].name}: Exception - {str(result)}")
            else:
                self.results.append(result)
                status = "✅ PASSED" if result.success else "❌ FAILED"
                duration_str = f"{result.duration:.2f}s"
                print(f"   {status} {suites[i].name} ({duration_str})")

        print()

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests

        total_duration = sum(r.duration for r in self.results)
        overall_success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

        report = {
            "execution_summary": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_duration": self.end_time - self.start_time if self.end_time else 0,
                "total_test_suites": total_tests,
                "successful_suites": successful_tests,
                "failed_suites": failed_tests,
                "overall_success_rate": overall_success_rate,
                "total_test_execution_time": total_duration
            },
            "test_suite_results": [result.to_dict() for result in self.results],
            "performance_metrics": self._calculate_performance_metrics(),
            "failure_analysis": self._analyze_failures(),
            "recommendations": self._generate_recommendations()
        }

        # Save comprehensive report
        report_file = self.output_dir / f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.results:
            return {}

        durations = [r.duration for r in self.results]

        return {
            "fastest_suite": min(durations),
            "slowest_suite": max(durations),
            "average_duration": sum(durations) / len(durations),
            "total_execution_time": sum(durations),
            "parallel_efficiency": self._calculate_parallel_efficiency()
        }

    def _calculate_parallel_efficiency(self) -> float:
        """Calculate theoretical parallel efficiency"""
        if len(self.results) <= 1:
            return 1.0

        total_sequential_time = sum(r.duration for r in self.results)
        actual_total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0

        if actual_total_time == 0:
            return 0.0

        return total_sequential_time / actual_total_time

    def _analyze_failures(self) -> Dict[str, Any]:
        """Analyze test failures"""
        failed_results = [r for r in self.results if not r.success]

        if not failed_results:
            return {"failed_suites": 0, "failure_patterns": []}

        failure_patterns = []
        for result in failed_results:
            pattern = {
                "suite_name": result.test_name,
                "error_type": type(result.exception).__name__ if result.exception else "Unknown",
                "error_message": result.error_output[:200] if result.error_output else "No error message",
                "duration": result.duration
            }
            failure_patterns.append(pattern)

        return {
            "failed_suites": len(failed_results),
            "failure_rate": len(failed_results) / len(self.results) * 100,
            "failure_patterns": failure_patterns
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        failed_results = [r for r in self.results if not r.success]

        if not failed_results:
            recommendations.append("🎉 All tests passed! System is performing excellently.")
        else:
            recommendations.append(f"🔧 {len(failed_results)} test suite(s) failed. Review and fix issues.")

            # Analyze specific failure patterns
            for result in failed_results:
                if "timeout" in result.error_output.lower():
                    recommendations.append(f"⏱️  Consider increasing timeout for {result.test_name}")
                elif "import" in result.error_output.lower():
                    recommendations.append(f"📦 Check dependencies and imports for {result.test_name}")
                elif "memory" in result.error_output.lower():
                    recommendations.append(f"💾 Monitor memory usage in {result.test_name}")

        # Performance recommendations
        if self.results:
            avg_duration = sum(r.duration for r in self.results) / len(self.results)
            if avg_duration > 60:
                recommendations.append("⚡ Consider optimizing test execution performance")

        return recommendations

    def _save_individual_results(self):
        """Save individual test results to separate files"""
        for result in self.results:
            # Save detailed output
            output_file = self.output_dir / f"{result.test_name.lower().replace(' ', '_')}_output.txt"
            with open(output_file, 'w') as f:
                f.write(f"Test Suite: {result.test_name}\n")
                f.write(f"Duration: {result.duration:.2f}s\n")
                f.write(f"Success: {result.success}\n")
                f.write(f"Exit Code: {result.exit_code}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.output)
                f.write("\n=== STDERR ===\n")
                f.write(result.error_output)

    def print_summary(self, report: Dict[str, Any]):
        """Print test execution summary"""
        print("=" * 80)
        print("🏁 COMPREHENSIVE TEST EXECUTION SUMMARY")
        print("=" * 80)

        summary = report["execution_summary"]
        print(f"📊 Total Test Suites: {summary['total_test_suites']}")
        print(f"✅ Successful Suites: {summary['successful_suites']}")
        print(f"❌ Failed Suites: {summary['failed_suites']}")
        print(f"📈 Overall Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"⏱️  Total Duration: {summary['total_duration']:.2f}s")
        print(f"🔧 Test Execution Time: {summary['total_test_execution_time']:.2f}s")

        print("\n📋 Test Suite Results:")
        for result in self.results:
            status = "✅ PASSED" if result.success else "❌ FAILED"
            duration_str = f"{result.duration:.2f}s"
            print(f"   {status} {result.test_name} ({duration_str})")

        if report["failure_analysis"]["failed_suites"] > 0:
            print("\n🚨 Failure Analysis:")
            for pattern in report["failure_analysis"]["failure_patterns"]:
                print(f"   • {pattern['suite_name']}: {pattern['error_message']}")

        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"   {rec}")

        print(f"\n📁 Detailed reports saved to: {self.output_dir}")
        print("=" * 80)


class CoverageRunner:
    """Run test coverage analysis"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    async def run_coverage(self, verbose: bool = False) -> bool:
        """Run coverage analysis"""
        try:
            print("🔍 Running coverage analysis...")

            # Prepare coverage command
            cmd = [
                sys.executable, "-m", "pytest",
                str(Path(__file__).parent),
                "--cov=integration",
                "--cov-report=html",
                f"--cov-report=html:{self.output_dir}/coverage_html",
                f"--cov-report=xml:{self.output_dir}/coverage.xml",
                "--cov-report=term-missing"
            ]

            if verbose:
                cmd.append("-v")

            # Execute coverage
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                print("✅ Coverage analysis completed successfully")
                print(f"📊 HTML coverage report: {self.output_dir}/coverage_html/index.html")
                return True
            else:
                print("❌ Coverage analysis failed")
                print(f"Error: {stderr.decode('utf-8', errors='replace')}")
                return False

        except Exception as e:
            print(f"❌ Coverage analysis failed: {e}")
            return False


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Runner for Agent-Based Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tests sequentially
    python integration/run_all_tests.py

    # Run tests in parallel with coverage
    python integration/run_all_tests.py --parallel --coverage

    # Run only performance tests with verbose output
    python integration/run_all_tests.py --performance-only --verbose

    # Run tests with custom timeout and output directory
    python integration/run_all_tests.py --timeout 600 --output-dir custom_results
        """
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )

    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )

    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Run only performance tests"
    )

    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration tests"
    )

    parser.add_argument(
        "--error-only",
        action="store_true",
        help="Run only error scenario tests"
    )

    parser.add_argument(
        "--comprehensive-only",
        action="store_true",
        help="Run only comprehensive tests"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results",
        help="Specify output directory for reports (default: test_results)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Set test timeout in seconds (default: 300)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Determine which suites to run
    filter_suites = []
    if args.performance_only:
        filter_suites.append("Performance Tests")
    elif args.integration_only:
        filter_suites.append("Integration Tests")
    elif args.error_only:
        filter_suites.append("Error Scenario Tests")
    elif args.comprehensive_only:
        filter_suites.append("Comprehensive Tests")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = output_dir / f"test_run_{timestamp}"
    timestamped_dir.mkdir(exist_ok=True)

    # Run tests
    runner = TestRunner(timestamped_dir)

    try:
        report = await runner.run_all_tests(
            timeout=args.timeout,
            parallel=args.parallel,
            verbose=args.verbose,
            debug=args.debug,
            filter_suites=filter_suites if filter_suites else None
        )

        # Print summary
        runner.print_summary(report)

        # Run coverage if requested
        if args.coverage:
            coverage_runner = CoverageRunner(timestamped_dir)
            await coverage_runner.run_coverage(args.verbose)

        # Determine exit code
        success_rate = report["execution_summary"]["overall_success_rate"]
        if success_rate >= 95:
            print("\n🎉 Excellent! All critical systems are functioning properly.")
            exit_code = 0
        elif success_rate >= 80:
            print("\n✅ Good! Most systems are functioning with minor issues.")
            exit_code = 0
        else:
            print("\n❌ Poor! Significant issues need immediate attention.")
            exit_code = 1

        # Exit with appropriate code
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())