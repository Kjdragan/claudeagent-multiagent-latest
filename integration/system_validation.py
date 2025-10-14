#!/usr/bin/env python3
"""
System Validation Script for Agent-Based Research System

This script performs comprehensive validation of the agent-based research system
without requiring complex dependencies or async operations.

Usage:
    python integration/system_validation.py [options]

Options:
    --verbose, -v          Enable verbose output
    --quick                Run quick validation only
    --comprehensive        Run comprehensive validation
    --output-dir DIR       Specify output directory for reports
    --help, -h             Show this help message

Author: Claude Code Assistant
Version: 1.0.0
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class SystemValidator:
    """Comprehensive system validator"""

    def __init__(self, output_dir: Path = None, verbose: bool = False):
        self.output_dir = output_dir or Path("validation_results")
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        self.results = {
            "validation_summary": {
                "start_time": datetime.now().isoformat(),
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "warnings": []
            },
            "check_results": {},
            "system_components": {},
            "performance_metrics": {},
            "recommendations": []
        }

    def log(self, message: str, level: str = "INFO"):
        """Log validation messages"""
        if self.verbose or level in ["ERROR", "WARNING", "SUCCESS"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {
                "INFO": "ℹ️",
                "SUCCESS": "✅",
                "WARNING": "⚠️",
                "ERROR": "❌"
            }.get(level, "ℹ️")
            print(f"[{timestamp}] {prefix} {message}")

    def validate_system_components(self) -> Dict[str, Any]:
        """Validate system components exist and are accessible"""
        self.log("Validating system components...")

        components = {
            "main_entry_point": ["main_comprehensive_research.py"],
            "agents": [
                "agents/comprehensive_research_agent.py"
            ],
            "integration": [
                "integration/agent_session_manager.py",
                "integration/query_processor.py",
                "integration/research_orchestrator.py",
                "integration/mcp_tool_integration.py",
                "integration/kevin_directory_integration.py",
                "integration/quality_assurance_integration.py",
                "integration/error_handling_integration.py"
            ],
            "testing": [
                "integration/comprehensive_test_suite.py",
                "integration/integration_tests.py",
                "integration/performance_tests.py",
                "integration/error_scenario_tests.py",
                "integration/simple_performance_tests.py"
            ]
        }

        component_results = {}
        total_components = 0
        passed_components = 0

        for category, files in components.items():
            self.log(f"  Checking {category}...", "INFO")
            category_results = []

            for file_path in files:
                total_components += 1
                full_path = Path(file_path)

                if full_path.exists():
                    result = {
                        "file": str(full_path),
                        "exists": True,
                        "readable": os.access(full_path, os.R_OK),
                        "size_bytes": full_path.stat().st_size if full_path.exists() else 0
                    }
                    passed_components += 1
                    self.log(f"    ✓ {full_path} ({result['size_bytes']} bytes)", "SUCCESS")
                else:
                    result = {
                        "file": str(full_path),
                        "exists": False,
                        "readable": False,
                        "size_bytes": 0
                    }
                    self.log(f"    ❌ {full_path} - Not found", "ERROR")

                category_results.append(result)

            component_results[category] = category_results

        # Update summary
        self.results["validation_summary"]["total_checks"] += total_components
        self.results["validation_summary"]["passed_checks"] += passed_components
        self.results["validation_summary"]["failed_checks"] += (total_components - passed_components)
        self.results["system_components"] = component_results

        success_rate = (passed_components / total_components * 100) if total_components > 0 else 0
        self.log(f"Component validation complete: {passed_components}/{total_components} ({success_rate:.1f}%)")

        return component_results

    def validate_code_quality(self) -> Dict[str, Any]:
        """Validate basic code quality"""
        self.log("Validating code quality...")

        quality_checks = {
            "python_syntax": [],
            "import_structure": [],
            "documentation": []
        }

        # Check Python syntax for key files
        key_files = [
            "main_comprehensive_research.py",
            "agents/comprehensive_research_agent.py",
            "integration/research_orchestrator.py",
            "integration/agent_session_manager.py"
        ]

        syntax_passed = 0
        syntax_total = 0

        for file_path in key_files:
            if Path(file_path).exists():
                syntax_total += 1
                try:
                    with open(file_path, 'r') as f:
                        code = f.read()
                    compile(code, file_path, 'exec')
                    syntax_passed += 1
                    self.log(f"    ✓ {file_path} - Valid Python syntax", "SUCCESS")
                    quality_checks["python_syntax"].append({
                        "file": file_path,
                        "valid": True
                    })
                except SyntaxError as e:
                    self.log(f"    ❌ {file_path} - Syntax error: {e}", "ERROR")
                    quality_checks["python_syntax"].append({
                        "file": file_path,
                        "valid": False,
                        "error": str(e)
                    })

        # Check basic documentation
        doc_files = [
            "README.md",
            "integration/TESTING_GUIDELINES.md"
        ]

        doc_passed = 0
        doc_total = 0

        for file_path in doc_files:
            if Path(file_path).exists():
                doc_total += 1
                with open(file_path, 'r') as f:
                    content = f.read()
                if len(content) > 1000:  # Reasonable documentation length
                    doc_passed += 1
                    self.log(f"    ✓ {file_path} - Adequate documentation", "SUCCESS")
                else:
                    self.log(f"    ⚠️ {file_path} - Minimal documentation", "WARNING")

        # Update summary
        total_checks = syntax_total + doc_total
        passed_checks = syntax_passed + doc_passed

        self.results["validation_summary"]["total_checks"] += total_checks
        self.results["validation_summary"]["passed_checks"] += passed_checks
        self.results["validation_summary"]["failed_checks"] += (total_checks - passed_checks)

        quality_results = {
            "syntax_checks": {"total": syntax_total, "passed": syntax_passed},
            "documentation_checks": {"total": doc_total, "passed": doc_passed},
            "details": quality_checks
        }

        self.log(f"Code quality validation complete: {passed_checks}/{total_checks} checks passed")
        return quality_results

    def validate_directory_structure(self) -> Dict[str, Any]:
        """Validate expected directory structure"""
        self.log("Validating directory structure...")

        expected_dirs = [
            "agents",
            "integration",
            "tools",
            "utils",
            "config",
            "core",
            "KEVIN"
        ]

        dir_results = []
        passed_dirs = 0

        for dir_name in expected_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists() and dir_path.is_dir():
                passed_dirs += 1
                file_count = len(list(dir_path.glob("*.py")))
                self.log(f"    ✓ {dir_name}/ - {file_count} Python files", "SUCCESS")
                dir_results.append({
                    "name": dir_name,
                    "exists": True,
                    "file_count": file_count
                })
            else:
                self.log(f"    ❌ {dir_name}/ - Directory not found", "ERROR")
                dir_results.append({
                    "name": dir_name,
                    "exists": False,
                    "file_count": 0
                })

        # Check KEVIN subdirectories
        kevin_dir = Path("KEVIN")
        if kevin_dir.exists():
            kevin_subdirs = ["sessions", "Project Documentation"]
            for subdir in kevin_subdirs:
                subdir_path = kevin_dir / subdir
                if subdir_path.exists():
                    self.log(f"    ✓ KEVIN/{subdir}/ - Found", "SUCCESS")
                else:
                    self.log(f"    ⚠️ KEVIN/{subdir}/ - Not found (optional)", "WARNING")

        # Update summary
        self.results["validation_summary"]["total_checks"] += len(expected_dirs)
        self.results["validation_summary"]["passed_checks"] += passed_dirs
        self.results["validation_summary"]["failed_checks"] += (len(expected_dirs) - passed_dirs)

        structure_results = {
            "expected_directories": len(expected_dirs),
            "found_directories": passed_dirs,
            "details": dir_results
        }

        self.log(f"Directory structure validation complete: {passed_dirs}/{len(expected_dirs)} directories found")
        return structure_results

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration"""
        self.log("Validating configuration...")

        config_checks = {
            "environment_variables": [],
            "configuration_files": []
        }

        # Check important environment variables
        important_env_vars = [
            "PYTHONPATH",
            "PATH"
        ]

        env_passed = 0
        for var in important_env_vars:
            if var in os.environ:
                env_passed += 1
                self.log(f"    ✓ {var} - Set", "SUCCESS")
                config_checks["environment_variables"].append({
                    "name": var,
                    "set": True,
                    "value": os.environ[var] if not self.verbose else "[REDACTED]"
                })
            else:
                self.log(f"    ⚠️ {var} - Not set", "WARNING")
                config_checks["environment_variables"].append({
                    "name": var,
                    "set": False
                })

        # Check for configuration files
        config_files = [
            "requirements.txt",
            ".gitignore",
            "CLAUDE.md"
        ]

        config_file_passed = 0
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                config_file_passed += 1
                self.log(f"    ✓ {config_file} - Found", "SUCCESS")
                config_checks["configuration_files"].append({
                    "file": config_file,
                    "exists": True
                })
            else:
                self.log(f"    ⚠️ {config_file} - Not found", "WARNING")
                config_checks["configuration_files"].append({
                    "file": config_file,
                    "exists": False
                })

        # Update summary
        total_config_checks = len(important_env_vars) + len(config_files)
        passed_config_checks = env_passed + config_file_passed

        self.results["validation_summary"]["total_checks"] += total_config_checks
        self.results["validation_summary"]["passed_checks"] += passed_config_checks
        self.results["validation_summary"]["failed_checks"] += (total_config_checks - passed_config_checks)

        config_results = {
            "environment_variables": {"total": len(important_env_vars), "passed": env_passed},
            "configuration_files": {"total": len(config_files), "passed": config_file_passed},
            "details": config_checks
        }

        self.log(f"Configuration validation complete: {passed_config_checks}/{total_config_checks} checks passed")
        return config_results

    def validate_testing_framework(self) -> Dict[str, Any]:
        """Validate testing framework"""
        self.log("Validating testing framework...")

        test_framework_checks = {
            "test_suites": [],
            "test_execution": None
        }

        # Check test suites
        test_suites = [
            "integration/simple_performance_tests.py",
            "integration/comprehensive_test_suite.py",
            "integration/integration_tests.py"
        ]

        test_suite_passed = 0
        for test_suite in test_suites:
            if Path(test_suite).exists():
                test_suite_passed += 1
                self.log(f"    ✓ {test_suite} - Available", "SUCCESS")
                test_framework_checks["test_suites"].append({
                    "suite": test_suite,
                    "available": True
                })
            else:
                self.log(f"    ❌ {test_suite} - Not found", "ERROR")
                test_framework_checks["test_suites"].append({
                    "suite": test_suite,
                    "available": False
                })

        # Try to run simple performance test
        try:
            self.log("    Running simple performance test...", "INFO")
            import subprocess
            result = subprocess.run(
                [sys.executable, "integration/simple_performance_tests.py"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                self.log("    ✓ Simple performance test - PASSED", "SUCCESS")
                test_framework_checks["test_execution"] = {
                    "simple_performance_test": "passed",
                    "output": result.stdout[-200:] if len(result.stdout) > 200 else result.stdout
                }
            else:
                self.log("    ❌ Simple performance test - FAILED", "ERROR")
                test_framework_checks["test_execution"] = {
                    "simple_performance_test": "failed",
                    "error": result.stderr[-200:] if len(result.stderr) > 200 else result.stderr
                }
        except Exception as e:
            self.log(f"    ⚠️ Could not run performance test: {e}", "WARNING")
            test_framework_checks["test_execution"] = {
                "simple_performance_test": "error",
                "error": str(e)
            }

        # Update summary
        total_framework_checks = len(test_suites) + 1
        passed_framework_checks = test_suite_passed + (1 if test_framework_checks.get("test_execution", {}).get("simple_performance_test") == "passed" else 0)

        self.results["validation_summary"]["total_checks"] += total_framework_checks
        self.results["validation_summary"]["passed_checks"] += passed_framework_checks
        self.results["validation_summary"]["failed_checks"] += (total_framework_checks - passed_framework_checks)

        framework_results = {
            "test_suites": {"total": len(test_suites), "passed": test_suite_passed},
            "test_execution": test_framework_checks["test_execution"],
            "details": test_framework_checks
        }

        self.log(f"Testing framework validation complete: {passed_framework_checks}/{total_framework_checks} checks passed")
        return framework_results

    def perform_basic_performance_test(self) -> Dict[str, Any]:
        """Perform basic performance test"""
        self.log("Performing basic performance test...")

        performance_tests = {
            "file_operations": self._test_file_operations(),
            "memory_usage": self._test_memory_usage(),
            "processing_speed": self._test_processing_speed()
        }

        # Calculate overall performance score
        performance_scores = []
        for test_name, test_result in performance_tests.items():
            if test_result.get("success", False):
                performance_scores.append(test_result.get("score", 0))

        overall_score = sum(performance_scores) / len(performance_scores) if performance_scores else 0

        performance_results = {
            "overall_score": overall_score,
            "tests": performance_tests,
            "assessment": "excellent" if overall_score >= 90 else "good" if overall_score >= 70 else "needs_improvement"
        }

        self.log(f"Basic performance test complete: {performance_results['assessment']} ({overall_score:.1f}/100)")
        return performance_results

    def _test_file_operations(self) -> Dict[str, Any]:
        """Test file operation performance"""
        try:
            import tempfile
            import shutil

            temp_dir = Path(tempfile.mkdtemp())

            # Test file creation and writing
            start_time = time.time()

            for i in range(10):
                file_path = temp_dir / f"test_{i}.txt"
                with open(file_path, 'w') as f:
                    f.write(f"Test content {i}\n" * 100)

            write_time = time.time() - start_time

            # Test file reading
            start_time = time.time()

            for i in range(10):
                file_path = temp_dir / f"test_{i}.txt"
                with open(file_path, 'r') as f:
                    content = f.read()

            read_time = time.time() - start_time

            # Cleanup
            shutil.rmtree(temp_dir)

            # Calculate score based on performance
            score = 100 - ((write_time + read_time) * 1000)  # Convert to milliseconds
            score = max(0, min(100, score))

            return {
                "success": True,
                "write_time": write_time,
                "read_time": read_time,
                "total_time": write_time + read_time,
                "score": score
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "score": 0
            }

    def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns"""
        try:
            # Test creating and managing data structures
            data = []

            start_time = time.time()

            for i in range(1000):
                data.append({
                    "id": i,
                    "content": f"Item {i}",
                    "metadata": {"created": time.time(), "type": "test"}
                })

            creation_time = time.time() - start_time

            # Test processing
            start_time = time.time()

            processed = [item for item in data if item["id"] % 2 == 0]

            processing_time = time.time() - start_time

            # Calculate score
            total_time = creation_time + processing_time
            score = 100 - (total_time * 500)  # Adjust scoring as needed
            score = max(0, min(100, score))

            return {
                "success": True,
                "items_created": len(data),
                "items_processed": len(processed),
                "creation_time": creation_time,
                "processing_time": processing_time,
                "total_time": total_time,
                "score": score
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "score": 0
            }

    def _test_processing_speed(self) -> Dict[str, Any]:
        """Test processing speed"""
        try:
            # Test computational performance
            start_time = time.time()

            # Perform some calculations
            result = sum(i * i for i in range(10000))

            calculation_time = time.time() - start_time

            # Test string processing
            start_time = time.time()

            text = "This is a test string for processing performance evaluation."
            processed = [word.upper() for word in text.split()]

            string_processing_time = time.time() - start_time

            # Calculate score
            total_time = calculation_time + string_processing_time
            score = 100 - (total_time * 1000)
            score = max(0, min(100, score))

            return {
                "success": True,
                "calculation_result": result,
                "calculation_time": calculation_time,
                "string_processing_time": string_processing_time,
                "total_time": total_time,
                "score": score
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "score": 0
            }

    def generate_recommendations(self):
        """Generate recommendations based on validation results"""
        self.log("Generating recommendations...")

        recommendations = []

        # Component recommendations
        components = self.results.get("system_components", {})
        for category, files in components.items():
            failed_files = [f for f in files if not f.get("exists", False)]
            if failed_files:
                recommendations.append({
                    "type": "missing_components",
                    "priority": "high",
                    "message": f"Missing {len(failed_files)} file(s) in {category}",
                    "details": [f["file"] for f in failed_files]
                })

        # Performance recommendations
        performance = self.results.get("performance_metrics", {})
        if performance.get("assessment") == "needs_improvement":
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "message": "System performance needs improvement",
                "details": [f"Overall score: {performance.get('overall_score', 0):.1f}/100"]
            })

        # Testing framework recommendations
        testing = self.results.get("check_results", {}).get("testing_framework", {})
        if testing.get("test_execution", {}).get("simple_performance_test") != "passed":
            recommendations.append({
                "type": "testing",
                "priority": "medium",
                "message": "Testing framework validation failed",
                "details": ["Check test suite compatibility and dependencies"]
            })

        # General recommendations
        success_rate = self._calculate_success_rate()
        if success_rate < 80:
            recommendations.append({
                "type": "general",
                "priority": "high",
                "message": f"System validation success rate is low ({success_rate:.1f}%)",
                "details": ["Address failed validation checks before production deployment"]
            })
        elif success_rate >= 95:
            recommendations.append({
                "type": "general",
                "priority": "low",
                "message": "System validation looks excellent!",
                "details": [f"Success rate: {success_rate:.1f}% - Ready for production"]
            })

        self.results["recommendations"] = recommendations

    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        summary = self.results["validation_summary"]
        total = summary["total_checks"]
        passed = summary["passed_checks"]
        return (passed / total * 100) if total > 0 else 0

    def save_results(self):
        """Save validation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"system_validation_{timestamp}.json"

        # Add completion timestamp
        self.results["validation_summary"]["end_time"] = datetime.now().isoformat()
        self.results["validation_summary"]["success_rate"] = self._calculate_success_rate()

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        self.log(f"Validation results saved to: {results_file}")
        return results_file

    def run_validation(self, quick: bool = False) -> Dict[str, Any]:
        """Run complete system validation"""
        self.log("🚀 Starting Agent-Based Research System Validation", "INFO")
        self.log(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"📁 Output directory: {self.output_dir}")

        if quick:
            self.log("🏃 Running quick validation...")
            self.validate_system_components()
            self.validate_directory_structure()
        else:
            self.log("🔍 Running comprehensive validation...")
            self.validate_system_components()
            self.validate_code_quality()
            self.validate_directory_structure()
            self.validate_configuration()
            self.validate_testing_framework()
            self.perform_basic_performance_test()

        self.generate_recommendations()
        results_file = self.save_results()

        # Print summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print validation summary"""
        summary = self.results["validation_summary"]
        success_rate = self._calculate_success_rate()

        print("\n" + "=" * 80)
        print("🏁 SYSTEM VALIDATION SUMMARY")
        print("=" * 80)
        print(f"📊 Total Checks: {summary['total_checks']}")
        print(f"✅ Passed Checks: {summary['passed_checks']}")
        print(f"❌ Failed Checks: {summary['failed_checks']}")
        print(f"📈 Success Rate: {success_rate:.1f}%")

        if success_rate >= 95:
            print("\n🎉 EXCELLENT! System is ready for production deployment.")
        elif success_rate >= 80:
            print("\n✅ GOOD! System is mostly ready with minor issues to address.")
        elif success_rate >= 60:
            print("\n⚠️  FAIR! System needs attention before production deployment.")
        else:
            print("\n❌ POOR! System requires significant fixes before production use.")

        # Print recommendations
        if self.results["recommendations"]:
            print(f"\n💡 Recommendations ({len(self.results['recommendations'])}):")
            for i, rec in enumerate(self.results["recommendations"], 1):
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec["priority"], "⚪")
                print(f"   {i}. {priority_icon} {rec['message']}")

        print("\n" + "=" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="System Validation Script for Agent-Based Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run comprehensive validation
    python integration/system_validation.py

    # Run quick validation
    python integration/system_validation.py --quick

    # Run with verbose output
    python integration/system_validation.py --verbose

    # Specify custom output directory
    python integration/system_validation.py --output-dir custom_validation
        """
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation only"
    )

    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive validation (default)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_results",
        help="Specify output directory for reports"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Run validation
    validator = SystemValidator(output_dir, args.verbose)

    try:
        results = validator.run_validation(quick=args.quick)

        # Determine exit code based on success rate
        success_rate = validator._calculate_success_rate()
        if success_rate >= 80:
            print("\n✅ System validation completed successfully!")
            exit_code = 0
        else:
            print("\n❌ System validation failed with significant issues.")
            exit_code = 1

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n⚠️  System validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 System validation failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()