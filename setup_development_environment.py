#!/usr/bin/env python3
"""
Development Environment Setup Script

This script sets up the enhanced development environment for the multi-agent
research system with Claude Agent SDK integration, logging, monitoring, and
observability infrastructure.

Phase 1.1.3 Implementation: Development Environment Setup

Features:
- Automatic environment detection and configuration
- Enhanced logging and monitoring setup
- Configuration validation and testing
- Development tools initialization
- Health checks and diagnostics
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from multi_agent_research_system.config import (
        initialize_configuration, get_configuration_summary,
        validate_system_configuration
    )
    from multi_agent_research_system.agent_logging import (
        get_enhanced_logger, setup_logging_for_session,
        start_monitoring, stop_monitoring,
        LogLevel, LogCategory, AgentEventType
    )
    from multi_agent_research_system.config.sdk_config import (
        get_sdk_config, DEVELOPMENT_CONFIG, TESTING_CONFIG, PRODUCTION_CONFIG
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this script from the project root and dependencies are installed.")
    sys.exit(1)


class DevelopmentEnvironmentSetup:
    """Handles development environment setup and configuration."""

    def __init__(self, environment: str = "development", config_dir: Optional[str] = None):
        self.environment = environment.lower()
        self.config_dir = Path(config_dir) if config_dir else None
        self.project_root = project_root

        # Initialize logging for setup process
        self.setup_logger = get_enhanced_logger("environment_setup")
        self.setup_logger.set_session_context(
            session_id=f"setup-{int(time.time())}",
            agent_id="setup_script",
            agent_type="setup"
        )

        self.setup_results = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "steps_completed": [],
            "errors": [],
            "warnings": [],
            "configuration_status": {},
            "health_checks": {}
        }

    def run_setup(self) -> bool:
        """Run the complete setup process."""
        self.setup_logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.SESSION_START,
            f"Starting development environment setup for: {self.environment}"
        )

        try:
            # Setup steps
            self._validate_python_version()
            self._validate_dependencies()
            self._setup_configuration()
            self._setup_directories()
            self._setup_logging_infrastructure()
            self._setup_monitoring()
            self._validate_configuration()
            self._run_health_checks()
            self._generate_setup_report()

            success = len(self.setup_results["errors"]) == 0

            if success:
                self.setup_logger.log_event(
                    LogLevel.INFO,
                    LogCategory.SYSTEM,
                    AgentEventType.SESSION_END,
                    "Development environment setup completed successfully"
                )
                print("\n✅ Development environment setup completed successfully!")
            else:
                self.setup_logger.log_event(
                    LogLevel.ERROR,
                    LogCategory.SYSTEM,
                    AgentEventType.ERROR,
                    f"Setup completed with {len(self.setup_results['errors'])} errors"
                )
                print(f"\n❌ Setup completed with {len(self.setup_results['errors'])} errors")

            return success

        except Exception as e:
            error_msg = f"Setup failed: {str(e)}"
            self.setup_results["errors"].append(error_msg)
            self.setup_logger.log_event(
                LogLevel.CRITICAL,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                error_msg,
                error_details={"error_type": type(e).__name__, "error_message": str(e)}
            )
            print(f"\n💥 Setup failed: {e}")
            return False

    def _validate_python_version(self):
        """Validate Python version requirements."""
        self.setup_logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.TASK_START,
            "Validating Python version"
        )

        try:
            python_version = sys.version_info
            required_version = (3, 10)

            if python_version >= required_version:
                self.setup_results["steps_completed"].append("Python version validation")
                self.setup_logger.log_event(
                    LogLevel.INFO,
                    LogCategory.SYSTEM,
                    AgentEventType.TASK_END,
                    f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} is valid"
                )
                print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            else:
                error_msg = f"Python {required_version[0]}.{required_version[1]}+ required, found {python_version.major}.{python_version.minor}"
                self.setup_results["errors"].append(error_msg)
                self.setup_logger.log_event(
                    LogLevel.ERROR,
                    LogCategory.SYSTEM,
                    AgentEventType.ERROR,
                    error_msg
                )
                print(f"❌ {error_msg}")

        except Exception as e:
            error_msg = f"Failed to validate Python version: {str(e)}"
            self.setup_results["errors"].append(error_msg)
            self.setup_logger.log_event(
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                error_msg
            )

    def _validate_dependencies(self):
        """Validate that all required dependencies are installed."""
        self.setup_logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.TASK_START,
            "Validating dependencies"
        )

        critical_dependencies = [
            "claude-agent-sdk",
            "anthropic",
            "pydantic-ai",
            "structlog",
            "rich",
            "psutil",
            "pandas",
            "numpy"
        ]

        missing_deps = []
        installed_deps = []

        for dep in critical_dependencies:
            try:
                __import__(dep.replace("-", "_"))
                installed_deps.append(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            error_msg = f"Missing critical dependencies: {', '.join(missing_deps)}"
            self.setup_results["errors"].append(error_msg)
            self.setup_logger.log_event(
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                error_msg
            )
            print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
            print("   Run: uv sync --dev to install dependencies")
        else:
            self.setup_results["steps_completed"].append("Dependency validation")
            self.setup_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.TASK_END,
                f"All {len(installed_deps)} critical dependencies are installed"
            )
            print(f"✅ All {len(installed_deps)} critical dependencies installed")

    def _setup_configuration(self):
        """Setup configuration for the specified environment."""
        self.setup_logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.TASK_START,
            f"Setting up {self.environment} configuration"
        )

        try:
            # Initialize configuration
            config_manager = initialize_configuration(
                config_dir=self.config_dir,
                environment=self.environment
            )

            # Get configuration summary
            config_summary = get_configuration_summary()
            self.setup_results["configuration_status"] = config_summary

            self.setup_results["steps_completed"].append("Configuration setup")
            self.setup_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.TASK_END,
                f"Configuration initialized for {self.environment} environment"
            )
            print(f"✅ Configuration initialized for {self.environment} environment")

            # Display configuration summary
            print(f"   SDK Version: {config_summary['sdk_config']['version']}")
            print(f"   Default Model: {config_summary['sdk_config']['model']}")
            print(f"   Max Tokens: {config_summary['sdk_config']['max_tokens']}")
            print(f"   Debug Mode: {config_summary['sdk_config']['debug_mode']}")
            print(f"   Enhanced Agents: {config_summary['enhanced_agents']['total_agents']} configured")

        except Exception as e:
            error_msg = f"Failed to setup configuration: {str(e)}"
            self.setup_results["errors"].append(error_msg)
            self.setup_logger.log_event(
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                error_msg,
                error_details={"error_type": type(e).__name__, "error_message": str(e)}
            )
            print(f"❌ {error_msg}")

    def _setup_directories(self):
        """Setup required directories for the development environment."""
        self.setup_logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.TASK_START,
            "Setting up directory structure"
        )

        required_dirs = [
            "KEVIN",
            "KEVIN/sessions",
            "KEVIN/work_products",
            "KEVIN/logs",
            "KEVIN/config",
            "KEVIN/monitoring",
            "KEVIN/exports"
        ]

        created_dirs = []
        existing_dirs = []

        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                existing_dirs.append(dir_path)
            else:
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_path)
                except Exception as e:
                    error_msg = f"Failed to create directory {dir_path}: {str(e)}"
                    self.setup_results["errors"].append(error_msg)
                    self.setup_logger.log_event(
                        LogLevel.ERROR,
                        LogCategory.SYSTEM,
                        AgentEventType.ERROR,
                        error_msg
                    )

        if not any("Failed to create directory" in error for error in self.setup_results["errors"]):
            self.setup_results["steps_completed"].append("Directory setup")
            self.setup_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.TASK_END,
                f"Directory structure setup complete (created: {len(created_dirs)}, existing: {len(existing_dirs)})"
            )
            print(f"✅ Directory structure ready")
            if created_dirs:
                print(f"   Created: {len(created_dirs)} new directories")
            if existing_dirs:
                print(f"   Existing: {len(existing_dirs)} directories")

    def _setup_logging_infrastructure(self):
        """Setup enhanced logging infrastructure."""
        self.setup_logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.TASK_START,
            "Setting up logging infrastructure"
        )

        try:
            # Test enhanced logging
            test_logger = get_enhanced_logger("test_logging")
            test_logger.set_session_context(
                session_id="setup-test",
                agent_id="setup_test",
                agent_type="test"
            )

            # Test different log levels and categories
            test_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.SESSION_START,
                "Testing logging infrastructure"
            )

            # Test performance tracking
            with test_logger.task_timer("setup_test_task"):
                time.sleep(0.1)  # Simulate work
                test_logger.log_event(
                    LogLevel.DEBUG,
                    LogCategory.PERFORMANCE,
                    AgentEventType.MESSAGE_PROCESSED,
                    "Performance tracking test"
                )

            # Test quality assessment logging
            test_logger.log_quality_assessment(
                content="Test content for quality assessment",
                quality_score=8.5,
                dimensions={
                    "accuracy": 9.0,
                    "completeness": 8.0,
                    "clarity": 8.5
                }
            )

            # Test flow compliance logging
            test_logger.log_flow_compliance(
                compliance_status="compliant",
                violations=[],
                enforcement_actions=[]
            )

            self.setup_results["steps_completed"].append("Logging infrastructure setup")
            self.setup_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.TASK_END,
                "Logging infrastructure setup and tested successfully"
            )
            print("✅ Enhanced logging infrastructure ready")

        except Exception as e:
            error_msg = f"Failed to setup logging infrastructure: {str(e)}"
            self.setup_results["errors"].append(error_msg)
            self.setup_logger.log_event(
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                error_msg,
                error_details={"error_type": type(e).__name__, "error_message": str(e)}
            )
            print(f"❌ {error_msg}")

    def _setup_monitoring(self):
        """Setup monitoring and metrics collection."""
        self.setup_logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.TASK_START,
            "Setting up monitoring system"
        )

        try:
            # Start monitoring system
            start_monitoring()

            # Test metrics recording
            from multi_agent_research_system.agent_logging import record_agent_task, record_tool_execution

            record_agent_task("test_agent", "test_task", 100.0, True, 8.5)
            record_tool_execution("test_tool", 50.0, True)

            # Wait a moment for metrics collection
            time.sleep(1)

            # Get system status
            from multi_agent_research_system.agent_logging import get_monitoring_system
            monitoring_system = get_monitoring_system()
            system_status = monitoring_system.get_system_status()

            self.setup_results["health_checks"]["monitoring"] = system_status

            self.setup_results["steps_completed"].append("Monitoring setup")
            self.setup_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.TASK_END,
                "Monitoring system setup and tested successfully"
            )
            print("✅ Monitoring system active")
            print(f"   System health: {system_status['health']['healthy']}")
            print(f"   Active alerts: {system_status['active_alerts']}")

        except Exception as e:
            error_msg = f"Failed to setup monitoring: {str(e)}"
            self.setup_results["errors"].append(error_msg)
            self.setup_logger.log_event(
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                error_msg,
                error_details={"error_type": type(e).__name__, "error_message": str(e)}
            )
            print(f"❌ {error_msg}")

    def _validate_configuration(self):
        """Validate system configuration."""
        self.setup_logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.TASK_START,
            "Validating system configuration"
        )

        try:
            validation_results = validate_system_configuration()
            self.setup_results["configuration_validation"] = validation_results

            if validation_results["compatible"]:
                self.setup_results["steps_completed"].append("Configuration validation")
                self.setup_logger.log_event(
                    LogLevel.INFO,
                    LogCategory.SYSTEM,
                    AgentEventType.TASK_END,
                    "System configuration validation passed"
                )
                print("✅ Configuration validation passed")
            else:
                for issue in validation_results["issues"]:
                    self.setup_results["errors"].append(f"Configuration issue: {issue}")
                    self.setup_logger.log_event(
                        LogLevel.WARNING,
                        LogCategory.SYSTEM,
                        AgentEventType.ERROR,
                        f"Configuration issue: {issue}"
                    )

                for warning in validation_results["warnings"]:
                    self.setup_results["warnings"].append(f"Configuration warning: {warning}")
                    self.setup_logger.log_event(
                        LogLevel.INFO,
                        LogCategory.SYSTEM,
                        AgentEventType.ERROR,
                        f"Configuration warning: {warning}"
                    )

                print(f"⚠️  Configuration validation completed with {len(validation_results['issues'])} issues and {len(validation_results['warnings'])} warnings")

        except Exception as e:
            error_msg = f"Failed to validate configuration: {str(e)}"
            self.setup_results["errors"].append(error_msg)
            self.setup_logger.log_event(
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                error_msg
            )
            print(f"❌ {error_msg}")

    def _run_health_checks(self):
        """Run system health checks."""
        self.setup_logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.TASK_START,
            "Running system health checks"
        )

        try:
            from multi_agent_research_system.agent_logging.monitoring import check_system_resources, check_process_health

            # System resources check
            system_health = check_system_resources()
            self.setup_results["health_checks"]["system_resources"] = system_health

            # Process health check
            process_health = check_process_health()
            self.setup_results["health_checks"]["process"] = process_health

            overall_healthy = system_health["healthy"] and process_health["healthy"]

            if overall_healthy:
                self.setup_results["steps_completed"].append("Health checks")
                self.setup_logger.log_event(
                    LogLevel.INFO,
                    LogCategory.SYSTEM,
                    AgentEventType.TASK_END,
                    "All health checks passed"
                )
                print("✅ All health checks passed")
                print(f"   CPU usage: {system_health.get('cpu_percent', 'N/A')}%")
                print(f"   Memory usage: {system_health.get('memory_percent', 'N/A')}%")
                print(f"   Disk usage: {system_health.get('disk_percent', 'N/A')}%")
            else:
                issues = []
                if not system_health["healthy"]:
                    issues.extend(system_health.get("issues", []))
                if not process_health["healthy"]:
                    issues.append(f"Process issue: {process_health.get('status', 'unknown')}")

                for issue in issues:
                    self.setup_results["warnings"].append(f"Health check warning: {issue}")
                    self.setup_logger.log_event(
                        LogLevel.WARNING,
                        LogCategory.SYSTEM,
                        AgentEventType.ERROR,
                        f"Health check warning: {issue}"
                    )

                print(f"⚠️  Health checks completed with {len(issues)} warnings")

        except Exception as e:
            error_msg = f"Failed to run health checks: {str(e)}"
            self.setup_results["errors"].append(error_msg)
            self.setup_logger.log_event(
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                error_msg
            )
            print(f"❌ {error_msg}")

    def _generate_setup_report(self):
        """Generate and save setup report."""
        self.setup_logger.log_event(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            AgentEventType.TASK_START,
            "Generating setup report"
        )

        try:
            # Save setup report
            reports_dir = self.project_root / "KEVIN" / "reports"
            reports_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"setup_report_{self.environment}_{timestamp}.json"

            with open(report_file, 'w') as f:
                json.dump(self.setup_results, f, indent=2, default=str)

            self.setup_results["steps_completed"].append("Setup report generation")
            self.setup_logger.log_event(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                AgentEventType.TASK_END,
                f"Setup report saved to: {report_file}"
            )
            print(f"📋 Setup report saved to: {report_file}")

            # Export logs
            from multi_agent_research_system.agent_logging import export_all_logs
            log_export_dir = reports_dir / "setup_logs"
            export_success = export_all_logs(log_export_dir)

            if export_success:
                print(f"📊 Setup logs exported to: {log_export_dir}")

        except Exception as e:
            error_msg = f"Failed to generate setup report: {str(e)}"
            self.setup_results["errors"].append(error_msg)
            self.setup_logger.log_event(
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                AgentEventType.ERROR,
                error_msg
            )
            print(f"❌ {error_msg}")

    def print_summary(self):
        """Print setup summary."""
        print("\n" + "="*60)
        print("DEVELOPMENT ENVIRONMENT SETUP SUMMARY")
        print("="*60)

        print(f"Environment: {self.environment}")
        print(f"Setup Time: {self.setup_results['timestamp']}")
        print(f"Steps Completed: {len(self.setup_results['steps_completed'])}")
        print(f"Errors: {len(self.setup_results['errors'])}")
        print(f"Warnings: {len(self.setup_results['warnings'])}")

        if self.setup_results["steps_completed"]:
            print("\n✅ Completed Steps:")
            for step in self.setup_results["steps_completed"]:
                print(f"   • {step}")

        if self.setup_results["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in self.setup_results["warnings"]:
                print(f"   • {warning}")

        if self.setup_results["errors"]:
            print("\n❌ Errors:")
            for error in self.setup_results["errors"]:
                print(f"   • {error}")

        # Configuration status
        if "sdk_config" in self.setup_results.get("configuration_status", {}):
            config = self.setup_results["configuration_status"]["sdk_config"]
            print(f"\n🔧 Configuration:")
            print(f"   • SDK Version: {config.get('version', 'N/A')}")
            print(f"   • Model: {config.get('model', 'N/A')}")
            print(f"   • Max Tokens: {config.get('max_tokens', 'N/A')}")
            print(f"   • Debug Mode: {config.get('debug_mode', 'N/A')}")

        # Health check summary
        if self.setup_results["health_checks"]:
            print(f"\n🏥 Health Status:")
            for check_name, check_result in self.setup_results["health_checks"].items():
                status = "✅ Healthy" if check_result.get("healthy", False) else "⚠️  Issues"
                print(f"   • {check_name.replace('_', ' ').title()}: {status}")

        print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup development environment for Multi-Agent Research System"
    )
    parser.add_argument(
        "--environment",
        choices=["development", "testing", "staging", "production"],
        default="development",
        help="Target environment (default: development)"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        help="Configuration directory path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    print("🚀 Multi-Agent Research System - Development Environment Setup")
    print(f"Environment: {args.environment}")
    print(f"Project Root: {project_root}")
    print("-" * 60)

    # Run setup
    setup = DevelopmentEnvironmentSetup(
        environment=args.environment,
        config_dir=args.config_dir
    )

    success = setup.run_setup()
    setup.print_summary()

    # Cleanup monitoring
    try:
        stop_monitoring()
    except:
        pass

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()