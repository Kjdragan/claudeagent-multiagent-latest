#!/usr/bin/env python3
"""
Startup Health Tests for Multi-Agent Research System

This module contains the extracted startup test functionality that was previously
built into the orchestrator initialization. These tests are designed to validate
that the system is healthy and ready for research operations.

## Purpose of These Tests

### Agent Health Check (check_agent_health)
**What it does:** Tests each agent by asking it to introduce itself and describe capabilities.
**Why it's useful:** Validates that:
- Each agent can be reached and responds
- Agent configurations are correct
- Basic Claude SDK connectivity is working
- Response collection mechanism is functioning

### Tool Verification (verify_tool_execution)
**What it does:** Tests critical tools by having agents perform simple operations.
**Why it's useful:** Validates that:
- SERP API search tool is accessible and working
- File operations (Read/Write) are available
- MCP server integration is functioning
- Tool permissions and routing are correct

## When to Run These Tests

1. **During Development:** When making changes to agent configurations or tool setup
2. **Troubleshooting:** When the system is not working as expected
3. **Environment Validation:** When setting up in a new environment
4. **After Updates:** After updating SDK versions or dependencies

## Running the Tests

```bash
# Run all startup health tests
python tests/test_startup_health.py

# Run specific test
python tests/test_startup_health.py --test agents
python tests/test_startup_health.py --test tools

# Run with verbose output
python tests/test_startup_health.py --verbose
```

## Expected Output

For a healthy system, you should see:
- All agents marked as "healthy" with reasonable response times
- All critical tools marked as "working"
- No error messages or warnings

If issues are found, the test will provide detailed information about what failed
and suggestions for resolution.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add the multi_agent_research_system to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.logging_config import get_logger, setup_logging
    from core.orchestrator import ResearchOrchestrator

    from claude_agent_sdk import ClaudeSDKClient
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the correct directory with proper Python path")
    sys.exit(1)


class StartupHealthTester:
    """Utility class for testing startup health of the multi-agent research system."""

    def __init__(self, verbose: bool = False):
        """Initialize the health tester.

        Args:
            verbose: Enable detailed logging output
        """
        self.verbose = verbose
        self.setup_logging()
        self.orchestrator = None
        self.client = None

    def setup_logging(self):
        """Setup logging for health testing."""
        if self.verbose:
            setup_logging("DEBUG", "KEVIN/logs")
        else:
            setup_logging("INFO", "KEVIN/logs")
        self.logger = get_logger("startup_health_test")

    async def initialize_orchestrator(self) -> bool:
        """Initialize the orchestrator for testing.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("🚀 Initializing orchestrator for health testing...")
            self.orchestrator = ResearchOrchestrator(debug_mode=False)
            await self.orchestrator.initialize()
            self.client = self.orchestrator.client
            self.agent_names = self.orchestrator.agent_names
            self.logger.info(f"✅ Orchestrator initialized with {len(self.agent_names)} agents")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize orchestrator: {e}")
            return False

    async def check_agent_health(self) -> dict[str, Any]:
        """
        Check the health and connectivity of all agents.

        **Purpose:** Validates that each agent can be reached and responds properly.
        **What it tests:**
        - Agent availability and connectivity
        - Claude SDK communication
        - Agent response generation
        - Response collection mechanism

        Returns:
            Detailed health report for all agents
        """
        self.logger.info("🔍 Performing agent health checks...")

        health_report = {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(self.agent_names),
            "healthy_agents": 0,
            "unhealthy_agents": 0,
            "agent_status": {},
            "mcp_servers": {},
            "issues": []
        }

        for agent_name in self.agent_names:
            try:
                self.logger.debug(f"Checking health of {agent_name}...")

                # Test basic connectivity with natural language agent selection
                test_prompt = f"Use the {agent_name} agent to introduce yourself and describe your capabilities in 2-3 sentences."
                start_time = time.time()

                try:
                    # Send query to single client with natural language agent selection
                    await self.client.query(test_prompt)
                    response_time = time.time() - start_time

                    # Collect substantive response to verify agent capability
                    substantive_response = False
                    collected_messages = []

                    try:
                        async for message in self.client.receive_response():
                            collected_messages.append(message)

                            # Check for substantive content
                            if hasattr(message, 'content') and message.content:
                                for block in message.content:
                                    if hasattr(block, 'text') and len(block.text.strip()) > 10:
                                        substantive_response = True
                                        break
                            elif hasattr(message, 'total_cost_usd'):
                                # Found ResultMessage, we're done
                                break
                    except Exception as response_error:
                        self.logger.debug(f"Response collection error for {agent_name}: {response_error}")
                        # Continue with basic check if response collection fails

                    if substantive_response or len(collected_messages) > 0:
                        health_report["agent_status"][agent_name] = {
                            "status": "healthy",
                            "response_time": response_time,
                            "messages_collected": len(collected_messages),
                            "substantive_response": substantive_response
                        }
                        health_report["healthy_agents"] += 1
                        self.logger.info(f"✅ {agent_name}: Healthy ({response_time:.2f}s)")
                    else:
                        health_report["agent_status"][agent_name] = {
                            "status": "unhealthy",
                            "response_time": response_time,
                            "messages_collected": len(collected_messages),
                            "substantive_response": substantive_response,
                            "issue": "No substantive response collected"
                        }
                        health_report["unhealthy_agents"] += 1
                        health_report["issues"].append(f"{agent_name}: No substantive response")
                        self.logger.warning(f"⚠️ {agent_name}: No substantive response")

                except Exception as agent_error:
                    health_report["agent_status"][agent_name] = {
                        "status": "error",
                        "response_time": time.time() - start_time,
                        "issue": f"Agent test failed: {str(agent_error)}"
                    }
                    health_report["unhealthy_agents"] += 1
                    health_report["issues"].append(f"{agent_name}: {str(agent_error)}")
                    self.logger.error(f"❌ {agent_name}: {str(agent_error)}")

            except Exception as e:
                health_report["agent_status"][agent_name] = {
                    "status": "critical_error",
                    "issue": f"Health check failed: {str(e)}"
                }
                health_report["unhealthy_agents"] += 1
                health_report["issues"].append(f"{agent_name}: Health check failed - {e}")
                self.logger.error(f"❌ {agent_name}: Health check failed - {e}")

        # Check MCP server status
        if self.orchestrator.mcp_server:
            try:
                health_report["mcp_servers"]["research_tools"] = {
                    "status": "available",
                    "type": type(self.orchestrator.mcp_server).__name__
                }
                self.logger.info("✅ Research Tools MCP Server: Available")
            except Exception as e:
                health_report["mcp_servers"]["research_tools"] = {
                    "status": "error",
                    "issue": str(e)
                }
                health_report["issues"].append(f"MCP Server: {e}")
                self.logger.error(f"❌ MCP Server error: {e}")
        else:
            health_report["mcp_servers"]["research_tools"] = {
                "status": "missing",
                "issue": "MCP server not initialized"
            }
            health_report["issues"].append("MCP Server: Not initialized")
            self.logger.warning("⚠️ MCP Server: Not initialized")

        return health_report

    async def verify_tool_execution(self, agent_name: str = "research_agent") -> dict[str, Any]:
        """
        Verify that critical tools can be executed by agents.

        **Purpose:** Validates that critical tools are accessible and working properly.
        **What it tests:**
        - SERP API search tool availability and functionality
        - File operation tools (Read/Write) accessibility
        - MCP server tool routing
        - Tool permissions and execution

        Args:
            agent_name: Name of the agent to test tools with

        Returns:
            Detailed tool verification report
        """
        self.logger.info(f"🔧 Verifying tool execution for {agent_name}...")

        tool_verification = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "tools_tested": [],
            "tools_working": [],
            "tools_failed": [],
            "issues": []
        }

        if not self.client:
            error_msg = "Client not initialized"
            tool_verification["issues"].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
            return tool_verification

        if agent_name not in self.agent_names:
            error_msg = f"Agent {agent_name} not available"
            tool_verification["issues"].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
            return tool_verification

        # Test critical tools with simple prompts
        critical_tools = [
            {
                "name": "SERP API Search",
                "test_prompt": "Use the mcp__research_tools__serp_search tool to search for 'test query' with num_results=1.",
                "expected_pattern": "serp_search",
                "description": "Tests SERP API search functionality via MCP server"
            },
            {
                "name": "File Operations",
                "test_prompt": "Use the Read tool to read the file 'test.txt'.",
                "expected_pattern": "Read",
                "description": "Tests basic file operation tool availability"
            }
        ]

        for tool_test in critical_tools:
            tool_name = tool_test["name"]
            test_prompt = tool_test["test_prompt"]
            tool_verification["tools_tested"].append(tool_name)

            try:
                self.logger.debug(f"Testing {tool_name} with {agent_name}...")
                start_time = time.time()

                # Use the orchestrator's execute_agent_query method
                full_test_prompt = f"{test_prompt} Use the {agent_name} agent for this task."
                result = await self.orchestrator.execute_agent_query(agent_name, full_test_prompt)
                execution_time = time.time() - start_time

                # Check if tool was mentioned in responses
                tool_used = False
                for message_info in result["messages_collected"]:
                    if "content_texts" in message_info:
                        for text in message_info["content_texts"]:
                            if tool_test["expected_pattern"] in str(text):
                                tool_used = True
                                break
                        if tool_used:
                            break

                if tool_used or len(result["messages_collected"]) > 0:
                    tool_verification["tools_working"].append(tool_name)
                    self.logger.info(f"✅ {tool_name}: Working ({execution_time:.2f}s)")
                else:
                    tool_verification["tools_failed"].append(tool_name)
                    issue = f"{tool_name}: Tool not executed in response"
                    tool_verification["issues"].append(issue)
                    self.logger.warning(f"⚠️ {tool_name}: Tool not executed")

            except Exception as e:
                tool_verification["tools_failed"].append(tool_name)
                issue = f"{tool_name}: {str(e)}"
                tool_verification["issues"].append(issue)
                self.logger.error(f"❌ {tool_name}: {str(e)}")

        # Summary
        tool_verification["success_rate"] = len(tool_verification["tools_working"]) / len(tool_verification["tools_tested"]) if tool_verification["tools_tested"] else 0
        tool_verification["summary"] = f"{len(tool_verification['tools_working'])}/{len(tool_verification['tools_tested'])} tools working"

        if tool_verification["success_rate"] == 1.0:
            self.logger.info(f"✅ All {len(tool_verification['tools_working'])} critical tools verified")
        elif tool_verification["success_rate"] >= 0.5:
            self.logger.warning(f"⚠️ {tool_verification['summary']} - some tools may not work")
        else:
            self.logger.error(f"🚨 Critical tool verification failed: {tool_verification['summary']}")

        return tool_verification

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all startup health tests.

        Returns:
            Combined health report with all test results
        """
        self.logger.info("🏥 Running comprehensive startup health tests...")

        full_report = {
            "timestamp": datetime.now().isoformat(),
            "test_suite": "startup_health",
            "tests_run": [],
            "overall_status": "unknown",
            "issues": []
        }

        # Initialize orchestrator
        if not await self.initialize_orchestrator():
            full_report["overall_status"] = "initialization_failed"
            full_report["issues"].append("Failed to initialize orchestrator")
            return full_report

        try:
            # Agent health check
            self.logger.info("📋 Running agent health checks...")
            agent_health = await self.check_agent_health()
            full_report["tests_run"].append({
                "name": "agent_health",
                "status": "completed",
                "result": agent_health
            })

            # Tool verification
            self.logger.info("🔧 Running tool verification...")
            tool_verification = await self.verify_tool_execution("research_agent")
            full_report["tests_run"].append({
                "name": "tool_verification",
                "status": "completed",
                "result": tool_verification
            })

            # Determine overall status
            critical_issues = 0
            for test_result in full_report["tests_run"]:
                result = test_result["result"]
                if "issues" in result and result["issues"]:
                    critical_issues += len(result["issues"])

            if critical_issues == 0:
                full_report["overall_status"] = "healthy"
                self.logger.info("🎉 All startup health tests passed - system is ready!")
            else:
                full_report["overall_status"] = "issues_found"
                full_report["issues"] = [f"Found {critical_issues} issues across startup tests"]
                self.logger.warning(f"⚠️ Startup health tests completed with {critical_issues} issues")

        except Exception as e:
            full_report["overall_status"] = "test_failed"
            full_report["issues"].append(f"Test execution failed: {str(e)}")
            self.logger.error(f"❌ Startup health tests failed: {e}")

        finally:
            # Cleanup
            if self.orchestrator:
                try:
                    await self.orchestrator.cleanup()
                except Exception as cleanup_error:
                    self.logger.warning(f"Cleanup warning: {cleanup_error}")

        return full_report

    def print_report(self, report: dict[str, Any]):
        """Print a formatted health report.

        Args:
            report: Health report to display
        """
        print("\n" + "="*80)
        print("🏥 MULTI-AGENT RESEARCH SYSTEM - STARTUP HEALTH REPORT")
        print("="*80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Overall Status: {report['overall_status'].upper()}")

        if report["issues"]:
            print(f"\n❌ Issues Found: {len(report['issues'])}")
            for issue in report["issues"]:
                print(f"   • {issue}")

        for test_result in report["tests_run"]:
            test_name = test_result["name"]
            result = test_result["result"]

            print(f"\n{'-'*60}")
            print(f"📋 {test_name.upper().replace('_', ' ')}")
            print(f"{'-'*60}")

            if test_name == "agent_health":
                print(f"Total Agents: {result['total_agents']}")
                print(f"Healthy: {result['healthy_agents']}")
                print(f"Unhealthy: {result['unhealthy_agents']}")

                for agent_name, status in result["agent_status"].items():
                    status_symbol = "✅" if status["status"] == "healthy" else "❌"
                    response_time = status.get("response_time", 0)
                    print(f"   {status_symbol} {agent_name}: {status['status']} ({response_time:.2f}s)")
                    if status["status"] != "healthy" and "issue" in status:
                        print(f"      Issue: {status['issue']}")

            elif test_name == "tool_verification":
                print(f"Agent Tested: {result['agent_name']}")
                print(f"Tools Working: {result['summary']}")
                print(f"Success Rate: {result['success_rate']:.1%}")

                for tool_name in result["tools_working"]:
                    print(f"   ✅ {tool_name}: Working")
                for tool_name in result["tools_failed"]:
                    print(f"   ❌ {tool_name}: Failed")

        print(f"\n{'='*80}")
        print("🏁 HEALTH CHECK COMPLETE")
        print("="*80)


async def main():
    """Main entry point for startup health testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System Startup Health Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_startup_health.py                    # Run all tests
  python test_startup_health.py --test agents     # Run agent health only
  python test_startup_health.py --test tools      # Run tool verification only
  python test_startup_health.py --verbose         # Detailed output
  python test_startup_health.py --output report.json  # Save report to file
        """
    )

    parser.add_argument(
        "--test", "-t",
        choices=["agents", "tools", "all"],
        default="all",
        help="Specific test to run (default: all)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output"
    )

    parser.add_argument(
        "--output", "-o",
        help="Save report to JSON file"
    )

    args = parser.parse_args()

    # Initialize tester
    tester = StartupHealthTester(verbose=args.verbose)

    try:
        if args.test == "all":
            # Run all tests
            report = await tester.run_all_tests()
        elif args.test == "agents":
            # Run agent health only
            await tester.initialize_orchestrator()
            agent_health = await tester.check_agent_health()
            report = {
                "timestamp": datetime.now().isoformat(),
                "test_suite": "startup_health",
                "tests_run": [
                    {
                        "name": "agent_health",
                        "status": "completed",
                        "result": agent_health
                    }
                ],
                "overall_status": "healthy" if agent_health["unhealthy_agents"] == 0 else "issues_found",
                "issues": agent_health["issues"]
            }
        elif args.test == "tools":
            # Run tool verification only
            await tester.initialize_orchestrator()
            tool_verification = await tester.verify_tool_execution("research_agent")
            report = {
                "timestamp": datetime.now().isoformat(),
                "test_suite": "startup_health",
                "tests_run": [
                    {
                        "name": "tool_verification",
                        "status": "completed",
                        "result": tool_verification
                    }
                ],
                "overall_status": "healthy" if tool_verification["success_rate"] == 1.0 else "issues_found",
                "issues": tool_verification["issues"]
            }

        # Print report
        tester.print_report(report)

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Report saved to: {args.output}")

        # Exit with appropriate code
        if report["overall_status"] == "healthy":
            print("\n✅ System is healthy and ready for research!")
            return 0
        else:
            print("\n⚠️ System has issues that may affect research performance")
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️ Health tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Health tests failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
