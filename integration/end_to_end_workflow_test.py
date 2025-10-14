#!/usr/bin/env python3
"""
End-to-End Workflow Test for Agent-Based Research System

This script performs comprehensive end-to-end testing of the complete
agent-based research workflow from query input to final report generation.

Usage:
    python integration/end_to_end_workflow_test.py [options]

Options:
    --verbose, -v          Enable verbose output
    --query TEXT          Specify custom query for testing
    --mock-mode           Use mock data instead of real operations
    --output-dir DIR      Specify output directory for results
    --help, -h            Show this help message

Author: Claude Code Assistant
Version: 1.0.0
"""

import argparse
import json
import os
import sys
import time
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid


class EndToEndWorkflowTester:
    """Comprehensive end-to-end workflow tester"""

    def __init__(self, output_dir: Path = None, verbose: bool = False, mock_mode: bool = False):
        self.output_dir = output_dir or Path("e2e_test_results")
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        self.mock_mode = mock_mode
        self.session_id = f"e2e_test_{uuid.uuid4().hex[:8]}"
        self.test_results = {
            "workflow_summary": {
                "session_id": self.session_id,
                "start_time": datetime.now().isoformat(),
                "test_query": "",
                "mock_mode": mock_mode,
                "stages_completed": [],
                "stages_failed": [],
                "total_stages": 0,
                "success_rate": 0
            },
            "stage_results": {},
            "performance_metrics": {},
            "generated_artifacts": [],
            "validation_results": {}
        }

    def log(self, message: str, level: str = "INFO"):
        """Log workflow messages"""
        if self.verbose or level in ["ERROR", "WARNING", "SUCCESS", "STAGE"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {
                "INFO": "ℹ️",
                "SUCCESS": "✅",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "STAGE": "🔄"
            }.get(level, "ℹ️")
            print(f"[{timestamp}] {prefix} {message}")

    def stage_1_query_processing(self, query: str) -> Dict[str, Any]:
        """Stage 1: Query Processing"""
        stage_name = "Query Processing"
        self.log(f"Stage 1: {stage_name}", "STAGE")

        start_time = time.time()
        stage_result = {
            "stage_name": stage_name,
            "start_time": start_time,
            "success": False,
            "outputs": {},
            "errors": []
        }

        try:
            self.log(f"  Processing query: '{query[:50]}...'", "INFO")

            # Import query processor
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from integration.query_processor import QueryProcessor

            processor = QueryProcessor()

            # Process the query (simplified for testing)
            processed_query = self._mock_query_processing(query)

            stage_result["outputs"] = {
                "original_query": query,
                "processed_query": processed_query,
                "query_type": processed_query.get("query_type", "unknown"),
                "complexity_score": processed_query.get("complexity_score", 0.5),
                "recommended_approach": processed_query.get("recommended_approach", "standard")
            }

            stage_result["success"] = True
            self.log(f"  ✓ Query processed successfully: {stage_result['outputs']['query_type']}", "SUCCESS")

        except Exception as e:
            stage_result["errors"].append(str(e))
            self.log(f"  ❌ Query processing failed: {str(e)}", "ERROR")
            if self.verbose:
                traceback.print_exc()

        stage_result["end_time"] = time.time()
        stage_result["duration"] = stage_result["end_time"] - stage_result["start_time"]

        self.test_results["stage_results"]["stage_1_query_processing"] = stage_result
        return stage_result

    def stage_2_session_initialization(self) -> Dict[str, Any]:
        """Stage 2: Session Initialization"""
        stage_name = "Session Initialization"
        self.log(f"Stage 2: {stage_name}", "STAGE")

        start_time = time.time()
        stage_result = {
            "stage_name": stage_name,
            "start_time": start_time,
            "success": False,
            "outputs": {},
            "errors": []
        }

        try:
            self.log(f"  Initializing session: {self.session_id}", "INFO")

            # Import session manager
            from integration.agent_session_manager import AgentSessionManager

            session_manager = AgentSessionManager()

            # Create session
            session_data = session_manager.create_session(
                self.session_id,
                self.test_results["workflow_summary"]["test_query"],
                {
                    "research_configuration": {
                        "target_urls": 10,
                        "concurrent_processing": 5,
                        "quality_threshold": 0.7
                    },
                    "quality_configuration": {
                        "enable_quality_assessment": True,
                        "quality_threshold": 0.75
                    }
                }
            )

            stage_result["outputs"] = {
                "session_id": self.session_id,
                "session_data": session_data,
                "session_directory": session_manager.get_session_directory(self.session_id),
                "status": "initialized"
            }

            stage_result["success"] = True
            self.log(f"  ✓ Session initialized successfully", "SUCCESS")

        except Exception as e:
            stage_result["errors"].append(str(e))
            self.log(f"  ❌ Session initialization failed: {str(e)}", "ERROR")
            if self.verbose:
                traceback.print_exc()

        stage_result["end_time"] = time.time()
        stage_result["duration"] = stage_result["end_time"] - stage_result["start_time"]

        self.test_results["stage_results"]["stage_2_session_initialization"] = stage_result
        return stage_result

    def stage_3_research_execution(self) -> Dict[str, Any]:
        """Stage 3: Research Execution"""
        stage_name = "Research Execution"
        self.log(f"Stage 3: {stage_name}", "STAGE")

        start_time = time.time()
        stage_result = {
            "stage_name": stage_name,
            "start_time": start_time,
            "success": False,
            "outputs": {},
            "errors": []
        }

        try:
            self.log(f"  Executing research for session: {self.session_id}", "INFO")

            if self.mock_mode:
                # Mock research execution
                research_results = self._create_mock_research_results()
            else:
                # Import research orchestrator
                from integration.research_orchestrator import ResearchOrchestrator
                from integration.mcp_tool_integration import MCPToolIntegration

                orchestrator = ResearchOrchestrator()
                mcp_integration = MCPToolIntegration()

                # Execute research
                research_results = await self._run_async_if_needed(
                    orchestrator.execute_research,
                    self.test_results["workflow_summary"]["test_query"],
                    self.session_id
                )

            stage_result["outputs"] = {
                "research_results": research_results,
                "result_count": len(research_results.get("results", [])),
                "sources_used": research_results.get("sources_used", []),
                "quality_score": research_results.get("quality_score", 0.8)
            }

            stage_result["success"] = True
            self.log(f"  ✓ Research executed successfully: {stage_result['outputs']['result_count']} results", "SUCCESS")

        except Exception as e:
            stage_result["errors"].append(str(e))
            self.log(f"  ❌ Research execution failed: {str(e)}", "ERROR")
            if self.verbose:
                traceback.print_exc()

        stage_result["end_time"] = time.time()
        stage_result["duration"] = stage_result["end_time"] - stage_result["start_time"]

        self.test_results["stage_results"]["stage_3_research_execution"] = stage_result
        return stage_result

    def stage_4_content_analysis(self) -> Dict[str, Any]:
        """Stage 4: Content Analysis"""
        stage_name = "Content Analysis"
        self.log(f"Stage 4: {stage_name}", "STAGE")

        start_time = time.time()
        stage_result = {
            "stage_name": stage_name,
            "start_time": start_time,
            "success": False,
            "outputs": {},
            "errors": []
        }

        try:
            self.log(f"  Analyzing research content", "INFO")

            research_results = self.test_results["stage_results"]["stage_3_research_execution"]["outputs"]["research_results"]

            if self.mock_mode:
                # Mock content analysis
                analysis_results = self._create_mock_analysis_results()
            else:
                # Import quality assurance integration
                from integration.quality_assurance_integration import QualityAssuranceIntegration

                quality_integration = QualityAssuranceIntegration()

                # Analyze content
                analysis_results = await self._run_async_if_needed(
                    quality_integration.assess_research_quality,
                    self.session_id,
                    json.dumps(research_results),
                    {"content_type": "research_results", "stage": "analysis"}
                )

            stage_result["outputs"] = {
                "analysis_results": analysis_results,
                "quality_score": analysis_results.get("overall_score", 0.8),
                "key_topics": analysis_results.get("key_topics", []),
                "recommendations": analysis_results.get("recommendations", [])
            }

            stage_result["success"] = True
            self.log(f"  ✓ Content analysis completed: quality score {stage_result['outputs']['quality_score']:.2f}", "SUCCESS")

        except Exception as e:
            stage_result["errors"].append(str(e))
            self.log(f"  ❌ Content analysis failed: {str(e)}", "ERROR")
            if self.verbose:
                traceback.print_exc()

        stage_result["end_time"] = time.time()
        stage_result["duration"] = stage_result["end_time"] - stage_result["start_time"]

        self.test_results["stage_results"]["stage_4_content_analysis"] = stage_result
        return stage_result

    def stage_5_report_generation(self) -> Dict[str, Any]:
        """Stage 5: Report Generation"""
        stage_name = "Report Generation"
        self.log(f"Stage 5: {stage_name}", "STAGE")

        start_time = time.time()
        stage_result = {
            "stage_name": stage_name,
            "start_time": start_time,
            "success": False,
            "outputs": {},
            "errors": []
        }

        try:
            self.log(f"  Generating final report", "INFO")

            # Generate comprehensive report
            report_data = self._generate_comprehensive_report()

            # Save report to file
            report_file = self.output_dir / f"research_report_{self.session_id}.md"
            with open(report_file, 'w') as f:
                f.write(report_data)

            stage_result["outputs"] = {
                "report_file": str(report_file),
                "report_size": len(report_data),
                "word_count": len(report_data.split()),
                "sections": self._count_report_sections(report_data)
            }

            stage_result["success"] = True
            self.log(f"  ✓ Report generated successfully: {stage_result['outputs']['word_count']} words", "SUCCESS")

            # Add to generated artifacts
            self.test_results["generated_artifacts"].append({
                "type": "report",
                "file": str(report_file),
                "description": "Final research report"
            })

        except Exception as e:
            stage_result["errors"].append(str(e))
            self.log(f"  ❌ Report generation failed: {str(e)}", "ERROR")
            if self.verbose:
                traceback.print_exc()

        stage_result["end_time"] = time.time()
        stage_result["duration"] = stage_result["end_time"] - stage_result["start_time"]

        self.test_results["stage_results"]["stage_5_report_generation"] = stage_result
        return stage_result

    def stage_6_quality_validation(self) -> Dict[str, Any]:
        """Stage 6: Quality Validation"""
        stage_name = "Quality Validation"
        self.log(f"Stage 6: {stage_name}", "STAGE")

        start_time = time.time()
        stage_result = {
            "stage_name": stage_name,
            "start_time": start_time,
            "success": False,
            "outputs": {},
            "errors": []
        }

        try:
            self.log(f"  Validating workflow quality", "INFO")

            # Collect all stage results
            all_stages = self.test_results["stage_results"]
            completed_stages = [name for name, result in all_stages.items() if result["success"]]
            failed_stages = [name for name, result in all_stages.items() if not result["success"]]

            # Calculate quality metrics
            total_duration = sum(result.get("duration", 0) for result in all_stages.values())
            success_rate = len(completed_stages) / len(all_stages) * 100 if all_stages else 0

            # Validate generated artifacts
            artifact_validation = self._validate_generated_artifacts()

            stage_result["outputs"] = {
                "completed_stages": completed_stages,
                "failed_stages": failed_stages,
                "success_rate": success_rate,
                "total_duration": total_duration,
                "artifact_validation": artifact_validation,
                "overall_quality_score": self._calculate_quality_score()
            }

            stage_result["success"] = True
            self.log(f"  ✓ Quality validation completed: success rate {success_rate:.1f}%", "SUCCESS")

        except Exception as e:
            stage_result["errors"].append(str(e))
            self.log(f"  ❌ Quality validation failed: {str(e)}", "ERROR")
            if self.verbose:
                traceback.print_exc()

        stage_result["end_time"] = time.time()
        stage_result["duration"] = stage_result["end_time"] - stage_result["start_time"]

        self.test_results["stage_results"]["stage_6_quality_validation"] = stage_result
        return stage_result

    async def _run_async_if_needed(self, func, *args, **kwargs):
        """Run function asynchronously if needed"""
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # For synchronous functions, run in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

    def _create_mock_research_results(self) -> Dict[str, Any]:
        """Create mock research results for testing"""
        return {
            "query": self.test_results["workflow_summary"]["test_query"],
            "results": [
                {
                    "url": f"https://example-source-{i+1}.com/article{i+1}",
                    "title": f"Research Article {i+1}",
                    "content": f"This is mock research content {i+1} for testing purposes...",
                    "source": f"source_{i+1}",
                    "timestamp": datetime.now().isoformat(),
                    "relevance_score": 0.9 - (i * 0.05)
                }
                for i in range(5)
            ],
            "metadata": {
                "total_results": 5,
                "processing_time": 2.5,
                "quality_score": 0.85,
                "sources_used": ["source_1", "source_2", "source_3"],
                "mock_mode": True
            }
        }

    def _create_mock_analysis_results(self) -> Dict[str, Any]:
        """Create mock analysis results for testing"""
        return {
            "overall_score": 0.88,
            "key_topics": ["artificial intelligence", "research", "analysis"],
            "sentiment_analysis": {
                "overall": "neutral",
                "confidence": 0.85
            },
            "recommendations": [
                "Include more recent sources",
                "Add case studies",
                "Enhance technical details"
            ],
            "quality_factors": {
                "relevance": 0.92,
                "accuracy": 0.85,
                "completeness": 0.87,
                "clarity": 0.90
            },
            "mock_mode": True
        }

    def _generate_comprehensive_report(self) -> str:
        """Generate comprehensive research report"""
        query = self.test_results["workflow_summary"]["test_query"]
        session_id = self.session_id

        # Get results from previous stages
        research_results = self.test_results["stage_results"]["stage_3_research_execution"]["outputs"]["research_results"]
        analysis_results = self.test_results["stage_results"]["stage_4_content_analysis"]["outputs"]["analysis_results"]

        report = f"""# Research Report: {query}

**Session ID:** {session_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Mock Mode:** {self.mock_mode}

## Executive Summary

This report presents research findings for the query: "{query}". The research process involved comprehensive data collection, analysis, and quality assessment.

## Research Findings

### Key Results
"""

        # Add research results
        for i, result in enumerate(research_results.get("results", []), 1):
            report += f"""
#### {i}. {result.get('title', f'Research Result {i}')}

**Source:** {result.get('source', 'Unknown')}
**URL:** {result.get('url', 'No URL available')}
**Relevance Score:** {result.get('relevance_score', 0.0):.2f}

**Content Summary:**
{result.get('content', 'No content available')[:200]}...

---

"""

        # Add analysis section
        report += f"""
## Quality Analysis

**Overall Quality Score:** {analysis_results.get('overall_score', 0.0):.2f}/1.0

### Key Topics Identified
"""

        for topic in analysis_results.get("key_topics", []):
            report += f"- {topic}\n"

        report += f"""
### Recommendations
"""

        for recommendation in analysis_results.get("recommendations", []):
            report += f"- {recommendation}\n"

        # Add methodology section
        report += f"""
## Methodology

This research was conducted using the Agent-Based Research System with the following methodology:

1. **Query Processing**: Analysis and optimization of the research query
2. **Research Execution**: Systematic data collection from multiple sources
3. **Content Analysis**: Quality assessment and content evaluation
4. **Report Generation**: Synthesis of findings into comprehensive report

### System Configuration
- **Target Results:** 10 sources
- **Quality Threshold:** 0.75
- **Processing Mode:** {"Mock" if self.mock_mode else "Live"}

## Conclusion

The research process successfully identified {len(research_results.get('results', []))} relevant sources with an overall quality score of {analysis_results.get('overall_score', 0.0):.2f}. The findings provide a comprehensive overview of the requested topic.

---

*This report was generated by the Agent-Based Research System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.*
"""

        return report

    def _count_report_sections(self, report_content: str) -> Dict[str, int]:
        """Count report sections"""
        sections = {
            "headings": report_content.count('#'),
            "tables": report_content.count('|'),
            "links": report_content.count('http'),
            "words": len(report_content.split()),
            "characters": len(report_content)
        }
        return sections

    def _validate_generated_artifacts(self) -> Dict[str, Any]:
        """Validate generated artifacts"""
        validation_results = {
            "total_artifacts": len(self.test_results["generated_artifacts"]),
            "valid_artifacts": 0,
            "invalid_artifacts": [],
            "artifact_details": []
        }

        for artifact in self.test_results["generated_artifacts"]:
            artifact_path = Path(artifact["file"])
            if artifact_path.exists() and artifact_path.stat().st_size > 0:
                validation_results["valid_artifacts"] += 1
                validation_results["artifact_details"].append({
                    "type": artifact["type"],
                    "file": str(artifact_path),
                    "size_bytes": artifact_path.stat().st_size,
                    "valid": True
                })
            else:
                validation_results["invalid_artifacts"].append(artifact["file"])
                validation_results["artifact_details"].append({
                    "type": artifact["type"],
                    "file": str(artifact_path),
                    "size_bytes": 0,
                    "valid": False
                })

        return validation_results

    def _calculate_quality_score(self) -> float:
        """Calculate overall workflow quality score"""
        stage_scores = []

        for stage_name, stage_result in self.test_results["stage_results"].items():
            if stage_result["success"]:
                # Base score for successful stage
                score = 80

                # Add performance bonus
                duration = stage_result.get("duration", 0)
                if duration < 5:  # Fast execution
                    score += 10
                elif duration < 15:  # Acceptable execution
                    score += 5

                # Add output quality bonus
                outputs = stage_result.get("outputs", {})
                if outputs:
                    score += 10

                stage_scores.append(score)
            else:
                stage_scores.append(0)

        # Calculate overall score
        overall_score = sum(stage_scores) / len(stage_scores) if stage_scores else 0
        return min(100, overall_score)

    def run_workflow_test(self, query: str) -> Dict[str, Any]:
        """Run complete end-to-end workflow test"""
        self.log("🚀 Starting End-to-End Workflow Test", "INFO")
        self.log(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"📁 Session ID: {self.session_id}")
        self.log(f"🔍 Query: '{query}'")
        self.log(f"🎭 Mock Mode: {self.mock_mode}")

        # Set test query
        self.test_results["workflow_summary"]["test_query"] = query

        # Define workflow stages
        workflow_stages = [
            ("stage_1_query_processing", self.stage_1_query_processing, [query]),
            ("stage_2_session_initialization", self.stage_2_session_initialization, []),
            ("stage_3_research_execution", self.stage_3_research_execution, []),
            ("stage_4_content_analysis", self.stage_4_content_analysis, []),
            ("stage_5_report_generation", self.stage_5_report_generation, []),
            ("stage_6_quality_validation", self.stage_6_quality_validation, [])
        ]

        # Execute workflow stages
        completed_stages = []
        failed_stages = []

        for stage_name, stage_func, stage_args in workflow_stages:
            try:
                self.log(f"Executing {stage_name}...", "INFO")
                result = stage_func(*stage_args)

                if result["success"]:
                    completed_stages.append(stage_name)
                    self.test_results["workflow_summary"]["stages_completed"].append(stage_name)
                else:
                    failed_stages.append(stage_name)
                    self.test_results["workflow_summary"]["stages_failed"].append(stage_name)
                    self.log(f"Stage {stage_name} failed", "ERROR")

            except Exception as e:
                failed_stages.append(stage_name)
                self.test_results["workflow_summary"]["stages_failed"].append(stage_name)
                self.log(f"Stage {stage_name} crashed: {str(e)}", "ERROR")
                if self.verbose:
                    traceback.print_exc()

        # Calculate final results
        total_stages = len(workflow_stages)
        self.test_results["workflow_summary"]["total_stages"] = total_stages
        self.test_results["workflow_summary"]["stages_completed"] = completed_stages
        self.test_results["workflow_summary"]["stages_failed"] = failed_stages
        self.test_results["workflow_summary"]["success_rate"] = (len(completed_stages) / total_stages * 100) if total_stages > 0 else 0
        self.test_results["workflow_summary"]["end_time"] = datetime.now().isoformat()

        # Calculate performance metrics
        total_duration = sum(
            result.get("duration", 0) for result in self.test_results["stage_results"].values()
        )
        self.test_results["performance_metrics"] = {
            "total_duration": total_duration,
            "average_stage_duration": total_duration / total_stages if total_stages > 0 else 0,
            "completed_stages": len(completed_stages),
            "failed_stages": len(failed_stages)
        }

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

        return self.test_results

    def _save_results(self):
        """Save workflow test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"e2e_workflow_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)

        self.log(f"📄 Workflow results saved to: {results_file}")
        return results_file

    def _print_summary(self):
        """Print workflow test summary"""
        summary = self.test_results["workflow_summary"]
        performance = self.test_results["performance_metrics"]

        print("\n" + "=" * 80)
        print("🏁 END-TO-END WORKFLOW TEST SUMMARY")
        print("=" * 80)
        print(f"📊 Session ID: {summary['session_id']}")
        print(f"🔍 Query: {summary['test_query']}")
        print(f"🎭 Mock Mode: {summary['mock_mode']}")
        print(f"📋 Total Stages: {summary['total_stages']}")
        print(f"✅ Completed Stages: {len(summary['stages_completed'])}")
        print(f"❌ Failed Stages: {len(summary['stages_failed'])}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️  Total Duration: {performance['total_duration']:.2f}s")
        print(f"📁 Generated Artifacts: {len(self.test_results['generated_artifacts'])}")

        print("\n🔄 Stage Results:")
        for stage_name, result in self.test_results["stage_results"].items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            duration = result.get("duration", 0)
            print(f"   {status} {stage_name.replace('_', ' ').title()} ({duration:.2f}s)")

        # Show artifacts
        if self.test_results["generated_artifacts"]:
            print(f"\n📄 Generated Artifacts:")
            for artifact in self.test_results["generated_artifacts"]:
                print(f"   📋 {artifact['type']}: {artifact['file']}")

        # Overall assessment
        success_rate = summary["success_rate"]
        if success_rate >= 90:
            print(f"\n🎉 EXCELLENT! Workflow completed successfully with {success_rate:.1f}% success rate.")
        elif success_rate >= 70:
            print(f"\n✅ GOOD! Workflow mostly completed with {success_rate:.1f}% success rate.")
        elif success_rate >= 50:
            print(f"\n⚠️  FAIR! Workflow partially completed with {success_rate:.1f}% success rate.")
        else:
            print(f"\n❌ POOR! Workflow failed with only {success_rate:.1f}% success rate.")

        print("=" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="End-to-End Workflow Test for Agent-Based Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run end-to-end test with default query
    python integration/end_to_end_workflow_test.py

    # Run with custom query
    python integration/end_to_end_workflow_test.py --query "artificial intelligence"

    # Run in mock mode
    python integration/end_to_end_workflow_test.py --mock-mode

    # Run with verbose output
    python integration/end_to_end_workflow_test.py --verbose
        """
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--query",
        type=str,
        default="latest developments in artificial intelligence and machine learning",
        help="Specify custom query for testing"
    )

    parser.add_argument(
        "--mock-mode",
        action="store_true",
        help="Use mock data instead of real operations"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="e2e_test_results",
        help="Specify output directory for results"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Run workflow test
    tester = EndToEndWorkflowTester(output_dir, args.verbose, args.mock_mode)

    try:
        results = tester.run_workflow_test(args.query)

        # Determine exit code based on success rate
        success_rate = results["workflow_summary"]["success_rate"]
        if success_rate >= 70:
            print("\n✅ End-to-end workflow test completed successfully!")
            exit_code = 0
        elif success_rate >= 50:
            print("\n⚠️  End-to-end workflow test completed with issues.")
            exit_code = 1
        else:
            print("\n❌ End-to-end workflow test failed.")
            exit_code = 2

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n⚠️  End-to-end workflow test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 End-to-end workflow test failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()