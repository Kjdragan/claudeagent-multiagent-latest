#!/usr/bin/env python3
"""
Test script for the enhanced multi-agent research workflow.

This script demonstrates the complete multi-agent workflow:
Research → Report → Editorial → Enhanced Report
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from main_comprehensive_research import ComprehensiveResearchCLI


async def test_multi_agent_workflow():
    """Test the complete multi-agent workflow"""

    print("🚀 Testing Enhanced Multi-Agent Research Workflow")
    print("=" * 60)

    # Initialize CLI
    cli = ComprehensiveResearchCLI()

    # Setup logging
    cli.setup_logging("INFO")

    # Test query
    test_query = "latest developments in artificial intelligence"

    try:
        # Initialize SDK client
        await cli.initialize_sdk_client()

        # Initialize system components
        await cli.initialize_system_components()

        print(f"🔍 Testing query: {test_query}")
        print()

        # Create user requirements
        user_requirements = {
            "depth": "Comprehensive Analysis",
            "audience": "General",
            "format": "Detailed Report",
            "mode": "comprehensive",
            "target_results": 10,
            "session_id": None,
            "debug_mode": True,
            "dry_run": False
        }

        # Execute the multi-agent workflow
        result = await cli.process_query(
            query=test_query,
            mode="comprehensive",
            num_results=10,
            user_requirements=user_requirements
        )

        # Display results
        print("🎉 Multi-Agent Workflow Results")
        print("=" * 40)
        print(f"Status: {result['status']}")
        print(f"Session ID: {result['session_id']}")
        print(f"Query: {result['query']}")
        print(f"Total Time: {result['total_time']:.2f} seconds")

        if result['status'] == 'success':
            workflow_summary = result['workflow_result']['workflow_summary']

            print("\n📊 Workflow Summary:")
            print(f"Total Duration: {workflow_summary['total_duration']:.2f} seconds")

            print("\n⏱️ Stage Durations:")
            for stage, duration in workflow_summary['stage_durations'].items():
                print(f"  {stage}: {duration:.2f} seconds")

            print("\n📁 Files Generated:")
            for file_type, filepath in workflow_summary['files_generated'].items():
                if filepath:
                    print(f"  {file_type}: {filepath}")

            print("\n📈 Quality Metrics:")
            for metric, value in workflow_summary['quality_metrics'].items():
                print(f"  {metric}: {value}")

            print("\n✅ Workflow Stages:")
            for stage, info in workflow_summary['workflow_stages'].items():
                status = info['status']
                if status == 'completed':
                    print(f"  {stage}: ✅ {status.upper()}")
                elif status == 'failed':
                    print(f"  {stage}: ❌ {status.upper()}")
                else:
                    print(f"  {stage}: ⏳ {status.upper()}")

            # Display final report location
            final_result = result['workflow_result']['final_result']
            final_report_path = final_result.get('final_report_path')
            if final_report_path:
                print(f"\n📄 Final Report: {final_report_path}")
                print("🎯 Enhanced multi-agent workflow completed successfully!")

        else:
            print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🧪 Enhanced Multi-Agent Research System Test")
    print("Testing complete workflow: Research → Report → Editorial → Enhanced Report")
    print()

    asyncio.run(test_multi_agent_workflow())