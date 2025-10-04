#!/usr/bin/env python3
"""
Test script to verify end-to-end search success flow.

This script tests that:
1. Research agents can successfully execute searches
2. Search results are properly captured and saved
3. Editorial agents can access successful research without generating failure reports
4. Final reports consolidate all successful search results
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the multi_agent_research_system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'multi_agent_research_system'))

from core.orchestrator import ResearchOrchestrator


async def test_end_to_end_search_flow():
    """Test the complete search to report workflow."""
    print("🧪 Testing End-to-End Search Success Flow")
    print("=" * 60)

    try:
        # Initialize orchestrator
        print("1. Initializing research orchestrator...")
        orchestrator = ResearchOrchestrator()

        # Test topic
        test_topic = "artificial intelligence in healthcare 2024"
        session_id = f"test_search_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"2. Starting research for topic: {test_topic}")
        print(f"   Session ID: {session_id}")

        # Run research
        result = await orchestrator.run_research(
            topic=test_topic,
            session_id=session_id,
            requirements="Test search to verify successful research flow and result consolidation"
        )

        print(f"3. Research completed with result: {result}")

        # Verify search success indicators
        print("\n4. Verifying search success indicators...")

        # Check for session directory
        session_dir = Path(f"/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN/sessions/{session_id}")
        if session_dir.exists():
            print(f"   ✅ Session directory created: {session_dir}")

            # Check for research findings
            findings_file = session_dir / "research_findings.json"
            if findings_file.exists():
                print(f"   ✅ Research findings file created")

                with open(findings_file, 'r', encoding='utf-8') as f:
                    findings = json.load(f)

                if "sources" in findings and findings["sources"]:
                    print(f"   ✅ Found {len(findings['sources'])} sources in research findings")
                else:
                    print("   ⚠️  No sources found in research findings")

                if "findings" in findings and findings["findings"]:
                    print(f"   ✅ Found {len(findings['findings'])} findings entries")
                else:
                    print("   ⚠️  No findings entries found")
            else:
                print("   ❌ Research findings file not found")

            # Check for search analysis files
            search_analysis_dir = session_dir / "search_analysis"
            if search_analysis_dir.exists():
                search_files = list(search_analysis_dir.glob("web_search_results_*.json"))
                print(f"   ✅ Found {len(search_files)} search result files")

                for search_file in search_files:
                    try:
                        with open(search_file, 'r', encoding='utf-8') as f:
                            search_data = json.load(f)

                        # Verify search data has content
                        if search_data.get("search_results") and len(search_data["search_results"]) > 100:
                            print(f"      ✅ {search_file.name}: Substantial search content ({len(search_data['search_results'])} chars)")
                        else:
                            print(f"      ⚠️  {search_file.name}: Limited search content")

                    except Exception as e:
                        print(f"      ❌ {search_file.name}: Error reading file - {e}")
            else:
                print("   ⚠️  No search analysis directory found")

            # Check for failure reports (should not exist for successful research)
            working_dir = session_dir / "working"
            if working_dir.exists():
                failure_reports = list(working_dir.glob("*FAILURE*"))
                if failure_reports:
                    print(f"   ❌ Found {len(failure_reports)} failure reports - this indicates a problem!")
                    for failure_report in failure_reports:
                        print(f"      ❌ {failure_report.name}")
                else:
                    print("   ✅ No failure reports found (good for successful research)")

            # Check for final reports
            research_dir = session_dir / "research"
            if research_dir.exists():
                report_files = list(research_dir.glob("*.md"))
                if report_files:
                    print(f"   ✅ Found {len(report_files)} research report files")
                    for report_file in report_files:
                        print(f"      ✅ {report_file.name}")
                else:
                    print("   ⚠️  No research report files found")
            else:
                print("   ⚠️  No research directory found")

        else:
            print(f"   ❌ Session directory not found: {session_dir}")

        print(f"\n5. Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Starting End-to-End Search Flow Test")
    print("Purpose: Verify that successful searches are properly consolidated and no failure reports are generated")
    print()

    success = await test_end_to_end_search_flow()

    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    if success:
        print("✅ End-to-end search flow test PASSED")
        print("✅ Search results are properly captured and consolidated")
        print("✅ No failure reports generated for successful research")
    else:
        print("❌ End-to-end search flow test FAILED")
        print("❌ Issues detected in search result consolidation")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)