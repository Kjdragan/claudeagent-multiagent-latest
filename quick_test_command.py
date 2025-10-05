#!/usr/bin/env python3
"""
Quick test command for quality system with reduced scope.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multi_agent_research_system.core.orchestrator import ResearchOrchestrator


async def quick_test():
    """Quick test with reduced research scope and shorter report."""

    # Initialize orchestrator
    orchestrator = ResearchOrchestrator(debug_mode=True)

    # Test topic - current events
    topic = "US military response to Venezuela boat attacks"
    session_id = "quick_test_venezuela"

    # User requirements for quick testing
    user_requirements = {
        "scope": "limited",           # Reduced research scope
        "report_length": "brief",     # Shorter report
        "sources_needed": 3,          # Fewer sources
        "depth": "overview",          # Not comprehensive
        "style": "news_summary",      # Quick news format
        "use_multimedia_exclusion": True,  # Faster crawling
        "quality_threshold": 70,      # Lower threshold for quicker completion
        "max_enhancement_cycles": 1   # Limit enhancement cycles
    }

    print("🚀 Starting Quick Quality System Test")
    print(f"📰 Topic: {topic}")
    print(f"🎯 Scope: Limited research, brief report")
    print(f"⚡ Optimizations: Multimedia exclusion, lower quality threshold")
    print("-" * 50)

    try:
        # Initialize session
        orchestrator.active_sessions[session_id] = {
            "topic": topic,
            "user_requirements": user_requirements
        }

        # Use the NEW quality-gated workflow
        result = await orchestrator.execute_quality_gated_research_workflow(
            session_id=session_id,
            topic=topic,
            user_requirements=user_requirements
        )

        print("\n" + "="*60)
        print("🎉 QUICK TEST RESULTS")
        print("="*60)

        if result.get("success", False):
            print("✅ Quality-gated workflow completed successfully!")
            print(f"📊 Final Quality Score: {result.get('quality_assessment', {}).get('overall_score', 'N/A')}/100")

            # Show workflow stages executed
            workflow_session = result.get('workflow_session', {})
            if 'stages' in workflow_session:
                print("\n🔄 Workflow Stages:")
                for stage_name, stage_data in workflow_session['stages'].items():
                    status = stage_data.get('status', 'unknown')
                    if stage_data.get('quality_metrics'):
                        quality = stage_data['quality_metrics'].get('overall_score', 'N/A')
                        print(f"  • {stage_name}: {status} (Quality: {quality})")
                    else:
                        print(f"  • {stage_name}: {status}")

            # Show key findings
            results = result.get('results', {})
            if 'research' in results:
                print(f"\n📰 Key Research Findings:")
                research_data = results['research']
                if isinstance(research_data, dict) and 'data' in research_data:
                    data = research_data['data']
                    if 'findings' in data and data['findings']:
                        for i, finding in enumerate(data['findings'][:3], 1):  # Top 3 findings
                            print(f"  {i}. {finding}")

                if isinstance(research_data, dict) and 'data' in research_data:
                    data = research_data['data']
                    if 'sources' in data and data['sources']:
                        print(f"\n📚 Sources Found: {len(data['sources'])}")
                        for source in data['sources'][:3]:  # Top 3 sources
                            print(f"  • {source}")

            print(f"\n⏱️  Session completed with quality management")
            print(f"🔧 Quality gates, enhancement, and monitoring active")

        else:
            print("❌ Quality-gated workflow failed")
            print(f"Error: {result.get('error', 'Unknown error')}")

            # Show what stages were attempted
            workflow_session = result.get('workflow_session', {})
            if 'stages' in workflow_session:
                print("\n🔄 Stages Attempted:")
                for stage_name, stage_data in workflow_session['stages'].items():
                    status = stage_data.get('status', 'unknown')
                    if stage_data.get('error_message'):
                        print(f"  • {stage_name}: {status} - {stage_data['error_message']}")
                    else:
                        print(f"  • {stage_name}: {status}")

        print("="*60)
        return result

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Cleanup
        if session_id in orchestrator.active_sessions:
            del orchestrator.active_sessions[session_id]


if __name__ == "__main__":
    print("Quick Quality System Test")
    print("Testing reduced scope research with quality gates")
    print("Topic: US military response to Venezuela boat attacks")
    print("-" * 50)

    result = asyncio.run(quick_test())

    if result and result.get("success"):
        print("\n🎉 Quick test completed successfully!")
        print("Quality system is working as expected.")
    else:
        print("\n⚠️  Test encountered issues.")
        print("Check the error messages above for debugging.")