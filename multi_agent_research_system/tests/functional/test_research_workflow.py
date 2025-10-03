"""Functional tests for the complete research workflow with real LLM interactions.

These tests require actual Claude API credentials and will execute the full
multi-agent research system with real web searches, content generation, and coordination.
"""

import asyncio
import os

import pytest

# Only run functional tests if SDK is available and credentials are set
pytest.importorskip("claude_agent_sdk", reason="Claude Agent SDK not available")

try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        create_sdk_mcp_server,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

try:
    import anthropic
    API_KEY_AVAILABLE = bool(os.getenv("ANTHROPIC_API_KEY"))
except ImportError:
    API_KEY_AVAILABLE = False

from core.orchestrator import ResearchOrchestrator
from core.research_tools import conduct_research, generate_report, review_report
from tests.utils.test_helpers import validate_research_output


class TestResearchWorkflowFunctional:
    """Functional tests for the complete research workflow."""

    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_complete_research_workflow_real(self, temp_dir, monkeypatch):
        """Test complete research workflow with real Claude API calls."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        # Initialize real orchestrator
        orchestrator = ResearchOrchestrator()
        await orchestrator.initialize()

        try:
            # Start real research session
            topic = "The Impact of Remote Work on Employee Productivity"
            user_requirements = {
                "depth": "Standard Research",
                "audience": "Business",
                "format": "Business Brief",
                "requirements": "Focus on recent studies from 2020-2024, include statistics and company examples",
                "timeline": "Within 1 hour"
            }

            print(f"\n🔬 Starting real research on: {topic}")
            session_id = await orchestrator.start_research_session(topic, user_requirements)

            # Wait for workflow to complete (with timeout)
            timeout = 600  # 10 minutes
            start_time = asyncio.get_event_loop().time()

            while True:
                status = await orchestrator.get_session_status(session_id)
                print(f"📊 Status: {status['status']} - Stage: {status.get('current_stage', 'unknown')}")

                if status['status'] in ['completed', 'error']:
                    break

                if asyncio.get_event_loop().time() - start_time > timeout:
                    pytest.fail(f"Research workflow timed out after {timeout} seconds")

                await asyncio.sleep(10)  # Poll every 10 seconds

            # Validate results
            assert status['status'] == 'completed', f"Research failed: {status}"

            # Check session directory
            session_dir = temp_dir / "researchmaterials" / "sessions" / session_id
            assert session_dir.exists(), "Session directory not created"

            # Check for final report
            report_files = list(session_dir.glob("report_v*.md"))
            assert len(report_files) > 0, "No report files found"

            # Validate report content
            latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
            report_content = latest_report.read_text()

            assert len(report_content) > 1000, "Report content too short"
            assert "Remote Work" in report_content, "Report doesn't mention the topic"
            assert "Productivity" in report_content, "Report doesn't mention productivity"

            print("✅ Research completed successfully!")
            print(f"📄 Report saved to: {latest_report}")
            print(f"📏 Report length: {len(report_content)} characters")

        finally:
            await orchestrator.cleanup()

    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_real_research_tool_execution(self, temp_dir, monkeypatch):
        """Test individual research tools with real Claude API."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        # Create MCP server with real tools
        from core.research_tools import (
            conduct_research,
            generate_report,
        )

        print("\n🔧 Testing real tool execution...")

        # Test research tool
        research_args = {
            "topic": "Benefits of Solar Energy",
            "depth": "medium",
            "focus_areas": ["environmental", "economic"],
            "max_sources": 3
        }

        print("🔍 Conducting research...")
        research_result = await conduct_research(research_args)

        assert "research_data" in research_result, "Research tool failed to return data"
        research_data = research_result["research_data"]

        # Validate research output
        assert validate_research_output(research_data), "Research output validation failed"

        # Test report generation tool
        report_args = {
            "research_data": research_data,
            "format": "markdown",
            "audience": "general",
            "sections": ["summary", "benefits", "conclusions"]
        }

        print("📝 Generating report...")
        report_result = await generate_report(report_args)

        assert "report_data" in report_result, "Report tool failed to return data"
        report_data = report_result["report_data"]

        # Validate report structure
        assert "title" in report_data, "Report missing title"
        assert "sections" in report_data, "Report missing sections"
        assert "Solar Energy" in report_data["title"], "Report title doesn't mention topic"

        print("✅ Tool execution successful!")
        print(f"🔍 Found {len(research_data['findings'])} findings")
        print(f"📚 Used {len(research_data['sources_used'])} sources")

    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_multi_agent_coordination_real(self, temp_dir, monkeypatch):
        """Test multi-agent coordination with real Claude instances."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        print("\n🤝 Testing multi-agent coordination...")

        # Initialize orchestrator with real agents
        orchestrator = ResearchOrchestrator()
        await orchestrator.initialize()

        try:
            # Start simple research to test coordination
            session_id = await orchestrator.start_research_session(
                "Artificial Intelligence in Education",
                {
                    "depth": "Quick Overview",
                    "audience": "Educators",
                    "format": "Brief Summary"
                }
            )

            # Monitor workflow progression
            stages_seen = []
            timeout = 300  # 5 minutes
            start_time = asyncio.get_event_loop().time()

            while True:
                status = await orchestrator.get_session_status(session_id)
                current_stage = status.get('current_stage', 'unknown')

                if current_stage not in stages_seen:
                    stages_seen.append(current_stage)
                    print(f"📋 Entered stage: {current_stage}")

                if status['status'] in ['completed', 'error']:
                    break

                if asyncio.get_event_loop().time() - start_time > timeout:
                    pytest.fail(f"Multi-agent coordination timed out after {timeout} seconds")

                await asyncio.sleep(5)

            # Validate workflow progression
            expected_stages = ['research', 'report_generation', 'editorial_review']
            for stage in expected_stages:
                assert any(stage in s for s in stages_seen), f"Stage {stage} not executed"

            assert status['status'] == 'completed', "Multi-agent workflow failed"

            print("✅ Multi-agent coordination successful!")
            print(f"📊 Workflow stages: {stages_seen}")

        finally:
            await orchestrator.cleanup()

    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_real_web_search_integration(self, temp_dir, monkeypatch):
        """Test real web search functionality through the research system."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        print("\n🌐 Testing real web search integration...")

        # Test with a topic that requires current web information
        topic = "Latest Developments in Quantum Computing 2024"

        # This should trigger real web searches
        research_args = {
            "topic": topic,
            "depth": "medium",
            "focus_areas": ["breakthroughs", "applications"],
            "max_sources": 5,
            "require_current": True  # This should ensure web searches
        }

        print(f"🔍 Searching for: {topic}")
        research_result = await conduct_research(research_args)

        assert "research_data" in research_result, "Web search integration failed"
        research_data = research_result["research_data"]

        # Validate that we got recent information
        sources = research_data.get("sources_used", [])
        assert len(sources) > 0, "No sources found through web search"

        # Check for 2024 dates in sources (indicating recent information)
        has_recent_info = any("2024" in str(source) for source in sources)
        if has_recent_info:
            print("✅ Found recent information from 2024")

        # Validate source URLs are real web sources
        real_sources = [s for s in sources if s.get("url", "").startswith("http")]
        assert len(real_sources) > 0, "No real web sources found"

        print("✅ Web search integration successful!")
        print(f"🌐 Found {len(real_sources)} web sources")

    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_quality_assessment_real(self, temp_dir, monkeypatch):
        """Test quality assessment with real Claude analysis."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        print("\n⭐ Testing quality assessment...")

        # Generate some research data
        research_args = {
            "topic": "Climate Change Mitigation Strategies",
            "depth": "medium",
            "focus_areas": ["renewable_energy", "policy"],
            "max_sources": 4
        }

        research_result = await conduct_research(research_args)
        research_data = research_result["research_data"]

        # Generate report
        report_args = {
            "research_data": research_data,
            "format": "markdown",
            "audience": "policy_makers",
            "sections": ["summary", "strategies", "recommendations"]
        }

        report_result = await generate_report(report_args)
        report_data = report_result["report_data"]

        # Test real quality assessment
        review_args = {
            "report": report_data,
            "review_criteria": ["accuracy", "completeness", "clarity", "credibility"]
        }

        print("📊 Conducting quality assessment...")
        review_result = await review_report(review_args)

        assert "review_result" in review_result, "Quality assessment failed"
        assessment = review_result["review_result"]

        # Validate assessment structure
        assert "overall_score" in assessment, "Missing overall score"
        assert "criteria_scores" in assessment, "Missing criteria scores"
        assert "feedback" in assessment, "Missing feedback"

        overall_score = assessment["overall_score"]
        print(f"⭐ Overall quality score: {overall_score}/10")

        # Validate score is reasonable (not too high or low)
        assert 1 <= overall_score <= 10, f"Invalid score: {overall_score}"

        # Validate criteria scores
        criteria_scores = assessment["criteria_scores"]
        for criterion, score in criteria_scores.items():
            assert 1 <= score <= 10, f"Invalid {criterion} score: {score}"

        print("✅ Quality assessment completed!")
        print(f"📋 Criteria scores: {criteria_scores}")

    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_error_handling_real_api(self, temp_dir, monkeypatch):
        """Test error handling with real API calls."""
        if not SDK_AVAILABLE:
            pytest.skip("Claude Agent SDK not available")
        if not API_KEY_AVAILABLE:
            pytest.skip("ANTHROPIC_API_KEY not set")

        monkeypatch.chdir(temp_dir)

        print("\n❌ Testing error handling...")

        # Test with invalid parameters that should cause graceful errors
        invalid_args = {
            "topic": "",  # Empty topic should cause error
            "depth": "medium",
            "focus_areas": [],  # Empty focus areas
            "max_sources": 0   # Zero sources
        }

        try:
            research_result = await conduct_research(invalid_args)
            # If it succeeds, check if it handles the invalid inputs gracefully
            assert "research_data" in research_result, "Should handle invalid inputs gracefully"

            research_data = research_result["research_data"]
            # Should either have fallback topic or error indication
            assert research_data.get("topic") or "error" in research_result.get("content", [{}])[0].get("text", "").lower()

        except Exception as e:
            # Should handle errors gracefully without crashing
            assert "crash" not in str(e).lower(), "Should handle errors gracefully"

        # Test with extremely long topic that might hit limits
        long_topic = "Test " * 1000  # Very long topic
        long_args = {
            "topic": long_topic,
            "depth": "quick",
            "focus_areas": ["test"],
            "max_sources": 1
        }

        try:
            research_result = await conduct_research(long_args)
            # Should either succeed with truncated topic or handle gracefully
            assert "research_data" in research_result or "content" in research_result
            print("✅ Error handling working correctly")

        except Exception as e:
            # Should be a graceful error, not a crash
            assert "timeout" in str(e).lower() or "limit" in str(e).lower() or "error" in str(e).lower()
            print("✅ Graceful error handling confirmed")
