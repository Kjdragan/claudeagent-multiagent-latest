"""
Unit tests for tool_execution_tracker.py

Tests the ToolExecutionTracker class for monitoring MCP tool lifecycle.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multi_agent_research_system', 'core'))

import tool_execution_tracker

ToolExecutionState = tool_execution_tracker.ToolExecutionState
ToolExecution = tool_execution_tracker.ToolExecution
ToolExecutionTracker = tool_execution_tracker.ToolExecutionTracker
get_tool_execution_tracker = tool_execution_tracker.get_tool_execution_tracker


def test_tool_execution_state_enum():
    """Test ToolExecutionState enum values."""
    states = [
        ToolExecutionState.PENDING,
        ToolExecutionState.RUNNING,
        ToolExecutionState.COMPLETED,
        ToolExecutionState.FAILED,
        ToolExecutionState.TIMEOUT,
        ToolExecutionState.CANCELLED,
        ToolExecutionState.INDETERMINATE
    ]

    assert len(states) == 7
    assert ToolExecutionState.RUNNING.value == "running"
    assert ToolExecutionState.COMPLETED.value == "completed"

    print("✅ Test passed: ToolExecutionState enum correct")


def test_tool_execution_dataclass():
    """Test ToolExecution dataclass."""
    execution = ToolExecution(
        tool_name="test_tool",
        tool_use_id="test_123",
        session_id="session_001",
        start_time=time.time(),
        input_data={"query": "test"}
    )

    assert execution.tool_name == "test_tool"
    assert execution.state == ToolExecutionState.PENDING
    assert execution.is_active() == True
    assert execution.end_time is None

    # Test elapsed time
    time.sleep(0.1)
    elapsed = execution.elapsed_time()
    assert elapsed >= 0.1

    # Test mark completed
    execution.mark_completed({"result": "success"})
    assert execution.state == ToolExecutionState.COMPLETED
    assert execution.is_active() == False
    assert execution.result_data == {"result": "success"}

    print(f"✅ Test passed: ToolExecution dataclass works (elapsed: {elapsed:.2f}s)")


def test_tracker_initialization():
    """Test ToolExecutionTracker initialization."""
    tracker = ToolExecutionTracker(default_timeout=120, warning_threshold=60)

    assert tracker.default_timeout == 120
    assert tracker.warning_threshold == 60
    assert len(tracker.active_executions) == 0
    assert tracker.stats["total_executions"] == 0

    print("✅ Test passed: Tracker initialization correct")


def test_track_tool_start():
    """Test starting tool execution tracking."""
    tracker = ToolExecutionTracker()

    execution = tracker.track_tool_start(
        tool_name="test_search",
        tool_use_id="abc123",
        session_id="session_001",
        input_data={"query": "AI trends"},
        timeout_seconds=180
    )

    assert execution.tool_name == "test_search"
    assert execution.state == ToolExecutionState.RUNNING
    assert "abc123" in tracker.active_executions
    assert tracker.stats["total_executions"] == 1

    print("✅ Test passed: Tool tracking started successfully")


def test_track_tool_completion():
    """Test completing tool execution tracking."""
    tracker = ToolExecutionTracker()

    execution = tracker.track_tool_start(
        tool_name="test_tool",
        tool_use_id="test_001",
        session_id="session_001",
        input_data={}
    )

    time.sleep(0.1)

    tracker.track_tool_completion(
        tool_use_id="test_001",
        result_data={"success": True},
        success=True
    )

    assert "test_001" not in tracker.active_executions
    assert len(tracker.execution_history) == 1
    assert tracker.stats["completed"] == 1
    assert tracker.execution_history[0].state == ToolExecutionState.COMPLETED

    print("✅ Test passed: Tool completion tracked successfully")


def test_track_tool_failure():
    """Test tracking tool failure."""
    tracker = ToolExecutionTracker()

    tracker.track_tool_start(
        tool_name="test_tool",
        tool_use_id="test_002",
        session_id="session_001",
        input_data={}
    )

    tracker.track_tool_completion(
        tool_use_id="test_002",
        success=False,
        error="Connection timeout"
    )

    assert tracker.stats["failed"] == 1
    assert tracker.execution_history[0].error == "Connection timeout"

    print("✅ Test passed: Tool failure tracked successfully")


def test_timeout_detection():
    """Test timeout detection with short timeout."""
    tracker = ToolExecutionTracker(default_timeout=1, warning_threshold=0.5)

    execution = tracker.track_tool_start(
        tool_name="slow_tool",
        tool_use_id="test_003",
        session_id="session_001",
        input_data={},
        timeout_seconds=1
    )

    # Wait for timeout
    time.sleep(1.2)

    timed_out = tracker.check_tool_timeout("test_003")

    assert timed_out == True
    assert execution.state == ToolExecutionState.TIMEOUT
    assert tracker.stats["timeout"] == 1
    assert "test_003" not in tracker.active_executions

    print("✅ Test passed: Timeout detected correctly")


def test_long_running_detection():
    """Test long-running tool detection."""
    tracker = ToolExecutionTracker(default_timeout=10, warning_threshold=0.5)

    tracker.track_tool_start(
        tool_name="long_tool",
        tool_use_id="test_004",
        session_id="session_001",
        input_data={}
    )

    # Wait past warning threshold but before timeout
    time.sleep(0.6)

    results = tracker.check_all_active_tools()

    assert len(results["long_running"]) == 1
    assert results["long_running"][0].tool_name == "long_tool"
    assert "test_004" in tracker.active_executions  # Still active, not timed out

    print("✅ Test passed: Long-running tool detected")


def test_orphaned_tools():
    """Test orphaned tool detection."""
    tracker = ToolExecutionTracker()

    # Start multiple tools for same session
    tracker.track_tool_start("tool1", "id1", "session_001", {})
    tracker.track_tool_start("tool2", "id2", "session_001", {})
    tracker.track_tool_start("tool3", "id3", "session_002", {})

    # Session ends, handle orphaned tools
    tracker.handle_orphaned_tools("session_001")

    assert tracker.stats["indeterminate"] == 2
    assert "id1" not in tracker.active_executions
    assert "id2" not in tracker.active_executions
    assert "id3" in tracker.active_executions  # Different session

    print("✅ Test passed: Orphaned tools handled correctly")


def test_statistics():
    """Test statistics collection."""
    tracker = ToolExecutionTracker()

    # Execute several tools
    for i in range(5):
        tool_id = f"tool_{i}"
        tracker.track_tool_start(f"test_{i}", tool_id, "session", {})
        time.sleep(0.01)
        tracker.track_tool_completion(tool_id, success=True)

    # Execute one failure
    tracker.track_tool_start("fail_tool", "fail_1", "session", {})
    tracker.track_tool_completion("fail_1", success=False, error="Test error")

    stats = tracker.get_statistics()

    assert stats["total_executions"] == 6
    assert stats["completed"] == 5
    assert stats["failed"] == 1
    assert stats["active_count"] == 0
    assert stats["average_execution_time"] > 0

    print(f"✅ Test passed: Statistics collected (avg time: {stats['average_execution_time']:.3f}s)")


def test_singleton_pattern():
    """Test get_tool_execution_tracker singleton."""
    tracker1 = get_tool_execution_tracker()
    tracker2 = get_tool_execution_tracker()

    assert tracker1 is tracker2

    print("✅ Test passed: Singleton pattern works")


def test_execution_state_query():
    """Test getting execution state."""
    tracker = ToolExecutionTracker()

    tracker.track_tool_start("test_tool", "query_1", "session", {})

    state = tracker.get_execution_state("query_1")
    assert state == ToolExecutionState.RUNNING

    tracker.track_tool_completion("query_1", success=True)

    state = tracker.get_execution_state("query_1")
    assert state == ToolExecutionState.COMPLETED

    state = tracker.get_execution_state("nonexistent")
    assert state is None

    print("✅ Test passed: Execution state query works")


def test_active_tool_summary():
    """Test active tool summary generation."""
    tracker = ToolExecutionTracker()

    # No active tools
    summary = tracker.get_active_tool_summary()
    assert "No active tools" in summary

    # Add some tools
    tracker.track_tool_start("tool1", "id1", "session", {})
    tracker.track_tool_start("tool2", "id2", "session", {})

    summary = tracker.get_active_tool_summary()
    assert "Active tools: 2" in summary
    assert "tool1" in summary
    assert "tool2" in summary

    print("✅ Test passed: Active tool summary generated")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("TOOL EXECUTION TRACKER UNIT TESTS")
    print("="*60 + "\n")

    tests = [
        ("ToolExecutionState Enum", test_tool_execution_state_enum),
        ("ToolExecution Dataclass", test_tool_execution_dataclass),
        ("Tracker Initialization", test_tracker_initialization),
        ("Track Tool Start", test_track_tool_start),
        ("Track Tool Completion", test_track_tool_completion),
        ("Track Tool Failure", test_track_tool_failure),
        ("Timeout Detection", test_timeout_detection),
        ("Long-Running Detection", test_long_running_detection),
        ("Orphaned Tools", test_orphaned_tools),
        ("Statistics Collection", test_statistics),
        ("Singleton Pattern", test_singleton_pattern),
        ("Execution State Query", test_execution_state_query),
        ("Active Tool Summary", test_active_tool_summary),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ Test failed: {test_name}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ Test error: {test_name}")
            print(f"   Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
