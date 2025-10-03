#!/usr/bin/env python3
"""
Test script to verify SDK pattern compliance in our hook system.

This script verifies that our hook system properly integrates with
the Claude Agent SDK patterns and follows async compatibility requirements.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our hook system directly
sys.path.insert(0, str(project_root / "multi_agent_research_system"))

from hooks.base_hooks import HookManager, HookContext
from hooks.tool_hooks import ToolExecutionHook
from hooks.sdk_integration import SDKHookBridge, SDKMessageProcessingHook

# Import SDK types if available
try:
    from claude_agent_sdk.types import HookMatcher, HookContext as SDKHookContext, HookJSONOutput
    SDK_AVAILABLE = True
    print("✅ Claude Agent SDK is available")
except ImportError:
    SDK_AVAILABLE = False
    print("⚠️  Claude Agent SDK not available - using fallback types")


async def test_sdk_hook_bridge():
    """Test the SDK hook bridge functionality."""
    print("\n🔧 Testing SDK Hook Bridge...")

    # Create hook manager
    hook_manager = HookManager()

    # Register some test hooks
    tool_hook = ToolExecutionHook(enabled=True)
    sdk_message_hook = SDKMessageProcessingHook(enabled=True)

    hook_manager.register_hook(tool_hook, ["PreToolUse", "PostToolUse"])
    hook_manager.register_hook(sdk_message_hook, ["sdk_message_processing"])

    # Create SDK bridge
    bridge = SDKHookBridge(hook_manager)

    # Test SDK hook callback creation
    if SDK_AVAILABLE:
        pre_tool_callback = bridge.create_sdk_hook_callback("PreToolUse")
        post_tool_callback = bridge.create_sdk_hook_callback("PostToolUse")

        # Verify callback signature matches SDK expectations
        import inspect
        sig = inspect.signature(pre_tool_callback)
        expected_params = ['input_data', 'tool_use_id', 'context']
        actual_params = list(sig.parameters.keys())

        if actual_params == expected_params:
            print("✅ SDK hook callback signature is correct")
        else:
            print(f"❌ SDK hook callback signature mismatch: expected {expected_params}, got {actual_params}")
            return False

        # Test hook callback execution
        sdk_context = SDKHookContext()
        test_input = {
            "tool_name": "TestTool",
            "tool_input": {"param": "value"}
        }

        try:
            result = await pre_tool_callback(test_input, "test_tool_id", sdk_context)
            if isinstance(result, dict) and "hookSpecificOutput" in result:
                print("✅ SDK hook callback execution successful")
            else:
                print("❌ SDK hook callback returned invalid format")
                return False
        except Exception as e:
            print(f"❌ SDK hook callback execution failed: {e}")
            return False

    print("✅ SDK Hook Bridge test passed")
    return True


async def test_hook_execution_patterns():
    """Test hook execution patterns for async compatibility."""
    print("\n⚡ Testing Hook Execution Patterns...")

    hook_manager = HookManager()

    # Register multiple hooks
    hooks = [
        ToolExecutionHook(enabled=True),
        ToolExecutionHook(enabled=True, name="secondary_tool_hook"),
        SDKMessageProcessingHook(enabled=True)
    ]

    for hook in hooks:
        hook_manager.register_hook(hook, ["test_execution"])

    # Create test context
    context = HookContext(
        hook_name="test_execution",
        hook_type="test_execution",
        session_id="test_session"
    )

    # Test parallel execution
    try:
        results = await hook_manager.execute_hooks(
            "test_execution",
            context,
            parallel=True
        )

        if len(results) == len(hooks):
            print("✅ Parallel hook execution successful")
        else:
            print(f"❌ Parallel execution returned {len(results)} results, expected {len(hooks)}")
            return False

    except Exception as e:
        print(f"❌ Parallel hook execution failed: {e}")
        return False

    # Test sequential execution
    try:
        results = await hook_manager.execute_hooks(
            "test_execution",
            context,
            parallel=False
        )

        if len(results) == len(hooks):
            print("✅ Sequential hook execution successful")
        else:
            print(f"❌ Sequential execution returned {len(results)} results, expected {len(hooks)}")
            return False

    except Exception as e:
        print(f"❌ Sequential hook execution failed: {e}")
        return False

    print("✅ Hook Execution Patterns test passed")
    return True


async def test_sdk_type_integration():
    """Test SDK type integration and fallback handling."""
    print("\n🔗 Testing SDK Type Integration...")

    # Test with fallback types
    from multi_agent_research_system.hooks.sdk_integration import SDK_AVAILABLE

    if SDK_AVAILABLE:
        print("✅ SDK types are properly imported")
    else:
        print("✅ Fallback types are properly configured")

    # Test message processing hook
    hook = SDKMessageProcessingHook(enabled=True)

    # Create test context with mock message data
    test_message_data = {
        "message": {
            "content": [{"text": "test message"}],
            "model": "claude-sonnet-4-5"
        },
        "message_type": "assistant_message"
    }

    context = HookContext(
        hook_name="sdk_message_test",
        hook_type="sdk_message_processing",
        session_id="test_session",
        metadata={
            "sdk_message": test_message_data,
            "message_type": "assistant_message"
        }
    )

    try:
        result = await hook.execute(context)
        if result.success:
            print("✅ SDK message processing successful")
        else:
            print(f"❌ SDK message processing failed: {result.error_message}")
            return False
    except Exception as e:
        print(f"❌ SDK message processing error: {e}")
        return False

    print("✅ SDK Type Integration test passed")
    return True


async def test_error_handling():
    """Test error handling and resilience."""
    print("\n🛡️  Testing Error Handling...")

    hook_manager = HookManager()

    # Create a hook that will fail
    class FailingHook:
        def __init__(self):
            self.name = "failing_hook"
            self.hook_type = "test_error"
            self.enabled = True
            self.timeout = 1.0
            self.retry_count = 1
            self.retry_delay = 0.1

        async def execute(self, context):
            raise ValueError("Test error")

        def can_execute(self, context):
            return self.enabled

        async def safe_execute(self, context):
            # Simplified version for testing
            try:
                result = await asyncio.wait_for(self.execute(context), timeout=self.timeout)
                return result
            except Exception as e:
                from hooks.base_hooks import HookResult, HookStatus
                return HookResult(
                    hook_name=self.name,
                    hook_type=self.hook_type,
                    status=HookStatus.FAILED,
                    execution_id=context.execution_id,
                    start_time=context.timestamp,
                    error_message=str(e),
                    error_type=type(e).__name__
                )

    failing_hook = FailingHook()
    hook_manager.register_hook(failing_hook, ["test_error"])

    context = HookContext(
        hook_name="test_error",
        hook_type="test_error",
        session_id="test_session"
    )

    try:
        results = await hook_manager.execute_hooks("test_error", context)

        # Should return results even with failure
        if len(results) == 1 and results[0].failed:
            print("✅ Error handling works correctly")
        else:
            print("❌ Error handling failed")
            return False

    except Exception as e:
        print(f"❌ Unexpected error in error handling: {e}")
        return False

    print("✅ Error Handling test passed")
    return True


async def main():
    """Run all SDK pattern compliance tests."""
    print("🚀 Starting SDK Pattern Compliance Tests...")
    print("=" * 50)

    tests = [
        test_sdk_hook_bridge,
        test_hook_execution_patterns,
        test_sdk_type_integration,
        test_error_handling
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if await test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All SDK pattern compliance tests passed!")
        return True
    else:
        print("⚠️  Some tests failed - review implementation")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)