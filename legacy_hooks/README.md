# Legacy Hook System Preservation

**Date**: October 2, 2025
**Reason**: Preserved before implementing simplified logging approach due to critical JavaScript parsing errors

## Issue Summary

The original hook system was causing **262 JavaScript parsing errors** per session, with patterns like:
- `'return "\\r" } else if(Y==='`
- JSON parsing failures in hook callbacks
- 91.7% of log entries being debug messages
- Major visibility gaps despite functional core workflow

## Root Cause (DeepWiki Investigation)

According to Claude Agent SDK documentation, the issue is caused by:
- **Malformed JSON output** from the `claude` CLI subprocess
- **Invalid JSON strings** being parsed by Python SDK
- **SubprocessCLITransport buffer issues** when reading stdout
- **Hook callback output** not conforming to expected `HookJSONOutput` structure

## Potential Solutions Investigated

1. **Increase `max_buffer_size`** in ClaudeAgentOptions
2. **Inspect raw CLI stdout** during hook callbacks
3. **Ensure HookJSONOutput TypedDict compliance**
4. **Capture subprocess output for debugging**

## Files Preserved

- `agent_hooks.py` - Agent-specific hook implementations
- `base_hooks.py` - Base hook functionality
- `hook_integration_manager.py` - Hook integration logic
- `mcp_hooks.py` - MCP server hooks
- `monitoring_hooks.py` - System monitoring hooks
- `sdk_integration.py` - SDK integration hooks
- `session_hooks.py` - Session lifecycle hooks
- `tool_hooks.py` - Tool execution hooks
- `workflow_hooks.py` - Workflow management hooks

## Decision

Due to the complexity of fixing the JavaScript parsing issues and the limited value provided by the current hook system, we are implementing a **simplified logging approach** that focuses on the most valuable outputs:

1. **session_state.json** - Complete session narrative
2. **orchestrator.jsonl** - Workflow progression
3. **agent_summary.json** - Performance metrics

The legacy system is preserved here for future reference if needed.