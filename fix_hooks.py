#!/usr/bin/env python3
"""
Script to fix all Hook __init__ methods to accept timeout parameter.
"""
import re
from pathlib import Path

hooks_dir = Path("multi_agent_research_system/hooks")

# Hook classes that need fixing
hooks_to_fix = [
    ("agent_hooks.py", "AgentCommunicationHook"),
    ("agent_hooks.py", "AgentHandoffHook"),
    ("agent_hooks.py", "AgentStateMonitor"),
    ("workflow_hooks.py", "WorkflowOrchestrationHook"),
    ("workflow_hooks.py", "StageTransitionHook"),
    ("workflow_hooks.py", "DecisionPointHook"),
    ("session_hooks.py", "SessionLifecycleHook"),
    ("session_hooks.py", "SessionStateMonitor"),
    ("session_hooks.py", "SessionRecoveryHook"),
    ("monitoring_hooks.py", "SystemHealthHook"),
    ("monitoring_hooks.py", "PerformanceMonitorHook"),
    ("monitoring_hooks.py", "ErrorTrackingHook"),
    ("sdk_integration.py", "SDKMessageProcessingHook"),
    ("sdk_integration.py", "SDKHookIntegration"),
    ("mcp_hooks.py", "MCPMessageHook"),
    ("mcp_hooks.py", "MCPSessionHook"),
]

for filename, classname in hooks_to_fix:
    filepath = hooks_dir / filename
    if not filepath.exists():
        print(f"⚠️  File not found: {filepath}")
        continue

    content = filepath.read_text()

    # Pattern to match __init__ that doesn't have timeout parameter
    # Looking for: def __init__(self, ..., enabled: bool = True):
    # Where ... doesn't contain "timeout"

    pattern = rf'(class {classname}\(BaseHook\):.*?def __init__\(self,\s*(?:.*?,\s*)?)(enabled:\s*bool\s*=\s*True\))'

    def add_timeout(match):
        before = match.group(1)
        enabled_part = match.group(2)

        # Check if timeout already exists
        if 'timeout' in before:
            return match.group(0)  # Already has timeout, don't modify

        # Add timeout parameter before enabled
        return f"{before}timeout: float = 30.0, {enabled_part}"

    new_content = re.sub(pattern, add_timeout, content, flags=re.DOTALL)

    if new_content != content:
        filepath.write_text(new_content)
        print(f"✅ Fixed {classname} in {filename}")
    else:
        print(f"⏭️  {classname} in {filename} - no changes needed or pattern not matched")

print("\n✨ Hook fixing complete!")
