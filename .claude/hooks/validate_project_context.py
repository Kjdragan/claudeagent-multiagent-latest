#!/usr/bin/env python3
"""
Project Context Validation Hook

Validates that the hook is running in the correct project context
and prevents execution in unrelated environments like Claude Code.
"""

import json
import os
import sys
from pathlib import Path

def detect_project_root(current_dir: Path) -> Path:
    """Detect the project root directory."""

    # Look for project indicators
    project_indicators = [
        "multi_agent_research_system",
        "KEVIN",
        ".claude",
        "main_comprehensive_research.py"
    ]

    # Check current directory
    for indicator in project_indicators:
        if (current_dir / indicator).exists():
            # Found project indicator, check if this is the root
            # by looking for parent directories without the indicator
            parent = current_dir.parent
            has_parent_indicator = any((parent / ind).exists() for ind in project_indicators)

            if not has_parent_indicator:
                return current_dir

    # Check parent directories
    parent = current_dir.parent
    while parent != current_dir.root and parent != Path("/"):
        for indicator in project_indicators:
            if (parent / indicator).exists():
                return parent
        parent = parent.parent

    return None

def validate_project_environment(current_dir: Path) -> bool:
    """Validate that we're in the correct project environment."""

    # Check if we're in a project directory
    required_dirs = [
        "multi_agent_research_system",
        "KEVIN",
        ".claude"
    ]

    found_dirs = [d for d in required_dirs if (current_dir / d).exists()]

    if not found_dirs:
        print("❌ Not in multi-agent research system directory")
        print(f"Current directory: {current_dir}")
        print(f"Looking for: {required_dirs}")
        return False

    # Check if we're not in a subdirectory that shouldn't trigger hooks
    problematic_dirs = [".git", "node_modules", "__pycache__", ".pytest_cache"]
    current_dir_name = current_dir.name

    if current_dir_name in problematic_dirs:
        print(f"❌ In problematic directory: {current_dir_name}")
        return False

    return True

def validate_session_data(session_data: dict) -> tuple[bool, dict]:
    """Validate the session data passed to the hook. Returns (is_valid, result_dict)."""

    session_id = session_data.get("session_id")
    base_dir = session_data.get("base_dir", "KEVIN/sessions")

    if not session_id:
        return False, {
            "status": "error",
            "error": "No session_id provided in hook data",
            "validation_passed": False
        }

    # Check for unexpanded shell parameters - common issue when hooks run in wrong context
    unexpanded_patterns = ["$(basename $(pwd))", "$(pwd)", "$(shell", "${", "`", "\\$"]

    for pattern in unexpanded_patterns:
        if pattern in session_id:
            return False, {
                "status": "skipped",
                "message": f"Unexpanded shell parameter detected: {pattern}",
                "session_id": session_id,
                "pattern_found": pattern,
                "validation_passed": False,
                "skip_reason": "unexpanded_shell_parameters"
            }

    if not base_dir:
        return False, {
            "status": "error",
            "error": "No base_dir provided in hook data",
            "validation_passed": False
        }

    return True, {
        "status": "success",
        "session_id": session_id,
        "base_dir": base_dir,
        "validation_passed": True
    }

def main():
    """Main validation function."""
    try:
        # Check if we're in a Claude Code environment that shouldn't trigger workflow hooks
        current_dir = Path.cwd()

        # Additional check: If we're in the project root but no active session exists,
        # this is likely just Claude Code starting up, not a workflow completion
        project_root = detect_project_root(current_dir)
        if project_root:
            sessions_dir = project_root / "KEVIN" / "sessions"
            if not sessions_dir.exists() or not any(sessions_dir.iterdir()):
                print("ℹ️  No active sessions found - likely Claude Code startup, not workflow completion")
                print("ℹ️  Gracefully skipping workflow organization")
                # Output valid JSON for downstream processing
                print(json.dumps({
                    "status": "skipped",
                    "message": "No active sessions - Claude Code startup detected",
                    "current_dir": str(current_dir),
                    "validation_passed": False,
                    "skip_reason": "no_active_sessions"
                }))
                sys.exit(0)  # Exit gracefully, not as an error

  
        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())

        # Validate project environment
        if not validate_project_environment(current_dir):
            result = {
                "status": "error",
                "error": "Not in valid project environment",
                "current_dir": str(current_dir),
                "validation_passed": False
            }
            print(json.dumps(result))
            sys.exit(1)

        # Detect project root
        if not project_root:
            result = {
                "status": "error",
                "error": "Could not detect project root",
                "current_dir": str(current_dir),
                "validation_passed": False
            }
            print(json.dumps(result))
            sys.exit(1)

        # Validate session data
        is_valid, session_result = validate_session_data(input_data)

        # Check if the session_id looks like a UUID (actual project session)
        # vs a directory name (Claude Code startup)
        session_id = session_result.get("session_id", "")
        if not session_id or len(session_id) < 10 or "-" not in session_id:
            print("ℹ️  Invalid session ID format - likely not a real project session")
            print("ℹ️  Gracefully skipping workflow organization")
            print(json.dumps({
                "status": "skipped",
                "message": f"Invalid session ID format: {session_id}",
                "current_dir": str(current_dir),
                "validation_passed": False,
                "skip_reason": "invalid_session_format"
            }))
            sys.exit(0)  # Exit gracefully, not as an error

        if not is_valid:
            # session_result already contains the proper JSON structure
            session_result["current_dir"] = str(current_dir)
            session_result["project_root"] = str(project_root)
            print(json.dumps(session_result))

            # Exit with different codes based on error type
            if session_result.get("status") == "skipped":
                sys.exit(0)  # Graceful skip
            else:
                sys.exit(1)  # Error

        # Success case - combine all data
        result = {
            "status": "success",
            "session_id": session_result["session_id"],
            "base_dir": session_result["base_dir"],
            "project_root": str(project_root),
            "current_dir": str(current_dir),
            "validation_passed": True
        }

        # Write validated data back to stdout for downstream processing
        print(json.dumps(result))

    except json.JSONDecodeError as e:
        error_result = {
            "status": "error",
            "error": f"Invalid JSON input: {e}",
            "validation_passed": False
        }
        print(json.dumps(error_result))
        sys.exit(1)
    except Exception as e:
        error_result = {
            "status": "error",
            "error": f"Unexpected error: {e}",
            "validation_passed": False
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()