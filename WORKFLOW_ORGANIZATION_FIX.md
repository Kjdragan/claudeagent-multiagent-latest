# Workflow File Organization Fix

## Problem Description

The workflow file organization hook was failing with an `Error in workflow file organization: 'organized_files'` error when trying to run:

```bash
echo '{"session_id": "$(basename $(pwd))", "base_dir": "KEVIN/sessions"}' | uv run python .claude/hooks/organize_workflow_files.py
```

## Root Cause Analysis

The issue had two parts:

1. **Command Syntax Issue**: The shell substitution `$(basename $(pwd))` was not being properly expanded due to being enclosed in single quotes, causing the script to receive the literal string `$(basename $(pwd))` as the session ID instead of the actual directory name.

2. **Script Error Handling Issue**: The organize_workflow_files.py script was trying to access the `organized_files` key before checking if the operation status was an error. When a session directory didn't exist, the script returned a minimal error dictionary without the `organized_files` key, causing a KeyError.

## Solution Implemented

### 1. Fixed the Script Error Handling

Modified `.claude/hooks/organize_workflow_files.py` to check the operation status before accessing the organized_files:

**Before:**
```python
# Print results for logging
print(f"Workflow organization completed for session {session_id}")
print(f"Files organized: {len(results['organized_files'])}")
print(f"Status: {results['status']}")

if results['status'] == 'error':
    print(f"Error: {results['message']}", file=os.sys.stderr)
    os.sys.exit(1)
```

**After:**
```python
# Print results for logging
print(f"Workflow organization completed for session {session_id}")
print(f"Status: {results['status']}")

if results['status'] == 'error':
    print(f"Error: {results['message']}", file=os.sys.stderr)
    os.sys.exit(1)

print(f"Files organized: {len(results['organized_files'])}")
```

### 2. Created a Wrapper Script

Created `organize_latest_session.py` that automatically finds the latest session directory and runs the organization:

```python
#!/usr/bin/env python3
"""
Wrapper script to organize the latest session's workflow files
"""

import json
import os
import sys
from pathlib import Path

def find_latest_session(base_dir="KEVIN/sessions"):
    """Find the most recent session directory"""
    sessions_dir = Path(base_dir)

    if not sessions_dir.exists():
        print(f"Sessions directory not found: {sessions_dir}", file=sys.stderr)
        return None

    # Find all session directories
    session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]

    if not session_dirs:
        print("No session directories found", file=sys.stderr)
        return None

    # Sort by modification time (most recent first)
    session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Return the most recent session
    return session_dirs[0].name

def main():
    base_dir = "KEVIN/sessions"

    # Find the latest session
    session_id = find_latest_session(base_dir)
    if not session_id:
        sys.exit(1)

    # Import and run the organization script
    script_dir = Path(__file__).parent / ".claude" / "hooks"
    sys.path.insert(0, str(script_dir))

    try:
        from organize_workflow_files import organize_workflow_files

        print(f"Organizing workflow files for session: {session_id}")

        # Organize workflow files
        results = organize_workflow_files(session_id, base_dir)

        # Print results for logging
        print(f"Workflow organization completed for session {session_id}")
        print(f"Status: {results['status']}")

        if results['status'] == 'error':
            print(f"Error: {results['message']}", file=sys.stderr)
            sys.exit(1)

        print(f"Files organized: {len(results['organized_files'])}")

        # Print detailed results
        if results.get('organized_files'):
            print("\nOrganized files:")
            for file_info in results['organized_files']:
                print(f"  {file_info['original']} -> {file_info['organized']} ({file_info['stage']})")

    except Exception as e:
        print(f"Error in workflow file organization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Usage Instructions

### Option 1: Use the Wrapper Script (Recommended)

```bash
uv run python organize_latest_session.py
```

This automatically finds the latest session and organizes its files.

### Option 2: Use the Original Hook with Manual Session ID

```bash
# Find the session ID first
ls KEVIN/sessions/

# Use the actual session ID
printf '{"session_id": "%s", "base_dir": "KEVIN/sessions"}\n' "1815d63a-81d4-4687-9a30-c428794a6a8c" | uv run python .claude/hooks/organize_workflow_files.py
```

### Option 3: Use the Original Hook with Correct Shell Expansion

```bash
# Using printf for proper JSON formatting
LATEST_SESSION=$(ls -t KEVIN/sessions/ | head -1)
printf '{"session_id": "%s", "base_dir": "KEVIN/sessions"}\n' "$LATEST_SESSION" | uv run python .claude/hooks/organize_workflow_files.py
```

## Testing the Fix

The fix was tested with both existing and non-existing sessions:

1. **Non-existing session**: Properly displays error message and exits with code 1
2. **Existing session**: Successfully organizes files and displays detailed results

## File Changes Made

1. **Modified**: `.claude/hooks/organize_workflow_files.py` - Fixed error handling order
2. **Added**: `organize_latest_session.py` - Wrapper script for automatic session detection
3. **Added**: `WORKFLOW_ORGANIZATION_FIX.md` - This documentation file

## Verification

To verify the fix works correctly:

```bash
# Test with the wrapper script
uv run python organize_latest_session.py

# Expected output:
# Organizing workflow files for session: 1815d63a-81d4-4687-9a30-c428794a6a8c
# Workflow organization completed for session 1815d63a-81d4-4687-9a30-c428794a6a8c
# Status: success
# Files organized: X
```

The workflow file organization now works correctly and provides clear error messages when session directories don't exist.