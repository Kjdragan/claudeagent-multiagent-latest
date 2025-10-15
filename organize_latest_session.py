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

    # Prepare the JSON payload
    payload = {
        "session_id": session_id,
        "base_dir": base_dir
    }

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