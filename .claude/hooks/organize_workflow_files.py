#!/usr/bin/env python3
"""
Workflow File Organization Hook

This hook organizes files generated during the multi-agent research workflow
into proper KEVIN directory structure with correct labeling and stage placement.

It processes workflow outputs and ensures files are organized correctly:
- Research stage → research/ directory
- Report stage → working/ directory with proper labeling
- Editorial stage → working/ directory with proper labeling
- Final enhanced report → complete/ directory
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def organize_workflow_files(session_id: str, base_dir: str = "KEVIN/sessions") -> Dict[str, Any]:
    """
    Organize workflow files into proper KEVIN directory structure with enhanced research work product handling.

    This function now validates the project context before proceeding to ensure
    it only runs in the correct project environment and not in unrelated contexts
    like Claude Code.

    Args:
        session_id: The workflow session ID
        base_dir: Base KEVIN directory path

    Returns:
        Dictionary with organization results and file mappings
    """

    # Validate project context before proceeding
    current_dir = Path.cwd()

    # Check if we're in the correct project environment
    required_dirs = ["multi_agent_research_system", "KEVIN", ".claude"]
    found_dirs = [d for d in required_dirs if (current_dir / d).exists()]

    if not found_dirs:
        return {
            "status": "error",
            "message": f"Not in multi-agent research system directory. Current: {current_dir}",
            "current_dir": str(current_dir),
            "required_dirs": required_dirs
        }

    # Check for problematic subdirectories
    problematic_subdirs = [".git", "node_modules", "__pycache__", ".pytest_cache", "venv", "env"]
    if current_dir.name in problematic_subdirs:
        return {
            "status": "error",
            "message": f"Hooks should not run in subdirectory: {current_dir.name}",
            "current_dir": str(current_dir)
        }

    # Validate session_id
    if not session_id or session_id == "$(basename $(pwd))":
        return {
            "status": "error",
            "message": f"Invalid or unexpanded session_id: {session_id}",
            "session_id": session_id
        }

    session_path = Path(base_dir) / session_id
    if not session_path.exists():
        return {
            "status": "error",
            "message": f"Session directory not found: {session_path}"
        }

    # Ensure subdirectories exist
    subdirs = {
        "working": session_path / "working",
        "research": session_path / "research",
        "complete": session_path / "complete",
        "logs": session_path / "logs",
        "agent_logs": session_path / "agent_logs",
        "sub_sessions": session_path / "sub_sessions"
    }

    created_directories = []
    for subdir_name, subdir_path in subdirs.items():
        if not subdir_path.exists():
            subdir_path.mkdir(parents=True, exist_ok=True)
            created_directories.append(subdir_name)

    organization_results = {
        "status": "success",
        "session_id": session_id,
        "organized_files": [],
        "created_directories": created_directories,
        "file_mappings": {},
        "research_work_products": [],
        "enhanced_workflow_files": []
    }

    # Find and organize files by workflow stage
    for file_path in session_path.rglob("*"):
        if file_path.is_file():
            stage = detect_workflow_stage(file_path)
            organized_path = organize_single_file(file_path, subdirs, session_id)

            if organized_path != file_path:
                file_info = {
                    "original": str(file_path),
                    "organized": str(organized_path),
                    "stage": stage,
                    "file_type": file_path.suffix
                }

                organization_results["organized_files"].append(file_info)
                organization_results["file_mappings"][str(file_path)] = str(organized_path)

                # Track research work products specifically
                if stage == "research" and "workproduct" in file_path.name.lower():
                    organization_results["research_work_products"].append(file_info)

                # Track enhanced workflow files
                if stage in ["editorial", "enhanced", "final"]:
                    organization_results["enhanced_workflow_files"].append(file_info)

    # Handle sub-session organization if any exist
    organize_sub_sessions(session_path, subdirs["sub_sessions"], organization_results)

    # Create workflow summary file
    create_workflow_summary(organization_results, subdirs["working"])

    # Validate session structure
    structure_validation = validate_session_structure(session_path, subdirs)
    organization_results["structure_validation"] = structure_validation

    return organization_results


def organize_single_file(file_path: Path, subdirs: Dict[str, Path], session_id: str) -> Path:
    """
    Organize a single file into the appropriate directory based on its type and stage.

    Args:
        file_path: Path to the file to organize
        subdirs: Dictionary of subdirectory paths
        session_id: Session ID for naming

    Returns:
        New organized file path
    """

    stage = detect_workflow_stage(file_path)
    filename = generate_proper_filename(file_path, stage, session_id)

    if stage == "research":
        target_dir = subdirs["research"]
    elif stage in ["report", "editorial", "draft"]:
        target_dir = subdirs["working"]
    elif stage == "final":
        target_dir = subdirs["complete"]
    elif stage == "log":
        target_dir = subdirs["logs"]
    else:
        # Keep in original location if stage is unclear
        return file_path

    target_path = target_dir / filename

    # Move file if location is different
    if target_path != file_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(target_path))

    return target_path


def detect_workflow_stage(file_path: Path) -> str:
    """
    Detect which workflow stage a file belongs to based on its name and content.

    Args:
        file_path: Path to the file

    Returns:
        Stage name: "research", "report", "editorial", "final", "log", or "other"
    """

    filename_lower = file_path.name.lower()

    # Enhanced pattern detection for research work products
    research_patterns = [
        "research", "workproduct", "initial_search", "editor-gap", "gap_research"
    ]

    # Enhanced pattern detection for workflow stages
    report_patterns = ["draft", "initial", "first_report", "research_report"]
    editorial_patterns = ["editorial", "review", "analysis", "gap_decision", "confidence"]
    final_patterns = ["final", "enhanced", "complete", "integrated"]

    # Check filename patterns
    if any(pattern in filename_lower for pattern in research_patterns):
        return "research"
    elif any(pattern in filename_lower for pattern in report_patterns):
        return "report"
    elif any(pattern in filename_lower for pattern in editorial_patterns):
        return "editorial"
    elif any(pattern in filename_lower for pattern in final_patterns):
        return "final"
    elif file_path.suffix == ".log":
        return "log"

    # Check content patterns for better detection
    try:
        if file_path.suffix in ['.md', '.txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content_preview = f.read(1000).lower()

                if any(pattern in content_preview for pattern in research_patterns):
                    return "research"
                elif any(pattern in content_preview for pattern in report_patterns):
                    return "report"
                elif any(pattern in content_preview for pattern in editorial_patterns):
                    return "editorial"
                elif any(pattern in content_preview for pattern in final_patterns):
                    return "final"
    except Exception:
        pass  # If we can't read content, stick with filename detection

    return "other"


def organize_sub_sessions(session_path: Path, sub_sessions_dir: Path, results: Dict[str, Any]):
    """
    Organize sub-session directories and files if they exist.

    Args:
        session_path: Main session directory path
        sub_sessions_dir: Sub-sessions directory path
        results: Results dictionary to update
    """

    if not sub_sessions_dir.exists():
        return

    sub_session_dirs = [d for d in sub_sessions_dir.iterdir() if d.is_dir()]

    if not sub_session_dirs:
        return

    results["sub_sessions_found"] = len(sub_session_dirs)
    results["sub_session_files"] = []

    for sub_session_dir in sub_session_dirs:
        sub_session_id = sub_session_dir.name

        # Find files in sub-session
        for file_path in sub_session_dir.rglob("*"):
            if file_path.is_file():
                stage = detect_workflow_stage(file_path)

                # Organize within sub-session structure
                organized_info = {
                    "original": str(file_path),
                    "organized": str(file_path),  # Keep in sub-session
                    "stage": stage,
                    "sub_session": sub_session_id,
                    "file_type": file_path.suffix
                }

                results["sub_session_files"].append(organized_info)


def validate_session_structure(session_path: Path, subdirs: Dict[str, Path]) -> Dict[str, Any]:
    """
    Validate the session directory structure meets requirements.

    Args:
        session_path: Session directory path
        subdirs: Expected subdirectories

    Returns:
        Validation result dictionary
    """

    validation = {
        "valid": True,
        "issues": [],
        "missing_directories": [],
        "unexpected_directories": [],
        "file_counts": {}
    }

    # Check expected directories
    for subdir_name, subdir_path in subdirs.items():
        if subdir_path.exists():
            file_count = len([f for f in subdir_path.rglob("*") if f.is_file()])
            validation["file_counts"][subdir_name] = file_count
        else:
            validation["valid"] = False
            validation["missing_directories"].append(subdir_name)
            validation["issues"].append(f"Missing directory: {subdir_name}")

    # Check for unexpected directories (excluding hidden ones)
    all_dirs = [d for d in session_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    expected_dir_names = set(subdirs.keys())

    for dir_path in all_dirs:
        if dir_path.name not in expected_dir_names:
            validation["unexpected_directories"].append(dir_path.name)
            validation["issues"].append(f"Unexpected directory: {dir_path.name}")

    # Validate research work products
    research_dir = subdirs.get("research")
    if research_dir and research_dir.exists():
        workproduct_files = list(research_dir.glob("*workproduct*"))
        if not workproduct_files:
            validation["issues"].append("No research workproducts found in research directory")
        else:
            validation["file_counts"]["research_workproducts"] = len(workproduct_files)

    return validation


def generate_proper_filename(file_path: Path, stage: str, session_id: str) -> str:
    """
    Generate a properly formatted filename for a workflow stage file.

    Args:
        file_path: Original file path
        stage: Workflow stage
        session_id: Session ID

    Returns:
        Properly formatted filename
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = file_path.stem

    if stage == "research":
        if "workproduct" in original_name.lower():
            return f"RESEARCH_WORKPRODUCT_{timestamp}.md"
        else:
            return f"RESEARCH_ANALYSIS_{timestamp}.md"

    elif stage == "report":
        if "draft" in original_name.lower():
            return f"INITIAL_REPORT_DRAFT_{timestamp}.md"
        else:
            return f"REPORT_ANALYSIS_{timestamp}.md"

    elif stage == "editorial":
        if "review" in original_name.lower():
            return f"EDITORIAL_REVIEW_{timestamp}.md"
        else:
            return f"EDITORIAL_ANALYSIS_{timestamp}.md"

    elif stage == "final":
        return f"FINAL_ENHANCED_REPORT_{timestamp}.md"

    elif stage == "log":
        return f"workflow_{timestamp}_{file_path.name}"

    else:
        return f"{original_name}_{timestamp}{file_path.suffix}"


def create_workflow_summary(results: Dict[str, Any], working_dir: Path):
    """
    Create a summary file showing the workflow organization results.

    Args:
        results: Organization results dictionary
        working_dir: Working directory to save summary in
    """

    summary_content = f"""# Workflow File Organization Summary

**Session ID:** {results['session_id']}
**Organization Time:** {datetime.now().isoformat()}
**Status:** {results['status']}

## Files Organized: {len(results['organized_files'])}

"""

    if results['organized_files']:
        for file_info in results['organized_files']:
            stage = file_info['stage']
            original = file_info['original']
            organized = file_info['organized']

            summary_content += f"""
### {stage.title()} Stage
- **From:** `{original}`
- **To:** `{organized}`
"""

    summary_content += f"""

## Directory Structure Created

```
KEVIN/sessions/{results['session_id']}/
├── working/          # Report and editorial analysis files
├── research/         # Research workproducts and data
├── complete/         # Final enhanced reports
└── logs/            # Workflow and operation logs
```

## Workflow Stages

1. **Research** - Initial research data and workproducts
2. **Report** - First draft reports and analysis
3. **Editorial** - Editorial review and enhancement
4. **Final** - Enhanced final reports

---
*Generated by Workflow File Organization Hook*
"""

    summary_path = working_dir / f"WORKFLOW_ORGANIZATION_SUMMARY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)


if __name__ == "__main__":
    try:
        # Read session data from stdin with robust UTF-8 handling
        try:
            # Try reading with UTF-8 encoding first
            input_bytes = sys.stdin.buffer.read()
            if not input_bytes:
                raise ValueError("No input data provided")

            # Decode with UTF-8 and error handling
            input_data = input_bytes.decode('utf-8', errors='replace').strip()
        except (AttributeError, UnicodeDecodeError):
            # Fallback to standard read with encoding handling
            try:
                with sys.stdin as f:
                    input_data = f.read().strip()
            except UnicodeDecodeError:
                # Last resort: read as bytes and decode loosely
                input_data = sys.stdin.buffer.read().decode('utf-8', errors='replace').strip()

        if not input_data:
            raise ValueError("No input data provided")

        # Parse JSON with error handling
        try:
            payload: dict = json.loads(input_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}. Input data: {input_data[:200]}...")

        # Handle input from validation script
        if payload.get("validation_passed"):
            # Input from validation script, extract original session data
            session_id = payload.get('session_id', '')
            base_dir = payload.get('base_dir', 'KEVIN/sessions')
            print(f"✅ Validation passed for session: {session_id}")
        elif payload.get("status") == "skipped":
            # Validation script skipped execution (e.g., Claude Code startup)
            skip_reason = payload.get("skip_reason", "unknown")
            message = payload.get("message", "Workflow organization skipped")
            print(f"ℹ️  {message}")
            print(f"   Skip reason: {skip_reason}")
            # Exit gracefully without processing
            sys.exit(0)
        else:
            # Direct input (original behavior)
            session_id = payload.get('session_id', '')
            base_dir = payload.get('base_dir', 'KEVIN/sessions')

        # Additional safety check for unexpanded shell parameters
        if not session_id or session_id == "$(basename $(pwd))":
            print(f"ℹ️  Invalid or unexpanded session_id detected: {session_id}")
            print("   This appears to be a Claude Code environment startup - gracefully skipping")
            sys.exit(0)

        if not session_id:
            raise ValueError("No session ID provided")

        # Organize workflow files
        results = organize_workflow_files(session_id, base_dir)

        # Print results for logging
        print(f"Workflow organization completed for session {session_id}")
        print(f"Status: {results['status']}")

        if results['status'] == 'error':
            print(f"Error: {results['message']}", file=sys.stderr)
            sys.exit(1)

        print(f"Files organized: {len(results['organized_files'])}")

    except Exception as e:
        print(f"Error in workflow file organization: {e}", file=sys.stderr)
        sys.exit(1)