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
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def organize_workflow_files(session_id: str, base_dir: str = "KEVIN/sessions") -> Dict[str, Any]:
    """
    Organize workflow files into proper KEVIN directory structure.

    Args:
        session_id: The workflow session ID
        base_dir: Base KEVIN directory path

    Returns:
        Dictionary with organization results and file mappings
    """

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
        "logs": session_path / "logs"
    }

    for subdir_path in subdirs.values():
        subdir_path.mkdir(parents=True, exist_ok=True)

    organization_results = {
        "status": "success",
        "session_id": session_id,
        "organized_files": [],
        "created_directories": [],
        "file_mappings": {}
    }

    # Find and organize files by workflow stage
    for file_path in session_path.rglob("*"):
        if file_path.is_file():
            organized_path = organize_single_file(file_path, subdirs, session_id)
            if organized_path != file_path:
                organization_results["organized_files"].append({
                    "original": str(file_path),
                    "organized": str(organized_path),
                    "stage": detect_workflow_stage(file_path)
                })
                organization_results["file_mappings"][str(file_path)] = str(organized_path)

    # Create workflow summary file
    create_workflow_summary(organization_results, subdirs["working"])

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

    # Check filename patterns
    if "research" in filename_lower or "workproduct" in filename_lower:
        return "research"
    elif "draft" in filename_lower or "initial" in filename_lower:
        return "report"
    elif "editorial" in filename_lower or "review" in filename_lower:
        return "editorial"
    elif "final" in filename_lower or "enhanced" in filename_lower:
        return "final"
    elif file_path.suffix == ".log":
        return "log"

    # Check content patterns for better detection
    try:
        if file_path.suffix in ['.md', '.txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content_preview = f.read(1000).lower()

                if "research workproduct" in content_preview:
                    return "research"
                elif "first draft" in content_preview or "initial report" in content_preview:
                    return "report"
                elif "editorial review" in content_preview or "editorial analysis" in content_preview:
                    return "editorial"
                elif "final enhanced report" in content_preview:
                    return "final"
    except Exception:
        pass  # If we can't read content, stick with filename detection

    return "other"


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
        # Read session data from stdin
        payload: dict = json.load(os.sys.stdin)
        session_id = payload.get('session_id', '')
        base_dir = payload.get('base_dir', 'KEVIN/sessions')

        if not session_id:
            raise ValueError("No session ID provided")

        # Organize workflow files
        results = organize_workflow_files(session_id, base_dir)

        # Print results for logging
        print(f"Workflow organization completed for session {session_id}")
        print(f"Files organized: {len(results['organized_files'])}")
        print(f"Status: {results['status']}")

        if results['status'] == 'error':
            print(f"Error: {results['message']}", file=os.sys.stderr)
            os.sys.exit(1)

    except Exception as e:
        print(f"Error in workflow file organization: {e}", file=os.sys.stderr)
        os.sys.exit(1)