#!/usr/bin/env python3
"""Standalone test for research tools file operations.

This test isolates the file writing logic from the MCP server environment
to determine if the issue is with the tool functions themselves or with
the MCP server context.
"""

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

def test_save_research_findings():
    """Test the save_research_findings function in isolation."""
    print("🔧 Testing save_research_findings function...")

    # Test arguments
    topic = "Test Research Topic for File Operations"
    findings = "This is test research content to verify file writing capabilities. " * 20  # Make it substantial
    sources = "['https://example.com/source1', 'https://example.com/source2']"
    session_id = str(uuid.uuid4())

    print(f"🔧 Test args:")
    print(f"  Topic: {topic}")
    print(f"  Findings length: {len(findings)}")
    print(f"  Sources: {sources}")
    print(f"  Session ID: {session_id[:8]}...")

    # Create session directory
    session_path = Path(f"researchmaterials/sessions/{session_id}")
    session_path.mkdir(parents=True, exist_ok=True)
    print(f"🔧 Created session directory: {session_path}")
    print(f"🔧 Session directory exists: {session_path.exists()}")

    # Create KEVIN directory
    kevin_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN")
    kevin_dir.mkdir(parents=True, exist_ok=True)
    print(f"🔧 KEVIN directory: {kevin_dir}")
    print(f"🔧 KEVIN directory exists: {kevin_dir.exists()}")
    print(f"🔧 KEVIN directory permissions: {oct(kevin_dir.stat().st_mode)}")

    # Test research data structure
    research_data = {
        "topic": topic,
        "findings": findings,
        "sources": sources,
        "saved_at": datetime.now().isoformat(),
        "session_id": session_id
    }

    # Test 1: Save to session path
    findings_file = session_path / "research_findings.json"
    print(f"🔧 Attempting to save findings to: {findings_file}")

    try:
        with open(findings_file, 'w', encoding='utf-8') as f:
            json.dump(research_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully saved to session path: {findings_file}")
        print(f"🔧 File exists check: {findings_file.exists()}")
        if findings_file.exists():
            print(f"🔧 File size: {findings_file.stat().st_size} bytes")
            print(f"🔧 File content preview:")
            with open(findings_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"  First 200 chars: {content[:200]}...")
    except Exception as e:
        print(f"❌ ERROR: Failed to save to session path: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Save to KEVIN directory
    timestamp = datetime.now().strftime('%H%M%S')
    kevin_findings_file = kevin_dir / f"test_research_findings_{session_id[:8]}_{timestamp}.json"
    print(f"🔧 Attempting to save findings to KEVIN: {kevin_findings_file}")

    try:
        with open(kevin_findings_file, 'w', encoding='utf-8') as f:
            json.dump(research_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully saved to KEVIN: {kevin_findings_file}")
        print(f"🔧 KEVIN file exists: {kevin_findings_file.exists()}")
        if kevin_findings_file.exists():
            print(f"🔧 KEVIN file size: {kevin_findings_file.stat().st_size} bytes")
    except Exception as e:
        print(f"❌ ERROR: Failed to save to KEVIN: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Simple text file write
    simple_file = kevin_dir / "simple_test.txt"
    print(f"🔧 Testing simple text file write to: {simple_file}")

    try:
        with open(simple_file, 'w', encoding='utf-8') as f:
            f.write("This is a simple test file.\n")
            f.write(f"Created at: {datetime.now().isoformat()}\n")
            f.write(f"Session ID: {session_id}\n")
        print(f"✅ Successfully created simple text file: {simple_file}")
        print(f"🔧 Simple file exists: {simple_file.exists()}")
        if simple_file.exists():
            print(f"🔧 Simple file size: {simple_file.stat().st_size} bytes")
    except Exception as e:
        print(f"❌ ERROR: Failed to create simple text file: {e}")
        import traceback
        traceback.print_exc()

    return {
        "session_file": str(findings_file),
        "session_file_exists": findings_file.exists(),
        "kevin_file": str(kevin_findings_file),
        "kevin_file_exists": kevin_findings_file.exists(),
        "simple_file": str(simple_file),
        "simple_file_exists": simple_file.exists(),
    }

def test_create_research_report():
    """Test the create_research_report function in isolation."""
    print("\n🔧 Testing create_research_report function...")

    topic = "Test Report Topic"
    content = "This is test report content with detailed analysis and findings. " * 30
    session_id = str(uuid.uuid4())
    format_type = "markdown"

    print(f"🔧 Report test args:")
    print(f"  Topic: {topic}")
    print(f"  Content length: {len(content)}")
    print(f"  Format: {format_type}")
    print(f"  Session ID: {session_id[:8]}...")

    # Create session directory
    session_path = Path(f"researchmaterials/sessions/{session_id}")
    session_path.mkdir(parents=True, exist_ok=True)

    # Create KEVIN directory
    kevin_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN")

    # Create formatted report
    report_content = f"""# Research Report: {topic}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Session ID:** {session_id}

---

{content}

---

*This report was generated by the Multi-Agent Research System using Claude Agent SDK.*
"""

    # Test 1: Save to session path
    extension = "md" if format_type == "markdown" else "txt"
    report_file = session_path / f"research_report.{extension}"

    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"✅ Successfully saved report to session: {report_file}")
        print(f"🔧 Report file exists: {report_file.exists()}")
        if report_file.exists():
            print(f"🔧 Report file size: {report_file.stat().st_size} bytes")
    except Exception as e:
        print(f"❌ ERROR: Failed to save report to session: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Save to KEVIN directory
    timestamp = datetime.now().strftime('%H%M%S')
    kevin_report_file = kevin_dir / f"test_research_report_{topic.replace(' ', '_')}_{timestamp}.{extension}"

    try:
        with open(kevin_report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"✅ Successfully saved report to KEVIN: {kevin_report_file}")
        print(f"🔧 KEVIN report exists: {kevin_report_file.exists()}")
        if kevin_report_file.exists():
            print(f"🔧 KEVIN report size: {kevin_report_file.stat().st_size} bytes")
    except Exception as e:
        print(f"❌ ERROR: Failed to save report to KEVIN: {e}")
        import traceback
        traceback.print_exc()

    return {
        "report_file": str(report_file),
        "report_file_exists": report_file.exists(),
        "kevin_report_file": str(kevin_report_file),
        "kevin_report_file_exists": kevin_report_file.exists(),
    }

def test_environment():
    """Test the current environment and permissions."""
    print("\n🔧 Testing environment...")

    # Current working directory
    cwd = Path.cwd()
    print(f"🔧 Current working directory: {cwd}")
    print(f"🔧 CWD exists: {cwd.exists()}")
    print(f"🔧 CWD permissions: {oct(cwd.stat().st_mode)}")
    print(f"🔧 CWD writable: {os.access(cwd, os.W_OK)}")

    # Test directories
    test_dirs = [
        "researchmaterials",
        "researchmaterials/sessions",
        "/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN"
    ]

    for dir_path in test_dirs:
        path = Path(dir_path)
        print(f"🔧 Directory: {dir_path}")
        print(f"  Exists: {path.exists()}")
        if path.exists():
            print(f"  Permissions: {oct(path.stat().st_mode)}")
            print(f"  Writable: {os.access(path, os.W_OK)}")
            print(f"  Readable: {os.access(path, os.R_OK)}")
        else:
            print(f"  Creating directory...")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created successfully")
                print(f"  Now exists: {path.exists()}")
                print(f"  Permissions: {oct(path.stat().st_mode)}")
                print(f"  Writable: {os.access(path, os.W_OK)}")
            except Exception as e:
                print(f"  ❌ Failed to create: {e}")

if __name__ == "__main__":
    print("🔧 Starting standalone file operations test...")
    print(f"🔧 Python version: {sys.version}")
    print(f"🔧 Current directory: {os.getcwd()}")

    # Test environment first
    test_environment()

    # Test research findings save
    findings_result = test_save_research_findings()

    # Test report creation
    report_result = test_create_research_report()

    # Summary
    print("\n🔧 Test Summary:")
    print(f"Findings session file created: {findings_result['session_file_exists']}")
    print(f"Findings KEVIN file created: {findings_result['kevin_file_exists']}")
    print(f"Simple test file created: {findings_result['simple_file_exists']}")
    print(f"Report session file created: {report_result['report_file_exists']}")
    print(f"Report KEVIN file created: {report_result['kevin_report_file_exists']}")

    all_success = all([
        findings_result['session_file_exists'],
        findings_result['kevin_file_exists'],
        findings_result['simple_file_exists'],
        report_result['report_file_exists'],
        report_result['kevin_report_file_exists']
    ])

    if all_success:
        print("✅ All file operations successful!")
    else:
        print("❌ Some file operations failed!")

    print("\n🔧 Check KEVIN directory for created files:")
    kevin_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN")
    if kevin_dir.exists():
        for file_path in kevin_dir.iterdir():
            if file_path.is_file():
                print(f"  📄 {file_path.name} ({file_path.stat().st_size} bytes)")