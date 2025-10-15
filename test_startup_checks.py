#!/usr/bin/env python3
"""Test script to verify startup checks work correctly without full system initialization."""

import sys
from pathlib import Path

# Add the utils directory to path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system" / "utils"))

# Import directly from the module file to avoid full system initialization
from startup_checks import run_all_startup_checks

print("Testing startup checks system...\n")

# Run the checks
success = run_all_startup_checks(verbose=True)

if success:
    print("\n✅ Test passed! Startup checks system is working correctly.")
    sys.exit(0)
else:
    print("\n❌ Test failed! Some startup checks did not pass.")
    sys.exit(1)
