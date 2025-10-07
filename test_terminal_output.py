#!/usr/bin/env python3
"""
Test script for terminal output logging functionality.

This script demonstrates how terminal output is captured to timestamped files
in the session's working directory.
"""

import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "multi_agent_research_system"))

from utils.terminal_output_logger import TerminalOutputLogger


def main():
    """Test terminal output logging."""
    print("=" * 80)
    print("Terminal Output Logger Test")
    print("=" * 80)
    print()

    # Create a test session directory
    test_session_id = "test_session_12345"
    test_working_dir = Path("./test_output/working")
    test_working_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Test session ID: {test_session_id}")
    print(f"📁 Test working directory: {test_working_dir}")
    print()

    # Test with context manager
    print("🔧 Starting terminal output capture...")
    print()

    with TerminalOutputLogger(test_session_id, test_working_dir) as logger:
        print("✅ Terminal output capture is now active!")
        print()

        # Generate various types of output
        print("🧪 Testing different output types:")
        print("-" * 40)
        print("1. Regular stdout output")
        print("2. Multi-line output:\n   Line 1\n   Line 2\n   Line 3")

        # Stderr output
        sys.stderr.write("⚠️  Warning: This is stderr output\n")
        sys.stderr.flush()

        print("3. Output with special characters: ✓ ✗ ⚡ 🚀 📊")
        print("4. Numeric data: 12345, 67.89, 0.001")

        # Simulate progress messages
        print()
        print("⏱️  Simulating progress output...")
        for i in range(5):
            print(f"   Progress: {i+1}/5 ({(i+1)*20}%)")
            time.sleep(0.5)

        print()
        print("✅ Test output complete!")
        print()
        print(f"📄 Output has been saved to: {logger.output_file}")

    print()
    print("=" * 80)
    print("Terminal output logging has stopped")
    print(f"Check the file: {test_working_dir / 'terminal_output_*.log'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
