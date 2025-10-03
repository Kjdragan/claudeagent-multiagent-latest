#!/usr/bin/env python3
"""
Startup script for the Streamlit UI with automatic port management.
"""

import subprocess
import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.port_manager import ensure_port_available


def main():
    """Start the Streamlit UI with automatic port management."""
    default_port = 8501

    # Ensure the default port is available
    if not ensure_port_available(default_port):
        print(f"❌ Failed to ensure port {default_port} is available", file=sys.stderr)
        sys.exit(1)

    # Change to the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Start Streamlit
    cmd = [
        "uv", "run", "streamlit", "run",
        "multi_agent_research_system/ui/streamlit_app.py",
        "--server.port", str(default_port)
    ]

    print(f"🚀 Starting Streamlit UI on port {default_port}...", file=sys.stderr)
    print(f"📍 Local URL: http://localhost:{default_port}", file=sys.stderr)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Streamlit UI...", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()