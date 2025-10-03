#!/usr/bin/env python3
"""
Port management utilities for the multi-agent research system.
"""

import subprocess
import sys
import time
from typing import Optional


def find_process_using_port(port: int) -> Optional[int]:
    """Find the PID of the process using the given port."""
    try:
        if sys.platform == "linux" or sys.platform == "darwin":
            # Unix-like systems
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip():
                return int(result.stdout.strip())
        elif sys.platform == "win32":
            # Windows systems
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        return int(parts[-1])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        pass

    return None


def kill_process_using_port(port: int, force: bool = False) -> bool:
    """Kill the process using the given port."""
    pid = find_process_using_port(port)
    if not pid:
        return True  # Port is already free

    try:
        print(f"🔧 Found process {pid} using port {port}", file=sys.stderr)

        if sys.platform == "linux" or sys.platform == "darwin":
            # Unix-like systems
            signal = "-9" if force else "-15"  # SIGKILL if force, SIGTERM otherwise
            subprocess.run(["kill", signal, str(pid)], check=True)
        elif sys.platform == "win32":
            # Windows systems
            subprocess.run(["taskkill", "/PID", str(pid), "/F" if force else "/PID"], check=True)

        print(f"✅ Successfully killed process {pid} using port {port}", file=sys.stderr)

        # Wait a moment for the process to fully terminate
        time.sleep(2)

        # Verify the port is now free
        if find_process_using_port(port) is None:
            return True
        else:
            print(f"⚠️ Process {pid} may still be running", file=sys.stderr)
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to kill process {pid}: {e}", file=sys.stderr)
        return False


def ensure_port_available(port: int, max_attempts: int = 3) -> bool:
    """Ensure the given port is available, killing processes if necessary."""
    print(f"🔍 Checking port {port} availability...", file=sys.stderr)

    for attempt in range(max_attempts):
        pid = find_process_using_port(port)
        if pid is None:
            print(f"✅ Port {port} is available", file=sys.stderr)
            return True

        print(f"⚠️ Port {port} is in use by process {pid} (attempt {attempt + 1}/{max_attempts})", file=sys.stderr)

        # Try graceful kill first, then force kill on subsequent attempts
        force = attempt > 0
        if kill_process_using_port(port, force=force):
            return True

    print(f"❌ Failed to free port {port} after {max_attempts} attempts", file=sys.stderr)
    return False


def get_available_port(start_port: int = 8501, max_attempts: int = 10) -> int:
    """Find an available port starting from the given port."""
    for port in range(start_port, start_port + max_attempts):
        if find_process_using_port(port) is None:
            return port

    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}")


def main():
    """Command line interface for port management."""
    if len(sys.argv) < 2:
        print("Usage: python port_manager.py <port> [--kill] [--find-available]", file=sys.stderr)
        sys.exit(1)

    port = int(sys.argv[1])

    if "--kill" in sys.argv:
        success = kill_process_using_port(port, force="--force" in sys.argv)
        sys.exit(0 if success else 1)
    elif "--find-available" in sys.argv:
        available_port = get_available_port(port)
        print(f"Available port found: {available_port}")
    else:
        success = ensure_port_available(port)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()