#!/usr/bin/env python3
"""Startup checks for the Multi-Agent Research System

This module performs essential startup checks that should run before
the main system initialization. It can be imported early without triggering
full system initialization.
"""

import logging
import sys
from pathlib import Path

# Import the Playwright checker - handle both package and standalone execution
try:
    from .playwright_setup import ensure_playwright_installed
except ImportError:
    # Fallback for standalone execution
    from playwright_setup import ensure_playwright_installed

logger = logging.getLogger(__name__)


def run_all_startup_checks(verbose: bool = True) -> bool:
    """Run all startup checks required before system initialization.

    Args:
        verbose: If True, print progress messages to stdout

    Returns:
        bool: True if all checks pass, False otherwise
    """
    checks_passed = True

    if verbose:
        print("=" * 60)
        print("🚀 Multi-Agent Research System - Startup Checks")
        print("=" * 60)

    # Check 1: Playwright browsers
    if verbose:
        print("\n1️⃣  Checking Playwright browser installation...")

    try:
        if not ensure_playwright_installed():
            logger.error("Playwright browser check failed")
            if verbose:
                print("❌ Playwright browser check FAILED")
            checks_passed = False
        else:
            if verbose:
                print("✅ Playwright browsers ready")
    except Exception as e:
        logger.error(f"Error during Playwright check: {e}")
        if verbose:
            print(f"❌ Playwright check error: {e}")
        checks_passed = False

    # Check 2: Python version (we need 3.10+)
    if verbose:
        print("\n2️⃣  Checking Python version...")

    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
        logger.error(f"Python 3.10+ required, got {python_version.major}.{python_version.minor}")
        if verbose:
            print(f"❌ Python 3.10+ required (found {python_version.major}.{python_version.minor})")
        checks_passed = False
    else:
        if verbose:
            print(f"✅ Python {python_version.major}.{python_version.minor} detected")

    # Check 3: KEVIN directory exists
    if verbose:
        print("\n3️⃣  Checking KEVIN directory structure...")

    kevin_dir = Path("KEVIN")
    if not kevin_dir.exists():
        logger.info("Creating KEVIN directory for session data")
        try:
            kevin_dir.mkdir(parents=True, exist_ok=True)
            if verbose:
                print("✅ KEVIN directory created")
        except Exception as e:
            logger.error(f"Failed to create KEVIN directory: {e}")
            if verbose:
                print(f"❌ Failed to create KEVIN directory: {e}")
            checks_passed = False
    else:
        if verbose:
            print("✅ KEVIN directory exists")

    # Check 4: Logs directory exists
    if verbose:
        print("\n4️⃣  Checking Logs directory structure...")

    logs_dir = Path("Logs")
    if not logs_dir.exists():
        logger.info("Creating Logs directory")
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            if verbose:
                print("✅ Logs directory created")
        except Exception as e:
            logger.error(f"Failed to create Logs directory: {e}")
            if verbose:
                print(f"❌ Failed to create Logs directory: {e}")
            # Don't fail on logs directory - just warn
            logger.warning("Continuing without logs directory")
            if verbose:
                print("⚠️  Continuing without logs directory")
    else:
        if verbose:
            print("✅ Logs directory exists")

    # Final summary
    if verbose:
        print("\n" + "=" * 60)
        if checks_passed:
            print("✅ All startup checks passed!")
        else:
            print("❌ Some startup checks failed. Please review above.")
        print("=" * 60 + "\n")

    return checks_passed


def quick_startup_check() -> bool:
    """Quick startup check without verbose output.

    Returns:
        bool: True if critical checks pass (Playwright only)
    """
    try:
        return ensure_playwright_installed()
    except Exception as e:
        logger.error(f"Quick startup check failed: {e}")
        return False


def main():
    """Command-line interface for running startup checks."""
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    success = run_all_startup_checks(verbose=True)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
