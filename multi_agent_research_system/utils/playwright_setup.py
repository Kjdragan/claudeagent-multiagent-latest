#!/usr/bin/env python3
"""Playwright Installation Checker and Auto-installer

This utility checks if Playwright browsers are properly installed and cached.
If not, it automatically runs the installation command to set up the browsers.
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def check_playwright_browsers_installed() -> bool:
    """Check if Playwright browsers are installed in the cache directory.

    Returns:
        bool: True if browsers are installed, False otherwise
    """
    # Check the standard Playwright cache directory
    cache_dir = Path.home() / ".cache" / "ms-playwright"

    if not cache_dir.exists():
        logger.warning(f"Playwright cache directory not found: {cache_dir}")
        return False

    # Check for essential browser directories
    required_browsers = ["chromium", "firefox", "webkit"]
    found_browsers = []

    for browser in required_browsers:
        # Look for browser directories (they include version numbers)
        browser_dirs = list(cache_dir.glob(f"{browser}-*"))
        if browser_dirs:
            found_browsers.append(browser)
            logger.debug(f"Found {browser} installation: {browser_dirs[0]}")

    # All three major browsers should be present
    if len(found_browsers) >= 3:
        logger.info("✅ Playwright browsers are properly installed")
        return True
    else:
        missing = set(required_browsers) - set(found_browsers)
        logger.warning(f"❌ Missing Playwright browsers: {missing}")
        return False


def install_playwright_browsers() -> bool:
    """Install Playwright browsers using uv run playwright install.

    Returns:
        bool: True if installation succeeded, False otherwise
    """
    try:
        logger.info("🔧 Installing Playwright browsers...")
        logger.info("This may take a few minutes on first run...")

        # Run the installation command
        result = subprocess.run(
            ["uv", "run", "playwright", "install"],
            capture_output=True,
            text=True,
            check=True
        )

        # Log the output
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")

        logger.info("✅ Playwright browsers installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install Playwright browsers: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("❌ 'uv' command not found. Please ensure uv is installed.")
        logger.error("Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during Playwright installation: {e}")
        return False


def ensure_playwright_installed() -> bool:
    """Ensure Playwright browsers are installed, installing if necessary.

    This is the main entry point that should be called at startup.

    Returns:
        bool: True if browsers are available (either already installed or just installed)
    """
    logger.info("🔍 Checking Playwright browser installation...")

    if check_playwright_browsers_installed():
        return True

    logger.info("📦 Playwright browsers not found. Installing now...")

    if install_playwright_browsers():
        # Verify installation succeeded
        if check_playwright_browsers_installed():
            logger.info("✅ Playwright setup complete and verified")
            return True
        else:
            logger.error("❌ Installation completed but browsers still not detected")
            return False
    else:
        logger.error("❌ Failed to install Playwright browsers")
        logger.error("Please run manually: uv run playwright install")
        return False


def main():
    """Command-line interface for testing the Playwright setup."""
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("🎭 Playwright Setup Checker")
    print("=" * 60)

    success = ensure_playwright_installed()

    if success:
        print("\n✅ Success! Playwright is ready for use.")
        sys.exit(0)
    else:
        print("\n❌ Failed to setup Playwright. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
