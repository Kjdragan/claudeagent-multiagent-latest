# Startup Checks System

## Overview

The startup checks system ensures that all required dependencies and system components are properly installed and configured before the Multi-Agent Research System begins operation. This prevents runtime errors related to missing dependencies like Playwright browsers.

## Features

### Automated Playwright Installation

The system automatically checks for Playwright browser installation and installs browsers if needed:

- **Automatic Detection**: Checks the Playwright cache directory (`~/.cache/ms-playwright/`) for browser installations
- **Automatic Installation**: If browsers are missing, automatically runs `uv run playwright install`
- **Browser Coverage**: Ensures Chromium, Firefox, and Webkit are all installed
- **First-Run Optimization**: Only installs on first run or when browsers are missing

### Comprehensive System Checks

The startup checks verify:

1. **Playwright Browsers** ✅ Critical - Ensures all browsers are installed
2. **Python Version** ✅ Verifies Python 3.10+ is available
3. **KEVIN Directory** ✅ Creates session storage directory if needed
4. **Logs Directory** ✅ Creates logging directory if needed

## Usage

### Automatic Integration

The startup checks run automatically when you use any of the main entry points:

```bash
# Using main_comprehensive_research.py (recommended)
uv run python main_comprehensive_research.py "your query here"

# Using main.py
uv run python multi_agent_research_system/main.py

# Using run_research.py
uv run python multi_agent_research_system/run_research.py "your query here"
```

### Manual Testing

You can test the startup checks independently:

```bash
# Run all startup checks with verbose output
uv run python multi_agent_research_system/utils/startup_checks.py

# Or from the utils directory
cd multi_agent_research_system/utils
uv run python startup_checks.py
```

### Programmatic Usage

You can also use the startup checks in your own scripts:

```python
from multi_agent_research_system.utils.startup_checks import run_all_startup_checks

# Run all checks with verbose output
if not run_all_startup_checks(verbose=True):
    print("Startup checks failed!")
    exit(1)

print("All checks passed, ready to proceed!")
```

Or for a quick, silent check:

```python
from multi_agent_research_system.utils.startup_checks import quick_startup_check

# Quick check (Playwright only, no verbose output)
if not quick_startup_check():
    print("Playwright not ready!")
    exit(1)
```

## How It Works

### Playwright Installation Check

1. **Cache Detection**: Checks `~/.cache/ms-playwright/` for browser directories
2. **Browser Verification**: Looks for `chromium-*`, `firefox-*`, and `webkit-*` directories
3. **Automatic Installation**: If missing, runs `uv run playwright install` command
4. **Installation Verification**: Confirms browsers are properly installed after setup

### First-Run Experience

On the first run or when Playwright browsers are not installed:

```
============================================================
🚀 Multi-Agent Research System - Startup Checks
============================================================

1️⃣  Checking Playwright browser installation...
❌ Missing Playwright browsers: {'chromium', 'firefox', 'webkit'}
🔧 Installing Playwright browsers...
This may take a few minutes on first run...
  Downloading Chromium 140.0.7339.16 (playwright build v1187)...
  Chromium 140.0.7339.16 downloaded to ~/.cache/ms-playwright/chromium-1187
  Downloading Firefox 141.0 (playwright build v1490)...
  Firefox 141.0 downloaded to ~/.cache/ms-playwright/firefox-1490
  Downloading Webkit 26.0 (playwright build v2203)...
  Webkit 26.0 downloaded to ~/.cache/ms-playwright/webkit-2203
✅ Playwright browsers installed successfully
✅ Playwright setup complete and verified

2️⃣  Checking Python version...
✅ Python 3.13 detected

3️⃣  Checking KEVIN directory structure...
✅ KEVIN directory created

4️⃣  Checking Logs directory structure...
✅ Logs directory created

============================================================
✅ All startup checks passed!
============================================================
```

### Subsequent Runs

On subsequent runs when everything is already installed:

```
============================================================
🚀 Multi-Agent Research System - Startup Checks
============================================================

1️⃣  Checking Playwright browser installation...
✅ Playwright browsers ready

2️⃣  Checking Python version...
✅ Python 3.13 detected

3️⃣  Checking KEVIN directory structure...
✅ KEVIN directory exists

4️⃣  Checking Logs directory structure...
✅ Logs directory exists

============================================================
✅ All startup checks passed!
============================================================
```

This runs almost instantly as it only checks for existing installations.

## Implementation Details

### Module Structure

```
multi_agent_research_system/utils/
├── playwright_setup.py       # Playwright-specific installation checker
├── startup_checks.py          # Comprehensive startup checks system
└── README_STARTUP_CHECKS.md   # This documentation
```

### Key Functions

#### `playwright_setup.py`

- `check_playwright_browsers_installed()`: Checks if browsers are in cache
- `install_playwright_browsers()`: Installs browsers using `uv run playwright install`
- `ensure_playwright_installed()`: Main entry point - checks and installs if needed

#### `startup_checks.py`

- `run_all_startup_checks(verbose=True)`: Runs all system checks with optional verbosity
- `quick_startup_check()`: Fast Playwright-only check without verbose output

### Integration Points

The startup checks are integrated at the following entry points:

1. **main_comprehensive_research.py** (line 1845-1856): Primary entry point
2. **multi_agent_research_system/main.py** (line 26-29): Alternative entry point
3. **multi_agent_research_system/run_research.py** (line 57-61): CLI entry point

## Error Handling

The system handles various failure scenarios gracefully:

### Missing `uv` Command

```
❌ 'uv' command not found. Please ensure uv is installed.
Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Network Issues During Installation

```
❌ Failed to install Playwright browsers: [error details]
Please run manually: uv run playwright install
```

### Permission Issues

```
❌ Failed to create KEVIN directory: Permission denied
```

## Benefits

1. **Zero Manual Setup**: Users never need to manually run `playwright install`
2. **Fast Subsequent Runs**: Checks are near-instantaneous when already installed
3. **Clear Feedback**: Verbose output shows exactly what's being checked and installed
4. **Graceful Failure**: Clear error messages guide users to resolve issues
5. **Prevents Runtime Errors**: Catches missing dependencies before they cause problems

## Troubleshooting

### If Playwright installation fails

1. Check network connectivity
2. Ensure sufficient disk space (~400MB for all browsers)
3. Try manual installation: `uv run playwright install`
4. Check the logs for detailed error messages

### If checks fail despite successful installation

1. Verify cache directory exists: `ls -la ~/.cache/ms-playwright/`
2. Check browser directories are present
3. Re-run installation: `uv run playwright install --force`

### If you need to reinstall browsers

```bash
# Remove existing browsers
rm -rf ~/.cache/ms-playwright/

# Re-run your script (will auto-install)
uv run python main_comprehensive_research.py "your query"
```

## Future Enhancements

Potential additions to the startup checks system:

- [ ] API key validation (check for required environment variables)
- [ ] Disk space verification
- [ ] Network connectivity check
- [ ] Optional dependency verification (PDF processing, etc.)
- [ ] System resource checks (memory, CPU availability)
- [ ] Configuration file validation

## Contributing

When adding new startup checks:

1. Add the check to `run_all_startup_checks()` in `startup_checks.py`
2. Follow the existing pattern for verbose output and error handling
3. Update this documentation
4. Test both success and failure scenarios
5. Ensure the check doesn't slow down startup significantly

## License

This component is part of the Multi-Agent Research System and follows the same license as the main project.
