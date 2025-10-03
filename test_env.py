#!/usr/bin/env python3
"""Test script to check environment variable loading."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("🔍 Testing environment variable loading...")
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")

# Load environment variables
try:
    from dotenv import load_dotenv
    result = load_dotenv()
    print(f"✅ load_dotenv() returned: {result}")
    print(f"✅ python-dotenv imported successfully")
except ImportError:
    print("❌ python-dotenv not found")
    sys.exit(1)

# Check for .env file
env_file = project_root / ".env"
print(f"📁 .env file path: {env_file}")
print(f"📁 .env file exists: {env_file.exists()}")

if env_file.exists():
    print(f"📁 .env file size: {env_file.stat().st_size} bytes")

# Check specific environment variables
vars_to_check = [
    'SERP_API_KEY',
    'OPENAI_API_KEY',
    'ZAI_API_KEY',
    'ANTHROPIC_API_KEY'
]

print("\n🔍 Environment variables:")
for var in vars_to_check:
    value = os.getenv(var)
    if value:
        # Show first few characters and length for security
        masked = value[:8] + "..." + value[-4:] if len(value) > 12 else value[:8] + "..."
        print(f"  ✅ {var}: {masked} (length: {len(value)})")
    else:
        print(f"  ❌ {var}: NOT_SET")

print(f"\n📝 All environment variables (names only):")
all_env_vars = [k for k in os.environ.keys() if 'API' in k.upper() or 'SERP' in k.upper() or 'ZAI' in k.upper()]
for var in sorted(all_env_vars):
    print(f"  - {var}")