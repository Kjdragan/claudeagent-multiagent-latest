"""Research tools that analyze and verify web search activity.

This module creates tools that will save WebFetch content and generate verification reports
to verify we're getting real web search data, not just LLM knowledge.
"""

import json
import os

# Import from parent directory structure
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found. Using environment variables only.")

# Import logging when available
try:
    from .logging_config import get_logger
    _logger = get_logger("search_analysis")
    _logger.debug("Search analysis tools logging initialized")
except ImportError:
    _logger = None

try:
    # Try to import from global context first (main script already imported it)
    import importlib
    sdk_tool = importlib.import_module('claude_agent_sdk').tool
except (ImportError, AttributeError):
    try:
        # Fallback to direct import
        from claude_agent_sdk import tool
        sdk_tool = tool
    except ImportError:
        # Fallback decorator for when the SDK is not available
        def tool(name, description, parameters):
            def decorator(func):
                return func
            return decorator
        sdk_tool = tool
        print("Warning: claude_agent_sdk not found. Using fallback tool decorator.")




@sdk_tool("save_webfetch_content", "Save WebFetch content with source URLs", {
    "url": str,
    "content": str,
    "session_id": str,
    "content_type": str
})
async def save_webfetch_content(args: dict[str, Any]) -> dict[str, Any]:
    """Save actual WebFetch content with full source information."""
    if _logger:
        _logger.info(f"Saving WebFetch content from URL: {args.get('url', 'Unknown')}")

    url = args["url"]
    content = args["content"]
    session_id = args.get("session_id", str(uuid.uuid4()))
    content_type = args.get("content_type", "web_content")

    # Create session-based directory structure
    base_sessions_dir = Path("/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions")
    session_dir = base_sessions_dir / session_id
    research_dir = session_dir / "search_analysis"
    research_dir.mkdir(parents=True, exist_ok=True)

    # Save WebFetch data with metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fetch_data = {
        "url": url,
        "content": content,
        "content_type": content_type,
        "fetched_at": datetime.now().isoformat(),
        "session_id": session_id,
        "fetch_type": "web_fetch_analysis",
        "verification_data": {
            "content_length": len(content),
            "url_domain": url.split('/')[2] if len(url.split('/')) > 2 else 'unknown',
            "fetch_timestamp": timestamp
        }
    }

    # Save to session-based directory
    fetch_file = research_dir / f"web_fetch_content_{timestamp}.json"
    with open(fetch_file, 'w', encoding='utf-8') as f:
        json.dump(fetch_data, f, indent=2, ensure_ascii=False)

    if _logger:
        _logger.info(f"WebFetch content saved to {fetch_file}")

    return {
        "content": [{
            "type": "text",
            "text": f"WebFetch content from {url} saved successfully. Content length: {len(content)} characters."
        }],
        "fetch_file": str(fetch_file),
        "session_id": session_id,
        "content_length": len(content)
    }


@sdk_tool("create_search_verification_report", "Create verification report showing real search vs LLM content", {
    "topic": str,
    "session_id": str,
    "verification_data": str
})
async def create_search_verification_report(args: dict[str, Any]) -> dict[str, Any]:
    """Create a verification report showing what was actually searched vs generated."""
    if _logger:
        _logger.info(f"Creating search verification report for topic: {args.get('topic', 'Unknown')}")

    topic = args["topic"]
    session_id = args.get("session_id", str(uuid.uuid4()))
    verification_data = args.get("verification_data", "")

    # Create KEVIN directory if it doesn't exist
    kevin_dir = Path("/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN")
    kevin_dir.mkdir(parents=True, exist_ok=True)

    # Scan for WebFetch content files (search verification files removed)
    fetch_files = []

    # Check session directories for WebFetch content files
    sessions_dir = kevin_dir / "sessions"
    if sessions_dir.exists():
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir():
                search_analysis_dir = session_dir / "search_analysis"
                if search_analysis_dir.exists():
                    fetch_files.extend(search_analysis_dir.glob("web_fetch_content_*.json"))

    # Also check KEVIN root for backward compatibility (old WebFetch files)
    fetch_files.extend(kevin_dir.glob("web_fetch_content_*.json"))

    # Remove duplicates (only WebFetch files now)
    fetch_files = list(set(fetch_files))

    # Create verification report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    verification_report = f"""# Search Verification Report: {topic}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Session ID:** {session_id}
**Report Type:** Search Result Verification

---

## VERIFICATION SUMMARY

This report analyzes whether the research system is using actual web search results or LLM-generated content.

### Web Search Activity
- **WebFetch Operations:** {len(fetch_files)}
- **Note:** Search verification metadata has been streamlined - focusing on actual content extraction

### WebFetch Content Analysis
"""

    if fetch_files:
        verification_report += "\n**WebFetch Content Files:**\n"
        for fetch_file in fetch_files[-5:]:  # Show last 5 files
            try:
                with open(fetch_file, encoding='utf-8') as f:
                    data = json.load(f)
                verification_report += f"\n- **File:** {fetch_file.name}"
                verification_report += f"\n  - URL: {data.get('url', 'N/A')}"
                verification_report += f"\n  - Content Length: {data.get('verification_data', {}).get('content_length', 'N/A')} chars"
                verification_report += f"\n  - Domain: {data.get('verification_data', {}).get('url_domain', 'N/A')}"
                verification_report += f"\n  - Fetched: {data.get('fetched_at', 'N/A')}"
            except Exception as e:
                verification_report += f"\n- **File:** {fetch_file.name} (Error reading: {e})"
    else:
        verification_report += "\n❌ **NO ACTUAL WEB FETCH CONTENT FOUND**"

    verification_report += f"""

## CONCLUSION

{'✅ REAL WEB SEARCH DETECTED' if fetch_files else '❌ NO REAL WEB SEARCH - LIKELY LLM GENERATED'}

The system {'IS' if fetch_files else 'IS NOT'} performing actual web searches and retrieving real content from the internet.

{'WebFetch operations with real content extraction are captured and saved for verification.' if fetch_files else 'No WebFetch content was found, suggesting content may be from LLM knowledge rather than live web search.'}

---

*This verification report was generated to determine if the research system uses real web search results or LLM-generated content.*
"""

    # Save verification report
    report_file = kevin_dir / f"search_verification_report_{timestamp}_{session_id[:8]}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(verification_report)

    if _logger:
        _logger.info(f"Search verification report created: {report_file}")

    return {
        "content": [{
            "type": "text",
            "text": f"Search verification report created. Real web search detected: {bool(fetch_files)}"
        }],
        "report_file": str(report_file),
        "verification_result": "REAL_SEARCH" if fetch_files else "LLM_ONLY",
        "session_id": session_id
    }
