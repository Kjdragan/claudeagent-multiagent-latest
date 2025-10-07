"""Research tools that capture and analyze actual web search results.

This module creates tools that will save raw search results and metadata
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
    from claude_agent_sdk import tool
except ImportError:
    # Fallback decorator for when the SDK is not available
    def tool(name, description, parameters):
        def decorator(func):
            return func
        return decorator
    print("Warning: claude_agent_sdk not found. Using fallback tool decorator.")


@tool("capture_search_results", "Capture and save web search results with metadata", {
    "search_query": str,
    "search_results": str,
    "sources_found": str,
    "session_id": str
})
async def capture_search_results(args: dict[str, Any]) -> dict[str, Any]:
    """Capture and save actual web search results with full metadata."""
    if _logger:
        _logger.info(f"Capturing search results for query: {args.get('search_query', 'Unknown')}")

    search_query = args["search_query"]
    search_results = args["search_results"]
    sources_found = args["sources_found"]
    session_id = args.get("session_id", str(uuid.uuid4()))

    # Create session-based directory structure
    base_sessions_dir = Path("/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions")
    session_dir = base_sessions_dir / session_id
    research_dir = session_dir / "search_analysis"
    research_dir.mkdir(parents=True, exist_ok=True)

    # Save raw search data with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    search_data = {
        "search_query": search_query,
        "search_results": search_results,
        "sources_found": sources_found,
        "captured_at": datetime.now().isoformat(),
        "session_id": session_id,
        "search_type": "web_search_analysis",
        "verification_data": {
            "query_timestamp": timestamp,
            "result_length": len(search_results),
            "sources_count": len(sources_found.split('\n')) if sources_found else 0
        }
    }

    # Save to session-based directory
    search_file = research_dir / f"web_search_results_{timestamp}.json"
    with open(search_file, 'w', encoding='utf-8') as f:
        json.dump(search_data, f, indent=2, ensure_ascii=False)

    if _logger:
        _logger.info(f"Search results captured and saved to {search_file}")

    return {
        "content": [{
            "type": "text",
            "text": f"Web search results for '{search_query}' captured and saved. Found {len(search_results)} characters of search data with {len(sources_found.split()) if sources_found else 0} sources."
        }],
        "search_file": str(search_file),
        "session_id": session_id,
        "results_length": len(search_results)
    }


@tool("save_webfetch_content", "Save WebFetch content with source URLs", {
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


@tool("create_search_verification_report", "Create verification report showing real search vs LLM content", {
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

    # Scan for search and fetch files in both KEVIN root and session directories
    search_files = []
    fetch_files = []

    # Check session directories first (organized approach)
    sessions_dir = kevin_dir / "sessions"
    if sessions_dir.exists():
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir():
                search_analysis_dir = session_dir / "search_analysis"
                if search_analysis_dir.exists():
                    search_files.extend(search_analysis_dir.glob("web_search_results_*.json"))
                    fetch_files.extend(search_analysis_dir.glob("web_fetch_content_*.json"))

    # Also check KEVIN root for backward compatibility (old files)
    search_files.extend(kevin_dir.glob("web_search_results_*.json"))
    fetch_files.extend(kevin_dir.glob("web_fetch_content_*.json"))

    # Remove duplicates
    search_files = list(set(search_files))
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
- **Search Queries Found:** {len(search_files)}
- **WebFetch Operations:** {len(fetch_files)}
- **Total Search Files:** {len(search_files) + len(fetch_files)}

### Actual Search Results Found
"""

    if search_files:
        verification_report += "\n**Raw Search Data Files:**\n"
        for search_file in search_files[-5:]:  # Show last 5 files
            try:
                with open(search_file, encoding='utf-8') as f:
                    data = json.load(f)
                verification_report += f"\n- **File:** {search_file.name}"
                verification_report += f"\n  - Query: {data.get('search_query', 'N/A')}"
                verification_report += f"\n  - Results Length: {data.get('verification_data', {}).get('result_length', 'N/A')} chars"
                verification_report += f"\n  - Captured: {data.get('captured_at', 'N/A')}"
            except Exception as e:
                verification_report += f"\n- **File:** {search_file.name} (Error reading: {e})"
    else:
        verification_report += "\n❌ **NO ACTUAL WEB SEARCH RESULTS FOUND**"

    if fetch_files:
        verification_report += "\n\n**WebFetch Content Files:**\n"
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

{'✅ REAL WEB SEARCH DETECTED' if search_files or fetch_files else '❌ NO REAL WEB SEARCH - LIKELY LLM GENERATED'}

The system {'IS' if search_files or fetch_files else 'IS NOT'} performing actual web searches and retrieving real content from the internet.

{'Search results and source URLs are captured and saved for verification.' if search_files or fetch_files else 'No search results or source URLs were found, suggesting content may be from LLM knowledge rather than live web search.'}

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
            "text": f"Search verification report created. Real web search detected: {bool(search_files or fetch_files)}"
        }],
        "report_file": str(report_file),
        "verification_result": "REAL_SEARCH" if search_files or fetch_files else "LLM_ONLY",
        "session_id": session_id
    }
