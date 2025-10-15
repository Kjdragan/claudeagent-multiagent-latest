"""
Research Threshold Tracker

A simple utility to track research progress across multiple tool calls
and prevent unnecessary searches when the success threshold is met.

This provides a lightweight alternative to the complex hook system for
threshold monitoring in MCP tools.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

# Import logging
try:
    from agent_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)


class ResearchThresholdTracker:
    """
    Simple threshold tracker for research sessions.

    Tracks successful scrapes across multiple tool calls and provides
    threshold checking functionality to prevent unnecessary searches.
    """

    def __init__(self, default_threshold: int = 10):
        """Initialize the threshold tracker."""
        self.default_threshold = default_threshold
        # Different thresholds for different search types
        self.thresholds = {
            "default": default_threshold,           # Main initial research
            "enhanced_search": 5,                  # Enhanced search tools
            "news_search": 5,                      # News search
            "expanded_query": 5,                   # Expanded query search
            "gap_research": 3,                      # Gap research (editorial)
            "zplayground1": default_threshold       # zplayground1 search
        }
        self.logger = get_logger("research_threshold_tracker")
        self.session_file_pattern = "session_state.json"
        self.intervention_file_pattern = "THRESHOLD_INTERVENTION_*.md"

    def _get_session_state_file(self, session_id: str) -> Path:
        """Get the session state file path."""
        # Try different possible base directories
        base_dirs = [
            Path.cwd() / "KEVIN" / "sessions" / session_id,
            Path("/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions") / session_id,
            Path.home() / "claudeagent-multiagent-latest/KEVIN/sessions" / session_id,
        ]

        for base_dir in base_dirs:
            if base_dir.exists():
                return base_dir / self.session_file_pattern

        # Fallback - create the directory
        fallback_dir = Path("/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions") / session_id
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir / self.session_file_pattern

    def _get_working_directory(self, session_id: str) -> Path:
        """Get the working directory for session files."""
        base_dirs = [
            Path.cwd() / "KEVIN" / "sessions" / session_id / "working",
            Path("/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions") / session_id / "working",
        ]

        for base_dir in base_dirs:
            if base_dir.exists():
                return base_dir

        # Fallback - create the directory
        fallback_dir = Path("/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions") / session_id / "working"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir

    def load_session_state(self, session_id: str) -> Dict[str, Any]:
        """Load session state from file."""
        try:
            state_file = self._get_session_state_file(session_id)
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"⚠️  Could not load session state for {session_id}: {e}")

        # Return default state
        return {
            "session_id": session_id,
            "successful_scrapes": 0,
            "total_searches": 0,
            "threshold_met": False,
            "intervention_sent": False,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }

    def save_session_state(self, session_id: str, state: Dict[str, Any]):
        """Save session state to file."""
        try:
            state_file = self._get_session_state_file(session_id)
            state["last_updated"] = datetime.now().isoformat()

            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.warning(f"⚠️  Could not save session state for {session_id}: {e}")

    def update_progress_from_workproduct(self, session_id: str) -> Dict[str, Any]:
        """Update progress by analyzing recent work products."""
        state = self.load_session_state(session_id)

        try:
            # Get research directory
            session_dir = self._get_session_state_file(session_id).parent
            research_dir = session_dir / "research"

            if not research_dir.exists():
                return state

            # Find recent work product files
            import glob
            workproduct_files = []
            for pattern in ["*search_workproduct_*.md", "*INITIAL_SEARCH_*.md"]:
                workproduct_files.extend(glob.glob(str(research_dir / pattern)))

            # Sort by modification time
            workproduct_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

            # Analyze the most recent work product
            if workproduct_files:
                latest_file = workproduct_files[0]
                file_mtime = datetime.fromtimestamp(os.path.getmtime(latest_file))

                # Only re-analyze if file is newer than last update
                last_updated = datetime.fromisoformat(state["last_updated"])
                if file_mtime > last_updated:
                    self._analyze_work_product_file(latest_file, state)

        except Exception as e:
            self.logger.warning(f"⚠️  Error updating progress from workproducts: {e}")

        return state

    def _analyze_work_product_file(self, file_path: str, state: Dict[str, Any]):
        """Analyze a work product file to extract progress."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for success indicators
            success_patterns = [
                r'(?:successful scrapes|successfully scraped|URLs Crawled):\s*(\d+)',
                r'✅\s*\d+\s*[^\n]*scrapes?[^:]*:\s*(\d+)',
                r'\[(?:SUCCESS|COMPLETED)\].*?(\d+)\s*scrapes?',
            ]

            max_successes = 0
            for pattern in success_patterns:
                import re
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    try:
                        max_successes = max(max_successes, int(match))
                    except ValueError:
                        continue

            # Also count individual success markers
            individual_successes = content.count("✅ Scraped:")
            max_successes = max(max_successes, individual_successes)

            # Update state if we found more successes
            if max_successes > state["successful_scrapes"]:
                old_count = state["successful_scrapes"]
                state["successful_scrapes"] = max_successes
                self.logger.info(f"📈 Updated successful scrapes: {old_count} → {max_successes}")

            # Determine appropriate threshold (default to main research threshold)
            current_threshold = self.thresholds.get("default", self.default_threshold)

            # Check if threshold is met
            if max_successes >= current_threshold:
                state["threshold_met"] = True
                self.logger.info(f"🎯 Threshold met: {max_successes} >= {current_threshold}")

        except Exception as e:
            self.logger.warning(f"⚠️  Error analyzing work product {file_path}: {e}")

    def should_continue_search(self, session_id: str, search_type: str = "default") -> bool:
        """
        Check if search should continue based on threshold.

        Args:
            session_id: Research session ID
            search_type: Type of search being performed

        Returns:
            False if threshold is met and intervention should be sent, True otherwise
        """
        state = self.update_progress_from_workproduct(session_id)

        # Get appropriate threshold for this search type
        threshold = self.thresholds.get(search_type, self.default_threshold)

        # Check if threshold is met
        threshold_met = state["successful_scrapes"] >= threshold

        # Save updated state
        self.save_session_state(session_id, state)

        if threshold_met and not state["intervention_sent"]:
            self.logger.info(f"🎯 {search_type} threshold met: {state['successful_scrapes']} >= {threshold}")
            return False  # Should not continue - need to send intervention

        return True  # Should continue

    def send_intervention(self, session_id: str, query: str, search_type: str = "default") -> str:
        """
        Generate and save intervention message.

        Args:
            session_id: Research session ID
            query: Current search query
            search_type: Type of search being performed

        Returns:
            Intervention message that should be returned to the agent
        """
        state = self.load_session_state(session_id)
        threshold = self.thresholds.get(search_type, self.default_threshold)

        intervention_message = f"""
🎯 **SUCCESS THRESHOLD ACHIEVED** - STOP SEARCHING

Your {search_type} research has achieved the target success threshold:
- Successful scrapes: {state['successful_scrapes']} (target: {threshold})
- Query: {query}

**IMMEDIATE ACTION REQUIRED:**
STOP making additional search calls! You have gathered sufficient research.

**NEXT STEPS:**
1. Proceed to analyze the research you've collected
2. Begin drafting your research report
3. DO NOT make any more search queries

The research goal has been achieved efficiently. Move on to analysis and reporting.
"""

        # Save intervention to working directory
        try:
            working_dir = self._get_working_directory(session_id)
            intervention_file = working_dir / f"THRESHOLD_INTERVENTION_{search_type}_{int(time.time())}.md"

            with open(intervention_file, 'w', encoding='utf-8') as f:
                f.write(f"# Research Threshold Intervention\n\n")
                f.write(f"**Timestamp**: {datetime.now().isoformat()}\n")
                f.write(f"**Session ID**: {session_id}\n")
                f.write(f"**Search Type**: {search_type}\n")
                f.write(f"**Threshold Met**: {state['successful_scrapes']} >= {threshold}\n\n")
                f.write("## Intervention Message\n\n")
                f.write(intervention_message)

            self.logger.info(f"💾 Intervention saved to: {intervention_file}")

        except Exception as e:
            self.logger.warning(f"⚠️  Could not save intervention message: {e}")

        # Mark intervention as sent
        state["intervention_sent"] = True
        state["intervention_sent_at"] = datetime.now().isoformat()
        self.save_session_state(session_id, state)

        return intervention_message

    def reset_session(self, session_id: str):
        """Reset session state."""
        try:
            state_file = self._get_session_state_file(session_id)
            if state_file.exists():
                state_file.unlink()

            # Also remove intervention files
            working_dir = self._get_working_directory(session_id)
            for intervention_file in working_dir.glob(self.intervention_file_pattern):
                intervention_file.unlink()

            self.logger.info(f"🗑️  Reset session state for: {session_id}")

        except Exception as e:
            self.logger.warning(f"⚠️  Error resetting session {session_id}: {e}")


# Global tracker instance
_global_tracker: Optional[ResearchThresholdTracker] = None


def get_research_threshold_tracker(threshold: int = 10) -> ResearchThresholdTracker:
    """Get the global threshold tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ResearchThresholdTracker(threshold)
    return _global_tracker


def check_search_threshold(session_id: str, query: str, search_type: str = "default") -> Optional[str]:
    """
    Check if search should continue based on threshold.

    Args:
        session_id: Research session ID
        query: Current search query
        search_type: Type of search being performed

    Returns:
        Intervention message if threshold is met, None if search should continue
    """
    tracker = get_research_threshold_tracker()

    if not tracker.should_continue_search(session_id, search_type):
        return tracker.send_intervention(session_id, query, search_type)

    return None


def reset_session_threshold(session_id: str):
    """Reset threshold tracking for a session."""
    tracker = get_research_threshold_tracker()
    tracker.reset_session(session_id)