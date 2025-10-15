"""
Research Threshold Monitor Hook

This hook monitors research progress in real-time and automatically stops
additional searches when the success threshold is achieved, improving
efficiency and preventing unnecessary resource consumption.

Features:
- Real-time monitoring of successful scrapes
- Automatic intervention when threshold is met
- Course correction to stop further searches
- Integration with Claude Agent SDK for agent guidance
- Configurable thresholds and research parameters
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, List
from pathlib import Path

# Import hook infrastructure
from .base_hooks import BaseHook, HookContext, HookPriority, HookResult, HookStatus
from .sdk_integration import SDKHookBridge

# Import logging with proper path handling
try:
    from agent_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)


@dataclass
class ResearchThresholdConfig:
    """Configuration for research threshold monitoring."""
    success_threshold: int = 10  # Stop when this many successful scrapes are achieved
    check_interval: float = 5.0  # Check every 5 seconds
    max_search_time: float = 300.0  # Maximum time for research (5 minutes)
    enable_auto_intervention: bool = True  # Automatically stop when threshold met
    agent_guidance_message: str = "SUCCESS_THRESHOLD_ACHIEVED"

    # Research session tracking
    session_file_patterns: List[str] = field(default_factory=lambda: [
        "1-search_workproduct_*.md",
        "1-expanded_search_workproduct_*.md",
        "INITIAL_SEARCH_*.md"
    ])


@dataclass
class ResearchProgress:
    """Current research progress tracking."""
    session_id: str
    total_searches: int = 0
    successful_scrapes: int = 0
    current_searches: int = 0
    last_check: datetime = field(default_factory=datetime.now)
    start_time: datetime = field(default_factory=datetime.now)
    threshold_met: bool = False
    intervention_sent: bool = False
    session_files: List[str] = field(default_factory=list)


class ResearchThresholdMonitorHook(BaseHook):
    """
    Hook that monitors research progress and intervenes when success threshold is met.

    This hook continuously monitors the research session's progress and automatically
    sends intervention signals to the agent when the configured success threshold
    is achieved, preventing unnecessary additional searches.
    """

    def __init__(self, config: Optional[ResearchThresholdConfig] = None):
        """Initialize the research threshold monitor hook.

        Monitors research progress and intervenes when success threshold is met.
        """
        super().__init__(
            name="research_threshold_monitor",
            hook_type="research_monitoring",
            priority=HookPriority.HIGH
        )

        self.config = config or ResearchThresholdConfig()
        self.logger = get_logger("research_threshold_monitor")

        # Track active research sessions
        self.active_sessions: Dict[str, ResearchProgress] = {}

        # SDK bridge for agent intervention
        self.sdk_bridge: Optional[SDKHookBridge] = None

        self.logger.info(f"🎯 Research Threshold Monitor Hook initialized")
        self.logger.info(f"   Success threshold: {self.config.success_threshold}")
        self.logger.info(f"   Check interval: {self.config.check_interval}s")
        self.logger.info(f"   Auto intervention: {self.config.enable_auto_intervention}")

    def set_sdk_bridge(self, sdk_bridge: SDKHookBridge):
        """Set the SDK bridge for agent intervention."""
        self.sdk_bridge = sdk_bridge
        self.logger.info("🔗 SDK bridge configured for agent intervention")

    async def execute(self, context: HookContext) -> HookResult:
        """
        Execute the research threshold monitoring logic.

        Args:
            context: Hook execution context with session information

        Returns:
            HookResult with monitoring status and any intervention actions taken
        """
        start_time = time.time()

        try:
            session_id = context.session_id

            # Initialize session tracking if needed
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = ResearchProgress(
                    session_id=session_id,
                    start_time=datetime.now()
                )
                self.logger.info(f"📊 Started monitoring research session: {session_id}")

            progress = self.active_sessions[session_id]

            # Update research progress
            await self._update_research_progress(progress, context)

            # Check if threshold is met
            threshold_met = progress.successful_scrapes >= self.config.success_threshold

            if threshold_met and not progress.threshold_met:
                progress.threshold_met = True
                self.logger.info(
                    f"🎯 SUCCESS THRESHOLD MET: {progress.successful_scrapes} successful scrapes "
                    f"(threshold: {self.config.success_threshold})"
                )

                # Send intervention if enabled
                if self.config.enable_auto_intervention and not progress.intervention_sent:
                    await self._send_threshold_intervention(progress, context)
                    progress.intervention_sent = True

            # Check for timeout
            elapsed_time = (datetime.now() - progress.start_time).total_seconds()
            timeout_reached = elapsed_time > self.config.max_search_time

            if timeout_reached and not progress.intervention_sent:
                self.logger.warning(
                    f"⏰ RESEARCH TIMEOUT: {elapsed_time:.1f}s elapsed (max: {self.config.max_search_time}s)"
                )
                await self._send_timeout_intervention(progress, context)
                progress.intervention_sent = True

            # Prepare result data
            result_data = {
                "session_id": session_id,
                "successful_scrapes": progress.successful_scrapes,
                "total_searches": progress.total_searches,
                "threshold_met": progress.threshold_met,
                "intervention_sent": progress.intervention_sent,
                "elapsed_time": elapsed_time,
                "threshold_percentage": (progress.successful_scrapes / self.config.success_threshold) * 100,
                "monitoring_active": not (progress.threshold_met or timeout_reached)
            }

            # Determine if monitoring should continue
            should_continue = not (progress.threshold_met or timeout_reached)

            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.COMPLETED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                execution_time=time.time() - start_time,
                result_data=result_data,
                metadata={
                    "should_continue": should_continue,
                    "next_check_in": self.config.check_interval if should_continue else None
                }
            )

        except Exception as e:
            self.logger.error(f"❌ Research threshold monitoring failed: {e}")
            return HookResult(
                hook_name=self.name,
                hook_type=self.hook_type,
                status=HookStatus.FAILED,
                execution_id=context.execution_id,
                start_time=context.timestamp,
                end_time=datetime.now(),
                execution_time=time.time() - start_time,
                error_message=str(e),
                error_type=type(e).__name__
            )

    async def _update_research_progress(self, progress: ResearchProgress, context: HookContext):
        """Update research progress by analyzing session files and metadata."""
        try:
            # Get session directory
            session_dir = self._get_session_directory(context.session_id)
            if not session_dir or not session_dir.exists():
                return

            # Look for research work products
            research_dir = session_dir / "research"
            if not research_dir.exists():
                return

            # Find recent work product files
            current_files = []
            for pattern in self.config.session_file_patterns:
                current_files.extend(research_dir.glob(pattern))

            # Sort by modification time
            current_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Analyze the most recent work product
            if current_files:
                latest_file = current_files[0]

                # Only re-analyze if file is newer than last check
                if (not progress.session_files or
                    str(latest_file) not in progress.session_files or
                    latest_file.stat().st_mtime > progress.last_check.timestamp()):

                    await self._analyze_work_product(latest_file, progress)
                    progress.session_files = [str(f) for f in current_files[:3]]  # Keep top 3

            progress.last_check = datetime.now()

        except Exception as e:
            self.logger.warning(f"⚠️  Failed to update research progress: {e}")

    async def _analyze_work_product(self, file_path: Path, progress: ResearchProgress):
        """Analyze a work product file to extract research progress."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for success indicators in the content
            success_indicators = [
                "successful scrapes:",
                "✅ successful",
                "successfully scraped",
                "completed scrape",
                "scrape successful"
            ]

            total_successes = 0
            for indicator in success_indicators:
                if indicator.lower() in content.lower():
                    # Try to extract numbers
                    import re
                    matches = re.findall(r'(\d+).*' + re.escape(indicator.lower()), content.lower())
                    for match in matches:
                        try:
                            total_successes = max(total_successes, int(match))
                        except ValueError:
                            continue

            # Also count individual success markers
            individual_successes = content.count("✅") + content.count("[SUCCESS]")
            total_successes = max(total_successes, individual_successes)

            # Look for search counts
            search_indicators = ["searches:", "search results:", "urls found:"]
            total_searches = 0
            for indicator in search_indicators:
                if indicator.lower() in content.lower():
                    import re
                    matches = re.findall(r'(\d+).*' + re.escape(indicator.lower()), content.lower())
                    for match in matches:
                        try:
                            total_searches = max(total_searches, int(match))
                        except ValueError:
                            continue

            # Update progress
            if total_successes > progress.successful_scrapes:
                old_count = progress.successful_scrapes
                progress.successful_scrapes = total_successes
                self.logger.info(f"📈 Updated successful scrapes: {old_count} → {total_successes}")

            if total_searches > progress.total_searches:
                progress.total_searches = total_searches
                self.logger.info(f"🔍 Updated total searches: {total_searches}")

        except Exception as e:
            self.logger.warning(f"⚠️  Failed to analyze work product {file_path}: {e}")

    def _get_session_directory(self, session_id: str) -> Optional[Path]:
        """Get the session directory for the given session ID."""
        try:
            # Try different possible base directories
            base_dirs = [
                Path.cwd() / "KEVIN" / "sessions" / session_id,
                Path("/home/kjdragan/lrepos/claudeagent-multiagent-latest/KEVIN/sessions") / session_id,
                Path.home() / "claudeagent-multiagent-latest/KEVIN/sessions" / session_id,
            ]

            for base_dir in base_dirs:
                if base_dir.exists():
                    return base_dir

            return None

        except Exception as e:
            self.logger.warning(f"⚠️  Could not determine session directory: {e}")
            return None

    async def _send_threshold_intervention(self, progress: ResearchProgress, context: HookContext):
        """Send intervention to agent when threshold is met."""
        try:
            intervention_message = f"""
🎯 **SUCCESS THRESHOLD ACHIEVED** - INTERVENTION REQUIRED

Your research has achieved the target success threshold:
- Successful scrapes: {progress.successful_scrapes} (target: {self.config.success_threshold})
- Total searches: {progress.total_searches}
- Elapsed time: {(datetime.now() - progress.start_time).total_seconds():.1f}s

**IMMEDIATE ACTION REQUIRED:**
STOP making additional search calls! You have gathered sufficient research.

**NEXT STEPS:**
1. Proceed to analyze the research you've collected
2. Begin drafting your research report
3. DO NOT make any more search queries

The research goal has been achieved efficiently. Move on to analysis and reporting.
"""

            # Send through SDK bridge if available
            if self.sdk_bridge:
                await self._send_sdk_intervention(intervention_message, context)
            else:
                # Log the intervention for agent to see
                self.logger.info("="*80)
                self.logger.info("🎯 THRESHOLD INTERVENTION MESSAGE")
                self.logger.info("="*80)
                self.logger.info(intervention_message)
                self.logger.info("="*80)

            # Also save to session file for persistence
            await self._save_intervention_message(progress.session_id, intervention_message, "threshold_met")

        except Exception as e:
            self.logger.error(f"❌ Failed to send threshold intervention: {e}")

    async def _send_timeout_intervention(self, progress: ResearchProgress, context: HookContext):
        """Send timeout intervention to agent."""
        try:
            intervention_message = f"""
⏰ **RESEARCH TIMEOUT** - INTERVENTION REQUIRED

Your research has exceeded the maximum allowed time:
- Elapsed time: {(datetime.now() - progress.start_time).total_seconds():.1f}s (max: {self.config.max_search_time}s)
- Successful scrapes: {progress.successful_scrapes}
- Total searches: {progress.total_searches}

**IMMEDIATE ACTION REQUIRED:**
STOP making additional search calls! Time limit exceeded.

**NEXT STEPS:**
1. Proceed with the research you've collected so far
2. Begin analysis and reporting
3. DO NOT make any more search queries

The research time budget has been exhausted. Work with what you have gathered.
"""

            # Send through SDK bridge if available
            if self.sdk_bridge:
                await self._send_sdk_intervention(intervention_message, context)
            else:
                # Log the intervention
                self.logger.info("="*80)
                self.logger.info("⏰ TIMEOUT INTERVENTION MESSAGE")
                self.logger.info("="*80)
                self.logger.info(intervention_message)
                self.logger.info("="*80)

            # Save to session file
            await self._save_intervention_message(progress.session_id, intervention_message, "timeout")

        except Exception as e:
            self.logger.error(f"❌ Failed to send timeout intervention: {e}")

    async def _send_sdk_intervention(self, message: str, context: HookContext):
        """Send intervention through Claude Agent SDK."""
        try:
            if self.sdk_bridge and hasattr(self.sdk_bridge, 'send_agent_message'):
                await self.sdk_bridge.send_agent_message(
                    session_id=context.session_id,
                    agent_name=context.agent_name,
                    message=message,
                    message_type="intervention"
                )
                self.logger.info("📤 Intervention sent via SDK bridge")
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to send SDK intervention: {e}")

    async def _save_intervention_message(self, session_id: str, message: str, reason: str):
        """Save intervention message to session file."""
        try:
            session_dir = self._get_session_directory(session_id)
            if session_dir:
                intervention_file = session_dir / "working" / f"INTERVENTION_{reason.upper()}_{int(time.time())}.md"

                with open(intervention_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Research Intervention - {reason.upper()}\n\n")
                    f.write(f"**Timestamp**: {datetime.now().isoformat()}\n")
                    f.write(f"**Session ID**: {session_id}\n")
                    f.write(f"**Reason**: {reason}\n\n")
                    f.write("## Intervention Message\n\n")
                    f.write(message)

                self.logger.info(f"💾 Intervention saved to: {intervention_file}")

        except Exception as e:
            self.logger.warning(f"⚠️  Failed to save intervention message: {e}")

    def get_session_progress(self, session_id: str) -> Optional[ResearchProgress]:
        """Get current progress for a session."""
        return self.active_sessions.get(session_id)

    def clear_session(self, session_id: str):
        """Clear session from monitoring."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"🗑️  Cleared session {session_id} from monitoring")

    def get_all_sessions(self) -> Dict[str, ResearchProgress]:
        """Get all active sessions."""
        return self.active_sessions.copy()


# Factory function for easy instantiation
def create_research_threshold_monitor_hook(
    success_threshold: int = 10,
    check_interval: float = 5.0,
    max_search_time: float = 300.0,
    enable_auto_intervention: bool = True
) -> ResearchThresholdMonitorHook:
    """
    Factory function to create a configured research threshold monitor hook.

    Args:
        success_threshold: Number of successful scrapes before intervention
        check_interval: How often to check progress (seconds)
        max_search_time: Maximum time allowed for research (seconds)
        enable_auto_intervention: Whether to automatically intervene

    Returns:
        Configured ResearchThresholdMonitorHook instance
    """
    config = ResearchThresholdConfig(
        success_threshold=success_threshold,
        check_interval=check_interval,
        max_search_time=max_search_time,
        enable_auto_intervention=enable_auto_intervention
    )

    return ResearchThresholdMonitorHook(config)