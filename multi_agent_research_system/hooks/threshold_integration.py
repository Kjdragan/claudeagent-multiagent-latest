"""
Threshold Integration Hook Manager

Integrates the research threshold monitor hook with the main research system
and Claude Agent SDK for real-time monitoring and intervention.

This module provides the integration layer needed to hook into the research
workflow and automatically stop searches when the success threshold is met.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional, List

# Import hook infrastructure
from .research_threshold_monitor import (
    ResearchThresholdMonitorHook,
    ResearchThresholdConfig,
    create_research_threshold_monitor_hook
)
from .comprehensive_hooks import ComprehensiveHookManager, HookCategory
from .sdk_integration import SDKHookBridge

# Import logging
try:
    from agent_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)


class ThresholdIntegrationManager:
    """
    Integration manager for research threshold monitoring.

    This class manages the integration of the threshold monitor hook with the
    research system, providing setup, configuration, and coordination services.
    """

    def __init__(self):
        """Initialize the threshold integration manager."""
        self.logger = get_logger("threshold_integration_manager")
        self.hook_manager: Optional[ComprehensiveHookManager] = None
        self.threshold_hook: Optional[ResearchThresholdMonitorHook] = None
        self.sdk_bridge: Optional[SDKHookBridge] = None
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.is_initialized = False

    async def initialize(self, hook_manager: ComprehensiveHookManager) -> bool:
        """
        Initialize the threshold integration with the hook manager.

        Args:
            hook_manager: The comprehensive hook manager to integrate with

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            self.hook_manager = hook_manager

            # Create and configure the threshold monitor hook
            self.threshold_hook = create_research_threshold_monitor_hook(
                success_threshold=10,  # Stop after 10 successful scrapes
                check_interval=3.0,    # Check every 3 seconds
                max_search_time=240.0, # 4 minute max
                enable_auto_intervention=True
            )

            # Create SDK bridge if available
            try:
                self.sdk_bridge = SDKHookBridge(hook_manager)
                self.threshold_hook.set_sdk_bridge(self.sdk_bridge)
                self.logger.info("🔗 SDK bridge configured for threshold interventions")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not create SDK bridge: {e}")

            # Note: Hook registration is handled elsewhere in the system
            # The threshold hook will be available through the hook manager
            self.logger.info("✅ Threshold hook integration completed")

            self.is_initialized = True
            self.logger.info("✅ Threshold integration manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize threshold integration manager: {e}")
            return False

    async def start_monitoring_session(self, session_id: str, agent_name: Optional[str] = None) -> bool:
        """
        Start monitoring a research session for threshold compliance.

        Args:
            session_id: The research session ID to monitor
            agent_name: Name of the agent conducting the research

        Returns:
            True if monitoring started successfully, False otherwise
        """
        if not self.is_initialized:
            self.logger.error("❌ Threshold integration manager not initialized")
            return False

        try:
            # Create monitoring context
            from .base_hooks import HookContext
            context = HookContext(
                hook_name="research_threshold_monitor",
                hook_type="research_monitoring",
                session_id=session_id,
                agent_name=agent_name,
                workflow_stage="research"
            )

            # Start periodic monitoring task
            if session_id not in self.monitoring_tasks:
                task = asyncio.create_task(
                    self._monitor_session_loop(session_id, context)
                )
                self.monitoring_tasks[session_id] = task
                self.logger.info(f"🔍 Started threshold monitoring for session: {session_id}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to start monitoring session {session_id}: {e}")
            return False

    async def stop_monitoring_session(self, session_id: str) -> bool:
        """
        Stop monitoring a research session.

        Args:
            session_id: The research session ID to stop monitoring

        Returns:
            True if monitoring stopped successfully, False otherwise
        """
        try:
            if session_id in self.monitoring_tasks:
                task = self.monitoring_tasks[session_id]
                task.cancel()
                del self.monitoring_tasks[session_id]

                # Clear session from hook
                if self.threshold_hook:
                    self.threshold_hook.clear_session(session_id)

                self.logger.info(f"⏹️  Stopped threshold monitoring for session: {session_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"❌ Failed to stop monitoring session {session_id}: {e}")
            return False

    async def _monitor_session_loop(self, session_id: str, context):
        """
        Periodic monitoring loop for a research session.

        Args:
            session_id: The session ID to monitor
            context: Hook context for monitoring
        """
        try:
            while True:
                # Check if session is still active
                if not self._is_session_active(session_id):
                    self.logger.info(f"📋 Session {session_id} appears inactive, stopping monitoring")
                    await self.stop_monitoring_session(session_id)
                    break

                # Execute threshold monitoring hook
                if self.threshold_hook:
                    result = await self.threshold_hook.execute(context)

                    # Check if monitoring should continue
                    if (result.result_data and
                        not result.result_data.get("monitoring_active", True)):
                        self.logger.info(
                            f"🎯 Threshold monitoring completed for session {session_id}: "
                            f"{'Threshold met' if result.result_data.get('threshold_met') else 'Timeout'}"
                        )
                        await self.stop_monitoring_session(session_id)
                        break

                    # Log progress
                    if result.result_data:
                        progress = result.result_data
                        self.logger.info(
                            f"📊 Session {session_id[:8]} progress: "
                            f"{progress['successful_scrapes']}/{progress['threshold_percentage']:.1f}% "
                            f"({'✅ Done' if progress['threshold_met'] else '🔄 Active'})"
                        )

                # Wait for next check
                await asyncio.sleep(3.0)  # Check every 3 seconds

        except asyncio.CancelledError:
            self.logger.info(f"📋 Monitoring loop cancelled for session {session_id}")
        except Exception as e:
            self.logger.error(f"❌ Monitoring loop error for session {session_id}: {e}")

    def _is_session_active(self, session_id: str) -> bool:
        """
        Check if a research session is still active.

        Args:
            session_id: The session ID to check

        Returns:
            True if session appears active, False otherwise
        """
        try:
            # Check if we have recent progress
            if self.threshold_hook:
                progress = self.threshold_hook.get_session_progress(session_id)
                if progress:
                    # Consider session active if we've seen activity in the last 2 minutes
                    time_since_last_check = (datetime.now() - progress.last_check).total_seconds()
                    return time_since_last_check < 120

            return False

        except Exception as e:
            self.logger.warning(f"⚠️  Error checking session activity: {e}")
            return False

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a monitored session.

        Args:
            session_id: The session ID to check

        Returns:
            Session status dictionary or None if not found
        """
        if not self.threshold_hook:
            return None

        progress = self.threshold_hook.get_session_progress(session_id)
        if not progress:
            return None

        return {
            "session_id": session_id,
            "successful_scrapes": progress.successful_scrapes,
            "total_searches": progress.total_searches,
            "threshold_met": progress.threshold_met,
            "intervention_sent": progress.intervention_sent,
            "elapsed_time": (datetime.now() - progress.start_time).total_seconds(),
            "monitoring_active": session_id in self.monitoring_tasks,
            "last_check": progress.last_check.isoformat(),
            "threshold_percentage": min(100, (progress.successful_scrapes / 10) * 100)
        }

    def get_all_sessions_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all monitored sessions.

        Returns:
            Dictionary of session statuses
        """
        if not self.threshold_hook:
            return {}

        all_sessions = self.threshold_hook.get_all_sessions()
        return {
            session_id: self.get_session_status(session_id)
            for session_id in all_sessions.keys()
        }

    async def configure_threshold(self, session_id: str, **kwargs) -> bool:
        """
        Configure threshold settings for a specific session.

        Args:
            session_id: The session to configure
            **kwargs: Configuration parameters to override

        Returns:
            True if configuration was successful, False otherwise
        """
        try:
            if not self.threshold_hook:
                return False

            # Update configuration
            for key, value in kwargs.items():
                if hasattr(self.threshold_hook.config, key):
                    setattr(self.threshold_hook.config, key, value)
                    self.logger.info(f"⚙️  Updated {key} = {value} for session {session_id}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to configure threshold for session {session_id}: {e}")
            return False

    async def cleanup(self):
        """Clean up all monitoring tasks and resources."""
        try:
            # Cancel all monitoring tasks
            for session_id in list(self.monitoring_tasks.keys()):
                await self.stop_monitoring_session(session_id)

            # Clear hook sessions
            if self.threshold_hook:
                all_sessions = self.threshold_hook.get_all_sessions()
                for session_id in all_sessions.keys():
                    self.threshold_hook.clear_session(session_id)

            self.logger.info("🧹 Threshold integration manager cleaned up")

        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")


# Global instance for easy access
_threshold_integration_manager: Optional[ThresholdIntegrationManager] = None


def get_threshold_integration_manager() -> ThresholdIntegrationManager:
    """
    Get the global threshold integration manager instance.

    Returns:
        ThresholdIntegrationManager instance
    """
    global _threshold_integration_manager
    if _threshold_integration_manager is None:
        _threshold_integration_manager = ThresholdIntegrationManager()
    return _threshold_integration_manager


async def setup_threshold_monitoring(
    hook_manager: ComprehensiveHookManager,
    success_threshold: int = 10,
    check_interval: float = 3.0,
    max_search_time: float = 240.0
) -> ThresholdIntegrationManager:
    """
    Set up threshold monitoring with the hook manager.

    Args:
        hook_manager: The comprehensive hook manager to integrate with
        success_threshold: Number of successful scrapes before intervention
        check_interval: How often to check progress (seconds)
        max_search_time: Maximum time allowed for research (seconds)

    Returns:
        Configured ThresholdIntegrationManager instance
    """
    manager = get_threshold_integration_manager()

    # Initialize with hook manager
    success = await manager.initialize(hook_manager)
    if not success:
        raise RuntimeError("Failed to initialize threshold integration manager")

    # Configure threshold settings
    await manager.configure_threshold(
        session_id="default",
        success_threshold=success_threshold,
        check_interval=check_interval,
        max_search_time=max_search_time
    )

    return manager