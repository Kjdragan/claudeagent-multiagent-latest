"""
Performance Timer Infrastructure

Provides lightweight timer decorators and context managers for tracking
async function performance without real-time dashboards. Designed for
end-of-run performance reporting as specified in asynceval1.md.

Features:
- @async_timed decorator for async functions
- timed_block() context manager for code blocks
- Session-based performance tracking
- JSON report generation
- Thread-safe operation
"""

import functools
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TimerResult:
    """Single timer measurement."""
    function_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTimer:
    """Global performance timer collector for session-based tracking."""

    def __init__(self):
        self.timers: List[TimerResult] = []
        self.session_start: Optional[float] = None
        self.session_id: Optional[str] = None
        self._enabled = True  # Can be disabled for testing

    def record(self, timer_result: TimerResult):
        """Record a timer result."""
        if self._enabled:
            self.timers.append(timer_result)

    def start_session(self, session_id: str):
        """Start timing a session."""
        self.session_id = session_id
        self.session_start = time.time()
        logger.info(f"⏱️  Performance tracking started for session {session_id}")

    def reset(self):
        """Reset timer for new session."""
        self.timers.clear()
        self.session_start = None
        self.session_id = None

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Dictionary with session statistics, function-level metrics, and timeline
        """
        if not self.timers:
            return {"error": "No timing data collected"}

        session_duration = time.time() - self.session_start if self.session_start else 0

        # Aggregate by function name
        function_stats = {}
        for timer in self.timers:
            if timer.function_name not in function_stats:
                function_stats[timer.function_name] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "successful": 0,
                    "failed": 0,
                    "min_duration": float('inf'),
                    "max_duration": 0.0,
                    "errors": []
                }

            stats = function_stats[timer.function_name]
            stats["count"] += 1
            stats["total_duration"] += timer.duration
            stats["min_duration"] = min(stats["min_duration"], timer.duration)
            stats["max_duration"] = max(stats["max_duration"], timer.duration)

            if timer.success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
                if timer.error:
                    stats["errors"].append(timer.error)

        # Calculate averages
        for stats in function_stats.values():
            stats["avg_duration"] = stats["total_duration"] / stats["count"]
            # Clean up errors list if empty
            if not stats["errors"]:
                del stats["errors"]

        # Build timeline
        timeline = [
            {
                "function": t.function_name,
                "start": t.start_time,
                "duration": t.duration,
                "success": t.success,
                "metadata": t.metadata
            }
            for t in self.timers
        ]

        # Calculate performance insights
        insights = self._calculate_insights(function_stats, session_duration)

        return {
            "session_id": self.session_id,
            "session_duration": round(session_duration, 2),
            "total_operations": len(self.timers),
            "function_statistics": function_stats,
            "timeline": timeline,
            "performance_insights": insights
        }

    def _calculate_insights(self, function_stats: dict, session_duration: float) -> dict:
        """Calculate performance insights from statistics."""
        insights = {}

        # Identify scraping operations
        scrape_ops = [k for k in function_stats.keys() if 'crawl' in k.lower() or 'scrape' in k.lower()]
        if scrape_ops:
            scrape_total = sum(function_stats[op]["total_duration"] for op in scrape_ops)
            scrape_count = sum(function_stats[op]["count"] for op in scrape_ops)
            insights["scraping_efficiency"] = f"{scrape_count} operations in {scrape_total:.1f}s (avg: {scrape_total/scrape_count:.2f}s/op)"

        # Identify cleaning operations
        clean_ops = [k for k in function_stats.keys() if 'clean' in k.lower()]
        if clean_ops:
            clean_total = sum(function_stats[op]["total_duration"] for op in clean_ops)
            clean_count = sum(function_stats[op]["count"] for op in clean_ops)
            insights["cleaning_efficiency"] = f"{clean_count} operations in {clean_total:.1f}s (avg: {clean_total/clean_count:.2f}s/op)"

        # Check for sequential processing (bottleneck)
        if scrape_ops and clean_ops:
            scrape_batch_ops = [k for k in function_stats.keys() if 'batch' in k.lower() and 'scrape' in k.lower()]
            clean_batch_ops = [k for k in function_stats.keys() if 'batch' in k.lower() and 'clean' in k.lower()]

            if scrape_batch_ops and clean_batch_ops:
                insights["bottleneck"] = "Sequential processing - scraping and cleaning in separate batches"
                scrape_time = function_stats[scrape_batch_ops[0]]["total_duration"]
                clean_time = function_stats[clean_batch_ops[0]]["total_duration"]
                potential_savings = max(scrape_time, clean_time) - (scrape_time + clean_time - max(scrape_time, clean_time))
                insights["optimization_potential"] = f"~{potential_savings:.0f}s savings possible with streaming processing"
            else:
                insights["processing_mode"] = "Streaming parallel processing detected"

        return insights

    def save_report(self, filepath: str | Path):
        """
        Save performance report to JSON file.

        Args:
            filepath: Path to save the report
        """
        report = self.generate_report()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"📊 Performance report saved to: {filepath}")

    def print_summary(self):
        """Print a concise performance summary to console."""
        report = self.generate_report()

        if "error" in report:
            logger.warning(f"No performance data: {report['error']}")
            return

        print("\n" + "=" * 80)
        print("Performance Report Summary")
        print("=" * 80)
        print(f"Session ID: {report['session_id']}")
        print(f"Total Duration: {report['session_duration']:.2f}s")
        print(f"Total Operations: {report['total_operations']}")
        print()

        # Function statistics
        print("Function Statistics:")
        print("-" * 80)
        for func_name, stats in report["function_statistics"].items():
            success_rate = (stats["successful"] / stats["count"]) * 100
            print(f"  {func_name}:")
            print(f"    Count: {stats['count']} | Success Rate: {success_rate:.1f}%")
            print(f"    Duration: avg={stats['avg_duration']:.2f}s, min={stats['min_duration']:.2f}s, max={stats['max_duration']:.2f}s")

        # Performance insights
        if report["performance_insights"]:
            print()
            print("Performance Insights:")
            print("-" * 80)
            for key, value in report["performance_insights"].items():
                print(f"  {key}: {value}")

        print("=" * 80)
        print()


# Global timer instance (singleton pattern)
_performance_timer: Optional[PerformanceTimer] = None


def get_performance_timer() -> PerformanceTimer:
    """Get global performance timer instance (singleton)."""
    global _performance_timer
    if _performance_timer is None:
        _performance_timer = PerformanceTimer()
    return _performance_timer


def reset_performance_timer():
    """Reset the global performance timer (useful for testing)."""
    global _performance_timer
    _performance_timer = None


# Decorator for timing async functions
def async_timed(metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to time async functions and record results.

    Usage:
        @async_timed(metadata={"category": "scraping"})
        async def scrape_url(url: str):
            ...

    Args:
        metadata: Optional metadata to attach to timer result
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            timer = get_performance_timer()
            start_time = time.time()
            success = True
            error_msg = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time

                timer_result = TimerResult(
                    function_name=func.__name__,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    success=success,
                    error=error_msg,
                    metadata=metadata or {}
                )
                timer.record(timer_result)

                # Log completion
                status = "✅" if success else "❌"
                logger.debug(f"{status} {func.__name__} completed in {duration:.2f}s")

        return wrapper
    return decorator


# Context manager for timing code blocks
@asynccontextmanager
async def timed_block(block_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager for timing code blocks.

    Usage:
        async with timed_block("url_scraping_batch", metadata={"urls": 10}):
            results = await scrape_urls(urls)

    Args:
        block_name: Name to identify this code block
        metadata: Optional metadata to attach to timer result
    """
    timer = get_performance_timer()
    start_time = time.time()
    success = True
    error_msg = None

    try:
        yield
    except Exception as e:
        success = False
        error_msg = str(e)
        raise
    finally:
        end_time = time.time()
        duration = end_time - start_time

        timer_result = TimerResult(
            function_name=block_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success=success,
            error=error_msg,
            metadata=metadata or {}
        )
        timer.record(timer_result)

        # Log completion
        status = "✅" if success else "❌"
        logger.debug(f"{status} {block_name} completed in {duration:.2f}s")


# Convenience function for saving performance report at session end
def save_session_performance_report(session_id: str, working_dir: str | Path):
    """
    Save performance report for a session to its working directory.

    Args:
        session_id: Session identifier
        working_dir: Path to session's working directory
    """
    timer = get_performance_timer()

    if timer.session_id != session_id:
        logger.warning(f"Timer session ID mismatch: expected {session_id}, got {timer.session_id}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(working_dir) / f"performance_report_{timestamp}.json"

    try:
        timer.save_report(report_file)
        timer.print_summary()
    except Exception as e:
        logger.error(f"Failed to save performance report: {e}")
