"""
Log Aggregator for centralized log collection and indexing.

This module provides centralized log collection from all system components,
including agent logs, monitoring data, and system events.
"""

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agent_logging import StructuredLogger


@dataclass
class LogEntry:
    """Unified log entry structure for all log sources."""
    timestamp: datetime
    level: str
    source: str  # Component that generated the log
    session_id: str
    agent_name: Optional[str]
    activity_type: Optional[str]
    message: str
    metadata: Dict[str, Any]
    correlation_id: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class LogSource:
    """Configuration for a log source."""
    name: str
    path: Path
    pattern: str  # File pattern to match
    format: str  # 'json', 'structured', 'plain'
    enabled: bool = True
    priority: int = 1  # Higher priority sources are processed first


class LogAggregator:
    """Centralized log aggregation and indexing system."""

    def __init__(self,
                 session_id: str,
                 aggregation_dir: str = "log_aggregation",
                 max_entries: int = 100000,
                 retention_days: int = 30):
        """
        Initialize the log aggregator.

        Args:
            session_id: Session identifier for log grouping
            aggregation_dir: Directory to store aggregated logs
            max_entries: Maximum number of log entries to keep in memory
            retention_days: Days to retain log entries
        """
        self.session_id = session_id
        self.aggregation_dir = Path(aggregation_dir)
        self.aggregation_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.retention_days = retention_days

        # Initialize structured logger
        self.logger = StructuredLogger(
            name="log_aggregator",
            log_dir=self.aggregation_dir
        )

        # Log storage
        self.log_entries: List[LogEntry] = []
        self.log_sources: Dict[str, LogSource] = {}
        self.indexed_fields: Dict[str, Dict[str, Set]] = defaultdict(lambda: defaultdict(set))
        self.session_activities: Dict[str, List[LogEntry]] = defaultdict(list)

        # Aggregation statistics
        self.stats = {
            'total_entries': 0,
            'entries_by_source': defaultdict(int),
            'entries_by_level': defaultdict(int),
            'last_aggregation': None,
            'aggregation_errors': 0
        }

        # Background aggregation task
        self.aggregation_task: Optional[asyncio.Task] = None
        self.is_aggregating = False

        self._initialize_default_sources()

    def _initialize_default_sources(self) -> None:
        """Initialize default log sources from the system."""
        base_dir = Path.cwd()

        # Agent log sources
        agent_log_dirs = [
            ("research_agent", base_dir / "logs" / "research_agent"),
            ("report_agent", base_dir / "logs" / "report_agent"),
            ("editor_agent", base_dir / "logs" / "editor_agent"),
            ("ui_coordinator", base_dir / "logs" / "ui_coordinator")
        ]

        for agent_name, log_dir in agent_log_dirs:
            if log_dir.exists():
                self.add_log_source(LogSource(
                    name=f"{agent_name}_logs",
                    path=log_dir,
                    pattern="*.jsonl",
                    format="json",
                    priority=2
                ))

        # System monitoring log sources
        monitoring_dirs = [
            ("metrics", base_dir / "metrics"),
            ("health", base_dir / "health"),
            ("diagnostics", base_dir / "diagnostics")
        ]

        for source_name, log_dir in monitoring_dirs:
            if log_dir.exists():
                self.add_log_source(LogSource(
                    name=f"{source_name}_logs",
                    path=log_dir,
                    pattern="*.json",
                    format="json",
                    priority=1
                ))

        # Hook logs
        hooks_dir = base_dir / "logs" / "hooks"
        if hooks_dir.exists():
            self.add_log_source(LogSource(
                name="hooks_logs",
                path=hooks_dir,
                pattern="*.jsonl",
                format="json",
                priority=3
            ))

        self.logger.info("Default log sources initialized",
                        total_sources=len(self.log_sources))

    def add_log_source(self, log_source: LogSource) -> None:
        """
        Add a new log source to monitor.

        Args:
            log_source: LogSource configuration
        """
        self.log_sources[log_source.name] = log_source
        self.logger.info(f"Added log source: {log_source.name}",
                        source_name=log_source.name,
                        source_path=str(log_source.path),
                        pattern=log_source.pattern)

    def remove_log_source(self, source_name: str) -> None:
        """
        Remove a log source from monitoring.

        Args:
            source_name: Name of the log source to remove
        """
        if source_name in self.log_sources:
            del self.log_sources[source_name]
            self.logger.info(f"Removed log source: {source_name}")

    async def start_aggregation(self, interval_seconds: int = 60) -> None:
        """
        Start the background log aggregation task.

        Args:
            interval_seconds: Seconds between aggregation cycles
        """
        if self.is_aggregating:
            return

        self.is_aggregating = True
        self.aggregation_task = asyncio.create_task(self._aggregation_loop(interval_seconds))
        self.logger.info("Log aggregation started",
                        session_id=self.session_id,
                        interval_seconds=interval_seconds)

    async def stop_aggregation(self) -> None:
        """Stop the background log aggregation task."""
        self.is_aggregating = False
        if self.aggregation_task:
            self.aggregation_task.cancel()
            try:
                await self.aggregation_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Log aggregation stopped",
                        session_id=self.session_id)

    async def _aggregation_loop(self, interval_seconds: int) -> None:
        """Main aggregation loop that runs periodically."""
        while self.is_aggregating:
            try:
                await self.aggregate_logs()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                self.stats['aggregation_errors'] += 1
                self.logger.error(f"Error in aggregation loop: {e}",
                                session_id=self.session_id)
                await asyncio.sleep(interval_seconds)

    async def aggregate_logs(self) -> Dict[str, Any]:
        """
        Aggregate logs from all configured sources.

        Returns:
            Aggregation statistics
        """
        aggregation_start = datetime.now()
        new_entries = []

        # Sort sources by priority
        sorted_sources = sorted(
            self.log_sources.values(),
            key=lambda x: x.priority,
            reverse=True
        )

        for log_source in sorted_sources:
            if not log_source.enabled:
                continue

            try:
                source_entries = await self._aggregate_from_source(log_source)
                new_entries.extend(source_entries)
            except Exception as e:
                self.logger.error(f"Error aggregating from {log_source.name}: {e}",
                                source_name=log_source.name)
                self.stats['aggregation_errors'] += 1

        # Process and index new entries
        for entry in new_entries:
            await self._process_log_entry(entry)

        # Cleanup old entries
        await self._cleanup_old_entries()

        # Update statistics
        self.stats['last_aggregation'] = aggregation_start
        self.stats['total_entries'] = len(self.log_entries)

        aggregation_stats = {
            'aggregation_time': (datetime.now() - aggregation_start).total_seconds(),
            'new_entries_count': len(new_entries),
            'total_entries': self.stats['total_entries'],
            'sources_processed': len(sorted_sources),
            'errors': self.stats['aggregation_errors']
        }

        self.logger.info("Log aggregation completed",
                        session_id=self.session_id,
                        **aggregation_stats)

        return aggregation_stats

    async def _aggregate_from_source(self, log_source: LogSource) -> List[LogEntry]:
        """Aggregate logs from a specific source."""
        entries = []

        if not log_source.path.exists():
            return entries

        # Find matching files
        pattern_files = list(log_source.path.glob(log_source.pattern))

        for file_path in pattern_files:
            try:
                file_entries = await self._process_log_file(file_path, log_source)
                entries.extend(file_entries)
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}",
                                source_name=log_source.name,
                                file_path=str(file_path))

        return entries

    async def _process_log_file(self, file_path: Path, log_source: LogSource) -> List[LogEntry]:
        """Process a single log file and convert to LogEntry objects."""
        entries = []

        try:
            if log_source.format == 'json':
                entries = await self._process_json_file(file_path, log_source)
            elif log_source.format == 'structured':
                entries = await self._process_structured_file(file_path, log_source)
            elif log_source.format == 'plain':
                entries = await self._process_plain_file(file_path, log_source)

        except Exception as e:
            self.logger.error(f"Error processing log file {file_path}: {e}",
                            source_name=log_source.name,
                            file_path=str(file_path))

        return entries

    async def _process_json_file(self, file_path: Path, log_source: LogSource) -> List[LogEntry]:
        """Process JSON log file."""
        entries = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    entry = self._convert_to_log_entry(data, log_source, file_path, line_num)
                    if entry:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON lines
                except Exception as e:
                    self.logger.debug(f"Error processing line {line_num}: {e}")

        return entries

    async def _process_structured_file(self, file_path: Path, log_source: LogSource) -> List[LogEntry]:
        """Process structured log file (JSONL format)."""
        entries = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    entry = self._convert_to_log_entry(data, log_source, file_path, line_num)
                    if entry:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.logger.debug(f"Error processing line {line_num}: {e}")

        return entries

    async def _process_plain_file(self, file_path: Path, log_source: LogSource) -> List[LogEntry]:
        """Process plain text log file."""
        entries = []

        # Common log format patterns
        patterns = [
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),?\d{3}\s*-\s*(\w+)\s*-\s*(\w+)\s*-\s*(.*)',
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\.\d]*Z?)\s+(\w+)\s+(.*)',
            r'(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\w+)\s+(.*)'
        ]

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                for pattern in patterns:
                    match = re.match(pattern, line)
                    if match:
                        try:
                            timestamp_str, level, message = match.groups()[:3]

                            # Parse timestamp
                            timestamp = self._parse_timestamp(timestamp_str)

                            entry = LogEntry(
                                timestamp=timestamp,
                                level=level.upper(),
                                source=log_source.name,
                                session_id=self.session_id,
                                agent_name=None,
                                activity_type=None,
                                message=message,
                                metadata={
                                    'file_path': str(file_path),
                                    'line_number': line_num
                                }
                            )
                            entries.append(entry)
                        except Exception as e:
                            self.logger.debug(f"Error parsing line {line_num}: {e}")
                        break

        return entries

    def _convert_to_log_entry(self, data: Dict[str, Any], log_source: LogSource, file_path: Path, line_num: int) -> Optional[LogEntry]:
        """Convert log data to LogEntry format."""
        try:
            # Extract timestamp
            timestamp = self._extract_timestamp(data)

            # Extract level
            level = data.get('level', 'INFO').upper()

            # Extract message
            message = data.get('message', data.get('msg', str(data)))

            # Extract agent information
            agent_name = data.get('agent_name') or data.get('agent')
            activity_type = data.get('activity_type') or data.get('event_type')

            # Extract correlation ID
            correlation_id = data.get('correlation_id') or data.get('session_id')

            # Extract tags
            tags = data.get('tags', [])

            entry = LogEntry(
                timestamp=timestamp,
                level=level,
                source=log_source.name,
                session_id=self.session_id,
                agent_name=agent_name,
                activity_type=activity_type,
                message=message,
                metadata={
                    'file_path': str(file_path),
                    'line_number': line_num,
                    'raw_data': data
                },
                correlation_id=correlation_id,
                tags=tags
            )

            return entry

        except Exception as e:
            self.logger.debug(f"Error converting log entry: {e}")
            return None

    def _extract_timestamp(self, data: Dict[str, Any]) -> datetime:
        """Extract timestamp from log data."""
        # Try various timestamp fields
        timestamp_fields = ['timestamp', 'time', 'datetime', 'created', '@timestamp']

        for field in timestamp_fields:
            if field in data:
                timestamp_value = data[field]
                if isinstance(timestamp_value, str):
                    return self._parse_timestamp(timestamp_value)
                elif isinstance(timestamp_value, (int, float)):
                    return datetime.fromtimestamp(timestamp_value)

        # Default to current time
        return datetime.now()

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string in various formats."""
        # Common timestamp formats
        formats = [
            '%Y-%m-%d %H:%M:%S,%f',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%S.%f%z'
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        # If all formats fail, return current time
        return datetime.now()

    async def _process_log_entry(self, entry: LogEntry) -> None:
        """Process and index a log entry."""
        # Add to main storage
        self.log_entries.append(entry)

        # Update statistics
        self.stats['entries_by_source'][entry.source] += 1
        self.stats['entries_by_level'][entry.level] += 1

        # Index by various fields
        if entry.agent_name:
            self.indexed_fields['agent_name'][entry.agent_name].add(len(self.log_entries) - 1)

        if entry.activity_type:
            self.indexed_fields['activity_type'][entry.activity_type].add(len(self.log_entries) - 1)

        if entry.session_id:
            self.indexed_fields['session_id'][entry.session_id].add(len(self.log_entries) - 1)
            self.session_activities[entry.session_id].append(entry)

        # Index tags
        for tag in entry.tags:
            self.indexed_fields['tags'][tag].add(len(self.log_entries) - 1)

        # Check memory limit
        if len(self.log_entries) > self.max_entries:
            # Remove oldest entries
            removed_count = len(self.log_entries) - self.max_entries
            self.log_entries = self.log_entries[-self.max_entries:]
            self.logger.info(f"Removed {removed_count} old log entries to maintain memory limit")

    async def _cleanup_old_entries(self) -> None:
        """Remove old entries based on retention policy."""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)

        # Find entries to remove
        indices_to_remove = []
        for i, entry in enumerate(self.log_entries):
            if entry.timestamp < cutoff_time:
                indices_to_remove.append(i)
            else:
                break  # Entries are sorted by timestamp

        if indices_to_remove:
            # Remove old entries
            for i in reversed(indices_to_remove):
                del self.log_entries[i]

            self.logger.info(f"Cleaned up {len(indices_to_remove)} old log entries",
                            cutoff_time=cutoff_time.isoformat())

    def get_entries(self,
                   limit: Optional[int] = None,
                   offset: int = 0,
                   level_filter: Optional[str] = None,
                   source_filter: Optional[str] = None,
                   agent_filter: Optional[str] = None,
                   session_filter: Optional[str] = None,
                   tag_filter: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[LogEntry]:
        """
        Get log entries with filtering options.

        Args:
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            level_filter: Filter by log level
            source_filter: Filter by source
            agent_filter: Filter by agent name
            session_filter: Filter by session ID
            tag_filter: Filter by tag
            start_time: Filter entries after this time
            end_time: Filter entries before this time

        Returns:
            Filtered list of log entries
        """
        entries = self.log_entries

        # Apply filters
        if level_filter:
            entries = [e for e in entries if e.level == level_filter.upper()]

        if source_filter:
            entries = [e for e in entries if e.source == source_filter]

        if agent_filter:
            entries = [e for e in entries if e.agent_name == agent_filter]

        if session_filter:
            entries = [e for e in entries if e.session_id == session_filter]

        if tag_filter:
            entries = [e for e in entries if tag_filter in e.tags]

        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]

        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        # Apply pagination
        if offset > 0:
            entries = entries[offset:]

        if limit:
            entries = entries[:limit]

        return entries

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get current aggregation statistics."""
        return {
            'session_id': self.session_id,
            'total_entries': len(self.log_entries),
            'entries_by_source': dict(self.stats['entries_by_source']),
            'entries_by_level': dict(self.stats['entries_by_level']),
            'configured_sources': len(self.log_sources),
            'active_sources': len([s for s in self.log_sources.values() if s.enabled]),
            'last_aggregation': self.stats['last_aggregation'].isoformat() if self.stats['last_aggregation'] else None,
            'aggregation_errors': self.stats['aggregation_errors'],
            'is_aggregating': self.is_aggregating
        }

    def export_aggregated_logs(self,
                              file_path: Optional[str] = None,
                              format: str = 'json',
                              include_metadata: bool = True) -> str:
        """
        Export aggregated logs to file.

        Args:
            file_path: Optional custom file path
            format: Export format ('json', 'csv')
            include_metadata: Whether to include metadata

        Returns:
            Path to exported file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self.aggregation_dir / f"aggregated_logs_{self.session_id}_{timestamp}.{format}")

        if format.lower() == 'json':
            self._export_json(file_path, include_metadata)
        elif format.lower() == 'csv':
            self._export_csv(file_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Aggregated logs exported to: {file_path}",
                        session_id=self.session_id,
                        format=format,
                        entries_count=len(self.log_entries))

        return file_path

    def _export_json(self, file_path: str, include_metadata: bool) -> None:
        """Export logs in JSON format."""
        export_data = {
            'session_id': self.session_id,
            'export_timestamp': datetime.now().isoformat(),
            'total_entries': len(self.log_entries),
            'statistics': self.get_aggregation_stats()
        }

        if include_metadata:
            export_data['metadata'] = {
                'log_sources': {name: asdict(source) for name, source in self.log_sources.items()},
                'indexed_fields': {field: dict(values) for field, values in self.indexed_fields.items()}
            }

        export_data['entries'] = [asdict(entry) for entry in self.log_entries]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)

    def _export_csv(self, file_path: str) -> None:
        """Export logs in CSV format."""
        import csv

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'timestamp', 'level', 'source', 'session_id', 'agent_name',
                'activity_type', 'message', 'correlation_id', 'tags'
            ])

            # Write entries
            for entry in self.log_entries:
                writer.writerow([
                    entry.timestamp.isoformat(),
                    entry.level,
                    entry.source,
                    entry.session_id,
                    entry.agent_name or '',
                    entry.activity_type or '',
                    entry.message,
                    entry.correlation_id or '',
                    ';'.join(entry.tags)
                ])

    async def cleanup(self) -> None:
        """Clean up resources and stop aggregation."""
        await self.stop_aggregation()

        # Export final aggregated data
        try:
            self.export_aggregated_logs()
        except Exception as e:
            self.logger.error(f"Error exporting final aggregated logs: {e}",
                            session_id=self.session_id)

        self.logger.info("LogAggregator cleanup completed",
                        session_id=self.session_id)