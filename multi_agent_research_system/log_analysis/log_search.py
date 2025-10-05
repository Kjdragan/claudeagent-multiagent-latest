"""
Log Search Engine for advanced log searching and filtering.

This module provides powerful search capabilities for aggregated logs,
including full-text search, field-based filtering, and pattern matching.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .log_aggregator import LogEntry


class SearchOperator(Enum):
    """Search operators for advanced queries."""
    AND = "and"
    OR = "or"
    NOT = "not"
    NEAR = "near"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    EQUALS = "equals"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"


@dataclass
class SearchQuery:
    """Search query structure."""
    field: str | None  # Field to search in, None for full-text
    operator: SearchOperator
    value: str | int | float | list
    case_sensitive: bool = False
    boost: float = 1.0


@dataclass
class SearchResult:
    """Single search result."""
    entry: LogEntry
    score: float
    highlights: dict[str, list[str]]


@dataclass
class SearchStats:
    """Search execution statistics."""
    total_entries: int
    matched_entries: int
    execution_time_ms: float
    query_complexity: int
    cache_hit: bool = False


class LogSearchEngine:
    """Advanced log search engine with full-text and field-based search."""

    def __init__(self):
        """Initialize the search engine."""
        self.index_cache: dict[str, dict[Any, set]] = {}
        self.search_cache: dict[str, tuple[list[SearchResult], SearchStats]] = {}
        self.cache_ttl_minutes = 10

    def build_index(self, entries: list[LogEntry]) -> None:
        """
        Build search index from log entries.

        Args:
            entries: List of log entries to index
        """
        self.index_cache = {
            'level': {},
            'source': {},
            'agent_name': {},
            'activity_type': {},
            'session_id': {},
            'correlation_id': {},
            'tags': {},
            'message_tokens': {},
            'metadata': {}
        }

        for i, entry in enumerate(entries):
            # Index exact matches
            self._index_field('level', entry.level, i)
            self._index_field('source', entry.source, i)
            if entry.agent_name:
                self._index_field('agent_name', entry.agent_name, i)
            if entry.activity_type:
                self._index_field('activity_type', entry.activity_type, i)
            self._index_field('session_id', entry.session_id, i)
            if entry.correlation_id:
                self._index_field('correlation_id', entry.correlation_id, i)

            # Index tags
            for tag in entry.tags:
                self._index_field('tags', tag, i)

            # Index message tokens for full-text search
            tokens = self._tokenize_message(entry.message)
            for token in tokens:
                self._index_field('message_tokens', token, i)

            # Index metadata fields
            for key, value in entry.metadata.items():
                if isinstance(value, str):
                    self._index_field(f'metadata.{key}', value, i)

    def _index_field(self, field: str, value: Any, entry_index: int) -> None:
        """Index a field value for fast lookup."""
        if field not in self.index_cache:
            self.index_cache[field] = {}
        if value not in self.index_cache[field]:
            self.index_cache[field][value] = set()
        self.index_cache[field][value].add(entry_index)

    def _tokenize_message(self, message: str) -> list[str]:
        """Tokenize message for full-text search."""
        # Simple tokenization - split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', message.lower())
        return tokens

    def search(self,
               entries: list[LogEntry],
               query: str | SearchQuery | list[SearchQuery],
               limit: int | None = None,
               offset: int = 0,
               sort_by: str = 'score',
               include_highlights: bool = True) -> tuple[list[SearchResult], SearchStats]:
        """
        Search log entries with advanced query capabilities.

        Args:
            entries: List of log entries to search
            query: Search query (string or structured query)
            limit: Maximum number of results
            offset: Results offset
            sort_by: Sort results by ('score', 'timestamp')
            include_highlights: Whether to include search highlights

        Returns:
            Tuple of (search results, search statistics)
        """
        start_time = datetime.now()

        # Check cache
        cache_key = self._generate_cache_key(entries, query, limit, offset, sort_by)
        if cache_key in self.search_cache:
            cached_results, cached_stats = self.search_cache[cache_key]
            cached_stats.cache_hit = True
            return cached_results, cached_stats

        # Convert string query to structured query
        if isinstance(query, str):
            search_queries = self._parse_string_query(query)
        elif isinstance(query, SearchQuery):
            search_queries = [query]
        else:
            search_queries = query

        # Execute search
        matched_results = []
        for i, entry in enumerate(entries):
            score, highlights = self._match_entry(entry, search_queries, include_highlights)
            if score > 0:
                result = SearchResult(
                    entry=entry,
                    score=score,
                    highlights=highlights
                )
                matched_results.append(result)

        # Sort results
        if sort_by == 'score':
            matched_results.sort(key=lambda x: x.score, reverse=True)
        elif sort_by == 'timestamp':
            matched_results.sort(key=lambda x: x.entry.timestamp, reverse=True)

        # Apply pagination
        if offset > 0:
            matched_results = matched_results[offset:]
        if limit:
            matched_results = matched_results[:limit]

        # Calculate statistics
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        stats = SearchStats(
            total_entries=len(entries),
            matched_entries=len(matched_results),
            execution_time_ms=execution_time,
            query_complexity=len(search_queries),
            cache_hit=False
        )

        # Cache results
        self.search_cache[cache_key] = (matched_results, stats)

        return matched_results, stats

    def _parse_string_query(self, query: str) -> list[SearchQuery]:
        """Parse string query into structured search queries."""
        queries = []

        # Simple parsing for now - can be enhanced with proper query language
        # Format examples:
        # "error" - searches for 'error' in message
        # "level:ERROR" - searches for ERROR level
        # "agent:research_agent" - searches for research_agent
        # "ERROR AND agent:research_agent" - combines conditions

        # Parse field:value patterns
        field_pattern = r'(\w+):([^\s]+)'
        field_matches = re.findall(field_pattern, query)

        remaining_query = query
        for field, value in field_matches:
            operator = SearchOperator.CONTAINS
            if field in ['level', 'source', 'agent_name', 'activity_type', 'session_id']:
                operator = SearchOperator.CONTAINS

            search_query = SearchQuery(
                field=field,
                operator=operator,
                value=value
            )
            queries.append(search_query)
            remaining_query = remaining_query.replace(f"{field}:{value}", "").strip()

        # Parse remaining text as full-text search
        if remaining_query:
            # Handle AND/OR operators
            if ' AND ' in remaining_query:
                parts = remaining_query.split(' AND ')
                for part in parts:
                    if part.strip():
                        queries.append(SearchQuery(
                            field=None,
                            operator=SearchOperator.CONTAINS,
                            value=part.strip()
                        ))
            elif ' OR ' in remaining_query:
                # OR logic - create multiple queries (simplified)
                parts = remaining_query.split(' OR ')
                value_list = [part.strip() for part in parts if part.strip()]
                if value_list:
                    queries.append(SearchQuery(
                        field=None,
                        operator=SearchOperator.IN,
                        value=value_list
                    ))
            else:
                # Simple text search
                queries.append(SearchQuery(
                    field=None,
                    operator=SearchOperator.CONTAINS,
                    value=remaining_query.strip()
                ))

        return queries

    def _match_entry(self, entry: LogEntry, queries: list[SearchQuery], include_highlights: bool) -> tuple[float, dict[str, list[str]]]:
        """Check if an entry matches the search queries."""
        total_score = 0.0
        highlights = {}

        for query in queries:
            score, entry_highlights = self._match_query(entry, query, include_highlights)
            total_score += score * query.boost

            # Merge highlights
            for field, hls in entry_highlights.items():
                if field not in highlights:
                    highlights[field] = []
                highlights[field].extend(hls)

        return total_score, highlights

    def _match_query(self, entry: LogEntry, query: SearchQuery, include_highlights: bool) -> tuple[float, dict[str, list[str]]]:
        """Check if an entry matches a single search query."""
        if query.field is None:
            # Full-text search
            return self._match_full_text(entry, query, include_highlights)
        else:
            # Field-based search
            return self._match_field(entry, query, include_highlights)

    def _match_full_text(self, entry: LogEntry, query: SearchQuery, include_highlights: bool) -> tuple[float, dict[str, list[str]]]:
        """Match entry against full-text search query."""
        search_text = entry.message.lower()
        search_value = str(query.value).lower()

        if not query.case_sensitive:
            search_text = search_text.lower()
            search_value = search_value.lower()

        if query.operator == SearchOperator.CONTAINS:
            if search_value in search_text:
                score = 1.0
                highlights = {'message': [search_value]} if include_highlights else {}
                return score, highlights

        elif query.operator == SearchOperator.REGEX:
            try:
                pattern = re.compile(search_value, re.IGNORECASE if not query.case_sensitive else 0)
                if pattern.search(entry.message):
                    score = 1.0
                    highlights = {'message': [search_value]} if include_highlights else {}
                    return score, highlights
            except re.error:
                pass

        elif query.operator == SearchOperator.IN:
            if isinstance(query.value, list):
                for value in query.value:
                    if str(value).lower() in search_text:
                        score = 1.0
                        highlights = {'message': [str(value)]} if include_highlights else {}
                        return score, highlights

        return 0.0, {}

    def _match_field(self, entry: LogEntry, query: SearchQuery, include_highlights: bool) -> tuple[float, dict[str, list[str]]]:
        """Match entry against field-based search query."""
        field_value = self._get_field_value(entry, query.field)
        if field_value is None:
            return 0.0, {}

        search_value = str(query.value)
        field_str = str(field_value)

        if not query.case_sensitive:
            field_str = field_str.lower()
            search_value = search_value.lower()

        score = 0.0
        highlights = {}

        if query.operator == SearchOperator.CONTAINS:
            if search_value in field_str:
                score = 1.0
                highlights = {query.field: [search_value]} if include_highlights else {}

        elif query.operator == SearchOperator.EQUALS:
            if field_str == search_value:
                score = 1.0
                highlights = {query.field: [search_value]} if include_highlights else {}

        elif query.operator == SearchOperator.STARTS_WITH:
            if field_str.startswith(search_value):
                score = 1.0
                highlights = {query.field: [search_value]} if include_highlights else {}

        elif query.operator == SearchOperator.ENDS_WITH:
            if field_str.endswith(search_value):
                score = 1.0
                highlights = {query.field: [search_value]} if include_highlights else {}

        elif query.operator == SearchOperator.IN:
            if isinstance(query.value, list) and field_str in [str(v).lower() for v in query.value]:
                score = 1.0
                highlights = {query.field: [field_str]} if include_highlights else {}

        elif query.operator in [SearchOperator.GT, SearchOperator.GTE, SearchOperator.LT, SearchOperator.LTE]:
            try:
                field_num = float(field_str)
                value_num = float(search_value)

                if query.operator == SearchOperator.GT and field_num > value_num or query.operator == SearchOperator.GTE and field_num >= value_num or query.operator == SearchOperator.LT and field_num < value_num or query.operator == SearchOperator.LTE and field_num <= value_num:
                    score = 1.0

                if score > 0:
                    highlights = {query.field: [search_value]} if include_highlights else {}

            except ValueError:
                pass  # Can't compare non-numeric values

        elif query.operator == SearchOperator.REGEX:
            try:
                pattern = re.compile(search_value, re.IGNORECASE if not query.case_sensitive else 0)
                if pattern.search(field_str):
                    score = 1.0
                    highlights = {query.field: [search_value]} if include_highlights else {}
            except re.error:
                pass

        return score, highlights

    def _get_field_value(self, entry: LogEntry, field: str) -> Any:
        """Get the value of a field from a log entry."""
        # Direct fields
        if hasattr(entry, field):
            return getattr(entry, field)

        # Metadata fields
        if field.startswith('metadata.'):
            metadata_field = field[9:]  # Remove 'metadata.' prefix
            return entry.metadata.get(metadata_field)

        # Special cases
        if field == 'timestamp':
            return entry.timestamp
        elif field == 'tags':
            return entry.tags

        return None

    def _generate_cache_key(self, entries: list[LogEntry], query: Any, limit: int | None, offset: int, sort_by: str) -> str:
        """Generate cache key for search results."""
        # Simple cache key generation - can be improved
        query_str = str(query) if not isinstance(query, str) else query
        entries_hash = str(hash(tuple(str(e) for e in entries[-100:])))  # Hash last 100 entries
        return f"{query_str}_{limit}_{offset}_{sort_by}_{entries_hash}"

    def clear_cache(self) -> None:
        """Clear search cache."""
        self.search_cache.clear()

    def get_field_suggestions(self, field: str, entries: list[LogEntry], limit: int = 10) -> list[str]:
        """Get suggestions for field values."""
        suggestions = set()

        for entry in entries:
            value = self._get_field_value(entry, field)
            if value:
                if isinstance(value, list):
                    suggestions.update(str(v) for v in value)
                else:
                    suggestions.add(str(value))

        return list(suggestions)[:limit]

    def get_search_stats(self) -> dict[str, Any]:
        """Get search engine statistics."""
        return {
            'indexed_fields': list(self.index_cache.keys()),
            'cache_size': len(self.search_cache),
            'index_size': {field: len(values) for field, values in self.index_cache.items()},
            'cache_ttl_minutes': self.cache_ttl_minutes
        }
