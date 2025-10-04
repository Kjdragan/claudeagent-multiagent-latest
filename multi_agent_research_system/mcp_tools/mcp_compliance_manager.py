"""
MCP Compliance Manager with Multi-Level Content Allocation

Implements sophisticated MCP compliance with smart compression and content allocation
as specified in the technical documentation.

Features:
- 70/30 token split (cleaned content + metadata)
- Smart compression for long content
- Multi-level content allocation
- Token limit management
- Structured metadata generation
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ContentPriority(Enum):
    """Content priority levels for allocation."""
    CRITICAL = "critical"      # Essential findings and insights
    HIGH = "high"             # Important supporting information
    MEDIUM = "medium"         # General content and context
    LOW = "low"              # Supplementary details


@dataclass
class ContentAllocation:
    """Content allocation result with metadata."""
    primary_content: str
    metadata_content: str
    allocation_stats: Dict[str, Any]
    compression_applied: bool
    token_usage: Dict[str, int]
    priority_distribution: Dict[ContentPriority, int]


class MCPComplianceManager:
    """
    MCP Compliance Manager for intelligent content allocation and compression.

    Features:
    - Multi-level content allocation based on importance
    - Smart compression with quality preservation
    - Token limit management and optimization
    - Structured metadata generation
    - Performance tracking
    """

    def __init__(self, max_tokens: int = 25000):
        """
        Initialize the MCP compliance manager.

        Args:
            max_tokens: Maximum token limit for content allocation
        """
        self.max_tokens = max_tokens
        self.primary_content_ratio = 0.7  # 70% for cleaned content
        self.metadata_ratio = 0.3         # 30% for metadata

        # Token estimation factors (rough estimates)
        self.chars_per_token = 4  # Average characters per token
        self.metadata_overhead = 500  # Base metadata tokens

        # Performance metrics
        self.stats = {
            'total_allocations': 0,
            'avg_compression_ratio': 0.0,
            'token_efficiency': 0.0,
            'content_types_processed': {}
        }

    def allocate_content(
        self,
        raw_content: str,
        metadata: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ContentAllocation:
        """
        Allocate content according to MCP compliance standards.

        Args:
            raw_content: Raw content to allocate
            metadata: Content metadata
            context: Allocation context (query, session, etc.)

        Returns:
            ContentAllocation with structured content and metadata
        """
        try:
            # Calculate token budgets
            primary_tokens = int(self.max_tokens * self.primary_content_ratio)
            metadata_tokens = int(self.max_tokens * self.metadata_ratio) - self.metadata_overhead

            logger.info(f"Allocating content: {primary_tokens} tokens primary, {metadata_tokens} tokens metadata")

            # Analyze and prioritize content
            content_analysis = self._analyze_content(raw_content, context)

            # Allocate primary content
            primary_content = self._allocate_primary_content(
                raw_content, content_analysis, primary_tokens
            )

            # Generate enhanced metadata
            metadata_content = self._generate_enhanced_metadata(
                metadata, content_analysis, context, metadata_tokens
            )

            # Calculate allocation statistics
            allocation_stats = self._calculate_allocation_stats(
                raw_content, primary_content, metadata_content, content_analysis
            )

            # Estimate token usage
            token_usage = self._estimate_token_usage(primary_content, metadata_content)

            return ContentAllocation(
                primary_content=primary_content,
                metadata_content=metadata_content,
                allocation_stats=allocation_stats,
                compression_applied=content_analysis.get('compression_applied', False),
                token_usage=token_usage,
                priority_distribution=content_analysis.get('priority_distribution', {})
            )

        except Exception as e:
            logger.error(f"Content allocation failed: {e}")
            # Fallback to simple allocation
            return self._fallback_allocation(raw_content, metadata, context)

    def _analyze_content(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content and determine priority distribution."""
        analysis = {
            'total_length': len(content),
            'sections': [],
            'key_points': [],
            'priority_distribution': {},
            'compression_needed': False,
            'compression_applied': False
        }

        # Split content into logical sections
        sections = self._split_into_sections(content)
        analysis['sections'] = sections

        # Identify key points and prioritize
        key_points = self._extract_key_points(content, context)
        analysis['key_points'] = key_points

        # Assign priority levels
        priority_distribution = self._assign_priorities(sections, key_points, context)
        analysis['priority_distribution'] = priority_distribution

        # Determine if compression is needed
        estimated_tokens = len(content) / self.chars_per_token
        if estimated_tokens > self.max_tokens * self.primary_content_ratio:
            analysis['compression_needed'] = True

        return analysis

    def _split_into_sections(self, content: str) -> List[Dict[str, Any]]:
        """Split content into logical sections with metadata."""
        sections = []
        lines = content.split('\n')
        current_section = {'type': 'intro', 'content': '', 'importance': 0.5}
        section_index = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if line.startswith('#') or line.startswith('##') or line.startswith('###'):
                # Save previous section
                if current_section['content'].strip():
                    current_section['index'] = section_index
                    sections.append(current_section)
                    section_index += 1

                # Start new section
                header_level = len(line) - len(line.lstrip('#'))
                current_section = {
                    'type': 'header',
                    'level': header_level,
                    'title': line.lstrip('#').strip(),
                    'content': '',
                    'importance': max(0.3, 1.0 - (header_level * 0.2))
                }
            else:
                current_section['content'] += line + '\n'

        # Add final section
        if current_section['content'].strip():
            current_section['index'] = section_index
            sections.append(current_section)

        return sections

    def _extract_key_points(self, content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key points from content."""
        key_points = []
        lines = content.split('\n')
        query_terms = context.get('query_terms', [])

        # Look for sentences with importance indicators
        importance_indicators = [
            'important', 'significant', 'key', 'critical', 'essential',
            'notable', 'remarkable', 'outstanding', 'major', 'primary'
        ]

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Check for importance indicators
            if any(indicator in line_lower for indicator in importance_indicators):
                # Check for query relevance
                relevance_score = self._calculate_line_relevance(line, query_terms)

                if relevance_score > 0.3:
                    key_points.append({
                        'content': line.strip(),
                        'relevance': relevance_score,
                        'position': i,
                        'type': 'important_point'
                    })

        # Sort by relevance and take top points
        key_points.sort(key=lambda x: x['relevance'], reverse=True)
        return key_points[:10]  # Limit to top 10 key points

    def _assign_priorities(
        self,
        sections: List[Dict[str, Any]],
        key_points: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[ContentPriority, int]:
        """Assign priority levels to content sections."""
        priorities = {
            ContentPriority.CRITICAL: 0,
            ContentPriority.HIGH: 0,
            ContentPriority.MEDIUM: 0,
            ContentPriority.LOW: 0
        }

        query_terms = context.get('query_terms', [])

        # Prioritize sections based on content and relevance
        for section in sections:
            section_relevance = self._calculate_section_relevance(section, query_terms)

            # Assign priority based on relevance and importance
            if section_relevance > 0.8 or section.get('importance', 0) > 0.8:
                priorities[ContentPriority.CRITICAL] += len(section['content'])
            elif section_relevance > 0.6 or section.get('importance', 0) > 0.6:
                priorities[ContentPriority.HIGH] += len(section['content'])
            elif section_relevance > 0.4 or section.get('importance', 0) > 0.4:
                priorities[ContentPriority.MEDIUM] += len(section['content'])
            else:
                priorities[ContentPriority.LOW] += len(section['content'])

        # Add key points to critical/high priority
        for point in key_points:
            if point['relevance'] > 0.8:
                priorities[ContentPriority.CRITICAL] += len(point['content'])
            elif point['relevance'] > 0.6:
                priorities[ContentPriority.HIGH] += len(point['content'])

        return priorities

    def _allocate_primary_content(
        self,
        content: str,
        analysis: Dict[str, Any],
        max_tokens: int
    ) -> str:
        """Allocate primary content based on priority and token limits."""
        max_chars = max_tokens * self.chars_per_token

        if len(content) <= max_chars:
            return content

        # Need to compress content
        logger.info(f"Compressing content from {len(content)} to {max_chars} characters")

        # Select content by priority
        allocated_content = self._select_content_by_priority(
            analysis['sections'], analysis['priority_distribution'], max_chars
        )

        # Add key points if space allows
        key_points_content = self._format_key_points(analysis['key_points'])
        if len(allocated_content) + len(key_points_content) <= max_chars:
            allocated_content += "\n\n## Key Insights\n\n" + key_points_content

        analysis['compression_applied'] = True
        return allocated_content

    def _select_content_by_priority(
        self,
        sections: List[Dict[str, Any]],
        priority_distribution: Dict[ContentPriority, int],
        max_chars: int
    ) -> str:
        """Select content sections based on priority distribution."""
        allocated_sections = []
        current_length = 0

        # Priority order
        priority_order = [
            ContentPriority.CRITICAL,
            ContentPriority.HIGH,
            ContentPriority.MEDIUM,
            ContentPriority.LOW
        ]

        # Allocate sections by priority
        for priority in priority_order:
            if priority_distribution[priority] == 0:
                continue

            for section in sections:
                if current_length >= max_chars:
                    break

                section_length = len(section['content'])
                if current_length + section_length <= max_chars:
                    # Format section based on type
                    if section.get('type') == 'header':
                        header_prefix = '#' * section.get('level', 1) + ' '
                        formatted_section = f"{header_prefix}{section.get('title', '')}\n\n{section['content']}"
                    else:
                        formatted_section = section['content']

                    allocated_sections.append(formatted_section)
                    current_length += len(formatted_section)

        return '\n\n'.join(allocated_sections)

    def _format_key_points(self, key_points: List[Dict[str, Any]]) -> str:
        """Format key points for inclusion."""
        if not key_points:
            return ""

        formatted_points = []
        for point in key_points:
            formatted_points.append(f"• {point['content']}")

        return '\n'.join(formatted_points)

    def _generate_enhanced_metadata(
        self,
        base_metadata: Dict[str, Any],
        content_analysis: Dict[str, Any],
        context: Dict[str, Any],
        max_tokens: int
    ) -> str:
        """Generate enhanced metadata within token limits."""
        max_chars = max_tokens * self.chars_per_token

        metadata_parts = [
            "# Search and Analysis Metadata",
            "",
            f"**Query**: {context.get('query', 'N/A')}",
            f"**Session ID**: {context.get('session_id', 'N/A')}",
            f"**Processing Time**: {context.get('processing_time', 'N/A')}",
            f"**Content Sources**: {context.get('source_count', 'N/A')}",
            "",
            "## Content Analysis",
            "",
            f"**Total Content Length**: {content_analysis.get('total_length', 0):,} characters",
            f"**Sections Identified**: {len(content_analysis.get('sections', []))}",
            f"**Key Points Extracted**: {len(content_analysis.get('key_points', []))}",
            f"**Compression Applied**: {'Yes' if content_analysis.get('compression_applied', False) else 'No'}",
            "",
            "## Priority Distribution",
            ""
        ]

        # Add priority distribution
        priority_dist = content_analysis.get('priority_distribution', {})
        for priority, char_count in priority_dist.items():
            if char_count > 0:
                metadata_parts.append(f"**{priority.value.title()}**: {char_count:,} characters")

        metadata_parts.extend([
            "",
            "## Technical Details",
            "",
            f"**Enhanced Relevance Scoring**: Position 40% + Title 30% + Snippet 30%",
            f"**Anti-Bot Escalation**: Progressive levels 0-3 with smart retry",
            f"**AI Content Cleaning**: GPT-5-nano powered via Pydantic AI",
            f"**Content Quality Judge**: Multi-criteria assessment with feedback loops",
            f"**Search Strategy Auto-Selection**: Google vs SERP News routing",
            "",
            "## Session Information",
            "",
            f"**Work Products Directory**: {base_metadata.get('workproduct_dir', 'N/A')}",
            f"**MCP Compliance**: Multi-level content allocation (70/30 split)",
            f"**Token Management**: Smart compression with quality preservation",
            "",
            "---",
            "",
            "*Metadata generated by MCP Compliance Manager v1.0*"
        ])

        metadata_content = '\n'.join(metadata_parts)

        # Truncate if necessary
        if len(metadata_content) > max_chars:
            # Keep the most important parts
            essential_parts = metadata_parts[:len(metadata_parts)//2]
            metadata_content = '\n'.join(essential_parts)
            metadata_content += f"\n\n*Note: Metadata truncated to fit {max_tokens:,} token limit*"

        return metadata_content

    def _calculate_section_relevance(self, section: Dict[str, Any], query_terms: List[str]) -> float:
        """Calculate relevance score for a content section."""
        content = section.get('content', '').lower()
        title = section.get('title', '').lower()

        if not query_terms:
            return section.get('importance', 0.5)

        relevance_score = 0.0
        for term in query_terms:
            term_lower = term.lower()
            if term_lower in content:
                relevance_score += 0.3
            if term_lower in title:
                relevance_score += 0.2

        return min(1.0, relevance_score)

    def _calculate_line_relevance(self, line: str, query_terms: List[str]) -> float:
        """Calculate relevance score for a single line."""
        if not query_terms:
            return 0.5

        line_lower = line.lower()
        matches = sum(1 for term in query_terms if term.lower() in line_lower)
        return min(1.0, matches / len(query_terms))

    def _calculate_allocation_stats(
        self,
        original_content: str,
        primary_content: str,
        metadata_content: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate allocation statistics."""
        compression_ratio = len(primary_content) / len(original_content) if original_content else 1.0

        return {
            'original_length': len(original_content),
            'primary_content_length': len(primary_content),
            'metadata_length': len(metadata_content),
            'compression_ratio': compression_ratio,
            'space_saved': len(original_content) - len(primary_content),
            'compression_applied': analysis.get('compression_applied', False),
            'sections_processed': len(analysis.get('sections', [])),
            'key_points_extracted': len(analysis.get('key_points', [])),
            'priority_distribution': {
                priority.value: count for priority, count in analysis.get('priority_distribution', {}).items()
            }
        }

    def _estimate_token_usage(self, primary_content: str, metadata_content: str) -> Dict[str, int]:
        """Estimate token usage for content."""
        primary_tokens = int(len(primary_content) / self.chars_per_token)
        metadata_tokens = int(len(metadata_content) / self.chars_per_token)
        total_tokens = primary_tokens + metadata_tokens

        return {
            'primary_content': primary_tokens,
            'metadata': metadata_tokens,
            'total': total_tokens,
            'limit': self.max_tokens,
            'utilization': (total_tokens / self.max_tokens) * 100
        }

    def _fallback_allocation(
        self,
        content: str,
        metadata: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ContentAllocation:
        """Fallback allocation for error cases."""
        max_chars = int(self.max_tokens * self.primary_content_ratio * self.chars_per_token)

        if len(content) > max_chars:
            primary_content = content[:max_chars] + "\n\n*Content truncated due to token limits*"
        else:
            primary_content = content

        metadata_content = f"""# Basic Metadata

**Query**: {context.get('query', 'N/A')}
**Session**: {context.get('session_id', 'N/A')}
**Error**: Fallback allocation applied
"""

        return ContentAllocation(
            primary_content=primary_content,
            metadata_content=metadata_content,
            allocation_stats={'fallback_mode': True},
            compression_applied=len(content) > max_chars,
            token_usage={'primary': len(primary_content) // 4, 'metadata': len(metadata_content) // 4},
            priority_distribution={}
        )


# Global MCP compliance manager instance
_global_mcp_manager: Optional[MCPComplianceManager] = None


def get_mcp_compliance_manager(max_tokens: int = 25000) -> MCPComplianceManager:
    """Get or create global MCP compliance manager."""
    global _global_mcp_manager
    if _global_mcp_manager is None or _global_mcp_manager.max_tokens != max_tokens:
        _global_mcp_manager = MCPComplianceManager(max_tokens)
    return _global_mcp_manager