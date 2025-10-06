"""
Query Intent Analyzer for Multi-Agent Research System

Analyzes user queries to determine appropriate report format and processing approach.
"""

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class QueryIntent:
    """Query intent classification."""
    BRIEF = "brief"
    COMPREHENSIVE = "comprehensive"
    DEFAULT = "default"


class QueryIntentAnalyzer:
    """
    Analyzes user queries to determine the appropriate report format and approach.

    This helps eliminate confusion by selecting a single, appropriate format
    based on user intent rather than generating multiple formats.
    """

    def __init__(self):
        # Keywords indicating brief format preference
        self.brief_keywords = [
            "brief", "summary", "summarize", "quick", "short", "overview",
            "highlights", "key points", "bullet points", "concise",
            "tl;dr", "tldr", "basics", "essentials", "snapshot"
        ]

        # Keywords indicating comprehensive format preference
        self.comprehensive_keywords = [
            "comprehensive", "detailed", "in-depth", "indepth", "thorough",
            "extensive", "complete", "full", "exhaustive", "deep dive",
            "extensive", "detailed analysis", "comprehensive analysis",
            "deep analysis", "thorough analysis", "extensive review"
        ]

        # Keywords that might indicate specific formats
        self.format_indicators = {
            "academic": ["academic", "scholarly", "research paper", "peer review"],
            "business": ["business", "corporate", "executive", "professional"],
            "technical": ["technical", "engineering", "specifications", "details"],
        }

    def analyze_query_intent(self, query: str) -> Dict[str, any]:
        """
        Analyze user query to determine appropriate report format.

        Args:
            query: The user's search query

        Returns:
            Dict containing:
            - format: 'brief', 'comprehensive', or 'default'
            - confidence: Float from 0.0 to 1.0
            - reasoning: Explanation of the decision
            - keywords: Keywords that influenced the decision
        """
        try:
            query_lower = query.lower()

            # Count keyword matches for each format
            brief_matches = self._count_keyword_matches(query_lower, self.brief_keywords)
            comprehensive_matches = self._count_keyword_matches(query_lower, self.comprehensive_keywords)

            # Determine primary intent
            if brief_matches > comprehensive_matches and brief_matches > 0:
                return {
                    "format": QueryIntent.BRIEF,
                    "confidence": min(0.9, 0.5 + (brief_matches * 0.1)),
                    "reasoning": f"Query contains {brief_matches} brief-format keywords",
                    "keywords": self._find_matching_keywords(query_lower, self.brief_keywords)
                }
            elif comprehensive_matches > brief_matches and comprehensive_matches > 0:
                return {
                    "format": QueryIntent.COMPREHENSIVE,
                    "confidence": min(0.9, 0.5 + (comprehensive_matches * 0.1)),
                    "reasoning": f"Query contains {comprehensive_matches} comprehensive-format keywords",
                    "keywords": self._find_matching_keywords(query_lower, self.comprehensive_keywords)
                }
            else:
                # No clear intent detected - use default format
                return {
                    "format": QueryIntent.DEFAULT,
                    "confidence": 0.5,
                    "reasoning": "No clear format preference detected in query",
                    "keywords": []
                }

        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            # Fallback to default format on error
            return {
                "format": QueryIntent.DEFAULT,
                "confidence": 0.3,
                "reasoning": f"Error during analysis: {str(e)}",
                "keywords": []
            }

    def _count_keyword_matches(self, query: str, keywords: list) -> int:
        """Count how many keywords from the list appear in the query."""
        matches = 0
        for keyword in keywords:
            if keyword in query:
                matches += 1
        return matches

    def _find_matching_keywords(self, query: str, keywords: list) -> list:
        """Find which keywords from the list appear in the query."""
        matches = []
        for keyword in keywords:
            if keyword in query:
                matches.append(keyword)
        return matches

    def suggest_format(self, query: str, topic_complexity: Optional[str] = None) -> str:
        """
        Suggest the most appropriate format for a given query.

        Args:
            query: The user's search query
            topic_complexity: Optional assessment of topic complexity

        Returns:
            Recommended format: 'brief', 'comprehensive', or 'default'
        """
        intent = self.analyze_query_intent(query)

        # Consider topic complexity as a factor
        if topic_complexity == "complex" and intent["confidence"] < 0.7:
            # For complex topics with unclear intent, default to comprehensive
            return QueryIntent.COMPREHENSIVE
        elif topic_complexity == "simple" and intent["confidence"] < 0.7:
            # For simple topics with unclear intent, default to brief
            return QueryIntent.BRIEF
        else:
            # Use the detected intent
            return intent["format"]

    def get_format_description(self, format_type: str) -> str:
        """Get human-readable description of a format type."""
        descriptions = {
            QueryIntent.BRIEF: "Brief summary with key points and highlights",
            QueryIntent.COMPREHENSIVE: "Comprehensive detailed analysis with extensive coverage",
            QueryIntent.DEFAULT: "Standard research report with balanced detail"
        }
        return descriptions.get(format_type, "Standard research report")


# Global analyzer instance
_global_analyzer: Optional[QueryIntentAnalyzer] = None


def get_query_intent_analyzer() -> QueryIntentAnalyzer:
    """Get or create the global query intent analyzer."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = QueryIntentAnalyzer()
    return _global_analyzer


def analyze_query_intent(query: str) -> Dict[str, any]:
    """
    Convenience function to analyze query intent.

    Args:
        query: The user's search query

    Returns:
        Dict containing format analysis results
    """
    analyzer = get_query_intent_analyzer()
    return analyzer.analyze_query_intent(query)


def determine_report_format(query: str, topic_complexity: Optional[str] = None) -> str:
    """
    Convenience function to determine the appropriate report format.

    Args:
        query: The user's search query
        topic_complexity: Optional topic complexity assessment

    Returns:
        Recommended format: 'brief', 'comprehensive', or 'default'
    """
    analyzer = get_query_intent_analyzer()
    return analyzer.suggest_format(query, topic_complexity)