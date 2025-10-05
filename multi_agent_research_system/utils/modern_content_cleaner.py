"""
Modern Web Content Cleaner

Provides sophisticated pattern matching for modern web elements and noise removal.
"""

import logging
import re
from dataclasses import dataclass
from re import Pattern


@dataclass
class CleaningResult:
    """Result of content cleaning operation."""
    cleaned_content: str
    original_length: int
    cleaned_length: int
    noise_removed: int
    patterns_matched: list[str]
    quality_score: int


class ModernWebContentCleaner:
    """Advanced content cleaner for modern web elements."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

        # Initialize modern web patterns
        self.modern_patterns = self._initialize_modern_patterns()

        # Additional rule-based patterns
        self.noise_indicators = self._initialize_noise_indicators()
        self.content_preservers = self._initialize_content_preservers()

        # Pattern matching cache for performance
        self._compiled_patterns = self._compile_patterns()

    def _initialize_modern_patterns(self) -> dict[str, list[str]]:
        """Initialize sophisticated patterns for modern web elements."""
        return {
            # Cookie consent patterns
            'cookie_banners': [
                r'cookie.*consent', r'accept.*cookies', r'privacy.*policy',
                r'gdpr.*compliance', r'cookie.*settings', r'privacy.*notice',
                r'we use cookies', r'cookie.*preferences', r'personalize.*ads',
                r'cookie.*declaration', r'your.*privacy', r'data.*protection',
                r'consent.*management', r'privacy.*settings', r'third.*party.*cookies'
            ],

            # Navigation patterns
            'navigation_elements': [
                r'menu.*navigation', r'skip.*navigation', r'main.*menu',
                r'breadcrumb.*navigation', r'site.*navigation', r'category.*menu',
                r'search.*form', r'language.*selector', r'mobile.*menu',
                r'navigation.*bar', r'nav.*menu', r'primary.*navigation',
                r'secondary.*navigation', r'top.*menu', r'footer.*navigation'
            ],

            # Social media patterns
            'social_media': [
                r'facebook.*com', r'twitter\.com|x\.com', r'instagram\.com',
                r'youtube\.com', r'tiktok\.com', r'reddit\.com',
                r'share.*this', r'follow.*us', r'social.*media',
                r'newsletter.*signup', r'email.*subscribe', r'share.*article',
                r'connect.*with.*us', r'social.*sharing', r'follow.*on',
                r'like.*us.*on', r'share.*page', r'subscribe.*channel'
            ],

            # Footer patterns
            'footer_boilerplate': [
                r'about.*us', r'contact.*us', r'careers.*at',
                r'privacy.*policy', r'terms.*.*use', r'copyright.*\d{4}',
                r'all.*rights.*reserved', r'site.*map', r'help.*center',
                r'legal.*disclaimer', r'corporate.*info', r'investor.*relations',
                r'press.*room', r'advertising.*info', r'affiliate.*program'
            ],

            # Modern news site patterns
            'news_site_specific': [
                r'support.*our.*journalism', r'donate.*now', r'subscribe.*today',
                r'breaking.*news', r'trending.*now', r'most.*popular',
                r'editor.*picks', r'recommended.*for.*you', r'latest.*headlines',
                r'news.*alert', r'live.*updates', r'real.*time.*news',
                r'investigative.*reporting', r'in.*depth.*analysis'
            ],

            # Author and byline patterns
            'author_elements': [
                r'by\s+\w+\s+\w+', r'author.*:', r'reporter.*:',
                r'journalist.*:', r'correspondent.*:', r'contributor.*:',
                r'staff.*writer', r'senior.*editor', r'news.*editor',
                r'photo.*by', r'images.*by', r'graphics.*by'
            ],

            # Interactive and multimedia patterns
            'interactive_elements': [
                r'play.*video', r'watch.*now', r'view.*gallery',
                r'click.*to.*expand', r'click.*to.*enlarge', r'interactive.*map',
                r'data.*visualization', r'live.*chat', r'comments.*section',
                r'read.*more', r'continue.*reading', r'full.*story'
            ],

            # Advertising patterns
            'advertising_patterns': [
                r'sponsored.*content', r'paid.*promotion', r'advertisement',
                r'promoted.*article', r'featured.*partner', r'affiliated.*content',
                r'ad.*vertisement', r'sponsor.*message', r'brought.*to.*you.*by'
            ],

            # Mobile-specific patterns
            'mobile_elements': [
                r'download.*app', r'mobile.*app', r'get.*the.*app',
                r'app.*store', r'google.*play', r'available.*on.*app',
                r'mobile.*version', r'desktop.*version', r'view.*full.*site'
            ]
        }

    def _initialize_noise_indicators(self) -> set[str]:
        """Initialize indicators of noisy content lines."""
        return {
            'skip to content', 'navigation menu', 'cookie policy',
            'privacy policy', 'terms of use', 'all rights reserved',
            'copyright ©', 'facebook', 'twitter', 'instagram',
            'youtube', 'tiktok', 'reddit', 'linkedin', 'pinterest',
            'subscribe to our', 'sign up for', 'follow us on',
            'share this article', 'email newsletter', 'breaking news',
            'trending now', 'most popular', 'editor\'s choice',
            'recommended for you', 'related articles', 'more from',
            'support our journalism', 'donate now', 'become a member',
            'advertisement', 'sponsored content', 'paid promotion',
            'about us', 'contact us', 'careers', 'terms & conditions',
            'privacy settings', 'cookie preferences', 'accept cookies',
            'gdpr compliance', 'data protection', 'legal disclaimer',
            'site map', 'help center', 'faq', 'frequently asked questions'
        }

    def _initialize_content_preservers(self) -> set[str]:
        """Initialize indicators of valuable content to preserve."""
        return {
            'said', 'reported', 'according to', 'stated that', 'announced',
            'confirmed', 'revealed', 'discovered', 'found that', 'showed that',
            'indicated', 'suggested', 'believed', 'estimated', 'projected',
            'forecasted', 'predicted', 'expected', 'anticipated', 'planned',
            'proposed', 'recommended', 'warned', 'cautioned', 'advised',
            'urged', 'called for', 'demanded', 'requested', 'asked',
            'questioned', 'investigated', 'analyzed', 'examined', 'studied',
            'researched', 'surveyed', 'polled', 'interviewed', 'observed',
            'witnessed', 'experienced', 'described', 'explained', 'detailed',
            'outlined', 'specified', 'clarified', 'emphasized', 'highlighted',
            'underscored', 'noted', 'mentioned', 'added', 'concluded',
            'determined', 'decided', 'ruled', 'judged', 'found'
        }

    def _compile_patterns(self) -> dict[str, list[Pattern]]:
        """Compile regex patterns for better performance."""
        compiled = {}
        for category, patterns in self.modern_patterns.items():
            try:
                compiled[category] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                    for pattern in patterns
                ]
            except re.error as e:
                self.logger.warning(f"Failed to compile patterns for {category}: {e}")
                compiled[category] = []
        return compiled

    def apply_modern_cleaning(self, content: str) -> CleaningResult:
        """
        Apply advanced cleaning for modern web content.

        Args:
            content: Raw content to clean

        Returns:
            CleaningResult with cleaned content and metadata
        """
        original_length = len(content)
        lines = content.split('\n')
        cleaned_lines = []
        patterns_matched = []

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Skip lines matching any modern web pattern
            matched_category = self.matches_modern_patterns(line)
            if matched_category:
                patterns_matched.append(f"{matched_category}:{line_num}")
                continue

            # Skip lines that are mostly URLs or navigation
            if self.is_mostly_navigation(line):
                patterns_matched.append(f"navigation_heavy:{line_num}")
                continue

            # Skip very short lines that are likely boilerplate
            if len(line) < 20 and self.is_boilerplate(line):
                patterns_matched.append(f"short_boilerplate:{line_num}")
                continue

            # Skip lines with high noise indicator density
            if self.has_high_noise_density(line):
                patterns_matched.append(f"high_noise:{line_num}")
                continue

            cleaned_lines.append(line)

        cleaned_content = '\n'.join(cleaned_lines)

        # Apply final cleaning to remove any remaining artifacts
        cleaned_content = self.final_polish(cleaned_content)

        cleaned_length = len(cleaned_content)
        noise_removed = original_length - cleaned_length

        # Calculate quality score based on content preservation
        quality_score = self.calculate_quality_score(
            original_length, cleaned_length, patterns_matched
        )

        return CleaningResult(
            cleaned_content=cleaned_content,
            original_length=original_length,
            cleaned_length=cleaned_length,
            noise_removed=noise_removed,
            patterns_matched=patterns_matched,
            quality_score=quality_score
        )

    def matches_modern_patterns(self, text: str) -> str:
        """
        Check if text matches any modern web patterns.

        Returns:
            Category name if matched, empty string otherwise
        """
        text_lower = text.lower()

        for category, compiled_patterns in self._compiled_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(text_lower):
                    return category

        return ""

    def is_mostly_navigation(self, text: str) -> bool:
        """
        Check if line is primarily navigation links.

        Args:
            text: Line to analyze

        Returns:
            True if line appears to be navigation-heavy
        """
        # Count URLs vs regular text
        url_count = len(re.findall(r'https?://[^\s]+', text, re.IGNORECASE))
        words = len(text.split())

        # If more than 30% of "words" are URLs, it's navigation
        if url_count > 0 and (url_count / max(words, 1)) > 0.3:
            return True

        # Check for navigation-heavy patterns
        nav_indicators = ['home', 'news', 'world', 'politics', 'business',
                         'tech', 'science', 'health', 'sports', 'entertainment']
        nav_count = sum(1 for indicator in nav_indicators
                       if indicator.lower() in text.lower())

        # If line has many navigation indicators, it's likely navigation
        return nav_count >= 3

    def is_boilerplate(self, text: str) -> bool:
        """
        Check if line is typical boilerplate content.

        Args:
            text: Line to analyze

        Returns:
            True if line appears to be boilerplate
        """
        text_lower = text.lower()

        # Check for common boilerplate phrases
        boilerplate_indicators = [
            'all rights reserved', 'copyright', 'privacy policy',
            'terms of use', 'cookie policy', 'about us', 'contact us',
            'follow us', 'share this', 'newsletter', 'subscribe',
            'advertisement', 'sponsored', 'breaking news'
        ]

        return any(indicator in text_lower for indicator in boilerplate_indicators)

    def has_high_noise_density(self, text: str) -> bool:
        """
        Check if text has high density of noise indicators.

        Args:
            text: Line to analyze

        Returns:
            True if line has high noise density
        """
        words = text.lower().split()
        if not words:
            return True

        noise_count = sum(1 for word in words
                         if any(noise in word for noise in self.noise_indicators))

        # If more than 40% of words are noise indicators, skip the line
        return (noise_count / len(words)) > 0.4

    def final_polish(self, content: str) -> str:
        """
        Apply final polishing to cleaned content.

        Args:
            content: Content to polish

        Returns:
            Polished content
        """
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # Fix common formatting issues
        content = '\n'.join(lines)

        # Remove any remaining HTML tags or attributes
        content = re.sub(r'<[^>]*>', '', content)

        # Fix spacing around punctuation
        content = re.sub(r'\s+([.,;:!])', r'\1', content)

        # Remove duplicate spaces
        content = re.sub(r' +', ' ', content)

        return content.strip()

    def calculate_quality_score(
        self,
        original_length: int,
        cleaned_length: int,
        patterns_matched: list[str]
    ) -> int:
        """
        Calculate quality score for the cleaning operation.

        Args:
            original_length: Original content length
            cleaned_length: Cleaned content length
            patterns_matched: List of patterns that were matched

        Returns:
            Quality score (0-100)
        """
        # Base score starts at 100
        score = 100

        # Deduct points for excessive removal (over 90% removed is bad)
        removal_ratio = 1 - (cleaned_length / max(original_length, 1))
        if removal_ratio > 0.9:
            score -= 30
        elif removal_ratio > 0.8:
            score -= 15
        elif removal_ratio > 0.7:
            score -= 5

        # Add points for effective noise removal
        if len(patterns_matched) > 0:
            score += min(10, len(patterns_matched))

        # Ensure score is within bounds
        return max(0, min(100, score))

    def clean_article_content(self, content: str, search_query: str = "") -> str:
        """
        Clean article content with search query context.

        Args:
            content: Content to clean
            search_query: Search query for context-aware cleaning

        Returns:
            Cleaned article content
        """
        result = self.apply_modern_cleaning(content)

        # Additional context-aware filtering if search query provided
        if search_query:
            result = self._apply_context_filtering(result, search_query)

        return result.cleaned_content

    def _apply_context_filtering(self, result: CleaningResult, search_query: str) -> CleaningResult:
        """
        Apply context-aware filtering based on search query.

        Args:
            result: Current cleaning result
            search_query: Search query for context

        Returns:
            Updated cleaning result
        """
        if not search_query:
            return result

        query_terms = set(search_query.lower().split())
        lines = result.cleaned_content.split('\n')
        context_filtered_lines = []

        for line in lines:
            line_lower = line.lower()
            line_words = set(line_lower.split())

            # Keep line if it contains query terms or content preservers
            if (query_terms.intersection(line_words) or
                self.content_preservers.intersection(line_words)) or len(line) > 100:
                context_filtered_lines.append(line)

        filtered_content = '\n'.join(context_filtered_lines)

        return CleaningResult(
            cleaned_content=filtered_content,
            original_length=result.original_length,
            cleaned_length=len(filtered_content),
            noise_removed=result.original_length - len(filtered_content),
            patterns_matched=result.patterns_matched + ['context_filter'],
            quality_score=result.quality_score
        )


def clean_modern_web_content(content: str, search_query: str = "", logger=None) -> str:
    """
    Convenience function for modern web content cleaning.

    Args:
        content: Content to clean
        search_query: Optional search query for context
        logger: Optional logger

    Returns:
        Cleaned content
    """
    cleaner = ModernWebContentCleaner(logger)
    return cleaner.clean_article_content(content, search_query)
