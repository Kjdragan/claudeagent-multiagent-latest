"""
Enhanced Relevance Scoring with Domain Authority

Implements the sophisticated relevance scoring algorithm from the technical documentation:
- Position 60% + Title 20% + Snippet 20% base scoring (increased position weighting)
- Domain authority boost up to 25% for high-authority sites
- Comprehensive scoring logic for optimal search result selection
"""

import logging
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# High-authority domains and their authority boost percentages
HIGH_AUTHORITY_DOMAINS = {
    # Government domains
    'gov': 0.25,
    'mil': 0.25,
    # Educational domains
    'edu': 0.20,
    # Major news organizations
    'reuters.com': 0.20,
    'ap.org': 0.20,
    'bbc.com': 0.20,
    'cnn.com': 0.15,
    'nytimes.com': 0.15,
    'washingtonpost.com': 0.15,
    'wsj.com': 0.15,
    'theguardian.com': 0.15,
    # Academic and research
    'nature.com': 0.20,
    'science.org': 0.20,
    'nejm.org': 0.20,
    'thelancet.com': 0.20,
    'pubmed.ncbi.nlm.nih.gov': 0.20,
    'arxiv.org': 0.15,
    # Major tech publications
    'techcrunch.com': 0.15,
    'wired.com': 0.15,
    'venturebeat.com': 0.15,
    # Major organizations
    'who.int': 0.25,
    'un.org': 0.25,
    'worldbank.org': 0.20,
    'imf.org': 0.20,
    'oecd.org': 0.20
}

# Medium-authority domains
MEDIUM_AUTHORITY_DOMAINS = {
    'forbes.com': 0.10,
    'bloomberg.com': 0.10,
    'cnbc.com': 0.10,
    'fortune.com': 0.10,
    'time.com': 0.10,
    'theatlantic.com': 0.10,
    'newyorker.com': 0.10,
    'economist.com': 0.10,
    'hbr.org': 0.10
}


def extract_domain_info(url: str) -> tuple[str, str]:
    """
    Extract domain and TLD information from URL.

    Args:
        url: The URL to analyze

    Returns:
        Tuple of (domain, tld)
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Extract TLD (top-level domain)
        if '.' in domain:
            parts = domain.split('.')
            if len(parts) >= 2:
                tld = parts[-1]
                # Remove common subdomains for better matching
                if parts[0] in ['www', 'news', 'blog', 'tech']:
                    domain = '.'.join(parts[1:])
        else:
            tld = ''

        return domain, tld
    except Exception as e:
        logger.warning(f"Failed to parse URL {url}: {e}")
        return '', ''


def calculate_domain_authority_boost(url: str) -> float:
    """
    Calculate domain authority boost based on the URL domain.

    Args:
        url: The URL to analyze

    Returns:
        Authority boost value between 0.0 and 0.25
    """
    domain, tld = extract_domain_info(url)

    # Check for exact domain matches first
    if domain in HIGH_AUTHORITY_DOMAINS:
        boost = HIGH_AUTHORITY_DOMAINS[domain]
        logger.debug(f"High authority domain boost for {domain}: {boost}")
        return boost

    # Check for TLD matches
    if tld in HIGH_AUTHORITY_DOMAINS:
        boost = HIGH_AUTHORITY_DOMAINS[tld]
        logger.debug(f"High authority TLD boost for {tld}: {boost}")
        return boost

    # Check for medium authority domains
    if domain in MEDIUM_AUTHORITY_DOMAINS:
        boost = MEDIUM_AUTHORITY_DOMAINS[domain]
        logger.debug(f"Medium authority domain boost for {domain}: {boost}")
        return boost

    # No boost for other domains
    return 0.0


def calculate_term_frequency_score(text: str, query_terms: list[str]) -> float:
    """
    Calculate term frequency score with partial matching support.

    Args:
        text: Text to analyze
        query_terms: List of query terms

    Returns:
        Frequency score between 0.0 and 1.0
    """
    if not query_terms:
        return 0.0

    text_lower = text.lower()
    total_terms = len(query_terms)
    matched_terms = 0

    for term in query_terms:
        term_lower = term.lower()
        if term_lower in text_lower:
            matched_terms += 1
            # Extra weight for exact word boundaries
            word_pattern = r'\b' + re.escape(term_lower) + r'\b'
            if re.search(word_pattern, text_lower):
                matched_terms += 0.5  # Bonus for exact word match

    return min(1.0, matched_terms / total_terms)


def calculate_enhanced_relevance_score_with_domain_authority(
    title: str,
    snippet: str,
    position: int,
    query_terms: list[str],
    url: str = ""
) -> float:
    """
    Calculate comprehensive enhanced relevance score with domain authority.

    Formula:
    - Base scoring: Position 60% + Title 20% + Snippet 20% (increased position weighting)
    - Domain authority boost: Up to 25% additional
    - Final score: min(1.0, base_score + authority_boost)

    Args:
        title: Search result title
        snippet: Search result snippet
        position: Google search position (1-based)
        query_terms: List of query terms to match
        url: URL for domain authority calculation

    Returns:
        Enhanced relevance score between 0.0 and 1.0
    """
    try:
        # Ensure input types are correct (prevent type conversion errors)
        if not isinstance(title, str):
            title = str(title) if title is not None else ""
        if not isinstance(snippet, str):
            snippet = str(snippet) if snippet is not None else ""
        if not isinstance(query_terms, list):
            query_terms = []
        if not isinstance(url, str):
            url = str(url) if url is not None else ""

        # Ensure position is an integer
        position = int(position) if position is not None else 999

        # Calculate base relevance score using documented formula
        position_score = _calculate_position_score(position)
        title_score = calculate_term_frequency_score(title, query_terms)
        snippet_score = calculate_term_frequency_score(snippet, query_terms)

        # Base score with increased position weighting (60% position, 20% title, 20% snippet)
        base_score = (
            position_score * 0.60 +
            title_score * 0.20 +
            snippet_score * 0.20
        )

        # Apply domain authority boost
        authority_boost = 0.0
        if url:
            authority_boost = calculate_domain_authority_boost(url)

        # Final score with authority boost, capped at 1.0
        final_score = min(1.0, base_score + authority_boost)

        logger.debug(f"Relevance scoring - Position: {position_score:.3f}, "
                    f"Title: {title_score:.3f}, Snippet: {snippet_score:.3f}, "
                    f"Authority: {authority_boost:.3f}, Final: {final_score:.3f}")

        return round(final_score, 3)

    except Exception as e:
        logger.error(f"Error calculating enhanced relevance score: {e}")
        # Fallback to gentle position-only scoring
        if position <= 3:
            return 1.0
        elif position <= 6:
            return 0.95
        elif position <= 9:
            return 0.90
        else:
            return max(0.05, 0.90 - ((position - 9) * 0.0167))  # Gentle decay for positions 10+


def _calculate_position_score(position: int) -> float:
    """
    Calculate position score with gentle decay logic for multi-query collation.

    For multi-query research (e.g., 3 queries combined):
    - Positions 1-3: 1.0 (top results from each query, no decay)
    - Positions 4-6: 0.95 (second results, -0.05 decay)
    - Positions 7-9: 0.90 (third results, -0.05 decay)
    - Continue with 0.05 decay per position group

    Args:
        position: Search result position (1-based)

    Returns:
        Position score between 0.0 and 1.0
    """
    # Ensure position is an integer (fix type conversion issues)
    try:
        position = int(position)
    except (ValueError, TypeError):
        logger.warning(f"Invalid position value: {position}, using default position 999")
        position = 999  # Very low score for invalid positions

    # Gentle decay for multi-query collation
    if position <= 3:
        # Top 3 positions (top result from each query): no decay
        return 1.0
    elif position <= 6:
        # Positions 4-6 (second results): 0.95
        return 0.95
    elif position <= 9:
        # Positions 7-9 (third results): 0.90
        return 0.90
    elif position <= 12:
        # Positions 10-12 (fourth results): 0.85
        return 0.85
    elif position <= 15:
        # Positions 13-15 (fifth results): 0.80
        return 0.80
    else:
        # Positions 16+: Continue with 0.05 decay per 3 positions, minimum 0.05
        decay_groups = (position - 16) // 3
        score = 0.80 - (decay_groups * 0.05)
        return max(0.05, score)


def batch_calculate_relevance_scores(search_results: list[dict], query: str) -> list[dict]:
    """
    Calculate relevance scores for a batch of search results.

    Args:
        search_results: List of search result dictionaries
        query: Original search query

    Returns:
        Enhanced search results with relevance scores
    """
    # Extract query terms
    query_terms = query.split()

    enhanced_results = []
    for i, result in enumerate(search_results):
        # Calculate enhanced relevance score
        relevance_score = calculate_enhanced_relevance_score_with_domain_authority(
            title=result.get('title', ''),
            snippet=result.get('snippet', ''),
            position=i + 1,
            query_terms=query_terms,
            url=result.get('link', '')
        )

        # Enhance result with relevance score
        enhanced_result = result.copy()
        enhanced_result['relevance_score'] = relevance_score
        enhanced_results.append(enhanced_result)

    # Sort by relevance score (highest first)
    enhanced_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    return enhanced_results


def filter_results_by_relevance_threshold(
    results: list[dict],
    threshold: float = 0.3
) -> list[dict]:
    """
    Filter search results by minimum relevance threshold.

    Args:
        results: List of search results with relevance scores
        threshold: Minimum relevance score threshold (default 0.3)

    Returns:
        Filtered list of results
    """
    filtered_results = [
        result for result in results
        if result.get('relevance_score', 0) >= threshold and result.get('link')
    ]

    logger.info(f"Filtered {len(results)} results to {len(filtered_results)} "
               f"using threshold {threshold}")

    return filtered_results


# Backward compatibility function
def calculate_enhanced_relevance_score(
    title: str,
    snippet: str,
    position: int,
    query_terms: list[str]
) -> float:
    """
    Backward compatibility wrapper for the original function signature.

    Args:
        title: Search result title
        snippet: Search result snippet
        position: Google search position (1-based)
        query_terms: List of query terms to match

    Returns:
        Relevance score between 0.0 and 1.0
    """
    return calculate_enhanced_relevance_score_with_domain_authority(
        title=title,
        snippet=snippet,
        position=position,
        query_terms=query_terms,
        url=""  # No URL for backward compatibility
    )
