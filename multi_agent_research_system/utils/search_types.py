"""
Common search result data types used across multiple modules.
"""

class SearchResult:
    """Search result data structure with enhanced relevance scoring"""
    def __init__(self, title: str, link: str, snippet: str, position: int = 0,
                 date: str = None, source: str = None, relevance_score: float = 0.0):
        self.title = title
        self.link = link
        self.snippet = snippet
        self.position = position
        self.date = date
        self.source = source
        self.relevance_score = relevance_score