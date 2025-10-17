"""
Workproduct Reader Utility

This module provides utilities for reading and extracting data from research workproducts.
Replaces the corpus system with direct workproduct access.

Created: October 17, 2025
Purpose: Simplify data access, remove corpus complexity
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkproductReader:
    """
    Read and extract data from search workproducts.
    
    Workproducts are markdown files containing:
    - JSON metadata block with article info
    - Full scraped article content
    - Search metadata
    
    This replaces the complex corpus building system with simple direct access.
    """
    
    def __init__(self, workproduct_path: str):
        """
        Initialize reader with path to workproduct file.
        
        Args:
            workproduct_path: Path to workproduct markdown file
        """
        self.path = Path(workproduct_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Workproduct not found: {workproduct_path}")
        
        self.content = self._load_content()
        self.metadata = self._load_metadata()
        
        logger.info(f"Loaded workproduct: {self.path.name}")
        logger.info(f"  Articles: {len(self.metadata.get('articles', []))}")
        logger.info(f"  Content size: {len(self.content)} chars")
    
    @classmethod
    def from_session(cls, session_id: str) -> 'WorkproductReader':
        """
        Create reader from session ID by finding latest workproduct.
        
        Args:
            session_id: Session ID to find workproduct for
            
        Returns:
            WorkproductReader instance
            
        Raises:
            FileNotFoundError: If no workproduct found for session
        """
        workproduct_path = cls._find_session_workproduct(session_id)
        if not workproduct_path:
            raise FileNotFoundError(f"No workproduct found for session {session_id}")
        return cls(str(workproduct_path))
    
    @staticmethod
    def _find_session_workproduct(session_id: str) -> Optional[Path]:
        """
        Find most recent workproduct file for session.
        
        Priority:
        1. research/search_workproduct_*.md (most common)
        2. working/RESEARCH_*.md (alternate format)
        3. working/COMPREHENSIVE_*.md (fallback)
        """
        session_dir = Path("KEVIN/sessions") / session_id
        
        if not session_dir.exists():
            logger.warning(f"Session directory not found: {session_dir}")
            return None
        
        # Priority 1: Search workproducts
        research_dir = session_dir / "research"
        if research_dir.exists():
            workproducts = sorted(
                research_dir.glob("search_workproduct_*.md"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if workproducts:
                logger.info(f"Found search workproduct: {workproducts[0].name}")
                return workproducts[0]
        
        # Priority 2: Research files in working
        working_dir = session_dir / "working"
        if working_dir.exists():
            research_files = sorted(
                working_dir.glob("RESEARCH_*.md"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if research_files:
                logger.info(f"Found research file: {research_files[0].name}")
                return research_files[0]
            
            # Priority 3: Comprehensive reports
            comprehensive_files = sorted(
                working_dir.glob("COMPREHENSIVE_*.md"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if comprehensive_files:
                logger.info(f"Found comprehensive report: {comprehensive_files[0].name}")
                return comprehensive_files[0]
        
        logger.warning(f"No workproduct found for session {session_id}")
        return None
    
    def _load_content(self) -> str:
        """Load full workproduct content."""
        return self.path.read_text(encoding='utf-8')
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from workproduct.
        
        Tries multiple formats:
        1. JSON metadata block (enhanced format)
        2. Markdown headers (standard format)
        """
        # Try JSON metadata block first
        json_metadata = self._extract_json_metadata()
        if json_metadata:
            return json_metadata
        
        # Fall back to parsing markdown headers
        return self._extract_markdown_metadata()
    
    def _extract_json_metadata(self) -> Optional[Dict[str, Any]]:
        """Extract JSON metadata block if present."""
        # Look for ```json ... ``` block
        json_pattern = r'```json\s*\n(.*?)\n```'
        match = re.search(json_pattern, self.content, re.DOTALL)
        
        if match:
            try:
                metadata = json.loads(match.group(1))
                logger.debug("Extracted JSON metadata")
                return metadata
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON metadata: {e}")
        
        return None
    
    def _extract_markdown_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from markdown headers.
        
        Parses standard workproduct format:
        **Session ID**: value
        **Total Search Results**: value
        etc.
        """
        metadata = {
            "articles": [],
            "version": "1.0"
        }
        
        # Extract session ID
        session_match = re.search(r'\*\*Session ID\*\*:\s*(.+)', self.content)
        if session_match:
            metadata["session_id"] = session_match.group(1).strip()
        
        # Extract timestamp
        date_match = re.search(r'\*\*Export Date\*\*:\s*(.+)', self.content)
        if date_match:
            metadata["created_at"] = date_match.group(1).strip()
        
        # Extract query
        query_match = re.search(r'\*\*Search Query\*\*:\s*(.+)', self.content)
        if query_match:
            metadata["search_query"] = query_match.group(1).strip()
        
        # Extract article count
        count_match = re.search(r'\*\*Successfully Crawled\*\*:\s*(\d+)', self.content)
        if count_match:
            metadata["crawled_count"] = int(count_match.group(1))
        
        # Extract articles from content
        metadata["articles"] = self._parse_articles_from_content()
        
        return metadata
    
    def _parse_articles_from_content(self) -> List[Dict[str, Any]]:
        """
        Parse article information from workproduct content.
        
        Looks for article sections with URLs and titles.
        """
        articles = []
        
        # Pattern for article headings: ### 1. Title
        article_pattern = r'###\s+(\d+)\.\s+(.+?)(?=\n|$)'
        
        for match in re.finditer(article_pattern, self.content):
            index = int(match.group(1))
            title = match.group(2).strip()
            
            # Find URL for this article (usually in next few lines)
            start_pos = match.end()
            next_section = self.content[start_pos:start_pos+500]
            url_match = re.search(r'\*\*URL\*\*:\s*(.+?)(?=\n|$)', next_section)
            
            article_info = {
                "index": index,
                "title": title,
                "url": url_match.group(1).strip() if url_match else "",
            }
            
            # Extract source
            source_match = re.search(r'\*\*Source\*\*:\s*(.+?)(?=\n|$)', next_section)
            if source_match:
                article_info["source"] = source_match.group(1).strip()
            
            # Extract date
            date_match = re.search(r'\*\*Date\*\*:\s*(.+?)(?=\n|$)', next_section)
            if date_match:
                article_info["date"] = date_match.group(1).strip()
            
            articles.append(article_info)
        
        return articles
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get high-level summary of workproduct.
        
        Returns:
            Dictionary with article count, sources, dates, etc.
        """
        articles = self.metadata.get("articles", [])
        
        # Extract unique sources
        sources = list(set(
            a.get("source", "Unknown")
            for a in articles
            if a.get("source")
        ))
        
        # Count total words (approximate)
        word_count = len(self.content.split())
        
        # Get date range
        dates = [a.get("date") for a in articles if a.get("date")]
        date_range = (
            f"{min(dates)} to {max(dates)}"
            if dates and len(dates) > 1
            else dates[0] if dates else "Unknown"
        )
        
        return {
            "session_id": self.metadata.get("session_id", "Unknown"),
            "created_at": self.metadata.get("created_at", "Unknown"),
            "search_query": self.metadata.get("search_query", "Unknown"),
            "article_count": len(articles),
            "total_words": word_count,
            "sources": sources,
            "date_range": date_range,
            "file_path": str(self.path),
            "file_size_kb": self.path.stat().st_size / 1024
        }
    
    def get_article_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get specific article by URL.
        
        Args:
            url: URL of article to retrieve
            
        Returns:
            Dictionary with article data or None if not found
        """
        # Find article in metadata
        for article in self.metadata.get("articles", []):
            if article.get("url") == url:
                # Extract content for this article
                return self._extract_article_content(article)
        
        logger.warning(f"Article not found with URL: {url}")
        return None
    
    def get_article_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get article by position (1-indexed).
        
        Args:
            index: Article position (1-indexed)
            
        Returns:
            Dictionary with article data or None if not found
        """
        articles = self.metadata.get("articles", [])
        
        # Find article with matching index
        for article in articles:
            if article.get("index") == index:
                return self._extract_article_content(article)
        
        # Try by list position if index field not present
        if 0 < index <= len(articles):
            return self._extract_article_content(articles[index - 1])
        
        logger.warning(f"Article not found at index: {index}")
        return None
    
    def get_all_articles(self) -> List[Dict[str, Any]]:
        """
        Get all articles with full content.
        
        Returns:
            List of article dictionaries with content
        """
        articles = []
        for article in self.metadata.get("articles", []):
            article_data = self._extract_article_content(article)
            if article_data:
                articles.append(article_data)
        return articles
    
    def _extract_article_content(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract full content for an article.
        
        Args:
            article: Article metadata dictionary
            
        Returns:
            Dictionary with article metadata and full content
        """
        title = article.get("title", "")
        url = article.get("url", "")
        
        # Find article content in workproduct
        # Look for "## Full Content - Article X" or similar
        content = self._find_article_content_by_title_or_url(title, url)
        
        return {
            "index": article.get("index"),
            "title": title,
            "url": url,
            "source": article.get("source", "Unknown"),
            "date": article.get("date", "Unknown"),
            "relevance_score": article.get("relevance_score", 0.0),
            "content": content,
            "word_count": len(content.split()) if content else 0
        }
    
    def _find_article_content_by_title_or_url(self, title: str, url: str) -> str:
        """
        Find article content in workproduct by title or URL.
        
        Looks for content sections that match the article.
        """
        # Pattern for full content sections
        # Usually "## Full Content - Article X" or similar
        
        # Try to find by title
        if title:
            # Escape special regex characters in title
            safe_title = re.escape(title[:50])  # First 50 chars
            pattern = rf'##\s+Full Content.*?\n.*?{safe_title}(.*?)(?=\n##|\Z)'
            match = re.search(pattern, self.content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Try to find by URL
        if url:
            safe_url = re.escape(url)
            pattern = rf'{safe_url}.*?\n\n(.*?)(?=\n##|\n---|\Z)'
            match = re.search(pattern, self.content, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Remove metadata lines
                lines = content.split('\n')
                content_lines = [l for l in lines if not l.startswith('**')]
                return '\n'.join(content_lines).strip()
        
        # Fallback: return empty if not found
        logger.warning(f"Could not extract content for article: {title[:50]}")
        return ""
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get full metadata dictionary.
        
        Returns:
            Complete metadata from workproduct
        """
        return self.metadata.copy()
    
    def get_full_content(self) -> str:
        """
        Get complete workproduct content.
        
        Returns:
            Full markdown content as string
        """
        return self.content


def find_session_workproduct(session_id: str) -> Optional[str]:
    """
    Find workproduct file path for a session.
    
    Convenience function that returns path as string.
    
    Args:
        session_id: Session ID to find workproduct for
        
    Returns:
        Path to workproduct file or None if not found
    """
    path = WorkproductReader._find_session_workproduct(session_id)
    return str(path) if path else None
