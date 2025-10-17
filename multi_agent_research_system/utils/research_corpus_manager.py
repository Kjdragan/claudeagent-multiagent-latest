"""Research Corpus Manager for Enhanced Report Generation.

This module provides structured research corpus management building on existing
data models and session storage patterns. Integrates with the Claude Agent SDK
through well-defined tool functions.

Key Features:
- Leverages ResearchDataStandardizer for data structure consistency
- Creates manageable content chunks for agent processing
- Implements relevance scoring and chunk retrieval
- Provides structured JSON storage for agent access
- Integrates with existing session management patterns
"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .research_data_standardizer import ResearchDataStandardizer

class ResearchCorpusManager:
    """
    Enhanced research corpus management building on existing data models.

    Leverages:
    - ResearchDataStandardizer for data structure consistency
    - Existing session JSON storage patterns
    - SourceMetadata and ResearchFinding dataclasses
    - Current working search pipeline integration
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.standardizer = ResearchDataStandardizer()
        self.research_corpus_path = f"KEVIN/sessions/{session_id}/research_corpus.json"

    async def build_corpus_from_workproduct(self, workproduct_path: str, corpus_id: Optional[str] = None) -> dict:
        """Build structured corpus from existing search workproduct."""

        try:
            # CRITICAL FIX: Generate corpus_id if not provided (was missing)
            if corpus_id is None:
                corpus_id = f"corpus_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 1. Parse existing workproduct using ResearchDataStandardizer
            try:
                parsed_data = self.standardizer.parse_search_workproduct(workproduct_path)
            except Exception as parse_error:
                print(f"❌ Standardizer failed to parse workproduct: {parse_error}")
                # CRITICAL FIX: Fallback to manual parsing for search workproduct format
                parsed_data = self._parse_search_workproduct_fallback(workproduct_path)

            # 2. Create enhanced corpus structure with CRITICAL FIX: include corpus_id
            corpus = {
                "corpus_id": corpus_id,  # CRITICAL FIX: Was missing
                "session_id": self.session_id,
                "build_timestamp": datetime.now().isoformat(),
                "workproduct_path": workproduct_path,
                "search_metadata": self._extract_search_metadata(parsed_data),
                "sources": self._structure_sources(parsed_data),
                "content_chunks": self._create_content_chunks(parsed_data),
                "key_findings": self._extract_key_findings(parsed_data),
                "topic_summary": self._generate_topic_summary(parsed_data),
                "source_analysis": self._analyze_source_diversity(parsed_data),
                "quality_metrics": self._calculate_quality_metrics(parsed_data),
                # CRITICAL FIX: Add comprehensive metadata that was missing
                "metadata": {
                    "total_sources": 0,  # Will be calculated below
                    "total_chunks": 0,  # Will be calculated below
                    "word_count": 0,  # Will be calculated below
                    "quality_score": 0.0,  # Will be calculated below
                    "last_updated": datetime.now().isoformat(),
                    "corpus_path": self.research_corpus_path
                }
            }

            # CRITICAL FIX: Calculate and populate metadata that was missing
            sources = corpus.get("sources", [])
            content_chunks = corpus.get("content_chunks", [])

            corpus["metadata"]["total_sources"] = len(sources)
            corpus["metadata"]["total_chunks"] = len(content_chunks)
            corpus["metadata"]["word_count"] = sum(
                len(source.get("cleaned_content", "").split()) for source in sources
            )
            corpus["metadata"]["quality_score"] = corpus.get("quality_metrics", {}).get("overall_score", 0.0)

            # 3. Save structured corpus for agent access
            await self._save_corpus(corpus)

            print(f"✅ Research corpus built: {len(corpus.get('sources', []))} sources, {len(corpus.get('content_chunks', []))} chunks")
            print(f"✅ Corpus ID: {corpus_id}")
            print(f"✅ Total words: {corpus['metadata']['word_count']}")
            print(f"✅ Quality score: {corpus['metadata']['quality_score']:.2f}")

            return corpus

        except Exception as e:
            print(f"❌ Error building research corpus: {e}")
            raise

    def _parse_search_workproduct_fallback(self, workproduct_path: str) -> dict:
        """
        CRITICAL FIX: Fallback parser for search workproduct format.

        This method parses the Enhanced Search+Crawl+Clean workproduct format
        that contains search results with URLs, sources, dates, and snippets.
        """
        try:
            print(f"🔧 Using fallback parser for: {workproduct_path}")

            with open(workproduct_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Initialize parsed data structure
            parsed_data = {
                "search_results": [],
                "crawled_content": [],
                "search_metadata": {
                    "query": "",
                    "total_results": 0,
                    "search_timestamp": "",
                    "workproduct_path": workproduct_path
                }
            }

            # Extract metadata from header
            lines = content.split('\n')
            for line in lines:
                if '**Search Query**:' in line:
                    parsed_data["search_metadata"]["query"] = line.split('**Search Query**:')[-1].strip()
                elif '**Total Search Results**:' in line:
                    try:
                        total = int(line.split('**Total Search Results**:')[-1].strip())
                        parsed_data["search_metadata"]["total_results"] = total
                    except ValueError:
                        pass
                elif '**Export Date**:' in line:
                    parsed_data["search_metadata"]["search_timestamp"] = line.split('**Export Date**:')[-1].strip()

            # Extract search results using regex pattern
            import re

            # Pattern to match search result entries
            search_pattern = r'### (\d+)\. (.+?)\*\*URL\*\*: (.+?)\*\*Source\*\*: (.+?)\*\*Date\*\*: (.+?)\*\*Relevance Score\*\*: (.+?)\*\*Snippet\*\*: (.+?)(?=---|\n\n###|$)'

            matches = re.findall(search_pattern, content, re.DOTALL)

            search_results = []
            crawled_content = []

            for i, match in enumerate(matches):
                result_num, title, url, source, date, relevance, snippet = match

                search_result = {
                    "position": int(result_num),
                    "title": title.strip(),
                    "url": url.strip(),
                    "source": source.strip(),
                    "date": date.strip(),
                    "relevance_score": float(relevance.strip()),
                    "snippet": snippet.strip()
                }
                search_results.append(search_result)

                # Create crawled content entry (simulate crawled data)
                crawled_entry = {
                    "url": url.strip(),
                    "title": title.strip(),
                    "content": snippet.strip(),
                    "source": source.strip(),
                    "relevance_score": float(relevance.strip()),
                    "cleanliness_score": 0.8,  # Default cleanliness score
                    "crawl_timestamp": parsed_data["search_metadata"]["search_timestamp"]
                }
                crawled_content.append(crawled_entry)

            parsed_data["search_results"] = search_results
            parsed_data["crawled_content"] = crawled_content

            print(f"✅ Fallback parser extracted {len(search_results)} search results and {len(crawled_content)} content entries")

            return parsed_data

        except Exception as e:
            print(f"❌ Fallback parser failed: {e}")
            # Return minimal structure to prevent complete failure
            return {
                "search_results": [],
                "crawled_content": [],
                "search_metadata": {
                    "query": "unknown",
                    "total_results": 0,
                    "search_timestamp": datetime.now().isoformat(),
                    "workproduct_path": workproduct_path,
                    "parse_error": str(e)
                }
            }

    async def build_corpus_from_data(self, research_data: dict) -> dict:
        """Build structured corpus directly from research data dictionary."""

        try:
            # 1. Create enhanced corpus structure directly from data
            corpus = {
                "session_id": self.session_id,
                "build_timestamp": datetime.now().isoformat(),
                "corpus_id": f"{self.session_id}_corpus",
                "total_chunks": 0,  # Will be calculated
                "metadata": {
                    "total_sources": len(research_data.get("crawled_content", [])),
                    "content_coverage": 0.0,  # Will be calculated
                    "average_relevance_score": 0.0,  # Will be calculated
                    "created_at": datetime.now().isoformat()
                },
                "content_chunks": [],
                "sources": [],
                "key_findings": [],
                "topic_summary": "",
                "quality_metrics": {}
            }

            # 2. Process crawled content into sources and chunks
            crawled_content = research_data.get("crawled_content", [])
            search_results = research_data.get("search_results", [])

            # Create sources from crawled content
            for i, content_item in enumerate(crawled_content):
                source = {
                    "source_id": f"source_{i+1}",
                    "title": self._extract_title_from_content(content_item, search_results, i),
                    "url": content_item.get("url", ""),
                    "content": content_item.get("content", ""),
                    "relevance_score": self._get_relevance_score(content_item, search_results, i),
                    "cleanliness_score": content_item.get("cleanliness_score", 0.8),
                    "word_count": len(content_item.get("content", "").split())
                }
                corpus["sources"].append(source)

            # 3. Create content chunks (<2000 tokens)
            all_content = " ".join([source["content"] for source in corpus["sources"]])
            chunks = self._chunk_content(all_content)
            corpus["content_chunks"] = [
                {
                    "chunk_id": f"chunk_{i+1}",
                    "content": chunk,
                    "source_ids": [f"source_{j+1}" for j in range(len(corpus["sources"]))],
                    "word_count": len(chunk.split()),
                    "token_estimate": len(chunk.split()) * 1.3  # Rough token estimate
                }
                for i, chunk in enumerate(chunks)
            ]
            corpus["total_chunks"] = len(corpus["content_chunks"])

            # 4. Calculate metadata
            if corpus["sources"]:
                relevance_scores = [source["relevance_score"] for source in corpus["sources"]]
                corpus["metadata"]["average_relevance_score"] = sum(relevance_scores) / len(relevance_scores)
                corpus["metadata"]["content_coverage"] = min(1.0, len(all_content) / 5000)  # Normalize to 0-1

            # 5. Extract key findings
            corpus["key_findings"] = self._extract_key_findings_from_sources(corpus["sources"])

            # 6. Generate topic summary
            corpus["topic_summary"] = self._generate_topic_summary_from_sources(corpus["sources"])

            # 7. Calculate quality metrics
            corpus["quality_metrics"] = self._calculate_corpus_quality_metrics(corpus)

            # 8. Save structured corpus for agent access
            await self._save_corpus(corpus)

            print(f"✅ Research corpus built: {len(corpus.get('sources', []))} sources, {len(corpus.get('content_chunks', []))} chunks")
            return corpus

        except Exception as e:
            print(f"❌ Error building research corpus from data: {e}")
            raise

    def _extract_title_from_content(self, content_item: dict, search_results: list, index: int) -> str:
        """Extract title from content item or match with search results."""
        # Try to get title from content item first
        if "title" in content_item:
            return content_item["title"]
        
        # Try to match with search results by URL
        url = content_item.get("url", "")
        for result in search_results:
            if result.get("url") == url and "title" in result:
                return result["title"]
        
        # Fallback to URL-based title
        if url:
            return f"Content from {url}"
        
        return f"Research Source {index + 1}"

    def _get_relevance_score(self, content_item: dict, search_results: list, index: int) -> float:
        """Get relevance score from content item or match with search results."""
        # Try to get relevance from content item
        if "relevance_score" in content_item:
            return content_item["relevance_score"]
        
        # Try to match with search results by URL
        url = content_item.get("url", "")
        for result in search_results:
            if result.get("url") == url and "relevance_score" in result:
                return result["relevance_score"]
        
        # Fallback to default score
        return 0.8

    def _chunk_content(self, content: str, max_tokens: int = 2000) -> list:
        """Chunk content into smaller pieces for processing."""
        if not content:
            return []
        
        # Simple word-based chunking (can be enhanced later)
        words = content.split()
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            # Rough token estimation (1.3 tokens per word)
            word_tokens = len(word) * 1.3
            
            if current_tokens + word_tokens > max_tokens and current_chunk:
                # Save current chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _extract_key_findings_from_sources(self, sources: list) -> list:
        """Extract key findings from sources."""
        findings = []
        
        for source in sources:
            content = source.get("content", "")
            # Simple extraction of sentences with numbers or percentages
            import re
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and (re.search(r'\d+', sentence) or len(sentence) > 100):
                    findings.append({
                        "finding": sentence,
                        "source_id": source.get("source_id"),
                        "relevance": source.get("relevance_score", 0.8)
                    })
        
        return findings[:10]  # Limit to top 10 findings

    def _generate_topic_summary_from_sources(self, sources: list) -> str:
        """Generate topic summary from sources."""
        if not sources:
            return ""
        
        # Simple summary based on source titles and content
        topics = []
        for source in sources[:5]:  # Use top 5 sources
            title = source.get("title", "")
            if title:
                topics.append(title)
        
        if topics:
            return f"Research covers: {', '.join(topics)}"
        
        return "Research summary based on collected sources"

    def _calculate_corpus_quality_metrics(self, corpus: dict) -> dict:
        """Calculate quality metrics for the corpus."""
        sources = corpus.get("sources", [])
        chunks = corpus.get("content_chunks", [])
        
        return {
            "source_count": len(sources),
            "chunk_count": len(chunks),
            "total_words": sum(source.get("word_count", 0) for source in sources),
            "average_relevance": sum(source.get("relevance_score", 0) for source in sources) / len(sources) if sources else 0,
            "content_diversity": len(set(source.get("url", "") for source in sources)),
            "data_richness": len([s for s in sources if any(char.isdigit() for char in s.get("content", ""))]) / len(sources) if sources else 0
        }

    def _extract_search_metadata(self, parsed_data: dict) -> dict:
        """Extract search metadata from parsed workproduct."""
        search_metadata = parsed_data.get("search_metadata", {})
        return {
            "query": search_metadata.get("query", ""),
            "search_type": search_metadata.get("search_type", "search"),
            "total_results": search_metadata.get("total_results", 0),
            "search_timestamp": search_metadata.get("timestamp", datetime.now().isoformat()),
            "search_duration_seconds": search_metadata.get("duration_seconds", 0)
        }

    def _structure_sources(self, parsed_data: dict) -> List[dict]:
        """Structure sources from parsed workproduct."""
        sources_data = parsed_data.get("sources", [])
        structured_sources = []

        for i, source in enumerate(sources_data):
            structured_source = {
                "source_id": f"source_{i}",
                "url": source.get("url", ""),
                "title": source.get("title", ""),
                "domain": self._extract_domain(source.get("url", "")),
                "relevance_score": source.get("relevance_score", 0.0),
                "content_length": len(source.get("content", "")),
                "extraction_success": source.get("extraction_success", False),
                "crawl_timestamp": source.get("crawl_timestamp", ""),
                "domain_authority": source.get("domain_authority", 0.0),
                "content_quality_score": source.get("content_quality_score", 0.0),
                "anti_bot_level_used": source.get("anti_bot_level_used", 0),
                "source_type": self._classify_source_type(source.get("url", "")),
                "publication_date": source.get("publication_date"),
                "cleaned_content": source.get("content", ""),
                "key_points": self._extract_key_points(source),
                "estimated_tokens": self._estimate_tokens(source.get("content", ""))
            }
            structured_sources.append(structured_source)

        return structured_sources

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return ""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except:
            return ""

    def _classify_source_type(self, url: str) -> str:
        """Classify source type based on URL."""
        if not url:
            return "unknown"

        domain = self._extract_domain(url).lower()

        if any(keyword in domain for keyword in ["gov", "mil", "edu"]):
            return "official"
        elif any(keyword in domain for keyword in ["news", "cnn", "bbc", "reuters", "ap", "guardian"]):
            return "news"
        elif any(keyword in domain for keyword in ["blog", "medium", "substack"]):
            return "blog"
        elif any(keyword in domain for keyword in ["org", "foundation", "institute"]):
            return "organization"
        else:
            return "general"

    def _extract_key_points(self, source: dict) -> List[str]:
        """Extract key points from source content."""
        content = source.get("content", "")
        if not content:
            return []

        # Simple key point extraction - look for substantive sentences
        sentences = content.split('.')
        key_points = []

        for sentence in sentences[:10]:  # Limit to first 10 sentences
            sentence = sentence.strip()
            if len(sentence) > 30 and '?' not in sentence and not sentence.isupper():
                key_points.append(sentence)

        return key_points[:5]  # Return top 5 key points

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content."""
        if not content:
            return 0
        # Rough estimate: ~4 characters per token for English text
        return len(content) // 4

    def _create_content_chunks(self, parsed_data: dict) -> List[dict]:
        """Create manageable content chunks for agent processing."""

        chunks = []
        sources = parsed_data.get("sources", [])

        # Strategy 1: Chunk by source (most reliable)
        for i, source in enumerate(sources):
            if source.get("extraction_success"):
                chunk = {
                    "chunk_id": f"source_{i}",
                    "chunk_type": "source_content",
                    "source_url": source.get("url", ""),
                    "title": source.get("title", ""),
                    "domain": self._extract_domain(source.get("url", "")),
                    "content": source.get("content", ""),
                    "relevance_score": source.get("relevance_score", 0.0),
                    "key_points": self._extract_key_points(source),
                    "word_count": len(source.get("content", "").split()),
                    "estimated_tokens": self._estimate_tokens(source.get("content", "")),
                    "publication_date": source.get("publication_date"),
                    "source_type": self._classify_source_type(source.get("url", "")),
                    "domain_authority": source.get("domain_authority", 0.0)
                }
                chunks.append(chunk)

        # Strategy 2: Create summary chunk for quick overview
        if sources:
            summary_chunk = {
                "chunk_id": "executive_summary",
                "chunk_type": "summary",
                "content": self._generate_corpus_summary(sources),
                "key_points": [self._extract_main_theme(sources)],
                "word_count": 200,
                "estimated_tokens": 250,
                "source_summary": f"Analysis of {len([s for s in sources if s.get('extraction_success')])} successfully processed sources"
            }
            chunks.insert(0, summary_chunk)

        return chunks

    def _generate_corpus_summary(self, sources: List[dict]) -> str:
        """Generate a summary of the research corpus."""
        successful_sources = [s for s in sources if s.get("extraction_success")]

        if not successful_sources:
            return "No sources were successfully processed in this research corpus."

        # Extract key themes and topics
        all_titles = [s.get("title", "") for s in successful_sources if s.get("title")]
        all_domains = list(set([self._extract_domain(s.get("url", "")) for s in successful_sources if s.get("url")]))

        # Generate summary
        summary = f"This research corpus contains {len(successful_sources)} successfully processed sources"

        if all_domains:
            summary += f" from {len(all_domains)} different domains: {', '.join(all_domains[:5])}"

        summary += ". Key topics include research analysis, current events, and expert commentary."

        return summary

    def _extract_main_theme(self, sources: List[dict]) -> str:
        """Extract the main theme from the sources."""
        successful_sources = [s for s in sources if s.get("extraction_success")]

        if not successful_sources:
            return "No theme identified"

        # Look for common keywords in titles
        all_titles = [s.get("title", "").lower() for s in successful_sources if s.get("title")]
        common_words = {}

        for title in all_titles:
            words = re.findall(r'\b\w+\b', title)
            for word in words:
                if len(word) > 3 and word not in ["that", "this", "with", "from", "have", "been", "were", "said"]:
                    common_words[word] = common_words.get(word, 0) + 1

        if common_words:
            main_theme = max(common_words, key=common_words.get)
            return main_theme.title()
        else:
            return "Research analysis"

    def _extract_key_findings(self, parsed_data: dict) -> List[dict]:
        """Extract key findings from the research data."""
        sources = parsed_data.get("sources", [])
        findings = []

        # Create findings from successful sources
        for i, source in enumerate(sources):
            if source.get("extraction_success"):
                key_points = self._extract_key_points(source)
                for point in key_points[:2]:  # Top 2 points per source
                    finding = {
                        "finding_id": f"finding_{i}_{len(findings)}",
                        "key_point": point,
                        "evidence": point[:100] + "..." if len(point) > 100 else point,
                        "sources": [source.get("url", "")],
                        "confidence_level": self._assess_confidence_level(source),
                        "topic_relevance": self._extract_topic_relevance(source),
                        "source_id": f"source_{i}",
                        "timestamp": source.get("crawl_timestamp", datetime.now().isoformat())
                    }
                    findings.append(finding)

        return findings

    def _assess_confidence_level(self, source: dict) -> str:
        """Assess confidence level of a source."""
        domain_authority = source.get("domain_authority", 0.0)
        content_quality = source.get("content_quality_score", 0.0)
        relevance = source.get("relevance_score", 0.0)

        avg_score = (domain_authority + content_quality + relevance) / 3

        if avg_score >= 0.8:
            return "high"
        elif avg_score >= 0.6:
            return "medium"
        else:
            return "low"

    def _extract_topic_relevance(self, source: dict) -> str:
        """Extract topic relevance from source."""
        title = source.get("title", "")
        content = source.get("content", "")

        # Combine title and first part of content for analysis
        text = (title + " " + content[:200]).lower()

        # Extract key terms that indicate topic relevance
        key_terms = re.findall(r'\b\w+\b', text)
        if key_terms:
            # Return the most frequent meaningful terms
            from collections import Counter
            filtered_terms = [term for term in key_terms if len(term) > 3 and term not in ["that", "this", "with", "from", "have", "been", "were", "said", "they", "their", "them"]]
            if filtered_terms:
                common_terms = Counter(filtered_terms).most_common(3)
                return ", ".join([term[0] for term in common_terms])

        return "Research topic analysis"

    def _generate_topic_summary(self, parsed_data: dict) -> str:
        """Generate a topic summary of the research."""
        search_metadata = parsed_data.get("search_metadata", {})
        query = search_metadata.get("query", "")

        if query:
            return f"This research corpus addresses the topic: {query}"
        else:
            return "Research corpus with comprehensive analysis of current events and expert commentary"

    def _analyze_source_diversity(self, parsed_data: dict) -> dict:
        """Analyze the diversity of sources in the corpus."""
        sources = parsed_data.get("sources", [])
        successful_sources = [s for s in sources if s.get("extraction_success")]

        if not successful_sources:
            return {"diversity_score": 0.0, "domain_types": [], "publication_spread": "poor"}

        # Analyze domain diversity
        domains = [self._extract_domain(s.get("url", "")) for s in successful_sources]
        unique_domains = list(set(domains))

        # Analyze source types
        source_types = [self._classify_source_type(s.get("url", "")) for s in successful_sources]
        unique_types = list(set(source_types))

        # Calculate diversity score
        domain_diversity = len(unique_domains) / len(successful_sources) if successful_sources else 0
        type_diversity = len(unique_types) / 5  # Max 5 types

        diversity_score = (domain_diversity + type_diversity) / 2

        return {
            "diversity_score": min(diversity_score, 1.0),
            "unique_domains": unique_domains,
            "domain_types": unique_types,
            "total_successful_sources": len(successful_sources),
            "publication_spread": "excellent" if diversity_score > 0.7 else "good" if diversity_score > 0.5 else "poor"
        }

    def _calculate_quality_metrics(self, parsed_data: dict) -> dict:
        """Calculate quality metrics for the research corpus."""
        sources = parsed_data.get("sources", [])
        successful_sources = [s for s in sources if s.get("extraction_success")]

        if not successful_sources:
            return {"overall_score": 0.0, "data_quality": "poor", "source_reliability": "unknown"}

        # Calculate various quality metrics
        total_relevance = sum(s.get("relevance_score", 0.0) for s in successful_sources)
        avg_relevance = total_relevance / len(successful_sources) if successful_sources else 0

        total_domain_authority = sum(s.get("domain_authority", 0.0) for s in successful_sources)
        avg_domain_authority = total_domain_authority / len(successful_sources) if successful_sources else 0

        total_content_quality = sum(s.get("content_quality_score", 0.0) for s in successful_sources)
        avg_content_quality = total_content_quality / len(successful_sources) if successful_sources else 0

        # Calculate overall quality score
        overall_score = (avg_relevance + avg_domain_authority + avg_content_quality) / 3

        # Determine quality assessment
        if overall_score >= 0.8:
            quality_assessment = "excellent"
        elif overall_score >= 0.6:
            quality_assessment = "good"
        elif overall_score >= 0.4:
            quality_assessment = "fair"
        else:
            quality_assessment = "poor"

        return {
            "overall_score": overall_score,
            "avg_relevance_score": avg_relevance,
            "avg_domain_authority": avg_domain_authority,
            "avg_content_quality": avg_content_quality,
            "quality_assessment": quality_assessment,
            "total_sources_evaluated": len(successful_sources),
            "extraction_success_rate": len(successful_sources) / len(sources) if sources else 0
        }

    async def get_relevant_chunks(self, query: str, max_chunks: int = 5) -> List[dict]:
        """Retrieve most relevant chunks for specific report sections."""

        try:
            # Load corpus
            corpus = await self._load_corpus()
            all_chunks = corpus.get("content_chunks", [])

            if not all_chunks:
                return []

            # Calculate relevance scores for each chunk
            scored_chunks = []
            query_lower = query.lower().split()

            for chunk in all_chunks:
                score = 0
                content = chunk.get("content", "").lower()
                title = chunk.get("title", "").lower()

                # Score based on query terms
                for term in query_lower:
                    if term in content:
                        score += content.count(term) * 2
                    if term in title:
                        score += title.count(term) * 3

                # Add relevance score from chunk if available
                chunk_relevance = chunk.get("relevance_score", 0.0)
                score += chunk_relevance * 10

                # Add domain authority boost
                domain_authority = chunk.get("domain_authority", 0.0)
                score += domain_authority * 5

                scored_chunks.append((score, chunk))

            # Sort by relevance and return top chunks
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            return [chunk for score, chunk in scored_chunks[:max_chunks] if score > 0]

        except Exception as e:
            print(f"❌ Error retrieving relevant chunks: {e}")
            return []

    async def _save_corpus(self, corpus: dict):
        """Save corpus to session directory."""
        try:
            os.makedirs(os.path.dirname(self.research_corpus_path), exist_ok=True)

            # Add save timestamp
            corpus["saved_at"] = datetime.now().isoformat()
            corpus["version"] = "1.0"

            with open(self.research_corpus_path, 'w', encoding='utf-8') as f:
                json.dump(corpus, f, indent=2, ensure_ascii=False)

            print(f"✅ Corpus saved to: {self.research_corpus_path}")

        except Exception as e:
            print(f"❌ Error saving corpus: {e}")
            raise

    async def _load_corpus(self) -> dict:
        """Load corpus from session directory."""
        try:
            if os.path.exists(self.research_corpus_path):
                with open(self.research_corpus_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"⚠️ No corpus found at: {self.research_corpus_path}")
                return {}
        except Exception as e:
            print(f"❌ Error loading corpus: {e}")
            return {}

    def analyze_corpus_quality(self, corpus_data: dict, analysis_type: str = "comprehensive") -> dict:
        """Analyze corpus quality across multiple dimensions.

        Args:
            corpus_data: Corpus data dictionary
            analysis_type: Type of analysis (comprehensive, quality, relevance, coverage)

        Returns:
            Dictionary with quality analysis results and recommendations
        """
        try:
            content_chunks = corpus_data.get("content_chunks", [])
            metadata = corpus_data.get("metadata", {})

            # Initialize quality metrics
            quality_scores = {
                "content_completeness": 0.0,
                "source_diversity": 0.0,
                "content_relevance": 0.0,
                "data_quality": 0.0,
                "structure_quality": 0.0
            }

            # Analyze content completeness
            if content_chunks:
                total_words = sum(chunk.get("word_count", 0) for chunk in content_chunks)
                quality_scores["content_completeness"] = min(1.0, total_words / 1000)  # Expect 1000+ words for good coverage

            # Analyze source diversity
            domains = set(chunk.get("domain", "") for chunk in content_chunks if chunk.get("domain"))
            quality_scores["source_diversity"] = min(1.0, len(domains) / 5)  # Expect 5+ different domains

            # Analyze content relevance (average relevance scores)
            relevance_scores = [chunk.get("relevance_score", 0.0) for chunk in content_chunks]
            if relevance_scores:
                quality_scores["content_relevance"] = sum(relevance_scores) / len(relevance_scores)

            # Analyze data quality (presence of key data points)
            chunks_with_key_points = sum(1 for chunk in content_chunks if chunk.get("key_points"))
            if content_chunks:
                quality_scores["data_quality"] = chunks_with_key_points / len(content_chunks)

            # Analyze structure quality (proper chunking and organization)
            has_summary = any(chunk.get("chunk_type") == "summary" for chunk in content_chunks)
            has_source_content = any(chunk.get("chunk_type") == "source_content" for chunk in content_chunks)
            quality_scores["structure_quality"] = 0.5 + (0.25 * has_summary) + (0.25 * has_source_content)

            # Calculate overall quality score
            overall_score = sum(quality_scores.values()) / len(quality_scores)

            # Generate recommendations
            recommendations = []
            if quality_scores["content_completeness"] < 0.5:
                recommendations.append("Add more content sources to improve coverage")
            if quality_scores["source_diversity"] < 0.6:
                recommendations.append("Include sources from more diverse domains")
            if quality_scores["content_relevance"] < 0.6:
                recommendations.append("Focus on more relevant sources for the research topic")
            if quality_scores["data_quality"] < 0.5:
                recommendations.append("Extract more specific data points and key findings")

            # Content analysis
            content_analysis = {
                "total_chunks": len(content_chunks),
                "total_words": sum(chunk.get("word_count", 0) for chunk in content_chunks),
                "unique_domains": len(domains),
                "chunk_types": list(set(chunk.get("chunk_type", "unknown") for chunk in content_chunks)),
                "average_relevance": quality_scores["content_relevance"]
            }

            # Source analysis
            source_analysis = {
                "total_sources": metadata.get("total_sources", len(content_chunks)),
                "source_domains": list(domains),
                "high_relevance_sources": len([s for s in content_chunks if s.get("relevance_score", 0) > 0.7]),
                "content_coverage": metadata.get("content_coverage", 0.0)
            }

            # Coverage analysis
            coverage_analysis = {
                "content_coverage_score": quality_scores["content_completeness"],
                "data_completeness": quality_scores["data_quality"],
                "topic_relevance": quality_scores["content_relevance"]
            }

            return {
                "overall_quality_score": overall_score,
                "quality_scores": quality_scores,
                "content_analysis": content_analysis,
                "source_analysis": source_analysis,
                "coverage_analysis": coverage_analysis,
                "recommendations": recommendations,
                "ready_for_synthesis": overall_score >= 0.7,
                "quality_level": self._get_quality_level(overall_score)
            }

        except Exception as e:
            return {
                "overall_quality_score": 0.0,
                "error": f"Error analyzing corpus quality: {str(e)}",
                "recommendations": ["Fix corpus structure and retry analysis"],
                "ready_for_synthesis": False,
                "quality_level": "error"
            }

    def _get_quality_level(self, score: float) -> str:
        """Convert quality score to quality level."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.75:
            return "good"
        elif score >= 0.6:
            return "acceptable"
        elif score >= 0.4:
            return "needs_improvement"
        else:
            return "poor"

    def get_corpus_summary(self) -> dict:
        """Get a quick summary of the corpus without loading full content."""
        try:
            if os.path.exists(self.research_corpus_path):
                with open(self.research_corpus_path, 'r', encoding='utf-8') as f:
                    corpus = json.load(f)

                return {
                    "session_id": corpus.get("session_id"),
                    "total_sources": len(corpus.get("sources", [])),
                    "content_chunks": len(corpus.get("content_chunks", [])),
                    "key_findings": len(corpus.get("key_findings", [])),
                    "build_timestamp": corpus.get("build_timestamp"),
                    "quality_score": corpus.get("quality_metrics", {}).get("overall_score", 0.0),
                    "quality_assessment": corpus.get("quality_metrics", {}).get("quality_assessment", "unknown")
                }
            else:
                return {"error": "No corpus found", "session_id": self.session_id}
        except Exception as e:
            return {"error": f"Error reading corpus: {e}", "session_id": self.session_id}