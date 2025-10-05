"""Research Data Standardizer for Multi-Agent Research System.

This module creates standardized research data structures that can be consumed
by the report generation pipeline. It transforms raw search results into
structured session data that matches the expected format for report agents.

Based on the Report Generation and Editorial Review Technical Documentation,
this ensures that research data is properly formatted for seamless integration
between research agents and report generation agents.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

try:
    from .logging_config import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("research_data_standardizer")


@dataclass
class SourceMetadata:
    """Metadata for a single research source."""
    url: str
    title: str
    domain: str
    relevance_score: float
    content_length: int
    extraction_success: bool
    crawl_timestamp: str
    domain_authority: float | None = None
    content_quality_score: float | None = None
    anti_bot_level_used: int | None = None


@dataclass
class ResearchFinding:
    """A single research finding with source attribution."""
    key_point: str
    evidence: str
    sources: list[str]
    confidence_level: str  # "high", "medium", "low"
    topic_relevance: str
    timestamp: str


@dataclass
class SearchMetrics:
    """Metrics from the search process."""
    query: str
    search_mode: str
    total_results: int
    selected_urls: int
    successfully_crawled: int
    content_extracted: int
    total_content_chars: int
    average_relevance_score: float
    crawl_success_rate: float
    processing_time_seconds: float
    anti_bot_levels_used: list[int]


@dataclass
class StandardizedResearchData:
    """Standardized research data structure for report generation."""
    session_id: str
    research_topic: str
    research_timestamp: str
    search_metrics: list[SearchMetrics]
    sources: list[SourceMetadata]
    findings: list[ResearchFinding]
    content_summary: str
    key_themes: list[str]
    source_analysis: dict[str, Any]
    quality_assessment: dict[str, Any]
    research_metadata: dict[str, Any]


class ResearchDataStandardizer:
    """Standardizes research data for report generation integration."""

    def __init__(self):
        self.logger = get_logger("research_data_standardizer")

    def parse_search_workproduct(self, workproduct_path: str) -> dict[str, Any]:
        """Parse a search workproduct markdown file.

        Args:
            workproduct_path: Path to the workproduct markdown file

        Returns:
            Dictionary containing parsed search data
        """
        try:
            with open(workproduct_path, encoding='utf-8') as f:
                content = f.read()

            # Parse sections from the markdown
            sections = self._parse_markdown_sections(content)

            # Extract search results
            search_results = self._extract_search_results(sections.get("Search Results", ""))

            # Extract extracted content
            extracted_content = self._extract_content_sections(sections.get("Extracted Content", ""))

            # Extract metadata
            metadata = self._extract_metadata(sections.get("Search Metadata", ""))

            return {
                "search_results": search_results,
                "extracted_content": extracted_content,
                "metadata": metadata,
                "raw_content": content
            }

        except Exception as e:
            self.logger.error(f"Failed to parse workproduct {workproduct_path}: {e}")
            return {}

    def _parse_markdown_sections(self, content: str) -> dict[str, str]:
        """Parse markdown content into sections."""
        sections = {}
        current_section = ""
        current_content = []

        for line in content.split('\n'):
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()

                # Start new section
                current_section = line.strip('#').strip()
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def _extract_search_results(self, search_results_text: str) -> list[dict[str, Any]]:
        """Extract search results from the text."""
        results = []
        current_result = {}

        for line in search_results_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith(f"{len(results) + 1}."):
                # Save previous result
                if current_result:
                    results.append(current_result)

                # Start new result
                title_end = line.find(' - ')
                if title_end != -1:
                    current_result = {
                        "position": len(results) + 1,
                        "title": line[3:title_end].strip(),
                        "snippet": line[title_end + 3:].strip()
                    }
            elif line.startswith("URL: ") and current_result:
                current_result["url"] = line[4:].strip()
            elif line.startswith("Relevance Score: ") and current_result:
                try:
                    score = float(line.split(':')[1].strip())
                    current_result["relevance_score"] = score
                except:
                    pass

        # Save last result
        if current_result:
            results.append(current_result)

        return results

    def _extract_content_sections(self, content_text: str) -> list[dict[str, Any]]:
        """Extract content sections from the text."""
        sections = []
        current_section = {}

        for line in content_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith("## Content from:"):
                # Save previous section
                if current_section:
                    sections.append(current_section)

                # Start new section
                url = line.replace("## Content from:", "").strip()
                current_section = {"url": url, "content": ""}
            elif line.startswith("**Content Length:**") and current_section:
                try:
                    length = int(line.split(':')[1].strip().replace(',', ''))
                    current_section["content_length"] = length
                except:
                    pass
            elif line.startswith("**Processing Notes:**") and current_section:
                current_section["processing_notes"] = line.replace("**Processing Notes:**", "").strip()
            elif not line.startswith('#') and not line.startswith('**') and current_section:
                # Add content line
                if current_section["content"]:
                    current_section["content"] += "\n"
                current_section["content"] += line

        # Save last section
        if current_section:
            sections.append(current_section)

        return sections

    def _extract_metadata(self, metadata_text: str) -> dict[str, Any]:
        """Extract metadata from the text."""
        metadata = {}

        for line in metadata_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()

        return metadata

    def create_standardized_research_data(
        self,
        session_id: str,
        research_topic: str,
        workproduct_paths: list[str]
    ) -> StandardizedResearchData:
        """Create standardized research data from workproduct files.

        Args:
            session_id: Research session identifier
            research_topic: The main research topic/query
            workproduct_paths: List of paths to workproduct files

        Returns:
            StandardizedResearchData object
        """
        self.logger.info(f"Creating standardized research data for session {session_id}")

        # Parse all workproducts
        all_search_results = []
        all_extracted_content = []
        all_metadata = {}

        for workproduct_path in workproduct_paths:
            if os.path.exists(workproduct_path):
                parsed_data = self.parse_search_workproduct(workproduct_path)
                all_search_results.extend(parsed_data.get("search_results", []))
                all_extracted_content.extend(parsed_data.get("extracted_content", []))
                all_metadata.update(parsed_data.get("metadata", {}))

        # Create source metadata
        sources = self._create_source_metadata(all_search_results, all_extracted_content)

        # Create search metrics
        search_metrics = self._create_search_metrics(all_metadata, research_topic)

        # Extract research findings
        findings = self._extract_research_findings(all_extracted_content, research_topic)

        # Generate content summary and themes
        content_summary = self._generate_content_summary(all_extracted_content)
        key_themes = self._extract_key_themes(all_extracted_content, research_topic)

        # Analyze sources
        source_analysis = self._analyze_sources(sources)

        # Quality assessment
        quality_assessment = self._assess_quality(sources, findings, content_summary)

        # Research metadata
        research_metadata = {
            "total_sources": len(sources),
            "successful_extractions": len([s for s in sources if s.extraction_success]),
            "content_diversity": len(set(s.domain for s in sources)),
            "average_relevance": sum(s.relevance_score for s in sources) / len(sources) if sources else 0,
            "research_completion_time": datetime.now().isoformat(),
            "data_standardization_version": "1.0.0"
        }

        return StandardizedResearchData(
            session_id=session_id,
            research_topic=research_topic,
            research_timestamp=datetime.now().isoformat(),
            search_metrics=search_metrics,
            sources=sources,
            findings=findings,
            content_summary=content_summary,
            key_themes=key_themes,
            source_analysis=source_analysis,
            quality_assessment=quality_assessment,
            research_metadata=research_metadata
        )

    def _create_source_metadata(
        self,
        search_results: list[dict[str, Any]],
        extracted_content: list[dict[str, Any]]
    ) -> list[SourceMetadata]:
        """Create source metadata from search results and extracted content."""
        sources = []

        # Create a mapping from URL to extracted content
        content_map = {content["url"]: content for content in extracted_content}

        for result in search_results:
            url = result.get("url", "")
            if not url:
                continue

            content = content_map.get(url, {})

            # Extract domain from URL
            domain = url.split('/')[2] if '/' in url else url

            source = SourceMetadata(
                url=url,
                title=result.get("title", ""),
                domain=domain,
                relevance_score=result.get("relevance_score", 0.0),
                content_length=content.get("content_length", 0),
                extraction_success=bool(content),
                crawl_timestamp=datetime.now().isoformat(),
                domain_authority=self._calculate_domain_authority(domain),
                content_quality_score=self._calculate_content_quality_score(content),
                anti_bot_level_used=content.get("anti_bot_level")
            )

            sources.append(source)

        return sources

    def _create_search_metrics(self, metadata: dict[str, Any], research_topic: str) -> list[SearchMetrics]:
        """Create search metrics from metadata."""
        metrics = []

        # Extract search information from metadata
        queries = [research_topic]  # Primary query
        if "Additional Queries" in metadata:
            additional_queries = metadata["Additional Queries"].split(', ')
            queries.extend(additional_queries)

        for i, query in enumerate(queries):
            metric = SearchMetrics(
                query=query,
                search_mode=metadata.get("Search Mode", "web"),
                total_results=int(metadata.get("Total Results", 0)),
                selected_urls=int(metadata.get("Selected URLs", 0)),
                successfully_crawled=int(metadata.get("Successfully Crawled", 0)),
                content_extracted=int(metadata.get("Content Extracted", 0)),
                total_content_chars=int(metadata.get("Total Content Characters", 0)),
                average_relevance_score=float(metadata.get("Average Relevance Score", 0.0)),
                crawl_success_rate=float(metadata.get("Crawl Success Rate", 0.0)),
                processing_time_seconds=float(metadata.get("Processing Time", 0.0)),
                anti_bot_levels_used=[0, 1]  # Default levels used
            )
            metrics.append(metric)

        return metrics

    def _extract_research_findings(
        self,
        extracted_content: list[dict[str, Any]],
        research_topic: str
    ) -> list[ResearchFinding]:
        """Extract key research findings from content."""
        findings = []

        # Analyze content to extract key points
        for content_item in extracted_content:
            content_text = content_item.get("content", "")
            if not content_text:
                continue

            # Use simple heuristics to extract key findings
            sentences = content_text.split('. ')

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 50:  # Skip short sentences
                    continue

                # Check if sentence contains relevant information
                if self._is_relevant_finding(sentence, research_topic):
                    finding = ResearchFinding(
                        key_point=sentence,
                        evidence=self._extract_evidence_for_finding(sentence, content_text),
                        sources=[content_item["url"]],
                        confidence_level=self._assess_confidence_level(sentence),
                        topic_relevance=self._assess_topic_relevance(sentence, research_topic),
                        timestamp=datetime.now().isoformat()
                    )
                    findings.append(finding)

        # Remove duplicates and limit to top findings
        unique_findings = self._deduplicate_findings(findings)
        return unique_findings[:20]  # Limit to top 20 findings

    def _generate_content_summary(self, extracted_content: list[dict[str, Any]]) -> str:
        """Generate a summary of all extracted content."""
        all_content = " ".join([content.get("content", "") for content in extracted_content])

        # Create a summary by extracting key sentences
        sentences = all_content.split('. ')
        key_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 100 and len(key_sentences) < 10:
                key_sentences.append(sentence)

        return ". ".join(key_sentences) + "." if key_sentences else "No content summary available."

    def _extract_key_themes(self, extracted_content: list[dict[str, Any]], research_topic: str) -> list[str]:
        """Extract key themes from the content."""
        # Simple theme extraction based on common words and phrases
        all_content = " ".join([content.get("content", "") for content in extracted_content]).lower()

        # Common themes in healthcare/AI content
        potential_themes = [
            "artificial intelligence",
            "machine learning",
            "healthcare",
            "medical diagnosis",
            "patient care",
            "treatment",
            "technology",
            "innovation",
            "clinical trials",
            "research",
            "data analysis",
            "automation",
            "efficiency",
            "accuracy",
            "decision support"
        ]

        themes = []
        for theme in potential_themes:
            if theme in all_content:
                themes.append(theme.title())

        return themes[:10]  # Limit to top 10 themes

    def _analyze_sources(self, sources: list[SourceMetadata]) -> dict[str, Any]:
        """Analyze the sources used in research."""
        if not sources:
            return {}

        domains = [source.domain for source in sources]
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        return {
            "total_sources": len(sources),
            "unique_domains": len(set(domains)),
            "domain_distribution": domain_counts,
            "average_relevance_score": sum(source.relevance_score for source in sources) / len(sources),
            "average_content_length": sum(source.content_length for source in sources) / len(sources),
            "successful_extractions": len([source for source in sources if source.extraction_success]),
            "success_rate": len([source for source in sources if source.extraction_success]) / len(sources) * 100,
            "high_quality_sources": len([source for source in sources if source.relevance_score > 0.7])
        }

    def _assess_quality(
        self,
        sources: list[SourceMetadata],
        findings: list[ResearchFinding],
        content_summary: str
    ) -> dict[str, Any]:
        """Assess the quality of the research."""
        return {
            "source_quality_score": self._calculate_source_quality_score(sources),
            "finding_confidence_score": self._calculate_finding_confidence_score(findings),
            "content_completeness_score": self._calculate_completeness_score(content_summary),
            "overall_quality_score": 0.0,  # Will be calculated below
            "quality_indicators": {
                "authoritative_sources": len([s for s in sources if s.domain_authority and s.domain_authority > 0.7]),
                "high_confidence_findings": len([f for f in findings if f.confidence_level == "high"]),
                "comprehensive_content": len(content_summary) > 500,
                "diverse_domains": len(set(s.domain for s in sources)) > 3
            }
        }

    def _calculate_domain_authority(self, domain: str) -> float:
        """Calculate domain authority based on known authoritative domains."""
        authoritative_domains = {
            "harvard.edu": 0.95,
            "stanford.edu": 0.95,
            "mayo.edu": 0.95,
            "nature.com": 0.90,
            "science.org": 0.90,
            "nih.gov": 0.90,
            "pubmed.ncbi.nlm.nih.gov": 0.90,
            "pmc.ncbi.nlm.nih.gov": 0.90,
            "aapa.org": 0.80,
            "himss.org": 0.80
        }

        return authoritative_domains.get(domain.lower(), 0.5)

    def _calculate_content_quality_score(self, content: dict[str, Any]) -> float:
        """Calculate content quality score based on content characteristics."""
        content_text = content.get("content", "")
        content_length = len(content_text)

        # Base score from content length
        if content_length > 10000:
            length_score = 1.0
        elif content_length > 5000:
            length_score = 0.8
        elif content_length > 1000:
            length_score = 0.6
        else:
            length_score = 0.4

        # Adjust for processing notes
        processing_notes = content.get("processing_notes", "").lower()
        if "successfully cleaned" in processing_notes:
            length_score += 0.1

        return min(1.0, length_score)

    def _is_relevant_finding(self, sentence: str, research_topic: str) -> bool:
        """Check if a sentence contains a relevant research finding."""
        # Simple heuristic: look for keywords related to the research topic
        topic_words = research_topic.lower().split()
        sentence_lower = sentence.lower()

        # Check if sentence contains at least one topic word
        return any(word in sentence_lower for word in topic_words if len(word) > 3)

    def _extract_evidence_for_finding(self, finding: str, content: str) -> str:
        """Extract evidence supporting a finding."""
        # Simple implementation: return surrounding context
        sentences = content.split('. ')
        for i, sentence in enumerate(sentences):
            if finding in sentence:
                # Return surrounding sentences as evidence
                start = max(0, i - 1)
                end = min(len(sentences), i + 2)
                return '. '.join(sentences[start:end])

        return finding  # Fallback to the finding itself

    def _assess_confidence_level(self, finding: str) -> str:
        """Assess confidence level for a finding."""
        # Simple heuristic based on finding characteristics
        if any(indicator in finding.lower() for indicator in ["study shows", "research indicates", "evidence suggests"]):
            return "high"
        elif any(indicator in finding.lower() for indicator in ["might", "could", "potential"]):
            return "medium"
        else:
            return "medium"

    def _assess_topic_relevance(self, finding: str, research_topic: str) -> str:
        """Assess how relevant a finding is to the research topic."""
        topic_words = set(research_topic.lower().split())
        finding_words = set(finding.lower().split())

        overlap = topic_words.intersection(finding_words)
        if len(overlap) >= 3:
            return "high"
        elif len(overlap) >= 2:
            return "medium"
        else:
            return "low"

    def _deduplicate_findings(self, findings: list[ResearchFinding]) -> list[ResearchFinding]:
        """Remove duplicate findings."""
        unique_findings = []
        seen_texts = set()

        for finding in findings:
            # Use first 100 characters as a deduplication key
            key = finding.key_point[:100].lower()
            if key not in seen_texts:
                seen_texts.add(key)
                unique_findings.append(finding)

        return unique_findings

    def _calculate_source_quality_score(self, sources: list[SourceMetadata]) -> float:
        """Calculate overall source quality score."""
        if not sources:
            return 0.0

        total_score = sum(
            source.relevance_score * (source.domain_authority or 0.5)
            for source in sources
        )
        return min(1.0, total_score / len(sources))

    def _calculate_finding_confidence_score(self, findings: list[ResearchFinding]) -> float:
        """Calculate finding confidence score."""
        if not findings:
            return 0.0

        high_confidence = len([f for f in findings if f.confidence_level == "high"])
        medium_confidence = len([f for f in findings if f.confidence_level == "medium"])

        return (high_confidence * 1.0 + medium_confidence * 0.5) / len(findings)

    def _calculate_completeness_score(self, content_summary: str) -> float:
        """Calculate content completeness score."""
        length = len(content_summary)

        if length > 2000:
            return 1.0
        elif length > 1000:
            return 0.8
        elif length > 500:
            return 0.6
        else:
            return 0.4

    def save_standardized_data(
        self,
        standardized_data: StandardizedResearchData,
        output_path: str
    ) -> bool:
        """Save standardized research data to file.

        Args:
            standardized_data: The standardized data to save
            output_path: Path where to save the data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to dictionary
            data_dict = asdict(standardized_data)

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Standardized research data saved to: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save standardized data: {e}")
            return False


# Global standardizer instance
_research_standardizer = None


def get_research_standardizer() -> ResearchDataStandardizer:
    """Get the global research standardizer instance."""
    global _research_standardizer
    if _research_standardizer is None:
        _research_standardizer = ResearchDataStandardizer()
    return _research_standardizer


def standardize_and_save_research_data(
    session_id: str,
    research_topic: str,
    workproduct_dir: str,
    session_dir: str
) -> str | None:
    """Standardize research data and save to session directory.

    Args:
        session_id: Research session identifier
        research_topic: The research topic/query
        workproduct_dir: Directory containing workproduct files
        session_dir: Session directory to save standardized data

    Returns:
        Path to saved standardized data file, or None if failed
    """
    standardizer = get_research_standardizer()

    # Find all workproduct files
    workproduct_files = []
    if os.path.exists(workproduct_dir):
        for file in os.listdir(workproduct_dir):
            if file.startswith("search_workproduct_") and file.endswith(".md"):
                workproduct_files.append(os.path.join(workproduct_dir, file))

    if not workproduct_files:
        logger.warning(f"No workproduct files found in {workproduct_dir}")
        return None

    # Create standardized data
    standardized_data = standardizer.create_standardized_research_data(
        session_id=session_id,
        research_topic=research_topic,
        workproduct_paths=workproduct_files
    )

    # Save to session directory
    output_path = os.path.join(session_dir, "research_findings.json")
    success = standardizer.save_standardized_data(standardized_data, output_path)

    if success:
        logger.info(f"Research data standardized and saved to: {output_path}")
        return output_path
    else:
        logger.error(f"Failed to save standardized research data to: {output_path}")
        return None
