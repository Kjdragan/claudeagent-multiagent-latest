"""
Research Corpus Analysis and Sufficiency Assessment System

Phase 3.2.3: Sophisticated research corpus analysis with comprehensive
sufficiency assessment, coverage evaluation, and quality metrics.

This module provides advanced research corpus analysis capabilities that go
beyond basic content analysis to provide multi-dimensional assessment of
research completeness, quality, and sufficiency for editorial decision-making.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import asyncio
import hashlib
import json
import re
from collections import Counter, defaultdict
import math

from pydantic import BaseModel, Field


class CoverageType(str, Enum):
    """Types of coverage analysis"""
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    GEOGRAPHICAL = "geographical"
    COMPARATIVE = "comparative"
    TECHNICAL = "technical"
    PERSPECTIVE = "perspective"
    EVIDENCE = "evidence"
    SOURCE_DIVERSITY = "source_diversity"


class QualityDimension(str, Enum):
    """Dimensions of content quality assessment"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    AUTHORITY = "authority"
    RECENCY = "recency"
    DEPTH = "depth"
    CLARITY = "clarity"
    OBJECTIVITY = "objectivity"


class SufficiencyLevel(str, Enum):
    """Levels of research sufficiency"""
    INSUFFICIENT = "insufficient"
    MINIMAL = "minimal"
    ADEQUATE = "adequate"
    GOOD = "good"
    COMPREHENSIVE = "comprehensive"
    EXEMPLARY = "exemplary"


@dataclass
class SourceAssessment:
    """Assessment of individual research sources"""
    source_id: str
    url: str
    title: str
    domain_authority: float  # 0-1
    content_quality: float   # 0-1
    recency_score: float     # 0-1
    relevance_score: float   # 0-1
    uniqueness_score: float  # 0-1
    content_length: int
    key_topics: List[str]
    cited_sources: List[str]
    publication_date: Optional[datetime]
    source_type: str  # academic, news, blog, official, etc.
    bias_rating: Optional[str]  # left, center, right, unknown
    fact_check_rating: Optional[float]  # 0-1 if available

    @property
    def overall_quality(self) -> float:
        """Calculate overall source quality"""
        weights = {
            'authority': 0.25,
            'quality': 0.20,
            'recency': 0.15,
            'relevance': 0.25,
            'uniqueness': 0.15
        }

        return (
            self.domain_authority * weights['authority'] +
            self.content_quality * weights['quality'] +
            self.recency_score * weights['recency'] +
            self.relevance_score * weights['relevance'] +
            self.uniqueness_score * weights['uniqueness']
        )


@dataclass
class CoverageAnalysis:
    """Analysis of topic coverage across different dimensions"""
    coverage_type: CoverageType
    coverage_score: float      # 0-1
    covered_aspects: List[str]
    missing_aspects: List[str]
    weak_aspects: List[str]
    evidence_strength: float   # 0-1
    source_diversity: float    # 0-1
    depth_indicators: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]  # (lower, upper)
    coverage_gaps: List[Dict[str, Any]]

    @property
    def is_comprehensive(self) -> bool:
        """Check if coverage is comprehensive"""
        return (self.coverage_score >= 0.8 and
                len(self.missing_aspects) == 0 and
                self.evidence_strength >= 0.7)

    @property
    def has_significant_gaps(self) -> bool:
        """Check if there are significant coverage gaps"""
        return (len(self.missing_aspects) > 2 or
                any(gap['severity'] == 'high' for gap in self.coverage_gaps))


@dataclass
class QualityAssessment:
    """Multi-dimensional quality assessment of research corpus"""
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    quality_issues: List[Dict[str, Any]]
    quality_strengths: List[str]
    improvement_areas: List[str]
    confidence_level: float  # 0-1
    assessment_metadata: Dict[str, Any]

    @property
    def is_high_quality(self) -> bool:
        """Check if content meets high quality standards"""
        return (self.overall_score >= 0.8 and
                self.confidence_level >= 0.7)

    @property
    def quality_level(self) -> SufficiencyLevel:
        """Determine quality level based on scores"""
        if self.overall_score >= 0.9:
            return SufficiencyLevel.EXEMPLARY
        elif self.overall_score >= 0.8:
            return SufficiencyLevel.COMPREHENSIVE
        elif self.overall_score >= 0.7:
            return SufficiencyLevel.GOOD
        elif self.overall_score >= 0.6:
            return SufficiencyLevel.ADEQUATE
        elif self.overall_score >= 0.4:
            return SufficiencyLevel.MINIMAL
        else:
            return SufficiencyLevel.INSUFFICIENT


@dataclass
class SufficiencyAssessment:
    """Overall sufficiency assessment for editorial decisions"""
    topic: str
    research_purpose: str
    overall_sufficiency: SufficiencyLevel
    sufficiency_score: float  # 0-1
    coverage_assessments: Dict[CoverageType, CoverageAnalysis]
    quality_assessment: QualityAssessment
    source_analysis: Dict[str, Any]
    gap_analysis: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_intervals: Dict[str, Tuple[float, float]]
    assessment_timestamp: datetime
    corpus_metadata: Dict[str, Any]

    @property
    def is_sufficient_for_purpose(self) -> bool:
        """Check if research is sufficient for its intended purpose"""
        return (self.sufficiency_score >= 0.7 and
                self.quality_assessment.overall_score >= 0.65)

    @property
    def needs_gap_research(self) -> bool:
        """Determine if gap research is needed"""
        return (self.sufficiency_score < 0.8 or
                len([g for g in self.gap_analysis if g['severity'] in ['high', 'critical']]) > 0)

    @property
    def critical_gaps(self) -> List[Dict[str, Any]]:
        """Get list of critical gaps that need addressing"""
        return [gap for gap in self.gap_analysis if gap['severity'] == 'critical']

    @property
    def high_priority_gaps(self) -> List[Dict[str, Any]]:
        """Get list of high-priority gaps"""
        return [gap for gap in self.gap_analysis if gap['severity'] in ['high', 'critical']]


class ResearchCorpusAnalyzer:
    """
    Advanced research corpus analyzer with comprehensive sufficiency assessment.

    This analyzer provides multi-dimensional evaluation of research completeness,
    quality, and sufficiency through sophisticated algorithms and metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self._quality_weights = self.config.get('quality_weights', {})
        self._coverage_thresholds = self.config.get('coverage_thresholds', {})
        self._source_requirements = self.config.get('source_requirements', {})

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for corpus analysis"""
        return {
            'quality_weights': {
                QualityDimension.ACCURACY: 0.20,
                QualityDimension.COMPLETENESS: 0.18,
                QualityDimension.RELEVANCE: 0.22,
                QualityDimension.AUTHORITY: 0.15,
                QualityDimension.RECENCY: 0.10,
                QualityDimension.DEPTH: 0.10,
                QualityDimension.CLARITY: 0.03,
                QualityDimension.OBJECTIVITY: 0.02
            },
            'coverage_thresholds': {
                'comprehensive_min': 0.8,
                'adequate_min': 0.6,
                'minimal_min': 0.4
            },
            'source_requirements': {
                'min_sources': 3,
                'min_diversity_score': 0.6,
                'min_authority_avg': 0.5,
                'preferred_source_types': ['academic', 'official', 'news'],
                'min_recency_score': 0.4  # for current topics
            },
            'analysis_settings': {
                'min_confidence_threshold': 0.6,
                'max_gap_severity_threshold': 0.8,
                'coverage_aspect_weight': 0.1,
                'evidence_strength_weight': 0.3
            }
        }

    async def analyze_research_corpus(
        self,
        research_data: Dict[str, Any],
        topic: str,
        research_purpose: str = "general_research"
    ) -> SufficiencyAssessment:
        """
        Perform comprehensive analysis of research corpus

        Args:
            research_data: Research data including sources and content
            topic: Research topic
            research_purpose: Purpose of research (affects analysis criteria)

        Returns:
            Comprehensive sufficiency assessment
        """
        # Extract sources from research data
        sources = self._extract_sources(research_data)

        # Analyze individual sources
        source_assessments = await self._analyze_sources(sources, topic)

        # Analyze coverage across different dimensions
        coverage_assessments = await self._analyze_coverage(
            research_data, topic, source_assessments
        )

        # Assess overall quality
        quality_assessment = await self._assess_quality(
            research_data, source_assessments, coverage_assessments
        )

        # Analyze gaps
        gap_analysis = await self._analyze_gaps(
            research_data, coverage_assessments, quality_assessment
        )

        # Analyze source patterns
        source_analysis = await self._analyze_source_patterns(source_assessments)

        # Calculate overall sufficiency
        sufficiency_score, sufficiency_level = await self._calculate_sufficiency(
            coverage_assessments, quality_assessment, source_analysis, research_purpose
        )

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            coverage_assessments, quality_assessment, gap_analysis, sufficiency_level
        )

        # Calculate confidence intervals
        confidence_intervals = await self._calculate_confidence_intervals(
            coverage_assessments, quality_assessment, source_assessments
        )

        return SufficiencyAssessment(
            topic=topic,
            research_purpose=research_purpose,
            overall_sufficiency=sufficiency_level,
            sufficiency_score=sufficiency_score,
            coverage_assessments=coverage_assessments,
            quality_assessment=quality_assessment,
            source_analysis=source_analysis,
            gap_analysis=gap_analysis,
            recommendations=recommendations,
            confidence_intervals=confidence_intervals,
            assessment_timestamp=datetime.now(),
            corpus_metadata=self._generate_corpus_metadata(research_data, len(sources))
        )

    def _extract_sources(self, research_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source information from research data"""
        sources = []

        # Extract from different possible structures
        if 'sources' in research_data:
            sources.extend(research_data['sources'])

        if 'research_results' in research_data:
            results = research_data['research_results']
            if isinstance(results, dict) and 'sources' in results:
                sources.extend(results['sources'])
            elif isinstance(results, list):
                for result in results:
                    if isinstance(result, dict) and 'sources' in result:
                        sources.extend(result['sources'])

        if 'content_analysis' in research_data:
            content_analysis = research_data['content_analysis']
            if 'sources' in content_analysis:
                sources.extend(content_analysis['sources'])

        # Remove duplicates
        seen_urls = set()
        unique_sources = []
        for source in sources:
            url = source.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)

        return unique_sources

    async def _analyze_sources(
        self,
        sources: List[Dict[str, Any]],
        topic: str
    ) -> List[SourceAssessment]:
        """Analyze individual research sources"""
        assessments = []

        for source in sources:
            assessment = await self._assess_single_source(source, topic)
            assessments.append(assessment)

        return assessments

    async def _assess_single_source(
        self,
        source: Dict[str, Any],
        topic: str
    ) -> SourceAssessment:
        """Assess a single research source"""
        url = source.get('url', '')
        title = source.get('title', '')

        # Calculate domain authority
        domain_authority = await self._calculate_domain_authority(url)

        # Assess content quality
        content_quality = await self._assess_content_quality(source)

        # Calculate recency score
        recency_score = await self._calculate_recency_score(source)

        # Assess relevance
        relevance_score = await self._assess_relevance(source, topic)

        # Calculate uniqueness
        uniqueness_score = await self._calculate_uniqueness(source)

        # Extract metadata
        content_length = len(source.get('content', ''))
        key_topics = self._extract_key_topics(source)
        cited_sources = source.get('cited_sources', [])
        publication_date = self._parse_publication_date(source)
        source_type = self._determine_source_type(source)
        bias_rating = self._assess_bias(source)
        fact_check_rating = self._assess_fact_check(source)

        return SourceAssessment(
            source_id=self._generate_source_id(url),
            url=url,
            title=title,
            domain_authority=domain_authority,
            content_quality=content_quality,
            recency_score=recency_score,
            relevance_score=relevance_score,
            uniqueness_score=uniqueness_score,
            content_length=content_length,
            key_topics=key_topics,
            cited_sources=cited_sources,
            publication_date=publication_date,
            source_type=source_type,
            bias_rating=bias_rating,
            fact_check_rating=fact_check_rating
        )

    async def _calculate_domain_authority(self, url: str) -> float:
        """Calculate domain authority score"""
        # Extract domain
        domain = re.sub(r'https?://(?:www\.)?', '', url).split('/')[0]

        # Authority indicators
        high_authority_domains = {
            'nature.com', 'science.org', 'cell.com', 'nejm.org', 'bmj.com',
            'jama.com', 'thelancet.com', 'ieee.org', 'acm.org', 'springer.com',
            'sciencedirect.com', 'pubmed.ncbi.nlm.nih.gov', 'arxiv.org',
            'reuters.com', 'apnews.com', 'bbc.com', 'npr.org', 'pbs.org',
            'who.int', 'cdc.gov', 'nih.gov', 'nasa.gov', 'un.org',
            'worldbank.org', 'imf.org', 'oecd.org', 'europa.eu'
        }

        medium_authority_domains = {
            'forbes.com', 'bloomberg.com', 'washingtonpost.com', 'nytimes.com',
            'wsj.com', 'economist.com', 'ft.com', 'guardian.com', 'telegraph.co.uk',
            'cnn.com', 'msnbc.com', 'foxnews.com', 'cbsnews.com', 'abcnews.go.com',
            'technologyreview.com', 'wired.com', 'techcrunch.com', 'venturebeat.com'
        }

        if domain in high_authority_domains:
            return 0.9
        elif domain in medium_authority_domains:
            return 0.7
        elif any(ending in domain for ending in ['.edu', '.gov', '.org']):
            return 0.8
        elif any(ending in domain for ending in ['.com', '.net', '.co']):
            return 0.5
        else:
            return 0.3

    async def _assess_content_quality(self, source: Dict[str, Any]) -> float:
        """Assess content quality based on various factors"""
        content = source.get('content', '')
        title = source.get('title', '')

        if not content:
            return 0.0

        quality_score = 0.0

        # Length appropriateness (not too short, not too long)
        if len(content) > 500 and len(content) < 50000:
            quality_score += 0.2
        elif len(content) > 1000:
            quality_score += 0.15

        # Structure indicators
        if '##' in content or '###' in content:  # Has headers
            quality_score += 0.1
        if '###' in content:  # Has sub-headers
            quality_score += 0.05

        # Data and evidence indicators
        if any(indicator in content.lower() for indicator in
               ['study', 'research', 'analysis', 'data', 'statistics', 'survey']):
            quality_score += 0.15

        # Citation indicators
        if any(indicator in content.lower() for indicator in
               ['according to', 'research shows', 'study found', 'source']):
            quality_score += 0.1

        # Quality writing indicators
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if 10 <= avg_sentence_length <= 25:  # Reasonable sentence length
            quality_score += 0.1

        # Author credibility indicators
        if source.get('author') and source.get('author_credentials'):
            quality_score += 0.1

        # Editorial standards indicators
        if source.get('publication_date'):
            quality_score += 0.05

        return min(quality_score, 1.0)

    async def _calculate_recency_score(self, source: Dict[str, Any]) -> float:
        """Calculate recency score based on publication date"""
        pub_date = self._parse_publication_date(source)

        if not pub_date:
            return 0.5  # Neutral score for unknown dates

        days_old = (datetime.now() - pub_date).days

        # Recency scoring based on topic requirements
        if days_old <= 30:  # Very recent
            return 1.0
        elif days_old <= 90:  # Recent
            return 0.8
        elif days_old <= 365:  # Within year
            return 0.6
        elif days_old <= 1825:  # Within 5 years
            return 0.4
        else:  # Older than 5 years
            return 0.2

    async def _assess_relevance(self, source: Dict[str, Any], topic: str) -> float:
        """Assess relevance of source to research topic"""
        content = source.get('content', '').lower()
        title = source.get('title', '').lower()
        topic_lower = topic.lower()

        # Extract topic keywords
        topic_keywords = topic_lower.split()

        relevance_score = 0.0
        total_checks = 0

        # Title relevance (most important)
        title_matches = sum(1 for keyword in topic_keywords if keyword in title)
        if title_matches > 0:
            relevance_score += min(title_matches / len(topic_keywords), 1.0) * 0.4
        total_checks += 0.4

        # Content relevance
        content_matches = sum(1 for keyword in topic_keywords if keyword in content)
        if content_matches > 0:
            # Normalize by content length
            normalized_matches = min(content_matches / 10, 1.0)
            relevance_score += normalized_matches * 0.4
        total_checks += 0.4

        # Semantic relevance (related terms)
        related_terms = self._get_related_terms(topic)
        related_matches = sum(1 for term in related_terms if term in content)
        if related_matches > 0:
            relevance_score += min(related_matches / 5, 1.0) * 0.2
        total_checks += 0.2

        return relevance_score / total_checks if total_checks > 0 else 0.0

    async def _calculate_uniqueness(self, source: Dict[str, Any]) -> float:
        """Calculate uniqueness score based on content distinctiveness"""
        content = source.get('content', '')

        if not content:
            return 0.0

        # Uniqueness indicators
        uniqueness_score = 0.0

        # Unique data/statistics
        if any(indicator in content.lower() for indicator in
               ['original research', 'exclusive data', 'proprietary study']):
            uniqueness_score += 0.3

        # Unique perspective
        if any(indicator in content.lower() for indicator in
               ['unique perspective', 'novel approach', 'innovative']):
            uniqueness_score += 0.2

        # Primary source indicators
        if any(indicator in content.lower() for indicator in
               ['interview with', 'survey conducted', 'research team']):
            uniqueness_score += 0.2

        # Depth of analysis (beyond surface-level)
        if len(content) > 2000:
            uniqueness_score += 0.1

        # Citations to unique sources
        if source.get('cited_sources'):
            uniqueness_score += 0.2

        return min(uniqueness_score, 1.0)

    def _extract_key_topics(self, source: Dict[str, Any]) -> List[str]:
        """Extract key topics from source content"""
        content = source.get('content', '').lower()
        title = source.get('title', '').lower()
        full_text = f"{title} {content}"

        # Simple keyword extraction (could be enhanced with NLP)
        common_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', full_text)
        word_freq = Counter(word for word in words if word not in common_stopwords)

        # Get top 10 most frequent words
        key_topics = [word for word, count in word_freq.most_common(10)]

        return key_topics

    def _parse_publication_date(self, source: Dict[str, Any]) -> Optional[datetime]:
        """Parse publication date from source"""
        date_str = source.get('publication_date') or source.get('date')

        if not date_str:
            return None

        # Try different date formats
        date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y',
            '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S',
            '%B %d, %Y', '%d %B %Y'
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    def _determine_source_type(self, source: Dict[str, Any]) -> str:
        """Determine the type of source"""
        url = source.get('url', '').lower()
        title = source.get('title', '').lower()

        # Check for academic sources
        if any(indicator in url for indicator in ['.edu', 'arxiv.org', 'scholar.google.com']):
            return 'academic'

        # Check for official sources
        if any(indicator in url for indicator in ['.gov', '.org', 'who.int', 'un.org']):
            return 'official'

        # Check for news sources
        if any(indicator in url for indicator in ['news', 'reuters', 'apnews', 'bbc.com']):
            return 'news'

        # Check for company sources
        if any(indicator in url for indicator in ['company', 'corp', 'inc.com']):
            return 'company'

        # Check for blog sources
        if any(indicator in url for indicator in ['blog', 'medium.com', 'substack.com']):
            return 'blog'

        # Default based on content characteristics
        if any(indicator in title for indicator in ['study', 'research', 'analysis']):
            return 'academic'
        elif any(indicator in title for indicator in ['news', 'report', 'breaking']):
            return 'news'
        else:
            return 'general'

    def _assess_bias(self, source: Dict[str, Any]) -> Optional[str]:
        """Assess potential bias in source"""
        url = source.get('url', '').lower()
        content = source.get('content', '').lower()

        # Known bias indicators (simplified)
        left_sources = ['cnn.com', 'msnbc.com', 'huffpost.com', 'vice.com']
        right_sources = ['foxnews.com', 'breitbart.com', 'dailycaller.com']
        center_sources = ['reuters.com', 'apnews.com', 'bbc.com', 'npr.org']

        domain = re.sub(r'https?://(?:www\.)?', '', url).split('/')[0]

        if domain in left_sources:
            return 'left'
        elif domain in right_sources:
            return 'right'
        elif domain in center_sources:
            return 'center'

        # Analyze content for bias indicators
        bias_words_left = ['progressive', 'liberal', 'inclusive']
        bias_words_right = ['conservative', 'traditional', 'values']

        left_count = sum(1 for word in bias_words_left if word in content)
        right_count = sum(1 for word in bias_words_right if word in content)

        if left_count > right_count * 2:
            return 'left'
        elif right_count > left_count * 2:
            return 'right'
        elif abs(left_count - right_count) <= 1:
            return 'center'

        return 'unknown'

    def _assess_fact_check(self, source: Dict[str, Any]) -> Optional[float]:
        """Assess fact-check rating if available"""
        # This would typically integrate with fact-checking APIs
        # For now, return None to indicate not available
        return None

    def _generate_source_id(self, url: str) -> str:
        """Generate unique source ID"""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def _get_related_terms(self, topic: str) -> List[str]:
        """Get related terms for topic (simplified)"""
        related_terms_map = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks'],
            'climate': ['climate change', 'global warming', 'carbon emissions', 'renewable energy'],
            'healthcare': ['medical', 'health', 'patient care', 'treatment', 'medicine'],
            'technology': ['tech', 'innovation', 'digital', 'software', 'hardware'],
            'economy': ['economic', 'financial', 'market', 'business', 'trade']
        }

        topic_lower = topic.lower()
        for key, terms in related_terms_map.items():
            if key in topic_lower:
                return terms

        return []

    async def _analyze_coverage(
        self,
        research_data: Dict[str, Any],
        topic: str,
        source_assessments: List[SourceAssessment]
    ) -> Dict[CoverageType, CoverageAnalysis]:
        """Analyze coverage across different dimensions"""
        coverage_assessments = {}

        for coverage_type in CoverageType:
            assessment = await self._analyze_coverage_type(
                coverage_type, research_data, topic, source_assessments
            )
            coverage_assessments[coverage_type] = assessment

        return coverage_assessments

    async def _analyze_coverage_type(
        self,
        coverage_type: CoverageType,
        research_data: Dict[str, Any],
        topic: str,
        source_assessments: List[SourceAssessment]
    ) -> CoverageAnalysis:
        """Analyze specific coverage type"""

        # Get aspects for this coverage type
        expected_aspects = self._get_expected_aspects(coverage_type, topic)

        # Analyze covered aspects
        covered_aspects, missing_aspects, weak_aspects = await self._analyze_aspect_coverage(
            expected_aspects, research_data, coverage_type
        )

        # Calculate coverage score
        coverage_score = await self._calculate_coverage_score(
            covered_aspects, weak_aspects, missing_aspects, expected_aspects
        )

        # Analyze evidence strength
        evidence_strength = await self._analyze_evidence_strength(
            research_data, coverage_type, source_assessments
        )

        # Analyze source diversity
        source_diversity = await self._analyze_source_diversity(
            source_assessments, coverage_type
        )

        # Analyze depth indicators
        depth_indicators = await self._analyze_depth_indicators(
            research_data, coverage_type
        )

        # Calculate confidence intervals
        confidence_intervals = await self._calculate_coverage_confidence_intervals(
            coverage_score, evidence_strength, source_diversity
        )

        # Identify coverage gaps
        coverage_gaps = await self._identify_coverage_gaps(
            missing_aspects, weak_aspects, evidence_strength
        )

        return CoverageAnalysis(
            coverage_type=coverage_type,
            coverage_score=coverage_score,
            covered_aspects=covered_aspects,
            missing_aspects=missing_aspects,
            weak_aspects=weak_aspects,
            evidence_strength=evidence_strength,
            source_diversity=source_diversity,
            depth_indicators=depth_indicators,
            confidence_intervals=confidence_intervals,
            coverage_gaps=coverage_gaps
        )

    def _get_expected_aspects(self, coverage_type: CoverageType, topic: str) -> List[str]:
        """Get expected aspects for coverage type and topic"""
        aspect_templates = {
            CoverageType.FACTUAL: [
                "basic facts", "key statistics", "definitions", "historical context",
                "current status", "main components", "quantitative data"
            ],
            CoverageType.TEMPORAL: [
                "historical development", "recent changes", "future projections",
                "timeline of events", "trend analysis", "time series data"
            ],
            CoverageType.GEOGRAPHICAL: [
                "regional variations", "global perspective", "local impact",
                "geographical distribution", "location-specific factors"
            ],
            CoverageType.COMPARATIVE: [
                "comparisons with alternatives", "relative performance",
                "advantages and disadvantages", "benchmarking", "similar cases"
            ],
            CoverageType.TECHNICAL: [
                "technical specifications", "implementation details",
                "technical challenges", "methodology", "technical requirements"
            ],
            CoverageType.PERSPECTIVE: [
                "expert opinions", "stakeholder views", "public opinion",
                "controversies", "debates", "different viewpoints"
            ],
            CoverageType.EVIDENCE: [
                "research studies", "empirical data", "scientific evidence",
                "case studies", "experimental results", "peer-reviewed sources"
            ],
            CoverageType.SOURCE_DIVERSITY: [
                "academic sources", "industry sources", "government sources",
                "media coverage", "independent analysis", "primary sources"
            ]
        }

        base_aspects = aspect_templates.get(coverage_type, [])

        # Add topic-specific aspects
        topic_specific = self._get_topic_specific_aspects(topic, coverage_type)

        return base_aspects + topic_specific

    def _get_topic_specific_aspects(self, topic: str, coverage_type: CoverageType) -> List[str]:
        """Get topic-specific aspects for coverage analysis"""
        topic_lower = topic.lower()

        if 'ai' in topic_lower or 'artificial intelligence' in topic_lower:
            if coverage_type == CoverageType.TECHNICAL:
                return ["machine learning models", "neural network architectures", "training data"]
            elif coverage_type == CoverageType.FACTUAL:
                return ["AI capabilities", "limitations", "current applications"]

        elif 'climate' in topic_lower or 'environment' in topic_lower:
            if coverage_type == CoverageType.FACTUAL:
                return ["carbon emissions data", "temperature trends", "climate models"]
            elif coverage_type == CoverageType.TEMPORAL:
                return ["historical climate data", "future projections", "trend analysis"]

        elif 'health' in topic_lower or 'medical' in topic_lower:
            if coverage_type == CoverageType.EVIDENCE:
                return ["clinical trials", "medical studies", "peer-reviewed research"]
            elif coverage_type == CoverageType.FACTUAL:
                return ["treatment effectiveness", "side effects", "patient outcomes"]

        return []

    async def _analyze_aspect_coverage(
        self,
        expected_aspects: List[str],
        research_data: Dict[str, Any],
        coverage_type: CoverageType
    ) -> Tuple[List[str], List[str], List[str]]:
        """Analyze which aspects are covered, missing, or weak"""
        # Combine all content for analysis
        all_content = self._extract_all_content(research_data)
        content_lower = all_content.lower()

        covered_aspects = []
        missing_aspects = []
        weak_aspects = []

        for aspect in expected_aspects:
            coverage_strength = await self._assess_aspect_coverage_strength(
                aspect, content_lower, coverage_type
            )

            if coverage_strength >= 0.7:
                covered_aspects.append(aspect)
            elif coverage_strength >= 0.4:
                weak_aspects.append(aspect)
            else:
                missing_aspects.append(aspect)

        return covered_aspects, missing_aspects, weak_aspects

    def _extract_all_content(self, research_data: Dict[str, Any]) -> str:
        """Extract all content from research data"""
        content_parts = []

        if 'content' in research_data:
            content_parts.append(research_data['content'])

        if 'research_results' in research_data:
            results = research_data['research_results']
            if isinstance(results, dict):
                if 'content' in results:
                    content_parts.append(results['content'])
                if 'summary' in results:
                    content_parts.append(results['summary'])
            elif isinstance(results, list):
                for result in results:
                    if isinstance(result, dict) and 'content' in result:
                        content_parts.append(result['content'])

        if 'sources' in research_data:
            for source in research_data['sources']:
                if isinstance(source, dict) and 'content' in source:
                    content_parts.append(source['content'])

        return ' '.join(content_parts)

    async def _assess_aspect_coverage_strength(
        self,
        aspect: str,
        content: str,
        coverage_type: CoverageType
    ) -> float:
        """Assess how well an aspect is covered in content"""
        aspect_keywords = aspect.split()
        coverage_indicators = {
            CoverageType.FACTUAL: [
                'data', 'statistics', 'facts', 'figures', 'numbers', 'percent',
                'study shows', 'research indicates', 'according to'
            ],
            CoverageType.TEMPORAL: [
                'history', 'timeline', 'past', 'future', 'trend', 'evolution',
                'development', 'progression', 'over time', 'since', 'until'
            ],
            CoverageType.GEOGRAPHICAL: [
                'region', 'country', 'location', 'area', 'place', 'geography',
                'global', 'local', 'international', 'worldwide', 'regional'
            ],
            CoverageType.COMPARATIVE: [
                'compare', 'comparison', 'versus', 'compared to', 'relative to',
                'advantage', 'disadvantage', 'pro', 'con', 'better', 'worse'
            ],
            CoverageType.TECHNICAL: [
                'technical', 'specification', 'implementation', 'methodology',
                'algorithm', 'process', 'procedure', 'technique', 'approach'
            ],
            CoverageType.PERSPECTIVE: [
                'opinion', 'viewpoint', 'perspective', 'expert', 'analyst',
                'believe', 'think', 'argue', 'suggest', 'recommend'
            ],
            CoverageType.EVIDENCE: [
                'evidence', 'research', 'study', 'analysis', 'data', 'results',
                'findings', 'conclusion', 'method', 'experiment'
            ],
            CoverageType.SOURCE_DIVERSITY: [
                'source', 'according to', 'reported', 'stated', 'mentioned',
                'reference', 'citation', 'quote', 'said', 'announced'
            ]
        }

        # Calculate keyword matches
        keyword_matches = sum(1 for keyword in aspect_keywords if keyword in content)
        keyword_score = min(keyword_matches / len(aspect_keywords), 1.0) * 0.4

        # Calculate coverage indicator matches
        indicators = coverage_indicators.get(coverage_type, [])
        indicator_matches = sum(1 for indicator in indicators if indicator in content)
        indicator_score = min(indicator_matches / 3, 1.0) * 0.3

        # Calculate context relevance
        context_score = await self._calculate_context_relevance(aspect, content) * 0.3

        return keyword_score + indicator_score + context_score

    async def _calculate_context_relevance(self, aspect: str, content: str) -> float:
        """Calculate contextual relevance of aspect in content"""
        # Look for aspect in meaningful contexts (not just random mentions)
        aspect_words = aspect.split()

        # Find sentences containing aspect words
        sentences = re.split(r'[.!?]+', content)
        relevant_sentences = []

        for sentence in sentences:
            if any(word in sentence.lower() for word in aspect_words):
                # Check if sentence has sufficient context length
                if len(sentence.split()) > 10:
                    relevant_sentences.append(sentence)

        if not relevant_sentences:
            return 0.0

        # Calculate relevance based on context quality
        context_score = min(len(relevant_sentences) / 3, 1.0)

        return context_score

    async def _calculate_coverage_score(
        self,
        covered_aspects: List[str],
        weak_aspects: List[str],
        missing_aspects: List[str],
        expected_aspects: List[str]
    ) -> float:
        """Calculate overall coverage score"""
        total_aspects = len(expected_aspects)
        if total_aspects == 0:
            return 0.0

        # Weight different coverage levels
        covered_weight = 1.0
        weak_weight = 0.5
        missing_weight = 0.0

        score = (
            len(covered_aspects) * covered_weight +
            len(weak_aspects) * weak_weight +
            len(missing_aspects) * missing_weight
        ) / total_aspects

        return score

    async def _analyze_evidence_strength(
        self,
        research_data: Dict[str, Any],
        coverage_type: CoverageType,
        source_assessments: List[SourceAssessment]
    ) -> float:
        """Analyze strength of evidence for coverage type"""
        # Extract sources relevant to this coverage type
        relevant_sources = [
            source for source in source_assessments
            if await self._is_source_relevant_to_coverage(source, coverage_type)
        ]

        if not relevant_sources:
            return 0.0

        # Calculate evidence strength based on source quality and relevance
        evidence_strengths = []

        for source in relevant_sources:
            strength = (
                source.overall_quality * 0.4 +
                source.relevance_score * 0.3 +
                source.uniqueness_score * 0.2 +
                source.domain_authority * 0.1
            )
            evidence_strengths.append(strength)

        # Use weighted average (more weight to higher quality sources)
        if evidence_strengths:
            weighted_strength = sum(s ** 1.5 for s in evidence_strengths) / sum(s ** 0.5 for s in evidence_strengths)
            return min(weighted_strength, 1.0)

        return 0.0

    async def _is_source_relevant_to_coverage(
        self,
        source: SourceAssessment,
        coverage_type: CoverageType
    ) -> bool:
        """Check if source is relevant to specific coverage type"""
        # Simple heuristic based on source type and topics
        if coverage_type == CoverageType.EVIDENCE:
            return source.source_type in ['academic', 'official', 'research']
        elif coverage_type == CoverageType.PERSPECTIVE:
            return source.source_type in ['news', 'blog', 'general']
        elif coverage_type == CoverageType.TECHNICAL:
            return any(tech in ' '.join(source.key_topics).lower()
                      for tech in ['technical', 'implementation', 'methodology'])

        return True

    async def _analyze_source_diversity(
        self,
        source_assessments: List[SourceAssessment],
        coverage_type: CoverageType
    ) -> float:
        """Analyze diversity of sources for coverage type"""
        if not source_assessments:
            return 0.0

        # Calculate diversity metrics

        # 1. Type diversity
        source_types = set(source.source_type for source in source_assessments)
        type_diversity = min(len(source_types) / 4, 1.0)  # Max 4 types for full score

        # 2. Domain diversity
        domains = set(re.sub(r'https?://(?:www\.)?', '', source.url).split('/')[0]
                     for source in source_assessments)
        domain_diversity = min(len(domains) / len(source_assessments), 1.0)

        # 3. Quality range diversity
        quality_scores = [source.overall_quality for source in source_assessments]
        if quality_scores:
            quality_range = max(quality_scores) - min(quality_scores)
            quality_diversity = min(quality_range, 1.0)
        else:
            quality_diversity = 0.0

        # 4. Temporal diversity
        dates = [source.publication_date for source in source_assessments if source.publication_date]
        if len(dates) > 1:
            date_range = (max(dates) - min(dates)).days
            temporal_diversity = min(date_range / 365, 1.0)  # Full score for 1+ year spread
        else:
            temporal_diversity = 0.0

        # Weighted average
        diversity_score = (
            type_diversity * 0.3 +
            domain_diversity * 0.3 +
            quality_diversity * 0.2 +
            temporal_diversity * 0.2
        )

        return diversity_score

    async def _analyze_depth_indicators(
        self,
        research_data: Dict[str, Any],
        coverage_type: CoverageType
    ) -> Dict[str, float]:
        """Analyze depth indicators for coverage type"""
        content = self._extract_all_content(research_data).lower()

        depth_indicators = {
            CoverageType.FACTUAL: {
                'quantitative_data': len(re.findall(r'\d+%|\d+\.\d+|\b\d+\b', content)) / 10,
                'specific_examples': content.count('example') / 5,
                'detailed_explanations': content.count('explanation') / 3,
                'comprehensive_coverage': len(content.split()) / 1000
            },
            CoverageType.TEMPORAL: {
                'timeline_completeness': content.count('year') / 10,
                'trend_analysis': content.count('trend') / 3,
                'historical_depth': content.count('history') / 5,
                'future_projections': content.count('future') / 3
            },
            CoverageType.COMPARATIVE: {
                'direct_comparisons': content.count('compared') / 3,
                'relative_analysis': content.count('relative') / 3,
                'advantages_disadvantages': (content.count('advantage') + content.count('disadvantage')) / 5,
                'benchmarking_data': content.count('benchmark') / 2
            },
            CoverageType.TECHNICAL: {
                'technical_details': content.count('technical') / 5,
                'methodology_explanation': content.count('method') / 3,
                'implementation_specifics': content.count('implement') / 3,
                'specification_details': content.count('specification') / 2
            },
            CoverageType.EVIDENCE: {
                'research_citations': content.count('study') / 5,
                'data_support': content.count('data') / 10,
                'analytical_depth': content.count('analysis') / 5,
                'empirical_evidence': content.count('evidence') / 3
            }
        }

        indicators = depth_indicators.get(coverage_type, {})

        # Normalize scores to 0-1 range
        normalized_indicators = {}
        for indicator, raw_score in indicators.items():
            normalized_indicators[indicator] = min(raw_score, 1.0)

        return normalized_indicators

    async def _calculate_coverage_confidence_intervals(
        self,
        coverage_score: float,
        evidence_strength: float,
        source_diversity: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for coverage metrics"""
        # Simple confidence interval calculation based on supporting evidence
        confidence_factor = (evidence_strength + source_diversity) / 2

        # Width of interval inversely proportional to confidence
        interval_width = (1 - confidence_factor) * 0.3  # Max 30% width

        lower_bound = max(0, coverage_score - interval_width)
        upper_bound = min(1, coverage_score + interval_width)

        return {
            'coverage_score': (lower_bound, upper_bound),
            'evidence_strength': (max(0, evidence_strength - 0.1), min(1, evidence_strength + 0.1)),
            'source_diversity': (max(0, source_diversity - 0.1), min(1, source_diversity + 0.1))
        }

    async def _identify_coverage_gaps(
        self,
        missing_aspects: List[str],
        weak_aspects: List[str],
        evidence_strength: float
    ) -> List[Dict[str, Any]]:
        """Identify and categorize coverage gaps"""
        gaps = []

        # Missing aspects are critical gaps
        for aspect in missing_aspects:
            gaps.append({
                'aspect': aspect,
                'type': 'missing',
                'severity': 'critical',
                'description': f"Critical missing coverage: {aspect}",
                'impact': 'high',
                'recommendation': f"Research needed to address: {aspect}"
            })

        # Weak aspects are high or medium priority gaps
        for aspect in weak_aspects:
            severity = 'high' if evidence_strength < 0.5 else 'medium'
            gaps.append({
                'aspect': aspect,
                'type': 'weak',
                'severity': severity,
                'description': f"Weak coverage: {aspect}",
                'impact': severity,
                'recommendation': f"Enhance coverage of: {aspect}"
            })

        # Evidence gaps
        if evidence_strength < 0.6:
            gaps.append({
                'aspect': 'evidence_quality',
                'type': 'evidence',
                'severity': 'high' if evidence_strength < 0.4 else 'medium',
                'description': f"Low evidence strength: {evidence_strength:.2f}",
                'impact': 'high',
                'recommendation': 'Strengthen evidence with higher quality sources'
            })

        return gaps

    async def _assess_quality(
        self,
        research_data: Dict[str, Any],
        source_assessments: List[SourceAssessment],
        coverage_assessments: Dict[CoverageType, CoverageAnalysis]
    ) -> QualityAssessment:
        """Assess overall quality of research corpus"""

        # Calculate dimension scores
        dimension_scores = await self._calculate_quality_dimensions(
            research_data, source_assessments, coverage_assessments
        )

        # Calculate overall score
        overall_score = await self._calculate_overall_quality_score(dimension_scores)

        # Identify quality issues and strengths
        quality_issues, quality_strengths = await self._analyze_quality_characteristics(
            dimension_scores, source_assessments, coverage_assessments
        )

        # Identify improvement areas
        improvement_areas = await self._identify_improvement_areas(
            quality_issues, dimension_scores, coverage_assessments
        )

        # Calculate confidence level
        confidence_level = await self._calculate_quality_confidence(
            source_assessments, coverage_assessments, overall_score
        )

        # Generate assessment metadata
        assessment_metadata = {
            'source_count': len(source_assessments),
            'average_source_quality': sum(s.overall_quality for s in source_assessments) / len(source_assessments) if source_assessments else 0,
            'coverage_types_analyzed': len(coverage_assessments),
            'analysis_timestamp': datetime.now().isoformat()
        }

        return QualityAssessment(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            quality_issues=quality_issues,
            quality_strengths=quality_strengths,
            improvement_areas=improvement_areas,
            confidence_level=confidence_level,
            assessment_metadata=assessment_metadata
        )

    async def _calculate_quality_dimensions(
        self,
        research_data: Dict[str, Any],
        source_assessments: List[SourceAssessment],
        coverage_assessments: Dict[CoverageType, CoverageAnalysis]
    ) -> Dict[QualityDimension, float]:
        """Calculate scores for each quality dimension"""
        dimension_scores = {}

        for dimension in QualityDimension:
            score = await self._calculate_dimension_score(
                dimension, research_data, source_assessments, coverage_assessments
            )
            dimension_scores[dimension] = score

        return dimension_scores

    async def _calculate_dimension_score(
        self,
        dimension: QualityDimension,
        research_data: Dict[str, Any],
        source_assessments: List[SourceAssessment],
        coverage_assessments: Dict[CoverageType, CoverageAnalysis]
    ) -> float:
        """Calculate score for specific quality dimension"""

        if dimension == QualityDimension.ACCURACY:
            return await self._calculate_accuracy_score(source_assessments)
        elif dimension == QualityDimension.COMPLETENESS:
            return await self._calculate_completeness_score(coverage_assessments)
        elif dimension == QualityDimension.RELEVANCE:
            return await self._calculate_relevance_score(source_assessments)
        elif dimension == QualityDimension.AUTHORITY:
            return await self._calculate_authority_score(source_assessments)
        elif dimension == QualityDimension.RECENCY:
            return await self._calculate_recency_score(source_assessments)
        elif dimension == QualityDimension.DEPTH:
            return await self._calculate_depth_score(coverage_assessments)
        elif dimension == QualityDimension.CLARITY:
            return await self._calculate_clarity_score(research_data)
        elif dimension == QualityDimension.OBJECTIVITY:
            return await self._calculate_objectivity_score(source_assessments)

        return 0.0

    async def _calculate_accuracy_score(self, source_assessments: List[SourceAssessment]) -> float:
        """Calculate accuracy score based on source quality"""
        if not source_assessments:
            return 0.0

        # Accuracy based on domain authority and content quality
        accuracy_scores = []

        for source in source_assessments:
            accuracy_score = (
                source.domain_authority * 0.6 +
                source.content_quality * 0.4
            )
            accuracy_scores.append(accuracy_score)

        return sum(accuracy_scores) / len(accuracy_scores)

    async def _calculate_completeness_score(
        self,
        coverage_assessments: Dict[CoverageType, CoverageAnalysis]
    ) -> float:
        """Calculate completeness score based on coverage"""
        if not coverage_assessments:
            return 0.0

        completeness_scores = [assessment.coverage_score for assessment in coverage_assessments.values()]
        return sum(completeness_scores) / len(completeness_scores)

    async def _calculate_relevance_score(self, source_assessments: List[SourceAssessment]) -> float:
        """Calculate relevance score based on source relevance"""
        if not source_assessments:
            return 0.0

        relevance_scores = [source.relevance_score for source in source_assessments]
        return sum(relevance_scores) / len(relevance_scores)

    async def _calculate_authority_score(self, source_assessments: List[SourceAssessment]) -> float:
        """Calculate authority score based on source authority"""
        if not source_assessments:
            return 0.0

        authority_scores = [source.domain_authority for source in source_assessments]
        return sum(authority_scores) / len(authority_scores)

    async def _calculate_recency_score(self, source_assessments: List[SourceAssessment]) -> float:
        """Calculate recency score based on source recency"""
        if not source_assessments:
            return 0.0

        recency_scores = [source.recency_score for source in source_assessments]
        return sum(recency_scores) / len(recency_scores)

    async def _calculate_depth_score(
        self,
        coverage_assessments: Dict[CoverageType, CoverageAnalysis]
    ) -> float:
        """Calculate depth score based on coverage depth indicators"""
        if not coverage_assessments:
            return 0.0

        depth_scores = []

        for assessment in coverage_assessments.values():
            # Average of depth indicators
            if assessment.depth_indicators:
                avg_depth = sum(assessment.depth_indicators.values()) / len(assessment.depth_indicators)
                depth_scores.append(avg_depth)

        return sum(depth_scores) / len(depth_scores) if depth_scores else 0.0

    async def _calculate_clarity_score(self, research_data: Dict[str, Any]) -> float:
        """Calculate clarity score based on content clarity"""
        content = self._extract_all_content(research_data)

        if not content:
            return 0.0

        clarity_score = 0.0

        # Structure clarity
        if '##' in content:  # Has headers
            clarity_score += 0.2
        if '###' in content:  # Has sub-headers
            clarity_score += 0.1

        # Sentence length clarity
        sentences = content.split('.')
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 10 <= avg_sentence_length <= 25:  # Good readability
                clarity_score += 0.3
            elif 5 <= avg_sentence_length <= 35:
                clarity_score += 0.2

        # Paragraph structure
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 3:  # Has structure
            clarity_score += 0.2

        # Logical flow indicators
        flow_indicators = ['therefore', 'however', 'consequently', 'furthermore', 'moreover']
        flow_count = sum(1 for indicator in flow_indicators if indicator in content.lower())
        if flow_count > 0:
            clarity_score += min(flow_count / 5, 0.2)

        return min(clarity_score, 1.0)

    async def _calculate_objectivity_score(self, source_assessments: List[SourceAssessment]) -> float:
        """Calculate objectivity score based on source bias"""
        if not source_assessments:
            return 0.0

        objectivity_scores = []

        for source in source_assessments:
            if source.bias_rating == 'center':
                objectivity_scores.append(1.0)
            elif source.bias_rating == 'unknown':
                objectivity_scores.append(0.7)
            else:  # left or right
                objectivity_scores.append(0.5)

        return sum(objectivity_scores) / len(objectivity_scores)

    async def _calculate_overall_quality_score(
        self,
        dimension_scores: Dict[QualityDimension, float]
    ) -> float:
        """Calculate overall quality score from dimension scores"""
        overall_score = 0.0

        for dimension, score in dimension_scores.items():
            weight = self._quality_weights.get(dimension, 0.1)
            overall_score += score * weight

        return min(overall_score, 1.0)

    async def _analyze_quality_characteristics(
        self,
        dimension_scores: Dict[QualityDimension, float],
        source_assessments: List[SourceAssessment],
        coverage_assessments: Dict[CoverageType, CoverageAnalysis]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Analyze quality issues and strengths"""
        quality_issues = []
        quality_strengths = []

        # Analyze dimension scores
        for dimension, score in dimension_scores.items():
            if score < 0.5:
                quality_issues.append({
                    'dimension': dimension.value,
                    'issue': f'Low {dimension.value} score',
                    'score': score,
                    'severity': 'high' if score < 0.3 else 'medium',
                    'recommendation': f'Improve {dimension.value} through better sources or analysis'
                })
            elif score >= 0.8:
                quality_strengths.append(f'High {dimension.value} quality')

        # Analyze source issues
        low_quality_sources = [s for s in source_assessments if s.overall_quality < 0.4]
        if low_quality_sources:
            quality_issues.append({
                'dimension': 'source_quality',
                'issue': f'{len(low_quality_sources)} low quality sources',
                'severity': 'medium',
                'recommendation': 'Replace with higher quality sources'
            })

        # Analyze coverage issues
        poor_coverage = [ctype for ctype, assessment in coverage_assessments.items()
                        if assessment.coverage_score < 0.5]
        if poor_coverage:
            quality_issues.append({
                'dimension': 'coverage',
                'issue': f'Poor coverage in {", ".join(ctype.value for ctype in poor_coverage)}',
                'severity': 'high',
                'recommendation': 'Enhance research in these coverage areas'
            })

        return quality_issues, quality_strengths

    async def _identify_improvement_areas(
        self,
        quality_issues: List[Dict[str, Any]],
        dimension_scores: Dict[QualityDimension, float],
        coverage_assessments: Dict[CoverageType, CoverageAnalysis]
    ) -> List[str]:
        """Identify areas for improvement"""
        improvement_areas = []

        # From quality issues
        for issue in quality_issues:
            if issue['severity'] in ['high', 'critical']:
                improvement_areas.append(f"Address {issue['dimension']} issues")

        # From low dimension scores
        low_dimensions = [dim.value for dim, score in dimension_scores.items() if score < 0.6]
        if low_dimensions:
            improvement_areas.append(f"Improve {', '.join(low_dimensions)}")

        # From coverage gaps
        gap_areas = [ctype.value for ctype, assessment in coverage_assessments.items()
                    if assessment.has_significant_gaps]
        if gap_areas:
            improvement_areas.append(f"Address {', '.join(gap_areas)} coverage gaps")

        return list(set(improvement_areas))  # Remove duplicates

    async def _calculate_quality_confidence(
        self,
        source_assessments: List[SourceAssessment],
        coverage_assessments: Dict[CoverageType, CoverageAnalysis],
        overall_score: float
    ) -> float:
        """Calculate confidence level in quality assessment"""
        if not source_assessments or not coverage_assessments:
            return 0.0

        # Confidence factors
        source_confidence = min(len(source_assessments) / 5, 1.0)  # More sources = more confidence
        coverage_confidence = len(coverage_assessments) / len(CoverageType)  # More coverage types analyzed
        score_consistency = 1.0 - (max(assessment.coverage_score for assessment in coverage_assessments.values()) -
                                  min(assessment.coverage_score for assessment in coverage_assessments.values()))

        # Weighted average
        confidence = (
            source_confidence * 0.4 +
            coverage_confidence * 0.3 +
            score_consistency * 0.3
        )

        return min(confidence, 1.0)

    async def _analyze_gaps(
        self,
        research_data: Dict[str, Any],
        coverage_assessments: Dict[CoverageType, CoverageAnalysis],
        quality_assessment: QualityAssessment
    ) -> List[Dict[str, Any]]:
        """Comprehensive gap analysis"""
        gaps = []

        # Coverage gaps
        for coverage_type, assessment in coverage_assessments.items():
            if assessment.coverage_gaps:
                for gap in assessment.coverage_gaps:
                    gap['coverage_type'] = coverage_type.value
                    gap['category'] = 'coverage'
                    gaps.append(gap)

        # Quality gaps
        for issue in quality_assessment.quality_issues:
            if issue['severity'] in ['high', 'critical']:
                gaps.append({
                    'aspect': issue['dimension'],
                    'type': 'quality',
                    'severity': issue['severity'],
                    'description': issue['issue'],
                    'impact': issue['severity'],
                    'recommendation': issue['recommendation'],
                    'category': 'quality'
                })

        # Source gaps
        if quality_assessment.assessment_metadata['source_count'] < 3:
            gaps.append({
                'aspect': 'source_quantity',
                'type': 'source',
                'severity': 'high',
                'description': f"Insufficient sources: {quality_assessment.assessment_metadata['source_count']}",
                'impact': 'high',
                'recommendation': 'Add more diverse, high-quality sources',
                'category': 'source'
            })

        return gaps

    async def _analyze_source_patterns(self, source_assessments: List[SourceAssessment]) -> Dict[str, Any]:
        """Analyze patterns in sources"""
        if not source_assessments:
            return {}

        # Source type distribution
        type_distribution = Counter(source.source_type for source in source_assessments)

        # Quality distribution
        quality_ranges = {'high': 0, 'medium': 0, 'low': 0}
        for source in source_assessments:
            if source.overall_quality >= 0.8:
                quality_ranges['high'] += 1
            elif source.overall_quality >= 0.5:
                quality_ranges['medium'] += 1
            else:
                quality_ranges['low'] += 1

        # Recency distribution
        recency_ranges = {'recent': 0, 'moderate': 0, 'old': 0}
        for source in source_assessments:
            if source.recency_score >= 0.8:
                recency_ranges['recent'] += 1
            elif source.recency_score >= 0.5:
                recency_ranges['moderate'] += 1
            else:
                recency_ranges['old'] += 1

        # Bias distribution
        bias_distribution = Counter(source.bias_rating for source in source_assessments if source.bias_rating)

        return {
            'source_count': len(source_assessments),
            'type_distribution': dict(type_distribution),
            'quality_distribution': quality_ranges,
            'recency_distribution': recency_ranges,
            'bias_distribution': dict(bias_distribution),
            'average_quality': sum(s.overall_quality for s in source_assessments) / len(source_assessments),
            'domain_diversity': len(set(re.sub(r'https?://(?:www\.)?', '', s.url).split('/')[0] for s in source_assessments)),
            'unique_topics': len(set(topic for source in source_assessments for topic in source.key_topics))
        }

    async def _calculate_sufficiency(
        self,
        coverage_assessments: Dict[CoverageType, CoverageAnalysis],
        quality_assessment: QualityAssessment,
        source_analysis: Dict[str, Any],
        research_purpose: str
    ) -> Tuple[float, SufficiencyLevel]:
        """Calculate overall sufficiency score and level"""

        # Base coverage score
        coverage_score = sum(assessment.coverage_score for assessment in coverage_assessments.values()) / len(coverage_assessments)

        # Quality score
        quality_score = quality_assessment.overall_score

        # Source adequacy score
        source_adequacy = await self._calculate_source_adequacy(source_analysis, research_purpose)

        # Weighted combination
        sufficiency_score = (
            coverage_score * 0.4 +
            quality_score * 0.4 +
            source_adequacy * 0.2
        )

        # Determine sufficiency level
        if sufficiency_score >= 0.9:
            level = SufficiencyLevel.EXEMPLARY
        elif sufficiency_score >= 0.8:
            level = SufficiencyLevel.COMPREHENSIVE
        elif sufficiency_score >= 0.7:
            level = SufficiencyLevel.GOOD
        elif sufficiency_score >= 0.6:
            level = SufficiencyLevel.ADEQUATE
        elif sufficiency_score >= 0.4:
            level = SufficiencyLevel.MINIMAL
        else:
            level = SufficiencyLevel.INSUFFICIENT

        return sufficiency_score, level

    async def _calculate_source_adequacy(self, source_analysis: Dict[str, Any], research_purpose: str) -> float:
        """Calculate source adequacy score"""
        if not source_analysis:
            return 0.0

        adequacy_score = 0.0

        # Source count adequacy
        source_count = source_analysis.get('source_count', 0)
        if source_count >= 10:
            adequacy_score += 0.3
        elif source_count >= 5:
            adequacy_score += 0.2
        elif source_count >= 3:
            adequacy_score += 0.1

        # Quality distribution
        quality_dist = source_analysis.get('quality_distribution', {})
        high_quality_ratio = quality_dist.get('high', 0) / max(source_count, 1)
        adequacy_score += high_quality_ratio * 0.3

        # Type diversity
        type_dist = source_analysis.get('type_distribution', {})
        type_diversity = len(type_dist) / 4  # Normalize by ideal diversity
        adequacy_score += min(type_diversity, 1.0) * 0.2

        # Domain diversity
        domain_diversity = source_analysis.get('domain_diversity', 0)
        adequacy_score += min(domain_diversity / source_count, 1.0) * 0.2

        return min(adequacy_score, 1.0)

    async def _generate_recommendations(
        self,
        coverage_assessments: Dict[CoverageType, CoverageAnalysis],
        quality_assessment: QualityAssessment,
        gap_analysis: List[Dict[str, Any]],
        sufficiency_level: SufficiencyLevel
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Coverage recommendations
        critical_coverage_gaps = [
            f"Address critical {ctype.value} coverage gaps"
            for ctype, assessment in coverage_assessments.items()
            if assessment.has_significant_gaps
        ]
        recommendations.extend(critical_coverage_gaps)

        # Quality recommendations
        high_priority_quality_issues = [
            f"Improve {issue['dimension']} quality"
            for issue in quality_assessment.quality_issues
            if issue['severity'] in ['high', 'critical']
        ]
        recommendations.extend(high_priority_quality_issues)

        # Gap recommendations
        critical_gaps = [gap['recommendation'] for gap in gap_analysis if gap['severity'] == 'critical']
        recommendations.extend(critical_gaps)

        # General recommendations based on sufficiency level
        if sufficiency_level in [SufficiencyLevel.INSUFFICIENT, SufficiencyLevel.MINIMAL]:
            recommendations.append("Conduct comprehensive additional research")
            recommendations.append("Expand source diversity and quality")
        elif sufficiency_level == SufficiencyLevel.ADEQUATE:
            recommendations.append("Enhance research in identified weak areas")
        elif sufficiency_level == SufficiencyLevel.GOOD:
            recommendations.append("Fine-tune coverage of remaining gaps")

        # Remove duplicates and limit to top recommendations
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:8]  # Top 8 recommendations

    async def _calculate_confidence_intervals(
        self,
        coverage_assessments: Dict[CoverageType, CoverageAnalysis],
        quality_assessment: QualityAssessment,
        source_assessments: List[SourceAssessment]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics"""
        confidence_intervals = {}

        # Overall quality confidence interval
        quality_confidence = quality_assessment.confidence_level
        quality_range = (1 - quality_confidence) * 0.2
        quality_lower = max(0, quality_assessment.overall_score - quality_range)
        quality_upper = min(1, quality_assessment.overall_score + quality_range)
        confidence_intervals['overall_quality'] = (quality_lower, quality_upper)

        # Coverage confidence intervals
        for coverage_type, assessment in coverage_assessments.items():
            if assessment.coverage_score:
                # Use existing confidence intervals from coverage analysis
                if 'coverage_score' in assessment.confidence_intervals:
                    confidence_intervals[f'{coverage_type.value}_coverage'] = assessment.confidence_intervals['coverage_score']

        # Source quality confidence interval
        if source_assessments:
            source_scores = [s.overall_quality for s in source_assessments]
            avg_score = sum(source_scores) / len(source_scores)
            score_variance = sum((s - avg_score) ** 2 for s in source_scores) / len(source_scores)
            std_dev = math.sqrt(score_variance)

            # 95% confidence interval
            margin = 1.96 * (std_dev / math.sqrt(len(source_scores)))
            confidence_intervals['source_quality'] = (
                max(0, avg_score - margin),
                min(1, avg_score + margin)
            )

        return confidence_intervals

    def _generate_corpus_metadata(
        self,
        research_data: Dict[str, Any],
        source_count: int
    ) -> Dict[str, Any]:
        """Generate metadata about the research corpus"""
        content = self._extract_all_content(research_data)

        return {
            'content_length': len(content),
            'word_count': len(content.split()),
            'source_count': source_count,
            'analysis_timestamp': datetime.now().isoformat(),
            'analyzer_version': '3.2.3',
            'content_hash': hashlib.md5(content.encode()).hexdigest()[:16]
        }


# Factory function for easy instantiation
def create_research_corpus_analyzer(config: Optional[Dict[str, Any]] = None) -> ResearchCorpusAnalyzer:
    """Create a configured research corpus analyzer"""
    return ResearchCorpusAnalyzer(config)


# Utility function for quick analysis
async def quick_corpus_analysis(
    research_data: Dict[str, Any],
    topic: str,
    research_purpose: str = "general_research"
) -> SufficiencyAssessment:
    """Quick corpus analysis with default configuration"""
    analyzer = create_research_corpus_analyzer()
    return await analyzer.analyze_research_corpus(research_data, topic, research_purpose)