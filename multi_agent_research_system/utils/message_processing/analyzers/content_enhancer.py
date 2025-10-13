"""
Content Enhancement Engine - Advanced Message Analysis and Enhancement

This module provides sophisticated content analysis and enhancement capabilities
for improving message quality, readability, and effectiveness.

Key Features:
- Content quality analysis and improvement suggestions
- Text enhancement and formatting optimization
- Semantic analysis and entity extraction
- Language improvement and clarity enhancement
- Content structure optimization
- Context-aware enhancement strategies
"""

import asyncio
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from ..core.message_types import RichMessage, EnhancedMessageType


@dataclass
class EnhancementResult:
    """Result of content enhancement with details and recommendations."""

    enhanced_content: str
    enhancements_applied: List[str]
    quality_improvement: float
    recommendations: List[str]
    confidence: float
    processing_time: float


class ContentEnhancer:
    """Advanced content enhancement engine with multiple improvement strategies."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize content enhancer with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Enhancement strategies
        self.enhancement_strategies = {
            "text_clarity": self._enhance_text_clarity,
            "structure_optimization": self._optimize_structure,
            "formatting_improvement": self._improve_formatting,
            "content_expansion": self._expand_content,
            "quality_enhancement": self._enhance_quality,
            "language_improvement": self._improve_language,
            "semantic_enrichment": self._enrich_semantics,
            "accessibility_improvement": self._improve_accessibility
        }

        # Enhancement statistics
        self.enhancement_stats = {
            "total_enhanced": 0,
            "enhancements_by_type": {},
            "average_improvement": 0.0,
            "processing_time": 0.0
        }

        # Language patterns for improvement
        self.clarity_patterns = {
            "passive_voice": r'\b(is|are|was|were|be|been|being)\s+\w+\s+(by\s+\w+)?',
            "wordy_phrases": r'\b(in order to|due to the fact that|for the purpose of|as a result of)\b',
            "redundant_words": r'\b(very|really|quite|rather|somewhat|somehow)\s+\w+',
            "complex_sentences": r'.{75,}',  # Sentences over 75 characters
        }

        self.structure_patterns = {
            "headings": r'^(#{1,6})\s+(.+)$',
            "lists": r'^(\s*)([-*+]|\d+\.)\s+(.+)$',
            "code_blocks": r'^```(\w*)\n([\s\S]*?)```',
            "links": r'\[([^\]]+)\]\(([^)]+)\)',
            "emphasis": r'\*\*([^*]+)\*\*|__([^_]+)__|\*([^*]+)\*|_([^_]+)_'
        }

    async def enhance_message(self, message: RichMessage) -> List[str]:
        """Enhance message content using applicable strategies."""
        if not self.config.get("enable_enhancement", True):
            return []

        start_time = datetime.now()
        enhancements_applied = []

        try:
            # Determine applicable enhancement strategies
            applicable_strategies = self._get_applicable_strategies(message)

            # Apply enhancements
            original_content = message.content
            enhanced_content = original_content
            quality_improvement = 0.0

            for strategy_name in applicable_strategies:
                strategy_func = self.enhancement_strategies[strategy_name]

                try:
                    enhancement_result = await strategy_func(enhanced_content, message)
                    if enhancement_result:
                        enhanced_content = enhancement_result.get("content", enhanced_content)
                        strategy_improvement = enhancement_result.get("improvement", 0.0)
                        quality_improvement += strategy_improvement

                        if enhancement_result.get("applied", False):
                            enhancements_applied.append(strategy_name)
                            self.logger.debug(f"Applied {strategy_name} enhancement to message {message.id}")

                except Exception as e:
                    self.logger.warning(f"Enhancement strategy {strategy_name} failed: {str(e)}")

            # Update message if improvements were made
            if enhanced_content != original_content and enhancements_applied:
                message.update_content(enhanced_content, f"Applied enhancements: {', '.join(enhancements_applied)}")

                # Update quality metrics
                current_quality = message.metadata.quality_score or 0.5
                new_quality = min(1.0, current_quality + quality_improvement)
                message.set_quality_scores(quality=new_quality)

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_enhancement_stats(enhancements_applied, quality_improvement, processing_time)

            self.logger.debug(f"Enhanced message {message.id} with {len(enhancements_applied)} improvements")

            return enhancements_applied

        except Exception as e:
            self.logger.error(f"Failed to enhance message {message.id}: {str(e)}")
            return []

    def _get_applicable_strategies(self, message: RichMessage) -> List[str]:
        """Determine which enhancement strategies are applicable to the message."""
        applicable = []

        # Content-based strategy selection
        content = message.content.lower()

        # Text clarity - always applicable for text content
        if message.message_type in [EnhancedMessageType.TEXT, EnhancedMessageType.MARKDOWN]:
            applicable.append("text_clarity")

        # Structure optimization - applicable for structured content
        if any(pattern in message.content for pattern in ["#", "-", "*", "```", "[", "]"]):
            applicable.append("structure_optimization")

        # Formatting improvement - applicable for markdown and code
        if message.message_type in [EnhancedMessageType.MARKDOWN, EnhancedMessageType.CODE]:
            applicable.append("formatting_improvement")

        # Content expansion - for short messages that could be more detailed
        if len(message.content.split()) < 50 and message.message_type == EnhancedMessageType.TEXT:
            applicable.append("content_expansion")

        # Quality enhancement - always applicable
        applicable.append("quality_enhancement")

        # Language improvement - for text content
        if message.message_type in [EnhancedMessageType.TEXT, EnhancedMessageType.MARKDOWN]:
            applicable.append("language_improvement")

        # Semantic enrichment - for research and analysis content
        if message.message_type in [
            EnhancedMessageType.RESEARCH_RESULT,
            EnhancedMessageType.ANALYSIS_RESULT,
            EnhancedMessageType.CONTENT_SUMMARY
        ]:
            applicable.append("semantic_enrichment")

        # Accessibility improvement - always applicable for better user experience
        applicable.append("accessibility_improvement")

        # Filter by configuration
        disabled_strategies = self.config.get("disabled_strategies", [])
        applicable = [s for s in applicable if s not in disabled_strategies]

        return applicable

    async def _enhance_text_clarity(self, content: str, message: RichMessage) -> Dict[str, Any]:
        """Enhance text clarity by improving readability and reducing complexity."""
        enhanced_content = content
        improvements_made = []

        # Fix passive voice where appropriate
        passive_replacements = {
            "is implemented by": "implements",
            "was created by": "created",
            "are managed by": "manages",
            "were developed by": "developed"
        }

        for passive, active in passive_replacements.items():
            if passive in enhanced_content.lower():
                enhanced_content = enhanced_content.replace(passive, active)
                improvements_made.append("passive_voice_correction")

        # Replace wordy phrases with concise alternatives
        wordy_replacements = {
            "in order to": "to",
            "due to the fact that": "because",
            "for the purpose of": "for",
            "as a result of": "due to",
            "in the event that": "if",
            "on the basis of": "based on",
            "with regard to": "about",
            "in relation to": "regarding"
        }

        replacements_made = 0
        for wordy, concise in wordy_replacements.items():
            if wordy in enhanced_content.lower():
                enhanced_content = enhanced_content.replace(wordy, concise)
                replacements_made += 1

        if replacements_made > 0:
            improvements_made.append("wordiness_reduction")

        # Break down very long sentences
        sentences = enhanced_content.split('. ')
        improved_sentences = []

        for sentence in sentences:
            if len(sentence) > 100 and ',' in sentence:
                # Try to break at conjunctions
                clauses = re.split(r',\s+(and|but|or|so|yet|for)\s+', sentence, flags=re.IGNORECASE)
                if len(clauses) > 1:
                    improved_sentences.extend(clauses)
                    improvements_made.append("sentence_breakdown")
                else:
                    improved_sentences.append(sentence)
            else:
                improved_sentences.append(sentence)

        enhanced_content = '. '.join(improved_sentences)

        # Calculate improvement score
        original_complexity = self._calculate_complexity_score(content)
        new_complexity = self._calculate_complexity_score(enhanced_content)
        improvement = max(0, (original_complexity - new_complexity) / original_complexity)

        return {
            "content": enhanced_content,
            "improvement": improvement,
            "applied": len(improvements_made) > 0,
            "changes": improvements_made
        }

    async def _optimize_structure(self, content: str, message: RichMessage) -> Dict[str, Any]:
        """Optimize content structure for better readability."""
        enhanced_content = content
        improvements_made = []

        # Ensure proper heading hierarchy
        lines = enhanced_content.split('\n')
        improved_lines = []

        for i, line in enumerate(lines):
            # Add missing spacing around headings
            if line.startswith('#'):
                if i > 0 and lines[i-1].strip() != '':
                    improved_lines.append('')  # Add blank line before heading
                improved_lines.append(line)
                if i < len(lines) - 1 and lines[i+1].strip() != '' and not lines[i+1].startswith('#'):
                    improved_lines.append('')  # Add blank line after heading
            else:
                improved_lines.append(line)

        enhanced_content = '\n'.join(improved_lines)

        # Improve list formatting
        # Ensure consistent list markers
        list_patterns = [
            (r'^(\s*)\*\s+(.+)', r'\1• \2'),  # * → •
            (r'^(\s*)-\s+(.+)', r'\1• \2'),  # - → •
            (r'^(\s*)\+\s+(.+)', r'\1• \2'),  # + → •
        ]

        for pattern, replacement in list_patterns:
            if re.search(pattern, enhanced_content, re.MULTILINE):
                enhanced_content = re.sub(pattern, replacement, enhanced_content, flags=re.MULTILINE)
                improvements_made.append("list_formatting")

        # Enhance code blocks with language hints
        code_blocks = re.findall(r'```\w*\n([\s\S]*?)```', enhanced_content)
        if code_blocks:
            improvements_made.append("code_block_enhancement")

        # Calculate improvement score
        structure_score_before = self._calculate_structure_score(content)
        structure_score_after = self._calculate_structure_score(enhanced_content)
        improvement = max(0, (structure_score_after - structure_score_before) / max(structure_score_before, 1))

        return {
            "content": enhanced_content,
            "improvement": improvement,
            "applied": len(improvements_made) > 0,
            "changes": improvements_made
        }

    async def _improve_formatting(self, content: str, message: RichMessage) -> Dict[str, Any]:
        """Improve formatting for better visual presentation."""
        enhanced_content = content
        improvements_made = []

        # Ensure proper link formatting
        # Fix bare URLs that should be markdown links
        url_pattern = r'(?<!\]\()https?://[^\s\)]+(?!\))'
        bare_urls = re.findall(url_pattern, enhanced_content)

        for url in bare_urls:
            # Extract domain for link text
            domain = re.sub(r'https?://([^/]+).*', r'\1', url)
            replacement = f"[{domain}]({url})"
            enhanced_content = enhanced_content.replace(url, replacement)
            improvements_made.append("link_formatting")

        # Improve emphasis formatting
        # Ensure consistent use of ** for bold
        enhanced_content = re.sub(r'__([^_]+)__', r'**\1**', enhanced_content)
        enhanced_content = re.sub(r'\*([^*\n]+)\*', r'*\1*', enhanced_content)

        # Add proper spacing around code elements
        enhanced_content = re.sub(r'(\w)`([^`]+)`(\w)', r'\1 `\2` \3', enhanced_content)

        # Improve table formatting if present
        if '|' in enhanced_content:
            enhanced_content = self._enhance_table_formatting(enhanced_content)
            improvements_made.append("table_formatting")

        # Calculate improvement score
        formatting_score_before = self._calculate_formatting_score(content)
        formatting_score_after = self._calculate_formatting_score(enhanced_content)
        improvement = max(0, (formatting_score_after - formatting_score_before) / max(formatting_score_before, 1))

        return {
            "content": enhanced_content,
            "improvement": improvement,
            "applied": len(improvements_made) > 0,
            "changes": improvements_made
        }

    async def _expand_content(self, content: str, message: RichMessage) -> Dict[str, Any]:
        """Expand content with additional relevant information."""
        enhanced_content = content
        improvements_made = []

        # Check if content is too brief
        word_count = len(content.split())
        if word_count < 20:
            # Add contextual expansion based on message type
            if message.message_type == EnhancedMessageType.RESEARCH_RESULT:
                enhanced_content = self._expand_research_content(enhanced_content, message)
                improvements_made.append("research_expansion")
            elif message.message_type == EnhancedMessageType.ANALYSIS_RESULT:
                enhanced_content = self._expand_analysis_content(enhanced_content, message)
                improvements_made.append("analysis_expansion")

        # Add summary or clarification if beneficial
        if len(enhanced_content.split()) > 100 and "summary:" not in enhanced_content.lower():
            summary = self._generate_content_summary(enhanced_content)
            if summary:
                enhanced_content = f"{summary}\n\n{enhanced_content}"
                improvements_made.append("summary_addition")

        # Calculate improvement score
        length_improvement = max(0, (len(enhanced_content) - len(content)) / max(len(content), 1))
        improvement = min(0.3, length_improvement)  # Cap at 30% improvement

        return {
            "content": enhanced_content,
            "improvement": improvement,
            "applied": len(improvements_made) > 0,
            "changes": improvements_made
        }

    async def _enhance_quality(self, content: str, message: RichMessage) -> Dict[str, Any]:
        """Enhance overall content quality through various improvements."""
        enhanced_content = content
        improvements_made = []

        # Remove duplicate lines
        lines = enhanced_content.split('\n')
        unique_lines = []
        seen_lines = set()

        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen_lines:
                unique_lines.append(line)
                seen_lines.add(line_stripped)
            elif not line_stripped:  # Keep blank lines
                unique_lines.append(line)

        if len(unique_lines) != len(lines):
            enhanced_content = '\n'.join(unique_lines)
            improvements_made.append("duplicate_removal")

        # Fix common spelling and grammar issues
        grammar_fixes = {
            r'\bi\.e\.\s': 'i.e., ',
            r'\be\.g\.\s': 'e.g., ',
            r'\s+': ' ',  # Multiple spaces
            r'\n\s*\n\s*\n': '\n\n',  # Multiple blank lines
        }

        for pattern, replacement in grammar_fixes.items():
            if re.search(pattern, enhanced_content):
                enhanced_content = re.sub(pattern, replacement, enhanced_content)
                improvements_made.append("grammar_fix")

        # Ensure proper punctuation at sentence endings
        enhanced_content = self._fix_sentence_endings(enhanced_content)
        improvements_made.append("punctuation_fix")

        # Calculate quality improvement
        quality_before = self._assess_content_quality(content)
        quality_after = self._assess_content_quality(enhanced_content)
        improvement = max(0, (quality_after - quality_before) / max(quality_before, 0.1))

        return {
            "content": enhanced_content,
            "improvement": improvement,
            "applied": len(improvements_made) > 0,
            "changes": improvements_made
        }

    async def _improve_language(self, content: str, message: RichMessage) -> Dict[str, Any]:
        """Improve language usage and vocabulary."""
        enhanced_content = content
        improvements_made = []

        # Replace weak words with stronger alternatives
        word_improvements = {
            "good": ["excellent", "outstanding", "superior", "effective"],
            "bad": ["poor", "inadequate", "suboptimal", "ineffective"],
            "big": ["large", "substantial", "significant", "considerable"],
            "small": ["minor", "limited", "modest", "minimal"],
            "nice": ["pleasant", "appealing", "satisfactory", "appropriate"],
            "get": ["obtain", "acquire", "receive", "achieve"],
            "do": ["perform", "execute", "accomplish", "implement"],
            "show": ["demonstrate", "illustrate", "indicate", "reveal"]
        }

        improvements_count = 0
        for weak_word, strong_alternatives in word_improvements.items():
            pattern = r'\b' + weak_word + r'\b'
            if re.search(pattern, enhanced_content, re.IGNORECASE):
                # Replace with contextually appropriate alternative
                replacement = strong_alternatives[improvements_count % len(strong_alternatives)]
                enhanced_content = re.sub(pattern, replacement, enhanced_content, flags=re.IGNORECASE, count=1)
                improvements_count += 1

        if improvements_count > 0:
            improvements_made.append("vocabulary_enhancement")

        # Improve sentence variety
        enhanced_content = self._improve_sentence_variety(enhanced_content)
        improvements_made.append("sentence_variety")

        # Calculate improvement score
        language_score_before = self._assess_language_quality(content)
        language_score_after = self._assess_language_quality(enhanced_content)
        improvement = max(0, (language_score_after - language_score_before) / max(language_score_before, 0.1))

        return {
            "content": enhanced_content,
            "improvement": improvement,
            "applied": len(improvements_made) > 0,
            "changes": improvements_made
        }

    async def _enrich_semantics(self, content: str, message: RichMessage) -> Dict[str, Any]:
        """Enrich content with semantic improvements and context."""
        enhanced_content = content
        improvements_made = []

        # Add semantic markers for important concepts
        if message.message_type in [EnhancedMessageType.RESEARCH_RESULT, EnhancedMessageType.ANALYSIS_RESULT]:
            # Highlight key findings
            enhanced_content = self._highlight_key_findings(enhanced_content)
            improvements_made.append("key_finding_highlight")

        # Add contextual information
        enhanced_content = self._add_contextual_information(enhanced_content, message)
        improvements_made.append("contextual_enrichment")

        # Improve semantic coherence
        enhanced_content = self._improve_semantic_coherence(enhanced_content)
        improvements_made.append("semantic_coherence")

        # Calculate improvement score
        semantic_score_before = self._assess_semantic_quality(content)
        semantic_score_after = self._assess_semantic_quality(enhanced_content)
        improvement = max(0, (semantic_score_after - semantic_score_before) / max(semantic_score_before, 0.1))

        return {
            "content": enhanced_content,
            "improvement": improvement,
            "applied": len(improvements_made) > 0,
            "changes": improvements_made
        }

    async def _improve_accessibility(self, content: str, message: RichMessage) -> Dict[str, Any]:
        """Improve content accessibility for better user experience."""
        enhanced_content = content
        improvements_made = []

        # Add alt text suggestions for images
        if '!' in enhanced_content:
            enhanced_content = self._suggest_alt_text(enhanced_content)
            improvements_made.append("alt_text_suggestions")

        # Improve heading structure for screen readers
        enhanced_content = self._improve_heading_structure(enhanced_content)
        improvements_made.append("heading_structure")

        # Add descriptive text for links
        enhanced_content = self._improve_link_descriptions(enhanced_content)
        improvements_made.append("link_descriptions")

        # Ensure proper contrast considerations (metadata)
        if message.metadata:
            message.metadata["accessibility_features"] = improvements_made

        # Calculate improvement score
        accessibility_score_before = self._assess_accessibility(content)
        accessibility_score_after = self._assess_accessibility(enhanced_content)
        improvement = max(0, (accessibility_score_after - accessibility_score_before) / max(accessibility_score_before, 0.1))

        return {
            "content": enhanced_content,
            "improvement": improvement,
            "applied": len(improvements_made) > 0,
            "changes": improvements_made
        }

    # Helper methods for specific enhancement tasks
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate text complexity score."""
        words = content.split()
        sentences = content.count('.') + content.count('!') + content.count('?')

        if sentences == 0:
            return 0.0

        avg_words_per_sentence = len(words) / sentences
        long_words = len([w for w in words if len(w) > 6])

        # Complexity based on sentence length and word complexity
        complexity = (avg_words_per_sentence / 20.0 + long_words / len(words)) / 2
        return min(1.0, complexity)

    def _calculate_structure_score(self, content: str) -> float:
        """Calculate structure quality score."""
        score = 0.0
        max_score = 5.0

        # Has headings
        if re.search(r'^#{1,6}\s+', content, re.MULTILINE):
            score += 1.0

        # Has proper spacing
        if '\n\n' in content:
            score += 1.0

        # Has lists
        if re.search(r'^(\s*[-*+]|\d+\.)\s+', content, re.MULTILINE):
            score += 1.0

        # Has code blocks
        if '```' in content:
            score += 1.0

        # Has links
        if '[' in content and '](' in content:
            score += 1.0

        return score

    def _calculate_formatting_score(self, content: str) -> float:
        """Calculate formatting quality score."""
        score = 0.0
        checks = [
            (r'\*\*[^*]+\*\*', "bold_formatting"),
            (r'\*[^*]+\*', "italic_formatting"),
            (r'\[([^\]]+)\]\(([^)]+)\)', "link_formatting"),
            (r'^#{1,6}\s+', "heading_formatting"),
            (r'^\s*[-*+]\s+', "list_formatting"),
        ]

        for pattern, check_name in checks:
            if re.search(pattern, content, re.MULTILINE):
                score += 1.0

        return score

    def _enhance_table_formatting(self, content: str) -> str:
        """Enhance table formatting for better readability."""
        lines = content.split('\n')
        enhanced_lines = []

        for line in lines:
            if '|' in line:
                # Ensure proper spacing around table cells
                cells = line.split('|')
                if len(cells) > 2:
                    # Remove empty cells at start/end
                    if cells[0] == '':
                        cells = cells[1:]
                    if cells[-1] == '':
                        cells = cells[:-1]

                    # Pad cells for better alignment
                    max_cell_length = max(len(cell.strip()) for cell in cells)
                    padded_cells = [f" {cell.strip():<{max_cell_length}} " for cell in cells]
                    enhanced_line = '|' + '|'.join(padded_cells) + '|'
                    enhanced_lines.append(enhanced_line)
                    continue

            enhanced_lines.append(line)

        return '\n'.join(enhanced_lines)

    def _expand_research_content(self, content: str, message: RichMessage) -> str:
        """Expand research content with additional context."""
        expansion_phrases = [
            "Based on comprehensive analysis, ",
            "The research indicates that ",
            "Further investigation reveals ",
            "Key findings suggest that ",
        ]

        # Add contextual introduction if content is very brief
        if len(content.split()) < 15:
            introduction = expansion_phrases[hash(message.id) % len(expansion_phrases)]
            return f"{introduction}{content}"

        return content

    def _expand_analysis_content(self, content: str, message: RichMessage) -> str:
        """Expand analysis content with additional insights."""
        analysis_prefixes = [
            "Upon detailed analysis, ",
            "The analysis demonstrates that ",
            "Critical examination reveals ",
            "Systematic analysis indicates ",
        ]

        if len(content.split()) < 15:
            prefix = analysis_prefixes[hash(message.id) % len(analysis_prefixes)]
            return f"{prefix}{content}"

        return content

    def _generate_content_summary(self, content: str) -> str:
        """Generate a brief summary of the content."""
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) > 3:
            # Take first and last sentences for summary
            summary = f"**Summary:** {sentences[0].strip()}. {sentences[-1].strip()}."
            return summary
        return ""

    def _assess_content_quality(self, content: str) -> float:
        """Assess overall content quality."""
        quality_factors = {
            "length": min(1.0, len(content.split()) / 50),  # Ideal length around 50 words
            "structure": self._calculate_structure_score(content) / 5.0,
            "formatting": self._calculate_formatting_score(content) / 5.0,
            "readability": 1.0 - self._calculate_complexity_score(content),  # Lower complexity is better
        }

        return sum(quality_factors.values()) / len(quality_factors)

    def _fix_sentence_endings(self, content: str) -> str:
        """Fix sentence endings and punctuation."""
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.endswith(('.', '!', '?', ':', ';')):
                # Add appropriate ending
                if line.endswith(('```', '```', '*', '**')):
                    # Don't add punctuation to code or formatting endings
                    pass
                else:
                    line += '.'
            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _assess_language_quality(self, content: str) -> float:
        """Assess language quality score."""
        words = content.split()
        if not words:
            return 0.0

        # Check for vocabulary diversity
        unique_words = set(word.lower().strip('.,!?;:') for word in words)
        vocabulary_diversity = len(unique_words) / len(words)

        # Check for appropriate sentence length
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        sentence_quality = 1.0 - min(1.0, abs(avg_sentence_length - 15) / 15)  # Ideal around 15 words

        # Check for professional language
        professional_indicators = ['analysis', 'research', 'implementation', 'strategy', 'optimization']
        professional_score = sum(1 for word in words if any(indicator in word.lower() for indicator in professional_indicators)) / len(words)

        return (vocabulary_diversity + sentence_quality + professional_score) / 3

    def _improve_sentence_variety(self, content: str) -> str:
        """Improve sentence variety and flow."""
        sentences = re.split(r'([.!?]+)', content)
        improved_sentences = []

        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i].strip()
                punctuation = sentences[i + 1]

                if sentence:
                    # Vary sentence beginnings
                    sentence_starters = ["Additionally,", "Furthermore,", "Moreover,", "In addition,"]
                    if i > 0 and len(sentence.split()) > 5:
                        starter = sentence_starters[(i // 2) % len(sentence_starters)]
                        sentence = f"{starter} {sentence.lower()}"

                    improved_sentences.append(sentence + punctuation)
                else:
                    improved_sentences.append(punctuation)
            else:
                improved_sentences.append(sentences[i])

        return ''.join(improved_sentences)

    def _assess_semantic_quality(self, content: str) -> float:
        """Assess semantic quality and coherence."""
        # Simple semantic quality assessment
        indicators = {
            "coherence": self._assess_coherence(content),
            "relevance": self._assess_relevance(content),
            "clarity": self._assess_clarity(content),
        }

        return sum(indicators.values()) / len(indicators)

    def _assess_coherence(self, content: str) -> float:
        """Assess content coherence."""
        # Simple coherence check based on transition words
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 'additionally']
        transition_count = sum(1 for word in transition_words if word in content.lower())
        sentences = content.count('.') + content.count('!') + content.count('?')

        if sentences == 0:
            return 0.0

        return min(1.0, transition_count / (sentences / 3))  # Ideal: 1 transition per 3 sentences

    def _assess_relevance(self, content: str) -> float:
        """Assess content relevance (placeholder implementation)."""
        # In a real implementation, this would compare against a topic or context
        return 0.8  # Default good relevance score

    def _assess_clarity(self, content: str) -> float:
        """Assess content clarity."""
        # Clarity based on simplicity and directness
        complexity = self._calculate_complexity_score(content)
        return 1.0 - complexity

    def _highlight_key_findings(self, content: str) -> str:
        """Highlight key findings in research content."""
        # Look for sentences that contain findings or results
        finding_indicators = ['found that', 'results show', 'analysis reveals', 'study indicates']
        lines = content.split('\n')
        highlighted_lines = []

        for line in lines:
            if any(indicator in line.lower() for indicator in finding_indicators):
                # Highlight finding sentences
                highlighted_line = f"🔍 **Key Finding:** {line}"
                highlighted_lines.append(highlighted_line)
            else:
                highlighted_lines.append(line)

        return '\n'.join(highlighted_lines)

    def _add_contextual_information(self, content: str, message: RichMessage) -> str:
        """Add contextual information to enhance understanding."""
        # Add context based on message metadata
        if message.metadata and message.metadata.get("session_id"):
            context_info = f"\n\n*Context: Part of research session {message.metadata['session_id']}*"
            return content + context_info

        return content

    def _improve_semantic_coherence(self, content: str) -> str:
        """Improve semantic coherence of content."""
        # Add transitional phrases where needed
        sentences = re.split(r'([.!?]+)', content)
        improved_sentences = []

        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i].strip()
                punctuation = sentences[i + 1]

                if sentence and i > 2:  # Not the first sentence
                    # Add transition if appropriate
                    if not any(trans in sentence.lower() for trans in ['however', 'therefore', 'moreover']):
                        transitions = ['Additionally,', 'Furthermore,', 'In addition,']
                        if len(sentence.split()) > 10:  # Only for longer sentences
                            transition = transitions[(i // 2) % len(transitions)]
                            sentence = f"{transition} {sentence.lower()}"

                improved_sentences.append(sentence + punctuation)
            else:
                improved_sentences.append(sentences[i])

        return ''.join(improved_sentences)

    def _assess_accessibility(self, content: str) -> float:
        """Assess content accessibility."""
        accessibility_features = {
            "headings": 1.0 if re.search(r'^#{1,6}\s+', content, re.MULTILINE) else 0.0,
            "links": 1.0 if '[' in content and '](' in content else 0.0,
            "structure": 1.0 if '\n\n' in content else 0.5,
            "lists": 1.0 if re.search(r'^\s*[-*+]\s+', content, re.MULTILINE) else 0.0,
        }

        return sum(accessibility_features.values()) / len(accessibility_features)

    def _suggest_alt_text(self, content: str) -> str:
        """Suggest alt text for images."""
        # Find images without alt text
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        images = re.findall(image_pattern, content)

        enhanced_content = content
        for alt_text, url in images:
            if not alt_text or alt_text.strip() == '':
                # Generate alt text suggestion
                filename = url.split('/')[-1].split('.')[0]
                suggestion = f"[TODO: Add descriptive alt text for {filename}]"
                enhanced_content = enhanced_content.replace(
                    f"![{alt_text}]({url})",
                    f"![{suggestion}]({url})"
                )

        return enhanced_content

    def _improve_heading_structure(self, content: str) -> str:
        """Improve heading structure for accessibility."""
        lines = content.split('\n')
        improved_lines = []
        heading_level = 0

        for line in lines:
            if line.startswith('#'):
                # Count heading level
                current_level = len(line) - len(line.lstrip('#'))
                # Ensure heading levels don't skip (e.g., from # to ###)
                if current_level > heading_level + 1:
                    # Add intermediate headings
                    for level in range(heading_level + 1, current_level):
                        intermediate_heading = '#' * level + ' Section'
                        improved_lines.append(intermediate_heading)
                heading_level = current_level

            improved_lines.append(line)

        return '\n'.join(improved_lines)

    def _improve_link_descriptions(self, content: str) -> str:
        """Improve link descriptions for better accessibility."""
        # Find links with poor descriptions
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, content)

        enhanced_content = content
        for text, url in links:
            # Improve generic link text
            if text.lower() in ['click here', 'here', 'link', 'read more']:
                # Extract meaningful text from URL
                domain = re.sub(r'https?://([^/]+).*', r'\1', url)
                improved_text = f"Visit {domain}"
                enhanced_content = enhanced_content.replace(
                    f"[{text}]({url})",
                    f"[{improved_text}]({url})"
                )

        return enhanced_content

    def _update_enhancement_stats(self, enhancements: List[str], improvement: float, processing_time: float):
        """Update enhancement statistics."""
        self.enhancement_stats["total_enhanced"] += 1
        self.enhancement_stats["processing_time"] += processing_time

        # Update by type
        for enhancement in enhancements:
            if enhancement not in self.enhancement_stats["enhancements_by_type"]:
                self.enhancement_stats["enhancements_by_type"][enhancement] = 0
            self.enhancement_stats["enhancements_by_type"][enhancement] += 1

        # Update average improvement
        current_avg = self.enhancement_stats["average_improvement"]
        count = self.enhancement_stats["total_enhanced"]
        self.enhancement_stats["average_improvement"] = ((current_avg * (count - 1)) + improvement) / count

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhancement statistics."""
        stats = self.enhancement_stats.copy()

        # Calculate additional metrics
        if stats["total_enhanced"] > 0:
            stats["average_processing_time"] = stats["processing_time"] / stats["total_enhanced"]
            stats["enhancement_rate"] = 1.0  # All processed messages get enhancement consideration
        else:
            stats["average_processing_time"] = 0.0
            stats["enhancement_rate"] = 0.0

        return stats

    def reset_stats(self):
        """Reset enhancement statistics."""
        self.enhancement_stats = {
            "total_enhanced": 0,
            "enhancements_by_type": {},
            "average_improvement": 0.0,
            "processing_time": 0.0
        }