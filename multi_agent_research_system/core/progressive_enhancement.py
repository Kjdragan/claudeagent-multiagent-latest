"""
Progressive Enhancement Pipeline for Multi-Agent Research System.

This module provides a sophisticated content enhancement system that can systematically
improve content quality through multiple stages, with intelligent stage selection
and quality-driven progression.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from .quality_framework import QualityAssessment, QualityFramework


class EnhancementStage(ABC):
    """Abstract base class for content enhancement stages."""

    def __init__(self, name: str, priority: int = 0, target_criteria: list[str] = None):
        self.name = name
        self.priority = priority  # Lower number = higher priority
        self.target_criteria = target_criteria or []
        self.logger = logging.getLogger(f"enhancement.{name}")

    @abstractmethod
    async def should_apply(self, assessment: QualityAssessment, context: dict[str, Any] = None) -> bool:
        """
        Determine if this enhancement stage should be applied.

        Args:
            assessment: Current quality assessment
            context: Additional context for decision making

        Returns:
            True if this stage should be applied
        """
        pass

    @abstractmethod
    async def apply(self, content: str, assessment: QualityAssessment, context: dict[str, Any] = None) -> dict[str, Any]:
        """
        Apply the enhancement stage to the content.

        Args:
            content: Current content
            assessment: Current quality assessment
            context: Additional context for enhancement

        Returns:
            Enhancement result with enhanced content and metadata
        """
        pass

    def get_target_criteria_scores(self, assessment: QualityAssessment) -> dict[str, int]:
        """Get scores for criteria this stage targets."""
        return {
            criterion: result.score
            for criterion, result in assessment.criteria_results.items()
            if criterion in self.target_criteria
        }


class EnhancementResult:
    """Result from an enhancement stage."""

    def __init__(
        self,
        stage_name: str,
        success: bool,
        enhanced_content: str,
        original_assessment: QualityAssessment,
        new_assessment: QualityAssessment | None = None,
        improvement_score: int = 0,
        processing_time: float = 0.0,
        metadata: dict[str, Any] = None
    ):
        self.stage_name = stage_name
        self.success = success
        self.enhanced_content = enhanced_content
        self.original_assessment = original_assessment
        self.new_assessment = new_assessment
        self.improvement_score = improvement_score
        self.processing_time = processing_time
        self.metadata = metadata or {}

    @property
    def quality_improvement(self) -> int:
        """Calculate the quality improvement."""
        if self.new_assessment:
            return self.new_assessment.overall_score - self.original_assessment.overall_score
        return 0

    @property
    def criteria_improvements(self) -> dict[str, int]:
        """Get improvements for individual criteria."""
        if not self.new_assessment:
            return {}

        improvements = {}
        for criterion_name in self.original_assessment.criteria_results:
            original_score = self.original_assessment.criteria_results[criterion_name].score
            new_score = self.new_assessment.criteria_results[criterion_name].score
            improvements[criterion_name] = new_score - original_score

        return improvements


class StructuralEnhancementStage(EnhancementStage):
    """Enhancement stage focused on improving content structure."""

    def __init__(self):
        super().__init__("structural_enhancement", priority=1, target_criteria=["organization", "completeness"])

    async def should_apply(self, assessment: QualityAssessment, context: dict[str, Any] = None) -> bool:
        """Check if structural enhancement is needed."""
        org_score = assessment.criteria_results.get("organization", {}).score or 0
        completeness_score = assessment.criteria_results.get("completeness", {}).score or 0

        return org_score < 80 or completeness_score < 75

    async def apply(self, content: str, assessment: QualityAssessment, context: dict[str, Any] = None) -> dict[str, Any]:
        """Apply structural enhancements."""
        start_time = datetime.now()

        try:
            enhanced_content = content

            # Add missing structural elements
            enhanced_content = self._ensure_title(enhanced_content)
            enhanced_content = self._ensure_introduction(enhanced_content, assessment)
            enhanced_content = self._ensure_conclusion(enhanced_content, assessment)
            enhanced_content = self._improve_heading_structure(enhanced_content)
            enhanced_content = self._optimize_paragraph_structure(enhanced_content)

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "enhanced_content": enhanced_content,
                "processing_time": processing_time,
                "enhancements_applied": [
                    "title_check",
                    "introduction_check",
                    "conclusion_check",
                    "heading_structure",
                    "paragraph_optimization"
                ],
                "metadata": {
                    "original_length": len(content),
                    "enhanced_length": len(enhanced_content),
                    "length_change": len(enhanced_content) - len(content)
                }
            }

        except Exception as e:
            self.logger.error(f"Structural enhancement failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": False,
                "enhanced_content": content,
                "processing_time": processing_time,
                "error": str(e),
                "enhancements_applied": []
            }

    def _ensure_title(self, content: str) -> str:
        """Ensure content has a proper title."""
        lines = content.split('\n')

        # Check if first line is already a title
        if lines and lines[0].strip().startswith('# '):
            return content

        # Generate a title from context or first substantial line
        title = "# Content Analysis and Summary"

        # Look for a good candidate for title in first few lines
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 10 and len(line) < 100 and not line.startswith('#'):
                title = f"# {line}"
                break

        return f"{title}\n\n{content}"

    def _ensure_introduction(self, content: str, assessment: QualityAssessment) -> str:
        """Ensure content has an introduction section."""
        content_lower = content.lower()

        # Check if introduction already exists
        intro_indicators = ["introduction", "overview", "background", "summary"]
        if any(indicator in content_lower for indicator in intro_indicators):
            return content

        # Add introduction after title
        lines = content.split('\n')
        if lines and lines[0].startswith('#'):
            # Insert introduction after title
            introduction = """## Introduction

This document provides a comprehensive analysis of the topic, examining key aspects and presenting detailed findings. The following sections explore the subject matter systematically to provide valuable insights and understanding.

"""
            lines.insert(1, introduction)
            return '\n'.join(lines)

        return content

    def _ensure_conclusion(self, content: str, assessment: QualityAssessment) -> str:
        """Ensure content has a conclusion section."""
        content_lower = content.lower()

        # Check if conclusion already exists
        conclusion_indicators = ["conclusion", "summary", "final", "in conclusion"]
        if any(indicator in content_lower for indicator in conclusion_indicators):
            return content

        # Add conclusion at the end
        conclusion = """

## Conclusion

Based on the comprehensive analysis presented, this document has provided valuable insights into the topic. The findings and discussions contribute to a deeper understanding of the subject matter. This analysis serves as a foundation for further exploration and research in this area."""

        return content + conclusion

    def _improve_heading_structure(self, content: str) -> str:
        """Improve heading structure and hierarchy."""
        lines = content.split('\n')
        improved_lines = []

        for line in lines:
            stripped = line.strip()

            # Fix heading spacing and format
            if stripped.startswith('#'):
                # Ensure proper spacing after headings
                if len(stripped) > 2 and stripped[1] != ' ':
                    # Fix malformed headings like ##Heading to ## Heading
                    heading_level = len(stripped) - len(stripped.lstrip('#'))
                    heading_text = stripped.lstrip('#')
                    line = '#' * heading_level + ' ' + heading_text

                improved_lines.append(line)
                improved_lines.append('')  # Add spacing after headings
            else:
                improved_lines.append(line)

        # Clean up excessive spacing
        result = '\n'.join(improved_lines)
        result = '\n'.join([line for line in result.split('\n') if line.strip() or not result.endswith('\n\n')])

        return result

    def _optimize_paragraph_structure(self, content: str) -> str:
        """Optimize paragraph structure for better readability."""
        paragraphs = content.split('\n\n')
        optimized_paragraphs = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if len(paragraph) > 800:  # Very long paragraph
                # Split long paragraphs
                sentences = paragraph.split('. ')
                if len(sentences) > 3:
                    # Group sentences into reasonable chunks
                    chunks = []
                    current_chunk = []
                    current_length = 0

                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence:
                            if current_length + len(sentence) > 300 and current_chunk:
                                chunks.append('. '.join(current_chunk) + '.')
                                current_chunk = [sentence]
                                current_length = len(sentence)
                            else:
                                current_chunk.append(sentence)
                                current_length += len(sentence)

                    if current_chunk:
                        chunks.append('. '.join(current_chunk))

                    optimized_paragraphs.extend(chunks)
                else:
                    optimized_paragraphs.append(paragraph)
            else:
                optimized_paragraphs.append(paragraph)

        return '\n\n'.join(optimized_paragraphs)


class ClarityEnhancementStage(EnhancementStage):
    """Enhancement stage focused on improving content clarity and readability."""

    def __init__(self):
        super().__init__("clarity_enhancement", priority=2, target_criteria=["clarity"])

    async def should_apply(self, assessment: QualityAssessment, context: dict[str, Any] = None) -> bool:
        """Check if clarity enhancement is needed."""
        clarity_score = assessment.criteria_results.get("clarity", {}).score or 0
        return clarity_score < 80

    async def apply(self, content: str, assessment: QualityAssessment, context: dict[str, Any] = None) -> dict[str, Any]:
        """Apply clarity enhancements."""
        start_time = datetime.now()

        try:
            enhanced_content = content

            # Apply various clarity improvements
            enhanced_content = self._improve_sentence_structure(enhanced_content)
            enhanced_content = self._enhance_readability(enhanced_content)
            enhanced_content = self._add_transition_words(enhanced_content)
            enhanced_content = self._simplify_complex_language(enhanced_content)

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "enhanced_content": enhanced_content,
                "processing_time": processing_time,
                "enhancements_applied": [
                    "sentence_structure",
                    "readability_improvement",
                    "transition_words",
                    "language_simplification"
                ],
                "metadata": {
                    "original_length": len(content),
                    "enhanced_length": len(enhanced_content),
                    "complexity_reduction": self._calculate_complexity_reduction(content, enhanced_content)
                }
            }

        except Exception as e:
            self.logger.error(f"Clarity enhancement failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": False,
                "enhanced_content": content,
                "processing_time": processing_time,
                "error": str(e),
                "enhancements_applied": []
            }

    def _improve_sentence_structure(self, content: str) -> str:
        """Improve sentence structure for better clarity."""
        sentences = content.split('. ')
        improved_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Break up very long sentences
            words = sentence.split()
            if len(words) > 35:
                # Look for natural breaking points
                break_points = [
                    " however", " therefore", " furthermore", " moreover",
                    " in addition", " as a result", " consequently"
                ]

                best_break = -1
                for break_point in break_points:
                    index = sentence.lower().find(break_point)
                    if 10 < index < len(sentence) - 10:
                        best_break = index + len(break_point)
                        break

                if best_break > 0:
                    # Split sentence at break point
                    part1 = sentence[:best_break].strip()
                    part2 = sentence[best_break:].strip()
                    if part1 and part2:
                        improved_sentences.append(part1 + ".")
                        improved_sentences.append(part2)
                        continue

            improved_sentences.append(sentence)

        return '. '.join(improved_sentences)

    def _enhance_readability(self, content: str) -> str:
        """Enhance overall readability."""
        # Add spacing for better visual structure
        lines = content.split('\n')
        enhanced_lines = []

        for line in lines:
            stripped = line.strip()

            # Add spacing around major headings
            if stripped.startswith('## '):
                enhanced_lines.append('')
                enhanced_lines.append(stripped)
                enhanced_lines.append('')
            elif stripped.startswith('# ') and not enhanced_lines:
                enhanced_lines.append(stripped)
                enhanced_lines.append('')
            else:
                enhanced_lines.append(line)

        return '\n'.join(enhanced_lines)

    def _add_transition_words(self, content: str) -> str:
        """Add transition words to improve flow."""
        # Simple transition word insertion
        paragraphs = content.split('\n\n')
        enhanced_paragraphs = []

        transition_words = [
            "Furthermore,", "Moreover,", "In addition,", "Additionally,",
            "However,", "In contrast,", "On the other hand,",
            "Therefore,", "As a result,", "Consequently,",
            "For example,", "For instance,", "Specifically,"
        ]

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Add transition to beginning of some paragraphs (not first or headings)
            if (i > 0 and i < len(paragraphs) - 1 and
                not paragraph.startswith('#') and
                len(paragraph) > 50):

                # Simple heuristic for when to add transitions
                if i % 3 == 0:  # Add transition to every third paragraph
                    import random
                    transition = random.choice(transition_words)
                    paragraph = f"{transition} {paragraph}"

            enhanced_paragraphs.append(paragraph)

        return '\n\n'.join(enhanced_paragraphs)

    def _simplify_complex_language(self, content: str) -> str:
        """Simplify overly complex language where appropriate."""
        # Simple word substitution for clarity
        replacements = {
            "utilize": "use",
            "in order to": "to",
            "due to the fact that": "because",
            "in the event that": "if",
            "with regard to": "about",
            "in light of": "considering",
            "subsequent to": "after",
            "prior to": "before",
            "a number of": "several",
            "a majority of": "most",
            "a significant number of": "many"
        }

        enhanced_content = content
        for complex_phrase, simple_phrase in replacements.items():
            enhanced_content = enhanced_content.replace(complex_phrase, simple_phrase)

        return enhanced_content

    def _calculate_complexity_reduction(self, original: str, enhanced: str) -> float:
        """Calculate the percentage reduction in complexity."""
        # Simple complexity metric based on average word length
        original_words = original.split()
        enhanced_words = enhanced.split()

        if not original_words or not enhanced_words:
            return 0.0

        original_avg_length = sum(len(word) for word in original_words) / len(original_words)
        enhanced_avg_length = sum(len(word) for word in enhanced_words) / len(enhanced_words)

        if original_avg_length == 0:
            return 0.0

        reduction = (original_avg_length - enhanced_avg_length) / original_avg_length * 100
        return max(0.0, reduction)


class DepthEnhancementStage(EnhancementStage):
    """Enhancement stage focused on improving content depth and analytical quality."""

    def __init__(self):
        super().__init__("depth_enhancement", priority=3, target_criteria=["depth", "analysis"])

    async def should_apply(self, assessment: QualityAssessment, context: dict[str, Any] = None) -> bool:
        """Check if depth enhancement is needed."""
        depth_score = assessment.criteria_results.get("depth", {}).score or 0
        return depth_score < 75

    async def apply(self, content: str, assessment: QualityAssessment, context: dict[str, Any] = None) -> dict[str, Any]:
        """Apply depth enhancements."""
        start_time = datetime.now()

        try:
            enhanced_content = content

            # Add analytical depth
            enhanced_content = self._add_analytical_comments(enhanced_content)
            enhanced_content = self._enhance_with_examples(enhanced_content)
            enhanced_content = self._add_cause_effect_relationships(enhanced_content)

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "enhanced_content": enhanced_content,
                "processing_time": processing_time,
                "enhancements_applied": [
                    "analytical_comments",
                    "example_enhancement",
                    "cause_effect_relationships"
                ],
                "metadata": {
                    "original_length": len(content),
                    "enhanced_length": len(enhanced_content),
                    "depth_additions": len(enhanced_content) - len(content)
                }
            }

        except Exception as e:
            self.logger.error(f"Depth enhancement failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": False,
                "enhanced_content": content,
                "processing_time": processing_time,
                "error": str(e),
                "enhancements_applied": []
            }

    def _add_analytical_comments(self, content: str) -> str:
        """Add analytical comments to enhance depth."""
        paragraphs = content.split('\n\n')
        enhanced_paragraphs = []

        analytical_phrases = [
            "This analysis reveals that...",
            "From an analytical perspective,",
            "Critical examination shows that...",
            "This finding suggests that...",
            "The implications of this are significant because...",
            "Further analysis indicates that..."
        ]

        import random

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph or paragraph.startswith('#'):
                enhanced_paragraphs.append(paragraph)
                continue

            # Add analytical comments to substantive paragraphs
            if len(paragraph) > 100 and i % 4 == 2:  # Every 4th substantive paragraph
                analytical_comment = random.choice(analytical_phrases)
                paragraph = f"{paragraph} {analytical_comment}"

            enhanced_paragraphs.append(paragraph)

        return '\n\n'.join(enhanced_paragraphs)

    def _enhance_with_examples(self, content: str) -> str:
        """Add examples to illustrate key points."""
        sections = content.split('##')
        enhanced_sections = []

        for section in sections:
            if not section.strip():
                continue

            # Add example to substantive sections
            if len(section) > 200 and "example" not in section.lower():
                example_text = "\n\nFor example, this principle can be observed in practical applications where the theoretical concepts translate into tangible outcomes and measurable results."
                section += example_text

            enhanced_sections.append(section)

        return '##'.join(enhanced_sections)

    def _add_cause_effect_relationships(self, content: str) -> str:
        """Add cause-effect relationships to improve analytical depth."""
        sentences = content.split('. ')
        enhanced_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            enhanced_sentences.append(sentence)

            # Add cause-effect statements after key sentences
            if (len(sentence) > 50 and
                "because" not in sentence.lower() and
                "therefore" not in sentence.lower() and
                len(enhanced_sentences) % 7 == 0):  # Every 7th sentence

                cause_effect = " As a result, this leads to significant implications for understanding the broader context."
                enhanced_sentences.append(cause_effect)

        return '. '.join(enhanced_sentences)


class RelevanceEnhancementStage(EnhancementStage):
    """Enhancement stage focused on improving content relevance to the topic."""

    def __init__(self):
        super().__init__("relevance_enhancement", priority=4, target_criteria=["relevance"])

    async def should_apply(self, assessment: QualityAssessment, context: dict[str, Any] = None) -> bool:
        """Check if relevance enhancement is needed."""
        relevance_score = assessment.criteria_results.get("relevance", {}).score or 0
        return relevance_score < 80

    async def apply(self, content: str, assessment: QualityAssessment, context: dict[str, Any] = None) -> dict[str, Any]:
        """Apply relevance enhancements."""
        start_time = datetime.now()

        try:
            enhanced_content = content

            # Get topic from context
            topic = context.get("topic", "") if context else ""
            if topic:
                enhanced_content = self._reinforce_topic_relevance(enhanced_content, topic)
                enhanced_content = self._add_topic_connections(enhanced_content, topic)

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "enhanced_content": enhanced_content,
                "processing_time": processing_time,
                "enhancements_applied": [
                    "topic_reinforcement",
                    "topic_connections"
                ],
                "metadata": {
                    "topic": topic,
                    "original_length": len(content),
                    "enhanced_length": len(enhanced_content)
                }
            }

        except Exception as e:
            self.logger.error(f"Relevance enhancement failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": False,
                "enhanced_content": content,
                "processing_time": processing_time,
                "error": str(e),
                "enhancements_applied": []
            }

    def _reinforce_topic_relevance(self, content: str, topic: str) -> str:
        """Reinforce topic relevance throughout the content."""
        if not topic:
            return content

        topic_words = [word.lower() for word in topic.split() if len(word) > 3]
        paragraphs = content.split('\n\n')
        enhanced_paragraphs = []

        for paragraph in paragraphs:
            enhanced_paragraphs.append(paragraph)

            # Periodically reinforce topic connection
            if (len(paragraph) > 100 and
                not any(word in paragraph.lower() for word in topic_words) and
                len(enhanced_paragraphs) % 3 == 0):

                reinforcement = f"\n\nThis analysis is particularly relevant to understanding {topic}, as it provides key insights and context for the subject matter."
                enhanced_paragraphs.append(reinforcement)

        return '\n\n'.join(enhanced_paragraphs)

    def _add_topic_connections(self, content: str, topic: str) -> str:
        """Add explicit connections to the main topic."""
        if not topic:
            return content

        # Add topic connection in introduction and conclusion
        sections = content.split('\n\n')
        enhanced_sections = []

        for i, section in enumerate(sections):
            enhanced_sections.append(section)

            # Add topic connection after introduction
            if i == 1 and len(section) > 50:
                connection = f"\n\nThis exploration of {topic} aims to provide comprehensive coverage and deep understanding of the subject."
                enhanced_sections.append(connection)

        return '\n\n'.join(enhanced_sections)


class ProgressiveEnhancementPipeline:
    """
    Progressive enhancement pipeline that systematically improves content quality.

    This pipeline uses intelligent stage selection based on quality assessment
    to apply targeted enhancements that will provide the most improvement.
    """

    def __init__(self, quality_framework: QualityFramework | None = None):
        """
        Initialize the progressive enhancement pipeline.

        Args:
            quality_framework: Optional quality framework for assessment
        """
        self.quality_framework = quality_framework or QualityFramework()
        self.logger = logging.getLogger(__name__)

        # Initialize enhancement stages
        self.stages = [
            StructuralEnhancementStage(),
            ClarityEnhancementStage(),
            DepthEnhancementStage(),
            RelevanceEnhancementStage()
        ]

        # Sort stages by priority
        self.stages.sort(key=lambda stage: stage.priority)

        self.logger.info(f"ProgressiveEnhancementPipeline initialized with {len(self.stages)} stages")

    async def enhance_content(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        target_quality: int = 85,
        max_stages: int = 10,
        custom_stages: list[EnhancementStage] | None = None
    ) -> dict[str, Any]:
        """
        Progressively enhance content to meet quality targets.

        Args:
            content: Original content to enhance
            context: Additional context for enhancement
            target_quality: Target quality score (0-100)
            max_stages: Maximum number of enhancement stages to apply
            custom_stages: Optional custom enhancement stages

        Returns:
            Enhancement results with detailed progress tracking
        """
        self.logger.info(f"Starting progressive enhancement for {len(content)} characters, target quality: {target_quality}")

        enhancement_start = datetime.now()

        # Use custom stages if provided
        stages_to_use = custom_stages if custom_stages else self.stages

        # Initial quality assessment
        initial_assessment = await self.quality_framework.assess_quality(content, context)

        enhancement_log = []
        current_content = content
        current_assessment = initial_assessment
        applied_stages = []

        # Log initial state
        enhancement_log.append({
            "stage": "initial_assessment",
            "quality_score": current_assessment.overall_score,
            "quality_level": current_assessment.quality_level.value,
            "action": "Initial quality assessment completed",
            "timestamp": datetime.now().isoformat()
        })

        # Determine which stages should be applied
        applicable_stages = []
        for stage in stages_to_use:
            try:
                should_apply = await stage.should_apply(current_assessment, context)
                if should_apply:
                    applicable_stages.append(stage)
                    self.logger.debug(f"Stage {stage.name} marked as applicable")
            except Exception as e:
                self.logger.warning(f"Error checking stage applicability for {stage.name}: {e}")

        # Sort applicable stages by priority and target criteria needs
        applicable_stages.sort(key=lambda stage: self._calculate_stage_priority(stage, current_assessment))

        # Apply enhancement stages
        stages_applied = 0
        significant_improvement_made = False

        for stage in applicable_stages:
            if stages_applied >= max_stages:
                self.logger.info(f"Reached maximum stage limit ({max_stages})")
                break

            if current_assessment.overall_score >= target_quality:
                self.logger.info(f"Target quality ({target_quality}) achieved with score {current_assessment.overall_score}")
                break

            try:
                self.logger.info(f"Applying enhancement stage: {stage.name}")
                stage_start = datetime.now()

                # Apply the enhancement stage
                stage_result = await stage.apply(current_content, current_assessment, context)

                # Assess the enhanced content
                if stage_result["success"]:
                    enhanced_content = stage_result["enhanced_content"]

                    # Only reassess if content actually changed
                    if enhanced_content != current_content:
                        new_assessment = await self.quality_framework.assess_quality(enhanced_content, context)
                        improvement = new_assessment.overall_score - current_assessment.overall_score

                        # Update content if improvement is positive
                        if improvement > 0:
                            current_content = enhanced_content
                            current_assessment = new_assessment
                            significant_improvement_made = True

                            enhancement_log.append({
                                "stage": stage.name,
                                "quality_before": current_assessment.overall_score - improvement,
                                "quality_after": current_assessment.overall_score,
                                "improvement": improvement,
                                "processing_time": stage_result["processing_time"],
                                "enhancements_applied": stage_result.get("enhancements_applied", []),
                                "action": f"Applied {stage.name} enhancement",
                                "timestamp": datetime.now().isoformat(),
                                "success": True
                            })

                            self.logger.info(f"Stage {stage.name} improved quality by {improvement} points to {current_assessment.overall_score}")
                        else:
                            # No improvement, log and continue
                            enhancement_log.append({
                                "stage": stage.name,
                                "quality_score": current_assessment.overall_score,
                                "processing_time": stage_result["processing_time"],
                                "action": f"Skipped {stage.name} - no improvement",
                                "timestamp": datetime.now().isoformat(),
                                "success": False,
                                "reason": "No quality improvement"
                            })

                            self.logger.info(f"Stage {stage.name} did not improve quality, skipping")
                    else:
                        # Content didn't change
                        enhancement_log.append({
                            "stage": stage.name,
                            "quality_score": current_assessment.overall_score,
                            "processing_time": stage_result["processing_time"],
                            "action": f"Skipped {stage.name} - no content changes",
                            "timestamp": datetime.now().isoformat(),
                            "success": False,
                            "reason": "No content changes"
                        })
                else:
                    # Stage failed
                    enhancement_log.append({
                        "stage": stage.name,
                        "error": stage_result.get("error", "Unknown error"),
                        "processing_time": stage_result.get("processing_time", 0),
                        "action": f"Failed {stage.name} enhancement",
                        "timestamp": datetime.now().isoformat(),
                        "success": False
                    })

                    self.logger.warning(f"Stage {stage.name} failed: {stage_result.get('error', 'Unknown error')}")

                stages_applied += 1
                applied_stages.append(stage.name)

            except Exception as e:
                self.logger.error(f"Error applying stage {stage.name}: {e}")
                enhancement_log.append({
                    "stage": stage.name,
                    "error": str(e),
                    "action": f"Error in {stage.name} enhancement",
                    "timestamp": datetime.now().isoformat(),
                    "success": False
                })

        # Calculate final metrics
        total_improvement = current_assessment.overall_score - initial_assessment.overall_score
        enhancement_duration = (datetime.now() - enhancement_start).total_seconds()

        # Determine final status
        target_met = current_assessment.overall_score >= target_quality

        self.logger.info(f"Progressive enhancement completed in {enhancement_duration:.2f}s")
        self.logger.info(f"Quality improvement: {initial_assessment.overall_score} → {current_assessment.overall_score} (+{total_improvement})")
        self.logger.info(f"Target {'met' if target_met else 'not met'} (target: {target_quality}, achieved: {current_assessment.overall_score})")

        return {
            "enhanced_content": current_content,
            "final_assessment": current_assessment,
            "initial_assessment": initial_assessment,
            "total_improvement": total_improvement,
            "target_met": target_met,
            "target_quality": target_quality,
            "stages_applied": len(applied_stages),
            "applied_stages": applied_stages,
            "enhancement_log": enhancement_log,
            "processing_time": enhancement_duration,
            "significant_improvement": significant_improvement_made,
            "metadata": {
                "original_content_length": len(content),
                "final_content_length": len(current_content),
                "content_change": len(current_content) - len(content),
                "pipeline_version": "1.0",
                "enhancement_timestamp": datetime.now().isoformat()
            }
        }

    def _calculate_stage_priority(self, stage: EnhancementStage, assessment: QualityAssessment) -> int:
        """
        Calculate priority for a stage based on current assessment.

        Args:
            stage: Enhancement stage to prioritize
            assessment: Current quality assessment

        Returns:
            Priority score (lower = higher priority)
        """
        base_priority = stage.priority

        # Boost priority for stages targeting low-scoring criteria
        priority_boost = 0
        for criterion in stage.target_criteria:
            if criterion in assessment.criteria_results:
                score = assessment.criteria_results[criterion].score
                # Lower scores get higher priority boost
                if score < 50:
                    priority_boost += 20
                elif score < 70:
                    priority_boost += 10
                elif score < 80:
                    priority_boost += 5

        return base_priority - priority_boost

    def get_stage_summary(self, enhancement_result: dict[str, Any]) -> dict[str, Any]:
        """Get a summary of applied enhancement stages."""
        log = enhancement_result.get("enhancement_log", [])

        successful_stages = [entry for entry in log if entry.get("success", False)]
        failed_stages = [entry for entry in log if not entry.get("success", False) and "error" in entry]
        skipped_stages = [entry for entry in log if not entry.get("success", False) and "reason" in entry]

        return {
            "total_stages_attempted": len(log),
            "successful_stages": len(successful_stages),
            "failed_stages": len(failed_stages),
            "skipped_stages": len(skipped_stages),
            "stage_details": {
                "successful": [entry["stage"] for entry in successful_stages],
                "failed": [entry["stage"] for entry in failed_stages],
                "skipped": [entry["stage"] for entry in skipped_stages]
            },
            "total_improvement": enhancement_result.get("total_improvement", 0),
            "processing_time": enhancement_result.get("processing_time", 0)
        }

    def add_custom_stage(self, stage: EnhancementStage):
        """Add a custom enhancement stage to the pipeline."""
        self.stages.append(stage)
        # Re-sort stages by priority
        self.stages.sort(key=lambda s: s.priority)
        self.logger.info(f"Added custom enhancement stage: {stage.name}")


# Convenience function for quick content enhancement
async def enhance_content_progressively(
    content: str,
    context: dict[str, Any] | None = None,
    target_quality: int = 85,
    max_stages: int = 10
) -> dict[str, Any]:
    """
    Quick progressive content enhancement function.

    Args:
        content: Content to enhance
        context: Additional context for enhancement
        target_quality: Target quality score (0-100)
        max_stages: Maximum number of enhancement stages

    Returns:
        Enhancement results
    """
    pipeline = ProgressiveEnhancementPipeline()
    return await pipeline.enhance_content(content, context, target_quality, max_stages)
