"""
LLM-Based Gap Research Evaluator

Simple, fast LLM evaluation of search workproduct to determine if additional research is needed.
Replaces complex threshold-based systems with intelligent natural language understanding.

Key Features:
- Binary decision: MORE_RESEARCH_NEEDED or SUFFICIENT
- Provides specific search queries if research needed
- Configurable strictness via prompt adjustment
- Fast evaluation using GPT-5-nano
- Fallback to "SUFFICIENT" on errors
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GapResearchEvaluation:
    """Result of LLM gap research evaluation."""

    decision: str  # "MORE_RESEARCH_NEEDED" or "SUFFICIENT"
    reasoning: str
    suggested_queries: List[str]
    confidence: float  # 0.0 - 1.0
    evaluator: str = "llm_gap_research_evaluator"


class LLMGapResearchEvaluator:
    """
    Simple LLM-based gap research evaluator.

    Uses fast LLM evaluation to determine if additional research is needed,
    replacing complex threshold-based systems with intelligent understanding.
    """

    def __init__(self, model: str = "gpt-5-nano", strictness: str = "standard"):
        """
        Initialize LLM gap research evaluator.

        Args:
            model: LLM model to use for evaluation
            strictness: Evaluation strictness level ("lenient", "standard", "strict")
        """
        self.model = model
        self.strictness = strictness
        self.logger = logging.getLogger(f"{__name__}.{strictness}")

        # Prompts configured by strictness level
        self.prompts = {
            "lenient": {
                "system": "You are a research assistant evaluating if search results are sufficient.",
                "evaluation": """
Review this search workproduct and determine if additional research is needed.

CONSIDER:
- Are there MAJOR information gaps that would significantly impact understanding?
- Is the coverage substantially incomplete for the research topic?
- Are critical aspects completely missing?

RETURN DECISION:
- "SUFFICIENT" if research adequately covers the topic (even with minor gaps)
- "MORE_RESEARCH_NEEDED" only if major gaps significantly impact understanding

If MORE_RESEARCH_NEEDED, provide 1-2 specific search queries to fill gaps.

RESPOND IN EXACT FORMAT:
DECISION: [SUFFICIENT|MORE_RESEARCH_NEEDED]
REASONING: [Brief explanation]
QUERIES: [query1; query2] (only if MORE_RESEARCH_NEEDED)
CONFIDENCE: [0.0-1.0]
""",
                "expected_keywords": ["sufficient", "adequate", "adequately covers", "minor gaps"]
            },
            "standard": {
                "system": "You are a research evaluator assessing search result completeness.",
                "evaluation": """
Review this search workproduct and evaluate if additional research is needed.

ASSESSMENT CRITERIA:
- Information completeness for the research topic
- Source diversity and quality
- Coverage of key aspects and perspectives
- Temporal relevance (if applicable)

DECISION GUIDELINES:
- "SUFFICIENT" if research provides good coverage with decent sources
- "MORE_RESEARCH_NEEDED" if significant gaps exist that impact understanding

If MORE_RESEARCH_NEEDED, provide 1-2 specific search queries to address gaps.

RESPOND IN EXACT FORMAT:
DECISION: [SUFFICIENT|MORE_RESEARCH_NEEDED]
REASONING: [Brief explanation]
QUERIES: [query1; query2] (only if MORE_RESEARCH_NEEDED)
CONFIDENCE: [0.0-1.0]
""",
                "expected_keywords": ["good coverage", "decent sources", "sufficient", "adequate"]
            },
            "strict": {
                "system": "You are a rigorous research quality evaluator.",
                "evaluation": """
Review this search workproduct and determine if additional research is needed.

STRICT EVALUATION CRITERIA:
- Comprehensive coverage of all aspects of the topic
- High-quality, authoritative sources
- Multiple perspectives and viewpoints
- Current and relevant information
- Depth of analysis and insights

HIGH STANDARDS:
- "SUFFICIENT" only if research is comprehensive and high-quality
- "MORE_RESEARCH_NEEDED" if any significant gaps or quality issues exist

If MORE_RESEARCH_NEEDED, provide 1-2 specific search queries for comprehensive coverage.

RESPOND IN EXACT FORMAT:
DECISION: [SUFFICIENT|MORE_RESEARCH_NEEDED]
REASONING: [Brief explanation]
QUERIES: [query1; query2] (only if MORE_RESEARCH_NEEDED)
CONFIDENCE: [0.0-1.0]
""",
                "expected_keywords": ["comprehensive", "high-quality", "authoritative", "thorough"]
            }
        }

    async def evaluate_search_workproduct(
        self,
        session_id: str,
        workproduct_path: Optional[str] = None
    ) -> GapResearchEvaluation:
        """
        Evaluate search workproduct to determine if gap research is needed.

        Args:
            session_id: Research session ID
            workproduct_path: Optional path to specific workproduct file

        Returns:
            GapResearchEvaluation with decision and recommendations
        """
        try:
            # Step 1: Find and read search workproduct
            workproduct_content = await self._load_search_workproduct(session_id, workproduct_path)

            if not workproduct_content:
                self.logger.warning(f"No search workproduct found for session {session_id}")
                return GapResearchEvaluation(
                    decision="SUFFICIENT",
                    reasoning="No search workproduct available - assuming sufficient",
                    suggested_queries=[],
                    confidence=0.5
                )

            # Step 2: Prepare content for LLM evaluation
            evaluation_content = self._prepare_evaluation_content(workproduct_content)

            # Step 3: Execute LLM evaluation
            evaluation_result = await self._execute_llm_evaluation(evaluation_content)

            # Step 4: Parse and validate result
            evaluation = self._parse_evaluation_result(evaluation_result)

            self.logger.info(
                f"LLM gap research evaluation for session {session_id}: "
                f"{evaluation.decision} (confidence: {evaluation.confidence:.2f})"
            )

            if evaluation.decision == "MORE_RESEARCH_NEEDED":
                self.logger.info(f"Suggested queries: {evaluation.suggested_queries}")

            return evaluation

        except Exception as e:
            self.logger.error(f"Error in LLM gap research evaluation for session {session_id}: {e}")
            # Fail-safe: default to SUFFICIENT
            return GapResearchEvaluation(
                decision="SUFFICIENT",
                reasoning=f"Evaluation error: {str(e)} - defaulting to sufficient",
                suggested_queries=[],
                confidence=0.0
            )

    async def _load_search_workproduct(
        self,
        session_id: str,
        workproduct_path: Optional[str] = None
    ) -> Optional[str]:
        """Load search workproduct content from session directory."""
        try:
            # If specific path provided, use it
            if workproduct_path and Path(workproduct_path).exists():
                return Path(workproduct_path).read_text(encoding='utf-8')

            # Auto-discover search workproduct in session directory
            kevin_dir = Path("KEVIN")
            session_dir = kevin_dir / "sessions" / session_id

            # Priority 1: Research directory search workproducts
            research_files = list(session_dir.glob("research/search_workproduct_*.md"))
            if research_files:
                # Use the most recent
                latest_file = max(research_files, key=lambda x: x.stat().st_mtime)
                content = latest_file.read_text(encoding='utf-8')
                self.logger.debug(f"Loaded search workproduct: {latest_file.name}")
                return content

            # Priority 2: Working directory RESEARCH files
            working_files = list(session_dir.glob("working/RESEARCH_*.md"))
            if working_files:
                latest_file = max(working_files, key=lambda x: x.stat().st_mtime)
                content = latest_file.read_text(encoding='utf-8')
                self.logger.debug(f"Loaded working RESEARCH file: {latest_file.name}")
                return content

            # Priority 3: Any markdown with search results
            all_md_files = list(session_dir.glob("**/*.md"))
            for md_file in all_md_files:
                try:
                    content = md_file.read_text(encoding='utf-8')
                    if "## 🔍 Search Results Summary" in content or "search workproduct" in content.lower():
                        self.logger.debug(f"Found search content in: {md_file.name}")
                        return content
                except Exception:
                    continue

            self.logger.warning(f"No search workproduct found for session {session_id}")
            return None

        except Exception as e:
            self.logger.error(f"Error loading search workproduct for session {session_id}: {e}")
            return None

    def _prepare_evaluation_content(self, workproduct_content: str) -> str:
        """Prepare workproduct content for LLM evaluation."""
        try:
            # Extract key sections for evaluation
            content_sections = []

            # Extract search results summary
            search_section = self._extract_section(workproduct_content, "## 🔍 Search Results Summary")
            if search_section:
                content_sections.append("SEARCH RESULTS:")
                content_sections.append(search_section[:2000])  # Limit content length

            # Extract session metadata if available
            metadata_section = self._extract_section(workproduct_content, "Session ID")
            if metadata_section:
                content_sections.append("SESSION INFO:")
                content_sections.append(metadata_section[:500])

            # If no structured sections found, use first part of content
            if not content_sections:
                content_sections.append("WORKPRODUCT CONTENT:")
                content_sections.append(workproduct_content[:3000])

            return "\n\n".join(content_sections)

        except Exception as e:
            self.logger.error(f"Error preparing evaluation content: {e}")
            return workproduct_content[:3000]  # Fallback to truncated content

    def _extract_section(self, content: str, section_header: str) -> Optional[str]:
        """Extract a section from markdown content."""
        try:
            lines = content.split('\n')
            start_idx = None
            end_idx = None

            for i, line in enumerate(lines):
                if section_header in line:
                    start_idx = i
                elif start_idx is not None and line.startswith('##'):
                    end_idx = i
                    break

            if start_idx is not None:
                if end_idx is None:
                    section_lines = lines[start_idx:]
                else:
                    section_lines = lines[start_idx:end_idx]

                # Limit section length
                section_text = '\n'.join(section_lines[:50])  # Max 50 lines
                return section_text.strip()

            return None

        except Exception as e:
            self.logger.error(f"Error extracting section '{section_header}': {e}")
            return None

    async def _execute_llm_evaluation(self, evaluation_content: str) -> str:
        """Execute LLM evaluation using OpenAI API."""
        try:
            # Import here to avoid dependency issues
            import openai
            from openai import AsyncOpenAI

            # Initialize client
            client = AsyncOpenAI()

            # Get prompt configuration for strictness level
            prompt_config = self.prompts.get(self.strictness, self.prompts["standard"])

            # Create evaluation prompt
            full_prompt = f"""{prompt_config['evaluation']}

SEARCH WORKPRODUCT TO EVALUATE:

{evaluation_content}

"""

            # Execute LLM call
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_config['system']},
                    {"role": "user", "content": full_prompt}
                ],
                max_completion_tokens=500
            )

            result = response.choices[0].message.content.strip()
            self.logger.debug(f"LLM evaluation result: {result}")

            return result

        except ImportError:
            self.logger.error("OpenAI package not available for LLM evaluation")
            raise
        except Exception as e:
            self.logger.error(f"Error executing LLM evaluation: {e}")
            raise

    def _parse_evaluation_result(self, evaluation_result: str) -> GapResearchEvaluation:
        """Parse LLM evaluation result into structured format."""
        try:
            # Extract decision
            decision_match = re.search(r'DECISION:\s*(.+?)(?:\n|$)', evaluation_result, re.IGNORECASE)
            decision = decision_match.group(1).strip().upper() if decision_match else "SUFFICIENT"

            # Normalize decision
            if "MORE" in decision or "NEEDED" in decision:
                decision = "MORE_RESEARCH_NEEDED"
            else:
                decision = "SUFFICIENT"

            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n\w+:|$)', evaluation_result, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

            # Extract queries
            queries_match = re.search(r'QUERIES:\s*(.+?)(?:\n\w+:|$)', evaluation_result, re.IGNORECASE)
            suggested_queries = []
            if queries_match and decision == "MORE_RESEARCH_NEEDED":
                queries_text = queries_match.group(1).strip()
                # Split by semicolon or new lines
                suggested_queries = [
                    q.strip() for q in re.split(r'[;\n]+', queries_text)
                    if q.strip() and len(q.strip()) > 10
                ][:2]  # Max 2 queries

            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', evaluation_result)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            confidence = min(1.0, max(0.0, confidence))  # Clamp to valid range

            return GapResearchEvaluation(
                decision=decision,
                reasoning=reasoning,
                suggested_queries=suggested_queries,
                confidence=confidence
            )

        except Exception as e:
            self.logger.error(f"Error parsing evaluation result: {e}")
            # Return conservative default
            return GapResearchEvaluation(
                decision="SUFFICIENT",
                reasoning=f"Parse error: {str(e)} - defaulting to sufficient",
                suggested_queries=[],
                confidence=0.0
            )


# Factory function for easy instantiation
def create_gap_research_evaluator(
    model: str = "gpt-5-nano",
    strictness: str = "standard"
) -> LLMGapResearchEvaluator:
    """
    Create and configure a gap research evaluator.

    Args:
        model: LLM model to use
        strictness: Evaluation strictness level

    Returns:
        Configured LLMGapResearchEvaluator instance
    """
    return LLMGapResearchEvaluator(model=model, strictness=strictness)


# Convenience function for direct evaluation
async def evaluate_gap_research_need(
    session_id: str,
    workproduct_path: Optional[str] = None,
    strictness: str = "standard"
) -> GapResearchEvaluation:
    """
    Evaluate if gap research is needed for a session.

    Args:
        session_id: Research session ID
        workproduct_path: Optional path to specific workproduct
        strictness: Evaluation strictness level

    Returns:
        GapResearchEvaluation result
    """
    evaluator = create_gap_research_evaluator(strictness=strictness)
    return await evaluator.evaluate_search_workproduct(session_id, workproduct_path)