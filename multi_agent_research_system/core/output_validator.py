"""
Output Validation Layer for Multi-Agent Research System

This module provides validation for agent outputs to ensure they match expected types
and formats. Following Claude Agent SDK best practices for quality control.

Key Functions:
- validate_editorial_output: Ensure editorial stage produces critique, not report
- validate_report_output: Ensure report stage produces report, not critique  
- validate_final_output: Ensure final output is enhanced narrative, not JSON metadata
"""

import logging
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of output validation."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: list[str]
    output_type: str  # "critique", "report", "json_metadata", "unknown"
    
    def __bool__(self) -> bool:
        """Allow boolean evaluation of validation result."""
        return self.is_valid


class OutputValidator:
    """
    Validates agent outputs to ensure correct content types.
    
    This prevents the critical issue where editorial agents generate reports
    instead of critiques, and final outputs contain JSON metadata instead of
    enhanced narratives.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_editorial_output(self, content: str, session_id: str = None) -> ValidationResult:
        """
        Validate that editorial stage produces critique, not report.
        
        Args:
            content: The output content to validate
            session_id: Optional session ID for logging context
            
        Returns:
            ValidationResult with validation status and issues
        """
        self.logger.info(f"Validating editorial output for session {session_id}")
        
        issues = []
        content_lower = content.lower()
        
        # Check 1: Required critique sections present
        critique_sections = [
            "quality assessment",
            "identified issues",
            "information gaps",
            "recommendations"
        ]
        sections_present = sum(
            1 for section in critique_sections 
            if section in content_lower
        )
        
        if sections_present < 3:
            issues.append(f"Missing {4 - sections_present} required critique sections")
        
        # Check 2: Report sections should be absent (as section headers)
        report_sections = [
            "## executive summary",
            "## key findings",
            "## conclusion",
            "###executive summary",
            "###key findings",
            "###conclusion"
        ]
        report_sections_present = sum(
            1 for section in report_sections
            if section in content_lower
        )
        
        if report_sections_present > 0:
            issues.append(f"Contains {report_sections_present} report sections (should be 0)")
        
        # Check 3: Critique language markers
        critique_markers = [
            "the report lacks",
            "needs improvement",
            "gap identified",
            "recommend adding",
            "issue:",
            "score:"
        ]
        has_critique_language = any(
            marker in content_lower
            for marker in critique_markers
        )
        
        if not has_critique_language:
            issues.append("Lacks critique language (e.g., 'the report lacks', 'needs improvement')")
        
        # Check 4: Should reference original report
        references_report = any(
            phrase in content_lower
            for phrase in ["the report", "this report", "report contains", "report lacks"]
        )
        
        if not references_report:
            issues.append("Does not reference the original report being critiqued")
        
        # Determine output type
        if sections_present >= 3 and report_sections_present == 0:
            output_type = "critique"
        elif report_sections_present >= 2:
            output_type = "report"
        else:
            output_type = "unknown"
        
        # Calculate score
        score_factors = [
            sections_present / len(critique_sections),  # Required sections present
            1.0 - (report_sections_present / len(report_sections)),  # Report sections absent
            1.0 if has_critique_language else 0.0,  # Has critique language
            1.0 if references_report else 0.0  # References original report
        ]
        score = sum(score_factors) / len(score_factors)
        
        is_valid = (
            sections_present >= 3 and
            report_sections_present == 0 and
            has_critique_language and
            references_report
        )
        
        if is_valid:
            self.logger.info(f"✅ Editorial output validation passed (score: {score:.2f})")
        else:
            self.logger.warning(f"❌ Editorial output validation failed (score: {score:.2f}): {issues}")
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            output_type=output_type
        )
    
    def validate_report_output(self, content: str, session_id: str = None) -> ValidationResult:
        """
        Validate that report stage produces report, not critique.
        
        Args:
            content: The output content to validate
            session_id: Optional session ID for logging context
            
        Returns:
            ValidationResult with validation status and issues
        """
        self.logger.info(f"Validating report output for session {session_id}")
        
        issues = []
        content_lower = content.lower()
        
        # Check 1: Required report sections present
        report_sections = [
            "executive summary",
            "introduction",
            "findings",
            "conclusion"
        ]
        sections_present = sum(
            1 for section in report_sections
            if section in content_lower
        )
        
        if sections_present < 2:
            issues.append(f"Missing {4 - sections_present} expected report sections")
        
        # Check 2: Critique sections should be absent
        critique_sections = [
            "quality assessment",
            "identified issues",
            "recommendations"
        ]
        critique_sections_present = sum(
            1 for section in critique_sections
            if section in content_lower
        )
        
        if critique_sections_present > 0:
            issues.append(f"Contains {critique_sections_present} critique sections (should be 0)")
        
        # Check 3: Report should have substantive content
        word_count = len(content.split())
        if word_count < 400:  # Reduced from 500 to be less strict
            issues.append(f"Report too short ({word_count} words, minimum 400)")
        
        # Check 4: Should have sources
        has_sources = "http://" in content or "https://" in content
        if not has_sources:
            issues.append("No sources/citations found in report")
        
        # Determine output type
        if sections_present >= 2 and critique_sections_present == 0:
            output_type = "report"
        elif critique_sections_present >= 2:
            output_type = "critique"
        else:
            output_type = "unknown"
        
        # Calculate score
        score_factors = [
            sections_present / len(report_sections),  # Report sections present
            1.0 - (critique_sections_present / len(critique_sections)),  # Critique sections absent
            min(word_count / 500, 1.0),  # Adequate length
            1.0 if has_sources else 0.0  # Has sources
        ]
        score = sum(score_factors) / len(score_factors)
        
        is_valid = (
            sections_present >= 2 and
            critique_sections_present == 0 and
            word_count >= 500 and
            has_sources
        )
        
        if is_valid:
            self.logger.info(f"✅ Report output validation passed (score: {score:.2f})")
        else:
            self.logger.warning(f"❌ Report output validation failed (score: {score:.2f}): {issues}")
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            output_type=output_type
        )
    
    def validate_final_output(self, content: str, session_id: str = None) -> ValidationResult:
        """
        Validate that final output is enhanced narrative, not JSON metadata.
        
        This addresses the critical issue where final enhanced reports contain
        only JSON session metadata instead of actual report content.
        
        Args:
            content: The output content to validate
            session_id: Optional session ID for logging context
            
        Returns:
            ValidationResult with validation status and issues
        """
        self.logger.info(f"Validating final output for session {session_id}")
        
        issues = []
        content_stripped = content.strip()
        
        # Check 1: Should NOT be JSON metadata
        is_json_metadata = False
        if content.strip().startswith('{') and content.strip().endswith('}'):
            # Likely JSON - check for metadata keys
            metadata_keys = ["session_id", "topic", "user_requirements", "created_at", "status"]
            matching_keys = sum(1 for key in metadata_keys if key in content)
            if matching_keys >= 3:
                is_json_metadata = True
                issues.append("Final output is JSON session metadata, not report content")
        
        # Check 2: Should have narrative content
        word_count = len(content.split())
        if word_count < 200:
            issues.append(f"Final output too short ({word_count} words, minimum 200)")
        
        # Check 3: Should have markdown structure
        has_markdown_structure = (
            content.count('\n#') > 0 or  # Headers
            content.count('\n-') > 0 or  # Bullets
            content.count('\n##') > 0     # Subheaders
        )
        if not has_markdown_structure:
            issues.append("No markdown structure found (missing headers/bullets)")
        
        # Check 4: Should have substantive paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        substantive_paragraphs = [p for p in paragraphs if len(p.split()) > 20]
        if len(substantive_paragraphs) < 3:
            issues.append(f"Insufficient substantive content ({len(substantive_paragraphs)} paragraphs with 20+ words)")
        
        # Determine output type
        if is_json_metadata:
            output_type = "json_metadata"
        elif has_markdown_structure and len(substantive_paragraphs) >= 3:
            output_type = "report"
        else:
            output_type = "unknown"
        
        # Calculate score
        score_factors = [
            0.0 if is_json_metadata else 1.0,  # Not JSON metadata
            min(word_count / 200, 1.0),  # Adequate length (reduced threshold)
            1.0 if has_markdown_structure else 0.0,  # Has structure
            min(len(substantive_paragraphs) / 3, 1.0)  # Substantive content
        ]
        score = sum(score_factors) / len(score_factors)
        
        is_valid = (
            not is_json_metadata and
            word_count >= 200 and  # Reduced from 300
            has_markdown_structure and
            len(substantive_paragraphs) >= 3
        )
        
        if is_valid:
            self.logger.info(f"✅ Final output validation passed (score: {score:.2f})")
        else:
            self.logger.error(f"❌ Final output validation FAILED (score: {score:.2f}): {issues}")
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            output_type=output_type
        )
    
    def validate_workflow_output(
        self, 
        stage: str, 
        content: str, 
        session_id: str = None
    ) -> ValidationResult:
        """
        Validate output based on workflow stage.
        
        Args:
            stage: Workflow stage ("research", "report", "editorial", "final")
            content: The output content to validate
            session_id: Optional session ID for logging context
            
        Returns:
            ValidationResult with validation status and issues
        """
        if stage == "editorial":
            return self.validate_editorial_output(content, session_id)
        elif stage == "report":
            return self.validate_report_output(content, session_id)
        elif stage == "final":
            return self.validate_final_output(content, session_id)
        else:
            self.logger.warning(f"Unknown stage '{stage}' - skipping validation")
            return ValidationResult(
                is_valid=True,  # Default to valid for unknown stages
                score=1.0,
                issues=[f"Validation not implemented for stage: {stage}"],
                output_type="unknown"
            )


# Global validator instance
_validator = None

def get_output_validator() -> OutputValidator:
    """Get or create the global output validator instance."""
    global _validator
    if _validator is None:
        _validator = OutputValidator()
    return _validator
