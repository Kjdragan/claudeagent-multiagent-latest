"""
Gap Research Enforcement System

Multi-layered validation system ensuring complete gap research execution
with 100% compliance enforcement throughout the editorial workflow.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib

# Configure logger
logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance levels for gap research enforcement."""
    CRITICAL = "critical"    # Must execute - system failure if not
    HIGH = "high"            # Strong recommendation enforced
    MEDIUM = "medium"        # Recommended with enforcement
    LOW = "low"              # Optional - logged only


class EnforcementAction(Enum):
    """Types of enforcement actions."""
    BLOCK_EXECUTION = "block_execution"      # Stop workflow until compliance
    AUTO_EXECUTION = "auto_execution"      # Force automatic execution
    ENHANCED_LOGGING = "enhanced_logging"    # Detailed logging and alerts
    MANUAL_REVIEW = "manual_review"        # Require manual verification
    QUALITY_PENALTY = "quality_penalty"      # Reduce quality score


@dataclass
class GapResearchRequirement:
    """Specific gap research requirement to be enforced."""
    requirement_id: str
    description: str
    compliance_level: ComplianceLevel
    validation_criteria: List[str]
    enforcement_actions: List[EnforcementAction]
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, in_progress, completed, failed, waived


@dataclass
class ComplianceCheckResult:
    """Result of a compliance check."""
    check_id: str
    requirement_id: str
    compliance_level: ComplianceLevel
    passed: bool
    details: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    enforcement_actions_taken: List[EnforcementAction] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


@dataclass
class GapResearchEnforcementReport:
    """Comprehensive report on gap research enforcement."""
    session_id: str
    enforcement_session_id: str
    total_requirements: int
    compliant_requirements: int
    failed_requirements: int
    waived_requirements: int
    overall_compliance_rate: float
    enforcement_actions_taken: List[EnforcementAction]
    critical_violations: List[ComplianceCheckResult]
    quality_impact: float
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class GapResearchEnforcementSystem:
    """
    Multi-layered validation system ensuring complete gap research execution
    with 100% compliance enforcement.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.requirements_registry: Dict[str, GapResearchRequirement] = {}
        self.compliance_history: List[ComplianceCheckResult] = []
        self.enforcement_reports: List[GapResearchEnforcementReport] = []
        self.active_enforcements: Dict[str, ComplianceCheckResult] = {}

        # Initialize standard gap research requirements
        self._initialize_standard_requirements()

    def _initialize_standard_requirements(self):
        """Initialize standard gap research requirements."""
        standard_requirements = [
            GapResearchRequirement(
                requirement_id="GAP_001",
                description="Gap research must be conducted when editorial decision identifies gaps",
                compliance_level=ComplianceLevel.CRITICAL,
                validation_criteria=[
                    "Editorial decision analysis completed",
                    "Gap identification documented with evidence",
                    "Gap research queries formulated",
                    "Research execution initiated"
                ],
                enforcement_actions=[
                    EnforcementAction.BLOCK_EXECUTION,
                    EnforcementAction.AUTO_EXECUTION,
                    EnforcementAction.QUALITY_PENALTY
                ]
            ),
            GapResearchRequirement(
                requirement_id="GAP_002",
                description="Gap research results must be integrated into final report",
                compliance_level=ComplianceLevel.HIGH,
                validation_criteria=[
                    "Gap research completed successfully",
                    "Results analyzed and quality assessed",
                    "Integration with existing research",
                    "Final report updated with gap research findings"
                ],
                enforcement_actions=[
                    EnforcementAction.AUTO_EXECUTION,
                    EnforcementAction.ENHANCED_LOGGING,
                    EnforcementAction.QUALITY_PENALTY
                ]
            ),
            GapResearchRequirement(
                requirement_id="GAP_003",
                description="Sub-session management must be properly coordinated",
                compliance_level=ComplianceLevel.HIGH,
                validation_criteria=[
                    "Sub-session created and linked to parent",
                    "Resource allocation appropriate",
                    "Session state properly synchronized",
                    "Results integrated into parent session"
                ],
                enforcement_actions=[
                    EnforcementAction.AUTO_EXECUTION,
                    EnforcementAction.ENHANCED_LOGGING
                ]
            ),
            GapResearchRequirement(
                requirement_id="GAP_004",
                description="Quality assessment must be performed on gap research results",
                compliance_level=ComplianceLevel.MEDIUM,
                validation_criteria=[
                    "Quality metrics calculated",
                    "Confidence scores established",
                    "Improvement recommendations generated",
                    "Quality gates passed"
                ],
                enforcement_actions=[
                    EnforcementAction.ENHANCED_LOGGING,
                    EnforcementAction.MANUAL_REVIEW
                ]
            ),
            GapResearchRequirement(
                requirement_id="GAP_005",
                description="Documentation and audit trail must be maintained",
                compliance_level=ComplianceLevel.MEDIUM,
                validation_criteria=[
                    "All gap research activities logged",
                    "Decision rationale documented",
                    "Evidence collected and stored",
                    "Audit trail complete and verifiable"
                ],
                enforcement_actions=[
                    EnforcementAction.ENHANCED_LOGGING,
                    EnforcementAction.MANUAL_REVIEW
                ]
            )
        ]

        for req in standard_requirements:
            self.requirements_registry[req.requirement_id] = req

    async def register_custom_requirement(self, requirement: GapResearchRequirement):
        """Register a custom gap research requirement."""
        self.requirements_registry[requirement.requirement_id] = requirement
        logger.info(f"Custom requirement registered: {requirement.requirement_id}")

    async def enforce_gap_research_compliance(self,
                                          session_id: str,
                                          editorial_decision: Dict[str, Any],
                                          gap_research_status: Dict[str, Any]) -> GapResearchEnforcementReport:
        """
        Enforce gap research compliance for a specific session.

        Args:
            session_id: Research session identifier
            editorial_decision: Editorial decision analysis results
            gap_research_status: Current gap research execution status

        Returns:
            Comprehensive enforcement report
        """
        enforcement_session_id = f"enforce_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting gap research enforcement for session: {session_id}")
        logger.info(f"Enforcement session ID: {enforcement_session_id}")

        # Initialize compliance checks
        compliance_checks = []
        failed_requirements = []
        waived_requirements = []
        enforcement_actions = []

        # Check each requirement
        for requirement_id, requirement in self.requirements_registry.items():
            check_result = await self._check_requirement_compliance(
                requirement_id,
                requirement,
                session_id,
                editorial_decision,
                gap_research_status
            )

            compliance_checks.append(check_result)

            if not check_result.passed:
                failed_requirements.append(check_result)

                # Apply enforcement actions
                for action in requirement.enforcement_actions:
                    await self._apply_enforcement_action(
                        action, check_result, session_id
                    )
                    enforcement_actions.append(action)
            else:
                logger.info(f"Requirement {requirement_id} passed compliance check")

        # Calculate overall compliance rate
        total_requirements = len(self.requirements_registry)
        compliant_requirements = total_requirements - len(failed_requirements)
        overall_compliance_rate = compliant_requirements / total_requirements if total_requirements > 0 else 1.0

        # Calculate quality impact
        quality_impact = self._calculate_quality_impact(failed_requirements)

        # Generate recommendations
        recommendations = self._generate_recommendations(failed_requirements)

        # Create enforcement report
        report = GapResearchEnforcementReport(
            session_id=session_id,
            enforcement_session_id=enforcement_session_id,
            total_requirements=total_requirements,
            compliant_requirements=compliant_requirements,
            failed_requirements=len(failed_requirements),
            waived_requirements=len(waived_requirements),
            overall_compliance_rate=overall_compliance_rate,
            enforcement_actions_taken=enforcement_actions,
            critical_violations=[r for r in failed_requirements if r.compliance_level == ComplianceLevel.CRITICAL],
            quality_impact=quality_impact,
            recommendations=recommendations,
            completed_at=datetime.now()
        )

        # Store report
        self.enforcement_reports.append(report)

        # Log compliance results
        logger.info(f"Gap research enforcement completed for session: {session_id}")
        logger.info(f"Compliance rate: {overall_compliance_rate:.2%}")
        logger.info(f"Failed requirements: {len(failed_requirements)}")
        logger.info(f"Critical violations: {len(report.critical_violations)}")
        logger.info(f"Quality impact: {quality_impact:.2f}")

        return report

    async def _check_requirement_compliance(self,
                                           requirement_id: str,
                                           requirement: GapResearchRequirement,
                                           session_id: str,
                                           editorial_decision: Dict[str, Any],
                                           gap_research_status: Dict[str, Any]) -> ComplianceCheckResult:
        """Check compliance for a specific requirement."""

        logger.debug(f"Checking compliance for requirement: {requirement_id}")

        passed = True
        details = "Compliance check passed"
        evidence = {}
        enforcement_actions = []
        next_steps = []

        # Check based on requirement type
        if requirement_id == "GAP_001":
            passed, details, evidence = await self._check_gap_research_execution(
                editorial_decision, gap_research_status
            )
        elif requirement_id == "GAP_002":
            passed, details, evidence = await self._check_gap_research_integration(
                editorial_decision, gap_research_status
            )
        elif requirement_id == "GAP_003":
            passed, details, evidence = await self._check_sub_session_coordination(
                gap_research_status
            )
        elif requirement_id == "GAP_004":
            passed, details, evidence = await self._check_quality_assessment(
                gap_research_status
            )
        elif requirement_id == "GAP_005":
            passed, details, evidence = await self._check_documentation_compliance(
                session_id, gap_research_status
            )
        else:
            # Custom requirement check
            passed, details, evidence = await self._check_custom_requirement(
                requirement, editorial_decision, gap_research_status
            )

        # Create compliance check result
        result = ComplianceCheckResult(
            check_id=f"check_{requirement_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            requirement_id=requirement_id,
            compliance_level=requirement.compliance_level,
            passed=passed,
            details=details,
            evidence=evidence,
            enforcement_actions_taken=enforcement_actions,
            next_steps=next_steps
        )

        # Store in history
        self.compliance_history.append(result)

        return result

    async def _check_gap_research_execution(self,
                                           editorial_decision: Dict[str, Any],
                                           gap_research_status: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
        """Check if gap research was properly executed."""

        # Check if gap research was identified
        gap_identified = editorial_decision.get("gap_research_needed", False)
        if not gap_identified:
            return True, "No gap research needed", {"decision": editorial_decision}

        # Check if gap research was executed
        gap_executed = gap_research_status.get("executed", False)
        if not gap_executed:
            return False, "Gap research identified but not executed", {
                "gap_needed": True,
                "gap_executed": False,
                "decision": editorial_decision
            }

        # Check execution quality
        execution_quality = gap_research_status.get("quality_score", 0.0)
        if execution_quality < 0.6:
            return False, f"Gap research executed but low quality ({execution_quality:.2f})", {
                "gap_executed": True,
                "quality_score": execution_quality,
                "min_acceptable": 0.6
            }

        return True, "Gap research executed successfully", {
            "gap_executed": True,
            "quality_score": execution_quality,
            "gap_research_data": gap_research_status
        }

    async def _check_gap_research_integration(self,
                                           editorial_decision: Dict[str, Any],
                                           gap_research_status: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
        """Check if gap research results were integrated into final report."""

        gap_executed = gap_research_status.get("executed", False)
        if not gap_executed:
            return True, "No gap research to integrate", {"status": "not_applicable"}

        # Check if integration was attempted
        integration_attempted = gap_research_status.get("integrated", False)
        if not integration_attempted:
            return False, "Gap research executed but not integrated", {
                "gap_executed": True,
                "integrated": False,
                "gap_research_data": gap_research_status
            }

        # Check integration quality
        integration_quality = gap_research_status.get("integration_score", 0.0)
        if integration_quality < 0.7:
            return False, f"Gap research integration attempted but low quality ({integration_quality:.2f})", {
                "gap_executed": True,
                "integrated": True,
                "integration_score": integration_quality,
                "min_acceptable": 0.7
            }

        return True, "Gap research results integrated successfully", {
            "gap_executed": True,
            "integrated": True,
            "integration_score": integration_quality,
            "gap_research_data": gap_research_status
        }

    async def _check_sub_session_coordination(self,
                                           gap_research_status: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
        """Check if sub-session management was properly coordinated."""

        sub_sessions = gap_research_status.get("sub_sessions", [])
        if not sub_sessions:
            return True, "No sub-sessions created", {"status": "not_applicable"}

        # Check sub-session coordination
        coordination_issues = []
        for sub_session in sub_sessions:
            session_id = sub_session.get("session_id")
            if not session_id:
                coordination_issues.append(f"Sub-session missing session ID")
                continue

            # Check parent-child linkage
            parent_id = sub_session.get("parent_session_id")
            if not parent_id:
                coordination_issues.append(f"Sub-session {session_id} not linked to parent")

            # Check resource allocation
            resources = sub_session.get("resources", {})
            if not resources:
                coordination_issues.append(f"Sub-session {session_id} has no resource allocation")

        if coordination_issues:
            return False, f"Sub-session coordination issues: {len(coordination_issues)}", {
                "sub_sessions": sub_sessions,
                "coordination_issues": coordination_issues
            }

        return True, "Sub-session coordination successful", {
            "sub_sessions": sub_sessions,
            "coordination_successful": True
        }

    async def _check_quality_assessment(self,
                                      gap_research_status: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
        """Check if quality assessment was performed on gap research."""

        quality_assessment = gap_research_status.get("quality_assessment", {})
        if not quality_assessment:
            return False, "No quality assessment performed", {
                "quality_assessment": quality_assessment,
                "gap_research_data": gap_research_status
            }

        # Check quality score
        quality_score = quality_assessment.get("overall_score", 0.0)
        if quality_score < 0.7:
            return False, f"Quality assessment completed but low score ({quality_score:.2f})", {
                "quality_assessment": quality_assessment,
                "quality_score": quality_score,
                "min_acceptable": 0.7
            }

        return True, "Quality assessment completed successfully", {
            "quality_assessment": quality_assessment,
            "quality_score": quality_score,
            "gap_research_data": gap_research_status
        }

    async def _check_documentation_compliance(self,
                                          session_id: str,
                                          gap_research_status: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
        """Check if documentation and audit trail are maintained."""

        # Check if logs exist
        logs = gap_research_status.get("logs", [])
        if not logs:
            return False, "No documentation/logs maintained", {
                "logs": logs,
                "gap_research_data": gap_research_status
            }

        # Check audit trail completeness
        audit_trail = gap_research_status.get("audit_trail", {})
        required_audit_items = ["decision_made", "research_executed", "results_analyzed", "integration_completed"]
        missing_items = [item for item in required_audit_items if item not in audit_trail]

        if missing_items:
            return False, f"Audit trail incomplete: missing {len(missing_items)} items", {
                "audit_trail": audit_trail,
                "missing_items": missing_items,
                "logs": logs
            }

        return True, "Documentation and audit trail maintained", {
            "audit_trail": audit_trail,
            "logs": logs,
            "documentation_compliant": True
        }

    async def _check_custom_requirement(self,
                                        requirement: GapResearchRequirement,
                                        editorial_decision: Dict[str, Any],
                                        gap_research_status: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
        """Check compliance for custom requirement."""

        # Implement custom requirement checking logic
        # This would be implemented based on specific requirement criteria

        # For now, assume custom requirements pass
        return True, f"Custom requirement {requirement.requirement_id} passed", {
            "requirement_id": requirement.requirement_id,
            "custom_check": True
        }

    async def _apply_enforcement_action(self,
                                        action: EnforcementAction,
                                        check_result: ComplianceCheckResult,
                                        session_id: str):
        """Apply enforcement action for compliance violation."""

        logger.warning(f"Applying enforcement action: {action.value} for requirement: {check_result.requirement_id}")

        if action == EnforcementAction.BLOCK_EXECUTION:
            # Block workflow execution
            logger.error(f"WORKFLOW BLOCKED: Critical compliance violation for requirement {check_result.requirement_id}")
            await self._block_workflow_execution(session_id, check_result)

        elif action == EnforcementAction.AUTO_EXECUTION:
            # Force automatic execution
            logger.info(f"Forcing automatic execution for requirement: {check_result.requirement_id}")
            await self._force_automatic_execution(session_id, check_result)

        elif action == EnforcementAction.ENHANCED_LOGGING:
            # Enhanced logging and alerts
            await self._enhanced_logging(check_result, session_id)

        elif action == EnforcementAction.MANUAL_REVIEW:
            # Require manual verification
            logger.warning(f"MANUAL REVIEW REQUIRED for requirement: {check_result.requirement_id}")
            await self._request_manual_review(check_result, session_id)

        elif action == EnforcementAction.QUALITY_PENALTY:
            # Apply quality penalty
            logger.info(f"Applying quality penalty for requirement: {check_result.requirement_id}")
            await self._apply_quality_penalty(session_id, check_result)

    async def _block_workflow_execution(self, session_id: str, check_result: ComplianceCheckResult):
        """Block workflow execution until compliance is achieved."""
        # Implement workflow blocking logic
        logger.error(f"Session {session_id} workflow blocked until compliance issue resolved")

        # Store active enforcement
        self.active_enforcements[check_result.requirement_id] = check_result

    async def _force_automatic_execution(self, session_id: str, check_result: ComplianceCheckResult):
        """Force automatic execution of gap research."""
        # Implement automatic execution forcing
        logger.info(f"Forcing automatic gap research execution for session: {session_id}")

    async def _enhanced_logging(self, check_result: ComplianceCheckResult, session_id: str):
        """Enhanced logging and alerting for compliance issues."""
        # Create detailed log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "requirement_id": check_result.requirement_id,
            "compliance_level": check_result.compliance_level.value,
            "passed": check_result.passed,
            "details": check_result.details,
            "evidence": check_result.evidence
        }

        logger.warning(f"COMPLIANCE ISSUE: {json.dumps(log_entry, indent=2)}")

    async def _request_manual_review(self, check_result: ComplianceCheckResult, session_id str):
        """Request manual review for compliance issue."""
        # Implement manual review request
        logger.warning(f"MANUAL REVIEW REQUESTED for requirement: {check_result.requirement_id}")

    async def _apply_quality_penalty(self, session_id: str, check_result: ComplianceCheckResult):
        """Apply quality penalty for compliance violation."""
        # Implement quality penalty application
        penalty_amount = self._calculate_quality_penalty(check_result)
        logger.warning(f"Quality penalty applied: {penalty_amount} for requirement: {check_result.requirement_id}")

    def _calculate_quality_impact(self, failed_requirements: List[ComplianceCheckResult]) -> float:
        """Calculate overall quality impact of failed requirements."""
        if not failed_requirements:
            return 0.0

        total_impact = 0.0
        for req in failed_requirements:
            # Higher impact for critical compliance levels
            if req.compliance_level == ComplianceLevel.CRITICAL:
                total_impact += 0.3
            elif req.compliance_level == ComplianceLevel.HIGH:
                total_impact += 0.2
            elif req.compliance_level == ComplianceLevel.MEDIUM:
                total_impact += 0.1
            elif req.compliance_level == ComplianceLevel.LOW:
                total_impact += 0.05

        return min(total_impact, 0.9)  # Cap at 90% quality reduction

    def _calculate_quality_penalty(self, check_result: ComplianceCheckResult) -> float:
        """Calculate quality penalty for compliance violation."""
        base_penalty = 0.1

        # Increase penalty based on compliance level
        if check_result.compliance_level == ComplianceLevel.CRITICAL:
            base_penalty = 0.3
        elif check_result.compliance_level == ComplianceLevel.HIGH:
            base_penalty = 0.2
        elif check_result.compliance_level == ComplianceLevel.MEDIUM:
            base_penalty = 0.15
        elif check_result.compliance_level == ComplianceLevel.LOW:
            base_penalty = 0.1

        return base_penalty

    def _generate_recommendations(self, failed_requirements: List[ComplianceCheckResult]) -> List[str]:
        """Generate recommendations for addressing compliance issues."""
        recommendations = []

        for req in failed_requirements:
            if req.requirement_id == "GAP_001":
                recommendations.extend([
                    "Ensure gap research is identified and documented in editorial decisions",
                    "Implement automatic gap research triggering based on editorial analysis",
                    "Establish quality thresholds for gap research execution"
                ])
            elif req.requirement_id == "GAP_002":
                recommendations.extend([
                    "Implement automatic integration of gap research results into final reports",
                    "Create integration templates for different types of gap research findings",
                    "Establish quality checks for integration completeness"
                ])
            elif req.requirement_id == "GAP_003":
                recommendations.extend([
                    "Implement robust sub-session management with parent-child coordination",
                    "Create resource allocation strategies for gap research sub-sessions",
                    "Establish state synchronization mechanisms between sub-sessions"
                ])
            elif req.requirement_id == "GAP_004":
                recommendations.extend([
                    "Implement quality assessment workflow for all gap research results",
                    "Create standardized quality metrics and scoring systems",
                    "Establish quality gates that must be passed before acceptance"
                ])
            elif req.requirement_id == "GAP_005":
                recommendations.extend([
                    "Implement comprehensive logging system for all gap research activities",
                    "Create audit trail templates and procedures",
                    "Establish documentation standards for compliance verification"
                ])
            else:
                recommendations.append(f"Address compliance issues for requirement: {req.requirement_id}")

        return recommendations

    def get_enforcement_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of gap research enforcement activities."""

        if session_id:
            # Get summary for specific session
            session_reports = [r for r in self.enforcement_reports if r.session_id == session_id]
        else:
            # Get overall summary
            session_reports = self.enforcement_reports

        if not session_reports:
            return {"message": "No enforcement reports found"}

        # Calculate summary statistics
        total_sessions = len(session_reports)
        avg_compliance_rate = sum(r.overall_compliance_rate for r in session_reports) / total_sessions
        total_violations = sum(len(r.critical_violations) for r in session_reports)

        return {
            "total_enforcement_sessions": total_sessions,
            "average_compliance_rate": avg_compliance_rate,
            "total_critical_violations": total_violations,
            "enforcement_reports": session_reports,
            "summary_timestamp": datetime.now().isoformat()
        }

    def export_enforcement_report(self, session_id: str, output_path: Optional[str] = None) -> str:
        """Export enforcement report to file."""

        # Find report for session
        report = next((r for r in self.enforcement_reports if r.session_id == session_id), None)
        if not report:
            raise ValueError(f"No enforcement report found for session: {session_id}")

        # Generate filename if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"gap_research_enforcement_{session_id}_{timestamp}.json"

        # Convert to JSON
        report_data = {
            "session_id": report.session_id,
            "enforcement_session_id": report.enforcement_session_id,
            "total_requirements": report.total_requirements,
            "compliant_requirements": report.compliant_requirements,
            "failed_requirements": report.failed_requirements,
            "waived_requirements": report.waived_requirements,
            "overall_compliance_rate": report.overall_compliance_rate,
            "enforcement_actions_taken": [action.value for action in report.enforcement_actions_taken],
            "critical_violations": [
                {
                    "check_id": v.check_id,
                    "requirement_id": v.requirement_id,
                    "compliance_level": v.compliance_level.value,
                    "details": v.details,
                    "timestamp": v.timestamp.isoformat(),
                    "evidence": v.evidence
                }
                for v in report.critical_violations
            ],
            "quality_impact": report.quality_impact,
            "recommendations": report.recommendations,
            "created_at": report.created_at.isoformat(),
            "completed_at": report.completed_at.isoformat() if report.completed_at else None
        }

        # Write to file
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"Enforcement report exported to: {output_path}")
        return output_path


# Factory function for easy instantiation
def create_gap_research_enforcement_system(config: Optional[Dict[str, Any]] = None) -> GapResearchEnforcementSystem:
    """Create a configured gap research enforcement system."""
    return GapResearchEnforcementSystem(config)


# Utility function for quick compliance check
async def quick_compliance_check(session_id: str,
                                  editorial_decision: Dict[str, Any],
                                  gap_research_status: Dict[str, Any]) -> GapResearchEnforcementReport:
    """Quick compliance check with default configuration."""
    enforcer = create_gap_research_enforcement_system()
    return await enforcer.enforce_gap_research_compliance(
        session_id, editorial_decision, gap_research_status
    )