"""
Audit Trail Manager for compliance and audit trail features.

This module provides comprehensive audit trail capabilities including
immutable logging, compliance reporting, and security event tracking.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from .log_aggregator import LogEntry


class AuditEventType(Enum):
    """Types of audit events."""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR_EVENT = "error_event"
    AGENT_HANDOFF = "agent_handoff"
    TOOL_EXECUTION = "tool_execution"
    WORKFLOW_STAGE = "workflow_stage"


class ComplianceStandard(Enum):
    """Compliance standards that can be tracked."""
    GDPR = "gdpr"
    SOX = "sox"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    CUSTOM = "custom"


@dataclass
class AuditEvent:
    """Comprehensive audit event structure."""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    actor: Optional[str]  # User, agent, or system component
    action: str
    resource: Optional[str]  # Resource that was acted upon
    outcome: str  # 'success', 'failure', 'partial'
    details: Dict[str, Any]
    session_id: str
    correlation_id: Optional[str]
    source_ip: Optional[str]
    user_agent: Optional[str]
    compliance_tags: List[str]
    data_classification: Optional[str]  # 'public', 'internal', 'confidential', 'restricted'
    retention_period_days: Optional[int]
    checksum: str  # For integrity verification

    def __post_init__(self):
        if self.compliance_tags is None:
            self.compliance_tags = []


@dataclass
class ComplianceReport:
    """Compliance report structure."""
    report_id: str
    standard: ComplianceStandard
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    total_events: int
    compliance_score: float  # 0-100
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    evidence: Dict[str, Any]


class AuditTrailManager:
    """Comprehensive audit trail management system."""

    def __init__(self,
                 session_id: str,
                 audit_dir: str = "audit_trail",
                 retention_days: int = 2555,  # 7 years default
                 enable_integrity_checks: bool = True):
        """
        Initialize the audit trail manager.

        Args:
            session_id: Session identifier
            audit_dir: Directory to store audit data
            retention_days: Default retention period for audit events
            enable_integrity_checks: Whether to enable integrity verification
        """
        self.session_id = session_id
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.enable_integrity_checks = enable_integrity_checks

        # Audit event storage
        self.audit_events: List[AuditEvent] = []
        self.event_index: Dict[str, Set[int]] = {
            'event_type': set(),
            'actor': set(),
            'session_id': set(),
            'correlation_id': set(),
            'compliance_tags': set(),
            'data_classification': set()
        }

        # Compliance tracking
        self.compliance_standards: Dict[ComplianceStandard, Dict[str, Any]] = {}
        self.violations: List[Dict[str, Any]] = []

        # Security monitoring
        self.security_events: List[Dict[str, Any]] = []
        self.suspicious_activities: List[Dict[str, Any]] = []

        # Data retention policies
        self.retention_policies: Dict[str, int] = {
            'security_event': retention_days * 2,  # Keep security events longer
            'user_action': retention_days,
            'system_event': retention_days // 2,
            'error_event': retention_days // 4
        }

        # Immutable audit log
        self.immutable_log_file = self.audit_dir / "immutable_audit.log"
        self._initialize_immutable_log()

    def _initialize_immutable_log(self) -> None:
        """Initialize the immutable audit log."""
        if not self.immutable_log_file.exists():
            with open(self.immutable_log_file, 'w') as f:
                f.write(f"# Immutable Audit Log - Initialized {datetime.now().isoformat()}\n")

    def log_audit_event(self,
                       event_type: AuditEventType,
                       action: str,
                       actor: Optional[str] = None,
                       resource: Optional[str] = None,
                       outcome: str = "success",
                       details: Optional[Dict[str, Any]] = None,
                       correlation_id: Optional[str] = None,
                       source_ip: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       compliance_tags: Optional[List[str]] = None,
                       data_classification: str = "internal",
                       retention_period_days: Optional[int] = None) -> str:
        """
        Log an audit event.

        Args:
            event_type: Type of audit event
            action: Action performed
            actor: Entity that performed the action
            resource: Resource that was acted upon
            outcome: Outcome of the action
            details: Additional event details
            correlation_id: Correlation ID for related events
            source_ip: Source IP address
            user_agent: User agent string
            compliance_tags: Compliance-related tags
            data_classification: Data classification level
            retention_period_days: Custom retention period

        Returns:
            Event ID
        """
        event_id = self._generate_event_id()
        timestamp = datetime.now()

        event_details = details or {}

        # Create audit event
        audit_event = AuditEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            details=event_details,
            session_id=self.session_id,
            correlation_id=correlation_id,
            source_ip=source_ip,
            user_agent=user_agent,
            compliance_tags=compliance_tags or [],
            data_classification=data_classification,
            retention_period_days=retention_period_days or self.retention_days,
            checksum=self._calculate_checksum(event_id, timestamp, action, event_details)
        )

        # Add to storage
        self.audit_events.append(audit_event)
        self._update_indexes(audit_event)

        # Write to immutable log
        self._write_to_immutable_log(audit_event)

        # Check for security concerns
        self._check_security_concerns(audit_event)

        return event_id

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return str(uuid.uuid4())

    def _calculate_checksum(self, event_id: str, timestamp: datetime, action: str, details: Dict[str, Any]) -> str:
        """Calculate checksum for integrity verification."""
        data = f"{event_id}{timestamp.isoformat()}{action}{json.dumps(details, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _update_indexes(self, event: AuditEvent) -> None:
        """Update search indexes for the event."""
        event_index = len(self.audit_events) - 1

        self.event_index['event_type'].add((event.event_type.value, event_index))
        if event.actor:
            self.event_index['actor'].add((event.actor, event_index))
        self.event_index['session_id'].add((event.session_id, event_index))
        if event.correlation_id:
            self.event_index['correlation_id'].add((event.correlation_id, event_index))
        for tag in event.compliance_tags:
            self.event_index['compliance_tags'].add((tag, event_index))
        self.event_index['data_classification'].add((event.data_classification, event_index))

    def _write_to_immutable_log(self, event: AuditEvent) -> None:
        """Write event to immutable audit log."""
        log_entry = {
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type.value,
            'actor': event.actor,
            'action': event.action,
            'resource': event.resource,
            'outcome': event.outcome,
            'details': event.details,
            'session_id': event.session_id,
            'correlation_id': event.correlation_id,
            'source_ip': event.source_ip,
            'user_agent': event.user_agent,
            'compliance_tags': event.compliance_tags,
            'data_classification': event.data_classification,
            'checksum': event.checksum
        }

        with open(self.immutable_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _check_security_concerns(self, event: AuditEvent) -> None:
        """Check event for security concerns."""
        security_concerns = []

        # Check for failed authentication attempts
        if "login" in event.action.lower() and event.outcome == "failure":
            security_concerns.append("Failed authentication attempt")

        # Check for privilege escalation
        if "admin" in event.action.lower() or "privilege" in event.action.lower():
            security_concerns.append("Privilege escalation activity")

        # Check for data access violations
        if event.data_classification == "restricted" and event.outcome == "success":
            security_concerns.append("Access to restricted data")

        # Check for unusual activity patterns
        if self._detect_unusual_activity(event):
            security_concerns.append("Unusual activity pattern detected")

        # Log security events
        if security_concerns:
            security_event = {
                'timestamp': event.timestamp,
                'event_id': event.event_id,
                'concerns': security_concerns,
                'actor': event.actor,
                'source_ip': event.source_ip,
                'session_id': event.session_id
            }
            self.security_events.append(security_event)

    def _detect_unusual_activity(self, event: AuditEvent) -> bool:
        """Detect unusual activity patterns."""
        # Simple heuristic: check for rapid successive actions
        recent_events = [e for e in self.audit_events[-10:] if e.actor == event.actor]
        if len(recent_events) > 5:
            time_diffs = [(event.timestamp - e.timestamp).total_seconds() for e in recent_events[-5:]]
            if all(diff < 1.0 for diff in time_diffs):
                return True

        return False

    def search_audit_trail(self,
                          event_type: Optional[AuditEventType] = None,
                          actor: Optional[str] = None,
                          session_id: Optional[str] = None,
                          correlation_id: Optional[str] = None,
                          compliance_tag: Optional[str] = None,
                          data_classification: Optional[str] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: Optional[int] = None) -> List[AuditEvent]:
        """
        Search audit trail with various filters.

        Args:
            event_type: Filter by event type
            actor: Filter by actor
            session_id: Filter by session ID
            correlation_id: Filter by correlation ID
            compliance_tag: Filter by compliance tag
            data_classification: Filter by data classification
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum number of results

        Returns:
            Filtered list of audit events
        """
        events = self.audit_events

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if actor:
            events = [e for e in events if e.actor == actor]

        if session_id:
            events = [e for e in events if e.session_id == session_id]

        if correlation_id:
            events = [e for e in events if e.correlation_id == correlation_id]

        if compliance_tag:
            events = [e for e in events if compliance_tag in e.compliance_tags]

        if data_classification:
            events = [e for e in events if e.data_classification == data_classification]

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # Sort by timestamp (most recent first)
        events.sort(key=lambda x: x.timestamp, reverse=True)

        # Apply limit
        if limit:
            events = events[:limit]

        return events

    def verify_integrity(self, event_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify integrity of audit trail.

        Args:
            event_id: Specific event ID to verify, or None for all events

        Returns:
            Integrity verification results
        """
        verification_results = {
            'timestamp': datetime.now().isoformat(),
            'total_events_verified': 0,
            'integrity_violations': [],
            'verification_passed': True
        }

        events_to_verify = self.audit_events
        if event_id:
            events_to_verify = [e for e in self.audit_events if e.event_id == event_id]

        for event in events_to_verify:
            # Recalculate checksum
            expected_checksum = self._calculate_checksum(
                event.event_id, event.timestamp, event.action, event.details
            )

            if event.checksum != expected_checksum:
                violation = {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'expected_checksum': expected_checksum,
                    'actual_checksum': event.checksum,
                    'violation_type': 'checksum_mismatch'
                }
                verification_results['integrity_violations'].append(violation)
                verification_results['verification_passed'] = False

            verification_results['total_events_verified'] += 1

        return verification_results

    async def generate_compliance_report(self,
                                       standard: ComplianceStandard,
                                       period_start: datetime,
                                       period_end: datetime) -> ComplianceReport:
        """
        Generate compliance report for a specific standard and period.

        Args:
            standard: Compliance standard
            period_start: Start of reporting period
            period_end: End of reporting period

        Returns:
            Compliance report
        """
        # Get events for the period
        period_events = [
            e for e in self.audit_events
            if period_start <= e.timestamp <= period_end
        ]

        # Standard-specific compliance checks
        if standard == ComplianceStandard.GDPR:
            violations, score = self._check_gdpr_compliance(period_events)
        elif standard == ComplianceStandard.SOX:
            violations, score = self._check_sox_compliance(period_events)
        elif standard == ComplianceStandard.HIPAA:
            violations, score = self._check_hipaa_compliance(period_events)
        else:
            violations, score = self._check_custom_compliance(period_events)

        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(standard, violations)

        # Collect evidence
        evidence = self._collect_compliance_evidence(standard, period_events)

        report = ComplianceReport(
            report_id=self._generate_event_id(),
            standard=standard,
            period_start=period_start,
            period_end=period_end,
            generated_at=datetime.now(),
            total_events=len(period_events),
            compliance_score=score,
            violations=violations,
            recommendations=recommendations,
            evidence=evidence
        )

        return report

    def _check_gdpr_compliance(self, events: List[AuditEvent]) -> Tuple[List[Dict[str, Any]], float]:
        """Check GDPR compliance."""
        violations = []

        # Check for data access logging
        data_access_events = [e for e in events if e.event_type == AuditEventType.DATA_ACCESS]
        if not data_access_events:
            violations.append({
                'type': 'missing_data_access_logs',
                'description': 'No data access events found',
                'severity': 'high'
            })

        # Check for user consent tracking
        consent_events = [e for e in events if 'consent' in e.action.lower()]
        if not consent_events:
            violations.append({
                'type': 'missing_consent_tracking',
                'description': 'No user consent tracking events found',
                'severity': 'medium'
            })

        # Check for data retention compliance
        retention_violations = 0
        for event in events:
            age_days = (datetime.now() - event.timestamp).days
            if age_days > event.retention_period_days:
                retention_violations += 1

        if retention_violations > 0:
            violations.append({
                'type': 'data_retention_violation',
                'description': f'{retention_violations} events exceed retention period',
                'severity': 'high'
            })

        # Calculate compliance score
        base_score = 100.0
        base_score -= len(violations) * 10  # Deduct 10 points per violation
        score = max(0, base_score)

        return violations, score

    def _check_sox_compliance(self, events: List[AuditEvent]) -> Tuple[List[Dict[str, Any]], float]:
        """Check SOX compliance."""
        violations = []

        # Check for financial data access
        financial_events = [e for e in events if 'financial' in e.action.lower() or 'financial' in str(e.resource).lower()]
        for event in financial_events:
            if event.actor is None:
                violations.append({
                    'type': 'unauthorized_financial_access',
                    'description': f'Financial data access without actor attribution: {event.event_id}',
                    'severity': 'high'
                })

        # Check for change management
        change_events = [e for e in events if e.event_type == AuditEventType.CONFIGURATION_CHANGE]
        if not change_events:
            violations.append({
                'type': 'missing_change_management',
                'description': 'No configuration change events found',
                'severity': 'medium'
            })

        # Calculate compliance score
        base_score = 100.0
        base_score -= len(violations) * 15  # SOX violations are more serious
        score = max(0, base_score)

        return violations, score

    def _check_hipaa_compliance(self, events: List[AuditEvent]) -> Tuple[List[Dict[str, Any]], float]:
        """Check HIPAA compliance."""
        violations = []

        # Check for PHI access logging
        phi_events = [e for e in events if e.data_classification in ['confidential', 'restricted']]
        for event in phi_events:
            if not event.source_ip:
                violations.append({
                    'type': 'missing_source_ip',
                    'description': f'PHI access without source IP: {event.event_id}',
                    'severity': 'medium'
                })

        # Calculate compliance score
        base_score = 100.0
        base_score -= len(violations) * 12
        score = max(0, base_score)

        return violations, score

    def _check_custom_compliance(self, events: List[AuditEvent]) -> Tuple[List[Dict[str, Any]], float]:
        """Check custom compliance rules."""
        violations = []

        # Add custom compliance checks here
        # For now, basic checks
        error_events = [e for e in events if e.outcome == 'failure']
        if len(error_events) > len(events) * 0.1:  # More than 10% failure rate
            violations.append({
                'type': 'high_failure_rate',
                'description': f'High failure rate: {len(error_events)}/{len(events)}',
                'severity': 'medium'
            })

        base_score = 100.0
        base_score -= len(violations) * 8
        score = max(0, base_score)

        return violations, score

    def _generate_compliance_recommendations(self, standard: ComplianceStandard, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations based on violations."""
        recommendations = []

        violation_types = [v['type'] for v in violations]

        if 'missing_data_access_logs' in violation_types:
            recommendations.append("Implement comprehensive data access logging")

        if 'missing_consent_tracking' in violation_types:
            recommendations.append("Implement user consent tracking and management")

        if 'data_retention_violation' in violation_types:
            recommendations.append("Review and implement data retention policies")

        if 'unauthorized_financial_access' in violation_types:
            recommendations.append("Strengthen access controls for financial data")

        if 'missing_change_management' in violation_types:
            recommendations.append("Implement formal change management processes")

        if 'missing_source_ip' in violation_types:
            recommendations.append("Ensure source IP logging for all sensitive data access")

        if not recommendations:
            recommendations.append("Continue monitoring for compliance improvements")

        return recommendations

    def _collect_compliance_evidence(self, standard: ComplianceStandard, events: List[AuditEvent]) -> Dict[str, Any]:
        """Collect evidence for compliance reporting."""
        evidence = {
            'total_events': len(events),
            'events_by_type': {},
            'events_by_outcome': {},
            'data_classification_breakdown': {},
            'top_actors': {},
            'sample_events': []
        }

        # Event type breakdown
        for event in events:
            event_type = event.event_type.value
            evidence['events_by_type'][event_type] = evidence['events_by_type'].get(event_type, 0) + 1

        # Outcome breakdown
        for event in events:
            outcome = event.outcome
            evidence['events_by_outcome'][outcome] = evidence['events_by_outcome'].get(outcome, 0) + 1

        # Data classification breakdown
        for event in events:
            classification = event.data_classification
            evidence['data_classification_breakdown'][classification] = evidence['data_classification_breakdown'].get(classification, 0) + 1

        # Top actors
        actor_counts = {}
        for event in events:
            if event.actor:
                actor_counts[event.actor] = actor_counts.get(event.actor, 0) + 1

        evidence['top_actors'] = dict(sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        # Sample events (first 10)
        evidence['sample_events'] = [
            {
                'event_id': e.event_id,
                'timestamp': e.timestamp.isoformat(),
                'event_type': e.event_type.value,
                'actor': e.actor,
                'action': e.action,
                'outcome': e.outcome
            }
            for e in events[:10]
        ]

        return evidence

    async def cleanup_expired_events(self) -> Dict[str, Any]:
        """Clean up expired audit events based on retention policies."""
        cleanup_start = datetime.now()
        events_removed = 0
        events_kept = 0

        current_events = []
        for event in self.audit_events:
            age_days = (cleanup_start - event.timestamp).days
            retention_period = event.retention_period_days or self.retention_days

            # Apply specific retention policies
            for policy_type, policy_retention in self.retention_policies.items():
                if policy_type in event.event_type.value.lower():
                    retention_period = policy_retention
                    break

            if age_days <= retention_period:
                current_events.append(event)
                events_kept += 1
            else:
                events_removed += 1

        self.audit_events = current_events
        self._rebuild_indexes()

        cleanup_stats = {
            'cleanup_timestamp': cleanup_start.isoformat(),
            'events_removed': events_removed,
            'events_kept': events_kept,
            'cleanup_duration_seconds': (datetime.now() - cleanup_start).total_seconds()
        }

        return cleanup_stats

    def _rebuild_indexes(self) -> None:
        """Rebuild search indexes after cleanup."""
        self.event_index = {
            'event_type': set(),
            'actor': set(),
            'session_id': set(),
            'correlation_id': set(),
            'compliance_tags': set(),
            'data_classification': set()
        }

        for i, event in enumerate(self.audit_events):
            self._update_indexes(event)

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get comprehensive audit summary."""
        return {
            'session_id': self.session_id,
            'total_events': len(self.audit_events),
            'security_events': len(self.security_events),
            'compliance_violations': len(self.violations),
            'events_by_type': {
                event_type.value: len([e for e in self.audit_events if e.event_type == event_type])
                for event_type in AuditEventType
            },
            'events_by_outcome': {
                outcome: len([e for e in self.audit_events if e.outcome == outcome])
                for outcome in ['success', 'failure', 'partial']
            },
            'data_classification_breakdown': {
                classification: len([e for e in self.audit_events if e.data_classification == classification])
                for classification in ['public', 'internal', 'confidential', 'restricted']
            },
            'audit_directory': str(self.audit_dir),
            'immutable_log_exists': self.immutable_log_file.exists(),
            'retention_policy_days': self.retention_days
        }

    def export_audit_trail(self,
                           file_path: Optional[str] = None,
                           include_checksums: bool = True) -> str:
        """Export audit trail to file."""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self.audit_dir / f"audit_trail_export_{self.session_id}_{timestamp}.json")

        export_data = {
            'session_id': self.session_id,
            'export_timestamp': datetime.now().isoformat(),
            'audit_summary': self.get_audit_summary(),
            'audit_events': [],
            'security_events': self.security_events,
            'compliance_violations': self.violations
        }

        for event in self.audit_events:
            event_data = asdict(event)
            if not include_checksums:
                event_data.pop('checksum', None)
            export_data['audit_events'].append(event_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)

        return file_path