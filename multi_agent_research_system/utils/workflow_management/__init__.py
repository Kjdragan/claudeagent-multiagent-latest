"""
Phase 1.5: Enhanced Success Tracking and Early Termination System

This module implements intelligent success tracking, early termination logic, and
simple lifecycle management for the multi-agent research system.

Key Components:
- EnhancedSuccessTracker: Comprehensive failure analysis and pattern tracking
- WorkflowState: Early termination logic with target achievement detection
- SimpleLifecycleManager: Task coordination and lifecycle management
- TaskResult: Comprehensive task result tracking with failure analysis

Based on Technical Enhancements Section 6 & 7:
- Enhanced Success Tracking & Resource Management
- Lifecycle Management & Edge Cases
"""

from .success_tracker import (
    TaskResult,
    EnhancedSuccessTracker,
    FailureAnalysis,
    SuccessMetrics,
    PerformancePattern
)

from .workflow_state import (
    WorkflowState,
    TargetDefinition,
    TerminationCriteria,
    StateTransition
)

from .lifecycle_manager import (
    SimpleLifecycleManager,
    TaskLifecycle,
    LifecycleEvent,
    LifecycleStage
)

from .integration import (
    WorkflowIntegrationMixin,
    OrchestratorIntegration
)

__all__ = [
    # Success Tracking
    'TaskResult',
    'EnhancedSuccessTracker',
    'FailureAnalysis',
    'SuccessMetrics',
    'PerformancePattern',

    # Workflow State Management
    'WorkflowState',
    'TargetDefinition',
    'TerminationCriteria',
    'StateTransition',

    # Lifecycle Management
    'SimpleLifecycleManager',
    'TaskLifecycle',
    'LifecycleEvent',
    'LifecycleStage',

    # Integration
    'WorkflowIntegrationMixin',
    'OrchestratorIntegration'
]

# Version information
__version__ = "1.5.0"
__phase__ = "Phase 1.5: Enhanced Success Tracking and Early Termination"