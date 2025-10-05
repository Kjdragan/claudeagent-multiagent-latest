"""
Workflow State Management for Multi-Agent Research System

Provides comprehensive workflow state persistence and recovery capabilities.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class WorkflowStage(Enum):
    """Workflow stages in the research pipeline."""
    RESEARCH = "research"
    REPORT_GENERATION = "report_generation"
    EDITORIAL_REVIEW = "editorial_review"
    DECOUPLED_EDITORIAL_REVIEW = "decoupled_editorial_review"
    QUALITY_ASSESSMENT = "quality_assessment"
    PROGRESSIVE_ENHANCEMENT = "progressive_enhancement"
    FINAL_OUTPUT = "final_output"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(Enum):
    """Status of a workflow stage."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageState:
    """State information for a single workflow stage."""
    stage: WorkflowStage
    status: StageStatus
    start_time: datetime | None = None
    end_time: datetime | None = None
    attempt_count: int = 0
    max_attempts: int = 3
    result: dict[str, Any] | None = None
    error_message: str | None = None
    checkpoint_data: dict[str, Any] | None = None
    recovery_attempts: list[dict[str, Any]] = field(default_factory=list)

    # Enhanced tracking for quality and editorial processes
    quality_metrics: dict[str, Any] | None = None
    editorial_metadata: dict[str, Any] | None = None
    enhancement_log: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    processing_path: list[str] = field(default_factory=list)  # Track which processing paths were taken

    @property
    def duration(self) -> timedelta | None:
        """Calculate duration of stage execution."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def is_finished(self) -> bool:
        """Check if stage is finished (completed, failed, or skipped)."""
        return self.status in [StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.SKIPPED]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        # Convert enums to strings
        data['stage'] = self.stage.value
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'StageState':
        """Create from dictionary."""
        # Convert string enums back to enums
        data['stage'] = WorkflowStage(data['stage'])
        data['status'] = StageStatus(data['status'])
        # Convert ISO strings back to datetime objects
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**data)


@dataclass
class WorkflowSession:
    """Complete workflow session state."""
    session_id: str
    topic: str
    user_requirements: dict[str, Any]
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    current_stage: WorkflowStage = WorkflowStage.RESEARCH
    overall_status: StageStatus = StageStatus.PENDING
    stages: dict[WorkflowStage, StageState] = field(default_factory=dict)
    global_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Enhanced tracking for quality and editorial workflows
    quality_history: list[dict[str, Any]] = field(default_factory=list)
    editorial_path: list[str] = field(default_factory=list)  # Track editorial processing path
    enhancement_stages_applied: list[str] = field(default_factory=list)
    final_quality_score: int | None = None
    processing_statistics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize stage states after creation."""
        if not self.stages:
            # Initialize all stages
            for stage in WorkflowStage:
                if stage not in [WorkflowStage.COMPLETED, WorkflowStage.FAILED]:
                    self.stages[stage] = StageState(
                        stage=stage,
                        status=StageStatus.PENDING
                    )

    @property
    def duration(self) -> timedelta | None:
        """Calculate total workflow duration."""
        if self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return None

    @property
    def completed_stages(self) -> list[WorkflowStage]:
        """Get list of completed stages."""
        return [
            stage for stage, state in self.stages.items()
            if state.status == StageStatus.COMPLETED
        ]

    @property
    def failed_stages(self) -> list[WorkflowStage]:
        """Get list of failed stages."""
        return [
            stage for stage, state in self.stages.items()
            if state.status == StageStatus.FAILED
        ]

    @property
    def is_completed(self) -> bool:
        """Check if entire workflow is completed."""
        return self.overall_status == StageStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if entire workflow has failed."""
        return self.overall_status == StageStatus.FAILED

    def get_stage_state(self, stage: WorkflowStage) -> StageState:
        """Get state for a specific stage."""
        return self.stages.get(stage, StageState(stage=stage, status=StageStatus.PENDING))

    def update_stage_state(self, stage: WorkflowStage, **kwargs):
        """Update state for a specific stage."""
        if stage not in self.stages:
            self.stages[stage] = StageState(stage=stage, status=StageStatus.PENDING)

        stage_state = self.stages[stage]
        for key, value in kwargs.items():
            if hasattr(stage_state, key):
                setattr(stage_state, key, value)

        # Update current stage if this stage is now in progress
        if kwargs.get('status') == StageStatus.IN_PROGRESS:
            self.current_stage = stage
            if not stage_state.start_time:
                stage_state.start_time = datetime.now()

        # Update overall status if stage completed or failed
        if kwargs.get('status') == StageStatus.COMPLETED:
            stage_state.end_time = datetime.now()
            self.current_stage = self._get_next_stage(stage)
        elif kwargs.get('status') == StageStatus.FAILED:
            stage_state.end_time = datetime.now()
            stage_state.error_message = kwargs.get('error_message', 'Unknown error')

    def _get_next_stage(self, current_stage: WorkflowStage) -> WorkflowStage:
        """Get the next stage in the workflow."""
        stage_order = [
            WorkflowStage.RESEARCH,
            WorkflowStage.REPORT_GENERATION,
            WorkflowStage.EDITORIAL_REVIEW,
            WorkflowStage.DECOUPLED_EDITORIAL_REVIEW,
            WorkflowStage.QUALITY_ASSESSMENT,
            WorkflowStage.PROGRESSIVE_ENHANCEMENT,
            WorkflowStage.FINAL_OUTPUT
        ]

        try:
            current_index = stage_order.index(current_stage)
            if current_index + 1 < len(stage_order):
                return stage_order[current_index + 1]
        except ValueError:
            pass

        return WorkflowStage.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        # Convert enums
        data['current_stage'] = self.current_stage.value
        data['overall_status'] = self.overall_status.value
        # Convert stage states
        data['stages'] = {
            stage.value: state.to_dict()
            for stage, state in self.stages.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'WorkflowSession':
        """Create from dictionary."""
        # Convert datetime objects
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        # Convert enums
        data['current_stage'] = WorkflowStage(data['current_stage'])
        data['overall_status'] = StageStatus(data['overall_status'])
        # Convert stage states
        stages_data = data.pop('stages', {})
        workflow_session = cls(**data)

        for stage_str, stage_dict in stages_data.items():
            stage = WorkflowStage(stage_str)
            workflow_session.stages[stage] = StageState.from_dict(stage_dict)

        return workflow_session


class WorkflowStateManager:
    """Manages workflow state persistence and recovery."""

    def __init__(self, base_path: str = "KEVIN/sessions", logger: logging.Logger | None = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.active_sessions: dict[str, WorkflowSession] = {}

    def create_session(
        self,
        session_id: str,
        topic: str,
        user_requirements: dict[str, Any]
    ) -> WorkflowSession:
        """Create a new workflow session."""
        session = WorkflowSession(
            session_id=session_id,
            topic=topic,
            user_requirements=user_requirements
        )

        self.active_sessions[session_id] = session
        self.save_checkpoint(session_id, "session_created", {"session_data": session.to_dict()})

        self.logger.info(f"Created workflow session: {session_id}")
        return session

    def get_session(self, session_id: str) -> WorkflowSession | None:
        """Get active session or load from disk."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # Try to load from disk
        session = self.load_session(session_id)
        if session:
            self.active_sessions[session_id] = session

        return session

    def save_checkpoint(self, session_id: str, stage: str, data: dict[str, Any]):
        """Save workflow state checkpoint."""
        session_dir = self.base_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = session_dir / "workflow_state.json"

        try:
            # Load existing state if any
            if checkpoint_file.exists():
                with open(checkpoint_file, encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
            else:
                checkpoint_data = {
                    "session_id": session_id,
                    "checkpoints": {},
                    "created_at": datetime.now().isoformat()
                }

            # Add new checkpoint
            checkpoint_data["checkpoints"][stage] = {
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            checkpoint_data["last_updated"] = datetime.now().isoformat()

            # Save updated state
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Saved checkpoint for {session_id} at stage: {stage}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for {session_id}: {e}")

    def load_checkpoint(self, session_id: str, stage: str) -> dict[str, Any] | None:
        """Load specific checkpoint data."""
        checkpoint_file = self.base_path / session_id / "workflow_state.json"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            return checkpoint_data["checkpoints"].get(stage, {}).get("data")

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint for {session_id}, stage {stage}: {e}")
            return None

    def can_resume_from_stage(self, session_id: str, stage: str) -> bool:
        """Check if workflow can resume from a specific stage."""
        checkpoint = self.load_checkpoint(session_id, stage)
        return checkpoint is not None

    def get_completed_stages(self, session_id: str) -> list[str]:
        """Get list of completed stages for a session."""
        checkpoint_file = self.base_path / session_id / "workflow_state.json"

        if not checkpoint_file.exists():
            return []

        try:
            with open(checkpoint_file, encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            completed_stages = []
            for stage_name, checkpoint in checkpoint_data.get("checkpoints", {}).items():
                if checkpoint.get("data", {}).get("success", False):
                    completed_stages.append(stage_name)

            return completed_stages

        except Exception as e:
            self.logger.error(f"Failed to get completed stages for {session_id}: {e}")
            return []

    def save_session(self, session: WorkflowSession):
        """Save complete session state."""
        session_dir = self.base_path / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        session_file = session_dir / "session_state.json"

        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Saved session state: {session.session_id}")

        except Exception as e:
            self.logger.error(f"Failed to save session {session.session_id}: {e}")

    def load_session(self, session_id: str) -> WorkflowSession | None:
        """Load complete session state from disk."""
        session_file = self.base_path / session_id / "session_state.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file, encoding='utf-8') as f:
                session_data = json.load(f)

            session = WorkflowSession.from_dict(session_data)
            self.logger.info(f"Loaded session state: {session_id}")
            return session

        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def delete_session(self, session_id: str):
        """Delete session data from disk and memory."""
        try:
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            # Remove from disk
            session_dir = self.base_path / session_id
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)

            self.logger.info(f"Deleted session: {session_id}")

        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")

    def cleanup_old_sessions(self, days_old: int = 7):
        """Clean up old session data."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0

        try:
            for session_dir in self.base_path.iterdir():
                if session_dir.is_dir():
                    session_file = session_dir / "session_state.json"
                    if session_file.exists():
                        try:
                            with open(session_file, encoding='utf-8') as f:
                                session_data = json.load(f)

                            start_time = datetime.fromisoformat(session_data.get("start_time"))
                            if start_time < cutoff_date:
                                self.delete_session(session_dir.name)
                                cleaned_count += 1
                        except:
                            # If we can't read the session, try to remove it
                            import shutil
                            shutil.rmtree(session_dir)
                            cleaned_count += 1

            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old sessions")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {e}")

    def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get summary of session state."""
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "topic": session.topic,
            "duration": str(session.duration) if session.duration else None,
            "current_stage": session.current_stage.value,
            "overall_status": session.overall_status.value,
            "completed_stages": [stage.value for stage in session.completed_stages],
            "failed_stages": [stage.value for stage in session.failed_stages],
            "total_stages": len(session.stages),
            "progress_percentage": len(session.completed_stages) / len(session.stages) * 100
        }
