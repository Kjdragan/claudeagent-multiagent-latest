"""
Sub-Agent Coordinator

This module provides high-level orchestration and coordination for sub-agents,
managing complex workflows, task distribution, and inter-agent communication.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from .sub_agent_factory import (
    SubAgentFactory, SubAgentRequest, SubAgentResult, get_sub_agent_factory
)
from .sub_agent_types import SubAgentType, create_sub_agent_config
from .communication_protocols import (
    SubAgentCommunicationManager, MessageType, MessagePriority, SubAgentMessage
)
from .context_isolation import ContextIsolationManager
from .performance_monitor import SubAgentPerformanceMonitor


logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Stages in a coordinated sub-agent workflow."""

    INITIALIZATION = "initialization"
    RESEARCH = "research"
    REPORT_GENERATION = "report_generation"
    EDITORIAL_REVIEW = "editorial_review"
    GAP_RESEARCH = "gap_research"
    QUALITY_ASSESSMENT = "quality_assessment"
    CONTENT_ENHANCEMENT = "content_enhancement"
    STYLE_EDITING = "style_editing"
    FINALIZATION = "finalization"


class WorkflowStatus(Enum):
    """Status of a coordinated workflow."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_SUB_AGENTS = "waiting_for_sub_agents"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowTask:
    """Individual task within a coordinated workflow."""

    task_id: str
    workflow_id: str
    stage: WorkflowStage
    agent_type: SubAgentType
    task_description: str
    task_data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # task_ids this task depends on
    priority: int = 3
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"  # pending, assigned, in_progress, completed, failed
    assigned_agent_id: Optional[str] = None
    result: Optional[SubAgentResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def is_ready(self, completed_tasks: set[str]) -> bool:
        """Check if this task is ready to be executed."""
        return all(dep in completed_tasks for dep in self.dependencies)

    def can_retry(self) -> bool:
        """Check if this task can be retried."""
        return self.retry_count < self.max_retries

    def get_duration(self) -> Optional[float]:
        """Get the duration of this task in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class CoordinatedWorkflow:
    """A coordinated workflow involving multiple sub-agents."""

    workflow_id: str
    session_id: str
    topic: str
    description: str
    stages: List[WorkflowStage]
    tasks: List[WorkflowTask] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_stage: Optional[WorkflowStage] = None
    completed_tasks: set[str] = field(default_factory=set)
    failed_tasks: set[str] = field(default_factory=set)
    workflow_data: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    context_isolation: str = "moderate"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_task_by_id(self, task_id: str) -> Optional[WorkflowTask]:
        """Get a task by its ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_tasks_by_stage(self, stage: WorkflowStage) -> List[WorkflowTask]:
        """Get all tasks for a specific stage."""
        return [task for task in self.tasks if task.stage == stage]

    def get_ready_tasks(self) -> List[WorkflowTask]:
        """Get tasks that are ready to be executed."""
        return [
            task for task in self.tasks
            if task.status == "pending" and task.is_ready(self.completed_tasks)
        ]

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow."""
        total_tasks = len(self.tasks)
        completed_tasks = len(self.completed_tasks)
        failed_tasks = len(self.failed_tasks)

        duration = None
        if self.started_at:
            end_time = self.completed_at or datetime.now()
            duration = (end_time - self.started_at).total_seconds()

        return {
            "workflow_id": self.workflow_id,
            "session_id": self.session_id,
            "topic": self.topic,
            "status": self.status.value,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "progress_percentage": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "duration_seconds": duration,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class SubAgentCoordinator:
    """
    Coordinates and orchestrates sub-agent workflows, managing task distribution,
    inter-agent communication, and workflow progression.
    """

    def __init__(self):
        self.factory = get_sub_agent_factory()
        self.communication_manager = SubAgentCommunicationManager()
        self.isolation_manager = ContextIsolationManager()
        self.performance_monitor = SubAgentPerformanceMonitor()

        self.active_workflows: Dict[str, CoordinatedWorkflow] = {}
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        self.coordination_config = {
            "max_concurrent_workflows": 10,
            "max_concurrent_tasks_per_workflow": 5,
            "task_timeout_seconds": 300,
            "workflow_timeout_hours": 2,
            "enable_auto_retry": True,
            "enable_quality_gates": True,
            "enable_progressive_enhancement": True,
            "cleanup_interval": 300,  # seconds
            "workflow_templates_enabled": True
        }
        self._coordination_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self):
        """Initialize the sub-agent coordinator."""
        logger.info("Initializing Sub-Agent Coordinator")
        self._running = True

        # Initialize components
        await self.factory.initialize()
        await self.communication_manager.initialize()
        await self.isolation_manager.initialize()
        await self.performance_monitor.initialize()

        # Register communication handlers
        await self._register_communication_handlers()

        # Setup workflow templates
        if self.coordination_config["workflow_templates_enabled"]:
            await self._setup_workflow_templates()

        # Start coordination tasks
        self._coordination_task = asyncio.create_task(self._coordination_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Sub-agent coordinator initialized")

    async def shutdown(self):
        """Shutdown the sub-agent coordinator."""
        logger.info("Shutting down Sub-Agent Coordinator")
        self._running = False

        # Cancel coordination tasks
        if self._coordination_task:
            self._coordination_task.cancel()
            try:
                await self._coordination_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel all active workflows
        for workflow_id in list(self.active_workflows.keys()):
            await self.cancel_workflow(workflow_id)

        # Shutdown components
        await self.communication_manager.shutdown()
        await self.isolation_manager.shutdown()
        await self.performance_monitor.shutdown()
        await self.factory.shutdown()

        logger.info("Sub-agent coordinator shutdown complete")

    async def create_coordinated_workflow(
        self,
        session_id: str,
        topic: str,
        description: str,
        workflow_type: str = "standard_research",
        stages: Optional[List[WorkflowStage]] = None,
        quality_requirements: Optional[Dict[str, float]] = None,
        context_isolation: str = "moderate",
        **kwargs
    ) -> str:
        """
        Create a new coordinated workflow.

        Args:
            session_id: Session ID for the workflow
            topic: Research topic
            description: Workflow description
            workflow_type: Type of workflow template to use
            stages: Custom stages (overrides template)
            quality_requirements: Quality requirements for the workflow
            context_isolation: Context isolation level
            **kwargs: Additional workflow parameters

        Returns:
            Workflow ID for the created workflow
        """

        # Check concurrent workflow limit
        if len(self.active_workflows) >= self.coordination_config["max_concurrent_workflows"]:
            await self._cleanup_completed_workflows()
            if len(self.active_workflows) >= self.coordination_config["max_concurrent_workflows"]:
                raise RuntimeError("Maximum concurrent workflows limit reached")

        # Generate workflow ID
        workflow_id = str(uuid.uuid4())

        # Determine stages
        if stages is None:
            stages = self._get_workflow_stages(workflow_type)

        # Create workflow
        workflow = CoordinatedWorkflow(
            workflow_id=workflow_id,
            session_id=session_id,
            topic=topic,
            description=description,
            stages=stages,
            quality_requirements=quality_requirements or {},
            context_isolation=context_isolation,
            metadata=kwargs
        )

        # Create tasks for each stage
        await self._create_workflow_tasks(workflow, workflow_type)

        # Store workflow
        self.active_workflows[workflow_id] = workflow

        logger.info(f"Created coordinated workflow {workflow_id} for session {session_id}")
        return workflow_id

    async def start_workflow(self, workflow_id: str) -> bool:
        """Start a coordinated workflow."""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.IN_PROGRESS
        workflow.started_at = datetime.now()

        logger.info(f"Started workflow {workflow_id}")
        return True

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a coordinated workflow."""
        if workflow_id not in self.active_workflows:
            return False

        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.CANCELLED
        workflow.completed_at = datetime.now()

        # Cancel active tasks
        for task in workflow.tasks:
            if task.status in ["assigned", "in_progress"]:
                task.status = "cancelled"
                if task.assigned_agent_id:
                    await self.factory.cleanup_instance(task.assigned_agent_id)

        logger.info(f"Cancelled workflow {workflow_id}")
        return True

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        if workflow_id not in self.active_workflows:
            return None

        workflow = self.active_workflows[workflow_id]
        return workflow.get_workflow_summary()

    async def get_workflow_results(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the results of a completed workflow."""
        if workflow_id not in self.active_workflows:
            return None

        workflow = self.active_workflows[workflow_id]

        if workflow.status != WorkflowStatus.COMPLETED:
            return None

        # Collect results from all tasks
        results = {}
        for task in workflow.tasks:
            if task.result:
                results[task.task_id] = {
                    "stage": task.stage.value,
                    "agent_type": task.agent_type.value,
                    "success": task.result.success,
                    "execution_time": task.result.execution_time,
                    "data": task.result.result_data,
                    "quality_score": task.result.quality_score
                }

        return {
            "workflow_id": workflow_id,
            "session_id": workflow.session_id,
            "topic": workflow.topic,
            "completed_at": workflow.completed_at.isoformat(),
            "total_duration": (workflow.completed_at - workflow.started_at).total_seconds(),
            "task_results": results,
            "workflow_data": workflow.workflow_data
        }

    async def _create_workflow_tasks(self, workflow: CoordinatedWorkflow, workflow_type: str):
        """Create tasks for a workflow based on its type."""
        if workflow_type == "standard_research":
            await self._create_standard_research_tasks(workflow)
        elif workflow_type == "quick_analysis":
            await self._create_quick_analysis_tasks(workflow)
        elif workflow_type == "comprehensive_report":
            await self._create_comprehensive_report_tasks(workflow)
        else:
            await self._create_standard_research_tasks(workflow)  # Default

    async def _create_standard_research_tasks(self, workflow: CoordinatedWorkflow):
        """Create tasks for standard research workflow."""

        # Research task
        research_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.RESEARCH,
            agent_type=SubAgentType.RESEARCHER,
            task_description=f"Conduct comprehensive research on: {workflow.topic}",
            task_data={"topic": workflow.topic, "depth": "comprehensive"},
            priority=1
        )

        # Report generation task (depends on research)
        report_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.REPORT_GENERATION,
            agent_type=SubAgentType.REPORT_WRITER,
            task_description="Generate structured report from research findings",
            task_data={"audience": "general", "format": "standard"},
            dependencies=[research_task.task_id],
            priority=2
        )

        # Editorial review task (depends on report)
        editorial_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.EDITORIAL_REVIEW,
            agent_type=SubAgentType.EDITORIAL_REVIEWER,
            task_description="Conduct editorial review and identify gaps",
            task_data={"focus": "completeness", "gap_research": True},
            dependencies=[report_task.task_id],
            priority=2
        )

        # Quality assessment task (depends on editorial)
        quality_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.QUALITY_ASSESSMENT,
            agent_type=SubAgentType.QUALITY_ASSESSOR,
            task_description="Assess final content quality",
            task_data={"criteria": ["completeness", "accuracy", "clarity"]},
            dependencies=[editorial_task.task_id],
            priority=3
        )

        workflow.tasks = [research_task, report_task, editorial_task, quality_task]

    async def _create_quick_analysis_tasks(self, workflow: CoordinatedWorkflow):
        """Create tasks for quick analysis workflow."""

        # Combined research and report task
        research_report_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.RESEARCH,
            agent_type=SubAgentType.RESEARCHER,
            task_description=f"Conduct quick research and analysis on: {workflow.topic}",
            task_data={"topic": workflow.topic, "depth": "quick", "include_report": True},
            priority=1
        )

        # Quality assessment task
        quality_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.QUALITY_ASSESSMENT,
            agent_type=SubAgentType.QUALITY_ASSESSOR,
            task_description="Quick quality assessment",
            task_data={"criteria": ["accuracy", "relevance"]},
            dependencies=[research_report_task.task_id],
            priority=2
        )

        workflow.tasks = [research_report_task, quality_task]

    async def _create_comprehensive_report_tasks(self, workflow: CoordinatedWorkflow):
        """Create tasks for comprehensive report workflow."""

        # Research task
        research_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.RESEARCH,
            agent_type=SubAgentType.RESEARCHER,
            task_description=f"Conduct comprehensive research on: {workflow.topic}",
            task_data={"topic": workflow.topic, "depth": "comprehensive", "sources": "academic"},
            priority=1
        )

        # Report generation task
        report_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.REPORT_GENERATION,
            agent_type=SubAgentType.REPORT_WRITER,
            task_description="Generate comprehensive report",
            task_data={"audience": "academic", "format": "detailed", "sections": "all"},
            dependencies=[research_task.task_id],
            priority=2
        )

        # Editorial review task
        editorial_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.EDITORIAL_REVIEW,
            agent_type=SubAgentType.EDITORIAL_REVIEWER,
            task_description="Comprehensive editorial review with gap research",
            task_data={"focus": "completeness", "gap_research": True, "enhancement": True},
            dependencies=[report_task.task_id],
            priority=2
        )

        # Content enhancement task (if gaps were found)
        enhancement_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.CONTENT_ENHANCEMENT,
            agent_type=SubAgentType.CONTENT_ENHANCER,
            task_description="Enhance content based on gap research and editorial feedback",
            task_data={"enhancement_type": "comprehensive"},
            dependencies=[editorial_task.task_id],
            priority=3
        )

        # Style editing task
        style_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.STYLE_EDITING,
            agent_type=SubAgentType.STYLE_EDITOR,
            task_description="Final style editing and formatting",
            task_data={"style_guide": "academic", "formatting": "professional"},
            dependencies=[enhancement_task.task_id],
            priority=3
        )

        # Final quality assessment
        final_quality_task = WorkflowTask(
            task_id=str(uuid.uuid4()),
            workflow_id=workflow.workflow_id,
            stage=WorkflowStage.QUALITY_ASSESSMENT,
            agent_type=SubAgentType.QUALITY_ASSESSOR,
            task_description="Final comprehensive quality assessment",
            task_data={"criteria": ["all"], "comprehensive": True},
            dependencies=[style_task.task_id],
            priority=4
        )

        workflow.tasks = [
            research_task, report_task, editorial_task,
            enhancement_task, style_task, final_quality_task
        ]

    async def _coordination_loop(self):
        """Main coordination loop for managing workflows."""
        while self._running:
            try:
                # Process all active workflows
                for workflow_id, workflow in list(self.active_workflows.items()):
                    if workflow.status == WorkflowStatus.IN_PROGRESS:
                        await self._process_workflow(workflow)

                await asyncio.sleep(1)  # Check every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")

    async def _process_workflow(self, workflow: CoordinatedWorkflow):
        """Process a single workflow."""
        try:
            # Get ready tasks
            ready_tasks = workflow.get_ready_tasks()

            # Limit concurrent tasks per workflow
            active_tasks = [
                task for task in workflow.tasks
                if task.status in ["assigned", "in_progress"]
            ]

            available_slots = (
                self.coordination_config["max_concurrent_tasks_per_workflow"] - len(active_tasks)
            )

            if available_slots > 0 and ready_tasks:
                # Sort by priority and execute
                ready_tasks.sort(key=lambda t: t.priority)
                tasks_to_execute = ready_tasks[:available_slots]

                for task in tasks_to_execute:
                    await self._execute_workflow_task(workflow, task)

            # Check for workflow completion
            await self._check_workflow_completion(workflow)

            # Check for workflow timeout
            if workflow.started_at:
                duration = (datetime.now() - workflow.started_at).total_seconds()
                timeout_seconds = self.coordination_config["workflow_timeout_hours"] * 3600

                if duration > timeout_seconds:
                    logger.warning(f"Workflow {workflow.workflow_id} timed out")
                    await self.cancel_workflow(workflow.workflow_id)

        except Exception as e:
            logger.error(f"Error processing workflow {workflow.workflow_id}: {e}")
            workflow.status = WorkflowStatus.FAILED

    async def _execute_workflow_task(self, workflow: CoordinatedWorkflow, task: WorkflowTask):
        """Execute a single workflow task."""
        try:
            # Update task status
            task.status = "assigned"
            task.started_at = datetime.now()

            # Create isolation context
            isolation_context = await self.isolation_manager.create_isolation_context(
                agent_type=task.agent_type.value,
                session_id=workflow.session_id,
                parent_context=None,
                isolation_level=workflow.context_isolation
            )

            # Create sub-agent request
            request = SubAgentRequest(
                agent_type=task.agent_type,
                task_description=task.task_description,
                session_id=workflow.session_id,
                parent_agent="coordinator",
                context_data={
                    "workflow_id": workflow.workflow_id,
                    "task_id": task.task_id,
                    "stage": task.stage.value,
                    **workflow.workflow_data
                },
                priority=task.priority,
                timeout_seconds=task.timeout_seconds,
                isolation_level=workflow.context_isolation
            )

            # Create and execute sub-agent
            result = await self.factory.create_and_execute(request, task.task_description)

            # Update task with result
            task.result = result
            task.status = "completed" if result.success else "failed"
            task.completed_at = datetime.now()

            if result.success:
                workflow.completed_tasks.add(task.task_id)

                # Store task results in workflow data
                workflow.workflow_data[f"task_{task.task_id}"] = result.result_data

                logger.info(f"Task {task.task_id} completed successfully for workflow {workflow.workflow_id}")
            else:
                workflow.failed_tasks.add(task.task_id)
                logger.error(f"Task {task.task_id} failed for workflow {workflow.workflow_id}: {result.error_message}")

                # Retry logic
                if self.coordination_config["enable_auto_retry"] and task.can_retry():
                    task.retry_count += 1
                    task.status = "pending"
                    logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count + 1})")

        except Exception as e:
            task.status = "failed"
            task.completed_at = datetime.now()
            workflow.failed_tasks.add(task.task_id)
            logger.error(f"Error executing task {task.task_id}: {e}")

    async def _check_workflow_completion(self, workflow: CoordinatedWorkflow):
        """Check if a workflow has completed."""
        total_tasks = len(workflow.tasks)
        completed_tasks = len(workflow.completed_tasks)
        failed_tasks = len(workflow.failed_tasks)

        if completed_tasks + failed_tasks == total_tasks:
            if failed_tasks == 0:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = datetime.now()
                logger.info(f"Workflow {workflow.workflow_id} completed successfully")
            else:
                workflow.status = WorkflowStatus.FAILED
                workflow.completed_at = datetime.now()
                logger.warning(f"Workflow {workflow.workflow_id} completed with {failed_tasks} failed tasks")

    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.coordination_config["cleanup_interval"])
                await self._cleanup_completed_workflows()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_completed_workflows(self):
        """Cleanup completed workflows."""
        cutoff_time = datetime.now() - timedelta(hours=1)  # Keep for 1 hour

        completed_workflows = [
            workflow_id for workflow_id, workflow in self.active_workflows.items()
            if workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
            and workflow.completed_at
            and workflow.completed_at < cutoff_time
        ]

        for workflow_id in completed_workflows:
            del self.active_workflows[workflow_id]
            logger.debug(f"Cleaned up completed workflow {workflow_id}")

    async def _register_communication_handlers(self):
        """Register communication handlers for coordination."""
        await self.communication_manager.register_message_handler(
            agent_id="coordinator",
            agent_type="coordinator",
            message_types=[MessageType.TASK_COMPLETION, MessageType.ERROR_REPORT],
            handler_function=self._handle_coordination_message
        )

    async def _handle_coordination_message(self, message: SubAgentMessage) -> Optional[SubAgentMessage]:
        """Handle coordination-related messages."""
        try:
            if message.message_type == MessageType.TASK_COMPLETION:
                await self._handle_task_completion(message)
            elif message.message_type == MessageType.ERROR_REPORT:
                await self._handle_error_report(message)

        except Exception as e:
            logger.error(f"Error handling coordination message: {e}")

        return None

    async def _handle_task_completion(self, message: SubAgentMessage):
        """Handle task completion messages."""
        # This could be used to handle asynchronous task completion notifications
        logger.debug(f"Received task completion notification: {message.payload}")

    async def _handle_error_report(self, message: SubAgentMessage):
        """Handle error report messages."""
        logger.error(f"Received error report: {message.payload}")

    def _get_workflow_stages(self, workflow_type: str) -> List[WorkflowStage]:
        """Get the stages for a workflow type."""
        if workflow_type == "quick_analysis":
            return [WorkflowStage.RESEARCH, WorkflowStage.QUALITY_ASSESSMENT]
        elif workflow_type == "comprehensive_report":
            return [
                WorkflowStage.RESEARCH, WorkflowStage.REPORT_GENERATION,
                WorkflowStage.EDITORIAL_REVIEW, WorkflowStage.CONTENT_ENHANCEMENT,
                WorkflowStage.STYLE_EDITING, WorkflowStage.QUALITY_ASSESSMENT
            ]
        else:  # standard_research
            return [
                WorkflowStage.RESEARCH, WorkflowStage.REPORT_GENERATION,
                WorkflowStage.EDITORIAL_REVIEW, WorkflowStage.QUALITY_ASSESSMENT
            ]

    async def _setup_workflow_templates(self):
        """Setup workflow templates."""
        self.workflow_templates = {
            "standard_research": {
                "description": "Standard research workflow with report generation and quality assessment",
                "estimated_duration_minutes": 30,
                "quality_requirements": {
                    "min_success_rate": 80,
                    "min_quality_score": 70
                }
            },
            "quick_analysis": {
                "description": "Quick analysis for rapid insights",
                "estimated_duration_minutes": 10,
                "quality_requirements": {
                    "min_success_rate": 75,
                    "min_quality_score": 60
                }
            },
            "comprehensive_report": {
                "description": "Comprehensive report with full enhancement cycle",
                "estimated_duration_minutes": 60,
                "quality_requirements": {
                    "min_success_rate": 90,
                    "min_quality_score": 85
                }
            }
        }

    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get the current status of the coordinator."""
        workflows_by_status = {}
        for workflow in self.active_workflows.values():
            status = workflow.status.value
            if status not in workflows_by_status:
                workflows_by_status[status] = 0
            workflows_by_status[status] += 1

        return {
            "running": self._running,
            "active_workflows": len(self.active_workflows),
            "workflows_by_status": workflows_by_status,
            "max_concurrent_workflows": self.coordination_config["max_concurrent_workflows"],
            "factory_status": self.factory.get_factory_status(),
            "communication_stats": self.communication_manager.get_communication_stats(),
            "isolation_status": self.isolation_manager.get_isolation_status(),
            "monitoring_status": self.performance_monitor.get_monitoring_status()
        }