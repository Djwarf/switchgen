"""Job queue system for generation requests.

ComfyUI's PromptExecutor executes immediately without queueing (Gotcha #4).
This module provides a simple queue system for managing multiple generation requests.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue, Empty
from threading import Thread, Lock
from typing import Callable, Optional, Any
import uuid
import time

from .engine import GenerationEngine, GenerationResult, ProgressInfo

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a generation job."""
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class GenerationJob:
    """A generation job in the queue."""

    workflow: dict
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0  # Higher = more urgent
    status: JobStatus = JobStatus.QUEUED
    result: Optional[GenerationResult] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Callbacks
    on_progress: Optional[Callable[[ProgressInfo], None]] = None
    on_complete: Optional[Callable[["GenerationJob"], None]] = None

    def __lt__(self, other: "GenerationJob") -> bool:
        """For priority queue ordering (higher priority first)."""
        return self.priority > other.priority


class GenerationQueue:
    """Queue system for managing generation jobs.

    Since PromptExecutor is immediate, this provides:
    - Job queuing with priorities
    - Background worker thread
    - Progress and completion callbacks
    - Job cancellation
    """

    def __init__(self, engine: GenerationEngine):
        self.engine = engine
        self._queue: Queue[GenerationJob] = Queue()
        self._jobs: dict[str, GenerationJob] = {}
        self._jobs_lock = Lock()
        self._running = False
        self._worker_thread: Optional[Thread] = None
        self._current_job: Optional[GenerationJob] = None

        # Global callbacks
        self.on_job_started: Optional[Callable[[GenerationJob], None]] = None
        self.on_job_completed: Optional[Callable[[GenerationJob], None]] = None
        self.on_queue_empty: Optional[Callable[[], None]] = None

    def add(self, job: GenerationJob) -> str:
        """Add a job to the queue.

        Args:
            job: The generation job to queue

        Returns:
            The job ID
        """
        with self._jobs_lock:
            self._jobs[job.job_id] = job

        self._queue.put(job)
        logger.info("Job queued (job_id=%s, priority=%d, queue_size=%d)",
                    job.job_id, job.priority, self._queue.qsize())
        return job.job_id

    def submit(
        self,
        workflow: dict,
        priority: int = 0,
        on_progress: Optional[Callable[[ProgressInfo], None]] = None,
        on_complete: Optional[Callable[[GenerationJob], None]] = None,
    ) -> str:
        """Submit a workflow for generation.

        Convenience method that creates a job and adds it to the queue.

        Args:
            workflow: ComfyUI workflow in API format
            priority: Job priority (higher = more urgent)
            on_progress: Callback for progress updates
            on_complete: Callback when job completes

        Returns:
            The job ID
        """
        job = GenerationJob(
            workflow=workflow,
            priority=priority,
            on_progress=on_progress,
            on_complete=on_complete,
        )
        return self.add(job)

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Get a job by ID."""
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel a job.

        If the job is running, interrupts it.
        If queued, marks it as cancelled.

        Returns:
            True if job was found and cancelled
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.status == JobStatus.RUNNING:
                # Interrupt the running job
                self.engine.interrupt()
                job.status = JobStatus.CANCELLED
                return True
            elif job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                return True

        return False

    def start(self) -> None:
        """Start the queue worker thread."""
        if self._running:
            return

        logger.info("Starting queue worker thread")
        self._running = True
        self._worker_thread = Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def stop(self, wait: bool = True) -> None:
        """Stop the queue worker.

        Args:
            wait: If True, wait for current job to complete
        """
        self._running = False
        if wait and self._worker_thread:
            self._worker_thread.join(timeout=30.0)

    def clear(self) -> int:
        """Clear all queued (not running) jobs.

        Returns:
            Number of jobs cleared
        """
        cleared = 0
        with self._jobs_lock:
            for job_id, job in list(self._jobs.items()):
                if job.status == JobStatus.QUEUED:
                    job.status = JobStatus.CANCELLED
                    cleared += 1
        return cleared

    @property
    def queue_size(self) -> int:
        """Number of jobs waiting in queue."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Whether the worker is running."""
        return self._running

    @property
    def current_job(self) -> Optional[GenerationJob]:
        """Currently executing job, if any."""
        return self._current_job

    def _worker(self) -> None:
        """Background worker that processes the queue."""
        logger.debug("Queue worker started")
        while self._running:
            try:
                # Get next job with timeout
                job = self._queue.get(timeout=1.0)

                # Skip cancelled jobs
                if job.status == JobStatus.CANCELLED:
                    logger.debug("Skipping cancelled job: %s", job.job_id)
                    continue

                # Process the job
                self._process_job(job)

            except Empty:
                # No jobs in queue
                if self.on_queue_empty and self._current_job is None:
                    self.on_queue_empty()
                continue

            except Exception as e:
                logger.error("Queue worker error: %s", e, exc_info=True)
                continue

        logger.debug("Queue worker stopped")

    def _process_job(self, job: GenerationJob) -> None:
        """Process a single job."""
        self._current_job = job
        job.status = JobStatus.RUNNING
        job.started_at = time.time()

        logger.info("Job started (job_id=%s)", job.job_id)

        # Notify job started
        if self.on_job_started:
            self.on_job_started(job)

        # Set up progress callback
        if job.on_progress:
            self.engine.set_progress_callback(job.on_progress)

        try:
            # Execute the workflow
            result = self.engine.execute(job.workflow)

            job.result = result
            job.status = JobStatus.COMPLETED if result.success else JobStatus.FAILED
            job.error = result.error

        except Exception as e:
            logger.error("Job execution failed: %s", e, exc_info=True)
            job.status = JobStatus.FAILED
            job.error = str(e)

        finally:
            job.completed_at = time.time()
            elapsed = job.completed_at - job.started_at
            self._current_job = None

            # Clear progress callback
            self.engine.set_progress_callback(None)

            logger.info("Job completed (job_id=%s, status=%s, time=%.2fs)",
                        job.job_id, job.status.name, elapsed)

            # Notify completion
            if job.on_complete:
                job.on_complete(job)
            if self.on_job_completed:
                self.on_job_completed(job)


# Global queue instance
_queue: Optional[GenerationQueue] = None


def get_queue() -> GenerationQueue:
    """Get the global generation queue.

    Creates and starts the queue if not already running.
    """
    global _queue
    if _queue is None:
        from .engine import get_engine
        engine = get_engine()
        engine.initialize()
        _queue = GenerationQueue(engine)
        _queue.start()
    return _queue


# Simple synchronous API for testing
def generate_sync(workflow: dict) -> GenerationResult:
    """Simple synchronous generation without queue.

    For testing or simple scripts.
    """
    from .engine import get_engine
    engine = get_engine()
    engine.initialize()
    return engine.execute(workflow)
