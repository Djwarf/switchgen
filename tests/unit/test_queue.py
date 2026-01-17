"""Unit tests for switchgen.core.queue module."""

import pytest
import time
from unittest.mock import MagicMock, patch


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_enum_values(self):
        """Should have all expected status values."""
        from switchgen.core.queue import JobStatus

        assert JobStatus.QUEUED
        assert JobStatus.RUNNING
        assert JobStatus.COMPLETED
        assert JobStatus.FAILED
        assert JobStatus.CANCELLED

    def test_all_statuses_count(self):
        """Should have exactly 5 status values."""
        from switchgen.core.queue import JobStatus
        assert len(JobStatus) == 5


class TestGenerationJob:
    """Tests for GenerationJob dataclass."""

    def test_default_values(self, sample_workflow):
        """Should have sensible default values."""
        from switchgen.core.queue import GenerationJob, JobStatus

        job = GenerationJob(workflow=sample_workflow)

        assert job.workflow == sample_workflow
        assert job.job_id  # Should have auto-generated ID
        assert job.priority == 0
        assert job.status == JobStatus.QUEUED
        assert job.result is None
        assert job.error is None
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None

    def test_lt_comparison_by_priority(self, sample_workflow):
        """Jobs should be compared by priority (higher first)."""
        from switchgen.core.queue import GenerationJob

        low_priority = GenerationJob(workflow=sample_workflow, priority=1)
        high_priority = GenerationJob(workflow=sample_workflow, priority=10)

        # Higher priority job should be "less than" (come first in queue)
        assert high_priority < low_priority

    def test_same_priority_comparison(self, sample_workflow):
        """Jobs with same priority should not be less than each other."""
        from switchgen.core.queue import GenerationJob

        job1 = GenerationJob(workflow=sample_workflow, priority=5)
        job2 = GenerationJob(workflow=sample_workflow, priority=5)

        assert not (job1 < job2)
        assert not (job2 < job1)


class TestGenerationQueue:
    """Tests for GenerationQueue class."""

    def test_add_returns_job_id(self, mock_engine, sample_workflow):
        """add should return the job ID."""
        from switchgen.core.queue import GenerationQueue, GenerationJob

        queue = GenerationQueue(mock_engine)
        job = GenerationJob(workflow=sample_workflow)

        result = queue.add(job)

        assert result == job.job_id

    def test_add_increments_queue_size(self, mock_engine, sample_workflow):
        """add should increment queue size."""
        from switchgen.core.queue import GenerationQueue, GenerationJob

        queue = GenerationQueue(mock_engine)
        initial_size = queue.queue_size

        queue.add(GenerationJob(workflow=sample_workflow))

        assert queue.queue_size == initial_size + 1

    def test_submit_creates_job(self, mock_engine, sample_workflow):
        """submit should create and add a job."""
        from switchgen.core.queue import GenerationQueue

        queue = GenerationQueue(mock_engine)

        job_id = queue.submit(sample_workflow, priority=5)

        assert job_id is not None
        assert queue.queue_size == 1

        job = queue.get_job(job_id)
        assert job.priority == 5

    def test_get_job_returns_job(self, mock_engine, sample_workflow):
        """get_job should return the job for valid ID."""
        from switchgen.core.queue import GenerationQueue, GenerationJob

        queue = GenerationQueue(mock_engine)
        job = GenerationJob(workflow=sample_workflow)
        queue.add(job)

        result = queue.get_job(job.job_id)

        assert result is job

    def test_get_job_returns_none_for_invalid(self, mock_engine):
        """get_job should return None for invalid ID."""
        from switchgen.core.queue import GenerationQueue

        queue = GenerationQueue(mock_engine)

        result = queue.get_job("nonexistent_id")

        assert result is None

    def test_cancel_queued_job(self, mock_engine, sample_workflow):
        """cancel should mark queued job as cancelled."""
        from switchgen.core.queue import GenerationQueue, GenerationJob, JobStatus

        queue = GenerationQueue(mock_engine)
        job = GenerationJob(workflow=sample_workflow)
        queue.add(job)

        result = queue.cancel(job.job_id)

        assert result is True
        assert job.status == JobStatus.CANCELLED

    def test_cancel_returns_false_for_invalid(self, mock_engine):
        """cancel should return False for invalid ID."""
        from switchgen.core.queue import GenerationQueue

        queue = GenerationQueue(mock_engine)

        result = queue.cancel("nonexistent_id")

        assert result is False

    def test_clear_cancels_queued_jobs(self, mock_engine, sample_workflow):
        """clear should cancel all queued jobs."""
        from switchgen.core.queue import GenerationQueue, GenerationJob, JobStatus

        queue = GenerationQueue(mock_engine)
        job1 = GenerationJob(workflow=sample_workflow)
        job2 = GenerationJob(workflow=sample_workflow)
        queue.add(job1)
        queue.add(job2)

        cleared = queue.clear()

        assert cleared == 2
        assert job1.status == JobStatus.CANCELLED
        assert job2.status == JobStatus.CANCELLED

    def test_is_running_initially_false(self, mock_engine):
        """is_running should be False initially."""
        from switchgen.core.queue import GenerationQueue

        queue = GenerationQueue(mock_engine)

        assert queue.is_running is False

    def test_start_sets_running(self, mock_engine):
        """start should set is_running to True."""
        from switchgen.core.queue import GenerationQueue

        queue = GenerationQueue(mock_engine)
        queue.start()

        try:
            assert queue.is_running is True
        finally:
            queue.stop(wait=False)

    def test_stop_clears_running(self, mock_engine):
        """stop should set is_running to False."""
        from switchgen.core.queue import GenerationQueue

        queue = GenerationQueue(mock_engine)
        queue.start()
        queue.stop(wait=False)

        # Give thread time to notice
        time.sleep(0.1)
        assert queue.is_running is False

    def test_current_job_initially_none(self, mock_engine):
        """current_job should be None initially."""
        from switchgen.core.queue import GenerationQueue

        queue = GenerationQueue(mock_engine)

        assert queue.current_job is None

    def test_queue_size_property(self, mock_engine, sample_workflow):
        """queue_size should reflect number of queued jobs."""
        from switchgen.core.queue import GenerationQueue, GenerationJob

        queue = GenerationQueue(mock_engine)

        assert queue.queue_size == 0

        queue.add(GenerationJob(workflow=sample_workflow))
        assert queue.queue_size == 1

        queue.add(GenerationJob(workflow=sample_workflow))
        assert queue.queue_size == 2


class TestGenerateSyncFunction:
    """Tests for generate_sync helper function."""

    def test_calls_engine_execute(self, sample_workflow):
        """Should call engine.execute with workflow."""
        mock_result = MagicMock()
        mock_engine = MagicMock()
        mock_engine.execute.return_value = mock_result

        # get_engine is imported inside generate_sync, so we patch where it's looked up
        with patch('switchgen.core.engine.get_engine', return_value=mock_engine):
            from switchgen.core.queue import generate_sync
            result = generate_sync(sample_workflow)

        mock_engine.initialize.assert_called_once()
        mock_engine.execute.assert_called_once_with(sample_workflow)
        assert result == mock_result
