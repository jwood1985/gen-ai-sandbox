"""
File-based job queue for spin-cycle simulations.

Each job is a JSON file in queue/ with states:
  pending  → running → completed | failed

Crash-recoverable: a job stuck in "running" after a crash can be
reset to "pending" by the recovery sweep.

Job file naming: {job_id}.json
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


QUEUE_DIR = Path(__file__).parent.parent / "queue"


@dataclass
class Job:
    job_id: str
    isotope_config: str
    B10_fraction: float
    B11_fraction: float
    N14_fraction: float
    N15_fraction: float
    magnetic_field_mT: float
    pulse_sequence: str
    n_pulses: int
    cce_order: int
    bath_radius_A: float
    n_time_points: int
    time_range_us: list
    spatial_seed: int
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _job_path(job_id: str) -> Path:
    return QUEUE_DIR / f"{job_id}.json"


def save_job(job: Job) -> Path:
    """Write job to queue directory."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    path = _job_path(job.job_id)
    with open(path, "w") as f:
        json.dump(job.to_dict(), f, indent=2)
    return path


def load_job(job_id: str) -> Job:
    """Load a job by ID."""
    with open(_job_path(job_id)) as f:
        return Job.from_dict(json.load(f))


def create_job(
    isotope_config: str,
    B10_fraction: float,
    B11_fraction: float,
    N14_fraction: float,
    N15_fraction: float,
    magnetic_field_mT: float,
    pulse_sequence: str,
    n_pulses: int,
    cce_order: int,
    bath_radius_A: float,
    n_time_points: int,
    time_range_us: list,
    spatial_seed: int,
) -> Job:
    """Create and enqueue a new job."""
    job = Job(
        job_id=str(uuid.uuid4())[:8],
        isotope_config=isotope_config,
        B10_fraction=B10_fraction,
        B11_fraction=B11_fraction,
        N14_fraction=N14_fraction,
        N15_fraction=N15_fraction,
        magnetic_field_mT=magnetic_field_mT,
        pulse_sequence=pulse_sequence,
        n_pulses=n_pulses,
        cce_order=cce_order,
        bath_radius_A=bath_radius_A,
        n_time_points=n_time_points,
        time_range_us=time_range_us,
        spatial_seed=spatial_seed,
    )
    save_job(job)
    return job


def list_jobs(status: str | None = None) -> list[Job]:
    """List all jobs, optionally filtered by status."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in sorted(QUEUE_DIR.glob("*.json")):
        with open(path) as f:
            job = Job.from_dict(json.load(f))
        if status is None or job.status == status:
            jobs.append(job)
    return jobs


def claim_next_job() -> Job | None:
    """Atomically claim the next pending job for execution."""
    for job in list_jobs(status="pending"):
        job.status = "running"
        job.started_at = time.time()
        save_job(job)
        return job
    return None


def complete_job(job: Job, result_path: str) -> None:
    """Mark a job as completed with its result path."""
    job.status = "completed"
    job.completed_at = time.time()
    job.result_path = result_path
    save_job(job)


def fail_job(job: Job, error: str) -> None:
    """Mark a job as failed with an error message."""
    job.status = "failed"
    job.completed_at = time.time()
    job.error = error
    save_job(job)


def recover_stale_jobs(timeout_seconds: float = 3600.0) -> int:
    """Reset jobs stuck in 'running' state beyond the timeout."""
    recovered = 0
    for job in list_jobs(status="running"):
        if job.started_at and (time.time() - job.started_at) > timeout_seconds:
            job.status = "pending"
            job.started_at = None
            save_job(job)
            recovered += 1
    return recovered


def queue_summary() -> dict[str, int]:
    """Return counts by status."""
    counts: dict[str, int] = {}
    for job in list_jobs():
        counts[job.status] = counts.get(job.status, 0) + 1
    return counts
