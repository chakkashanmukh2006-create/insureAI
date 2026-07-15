from typing import Dict, Any, List
import uuid

# In-memory dictionary to track background job states.
# Format: { "job_id": { "status": "running"|"completed"|"failed", "logs": [...], "results": {...} } }
_jobs: Dict[str, Dict[str, Any]] = {}

def create_job() -> str:
    """Creates a new job and returns the job_id."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "logs": ["Job initialized."],
        "results": None
    }
    return job_id

def append_log(job_id: str, message: str) -> None:
    """Appends a log message to the specific job."""
    if job_id in _jobs:
        _jobs[job_id]["logs"].append(message)

def mark_completed(job_id: str, results: dict = None) -> None:
    """Marks the job as completed and stores any final results."""
    if job_id in _jobs:
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["results"] = results
        _jobs[job_id]["logs"].append("Job completed successfully.")

def mark_failed(job_id: str, error: str) -> None:
    """Marks the job as failed with an error message."""
    if job_id in _jobs:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["logs"].append(f"ERROR: {error}")

def get_job_status(job_id: str) -> Dict[str, Any]:
    """Retrieves the current job status and logs."""
    return _jobs.get(job_id, {"status": "not_found", "logs": [], "results": None})
