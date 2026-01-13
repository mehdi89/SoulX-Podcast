"""
TubeOnAI API client for worker communication.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import requests

from .config import WorkerConfig


logger = logging.getLogger(__name__)


@dataclass
class PodcastJob:
    """Represents a podcast generation job."""
    job_id: str
    script_text: str
    seed: int
    s3_upload_path: str


class TubeOnAIClient:
    """Client for communicating with TubeOnAI backend."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.base_url = config.api_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.timeout = 30  # seconds

    def claim_job(self) -> Optional[PodcastJob]:
        """
        Attempt to claim a pending job from the queue.
        Returns None if no jobs available.
        """
        try:
            response = requests.post(
                f"{self.base_url}/jobs/claim",
                json={"worker_id": self.config.worker_id},
                headers=self.headers,
                timeout=self.timeout,
            )

            if response.status_code == 204:
                # No jobs available
                return None

            response.raise_for_status()
            data = response.json()

            return PodcastJob(
                job_id=data["job_id"],
                script_text=data["script_text"],
                seed=data.get("seed", 1988),
                s3_upload_path=data["s3_upload_path"],
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to claim job: {e}")
            return None

    def complete_job(
        self,
        job_id: str,
        output_url: str,
        duration_seconds: int,
    ) -> bool:
        """Mark a job as successfully completed."""
        try:
            response = requests.post(
                f"{self.base_url}/jobs/{job_id}/complete",
                json={
                    "worker_id": self.config.worker_id,
                    "output_url": output_url,
                    "duration_seconds": duration_seconds,
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.info(f"Job {job_id} marked as complete")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to complete job {job_id}: {e}")
            return False

    def fail_job(self, job_id: str, error_message: str) -> bool:
        """Mark a job as failed."""
        try:
            response = requests.post(
                f"{self.base_url}/jobs/{job_id}/failed",
                json={
                    "worker_id": self.config.worker_id,
                    "error_message": error_message[:1000],  # Truncate long errors
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.info(f"Job {job_id} marked as failed: {error_message[:100]}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to mark job {job_id} as failed: {e}")
            return False

    def heartbeat(
        self,
        status: str = "idle",
        current_job_id: Optional[str] = None,
    ) -> bool:
        """Send heartbeat to TubeOnAI to indicate worker is alive."""
        try:
            response = requests.post(
                f"{self.base_url}/workers/heartbeat",
                json={
                    "worker_id": self.config.worker_id,
                    "server_ip": self.config.server_ip,
                    "server_name": self.config.server_name,
                    "status": status,
                    "current_job_id": current_job_id,
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False
