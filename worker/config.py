"""
Worker configuration - loads from environment variables.
"""

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class WorkerConfig:
    """Configuration for the podcast worker."""

    # Worker identity
    worker_id: str

    # TubeOnAI API
    api_url: str
    api_token: str

    # S3 Storage
    s3_bucket: str
    s3_access_key: str
    s3_secret_key: str
    s3_region: str
    s3_endpoint_url: str | None  # For S3-compatible storage

    # Azure Queue (for KEDA scaling message cleanup)
    azure_queue_enabled: bool
    azure_queue_connection_string: str
    azure_queue_name: str

    # Model
    model_path: str

    # Worker settings
    poll_interval: int  # seconds

    @classmethod
    def from_env(cls, env_path: str | None = None) -> "WorkerConfig":
        """Load configuration from environment variables."""

        # Load .env file if specified or from default location
        if env_path:
            load_dotenv(env_path)
        else:
            # Try worker/.env first, then project root .env
            worker_env = Path(__file__).parent / ".env"
            root_env = Path(__file__).parent.parent / ".env"

            if worker_env.exists():
                load_dotenv(worker_env)
            elif root_env.exists():
                load_dotenv(root_env)

        return cls(
            # Worker identity
            worker_id=os.getenv("WORKER_ID", f"worker-{socket.gethostname()}"),

            # TubeOnAI API
            api_url=os.getenv("TUBEONAI_API_URL", "http://localhost:8080/podcast-worker/v1"),
            api_token=os.getenv("TUBEONAI_API_TOKEN", ""),

            # S3 Storage
            s3_bucket=os.getenv("S3_BUCKET", ""),
            s3_access_key=os.getenv("S3_ACCESS_KEY", ""),
            s3_secret_key=os.getenv("S3_SECRET_KEY", ""),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            s3_endpoint_url=os.getenv("S3_ENDPOINT_URL"),  # Optional, for MinIO etc.

            # Azure Queue (for KEDA scaling message cleanup)
            azure_queue_enabled=os.getenv("AZURE_QUEUE_ENABLED", "false").lower() == "true",
            azure_queue_connection_string=os.getenv("AZURE_QUEUE_CONNECTION_STRING", ""),
            azure_queue_name=os.getenv("AZURE_QUEUE_NAME", "podcast-jobs"),

            # Model
            model_path=os.getenv("MODEL_PATH", "pretrained_models/SoulX-Podcast-1.7B"),

            # Worker settings
            poll_interval=int(os.getenv("POLL_INTERVAL", "10")),
        )

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []

        if not self.api_token:
            errors.append("TUBEONAI_API_TOKEN is required")

        if not self.s3_bucket:
            errors.append("S3_BUCKET is required")

        if not self.s3_access_key:
            errors.append("S3_ACCESS_KEY is required")

        if not self.s3_secret_key:
            errors.append("S3_SECRET_KEY is required")

        if not Path(self.model_path).exists():
            errors.append(f"MODEL_PATH does not exist: {self.model_path}")

        return errors
