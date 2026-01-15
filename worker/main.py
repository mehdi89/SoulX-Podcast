"""
Main worker loop - receives jobs from Azure Queue and processes them.
"""

import json
import logging
import os
import signal
import sys
import threading  # Still used for health server thread
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

from .api_client import TubeOnAIClient
from .config import WorkerConfig
from .processor import PodcastProcessor
from .queue_client import AzureQueueClient
from .s3_client import S3Client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Reduce Azure SDK logging verbosity
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


# Global reference for health check
_worker_instance = None


class HealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health" or self.path == "/":
            self._send_health_response()
        else:
            self.send_error(404)

    def _send_health_response(self):
        """Send health check response."""
        global _worker_instance

        status = "healthy"
        current_job = None

        if _worker_instance:
            if not _worker_instance.running:
                status = "stopping"
            current_job = _worker_instance.current_job_id

        response = {
            "status": status,
            "current_job": current_job,
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())


def start_health_server(port: int = 8080):
    """Start the health check HTTP server in a background thread."""
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health server started on port {port}")
    return server


class PodcastWorker:
    """Main worker that receives jobs from Azure Queue and processes them."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.api_client = TubeOnAIClient(config)
        self.s3_client = S3Client(config)
        self.queue_client = AzureQueueClient(config)
        self.processor = PodcastProcessor(config)

        self.running = False
        self.current_job_id = None

    def start(self):
        """Start the worker."""
        global _worker_instance
        _worker_instance = self

        logger.info(f"Starting worker: {self.config.worker_id}")
        logger.info(f"API URL: {self.config.api_url}")
        logger.info(f"S3 Bucket: {self.config.s3_bucket}")
        logger.info(f"Azure Queue: {self.config.azure_queue_name} (enabled: {self.config.azure_queue_enabled})")

        # Start health server for Azure liveness probes
        start_health_server(port=8080)

        # Validate config
        errors = self.config.validate()
        if errors:
            for error in errors:
                logger.error(f"Config error: {error}")
            sys.exit(1)

        # Initialize model (preload to avoid delay on first job)
        logger.info("Initializing model...")
        self.processor.initialize()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True

        # Main loop - queue-based or fallback to polling
        if self.config.azure_queue_enabled:
            self._run_queue_loop()
        else:
            self._run_poll_loop()

    def _run_queue_loop(self):
        """Main loop - receive jobs from Azure Queue."""
        logger.info("Worker ready, waiting for queue messages...")

        while self.running:
            try:
                # Receive message from queue
                message = self.queue_client.receive_message()

                if message is None:
                    # No messages, wait and retry
                    time.sleep(self.config.poll_interval)
                    continue

                logger.info(f"Received queue message for job: {message.job_id}")

                # Fetch job details from API
                job = self.api_client.get_job(message.job_id)

                if job is None:
                    # Job not found or already processed, delete message and continue
                    logger.warning(f"Job {message.job_id} not found, deleting queue message")
                    self.queue_client.delete_message(message.message_id, message.pop_receipt)
                    continue

                # Process job
                self._process_job(job)

                # Delete queue message after successful processing
                self.queue_client.delete_message(message.message_id, message.pop_receipt)
                logger.info(f"Deleted queue message for job: {job.job_id}")

            except Exception as e:
                logger.exception(f"Worker error: {e}")
                time.sleep(self.config.poll_interval)

        logger.info("Worker stopped")

    def _run_poll_loop(self):
        """Fallback polling loop when queue is disabled."""
        logger.info("Worker ready, polling for jobs...")

        while self.running:
            try:
                # Poll for job
                job = self.api_client.claim_job()

                if job is None:
                    # No jobs available, wait and retry
                    time.sleep(self.config.poll_interval)
                    continue

                # Process job
                self._process_job(job)

            except Exception as e:
                logger.exception(f"Worker error: {e}")
                time.sleep(self.config.poll_interval)

        logger.info("Worker stopped")

    def _process_job(self, job):
        """Process a single podcast job."""
        self.current_job_id = job.job_id
        logger.info(f"Processing job: {job.job_id}")

        try:
            # Generate podcast
            audio_path, duration = self.processor.generate(
                script=job.script_text,
                seed=job.seed,
            )

            # Upload to S3
            output_url = self.s3_client.upload(
                local_path=audio_path,
                s3_path=job.s3_upload_path,
            )

            # Mark complete
            self.api_client.complete_job(
                job_id=job.job_id,
                output_url=output_url,
                duration_seconds=duration,
            )

            logger.info(f"Job {job.job_id} completed successfully")

            # Cleanup temp file
            self._cleanup(audio_path)

        except Exception as e:
            logger.exception(f"Job {job.job_id} failed: {e}")
            self.api_client.fail_job(
                job_id=job.job_id,
                error_message=str(e),
            )

        finally:
            self.current_job_id = None

    def _cleanup(self, path: str):
        """Remove temporary file."""
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"Cleaned up: {path}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False


def run_worker(env_path: str = None):
    """Entry point for running the worker."""
    config = WorkerConfig.from_env(env_path)
    worker = PodcastWorker(config)
    worker.start()


if __name__ == "__main__":
    run_worker()
