"""
S3 client for uploading generated podcast audio.
"""

import logging
import mimetypes
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import WorkerConfig


logger = logging.getLogger(__name__)


class S3Client:
    """Client for uploading files to S3."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.bucket = config.s3_bucket

        # Create S3 client
        boto_config = BotoConfig(
            region_name=config.s3_region,
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        )

        client_kwargs = {
            "aws_access_key_id": config.s3_access_key,
            "aws_secret_access_key": config.s3_secret_key,
            "config": boto_config,
        }

        # Support S3-compatible storage (MinIO, etc.)
        if config.s3_endpoint_url:
            client_kwargs["endpoint_url"] = config.s3_endpoint_url

        self.client = boto3.client("s3", **client_kwargs)

    def upload(self, local_path: str, s3_path: str) -> str:
        """
        Upload a file to S3.

        Args:
            local_path: Path to the local file
            s3_path: Destination path in S3 bucket (without bucket name)

        Returns:
            Full S3 URL of the uploaded file
        """
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        # Determine content type
        content_type, _ = mimetypes.guess_type(str(local_path))
        if content_type is None:
            content_type = "audio/wav"  # Default for podcast files

        # Remove leading slash from s3_path if present
        s3_path = s3_path.lstrip("/")

        try:
            logger.info(f"Uploading {local_path} to s3://{self.bucket}/{s3_path}")

            self.client.upload_file(
                str(local_path),
                self.bucket,
                s3_path,
                ExtraArgs={
                    "ContentType": content_type,
                    "ACL": "private",  # Keep files private, access via signed URLs
                },
            )

            # Construct the URL
            if self.config.s3_endpoint_url:
                # S3-compatible storage
                url = f"{self.config.s3_endpoint_url}/{self.bucket}/{s3_path}"
            else:
                # AWS S3
                url = f"https://{self.bucket}.s3.{self.config.s3_region}.amazonaws.com/{s3_path}"

            logger.info(f"Upload complete: {url}")
            return url

        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise

    def delete(self, s3_path: str) -> bool:
        """Delete a file from S3."""
        s3_path = s3_path.lstrip("/")

        try:
            self.client.delete_object(Bucket=self.bucket, Key=s3_path)
            logger.info(f"Deleted s3://{self.bucket}/{s3_path}")
            return True

        except ClientError as e:
            logger.error(f"S3 delete failed: {e}")
            return False

    def file_exists(self, s3_path: str) -> bool:
        """Check if a file exists in S3."""
        s3_path = s3_path.lstrip("/")

        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_path)
            return True
        except ClientError:
            return False
