"""
Azure Storage Queue client for KEDA message cleanup.

When a job is processed, we delete the corresponding queue message
so KEDA can scale down the container faster.
"""

import json
import logging
from typing import Optional

from .config import WorkerConfig


logger = logging.getLogger(__name__)


class QueueMessage:
    """Represents a queue message with deletion info."""
    def __init__(self, message_id: str, pop_receipt: str, job_id: str):
        self.message_id = message_id
        self.pop_receipt = pop_receipt
        self.job_id = job_id


class AzureQueueClient:
    """Client for Azure Storage Queue operations."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        """Lazy initialization of queue client."""
        if self._client is None:
            if not self.config.azure_queue_enabled:
                return None

            try:
                from azure.storage.queue import QueueClient
                self._client = QueueClient.from_connection_string(
                    self.config.azure_queue_connection_string,
                    queue_name=self.config.azure_queue_name,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Azure Queue client: {e}")
                return None

        return self._client

    def receive_message(self) -> Optional[QueueMessage]:
        """
        Receive a single message from the queue.
        Returns the message with job_id and deletion info.
        """
        client = self._get_client()
        if client is None:
            return None

        try:
            # Receive one message with 5 minute visibility timeout
            messages = client.receive_messages(
                messages_per_page=1,
                visibility_timeout=300,  # 5 minutes
            )

            for msg in messages:
                try:
                    content = json.loads(msg.content)
                    job_id = content.get("job_id", "")

                    return QueueMessage(
                        message_id=msg.id,
                        pop_receipt=msg.pop_receipt,
                        job_id=job_id,
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Invalid queue message content: {msg.content}")
                    # Delete invalid message
                    self.delete_message(msg.id, msg.pop_receipt)

            return None

        except Exception as e:
            logger.warning(f"Failed to receive queue message: {e}")
            return None

    def delete_message(self, message_id: str, pop_receipt: str) -> bool:
        """Delete a message from the queue."""
        client = self._get_client()
        if client is None:
            return False

        try:
            client.delete_message(message_id, pop_receipt)
            logger.info(f"Deleted queue message: {message_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete queue message {message_id}: {e}")
            return False

    def clear_messages_for_job(self, job_id: str) -> int:
        """
        Clear all messages for a specific job_id.
        Returns the number of messages deleted.
        """
        client = self._get_client()
        if client is None:
            return 0

        deleted = 0
        try:
            # Peek at messages to find ones matching job_id
            messages = client.receive_messages(
                messages_per_page=32,
                visibility_timeout=60,
            )

            for msg in messages:
                try:
                    content = json.loads(msg.content)
                    if content.get("job_id") == job_id:
                        client.delete_message(msg.id, msg.pop_receipt)
                        deleted += 1
                        logger.info(f"Deleted queue message for job {job_id}")
                except (json.JSONDecodeError, Exception) as e:
                    logger.debug(f"Skipping message: {e}")

        except Exception as e:
            logger.warning(f"Error clearing messages for job {job_id}: {e}")

        return deleted
