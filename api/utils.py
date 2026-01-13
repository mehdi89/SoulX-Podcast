"""
Utility Functions for API
"""
import os
import re
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
from fastapi import UploadFile, HTTPException
import logging

from api.config import config

logger = logging.getLogger(__name__)


def generate_task_id() -> str:
    """Generate a unique task ID"""
    return str(uuid.uuid4())


def save_upload_file(upload_file: UploadFile, task_id: str, index: int) -> Path:
    """
    Save uploaded file to temporary directory

    Args:
        upload_file: FastAPI upload file object
        task_id: Task ID
        index: File index

    Returns:
        Path: Saved file path
    """
    try:
        # Get file extension
        file_extension = Path(upload_file.filename).suffix or ".wav"

        # Build save path
        filename = f"{task_id}_prompt_{index}{file_extension}"
        file_path = config.temp_dir / filename

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        logger.info(f"Saved upload file to {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Failed to save upload file: {e}")
        raise HTTPException(status_code=500, detail=f"File save failed: {str(e)}")
    finally:
        upload_file.file.close()


def validate_audio_files(files: List[UploadFile]) -> None:
    """
    Validate audio files

    Args:
        files: List of uploaded files

    Raises:
        HTTPException: If validation fails
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least 1 reference audio file is required")

    if len(files) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 speakers (4 audio files) supported")

    # Validate file type and size
    allowed_extensions = {".wav", ".mp3", ".flac", ".m4a"}
    for i, file in enumerate(files):
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} format not supported. Supported formats: {', '.join(allowed_extensions)}"
            )

        # Check file size (via content-length header, may be inaccurate)
        if hasattr(file, 'size') and file.size and file.size > config.max_upload_size:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds max size limit ({config.max_upload_size / 1024 / 1024}MB)"
            )


def validate_dialogue_format(dialogue_text: str, num_speakers: int) -> Tuple[bool, str]:
    """
    Validate dialogue text format

    Args:
        dialogue_text: Dialogue text
        num_speakers: Number of speakers

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    dialogue_text = dialogue_text.strip()

    # Single speaker mode: no special format required
    if num_speakers == 1:
        if len(dialogue_text) == 0:
            return False, "Dialogue text cannot be empty"
        # Single speaker can omit [S1] tag
        return True, ""

    # Multi-speaker mode: requires [S1][S2] etc. tags
    # Extract all speaker tags
    speaker_pattern = r'\[S[1-4]\]'
    matches = re.findall(speaker_pattern, dialogue_text)

    if not matches:
        return False, f"Multi-speaker mode requires speaker tags, e.g.: [S1]Hello[S2]Hi there"

    # Check if used speaker IDs are within valid range
    used_speakers = set()
    for match in matches:
        speaker_id = int(match[2])  # Extract 1 from [S1]
        used_speakers.add(speaker_id)

        if speaker_id > num_speakers:
            return False, f"Text uses speaker [S{speaker_id}], but only {num_speakers} reference audio(s) provided"

    return True, ""


def cleanup_old_files(directory: Path, minutes: int = 30) -> int:
    """
    Clean up expired files

    Args:
        directory: Directory to clean
        minutes: File retention time (minutes)

    Returns:
        int: Number of files cleaned
    """
    if not directory.exists():
        return 0

    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    cleaned_count = 0

    try:
        for file_path in directory.glob("*"):
            if file_path.is_file():
                # Get file modification time
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                if file_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    return cleaned_count


def format_audio_duration(seconds: float) -> str:
    """Format audio duration"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def parse_dialogue_text(dialogue_text: str, num_speakers: int) -> List[str]:
    """
    Parse dialogue text into list format

    Args:
        dialogue_text: Raw dialogue text
        num_speakers: Number of speakers

    Returns:
        List[str]: Segmented dialogue list, each segment includes speaker tag
    """
    # Single speaker: add [S1] tag directly
    if num_speakers == 1:
        if not dialogue_text.startswith("[S1]"):
            return [f"[S1]{dialogue_text}"]
        else:
            return [dialogue_text]

    # Multi-speaker: split by [S1][S2]
    pattern = r'(\[S[1-4]\][^\[\]]*)'
    segments = re.findall(pattern, dialogue_text)

    # Filter empty segments
    segments = [seg.strip() for seg in segments if seg.strip()]

    return segments
