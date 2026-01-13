"""
FastAPI Main Application for SoulX-Podcast Voice Cloning API
"""
import asyncio
import logging
import re
from contextlib import asynccontextmanager
from typing import List
import json
import threading

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import scipy.io.wavfile as wavfile

from api.config import config
from api.models import (
    TaskCreateResponse,
    TaskStatusResponse,
    HealthResponse,
    ErrorResponse,
    TaskStatus,
)
from api.service import get_service
from api.tasks import get_task_manager
from api.utils import (
    generate_task_id,
    save_upload_file,
    validate_audio_files,
    validate_dialogue_format,
    cleanup_old_files,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hardcoded English voice configuration
ENGLISH_VOICES = {
    "S1": {
        "path": "example/audios/en-Alice_woman.wav",
        "prompt_text": "Welcome to Tech Talk, where we discuss the latest developments in artificial intelligence and technology."
    },
    "S2": {
        "path": "example/audios/en-Frank_man.wav",
        "prompt_text": "I'm excited to share my thoughts on how AI is transforming our world and what the future might hold."
    }
}

# Global lock for concurrent inference control
inference_lock = threading.Lock()
active_inferences = 0
MAX_CONCURRENT_SYNC_INFERENCES = 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Starting SoulX-Podcast API...")

    # Initialize model (in main thread)
    logger.info("Loading model...")
    service = get_service()
    if not service.is_loaded():
        raise RuntimeError("Failed to load model")

    # Start task manager
    task_manager = get_task_manager()
    task_manager.start_workers(config.max_concurrent_tasks)

    # Start file cleanup task
    async def cleanup_task():
        while True:
            await asyncio.sleep(600)  # Clean up every 10 minutes
            count = cleanup_old_files(config.temp_dir, config.file_cleanup_minutes)
            count += cleanup_old_files(config.output_dir, config.file_cleanup_minutes)
            if count > 0:
                logger.info(f"Cleaned up {count} old files")

    cleanup_task_handle = asyncio.create_task(cleanup_task())

    logger.info("API started successfully!")

    yield

    # Shutdown
    logger.info("Shutting down API...")
    cleanup_task_handle.cancel()

    # Quick shutdown for task manager
    try:
        await asyncio.wait_for(task_manager.shutdown(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Task manager shutdown timeout, forcing exit")

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("API shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="SoulX-Podcast API",
    description="Podcast generation API with pre-configured English voices",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "name": "SoulX-Podcast API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "voices": {
            "S1": "Alice (female host)",
            "S2": "Frank (male guest)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    service = get_service()
    task_manager = get_task_manager()

    return HealthResponse(
        status="healthy",
        model_loaded=service.is_loaded(),
        gpu_available=torch.cuda.is_available(),
        llm_engine=config.llm_engine,
        active_tasks=task_manager.get_active_task_count(),
        version="1.0.0"
    )


@app.post("/generate-async", response_model=TaskCreateResponse, tags=["Generation"])
async def generate_async_simple(
    dialogue_text: str = Form(..., description="Podcast script with [S1] and [S2] speaker tags"),
    seed: int = Form(default=1988, description="Random seed for reproducibility"),
    temperature: float = Form(default=0.6, ge=0.1, le=2.0, description="Sampling temperature"),
    top_k: int = Form(default=100, ge=1, le=500, description="Top-K sampling"),
    top_p: float = Form(default=0.9, ge=0.0, le=1.0, description="Top-P sampling"),
):
    """
    Generate podcast audio asynchronously (returns task ID).

    Uses pre-configured English voices:
    - S1: Alice (female host)
    - S2: Frank (male guest)

    Script format: [S1] Hello! [S2] Hi there! [S1] Let's begin...
    """
    task_id = generate_task_id()

    try:
        # Validate dialogue format
        dialogue_text = dialogue_text.strip()
        if not dialogue_text:
            raise HTTPException(status_code=400, detail="Dialogue text cannot be empty")

        # Check for speaker tags
        if not re.search(r'\[S[1-2]\]', dialogue_text):
            raise HTTPException(
                status_code=400,
                detail="Script must use [S1] and [S2] speaker tags. Example: [S1] Hello! [S2] Hi!"
            )

        # Use hardcoded English voices
        audio_paths = [ENGLISH_VOICES["S1"]["path"], ENGLISH_VOICES["S2"]["path"]]
        prompt_texts = [ENGLISH_VOICES["S1"]["prompt_text"], ENGLISH_VOICES["S2"]["prompt_text"]]

        # Create async task
        task_manager = get_task_manager()
        task = await task_manager.create_task(
            task_id=task_id,
            prompt_audio_paths=audio_paths,
            prompt_texts=prompt_texts,
            dialogue_text=dialogue_text,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.25,
        )

        logger.info(f"Async task created: task_id={task_id}")

        return TaskCreateResponse(
            task_id=task_id,
            status=task.status,
            created_at=task.created_at,
            message=f"Task created. Queue position: {task_manager.queue.qsize()}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-async-custom", response_model=TaskCreateResponse, tags=["Generation"])
async def generate_async_custom(
    prompt_audio: List[UploadFile] = File(..., description="Reference audio files (1-4)"),
    prompt_texts: str = Form(..., description="Reference texts as JSON array"),
    dialogue_text: str = Form(..., description="Dialogue text to generate"),
    seed: int = Form(default=1988, description="Random seed"),
    temperature: float = Form(default=0.6, ge=0.1, le=2.0, description="Sampling temperature"),
    top_k: int = Form(default=100, ge=1, le=500, description="Top-K sampling"),
    top_p: float = Form(default=0.9, ge=0.0, le=1.0, description="Top-P sampling"),
    repetition_penalty: float = Form(default=1.25, ge=1.0, le=2.0, description="Repetition penalty"),
):
    """
    Generate podcast with custom voice files (advanced endpoint).

    Upload your own voice reference audio files.
    """
    task_id = generate_task_id()

    try:
        # Validate audio files
        validate_audio_files(prompt_audio)

        # Parse prompt_texts
        try:
            prompt_text_list = json.loads(prompt_texts)
            if not isinstance(prompt_text_list, list):
                raise ValueError("prompt_texts must be a JSON array")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"prompt_texts JSON parse error: {str(e)}")

        # Validate count match
        if len(prompt_audio) != len(prompt_text_list):
            raise HTTPException(
                status_code=400,
                detail=f"Audio count ({len(prompt_audio)}) doesn't match text count ({len(prompt_text_list)})"
            )

        # Validate dialogue format
        is_valid, error_msg = validate_dialogue_format(dialogue_text, len(prompt_audio))
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # Save uploaded files
        audio_paths = []
        for i, file in enumerate(prompt_audio):
            path = save_upload_file(file, task_id, i)
            audio_paths.append(str(path))

        # Create async task
        task_manager = get_task_manager()
        task = await task_manager.create_task(
            task_id=task_id,
            prompt_audio_paths=audio_paths,
            prompt_texts=prompt_text_list,
            dialogue_text=dialogue_text,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        logger.info(f"Custom async task created: task_id={task_id}")

        return TaskCreateResponse(
            task_id=task_id,
            status=task.status,
            created_at=task.created_at,
            message=f"Task created. Queue position: {task_manager.queue.qsize()}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
async def get_task_status(task_id: str):
    """Query task status"""
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    # Build result URL
    result_url = None
    if task.status == TaskStatus.COMPLETED and task.result_path:
        result_url = f"/download/{task.result_path.name}"

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        result_url=result_url,
        error=task.error,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


@app.get("/download/{filename}", tags=["Download"])
async def download_file(filename: str):
    """Download generated audio file"""
    file_path = config.output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level="info"
    )
