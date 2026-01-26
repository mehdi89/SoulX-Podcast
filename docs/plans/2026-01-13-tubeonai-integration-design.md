# TubeOnAI Podcast Integration Design

**Date:** 2026-01-13
**Status:** ⚠️ HISTORICAL - Partially Superseded
**Author:** Claude (with Mehdi)

> **Note (2026-01-15):** This document describes the original design. The implementation has evolved:
> - **Deployment**: Now uses Azure Container Apps with GPU (not manual VM setup)
> - **Job distribution**: Now uses Azure Storage Queue (not API polling)
> - **Worker tracking**: Removed - Azure Container Apps handles scaling/health
> - **Heartbeat system**: Removed - replaced by HTTP health endpoint for Azure liveness probes
>
> See `readme.md` for current deployment architecture.

## Overview

Integrate SoulX-Podcast with TubeOnAI to allow users to generate podcast audio from video summaries. The system uses a decoupled architecture with pull-based GPU workers that can scale across multiple servers.

## Requirements

- User clicks "Generate Podcast" on a summary page
- Background processing (user continues browsing)
- Notifications: in-app (WebSocket) + email fallback
- Expected volume: ~50 requests/day (low)
- Multi-GPU server support for easy scaling
- S3 storage for generated audio files
- Database queue (simple, no external dependencies)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           TubeOnAI                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐   │
│  │  Web App │───▶│   Laravel    │───▶│  Database Queue Table   │   │
│  │  (User)  │    │   Backend    │    │  (podcast_jobs)         │   │
│  └──────────┘    └──────────────┘    └─────────────────────────┘   │
│                         │                        ▲                  │
│                         │ Notifications          │ Poll/Complete    │
│                         ▼                        │                  │
│              ┌──────────────────┐                │                  │
│              │ WebSocket/Email  │                │                  │
│              └──────────────────┘                │                  │
└─────────────────────────────────────────────────┼──────────────────┘
                                                   │
                          ┌────────────────────────┼────────────────┐
                          │         Podcast Worker API              │
                          │  ┌─────────────┐  ┌─────────────┐       │
                          │  │ GPU Server 1│  │ GPU Server 2│  ...  │
                          │  └──────┬──────┘  └──────┬──────┘       │
                          │         │                │               │
                          │         └────────┬───────┘               │
                          │                  ▼                       │
                          │         ┌─────────────┐                  │
                          │         │     S3      │                  │
                          │         │   Storage   │                  │
                          │         └─────────────┘                  │
                          └─────────────────────────────────────────┘
```

### Components

1. **TubeOnAI Backend (Laravel)** - Manages job queue, exposes API for workers, sends notifications
2. **Podcast Workers (Python)** - GPU servers running SoulX-Podcast, poll for jobs, upload results
3. **S3 Storage** - Shared storage for generated audio files

### Flow

1. User clicks "Generate Podcast" → Job added to database queue
2. Worker polls TubeOnAI API for pending jobs → Claims a job
3. Worker generates audio using SoulX-Podcast model
4. Worker uploads WAV file to S3
5. Worker calls completion API → TubeOnAI notifies user via WebSocket + email

---

## Database Schema

### podcast_jobs

```sql
CREATE TABLE podcast_jobs (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,

    -- Job identification
    job_id          VARCHAR(36) UNIQUE NOT NULL,  -- UUID for external reference

    -- Source reference
    summary_id      BIGINT NOT NULL,              -- Link to TubeOnAI summary
    user_id         BIGINT NOT NULL,              -- User who requested

    -- Input
    script_text     TEXT NOT NULL,                -- "[S1] Hello... [S2] Hi..."
    seed            INT DEFAULT 1988,             -- Voice variation seed

    -- Status tracking
    status          ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    worker_id       VARCHAR(64) NULL,             -- Which worker claimed it
    retry_count     INT DEFAULT 0,                -- Number of retries

    -- Output
    output_url      VARCHAR(512) NULL,            -- S3 URL when complete
    duration_seconds INT NULL,                    -- Audio duration
    error_message   TEXT NULL,                    -- Error details if failed

    -- Timing
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at      TIMESTAMP NULL,               -- When worker started
    completed_at    TIMESTAMP NULL,               -- When finished

    -- Indexes
    INDEX idx_status (status),
    INDEX idx_user (user_id),
    INDEX idx_summary (summary_id)
);
```

**Status Flow:**
- `pending` → Job waiting for a worker
- `processing` → Worker claimed and working on it
- `completed` → Done, `output_url` contains S3 link
- `failed` → Error occurred, check `error_message`

### podcast_workers

```sql
CREATE TABLE podcast_workers (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    worker_id       VARCHAR(64) UNIQUE NOT NULL,

    -- Server info
    server_ip       VARCHAR(45) NOT NULL,         -- IPv4 or IPv6
    server_name     VARCHAR(128) NULL,            -- Optional friendly name

    -- Status
    status          ENUM('online', 'offline', 'processing') DEFAULT 'offline',
    current_job_id  VARCHAR(36) NULL,

    -- Stats
    jobs_completed  INT DEFAULT 0,
    jobs_failed     INT DEFAULT 0,
    last_heartbeat  TIMESTAMP NULL,
    last_job_at     TIMESTAMP NULL,

    -- Metadata
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_status (status)
);
```

---

## TubeOnAI API Endpoints (for Workers)

Base URL: `https://api.tubeonai.com/podcast-worker/v1`
Authentication: `Authorization: Bearer <worker-token>`

### POST /jobs/claim

Worker requests a pending job.

**Request:**
```json
{
    "worker_id": "gpu-server-1"
}
```

**Response (200 - job found):**
```json
{
    "job_id": "abc-123-def-456",
    "script_text": "[S1] Welcome to the show!\n[S2] Thanks for having me!",
    "seed": 1988,
    "s3_upload_path": "podcasts/2026/01/abc-123-def-456.wav"
}
```

**Response (204 - no jobs available):**
Empty body

### POST /jobs/{job_id}/complete

Worker marks job as successfully completed.

**Request:**
```json
{
    "worker_id": "gpu-server-1",
    "output_url": "https://s3.amazonaws.com/tubeonai-media/podcasts/2026/01/abc-123-def-456.wav",
    "duration_seconds": 145
}
```

**Response (200):**
```json
{
    "status": "ok"
}
```

### POST /jobs/{job_id}/failed

Worker marks job as failed.

**Request:**
```json
{
    "worker_id": "gpu-server-1",
    "error_message": "Out of GPU memory"
}
```

**Response (200):**
```json
{
    "status": "ok"
}
```

### POST /workers/heartbeat

Worker sends periodic health check.

**Request:**
```json
{
    "worker_id": "gpu-server-1",
    "server_ip": "20.85.123.45",
    "server_name": "Azure ML East US",
    "status": "idle",
    "current_job_id": null
}
```

**Response (200):**
```json
{
    "status": "ok",
    "registered": true
}
```

---

## Worker Application

### Directory Structure

```
SoulX-Podcast/
├── worker/
│   ├── __init__.py
│   ├── config.py          # Environment configuration
│   ├── api_client.py      # TubeOnAI API client
│   ├── s3_client.py       # S3 upload client
│   ├── processor.py       # Audio generation wrapper
│   ├── main.py            # Main worker loop
│   └── .env.example       # Environment template
├── run_worker.py          # Entry point
└── setup_worker.sh        # One-command setup script
```

### Worker Loop

```python
# Pseudocode for worker/main.py

while True:
    # 1. Send heartbeat
    api_client.heartbeat(worker_id, server_ip, "idle")

    # 2. Poll for job
    job = api_client.claim_job(worker_id)

    if job is None:
        sleep(POLL_INTERVAL)  # Default: 10 seconds
        continue

    try:
        # 3. Update status
        api_client.heartbeat(worker_id, server_ip, "processing", job.job_id)

        # 4. Generate podcast
        audio_path = processor.generate(
            script=job.script_text,
            seed=job.seed
        )

        # 5. Upload to S3
        output_url = s3_client.upload(
            local_path=audio_path,
            s3_path=job.s3_upload_path
        )

        # 6. Get audio duration
        duration = get_audio_duration(audio_path)

        # 7. Mark complete
        api_client.complete_job(
            job_id=job.job_id,
            output_url=output_url,
            duration_seconds=duration
        )

    except Exception as e:
        # 8. Mark failed
        api_client.fail_job(
            job_id=job.job_id,
            error_message=str(e)
        )

    # Cleanup temp files
    cleanup(audio_path)
```

### Environment Configuration

```bash
# worker/.env.example

# Worker identity (unique per server)
WORKER_ID=gpu-server-1
SERVER_NAME=Azure ML East US

# TubeOnAI API
TUBEONAI_API_URL=https://api.tubeonai.com/podcast-worker/v1
TUBEONAI_API_TOKEN=your-token-here

# S3 Storage
S3_BUCKET=tubeonai-media
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
S3_REGION=us-east-1

# Model
MODEL_PATH=pretrained_models/SoulX-Podcast-1.7B

# Worker settings
POLL_INTERVAL=10
HEARTBEAT_INTERVAL=60
```

### One-Command Setup

```bash
#!/bin/bash
# setup_worker.sh - Run on new GPU server

set -e

echo "=== SoulX-Podcast Worker Setup ==="

# 1. Clone repo
git clone https://github.com/mehdi89/SoulX-Podcast.git
cd SoulX-Podcast

# 2. Create conda environment
conda create -n soulxpodcast -y python=3.11
source $(conda info --base)/etc/profile.d/conda.sh
conda activate soulxpodcast

# 3. Install dependencies
pip install -r requirements.txt
pip install boto3 requests  # Worker dependencies

# 4. Download model
huggingface-cli download Soul-AILab/SoulX-Podcast-1.7B \
    --local-dir pretrained_models/SoulX-Podcast-1.7B

# 5. Create .env from template
cp worker/.env.example worker/.env

echo "✅ Setup complete!"
echo "📝 Edit worker/.env with your credentials, then run:"
echo "   python run_worker.py"
```

### Systemd Service (Production)

```ini
# worker/podcast-worker.service

[Unit]
Description=SoulX-Podcast Worker
After=network.target

[Service]
Type=simple
User=azureuser
WorkingDirectory=/home/azureuser/SoulX-Podcast
Environment="PATH=/home/azureuser/anaconda3/envs/soulxpodcast/bin"
ExecStart=/home/azureuser/anaconda3/envs/soulxpodcast/bin/python run_worker.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Install service:**
```bash
sudo cp worker/podcast-worker.service /etc/systemd/system/
sudo systemctl enable podcast-worker
sudo systemctl start podcast-worker
```

---

## TubeOnAI Web App Integration

### UI States

| State | Button Text | Action |
|-------|-------------|--------|
| No podcast | "Generate Podcast" | Clickable, creates job |
| Pending/Processing | "Generating..." | Disabled, shows spinner |
| Completed | "Play Podcast" | Opens audio player |
| Failed | "Retry Podcast" | Clickable, creates new job |

### User Flow

1. **User clicks "Generate Podcast"**
   - Frontend calls `POST /api/summaries/{id}/generate-podcast`
   - Backend validates and creates job in `podcast_jobs` table
   - Toast: "Podcast queued! We'll notify you when it's ready."
   - Button changes to "Generating..." (disabled)

2. **User continues browsing**
   - WebSocket connection already exists
   - Backend sends notification when job completes

3. **Job completes**
   - Backend broadcasts via WebSocket: `{ type: "podcast_ready", summary_id: 123 }`
   - Frontend shows toast: "Your podcast is ready! Click to listen"
   - Email sent with download link

4. **User plays podcast**
   - Button now shows "Play Podcast"
   - Clicking opens audio player with S3 URL

### In-App Notification (WebSocket)

```json
{
    "type": "podcast_ready",
    "summary_id": 123,
    "summary_title": "How AI Works",
    "podcast_url": "https://s3.amazonaws.com/...",
    "duration_seconds": 145
}
```

### Email Notification

```
Subject: Your podcast is ready!

Hi {user_name},

Your podcast for "{summary_title}" is ready to listen!

[Listen Now] → links to summary page

Duration: 2 minutes 45 seconds
```

---

## Error Handling

### Rate Limiting

| Rule | Value | Configurable |
|------|-------|--------------|
| Max pending jobs per user | 1 | Yes |
| Max podcasts per user per day | 5 | Yes |

**Response when limit exceeded:**
```json
{
    "error": "rate_limit_exceeded",
    "message": "Please wait for your current podcast to complete."
}
```

### Script Validation

Before queueing, validate:
- Script is not empty
- Contains at least one `[S1]` or `[S2]` tag
- Script length < 10,000 characters

### Stale Job Recovery

Jobs stuck in `processing` for > 30 minutes are reset to `pending`.

```php
// Laravel Scheduled Task - Run every 5 minutes
PodcastJob::where('status', 'processing')
    ->where('started_at', '<', now()->subMinutes(30))
    ->update([
        'status' => 'pending',
        'worker_id' => null,
        'started_at' => null
    ]);
```

### Retry Logic

- **Max retries:** 2 automatic retries
- **Backoff:** 1 minute between retries
- **After max retries:** Mark as permanently failed, notify user

### Worker Offline Detection

```php
// Laravel Scheduled Task - Run every 2 minutes
PodcastWorker::where('last_heartbeat', '<', now()->subMinutes(5))
    ->where('status', '!=', 'offline')
    ->update(['status' => 'offline']);

// Alert if all workers offline
if (PodcastWorker::where('status', '!=', 'offline')->count() === 0) {
    // Send alert to admin
}
```

---

## Admin Dashboard

Simple view for monitoring:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Podcast Workers                                                     │
├──────────────────────────────────────────────────────────────────────┤
│  Worker        IP              Status       Last Seen   Jobs Today   │
│  ────────────────────────────────────────────────────────────────    │
│  gpu-server-1  20.85.123.45    🟢 Online    2 min ago   12           │
│  gpu-server-2  20.85.124.89    🟢 Processing 30 sec     8            │
│  gpu-server-3  52.168.50.12    🔴 Offline   2 hrs ago   0            │
├──────────────────────────────────────────────────────────────────────┤
│  Queue: 3 pending │ Today: 23 completed │ 2 failed                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Checklist

### TubeOnAI Backend (Laravel)

- [ ] Create `podcast_jobs` migration
- [ ] Create `podcast_workers` migration
- [ ] Create `PodcastJob` model
- [ ] Create `PodcastWorker` model
- [ ] Implement worker API endpoints:
  - [ ] `POST /podcast-worker/v1/jobs/claim`
  - [ ] `POST /podcast-worker/v1/jobs/{id}/complete`
  - [ ] `POST /podcast-worker/v1/jobs/{id}/failed`
  - [ ] `POST /podcast-worker/v1/workers/heartbeat`
- [ ] Add authentication middleware for worker endpoints
- [ ] Implement job creation endpoint for frontend
- [ ] Add WebSocket broadcast on job completion
- [ ] Add email notification on job completion
- [ ] Add scheduled task for stale job recovery
- [ ] Add scheduled task for worker offline detection
- [ ] Add rate limiting (1 pending, 5/day)
- [ ] Add admin dashboard view

### TubeOnAI Web App

- [ ] Add "Generate Podcast" button to summary page
- [ ] Handle button states (pending, processing, completed, failed)
- [ ] Add WebSocket listener for `podcast_ready` event
- [ ] Show toast notification when podcast ready
- [ ] Add audio player for completed podcasts

### SoulX-Podcast Worker

- [ ] Create `worker/` directory structure
- [ ] Implement `config.py` - environment loading
- [ ] Implement `api_client.py` - TubeOnAI API calls
- [ ] Implement `s3_client.py` - S3 upload
- [ ] Implement `processor.py` - audio generation
- [ ] Implement `main.py` - worker loop
- [ ] Create `.env.example` template
- [ ] Create `setup_worker.sh` script
- [ ] Create `podcast-worker.service` systemd file
- [ ] Update `requirements.txt` with boto3, requests
- [ ] Test end-to-end flow

---

## Configuration Reference

### TubeOnAI Config

```php
// config/podcast.php
return [
    'max_pending_per_user' => env('PODCAST_MAX_PENDING', 1),
    'max_per_day_per_user' => env('PODCAST_MAX_DAILY', 5),
    'stale_job_timeout_minutes' => env('PODCAST_STALE_TIMEOUT', 30),
    'worker_offline_minutes' => env('PODCAST_WORKER_OFFLINE', 5),
    'max_retries' => env('PODCAST_MAX_RETRIES', 2),
];
```

### Worker Config

```bash
WORKER_ID=gpu-server-1
SERVER_NAME=Azure ML East US
TUBEONAI_API_URL=https://api.tubeonai.com/podcast-worker/v1
TUBEONAI_API_TOKEN=xxx
S3_BUCKET=tubeonai-media
S3_ACCESS_KEY=xxx
S3_SECRET_KEY=xxx
S3_REGION=us-east-1
MODEL_PATH=pretrained_models/SoulX-Podcast-1.7B
POLL_INTERVAL=10
HEARTBEAT_INTERVAL=60
```
