# CLAUDE.md — infrastructure/queue/

## Responsibility

ARQ job queue wiring. Defines how CPU-heavy signing and verification jobs
are enqueued, executed, and polled for status.

## Key design decisions

**CPU-bound work must run in ProcessPoolExecutor.**
ARQ runs on asyncio. Signing a video is 30–120 seconds of CPU work.
Blocking the event loop will starve all other jobs and requests.

```python
# jobs.py — correct pattern
async def process_sign_job(ctx: dict, content_id: str, input_path: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        ctx["process_pool"],   # ProcessPoolExecutor passed via WorkerSettings.ctx
        _sign_cpu_work,        # top-level function (picklable)
        content_id,
        input_path,
    )
    return result
```

**Jobs must be idempotent.**
ARQ requeues jobs if a worker dies mid-execution. The sign job must be safe
to run twice for the same `content_id` (check if already processed before writing to DB).

**Two-phase signing architecture (Phase 7.2):**

`_sign_sync()` (subprocess) calls `_sign_*_cpu()` and returns a `RawSigningPayload` dict —
no async I/O, no `_NullRegistry`. `process_sign_job` (parent async loop) calls
`_persist_payload(payload, storage, registry)` after `run_in_executor` returns:

```python
# jobs.py — correct CPU/IO split
def _sign_sync(media_path, cert_data, private_key_pem, pepper, org_id) -> dict:
    # Routes to _sign_audio_cpu / _sign_video_cpu / _sign_av_cpu based on media type
    # Returns RawSigningPayload — no storage, no DB, no asyncio.run()
    ...

async def process_sign_job(ctx, ...):
    payload = await loop.run_in_executor(process_pool, _sign_sync, ...)
    # I/O phase with real adapters from ctx
    await _persist_payload(payload, ctx["storage"], ctx["registry"])
```

**Status polling — Redis key takes precedence over ARQ (Phase 7.1):**

`process_sign_job` writes a Redis key `job:{job_id}:status` (TTL 3600 s) at each
milestone so the API endpoint can return a `progress` integer (0–100):

```python
# jobs.py — progress tracking pattern
async def _set_job_status(redis, job_id: str, status: dict) -> None:
    await redis.set(f"job:{job_id}:status", json.dumps(status), ex=3600)

# In process_sign_job:
await _set_job_status(redis, job_id, {"status": "processing", "progress": 0})
# ... CPU work in subprocess ...
await _set_job_status(redis, job_id, {"status": "processing", "progress": 20})
# ... I/O phase in parent loop ...
await _set_job_status(redis, job_id, {"status": "completed", "progress": 100, "result": ...})
```

`ctx["redis"]` and `ctx["job_id"]` are injected automatically by ARQ.

**Fallback: ARQ native Job.info() (no progress):**
```python
from arq.jobs import Job, JobStatus
job = Job(job_id=job_id, redis=redis_pool)
status = await job.status()    # JobStatus enum
info   = await job.info()      # includes result when complete
```

The `GET /sign/{job_id}` endpoint checks the Redis key first; only falls back to
ARQ's native status if the key is absent (e.g. jobs enqueued before Phase 7.1).

## redis_pool.py

```python
from arq.connections import RedisSettings

REDIS_SETTINGS = RedisSettings(
    host="your-endpoint.upstash.io",
    port=6379,
    password="YOUR_PASSWORD",
    ssl=True,
    conn_timeout=10,
)
```
Upstash free tier: 10k commands/day, 256 MB. Sufficient for MVP.
Use the TCP/TLS endpoint — NOT the REST API endpoint.

## WorkerSettings (worker.py)

```python
class WorkerSettings:
    functions    = [process_sign_job]
    redis_settings = REDIS_SETTINGS
    job_timeout  = 180        # 3 min — covers 150 MB video worst case
    max_jobs     = 4          # concurrent jobs per worker process
    retry_jobs   = True
    ctx          = {}         # populated in on_startup
```

## Validation

```bash
# Start worker in one terminal
make worker

# In another terminal, enqueue a test job
python -m pytest tests/unit/test_queue_jobs.py -v
```

Expected: job enqueues → status = "queued" → status = "in_progress" → status = "complete".
