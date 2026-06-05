"""
FinCodeBench API
FastAPI web service for triggering and retrieving eval runs on Render.
"""

import json
import os
import secrets
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────────
# On Render, use persistent disk at /data. Locally, use ./data
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"
TASKS_FILE = Path(__file__).parent / "tasks" / "tasks.json"

# Add harness to path, then load the provider registry (bring-your-own-key for
# Anthropic, OpenAI, OpenRouter, DeepSeek, Qwen, Kimi, Venice, …).
sys.path.insert(0, str(Path(__file__).parent / "harness"))
import providers

# ── Auth ──────────────────────────────────────────────────────────────────────
# Strip whitespace: a secret pasted into a hosting dashboard (e.g. Render) very
# often picks up a trailing newline or space, which would otherwise make every
# key compare unequal even when the user typed the right value.
API_KEY = os.environ.get("FINCODEBENCH_API_KEY", "").strip()

def verify_key(x_api_key: Optional[str] = Header(default=None)):
    if not API_KEY:
        return  # auth disabled — no key configured on the server
    provided = (x_api_key or "").strip()
    # Constant-time compare to avoid leaking the key via timing.
    if not secrets.compare_digest(provided.encode("utf-8"), API_KEY.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Run registry (JSON index on disk, self-healing from run folders) ──────────
# registry.json is just an index of runs. The source of truth for a *completed*
# run is its own folder (report.json + all_results.json), so we rebuild/repair
# the index from those folders on every load. That way a lost, corrupted, or
# half-written registry.json (e.g. a restart that interrupted a write — progress
# is saved to it every few seconds during a run) never hides runs whose data is
# still on disk.
def _created_at_from_run_id(run_id: str) -> str:
    # run_id looks like "YYYYmmdd_HHMMSS_<hex8>"; recover the queue time from it.
    try:
        stamp = "_".join(run_id.split("_")[:2])
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").isoformat()
    except Exception:
        return ""

def _run_entry_from_report(run_id: str, report: dict) -> dict:
    overall = report.get("overall") or {}
    return {
        "run_id": run_id,
        "status": "completed",
        "created_at": _created_at_from_run_id(run_id) or report.get("timestamp", ""),
        "completed_at": report.get("timestamp"),
        "n_tasks": report.get("n_tasks"),
        "mean_score": overall.get("mean_score"),
        "pass_rate": overall.get("pass_rate_75"),
        "provider": report.get("provider"),
        "model": report.get("model"),
        "judge_model": report.get("judge_model"),
        "cost_usd": (report.get("cost_usd") or {}).get("total"),
        "elapsed_seconds": (report.get("latency_seconds") or {}).get("total"),
        "recovered": True,
    }

def _heal_registry(reg: dict) -> dict:
    """Reconcile the registry with run folders on disk: add any completed run
    missing from the index, and promote a stale queued/running entry to
    completed once its report.json exists. Never deletes — only fills gaps."""
    if not RESULTS_DIR.exists():
        return reg
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        existing = reg.get(run_id)
        if existing and existing.get("status") == "completed":
            continue  # index already has the finished run
        report_path = run_dir / "report.json"
        if not report_path.exists():
            continue  # not finished (or nothing recoverable) — leave as-is
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            continue
        entry = _run_entry_from_report(run_id, report)
        if existing:
            # Promote in place: keep everything the index already had (provider,
            # model/judge_model overrides, task_ids, categories, queue time, …)
            # and overlay only the non-null report-derived fields. Overlaying
            # blindly would drop create-time metadata the report doesn't carry.
            merged = dict(existing)
            for k, v in entry.items():
                if v is not None:
                    merged[k] = v
            merged["status"] = "completed"
            merged["created_at"] = existing.get("created_at") or merged.get("created_at", "")
            merged.pop("recovered", None)  # was tracked all along, not recovered
            entry = merged
        reg[run_id] = entry
    return reg

# ── Storage backends ──────────────────────────────────────────────────────────
# Default: JSON files on the local disk (durable only with a persistent disk).
# If DATABASE_URL is set, runs live in Postgres instead — durable across restarts
# and shared across browsers, and the right choice on hosts without a persistent
# disk (e.g. Render's free tier). Both backends are interchangeable behind the
# _load_registry / _save_registry / _STORE.* calls the rest of the app uses.
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()


class _DiskBackend:
    """Run storage on the local filesystem. Self-heals the index from the run
    folders and writes it atomically."""

    kind = "disk"

    def load_registry(self) -> dict:
        path = RESULTS_DIR / "registry.json"
        reg = {}
        if path.exists():
            try:
                reg = json.loads(path.read_text())
            except Exception:
                reg = {}  # corrupted — rebuilt from run folders below
        return _heal_registry(reg)

    def save_registry(self, registry: dict):
        # Atomic write so a crash/restart mid-write can't corrupt the index.
        path = RESULTS_DIR / "registry.json"
        tmp = path.with_name("registry.json.tmp")
        tmp.write_text(json.dumps(registry, indent=2))
        os.replace(tmp, path)

    def get_report(self, run_id: str):
        p = RESULTS_DIR / run_id / "report.json"
        return json.loads(p.read_text()) if p.exists() else None

    def put_report(self, run_id: str, report: dict):
        d = RESULTS_DIR / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.json").write_text(json.dumps(report, indent=2))

    def get_results(self, run_id: str):
        p = RESULTS_DIR / run_id / "all_results.json"
        return json.loads(p.read_text()) if p.exists() else None

    def put_results(self, run_id: str, results: list):
        d = RESULTS_DIR / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "all_results.json").write_text(json.dumps(results, indent=2))

    def delete_run(self, run_id: str):
        import shutil
        d = RESULTS_DIR / run_id
        if d.exists():
            shutil.rmtree(d)

    def startup(self):
        if not os.environ.get("DATA_DIR"):
            print(
                "WARNING: DATA_DIR is not set — writing runs to ephemeral './data'. "
                "Run history will be LOST on every redeploy/restart. Attach a "
                "persistent disk and set DATA_DIR (e.g. /data), or set DATABASE_URL "
                "to a Postgres database for durable storage (works on Render free)."
            )
        else:
            print(f"Run storage: disk ({RESULTS_DIR.resolve()})")
        with registry_lock:                      # persist any runs recovered from disk
            self.save_registry(self.load_registry())


class _PgBackend:
    """Run storage in Postgres (DATABASE_URL). One row per run holds the index
    entry plus the report and per-task results as JSONB — durable across restarts
    and shared across browsers. A short-lived connection is opened per call, so
    it's safe to use from the background run thread (psycopg connections are not
    shareable across threads)."""

    kind = "postgres"

    def __init__(self, dsn: str):
        import psycopg
        from psycopg.types.json import Json
        self._psycopg = psycopg
        self._Json = Json
        self._dsn = dsn
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id      TEXT PRIMARY KEY,
                    created_at  TEXT,
                    meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
                    report      JSONB,
                    results     JSONB,
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    def load_registry(self) -> dict:
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT run_id, meta FROM runs")
            return {run_id: meta for run_id, meta in cur.fetchall()}

    def save_registry(self, registry: dict):
        # Upsert each entry's index metadata (report/results columns untouched).
        # Deletions are handled explicitly by delete_run, so no global wipe here.
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            for run_id, meta in registry.items():
                cur.execute(
                    """
                    INSERT INTO runs (run_id, created_at, meta, updated_at)
                    VALUES (%s, %s, %s, now())
                    ON CONFLICT (run_id) DO UPDATE
                        SET meta = EXCLUDED.meta,
                            created_at = EXCLUDED.created_at,
                            updated_at = now()
                    """,
                    (run_id, meta.get("created_at"), self._Json(meta)),
                )

    def _get_col(self, run_id, col):
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT {col} FROM runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def get_report(self, run_id): return self._get_col(run_id, "report")
    def get_results(self, run_id): return self._get_col(run_id, "results")

    def _put_col(self, run_id, col, value):
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO runs (run_id, {col}, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (run_id) DO UPDATE SET {col} = EXCLUDED.{col}, updated_at = now()
                """,
                (run_id, self._Json(value)),
            )

    def put_report(self, run_id, report): self._put_col(run_id, "report", report)
    def put_results(self, run_id, results): self._put_col(run_id, "results", results)

    def delete_run(self, run_id):
        with self._psycopg.connect(self._dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))

    def startup(self):
        print("Run storage: Postgres (DATABASE_URL) — durable across restarts.")


def _make_store():
    if DATABASE_URL:
        try:
            store = _PgBackend(DATABASE_URL)
            print("Initialized Postgres run storage from DATABASE_URL.")
            return store
        except Exception as e:
            print(f"WARNING: DATABASE_URL is set but Postgres init failed ({e}). "
                  f"Falling back to disk storage — runs may not persist.")
    return _DiskBackend()


_STORE = _make_store()


def _load_registry() -> dict:
    return _STORE.load_registry()

def _save_registry(registry: dict):
    _STORE.save_registry(registry)

registry_lock = threading.Lock()


# ── Run execution (background thread) ─────────────────────────────────────────
def _execute_run(run_id: str, task_ids: Optional[list], categories: Optional[list],
                 provider: str, api_key: str, model: str, judge_model: str):
    """Runs the full eval pipeline in a background thread on the visitor's own
    provider + key (bring-your-own-key). Updates registry on completion."""
    from runner import run_benchmark
    from scorer import score_task
    from judge import score_pending_judge_tasks
    from eval import generate_report

    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Update status: running
    with registry_lock:
        reg = _load_registry()
        reg[run_id]["status"] = "running"
        reg[run_id]["started_at"] = datetime.utcnow().isoformat()
        _save_registry(reg)

    try:
        # Override module globals so the runner writes to run_dir and every model
        # call uses the visitor's provider/key/model, not the server's (restored
        # in finally). A single client serves both runner and judge.
        import runner as runner_mod
        import judge as judge_mod
        import eval as eval_mod
        original_results = runner_mod.RESULTS_DIR
        original_runner_client = runner_mod.client
        original_judge_client = judge_mod.client
        original_model = runner_mod.MODEL
        original_provider = runner_mod.PROVIDER
        original_judge_model = judge_mod.JUDGE_MODEL
        runner_mod.RESULTS_DIR = run_dir
        eval_mod.RESULTS_DIR = run_dir
        run_client = providers.ChatClient(provider, api_key)
        runner_mod.client = run_client
        judge_mod.client = run_client
        runner_mod.PROVIDER = provider
        runner_mod.MODEL = model
        judge_mod.JUDGE_MODEL = judge_model

        # Live progress: write phase/counter to the registry so the dashboard
        # can show "Running 12/28 · agentic-003" instead of a bare "running".
        def _progress(phase):
            def cb(done, total, current):
                with registry_lock:
                    reg = _load_registry()
                    if run_id in reg:
                        reg[run_id]["progress"] = {
                            "phase": phase, "done": done, "total": total, "current": current,
                        }
                        _save_registry(reg)
            return cb

        # Load tasks
        with open(Path(__file__).parent / "tasks" / "tasks.json") as f:
            all_tasks = json.load(f)
        tasks_map = {t["id"]: t for t in all_tasks}

        # Execute
        results = run_benchmark(
            task_ids=task_ids,
            categories=categories,
            verbose=False,
            progress_callback=_progress("running"),
        )

        # Score deterministic
        _progress("scoring")(len(results), len(results), None)
        for result in results:
            task = tasks_map.get(result["task_id"])
            if task and task.get("scoring_type") != "llm_judge":
                result["score_result"] = score_task(task, result)

        # LLM judge
        calib_path = str(run_dir / "calibration_template.json")
        results = score_pending_judge_tasks(
            results, tasks_map, calibration_path=calib_path,
            progress_callback=_progress("judging"),
        )

        # Generate report and persist it + the per-task results via the store
        # (disk files, or Postgres rows when DATABASE_URL is set).
        report = generate_report(results, tasks_map)
        _STORE.put_report(run_id, report)
        _STORE.put_results(run_id, results)

        # Update registry: completed
        with registry_lock:
            reg = _load_registry()
            reg[run_id]["status"] = "completed"
            reg[run_id]["completed_at"] = datetime.utcnow().isoformat()
            reg[run_id]["n_tasks"] = report.get("n_tasks", 0)
            reg[run_id]["mean_score"] = report.get("overall", {}).get("mean_score")
            reg[run_id]["pass_rate"] = report.get("overall", {}).get("pass_rate_75")
            # Top-line model / cost / latency so the runs table can show them
            # without loading each run's full report.
            reg[run_id]["model"] = report.get("model")
            reg[run_id]["judge_model"] = report.get("judge_model")
            reg[run_id]["cost_usd"] = (report.get("cost_usd") or {}).get("total")
            reg[run_id]["elapsed_seconds"] = (report.get("latency_seconds") or {}).get("total")
            _save_registry(reg)

    except Exception as e:
        with registry_lock:
            reg = _load_registry()
            reg[run_id]["status"] = "failed"
            reg[run_id]["error"] = str(e)
            _save_registry(reg)
        raise
    finally:
        # Restore module globals
        runner_mod.RESULTS_DIR = original_results
        eval_mod.RESULTS_DIR = original_results
        runner_mod.client = original_runner_client
        judge_mod.client = original_judge_client
        runner_mod.MODEL = original_model
        runner_mod.PROVIDER = original_provider
        judge_mod.JUDGE_MODEL = original_judge_model


# ── FastAPI app ───────────────────────────────────────────────────────────────
def _startup_repair_and_warn():
    """Let the active storage backend run its boot tasks: the disk backend
    repairs the index and warns on ephemeral storage; Postgres just reports in."""
    try:
        _STORE.startup()
    except Exception as e:
        print(f"Storage startup failed (non-fatal): {e}")


@asynccontextmanager
async def _lifespan(app):
    _startup_repair_and_warn()
    yield


app = FastAPI(
    title="FinCodeBench",
    description="Financial coding agent eval suite — benchmark Claude Code on financial tasks",
    version="1.0.0",
    lifespan=_lifespan,
)


# ── Request/response models ───────────────────────────────────────────────────
VALID_CATEGORIES = {"extraction", "code_generation", "computation", "agentic", "debug"}

class RunRequest(BaseModel):
    task_ids: Optional[list[str]] = None
    categories: Optional[list[str]] = None
    provider: Optional[str] = None      # anthropic (default), openai, openrouter, deepseek, qwen, kimi, venice
    model: Optional[str] = None         # runner model override (defaults to provider's)
    judge_model: Optional[str] = None   # judge model override (defaults to provider's)

    def validate_categories(self):
        if self.categories:
            invalid = set(self.categories) - VALID_CATEGORIES
            if invalid:
                raise HTTPException(400, f"Invalid categories: {invalid}. Valid: {VALID_CATEGORIES}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/providers")
def list_providers():
    """Public, secrets-free registry of supported model providers — name, label,
    key hint, default models, and where to get a key. Drives the dashboard's
    bring-your-own-key selector."""
    return {"providers": providers.public_registry()}


@app.get("/providers/{provider}/models")
def list_provider_models(
    provider: str,
    x_provider_api_key: Optional[str] = Header(default=None),
    x_anthropic_api_key: Optional[str] = Header(default=None),
):
    """Model catalogue for a provider. With the caller's own key (in the
    X-Provider-Api-Key header, used only for this lookup and never stored), it
    returns the provider's full live /models list. Without a key it returns the
    full list anyway for providers whose /models is public (e.g. OpenRouter),
    otherwise the registry's static suggestions. Always 200 with a `source` of
    "live" or "static" so the dashboard can populate the dropdown regardless."""
    try:
        name, cfg = providers.resolve_provider(provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    api_key = (x_provider_api_key or x_anthropic_api_key or "").strip()

    # Fetch live when we have a key, or when the provider's listing is public.
    if api_key or cfg.get("public_models"):
        try:
            models = providers.ChatClient(name, api_key or "public").list_models()
            if models:
                return {"provider": name, "models": models, "source": "live"}
        except Exception as e:
            # Degrade gracefully — the dashboard still has the static suggestions.
            return {"provider": name, "models": cfg["models"], "source": "static",
                    "error": str(e)}

    return {"provider": name, "models": cfg["models"], "source": "static"}


# ── Dashboard (public, read-only) ─────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def dashboard_index():
    """Serve the single-page methodology + results dashboard."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "Dashboard not built")
    return FileResponse(index)


@app.get("/dashboard/data")
def dashboard_data():
    """
    Public payload for the dashboard: the full task set plus a summary of all
    runs. No secrets and no trajectories — safe to serve without the API key so
    the methodology, tasks, and result charts are publicly viewable. POST /runs
    is bring-your-own-key (the caller supplies their own provider key); the
    destructive DELETE stays gated by FINCODEBENCH_API_KEY.
    """
    with open(TASKS_FILE) as f:
        tasks = json.load(f)
    with registry_lock:
        reg = _load_registry()
    runs = sorted(reg.values(), key=lambda r: r.get("created_at", ""), reverse=True)
    return {"tasks": tasks, "runs": runs, "providers": providers.public_registry()}


@app.get("/dashboard/runs/{run_id}")
def dashboard_run_report(run_id: str):
    """Public aggregate report (scores only) for one completed run."""
    report = _STORE.get_report(run_id)
    if report is None:
        raise HTTPException(404, f"No report for run {run_id}")
    return report


@app.get("/tasks")
def list_tasks(
    category: Optional[str] = Query(default=None),
    difficulty: Optional[str] = Query(default=None),
    x_api_key: Optional[str] = Header(default=None)
):
    """List all available tasks with optional filters."""
    verify_key(x_api_key)
    with open(Path(__file__).parent / "tasks" / "tasks.json") as f:
        tasks = json.load(f)

    if category:
        tasks = [t for t in tasks if t["category"] == category]
    if difficulty:
        tasks = [t for t in tasks if t["difficulty"] == difficulty]

    return {
        "n": len(tasks),
        "tasks": [
            {
                "id": t["id"],
                "category": t["category"],
                "difficulty": t["difficulty"],
                "scoring_type": t["scoring_type"],
                "prompt_preview": t["prompt"][:100] + "..."
            }
            for t in tasks
        ]
    }


@app.post("/runs", status_code=202)
def create_run(
    req: RunRequest,
    background_tasks: BackgroundTasks,
    x_provider_api_key: Optional[str] = Header(default=None),
    x_anthropic_api_key: Optional[str] = Header(default=None),
):
    """
    Trigger a new eval run on the caller's own provider + key (bring-your-own-key).

    The provider is chosen in the request body (`provider`: anthropic [default],
    openai, openrouter, deepseek, qwen, kimi, venice). The key is supplied via
    the X-Provider-Api-Key header (X-Anthropic-Api-Key still works for Anthropic),
    used for every model call in the run, billed to the caller, and never stored
    or logged. Optionally override `model` / `judge_model`; otherwise the
    provider's defaults are used. Returns run_id immediately; execution is async.
    Poll GET /runs/{run_id} for status.
    """
    # Validate provider and resolve per-provider model defaults.
    try:
        provider, cfg = providers.resolve_provider(req.provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    api_key = (x_provider_api_key or x_anthropic_api_key or "").strip()
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail=f"Bring your own {cfg['label']} API key ({cfg['key_hint']}): "
                   f"send it in the X-Provider-Api-Key header.",
        )
    req.validate_categories()

    model = (req.model or "").strip() or cfg["default_model"]
    judge_model = (req.judge_model or "").strip() or cfg["default_judge_model"]

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    # Register (the API key is deliberately never written to the registry; the
    # provider/model are, so the dashboard can show what each run used).
    with registry_lock:
        reg = _load_registry()
        reg[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "task_ids": req.task_ids,
            "categories": req.categories,
            "provider": provider,
            "model": model,
            "judge_model": judge_model,
        }
        _save_registry(reg)

    # Fire background thread (Render web service stays alive between requests)
    background_tasks.add_task(
        _execute_run,
        run_id=run_id,
        task_ids=req.task_ids,
        categories=req.categories,
        provider=provider,
        api_key=api_key,
        model=model,
        judge_model=judge_model,
    )

    return {
        "run_id": run_id,
        "status": "queued",
        "provider": provider,
        "model": model,
        "poll_url": f"/runs/{run_id}"
    }


@app.get("/runs")
def list_runs(x_api_key: Optional[str] = Header(default=None)):
    """List all runs with their status and top-line scores."""
    verify_key(x_api_key)
    with registry_lock:
        reg = _load_registry()
    return {
        "n": len(reg),
        "runs": sorted(reg.values(), key=lambda r: r.get("created_at", ""), reverse=True)
    }


@app.get("/runs/{run_id}")
def get_run(run_id: str, x_api_key: Optional[str] = Header(default=None)):
    """Get run status. If completed, includes full report."""
    verify_key(x_api_key)
    with registry_lock:
        reg = _load_registry()

    if run_id not in reg:
        raise HTTPException(404, f"Run {run_id} not found")

    run_meta = reg[run_id]

    if run_meta["status"] != "completed":
        return run_meta

    # Load report from the active store (disk file or Postgres row)
    report = _STORE.get_report(run_id)
    if report is not None:
        return {**run_meta, "report": report}

    return run_meta


@app.get("/runs/{run_id}/results")
def get_run_results(run_id: str, x_api_key: Optional[str] = Header(default=None)):
    """Get per-task results for a completed run."""
    verify_key(x_api_key)
    with registry_lock:
        reg = _load_registry()

    if run_id not in reg:
        raise HTTPException(404, f"Run {run_id} not found")

    if reg[run_id]["status"] != "completed":
        raise HTTPException(409, f"Run {run_id} is {reg[run_id]['status']}, not completed yet")

    results = _STORE.get_results(run_id)
    if results is None:
        raise HTTPException(404, "Results not found")

    # Return lightweight version (no full trajectories by default)
    return {
        "run_id": run_id,
        "task_results": [
            {
                "task_id": r["task_id"],
                "model": r.get("model"),
                "category": r.get("category"),
                "difficulty": r.get("difficulty"),
                "turns": r.get("turns"),
                "elapsed_seconds": r.get("elapsed_seconds"),
                "cost_usd": round((r.get("cost_usd") or 0.0)
                                  + (r.get("score_result", {}).get("judge_cost_usd") or 0.0), 6)
                            if (r.get("cost_usd") is not None
                                or r.get("score_result", {}).get("judge_cost_usd") is not None)
                            else None,
                "score": r.get("score_result", {}).get("score"),
                "method": r.get("score_result", {}).get("method"),
                "reasoning": r.get("score_result", {}).get("reasoning"),
                "error": r.get("error")
            }
            for r in results
        ]
    }


@app.get("/runs/{run_id}/trajectory/{task_id}")
def get_trajectory(
    run_id: str,
    task_id: str,
    x_api_key: Optional[str] = Header(default=None)
):
    """Get full trajectory (all turns, tool calls, outputs) for a specific task."""
    verify_key(x_api_key)
    results = _STORE.get_results(run_id)
    if results is None:
        raise HTTPException(404, "Run results not found")

    match = next((r for r in results if r["task_id"] == task_id), None)
    if not match:
        raise HTTPException(404, f"Task {task_id} not found in run {run_id}")

    return match


@app.delete("/runs/{run_id}")
def delete_run(run_id: str, x_api_key: Optional[str] = Header(default=None)):
    """Delete a run and its results from the active store."""
    verify_key(x_api_key)
    with registry_lock:
        reg = _load_registry()
        if run_id not in reg:
            raise HTTPException(404, f"Run {run_id} not found")
        del reg[run_id]
        _save_registry(reg)

    # Remove the run's report/results (disk folder, or Postgres row)
    _STORE.delete_run(run_id)

    return {"deleted": run_id}
