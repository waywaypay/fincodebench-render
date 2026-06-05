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

import anthropic
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

# Add harness to path
sys.path.insert(0, str(Path(__file__).parent / "harness"))

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


# ── Anthropic error handling ──────────────────────────────────────────────────
# Bring-your-own-key runs fail for boring, fixable reasons — most often an invalid
# key or a key with no credits. The raw SDK exception ("Error code: 400 - {...your
# credit balance is too low...}") is opaque to a dashboard user, so map the known
# cases to a clear HTTP status + actionable message. Billing is the awkward one:
# "credit balance is too low" comes back as a 400 invalid_request_error (and
# occasionally a 403 billing_error), so it can't be told apart by HTTP status
# alone — inspect the message/type, not just the exception class.
def classify_anthropic_error(exc: Exception) -> tuple[int, str]:
    """Map an Anthropic SDK exception to (http_status, human_message)."""
    # `.message` is the clean inner message on APIStatusError subclasses; for
    # everything else fall back to str(exc).
    message = str(getattr(exc, "message", "") or exc)
    err_type = getattr(exc, "type", None)  # "invalid_request_error", "billing_error", ...
    low = message.lower()

    if err_type == "billing_error" or "credit balance" in low:
        return 402, (
            "The Anthropic API key you provided has insufficient credits. Add "
            "credits at https://console.anthropic.com/settings/billing, then "
            "start the run again."
        )
    if isinstance(exc, anthropic.AuthenticationError):
        return 401, (
            "The Anthropic API key is invalid or has been revoked. It should "
            "start with 'sk-ant-' — double-check it and try again."
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return 403, f"This Anthropic API key lacks permission for the requested model. {message}"
    if isinstance(exc, anthropic.NotFoundError):
        return 404, f"The requested model isn't available to this key. {message}"
    if isinstance(exc, anthropic.RateLimitError):
        return 429, "The Anthropic API rate limit was hit. Wait a moment, then start the run again."
    if isinstance(exc, anthropic.APIConnectionError):
        return 502, "Could not reach the Anthropic API. Check connectivity and try again."
    if isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code or 502
        # 529 overloaded / 5xx server errors are transient — tell the caller to retry.
        # (OverloadedError isn't re-exported at the top level in every SDK version,
        #  so classify by status/type rather than by the exception class.)
        if status == 529 or err_type == "overloaded_error":
            return 503, "The Anthropic API is temporarily overloaded. Try again in a few moments."
        if status >= 500:
            return 503, "The Anthropic API is temporarily unavailable. Try again in a few moments."
        return status, f"Anthropic API error {status}: {message}"
    # Not an Anthropic API error — surface the message without leaking a stack trace.
    return 500, f"Run failed: {message}"


def _validate_anthropic_key(api_key: str, model: str) -> None:
    """Fail fast before queuing a long run: a 1-token completion verifies the
    caller's key is valid and funded. A free count_tokens call would NOT surface
    the 'credit balance is too low' billing error, so use a real (tiny) call.
    The key is used transiently here and never stored. Raises HTTPException with
    a clear status + message on any failure; returns None on success."""
    probe = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=20.0)
    try:
        probe.messages.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
    except Exception as exc:
        status, message = classify_anthropic_error(exc)
        raise HTTPException(status_code=status, detail=message)


# ── In-memory run registry (survives restarts via JSON on disk) ───────────────
def _load_registry() -> dict:
    path = RESULTS_DIR / "registry.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

def _save_registry(registry: dict):
    path = RESULTS_DIR / "registry.json"
    path.write_text(json.dumps(registry, indent=2))

registry_lock = threading.Lock()


# ── Run execution (background thread) ─────────────────────────────────────────
def _execute_run(run_id: str, task_ids: Optional[list], categories: Optional[list],
                 anthropic_api_key: str):
    """Runs the full eval pipeline in a background thread on the visitor's own
    Anthropic key (bring-your-own-key). Updates registry on completion."""
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
        # Override module globals so the runner writes to run_dir and every
        # Anthropic call uses the visitor's key, not the server's (restored below).
        import runner as runner_mod
        import judge as judge_mod
        import eval as eval_mod
        original_results = runner_mod.RESULTS_DIR
        original_runner_client = runner_mod.client
        original_judge_client = judge_mod.client
        runner_mod.RESULTS_DIR = run_dir
        eval_mod.RESULTS_DIR = run_dir
        run_client = anthropic.Anthropic(api_key=anthropic_api_key)
        runner_mod.client = run_client
        judge_mod.client = run_client

        # Load tasks
        with open(Path(__file__).parent / "tasks" / "tasks.json") as f:
            all_tasks = json.load(f)
        tasks_map = {t["id"]: t for t in all_tasks}

        # Execute
        results = run_benchmark(
            task_ids=task_ids,
            categories=categories,
            verbose=False
        )

        # Score deterministic
        for result in results:
            task = tasks_map.get(result["task_id"])
            if task and task.get("scoring_type") != "llm_judge":
                result["score_result"] = score_task(task, result)

        # LLM judge
        calib_path = str(run_dir / "calibration_template.json")
        results = score_pending_judge_tasks(results, tasks_map, calibration_path=calib_path)

        # Generate report
        report = generate_report(results, tasks_map)
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2))

        # Save all results
        all_results_path = run_dir / "all_results.json"
        all_results_path.write_text(json.dumps(results, indent=2))

        # Update registry: completed
        with registry_lock:
            reg = _load_registry()
            reg[run_id]["status"] = "completed"
            reg[run_id]["completed_at"] = datetime.utcnow().isoformat()
            reg[run_id]["n_tasks"] = report.get("n_tasks", 0)
            reg[run_id]["mean_score"] = report.get("overall", {}).get("mean_score")
            reg[run_id]["pass_rate"] = report.get("overall", {}).get("pass_rate_75")
            _save_registry(reg)

    except Exception as e:
        # Store a clean, actionable message (e.g. credits ran out mid-run, or a
        # rate limit / overload) rather than the raw SDK repr the dashboard would
        # otherwise show verbatim.
        _, message = classify_anthropic_error(e)
        with registry_lock:
            reg = _load_registry()
            reg[run_id]["status"] = "failed"
            reg[run_id]["error"] = message
            _save_registry(reg)
        raise
    finally:
        # Restore module globals
        runner_mod.RESULTS_DIR = original_results
        eval_mod.RESULTS_DIR = original_results
        runner_mod.client = original_runner_client
        judge_mod.client = original_judge_client


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinCodeBench",
    description="Financial coding agent eval suite — benchmark Claude Code on financial tasks",
    version="1.0.0"
)


# ── Request/response models ───────────────────────────────────────────────────
VALID_CATEGORIES = {"extraction", "code_generation", "computation", "agentic", "debug"}

class RunRequest(BaseModel):
    task_ids: Optional[list[str]] = None
    categories: Optional[list[str]] = None

    def validate_categories(self):
        if self.categories:
            invalid = set(self.categories) - VALID_CATEGORIES
            if invalid:
                raise HTTPException(400, f"Invalid categories: {invalid}. Valid: {VALID_CATEGORIES}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


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
    is bring-your-own-key (the caller supplies their own Anthropic key); the
    destructive DELETE stays gated by FINCODEBENCH_API_KEY.
    """
    with open(TASKS_FILE) as f:
        tasks = json.load(f)
    with registry_lock:
        reg = _load_registry()
    runs = sorted(reg.values(), key=lambda r: r.get("created_at", ""), reverse=True)
    return {"tasks": tasks, "runs": runs}


@app.get("/dashboard/runs/{run_id}")
def dashboard_run_report(run_id: str):
    """Public aggregate report (scores only) for one completed run."""
    report_path = RESULTS_DIR / run_id / "report.json"
    if not report_path.exists():
        raise HTTPException(404, f"No report for run {run_id}")
    return json.loads(report_path.read_text())


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
    x_anthropic_api_key: Optional[str] = Header(default=None),
):
    """
    Trigger a new eval run on the caller's own Anthropic key (bring-your-own-key).
    The key is supplied via the X-Anthropic-Api-Key header, used for every model
    call in the run, billed to the caller, and never stored or logged. Returns
    run_id immediately; execution is async. Poll GET /runs/{run_id} for status.
    """
    anthropic_api_key = (x_anthropic_api_key or "").strip()
    if not anthropic_api_key:
        raise HTTPException(
            status_code=401,
            detail="Bring your own Anthropic API key: send it in the "
                   "X-Anthropic-Api-Key header (sk-ant-...).",
        )
    req.validate_categories()

    # Fail fast: verify the key is valid and funded before queuing a long run, so
    # the caller gets an immediate, clear error instead of a run that silently
    # flips to "failed" with an opaque SDK string after polling. Probe the same
    # model the runner will use, so a key without access to it is caught too.
    import runner as runner_mod
    _validate_anthropic_key(anthropic_api_key, runner_mod.MODEL)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    # Register (the Anthropic key is deliberately never written to the registry)
    with registry_lock:
        reg = _load_registry()
        reg[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "task_ids": req.task_ids,
            "categories": req.categories,
        }
        _save_registry(reg)

    # Fire background thread (Render web service stays alive between requests)
    background_tasks.add_task(
        _execute_run,
        run_id=run_id,
        task_ids=req.task_ids,
        categories=req.categories,
        anthropic_api_key=anthropic_api_key,
    )

    return {
        "run_id": run_id,
        "status": "queued",
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

    # Load report
    report_path = RESULTS_DIR / run_id / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
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

    results_path = RESULTS_DIR / run_id / "all_results.json"
    if not results_path.exists():
        raise HTTPException(404, "Results file not found")

    results = json.loads(results_path.read_text())

    # Return lightweight version (no full trajectories by default)
    return {
        "run_id": run_id,
        "task_results": [
            {
                "task_id": r["task_id"],
                "category": r.get("category"),
                "difficulty": r.get("difficulty"),
                "turns": r.get("turns"),
                "elapsed_seconds": r.get("elapsed_seconds"),
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
    results_path = RESULTS_DIR / run_id / "all_results.json"
    if not results_path.exists():
        raise HTTPException(404, "Run results not found")

    results = json.loads(results_path.read_text())
    match = next((r for r in results if r["task_id"] == task_id), None)
    if not match:
        raise HTTPException(404, f"Task {task_id} not found in run {run_id}")

    return match


@app.delete("/runs/{run_id}")
def delete_run(run_id: str, x_api_key: Optional[str] = Header(default=None)):
    """Delete a run and its results from disk."""
    verify_key(x_api_key)
    with registry_lock:
        reg = _load_registry()
        if run_id not in reg:
            raise HTTPException(404, f"Run {run_id} not found")
        del reg[run_id]
        _save_registry(reg)

    # Remove files
    import shutil
    run_dir = RESULTS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)

    return {"deleted": run_id}
