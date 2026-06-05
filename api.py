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
            # keep the queue metadata the index already had
            entry["created_at"] = existing.get("created_at") or entry["created_at"]
            for k in ("task_ids", "categories", "started_at"):
                if existing.get(k) is not None:
                    entry[k] = existing[k]
            entry.pop("recovered", None)
        reg[run_id] = entry
    return reg

def _load_registry() -> dict:
    path = RESULTS_DIR / "registry.json"
    reg = {}
    if path.exists():
        try:
            reg = json.loads(path.read_text())
        except Exception:
            reg = {}  # corrupted — rebuilt from run folders below
    return _heal_registry(reg)

def _save_registry(registry: dict):
    # Write atomically so a crash/restart mid-write can't corrupt the registry.
    path = RESULTS_DIR / "registry.json"
    tmp = path.with_name("registry.json.tmp")
    tmp.write_text(json.dumps(registry, indent=2))
    os.replace(tmp, path)

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


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinCodeBench",
    description="Financial coding agent eval suite — benchmark Claude Code on financial tasks",
    version="1.0.0"
)


@app.on_event("startup")
def _startup_repair_and_warn():
    """Repair the run index from disk on boot, and flag ephemeral storage —
    the usual reason runs 'disappear' after a redeploy."""
    if not os.environ.get("DATA_DIR"):
        print(
            "WARNING: DATA_DIR is not set — writing runs to ephemeral './data'. "
            "Run history will be LOST on every redeploy/restart. On Render, attach "
            "a persistent disk and set DATA_DIR to its mount path (e.g. /data)."
        )
    else:
        print(f"Run data directory: {RESULTS_DIR.resolve()}")
    try:
        with registry_lock:
            _save_registry(_load_registry())  # persist any runs recovered from disk
    except Exception as e:
        print(f"Registry self-heal at startup failed (non-fatal): {e}")


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
