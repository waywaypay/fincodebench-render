"""
FinCodeBench API
FastAPI web service for triggering and retrieving eval runs on Render.
"""

import json
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────────
# On Render, use persistent disk at /data. Locally, use ./data
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add harness to path
sys.path.insert(0, str(Path(__file__).parent / "harness"))

# ── Auth ──────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("FINCODEBENCH_API_KEY", "")

def verify_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


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
def _execute_run(run_id: str, task_ids: Optional[list], categories: Optional[list]):
    """Runs the full eval pipeline in a background thread. Updates registry on completion."""
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
        # Override results dir so runner writes to run_dir
        import runner as runner_mod
        import eval as eval_mod
        original_results = runner_mod.RESULTS_DIR
        runner_mod.RESULTS_DIR = run_dir
        eval_mod.RESULTS_DIR = run_dir

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
        with registry_lock:
            reg = _load_registry()
            reg[run_id]["status"] = "failed"
            reg[run_id]["error"] = str(e)
            _save_registry(reg)
        raise
    finally:
        # Restore paths
        runner_mod.RESULTS_DIR = original_results
        eval_mod.RESULTS_DIR = original_results


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
    x_api_key: Optional[str] = Header(default=None)
):
    """
    Trigger a new eval run. Returns run_id immediately; execution is async.
    Poll GET /runs/{run_id} for status and results.
    """
    verify_key(x_api_key)
    req.validate_categories()

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    # Register
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
        categories=req.categories
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
