"""
FinCodeBench Runner
Executes Claude against financial coding tasks and captures full trajectories.
"""

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import anthropic

# ── Config ──────────────────────────────────────────────────────────────────
MODEL = os.environ.get("FINCODEBENCH_MODEL", "claude-haiku-4-5")  # override via FINCODEBENCH_MODEL
MAX_TOKENS = 4096
RESULTS_DIR = Path("results")
TASKS_FILE = Path("tasks/tasks.json")

# Built from ANTHROPIC_API_KEY for direct CLI use. The web service overrides this
# per run with the caller's own key (bring-your-own-key), so a missing env key
# must not break import — fall back to a placeholder that gets replaced.
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY") or "placeholder")


# ── Tool definitions ─────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "execute_python",
        "description": (
            "Execute Python code and return stdout/stderr. "
            "Use this to run calculations, parse data, or test your code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
]


# ── Tool executor ─────────────────────────────────────────────────────────────
def execute_python(code: str, timeout: int = 30) -> str:
    """Run Python code in a subprocess, return combined stdout+stderr."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmpfile = f.name

    try:
        result = subprocess.run(
            ['python3', tmpfile],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if err:
            out = out + f"\n[STDERR]\n{err}" if out else f"[STDERR]\n{err}"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Execution timed out after {timeout}s"
    except Exception as e:
        return f"[ERROR] {str(e)}"
    finally:
        try:
            os.unlink(tmpfile)
        except Exception:
            pass


TOOL_EXECUTORS = {
    "execute_python": lambda inp: execute_python(inp["code"])
}


# ── Task runner ───────────────────────────────────────────────────────────────
def run_task(task: dict, verbose: bool = True) -> dict:
    """
    Run a single task against Claude.
    Returns a result dict with trajectory, final_response, turns, and metadata.
    """
    task_id = task["id"]
    if verbose:
        print(f"\n{'─'*60}")
        print(f"Running: {task_id} ({task['category']} / {task['difficulty']})")

    # Build initial prompt
    prompt = task["prompt"]
    if task.get("context"):
        prompt = f"**Context / Data:**\n```\n{task['context']}\n```\n\n**Task:**\n{prompt}"

    messages = [{"role": "user", "content": prompt}]
    trajectory = []
    final_response = ""
    error = None
    max_turns = task.get("max_turns", 8)

    start_time = time.time()

    for turn in range(1, max_turns + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                tools=TOOLS,
                messages=messages
            )
        except Exception as e:
            error = str(e)
            if verbose:
                print(f"  [API ERROR turn {turn}] {e}")
            break

        # Parse response content
        traj_entry = {
            "turn": turn,
            "stop_reason": response.stop_reason,
            "blocks": []
        }

        tool_calls = []
        text_blocks = []

        for block in response.content:
            if block.type == "text":
                text_blocks.append(block.text)
                traj_entry["blocks"].append({
                    "type": "text",
                    "text": block.text
                })
                if verbose:
                    preview = block.text[:120].replace('\n', ' ')
                    print(f"  [turn {turn} text] {preview}...")

            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
                traj_entry["blocks"].append({
                    "type": "tool_use",
                    "name": block.name,
                    "input": block.input,
                    "id": block.id
                })
                if verbose:
                    preview = str(block.input)[:80]
                    print(f"  [turn {turn} tool_use] {block.name}: {preview}...")

        trajectory.append(traj_entry)

        # Append assistant message (raw content blocks, as expected by Anthropic SDK)
        messages.append({"role": "assistant", "content": response.content})

        # Stop if no tool calls
        if response.stop_reason == "end_turn":
            final_response = "\n".join(text_blocks)
            if verbose:
                print(f"  [done in {turn} turns]")
            break

        # Execute tool calls and append results
        if response.stop_reason == "tool_use" and tool_calls:
            tool_result_blocks = []
            for tc in tool_calls:
                executor = TOOL_EXECUTORS.get(tc["name"])
                if executor:
                    tool_output = executor(tc["input"])
                else:
                    tool_output = f"[ERROR] Unknown tool: {tc['name']}"

                # Record in trajectory
                traj_entry["blocks"].append({
                    "type": "tool_result",
                    "name": tc["name"],
                    "result": tool_output
                })

                if verbose:
                    preview = tool_output[:100].replace('\n', ' ')
                    print(f"  [turn {turn} tool_result] {preview}")

                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": tool_output
                })

            messages.append({"role": "user", "content": tool_result_blocks})

    elapsed = time.time() - start_time

    return {
        "task_id": task_id,
        "category": task["category"],
        "difficulty": task["difficulty"],
        "scoring_type": task["scoring_type"],
        "final_response": final_response,
        "trajectory": trajectory,
        "turns": len(trajectory),
        "elapsed_seconds": round(elapsed, 2),
        "error": error,
        "timestamp": datetime.utcnow().isoformat()
    }


# ── Batch runner ──────────────────────────────────────────────────────────────
def run_benchmark(
    task_ids: Optional[list] = None,
    categories: Optional[list] = None,
    verbose: bool = True
) -> list:
    """
    Run all (or filtered) tasks, save raw results to results/raw/.
    Returns list of raw result dicts.
    """
    with open(TASKS_FILE) as f:
        tasks = json.load(f)

    # Filter
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]
    if categories:
        tasks = [t for t in tasks if t["category"] in categories]

    print(f"\nRunning {len(tasks)} tasks on model: {MODEL}")

    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task in tasks:
        result = run_task(task, verbose=verbose)
        results.append(result)

        # Save immediately — don't lose progress
        out_path = raw_dir / f"{task['id']}.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

    # Save full batch
    batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_path = RESULTS_DIR / f"batch_{batch_id}.json"
    with open(batch_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results → {batch_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FinCodeBench runner")
    parser.add_argument("--task", nargs="+", help="Run specific task IDs")
    parser.add_argument("--category", nargs="+",
                        choices=["extraction", "code_generation", "computation", "agentic", "debug"],
                        help="Run only specific categories")
    parser.add_argument("--quiet", action="store_true", help="Suppress turn-by-turn output")
    args = parser.parse_args()

    results = run_benchmark(
        task_ids=args.task,
        categories=args.category,
        verbose=not args.quiet
    )
    print(f"\n{'='*60}")
    print(f"Completed {len(results)} tasks.")
    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"Errors: {len(errors)} — {[e['task_id'] for e in errors]}")
