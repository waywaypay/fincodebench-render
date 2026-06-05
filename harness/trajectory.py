"""
FinCodeBench Trajectory Analyzer
The PM eval skill — read transcripts, classify failure modes, find patterns.
This is what you present at an Anthropic interview.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Optional


RESULTS_DIR = Path("results")


# ── Failure mode taxonomy ─────────────────────────────────────────────────────
FAILURE_MODES = {
    "wrong_output":       "Correct approach but arithmetic/logic error in final answer",
    "incomplete":         "Started correctly but stopped before completing the task",
    "wrong_json_format":  "Output not parseable as the requested JSON structure",
    "code_crash":         "Generated code that fails at runtime (syntax/import/runtime error)",
    "off_by_one":         "Correct formula but fencepost error (indexing, range, period)",
    "hallucinated_data":  "Invented numbers not present in the provided context",
    "tool_overuse":       "Used execute_python for trivial arithmetic that could be done inline",
    "unnecessary_hedging":"Correct answer buried in excessive caveats and disclaimers",
    "wrong_formula":      "Used the wrong financial formula or mis-specified a calculation",
    "gave_up":            "Refused or said it couldn't complete the task",
}


def load_results(results_dir: Optional[str] = None) -> list[dict]:
    """Load all raw result JSON files."""
    d = Path(results_dir) if results_dir else RESULTS_DIR / "raw"
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]


def load_tasks() -> dict:
    with open("tasks/tasks.json") as f:
        return {t["id"]: t for t in json.load(f)}


def analyze_trajectory(result: dict) -> dict:
    """
    Analyze a single task trajectory. Returns:
    - turn_count, tool_calls, retries, first_code_turn, correct_on_first_try
    - text length of final response
    """
    trajectory = result.get("trajectory", [])
    tool_call_count = 0
    first_code_turn = None
    retry_count = 0
    prev_was_error = False

    for entry in trajectory:
        for block in entry.get("blocks", []):
            if block["type"] == "tool_use" and block["name"] == "execute_python":
                tool_call_count += 1
                if first_code_turn is None:
                    first_code_turn = entry["turn"]

            if block["type"] == "tool_result":
                has_error = "[ERROR]" in block.get("result", "") or "[STDERR]" in block.get("result", "")
                if has_error:
                    retry_count += 1
                prev_was_error = has_error

    score = result.get("score_result", {}).get("score")
    if score is not None:
        normalized = float(score) if isinstance(score, (int, float)) else 0.0
    else:
        normalized = None

    return {
        "task_id": result["task_id"],
        "category": result.get("category"),
        "difficulty": result.get("difficulty"),
        "turns": result.get("turns", len(trajectory)),
        "tool_calls": tool_call_count,
        "code_errors": retry_count,
        "first_code_turn": first_code_turn,
        "final_response_length": len(result.get("final_response", "")),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "score": normalized,
        "passed": normalized is not None and normalized >= 0.75
    }


def summarize_trajectories(results: list[dict]) -> dict:
    """Aggregate trajectory stats across all tasks."""
    analyses = [analyze_trajectory(r) for r in results]

    scored = [a for a in analyses if a["score"] is not None]
    passed = [a for a in scored if a["passed"]]
    failed = [a for a in scored if not a["passed"]]

    def mean(lst): return round(sum(lst) / len(lst), 2) if lst else None

    return {
        "n_total": len(analyses),
        "n_scored": len(scored),
        "n_passed": len(passed),
        "overall_pass_rate": round(len(passed) / len(scored), 3) if scored else None,
        "avg_turns": {
            "passing": mean([a["turns"] for a in passed]),
            "failing": mean([a["turns"] for a in failed])
        },
        "avg_tool_calls": {
            "passing": mean([a["tool_calls"] for a in passed]),
            "failing": mean([a["tool_calls"] for a in failed])
        },
        "avg_code_errors": {
            "passing": mean([a["code_errors"] for a in passed]),
            "failing": mean([a["code_errors"] for a in failed])
        },
        "avg_elapsed": mean([a["elapsed_seconds"] for a in analyses if a["elapsed_seconds"]]),
        "by_category": _by_key(analyses, "category"),
        "by_difficulty": _by_key(analyses, "difficulty"),
        "task_detail": sorted(analyses, key=lambda x: (x["score"] or 0, x["task_id"]))
    }


def _by_key(analyses, key):
    groups = {}
    for a in analyses:
        k = a.get(key, "unknown")
        groups.setdefault(k, []).append(a)
    result = {}
    for k, group in groups.items():
        scored = [a for a in group if a["score"] is not None]
        passed = [a for a in scored if a["passed"]]
        result[k] = {
            "n": len(group),
            "pass_rate": round(len(passed) / len(scored), 3) if scored else None,
            "avg_turns": round(sum(a["turns"] for a in group) / len(group), 1)
        }
    return result


def classify_failure(result: dict, task: dict) -> list[str]:
    """
    Manually-assisted failure classification.
    Prints the failed task response for you to label.
    Returns a list of failure mode keys.
    """
    response = result.get("final_response", "")
    score_info = result.get("score_result", {})

    print(f"\n{'─'*60}")
    print(f"TASK: {result['task_id']}  ({task['category']} / {task['difficulty']})")
    print(f"SCORE: {score_info.get('score')}  METHOD: {score_info.get('method')}")
    print(f"\nPROMPT: {task['prompt'][:200]}...")
    if task.get("expected_output"):
        print(f"EXPECTED: {str(task['expected_output'])[:200]}")
    print(f"\nRESPONSE (first 600 chars):\n{response[:600]}")
    if score_info.get("error"):
        print(f"\nSCORER ERROR: {score_info['error']}")

    print("\nAvailable failure modes:")
    for i, (k, v) in enumerate(FAILURE_MODES.items()):
        print(f"  {i:2d}. {k:25s} — {v}")

    raw = input("\nEnter failure mode numbers (comma-separated), or press Enter to skip: ").strip()
    if not raw:
        return []

    modes = list(FAILURE_MODES.keys())
    selected = []
    for idx_str in raw.split(","):
        try:
            idx = int(idx_str.strip())
            selected.append(modes[idx])
        except (ValueError, IndexError):
            pass
    return selected


def run_failure_analysis(results: list[dict], tasks: dict) -> dict:
    """
    Interactive failure classification session.
    Walks through all failed tasks, prompts for classification.
    Saves labeled results.
    """
    failures = [r for r in results if r.get("score_result", {}).get("score", 1.0) < 0.75]
    print(f"\n{len(failures)} failed tasks to classify.")

    labels = {}
    for result in failures:
        task = tasks.get(result["task_id"])
        if not task:
            continue
        modes = classify_failure(result, task)
        labels[result["task_id"]] = modes

    # Aggregate
    all_modes = [m for modes in labels.values() for m in modes]
    mode_counts = Counter(all_modes)

    analysis = {
        "n_failures": len(failures),
        "failure_mode_counts": dict(mode_counts.most_common()),
        "by_task": labels,
        "top_failure_mode": mode_counts.most_common(1)[0][0] if mode_counts else None
    }

    out_path = RESULTS_DIR / "failure_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nFailure analysis saved → {out_path}")
    print("\nTop failure modes:")
    for mode, count in mode_counts.most_common(5):
        print(f"  {count:3d}x  {mode}")

    return analysis


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FinCodeBench trajectory analysis")
    parser.add_argument("--summary", action="store_true", help="Print trajectory summary stats")
    parser.add_argument("--classify", action="store_true", help="Interactive failure classification")
    args = parser.parse_args()

    results = load_results()
    tasks = load_tasks()

    if not results:
        print("No results found. Run eval.py first.")
    elif args.classify:
        run_failure_analysis(results, tasks)
    else:
        summary = summarize_trajectories(results)
        print(json.dumps(summary, indent=2))
