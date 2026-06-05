"""
FinCodeBench Eval Pipeline
Runs the full benchmark: execute → score → judge → report.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from runner import run_benchmark
from scorer import score_task
from judge import score_pending_judge_tasks


RESULTS_DIR = Path("results")
TASKS_FILE = Path("tasks/tasks.json")


def load_tasks() -> dict:
    with open(TASKS_FILE) as f:
        return {t["id"]: t for t in json.load(f)}


def run_full_eval(
    task_ids: Optional[list] = None,
    categories: Optional[list] = None,
    skip_judge: bool = False,
    verbose: bool = True
) -> dict:
    """
    Full pipeline:
    1. Run all tasks via runner
    2. Score deterministic tasks (exact, fuzzy, functional)
    3. Run LLM-as-judge on llm_judge tasks
    4. Produce summary report
    """
    tasks = load_tasks()

    # Step 1: Execute
    print("\n" + "="*60)
    print("STEP 1: Running tasks")
    print("="*60)
    results = run_benchmark(
        task_ids=task_ids,
        categories=categories,
        verbose=verbose
    )

    # Step 2: Score deterministic tasks
    print("\n" + "="*60)
    print("STEP 2: Scoring deterministic tasks")
    print("="*60)
    for result in results:
        task = tasks.get(result["task_id"])
        if not task:
            continue
        if task.get("scoring_type") != "llm_judge":
            score_result = score_task(task, result)
            result["score_result"] = score_result
            status = "✓" if score_result.get("score", 0) == 1.0 else "✗"
            print(f"  {status} {result['task_id']:25s} score={score_result.get('score')}")

    # Step 3: LLM-as-judge
    if not skip_judge:
        print("\n" + "="*60)
        print("STEP 3: LLM-as-judge scoring")
        print("="*60)
        calib_path = str(RESULTS_DIR / "calibration_template.json")
        results = score_pending_judge_tasks(results, tasks, calibration_path=calib_path)

    # Step 4: Report
    print("\n" + "="*60)
    print("STEP 4: Generating report")
    print("="*60)
    report = generate_report(results, tasks)

    # Save report
    report_path = RESULTS_DIR / f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print_report(report)
    print(f"\nFull report saved → {report_path}")
    return report


def generate_report(results: list, tasks: dict) -> dict:
    """Aggregate scores into summary statistics."""
    import runner  # read the (possibly per-run overridden) provider/model

    by_category = {}
    by_difficulty = {}
    all_scores = []
    task_scores = []

    for result in results:
        task = tasks.get(result["task_id"])
        if not task:
            continue

        sr = result.get("score_result", {})
        raw_score = sr.get("score")
        method = sr.get("method", "unknown")

        # Normalize: functional/exact = 0 or 1; fuzzy_dict = 0–1; llm_judge = normalized
        if method == "llm_judge":
            normalized = sr.get("normalized", 0.0)
        elif method == "fuzzy_dict":
            normalized = raw_score if raw_score is not None else 0.0
        elif raw_score is not None:
            normalized = float(raw_score)
        else:
            normalized = None

        cat = result["category"]
        diff = result["difficulty"]

        by_category.setdefault(cat, [])
        by_difficulty.setdefault(diff, [])

        if normalized is not None:
            by_category[cat].append(normalized)
            by_difficulty[diff].append(normalized)
            all_scores.append(normalized)

        task_scores.append({
            "task_id": result["task_id"],
            "category": cat,
            "difficulty": diff,
            "score": normalized,
            "method": method,
            "turns": result.get("turns", 0),
            "elapsed": result.get("elapsed_seconds"),
            "reasoning": sr.get("reasoning") if method == "llm_judge" else None
        })

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else None
    def pass_rate(lst, threshold=0.75): return round(sum(1 for x in lst if x >= threshold) / len(lst), 4) if lst else None

    return {
        "model": runner.MODEL,
        "provider": runner.PROVIDER,
        "judge_model": __import__("judge").JUDGE_MODEL,
        "timestamp": datetime.utcnow().isoformat(),
        "n_tasks": len(results),
        "overall": {
            "mean_score": avg(all_scores),
            "pass_rate_75": pass_rate(all_scores),
        },
        "by_category": {k: {"mean": avg(v), "n": len(v), "pass_rate": pass_rate(v)} for k, v in by_category.items()},
        "by_difficulty": {k: {"mean": avg(v), "n": len(v)} for k, v in by_difficulty.items()},
        "task_scores": sorted(task_scores, key=lambda x: (x["category"], x["task_id"]))
    }


def print_report(report: dict):
    """Pretty-print summary to console."""
    print(f"\n{'='*60}")
    print(f"FINCODEBENCH RESULTS")
    print(f"{'='*60}")
    overall = report["overall"]
    print(f"Tasks: {report['n_tasks']}  |  Mean score: {overall['mean_score']}  |  Pass rate (≥0.75): {overall['pass_rate_75']}")

    print(f"\nBy Category:")
    for cat, stats in sorted(report["by_category"].items()):
        bar = "█" * int((stats["mean"] or 0) * 20)
        print(f"  {cat:20s}  {bar:20s}  {stats['mean']:.3f}  (n={stats['n']})")

    print(f"\nBy Difficulty:")
    for diff, stats in sorted(report["by_difficulty"].items()):
        print(f"  {diff:10s}  mean={stats['mean']:.3f}  n={stats['n']}")

    print(f"\nTask-level:")
    for ts in report["task_scores"]:
        score_str = f"{ts['score']:.3f}" if ts["score"] is not None else "pending"
        flag = "✓" if ts["score"] and ts["score"] >= 0.75 else "✗"
        print(f"  {flag} {ts['task_id']:25s}  {score_str:6s}  turns={ts['turns']}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FinCodeBench eval pipeline")
    parser.add_argument("--task", nargs="+", help="Specific task IDs")
    parser.add_argument("--category", nargs="+",
                        choices=["extraction", "code_generation", "computation", "agentic", "debug"])
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge step")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_full_eval(
        task_ids=args.task,
        categories=args.category,
        skip_judge=args.skip_judge,
        verbose=not args.quiet
    )
