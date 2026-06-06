"""
FinCodeBench Eval Pipeline
Runs the full benchmark: execute → score → judge → report.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from runner import run_benchmark
from scorer import score_task
from judge import score_pending_judge_tasks


RESULTS_DIR = Path("results")
TASKS_FILE = Path("tasks/tasks.json")

# Bumped whenever tasks or scoring semantics change, so runs recorded under an
# older methodology are never silently compared against newer ones. Surfaced in
# the report as `benchmark_version`. 1.2.0: scoring-comparability work + the
# agentic_real expansion from 3 to 8 synthetic two-filing tasks.
BENCHMARK_VERSION = "1.2.0"


def _diagnostic_enabled() -> bool:
    """Opt-in failure diagnosis on functional tasks (extra judge calls on
    failures only). Enable with FINCODEBENCH_DIAGNOSTIC_JUDGE=1."""
    return os.environ.get("FINCODEBENCH_DIAGNOSTIC_JUDGE", "").lower() in ("1", "true", "yes")


# Scoring methods fall into two granularities. BINARY methods can only ever score
# 0 or 1, so a category mean built from them is really a pass rate. PARTIAL_CREDIT
# methods yield continuous 0–1 scores. Averaging across the two scales — or
# comparing a binary category's mean against a partial-credit one — overstates
# differences, which is exactly why code_generation (all-binary) looks far weaker
# than agentic (partial-credit) even at similar capability. The report flags this
# rather than hiding it.
_BINARY_METHODS = {"functional", "fuzzy_number", "exact_json"}
_PARTIAL_CREDIT_METHODS = {"fuzzy_dict", "llm_judge"}


def _scale_of(method: str) -> str:
    if method in _BINARY_METHODS:
        return "binary"
    if method in _PARTIAL_CREDIT_METHODS:
        return "partial_credit"
    return "unknown"


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
    run_diag = _diagnostic_enabled()
    for result in results:
        task = tasks.get(result["task_id"])
        if not task:
            continue
        if task.get("scoring_type") != "llm_judge":
            score_result = score_task(task, result, run_diagnostic=run_diag)
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
    """Aggregate scores, cost, and latency into summary statistics."""

    by_category = {}
    by_difficulty = {}
    by_category_methods = {}   # {cat: {method: count}} — which scales a category mixes
    functional_failures = {}   # {failure_type: count} — from the diagnostic judge
    all_scores = []
    task_scores = []

    runner_models = set()
    runner_providers = set()
    judge_models = set()
    runner_cost_total = 0.0
    judge_cost_total = 0.0
    cost_known = False
    total_elapsed = 0.0
    elapsed_n = 0

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
        by_category_methods.setdefault(cat, {})

        if normalized is not None:
            by_category[cat].append(normalized)
            by_difficulty[diff].append(normalized)
            all_scores.append(normalized)
            by_category_methods[cat][method] = by_category_methods[cat].get(method, 0) + 1

        # Tally why functional tasks failed, when the diagnostic judge ran. This
        # turns the per-task diagnoses into a category-agnostic failure profile
        # (e.g. "4 of 6 codegen failures were signature_mismatch" → benchmark bug,
        # not model weakness).
        diag = sr.get("diagnostic_judge_result")
        if diag and diag.get("failure_type"):
            ft = diag["failure_type"]
            functional_failures[ft] = functional_failures.get(ft, 0) + 1

        # ── cost (runner + judge) and latency ────────────────────────────────
        model = result.get("model")
        if model:
            runner_models.add(model)
        if result.get("provider"):
            runner_providers.add(result["provider"])
        runner_cost = result.get("cost_usd")
        judge_cost = sr.get("judge_cost_usd")
        if sr.get("judge_model"):
            judge_models.add(sr["judge_model"])
        if runner_cost is not None:
            runner_cost_total += runner_cost
            cost_known = True
        if judge_cost is not None:
            judge_cost_total += judge_cost
            cost_known = True
        task_cost = None
        if runner_cost is not None or judge_cost is not None:
            task_cost = round((runner_cost or 0.0) + (judge_cost or 0.0), 6)

        elapsed = result.get("elapsed_seconds")
        if elapsed is not None:
            total_elapsed += elapsed
            elapsed_n += 1

        task_scores.append({
            "task_id": result["task_id"],
            "category": cat,
            "difficulty": diff,
            "score": normalized,
            "method": method,
            "turns": result.get("turns", 0),
            "elapsed": elapsed,
            "cost": task_cost,
            "model": model,
            "tokens": result.get("usage"),
            "reasoning": sr.get("reasoning") if method == "llm_judge" else None,
            # The agent's final answer, surfaced so the dashboard can show what the
            # model actually produced for each task next to its score. This is the
            # final response only — full trajectories stay behind the keyed endpoint.
            "output": result.get("final_response"),
            "error": result.get("error"),
        })

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else None
    def pass_rate(lst, threshold=0.75): return round(sum(1 for x in lst if x >= threshold) / len(lst), 4) if lst else None

    def _category_methods(cat):
        """Per-category scoring-method breakdown + a flag for when one category
        mixes binary and partial-credit tasks (its mean then blends two scales)."""
        methods = by_category_methods.get(cat, {})
        total = sum(methods.values())
        breakdown = {
            m: {"count": c, "fraction": round(c / total, 3) if total else 0.0}
            for m, c in sorted(methods.items())
        }
        scales = sorted({_scale_of(m) for m in methods})
        return {
            "breakdown": breakdown,
            "scales": scales,
            "mixed_scoring_warning": "binary" in scales and "partial_credit" in scales,
        }

    # Classify each category by the granularity of its dominant scoring method, so
    # we can warn when the report compares binary-scored categories (means are pass
    # rates) against partial-credit ones (means are continuous).
    cat_scales = {
        cat: _scale_of(max(methods.items(), key=lambda kv: kv[1])[0])
        for cat, methods in by_category_methods.items() if methods
    }
    binary_cats = sorted(c for c, s in cat_scales.items() if s == "binary")
    partial_cats = sorted(c for c, s in cat_scales.items() if s == "partial_credit")
    cross_warn = bool(binary_cats) and bool(partial_cats)

    return {
        "model": ", ".join(sorted(runner_models)) if runner_models else "unknown",
        "provider": ", ".join(sorted(runner_providers)) if runner_providers else None,
        "judge_model": ", ".join(sorted(judge_models)) if judge_models else None,
        "benchmark_version": BENCHMARK_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "n_tasks": len(results),
        "overall": {
            "mean_score": avg(all_scores),
            "pass_rate_75": pass_rate(all_scores),
        },
        "scoring_methodology": {
            "cross_category_comparison_warning": cross_warn,
            "detail": (
                f"Binary-scored categories (each task is 0 or 1, so the mean is a "
                f"pass rate): {binary_cats}. Partial-credit categories (continuous "
                f"0-1 means): {partial_cats}. Comparing a binary category's mean "
                f"against a partial-credit one is not apples-to-apples — it "
                f"overstates the gap. Compare within a scale, or use pass_rate_75 "
                f"across all."
            ) if cross_warn else None,
            "binary_categories": binary_cats,
            "partial_credit_categories": partial_cats,
            "functional_failure_breakdown": functional_failures or None,
        },
        "cost_usd": {
            "total": round(runner_cost_total + judge_cost_total, 6) if cost_known else None,
            "runner": round(runner_cost_total, 6) if cost_known else None,
            "judge": round(judge_cost_total, 6) if cost_known else None,
        },
        "latency_seconds": {
            "total": round(total_elapsed, 2),
            "mean": round(total_elapsed / elapsed_n, 2) if elapsed_n else None,
        },
        "by_category": {k: {"mean": avg(v), "n": len(v), "pass_rate": pass_rate(v), **_category_methods(k)} for k, v in by_category.items()},
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

    cost = report.get("cost_usd") or {}
    lat = report.get("latency_seconds") or {}
    if cost.get("total") is not None:
        print(f"Model: {report.get('model')}  |  Cost: ${cost['total']:.4f}  |  Latency: {lat.get('total')}s")

    print(f"\nBy Category:")
    for cat, stats in sorted(report["by_category"].items()):
        bar = "█" * int((stats["mean"] or 0) * 20)
        mix = "  ⚠ mixed scales" if stats.get("mixed_scoring_warning") else ""
        print(f"  {cat:20s}  {bar:20s}  {stats['mean']:.3f}  (n={stats['n']}){mix}")

    methodology = report.get("scoring_methodology") or {}
    if methodology.get("cross_category_comparison_warning"):
        print(f"\n  ⚠ Scoring scales differ across categories — category means are not "
              f"directly comparable.\n    Binary (pass-rate) means: {methodology['binary_categories']}"
              f"\n    Partial-credit means:     {methodology['partial_credit_categories']}")

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
                        choices=["extraction", "code_generation", "computation", "workflow", "agentic", "agentic_real", "debug"])
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge step")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_full_eval(
        task_ids=args.task,
        categories=args.category,
        skip_judge=args.skip_judge,
        verbose=not args.quiet
    )
