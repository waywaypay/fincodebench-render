import json
import time

import runner


def _task(task_id):
    return {
        "id": task_id,
        "category": "computation",
        "difficulty": "easy",
        "scoring_type": "exact_json",
        "prompt": "p",
    }


def test_run_benchmark_concurrency_preserves_task_order_and_saves_raw(tmp_path, monkeypatch):
    tasks = [_task("slow"), _task("fast"), _task("medium")]
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps(tasks))
    results_dir = tmp_path / "results"

    monkeypatch.setattr(runner, "TASKS_FILE", tasks_file)
    monkeypatch.setattr(runner, "RESULTS_DIR", results_dir)

    delays = {"slow": 0.05, "medium": 0.02, "fast": 0.0}

    def fake_run_task(task, verbose=True):
        time.sleep(delays[task["id"]])
        return {
            "task_id": task["id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "scoring_type": task["scoring_type"],
            "error": None,
        }

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    progress = []
    results = runner.run_benchmark(
        verbose=True,
        concurrency=3,
        progress_callback=lambda done, total, current: progress.append((done, total, current)),
    )

    assert [r["task_id"] for r in results] == ["slow", "fast", "medium"]
    assert (results_dir / "raw" / "slow.json").exists()
    assert (results_dir / "raw" / "fast.json").exists()
    assert (results_dir / "raw" / "medium.json").exists()
    assert [p[0] for p in progress] == [1, 2, 3, 3]
    assert progress[-1] == (3, 3, None)


def test_coerce_concurrency_rejects_non_positive_values():
    assert runner._coerce_concurrency(0) == 1
    assert runner._coerce_concurrency(-2) == 1
    assert runner._coerce_concurrency("bad") == 1
