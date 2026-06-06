"""Offline tests for the per-task tool set the runner builds for agentic tasks.

Files land in a private working directory exposed via list_files / read_file,
execute_python runs there (so relative opens resolve), and fetch_filing serves
data that never touches disk — forcing the model to actually choose that tool.
build_task_tools and its executors are pure, so these run with no network.
"""
import json

import runner


def _task_with_tools():
    return {
        "id": "agentic-test",
        "category": "agentic",
        "difficulty": "medium",
        "scoring_type": "llm_judge",
        "tools_data": {
            "files": {"data.csv": "a,b\n1,2\n3,4\n"},
            "filings": {"Acme Inc": {"2024": {"revenue_usd_m": 100}}},
        },
    }


def test_tools_data_task_exposes_all_tools():
    tools, executors, cleanup = runner.build_task_tools(_task_with_tools())
    try:
        names = {t["name"] for t in tools}
        assert names == {"execute_python", "list_files", "read_file", "fetch_filing"}
        assert set(executors) == names
    finally:
        cleanup()


def test_list_and_read_file():
    _, ex, cleanup = runner.build_task_tools(_task_with_tools())
    try:
        assert "data.csv" in ex["list_files"]({})
        assert "a,b" in ex["read_file"]({"path": "data.csv"})
        # Missing file and a path-escape attempt both fail gracefully.
        assert "No such file" in ex["read_file"]({"path": "nope.csv"})
        assert "No such file" in ex["read_file"]({"path": "../secret"})
    finally:
        cleanup()


def test_execute_python_runs_in_task_workdir():
    _, ex, cleanup = runner.build_task_tools(_task_with_tools())
    try:
        out = ex["execute_python"]({"code": "print(open('data.csv').read().strip())"})
        assert "1,2" in out and "3,4" in out
    finally:
        cleanup()


def test_fetch_filing_hits_and_misses():
    _, ex, cleanup = runner.build_task_tools(_task_with_tools())
    try:
        hit = ex["fetch_filing"]({"company": "Acme Inc", "year": "2024"})
        assert json.loads(hit)["revenue_usd_m"] == 100
        assert "No filing found" in ex["fetch_filing"]({"company": "Acme Inc", "year": "1999"})
        assert "No filing found" in ex["fetch_filing"]({"company": "Unknown", "year": "2024"})
    finally:
        cleanup()


def test_cleanup_removes_workdir():
    _, ex, cleanup = runner.build_task_tools(_task_with_tools())
    assert "data.csv" in ex["list_files"]({})
    cleanup()
    # The directory is gone, so list_files now reports an error, not the listing.
    assert "data.csv" not in ex["list_files"]({})


def test_task_without_tools_data_gets_only_execute_python():
    tools, ex, cleanup = runner.build_task_tools({"id": "codegen-x"})
    try:
        assert [t["name"] for t in tools] == ["execute_python"]
        assert "hi" in ex["execute_python"]({"code": "print('hi')"})
    finally:
        cleanup()


# ── agentic_real snapshot tools ───────────────────────────────────────────────

def _agentic_real_task():
    return {
        "id": "agentic_real-test",
        "category": "agentic_real",
        "difficulty": "medium",
        "scoring_type": "llm_judge",
        "tools_data": {
            "snapshots": {
                "web_search": {
                    "Acme Corp FY2024 revenue": "Acme Corp reported FY2024 revenue of $1.2B.",
                },
                "fetch_sec_filing": {
                    "Acme Corp:10-K:2024": "ACME CORP 10-K FY2024\nRevenue: $1,200M\nOperating income: $240M",
                },
            }
        },
    }


def test_agentic_real_task_exposes_correct_tools():
    tools, executors, cleanup = runner.build_task_tools(_agentic_real_task())
    try:
        names = {t["name"] for t in tools}
        assert names == {"execute_python", "web_search", "fetch_sec_filing"}
        assert set(executors) == names
    finally:
        cleanup()


def test_web_search_snapshot_hit():
    _, ex, cleanup = runner.build_task_tools(_agentic_real_task())
    try:
        result, source = ex["web_search"]({"query": "Acme Corp FY2024 revenue"})
        assert "1.2B" in result
        assert source == "snapshot"
    finally:
        cleanup()


def test_web_search_snapshot_miss_returns_tuple():
    _, ex, cleanup = runner.build_task_tools(_agentic_real_task())
    try:
        # A query not in the snapshot — should return a tuple (str, source_str)
        raw = ex["web_search"]({"query": "unknown query not in snapshots"})
        assert isinstance(raw, tuple) and len(raw) == 2
        assert isinstance(raw[0], str)
        assert raw[1] in ("live", "error")
    finally:
        cleanup()


def test_fetch_sec_filing_snapshot_hit():
    _, ex, cleanup = runner.build_task_tools(_agentic_real_task())
    try:
        result, source = ex["fetch_sec_filing"]({"company": "Acme Corp", "form_type": "10-K", "year": "2024"})
        assert "Revenue: $1,200M" in result
        assert source == "snapshot"
    finally:
        cleanup()


def test_fetch_sec_filing_snapshot_miss_returns_tuple():
    _, ex, cleanup = runner.build_task_tools(_agentic_real_task())
    try:
        raw = ex["fetch_sec_filing"]({"company": "Unknown Corp", "form_type": "10-K", "year": "2024"})
        assert isinstance(raw, tuple) and len(raw) == 2
        assert isinstance(raw[0], str)
    finally:
        cleanup()


def test_real_tools_executor_returns_are_tuples():
    _, ex, cleanup = runner.build_task_tools(_agentic_real_task())
    try:
        ws_hit = ex["web_search"]({"query": "Acme Corp FY2024 revenue"})
        sec_hit = ex["fetch_sec_filing"]({"company": "Acme Corp", "form_type": "10-K", "year": "2024"})
        assert isinstance(ws_hit, tuple)
        assert isinstance(sec_hit, tuple)
    finally:
        cleanup()
