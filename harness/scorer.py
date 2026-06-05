"""
FinCodeBench Scorer
Deterministic scoring for exact, fuzzy-numerical, and functional test tasks.
LLM-as-judge tasks are handled separately in judge.py.
"""

import json
import os
import re
import subprocess
import tempfile
from typing import Any, Optional


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> Optional[Any]:
    """Try to parse the first JSON object or array found in text."""
    # Try code blocks first
    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

    # Try bare JSON object / array
    for pattern in [r'(\{[\s\S]*\})', r'(\[[\s\S]*\])']:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None


def _extract_numbers(text: str) -> list[float]:
    """Return all floats/ints found in text (handles commas in numbers)."""
    cleaned = text.replace(',', '')
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', cleaned)]


def _extract_code_blocks(text: str) -> list[str]:
    """Return all Python code blocks found in text."""
    blocks = re.findall(r'```python\s*([\s\S]*?)\s*```', text)
    if not blocks:
        blocks = re.findall(r'```\s*([\s\S]*?)\s*```', text)
    return blocks


def _float_close(a: float, b: float, tolerance: float) -> bool:
    """True if a and b are within tolerance of each other (relative if b≠0)."""
    if b == 0:
        return abs(a) <= tolerance
    return abs(a - b) / abs(b) <= tolerance


# ── Scoring functions ─────────────────────────────────────────────────────────
def score_exact_json(response: str, expected: Any) -> dict:
    """
    Parse JSON from response and compare to expected via equality.
    Returns score 0 or 1.
    """
    parsed = _extract_json(response)
    if parsed is None:
        return {"score": 0.0, "method": "exact_json", "error": "no JSON found", "parsed": None}

    match = parsed == expected
    return {
        "score": 1.0 if match else 0.0,
        "method": "exact_json",
        "parsed": parsed,
        "expected": expected,
        "match": match
    }


def score_fuzzy_number(response: str, expected: float, tolerance: float = 0.02) -> dict:
    """
    Look for a number in the response within tolerance of expected.
    Returns score 0 or 1.
    """
    numbers = _extract_numbers(response)
    for num in numbers:
        if _float_close(num, expected, tolerance):
            return {"score": 1.0, "method": "fuzzy_number", "found": num, "expected": expected}
    return {
        "score": 0.0,
        "method": "fuzzy_number",
        "found": numbers[:5] if numbers else [],
        "expected": expected
    }


def score_fuzzy_dict(response: str, expected: dict, tolerance: float = 0.005) -> dict:
    """
    Parse JSON from response, compare numeric values with tolerance.
    Handles None values (matched only against None).
    Returns score 0–1 as fraction of keys matched.
    """
    parsed = _extract_json(response)
    if parsed is None or not isinstance(parsed, dict):
        return {"score": 0.0, "method": "fuzzy_dict", "error": "no valid dict found"}

    matched = 0
    details = {}
    for key, exp_val in expected.items():
        got_val = parsed.get(key)
        if exp_val is None:
            ok = got_val is None
        elif isinstance(exp_val, (int, float)) and isinstance(got_val, (int, float)):
            ok = _float_close(float(got_val), float(exp_val), tolerance)
        else:
            ok = got_val == exp_val
        details[key] = {"expected": exp_val, "got": got_val, "match": ok}
        if ok:
            matched += 1

    score = matched / len(expected) if expected else 0.0
    return {"score": score, "method": "fuzzy_dict", "details": details, "matched": matched, "total": len(expected)}


def score_functional(response: str, test_code: str) -> dict:
    """
    Extract code from response, append test_code, run in subprocess.
    Returns score 1.0 if test prints 'PASS' and exits 0, else 0.0.
    """
    code_blocks = _extract_code_blocks(response)

    # Also try raw response if no code blocks found
    if not code_blocks:
        code_blocks = [response]

    last_error = "no code found"
    for code in code_blocks:
        full = code + "\n\n" + test_code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full)
            tmpfile = f.name

        try:
            result = subprocess.run(
                ['python3', tmpfile],
                capture_output=True, text=True, timeout=30
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            passed = result.returncode == 0 and 'PASS' in stdout

            if passed:
                return {"score": 1.0, "method": "functional", "stdout": stdout}
            last_error = stderr or stdout or "non-zero exit"
        except subprocess.TimeoutExpired:
            last_error = "timeout"
        except Exception as e:
            last_error = str(e)
        finally:
            try:
                os.unlink(tmpfile)
            except Exception:
                pass

    return {"score": 0.0, "method": "functional", "error": last_error}


# ── Dispatcher ────────────────────────────────────────────────────────────────
def score_task(task: dict, result: dict) -> dict:
    """
    Route to the appropriate scorer based on task['scoring_type'].
    LLM-judge tasks return {"score": None, "method": "llm_judge_pending"}.
    """
    scoring_type = task.get("scoring_type", "llm_judge")
    response = result.get("final_response", "")

    # If task errored out, score 0
    if result.get("error"):
        return {"score": 0.0, "method": scoring_type, "error": result["error"]}

    if scoring_type == "exact_json":
        return score_exact_json(response, task["expected_output"])

    elif scoring_type == "fuzzy_number":
        return score_fuzzy_number(
            response,
            task["expected_output"],
            task.get("tolerance", 0.02)
        )

    elif scoring_type == "fuzzy_dict":
        return score_fuzzy_dict(
            response,
            task["expected_output"],
            task.get("tolerance", 0.005)
        )

    elif scoring_type == "functional":
        return score_functional(response, task["test_code"])

    elif scoring_type == "llm_judge":
        # Deferred — call judge.py separately
        return {"score": None, "method": "llm_judge_pending"}

    else:
        return {"score": None, "method": f"unknown:{scoring_type}"}


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json

    # Score a single raw result file: python scorer.py results/raw/codegen-001.json
    if len(sys.argv) < 2:
        print("Usage: python scorer.py <raw_result.json>")
        sys.exit(1)

    result_path = sys.argv[1]
    tasks_path = "tasks/tasks.json"

    with open(result_path) as f:
        result = json.load(f)
    with open(tasks_path) as f:
        tasks = {t["id"]: t for t in json.load(f)}

    task = tasks.get(result["task_id"])
    if not task:
        print(f"Task {result['task_id']} not found in tasks.json")
        sys.exit(1)

    score_result = score_task(task, result)
    print(json.dumps(score_result, indent=2))
