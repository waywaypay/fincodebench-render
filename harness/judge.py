"""
FinCodeBench LLM-as-Judge
Uses a separate Claude call to score responses on complex/agentic tasks.

Key design choices:
- Separate model call from the task execution (avoids self-reinforcement)
- Structured 0–4 rubric with forced JSON output
- Calibration tracking: store human vs judge scores to measure agreement
"""

import json
import os
import re
from typing import Optional, Callable

import providers
from pricing import compute_cost

# Overridden per run by the web service with the caller's own key + provider
# (BYOK); falls back to a placeholder key so a missing env key can't break import.
_PROVIDER, _PROVIDER_CFG = providers.resolve_provider(os.environ.get("FINCODEBENCH_PROVIDER"))
client = providers.client_from_env(_PROVIDER)

# Kept distinct from the runner model to avoid self-grading; defaults to the
# provider's judge model. Override via FINCODEBENCH_JUDGE_MODEL.
JUDGE_MODEL = os.environ.get("FINCODEBENCH_JUDGE_MODEL") or _PROVIDER_CFG["default_judge_model"]

JUDGE_SYSTEM = """You are an expert judge evaluating AI assistant responses to financial analysis and coding tasks.

Your job is to score the response strictly against the provided rubric.

SCORING SCALE:
4 = Perfect — correct, complete, well-structured, no significant issues
3 = Good — mostly correct, minor issues or missing detail
2 = Partial — correct direction but missing key elements or has errors
1 = Poor — attempts the task but is substantially wrong
0 = Fail — wrong output, crashed code, refused task, or irrelevant

RULES:
- Base your score ONLY on the rubric provided, not your own opinion
- If the response contains code that crashes or produces wrong output, cap at 1
- If the response refuses or says it cannot do the task, score 0
- Be strict: a "pretty good" response with one clear factual error is a 2 or 3, not 4

Respond ONLY with a valid JSON object — no preamble, no markdown fences:
{"score": <0|1|2|3|4>, "reasoning": "<one concise sentence>", "key_issues": ["<issue1>", "<issue2>"]}"""


# ── Building what the judge sees ───────────────────────────────────────────────
# The runner records the model's full trajectory (its text, the code it ran via
# the execute_python tool, and the output that came back) but stores only the
# last text turn in `final_response`. Agentic rubrics grade the *code and its
# output* — which live in the trajectory — so judging `final_response` alone
# leaves the judge seeing "no code" and scoring 0. We rebuild a labeled
# transcript instead.
MAX_JUDGE_RESPONSE_CHARS = 8000   # cap the transcript handed to the judge
MAX_TOOL_OUTPUT_CHARS = 2000      # cap any single tool output within it


def _truncate_middle(text: str, limit: int) -> str:
    """Trim to `limit` chars keeping the head and tail, so a long transcript
    still shows both the code at the top and the conclusion at the bottom."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + "\n…[transcript truncated]…\n" + text[-tail:]


def build_judge_response(result: dict) -> str:
    """Render what the judge should evaluate.

    For tool-using (agentic) tasks the real work — the code and its execution
    output — lives in `result["trajectory"]`, not in `final_response`. Walk the
    trajectory in order and emit a labeled transcript: the assistant's text, the
    code it executed, and the output each execution produced. When a task used no
    tools (e.g. a pure extraction answer), the answer is fully contained in
    `final_response`, so return that unchanged.
    """
    trajectory = result.get("trajectory") or []
    final_response = (result.get("final_response") or "").strip()

    has_tool_use = any(
        b.get("type") == "tool_use"
        for entry in trajectory
        for b in entry.get("blocks", [])
    )
    if not has_tool_use:
        return final_response

    parts: list[str] = []
    for entry in trajectory:
        for b in entry.get("blocks", []):
            btype = b.get("type")
            if btype == "text":
                txt = (b.get("text") or "").strip()
                if txt:
                    parts.append(txt)
            elif btype == "tool_use":
                inp = b.get("input") or {}
                code = inp.get("code") if isinstance(inp, dict) else None
                if code:
                    parts.append(f"[executed code]\n```python\n{code.strip()}\n```")
                else:
                    parts.append(f"[tool call: {b.get('name', 'tool')}] "
                                 f"{json.dumps(inp)[:MAX_TOOL_OUTPUT_CHARS]}")
            elif btype == "tool_result":
                out = str(b.get("result", "")).strip()
                if len(out) > MAX_TOOL_OUTPUT_CHARS:
                    out = out[:MAX_TOOL_OUTPUT_CHARS] + "\n…[output truncated]…"
                name = b.get("name") or ""
                label = "execution output" if name in ("", "execute_python") else f"{name} result"
                parts.append(f"[{label}]\n{out}")

    body = "\n\n".join(parts)
    # The final answer is the last end_turn text, already included above; add it
    # back explicitly only if it somehow isn't present (e.g. an empty last turn).
    if final_response and final_response not in body:
        parts.append(f"[final answer]\n{final_response}")
        body = "\n\n".join(parts)
    return body.strip() or final_response


def _extract_judge_json(text: str):
    """Parse the judge's JSON verdict, tolerant of code fences and any
    preamble/postamble prose around it. Returns a dict, or None if nothing
    parseable is found."""
    if not text:
        return None
    candidates = [text]
    stripped = re.sub(r'```(?:json)?|```', '', text).strip()
    if stripped and stripped != text:
        candidates.append(stripped)
    m = re.search(r'\{[\s\S]*\}', text)   # first '{' … last '}'
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def llm_judge(
    task_prompt: str,
    context: Optional[str],
    response: str,
    rubric: str
) -> dict:
    """
    Score a response using LLM-as-judge.
    Returns dict with score (0–4), normalized (0–1), reasoning, key_issues.
    """
    # Build judge prompt
    parts = [f"**Task given to the AI:**\n{task_prompt}"]
    if context:
        parts.append(f"**Context/Data provided:**\n```\n{context[:2000]}\n```")
    parts.append(f"**Scoring rubric:**\n{rubric}")
    parts.append(
        "**AI response / execution transcript to evaluate:**\n"
        "(For tool-using tasks this includes the code the assistant executed and the "
        "output it produced — judge the code and its results, not just the prose.)\n"
        f"{_truncate_middle(response, MAX_JUDGE_RESPONSE_CHARS)}"
    )

    judge_prompt = "\n\n".join(parts)

    result = client.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    # Judge calls cost money too — track usage/cost so the run total includes them
    u = getattr(result, "usage", None)
    judge_usage = None
    judge_cost = None
    if u is not None:
        judge_usage = {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        }
        judge_cost = compute_cost(
            JUDGE_MODEL,
            judge_usage["input_tokens"], judge_usage["output_tokens"],
            judge_usage["cache_creation_input_tokens"], judge_usage["cache_read_input_tokens"],
        )
    judge_meta = {"judge_model": JUDGE_MODEL, "judge_usage": judge_usage, "judge_cost_usd": judge_cost}

    raw_text = result.text.strip()

    # Parse the judge's JSON verdict (tolerant of fences / surrounding prose).
    judgment = _extract_judge_json(raw_text)
    if judgment is None:
        return {
            "score": 0,
            "normalized": 0.0,
            "method": "llm_judge",
            "reasoning": "Judge parse error: no JSON object found in judge output",
            "key_issues": ["parse_error"],
            "raw": raw_text,
            **judge_meta,
        }

    try:
        score = int(round(float(judgment.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(4, score))   # clamp to 0–4
    judgment["score"] = score
    judgment["normalized"] = round(score / 4.0, 4)
    judgment["method"] = "llm_judge"
    judgment.update(judge_meta)
    return judgment


# ── Diagnostic judge (failure classification, NOT scoring) ─────────────────────
# Functional tasks are graded by execution (0 or 1). When one fails we still want
# to know *why*, so we can tell a benchmark-fault failure (the test demanded an
# exact signature the prompt never stated) from a genuine capability failure (the
# model used the wrong formula). This classifier runs ONLY on failures, never
# produces a score, and never changes one — its output is stored alongside the
# score for later analysis and aggregated into the report's failure breakdown.
DIAGNOSTIC_SYSTEM = """You classify why a Python solution failed a financial-coding unit test.

Given the task prompt, the code the model submitted, the hidden test harness, and the
error output, choose EXACTLY ONE category that best explains the failure:

- wrong_formula: the code runs but computes the wrong financial logic or result
- signature_mismatch: the logic looks right but the function name, arguments, or return type/shape don't match what the test calls
- edge_case: core logic is right but an edge case (division by zero, empty input, None handling) is mishandled
- extraction_failure: there is no usable Python code to run (the answer was prose, or the code block was malformed)
- runtime_error: the code crashes for a reason unrelated to financial correctness (syntax error, bad import, name error)

Respond with ONLY a JSON object, no preamble or fences:
{"failure_type": "<one category>", "explanation": "<one short sentence>"}"""

_DIAGNOSTIC_FAILURE_TYPES = {
    "wrong_formula", "signature_mismatch", "edge_case",
    "extraction_failure", "runtime_error",
}


def diagnostic_failure_judge(
    task_prompt: str,
    model_code: str,
    test_code: str,
    error_output: str,
) -> dict:
    """Categorize WHY a functional test failed. Diagnostic only — returns a
    {failure_type, explanation, method} dict, never a score. Safe by construction:
    on any error or unparseable verdict it falls back to 'runtime_error' rather
    than raising, so it can never break the scoring path that calls it."""
    parts = [
        f"**Task prompt:**\n{(task_prompt or '')[:1200]}",
        f"**Submitted code:**\n```python\n{(model_code or '')[:3000]}\n```",
        f"**Hidden test harness:**\n```python\n{(test_code or '')[:1200]}\n```",
        f"**Error / output:**\n{(error_output or '')[:600]}",
    ]
    try:
        result = client.create(
            model=JUDGE_MODEL,
            max_tokens=150,
            system=DIAGNOSTIC_SYSTEM,
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
        )
        parsed = _extract_judge_json((result.text or "").strip())
    except Exception as e:
        return {"failure_type": "runtime_error",
                "explanation": f"diagnostic call failed: {str(e)[:120]}",
                "method": "diagnostic_judge"}

    if not parsed or "failure_type" not in parsed:
        return {"failure_type": "runtime_error",
                "explanation": "diagnostic parse error",
                "method": "diagnostic_judge"}
    ft = parsed.get("failure_type")
    if ft not in _DIAGNOSTIC_FAILURE_TYPES:
        ft = "runtime_error"
    return {"failure_type": ft,
            "explanation": (parsed.get("explanation") or "")[:300],
            "method": "diagnostic_judge"}


def score_pending_judge_tasks(
    results: list[dict],
    tasks: dict,
    calibration_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> list[dict]:
    """
    For all results where scoring_type == 'llm_judge', run the judge and attach scores.
    Optionally saves to calibration_path for human vs judge tracking.
    """
    calibration = []

    total_judge = sum(
        1 for r in results
        if (tasks.get(r["task_id"]) or {}).get("scoring_type") == "llm_judge"
    )
    done = 0
    for result in results:
        task_id = result["task_id"]
        task = tasks.get(task_id)
        if not task or task.get("scoring_type") != "llm_judge":
            continue

        if progress_callback:
            try:
                progress_callback(done, total_judge, task_id)
            except Exception:
                pass
        print(f"  Judging: {task_id}")
        judgment = llm_judge(
            task_prompt=task["prompt"],
            context=task.get("context"),
            response=build_judge_response(result),
            rubric=task["rubric"]
        )

        result["score_result"] = judgment
        done += 1

        calibration.append({
            "task_id": task_id,
            "judge_score": judgment["score"],
            "judge_reasoning": judgment.get("reasoning"),
            "human_score": None   # fill in manually for calibration
        })

    if calibration_path and calibration:
        with open(calibration_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        print(f"  Calibration template saved → {calibration_path}")

    return results


# ── Calibration analysis ──────────────────────────────────────────────────────
def analyze_calibration(calibration_path: str) -> dict:
    """
    After you've filled in human_score values, run this to compute judge accuracy.
    Reports: mean absolute error, correlation, % within 1 point.
    """
    with open(calibration_path) as f:
        data = json.load(f)

    scored = [(d["human_score"], d["judge_score"]) for d in data if d["human_score"] is not None]
    if not scored:
        return {"error": "no human scores found — fill in human_score fields first"}

    n = len(scored)
    mae = sum(abs(h - j) for h, j in scored) / n
    within_1 = sum(1 for h, j in scored if abs(h - j) <= 1) / n

    # Pearson correlation
    h_vals = [h for h, _ in scored]
    j_vals = [j for _, j in scored]
    h_mean = sum(h_vals) / n
    j_mean = sum(j_vals) / n
    num = sum((h - h_mean) * (j - j_mean) for h, j in zip(h_vals, j_vals))
    denom = (sum((h - h_mean)**2 for h in h_vals) * sum((j - j_mean)**2 for j in j_vals)) ** 0.5
    corr = num / denom if denom > 0 else 0.0

    return {
        "n": n,
        "mae": round(mae, 3),
        "within_1_point": round(within_1, 3),
        "pearson_r": round(corr, 3),
        "interpretation": (
            "strong alignment" if corr > 0.8 else
            "moderate alignment" if corr > 0.5 else
            "weak alignment — review rubrics"
        )
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "calibrate":
        print(json.dumps(analyze_calibration(sys.argv[2]), indent=2))
        sys.exit(0)

    # Score a single result: python judge.py results/raw/agentic-001.json
    if len(sys.argv) < 2:
        print("Usage: python judge.py <raw_result.json>")
        print("       python judge.py calibrate <calibration.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        result = json.load(f)
    with open("tasks/tasks.json") as f:
        tasks = {t["id"]: t for t in json.load(f)}

    task = tasks.get(result["task_id"])
    if not task or task.get("scoring_type") != "llm_judge":
        print("Task not found or not an llm_judge task")
        sys.exit(1)

    judgment = llm_judge(
        task_prompt=task["prompt"],
        context=task.get("context"),
        response=build_judge_response(result),
        rubric=task["rubric"]
    )
    print(json.dumps(judgment, indent=2))
