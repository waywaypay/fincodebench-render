"""
FinCodeBench LLM-as-Judge
Uses a separate Claude call to score responses on complex/agentic tasks.

Key design choices:
- Separate model call from the task execution (avoids self-reinforcement)
- Structured 0–4 rubric with forced JSON output
- Calibration tracking: store human vs judge scores to measure agreement
"""

import json
import re
from typing import Optional
import anthropic

client = anthropic.Anthropic()

JUDGE_MODEL = "claude-sonnet-4-5"      # Intentionally different from runner model

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
    parts.append(f"**AI response to evaluate:**\n{response[:4000]}")

    judge_prompt = "\n\n".join(parts)

    result = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    raw_text = result.content[0].text.strip()

    # Parse JSON response
    try:
        # Strip any accidental markdown fences
        clean = re.sub(r'```(?:json)?|```', '', raw_text).strip()
        judgment = json.loads(clean)
        score = int(judgment.get("score", 0))
        score = max(0, min(4, score))   # clamp to 0–4
        judgment["score"] = score
        judgment["normalized"] = round(score / 4.0, 4)
        judgment["method"] = "llm_judge"
        return judgment
    except Exception as e:
        return {
            "score": 0,
            "normalized": 0.0,
            "method": "llm_judge",
            "reasoning": f"Judge parse error: {str(e)}",
            "key_issues": ["parse_error"],
            "raw": raw_text
        }


def score_pending_judge_tasks(
    results: list[dict],
    tasks: dict,
    calibration_path: Optional[str] = None
) -> list[dict]:
    """
    For all results where scoring_type == 'llm_judge', run the judge and attach scores.
    Optionally saves to calibration_path for human vs judge tracking.
    """
    calibration = []

    for result in results:
        task_id = result["task_id"]
        task = tasks.get(task_id)
        if not task or task.get("scoring_type") != "llm_judge":
            continue

        print(f"  Judging: {task_id}")
        judgment = llm_judge(
            task_prompt=task["prompt"],
            context=task.get("context"),
            response=result.get("final_response", ""),
            rubric=task["rubric"]
        )

        result["score_result"] = judgment

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
        response=result.get("final_response", ""),
        rubric=task["rubric"]
    )
    print(json.dumps(judgment, indent=2))
