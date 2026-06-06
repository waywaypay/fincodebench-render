"""Offline tests for the LLM-as-judge.

The judge must see the executed code and its output — not just the final text
turn — for tool-using (agentic) tasks, must parse verdicts robustly, and must
normalize the 0–4 rubric to 0–1. The judge model client is faked, so these run
with no network.
"""
from types import SimpleNamespace

import judge


def _agentic_result():
    """A trajectory shaped like the runner records for an agentic task: turn 1
    writes and runs code via execute_python; turn 2 is a short end_turn summary.
    The substance (code + output) lives in the trajectory, not final_response."""
    return {
        "task_id": "agentic-001",
        "final_response": "Beta Inc had the highest average YoY growth.",
        "trajectory": [
            {"turn": 1, "stop_reason": "tool_use", "blocks": [
                {"type": "text", "text": "Let me compute YoY growth."},
                {"type": "tool_use", "name": "execute_python", "id": "t1",
                 "input": {"code": "print('Beta Inc', 0.31)"}},
                {"type": "tool_result", "name": "execute_python", "result": "Beta Inc 0.31"},
            ]},
            {"turn": 2, "stop_reason": "end_turn", "blocks": [
                {"type": "text", "text": "Beta Inc had the highest average YoY growth."},
            ]},
        ],
    }


def test_build_judge_response_includes_code_and_output():
    transcript = judge.build_judge_response(_agentic_result())
    assert "print('Beta Inc', 0.31)" in transcript      # the code it executed
    assert "Beta Inc 0.31" in transcript                 # the execution output
    assert "Beta Inc had the highest" in transcript      # the final answer


def test_build_judge_response_falls_back_when_no_tools():
    # Pure extraction-style result: the answer is entirely in final_response and
    # there are no tool calls, so the transcript is just that response.
    result = {
        "final_response": '{"ticker": "AAPL"}',
        "trajectory": [
            {"turn": 1, "stop_reason": "end_turn",
             "blocks": [{"type": "text", "text": '{"ticker": "AAPL"}'}]},
        ],
    }
    assert judge.build_judge_response(result) == '{"ticker": "AAPL"}'


def test_build_judge_response_caps_huge_tool_output():
    big = "x" * (judge.MAX_TOOL_OUTPUT_CHARS + 5000)
    result = {
        "final_response": "done",
        "trajectory": [{"turn": 1, "stop_reason": "tool_use", "blocks": [
            {"type": "tool_use", "name": "execute_python", "input": {"code": "print(1)"}},
            {"type": "tool_result", "name": "execute_python", "result": big},
        ]}],
    }
    transcript = judge.build_judge_response(result)
    assert "output truncated" in transcript
    assert len(transcript) < len(big)


def _fake_judge_client(capture, raw_text):
    def create(**kwargs):
        capture.update(kwargs)
        return SimpleNamespace(text=raw_text, usage=None)
    return SimpleNamespace(create=create)


def test_judge_sees_transcript_and_scores(monkeypatch):
    """End-to-end against an agentic trajectory: the prompt the judge receives
    must contain the executed code + output, and a 4 must normalize to 1.0."""
    capture = {}
    monkeypatch.setattr(
        judge, "client",
        _fake_judge_client(capture, '{"score": 4, "reasoning": "ok", "key_issues": []}'),
    )
    out = judge.llm_judge(
        task_prompt="compute YoY growth and name the winner",
        context="company,quarter,revenue\n...",
        response=judge.build_judge_response(_agentic_result()),
        rubric="Score 4 if code runs and names Beta Inc. Score 0 if no code.",
    )
    judge_prompt = capture["messages"][0]["content"]
    assert "print('Beta Inc', 0.31)" in judge_prompt   # judge actually saw the code
    assert "Beta Inc 0.31" in judge_prompt              # ...and the execution output
    assert out["score"] == 4 and out["normalized"] == 1.0


def test_judge_json_parsing_is_robust(monkeypatch):
    # The judge wraps its JSON in prose and a fence — it must still parse, not
    # silently default to 0 the way the old fence-only strip did.
    raw = ("Here is my assessment:\n```json\n"
           '{"score": 3, "reasoning": "minor issue", "key_issues": ["x"]}\n```\nDone.')
    monkeypatch.setattr(judge, "client", _fake_judge_client({}, raw))
    out = judge.llm_judge("p", None, "resp", "rubric")
    assert out["score"] == 3 and out["normalized"] == 0.75
    assert out["key_issues"] == ["x"]


def test_judge_unparseable_defaults_to_zero(monkeypatch):
    monkeypatch.setattr(judge, "client", _fake_judge_client({}, "totally not json"))
    out = judge.llm_judge("p", None, "resp", "rubric")
    assert out["score"] == 0 and out["normalized"] == 0.0
    assert "parse_error" in out.get("key_issues", [])


# ── Diagnostic failure judge: classifies WHY, never scores ─────────────────────
def test_diagnostic_judge_classifies_and_never_scores(monkeypatch):
    capture = {}
    monkeypatch.setattr(judge, "client", _fake_judge_client(
        capture, '{"failure_type": "signature_mismatch", "explanation": "wrong arg name"}'))
    out = judge.diagnostic_failure_judge(
        task_prompt="write f(a, b)", model_code="def f(a): ...",
        test_code="f(1, b=2)", error_output="TypeError: unexpected keyword 'b'")
    assert out["failure_type"] == "signature_mismatch"
    assert out["method"] == "diagnostic_judge"
    assert "score" not in out and "normalized" not in out   # diagnosis is not a score
    # The judge must actually see the code and the error it's classifying.
    judge_prompt = capture["messages"][0]["content"]
    assert "def f(a)" in judge_prompt and "TypeError" in judge_prompt


def test_diagnostic_judge_rejects_unknown_category(monkeypatch):
    monkeypatch.setattr(judge, "client", _fake_judge_client({}, '{"failure_type": "banana"}'))
    out = judge.diagnostic_failure_judge("p", "c", "t", "e")
    assert out["failure_type"] == "runtime_error"   # out-of-vocab falls back, never raises


def test_diagnostic_judge_unparseable_falls_back(monkeypatch):
    monkeypatch.setattr(judge, "client", _fake_judge_client({}, "not json at all"))
    out = judge.diagnostic_failure_judge("p", "c", "t", "e")
    assert out["failure_type"] == "runtime_error"
    assert out["method"] == "diagnostic_judge"


def test_diagnostic_judge_survives_client_exception(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("network down")
    monkeypatch.setattr(judge, "client", SimpleNamespace(create=boom))
    out = judge.diagnostic_failure_judge("p", "c", "t", "e")
    assert out["failure_type"] == "runtime_error"   # swallowed, not propagated


# ── Scorer ↔ diagnostic integration: failure triggers diagnosis, pass doesn't ──
def test_scorer_attaches_diagnosis_only_on_functional_failure(monkeypatch):
    import scorer
    monkeypatch.setattr(
        judge, "diagnostic_failure_judge",
        lambda **kw: {"failure_type": "wrong_formula", "explanation": "x", "method": "diagnostic_judge"},
    )
    failing = {"scoring_type": "functional", "prompt": "p", "test_code": "assert False\nprint('PASS')"}
    res = {"final_response": "```python\nx = 1\n```"}

    # Enabled + failing → diagnosis attached, score untouched (still 0).
    sr = scorer.score_task(failing, res, run_diagnostic=True)
    assert sr["score"] == 0.0
    assert sr["diagnostic_judge_result"]["failure_type"] == "wrong_formula"

    # Disabled → never calls the judge, no diagnosis field.
    sr_off = scorer.score_task(failing, res, run_diagnostic=False)
    assert "diagnostic_judge_result" not in sr_off

    # Enabled but PASSING → no diagnosis (only failures are diagnosed).
    passing = {"scoring_type": "functional", "prompt": "p", "test_code": "assert True\nprint('PASS')"}
    sr_pass = scorer.score_task(passing, {"final_response": "```python\nx = 1\n```"}, run_diagnostic=True)
    assert sr_pass["score"] == 1.0 and "diagnostic_judge_result" not in sr_pass
