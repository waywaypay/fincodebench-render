"""End-to-end pipeline tests through the FastAPI app with a faked provider
client (no network). The Starlette TestClient runs background tasks inline, so a
run is already complete by the time POST /runs returns."""
import pytest
from fastapi.testclient import TestClient

import providers
import api


class FakeClient:
    """Stand-in for providers.ChatClient: emits one tool call then a final
    answer for task turns, and valid rubric JSON for judge calls."""
    def __init__(self, provider, api_key, base_url=None):
        self.provider, _ = providers.resolve_provider(provider)
        self.calls = 0

    def create(self, model, max_tokens, messages, tools=None, system=None):
        if system and not tools:  # judge call
            return providers.ChatResponse('{"score": 3, "reasoning": "ok", "key_issues": []}', [], "end_turn")
        self.calls += 1
        if self.calls == 1 and tools:
            return providers.ChatResponse(
                "Let me compute.",
                [{"id": "t1", "name": "execute_python", "input": {"code": "print('hi')"}}],
                "tool_use",
            )
        return providers.ChatResponse("```python\ndef f():\n    return 1\n```", [], "end_turn")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(providers, "ChatClient", FakeClient)
    return TestClient(api.app)


def test_providers_endpoint(client):
    r = client.get("/providers")
    assert r.status_code == 200
    names = [p["name"] for p in r.json()["providers"]]
    assert names[0] == "anthropic"
    assert {"openai", "openrouter", "deepseek", "qwen", "kimi", "venice"} <= set(names)


def test_dashboard_data_includes_providers(client):
    d = client.get("/dashboard/data").json()
    assert "providers" in d and len(d["providers"]) == 7
    assert "tasks" in d and "runs" in d


def test_missing_key_returns_401_with_provider_hint(client):
    r = client.post("/runs", json={"provider": "openai", "task_ids": ["codegen-001"]})
    assert r.status_code == 401
    assert "OpenAI" in r.json()["detail"]


def test_unknown_provider_returns_400(client):
    r = client.post("/runs", headers={"X-Provider-Api-Key": "k"}, json={"provider": "bogus"})
    assert r.status_code == 400
    assert "Unknown provider" in r.json()["detail"]


def test_full_run_with_model_override(client):
    r = client.post(
        "/runs",
        headers={"X-Provider-Api-Key": "sk-test"},
        json={"provider": "openai", "model": "gpt-4o-mini-test", "task_ids": ["codegen-001"]},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["provider"] == "openai" and body["model"] == "gpt-4o-mini-test"
    run_id = body["run_id"]

    g = client.get(f"/runs/{run_id}").json()
    assert g["status"] == "completed"
    assert g["provider"] == "openai" and g["model"] == "gpt-4o-mini-test"
    rep = g["report"]
    assert rep["provider"] == "openai" and rep["model"] == "gpt-4o-mini-test"

    res = client.get(f"/runs/{run_id}/trajectory/codegen-001").json()
    assert res["provider"] == "openai" and res["model"] == "gpt-4o-mini-test"
    assert res["turns"] == 2  # one tool turn + one final turn
    blocks = res["trajectory"][0]["blocks"]
    assert any(b["type"] == "tool_use" for b in blocks)
    assert any(b["type"] == "tool_result" for b in blocks)


def test_llm_judge_run_scores(client):
    r = client.post(
        "/runs",
        headers={"X-Provider-Api-Key": "k"},
        json={"provider": "kimi", "task_ids": ["agentic-001"]},
    )
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]
    g = client.get(f"/runs/{run_id}").json()
    assert g["status"] == "completed"
    ts = g["report"]["task_scores"][0]
    assert ts["method"] == "llm_judge"
    assert ts["score"] == 0.75  # judge returned 3/4
    assert g["report"]["provider"] == "kimi"


def test_legacy_anthropic_header_still_works(client):
    r = client.post(
        "/runs",
        headers={"X-Anthropic-Api-Key": "sk-ant-test"},
        json={"task_ids": ["codegen-001"]},
    )
    assert r.status_code == 202
    assert r.json()["provider"] == "anthropic"


def test_module_globals_restored_after_run(client):
    import runner
    import judge
    before = (runner.PROVIDER, runner.MODEL, judge.JUDGE_MODEL)
    r = client.post(
        "/runs",
        headers={"X-Provider-Api-Key": "sk-test"},
        json={"provider": "deepseek", "model": "deepseek-reasoner", "task_ids": ["codegen-001"]},
    )
    assert r.status_code == 202
    # background task already ran inline; globals must be back to import-time values
    assert (runner.PROVIDER, runner.MODEL, judge.JUDGE_MODEL) == before
    assert runner.PROVIDER == "anthropic"
