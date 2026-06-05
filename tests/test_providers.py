"""Offline tests for the provider abstraction: neutral<->native message
translation and tool-call handling for both the Anthropic and OpenAI paths,
with no network calls (the underlying SDK client is faked)."""
from types import SimpleNamespace

import pytest

import providers

TOOLS = [{
    "name": "execute_python",
    "description": "Run Python",
    "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}]


def test_registry_and_resolution():
    names = [p["name"] for p in providers.public_registry()]
    assert names == ["anthropic", "openai", "openrouter", "deepseek", "qwen", "kimi", "venice"]
    assert providers.resolve_provider(None)[0] == "anthropic"
    assert providers.resolve_provider("OpenAI")[0] == "openai"
    with pytest.raises(ValueError, match="Unknown provider"):
        providers.resolve_provider("nope")


def test_public_registry_is_secrets_free():
    for p in providers.public_registry():
        assert {"name", "label", "key_hint", "default_model", "default_judge_model", "models"} <= set(p)
        assert "api_key" not in p and "key_env" not in p  # never leak the env var name/secret


def test_openai_tool_loop_translation():
    captured = []

    def fake_create(**kwargs):
        captured.append(kwargs)
        if len(captured) == 1:  # first turn → emit a tool call
            tc = SimpleNamespace(id="call_1", type="function",
                function=SimpleNamespace(name="execute_python", arguments='{"code": "print(2+2)"}'))
            msg = SimpleNamespace(content=None, tool_calls=[tc])
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])
        msg = SimpleNamespace(content="The answer is 4", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    c = providers.ChatClient("openai", "sk-test")
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    msgs = [{"role": "user", "content": "What is 2+2?"}]
    r1 = c.create("gpt-4o-mini", 256, msgs, tools=TOOLS)
    assert r1.stop_reason == "tool_use"
    assert r1.tool_calls[0]["name"] == "execute_python"
    assert r1.tool_calls[0]["input"] == {"code": "print(2+2)"}
    # Anthropic tool schema → OpenAI function schema
    assert captured[0]["tools"][0]["type"] == "function"
    assert captured[0]["tools"][0]["function"]["name"] == "execute_python"
    assert captured[0]["tools"][0]["function"]["parameters"] == TOOLS[0]["input_schema"]

    msgs.append({"role": "assistant", "text": r1.text, "tool_calls": r1.tool_calls})
    msgs.append({"role": "tool", "results": [{"id": "call_1", "name": "execute_python", "output": "4"}]})
    r2 = c.create("gpt-4o-mini", 256, msgs, tools=TOOLS)
    assert r2.stop_reason == "end_turn" and r2.text == "The answer is 4"

    native = captured[1]["messages"]
    assert [m["role"] for m in native] == ["user", "assistant", "tool"]
    assert native[1]["tool_calls"][0]["id"] == "call_1"
    assert native[1]["tool_calls"][0]["function"]["arguments"] == '{"code": "print(2+2)"}'
    assert native[2]["tool_call_id"] == "call_1" and native[2]["content"] == "4"


def test_openai_system_and_no_tools():
    captured = []

    def fake_create(**kwargs):
        captured.append(kwargs)
        msg = SimpleNamespace(content='{"score": 4}', tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    c = providers.ChatClient("deepseek", "sk-test")
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    r = c.create("deepseek-chat", 400, [{"role": "user", "content": "grade"}], system="You are a judge")
    assert r.text == '{"score": 4}'
    assert captured[0]["messages"][0] == {"role": "system", "content": "You are a judge"}
    assert "tools" not in captured[0]


def test_openai_max_completion_tokens_fallback():
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if "max_tokens" in kwargs:
            raise Exception("Unsupported parameter: 'max_tokens' is not supported with this "
                            "model. Use 'max_completion_tokens' instead.")
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    c = providers.ChatClient("openai", "sk-test")
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    r = c.create("o4-mini", 512, [{"role": "user", "content": "hi"}])
    assert r.text == "ok"
    assert len(calls) == 2
    assert "max_tokens" not in calls[1] and calls[1]["max_completion_tokens"] == 512


def test_anthropic_tool_loop_translation():
    captured = []

    def fake_create(**kwargs):
        captured.append(kwargs)
        if len(captured) == 1:
            blocks = [SimpleNamespace(type="tool_use", id="toolu_1", name="execute_python",
                                      input={"code": "print(2+2)"})]
            return SimpleNamespace(content=blocks, stop_reason="tool_use")
        blocks = [SimpleNamespace(type="text", text="The answer is 4")]
        return SimpleNamespace(content=blocks, stop_reason="end_turn")

    c = providers.ChatClient("anthropic", "sk-ant-test")
    c._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))

    m = [{"role": "user", "content": "What is 2+2?"}]
    a1 = c.create("claude-haiku-4-5", 256, m, tools=TOOLS)
    assert a1.stop_reason == "tool_use" and a1.tool_calls[0]["input"] == {"code": "print(2+2)"}
    assert captured[0]["tools"] == TOOLS  # native Anthropic schema passed through

    m.append({"role": "assistant", "text": a1.text, "tool_calls": a1.tool_calls})
    m.append({"role": "tool", "results": [{"id": "toolu_1", "name": "execute_python", "output": "4"}]})
    a2 = c.create("claude-haiku-4-5", 256, m, tools=TOOLS)
    assert a2.stop_reason == "end_turn" and a2.text == "The answer is 4"

    native = captured[1]["messages"]
    assert native[1]["role"] == "assistant"
    assert native[1]["content"][0]["type"] == "tool_use" and native[1]["content"][0]["id"] == "toolu_1"
    assert native[2]["role"] == "user"
    tr = native[2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "toolu_1" and tr["content"] == "4"


def test_usage_is_carried_and_mapped():
    # Anthropic: native usage fields pass through unchanged
    c = providers.ChatClient("anthropic", "k")
    c._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")], stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                              cache_creation_input_tokens=5, cache_read_input_tokens=7))))
    r = c.create("claude-haiku-4-5", 100, [{"role": "user", "content": "x"}])
    assert (r.usage.input_tokens, r.usage.output_tokens) == (1000, 200)
    assert r.usage.cache_creation_input_tokens == 5 and r.usage.cache_read_input_tokens == 7

    # OpenAI: prompt/completion → input/output, cached → cache_read
    c2 = providers.ChatClient("openai", "k")
    c2._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **k: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="hi", tool_calls=None), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=500, completion_tokens=50,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=100))))))
    r2 = c2.create("gpt-4o-mini", 100, [{"role": "user", "content": "x"}])
    assert (r2.usage.input_tokens, r2.usage.output_tokens) == (500, 50)
    assert r2.usage.cache_read_input_tokens == 100

    # Missing usage → zeros, never raises (keeps cost reporting graceful)
    c3 = providers.ChatClient("openai", "k")
    c3._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **k: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="hi", tool_calls=None), finish_reason="stop")], usage=None))))
    r3 = c3.create("gpt-4o-mini", 100, [{"role": "user", "content": "x"}])
    assert r3.usage.input_tokens == 0 and r3.usage.output_tokens == 0


def test_anthropic_system_passthrough():
    captured = []

    def fake_create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], stop_reason="end_turn")

    c = providers.ChatClient("anthropic", "sk-ant-test")
    c._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    c.create("claude-sonnet-4-5", 400, [{"role": "user", "content": "grade"}], system="You are a judge")
    assert captured[0]["system"] == "You are a judge" and "tools" not in captured[0]
