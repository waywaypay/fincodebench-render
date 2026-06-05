"""
FinCodeBench provider registry + unified chat client.

The benchmark is bring-your-own-key. Originally that meant an Anthropic key
only; this module lets a run target any of several model providers without the
runner or judge having to know which one. Anthropic is called through its
native SDK; every other provider here speaks the OpenAI chat-completions API,
so they all share a single OpenAI-SDK code path that differs only by base_url.

Adding a provider is a one-line entry in PROVIDERS — no changes to runner.py,
judge.py, or api.py.
"""

import json
import os
from dataclasses import dataclass, field

# ── Provider registry ─────────────────────────────────────────────────────────
# kind:    "anthropic" (native SDK) or "openai" (OpenAI-compatible /chat/completions)
# base_url: API root for OpenAI-compatible providers (None for native Anthropic)
# key_env:  env var the CLI reads the key from (web runs pass the key per request)
# key_hint: placeholder shown in the dashboard key field
# default_model / default_judge_model: used when the caller doesn't override
# models:   a few suggested model ids (powers the dashboard datalist; not enforced)
PROVIDERS = {
    "anthropic": {
        "label": "Anthropic",
        "kind": "anthropic",
        "base_url": None,
        "key_env": "ANTHROPIC_API_KEY",
        "key_hint": "sk-ant-…",
        "default_model": "claude-haiku-4-5",
        "default_judge_model": "claude-sonnet-4-5",
        "models": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
        "docs": "https://console.anthropic.com/settings/keys",
    },
    "openai": {
        "label": "OpenAI",
        "kind": "openai",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "key_hint": "sk-…",
        "default_model": "gpt-4o-mini",
        "default_judge_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o4-mini"],
        "docs": "https://platform.openai.com/api-keys",
    },
    "openrouter": {
        "label": "OpenRouter",
        "kind": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "key_hint": "sk-or-…",
        "default_model": "openai/gpt-4o-mini",
        "default_judge_model": "openai/gpt-4o",
        "models": [
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-flash-001",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
        ],
        "docs": "https://openrouter.ai/keys",
    },
    "deepseek": {
        "label": "DeepSeek",
        "kind": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
        "key_hint": "sk-…",
        "default_model": "deepseek-chat",
        "default_judge_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "docs": "https://platform.deepseek.com/api_keys",
    },
    "qwen": {
        "label": "Qwen (DashScope)",
        "kind": "openai",
        # International DashScope endpoint; for mainland China use
        # https://dashscope.aliyuncs.com/compatible-mode/v1
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "key_env": "DASHSCOPE_API_KEY",
        "key_hint": "sk-…",
        "default_model": "qwen-plus",
        "default_judge_model": "qwen-max",
        "models": ["qwen-max", "qwen-plus", "qwen-turbo", "qwen2.5-72b-instruct"],
        "docs": "https://bailian.console.alibabacloud.com/?apiKey=1",
    },
    "kimi": {
        "label": "Kimi (Moonshot)",
        "kind": "openai",
        # International endpoint; for mainland China use https://api.moonshot.cn/v1
        "base_url": "https://api.moonshot.ai/v1",
        "key_env": "MOONSHOT_API_KEY",
        "key_hint": "sk-…",
        "default_model": "moonshot-v1-8k",
        "default_judge_model": "moonshot-v1-32k",
        "models": [
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
            "kimi-k2-0711-preview",
        ],
        "docs": "https://platform.moonshot.ai/console/api-keys",
    },
    "venice": {
        "label": "Venice",
        "kind": "openai",
        "base_url": "https://api.venice.ai/api/v1",
        "key_env": "VENICE_API_KEY",
        "key_hint": "your Venice API key",
        "default_model": "llama-3.3-70b",
        "default_judge_model": "llama-3.3-70b",
        "models": ["llama-3.3-70b", "qwen3-235b", "mistral-31-24b", "llama-3.1-405b"],
        "docs": "https://venice.ai/settings/api",
    },
}

DEFAULT_PROVIDER = "anthropic"


def resolve_provider(name):
    """Normalize a provider name and return (name, config). Raises ValueError on
    an unknown provider so callers can surface a clear 400."""
    name = (name or DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{name}'. Valid: {', '.join(PROVIDERS)}"
        )
    return name, PROVIDERS[name]


def env_api_key(name):
    """Read a provider's API key from its env var (for CLI use). '' if unset."""
    _, cfg = resolve_provider(name)
    return os.environ.get(cfg["key_env"], "")


def public_registry():
    """Secrets-free provider list for the dashboard / API consumers."""
    return [
        {
            "name": name,
            "label": cfg["label"],
            "key_hint": cfg["key_hint"],
            "default_model": cfg["default_model"],
            "default_judge_model": cfg["default_judge_model"],
            "models": cfg["models"],
            "docs": cfg.get("docs"),
        }
        for name, cfg in PROVIDERS.items()
    ]


# ── Unified response ──────────────────────────────────────────────────────────
@dataclass
class ChatResponse:
    """Provider-agnostic result of one model turn."""
    text: str                       # concatenated assistant text
    tool_calls: list = field(default_factory=list)  # [{"id","name","input"}]
    stop_reason: str = "end_turn"   # "end_turn" | "tool_use"


# ── Unified client ────────────────────────────────────────────────────────────
class ChatClient:
    """One tool-calling interface over Anthropic and OpenAI-compatible APIs.

    Conversation history is kept in a neutral shape and translated to each SDK's
    native format on every call, so runner.py / judge.py never branch on
    provider. Neutral history entries:

        {"role": "user",      "content": "<text>"}
        {"role": "assistant", "text": "<text>", "tool_calls": [{"id","name","input"}]}
        {"role": "tool",      "results": [{"id","name","output"}]}
    """

    def __init__(self, provider, api_key, base_url=None):
        self.provider, cfg = resolve_provider(provider)
        self.kind = cfg["kind"]
        self._cfg = cfg
        api_key = api_key or "placeholder"  # construction is offline; calls will fail loudly
        if self.kind == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url or cfg["base_url"])

    # -- public API --
    def create(self, model, max_tokens, messages, tools=None, system=None):
        """Run one turn. `messages` is neutral history; returns a ChatResponse."""
        if self.kind == "anthropic":
            return self._create_anthropic(model, max_tokens, messages, tools, system)
        return self._create_openai(model, max_tokens, messages, tools, system)

    # -- Anthropic path --
    def _create_anthropic(self, model, max_tokens, messages, tools, system):
        native = []
        for m in messages:
            if m["role"] == "user":
                native.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = []
                if m.get("text"):
                    blocks.append({"type": "text", "text": m["text"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({
                        "type": "tool_use", "id": tc["id"],
                        "name": tc["name"], "input": tc["input"],
                    })
                native.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                native.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["output"]}
                    for r in m["results"]
                ]})

        kwargs = {"model": model, "max_tokens": max_tokens, "messages": native}
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)

        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
        stop = "tool_use" if resp.stop_reason == "tool_use" else "end_turn"
        return ChatResponse("\n".join(text_parts), tool_calls, stop)

    # -- OpenAI-compatible path --
    @staticmethod
    def _openai_tools(tools):
        """Anthropic tool schema → OpenAI function-tool schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            }
            for t in tools or []
        ]

    def _create_openai(self, model, max_tokens, messages, tools, system):
        native = []
        if system:
            native.append({"role": "system", "content": system})
        for m in messages:
            if m["role"] == "user":
                native.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                # content may be null when the turn is purely tool calls (per spec)
                entry = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    entry["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"]),
                            },
                        }
                        for tc in m["tool_calls"]
                    ]
                native.append(entry)
            elif m["role"] == "tool":
                for r in m["results"]:
                    native.append({
                        "role": "tool", "tool_call_id": r["id"], "content": r["output"],
                    })

        kwargs = {"model": model, "max_tokens": max_tokens, "messages": native}
        if tools:
            kwargs["tools"] = self._openai_tools(tools)

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # Reasoning models (OpenAI o-series, gpt-5) reject the deprecated
            # max_tokens and require max_completion_tokens. Retry once.
            msg = str(e)
            if "max_completion_tokens" in msg and "max_tokens" in msg:
                kwargs.pop("max_tokens", None)
                kwargs["max_completion_tokens"] = max_tokens
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise

        choice = resp.choices[0]
        msg = choice.message
        tool_calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "input": args})
        stop = "tool_use" if tool_calls else "end_turn"
        return ChatResponse(msg.content or "", tool_calls, stop)


def client_from_env(provider=None):
    """Build a ChatClient for CLI use from FINCODEBENCH_PROVIDER + the provider's
    key env var. Falls back to a placeholder key so a missing key can't break
    import (the web service overrides this per run with the caller's key)."""
    name, _ = resolve_provider(provider or os.environ.get("FINCODEBENCH_PROVIDER"))
    return ChatClient(name, env_api_key(name) or "placeholder")
