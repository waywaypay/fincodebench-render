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
        "default_judge_model": "claude-sonnet-4-6",
        "models": ["claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"],
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
        "models": [
            "gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
            "gpt-4.1-nano", "o3", "o4-mini", "o3-mini", "gpt-4-turbo", "gpt-3.5-turbo",
        ],
        "docs": "https://platform.openai.com/api-keys",
    },
    "openrouter": {
        "label": "OpenRouter",
        "kind": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "key_hint": "sk-or-…",
        "public_models": True,   # /models is public — full list loads without a key
        "default_model": "openai/gpt-4o-mini",
        "default_judge_model": "openai/gpt-4o",
        "models": [
            "openai/gpt-4o", "openai/gpt-4o-mini", "openai/o3",
            "anthropic/claude-opus-4-8", "anthropic/claude-sonnet-4-6",
            "google/gemini-2.5-pro", "google/gemini-2.0-flash-001",
            "meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-chat",
            "deepseek/deepseek-r1", "qwen/qwen-2.5-72b-instruct",
            "mistralai/mistral-large",
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
        "models": [
            "qwen-max", "qwen-plus", "qwen-turbo", "qwen-long",
            "qwen2.5-72b-instruct", "qwen2.5-32b-instruct", "qwen2.5-14b-instruct",
            "qwen2.5-7b-instruct", "qwen2.5-coder-32b-instruct", "qwq-32b-preview",
        ],
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
            "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k",
            "moonshot-v1-auto", "kimi-k2-0711-preview", "kimi-latest",
            "kimi-thinking-preview",
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
        "default_judge_model": "llama-3.1-405b",
        # Venice's catalogue is large and changes often — these are common
        # starting points; the dashboard fetches the full live list from the key.
        "models": [
            "llama-3.3-70b",
            "llama-3.1-405b",
            "llama-3.2-3b",
            "qwen3-235b",
            "qwen3-4b",
            "qwen-2.5-qwq-32b",
            "mistral-31-24b",
            "deepseek-r1-671b",
            "venice-uncensored",
            "dolphin-2.9.2-qwen2-72b",
        ],
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
            "public_models": cfg.get("public_models", False),
            "docs": cfg.get("docs"),
        }
        for name, cfg in PROVIDERS.items()
    ]


# ── Unified response ──────────────────────────────────────────────────────────
@dataclass
class Usage:
    """Token counts in Anthropic's shape, so pricing.compute_cost and the
    runner/judge usage accumulators work the same across providers."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class ChatResponse:
    """Provider-agnostic result of one model turn."""
    text: str                       # concatenated assistant text
    tool_calls: list = field(default_factory=list)  # [{"id","name","input"}]
    stop_reason: str = "end_turn"   # "end_turn" | "tool_use"
    usage: Usage = field(default_factory=Usage)


def _anthropic_usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return Usage()
    return Usage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
    )


def _openai_usage(resp):
    # OpenAI-style: prompt/completion tokens, with cached reads nested under
    # prompt_tokens_details. No cache-creation concept.
    u = getattr(resp, "usage", None)
    if u is None:
        return Usage()
    details = getattr(u, "prompt_tokens_details", None)
    return Usage(
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
        cache_read_input_tokens=getattr(details, "cached_tokens", 0) or 0 if details else 0,
    )


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

    def list_models(self):
        """Live list of model ids the caller's key can use, sorted. Both the
        Anthropic and OpenAI SDKs expose client.models.list() with `.id` on each
        item. Raises on auth/network errors so callers can fall back to the
        static `models` suggestions in the registry."""
        resp = self._client.models.list()
        items = getattr(resp, "data", None)
        if items is None:
            items = list(resp)  # both SDKs' page objects are iterable
        ids = []
        for m in items:
            mid = getattr(m, "id", None)
            if mid is None and isinstance(m, dict):
                mid = m.get("id")
            if mid:
                ids.append(mid)
        return sorted(ids)

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
        return ChatResponse("\n".join(text_parts), tool_calls, stop, _anthropic_usage(resp))

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
        return ChatResponse(msg.content or "", tool_calls, stop, _openai_usage(resp))


def client_from_env(provider=None):
    """Build a ChatClient for CLI use from FINCODEBENCH_PROVIDER + the provider's
    key env var. Falls back to a placeholder key so a missing key can't break
    import (the web service overrides this per run with the caller's key)."""
    name, _ = resolve_provider(provider or os.environ.get("FINCODEBENCH_PROVIDER"))
    return ChatClient(name, env_api_key(name) or "placeholder")
