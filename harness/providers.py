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


def request_timeout_seconds() -> float:
    """Bound individual provider HTTP requests so one stalled model turn cannot
    leave an entire benchmark run stuck forever. Override with
    FINCODEBENCH_REQUEST_TIMEOUT_SECONDS; defaults to 180 seconds.
    """
    raw = os.environ.get("FINCODEBENCH_REQUEST_TIMEOUT_SECONDS", "180").strip()
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return 180.0
    return timeout if timeout > 0 else 180.0

# ── Provider registry ─────────────────────────────────────────────────────────
# kind:    "anthropic" (native SDK) or "openai" (OpenAI-compatible /chat/completions)
# base_url: API root for OpenAI-compatible providers (None for native Anthropic)
# key_env:  env var the CLI reads the key from (web runs pass the key per request)
# key_hint: placeholder shown in the dashboard key field
# default_model / default_judge_model: used when the caller doesn't override
# models:   a few suggested model ids (powers the dashboard datalist; not enforced)
# public_models: True if the provider's /models is reachable without a key, so the
#           dashboard can load the full live catalogue even before a key is entered
# models_query: optional query params forwarded to /models (e.g. {"type": "text"}
#           to keep Venice's listing to chat-capable models, not image/tts/etc.)
# require_tool_capable_models: when a public catalogue exposes capabilities, drop
#           models that cannot emit function/tool calls; every benchmark run
#           includes tool-enabled tasks, so offering incapable models creates
#           misleading all-zero "completed" runs.
# extra_body: provider-specific Chat Completions payload fields forwarded through
#           the OpenAI SDK (e.g. Venice venice_parameters).
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
        # qwen3 series is current; the qwen-max / qwen-plus / qwen-turbo / qwen-flash
        # aliases stay valid and track the latest snapshots (qwen2.5-* superseded).
        "models": [
            "qwen3-max", "qwen-max", "qwen-plus", "qwen-flash", "qwen-turbo",
            "qwen3.5-flash", "qwen3-coder-plus", "qwen3-coder-flash", "qwq-plus",
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
        # kimi-k2.6 (256k) is the current flagship; the moonshot-v1 family is
        # still served. The older kimi-k2-*-preview / kimi-latest / -thinking-preview
        # ids were retired, so they're dropped here.
        "models": [
            "kimi-k2.6", "kimi-k2.5",
            "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k",
            "moonshot-v1-8k-vision-preview", "moonshot-v1-32k-vision-preview",
            "moonshot-v1-128k-vision-preview",
        ],
        "docs": "https://platform.moonshot.ai/console/api-keys",
    },
    "venice": {
        "label": "Venice",
        "kind": "openai",
        "base_url": "https://api.venice.ai/api/v1",
        "key_env": "VENICE_API_KEY",
        "key_hint": "your Venice API key",
        # Venice's /models is public — mark it so the dashboard loads the full
        # live catalogue without a key (like OpenRouter), instead of falling back
        # to the static list below.
        "public_models": True,
        # Venice's /models can return image/tts/embedding/… models too; pin the
        # listing to text and then filter by supportsFunctionCalling so the
        # dropdown only offers models that can actually drive benchmark tools.
        "models_query": {"type": "text"},
        "require_tool_capable_models": True,
        # Venice injects a default Venice character/system prompt unless this is
        # disabled. That prompt is helpful in chat but can steer models away from
        # OpenAI-style function calls, producing fast all-text/all-zero runs.
        "extra_body": {
            "venice_parameters": {
                "include_venice_system_prompt": False,
                "enable_web_search": "off",
                "enable_web_scraping": False,
                "enable_web_citations": False,
            }
        },
        "default_model": "llama-3.3-70b",
        "default_judge_model": "deepseek-v3.2",
        # Venice's catalogue is large and rotates often — these are current
        # starting points; with public_models the dashboard shows the full live
        # list, so this is only a fallback if that fetch fails.
        "models": [
            "llama-3.3-70b",
            "llama-3.2-3b",
            "qwen3-235b-a22b-instruct-2507",
            "qwen3-coder-480b-a35b-instruct-turbo",
            "deepseek-v3.2",
            "mistral-small-3-2-24b-instruct",
            "zai-org-glm-5-1",
            "venice-uncensored-1-2",
            "claude-sonnet-4-6",
            "kimi-k2-6",
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


def _get_nested(obj, *keys):
    """Read nested attributes/keys from SDK objects or dictionaries."""
    cur = obj
    for key in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return cur


def _model_supports_tools(model_obj) -> bool:
    """Return True unless a provider explicitly says function calling is false.

    Public catalogues differ in shape: Venice nests capabilities under
    model_spec.capabilities, while other OpenAI-compatible APIs may expose flat
    dicts or SDK objects. Treat missing metadata as allowed so providers without
    capability flags don't accidentally hide every model.
    """
    flag = _get_nested(model_obj, "model_spec", "capabilities", "supportsFunctionCalling")
    if flag is None:
        flag = _get_nested(model_obj, "capabilities", "supportsFunctionCalling")
    return flag is not False

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
        timeout = request_timeout_seconds()
        if self.kind == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url or cfg["base_url"], timeout=timeout)

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
        static `models` suggestions in the registry.

        Providers that publish capability metadata can opt into a tool-calling
        filter. FinCodeBench tasks rely on function tools, so surfacing models
        that cannot call tools leads to misleading runs where every task gets a
        text-only answer and scores zero.
        """
        # Some providers expose typed catalogues (e.g. Venice's text/image/tts);
        # forward the provider's models_query so we only list chat-capable models.
        query = self._cfg.get("models_query")
        resp = self._client.models.list(extra_query=query) if query else self._client.models.list()
        items = getattr(resp, "data", None)
        if items is None:
            items = list(resp)  # both SDKs' page objects are iterable
        ids = []
        for m in items:
            mid = getattr(m, "id", None)
            if mid is None and isinstance(m, dict):
                mid = m.get("id")
            if not mid:
                continue
            if self._cfg.get("require_tool_capable_models") and not _model_supports_tools(m):
                continue
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
        extra_body = self._cfg.get("extra_body")
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools:
            kwargs["tools"] = self._openai_tools(tools)
            # Be explicit across OpenAI-compatible providers. Some gateways treat
            # the default as text-only even when tools are present, which makes
            # the model narrate work instead of emitting function calls.
            kwargs["tool_choice"] = "auto"

        while True:
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                # OpenAI-compatible providers are not perfectly uniform. Retry
                # only when the provider explicitly rejects an optional parameter
                # we added for compatibility; otherwise surface the real error.
                msg = str(e)
                changed = False
                # Reasoning models (OpenAI o-series, gpt-5) reject the deprecated
                # max_tokens and require max_completion_tokens.
                if "max_completion_tokens" in msg and "max_tokens" in msg and "max_tokens" in kwargs:
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = max_tokens
                    changed = True
                # Some OpenAI-compatible gateways accept tools but not an explicit
                # tool_choice. Dropping it preserves tool definitions while avoiding
                # a hard provider failure.
                elif "tool_choice" in msg and "tool_choice" in kwargs:
                    kwargs.pop("tool_choice", None)
                    changed = True
                # Provider-specific extras are best-effort: if a gateway rejects
                # the pass-through payload shape, retry the plain OpenAI request.
                elif ("extra_body" in msg or "venice_parameters" in msg) and "extra_body" in kwargs:
                    kwargs.pop("extra_body", None)
                    changed = True

                if not changed:
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
