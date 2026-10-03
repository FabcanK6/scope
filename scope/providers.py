"""LLM providers for SCOPE: Google Gemini, OpenAI, Anthropic Claude, and any OpenAI-compatible service.

SCOPE's engine, rubric, verification and scoring are the same whichever model reads the note; a provider only has
to turn (instructions, note, answer schema) into JSON text. Answers are constrained to SCOPE's schema with each
provider's structured-output feature:

* Gemini     - responseSchema (see ``scope.llm.GeminiClient``)
* OpenAI     - Chat Completions ``response_format`` = ``json_schema`` (strict)
* Anthropic  - Messages API ``output_config.format`` = ``json_schema``
* Compatible - Chat Completions with ``json_schema``; falls back to ``json_object`` + the schema in the
               instructions for services that do not support schemas (Azure OpenAI, Mistral, Groq, OpenRouter,
               a local Ollama or vLLM server, ...)

Standard library only (urllib). Keys are passed in by the caller and never stored.
"""

from __future__ import annotations

import copy
import json
import re
import time
import urllib.error
import urllib.request

from scope.llm import (
    BUSY_MESSAGE,
    GeminiClient,
    LLMClient,
    LLMError,
    ModelBusy,
    _message,
)

PROVIDERS = ["Google Gemini", "OpenAI", "Anthropic Claude", "Other (OpenAI-compatible)"]
OPENAI_BASE = "https://api.openai.com/v1"
ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


# ---------------------------------------------------------------------------
# schema dialects
# ---------------------------------------------------------------------------
def to_json_schema(schema: dict, strict: bool = True) -> dict:
    """SCOPE's schema dialect (Gemini style: OBJECT/STRING, ``nullable``, ``propertyOrdering``) -> standard JSON
    Schema. ``strict`` makes every property required and forbids extra properties, as OpenAI's strict mode needs;
    optional values are expressed as nullable types instead."""
    s = copy.deepcopy(schema)
    out: dict = {}
    typ = str(s.get("type", "")).lower() or None
    if typ:
        out["type"] = [typ, "null"] if s.get("nullable") else typ
    if "enum" in s:
        out["enum"] = list(s["enum"]) + ([None] if s.get("nullable") else [])
    if typ == "object":
        props = s.get("properties", {})
        order = s.get("propertyOrdering") or list(props)
        out["properties"] = {k: to_json_schema(props[k], strict) for k in order if k in props}
        out["required"] = list(out["properties"]) if strict else list(s.get("required", []))
        out["additionalProperties"] = False
    if typ == "array" and "items" in s:
        out["items"] = to_json_schema(s["items"], strict)
    for k in ("description",):
        if k in s:
            out[k] = s[k]
    return out


def _schema_hint(schema: dict) -> str:
    return ("\n\nAnswer with one JSON object only, no other text, matching this JSON Schema:\n"
            + json.dumps(to_json_schema(schema, strict=False)))


# ---------------------------------------------------------------------------
# shared HTTP handling
# ---------------------------------------------------------------------------
class _HTTPClient(LLMClient):
    retry_delays = (2.0, 5.0)
    timeout = 120

    def __init__(self, api_key: str, model: str | None, base_url: str):
        super().__init__()
        if not api_key and self.needs_key:
            raise LLMError(f"No {self.provider} API key configured.")
        self.api_key = api_key
        self.model = model or None
        self.base_url = base_url.rstrip("/")
        self._sleep = time.sleep

    needs_key = True

    def _headers(self) -> dict:  # pragma: no cover - overridden
        return {}

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/{path}", method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", **self._headers()})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:3000]
            raise self._error(e.code, detail) from e
        except TimeoutError as e:
            raise ModelBusy("timeout") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise ModelBusy("timeout") from e
            raise LLMError(f"Could not reach {self.provider} ({e.reason}).") from e

    def _error(self, code: int, detail: str) -> LLMError:
        msg = _message(detail)
        if code in (401, 403):
            return LLMError(f"The {self.provider} API key was rejected. Check the key in the sidebar.")
        if code == 404:
            return LLMError(f"{self.provider} does not know the model '{self.model}': {msg}")
        if code == 429 or code == 529:
            if re.search(r"insufficient_quota|billing|credit", detail, re.I):
                return LLMError(f"Your {self.provider} account has no credit or quota left.")
            return ModelBusy(f"HTTP {code}")
        if code >= 500:
            return ModelBusy(f"HTTP {code}")
        return LLMError(f"{self.provider} API error {code}: {msg}")

    def _post_with_retry(self, path: str, body: dict) -> dict:
        for delay in (*self.retry_delays, None):
            try:
                return self._request("POST", path, body)
            except ModelBusy:
                if delay is None:
                    raise LLMError(BUSY_MESSAGE.replace("Gemini", self.provider)
                                   .replace("Google's free tier is", "the service is")) from None
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _need_model(self) -> str:
        if not self.model:
            models = self.list_models()
            if not models:
                raise LLMError(f"Choose a {self.provider} model in the sidebar.")
            self.model = models[0]
        return self.model

    def list_models(self) -> list[str]:  # pragma: no cover - overridden
        return []


# ---------------------------------------------------------------------------
# OpenAI and OpenAI-compatible services
# ---------------------------------------------------------------------------
class OpenAIClient(_HTTPClient):
    provider = "OpenAI"

    def __init__(self, api_key: str, model: str | None = None, base_url: str = OPENAI_BASE):
        super().__init__(api_key, model, base_url)
        self.schema_mode = "json_schema"  # becomes "json_object" if the service rejects schemas

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def list_models(self) -> list[str]:
        data = self._request("GET", "models")
        ids = [m.get("id", "") for m in data.get("data", [])]
        skip = re.compile(r"embed|whisper|tts|dall|image|audio|realtime|transcri|moderation|search|davinci|babbage",
                          re.I)
        chat = [i for i in ids if i and not skip.search(i)]
        newest = sorted(chat, key=lambda i: next((m.get("created", 0) for m in data["data"] if m.get("id") == i), 0),
                        reverse=True)
        return newest

    def generate(self, system: str, prompt: str, schema: dict | None = None) -> str:
        model = self._need_model()
        body = {"model": model, "messages": [{"role": "system", "content": system},
                                              {"role": "user", "content": prompt}]}
        if schema is not None:
            if self.schema_mode == "json_schema":
                body["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "scope_answer", "strict": True, "schema": to_json_schema(schema, strict=True)}}
            else:
                body["messages"][0]["content"] = system + _schema_hint(schema)
                body["response_format"] = {"type": "json_object"}
        try:
            data = self._post_with_retry("chat/completions", body)
        except LLMError as e:
            if schema is not None and self.schema_mode == "json_schema" and re.search(
                    r"response_format|json_schema|not supported|unsupported", str(e), re.I):
                self.schema_mode = "json_object"  # this service has no schema support: ask for plain JSON
                return self.generate(system, prompt, schema)
            raise
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"{self.provider} returned no answer.") from e
        if message.get("refusal"):
            raise LLMError(f"{self.provider} refused to answer: {message['refusal']}")
        return message.get("content") or ""


class CompatibleClient(OpenAIClient):
    """Any service that speaks the OpenAI Chat Completions API (Azure OpenAI, Mistral, Groq, OpenRouter, Ollama...)."""

    provider = "OpenAI-compatible"
    needs_key = False  # a local server may not need one

    def __init__(self, api_key: str, model: str | None = None, base_url: str = ""):
        if not base_url:
            raise LLMError("Enter the service's base URL (for example https://api.mistral.ai/v1).")
        super().__init__(api_key, model, base_url)


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------
class AnthropicClient(_HTTPClient):
    provider = "Anthropic Claude"
    max_tokens = 8192

    def __init__(self, api_key: str, model: str | None = None, base_url: str = ANTHROPIC_BASE):
        super().__init__(api_key, model, base_url)
        self.schema_mode = "output_config"  # becomes "prompt" for models without structured outputs

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION}

    def list_models(self) -> list[str]:
        data = self._request("GET", "models?limit=100")
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]  # newest first

    def generate(self, system: str, prompt: str, schema: dict | None = None) -> str:
        model = self._need_model()
        body = {"model": model, "max_tokens": self.max_tokens, "system": system,
                "messages": [{"role": "user", "content": prompt}]}
        if schema is not None:
            if self.schema_mode == "output_config":
                body["output_config"] = {"format": {"type": "json_schema",
                                                    "schema": to_json_schema(schema, strict=True)}}
            else:
                body["system"] = system + _schema_hint(schema)
        try:
            data = self._post_with_retry("messages", body)
        except LLMError as e:
            if schema is not None and self.schema_mode == "output_config" and re.search(
                    r"output_config|structured|json_schema|not supported", str(e), re.I):
                self.schema_mode = "prompt"  # older model: describe the schema in the instructions instead
                return self.generate(system, prompt, schema)
            raise
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        if data.get("stop_reason") == "refusal":
            raise LLMError(f"{self.provider} declined to answer this note.")
        return text


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def make_client(provider: str, api_key: str, model: str | None = None, base_url: str | None = None) -> LLMClient:
    """A client for one of ``PROVIDERS``."""
    if provider == "Google Gemini":
        return GeminiClient(api_key, model=model)
    if provider == "OpenAI":
        return OpenAIClient(api_key, model=model)
    if provider == "Anthropic Claude":
        return AnthropicClient(api_key, model=model)
    if provider.startswith("Other"):
        return CompatibleClient(api_key, model=model, base_url=base_url or "")
    raise LLMError(f"Unknown provider: {provider}")


def list_models(client: LLMClient) -> list[str]:
    try:
        return client.list_models()
    except LLMError:
        return []
