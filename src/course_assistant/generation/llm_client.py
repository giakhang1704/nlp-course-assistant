"""Local LLM client (Ollama). No API key, no external network, no data leaves the machine.

The client sends the question and the retrieved evidence to a model served by
Ollama (http://localhost:11434) and constrains the reply with a JSON Schema
({status, answer, citations}); validation.py then checks the citations.
"""
import json
import os
import re
import time
from pathlib import Path
from .validation import response_schema, diagnostic_preview

# Errors meaning the local LLM is not set up (server not running, model not pulled).
UNAVAILABLE_ERRORS = {"llm_local_server_unreachable", "llm_local_model_not_found", "invalid_model_name",
                      "llm_api_key_missing", "llm_auth_error", "llm_remote_model_not_found",
                      "llm_remote_unreachable"}


class LLMError(RuntimeError):
    def __init__(self, code, diagnostic=None):
        super().__init__(code)
        self.diagnostic = diagnostic


def load_environment(root):
    """Minimal KEY=value .env support, preserving already configured environment."""
    path = Path(root) / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
                os.environ.setdefault(key, value.strip('"\''))


class OllamaClient:
    """Model served by Ollama, called through its native /api/chat endpoint."""

    def __init__(self, config, transport=None, sleep=None):
        self.config = config
        self.model = os.environ.get("OLLAMA_MODEL") or config["model"]
        self.base_url = (os.environ.get("OLLAMA_HOST") or config.get("base_url") or "http://localhost:11434").rstrip("/")
        self.transport = transport
        self.sleep = sleep or time.sleep
        self.requests_sent = 0

    def generate(self, system, question, context):
        if not re.fullmatch(r"[A-Za-z0-9._:/-]+", self.model):
            raise LLMError("invalid_model_name")
        import httpx
        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": json.dumps(
                                    {"QUESTION": question, "EVIDENCE": context}, ensure_ascii=False)}],
                   "format": response_schema(context),
                   "options": {"temperature": 0, "seed": self.config.get("seed", 0),
                               "num_predict": self.config["max_output_tokens"],
                               "num_ctx": self.config.get("num_ctx", 8192)}}
        if "think" in self.config:
            payload["think"] = self.config["think"]
        attempts = self.config.get("max_attempts", 2)
        output_text = None
        try:
            with httpx.Client(timeout=self.config["timeout_seconds"], transport=self.transport,
                              follow_redirects=False) as client:
                for attempt in range(attempts):
                    self.requests_sent += 1
                    try:
                        response = client.post(self.base_url + "/api/chat", json=payload)
                    except httpx.ConnectError as exc:
                        raise LLMError("llm_local_server_unreachable") from exc
                    if response.status_code == 404:
                        raise LLMError("llm_local_model_not_found")
                    if response.status_code >= 500 and attempt + 1 < attempts:
                        self.sleep(2 ** (attempt + 1))
                        continue
                    if response.status_code != 200:
                        raise LLMError(f"llm_http_{response.status_code}")
                    result = response.json()
                    if result.get("done_reason") not in (None, "stop"):
                        raise LLMError("llm_blocked_or_truncated")
                    output_text = result.get("message", {}).get("content", "")
                    return json.loads(output_text)
        except httpx.TimeoutException as exc:
            raise LLMError("llm_timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMError("llm_connection_error") from exc
        except (ValueError, KeyError, TypeError) as exc:
            diagnostic = diagnostic_preview(output_text) if output_text is not None else None
            raise LLMError("llm_invalid_json" if output_text is not None else "llm_invalid_response", diagnostic) from exc
        raise LLMError("llm_request_failed")


def _gemini_schema(schema):
    """Convert a JSON Schema to the OpenAPI subset accepted by responseSchema."""
    out = {}
    for key, value in schema.items():
        if key == "type":
            out["type"] = str(value).upper()
        elif key == "properties":
            out["properties"] = {name: _gemini_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out["items"] = _gemini_schema(value)
        elif key in ("enum", "required", "description"):
            out[key] = value
    return out


class GeminiClient:
    """Optional demo provider: Google Gemini through the generateContent REST API.

    Enabled with LLM_PROVIDER=gemini and GEMINI_API_KEY in the environment or .env.
    The prompt, JSON schema and output validation are the same as for Ollama.
    """

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, config, transport=None, sleep=None):
        self.config = config
        self.model = os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash-lite"
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.thinking_level = os.environ.get("GEMINI_THINKING_LEVEL", "minimal")
        self.transport = transport
        self.sleep = sleep or time.sleep
        self.requests_sent = 0

    def payload(self, system, question, context, compatible=False):
        schema = response_schema(context)
        generation = {"temperature": 0, "seed": self.config.get("seed", 0),
                      "maxOutputTokens": self.config["max_output_tokens"],
                      "responseMimeType": "application/json"}
        if compatible:
            # Older request shape: OpenAPI-subset schema and no thinking settings.
            generation["responseSchema"] = _gemini_schema(schema)
        else:
            generation["responseJsonSchema"] = schema
            if self.thinking_level:
                generation["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        return {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": json.dumps(
                    {"QUESTION": question, "EVIDENCE": context}, ensure_ascii=False)}]}],
                "generationConfig": generation}

    def generate(self, system, question, context):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.model):
            raise LLMError("invalid_model_name")
        if not self.api_key:
            raise LLMError("llm_api_key_missing")
        import httpx
        url = self.URL.format(model=self.model)
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        attempts = max(self.config.get("max_attempts", 2), 2)
        compatible = False
        output_text = None
        try:
            with httpx.Client(timeout=self.config["timeout_seconds"], transport=self.transport,
                              follow_redirects=False) as client:
                for attempt in range(attempts + 1):
                    self.requests_sent += 1
                    try:
                        response = client.post(url, headers=headers,
                                               json=self.payload(system, question, context, compatible))
                    except httpx.ConnectError as exc:
                        raise LLMError("llm_remote_unreachable") from exc
                    if response.status_code == 400 and not compatible:
                        compatible = True  # retry once with the older request shape
                        continue
                    if response.status_code in (401, 403):
                        raise LLMError("llm_auth_error")
                    if response.status_code == 404:
                        raise LLMError("llm_remote_model_not_found")
                    if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < attempts + 1:
                        self.sleep(2 ** (attempt + 1))
                        continue
                    if response.status_code != 200:
                        raise LLMError(f"llm_http_{response.status_code}")
                    result = response.json()
                    if (result.get("promptFeedback") or {}).get("blockReason"):
                        raise LLMError("llm_blocked_or_truncated")
                    candidate = (result.get("candidates") or [{}])[0]
                    if candidate.get("finishReason") not in (None, "STOP"):
                        raise LLMError("llm_blocked_or_truncated")
                    parts = (candidate.get("content") or {}).get("parts") or []
                    output_text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                    return json.loads(output_text)
        except httpx.TimeoutException as exc:
            raise LLMError("llm_timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMError("llm_connection_error") from exc
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            diagnostic = diagnostic_preview(output_text) if output_text is not None else None
            raise LLMError("llm_invalid_json" if output_text is not None else "llm_invalid_response", diagnostic) from exc
        raise LLMError("llm_request_failed")


def make_client(config):
    """Build the LLM client: Ollama by default, Gemini when LLM_PROVIDER=gemini."""
    provider = (os.environ.get("LLM_PROVIDER") or config.get("provider", "ollama")).lower()
    if provider == "gemini":
        return GeminiClient(config)
    if provider != "ollama":
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return OllamaClient(config)
