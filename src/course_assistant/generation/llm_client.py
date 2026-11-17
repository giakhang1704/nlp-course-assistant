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
UNAVAILABLE_ERRORS = {"llm_local_server_unreachable", "llm_local_model_not_found", "invalid_model_name"}


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


def make_client(config):
    """Build the LLM client named by config["provider"] (only "ollama")."""
    if config.get("provider", "ollama") != "ollama":
        raise ValueError(f"Unsupported LLM provider: {config.get('provider')}")
    return OllamaClient(config)
