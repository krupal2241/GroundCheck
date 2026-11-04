"""`OllamaProvider`: plain HTTP to a local Ollama server (default http://localhost:11434).

Endpoints used:

- `GET /api/version` → the runtime version (`{"version": "0.12.3"}`);
- `GET /api/tags` → installed models with their sha256 `digest`, so every response
  records exactly which model build produced it (a re-pulled `llama3.2:3b` can differ);
- `POST /api/generate` with `stream: false` and `options: {temperature, seed,
  num_predict, num_ctx}`; `format: "json"` when JSON output is requested.

Version and digest are fetched once per provider and remembered. Requests go through a
`requests.Session` that tests replace with a fake, so no test touches the network.
"""

import time
from typing import Any

import requests

from groundcheck.llm.base import (
    DEFAULT_MODEL,
    LLMError,
    LLMRequest,
    LLMResponse,
    ModelIdentity,
    build_metadata,
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaProvider:
    name = "ollama"
    paid = False

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout_s: float = 300.0,
        session: Any = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = session if session is not None else requests.Session()
        self._identity: ModelIdentity | None = None

    def _get(self, path: str) -> dict[str, Any]:
        return self._call("GET", path)

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise LLMError(
                f"cannot reach Ollama at {self.base_url} ({exc.__class__.__name__}). "
                "Is it running? Start the Ollama app or run `ollama serve`."
            ) from exc
        if response.status_code == 404 and path == "/api/generate":
            raise LLMError(f"Ollama has no model {self.model!r}; run `ollama pull {self.model}`")
        if response.status_code != 200:
            raise LLMError(
                f"Ollama {method} {path} returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError(f"Ollama {path} returned invalid JSON") from exc

    def _digest(self) -> str:
        models = self._get("/api/tags").get("models", [])
        wanted = {self.model, f"{self.model}:latest"}
        for entry in models:
            if entry.get("name") in wanted or entry.get("model") in wanted:
                return entry.get("digest", "")
        raise LLMError(f"Ollama has no model {self.model!r}; run `ollama pull {self.model}`")

    def identity(self) -> ModelIdentity:
        if self._identity is None:
            version = self._get("/api/version").get("version", "unknown")
            self._identity = ModelIdentity(
                provider=self.name,
                model=self.model,
                model_digest=self._digest(),
                runtime="ollama",
                runtime_version=version,
            )
        return self._identity

    def generate(self, request: LLMRequest) -> LLMResponse:
        identity = self.identity()
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "seed": request.seed,
                "num_predict": request.max_tokens,
            },
        }
        if request.system:  # an empty "system" would replace the model's default one
            payload["system"] = request.system
        if request.context_window:
            payload["options"]["num_ctx"] = request.context_window
        if request.json_mode:
            payload["format"] = "json"
        started = time.monotonic()
        body = self._call("POST", "/api/generate", payload)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if "response" not in body:
            raise LLMError(f"Ollama response has no 'response' field: {str(body)[:200]}")
        total_ns = body.get("total_duration")
        metadata = build_metadata(
            identity,
            request,
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            duration_ms=int(total_ns / 1e6) if total_ns else elapsed_ms,
            extra={"done_reason": body.get("done_reason")} if body.get("done_reason") else {},
        )
        return LLMResponse(text=body["response"], metadata=metadata, raw=body)
