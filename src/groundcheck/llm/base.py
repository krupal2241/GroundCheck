"""The `LLMProvider` interface, its request/response records, and shared errors.

Every provider takes one `LLMRequest` (system prompt, user prompt, temperature, seed,
max tokens, optional JSON mode) and returns an `LLMResponse`: the generated text plus
`LLMMetadata` saying exactly what produced it (provider, model name, model digest,
runtime and runtime version, the sampling parameters, token counts). That metadata is
written next to every generated answer, so a candidate file always says which model
build and which settings made it.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_SEED = 42


class LLMError(Exception):
    """A provider could not produce a response (server down, model missing, ...)."""


class PaidProviderRefused(LLMError):
    """A paid provider was requested without `--allow-paid`."""


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    system: str = ""
    temperature: float = 0.0
    seed: int = DEFAULT_SEED
    max_tokens: int = 768
    json_mode: bool = False  # ask the runtime to constrain output to JSON, if it can
    # Context window to allocate (Ollama `num_ctx`). Ollama's default (2,048 tokens in
    # many versions) silently truncates a prompt with 5 retrieved chunks. None = default.
    context_window: int | None = None

    def params(self) -> dict[str, Any]:
        """The sampling parameters, as recorded in metadata and in the cache key."""
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "context_window": self.context_window,
        }


class LLMMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str  # "ollama", "hf-local", "fake"
    model: str  # e.g. "llama3.2:3b"
    model_digest: str  # Ollama's sha256 digest, or the HF commit hash
    runtime: str  # "ollama", "transformers", "fake"
    runtime_version: str
    temperature: float
    seed: int
    max_tokens: int
    json_mode: bool = False
    context_window: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_ms: int | None = None
    cached: bool = False  # served from the disk cache
    attempts: int = 1  # structured output: how many tries it took
    extra: dict[str, Any] = {}  # provider-specific details (e.g. device)


@dataclass
class LLMResponse:
    text: str
    metadata: LLMMetadata
    raw: dict[str, Any] = field(default_factory=dict)  # not cached or recorded


@dataclass(frozen=True)
class ModelIdentity:
    """What the cache key and metadata need to know about the model before a call."""

    provider: str
    model: str
    model_digest: str
    runtime: str
    runtime_version: str


class LLMProvider(Protocol):
    """Anything with these attributes and methods is a provider."""

    name: str  # registry name, e.g. "ollama"
    model: str
    paid: bool  # True for any API that costs money; refused without --allow-paid

    def identity(self) -> ModelIdentity: ...

    def generate(self, request: LLMRequest) -> LLMResponse: ...


def build_metadata(identity: ModelIdentity, request: LLMRequest, **fields: Any) -> LLMMetadata:
    return LLMMetadata(
        provider=identity.provider,
        model=identity.model,
        model_digest=identity.model_digest,
        runtime=identity.runtime,
        runtime_version=identity.runtime_version,
        **request.params(),
        **fields,
    )
