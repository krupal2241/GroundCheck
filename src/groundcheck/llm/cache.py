"""Disk cache for LLM responses, under `data/cache/llm/` (gitignored).

The key is the sha256 of a canonical JSON object holding everything that can change
the output: provider, model, model digest, runtime version, system prompt, prompt, and
the sampling parameters (temperature, seed, max tokens, JSON mode). A re-pulled model
gets a new digest, so it never reuses answers from the old build.

Each entry is one JSON file, `<key[:2]>/<key>.json`, written atomically (`.part` then
rename), so an interrupted run never leaves a half-written entry that later looks valid.
"""

import hashlib
import json
from pathlib import Path

from groundcheck.config import REPO_ROOT
from groundcheck.llm.base import LLMMetadata, LLMProvider, LLMRequest, LLMResponse, ModelIdentity

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache" / "llm"
CACHE_FORMAT = 1


def cache_key(identity: ModelIdentity, request: LLMRequest) -> str:
    material = {
        "format": CACHE_FORMAT,
        "provider": identity.provider,
        "model": identity.model,
        "model_digest": identity.model_digest,
        "runtime_version": identity.runtime_version,
        "system": request.system,
        "prompt": request.prompt,
        **request.params(),
    }
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class DiskCache:
    def __init__(self, root: Path = DEFAULT_CACHE_DIR):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        path = self.path(key)
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            metadata = LLMMetadata.model_validate(entry["metadata"])
        except (ValueError, KeyError):
            return None  # a corrupt entry is a miss; it is overwritten on the next put
        return LLMResponse(
            text=entry["text"], metadata=metadata.model_copy(update={"cached": True})
        )

    def put(self, key: str, response: LLMResponse) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"text": response.text, "metadata": response.metadata.model_dump()}
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)


class CachedProvider:
    """Wraps a provider: identical requests to the identical model build hit the disk."""

    def __init__(self, inner: LLMProvider, cache: DiskCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.model = inner.model
        self.paid = inner.paid
        self.hits = 0
        self.misses = 0

    def identity(self) -> ModelIdentity:
        return self.inner.identity()

    def generate(self, request: LLMRequest) -> LLMResponse:
        key = cache_key(self.identity(), request)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        response = self.inner.generate(request)
        self.cache.put(key, response)
        return response
