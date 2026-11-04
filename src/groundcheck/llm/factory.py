"""Build a provider by name, with the paid-call guard and the disk cache.

The guard lives here, so every code path that gets a provider goes through it: a
provider whose `paid` attribute is True is refused unless `allow_paid=True`
(`--allow-paid` on the command line). No paid provider exists yet; the registry is
open so one can be added later, and the guard is tested with a stand-in.
"""

from collections.abc import Callable
from typing import Any

from groundcheck.config import ConfigError, LLMConfig
from groundcheck.llm.base import LLMProvider, PaidProviderRefused
from groundcheck.llm.cache import CachedProvider, DiskCache
from groundcheck.llm.fake import FakeProvider

ProviderFactory = Callable[..., LLMProvider]


def _ollama(config: LLMConfig, model: str | None, **kwargs: Any) -> LLMProvider:
    from groundcheck.llm.ollama import OllamaProvider

    return OllamaProvider(
        model=model or config.model,
        base_url=config.ollama_url,
        timeout_s=config.timeout_s,
        **kwargs,
    )


def _hf_local(config: LLMConfig, model: str | None, **kwargs: Any) -> LLMProvider:
    from groundcheck.llm.hf_local import HFLocalProvider

    return HFLocalProvider(model=model or config.hf_model, **kwargs)


def _fake(config: LLMConfig, model: str | None, **kwargs: Any) -> LLMProvider:
    return FakeProvider(model=model or "fake-model", **kwargs)


# name -> (factory, paid). `paid` is declared here as well as on the class, so the
# guard runs *before* a paid provider is even constructed (no client, no API key read).
PROVIDERS: dict[str, tuple[ProviderFactory, bool]] = {
    "ollama": (_ollama, False),
    "hf-local": (_hf_local, False),
    "fake": (_fake, False),
}


def register_provider(name: str, factory: ProviderFactory, paid: bool) -> None:
    PROVIDERS[name] = (factory, paid)


def check_paid(name: str, paid: bool, allow_paid: bool) -> None:
    if paid and not allow_paid:
        raise PaidProviderRefused(
            f"provider {name!r} is a paid API and this project has a $0 budget; "
            "pass --allow-paid if you really mean to spend money"
        )


def make_provider(
    config: LLMConfig,
    name: str | None = None,
    model: str | None = None,
    allow_paid: bool = False,
    use_cache: bool = True,
    **kwargs: Any,
) -> LLMProvider:
    """The provider `name` (default: config.provider), cached unless `use_cache=False`."""
    name = name or config.provider
    if name not in PROVIDERS:
        raise ConfigError(f"unknown LLM provider {name!r}; choose from {sorted(PROVIDERS)}")
    factory, paid = PROVIDERS[name]
    check_paid(name, paid, allow_paid)
    provider = factory(config, model, **kwargs)
    check_paid(name, bool(getattr(provider, "paid", False)), allow_paid)  # belt and braces
    if use_cache:
        return CachedProvider(provider, DiskCache(config.cache_path))
    return provider
