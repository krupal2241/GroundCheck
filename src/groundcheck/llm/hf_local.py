"""`HFLocalProvider`: a local Hugging Face `transformers` model, for Colab's free GPU.

Default model: `Qwen/Qwen2.5-3B-Instruct` (ungated, Apache-2.0, fits a T4 in fp16).
The prompt goes through the model's own chat template. Temperature 0 means greedy
decoding (`do_sample=False`); otherwise sampling with `torch.manual_seed(seed)`.

The "digest" recorded in metadata is the Hub commit hash the weights were loaded from
(`config._commit_hash`), the closest match to Ollama's model digest. `transformers`
and `torch` are in the `[hf]` extra and imported only when the provider is created,
so CI never needs them (tests use a stand-in module).
"""

import time
from typing import Any

from groundcheck.config import ConfigError
from groundcheck.llm.base import (
    LLMRequest,
    LLMResponse,
    ModelIdentity,
    build_metadata,
)

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"


class HFLocalProvider:
    name = "hf-local"
    paid = False

    def __init__(self, model: str = DEFAULT_HF_MODEL, device: str | None = None):
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ConfigError(
                'HFLocalProvider needs the [hf] extra: pip install -e ".[hf]"'
            ) from exc
        self.model = model
        self._torch = torch
        self._transformers = transformers
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        self.lm = transformers.AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype)
        self.lm.to(device)
        self.lm.eval()
        commit = getattr(self.lm.config, "_commit_hash", None) or "unknown"
        self._identity = ModelIdentity(
            provider=self.name,
            model=model,
            model_digest=commit,
            runtime="transformers",
            runtime_version=f"transformers {transformers.__version__}, torch {torch.__version__}",
        )

    def identity(self) -> ModelIdentity:
        return self._identity

    def generate(self, request: LLMRequest) -> LLMResponse:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        kwargs: dict[str, Any] = {"max_new_tokens": request.max_tokens}
        if request.temperature > 0:
            self._torch.manual_seed(request.seed)
            kwargs.update(do_sample=True, temperature=request.temperature)
        else:
            kwargs.update(do_sample=False)
        started = time.monotonic()
        with self._torch.no_grad():
            output = self.lm.generate(**inputs, **kwargs)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        metadata = build_metadata(
            self._identity,
            request,
            prompt_tokens=int(prompt_len),
            completion_tokens=int(len(new_tokens)),
            duration_ms=elapsed_ms,
            extra={"device": self.device},
        )
        return LLMResponse(text=text.strip(), metadata=metadata)
