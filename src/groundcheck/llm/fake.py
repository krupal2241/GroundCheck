"""`FakeProvider`: a deterministic stand-in for tests and `--dry-run`.

It answers with a function of the request (`responder`), or with scripted responses
in order (`script`), or with a fixed string. Every request is recorded in `calls`, so
tests can assert on prompts and parameters.
"""

from collections.abc import Callable, Iterable

from groundcheck.llm.base import LLMRequest, LLMResponse, ModelIdentity, build_metadata


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        responder: Callable[[LLMRequest], str] | None = None,
        script: Iterable[str] | None = None,
        model: str = "fake-model",
        paid: bool = False,
    ):
        self.responder = responder
        self.script = list(script) if script is not None else None
        self.model = model
        self.paid = paid
        self.calls: list[LLMRequest] = []

    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            provider=self.name,
            model=self.model,
            model_digest="fake-digest",
            runtime="fake",
            runtime_version="0",
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self.script is not None:
            if not self.script:
                raise AssertionError("FakeProvider script exhausted")
            text = self.script.pop(0)
        elif self.responder is not None:
            text = self.responder(request)
        else:
            text = "fake response"
        metadata = build_metadata(
            self.identity(),
            request,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(text.split()),
            duration_ms=0,
        )
        return LLMResponse(text=text, metadata=metadata)
