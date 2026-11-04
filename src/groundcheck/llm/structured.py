"""Structured JSON output: ask for JSON, validate it with pydantic, retry a few times.

Small local models often wrap JSON in prose or a ```json fence, or miss a field. So:

1. the request is sent with `json_mode=True` (Ollama's `format: "json"`);
2. the first JSON object in the text is extracted (fences and surrounding prose are
   ignored) and validated against the pydantic model;
3. on failure, the prompt is re-sent with the validation error appended, at most
   `max_attempts` times in total. At temperature 0 an identical retry would give the
   identical answer, which is why the retry prompt includes the error.

If every attempt fails, `StructuredOutputError` carries all the errors and raw texts.
"""

import json
from dataclasses import replace
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from groundcheck.llm.base import LLMError, LLMProvider, LLMRequest, LLMResponse

T = TypeVar("T", bound=BaseModel)
DEFAULT_MAX_ATTEMPTS = 3


class StructuredOutputError(LLMError):
    def __init__(self, message: str, errors: list[str], texts: list[str]):
        super().__init__(message)
        self.errors = errors
        self.texts = texts


def extract_json_object(text: str) -> str:
    """The first balanced `{...}` in `text` (strings and escapes respected)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unterminated JSON object")


def parse_structured(text: str, model: type[T]) -> T:
    """Extract and validate; raises ValueError with a short, model-readable reason."""
    blob = extract_json_object(text)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg} at position {exc.pos}") from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors()
        )
        raise ValueError(f"JSON does not match the schema: {details}") from exc


def retry_prompt(prompt: str, error: str) -> str:
    return (
        f"{prompt}\n\nYour previous reply could not be used: {error}\n"
        "Reply again with only the JSON object, exactly in the requested format."
    )


def generate_structured(
    provider: LLMProvider,
    request: LLMRequest,
    model: type[T],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> tuple[T, LLMResponse]:
    """Return the validated object and the response that produced it.

    The response's metadata has `attempts` set to the number of tries used.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    base = replace(request, json_mode=True)
    current = base
    errors: list[str] = []
    texts: list[str] = []
    for attempt in range(1, max_attempts + 1):
        response = provider.generate(current)
        try:
            value = parse_structured(response.text, model)
        except ValueError as exc:
            errors.append(str(exc))
            texts.append(response.text)
            current = replace(base, prompt=retry_prompt(base.prompt, str(exc)))
            continue
        response.metadata = response.metadata.model_copy(update={"attempts": attempt})
        return value, response
    raise StructuredOutputError(
        f"no valid {model.__name__} after {max_attempts} attempt(s); last error: {errors[-1]}",
        errors,
        texts,
    )
