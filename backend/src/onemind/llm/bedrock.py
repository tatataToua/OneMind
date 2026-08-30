"""Amazon Bedrock provider - the production path.

Not exercised on the demo machine, which has no AWS credentials. It exists
because "how would this run in production?" deserves a concrete answer, and
because writing it is what proves the `LLMProvider` seam is real: the router,
the specialists, and the synthesiser are untouched by which of these two files
is loaded.

Bedrock has no equivalent of Ollama's schema-constrained decoding, so
`structured` uses tool-use forcing instead - the model is given exactly one tool
whose input schema is the target model and told it must call it. That is the
supported way to get guaranteed-shape JSON out of Claude, and it is why this
method is not simply "prompt and hope".

`boto3` is imported lazily and is not a project dependency; install it only if
you actually target Bedrock.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from ..config import settings
from .base import Message

T = TypeVar("T", bound=BaseModel)

_STRUCTURED_TOOL = "emit_result"


class BedrockProvider:
    name = "bedrock"

    def __init__(self, region: str | None = None, model_id: str | None = None) -> None:
        self.region = region or settings.bedrock_region
        self.model_id = model_id or settings.bedrock_model_id
        # `model` is the name the rest of the codebase reads (`live_identity`,
        # the eval report); `model_id` is Bedrock's own term, kept because the
        # API calls below use it. One value, two names.
        self.model = self.model_id
        self._client: Any = None

    def _bedrock(self) -> Any:
        if self._client is None:
            try:
                import boto3  # noqa: PLC0415 - optional dependency, loaded on use
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "BedrockProvider requires boto3. Install it with "
                    "`uv add boto3`, or set ONEMIND_LLM_PROVIDER=ollama."
                ) from exc
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    @staticmethod
    def _split(messages: Sequence[Message]) -> tuple[list[dict], list[dict]]:
        """Bedrock's Converse API takes the system prompt out of band."""
        system = [{"text": m.content} for m in messages if m.role == "system"]
        turns = [
            {"role": m.role, "content": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        return system, turns

    async def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        system, turns = self._split(messages)
        response = self._bedrock().converse(
            modelId=self.model_id,
            system=system,
            messages=turns,
            inferenceConfig={"temperature": temperature},
        )
        return response["output"]["message"]["content"][0]["text"]

    async def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        system, turns = self._split(messages)
        response = self._bedrock().converse_stream(
            modelId=self.model_id,
            system=system,
            messages=turns,
            inferenceConfig={"temperature": temperature},
        )
        for event in response["stream"]:
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if "text" in delta:
                yield delta["text"]

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float = 0.0,
    ) -> T:
        system, turns = self._split(messages)
        response = self._bedrock().converse(
            modelId=self.model_id,
            system=system,
            messages=turns,
            inferenceConfig={"temperature": temperature},
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": _STRUCTURED_TOOL,
                            "description": f"Return the result as {schema.__name__}.",
                            "inputSchema": {"json": schema.model_json_schema()},
                        }
                    }
                ],
                # Forcing the tool is what makes the shape a guarantee.
                "toolChoice": {"tool": {"name": _STRUCTURED_TOOL}},
            },
        )
        for block in response["output"]["message"]["content"]:
            if "toolUse" in block:
                return schema.model_validate(block["toolUse"]["input"])

        raw = json.dumps(response["output"], default=str)[:400]
        raise RuntimeError(f"{self.model_id} did not call {_STRUCTURED_TOOL}: {raw}")

    async def aclose(self) -> None:
        self._client = None
