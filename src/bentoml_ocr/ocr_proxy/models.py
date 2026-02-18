from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from ocr_proxy.constants import FULL_MODEL_NAME


class ImageUrl(BaseModel):
    url: str


class ContentTextPart(BaseModel):
    type: Literal["text"]
    text: str


class ContentImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ChatContentPart = ContentTextPart | ContentImagePart


class ChatMessage(BaseModel):
    role: str
    content: str | list[ChatContentPart]


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=FULL_MODEL_NAME)
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None)
    temperature: float | None = Field(default=None)
    stream: bool | None = Field(default=False)


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop"] = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage = Field(default_factory=ChatUsage)


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "bentoml-ocr"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class OCRBackend(Protocol):
    async def process_full(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...
    async def process_raw(self, request: ChatCompletionRequest) -> dict[str, Any]: ...
    async def close(self) -> None: ...
