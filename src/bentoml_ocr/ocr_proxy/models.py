"""Pydantic models and protocols for the OpenAI-compatible OCR proxy API."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field

MODEL_NAME = "glm-ocr"


class ImageUrl(BaseModel):
    """URL reference to an image resource."""

    url: str


class ContentTextPart(BaseModel):
    """Text content part within a chat message."""

    type: Literal["text"]
    text: str


class ContentImagePart(BaseModel):
    """Image content part within a chat message, referencing an image URL."""

    type: Literal["image_url"]
    image_url: ImageUrl


ChatContentPart = ContentTextPart | ContentImagePart


class ChatMessage(BaseModel):
    """Single message in a chat conversation, with text or multimodal content."""

    role: str
    content: str | list[ChatContentPart]


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request payload."""

    model: str = Field(default=MODEL_NAME)
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None)
    temperature: float | None = Field(default=None)
    stream: bool | None = Field(default=False)


class ResponseMessage(BaseModel):
    """Assistant message returned in a chat completion response."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    """Single completion choice within a chat completion response."""

    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop"] = "stop"


class ChatUsage(BaseModel):
    """Token usage statistics for a chat completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage = Field(default_factory=ChatUsage)


class ModelCard(BaseModel):
    """Descriptor for a single model exposed by the API."""

    id: str
    object: Literal["model"] = "model"
    owned_by: str = "bentoml-ocr"


class ModelListResponse(BaseModel):
    """Response payload for the /v1/models listing endpoint."""

    object: Literal["list"] = "list"
    data: list[ModelCard]


class OCRBackend(Protocol):
    """Protocol defining the interface for OCR processing backends."""

    async def process_full(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...
    async def close(self) -> None: ...
