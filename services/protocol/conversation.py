from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator

import tiktoken
from PIL import Image

from services.account_service import account_service
from services.config import config
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol.chatgpt_markup import collect_references, sanitize
from utils.helper import (
    IMAGE_MODELS,
    anonymize_token,
    extract_image_from_message_content,
    is_codex_image_model,
    is_supported_image_model,
    split_image_model,
)
from utils.log import logger


class ImageGenerationError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        error_type: str = "server_error",
        code: str | None = "upstream_error",
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param

    def to_openai_error(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def is_token_invalid_error(message: str) -> bool:
    text = str(message or "").lower()
    return (
        "token_invalidated" in text
        or "token_revoked" in text
        or "authentication token has been invalidated" in text
        or "invalidated oauth token" in text
    )


def is_rate_limit_error(exc: Exception | str) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    text = str(exc or "").lower()
    return (
        "status=429" in text
        or "http 429" in text
        or "too many requests" in text
        or "rate_limit_exceeded" in text
        or "usage_limit_reached" in text
    )


def _is_auth_error(exc: Exception | str) -> bool:
    try:
        status_code = int(getattr(exc, "status_code", 0) or 0)
    except (TypeError, ValueError):
        status_code = 0
    return status_code == 401 or is_token_invalid_error(str(exc))


def _probe_image_account_models(
        backend: OpenAIBackendAPI,
        access_token: str,
        *,
        source_type: str,
        model: str,
        requested_resolution: str = "",
        requested_size: str = "",
        codex_size: str = "",
) -> tuple[bool, str]:
    try:
        backend.list_models()
        payload = {
            "event": "image_account_model_probe_ok",
            "source_type": source_type,
            "model": model,
            "requested_resolution": requested_resolution,
            "requested_size": requested_size,
            "token": anonymize_token(access_token),
        }
        if codex_size:
            payload["codex_size"] = codex_size
        logger.info(payload)
        return True, ""
    except Exception as exc:
        error = str(exc) or exc.__class__.__name__
        action = "skipped"
        if is_rate_limit_error(exc):
            account_service.mark_image_rate_limited(
                access_token,
                error=error,
                headers=getattr(exc, "headers", None),
                body=getattr(exc, "body", None),
            )
            action = "rate_limited"
        else:
            account_service.release_image_slot(access_token)
            if _is_auth_error(exc):
                account_service.remove_invalid_token(access_token, f"{source_type}_image_model_probe")
                action = "invalid"
        payload = {
            "event": "image_account_model_probe_failed",
            "source_type": source_type,
            "model": model,
            "requested_resolution": requested_resolution,
            "requested_size": requested_size,
            "token": anonymize_token(access_token),
            "action": action,
            "error": error,
        }
        if codex_size:
            payload["codex_size"] = codex_size
        logger.warning(payload)
        return False, error


def mark_image_failure(access_token: str, exc: Exception | None = None) -> None:
    if exc is not None and is_rate_limit_error(exc):
        account_service.mark_image_rate_limited(
            access_token,
            error=str(exc),
            headers=getattr(exc, "headers", None),
            body=getattr(exc, "body", None),
        )
        return
    account_service.mark_image_result(access_token, False)


def image_stream_error_message(message: str) -> str:
    text = str(message or "")
    lower = text.lower()
    if "curl: (35)" in lower or "tls connect error" in lower or "openssl_internal" in lower:
        return "upstream image connection failed, please retry later"
    return text or "image generation failed"


def encode_images(images: Iterable[tuple[bytes, str, str]]) -> list[str]:
    return [base64.b64encode(data).decode("ascii") for data, _, _ in images if data]


def save_image_bytes(image_data: bytes, base_url: str | None = None) -> str:
    config.cleanup_old_images()
    file_hash = hashlib.md5(image_data).hexdigest()
    filename = f"{int(time.time())}_{file_hash}.png"
    relative_dir = Path(time.strftime("%Y"), time.strftime("%m"), time.strftime("%d"))
    file_path = config.images_dir / relative_dir / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(image_data)
    return f"{(base_url or config.base_url)}/images/{relative_dir.as_posix()}/{filename}"


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and str(item.get("type") or "") in {"text", "input_text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def normalize_messages(messages: object, system: Any = None) -> list[dict[str, Any]]:
    normalized = []
    if config.global_system_prompt:
        normalized.append({"role": "system", "content": config.global_system_prompt})
    system_text = message_text(system)
    if system_text:
        normalized.append({"role": "system", "content": system_text})
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "user")
            content = message.get("content", "")
            text = message_text(content)
            images: list[tuple[bytes, str]] = []
            if role == "user":
                images.extend(extract_image_from_message_content(content))
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict) or part.get("type") != "image":
                            continue
                        data = part.get("data")
                        if isinstance(data, (bytes, bytearray)):
                            images.append((bytes(data), str(part.get("mime") or "image/png")))
            if images:
                parts: list[Any] = []
                if text:
                    parts.append({"type": "text", "text": text})
                for data, mime in images:
                    parts.append({"type": "image", "data": data, "mime": mime})
                normalized.append({"role": role, "content": parts})
            else:
                normalized.append({"role": role, "content": text})
    return normalized


def prompt_with_global_system(prompt: str) -> str:
    return f"{config.global_system_prompt}\n\n{prompt}" if config.global_system_prompt else prompt


def assistant_history_text(messages: list[dict[str, Any]]) -> str:
    return "".join(str(item.get("content") or "") for item in messages if item.get("role") == "assistant")


def assistant_history_messages(messages: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("content") or "") for item in messages if item.get("role") == "assistant" and item.get("content")]


def build_image_prompt(prompt: str, size: str | None) -> str:
    if not size:
        return prompt
    if size not in {"1:1", "16:9", "9:16", "4:3", "3:4"}:
        return f"{prompt.strip()}\n\n输出图片，宽高比为 {size}。"
    hint = {
        "1:1": "输出为 1:1 正方形构图，主体居中，适合正方形画幅。",
        "16:9": "输出为 16:9 横屏构图，适合宽画幅展示。",
        "9:16": "输出为 9:16 竖屏构图，适合竖版画幅展示。",
        "4:3": "输出为 4:3 比例，兼顾宽度与高度，适合展示画面细节。",
        "3:4": "输出为 3:4 比例，纵向构图，适合人物肖像或竖向场景。",
    }[size]
    return f"{prompt.strip()}\n\n{hint}"


IMAGE_RESOLUTION_HINTS = {
    "1k": "目标输出分辨率为 1K 级别，优先保证构图稳定和主体清晰。",
    "2k": "目标输出分辨率为 2K 级别，尽可能输出长边约 2048px 的高清图片，保留细节纹理。",
    "4k": "目标输出分辨率为 4K 级别，尽可能输出接近 3840px 长边的超清图片，保留丰富细节和干净边缘。",
}


def normalize_image_resolution(value: object) -> str | None:
    normalized = str(value or "").strip().lower().replace(" ", "").replace("-", "")
    aliases = {
        "auto": "",
        "default": "",
        "": "",
        "1": "1k",
        "1024": "1k",
        "1024px": "1k",
        "1024x1024": "1k",
        "1k": "1k",
        "2": "2k",
        "2048": "2k",
        "2048px": "2k",
        "2048x2048": "2k",
        "2k": "2k",
        "4": "4k",
        "3840": "4k",
        "3840px": "4k",
        "3840x2160": "4k",
        "2160x3840": "4k",
        "4096": "4k",
        "4096px": "4k",
        "4096x4096": "4k",
        "4k": "4k",
    }
    result = aliases.get(normalized)
    if result is None:
        return None
    return result or None


def build_image_prompt_with_options(prompt: str, size: str | None, resolution: str | None = None) -> str:
    final_prompt = build_image_prompt(prompt, size).strip()
    normalized_resolution = normalize_image_resolution(resolution)
    if not normalized_resolution:
        return final_prompt
    return f"{final_prompt}\n\n{IMAGE_RESOLUTION_HINTS[normalized_resolution]}"


def image_plan_candidates(request: "ConversationRequest") -> list[str | None]:
    model_plan_type, _ = split_image_model(request.model)
    allowed = normalize_allowed_image_plan_types(request.allowed_plan_types)
    if allowed:
        base_candidates = list(allowed)
        if request.plan_type and (normalized_plan := normalize_image_plan_type(request.plan_type)):
            base_candidates = [normalized_plan]
        elif model_plan_type and (normalized_plan := normalize_image_plan_type(model_plan_type)):
            base_candidates = [normalized_plan]
        elif normalize_image_resolution(request.resolution) in {"2k", "4k"}:
            base_candidates = [plan for plan in ("Pro", "Plus", "Team") if plan in allowed]
        return [plan for plan in base_candidates if plan in allowed]
    if request.plan_type:
        return [request.plan_type]
    if model_plan_type:
        return [model_plan_type]
    if normalize_image_resolution(request.resolution) in {"2k", "4k"}:
        return ["Pro", "Plus", None]
    return [None]


def normalize_image_plan_type(value: object) -> str | None:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    compact = raw.replace("_", "")
    return {
        "free": "free",
        "plus": "Plus",
        "pro": "Pro",
        "team": "Team",
        "business": "Team",
    }.get(compact)


def normalize_allowed_image_plan_types(value: object) -> tuple[str, ...]:
    if not value:
        return ()
    raw_items = value if isinstance(value, (list, tuple, set)) else [value]
    order = ["Pro", "Plus", "Team", "free"]
    normalized: list[str] = []
    for item in raw_items:
        plan = normalize_image_plan_type(item)
        if plan and plan not in normalized:
            normalized.append(plan)
    return tuple(sorted(normalized, key=lambda item: order.index(item) if item in order else len(order)))


def image_bytes_dimensions(image_data: bytes) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(image_data)) as image:
            width, height = image.size
            return int(width), int(height)
    except Exception:
        return 0, 0


def codex_image_size_for_request(request: "ConversationRequest") -> str | None:
    resolution = normalize_image_resolution(request.resolution)
    if resolution not in {"2k", "4k"}:
        return None
    aspect = str(request.size or "").strip()
    table = {
        "2k": {
            "1:1": "2048x2048",
            "16:9": "2048x1152",
            "9:16": "1152x2048",
            "4:3": "2048x1536",
            "3:4": "1536x2048",
            "": "2048x2048",
        },
        "4k": {
            "1:1": "2048x2048",
            "16:9": "3840x2160",
            "9:16": "2160x3840",
            "4:3": "3072x2304",
            "3:4": "2304x3072",
            "": "3840x2160",
        },
    }
    return table[resolution].get(aspect) or table[resolution][""]


def image_generation_model_for_tool(model: str) -> str:
    _, base_model = split_image_model(model)
    if base_model == "gpt-image-2":
        return "gpt-image-2"
    if base_model == "codex-gpt-image-2":
        return "gpt-image-2"
    return "gpt-image-2"


def encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("o200k_base")
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


def count_message_tokens(messages: list[dict[str, Any]], model: str) -> int:
    encoding = encoding_for_model(model)
    total = 0
    for message in messages:
        total += 3
        for key, value in message.items():
            if not isinstance(value, str):
                continue
            total += len(encoding.encode(value))
            if key == "name":
                total += 1
    return total + 3


def count_text_tokens(text: str, model: str) -> int:
    return len(encoding_for_model(model).encode(text))


def format_image_result(
    items: list[dict[str, Any]],
    prompt: str,
    response_format: str,
    base_url: str | None = None,
    created: int | None = None,
    message: str = "",
) -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    for item in items:
        b64_json = str(item.get("b64_json") or "").strip()
        if not b64_json:
            continue
        revised_prompt = str(item.get("revised_prompt") or prompt).strip() or prompt
        if response_format == "b64_json":
            data.append({
                "b64_json": b64_json,
                "url": save_image_bytes(base64.b64decode(b64_json), base_url),
                "revised_prompt": revised_prompt,
            })
        else:
            data.append({
                "url": save_image_bytes(base64.b64decode(b64_json), base_url),
                "revised_prompt": revised_prompt,
            })
    result: dict[str, Any] = {"created": created or int(time.time()), "data": data}
    if message and not data:
        result["message"] = message
    return result


@dataclass
class ConversationRequest:
    model: str = "auto"
    prompt: str = ""
    messages: list[dict[str, Any]] | None = None
    conversation_id: str = ""
    plan_type: str | None = None
    images: list[str] | None = None
    n: int = 1
    size: str | None = None
    resolution: str | None = None
    allowed_plan_types: tuple[str, ...] | list[str] | set[str] | None = None
    response_format: str = "b64_json"
    base_url: str | None = None
    message_as_error: bool = False


@dataclass
class ConversationState:
    text: str = ""
    clean_text: str = ""
    conversation_id: str = ""
    file_ids: list[str] = field(default_factory=list)
    sediment_ids: list[str] = field(default_factory=list)
    blocked: bool = False
    tool_invoked: bool | None = None
    turn_use_case: str = ""
    references: dict[str, dict[str, Any]] = field(default_factory=dict)
    cite_numbers: dict[str, int] = field(default_factory=dict)
    cite_counter: list[int] = field(default_factory=lambda: [0])


@dataclass
class ImageOutput:
    kind: str
    model: str
    index: int
    total: int
    created: int = field(default_factory=lambda: int(time.time()))
    text: str = ""
    upstream_event_type: str = ""
    data: list[dict[str, Any]] = field(default_factory=list)

    def to_chunk(self) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "object": "image.generation.chunk",
            "created": self.created,
            "model": self.model,
            "index": self.index,
            "total": self.total,
            "progress_text": self.text,
            "upstream_event_type": self.upstream_event_type,
            "data": [],
        }
        if self.kind == "message":
            chunk.update({
                "object": "image.generation.message",
                "message": self.text,
            })
            chunk.pop("progress_text", None)
            chunk.pop("upstream_event_type", None)
        elif self.kind == "result":
            chunk.update({
                "object": "image.generation.result",
                "data": self.data,
            })
            chunk.pop("progress_text", None)
            chunk.pop("upstream_event_type", None)
        return chunk


def assistant_message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or {}
    parts = content.get("parts") or []
    if not isinstance(parts, list):
        return ""
    return "".join(part for part in parts if isinstance(part, str))


def strip_history(text: str, history_text: str = "") -> str:
    text = str(text or "")
    history_text = str(history_text or "")
    while history_text and text.startswith(history_text):
        text = text[len(history_text):]
    return text


def assistant_text(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if not isinstance(message, dict):
            continue
        role = str((message.get("author") or {}).get("role") or "").strip().lower()
        if role != "assistant":
            continue
        text = assistant_message_text(message)
        if text:
            return strip_history(text, history_text)
    return apply_text_patch(event, current_text, history_text)


def event_assistant_text(event: dict[str, Any], history_text: str = "") -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if isinstance(message, dict) and (message.get("author") or {}).get("role") == "assistant":
            return strip_history(assistant_message_text(message), history_text)
    return ""


def apply_text_patch(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    if event.get("p") == "/message/content/parts/0":
        return apply_patch_op(event, current_text, history_text)

    operations = event.get("v")
    if isinstance(operations, str) and current_text and not event.get("p") and not event.get("o"):
        return current_text + operations

    if event.get("o") == "patch" and isinstance(operations, list):
        text = current_text
        for item in operations:
            if isinstance(item, dict):
                text = apply_text_patch(item, text, history_text)
        return text

    if not isinstance(operations, list):
        return current_text

    text = current_text
    for item in operations:
        if isinstance(item, dict):
            text = apply_text_patch(item, text, history_text)
    return text


def apply_patch_op(operation: dict[str, Any], current_text: str, history_text: str = "") -> str:
    op = operation.get("o")
    value = str(operation.get("v") or "")
    if op == "append":
        return current_text + value
    if op == "replace":
        return strip_history(value, history_text)
    return current_text


def add_unique(values: list[str], candidates: list[str]) -> None:
    for candidate in candidates:
        if candidate and candidate not in values:
            values.append(candidate)


def extract_conversation_ids(payload: str) -> tuple[str, list[str], list[str]]:
    conversation_match = re.search(r'"conversation_id"\s*:\s*"([^"]+)"', payload)
    conversation_id = conversation_match.group(1) if conversation_match else ""
    file_ids = re.findall(r"(file[-_][A-Za-z0-9]+)", payload)
    sediment_ids = re.findall(r"sediment://([A-Za-z0-9_-]+)", payload)
    return conversation_id, file_ids, sediment_ids


def is_image_tool_event(event: dict[str, Any]) -> bool:
    value = event.get("v")
    message = event.get("message") or (value.get("message") if isinstance(value, dict) else None)
    if not isinstance(message, dict):
        return False
    metadata = message.get("metadata") or {}
    author = message.get("author") or {}
    return author.get("role") == "tool" and metadata.get("async_task_type") == "image_gen"


def update_conversation_state(state: ConversationState, payload: str, event: dict[str, Any] | None = None) -> None:
    conversation_id, file_ids, sediment_ids = extract_conversation_ids(payload)
    if conversation_id and not state.conversation_id:
        state.conversation_id = conversation_id
    if isinstance(event, dict) and is_image_tool_event(event):
        add_unique(state.file_ids, file_ids)
        add_unique(state.sediment_ids, sediment_ids)
    if not isinstance(event, dict):
        return
    collect_references(event, state.references)
    state.conversation_id = str(event.get("conversation_id") or state.conversation_id)
    value = event.get("v")
    if isinstance(value, dict):
        state.conversation_id = str(value.get("conversation_id") or state.conversation_id)
    if event.get("type") == "moderation":
        moderation = event.get("moderation_response")
        if isinstance(moderation, dict) and moderation.get("blocked") is True:
            state.blocked = True
    if event.get("type") == "server_ste_metadata":
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            if isinstance(metadata.get("tool_invoked"), bool):
                state.tool_invoked = metadata["tool_invoked"]
            state.turn_use_case = str(metadata.get("turn_use_case") or state.turn_use_case)


def conversation_base_event(event_type: str, state: ConversationState, **extra: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "text": state.clean_text or state.text,
        "raw_text": state.text,
        "conversation_id": state.conversation_id,
        "file_ids": list(state.file_ids),
        "sediment_ids": list(state.sediment_ids),
        "blocked": state.blocked,
        "tool_invoked": state.tool_invoked,
        "turn_use_case": state.turn_use_case,
        **extra,
    }


def iter_conversation_payloads(payloads: Iterator[str], history_text: str = "",
                               history_messages: list[str] | None = None) -> Iterator[dict[str, Any]]:
    state = ConversationState()
    history_messages = history_messages or []
    history_index = 0
    for payload in payloads:
        # print(f"[upstream_sse] {payload}", flush=True)
        if not payload:
            continue
        if payload == "[DONE]":
            yield conversation_base_event("conversation.done", state, done=True)
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            update_conversation_state(state, payload)
            yield conversation_base_event("conversation.raw", state, payload=payload)
            continue
        if not isinstance(event, dict):
            yield conversation_base_event("conversation.event", state, raw=event)
            continue
        update_conversation_state(state, payload, event)
        if history_index < len(history_messages) and event_assistant_text(event, history_text) == history_messages[history_index]:
            history_index += 1
            state.text = ""
            state.clean_text = ""
            continue
        next_text = assistant_text(event, state.text, history_text)
        if next_text != state.text:
            state.text = next_text
            next_clean = sanitize(next_text, state.references, state.cite_numbers, state.cite_counter)
            delta = next_clean[len(state.clean_text):] if next_clean.startswith(state.clean_text) else next_clean
            state.clean_text = next_clean
            if delta:
                yield conversation_base_event("conversation.delta", state, raw=event, delta=delta)
                continue
            yield conversation_base_event("conversation.event", state, raw=event)
            continue
        yield conversation_base_event("conversation.event", state, raw=event)


def conversation_events(
    backend: OpenAIBackendAPI,
    messages: list[dict[str, Any]] | None = None,
    model: str = "auto",
    prompt: str = "",
    images: list[str] | None = None,
    size: str | None = None,
    resolution: str | None = None,
) -> Iterator[dict[str, Any]]:
    normalized = normalize_messages(messages or ([{"role": "user", "content": prompt}] if prompt else []))
    image_model = is_supported_image_model(model)
    history_text = "" if image_model else assistant_history_text(normalized)
    history_messages = [] if image_model else assistant_history_messages(normalized)
    final_prompt = prompt_with_global_system(build_image_prompt_with_options(prompt, size, resolution)) if image_model else prompt
    if image_model:
        logger.info({
            "event": "image_conversation_options",
            "model": model,
            "requested_resolution": resolution or "",
            "normalized_resolution": normalize_image_resolution(resolution) or "",
            "requested_size": size or "",
            "reference_count": len(images or []),
            "prompt_length": len(prompt or ""),
            "final_prompt_length": len(final_prompt or ""),
        })
    payloads = backend.stream_conversation(
        messages=normalized,
        model=model,
        prompt=final_prompt,
        images=images if image_model else None,
        system_hints=["picture_v2"] if image_model else None,
        image_size=size if image_model else None,
        image_resolution=resolution if image_model else None,
    )
    yield from iter_conversation_payloads(payloads, history_text, history_messages)


def text_backend() -> OpenAIBackendAPI:
    return OpenAIBackendAPI(access_token=account_service.get_text_access_token())


def stream_text_deltas(backend: OpenAIBackendAPI, request: ConversationRequest) -> Iterator[str]:
    attempted_tokens: set[str] = set()
    token = getattr(backend, "access_token", "")
    emitted = False
    while True:
        if token and token in attempted_tokens:
            raise RuntimeError("no available text account")
        if token:
            attempted_tokens.add(token)
        try:
            active_backend = OpenAIBackendAPI(access_token=token)
            for event in conversation_events(active_backend, messages=request.messages, model=request.model, prompt=request.prompt):
                if event.get("type") != "conversation.delta":
                    continue
                delta = str(event.get("delta") or "")
                if delta:
                    emitted = True
                    yield delta
            account_service.mark_text_used(token)
            return
        except Exception as exc:
            error_message = str(exc)
            if token and not emitted and is_token_invalid_error(error_message):
                account_service.remove_invalid_token(token, "text_stream")
                token = account_service.get_text_access_token(attempted_tokens)
                if token:
                    continue
            raise


def collect_text(backend: OpenAIBackendAPI, request: ConversationRequest) -> str:
    return "".join(stream_text_deltas(backend, request))


def stream_image_outputs(
        backend: OpenAIBackendAPI,
        request: ConversationRequest,
        index: int = 1,
        total: int = 1,
) -> Iterator[ImageOutput]:
    last: dict[str, Any] = {}
    for event in conversation_events(
            backend,
            prompt=request.prompt,
            model=request.model,
            images=request.images or [],
            size=request.size,
            resolution=request.resolution,
    ):
        last = event
        if event.get("type") == "conversation.delta":
            yield ImageOutput(
                kind="progress",
                model=request.model,
                index=index,
                total=total,
                text=str(event.get("delta") or ""),
                upstream_event_type="conversation.delta",
            )
            continue
        if event.get("type") == "conversation.event":
            raw = event.get("raw")
            raw_type = str(raw.get("type") or "") if isinstance(raw, dict) else ""
            yield ImageOutput(
                kind="progress",
                model=request.model,
                index=index,
                total=total,
                upstream_event_type=raw_type,
            )

    conversation_id = str(last.get("conversation_id") or "")
    file_ids = [str(item) for item in last.get("file_ids") or []]
    sediment_ids = [str(item) for item in last.get("sediment_ids") or []]
    message = str(last.get("text") or "").strip()
    is_text_response = last.get("tool_invoked") is False or last.get("turn_use_case") == "text"
    logger.info({
        "event": "image_stream_resolve_start",
        "conversation_id": conversation_id,
        "file_ids": file_ids,
        "sediment_ids": sediment_ids,
        "tool_invoked": last.get("tool_invoked"),
        "turn_use_case": last.get("turn_use_case"),
    })
    if message and not file_ids and not sediment_ids and (last.get("blocked") or is_text_response):
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message)
        return

    image_urls = backend.resolve_conversation_image_urls(conversation_id, file_ids, sediment_ids)
    if image_urls:
        downloaded_images = list(backend.download_image_bytes(image_urls))
        dimensions = [image_bytes_dimensions(image_data) for image_data in downloaded_images]
        logger.info({
            "event": "image_stream_downloaded",
            "conversation_id": conversation_id,
            "requested_resolution": request.resolution or "",
            "requested_size": request.size or "",
            "image_count": len(downloaded_images),
            "dimensions": [
                {"width": width, "height": height}
                for width, height in dimensions
            ],
        })
        image_items = [
            {"b64_json": base64.b64encode(image_data).decode("ascii")}
            for image_data in downloaded_images
        ]
        data = format_image_result(
            image_items,
            request.prompt,
            request.response_format,
            request.base_url,
            int(time.time()),
        )["data"]
        if data:
            yield ImageOutput(kind="result", model=request.model, index=index, total=total, data=data)
        return

    if message:
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message)


def stream_codex_image_outputs(
        backend: OpenAIBackendAPI,
        request: ConversationRequest,
        image_size: str,
        index: int = 1,
        total: int = 1,
) -> Iterator[ImageOutput]:
    items = backend.generate_codex_image(
        prompt=build_image_prompt(request.prompt, request.size),
        image_size=image_size,
        model=image_generation_model_for_tool(request.model),
        images=request.images or [],
    )
    image_items = []
    dimensions = []
    for item in items:
        b64_json = str(item.get("result") or "").strip()
        if not b64_json:
            continue
        try:
            image_data = base64.b64decode(b64_json)
        except Exception:
            continue
        dimensions.append(image_bytes_dimensions(image_data))
        image_items.append({
            "b64_json": b64_json,
            "revised_prompt": str(item.get("revised_prompt") or request.prompt).strip() or request.prompt,
        })
    logger.info({
        "event": "codex_image_stream_downloaded",
        "requested_resolution": request.resolution or "",
        "requested_size": request.size or "",
        "codex_size": image_size,
        "image_count": len(image_items),
        "dimensions": [
            {"width": width, "height": height}
            for width, height in dimensions
        ],
    })
    data = format_image_result(
        image_items,
        request.prompt,
        request.response_format,
        request.base_url,
        int(time.time()),
    )["data"]
    if data:
        yield ImageOutput(kind="result", model=request.model, index=index, total=total, data=data)
        return
    raise RuntimeError("Codex image generation returned no image")


def try_stream_codex_image_outputs_with_pool(
        request: ConversationRequest,
        index: int,
        total: int,
) -> tuple[bool, Iterator[ImageOutput] | None]:
    image_size = codex_image_size_for_request(request)
    if not image_size:
        return False, None
    candidates = image_plan_candidates(request)
    logger.info({
        "event": "codex_image_account_select_start",
        "model": request.model,
        "requested_resolution": request.resolution or "",
        "requested_size": request.size or "",
        "codex_size": image_size,
        "candidate_plan_types": [plan_type or "auto" for plan_type in candidates],
        "source_type": "codex",
    })

    def select_token(excluded_tokens: set[str]) -> tuple[str, str | None, str]:
        plan_error = ""
        for plan_type in candidates:
            try:
                access_token = account_service.get_available_access_token(
                    plan_type=plan_type,
                    source_type="codex",
                    plan_types={"Plus", "Team", "Pro"} if not plan_type else None,
                    excluded_tokens=excluded_tokens,
                )
                return access_token, plan_type, ""
            except RuntimeError as exc:
                plan_error = str(exc)
                logger.info({
                    "event": "codex_image_account_select_plan_unavailable",
                    "model": request.model,
                    "requested_resolution": request.resolution or "",
                    "plan_type": plan_type or "auto",
                    "error": plan_error,
                })
        return "", None, plan_error

    excluded_tokens: set[str] = set()
    token, selected_plan_type, plan_error = select_token(excluded_tokens)
    if not token:
        logger.info({
            "event": "codex_image_account_unavailable",
            "model": request.model,
            "requested_resolution": request.resolution or "",
            "codex_size": image_size,
            "error": plan_error or "no available codex image quota",
        })
        return False, None

    def iterator() -> Iterator[ImageOutput]:
        nonlocal token, selected_plan_type, plan_error
        last_rate_limit_error = ""
        last_probe_error = ""
        while token:
            returned_result = False
            original_token = token
            active_token = token
            refreshed_token = account_service.refresh_oauth_access_token(token)
            if refreshed_token:
                active_token = refreshed_token
                excluded_tokens.add(token)
                token = active_token
            selected_account = account_service.get_account(active_token) or {}
            logger.info({
                "event": "codex_image_account_selected",
                "model": request.model,
                "requested_resolution": request.resolution or "",
                "requested_size": request.size or "",
                "codex_size": image_size,
                "selected_plan_type": selected_plan_type or "auto",
                "account_type": selected_account.get("type") or "",
                "account_status": selected_account.get("status") or "",
                "account_source_type": selected_account.get("source_type") or "",
                "account_quota": selected_account.get("quota"),
                "account_image_quota_unknown": bool(selected_account.get("image_quota_unknown")),
                "token": anonymize_token(active_token),
                "oauth_refreshed": active_token != original_token,
            })
            try:
                backend = OpenAIBackendAPI(access_token=active_token)
                probe_ok, probe_error = _probe_image_account_models(
                    backend,
                    active_token,
                    source_type="codex",
                    model=request.model,
                    requested_resolution=request.resolution or "",
                    requested_size=request.size or "",
                    codex_size=image_size,
                )
                if not probe_ok:
                    last_probe_error = probe_error
                    excluded_tokens.add(active_token)
                    token, selected_plan_type, plan_error = select_token(excluded_tokens)
                    if token:
                        continue
                    break
                for output in stream_codex_image_outputs(backend, request, image_size, index, total):
                    returned_result = returned_result or output.kind == "result"
                    yield output
                account_service.mark_image_result(active_token, returned_result)
                return
            except Exception as exc:
                if is_rate_limit_error(exc):
                    last_rate_limit_error = str(exc)
                    account_service.mark_image_rate_limited(
                        active_token,
                        error=last_rate_limit_error,
                        headers=getattr(exc, "headers", None),
                        body=getattr(exc, "body", None),
                    )
                    excluded_tokens.add(active_token)
                    logger.warning({
                        "event": "codex_image_account_rate_limited",
                        "model": request.model,
                        "requested_resolution": request.resolution or "",
                        "requested_size": request.size or "",
                        "codex_size": image_size,
                        "token": anonymize_token(active_token),
                        "error": last_rate_limit_error,
                    })
                    token, selected_plan_type, plan_error = select_token(excluded_tokens)
                    if token:
                        continue
                    raise RuntimeError(
                        last_rate_limit_error or last_probe_error or plan_error or "no available codex image quota"
                    ) from exc
                mark_image_failure(active_token, exc)
                raise
        raise RuntimeError(last_rate_limit_error or last_probe_error or plan_error or "no available codex image quota")

    return True, iterator()


def stream_image_outputs_with_pool(request: ConversationRequest) -> Iterator[ImageOutput]:
    if not is_supported_image_model(request.model):
        raise ImageGenerationError("unsupported image model,supported models: " + ", ".join(sorted(IMAGE_MODELS)))

    emitted = False
    last_error = ""
    normalized_resolution = normalize_image_resolution(request.resolution)
    high_resolution_requested = normalized_resolution in {"2k", "4k"}
    for index in range(1, request.n + 1):
        used_codex, codex_outputs = try_stream_codex_image_outputs_with_pool(request, index, request.n)
        if codex_outputs is not None:
            try:
                for output in codex_outputs:
                    emitted = True
                    yield output
                if used_codex:
                    continue
            except Exception as exc:
                last_error = str(exc)
                logger.warning({
                    "event": "codex_image_stream_fail",
                    "model": request.model,
                    "requested_resolution": request.resolution or "",
                    "error": last_error,
                })
                if used_codex and high_resolution_requested:
                    message = image_stream_error_message(last_error)
                    logger.warning({
                        "event": "codex_image_high_resolution_failed_no_fallback",
                        "model": request.model,
                        "requested_resolution": request.resolution or "",
                        "requested_size": request.size or "",
                        "error": last_error,
                    })
                    raise ImageGenerationError(
                        f"{str(normalized_resolution).upper()} 高清生成失败，未降级为普通清晰度。原因：{message}",
                        code="high_resolution_generation_failed",
                    ) from exc
        attempted_image_tokens: set[str] = set()
        while True:
            try:
                codex_model = is_codex_image_model(request.model)
                plan_error = ""
                selected_plan_type: str | None = None
                candidates = image_plan_candidates(request)
                logger.info({
                    "event": "image_account_select_start",
                    "model": request.model,
                    "requested_resolution": request.resolution or "",
                    "requested_size": request.size or "",
                    "candidate_plan_types": [plan_type or "auto" for plan_type in candidates],
                    "source_type": "codex" if codex_model else "web",
                })
                for plan_type in candidates:
                    try:
                        token = account_service.get_available_access_token(
                            plan_type=plan_type,
                            source_type="codex" if codex_model else "web",
                            plan_types={"Plus", "Team", "Pro"} if codex_model and not plan_type else None,
                            excluded_tokens=attempted_image_tokens,
                        )
                        selected_plan_type = plan_type
                        break
                    except RuntimeError as exc:
                        plan_error = str(exc)
                        logger.info({
                            "event": "image_account_select_plan_unavailable",
                            "model": request.model,
                            "requested_resolution": request.resolution or "",
                            "plan_type": plan_type or "auto",
                            "error": plan_error,
                        })
                        token = ""
                        continue
                if not token:
                    raise RuntimeError(last_error or plan_error or "no available image quota")
                attempted_image_tokens.add(token)
                selected_account = account_service.get_account(token) or {}
                logger.info({
                    "event": "image_account_selected",
                    "model": request.model,
                    "requested_resolution": request.resolution or "",
                    "requested_size": request.size or "",
                    "selected_plan_type": selected_plan_type or "auto",
                    "account_type": selected_account.get("type") or "",
                    "account_status": selected_account.get("status") or "",
                    "account_source_type": selected_account.get("source_type") or "web",
                    "account_quota": selected_account.get("quota"),
                    "account_image_quota_unknown": bool(selected_account.get("image_quota_unknown")),
                    "token": anonymize_token(token),
                })
            except RuntimeError as exc:
                if emitted:
                    return
                raise ImageGenerationError(str(exc) or "image generation failed") from exc

            emitted_for_token = False
            returned_message = False
            returned_result = False
            try:
                backend = OpenAIBackendAPI(access_token=token)
                probe_ok, probe_error = _probe_image_account_models(
                    backend,
                    token,
                    source_type="codex" if codex_model else "web",
                    model=request.model,
                    requested_resolution=request.resolution or "",
                    requested_size=request.size or "",
                )
                if not probe_ok:
                    last_error = probe_error
                    continue
                for output in stream_image_outputs(backend, request, index, request.n):
                    if output.kind == "message" and request.message_as_error:
                        raise ImageGenerationError(
                            output.text or "Image generation was rejected by upstream policy.",
                            status_code=400,
                            error_type="invalid_request_error",
                            code="content_policy_violation",
                        )
                    emitted = True
                    emitted_for_token = True
                    returned_message = output.kind == "message"
                    returned_result = returned_result or output.kind == "result"
                    yield output
                if returned_message or not returned_result:
                    account_service.mark_image_result(token, False)
                    return
                account_service.mark_image_result(token, True)
                break
            except ImageGenerationError as exc:
                mark_image_failure(token, exc)
                raise
            except Exception as exc:
                last_error = str(exc)
                logger.warning({"event": "image_stream_fail", "request_token": anonymize_token(token), "error": last_error})
                if not emitted_for_token and is_token_invalid_error(last_error):
                    account_service.mark_image_result(token, False)
                    account_service.remove_invalid_token(token, "image_stream")
                    continue
                if is_rate_limit_error(exc):
                    account_service.mark_image_rate_limited(
                        token,
                        error=last_error,
                        headers=getattr(exc, "headers", None),
                        body=getattr(exc, "body", None),
                    )
                    logger.warning({
                        "event": "image_account_rate_limited",
                        "model": request.model,
                        "requested_resolution": request.resolution or "",
                        "requested_size": request.size or "",
                        "token": anonymize_token(token),
                        "error": last_error,
                    })
                    continue
                account_service.mark_image_result(token, False)
                raise ImageGenerationError(image_stream_error_message(last_error)) from exc

    if not emitted:
        raise ImageGenerationError(image_stream_error_message(last_error))


def stream_image_chunks(outputs: Iterable[ImageOutput]) -> Iterator[dict[str, Any]]:
    for output in outputs:
        yield output.to_chunk()


def stream_chat_events(
    request: ConversationRequest,
    *,
    preferred_token: str = "",
    excluded_tokens: set[str] | None = None,
    plan_type: str | None = None,
    plan_types: set[str] | tuple[str, ...] | None = None,
) -> Iterator[dict[str, Any]]:
    """/api/chat/stream 专用通道：history_and_training_disabled=False，
    暴露 conversation_id 供调用方做异步 DELETE。失效 token 单次轮换。
    preferred_token 用于续聊场景'粘住'原账号；excluded_tokens 用于'手动换号'
    排除旧号；上游每轮都开新 cid，靠完整历史维持上下文，避免和 done 后的 DELETE
    自相矛盾。"""
    attempted: set[str] = set(excluded_tokens or ())
    token = ""
    if preferred_token and preferred_token not in attempted:
        account = account_service.get_account(preferred_token)
        if (
                account
                and str(account.get("status") or "") not in {"禁用", "异常"}
                and account_service._account_matches_plan_type(account, plan_type)
                and account_service._account_matches_any_plan_type(account, plan_types)
        ):
            token = preferred_token
    if not token:
        token = account_service.get_text_access_token(attempted, plan_type=plan_type, plan_types=plan_types)
    if plan_type and not token:
        raise RuntimeError(f"no available {plan_type} text account")
    emitted = False
    while True:
        if token and token in attempted:
            raise RuntimeError("no available text account")
        if token:
            attempted.add(token)
        backend = OpenAIBackendAPI(access_token=token)
        normalized = normalize_messages(request.messages or ([{"role": "user", "content": request.prompt}] if request.prompt else []))
        history_text = assistant_history_text(normalized)
        history_messages = assistant_history_messages(normalized)
        try:
            payloads = backend.stream_conversation(
                messages=normalized,
                model=request.model,
                prompt=request.prompt,
                history_and_training_disabled=False,
            )
            for event in iter_conversation_payloads(payloads, history_text, history_messages):
                emitted = True
                event["account_token"] = token
                yield event
            account_service.mark_text_used(token)
            return
        except Exception as exc:
            error_message = str(exc)
            if token and not emitted and is_token_invalid_error(error_message):
                account_service.remove_invalid_token(token, "chat_stream")
                token = account_service.get_text_access_token(attempted, plan_type=plan_type, plan_types=plan_types)
                if plan_type and not token:
                    raise RuntimeError(f"no available {plan_type} text account")
                if token:
                    continue
            raise


def delete_conversation_safely(token: str, conversation_id: str) -> None:
    """异步 DELETE 用：失败吞掉。"""
    if not token or not conversation_id:
        return
    try:
        OpenAIBackendAPI(access_token=token).delete_conversation(conversation_id)
    except Exception:
        pass




def collect_image_outputs(outputs: Iterable[ImageOutput]) -> dict[str, Any]:
    created = None
    data: list[dict[str, Any]] = []
    message = ""
    progress_parts: list[str] = []
    for output in outputs:
        created = created or output.created
        if output.kind == "progress" and output.text:
            progress_parts.append(output.text)
        elif output.kind == "message":
            message = output.text
        elif output.kind == "result":
            data.extend(output.data)

    result: dict[str, Any] = {"created": created or int(time.time()), "data": data}
    if not data:
        text = message or "".join(progress_parts).strip()
        if text:
            result["message"] = text
    return result
