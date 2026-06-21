from __future__ import annotations

import base64
import unittest
from io import BytesIO
from unittest import mock

from PIL import Image

from services.protocol import conversation
from services.protocol.conversation import ConversationRequest, collect_image_outputs, stream_image_outputs_with_pool
from utils.helper import UpstreamHTTPError


def tiny_png_b64() -> str:
    buffer = BytesIO()
    Image.new("RGB", (16, 9), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeProbeAccountService:
    def __init__(
            self,
            calls,
            tokens: list[str],
            *,
            source_type: str,
            account_type: str,
            image_quota_unknown: bool,
    ):
        self.calls = calls
        self.tokens = tokens
        self.source_type = source_type
        self.account_type = account_type
        self.image_quota_unknown = image_quota_unknown

    def get_available_access_token(self, **kwargs):
        self.calls.append(("token", kwargs))
        excluded = kwargs.get("excluded_tokens") or set()
        for token in self.tokens:
            if token not in excluded:
                return token
        raise RuntimeError(f"no available {self.source_type} image quota")

    def get_account(self, token):
        return {
            "access_token": token,
            "type": self.account_type,
            "status": "正常",
            "source_type": self.source_type,
            "quota": 1,
            "image_quota_unknown": self.image_quota_unknown,
        }

    def refresh_oauth_access_token(self, token):
        return ""

    def release_image_slot(self, token):
        self.calls.append(("release", {"token": token}))

    def mark_image_result(self, token, success):
        self.calls.append(("mark", {"token": token, "success": success}))

    def mark_image_rate_limited(self, token, **kwargs):
        self.calls.append(("limited", {"token": token, "error": kwargs.get("error", "")}))

    def remove_invalid_token(self, token, event):
        self.calls.append(("remove", {"token": token, "event": event}))


def fake_result_stream(calls):
    def fake_stream_image_outputs(backend, request, index=1, total=1):
        calls.append(("generate", {"token": backend.access_token}))
        yield conversation.ImageOutput(
            kind="result",
            model=request.model,
            index=index,
            total=total,
            data=[{"b64_json": tiny_png_b64()}],
        )

    return fake_stream_image_outputs


class CodexImageRouteTests(unittest.TestCase):
    def test_high_resolution_uses_codex_responses_size(self):
        calls = []

        class FakeAccountService:
            def get_available_access_token(self, **kwargs):
                calls.append(("token", kwargs))
                return "codex-token"

            def get_account(self, token):
                return {
                    "access_token": token,
                    "type": "Plus",
                    "status": "正常",
                    "source_type": "codex",
                    "quota": 1,
                    "image_quota_unknown": True,
                }

            def refresh_oauth_access_token(self, token):
                calls.append(("refresh", {"token": token}))
                return ""

            def mark_image_result(self, token, success):
                calls.append(("mark", {"token": token, "success": success}))

        class FakeBackend:
            def __init__(self, access_token):
                self.access_token = access_token

            def list_models(self):
                return {"object": "list", "data": []}

            def generate_codex_image(self, **kwargs):
                calls.append(("generate", kwargs))
                return [{"type": "image_generation_call", "result": tiny_png_b64()}]

        with (
            mock.patch.object(conversation, "account_service", FakeAccountService()),
            mock.patch.object(conversation, "OpenAIBackendAPI", FakeBackend),
        ):
            result = collect_image_outputs(stream_image_outputs_with_pool(
                ConversationRequest(
                    model="gpt-image-2",
                    prompt="cat",
                    resolution="4k",
                    size="16:9",
                    response_format="b64_json",
                )
            ))

        generate_call = next(payload for kind, payload in calls if kind == "generate")
        self.assertEqual(generate_call["image_size"], "3840x2160")
        self.assertEqual(generate_call["model"], "gpt-image-2")
        self.assertEqual(result["data"][0]["b64_json"], tiny_png_b64())
        self.assertIn(("mark", {"token": "codex-token", "success": True}), calls)

    def test_high_resolution_codex_failure_does_not_fallback_to_picture_v2(self):
        calls = []

        class FakeAccountService:
            def get_available_access_token(self, **kwargs):
                calls.append(("token", kwargs))
                return "codex-token"

            def get_account(self, token):
                return {
                    "access_token": token,
                    "type": "Plus",
                    "status": "正常",
                    "source_type": "codex",
                    "quota": 1,
                    "image_quota_unknown": True,
                }

            def refresh_oauth_access_token(self, token):
                return ""

            def mark_image_result(self, token, success):
                calls.append(("mark", {"token": token, "success": success}))

        class FakeBackend:
            def __init__(self, access_token):
                self.access_token = access_token

            def list_models(self):
                return {"object": "list", "data": []}

            def generate_codex_image(self, **kwargs):
                calls.append(("generate", kwargs))
                raise RuntimeError("codex upstream rejected size")

        with (
            mock.patch.object(conversation, "account_service", FakeAccountService()),
            mock.patch.object(conversation, "OpenAIBackendAPI", FakeBackend),
        ):
            with self.assertRaises(conversation.ImageGenerationError) as context:
                collect_image_outputs(stream_image_outputs_with_pool(
                    ConversationRequest(
                        model="gpt-image-2",
                        prompt="cat",
                        resolution="4k",
                        size="9:16",
                        response_format="b64_json",
                    )
                ))

        self.assertIn("4K 高清生成失败", str(context.exception))
        generate_call = next(payload for kind, payload in calls if kind == "generate")
        self.assertEqual(generate_call["image_size"], "2160x3840")
        self.assertIn(("mark", {"token": "codex-token", "success": False}), calls)

    def test_high_resolution_429_marks_account_and_retries_next_codex_account(self):
        calls = []

        class FakeAccountService:
            def get_available_access_token(self, **kwargs):
                calls.append(("token", kwargs))
                excluded = kwargs.get("excluded_tokens") or set()
                if "codex-token-1" not in excluded:
                    return "codex-token-1"
                if "codex-token-2" not in excluded:
                    return "codex-token-2"
                raise RuntimeError("no available codex image quota")

            def get_account(self, token):
                return {
                    "access_token": token,
                    "type": "Plus",
                    "status": "正常",
                    "source_type": "codex",
                    "quota": 1,
                    "image_quota_unknown": True,
                }

            def refresh_oauth_access_token(self, token):
                return ""

            def mark_image_rate_limited(self, token, **kwargs):
                calls.append(("limited", {"token": token, "error": kwargs.get("error", "")}))

            def mark_image_result(self, token, success):
                calls.append(("mark", {"token": token, "success": success}))

        class FakeBackend:
            def __init__(self, access_token):
                self.access_token = access_token

            def list_models(self):
                return {"object": "list", "data": []}

            def generate_codex_image(self, **kwargs):
                calls.append(("generate", {"token": self.access_token, **kwargs}))
                if self.access_token == "codex-token-1":
                    raise UpstreamHTTPError(
                        "/backend-api/codex/responses",
                        429,
                        {"error": {"type": "rate_limit_exceeded"}},
                        {"x-codex-primary-used-percent": "100", "x-codex-primary-reset-after-seconds": "60", "x-codex-primary-window-minutes": "300"},
                    )
                return [{"type": "image_generation_call", "result": tiny_png_b64()}]

        with (
            mock.patch.object(conversation, "account_service", FakeAccountService()),
            mock.patch.object(conversation, "OpenAIBackendAPI", FakeBackend),
        ):
            result = collect_image_outputs(stream_image_outputs_with_pool(
                ConversationRequest(
                    model="gpt-image-2",
                    prompt="cat",
                    resolution="4k",
                    size="9:16",
                    response_format="b64_json",
                )
            ))

        generated_tokens = [payload["token"] for kind, payload in calls if kind == "generate"]
        self.assertEqual(generated_tokens, ["codex-token-1", "codex-token-2"])
        self.assertIn(("limited", {"token": "codex-token-1", "error": "/backend-api/codex/responses failed: status=429, body={'error': {'type': 'rate_limit_exceeded'}}"}), calls)
        self.assertIn(("mark", {"token": "codex-token-2", "success": True}), calls)
        self.assertEqual(result["data"][0]["b64_json"], tiny_png_b64())

    def test_web_model_probe_failure_skips_to_next_account(self):
        calls = []
        account_service = FakeProbeAccountService(
            calls,
            ["web-token-1", "web-token-2"],
            source_type="web",
            account_type="free",
            image_quota_unknown=False,
        )

        class FakeBackend:
            def __init__(self, access_token):
                self.access_token = access_token

            def list_models(self):
                calls.append(("probe", {"token": self.access_token}))
                if self.access_token == "web-token-1":
                    raise RuntimeError("temporary model probe failure")
                return {"object": "list", "data": []}

        with (
            mock.patch.object(conversation, "account_service", account_service),
            mock.patch.object(conversation, "OpenAIBackendAPI", FakeBackend),
            mock.patch.object(conversation, "stream_image_outputs", fake_result_stream(calls)),
        ):
            result = collect_image_outputs(stream_image_outputs_with_pool(
                ConversationRequest(
                    model="gpt-image-2",
                    prompt="cat",
                    response_format="b64_json",
                )
            ))

        probed_tokens = [payload["token"] for kind, payload in calls if kind == "probe"]
        generated_tokens = [payload["token"] for kind, payload in calls if kind == "generate"]
        self.assertEqual(probed_tokens, ["web-token-1", "web-token-2"])
        self.assertEqual(generated_tokens, ["web-token-2"])
        self.assertIn(("release", {"token": "web-token-1"}), calls)
        self.assertIn(("mark", {"token": "web-token-2", "success": True}), calls)
        self.assertEqual(result["data"][0]["b64_json"], tiny_png_b64())

    def test_high_resolution_model_probe_failure_retries_next_codex_account(self):
        calls = []
        account_service = FakeProbeAccountService(
            calls,
            ["codex-token-1", "codex-token-2"],
            source_type="codex",
            account_type="Plus",
            image_quota_unknown=True,
        )

        class FakeBackend:
            def __init__(self, access_token):
                self.access_token = access_token

            def list_models(self):
                calls.append(("probe", {"token": self.access_token}))
                if self.access_token == "codex-token-1":
                    raise RuntimeError("temporary model probe failure")
                return {"object": "list", "data": []}

            def generate_codex_image(self, **kwargs):
                calls.append(("generate", {"token": self.access_token, **kwargs}))
                return [{"type": "image_generation_call", "result": tiny_png_b64()}]

        with (
            mock.patch.object(conversation, "account_service", account_service),
            mock.patch.object(conversation, "OpenAIBackendAPI", FakeBackend),
        ):
            result = collect_image_outputs(stream_image_outputs_with_pool(
                ConversationRequest(
                    model="gpt-image-2",
                    prompt="cat",
                    resolution="4k",
                    size="16:9",
                    response_format="b64_json",
                )
            ))

        probed_tokens = [payload["token"] for kind, payload in calls if kind == "probe"]
        generated_tokens = [payload["token"] for kind, payload in calls if kind == "generate"]
        self.assertEqual(probed_tokens, ["codex-token-1", "codex-token-2"])
        self.assertEqual(generated_tokens, ["codex-token-2"])
        self.assertIn(("release", {"token": "codex-token-1"}), calls)
        self.assertIn(("mark", {"token": "codex-token-2", "success": True}), calls)
        self.assertEqual(result["data"][0]["b64_json"], tiny_png_b64())

    def test_model_probe_429_marks_account_and_retries_next_web_account(self):
        calls = []
        account_service = FakeProbeAccountService(
            calls,
            ["web-token-1", "web-token-2"],
            source_type="web",
            account_type="free",
            image_quota_unknown=False,
        )

        class FakeBackend:
            def __init__(self, access_token):
                self.access_token = access_token

            def list_models(self):
                calls.append(("probe", {"token": self.access_token}))
                if self.access_token == "web-token-1":
                    raise UpstreamHTTPError(
                        "/backend-api/models",
                        429,
                        {"error": {"type": "rate_limit_exceeded"}},
                    )
                return {"object": "list", "data": []}

        with (
            mock.patch.object(conversation, "account_service", account_service),
            mock.patch.object(conversation, "OpenAIBackendAPI", FakeBackend),
            mock.patch.object(conversation, "stream_image_outputs", fake_result_stream(calls)),
        ):
            result = collect_image_outputs(stream_image_outputs_with_pool(
                ConversationRequest(
                    model="gpt-image-2",
                    prompt="cat",
                    response_format="b64_json",
                )
            ))

        generated_tokens = [payload["token"] for kind, payload in calls if kind == "generate"]
        self.assertEqual(generated_tokens, ["web-token-2"])
        self.assertIn(("limited", {"token": "web-token-1", "error": "/backend-api/models failed: status=429, body={'error': {'type': 'rate_limit_exceeded'}}"}), calls)
        self.assertNotIn(("release", {"token": "web-token-1"}), calls)
        self.assertIn(("mark", {"token": "web-token-2", "success": True}), calls)
        self.assertEqual(result["data"][0]["b64_json"], tiny_png_b64())


if __name__ == "__main__":
    unittest.main()
