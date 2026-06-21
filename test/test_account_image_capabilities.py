from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("LY2CHATGPT2API_AUTH_KEY", "test-auth")

from services.account_service import AccountService
from services.auth_service import AuthService
from services.cpa_service import _account_payload_from_auth_file
from services.sub2api_service import _account_payload_from_remote
from services.storage.json_storage import JSONStorageBackend
from utils.helper import anonymize_token


class AccountCapabilityTests(unittest.TestCase):
    def test_unknown_quota_accounts_are_available_only_when_not_throttled(self) -> None:
        self.assertFalse(
            AccountService._is_image_account_available(
                {"status": "限流", "image_quota_unknown": True, "quota": 0}
            )
        )
        self.assertTrue(
            AccountService._is_image_account_available(
                {"status": "正常", "image_quota_unknown": True, "quota": 0}
            )
        )

    def test_runtime_rate_limit_blocks_until_reset_time(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        past = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()

        self.assertFalse(
            AccountService._is_image_account_available(
                {"status": "限流", "image_quota_unknown": True, "quota": 0, "rate_limit_reset_at": future}
            )
        )
        self.assertTrue(
            AccountService._is_image_account_available(
                {"status": "限流", "image_quota_unknown": True, "quota": 0, "rate_limit_reset_at": past}
            )
        )

    def test_prolite_variants_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            self.assertEqual(service._normalize_account_type("prolite"), "ProLite")
            self.assertEqual(service._normalize_account_type("pro_lite"), "ProLite")

    def test_search_account_type_ignores_unrelated_scalar_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            self.assertIsNone(
                service._search_account_type(
                    {
                        "amr": ["pwd", "otp", "mfa"],
                        "chatgpt_compute_residency": "no_constraint",
                        "chatgpt_data_residency": "no_constraint",
                        "user_id": "user-I52GFfLGFM0dokFk2dBiKEBn",
                    }
                )
            )

    def test_add_account_items_preserves_codex_source_and_plan_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))

            result = service.add_account_items([
                {"accessToken": "token-1", "type": "codex", "plan_type": "plus"}
            ])

            self.assertEqual(result["added"], 1)
            account = service.get_account("token-1")
            self.assertIsNotNone(account)
            self.assertEqual(account["source_type"], "codex")
            self.assertEqual(account["export_type"], "codex")
            self.assertEqual(account["type"], "Plus")

    def test_add_accounts_uses_account_records_payload_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))

            result = service.add_accounts(
                ["token-1"],
                [{"accessToken": "token-1", "source_type": "codex", "plan_type": "team"}],
            )

            self.assertEqual(result["added"], 1)
            account = service.get_account("token-1")
            self.assertIsNotNone(account)
            self.assertEqual(account["source_type"], "codex")
            self.assertEqual(account["type"], "Team")

    def test_oauth_payload_with_refresh_token_is_codex_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))

            result = service.add_account_items([{
                "access_token": "token-1",
                "refresh_token": "rt-1",
                "id_token": "id-1",
                "account_id": "acc-1",
                "plan_type": "plus",
            }])

            self.assertEqual(result["added"], 1)
            account = service.get_account("token-1")
            self.assertIsNotNone(account)
            self.assertEqual(account["source_type"], "codex")
            self.assertEqual(account["refresh_token"], "rt-1")
            self.assertEqual(account["id_token"], "id-1")
            self.assertEqual(account["account_id"], "acc-1")
            self.assertEqual(account["type"], "Plus")

    def test_cpa_auth_file_payload_preserves_codex_oauth_fields(self) -> None:
        payload = _account_payload_from_auth_file(
            {
                "type": "codex",
                "access_token": "token-1",
                "refresh_token": "rt-1",
                "id_token": "id-1",
                "account_id": "acc-1",
                "email": "a@example.com",
            },
            "codex-account.json",
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["access_token"], "token-1")
        self.assertEqual(payload["refresh_token"], "rt-1")
        self.assertEqual(payload["id_token"], "id-1")
        self.assertEqual(payload["account_id"], "acc-1")
        self.assertEqual(payload["source_type"], "codex")
        self.assertEqual(payload["cpa_file_name"], "codex-account.json")

    def test_sub2api_account_payload_preserves_credentials(self) -> None:
        payload = _account_payload_from_remote(
            {
                "id": 42,
                "name": "a@example.com",
                "platform": "openai",
                "type": "oauth",
                "credentials": {
                    "access_token": "token-1",
                    "refresh_token": "rt-1",
                    "id_token": "id-1",
                    "chatgpt_account_id": "acc-1",
                    "plan_type": "team",
                },
            },
            "42",
        )

        self.assertEqual(payload["access_token"], "token-1")
        self.assertEqual(payload["refresh_token"], "rt-1")
        self.assertEqual(payload["id_token"], "id-1")
        self.assertEqual(payload["chatgpt_account_id"], "acc-1")
        self.assertEqual(payload["source_type"], "codex")
        self.assertEqual(payload["plan_type"], "team")
        self.assertEqual(payload["sub2api_account_id"], "42")

    def test_mark_image_result_does_not_consume_unknown_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_accounts(["token-1"])
            service.update_account(
                "token-1",
                {
                    "status": "正常",
                    "quota": 0,
                    "image_quota_unknown": True,
                },
            )

            updated = service.mark_image_result("token-1", success=True)

            self.assertIsNotNone(updated)
            self.assertEqual(updated["quota"], 0)
            self.assertEqual(updated["status"], "正常")
            self.assertTrue(updated["image_quota_unknown"])

    def test_mark_image_rate_limited_uses_codex_reset_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_accounts(["token-1"])
            service.update_account(
                "token-1",
                {
                    "status": "正常",
                    "quota": 0,
                    "image_quota_unknown": True,
                },
            )

            before = datetime.now(timezone.utc)
            updated = service.mark_image_rate_limited(
                "token-1",
                error="status=429",
                headers={
                    "x-codex-primary-used-percent": "100",
                    "x-codex-primary-reset-after-seconds": "60",
                    "x-codex-primary-window-minutes": "300",
                    "x-codex-secondary-used-percent": "10",
                    "x-codex-secondary-reset-after-seconds": "3600",
                    "x-codex-secondary-window-minutes": "10080",
                },
            )
            after = datetime.now(timezone.utc)

            self.assertIsNotNone(updated)
            self.assertEqual(updated["status"], "限流")
            self.assertEqual(updated["fail"], 1)
            reset_at = AccountService._parse_datetime(updated["rate_limit_reset_at"])
            self.assertIsNotNone(reset_at)
            self.assertGreaterEqual(reset_at, before + timedelta(seconds=60))
            self.assertLessEqual(reset_at, after + timedelta(seconds=60))
            self.assertFalse(AccountService._is_image_account_available(updated))


class TokenLogTests(unittest.TestCase):
    def test_anonymize_token_hides_raw_value(self) -> None:
        token = "super-secret-token"
        token_ref = anonymize_token(token)

        self.assertTrue(token_ref.startswith("token:"))
        self.assertNotIn(token, token_ref)


class AuthServiceTests(unittest.TestCase):
    def test_create_authenticate_disable_and_delete_user_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))

            item, raw_key = service.create_key(role="user", name="Alice")

            self.assertEqual(item["role"], "user")
            self.assertEqual(item["name"], "Alice")
            self.assertEqual(item["account_tier"], "free")
            self.assertFalse(item["can_use_high_resolution"])
            self.assertTrue(item["enabled"])
            self.assertTrue(raw_key.startswith("sk-"))

            authed = service.authenticate(raw_key)
            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])
            self.assertEqual(authed["role"], "user")
            self.assertEqual(authed["account_tier"], "free")
            self.assertIsNotNone(authed["last_used_at"])

            updated = service.update_key(item["id"], {"enabled": False}, role="user")
            self.assertIsNotNone(updated)
            self.assertFalse(updated["enabled"])
            self.assertIsNone(service.authenticate(raw_key))

            self.assertTrue(service.delete_key(item["id"], role="user"))
            self.assertFalse(service.delete_key(item["id"], role="user"))
            self.assertEqual(service.list_keys(role="user"), [])

    def test_authenticate_ignores_last_used_save_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            item, raw_key = service.create_key(role="user", name="Alice")

            def fail_save() -> None:
                raise OSError("disk unavailable")

            service._save = fail_save

            authed = service.authenticate(raw_key)

            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])

    def test_user_key_account_tier_controls_public_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))

            item, raw_key = service.create_key(role="user", name="Alice", account_tier="premium")

            self.assertEqual(item["account_tier"], "premium")
            self.assertTrue(item["can_use_paid_image_accounts"])
            self.assertTrue(item["can_use_high_resolution"])

            authed = service.authenticate(raw_key)
            self.assertIsNotNone(authed)
            self.assertEqual(authed["account_tier"], "premium")
            self.assertTrue(authed["can_use_high_resolution"])

            updated = service.update_key(item["id"], {"account_tier": "free"}, role="user")
            self.assertIsNotNone(updated)
            self.assertEqual(updated["account_tier"], "free")
            self.assertFalse(updated["can_use_high_resolution"])
            self.assertIsNotNone(authed["last_used_at"])

    def test_update_user_key_replaces_raw_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            item, raw_key = service.create_key(role="user", name="Alice")

            updated = service.update_key(item["id"], {"key": "sk-user-custom-key"}, role="user")

            self.assertIsNotNone(updated)
            self.assertIsNone(service.authenticate(raw_key))

            authed = service.authenticate("sk-user-custom-key")
            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])

    def test_user_key_name_must_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            first, _ = service.create_key(role="user", name="Alice")
            second, _ = service.create_key(role="user", name="Bob")

            with self.assertRaisesRegex(ValueError, "这个名称已经在使用中了"):
                service.create_key(role="user", name="Alice")

            with self.assertRaisesRegex(ValueError, "这个名称已经在使用中了"):
                service.update_key(second["id"], {"name": "Alice"}, role="user")

            updated = service.update_key(first["id"], {"name": "Alice"}, role="user")
            self.assertIsNotNone(updated)
            self.assertEqual(updated["name"], "Alice")

    def test_user_key_quota_hierarchy_is_validated_on_create(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))

            with self.assertRaisesRegex(ValueError, "画图月限额不能大于画图总额度"):
                service.create_key(
                    role="user",
                    name="Alice",
                    image_monthly_quota=200,
                    image_monthly_unlimited=False,
                    image_total_quota=12,
                    image_total_unlimited=False,
                )

    def test_user_key_quota_hierarchy_is_validated_on_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            item, _ = service.create_key(
                role="user",
                name="Alice",
                image_monthly_quota=10,
                image_monthly_unlimited=False,
                image_total_quota=100,
                image_total_unlimited=False,
            )

            with self.assertRaisesRegex(ValueError, "画图月限额不能大于画图总额度"):
                service.update_key(item["id"], {"image_monthly_quota": 200}, role="user")


if __name__ == "__main__":
    unittest.main()
