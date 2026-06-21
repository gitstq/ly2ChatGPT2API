from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from api.support import can_use_paid_image_accounts, require_admin, require_identity, resolve_image_base_url
from services.auth_service import auth_service
from services.backup_service import BackupError, backup_service
from services.config import config
from services.image_owners_service import get_owner, owner_counts
from services.image_service import count_total_images, delete_images, download_images_zip, get_image_download_response, get_thumbnail_response, list_images
from services.image_tags_service import delete_tag, get_all_tags, set_tags
from services.log_service import log_service
from services.proxy_service import test_proxy


def _admin_owner_ids() -> set[str]:
    """收集所有可能落在 image_owners.json 里的 admin id：
    - "admin"：旧 auth_key（LY2CHATGPT2API_AUTH_KEY / config.json.auth-key）的固定 id
    - 其余：通过 auth_service 创建的 admin 角色密钥
    用来把"管理员生成"和"真孤儿"两个桶区分开，别再混在一起。
    """
    ids: set[str] = {"admin"}
    for item in auth_service.list_keys(role="admin"):
        uid = str(item.get("id") or "").strip()
        if uid:
            ids.add(uid)
    return ids


class SettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class ProxyTestRequest(BaseModel):
    url: str = ""


class ImageDeleteRequest(BaseModel):
    paths: list[str] = []
    start_date: str = ""
    end_date: str = ""
    owner: str = ""
    all_matching: bool = False

class ImageDownloadRequest(BaseModel):
    paths: list[str]

class ImageTagsRequest(BaseModel):
    path: str
    tags: list[str]

class LogDeleteRequest(BaseModel):
    ids: list[str] = []
class BackupDeleteRequest(BaseModel):
    key: str = ""


def create_router(app_version: str) -> APIRouter:
    router = APIRouter()

    @router.post("/auth/login")
    async def login(authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        return {
            "ok": True,
            "version": app_version,
            "role": identity.get("role"),
            "subject_id": identity.get("id"),
            "name": identity.get("name"),
            "account_tier": "premium" if can_use_paid_image_accounts(identity) else "free",
            "can_use_paid_image_accounts": can_use_paid_image_accounts(identity),
            "can_use_high_resolution": can_use_paid_image_accounts(identity),
        }

    @router.get("/version")
    async def get_version():
        return {"version": app_version}

    @router.get("/api/settings")
    async def get_settings(authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return {"config": config.get()}

    @router.post("/api/settings")
    async def save_settings(body: SettingsUpdateRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return {"config": config.update(body.model_dump(mode="python"))}

    @router.get("/api/images")
    async def get_images(
        request: Request,
        start_date: str = "",
        end_date: str = "",
        owner: str = "",
        authorization: str | None = Header(default=None),
    ):
        require_admin(authorization)
        return list_images(
            resolve_image_base_url(request),
            start_date=start_date.strip(),
            end_date=end_date.strip(),
            owner=owner.strip(),
            admin_ids=_admin_owner_ids(),
        )

    @router.get("/api/me/images")
    async def get_my_images(
        request: Request,
        start_date: str = "",
        end_date: str = "",
        authorization: str | None = Header(default=None),
    ):
        """登录用户视角的"我的图片"。

        鉴权用 require_identity，普通 user 密钥也能调；按 identity.id 自动过滤
        image_owners.json 里挂在自己名下的图。Admin 调时退化为 owner=__admin__,
        把所有 admin 生成的图聚合返回（语义上"我"= 管理员这个角色）。

        - Android / 未来其他客户端启动时 fetch 这个端点把云端历史合并进本地 Room
        - 不开放 owner 参数，避免用户冒名查别人的图
        """
        identity = require_identity(authorization)
        admin_ids = _admin_owner_ids()
        role = str(identity.get("role") or "").strip()
        identity_id = str(identity.get("id") or "").strip()
        if role == "admin" or identity_id in admin_ids:
            owner_filter = "__admin__"
        else:
            owner_filter = identity_id
        return list_images(
            resolve_image_base_url(request),
            start_date=start_date.strip(),
            end_date=end_date.strip(),
            owner=owner_filter,
            admin_ids=admin_ids,
        )

    @router.get("/api/images/owners")
    async def get_image_owners(authorization: str | None = Header(default=None)):
        """图片管理页用户筛选下拉的数据源。
        三类语义，互不混淆：
        1. 普通用户：列出所有用户密钥（即便 count=0），admin 期望"我建过的密钥都能筛"
        2. 管理员（__admin__）：所有 admin 角色（含旧 auth_key 的 "admin" id）生成的图聚合
        3. 未归属（__unowned__）：image_owners.json 里没记录的真孤儿，多半是老数据
        孤儿 user id（用户密钥已被删但归属表还留着）单列出来，标记 deleted=true。
        """
        require_admin(authorization)
        counts = owner_counts()
        admin_ids = _admin_owner_ids()
        users = auth_service.list_keys(role="user")
        items: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for user in users:
            uid = str(user.get("id") or "").strip()
            if not uid:
                continue
            seen_ids.add(uid)
            items.append({
                "id": uid,
                "name": user.get("name") or uid,
                "deleted": False,
                "count": int(counts.get(uid, 0)),
            })
        # admin 集合本身已经独立成一桶，所以 seen_ids 里要带上 admin_ids 防止重复
        seen_ids |= admin_ids
        admin_count = sum(int(c) for k, c in counts.items() if k in admin_ids)
        for owner_id, count in counts.items():
            if not owner_id or owner_id in seen_ids:
                continue
            items.append({
                "id": owner_id,
                "name": owner_id,
                "deleted": True,
                "count": int(count),
            })
        items.sort(key=lambda x: (-int(x.get("count") or 0), str(x.get("name") or "")))
        # 真孤儿 = 总图片数 − 已挂归属的所有图（含 admin / 用户 / 已删用户）
        owned_total = sum(int(v) for v in counts.values())
        unowned_count = max(0, count_total_images() - owned_total)
        # 两个固定桶；前端会把它们置顶到列表最上方。
        items.append({"id": "__admin__", "name": "管理员", "deleted": False, "count": admin_count})
        items.append({"id": "__unowned__", "name": "未归属", "deleted": False, "count": unowned_count})
        return {"items": items}

    @router.get("/image-thumbnails/{image_path:path}", include_in_schema=False)
    async def get_image_thumbnail(image_path: str):
        return get_thumbnail_response(image_path)

    @router.post("/api/images/delete")
    async def delete_images_endpoint(body: ImageDeleteRequest, authorization: str | None = Header(default=None)):
        """图片删除：
          - admin：全权，可按路径 / 按 owner / all_matching 任意筛选删
          - user：只能按路径删自己的图（image_owners.json 里 owner == identity.id）
            其余筛选参数 (start_date / end_date / owner / all_matching) 一律忽略，
            避免误把 all_matching=true 当成"清空所有"操作。
        """
        identity = require_identity(authorization)
        role = str(identity.get("role") or "").lower()
        if role == "admin":
            return delete_images(
                body.paths,
                start_date=body.start_date.strip(),
                end_date=body.end_date.strip(),
                owner=body.owner.strip(),
                all_matching=body.all_matching,
                admin_ids=_admin_owner_ids(),
            )
        # 普通用户路径：只允许按 paths 删自己拥有的图
        user_id = str(identity.get("id") or "").strip()
        if not user_id:
            raise HTTPException(status_code=403, detail={"error": "无权删除"})
        requested = [p.strip().lstrip("/") for p in (body.paths or []) if p and p.strip()]
        # owner 校验：每条 path 都必须 owner == 自己；不是的直接丢弃
        # 这样客户端误传别人的图也只是不删，不会泄露归属
        owned = [rel for rel in requested if get_owner(rel) == user_id]
        if not owned:
            return {"removed": 0}
        return delete_images(owned)

    @router.post("/api/images/download")
    async def download_images_endpoint(body: ImageDownloadRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        buf = download_images_zip(body.paths)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="images.zip"'},
        )

    @router.get("/api/images/download/{image_path:path}")
    async def download_single_image_endpoint(image_path: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return get_image_download_response(image_path)

    @router.get("/api/logs")
    async def get_logs(type: str = "", start_date: str = "", end_date: str = "", authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return {"items": log_service.list(type=type.strip(), start_date=start_date.strip(), end_date=end_date.strip())}

    @router.post("/api/logs/delete")
    async def delete_logs(body: LogDeleteRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return log_service.delete(body.ids)

    @router.post("/api/proxy/test")
    async def test_proxy_endpoint(body: ProxyTestRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        candidate = (body.url or "").strip() or config.get_proxy_settings()
        if not candidate:
            raise HTTPException(status_code=400, detail={"error": "proxy url is required"})
        return {"result": await run_in_threadpool(test_proxy, candidate)}

    @router.get("/api/storage/info")
    async def get_storage_info(authorization: str | None = Header(default=None)):
        require_admin(authorization)
        storage = config.get_storage_backend()
        return {
            "backend": storage.get_backend_info(),
            "health": storage.health_check(),
        }

    @router.post("/api/backup/test")
    async def test_backup_connection(authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            return {"result": await run_in_threadpool(backup_service.test_connection)}
        except BackupError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    @router.get("/api/backups")
    async def get_backups(authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            return {
                "items": await run_in_threadpool(backup_service.list_backups),
                "state": backup_service.get_status(),
                "settings": backup_service.get_settings(),
            }
        except BackupError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    @router.post("/api/backups/run")
    async def run_backup_endpoint(authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            return {"result": await run_in_threadpool(backup_service.run_backup)}
        except BackupError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    @router.post("/api/backups/delete")
    async def delete_backup_endpoint(body: BackupDeleteRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            await run_in_threadpool(backup_service.delete_backup, body.key)
            return {"ok": True}
        except BackupError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    @router.get("/api/backups/detail")
    async def get_backup_detail(key: str = "", authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            return {"item": await run_in_threadpool(backup_service.get_backup_detail, key)}
        except BackupError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    @router.get("/api/backups/download")
    async def download_backup_endpoint(key: str = "", authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            item = await run_in_threadpool(backup_service.download_backup, key)
        except BackupError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        filename = str(item.get("name") or "backup.bin")
        quoted = quote(filename)
        headers = {
            "Content-Disposition": f"attachment; filename*=UTF-8''{quoted}",
            "Content-Length": str(int(item.get("size") or 0)),
        }
        return Response(
            content=bytes(item.get("payload") or b""),
            media_type=str(item.get("content_type") or "application/octet-stream"),
            headers=headers,
        )


    @router.get("/api/images/tags")
    async def list_image_tags(authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return {"tags": get_all_tags()}

    @router.post("/api/images/tags")
    async def update_image_tags(body: ImageTagsRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        rel = body.path.strip().lstrip("/")
        if not rel:
            raise HTTPException(status_code=400, detail={"error": "path is required"})
        tags = set_tags(rel, body.tags)
        return {"ok": True, "tags": tags}

    @router.delete("/api/images/tags/{tag}")
    async def delete_image_tag(tag: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        count = delete_tag(tag)
        return {"ok": True, "removed_from": count}

    return router
