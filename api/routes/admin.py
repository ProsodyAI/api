"""
Admin API for enterprise tenants: API key CRUD and RBAC.

Protected by ADMIN_API_KEY (env: PROSODYAI_ADMIN_API_KEY or ADMIN_API_KEY).
Use header X-Admin-Key or Authorization: Bearer <key>.
"""

import hashlib
import logging
import secrets
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, status
from pydantic import BaseModel, Field

from api.config import settings
from api.kpis import get_kpi_loader

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Admin auth
# ---------------------------------------------------------------------------

def _get_admin_key_from_header(
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
    authorization: Optional[str] = Header(None),
) -> Optional[str]:
    """Extract admin key from X-Admin-Key or Authorization: Bearer."""
    if x_admin_key:
        return x_admin_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


async def require_admin(
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
    authorization: Optional[str] = Header(None),
) -> None:
    """Dependency: require valid admin API key for admin routes."""
    key = _get_admin_key_from_header(x_admin_key, authorization)
    admin_key = settings.admin_api_key or __import__("os").environ.get("ADMIN_API_KEY")
    if not admin_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API is not configured (ADMIN_API_KEY not set)",
        )
    if not key or not secrets.compare_digest(key, admin_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing admin API key. Use X-Admin-Key or Authorization: Bearer.",
        )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CreateApiKeyResponse(BaseModel):
    id: str
    key: str = Field(description="The raw API key; only returned once. Store it securely.")
    name: Optional[str] = None


class ApiKeyListItem(BaseModel):
    id: str
    name: Optional[str] = None
    key_preview: str = Field(description="Last 4 characters, e.g. …abc1")
    created_at: Optional[str] = None


class AssignRolesRequest(BaseModel):
    role_ids: list[str] = Field(description="Role IDs to assign to the user")


# ---------------------------------------------------------------------------
# API key CRUD
# ---------------------------------------------------------------------------

@router.post(
    "/tenants/{organization_id}/api-keys",
    response_model=CreateApiKeyResponse,
    summary="Create API key for organization",
)
async def create_api_key(
    organization_id: str,
    name: Optional[str] = None,
    _: None = Depends(require_admin),
):
    """Create a new API key for the tenant. The raw key is returned only in this response."""
    raw_key = f"prosody_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[:8]
    key_id = f"key_{uuid.uuid4().hex[:24]}"

    loader = get_kpi_loader()
    pool = await loader._get_pool()
    async with pool.acquire() as conn:
        try:
            await conn.execute(
                """
                INSERT INTO "ApiKey" (id, "organizationId", "keyHash", "keyPrefix", name, "createdAt", "updatedAt")
                VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                """,
                key_id,
                organization_id,
                key_hash,
                key_prefix,
                name or "API Key",
            )
        except Exception as e:
            if "keyPrefix" in str(e) or "does not exist" in str(e).lower():
                await conn.execute(
                    """
                    INSERT INTO "ApiKey" (id, "organizationId", "keyHash", name, "createdAt", "updatedAt")
                    VALUES ($1, $2, $3, $4, NOW(), NOW())
                    """,
                    key_id,
                    organization_id,
                    key_hash,
                    name or "API Key",
                )
            else:
                raise
    return CreateApiKeyResponse(id=key_id, key=raw_key, name=name or None)


@router.get(
    "/tenants/{organization_id}/api-keys",
    response_model=list[ApiKeyListItem],
    summary="List API keys for organization",
)
async def list_api_keys(
    organization_id: str,
    _: None = Depends(require_admin),
):
    """List API keys for the tenant (masked)."""
    loader = get_kpi_loader()
    pool = await loader._get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, "keyHash", "createdAt"
            FROM "ApiKey"
            WHERE "organizationId" = $1
            ORDER BY "createdAt" DESC
            """,
            organization_id,
        )
    out = []
    for row in rows:
        # Show last 4 chars of hash as preview (not the key itself)
        h = row["keyHash"] or ""
        preview = f"…{h[-4:]}" if len(h) >= 4 else "…****"
        out.append(ApiKeyListItem(
            id=row["id"],
            name=row["name"],
            key_preview=preview,
            created_at=row["createdAt"].isoformat() if row.get("createdAt") else None,
        ))
    return out


@router.delete(
    "/tenants/{organization_id}/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API key",
)
async def revoke_api_key(
    organization_id: str,
    key_id: str,
    _: None = Depends(require_admin),
):
    """Revoke an API key by ID."""
    loader = get_kpi_loader()
    pool = await loader._get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM "ApiKey"
            WHERE id = $1 AND "organizationId" = $2
            """,
            key_id,
            organization_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")


# ---------------------------------------------------------------------------
# RBAC: users, roles, permissions
# ---------------------------------------------------------------------------

@router.get(
    "/tenants/{organization_id}/users",
    summary="List users in organization",
)
async def list_users(
    organization_id: str,
    _: None = Depends(require_admin),
):
    """List users for the tenant (requires User table from schema)."""
    loader = get_kpi_loader()
    pool = await loader._get_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, "organizationId", email, name, "createdAt"
                FROM "User"
                WHERE "organizationId" = $1
                ORDER BY "createdAt" DESC
                """,
                organization_id,
            )
    except Exception as e:
        if "does not exist" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="User table not found. Run schema migrations (e.g. schema/migrations/002_rbac.sql).",
            ) from e
        raise
    return [dict(r) for r in rows]


@router.get(
    "/tenants/{organization_id}/roles",
    summary="List roles in organization",
)
async def list_roles(
    organization_id: str,
    _: None = Depends(require_admin),
):
    """List roles for the tenant."""
    loader = get_kpi_loader()
    pool = await loader._get_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, "organizationId", name, "createdAt"
                FROM "Role"
                WHERE "organizationId" = $1
                ORDER BY name
                """,
                organization_id,
            )
    except Exception as e:
        if "does not exist" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Role table not found. Run schema migrations (e.g. schema/migrations/002_rbac.sql).",
            ) from e
        raise
    return [dict(r) for r in rows]


@router.post(
    "/tenants/{organization_id}/users/{user_id}/roles",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Assign roles to user",
)
async def assign_roles_to_user(
    organization_id: str,
    user_id: str,
    body: AssignRolesRequest,
    _: None = Depends(require_admin),
):
    """Assign one or more roles to a user. Replaces existing assignments if needed."""
    loader = get_kpi_loader()
    pool = await loader._get_pool()
    try:
        async with pool.acquire() as conn:
            # Verify user belongs to org
            u = await conn.fetchrow(
                'SELECT id FROM "User" WHERE id = $1 AND "organizationId" = $2',
                user_id,
                organization_id,
            )
            if not u:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
            # Remove existing role assignments for this user
            await conn.execute('DELETE FROM "UserRole" WHERE "userId" = $1', user_id)
            # Insert new assignments
            for role_id in body.role_ids:
                r = await conn.fetchrow(
                    'SELECT id FROM "Role" WHERE id = $1 AND "organizationId" = $2',
                    role_id,
                    organization_id,
                )
                if not r:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Role {role_id} not found in organization",
                    )
                await conn.execute(
                    'INSERT INTO "UserRole" ("userId", "roleId", "createdAt") VALUES ($1, $2, NOW())',
                    user_id,
                    role_id,
                )
    except HTTPException:
        raise
    except Exception as e:
        if "does not exist" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="RBAC tables not found. Run schema migrations (e.g. schema/migrations/002_rbac.sql).",
            ) from e
        raise


@router.get(
    "/tenants/{organization_id}/permissions",
    summary="List permissions (global or role–permission mapping)",
)
async def list_permissions(
    organization_id: str,
    _: None = Depends(require_admin),
):
    """List all permissions (global list). Optionally include which roles have which permissions."""
    loader = get_kpi_loader()
    pool = await loader._get_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch('SELECT id, name FROM "Permission" ORDER BY name')
    except Exception as e:
        if "does not exist" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Permission table not found. Run schema migrations (e.g. schema/migrations/002_rbac.sql).",
            ) from e
        raise
    return [dict(r) for r in rows]
