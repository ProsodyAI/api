"""
Shared database helpers for ConversationSession, Transcript, and Audio.

Uses the same asyncpg pool as kpis (get_kpi_loader()._get_pool()).
Table names match website/prisma (ConversationSession = conversation session).
"""

import logging
from typing import Optional

from kpis import get_kpi_loader

logger = logging.getLogger(__name__)


async def create_session(organization_id: str, metadata: Optional[dict] = None) -> Optional[str]:
    """
    Create a ConversationSession row and return its id.
    Returns None if the table does not exist or on error.
    """
    import json

    loader = get_kpi_loader()
    try:
        pool = await loader._get_pool()
    except Exception as e:
        logger.debug("No database pool for create_session: %s", e)
        return None

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO "ConversationSession" ("organizationId", metadata, "createdAt")
                VALUES ($1, $2::jsonb, NOW())
                RETURNING id
                """,
                organization_id,
                json.dumps(metadata) if metadata else None,
            )
            return str(row["id"]) if row else None
    except Exception as e:
        if "does not exist" in str(e).lower():
            logger.debug("ConversationSession table not found: %s", e)
            return None
        logger.warning("create_session failed: %s", e)
        return None


async def get_session_organization_id(session_id: str) -> Optional[str]:
    """
    Return the organizationId that owns the conversation session.
    Used for tenant isolation.
    """
    loader = get_kpi_loader()
    try:
        pool = await loader._get_pool()
    except Exception:
        return None

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT "organizationId" FROM "ConversationSession" WHERE id = $1',
                session_id,
            )
            return row["organizationId"] if row else None
    except Exception as e:
        if "does not exist" in str(e).lower():
            return None
        logger.warning("get_session_organization_id failed: %s", e)
        return None
