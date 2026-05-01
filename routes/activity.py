"""
Admin activity endpoints: operator pulse for the admin dashboard.

Aggregates recent activity across all tenants from the shared Postgres DB.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from kpis import get_kpi_loader
from routes.admin import require_admin

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/activity/pulse", dependencies=[Depends(require_admin)])
async def activity_pulse(hours: int = Query(24, ge=1, le=168)) -> dict[str, Any]:
    """
    Aggregated platform activity for the last N hours.

    Returns totals, hourly buckets for a sparkline, and the top-5 most active
    organizations.
    """
    loader = get_kpi_loader()
    try:
        pool = await loader._get_pool()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")

    async with pool.acquire() as conn:
        # Totals
        totals_row = await conn.fetchrow(
            """
            SELECT
              (SELECT COUNT(*) FROM "UsageEvent" WHERE "createdAt" >= NOW() - make_interval(hours => $1)) AS api_calls,
              (SELECT COUNT(*) FROM "Transcript" WHERE "createdAt" >= NOW() - make_interval(hours => $1)) AS transcripts,
              (SELECT COUNT(*) FROM "User" WHERE "createdAt" >= NOW() - make_interval(hours => $1)) AS new_users,
              (SELECT COUNT(*) FROM "Organization" WHERE "createdAt" >= NOW() - make_interval(hours => $1)) AS new_orgs,
              (SELECT COUNT(DISTINCT "organizationId") FROM "UsageEvent" WHERE "createdAt" >= NOW() - make_interval(hours => $1)) AS active_orgs,
              (SELECT COUNT(*) FROM "Organization") AS total_orgs,
              (SELECT COUNT(*) FROM "User") AS total_users,
              (SELECT COUNT(*) FROM "Transcript") AS total_transcripts,
              (SELECT COUNT(*) FROM "ApiKey") AS total_api_keys
            """,
            hours,
        )

        # Hourly buckets (event counts per hour for sparkline)
        bucket_rows = await conn.fetch(
            """
            WITH hours AS (
              SELECT generate_series(
                date_trunc('hour', NOW() - make_interval(hours => $1 - 1)),
                date_trunc('hour', NOW()),
                interval '1 hour'
              ) AS hour
            )
            SELECT
              h.hour,
              COALESCE(SUM(CASE WHEN ue.id IS NOT NULL THEN 1 ELSE 0 END), 0) AS api_calls,
              COALESCE(SUM(CASE WHEN t.id IS NOT NULL THEN 1 ELSE 0 END), 0) AS transcripts
            FROM hours h
            LEFT JOIN "UsageEvent" ue ON date_trunc('hour', ue."createdAt") = h.hour
            LEFT JOIN "Transcript" t ON date_trunc('hour', t."createdAt") = h.hour
            GROUP BY h.hour
            ORDER BY h.hour
            """,
            hours,
        )

        # Top 5 orgs by activity (use UsageEvent + Transcript, joined to org name)
        top_orgs = await conn.fetch(
            """
            SELECT
              o.id,
              o.name,
              o.slug,
              o.plan,
              (SELECT COUNT(*) FROM "UsageEvent" ue
                 WHERE ue."organizationId" = o.id
                   AND ue."createdAt" >= NOW() - make_interval(hours => $1)) AS api_calls,
              (SELECT COUNT(*) FROM "Transcript" t
                 WHERE t."organizationId" = o.id
                   AND t."createdAt" >= NOW() - make_interval(hours => $1)) AS transcripts
            FROM "Organization" o
            ORDER BY api_calls DESC, transcripts DESC
            LIMIT 5
            """,
            hours,
        )

        # Recent activity timeline: 20 most recent events + transcripts with org name
        recent = await conn.fetch(
            """
            (SELECT
              'usage' AS kind,
              ue.id,
              ue.type,
              ue."createdAt" AS ts,
              o.name AS org_name,
              o.slug AS org_slug,
              o.id AS org_id,
              NULL::text AS detail
            FROM "UsageEvent" ue
            LEFT JOIN "Organization" o ON o.id = ue."organizationId"
            ORDER BY ue."createdAt" DESC
            LIMIT 20)
            UNION ALL
            (SELECT
              'transcript' AS kind,
              t.id,
              'transcript_created' AS type,
              t."createdAt" AS ts,
              o.name AS org_name,
              o.slug AS org_slug,
              o.id AS org_id,
              COALESCE(t.source::text, 'UPLOAD') AS detail
            FROM "Transcript" t
            LEFT JOIN "Organization" o ON o.id = t."organizationId"
            ORDER BY t."createdAt" DESC
            LIMIT 20)
            ORDER BY ts DESC
            LIMIT 20
            """,
        )

        # Recent signups
        signups = await conn.fetch(
            """
            SELECT u.id, u.email, u.name, u."createdAt", o.name AS org_name, o.slug AS org_slug
            FROM "User" u
            LEFT JOIN "Organization" o ON o.id = u."organizationId"
            ORDER BY u."createdAt" DESC
            LIMIT 10
            """,
        )

    return {
        "hours": hours,
        "totals": {
            "api_calls": totals_row["api_calls"],
            "transcripts": totals_row["transcripts"],
            "new_users": totals_row["new_users"],
            "new_orgs": totals_row["new_orgs"],
            "active_orgs": totals_row["active_orgs"],
            "total_orgs": totals_row["total_orgs"],
            "total_users": totals_row["total_users"],
            "total_transcripts": totals_row["total_transcripts"],
            "total_api_keys": totals_row["total_api_keys"],
        },
        "buckets": [
            {
                "t": r["hour"].isoformat(),
                "api_calls": r["api_calls"],
                "transcripts": r["transcripts"],
            }
            for r in bucket_rows
        ],
        "top_orgs": [
            {
                "id": r["id"],
                "name": r["name"],
                "slug": r["slug"],
                "plan": r["plan"],
                "api_calls": r["api_calls"],
                "transcripts": r["transcripts"],
            }
            for r in top_orgs
        ],
        "recent": [
            {
                "kind": r["kind"],
                "id": r["id"],
                "type": r["type"],
                "ts": r["ts"].isoformat() if r["ts"] else None,
                "org_name": r["org_name"],
                "org_slug": r["org_slug"],
                "org_id": r["org_id"],
                "detail": r["detail"],
            }
            for r in recent
        ],
        "recent_signups": [
            {
                "id": r["id"],
                "email": r["email"],
                "name": r["name"],
                "created_at": r["createdAt"].isoformat() if r["createdAt"] else None,
                "org_name": r["org_name"],
                "org_slug": r["org_slug"],
            }
            for r in signups
        ],
    }
