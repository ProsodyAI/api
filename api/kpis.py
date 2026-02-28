"""
Client-defined KPI system for ProsodyAI.

Reads KPI definitions from the shared PostgreSQL database written by the
Next.js dashboard (../prosodyai-website).  This module is READ-ONLY — the
Python API never mutates KPI configuration.

KPI types:
    - scalar:      Continuous value with a range (e.g., CSAT 1-5)
    - binary:      Yes/no outcome (e.g., deal_closed, escalated)
    - categorical: One of N labels (e.g., resolution_type)

Each KPI has a direction (higher_is_better / lower_is_better) and optional
alert thresholds so the API can proactively warn when a predicted outcome is
heading in the wrong direction.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KPI Definition (mirrors the Prisma Kpi model)
# ---------------------------------------------------------------------------


class KPIType(str, Enum):
    SCALAR = "SCALAR"
    BINARY = "BINARY"
    CATEGORICAL = "CATEGORICAL"


class KPIDirection(str, Enum):
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"


class AlertDirection(str, Enum):
    ABOVE = "ABOVE"
    BELOW = "BELOW"


@dataclass
class KPIDefinition:
    """A single client-defined KPI, loaded from the database."""

    id: str
    name: str
    kpi_type: KPIType
    direction: KPIDirection = KPIDirection.HIGHER_IS_BETTER
    description: str = ""
    enabled: bool = True

    # Scalar KPIs: valid range
    range_min: Optional[float] = None
    range_max: Optional[float] = None

    # Categorical KPIs: allowed labels
    categories: list[str] = field(default_factory=list)

    # Alert config
    alert_threshold: Optional[float] = None
    alert_direction: AlertDirection = AlertDirection.BELOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.kpi_type.value,
            "direction": self.direction.value,
            "description": self.description,
            "enabled": self.enabled,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "categories": self.categories,
            "alert_threshold": self.alert_threshold,
            "alert_direction": self.alert_direction.value,
        }


# ---------------------------------------------------------------------------
# Database-backed KPI loader (read-only)
# ---------------------------------------------------------------------------

# SQL to fetch enabled KPIs for an organization, resolved via API key hash
_KPI_QUERY = """
SELECT
    k.id,
    k.name,
    k.type,
    k.direction,
    k.description,
    k.enabled,
    k."rangeMin",
    k."rangeMax",
    k.categories,
    k."alertThreshold",
    k."alertDirection"
FROM "Kpi" k
JOIN "ApiKey" ak ON ak."organizationId" = k."organizationId"
WHERE ak."keyHash" = $1
  AND k.enabled = true
ORDER BY k.name
"""

_ORG_ID_FROM_KEY_QUERY = """
SELECT "organizationId" FROM "ApiKey" WHERE "keyHash" = $1
"""


def _row_to_kpi(row) -> KPIDefinition:
    """Convert a database row (asyncpg Record) to a KPIDefinition."""
    return KPIDefinition(
        id=row["id"],
        name=row["name"],
        kpi_type=KPIType(row["type"]),
        direction=KPIDirection(row["direction"]),
        description=row["description"] or "",
        enabled=row["enabled"],
        range_min=row["rangeMin"],
        range_max=row["rangeMax"],
        categories=list(row["categories"]) if row["categories"] else [],
        alert_threshold=row["alertThreshold"],
        alert_direction=AlertDirection(row["alertDirection"]),
    )


class KPILoader:
    """
    Reads KPI definitions from the shared PostgreSQL database.

    The Next.js dashboard writes KPIs; this loader only reads them.
    Uses asyncpg for async PostgreSQL access, matching FastAPI's async model.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._database_url = database_url or os.getenv(
            "DATABASE_URL",
            os.getenv("PROSODYAI_DATABASE_URL", ""),
        )
        self._pool = None

    async def _get_pool(self):
        """Lazy-initialize the connection pool."""
        if self._pool is None:
            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "asyncpg is required for KPI loading. "
                    "Install with: pip install asyncpg"
                )

            if not self._database_url:
                raise RuntimeError(
                    "DATABASE_URL is not set. The Python API needs access to "
                    "the same PostgreSQL database as the Next.js dashboard."
                )

            self._pool = await asyncpg.create_pool(
                self._database_url,
                min_size=2,
                max_size=10,
            )
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def get_kpis_for_api_key(self, api_key_hash: str) -> list[KPIDefinition]:
        """
        Fetch all enabled KPIs for the organization that owns this API key.

        Args:
            api_key_hash: SHA-256 hash of the caller's API key.

        Returns:
            List of enabled KPIDefinitions (empty if no KPIs configured).
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(_KPI_QUERY, api_key_hash)
            return [_row_to_kpi(row) for row in rows]

    async def get_organization_id(self, api_key_hash: str) -> Optional[str]:
        """Resolve an API key hash to an organization ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(_ORG_ID_FROM_KEY_QUERY, api_key_hash)
            return row["organizationId"] if row else None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_loader: Optional[KPILoader] = None


def get_kpi_loader() -> KPILoader:
    """Get the singleton KPILoader instance."""
    global _loader
    if _loader is None:
        _loader = KPILoader()
    return _loader
