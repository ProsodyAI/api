"""
Webhook delivery for tenant integrations.

After a streaming session ends, deliver the enriched transcript to the
organization's CUSTOM_WEBHOOK integration (if configured and enabled).
"""

import hashlib
import hmac
import json
import logging
import os
from typing import Optional

import httpx
from kpis import get_kpi_loader

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT = 10.0

# Treat an unset OR empty env var as "use default" — Cloud Run's
# --env-vars-file path will inject INTEGRATION_ENCRYPTION_KEY="" when the
# corresponding GitHub secret hasn't been set yet, and `os.getenv` would
# happily return that empty string. Falling back to the default keeps
# existing encrypted Integration rows readable until the rotation runs.
_ENCRYPTION_KEY = os.getenv("INTEGRATION_ENCRYPTION_KEY") or "prosody-dev-key-32-bytes-long!!"


def _decrypt_config(raw_config) -> dict:
    """Decrypt AES-256-GCM config stored by the Next.js dashboard.

    Prisma stores `Json` fields as jsonb, so *raw_config* may be a dict (plain
    JSON, unencrypted) or a string (the ``iv_hex:auth_tag_hex:ciphertext_hex``
    encrypted format).
    """
    if isinstance(raw_config, dict):
        return raw_config

    encrypted_text = str(raw_config)
    # jsonb may store the encrypted string with JSON quotes
    if encrypted_text.startswith('"') and encrypted_text.endswith('"'):
        encrypted_text = json.loads(encrypted_text)

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    parts = encrypted_text.split(":")
    if len(parts) != 3:
        return json.loads(encrypted_text)

    iv = bytes.fromhex(parts[0])
    auth_tag = bytes.fromhex(parts[1])
    ciphertext = bytes.fromhex(parts[2])
    key = _ENCRYPTION_KEY.ljust(32)[:32].encode()
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(iv, ciphertext + auth_tag, None)
    return json.loads(plaintext.decode())


async def _get_webhook_config(org_id: str) -> Optional[dict]:
    """Fetch the CUSTOM_WEBHOOK integration for *org_id*, or ``None``."""
    loader = get_kpi_loader()
    try:
        pool = await loader._get_pool()
    except Exception:
        return None

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT config FROM "Integration"
                WHERE "organizationId" = $1
                  AND type = 'CUSTOM_WEBHOOK'
                  AND enabled = true
                """,
                org_id,
            )
            if not row or not row["config"]:
                return None
            return _decrypt_config(row["config"])
    except Exception:
        logger.debug("Failed to load CUSTOM_WEBHOOK for org %s", org_id, exc_info=True)
        return None


async def _update_sync_status(org_id: str, error: Optional[str] = None) -> None:
    loader = get_kpi_loader()
    try:
        pool = await loader._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE "Integration"
                SET "lastSyncAt" = NOW(), "lastError" = $1
                WHERE "organizationId" = $2
                  AND type = 'CUSTOM_WEBHOOK'
                """,
                error,
                org_id,
            )
    except Exception:
        logger.debug("Failed to update Integration sync status", exc_info=True)


def _sign_payload(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def deliver_webhook(
    org_id: str,
    session_id: str,
    transcript: dict,
    *,
    prosody_summary: Optional[dict] = None,
) -> None:
    """POST the enriched transcript to the org's CUSTOM_WEBHOOK, if configured."""
    config = await _get_webhook_config(org_id)
    if not config:
        return

    url = config.get("url", "").strip()
    if not url:
        return
    if not (url.startswith("http://") or url.startswith("https://")):
        # Stored URL is missing a scheme; httpx would raise on send. Surface a
        # clear, actionable error in lastError instead of the cryptic httpx one.
        msg = "Webhook URL must start with http:// or https://"
        logger.warning("Webhook delivery skipped for session %s: %s", session_id, msg)
        await _update_sync_status(org_id, error=msg)
        return

    secret = config.get("secret", "")
    headers_extra: dict = config.get("headers") or {}

    payload = {
        "event": "session.transcript",
        "session_id": session_id,
        "transcript": transcript,
    }
    if prosody_summary:
        payload["prosody_summary"] = prosody_summary

    body = json.dumps(payload, default=str).encode()

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ProsodyAI-Webhook/1.0",
        **headers_extra,
    }
    if secret:
        headers["X-ProsodyAI-Signature"] = _sign_payload(secret, body)

    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
            resp = await client.post(url, content=body, headers=headers)
            resp.raise_for_status()
        logger.info("Webhook delivered for session %s → %s (%d)", session_id, url, resp.status_code)
        await _update_sync_status(org_id)
    except Exception as e:
        error_msg = str(e)[:500]
        logger.warning("Webhook delivery failed for session %s: %s", session_id, error_msg)
        await _update_sync_status(org_id, error=error_msg)
