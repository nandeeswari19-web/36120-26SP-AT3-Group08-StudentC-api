"""Calls to peer services' POST /features (CONTRACT.md section 7).

- Base URLs come from environment variables, never hard-coded:
  STUDENT_A_URL (Return20, Return5) and STUDENT_B_URL (MA5, MA20).
- Peers are called IN PARALLEL with httpx.AsyncClient, 10 s timeout, 1 retry.
- Successful responses are cached for 15 minutes per (peer, token).
- Any failure (not configured, timeout, HTTP error, bad JSON, missing key)
  makes that peer's features missing; the prediction still returns 200.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Final

import httpx

from app.features import PEER_FEATURES

logger = logging.getLogger(__name__)

PEER_ENV: Final[dict[str, str]] = {"student_a": "STUDENT_A_URL", "student_b": "STUDENT_B_URL"}
TIMEOUT_SECONDS: Final = float(os.getenv("PEER_TIMEOUT_SECONDS", "10"))
RETRIES: Final = 1
CACHE_SECONDS: Final = 15 * 60


@dataclass
class PeerResult:
    peer: str
    ok: bool
    values: dict[str, float | None] = field(default_factory=dict)
    as_of_date: str | None = None
    error: str | None = None
    cached: bool = False


_cache: dict[tuple[str, str], tuple[float, PeerResult]] = {}


def peer_url(peer: str) -> str | None:
    value = os.getenv(PEER_ENV[peer], "").strip().rstrip("/")
    return value or None


def configured_peers() -> dict[str, str | None]:
    return {peer: peer_url(peer) for peer in PEER_ENV}


async def _fetch_one(client: httpx.AsyncClient, peer: str, token: str) -> PeerResult:
    base = peer_url(peer)
    if base is None:
        return PeerResult(peer, ok=False, error=f"{PEER_ENV[peer]} is not configured")

    key = (peer, token)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        result = hit[1]
        return PeerResult(peer, True, dict(result.values), result.as_of_date, cached=True)

    wanted = PEER_FEATURES[peer]
    last_error = "unknown error"
    for attempt in range(RETRIES + 1):
        try:
            response = await client.post(
                f"{base}/features",
                json={"token": token, "forecast_horizon": 7, "history_days": 1},
                timeout=TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
                if response.status_code < 500:  # a bad request will not improve on retry
                    break
            else:
                body = response.json()
                features = body.get("features") or {}
                values = {name: features.get(name) for name in wanted}
                if all(v is None for v in values.values()):
                    last_error = "response contained none of the expected features"
                    break
                result = PeerResult(peer, True, values, body.get("as_of_date"))
                _cache[key] = (time.monotonic(), result)
                return result
        except (httpx.HTTPError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < RETRIES:
            await asyncio.sleep(0.5)

    logger.warning("Peer %s unavailable for %s: %s", peer, token, last_error)
    return PeerResult(peer, ok=False, error=last_error)


async def fetch_peer_features(token: str) -> dict[str, PeerResult]:
    """Fetch every peer in parallel; never raises."""
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(_fetch_one(client, p, token) for p in PEER_ENV))
    return {r.peer: r for r in results}


def clear_cache() -> None:
    _cache.clear()
