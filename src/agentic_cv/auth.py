"""Small provider-neutral authorization boundary for local API deployments."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Mapping
from uuid import UUID


@dataclass(frozen=True, kw_only=True)
class Principal:
    tenant_id: UUID
    roles: frozenset[str]


class StaticApiKeyAuthorizer:
    """Authorize local API keys without changing the production auth boundary."""

    def __init__(self, credentials: Mapping[str, Principal]) -> None:
        if not credentials:
            raise ValueError("at least one API credential is required")
        self._credentials = dict(credentials)

    def authorize(self, *, api_key: str | None, tenant_id: UUID, role: str) -> Principal:
        if not api_key:
            raise PermissionError("X-API-Key is required")
        principal = next(
            (candidate for key, candidate in self._credentials.items() if hmac.compare_digest(key, api_key)),
            None,
        )
        if principal is None:
            raise PermissionError("invalid API key")
        if principal.tenant_id != tenant_id:
            raise PermissionError("API key does not belong to tenant")
        if role not in principal.roles:
            raise PermissionError(f"API key lacks role: {role}")
        return principal