"""Small provider-neutral authorization boundary for local API deployments."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol
from uuid import UUID


@dataclass(frozen=True, kw_only=True)
class Principal:
    tenant_id: UUID
    roles: frozenset[str]
    subject: str | None = None


class Authorizer(Protocol):
    def authorize(
        self, *, credential: str | None, tenant_id: UUID, role: str
    ) -> Principal: ...


class StaticApiKeyAuthorizer:
    """Authorize local API keys without changing the production auth boundary."""

    def __init__(self, credentials: Mapping[str, Principal]) -> None:
        if not credentials:
            raise ValueError("at least one API credential is required")
        self._credentials = dict(credentials)

    def authorize(
        self,
        *,
        credential: str | None = None,
        tenant_id: UUID,
        role: str,
        api_key: str | None = None,
    ) -> Principal:
        credential = credential if credential is not None else api_key
        if credential is not None and credential.startswith("Bearer "):
            credential = credential[7:].strip()
        if not credential:
            raise PermissionError("X-API-Key is required")
        principal = next(
            (candidate for key, candidate in self._credentials.items() if hmac.compare_digest(key, credential)),
            None,
        )
        if principal is None:
            raise PermissionError("invalid API key")
        if principal.tenant_id != tenant_id:
            raise PermissionError("API key does not belong to tenant")
        if role not in principal.roles:
            raise PermissionError(f"API key lacks role: {role}")
        return principal


class OidcJwtAuthorizer:
    """Validate OIDC JWTs and map tenant/role claims to platform principals."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str,
        tenant_claim: str = "tenant_id",
        roles_claim: str = "roles",
        key_provider: Callable[[str], Any] | None = None,
        algorithms: tuple[str, ...] = ("RS256", "ES256"),
        clock_skew_seconds: int = 30,
    ) -> None:
        if not issuer.strip() or not audience.strip() or not jwks_url.strip():
            raise ValueError("issuer, audience, and jwks_url must not be empty")
        if not tenant_claim.strip() or not roles_claim.strip():
            raise ValueError("tenant_claim and roles_claim must not be empty")
        if not algorithms:
            raise ValueError("at least one JWT algorithm is required")
        self.issuer = issuer
        self.audience = audience
        self.jwks_url = jwks_url
        self.tenant_claim = tenant_claim
        self.roles_claim = roles_claim
        self.key_provider = key_provider
        self.algorithms = algorithms
        self.clock_skew_seconds = clock_skew_seconds

    def authorize(
        self, *, credential: str | None, tenant_id: UUID, role: str
    ) -> Principal:
        if not credential:
            raise PermissionError("Bearer token is required")
        if not credential.startswith("Bearer "):
            raise PermissionError("Authorization must use the Bearer scheme")
        token = credential[7:].strip()
        if not token:
            raise PermissionError("Bearer token is required")
        try:
            import jwt

            claims = jwt.decode(
                token,
                self._signing_key(jwt, token),
                algorithms=list(self.algorithms),
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.clock_skew_seconds,
                options={"require": ["exp", "iat", "sub"]},
            )
        except Exception as exc:
            if exc.__class__.__module__.startswith("jwt"):
                raise PermissionError("invalid OIDC access token") from exc
            raise

        try:
            principal_tenant = UUID(str(claims[self.tenant_claim]))
        except (KeyError, ValueError, TypeError) as exc:
            raise PermissionError("token is missing a valid tenant claim") from exc
        roles = self._roles(claims.get(self.roles_claim))
        if principal_tenant != tenant_id:
            raise PermissionError("token does not belong to tenant")
        if role not in roles:
            raise PermissionError(f"token lacks role: {role}")
        return Principal(
            tenant_id=principal_tenant,
            roles=frozenset(roles),
            subject=str(claims["sub"]),
        )

    def _signing_key(self, jwt: Any, token: str) -> Any:
        if self.key_provider is not None:
            return self.key_provider(token)
        return jwt.PyJWKClient(self.jwks_url).get_signing_key_from_jwt(token).key

    @staticmethod
    def _roles(value: Any) -> set[str]:
        if isinstance(value, list) and all(isinstance(role, str) for role in value):
            return set(value)
        if isinstance(value, dict) and isinstance(value.get("roles"), list):
            return {role for role in value["roles"] if isinstance(role, str)}
        raise PermissionError("token is missing a valid roles claim")