"""Central role-to-permission policy for tenant APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


DEFAULT_ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "reader": frozenset({
        "search:read", "preview:read", "exports:read", "labels:read",
    }),
    "writer": frozenset({
        "assets:write", "uploads:write", "search:read", "subassets:write",
        "preview:read", "exports:read", "exports:write",
        "labels:read", "labels:write",
    }),
    "admin": frozenset({"*"}),
}


@dataclass(frozen=True, kw_only=True)
class RolePolicy:
    grants: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: dict(DEFAULT_ROLE_PERMISSIONS)
    )

    def allows(self, roles: Iterable[str], permission: str) -> bool:
        role_set = set(roles)
        if permission in role_set:
            return True
        return any(
            permission in self.grants.get(role, frozenset())
            or "*" in self.grants.get(role, frozenset())
            for role in role_set
        )

    def validate(self) -> None:
        for role, permissions in self.grants.items():
            if not role.strip() or any(not permission.strip() for permission in permissions):
                raise ValueError("roles and permissions must not be empty")
        if "admin" not in self.grants:
            raise ValueError("policy must define an admin role")
