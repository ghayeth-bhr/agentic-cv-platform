from uuid import uuid4

import pytest

import jwt

from agentic_cv.auth import OidcJwtAuthorizer, Principal, StaticApiKeyAuthorizer


def test_static_api_key_authorizer_binds_tenant_and_role():
    tenant_id = uuid4()
    authorizer = StaticApiKeyAuthorizer({
        "writer-key": Principal(tenant_id=tenant_id, roles=frozenset({"writer"})),
    })

    assert authorizer.authorize(api_key="writer-key", tenant_id=tenant_id, role="writer").tenant_id == tenant_id
    with pytest.raises(PermissionError, match="tenant"):
        authorizer.authorize(api_key="writer-key", tenant_id=uuid4(), role="writer")
    with pytest.raises(PermissionError, match="role"):
        authorizer.authorize(api_key="writer-key", tenant_id=tenant_id, role="admin")
    with pytest.raises(PermissionError, match="required"):
        authorizer.authorize(api_key=None, tenant_id=tenant_id, role="writer")


def test_oidc_authorizer_validates_claims_and_maps_bearer_principal(monkeypatch):
    tenant_id = uuid4()
    captured = {}

    def fake_decode(token, key, **kwargs):
        captured.update(kwargs)
        assert token == "signed-token"
        assert key == "jwks-key"
        return {
            "sub": "user-123",
            "tenant_id": str(tenant_id),
            "roles": ["reader", "writer"],
        }

    monkeypatch.setattr(jwt, "decode", fake_decode)
    authorizer = OidcJwtAuthorizer(
        issuer="https://issuer.example",
        audience="agentic-cv",
        jwks_url="https://issuer.example/.well-known/jwks.json",
        key_provider=lambda token: "jwks-key",
    )

    principal = authorizer.authorize(
        credential="Bearer signed-token", tenant_id=tenant_id, role="reader"
    )

    assert principal.subject == "user-123"
    assert captured["issuer"] == "https://issuer.example"
    assert captured["audience"] == "agentic-cv"
    assert captured["algorithms"] == ["RS256", "ES256"]


def test_oidc_authorizer_rejects_bad_scheme_tenant_and_role(monkeypatch):
    tenant_id = uuid4()
    authorizer = OidcJwtAuthorizer(
        issuer="issuer", audience="audience", jwks_url="jwks",
        key_provider=lambda token: "key",
    )
    with pytest.raises(PermissionError, match="Bearer"):
        authorizer.authorize(credential="Basic token", tenant_id=tenant_id, role="reader")

    monkeypatch.setattr(
        jwt, "decode", lambda *args, **kwargs: {
            "sub": "user", "tenant_id": str(uuid4()), "roles": ["reader"]
        },
    )
    with pytest.raises(PermissionError, match="tenant"):
        authorizer.authorize(credential="Bearer token", tenant_id=tenant_id, role="reader")
