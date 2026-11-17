from uuid import uuid4

import pytest

from agentic_cv.auth import Principal, StaticApiKeyAuthorizer


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