"""H200 modification: Messages clients discover models with pagination query strings."""

from types import SimpleNamespace

import pytest

from tensorfold.cuda.http import make_handler


@pytest.mark.parametrize("path", ["/v1/models", "/v1/models?limit=1000",
                                  "/v1/models/?limit=1", "/models?limit=1000"])
def test_model_listing_accepts_query(path, monkeypatch):
    monkeypatch.delenv("TF_API_KEY", raising=False)
    monkeypatch.delenv("TF_API_KEY_FILE", raising=False)
    app = SimpleNamespace(model_ids=["H200-test-model"], effective_context_window=1048576)
    handler_type = make_handler(app)
    handler = object.__new__(handler_type)
    handler.path = path
    handler._authorized = lambda: True
    responses = []
    handler._json = lambda status, body: responses.append((status, body))
    handler.do_GET()
    assert responses == [(200, {"object": "list", "data": [
        {"id": "H200-test-model", "object": "model", "owned_by": "tensorfold", "max_model_len": 1048576}
    ]})]
