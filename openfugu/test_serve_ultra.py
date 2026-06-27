#!/usr/bin/env python3
"""
Offline functional tests for serve_ultra.py — no API keys, no GPU, no network.

Uses ultra.CANNED (a pre-baked parseable 3-step workflow) and ultra.MockWorker
(deterministic stub replies) to exercise the real Handler request path.

Run:
    cd D:/Devel/OpenFugu/openfugu
    python test_serve_ultra.py
"""
from __future__ import annotations
import http.client
import json
import os
import sys
import threading

# Run from the openfugu/ directory so ultra's sys.path.insert finds its siblings.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serve_ultra
import ultra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start_server() -> tuple:
    """Bind :0 (OS assigns a free port), serve in a daemon thread."""
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_ultra.Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def _post(port: int, payload: dict) -> tuple:
    """POST JSON to /v1/chat/completions; return (response, parsed_body)."""
    data = json.dumps(payload).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=data,
        headers={"Content-Type": "application/json", "Content-Length": str(len(data))},
    )
    resp = conn.getresponse()
    body = json.loads(resp.read())
    return resp, body


def _with_globals(conductor_fn, worker, slot_labels=None):
    """Context manager: temporarily set module-level globals."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        old_c, old_w, old_s = serve_ultra.CONDUCTOR_FN, serve_ultra.WORKER, serve_ultra.SLOT_LABELS
        serve_ultra.CONDUCTOR_FN = conductor_fn
        serve_ultra.WORKER = worker
        serve_ultra.SLOT_LABELS = slot_labels or ultra.DEFAULT_SLOT_LABELS
        try:
            yield
        finally:
            serve_ultra.CONDUCTOR_FN, serve_ultra.WORKER, serve_ultra.SLOT_LABELS = old_c, old_w, old_s

    return _ctx()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_normal_flow():
    """
    Happy path: CANNED conductor output → 3-step DAG executed by MockWorker
    → HTTP 200, non-empty content, no warning header.
    """
    with _with_globals(lambda msgs: ultra.CANNED, ultra.MockWorker()):
        srv, port = _start_server()
        try:
            resp, body = _post(port, {
                "messages": [{"role": "user", "content": "What is 2+2?"}]
            })
            assert resp.status == 200, f"expected 200, got {resp.status}"
            content = body["choices"][0]["message"]["content"]
            assert content, "content must be non-empty"
            assert resp.getheader("X-Fugu-Warning") is None, "unexpected warning header"
            # steps count should match the 3-step CANNED workflow
            assert body["usage"]["total_steps"] == 3, f"expected 3 steps, got {body['usage']['total_steps']}"
        finally:
            srv.shutdown()
    print("  PASS  test_normal_flow")


def test_no_workflow_parsed():
    """
    Degraded path: conductor emits unparseable text → HTTP 200 with
    X-Fugu-Warning: no-workflow-parsed header; raw completion returned as content.
    """
    raw = "I cannot produce a workflow right now."
    with _with_globals(lambda msgs: raw, ultra.MockWorker()):
        srv, port = _start_server()
        try:
            resp, body = _post(port, {
                "messages": [{"role": "user", "content": "Anything"}]
            })
            assert resp.status == 200, f"expected 200, got {resp.status}"
            warning = resp.getheader("X-Fugu-Warning")
            assert warning == "no-workflow-parsed", (
                f"expected 'no-workflow-parsed', got {warning!r}"
            )
            content = body["choices"][0]["message"]["content"]
            assert content == raw, f"expected raw completion as content, got {content!r}"
            assert body["usage"]["total_steps"] == 0
        finally:
            srv.shutdown()
    print("  PASS  test_no_workflow_parsed")


def test_health_always_200():
    """GET /health always returns HTTP 200 with status=ok."""
    with _with_globals(lambda msgs: ultra.CANNED, ultra.MockWorker()):
        srv, port = _start_server()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            body = json.loads(resp.read())
            assert resp.status == 200, f"expected 200, got {resp.status}"
            assert body["status"] == "ok"
        finally:
            srv.shutdown()
    print("  PASS  test_health_always_200")


def test_missing_messages_returns_400():
    """POST with an empty messages list → 400."""
    with _with_globals(lambda msgs: ultra.CANNED, ultra.MockWorker()):
        srv, port = _start_server()
        try:
            resp, body = _post(port, {"messages": []})
            assert resp.status == 400, f"expected 400, got {resp.status}"
            assert "error" in body
        finally:
            srv.shutdown()
    print("  PASS  test_missing_messages_returns_400")


def test_unknown_path_returns_404():
    """POST to an unknown path → 404."""
    with _with_globals(lambda msgs: ultra.CANNED, ultra.MockWorker()):
        srv, port = _start_server()
        try:
            data = json.dumps({}).encode()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("POST", "/not/a/real/path", body=data,
                         headers={"Content-Length": str(len(data))})
            resp = conn.getresponse()
            assert resp.status == 404, f"expected 404, got {resp.status}"
        finally:
            srv.shutdown()
    print("  PASS  test_unknown_path_returns_404")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("serve_ultra.py functional tests (offline — no keys, no GPU)\n")
    test_normal_flow()
    test_no_workflow_parsed()
    test_health_always_200()
    test_missing_messages_returns_400()
    test_unknown_path_returns_404()
    print("\nAll 5 tests passed.")
