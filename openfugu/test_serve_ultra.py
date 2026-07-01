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
    with _with_globals(lambda msgs: (ultra.CANNED, 0, 0), ultra.MockWorker()):
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
    with _with_globals(lambda msgs: (raw, 0, 0), ultra.MockWorker()):
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
    with _with_globals(lambda msgs: (ultra.CANNED, 0, 0), ultra.MockWorker()):
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
    with _with_globals(lambda msgs: (ultra.CANNED, 0, 0), ultra.MockWorker()):
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
    with _with_globals(lambda msgs: (ultra.CANNED, 0, 0), ultra.MockWorker()):
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



def test_v1_workers_returns_valid_structure():
    """GET /v1/workers returns valid structure with conductor slot_id=0 before any requests."""
    import importlib, sys, threading, json, urllib.request
    from http.server import HTTPServer

    # Fresh import to get clean counter state
    for key in [k for k in sys.modules if "serve_ultra" in k]:
        del sys.modules[key]

    import serve_ultra as su

    server = HTTPServer(("127.0.0.1", 0), su.Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/workers", timeout=3)
        data = json.loads(resp.read())
        assert data["model"] == "fugu-ultra", f"unexpected model: {data['model']}"
        assert isinstance(data["workers"], list), "workers must be a list"
        # Conductor slot must always appear
        assert len(data["workers"]) >= 1, "at least conductor slot expected"
        w0 = data["workers"][0]
        assert w0["slot_id"] == 0, f"slot 0 must be conductor, got {w0['slot_id']}"
        assert all(f"lat_h{i}" in w0 for i in range(11)), f"missing lat_hN fields: {list(w0)}"
    finally:
        server.shutdown()
    print("  PASS  test_v1_workers_returns_valid_structure")


def test_conductor_token_accounting():
    """Conductor prompt/completion tokens from CONDUCTOR_FN are recorded and
    surfaced via /v1/workers (regression guard: these were previously never
    incremented and always reported 0)."""
    import sys, threading, json, urllib.request
    from http.server import HTTPServer

    for key in [k for k in sys.modules if "serve_ultra" in k]:
        del sys.modules[key]
    import serve_ultra as su

    su.CONDUCTOR_FN = lambda msgs: (ultra.CANNED, 17, 29)
    su.WORKER = ultra.MockWorker()
    su.SLOT_LABELS = ultra.DEFAULT_SLOT_LABELS

    server = HTTPServer(("127.0.0.1", 0), su.Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        data = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=5).read()

        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/workers", timeout=3)
        w0 = json.loads(resp.read())["workers"][0]
        assert w0["slot_id"] == 0, f"slot 0 must be conductor, got {w0['slot_id']}"
        assert w0["prompt_tokens"] == 17, f"expected 17 prompt tokens, got {w0['prompt_tokens']}"
        assert w0["compl_tokens"] == 29, f"expected 29 compl tokens, got {w0['compl_tokens']}"
    finally:
        server.shutdown()
    print("  PASS  test_conductor_token_accounting")


def test_cond_metrics_ensure_resyncs_model_id():
    """Regression: SlotMetrics.ensure() must refresh model_id on repeat calls,
    not just on first insert. serve_ultra pre-registers slot 0 with a "" model
    id at module load (so /v1/workers always lists the conductor row, even
    before main() runs), then main() calls ensure(0, real_id) once CLI args
    are parsed. If ensure() were a first-seen-only no-op, the conductor's
    model_id would stay "" forever in production."""
    from serving import SlotMetrics
    m = SlotMetrics()
    m.ensure(0, "")
    assert m.snapshot_rows()[0]["model_id"] == ""
    m.ensure(0, "gpt-5-conductor")
    row = m.snapshot_rows()[0]
    assert row["model_id"] == "gpt-5-conductor", row
    assert row["hits"] == 0, "ensure() must not touch counters"
    print("  PASS  test_cond_metrics_ensure_resyncs_model_id")


def test_v1_workers_handles_uninstrumented_worker():
    """Regression: GET /v1/workers must not 500 when WORKER is a raw callable
    (no .metrics attribute) — e.g. before main() wraps it in
    InstrumentedWorker, or in a test/script that assigns WORKER directly."""
    import sys, threading, json, urllib.request
    from http.server import HTTPServer

    for key in [k for k in sys.modules if "serve_ultra" in k]:
        del sys.modules[key]
    import serve_ultra as su

    su.WORKER = ultra.MockWorker()  # no .metrics — raw, uninstrumented

    server = HTTPServer(("127.0.0.1", 0), su.Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/workers", timeout=3)
        assert resp.status == 200
        body = json.loads(resp.read())
        assert isinstance(body["workers"], list)
    finally:
        server.shutdown()
    print("  PASS  test_v1_workers_handles_uninstrumented_worker")


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
    test_v1_workers_returns_valid_structure()
    test_conductor_token_accounting()
    test_cond_metrics_ensure_resyncs_model_id()
    test_v1_workers_handles_uninstrumented_worker()
    print("\nAll 9 tests passed.")
