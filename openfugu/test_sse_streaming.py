#!/usr/bin/env python3
"""Offline tests for SSE streaming: ultra.py, serve_ultra.py, serve.py.
Run: python test_sse_streaming.py
No API keys or GPU needed.
"""
from __future__ import annotations
import http.client, json, os, sys, threading, contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ultra
import serve_ultra
import serve
from mini import MockWorker


# ── SSE helpers ────────────────────────────────────────────────────────────────

def _parse_sse(raw: str) -> list[dict]:
    """Return list of parsed JSON objects from an SSE response body, skipping [DONE]."""
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if block.startswith("data: ") and block[6:] != "[DONE]":
            try:
                events.append(json.loads(block[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _post_stream(port: int, payload: dict) -> tuple:
    """POST JSON to /v1/chat/completions; return (response, [parsed SSE events])."""
    data = json.dumps(payload).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("POST", "/v1/chat/completions", body=data,
                 headers={"Content-Type": "application/json",
                          "Content-Length": str(len(data))})
    resp = conn.getresponse()
    raw = resp.read().decode()
    return resp, _parse_sse(raw)


# ── ultra.py tests ─────────────────────────────────────────────────────────────

def test_litellm_worker_stream_yields_tokens():
    """LiteLLMWorker.stream() calls litellm with stream=True and yields non-empty tokens."""

    class _Delta:
        def __init__(self, content): self.content = content
    class _Choice:
        def __init__(self, c): self.delta = _Delta(c)
    class _Chunk:
        def __init__(self, c): self.choices = [_Choice(c)]
    class _FakeLiteLLM:
        @staticmethod
        def completion(**kw):
            assert kw.get("stream") is True, "stream=True must be passed"
            assert kw["model"] == "fake-model"
            return [_Chunk("Hello"), _Chunk(""), _Chunk(" world")]

    w = ultra.LiteLLMWorker(slot_models=["fake-model"])
    w.litellm = _FakeLiteLLM()

    tokens = list(w.stream("subtask", [{"role": "user", "content": "hi"}], 0))
    assert tokens == ["Hello", " world"], f"got {tokens!r}"
    print("  PASS  test_litellm_worker_stream_yields_tokens")


def test_litellm_worker_stream_passes_api_kwargs():
    """LiteLLMWorker.stream() forwards api_key and api_base when set."""

    class _Chunk:
        choices = [type("C", (), {"delta": type("D", (), {"content": "x"})()})()]

    captured = {}
    class _FakeLiteLLM:
        @staticmethod
        def completion(**kw):
            captured.update(kw)
            return [_Chunk()]

    w = ultra.LiteLLMWorker(slot_models=["m"], api_key="k", api_base="http://b")
    w.litellm = _FakeLiteLLM()
    list(w.stream("s", [], 0))

    assert captured.get("api_key") == "k"
    assert captured.get("api_base") == "http://b"
    assert captured.get("stream") is True
    print("  PASS  test_litellm_worker_stream_passes_api_kwargs")


def test_conductor_executor_streams_last_step():
    """ConductorExecutor calls stream_last_step for step N-1 and accumulates tokens."""
    calls = []

    def buffered_worker(subtask, messages, agent_id):
        calls.append(("buf", agent_id % 7))
        return f"r{agent_id}"

    def streaming_worker(subtask, messages, agent_id):
        calls.append(("stream", agent_id % 7))
        yield "final "
        yield "answer"

    tokens = []
    ex = ultra.ConductorExecutor(buffered_worker)
    res = ex.execute([0, 1, 2], ["s0", "s1", "s2"], [[], [0], [0, 1]],
                     stream_last_step=streaming_worker,
                     on_token=tokens.append)

    assert calls == [("buf", 0), ("buf", 1), ("stream", 2)], calls
    assert tokens == ["final ", "answer"], tokens
    assert res.final == "final answer"
    assert len(res.steps) == 3
    print("  PASS  test_conductor_executor_streams_last_step")


def test_conductor_executor_no_stream_unchanged():
    """Without stream_last_step, ConductorExecutor behaves exactly as before."""
    def worker(subtask, messages, agent_id):
        return f"r{agent_id % 7}"

    ex = ultra.ConductorExecutor(worker)
    res = ex.execute([0, 1], ["a", "b"], [[], [0]])
    assert res.final == "r1"
    print("  PASS  test_conductor_executor_no_stream_unchanged")


def test_conductor_executor_single_step_streams():
    """A single-step workflow uses stream_last_step for step 0."""
    collected = []
    def streaming_worker(subtask, messages, agent_id):
        yield "only"
        yield " step"

    ex = ultra.ConductorExecutor(lambda *a: "unused")
    res = ex.execute([0], ["s0"], [[]], stream_last_step=streaming_worker,
                     on_token=collected.append)
    assert res.final == "only step"
    assert collected == ["only", " step"]
    print("  PASS  test_conductor_executor_single_step_streams")


# ── serve_ultra.py tests ────────────────────────────────────────────────────────

@contextlib.contextmanager
def _ultra_globals(conductor_fn, worker, slot_labels=None):
    old_c = serve_ultra.CONDUCTOR_FN
    old_w = serve_ultra.WORKER
    old_s = serve_ultra.SLOT_LABELS
    serve_ultra.CONDUCTOR_FN = conductor_fn
    serve_ultra.WORKER       = worker
    serve_ultra.SLOT_LABELS  = slot_labels or ultra.DEFAULT_SLOT_LABELS
    try:
        yield
    finally:
        serve_ultra.CONDUCTOR_FN = old_c
        serve_ultra.WORKER       = old_w
        serve_ultra.SLOT_LABELS  = old_s


def _start_ultra_server():
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_ultra.Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


CANNED_PLAN = """
model_id: [0, 1]
subtasks: ["step one", "step two"]
access_list: [[], [0]]
"""


class _MockStreamWorker:
    """Step 0 buffered, step 1 streaming."""
    def __call__(self, subtask, messages, agent_id):
        return f"buffered_result_{agent_id % 7}"

    def stream(self, subtask, messages, agent_id):
        yield "final "
        yield "token"


def test_ultra_streaming_returns_sse():
    """serve_ultra: stream=True returns text/event-stream with role + content + stop events."""
    srv, port = _start_ultra_server()
    try:
        with _ultra_globals(lambda _: (CANNED_PLAN, 0, 0), _MockStreamWorker()):
            resp, events = _post_stream(port, {
                "model": "fugu-ultra",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            })
        assert resp.status == 200, resp.status
        assert "text/event-stream" in (resp.getheader("Content-Type") or ""), \
            resp.getheader("Content-Type")
        assert events, "no SSE events received"
        assert events[0]["choices"][0]["delta"].get("role") == "assistant", events[0]
        content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        assert "final" in content and "token" in content, f"content={content!r}"
        last = events[-1]
        assert last["choices"][0]["finish_reason"] == "stop", last
        assert "usage" in last, last
    finally:
        srv.shutdown()
    print("  PASS  test_ultra_streaming_returns_sse")


def test_ultra_streaming_no_workflow_warning_header():
    """serve_ultra: stream=True with unparseable plan → X-Fugu-Warning response header."""
    srv, port = _start_ultra_server()
    try:
        with _ultra_globals(lambda _: ("I cannot plan this.", 0, 0), _MockStreamWorker()):
            resp, events = _post_stream(port, {
                "model": "fugu-ultra",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
        assert resp.status == 200
        warn = resp.getheader("X-Fugu-Warning") or resp.getheader("x-fugu-warning")
        assert warn == "no-workflow-parsed", f"X-Fugu-Warning={warn!r}"
        assert events, "no SSE events received"
        assert events[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        srv.shutdown()
    print("  PASS  test_ultra_streaming_no_workflow_warning_header")


def test_ultra_non_streaming_unchanged():
    """serve_ultra: stream absent still returns buffered JSON."""
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_ultra.Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with _ultra_globals(lambda _: (CANNED_PLAN, 0, 0), _MockStreamWorker()):
            data = json.dumps({
                "model": "fugu-ultra",
                "messages": [{"role": "user", "content": "hello"}],
            }).encode()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            conn.request("POST", "/v1/chat/completions", body=data,
                         headers={"Content-Type": "application/json",
                                  "Content-Length": str(len(data))})
            resp = conn.getresponse()
            body = json.loads(resp.read())
        assert resp.status == 200
        assert "application/json" in (resp.getheader("Content-Type") or "")
        assert "choices" in body
    finally:
        srv.shutdown()
    print("  PASS  test_ultra_non_streaming_unchanged")


def test_ultra_streaming_production_wiring_falls_back_safely():
    """Regression (H1): _InstrumentedWorker always defines .stream, so stream
    detection must check the INNER worker, not the wrapper — else a
    non-streaming pool (MockWorker/LocalPoolWorker) crashes mid-stream in
    production instead of taking the word-chunk fallback."""
    srv, port = _start_ultra_server()
    try:
        wrapped = serve_ultra._InstrumentedWorker(ultra.MockWorker(), ultra.DEFAULT_SLOT_LABELS)
        with _ultra_globals(lambda _: (CANNED_PLAN, 0, 0), wrapped):
            resp, events = _post_stream(port, {
                "model": "fugu-ultra",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            })
        assert resp.status == 200, resp.status
        assert events, "no SSE events received"
        assert not any("error" in e for e in events), f"unexpected error frame: {events}"
        content = "".join(e.get("choices", [{}])[0].get("delta", {}).get("content", "") for e in events)
        assert content, "expected non-empty content from MockWorker fallback"
        assert events[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        srv.shutdown()
    print("  PASS  test_ultra_streaming_production_wiring_falls_back_safely")


def test_ultra_streaming_production_wiring_streams_when_capable():
    """When the inner worker DOES support streaming, production wiring
    (_InstrumentedWorker-wrapped WORKER) must still take the real streaming
    path (not the fallback) and record the slot hit."""
    srv, port = _start_ultra_server()
    try:
        wrapped = serve_ultra._InstrumentedWorker(_MockStreamWorker(), ultra.DEFAULT_SLOT_LABELS)
        with _ultra_globals(lambda _: (CANNED_PLAN, 0, 0), wrapped):
            resp, events = _post_stream(port, {
                "model": "fugu-ultra",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            })
        assert resp.status == 200, resp.status
        content = "".join(e.get("choices", [{}])[0].get("delta", {}).get("content", "") for e in events)
        assert "final" in content and "token" in content, f"content={content!r}"
        import urllib.request as _ur
        workers = json.loads(_ur.urlopen(f"http://127.0.0.1:{port}/v1/workers", timeout=3).read())["workers"]
        # step 1 (last, mid=1) streamed -> raw_sid=1 -> slot_id=2 (slot0=conductor)
        streamed_slot = next((w for w in workers if w["slot_id"] == 2), None)
        assert streamed_slot is not None and streamed_slot["hits"] >= 1, workers
    finally:
        srv.shutdown()
    print("  PASS  test_ultra_streaming_production_wiring_streams_when_capable")


MULTILINE_PLAN = """
model_id: [0]
subtasks: ["write code"]
access_list: [[]]
"""


class _MultilineWorker:
    """Deterministic worker returning whitespace-sensitive text (no .stream)."""
    def __call__(self, subtask, messages, agent_id):
        return "def f(n):\n    return n*2\n\nResult:  72"


def test_ultra_streaming_preserves_whitespace():
    """Regression (H2): the SSE fallback must reconstruct res.final
    byte-for-byte, not collapse newlines/indentation/multi-space runs the way
    str.split() + ' '.join(...) did."""
    srv, port = _start_ultra_server()
    try:
        wrapped = serve_ultra._InstrumentedWorker(_MultilineWorker(), ultra.DEFAULT_SLOT_LABELS)
        with _ultra_globals(lambda _: (MULTILINE_PLAN, 0, 0), wrapped):
            resp, events = _post_stream(port, {
                "model": "fugu-ultra",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            })
        assert resp.status == 200, resp.status
        content = "".join(e.get("choices", [{}])[0].get("delta", {}).get("content", "") for e in events)
        expected = "def f(n):\n    return n*2\n\nResult:  72"
        assert content == expected, f"whitespace corrupted: {content!r} != {expected!r}"
    finally:
        srv.shutdown()
    print("  PASS  test_ultra_streaming_preserves_whitespace")


# ── serve.py tests ─────────────────────────────────────────────────────────────

class _MockTrinityRouter:
    """Always routes to Worker/agent_id=0."""
    def route(self, messages, sample=True, agent_mask=None):
        return {"role_name": "Worker", "agent_id": 0}


@contextlib.contextmanager
def _trinity_globals(router, worker, max_turns=1):
    old_r, old_w, old_t = serve.ROUTER, serve.WORKER, serve.MAX_TURNS
    serve.ROUTER    = router
    serve.WORKER    = worker
    serve.MAX_TURNS = max_turns
    try:
        yield
    finally:
        serve.ROUTER, serve.WORKER, serve.MAX_TURNS = old_r, old_w, old_t


def _start_trinity_server():
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def test_trinity_streaming_returns_sse():
    """serve.py: stream=True word-chunks res.final as SSE delta events."""
    labels = [f"slot-{i}" for i in range(7)]
    worker = serve._InstrumentedWorker(MockWorker(), labels)
    srv, port = _start_trinity_server()
    try:
        with _trinity_globals(_MockTrinityRouter(), worker, max_turns=1):
            resp, events = _post_stream(port, {
                "model": "fugu",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            })
        assert resp.status == 200, resp.status
        assert "text/event-stream" in (resp.getheader("Content-Type") or "")
        assert events
        assert events[0]["choices"][0]["delta"].get("role") == "assistant"
        content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        assert len(content) > 0, "expected non-empty content from MockWorker"
        last = events[-1]
        assert last["choices"][0]["finish_reason"] == "stop"
        assert "usage" in last
    finally:
        srv.shutdown()
    print("  PASS  test_trinity_streaming_returns_sse")


def test_trinity_non_streaming_unchanged():
    """serve.py: stream absent still returns buffered JSON."""
    labels = [f"slot-{i}" for i in range(7)]
    worker = serve._InstrumentedWorker(MockWorker(), labels)
    srv, port = _start_trinity_server()
    try:
        with _trinity_globals(_MockTrinityRouter(), worker, max_turns=1):
            data = json.dumps({
                "model": "fugu",
                "messages": [{"role": "user", "content": "hi"}],
            }).encode()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            conn.request("POST", "/v1/chat/completions", body=data,
                         headers={"Content-Type": "application/json",
                                  "Content-Length": str(len(data))})
            resp = conn.getresponse()
            body = json.loads(resp.read())
        assert resp.status == 200
        assert "choices" in body
    finally:
        srv.shutdown()
    print("  PASS  test_trinity_non_streaming_unchanged")


class _MultilineTrinityWorker:
    """Deterministic worker returning whitespace-sensitive text."""
    def __call__(self, role_name, messages, agent_id):
        return "def f(n):\n    return n*2\n\nResult:  72"


def test_trinity_streaming_matches_buffered_content():
    """Regression (H2): serve.py's SSE fallback used str.split() + ' '.join,
    which collapses newlines/indentation/multi-space runs. The concatenated
    streamed delta.content must equal the buffered res.final exactly."""
    labels = [f"slot-{i}" for i in range(7)]

    # Buffered
    worker_a = serve._InstrumentedWorker(_MultilineTrinityWorker(), labels)
    srv_a, port_a = _start_trinity_server()
    try:
        with _trinity_globals(_MockTrinityRouter(), worker_a, max_turns=1):
            data = json.dumps({"model": "fugu",
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            conn = http.client.HTTPConnection("127.0.0.1", port_a, timeout=30)
            conn.request("POST", "/v1/chat/completions", body=data,
                         headers={"Content-Type": "application/json",
                                  "Content-Length": str(len(data))})
            buffered_body = json.loads(conn.getresponse().read())
        buffered_content = buffered_body["choices"][0]["message"]["content"]
    finally:
        srv_a.shutdown()

    # Streamed
    worker_b = serve._InstrumentedWorker(_MultilineTrinityWorker(), labels)
    srv_b, port_b = _start_trinity_server()
    try:
        with _trinity_globals(_MockTrinityRouter(), worker_b, max_turns=1):
            resp, events = _post_stream(port_b, {
                "model": "fugu",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
        assert resp.status == 200, resp.status
        streamed_content = "".join(
            e["choices"][0]["delta"].get("content", "") for e in events)
    finally:
        srv_b.shutdown()

    assert streamed_content == buffered_content, (
        f"streamed != buffered: {streamed_content!r} != {buffered_content!r}")
    print("  PASS  test_trinity_streaming_matches_buffered_content")


def test_trinity_v1_workers_reflects_execution():
    """serve.py /v1/workers (previously untested) must reflect real request
    traffic: hits, model_id, and lat_hN bins after a served request."""
    labels = [f"slot-{i}" for i in range(7)]
    worker = serve._InstrumentedWorker(MockWorker(), labels)
    srv, port = _start_trinity_server()
    try:
        with _trinity_globals(_MockTrinityRouter(), worker, max_turns=1):
            data = json.dumps({"model": "fugu",
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            conn.request("POST", "/v1/chat/completions", body=data,
                         headers={"Content-Type": "application/json",
                                  "Content-Length": str(len(data))})
            conn.getresponse().read()

            conn2 = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn2.request("GET", "/v1/workers")
            resp = conn2.getresponse()
            body = json.loads(resp.read())
        assert resp.status == 200
        assert body["model"] == "fugu"
        slot0 = next((w for w in body["workers"] if w["slot_id"] == 0), None)
        assert slot0 is not None, f"expected slot_id 0 in {body['workers']}"
        assert slot0["hits"] >= 1, slot0
        assert slot0["model_id"] == "slot-0", slot0
        assert all(f"lat_h{i}" in slot0 for i in range(11)), slot0
        assert sum(slot0[f"lat_h{i}"] for i in range(11)) >= 1, slot0
    finally:
        srv.shutdown()
    print("  PASS  test_trinity_v1_workers_reflects_execution")


# ── runner ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SSE streaming tests (offline — no keys, no GPU)\n")
    tests = [
        test_litellm_worker_stream_yields_tokens,
        test_litellm_worker_stream_passes_api_kwargs,
        test_conductor_executor_streams_last_step,
        test_conductor_executor_no_stream_unchanged,
        test_conductor_executor_single_step_streams,
        test_ultra_streaming_returns_sse,
        test_ultra_streaming_no_workflow_warning_header,
        test_ultra_non_streaming_unchanged,
        test_ultra_streaming_production_wiring_falls_back_safely,
        test_ultra_streaming_production_wiring_streams_when_capable,
        test_ultra_streaming_preserves_whitespace,
        test_trinity_streaming_returns_sse,
        test_trinity_non_streaming_unchanged,
        test_trinity_streaming_matches_buffered_content,
        test_trinity_v1_workers_reflects_execution,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            raise SystemExit(1)
    print(f"\nAll {len(tests)} tests passed.")
