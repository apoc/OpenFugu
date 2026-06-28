#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
"""
serve_ultra.py — Fugu-Ultra as a single OpenAI-compatible model endpoint.

Mirrors serve.py's ThreadingHTTPServer shape.  Uses ultra.py's API:
  conductor_prompt / LiteLLMWorker.conduct (or LocalConductor.conduct)
  / parse_workflow / ConductorExecutor.execute

Run (litellm conductor + litellm workers):
  python serve_ultra.py \\
    --conductor openai/gpt-4o \\
    --slot-models "openai/gpt-4o,anthropic/claude-3-5-haiku" \\
    --port 8089

Run (local conductor + local workers):
  python serve_ultra.py \\
    --local-conductor /path/to/ckpt \\
    --local-models "/path/llama@cuda:0,/path/gemma@cuda:1" \\
    --port 8089
"""
from __future__ import annotations
import argparse, json, os, sys, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultra import (
    ConductorExecutor,
    LiteLLMWorker,
    LocalConductor,
    LocalPoolWorker,
    MockWorker,
    conductor_prompt,
    parse_workflow,
    DEFAULT_SLOT_LABELS,
    _parse_local_specs,  # module-level helper; same package, intentional import
)

import threading as _threading, time as _time

# ── Conductor slot (slot_id=0) ──────────────────────────────────────────────────
_COND_LOCK       = _threading.Lock()
_cond_hits        = 0
_cond_prompt_tok  = 0
_cond_compl_tok   = 0
_cond_latency_ms  = 0
_cond_lat_hist    = [0] * 11
_cond_model_id    = ""  # set in main()

# ── DAG execution worker slots (slot_id = internal_agent_id + 1) ───────────────
_SLOT_LOCK    = _threading.Lock()
_slot_hits       = {}
_slot_prompt_tok = {}
_slot_compl_tok  = {}
_slot_latency_ms = {}
_slot_lat_hist   = {}
_slot_model_ids  = {}
_slot_status     = {}

_LAT_BINS = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 15_000, 20_000, 30_000]


def _lat_bin(ms: float) -> int:
    for i, upper in enumerate(_LAT_BINS):
        if ms < upper:
            return i
    return 10


def _ensure_slot(sid: int, mid: str) -> None:
    if sid not in _slot_hits:
        _slot_hits[sid]       = 0
        _slot_prompt_tok[sid] = 0
        _slot_compl_tok[sid]  = 0
        _slot_latency_ms[sid] = 0
        _slot_lat_hist[sid]   = [0] * 11
        _slot_model_ids[sid]  = mid
        _slot_status[sid]     = "ok"


def _record_slot(sid: int, mid: str, lat_ms: float,
                 prompt_tok: int, compl_tok: int, ok: bool) -> None:
    with _SLOT_LOCK:
        _ensure_slot(sid, mid)
        _slot_hits[sid]       += 1
        _slot_prompt_tok[sid] += prompt_tok
        _slot_compl_tok[sid]  += compl_tok
        _slot_latency_ms[sid] += int(lat_ms)
        _slot_lat_hist[sid][_lat_bin(lat_ms)] += 1
        _slot_status[sid] = "ok" if ok else "error"


class _InstrumentedWorker:
    """Wraps WORKER; slot_id in /v1/workers = internal_agent_id + 1 (slot 0 = conductor)."""
    def __init__(self, inner, slot_labels):
        self.inner       = inner
        self.slot_labels = slot_labels
        with _SLOT_LOCK:
            for i, mid in enumerate(slot_labels):
                _ensure_slot(i, mid)

    def __call__(self, subtask, messages, agent_id):
        raw_sid  = agent_id % len(self.slot_labels)
        model_id = self.slot_labels[raw_sid]
        t0 = _time.monotonic()
        ok = True
        prompt_tok = compl_tok = 0
        try:
            if isinstance(self.inner, LiteLLMWorker):
                kw = dict(
                    model=self.inner.slot_models[raw_sid],
                    messages=[{"role": m["role"], "content": m["content"]}
                               for m in messages],
                    max_tokens=self.inner.max_tokens,
                    temperature=self.inner.temperature,
                )
                if self.inner.api_key:  kw["api_key"]  = self.inner.api_key
                if self.inner.api_base: kw["api_base"] = self.inner.api_base
                r = self.inner.litellm.completion(**kw)
                usage      = getattr(r, "usage", None) or object()
                prompt_tok = getattr(usage, "prompt_tokens",     0) or 0
                compl_tok  = getattr(usage, "completion_tokens", 0) or 0
                result = r.choices[0].message.content or ""
            else:
                result = self.inner(subtask, messages, agent_id)
        except Exception:
            ok = False
            raise
        finally:
            _record_slot(raw_sid, model_id,
                         (_time.monotonic() - t0) * 1_000,
                         prompt_tok, compl_tok, ok)
        return result

    def stream(self, subtask, messages, agent_id):
        """Streaming variant: records total latency and token count across the call."""
        raw_sid  = agent_id % len(self.slot_labels)
        model_id = self.slot_labels[raw_sid]
        t0 = _time.monotonic()
        tokens: list[str] = []
        ok = True
        try:
            for tok in self.inner.stream(subtask, messages, raw_sid):
                tokens.append(tok)
                yield tok
        except Exception:
            ok = False
            raise
        finally:
            lat_ms = (_time.monotonic() - t0) * 1_000
            _record_slot(raw_sid, model_id, lat_ms, 0, len(tokens), ok)

# Set by main() before the server starts.
CONDUCTOR_FN = None  # callable(messages: list[dict]) -> str
WORKER = None  # callable(subtask, messages, agent_id) -> str
SLOT_LABELS = DEFAULT_SLOT_LABELS
MODEL_NAME = "fugu-ultra"


def _chat_response(text: str, model: str, steps: int) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_steps": steps},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: dict, extra_headers: dict | None = None):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _sse_chunk(self, req_id: str, created: int, model: str,
                   delta: dict, finish_reason=None, usage=None):
        """Write one OpenAI chat.completion.chunk SSE event and flush."""
        event: dict = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            event["usage"] = usage
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": MODEL_NAME, "object": "model", "owned_by": "openfugu"}
                    ],
                },
            )
        elif self.path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": MODEL_NAME})
        elif self.path == "/v1/workers":
            workers = []
            # Slot 0: conductor
            with _COND_LOCK:
                ch = list(_cond_lat_hist)
            workers.append({
                "slot_id": 0, "model_id": _cond_model_id, "status": "ok",
                "hits": _cond_hits,
                "prompt_tokens": _cond_prompt_tok,
                "compl_tokens":  _cond_compl_tok,
                "total_latency_ms": _cond_latency_ms,
                "lat_h0": ch[0],  "lat_h1": ch[1],  "lat_h2": ch[2],
                "lat_h3": ch[3],  "lat_h4": ch[4],  "lat_h5": ch[5],
                "lat_h6": ch[6],  "lat_h7": ch[7],  "lat_h8": ch[8],
                "lat_h9": ch[9],  "lat_h10": ch[10],
            })
            # Slots 1+: DAG execution workers
            with _SLOT_LOCK:
                for raw_sid in sorted(_slot_hits):
                    h = _slot_lat_hist[raw_sid]
                    workers.append({
                        "slot_id":          raw_sid + 1,
                        "model_id":         _slot_model_ids[raw_sid],
                        "status":           _slot_status[raw_sid],
                        "hits":             _slot_hits[raw_sid],
                        "prompt_tokens":    _slot_prompt_tok[raw_sid],
                        "compl_tokens":     _slot_compl_tok[raw_sid],
                        "total_latency_ms": _slot_latency_ms[raw_sid],
                        "lat_h0": h[0],  "lat_h1": h[1],  "lat_h2": h[2],
                        "lat_h3": h[3],  "lat_h4": h[4],  "lat_h5": h[5],
                        "lat_h6": h[6],  "lat_h7": h[7],  "lat_h8": h[8],
                        "lat_h9": h[9],  "lat_h10": h[10],
                    })
            self._send(200, {"model": MODEL_NAME, "workers": workers})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            messages = req.get("messages", [])
            if not messages:
                self._send(400, {"error": "messages required"})
                return

            query = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            stream = bool(req.get("stream", False))
            model  = req.get("model", MODEL_NAME)

            # ── conductor plan (always buffered) ──────────────────────────────
            _t_cond = _time.monotonic()
            try:
                completion = CONDUCTOR_FN(conductor_prompt(query, SLOT_LABELS))
            finally:
                _lat_cond_ms = (_time.monotonic() - _t_cond) * 1_000
                with _COND_LOCK:
                    global _cond_hits, _cond_latency_ms
                    _cond_hits       += 1
                    _cond_latency_ms += int(_lat_cond_ms)
                    _cond_lat_hist[_lat_bin(_lat_cond_ms)] += 1

            mids, subs, acc = parse_workflow(completion)

            if not stream:
                # ── buffered path (unchanged) ──────────────────────────────────
                if not subs:
                    self._send(200, _chat_response(completion, model, 0),
                               {"X-Fugu-Warning": "no-workflow-parsed"})
                    return
                res = ConductorExecutor(WORKER, slot_labels=SLOT_LABELS).execute(mids, subs, acc)
                self._send(200, _chat_response(res.final, model, len(res.steps)))
                return

            # ── SSE streaming path ─────────────────────────────────────────────
            # Headers sent after planning so X-Fugu-Warning can be a real header.
            req_id  = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(_time.time())
            no_wf   = not subs

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            if no_wf:
                self.send_header("X-Fugu-Warning", "no-workflow-parsed")
            self.end_headers()

            try:
                if no_wf:
                    self._sse_chunk(req_id, created, model, {"role": "assistant", "content": ""})
                    words = completion.split()
                    for i, word in enumerate(words):
                        self._sse_chunk(req_id, created, model,
                                        {"content": word if i == 0 else " " + word})
                    n_words = len(words)
                else:
                    self._sse_chunk(req_id, created, model, {"role": "assistant", "content": ""})
                    tok_count: list[int] = [0]

                    def on_token(tok: str) -> None:
                        tok_count[0] += 1
                        self._sse_chunk(req_id, created, model, {"content": tok})

                    has_stream = hasattr(WORKER, "stream")
                    res = ConductorExecutor(WORKER, slot_labels=SLOT_LABELS).execute(
                        mids, subs, acc,
                        stream_last_step=WORKER.stream if has_stream else None,
                        on_token=on_token,
                    )

                    if not has_stream:
                        words = res.final.split()
                        for i, word in enumerate(words):
                            self._sse_chunk(req_id, created, model,
                                            {"content": word if i == 0 else " " + word})
                        n_words = len(words)
                    else:
                        n_words = tok_count[0]

                self._sse_chunk(req_id, created, model, {}, finish_reason="stop",
                                usage={"prompt_tokens": 0,
                                       "completion_tokens": n_words,
                                       "total_tokens": n_words})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            except Exception as e:
                err = {"error": {"message": str(e), "type": "sidecar_error"}}
                self.wfile.write(f"data: {json.dumps(err)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass  # quiet by default


def main():
    global CONDUCTOR_FN, WORKER, SLOT_LABELS, _cond_model_id

    ap = argparse.ArgumentParser(
        description="Serve Fugu-Ultra as one OpenAI-compatible model endpoint."
    )
    ap.add_argument(
        "--conductor",
        metavar="LITELLM_MODEL",
        help="litellm model id for the conductor (prompt-engineered stand-in)",
    )
    ap.add_argument(
        "--local-conductor",
        metavar="PATH",
        help="path to a trained local Conductor checkpoint",
    )
    ap.add_argument("--conductor-device", default="cuda:0")
    ap.add_argument(
        "--slot-models",
        metavar="CSV",
        help="litellm worker model ids (comma-separated)",
    )
    ap.add_argument(
        "--local-models",
        metavar="CSV",
        help="local HF worker model paths (path or path@device)",
    )
    ap.add_argument("--port", type=int, default=8089)
    args = ap.parse_args()

    if not args.conductor and not args.local_conductor:
        ap.error("need --conductor <litellm_id> or --local-conductor <path>")

    # --- Worker pool ---
    if args.local_models:
        try:
            import torch

            n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            n_gpu = 0
        specs = _parse_local_specs(args.local_models, n_gpu)
        WORKER = LocalPoolWorker(specs)
        SLOT_LABELS = [name for name, _, _ in specs]
        print(
            f"[serve_ultra] worker pool: LOCAL ({len(specs)}): {SLOT_LABELS}",
            flush=True,
        )
    elif args.slot_models:
        slots = args.slot_models.split(",")
        WORKER = LiteLLMWorker(slot_models=slots)
        SLOT_LABELS = slots
        print(
            f"[serve_ultra] worker pool: litellm ({len(slots)} slots)", flush=True
        )
    else:
        WORKER = MockWorker()
        SLOT_LABELS = DEFAULT_SLOT_LABELS
        print(
            "[serve_ultra] worker pool: MOCK (no --slot-models / --local-models)",
            flush=True,
        )

    # --- Conductor callable ---
    # Both paths produce callable(messages: list[dict]) -> str.
    if args.local_conductor:
        cond = LocalConductor(args.local_conductor, device=args.conductor_device)
        CONDUCTOR_FN = cond.conduct  # conduct(messages) -> str
        print(f"[serve_ultra] conductor: LOCAL {args.local_conductor}", flush=True)
    else:
        # LiteLLMWorker.conduct(model, messages) bumps max_tokens to 2048 for
        # planning.  Reuse WORKER's client when it's already a LiteLLMWorker;
        # else create a fresh one.
        lw = WORKER if isinstance(WORKER, LiteLLMWorker) else LiteLLMWorker()
        conductor_model = args.conductor
        CONDUCTOR_FN = lambda msgs: lw.conduct(conductor_model, msgs)
        print(f"[serve_ultra] conductor: litellm {conductor_model}", flush=True)

    _cond_model_id = args.conductor or (args.local_conductor or "conductor")
    _slot_labels = list(SLOT_LABELS) if SLOT_LABELS else ["slot-0"]
    WORKER = _InstrumentedWorker(WORKER, _slot_labels)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(
        f"[serve_ultra] Fugu-Ultra listening on :{args.port} "
        f"— POST /v1/chat/completions",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
