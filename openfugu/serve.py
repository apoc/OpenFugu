#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: OpenAI-compatible serving layer for the OpenFugu TRINITY coordinator. Original code.
"""
serve.py — Fugu as a single OpenAI-compatible model endpoint.

This is Fugu's real product surface: "one model to command them all". A client
POSTs to /v1/chat/completions as if calling one model; internally the TRINITY
coordinator (Qwen3-0.6B + model_iter_60.npy) routes each turn to a worker from a
real pool (via litellm) and runs the step_trinity loop until a verifier accepts.
The caller never sees the pool.

stdlib http.server only — no FastAPI/uvicorn (ponytail: a router endpoint needs
a socket and a JSON handler, not a web framework).

Serving-layer plumbing (per-slot metrics, SSE streaming, worker instrumentation,
local-model pool loading, CLI/boot boilerplate) lives in serving.py, shared with
serve_ultra.py — see that module's docstring for the split rationale.

Run:
  FUGU_API_KEY=... FUGU_BASE_URL=... \
  python serve.py --model <qwen3-0.6b dir> --vector model_iter_60.npy \
                  --slot-models <csv of litellm worker ids> --port 8088

Query:
  curl localhost:8088/v1/chat/completions -d '{"messages":[{"role":"user","content":"..."}]}'
"""
from __future__ import annotations
import argparse, os, sys, time, uuid
import numpy as np

# reuse the faithful implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini import (FuguRouter, Coordinator, LiteLLMWorker, MockWorker,
                  DEFAULT_SLOT_LABELS, HEAD_ROWS, HIDDEN)
from serving import (InstrumentedWorker, JsonSSEHandler,
                     stream_chunks, build_worker_pool, run_threaded_server)

import time as _time

ROUTER: FuguRouter | None = None
WORKER = None
MODEL_NAME = "fugu"
MAX_TURNS = 5


def _chat_response(text: str, model: str, usage_turns: int) -> dict:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        # surface the orchestration depth without exposing which workers ran
        "usage": {"fugu_turns": usage_turns},
    }


class Handler(JsonSSEHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "openfugu"}]})
        elif self.path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": MODEL_NAME})
        elif self.path == "/v1/workers":
            rows = getattr(WORKER, "metrics", None)
            rows = rows.snapshot_rows() if rows is not None else []
            self._send(200, {"model": MODEL_NAME, "workers": rows})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._drain_body()
            self._send(404, {"error": "not found"}); return
        try:
            req = self._read_json_body()
            messages = self._extract_messages(req)
            if messages is None:
                self._send(400, {"error": "messages required"}); return

            query  = self.last_user_content(messages)
            stream = bool(req.get("stream", False))
            model  = req.get("model", MODEL_NAME)

            # Coordinator runs fully buffered — TRINITY cannot predict the final
            # Worker turn until the Verifier terminates, so streaming mid-run
            # would corrupt delta.content reconstruction.
            coord = Coordinator(ROUTER, WORKER, max_turns=MAX_TURNS, sample=True)
            res   = coord.run(query, verbose=False)

            if not stream:
                self._send(200, _chat_response(res.final, model, len(res.turns)))
                return

            # SSE path: headers sent after coord.run() — no exception can occur
            # after end_headers(), keeping error handling clean.
            req_id  = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(_time.time())

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                self._sse_chunk(req_id, created, model, {"role": "assistant", "content": ""})
                chunks, n_words = stream_chunks(res.final)
                for chunk in chunks:
                    self._sse_chunk(req_id, created, model, {"content": chunk})
                self._sse_chunk(req_id, created, model, {}, finish_reason="stop",
                                usage={"prompt_tokens": 0,
                                       "completion_tokens": n_words,
                                       "total_tokens": n_words})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as e:
                self._sse_error_and_done(str(e))

        except Exception as e:
            self._send(500, {"error": str(e)})


def main():
    global ROUTER, WORKER, MAX_TURNS
    ap = argparse.ArgumentParser(description="Serve Fugu as one OpenAI-compatible model.")
    ap.add_argument("--model", required=True, help="Qwen3-0.6B dir")
    ap.add_argument("--vector", default="model_iter_60.npy",
                    help="base vector (19456) — SVF + head")
    ap.add_argument("--head", default=None,
                    help="optional trained head-only vector (10240); overrides the "
                         "head from --vector after SVF is applied")
    ap.add_argument("--slot-models", metavar="CSV", help="litellm worker ids; omit for mock")
    ap.add_argument("--local-models", metavar="CSV",
                    help="local HF worker model paths (real per-step pool, no API). "
                         "Optional 'path@device' per entry; default round-robin GPUs.")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--max-turns", type=int, default=5)
    args = ap.parse_args()
    MAX_TURNS = args.max_turns

    print(f"[serve] loading TRINITY router ({args.model}) ...", flush=True)
    ROUTER = FuguRouter(args.model, args.vector, seed=0)
    if args.head:                                  # layer a trained head over base SVF
        h = np.load(args.head).astype(np.float64)
        if h.shape != (HEAD_ROWS * HIDDEN,):
            raise ValueError(f"--head must be {HEAD_ROWS * HIDDEN} floats, got {h.shape}")
        ROUTER.head = ROUTER.torch.from_numpy(h.copy()).float().reshape(
            HEAD_ROWS, HIDDEN).to(ROUTER.device)
        print(f"[serve] applied trained head from {args.head}", flush=True)

    WORKER, slot_labels = build_worker_pool(
        args.local_models, args.slot_models,
        mock_worker_cls=MockWorker, litellm_worker_cls=LiteLLMWorker,
        default_slot_labels=DEFAULT_SLOT_LABELS, log_prefix="serve",
    )
    WORKER = InstrumentedWorker(WORKER, slot_labels)

    run_threaded_server(Handler, args.port, "serve")


if __name__ == "__main__":
    main()
