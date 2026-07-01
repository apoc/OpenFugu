#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
"""
serve_ultra.py — Fugu-Ultra as a single OpenAI-compatible model endpoint.

Mirrors serve.py's ThreadingHTTPServer shape.  Uses ultra.py's API:
  conductor_prompt / LiteLLMWorker.conduct (or LocalConductor.conduct)
  / parse_workflow / ConductorExecutor.execute

Serving-layer plumbing (per-slot metrics, SSE streaming, worker instrumentation,
local-model pool loading, CLI/boot boilerplate) lives in serving.py, shared with
serve.py — see that module's docstring for the split rationale.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultra import (
    ConductorExecutor,
    LiteLLMWorker,
    LocalConductor,
    MockWorker,
    conductor_prompt,
    parse_workflow,
    DEFAULT_SLOT_LABELS,
)
from serving import (SlotMetrics, InstrumentedWorker, JsonSSEHandler,
                     stream_chunks, build_worker_pool, run_threaded_server)

import time as _time

# Slot 0 is reserved for the conductor (a module global — there's no wrapper
# object to hang its metrics off). DAG worker slots live on WORKER.metrics
# (InstrumentedWorker's own SlotMetrics) and report raw_sid + 1, read
# dynamically in do_GET so "whatever WORKER is currently wired up" and "what
# /v1/workers reports" can never drift apart — see serving.InstrumentedWorker.
COND_METRICS = SlotMetrics()
COND_METRICS.ensure(0, "")  # always listed, even before main() sets the real model_id

# Set by main() before the server starts.
CONDUCTOR_FN = None  # callable(messages: list[dict]) -> (str, prompt_tok, compl_tok)
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


class Handler(JsonSSEHandler):
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
            # Slot 0: conductor, then DAG execution workers at raw_sid + 1.
            metrics = getattr(WORKER, "metrics", None)
            worker_rows = metrics.snapshot_rows(slot_id_offset=1) if metrics is not None else []
            workers = COND_METRICS.snapshot_rows() + worker_rows
            self._send(200, {"model": MODEL_NAME, "workers": workers})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._drain_body()
            self._send(404, {"error": "not found"})
            return
        try:
            req = self._read_json_body()
            messages = self._extract_messages(req)
            if messages is None:
                self._send(400, {"error": "messages required"})
                return

            query = self.last_user_content(messages)
            stream = bool(req.get("stream", False))
            model  = req.get("model", MODEL_NAME)

            # ── conductor plan (always buffered) ──────────────────────────────
            _t_cond = _time.monotonic()
            cond_ptok = cond_ctok = 0
            cond_ok = True
            try:
                completion, cond_ptok, cond_ctok = CONDUCTOR_FN(conductor_prompt(query, SLOT_LABELS))
            except Exception:
                cond_ok = False
                raise
            finally:
                lat_cond_ms = (_time.monotonic() - _t_cond) * 1_000
                COND_METRICS.record(0, _cond_model_id, lat_cond_ms, cond_ptok, cond_ctok, cond_ok)

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
                    chunks, n_words = stream_chunks(completion)
                    for chunk in chunks:
                        self._sse_chunk(req_id, created, model, {"content": chunk})
                else:
                    self._sse_chunk(req_id, created, model, {"role": "assistant", "content": ""})
                    tok_count: list[int] = [0]

                    def on_token(tok: str) -> None:
                        tok_count[0] += 1
                        self._sse_chunk(req_id, created, model, {"content": tok})

                    # Detect streaming capability on the INNER worker, not the
                    # always-.stream-defining InstrumentedWorker wrapper — else
                    # a non-streaming pool (MockWorker/LocalPoolWorker) crashes
                    # mid-stream when the wrapper's .stream() delegates down.
                    inner_worker = getattr(WORKER, "inner", WORKER)
                    has_stream = hasattr(inner_worker, "stream")
                    res = ConductorExecutor(WORKER, slot_labels=SLOT_LABELS).execute(
                        mids, subs, acc,
                        stream_last_step=WORKER.stream if has_stream else None,
                        on_token=on_token,
                    )

                    if not has_stream:
                        chunks, n_words = stream_chunks(res.final)
                        for chunk in chunks:
                            self._sse_chunk(req_id, created, model, {"content": chunk})
                    else:
                        n_words = tok_count[0]

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


_cond_model_id = ""  # set in main(); read by do_POST's COND_METRICS.record calls


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

    WORKER, SLOT_LABELS = build_worker_pool(
        args.local_models, args.slot_models,
        mock_worker_cls=MockWorker, litellm_worker_cls=LiteLLMWorker,
        default_slot_labels=DEFAULT_SLOT_LABELS, log_prefix="serve_ultra",
    )

    # --- Conductor callable ---
    # Both paths produce callable(messages: list[dict]) -> (str, prompt_tok, compl_tok).
    if args.local_conductor:
        cond = LocalConductor(args.local_conductor, device=args.conductor_device)
        CONDUCTOR_FN = cond.conduct_with_usage
        print(f"[serve_ultra] conductor: LOCAL {args.local_conductor}", flush=True)
    else:
        # LiteLLMWorker.conduct(model, messages) bumps max_tokens to 2048 for
        # planning.  Reuse WORKER's client when it's already a LiteLLMWorker;
        # else create a fresh one.
        lw = WORKER if isinstance(WORKER, LiteLLMWorker) else LiteLLMWorker()
        conductor_model = args.conductor
        CONDUCTOR_FN = lambda msgs: lw.conduct_with_usage(conductor_model, msgs)
        print(f"[serve_ultra] conductor: litellm {conductor_model}", flush=True)

    _cond_model_id = args.conductor or (args.local_conductor or "conductor")
    COND_METRICS.ensure(0, _cond_model_id)
    WORKER = InstrumentedWorker(WORKER, list(SLOT_LABELS))

    run_threaded_server(Handler, args.port, "serve_ultra")


if __name__ == "__main__":
    main()
