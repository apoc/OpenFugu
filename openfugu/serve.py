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

Run:
  FUGU_API_KEY=... FUGU_BASE_URL=... \
  python serve.py --model <qwen3-0.6b dir> --vector model_iter_60.npy \
                  --slot-models <csv of litellm worker ids> --port 8088

Query:
  curl localhost:8088/v1/chat/completions -d '{"messages":[{"role":"user","content":"..."}]}'
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, uuid
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# reuse the faithful implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini import (FuguRouter, Coordinator, LiteLLMWorker, MockWorker,
                  DEFAULT_SLOT_LABELS, HEAD_ROWS, HIDDEN)

import threading as _threading, time as _time

# ── Per-slot cumulative counters ─────────────────────────────────────────────────
# key: slot_id = agent_id % len(slot_models). Updated under _SLOT_LOCK.
_SLOT_LOCK    = _threading.Lock()
_slot_hits       = {}  # slot_id -> int
_slot_prompt_tok = {}  # slot_id -> int
_slot_compl_tok  = {}  # slot_id -> int
_slot_latency_ms = {}  # slot_id -> int
_slot_lat_hist   = {}  # slot_id -> list[int] (11 bins)
_slot_model_ids  = {}  # slot_id -> str
_slot_status     = {}  # slot_id -> str

_LAT_BINS = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 15_000, 20_000, 30_000]


def _lat_bin(ms: float) -> int:
    for i, upper in enumerate(_LAT_BINS):
        if ms < upper:
            return i
    return 10


def _ensure_slot(slot_id: int, model_id: str) -> None:
    """Initialise counters for slot_id if first seen. Call while holding _SLOT_LOCK."""
    if slot_id not in _slot_hits:
        _slot_hits[slot_id]       = 0
        _slot_prompt_tok[slot_id] = 0
        _slot_compl_tok[slot_id]  = 0
        _slot_latency_ms[slot_id] = 0
        _slot_lat_hist[slot_id]   = [0] * 11
        _slot_model_ids[slot_id]  = model_id
        _slot_status[slot_id]     = "ok"


def _record_slot(slot_id: int, model_id: str, lat_ms: float,
                 prompt_tok: int, compl_tok: int, ok: bool) -> None:
    with _SLOT_LOCK:
        _ensure_slot(slot_id, model_id)
        _slot_hits[slot_id]       += 1
        _slot_prompt_tok[slot_id] += prompt_tok
        _slot_compl_tok[slot_id]  += compl_tok
        _slot_latency_ms[slot_id] += int(lat_ms)
        _slot_lat_hist[slot_id][_lat_bin(lat_ms)] += 1
        _slot_status[slot_id] = "ok" if ok else "error"


class _InstrumentedWorker:
    """Wraps any worker callable to record per-slot timing and usage.

    For LiteLLMWorker: re-implements the litellm call to capture usage fields.
    For MockWorker / LocalPoolWorker: records timing only (token counts = 0).
    """
    def __init__(self, inner, slot_labels):
        self.inner = inner
        self.slot_labels = slot_labels
        # Pre-initialise so /v1/workers shows all slots at startup.
        with _SLOT_LOCK:
            for sid, mid in enumerate(slot_labels):
                _ensure_slot(sid, mid)

    def __call__(self, role_name, messages, agent_id):
        slot_id  = agent_id % len(self.slot_labels)
        model_id = self.slot_labels[slot_id]
        t0 = _time.monotonic()
        ok = True
        prompt_tok = compl_tok = 0
        try:
            if isinstance(self.inner, LiteLLMWorker):
                kw = dict(
                    model=self.inner.slot_models[slot_id],
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
                result = self.inner(role_name, messages, agent_id)
        except Exception:
            ok = False
            raise
        finally:
            lat_ms = (_time.monotonic() - t0) * 1_000
            _record_slot(slot_id, model_id, lat_ms, prompt_tok, compl_tok, ok)
        return result

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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "openfugu"}]})
        elif self.path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": MODEL_NAME})
        elif self.path == "/v1/workers":
            with _SLOT_LOCK:
                workers = []
                for sid in sorted(_slot_hits):
                    h = _slot_lat_hist[sid]
                    workers.append({
                        "slot_id":          sid,
                        "model_id":         _slot_model_ids[sid],
                        "status":           _slot_status[sid],
                        "hits":             _slot_hits[sid],
                        "prompt_tokens":    _slot_prompt_tok[sid],
                        "compl_tokens":     _slot_compl_tok[sid],
                        "total_latency_ms": _slot_latency_ms[sid],
                        "lat_h0":  h[0],  "lat_h1":  h[1],  "lat_h2":  h[2],
                        "lat_h3":  h[3],  "lat_h4":  h[4],  "lat_h5":  h[5],
                        "lat_h6":  h[6],  "lat_h7":  h[7],  "lat_h8":  h[8],
                        "lat_h9":  h[9],  "lat_h10": h[10],
                    })
            self._send(200, {"model": MODEL_NAME, "workers": workers})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"}); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            messages = req.get("messages", [])
            if not messages:
                self._send(400, {"error": "messages required"}); return
            # the user query = last user message; coordinator runs the full loop
            query = next((m["content"] for m in reversed(messages)
                          if m.get("role") == "user"), "")
            coord = Coordinator(ROUTER, WORKER, max_turns=MAX_TURNS, sample=True)
            res = coord.run(query, verbose=False)
            self._send(200, _chat_response(res.final, req.get("model", MODEL_NAME),
                                           len(res.turns)))
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):       # quiet
        pass


class LocalPoolWorker:
    """Serving-time local worker pool — the same protocol the per-step trainer
    used. The Coordinator calls (role_name, messages, agent_id) -> reply; we
    dispatch to model[agent_id % n], each model resident on its own GPU. Replies
    are decoded greedily so serving is deterministic. No external API."""
    def __init__(self, specs, max_new=384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.max_new = torch, max_new
        self.names, self.toks, self.models, self.devs = [], [], [], []
        for name, path, dev in specs:
            tk = AutoTokenizer.from_pretrained(path)
            if tk.pad_token is None:
                tk.pad_token = tk.eos_token
            try:
                m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
            except TypeError:
                m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(dev).eval()
            self.names.append(name); self.toks.append(tk); self.models.append(m); self.devs.append(dev)

    def __call__(self, role_name, messages, agent_id):
        torch = self.torch
        wid = agent_id % len(self.models)
        tk, model, dev = self.toks[wid], self.models[wid], self.devs[wid]
        try:
            text = tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "\n".join(m["content"] for m in messages)
        ids = tk(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=self.max_new, do_sample=False,
                                 pad_token_id=tk.pad_token_id)
        return tk.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


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

    if args.local_models:                          # real local worker pool (no API)
        specs = []
        n_gpu = ROUTER.torch.cuda.device_count() if ROUTER.torch.cuda.is_available() else 0
        for i, entry in enumerate(args.local_models.split(",")):
            if "@" in entry:
                path, dev = entry.rsplit("@", 1)
            else:
                path = entry
                dev = f"cuda:{(i % max(n_gpu - 1, 1)) + 1}" if n_gpu > 1 else "cpu"
            specs.append((os.path.basename(path.rstrip("/")) or f"w{i}", path, dev))
        WORKER = LocalPoolWorker(specs)
        print(f"[serve] worker pool: LOCAL ({len(specs)}): "
              f"{[n for n,_,_ in specs]}", flush=True)
    elif args.slot_models:
        WORKER = LiteLLMWorker(slot_models=args.slot_models.split(","))
        print(f"[serve] worker pool: litellm ({len(args.slot_models.split(','))} slots)", flush=True)
    else:
        WORKER = MockWorker()
        print("[serve] worker pool: MOCK (no --slot-models / --local-models given)", flush=True)

    # Determine slot labels for instrumentation.
    if isinstance(WORKER, LiteLLMWorker):
        _slot_labels = list(WORKER.slot_models)
    else:
        _slot_labels = [f"slot-{i}" for i in range(7)]
    WORKER = _InstrumentedWorker(WORKER, _slot_labels)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[serve] Fugu listening on :{args.port} — POST /v1/chat/completions", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
