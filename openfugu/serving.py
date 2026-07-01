#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
"""
serving.py — shared HTTP-serving infrastructure for OpenFugu's OpenAI-compatible
sidecars (serve.py / serve_ultra.py / future endpoints).

serve.py (TRINITY per-turn routing) and serve_ultra.py (Conductor DAG
execution) are two different orchestration engines wearing the same serving
skin: per-slot metrics, SSE streaming, worker instrumentation, local-model
pool loading, and CLI/boot boilerplate were identical or near-identical in
both. This module is the single source of truth for that skin so a third
endpoint doesn't re-copy it again.

Deliberately engine-agnostic: nothing here imports mini.py or ultra.py.
Worker-type dispatch (e.g. "is this a LiteLLMWorker") is duck-typed on shape,
not `isinstance`, so this module never needs to know which engine is calling
it.

What stays OUT of this module: anything that encodes an orchestration
algorithm (TRINITY's Coordinator loop, Conductor's DAG executor) or a
wire-response shape specific to one engine (each serve_*.py's _chat_response
differs: fugu_turns vs total_steps). Those stay in mini.py / ultra.py / each
serve_*.py.
"""
from __future__ import annotations
import json, os, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── Latency histogram ─────────────────────────────────────────────────────────
LAT_BINS = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 15_000, 20_000, 30_000]


def lat_bin(ms: float) -> int:
    """Map a latency in milliseconds to a histogram bin index 0..10 (10 = overflow)."""
    for i, upper in enumerate(LAT_BINS):
        if ms < upper:
            return i
    return 10


# ── Whitespace-preserving SSE chunking ────────────────────────────────────────
def stream_chunks(text: str) -> tuple[list[str], int]:
    """Split text into whitespace-preserving chunks for SSE fallback streaming.

    Concatenating the returned chunks reproduces `text` exactly (no whitespace
    collapse) — unlike `text.split()` + " ".join, which corrupts code/markdown
    containing newlines or multi-space runs. Returns (chunks, word_count)
    where word_count is used for the reported completion-token estimate.
    """
    return re.findall(r"\s+|\S+", text), len(text.split())


# ── Per-slot metrics registry ─────────────────────────────────────────────────
class SlotMetrics:
    """Thread-safe cumulative per-slot counters, polled by GET /v1/workers.

    One instance covers one flat namespace of slot ids starting at 0. A server
    with a single worker pool (serve.py) uses one instance. A server with a
    separate conductor "slot" plus a worker pool (serve_ultra.py) uses two
    instances — instance boundaries are the caller's choice, not this class's.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._hits: dict[int, int] = {}
        self._prompt_tok: dict[int, int] = {}
        self._compl_tok: dict[int, int] = {}
        self._latency_ms: dict[int, int] = {}
        self._lat_hist: dict[int, list[int]] = {}
        self._model_id: dict[int, str] = {}
        self._status: dict[int, str] = {}

    def _ensure_locked(self, slot_id: int, model_id: str) -> None:
        """Initialise counters for slot_id if first seen. Caller holds the lock."""
        if slot_id not in self._hits:
            self._hits[slot_id] = 0
            self._prompt_tok[slot_id] = 0
            self._compl_tok[slot_id] = 0
            self._latency_ms[slot_id] = 0
            self._lat_hist[slot_id] = [0] * 11
            self._model_id[slot_id] = model_id
            self._status[slot_id] = "ok"

    def ensure(self, slot_id: int, model_id: str) -> None:
        """Pre-register a slot so /v1/workers lists it with zero counters
        before any request lands (e.g. the conductor slot at startup).

        Idempotent but NOT a pure no-op on repeat calls: the label is
        resynced to `model_id` every time (counters are untouched). This
        lets a caller register a placeholder label immediately at import
        time — so the slot is always listed — then update it to the real
        label once known (e.g. after CLI args are parsed in main())."""
        with self._lock:
            self._ensure_locked(slot_id, model_id)
            self._model_id[slot_id] = model_id

    def ensure_many(self, slot_labels: list[str]) -> None:
        with self._lock:
            for slot_id, model_id in enumerate(slot_labels):
                self._ensure_locked(slot_id, model_id)

    def record(self, slot_id: int, model_id: str, lat_ms: float,
               prompt_tok: int, compl_tok: int, ok: bool) -> None:
        with self._lock:
            self._ensure_locked(slot_id, model_id)
            self._hits[slot_id] += 1
            self._prompt_tok[slot_id] += prompt_tok
            self._compl_tok[slot_id] += compl_tok
            self._latency_ms[slot_id] += int(lat_ms)
            self._lat_hist[slot_id][lat_bin(lat_ms)] += 1
            self._status[slot_id] = "ok" if ok else "error"

    def snapshot_rows(self, slot_id_offset: int = 0) -> list[dict]:
        """One dict per known slot, ascending slot-id order, shaped for the
        /v1/workers wire format. `slot_id_offset` shifts the reported
        slot_id (serve_ultra.py reserves 0 for the conductor, so worker
        slots report raw_sid + 1)."""
        with self._lock:
            rows = []
            for sid in sorted(self._hits):
                h = self._lat_hist[sid]
                rows.append({
                    "slot_id":          sid + slot_id_offset,
                    "model_id":         self._model_id[sid],
                    "status":           self._status[sid],
                    "hits":             self._hits[sid],
                    "prompt_tokens":    self._prompt_tok[sid],
                    "compl_tokens":     self._compl_tok[sid],
                    "total_latency_ms": self._latency_ms[sid],
                    "lat_h0": h[0],  "lat_h1": h[1],  "lat_h2": h[2],
                    "lat_h3": h[3],  "lat_h4": h[4],  "lat_h5": h[5],
                    "lat_h6": h[6],  "lat_h7": h[7],  "lat_h8": h[8],
                    "lat_h9": h[9],  "lat_h10": h[10],
                })
            return rows


# ── Worker instrumentation ────────────────────────────────────────────────────
def _is_litellm_worker(inner) -> bool:
    """Duck-type check: both mini.LiteLLMWorker and ultra.LiteLLMWorker have
    this shape; MockWorker/LocalPoolWorker don't. Avoids importing either
    engine module here (this module stays engine-agnostic)."""
    return hasattr(inner, "litellm") and hasattr(inner, "slot_models")


class InstrumentedWorker:
    """Wraps any worker callable to record per-slot timing and usage into a
    SlotMetrics registry.

    For a LiteLLMWorker (either engine's): re-implements the litellm call
    directly to capture usage fields. For any other worker: records timing
    only via __call__ (token counts = 0), or accumulated stream token count
    via .stream() when the inner worker supports it.

    `metrics` defaults to a fresh private SlotMetrics if not supplied. Callers
    that need /v1/workers to report these counters should read them back via
    `worker.metrics.snapshot_rows()` rather than holding a separate registry
    reference — that way "whatever WORKER is currently wired up" and "what
    /v1/workers reports" can never drift apart, even when a test wraps a
    throwaway worker instance.
    """

    def __init__(self, inner, slot_labels: list[str], metrics: SlotMetrics | None = None):
        self.inner = inner
        self.slot_labels = slot_labels
        self.metrics = metrics if metrics is not None else SlotMetrics()
        self.metrics.ensure_many(slot_labels)

    def __call__(self, task, messages, agent_id):
        slot_id  = agent_id % len(self.slot_labels)
        model_id = self.slot_labels[slot_id]
        t0 = time.monotonic()
        ok = True
        prompt_tok = compl_tok = 0
        try:
            if _is_litellm_worker(self.inner):
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
                result = self.inner(task, messages, agent_id)
        except Exception:
            ok = False
            raise
        finally:
            lat_ms = (time.monotonic() - t0) * 1_000
            self.metrics.record(slot_id, model_id, lat_ms, prompt_tok, compl_tok, ok)
        return result

    def stream(self, task, messages, agent_id):
        """Streaming variant: records total latency and token count across
        the call. Only meaningful when self.inner has a compatible .stream()
        — callers must check that before wiring this in (see H1 in
        serve_ultra.py's do_POST: detect on the INNER worker, not this
        wrapper, since this wrapper always defines .stream)."""
        slot_id  = agent_id % len(self.slot_labels)
        model_id = self.slot_labels[slot_id]
        t0 = time.monotonic()
        tokens: list[str] = []
        ok = True
        try:
            for tok in self.inner.stream(task, messages, slot_id):
                tokens.append(tok)
                yield tok
        except Exception:
            ok = False
            raise
        finally:
            lat_ms = (time.monotonic() - t0) * 1_000
            self.metrics.record(slot_id, model_id, lat_ms, 0, len(tokens), ok)


# ── Local HF worker pool ──────────────────────────────────────────────────────
def parse_local_specs(csv: str, n_gpu: int) -> list[tuple[str, str, str]]:
    """Parse --local-models CSV into (name, path, device) triples.

    Each entry is `path` or `path@device`; bare paths round-robin across
    GPUs 1..n_gpu-1 (GPU 0 reserved, e.g. for a router/conductor), or fall
    back to CPU when fewer than 2 GPUs are visible.
    """
    specs = []
    for i, entry in enumerate(csv.split(",")):
        if "@" in entry:
            path, dev = entry.rsplit("@", 1)
        else:
            path = entry
            dev = f"cuda:{(i % max(n_gpu - 1, 1)) + 1}" if n_gpu > 1 else "cpu"
        specs.append((os.path.basename(path.rstrip("/")) or f"w{i}", path, dev))
    return specs


class LocalPoolWorker:
    """Local worker pool for serving — the same protocol the per-step
    trainers used: (task, messages, agent_id) -> reply; dispatches to
    model[agent_id % n], each model resident on its own GPU. Replies are
    decoded greedily so serving is deterministic. No external API."""

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

    def __call__(self, task, messages, agent_id):
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


# ── main() boilerplate ────────────────────────────────────────────────────────
def build_worker_pool(local_models: str | None, slot_models: str | None,
                       mock_worker_cls, litellm_worker_cls,
                       default_slot_labels: list[str], log_prefix: str):
    """Construct a worker pool from CLI args, in priority order:
    --local-models > --slot-models > mock. Returns (worker, slot_labels).

    `mock_worker_cls`/`litellm_worker_cls`/`default_slot_labels` are supplied
    by the caller (each engine has its own) — this function stays
    engine-agnostic.
    """
    if local_models:
        try:
            import torch
            n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            n_gpu = 0
        specs = parse_local_specs(local_models, n_gpu)
        worker = LocalPoolWorker(specs)
        slot_labels = [name for name, _, _ in specs]
        print(f"[{log_prefix}] worker pool: LOCAL ({len(specs)}): {slot_labels}", flush=True)
    elif slot_models:
        slots = slot_models.split(",")
        worker = litellm_worker_cls(slot_models=slots)
        slot_labels = slots
        print(f"[{log_prefix}] worker pool: litellm ({len(slots)} slots)", flush=True)
    else:
        worker = mock_worker_cls()
        slot_labels = list(default_slot_labels)
        print(f"[{log_prefix}] worker pool: MOCK (no --slot-models / --local-models given)",
              flush=True)
    return worker, slot_labels


def run_threaded_server(handler_cls, port: int, label: str,
                         endpoint: str = "/v1/chat/completions") -> None:
    """Boot a ThreadingHTTPServer and block forever. Shared boot boilerplate
    for every OpenFugu sidecar."""
    srv = ThreadingHTTPServer(("0.0.0.0", port), handler_cls)
    print(f"[{label}] listening on :{port} — POST {endpoint}", flush=True)
    srv.serve_forever()


# ── HTTP handler mixin ────────────────────────────────────────────────────────
class JsonSSEHandler(BaseHTTPRequestHandler):
    """Mixin providing the OpenAI-compatible JSON + SSE plumbing shared by
    every OpenFugu sidecar Handler. Subclasses implement do_GET/do_POST
    (routing and orchestration are engine-specific) and call these helpers.
    """

    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: dict, extra_headers: dict | None = None) -> None:
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
                   delta: dict, finish_reason=None, usage=None) -> None:
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

    def _sse_error_and_done(self, message: str) -> None:
        """Emit a sidecar_error SSE frame + [DONE], swallowing OSError if the
        client already disconnected (nothing more to send)."""
        try:
            err = {"error": {"message": message, "type": "sidecar_error"}}
            self.wfile.write(f"data: {json.dumps(err)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass

    def _drain_body(self) -> None:
        """Read and discard the request body. Call before a short-circuit
        response (e.g. 404) so HTTP/1.1 keep-alive doesn't desync on the
        next request from leftover unread bytes."""
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n:
            self.rfile.read(n)

    def _read_json_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _extract_messages(self, req: dict):
        """Validate `req["messages"]` is a non-empty list of dicts.
        Returns the list, or None (caller should send a 400)."""
        messages = req.get("messages", [])
        if not isinstance(messages, list) or not messages or \
           not all(isinstance(m, dict) for m in messages):
            return None
        return messages

    @staticmethod
    def last_user_content(messages: list[dict]) -> str:
        return next((m["content"] for m in reversed(messages)
                     if m.get("role") == "user"), "")

    def log_message(self, *a):       # quiet by default
        pass
