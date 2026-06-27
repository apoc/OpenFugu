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
                (
                    m["content"]
                    for m in reversed(messages)
                    if m.get("role") == "user"
                ),
                "",
            )

            # 1. Ask conductor for a 3-list workflow plan
            completion = CONDUCTOR_FN(conductor_prompt(query, SLOT_LABELS))

            # 2. Parse the 3 lists
            mids, subs, acc = parse_workflow(completion)

            # 3. No parseable workflow — degrade gracefully rather than 500
            if not subs:
                self._send(
                    200,
                    _chat_response(completion, req.get("model", MODEL_NAME), 0),
                    {"X-Fugu-Warning": "no-workflow-parsed"},
                )
                return

            # 4. Execute the DAG
            res = ConductorExecutor(WORKER, slot_labels=SLOT_LABELS).execute(
                mids, subs, acc
            )
            self._send(
                200,
                _chat_response(
                    res.final, req.get("model", MODEL_NAME), len(res.steps)
                ),
            )
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass  # quiet by default


def main():
    global CONDUCTOR_FN, WORKER, SLOT_LABELS

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

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(
        f"[serve_ultra] Fugu-Ultra listening on :{args.port} "
        f"— POST /v1/chat/completions",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
