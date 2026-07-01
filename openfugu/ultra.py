#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: Learning to Orchestrate Agents in NL with the Conductor (arXiv:2512.04388, Sakana AI). Independent reimplementation of the workflow-DAG executor from the paper; no Sakana source code is copied.
"""
fugu_ultra.py — a faithful, runnable reconstruction of Sakana Fugu-Ultra's
Conductor line: instead of routing one worker per turn (that's fugu_mini.py /
TRINITY), a Conductor LM emits an ENTIRE agentic workflow in one shot — three
equal-length lists (model_id / subtasks / access_list) forming a DAG over a
worker pool — which is then executed in topological order.

Provenance, stated honestly:
  [EXEC]  the execution engine — 3-list parse, DAG order, access-list visibility
          injection — is a faithful reimplementation of the TRINITY/Conductor
          authors' conductor_engine.py + conductor_utils.py.
  [DOC]   the GRPO-trained 7B Conductor weights are NOT public, so here the
          Conductor is a *prompted off-the-shelf model*. The Conductor paper's
          own claim is that prompting works (just below the RL-optimized model);
          this reproduces the mechanism, not the trained policy.

Workers (and the Conductor) run through litellm, so any provider pool works.

Usage:
  python fugu_ultra.py --query "..." --conductor novita/deepseek/deepseek-v4-pro \
      --slot-models <csv of 7 worker model ids>
  python fugu_ultra.py --self-test     # offline: parser + DAG executor on a canned workflow
"""
from __future__ import annotations
import argparse, ast, json, os, re, sys
from dataclasses import dataclass, field
from typing import Callable

N_AGENTS = 7
MAX_STEPS = 5                      # [DOC] Conductor workflows up to 5 steps
_SMART = str.maketrans("“”‘’", "\"\"''")

DEFAULT_SLOT_LABELS = [            # [DATA] training metadata; remappable to any provider
    "gpt-5", "claude-sonnet-4", "gemini-2.5-pro",
    "deepseek-r1-distill-qwen-32b", "gemma-3-27b-it",
    "qwen3-32b-reasoning", "qwen3-32b-direct",
]

# ---- 3-list parsing (faithful to conductor_utils._extract_any) [EXEC] --------
def _balanced_list(after: str) -> str | None:
    """Extract the first balanced [...] list, respecting quotes/escapes."""
    depth = 0; start = None; q = None; esc = False
    for i, ch in enumerate(after):
        if esc: esc = False; continue
        if ch == "\\": esc = True; continue
        if q:
            if ch == q: q = None
            continue
        if ch in "\"'": q = ch; continue
        if ch == "[":
            if depth == 0: start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                return after[start:i + 1]
    return None


def extract_list(text: str, labels: list[str]) -> list:
    """Find 'label: [ ... ]' then parse via ast -> json -> CSV fallback. [EXEC]"""
    tag = "|".join(re.escape(l) for l in labels)
    m = re.search(rf"({tag})\s*[:=]\s*", text, re.I)
    if not m:
        return []
    raw = _balanced_list(text[m.end():])
    if not raw:
        return []
    raw = raw.translate(_SMART).strip()
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass
    try:
        return json.loads(re.sub(r"'", '"', raw))
    except Exception:
        pass
    items = [x.strip(" \"'") for x in raw.strip("[]").split(",") if x.strip()]
    return [int(x) if x.isdigit() else x for x in items]


def parse_workflow(text: str) -> tuple[list, list, list]:
    model_ids = extract_list(text, ["model_id", "model id", "model_ids", "model ids"])
    subtasks  = extract_list(text, ["subtasks", "subtask"])
    access    = extract_list(text, ["access_list", "access list", "access"])
    return model_ids, subtasks, access


def _is_all(x) -> bool:
    return isinstance(x, str) and x.strip().lower() in ("all", "[all]", "'all'")

# EXEC_MARKER

# ---- access-list visibility (choose_position: indices of earlier steps) [EXEC]
def visible_indices(access_list: list, step: int) -> list[int]:
    """Which earlier step outputs are visible to `step`. Forward references are
    rejected (topological order). 'all' => every earlier step. Faithful to
    _ascribe_history_positional_complex / _ascribe_history_binary."""
    if step == 0:
        return []
    a = access_list[step] if step < len(access_list) else []
    if _is_all(a):
        return list(range(step))
    if a in ([], "", None):
        return []
    out = []
    for pos in dict.fromkeys(a if isinstance(a, (list, tuple)) else [a]):
        if not isinstance(pos, int):
            continue
        if pos >= step:                                  # forward reference -> reject [EXEC]
            raise ValueError(f"step {step} references future/own step {pos} (not a DAG order)")
        if 0 <= pos < step:
            out.append(pos)
    return sorted(out)


# ---- the Conductor prompt (prompted stand-in for the RL-trained 7B) [DOC] ----
def conductor_prompt(query: str, slot_labels: list[str]) -> list[dict]:
    pool = "\n".join(f"  {i}: {name}" for i, name in enumerate(slot_labels))
    sys = (
        "You are a Conductor that orchestrates a pool of worker LLMs to solve a task. "
        "Design an agentic workflow as THREE equal-length Python lists:\n"
        "  model_id   = [int, ...]   # which worker (0-indexed) runs each step\n"
        "  subtasks   = [str, ...]   # the natural-language instruction for each step\n"
        "  access_list= [list, ...]  # for each step, the indices of EARLIER steps whose\n"
        "                            # outputs that step may see ([] = none, may use \"all\")\n"
        "Rules: lists must be equal length (<=5 steps); access_list may only reference "
        "strictly earlier steps (it is a DAG executed in order); the LAST step's output "
        "is the final answer. Pick workers to match each subtask's demands.\n\n"
        f"AVAILABLE LANGUAGE MODELS:\n{pool}\n\n"
        "Output the three lists explicitly as 'model_id: [...]', 'subtasks: [...]', "
        "'access_list: [...]'. You may reason first, but the three lists must appear."
    )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": f"USER QUESTION: {query}"}]


WorkerFn = Callable[[str, list, int], str]   # (subtask, messages, agent_id) -> reply


@dataclass
class Step:
    idx: int
    agent_id: int
    subtask: str
    sees: list[int]
    reply: str


@dataclass
class UltraResult:
    final: str
    steps: list[Step] = field(default_factory=list)
    workflow: dict = field(default_factory=dict)


class ConductorExecutor:
    """Parse a Conductor completion into a workflow DAG and execute it. [EXEC]

    Each step prompts its worker with its subtask plus the outputs of the steps
    named in access_list, injected as <Agent N response> blocks (the engine's
    exact context-assembly format)."""
    def __init__(self, worker: WorkerFn, slot_labels=None, max_steps=MAX_STEPS):
        self.worker = worker
        self.slot_labels = slot_labels or DEFAULT_SLOT_LABELS
        self.max_steps = max_steps

    def validate(self, model_ids, subtasks, access):
        if not (subtasks and model_ids and access):
            raise ValueError("workflow missing one of model_id/subtasks/access_list")
        if not (len(model_ids) == len(subtasks) == len(access)):
            raise ValueError(f"lists unequal length: "
                             f"{len(model_ids)}/{len(subtasks)}/{len(access)}")
        if len(subtasks) > self.max_steps:
            subtasks, model_ids, access = (subtasks[:self.max_steps],
                                           model_ids[:self.max_steps], access[:self.max_steps])
        return model_ids, subtasks, access

    def execute(self, model_ids, subtasks, access, verbose=False,
                stream_last_step=None, on_token=None) -> UltraResult:
        """Execute the DAG.

        stream_last_step: if provided, called instead of self.worker for the
            final step; must be callable(subtask, messages, agent_id) -> Iterator[str].
        on_token: called with each token string from stream_last_step.
        """
        model_ids, subtasks, access = self.validate(model_ids, subtasks, access)
        res = UltraResult(final="", workflow={"model_id": model_ids,
                                              "subtasks": subtasks, "access_list": access})
        outputs: list[str] = []
        last_idx = len(subtasks) - 1
        for t, (mid, sub) in enumerate(zip(model_ids, subtasks)):
            sees = visible_indices(access, t)
            ctx = ""
            for j in sees:
                ctx += (f"\n<Subtask assigned to Agent {model_ids[j]}>{subtasks[j]}"
                        f"</Subtask assigned to Agent {model_ids[j]}>"
                        f"\n<Agent {model_ids[j]} response>{outputs[j].strip()}"
                        f"</Agent {model_ids[j]} response>")
            user = (f"USER QUESTION context:\n{ctx}\n\nYour subtask: {sub}"
                    if ctx else f"Your subtask: {sub}")
            mid = int(mid) % len(self.slot_labels)
            messages = [{"role": "user", "content": user}]

            if t == last_idx and stream_last_step is not None:
                reply_parts: list[str] = []
                for tok in stream_last_step(sub, messages, mid):
                    if on_token is not None:
                        on_token(tok)
                    reply_parts.append(tok)
                reply = "".join(reply_parts)
            else:
                reply = self.worker(sub, messages, mid)

            outputs.append(reply)
            res.steps.append(Step(t, mid, sub, sees, reply))
            if verbose:
                print(f"  step {t}: agent={mid}({self.slot_labels[mid]}) sees={sees}")
                print(f"    subtask: {sub[:80]}")
                print(f"    -> {reply.strip()[:90]}")
        res.final = outputs[-1] if outputs else ""        # last step = answer [EXEC]
        return res

# CLI_MARKER

class LiteLLMWorker:
    """Provider-agnostic worker via litellm (same middle layer as fugu_mini)."""
    def __init__(self, slot_models=None, api_key=None, api_base=None,
                 max_tokens=1024, temperature=0.2):
        import litellm
        self.litellm = litellm
        default = os.environ.get("FUGU_WORKER_MODEL", "openai/gpt-4o-mini")
        self.slot_models = slot_models or [default] * N_AGENTS
        self.api_key = api_key or os.environ.get("FUGU_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("FUGU_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.max_tokens, self.temperature = max_tokens, temperature

    def _call(self, model, messages):
        kw = dict(model=model, messages=messages,
                  max_tokens=self.max_tokens, temperature=self.temperature)
        if self.api_key:  kw["api_key"] = self.api_key
        if self.api_base: kw["api_base"] = self.api_base
        r = self.litellm.completion(**kw)
        usage      = getattr(r, "usage", None) or object()
        prompt_tok = getattr(usage, "prompt_tokens",     0) or 0
        compl_tok  = getattr(usage, "completion_tokens", 0) or 0
        return r.choices[0].message.content or "", prompt_tok, compl_tok

    def __call__(self, subtask, messages, agent_id):
        text, _, _ = self._call(self.slot_models[agent_id % len(self.slot_models)], messages)
        return text

    def _stream_call(self, model, messages):
        kw = dict(model=model, messages=messages,
                  max_tokens=self.max_tokens, temperature=self.temperature,
                  stream=True)
        if self.api_key:  kw["api_key"]  = self.api_key
        if self.api_base: kw["api_base"] = self.api_base
        return self.litellm.completion(**kw)

    def stream(self, subtask: str, messages: list, agent_id: int):
        """Yield non-empty token strings for agent_id's slot model."""
        r = self._stream_call(self.slot_models[agent_id % len(self.slot_models)], messages)
        for chunk in r:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def conduct(self, model, messages):     # the Conductor call (more tokens)
        return self.conduct_with_usage(model, messages)[0]

    def conduct_with_usage(self, model, messages):
        """Like conduct(), but also returns (prompt_tokens, completion_tokens)."""
        old = self.max_tokens; self.max_tokens = 2048
        try:
            return self._call(model, messages)
        finally:
            self.max_tokens = old


class LocalConductor:
    """Our GRPO-trained Conductor, loaded locally with transformers. Emits the
    workflow completion (the 3-list DAG) from a chat-formatted prompt. This is
    the trained policy driving inference — not a prompted off-the-shelf API
    model. [EXEC: generation; the trained weights are ours]"""
    def __init__(self, ckpt, device="cuda:0", max_new=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.max_new, self.device = torch, max_new, device
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).to(device).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(device).eval()

    def conduct(self, messages):              # mirrors LiteLLMWorker.conduct signature use
        return self.conduct_with_usage(messages)[0]

    def conduct_with_usage(self, messages):
        """Like conduct(), but also returns (prompt_tokens, completion_tokens)."""
        torch = self.torch
        try:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "\n".join(m["content"] for m in messages)
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        prompt_tok = int(ids["input_ids"].shape[1])
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        compl_tok = int(out.shape[1] - prompt_tok)
        text_out = self.tok.decode(out[0, prompt_tok:], skip_special_tokens=True)
        return text_out, prompt_tok, compl_tok


class LocalPoolWorker:
    """Local worker pool for executing the workflow DAG steps — same protocol as
    the TRINITY serving side: (subtask, messages, agent_id) -> reply, each model
    resident on its own GPU. No external API."""
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

    def __call__(self, subtask, messages, agent_id):
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


class MockWorker:
    """Offline: deterministic replies so parser+DAG can be tested with no keys."""
    def __call__(self, subtask, messages, agent_id):
        return f"[agent {agent_id}] result for: {subtask[:50]}"


CANNED = (  # a Conductor-style completion for the offline self-test
    "Plan: derive then implement then verify.\n"
    "model_id: [2, 0, 1]\n"
    'subtasks: ["Devise an algorithm for the task", '
    '"Implement it in Python using the devised algorithm", '
    '"Verify the implementation is correct"]\n'
    "access_list: [[], [0], [0, 1]]\n"
)


def self_test() -> int:
    """Offline: parse a canned workflow and execute the DAG with a mock pool.
    Checks parsing, equal-length validation, topological visibility."""
    mids, subs, acc = parse_workflow(CANNED)
    print("parsed workflow:")
    print(f"  model_id   = {mids}")
    print(f"  subtasks   = {[s[:30]+'...' for s in subs]}")
    print(f"  access_list= {acc}")
    assert mids == [2, 0, 1], mids
    assert acc == [[], [0], [0, 1]], acc
    assert len(mids) == len(subs) == len(acc) == 3
    # visibility
    assert visible_indices(acc, 0) == []
    assert visible_indices(acc, 1) == [0]
    assert visible_indices(acc, 2) == [0, 1]
    # forward-ref rejection
    try:
        visible_indices([[], [2], []], 1); raise AssertionError("should have rejected")
    except ValueError:
        pass
    res = ConductorExecutor(MockWorker()).execute(mids, subs, acc, verbose=True)
    assert len(res.steps) == 3 and res.final
    print("\nPASS — parser, equal-length, DAG order, forward-ref ban, execution all OK")
    return 0


def _parse_local_specs(csv, n_gpu):
    specs = []
    for i, entry in enumerate(csv.split(",")):
        if "@" in entry:
            path, dev = entry.rsplit("@", 1)
        else:
            path = entry
            dev = f"cuda:{(i % max(n_gpu - 1, 1)) + 1}" if n_gpu > 1 else "cpu"
        specs.append((os.path.basename(path.rstrip("/")) or f"w{i}", path, dev))
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Faithful Fugu-Ultra (Conductor) workflow executor.")
    ap.add_argument("--query")
    ap.add_argument("--conductor", help="litellm model id acting as the Conductor")
    ap.add_argument("--local-conductor", help="path to a trained local Conductor checkpoint")
    ap.add_argument("--conductor-device", default="cuda:0")
    ap.add_argument("--slot-models", metavar="CSV", help="7 litellm worker model ids")
    ap.add_argument("--local-models", metavar="CSV",
                    help="local HF worker paths (path or path@device); no API needed")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.query or not (args.conductor or args.local_conductor):
        ap.error("need --query and (--conductor or --local-conductor), or --self-test")

    # worker pool: local resident models, or litellm
    if args.local_models:
        try:
            import torch
            n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            n_gpu = 0
        specs = _parse_local_specs(args.local_models, n_gpu)
        worker = LocalPoolWorker(specs)
        slot_labels = [n for n, _, _ in specs]
        print(f"workers: LOCAL ({len(specs)}): {slot_labels}")
    else:
        slots = args.slot_models.split(",") if args.slot_models else None
        worker = LiteLLMWorker(slot_models=slots)
        slot_labels = slots or DEFAULT_SLOT_LABELS
        print(f"workers: litellm ({len(slot_labels)} slots)")

    # Conductor: trained local checkpoint, or litellm
    if args.local_conductor:
        conductor = LocalConductor(args.local_conductor, device=args.conductor_device)
        print(f"conductor: LOCAL trained {args.local_conductor}")
        completion = conductor.conduct(conductor_prompt(args.query, slot_labels))
    else:
        print(f"conductor: {args.conductor}")
        completion = worker.conduct(args.conductor, conductor_prompt(args.query, slot_labels))

    print(f"query: {args.query}\n")
    mids, subs, acc = parse_workflow(completion)
    if not subs:
        print("Conductor did not emit a parseable workflow. Raw completion:\n")
        print(completion[:800]); return 1
    print(f"workflow: model_id={mids}  access_list={acc}")
    print(f"  ({len(subs)} steps)\n")
    res = ConductorExecutor(worker, slot_labels=slot_labels).execute(mids, subs, acc, verbose=True)
    print(f"\nfinal answer (step {len(res.steps)-1} output):\n{res.final.strip()[:600]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


