#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: end-to-end proof for the Fugu-Ultra Conductor — our GRPO-TRAINED
# local Conductor emits a workflow DAG that is executed over a real local worker
# pool to a final answer. Connects the trained Conductor weights to inference.
"""
ultra_e2e.py — end-to-end proof that the TRAINED Conductor drives a real workflow.

Loads our GRPO-trained Conductor checkpoint (LocalConductor) + a real local
worker pool (LocalPoolWorker), runs a real query through ConductorExecutor, and
asserts: a parseable workflow (>=1 step) was emitted, execution ran, and the
final answer is non-empty. Fails loudly if no workflow parses or the answer is
empty — the trained weights' job is to emit and drive a valid workflow.

  python ultra_e2e.py --conductor-ckpt <trained conductor dir> \
      --local-models "<llama dir>,<gemma dir>"
"""
from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "openfugu"))
sys.path.insert(0, "/root")
import ultra
from ultra import LocalConductor, ConductorExecutor, conductor_prompt, parse_workflow
from serving import LocalPoolWorker, parse_local_specs


def main():
    ap = argparse.ArgumentParser(description="End-to-end proof: trained local Conductor + local pool.")
    ap.add_argument("--conductor-ckpt", required=True, help="trained Conductor checkpoint dir")
    ap.add_argument("--conductor-device", default="cuda:0")
    ap.add_argument("--local-models", required=True, metavar="CSV")
    ap.add_argument("--query", default="Write a Python function that returns the n-th Fibonacci "
                                       "number, then verify it on n=10.")
    args = ap.parse_args()

    try:
        import torch
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_gpu = 0
    specs = parse_local_specs(args.local_models, n_gpu)
    print(f"[ultra-e2e] workers LOCAL ({len(specs)}): {[n for n,_,_ in specs]}", flush=True)
    worker = LocalPoolWorker(specs)
    slot_labels = [n for n, _, _ in specs]

    print(f"[ultra-e2e] loading TRAINED Conductor {args.conductor_ckpt}", flush=True)
    conductor = LocalConductor(args.conductor_ckpt, device=args.conductor_device)

    print(f"[ultra-e2e] query: {args.query}", flush=True)
    completion = conductor.conduct(conductor_prompt(args.query, slot_labels))
    mids, subs, acc = parse_workflow(completion)
    print(f"[ultra-e2e] emitted workflow: model_id={mids} access_list={acc} steps={len(subs)}", flush=True)

    if not subs:
        print("[ultra-e2e] raw completion (no parseable workflow):\n" + completion[:600], flush=True)
        print("\nFAIL — trained Conductor emitted no parseable workflow")
        return 1

    res = ConductorExecutor(worker, slot_labels=slot_labels).execute(mids, subs, acc, verbose=True)
    final = (res.final or "").strip()
    print(f"\n[ultra-e2e] executed {len(res.steps)} steps; final answer[:200]={final[:200]!r}", flush=True)

    ok = (len(subs) >= 1) and (len(res.steps) >= 1) and bool(final)
    if ok:
        print("\nPASS — trained local Conductor emitted a workflow that executed over a "
              "real local worker pool to a non-empty final answer")
        return 0
    print("\nFAIL — workflow did not execute to a non-empty answer")
    return 1


if __name__ == "__main__":
    sys.exit(main())
