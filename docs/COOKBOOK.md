# OpenFugu Cookbook — updating models, training selectors, benchmarking

A practical, source-grounded guide to operating OpenFugu: swapping in the latest
frontier models, (re)training the TRINITY router and the Conductor, and
benchmarking the result. Every command and flag below is taken from the actual
scripts in this repo — file references are given so you can verify.

> **Mental model — read this first.** The "model selector" never names a model.
> TRINITY's router emits a **slot index** (`agent_id`), and the Conductor emits a
> list of **slot indices** (`model_id: [...]`). Which concrete model sits in each
> slot is decided entirely at serve time by `--slot-models` / `--local-models`.
>
> Consequence: **putting GPT-5.5 / Sonnet-4.6 / GLM-5.2 into the pool is just a
> CLI change — no retraining required to *use* them.** You retrain the selector
> only to *re-optimize routing* for the new pool's competence profile (a strong
> new model should get more traffic). See Part B.

---

## Part A — Update the worker pool to the latest models

### A.1 The two pool flags (both `serve.py` and `serve_ultra.py`)

| Flag | Pool type | Format | Notes |
|---|---|---|---|
| `--slot-models` | Remote (litellm) | CSV of litellm model IDs | `openai/gpt-5.5,anthropic/claude-sonnet-4-6,...` |
| `--local-models` | Local HF | CSV of paths, optional `@device` | `/models/llama-3.3-8b@cuda:1,/models/gemma-3-27b@cuda:2` |

- Omit both → `MockWorker` (offline stand-in; the dashboard shows mock data).
- Pool size is **`N_AGENTS = 7`** slots (`openfugu/ultra.py:33`, `openfugu/mini.py`). Provide up to 7 models; if you give fewer, dispatch wraps with `agent_id % len(pool)` (`serve.py:107`, `serve_ultra.py:114`).
- Local device parsing: `path@cuda:N`, else round-robin across GPUs, else `cpu` (`serve.py:345-354`, `ultra.py:_parse_local_specs`).

### A.2 litellm IDs for the requested frontier models

litellm IDs are `provider/model`. Exact version strings must match what your
provider / litellm build exposes — **verify against `litellm.model_list` or the
provider's docs**, since these are bleeding-edge versions. The cleanest way to
run a *heterogeneous* frontier pool behind one key is **OpenRouter** (`openrouter/...`).

| You want | Direct litellm ID (example) | Via OpenRouter (recommended) |
|---|---|---|
| GPT 5.5 | `openai/gpt-5.5` | `openrouter/openai/gpt-5.5` |
| Claude Opus 4.8¹ | `anthropic/claude-opus-4-8` | `openrouter/anthropic/claude-opus-4.8` |
| Claude Sonnet 4.6 | `anthropic/claude-sonnet-4-6` | `openrouter/anthropic/claude-sonnet-4.6` |
| GLM 5.2 | `zhipuai/glm-5.2` | `openrouter/z-ai/glm-5.2` |
| GLM 5.1 | `zhipuai/glm-5.1` | `openrouter/z-ai/glm-5.1` |

¹ "Open 4.8" is read here as **Claude Opus 4.8**. If you meant something else
(e.g. an OpenAI `o`-series model), swap the ID — the mechanism is identical.

**Credentials** (read by `LiteLLMWorker`, `ultra.py:230-240` / `mini.py:236-256`):

| Env var | Used for |
|---|---|
| `FUGU_API_KEY` → falls back to `OPENAI_API_KEY` (and `NOVITA_API_KEY` in training) | provider key |
| `FUGU_BASE_URL` → falls back to `OPENAI_BASE_URL` | custom/base endpoint (e.g. OpenRouter `https://openrouter.ai/api/v1`) |
| `FUGU_WORKER_MODEL` | default per-slot model if `--slot-models` omitted (default `openai/gpt-4o-mini`) |

### A.3 Serve Fugu (TRINITY) with the new pool

```sh
export FUGU_API_KEY=sk-or-...                       # OpenRouter key
export FUGU_BASE_URL=https://openrouter.ai/api/v1

python openfugu/serve.py \
  --model /models/Qwen3-0.6B \
  --vector artifacts/model_iter_60.npy \
  --slot-models "openrouter/openai/gpt-5.5,openrouter/anthropic/claude-opus-4.8,openrouter/anthropic/claude-sonnet-4.6,openrouter/z-ai/glm-5.2,openrouter/z-ai/glm-5.1,openrouter/qwen/qwen3-32b,openrouter/deepseek/deepseek-r1" \
  --port 8088
```

`serve.py` flags (`serve.py:319-332`): `--model` (required, Qwen3-0.6B dir),
`--vector` (default `model_iter_60.npy`, the 19 456-float base = 9 216 SVF +
10 240 head), `--head` (optional trained head-only override, 10 240 floats),
`--slot-models`, `--local-models`, `--port` (8088), `--max-turns` (5).

### A.4 Serve Fugu-Ultra (Conductor) with the new pool

The Conductor itself is a model too. Two ways to supply it:

```sh
# Prompted frontier conductor (no training) — RECOMMENDED for newest models
python openfugu/serve_ultra.py \
  --conductor "openrouter/anthropic/claude-opus-4.8" \
  --slot-models "openrouter/openai/gpt-5.5,openrouter/anthropic/claude-sonnet-4.6,openrouter/z-ai/glm-5.2,openrouter/z-ai/glm-5.1,openrouter/qwen/qwen3-32b,openrouter/deepseek/deepseek-r1,openrouter/google/gemini-2.5-pro" \
  --port 8089

# OR a locally-trained conductor checkpoint
python openfugu/serve_ultra.py \
  --local-conductor /ckpts/conductor_workflow \
  --conductor-device cuda:0 \
  --local-models "/models/llama-3.3-8b@cuda:1,/models/gemma-3-27b@cuda:2" \
  --port 8089
```

`serve_ultra.py` flags (`serve_ultra.py:390-415`): `--conductor` (litellm ID) **or**
`--local-conductor` (checkpoint path) — one is required; `--conductor-device`
(cuda:0), `--slot-models`, `--local-models`, `--port` (8089).

> **Conductor must speak the workflow DSL.** It has to emit three equal-length
> lists `model_id: [...]`, `subtasks: [...]`, `access_list: [...]`
> (`ultra.py:121-139`). Capable instruction-tuned frontier models do this when
> prompted. The repo's GRPO checkpoint (`conductor_toolscale_100`) was trained on
> the **tool-call** DSL, not the workflow DSL, so it emits code and fails to parse
> (`results/conductor_e2e_run.txt`) — see Part B.2 for the fix.

### A.5 When do you actually need to retrain?

| Change | Retrain selector? |
|---|---|
| Swap a slot's model for a newer version of similar skill | No — routing still reasonable |
| Add a much stronger/weaker model, or reorder slots | **Yes** — re-optimize so traffic follows the new competences |
| Conductor is a prompted frontier model | No training; just `--conductor <id>` |
| Want a *small local* conductor to plan workflows | Yes — train on workflow-DSL data (B.2) |

The dashboard's worker cards read `/v1/workers` (`serve.py:186-212`,
`serve_ultra.py:216-264`), so new slot models appear automatically once served —
slot 0 on the Ultra side is the Conductor, slots 1+ are workers.

---

## Part B — Train the model selectors

### B.1 TRINITY router (sep-CMA-ES over a linear head)

**What is trained.** TRINITY = a frozen Qwen3-0.6B backbone + a bias-free linear
head over its penultimate hidden state (`mini.py`). The released base vector
`model_iter_60.npy` is 19 456 floats (9 216 SVF singular-value offsets + a
10 240 = 10×1024 head). Training optimizes the **head** with **sep-CMA-ES** (the
`cma` package); the head is saved as a `.npy` you pass to `serve.py --head`.

Three entry points, increasing fidelity:

| Script | Data / pool | Fitness | Key flags (defaults) |
|---|---|---|---|
| `train/train_trinity.py` | **mock** synthetic world | mean terminal reward | `--mock` (on), `--iters 60`, `--sigma0 0.3`, `--n-tasks 64`, `--repeats 4`, `--out trinity_mock.npy` |
| `train/train_trinity_real.py` | **real GSM8K** + litellm pool | numeric-answer match | `--slot-models` (**required**, CSV), `--model Qwen/Qwen3-0.6B`, `--n-train 12`, `--iters 8`, `--sigma0 0.5`, `--out trinity_gsm8k.npy` |
| `train/train_trinity_perstep.py` | real GSM8K, **full multi-turn rollout** | solved-rate of the Coordinator loop | `--router-model`, `--vector`, `--n-train 8`, `--iters 6`, `--max-turns 4`, `--out trinity_perstep.npy` |

Real-data run (this is the one to use when re-optimizing for a new pool):

```sh
export FUGU_API_KEY=...    # or OPENAI_API_KEY / NOVITA_API_KEY
export FUGU_BASE_URL=https://openrouter.ai/api/v1
python train/train_trinity_real.py \
  --slot-models "openrouter/openai/gpt-5.5,openrouter/anthropic/claude-sonnet-4.6,openrouter/z-ai/glm-5.2,openrouter/qwen/qwen3-32b" \
  --n-train 64 --iters 20 --out trinity_newpool.npy
# then serve with the trained head:
python openfugu/serve.py --model /models/Qwen3-0.6B \
  --vector artifacts/model_iter_60.npy --head trinity_newpool.npy \
  --slot-models "<same CSV, same order>" --port 8088
```

> **Slot order is the contract.** The head learns "slot 2 is strong at math".
> Serve with the **same `--slot-models` order** you trained with, or the routing
> is meaningless.

> **⚠ Hardcoded paths in the per-step trainers.** `train_trinity_perstep.py` and
> `train_adaptive_pool_perstep.py` load a **hardcoded local worker pool**
> (`/vePFS-Mindverse/share/huggingface/...`, e.g. deepseek-distill-7b@cuda:1,
> llama-3.2-3b@cuda:2, gemma-3-4b@cuda:3) and `--vector /root/model_iter_60.npy`.
> Edit those paths to your machine before running, or they will fail.

Other TRINITY variants:
- `train/train_adaptive_pool.py` (mock) / `train_adaptive_pool_perstep.py` (real) — train a router that respects an **availability mask** so it generalizes to arbitrary k-of-n worker subsets (swap/opt-out providers at runtime via `route(..., agent_mask=...)`).

### B.2 Conductor (GRPO)

`train/train_conductor.py` GRPO-trains a base LM with `trl.GRPOTrainer`.

- Base model: `FUGU_BASE_MODEL` (default `meta-llama/Llama-3.2-3B-Instruct`).
- Output dir: `FUGU_OUT` (default `conductor_out`).
- Config (in-script, `train_conductor.py:36-53`): `num_generations=8`,
  `max_steps=40`, `lr=1e-5`, `bf16=True`, `gradient_checkpointing=True`,
  `beta=0.0` (no KL, matches the Fugu-Ultra report), `use_vllm=False`.
- Data + reward: `nvidia/ToolScale` via `train/toolscale_data.py`
  (`make_datasets`, `make_reward_functions`): `format_reward` (checks
  `<answer>…</answer>`) + `action_reward` (tool-call sequence match).

```sh
export FUGU_BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export FUGU_OUT=/ckpts/conductor_toolscale
python train/train_conductor.py        # watch reward climb off zero
```

> **Known gap (be honest about it).** This trains the **tool-call** DSL
> (`<think>…</think><answer>[json]</answer>`), **not** the Conductor **workflow**
> DSL (`model_id/subtasks/access_list`). The resulting checkpoint therefore
> *fails* the workflow executor (`results/conductor_e2e_run.txt`,
> `results/README.md` §"Fugu-Ultra Conductor"). To get a *trained local* conductor
> that drives the DAG you must GRPO on the **workflow DSL itself** — a new dataset
> whose reward parses the 3-list workflow and scores executed correctness, reusing
> the GRPO scaffold in `train_conductor.py`. Until then, use a **prompted frontier
> conductor** (A.4), which speaks the workflow DSL out of the box.

Recursion finetune (test-time scaling — conductor revises its own plan):
- `train/train_recursion.py` (mock), `train/train_recursion_real.py` (real;
  `FUGU_BASE_CKPT`, `FUGU_OUT`, `FUGU_STEPS=30`, 256 ToolScale rows). Splices the
  round-0 output into the round-1 prompt and GRPO-trains the revision.

### B.3 One-command train → serve → verify

`pipeline/e2e_train_serve.py` chains `train_trinity_perstep.py` →
`eval/serve_e2e.py`:

```sh
python pipeline/e2e_train_serve.py \
  --model /models/Qwen3-0.6B \
  --local-models "/models/llama-3.3-8b@cuda:1,/models/gemma-3-27b@cuda:2" \
  --iters 6 --n-train 8 --port 8099
# --skip-train --head <existing.npy> to reuse a head instead of training
```

---

## Part C — Benchmark the models

### C.1 Routing regression — the 37-case fixture (`verify/verify_37.py`)

Fast, GPU-only, no API. Reconstructs the head from a vector and checks routing
decisions against a fixture; the released checkpoint scores **95% agent /
100% role** vs a ~51% class-prior baseline (`docs/ARCHITECTURE.md`).

```sh
export FUGU_MODEL=Qwen/Qwen3-0.6B
export FUGU_VECTOR=artifacts/model_iter_60.npy
export FUGU_FIXTURE=artifacts/qwen_router_prompt_eval_cases.json   # via scripts/fetch_artifacts.py
python verify/verify_37.py
```

Use it as a **regression guard**: after retraining a head, confirm it doesn't
regress routing. `verify/verify_margin.py` analyzes near-miss logit margins;
`verify/verify_trinity2.py` is a single-case reproduction guard.

### C.2 Orchestration-beats-best-single-model (the central claim)

- **Mock** (`eval/eval_orchestration.py`): runs with no GPU/API. Compares each
  worker alone, random routing, the trained coordinator, and the oracle ceiling;
  prints the lift `(coord − best_single)/best_single`. Flags: `--coordinator
  trinity_mock.npy`, `--n-tasks 5000`, `--seed 7`, `--world-seed 42` (**must
  match training**), `--train-iters 60`. Reported: **+107%** over best single on
  the mock world (`results/README.md`).
- **Real** (`train/train_trinity_real.py` prints the same comparison): it reports
  per-worker solo solved-rate vs the coordinator on real GSM8K. On GSM8K the
  result is often a **tie** (problems easy, all frontier workers strong); the
  multi-domain ToolScale run shows the coordinator beating the best single by a
  few percent where worker skills actually diverge (`results/README.md`).

> **To benchmark a NEW pool head-to-head:** run `train_trinity_real.py` with your
> new `--slot-models` — its own output already lists each worker's solo solved
> rate **and** the coordinator's, i.e. it doubles as a per-model benchmark plus a
> routing-gain measurement on the same task set.

### C.3 End-to-end serving proofs

```sh
# TRINITY end-to-end over a real local pool (boots serve.py, asks GSM8K, checks =72)
python eval/serve_e2e.py --model /models/Qwen3-0.6B \
  --vector artifacts/model_iter_60.npy --head trinity_newpool.npy \
  --local-models "/models/llama-3.3-8b@cuda:1,/models/gemma-3-27b@cuda:2" --port 8099

# Conductor end-to-end: emit + execute a workflow DAG, assert it parses & answers
python eval/ultra_e2e.py --conductor-ckpt /ckpts/conductor_workflow \
  --local-models "/models/llama-3.3-8b@cuda:1,/models/gemma-3-27b@cuda:2" \
  --query "Write a Fibonacci function, then verify it on n=10."
```

`serve_e2e.py` asserts `answer==gold`, `turns>0`, pool is LOCAL (not mock).
`ultra_e2e.py` asserts a parseable workflow (≥1 step), execution ran, non-empty
final answer — and **fails loudly** if the conductor doesn't speak the workflow DSL.

### C.4 Recursion (does a revise round help?)

`eval/eval_recursion_real.py --model <ckpt> --n 40` — scores round-0 vs round-1
on held-out ToolScale, reports mean delta + fix rate. Honest finding to date: a
**TIE** when the base is already strong/saturated (`results/recursion_real_run.txt`).

### C.5 Result-file convention

Every run is captured as a `results/*.txt` with the boot command, per-step
metrics, and a final `PASS`/`FAIL`/`TIE` line; `results/README.md` is the
narrative index with honest caveats. Follow the same pattern for new benchmarks:
log the command, the numbers, and an honest verdict.

---

## Reference

### Environment variables

| Var | Default | Role |
|---|---|---|
| `FUGU_API_KEY` / `OPENAI_API_KEY` / `NOVITA_API_KEY` | — | litellm provider key |
| `FUGU_BASE_URL` / `OPENAI_BASE_URL` | — | custom endpoint (e.g. OpenRouter) |
| `FUGU_WORKER_MODEL` | `openai/gpt-4o-mini` | default slot model if `--slot-models` omitted |
| `FUGU_MODEL` | `Qwen/Qwen3-0.6B` | router backbone |
| `FUGU_BASE_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | Conductor GRPO base |
| `FUGU_OUT` | `conductor_out` | training output dir |
| `FUGU_VECTOR` / `FUGU_FIXTURE` | artifacts/… | verify inputs |
| `FUGU_EVAL_CKPT` | — | recursion-eval checkpoint |

### Artifacts

| File | Meaning |
|---|---|
| `model_iter_60.npy` | released base vector, 19 456 = 9 216 SVF + 10 240 head |
| `trinity_*.npy` | trained head-only override (10 240), pass via `serve.py --head` |
| `conductor_*/` (dir) | trl/transformers Conductor checkpoint, pass via `--local-conductor` |
| `artifacts/qwen_router_prompt_eval_cases.json` | 37-case routing fixture (`scripts/fetch_artifacts.py`) |

### Honest caveats (carried from `results/README.md`)

- `eval_orchestration.py` and `train_trinity.py` are **mock**; only the `_real`
  scripts, `serve_e2e.py`, `ultra_e2e.py`, and `verify_*` touch real models.
- Per-step trainers report **in-sample** solved-rate on a handful of questions —
  proof of mechanism, not a held-out benchmark.
- The GRPO `train_conductor.py` checkpoint speaks the **tool-call** DSL, not the
  **workflow** DSL; prompted frontier conductors are the working path today.
- Match `--slot-models` **order** between training and serving, and match
  `--world-seed` between `train_trinity.py` and `eval_orchestration.py`.

### Dependencies

`torch>=2.4`, `transformers>=4.52,<5`, `trl>=0.19,<0.20`, `datasets>=3.6`,
`peft`, `accelerate`, `numpy`, `litellm`, `hydra-core>=1.3`, `omegaconf`,
`math_verify`, `huggingface_hub`, `cma` (`requirements.txt`). GPU needs: TRINITY
real ~1 GPU (+ API workers); per-step local pool ~3–4 GPUs; Conductor GRPO ~1–2
GPUs (bf16 + gradient checkpointing).
