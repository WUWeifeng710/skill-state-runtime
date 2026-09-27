---
name: skill-state-runtime
agent_created: true
version: 0.3.0
description: >
  Scalable long-horizon agent execution: replace append-only conversation
  history with an explicit mutable structured state (SKILL.state, arXiv:2608.26263).
  Use when an agent must run a long, multi-step procedural skill (bioinformatics
  pipelines, multi-stage workflows) where prompt growth / context-poisoning /
  cost blowup are risks. NOT for open-ended exploration or provenance/audit tasks.
---

# SKILL.state runtime

A **model-agnostic** execution pattern. It binds to NO specific LLM, vendor, or
API: the reasoning engine is supplied by the caller as either (a) a host agent
that answers a per-step prompt file, or (b) any `llm_call` callable
(OpenAI-compatible or otherwise). The runtime core is pure stdlib and makes no
network call by itself.

## When to use
- The task is a **long-horizon procedural skill** with a **known, fixed state schema**
  (you can list the fields that matter up front). Typical: RNA-seq / ChIP-seq /
  metagenomics assembly / molecular-docking pipelines, multi-step data wrangling.
- You care about **context-poisoning** (stale intermediate results corrupting later
  steps) and about keeping the **working memory lean** — the agent always reads the
  *current* compact state from a file, never a growing transcript.
- The environment is **noisy** (cluster stderr, slurm/sbatch logs) or **drifts**
  (jobs preempted, files overwritten) — SKILL.state filters distractors and recovers
  from drift in ~0 steps.

**No API, no network required by default.** The host agent (whatever model the
host process runs) is the reasoning engine: it reads the per-step prompt, returns
a state-patch + action JSON, the runtime validates/applies it, and continues.
A generic head-less path exists in `adapters.py` ONLY if you later want full
automation with no agent in the loop — see "Optional head-less model path" below.

## When NOT to use
- **Open-ended exploration** ("just look at this dataset and tell me what's interesting")
  where the state structure cannot be fixed in advance.
- **Provenance / audit / "why did step 3 fail last week"** tasks — these *require* the
  historical trajectory, which SKILL.state deliberately discards.
- Steps that must run on a **tiny local model** without grammar-constrained decoding:
  small models frequently emit invalid JSON / overwrite state keys (see Limitations).

## The loop (what the engine does)
At each step t the model sees ONLY three things and produces three things:

```
input :  P  (immutable skill spec)  +  Σt (current structured state)  +  Ot (latest obs)
output:  Rt (reasoning, DISCARDED)  +  ΔΣt (validated state patch)     +  at (action|terminate)
state :  Σt+1 = Σt ⊕ ΔΣt     (dict merge; null deletes a key)
```

- Reasoning `Rt` is **never persisted** — it cannot poison later steps and does not grow
  the prompt. This is the whole point.
- A rejected `ΔΣt` triggers an in-step **rollback-retry** (the validation error is fed
  back; state is not advanced until the patch validates).
- An external `DriftError` from the action layer becomes the next observation; the model
  re-derives `Σ` from reality — no stale history to hallucinate against.

## Patch discipline (the rules the agent must follow every step)

The runtime validates *shape* (keys/types); it cannot validate *judgement*. These rules are the
difference between a lean, trustworthy `Σ` and a poisoned one. Follow them exactly:

1. **Minimal delta.** Emit only the keys that changed this step. Unchanged keys are simply
   absent from `ΔΣt` — never re-emit the whole state.
2. **`null` deletes.** `null` is the only removal syntax and removes the key outright. Use it
   on purpose; a `null` you did not mean is data loss the validator will never warn about.
3. **`completed_steps` is rewritten whole.** The merge does not append. Copy the current list
   from `Σt` and add the newly finished stage — a patch containing only the new stage silently
   erases the history of completed work.
4. **Numbers come from `Ot` verbatim.** A `key_results` value must be a quantity you can point
   at in the latest observation (or the action's own report). Never derive, round, or reuse a
   number from memory of earlier steps.
5. **Reasoning and noise never enter `Σ`.** Your analysis lives in the `action`/answer field and
   is discarded. Do not launder explanations into `open_issues` or `key_results`, and do not
   paste cluster logs (slurm/sbatch output, grep noise, scheduler chatter) into the state —
   filtering distractors out is a core function of this runtime, and the schema has no place
   for them by design.
6. **On `DriftError`, reality wins.** Rebuild the affected keys from the fresh observation;
   do not trust the old `Σ` or resubmit the rejected patch unchanged.
7. **Advance the stage as soon as the work is verifiably done.** If the next stage is blocked
   (waiting on a job, a file, a decision), still move `current_stage` forward and record the
   blocker in `open_issues` — do not freeze the stage while completed work accumulates in
   `completed_steps`.
8. **A rejected patch is fixed, not argued.** Validation errors are typically an unknown key or
   a wrong type: correct that key and resubmit. Never delete schema keys just to make a patch
   pass.

## How to author a schema (once per DOMAIN, not per task)
Write a JSON file under `schemas/` describing the allowed state keys and their types.
A schema is **reused by every task in that domain**. Example (`schemas/rna_seq.json`):

```json
{
  "inputs":         {"type":"dict","doc":"fixed inputs, written once: sample_id/fq1/fq2/organism/genome_index"},
  "current_stage":  {"type":"str", "doc":"qc|trim|align|sort|dedup|quant|diff|enrich|done"},
  "completed_steps":{"type":"list","doc":"names of completed stages (model rewrites the full list each step)"},
  "key_results":    {"type":"dict","doc":"read_count/mapped_pct/expressed_genes/sig_degs"},
  "open_issues":    {"type":"list","doc":"open issues to resolve"}
}
```

Types: `str | int | float | bool | list | dict | any`. Keep it **small and stable** —
the schema is the contract between you and the model; changing it mid-run is unsupported.

## Running the host agent (default — NO API)
The host agent IS the model. Two complementary ways:

**1) Turn-by-turn, host agent in the loop** — best when the actions are real tools
(bash, slurm, conda) the agent executes itself. The runtime externalizes state and
hands the agent a tiny per-step prompt; the agent reasons and replies. No network, ever.

```python
from agent_loop import AgentDriver
drv = AgentDriver(spec, schema, workdir="./run", action_exec=my_tool)
drv.run_interactive(initial_observation="task handed over")
# each step: runtime writes _step_prompt.json  ->  agent reads it, reasons,
#             writes _step_answer.json  ->  runtime validates+applies, executes,
#             writes next observation. State lives in <workdir>/state.json.
```

You can also call the helpers directly from a tool step instead of files:
```python
from engine import StateStore, build_prompt_text, apply_patch_to_file
store = StateStore("./run/state.json")
prompt = build_prompt_text(spec, schema, store.load(), latest_observation)
# the agent reads `prompt`, produces patch+action JSON, then:
new_state = apply_patch_to_file(store, patch, schema)   # validates+merges+saves
```

**2) Offline self-test** — proves the loop/prompt-footprint/drift with a reference
`brain` (a callable that mimics how a host agent would answer), no model attached:
```bash
python example_usage.py       # minimal 4-stage loop
python live_demo.py           # minimal end-to-end, resumable file-handshake demo
python rna_seq_pilot.py       # full RNA-seq shape; real tools run when on PATH
```

## Optional head-less model path (opt-in, fully generic)
Only if you want the loop to run with NO agent in the loop (cron/service/CI),
supply any OpenAI-compatible model via `adapters.make_llm_call(...)`. The endpoint,
key, and model name are all caller-supplied parameters — nothing is hard-coded:

```python
from adapters import make_llm_call
from agent_loop import AgentDriver
llm = make_llm_call(base_url="<your-endpoint>", api_key="<your-key>", model="<any-model>")
drv = AgentDriver(spec, schema, workdir="./run", action_exec=my_tool, brain=llm)
drv.run("start")
```
This is the ONLY place an API is touched, and it is entirely optional.

## Integration with a dispatcher harness
If you route tasks through a scheduler that classifies/splits work and dispatches
experts (dsh-style or any other), wrap **each routed expert task** in an
`AgentDriver` and let the host agent or the routed model fill the step. The shared
`Σ` (in `state.json`) becomes the coordination surface; the aggregator consumes
structured state instead of an O(T^2) conversation history.

## Files in this skill
- `engine.py` — pure stdlib runtime: `SkillStateRuntime`, `validate_patch`,
  `merge_state`, `StateStore`, `build_prompt_text`, `apply_patch_to_file`,
  `SYSTEM_PROMPT`, `DriftError`.
- `agent_loop.py` — **the default driver**: `AgentDriver` runs the stepwise
  handshake with the host agent as the model. No network.
- `adapters.py` — OPTIONAL generic OpenAI-compatible model factory for head-less
  automation only. Off by default; no vendor is hard-coded.
- `schemas/*.json` — domain templates: `rna_seq`, `metagenomics`, `generic`.
- `example_usage.py`, `live_demo.py`, `rna_seq_pilot.py` — offline demos driven by
  a reference `brain` (stand-in for any host agent).

## Limitations (be honest)
1. Requires a **known schema** up front; fails for fully open tasks.
2. **Small models** (e.g. <=8B) struggle with structured JSON adherence — in the paper
   Gemma-4-31B hit only 0.42 at T=100, with 68% of errors being premature key
   overwrite/delete. If an expert step uses a local small model, add grammar-constrained
   decoding (outlines / llama.cpp GBNF) or a JSON-schema validator + retry layer.
3. **Recovery** is automatic only for *external* drift the environment surfaces; if an
   early observation's relevance was missed at observation time, that fact is gone (it was
   never committed to `Σ`).
4. Single-agent runtime today. A dispatcher fans out to experts, but they do **not** share
   one mutable `Σ` concurrently — each expert owns its own `Σ`; the scheduler/aggregator is
   the only shared surface. Concurrent write-conflict resolution is out of scope.
5. The task objective must NOT be "the trajectory itself" (auditing / debugging provenance
   / explaining past actions) — that history is intentionally discarded.

## Validation before trusting
Run `example_usage.py` (offline, no API) to confirm the loop, merge semantics,
the O(1) prompt report, and drift recovery. Then pilot on ONE real long pipeline
(recommend RNA-seq or metagenomics assembly) using `run_interactive()` with the
host agent executing the real tools, and compare against your current ReAct-style
loop for cleanliness / context-poisoning before rolling out to all domains.
