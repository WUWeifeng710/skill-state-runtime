# SKILL.state Runtime

**Scalable long-horizon agent skills: replace append-only conversation history with an explicit, mutable, structured execution state.**

A model-agnostic reference implementation of the architecture from
[*"SKILL.state: Scalable Long-Horizon Agent Skills"*](https://arxiv.org/abs/2608.26263)
(Google / Purdue, arXiv:2608.26263). Pure Python stdlib — no framework, no
hard-coded model, no required API.

---

## Why

When an LLM agent runs a long, multi-step procedure (a bioinformatics pipeline,
a multi-stage data workflow), the conventional append-only conversation history
grows without bound. SKILL.state replaces it with a compact execution state:

```
input :  P  (immutable skill spec)  +  Σt (current structured state)  +  Ot (latest obs)
output:  Rt (reasoning, DISCARDED)  +  ΔΣt (validated state patch)     +  at (action|terminate)
state :  Σt+1 = Σt ⊕ ΔΣt     (dict merge; null deletes a key)
```

| Dimension | Conversational baseline | SKILL.state |
|---|---|---|
| Complexity | cumulative tokens **O(T²)** | prompt **O(1)**, cumulative **O(T)** |
| T=100 total tokens | ~1.06M (Stateful baseline) | **~65K (16× less)** |
| High-noise environment | 0.68 → 0.53 accuracy | **≥0.97 maintained** |
| External state drift | hallucinated 5–14 steps before correcting | **0 recovery steps** |
| Budget-matched compression baselines | collapse to 0.18–0.22 | **0.94** (structure preserves relational dependencies) |

Practical consequences for real pipelines:

- **Cost & latency stop growing** — the prompt never accumulates reasoning traces
  or stale tool output.
- **No context poisoning** — stale intermediate results (old BLAST hits, outdated
  QC stats) never re-enter the prompt, so errors do not propagate across steps.
- **Noise immunity** — cluster stderr / slurm / conda chatter is filtered at
  state-patch generation and never persisted.
- **Instant drift recovery** — preempted jobs, overwritten files, changed inputs
  are re-derived from the fresh observation, not hallucinated against history.

---

## Install

### Option A — manual (any machine with git + Python 3.8+)

```bash
git clone https://github.com/WUWeifeng710/skill-state-runtime.git
cp -r skill-state-runtime/<skill files> ~/.your-agent/skills/skill-state-runtime/
```

For WorkBuddy, the target directory is `~/.workbuddy/skills/skill-state-runtime/`.
For Claude Code / other agent tools, use their equivalent user-level skills
directory. The skill itself is plain Python + JSON — nothing to compile, no
dependencies to install.

### Option B — let your AI agent install it (recommended)

Paste this prompt into your AI agent (WorkBuddy, Claude Code, Codex CLI, etc.):

```text
Install the "skill-state-runtime" skill for me:

1. Download the repository https://github.com/WUWeifeng710/skill-state-runtime
   (git clone, or fetch the files individually).
2. Copy the skill files into the user-level skills directory:
   ~/.workbuddy/skills/skill-state-runtime/
   Required files: SKILL.md, engine.py, agent_loop.py, adapters.py,
   example_usage.py, live_demo.py, rna_seq_pilot.py, schemas/*.json
3. Verify the install by running:
   python example_usage.py
   It must print a FINAL STATE and a TOKEN REPORT with
   "prompt_footprint_constant": true, and must complete without any network
   access or API key.
4. Do NOT modify engine.py semantics; domain schemas under schemas/ are safe
   to extend for my own domains.
```

The skill is intentionally dependency-free: the core (`engine.py` +
`agent_loop.py`) runs on the Python standard library alone. The optional
head-less model path (`adapters.py`) needs `pip install openai` only if you
actually use it.

---

## Quick start

### 1. Agent-driven (default — no API, no network)

The host agent IS the reasoning engine. The runtime externalizes state to a
file and hands the agent a tiny per-step prompt; the agent reasons and replies.

```python
from agent_loop import AgentDriver
drv = AgentDriver(spec, schema, workdir="./run", action_exec=my_tool)
drv.run_interactive(initial_observation="task handed over")
# each step: runtime writes _step_prompt.json  ->  agent reads it, reasons,
#             writes _step_answer.json  ->  runtime validates+applies, executes,
#             writes next observation. State lives in <workdir>/state.json.
```

Or drive the helpers directly from any tool step:

```python
from engine import StateStore, build_prompt_text, apply_patch_to_file
store = StateStore("./run/state.json")
prompt = build_prompt_text(spec, schema, store.load(), latest_observation)
# the agent reads `prompt`, produces patch+action JSON, then:
new_state = apply_patch_to_file(store, patch, schema)   # validates+merges+saves
```

### 2. Head-less with any OpenAI-compatible model (optional)

Only if you want the loop to run with no agent in the loop (cron, service, CI).
Endpoint, key, and model are all caller-supplied parameters — nothing is
hard-coded:

```python
from adapters import make_llm_call
from agent_loop import AgentDriver
llm = make_llm_call(base_url="<your-endpoint>", api_key="<your-key>", model="<any-model>")
drv = AgentDriver(spec, schema, workdir="./run", action_exec=my_tool, brain=llm)
drv.run("start")
```

### 3. Offline self-test (proves the machinery, no model needed)

```bash
python example_usage.py       # minimal 4-stage loop
python live_demo.py           # minimal end-to-end, resumable file-handshake demo
python rna_seq_pilot.py       # full RNA-seq shape; real tools run when on PATH
```

Expected output ends with a token report like:

```json
{
  "steps": 9,
  "prompt_growth_ratio_max_over_min": 1.577,
  "prompt_footprint_constant": true,
  "cumulative_tokens": 5443
}
```

`prompt_footprint_constant: true` is the O(1) proof: prompt size does not scale
with step count. (Note: the prompt may grow *slightly* because compact,
bounded `key_results` accumulate in `Σ` — this is structured state, not
unbounded history re-feeding.)

---

## Writing a domain schema (once per DOMAIN, not per task)

A schema is the contract between you and the model, **reused by every task in
that domain**. Example (`schemas/rna_seq.json`):

```json
{
  "inputs":         {"type":"dict","doc":"fixed inputs, written once: sample_id/fq1/fq2/organism/genome_index"},
  "current_stage":  {"type":"str", "doc":"qc|trim|align|sort|dedup|quant|diff|enrich|done"},
  "completed_steps":{"type":"list","doc":"names of completed stages (model rewrites the full list each step)"},
  "key_results":    {"type":"dict","doc":"read_count/mapped_pct/expressed_genes/sig_degs"},
  "open_issues":    {"type":"list","doc":"open issues to resolve"}
}
```

Types: `str | int | float | bool | list | dict | any`. Keep it **small and
stable** — changing it mid-run is unsupported.

---

## When NOT to use it (be honest)

1. **Open-ended exploration** — the state structure must be knowable up front.
2. **Provenance / audit tasks** — "why did step 3 fail last week" requires the
   trajectory, which SKILL.state deliberately discards.
3. **Tiny local models without grammar-constrained decoding** — small models
   frequently emit invalid JSON or prematurely overwrite state keys (in the
   paper, a 31B model reached only 0.42 at T=100; 68% of errors were premature
   key overwrite/delete). Add grammar-constrained decoding (outlines, llama.cpp
   GBNF) or a JSON-schema validator + retry layer for such models.
4. **Concurrent multi-agent writes** — each agent owns its own `Σ`; there is no
   built-in concurrent merge.

---

## Files

| File | Purpose |
|---|---|
| `engine.py` | Pure-stdlib runtime: `SkillStateRuntime`, `validate_patch`, `merge_state`, `StateStore`, `build_prompt_text`, `apply_patch_to_file`, `DriftError` |
| `agent_loop.py` | Default driver: `AgentDriver` stepwise file handshake with the host agent. No network. |
| `adapters.py` | OPTIONAL generic OpenAI-compatible model factory (head-less automation only) |
| `schemas/*.json` | Domain templates: `rna_seq`, `metagenomics`, `generic` |
| `example_usage.py` | Offline demo with a reference brain (no model needed) |
| `live_demo.py` | Minimal resumable file-handshake demo |
| `rna_seq_pilot.py` | Real RNA-seq pipeline shape; shells out to real tools when on PATH, records intended commands otherwise |

---

## Citation

If this implementation is useful to you, please cite the original paper:

```bibtex
@article{skillstate2026,
  title   = {SKILL.state: Scalable Long-Horizon Agent Skills},
  author  = {Google and Purdue University},
  year    = {2026},
  journal = {arXiv preprint arXiv:2608.26263},
  url     = {https://arxiv.org/abs/2608.26263}
}
```

## License

MIT — see [LICENSE](LICENSE).
