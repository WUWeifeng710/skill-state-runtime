"""
SKILL.state runtime engine -- reference implementation (pure stdlib).

Implements the architecture from:
    "SKILL.state: Scalable Long-Horizon Agent Skills" (arXiv:2608.26263)

Core idea: replace append-only conversational history with an explicit,
mutable, structured execution state. At each step t the model receives only:
    P   = immutable procedural skill specification
    Σt  = current structured execution state
    Ot  = latest observation
and must produce:
    Rt  = chain-of-thought reasoning   (DISCARDED after use)
    ΔΣt = structured state patch        (validated, then merged)
    at  = action to execute             (or terminate)

State update:  Σt+1 = Σt ⊕ ΔΣt   (dict merge; null deletes a key)

Complexity: prompt footprint is O(1); cumulative tokens O(T). The reasoning
trace is never re-fed, so context-poisoning and O(T^2) cost vanish.

Minimal example
---------------
    rt = SkillStateRuntime(
        spec=P,
        schema=SCHEMA,
        llm_call=my_llm,            # callable([{role,content}]) -> str
        action_exec=my_tool,        # callable(action_dict) -> observation_str
    )
    final_state = rt.run(initial_observation="start")
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional


class InvalidStatePatch(Exception):
    """Raised when a state patch cannot be validated against the schema."""


class DriftError(Exception):
    """Raise from action_exec when the external world changed under us.

    The runtime catches this, feeds a recovery observation, and lets the
    model re-derive state -- this is what gives SKILL.state its 0-step
    recovery property (no stale history to hallucinate against).
    """


# Fixed instruction given to the model at every step. The reasoning key is
# discarded; only the validated state_patch + action are persisted.
SYSTEM_PROMPT = (
    "You are executing a long-horizon procedural skill using the "
    "SKILL.state runtime. You are given: (1) the immutable skill "
    "specification P, (2) the current structured execution state Sigma, "
    "(3) the latest observation O. Output a SINGLE JSON object with "
    "exactly three keys:\n"
    "  - 'reasoning': your private chain-of-thought. It is discarded "
    "after use and never persisted.\n"
    "  - 'state_patch': a JSON object updating Sigma. Only keys present "
    "in the schema are allowed; set a key to null to delete it.\n"
    "  - 'action': either {'type':'terminate','result':'...'} when the "
    "task is done, or {'type':'<tool>','args':{...}} to act.\n"
    "Do not output anything outside the JSON object."
)


# --------------------------------------------------------------------------- #
# Schema + state helpers
# --------------------------------------------------------------------------- #
def _declared_type(spec: Any) -> str:
    """Extract the declared type from a schema entry (str or {type:...})."""
    if isinstance(spec, dict):
        return spec.get("type", "any")
    return spec


def validate_patch(patch: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable errors (empty list == valid)."""
    errors: List[str] = []
    if not isinstance(patch, dict):
        return ["state_patch must be a JSON object"]
    for key, value in patch.items():
        if key not in schema:
            errors.append(f"unknown key '{key}' is not in the schema")
            continue
        if value is None:
            continue  # null == delete, always allowed
        t = _declared_type(schema[key])
        if t == "any":
            continue
        if t == "str" and not isinstance(value, str):
            errors.append(f"key '{key}' expects str, got {type(value).__name__}")
        elif t == "int" and (not isinstance(value, int) or isinstance(value, bool)):
            errors.append(f"key '{key}' expects int, got {type(value).__name__}")
        elif t == "float" and not isinstance(value, (int, float)) or (
            t == "float" and isinstance(value, bool)
        ):
            errors.append(f"key '{key}' expects float, got {type(value).__name__}")
        elif t == "bool" and not isinstance(value, bool):
            errors.append(f"key '{key}' expects bool, got {type(value).__name__}")
        elif t == "list" and not isinstance(value, list):
            errors.append(f"key '{key}' expects list, got {type(value).__name__}")
        elif t == "dict" and not isinstance(value, dict):
            errors.append(f"key '{key}' expects dict, got {type(value).__name__}")
    return errors


def merge_state(state: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a state patch. null deletes a key; nested dicts merge recusively."""
    out = dict(state)
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_state(out[key], value)
        else:
            out[key] = value
    return out


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of a model response (fenced or raw)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j != -1 and j > i:
        return json.loads(text[i : j + 1])
    raise ValueError("no JSON object found in model output")


def _estimate_tokens(messages: List[Dict[str, str]]) -> int:
    """Rough token estimate (~4 chars/token) for cost accounting."""
    return sum(len(m.get("content", "")) // 4 + 1 for m in messages)


def build_prompt_text(
    spec: str, schema: Dict[str, Any], state: Dict[str, Any], observation: str
) -> str:
    """The exact text the model reads at step t (P + Σ + O + schema).

    Exposed as a standalone function so a *host agent* (any model or host
    process driving the loop) can call it to get the precise prompt to reason
    over, without any external API. The agent then returns a state_patch +
    action JSON which it validates/applies via `apply_patch_to_file`.
    """
    return (
        "## Skill specification (P)\n" + spec + "\n\n"
        "## Current execution state (Σ)\n"
        + json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"
        "## Latest observation (O)\n" + observation + "\n\n"
        "## State schema (allowed keys)\n"
        + json.dumps(schema, ensure_ascii=False, indent=2) + "\n\n"
        "Produce your JSON now."
    )


class StateStore:
    """File-backed canonical state. The agent always reads the *current*
    compact state from here -- never a growing transcript. This is what keeps
    the working memory O(1) per step even when the agent drives across turns."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> Dict[str, Any]:
        import os
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self, state: Dict[str, Any]) -> None:
        import os
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)


def apply_patch_to_file(
    store: StateStore, patch: Dict[str, Any], schema: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate + merge a patch against the stored state; raises on invalid.

    This is the single mutation point -- call it after the host agent
    returns its step answer. Returns the new merged state.
    """
    errs = validate_patch(patch, schema)
    if errs:
        raise InvalidStatePatch("; ".join(errs))
    state = merge_state(store.load(), patch)
    store.save(state)
    return state


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
class SkillStateRuntime:
    def __init__(
        self,
        spec: str,
        schema: Dict[str, Any],
        llm_call: Callable[[List[Dict[str, str]]], str],
        action_exec: Callable[[Dict[str, Any]], str],
        max_steps: int = 200,
        retries: int = 3,
        verbose: bool = True,
    ):
        self.spec = spec
        self.schema = schema
        self.llm_call = llm_call
        self.action_exec = action_exec
        self.max_steps = max_steps
        self.retries = retries
        self.verbose = verbose
        self.prompt_sizes: List[int] = []   # per-step prompt tokens (proof of O(1))
        self.steps_taken: int = 0

    # -- prompt construction ------------------------------------------------- #
    def _build_messages(self, state: dict, observation: str) -> List[Dict[str, str]]:
        user = build_prompt_text(self.spec, self.schema, state, observation)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    # -- main loop ----------------------------------------------------------- #
    def run(self, initial_observation: str) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        obs = initial_observation

        for t in range(1, self.max_steps + 1):
            patch, action = self._attempt_step(state, obs, t)
            if patch is None:
                # exhausted retries without a valid patch
                raise InvalidStatePatch(f"step {t}: could not produce valid state_patch")

            state = merge_state(state, patch)
            self._log(f"step {t}: state={state}")

            atype = action.get("type")
            if atype == "terminate":
                self.steps_taken = t
                self._log(f"terminate: {action.get('result')}")
                return state

            try:
                obs = self.action_exec(action)
            except DriftError as e:
                # external drift: re-derive state from the new reality
                obs = f"STATE DRIFT DETECTED: {e}. Re-derive Sigma from this."
                self._log(f"step {t}: drift -> {e}")
                # continue loop; prompt stays O(1), recovery is automatic

        self.steps_taken = self.max_steps
        self._log("max_steps reached; returning current state")
        return state

    def _attempt_step(self, state, obs, t):
        last_err = ""
        for attempt in range(self.retries):
            ob_for_model = obs if attempt == 0 else (
                obs + f"\n[PRIOR STATE PATCH REJECTED: {last_err}]"
            )
            messages = self._build_messages(state, ob_for_model)
            self.prompt_sizes.append(_estimate_tokens(messages))

            raw = self.llm_call(messages)
            obj = _extract_json(raw)
            reasoning = obj.get("reasoning", "")
            patch = obj.get("state_patch", {}) or {}
            action = obj.get("action", {}) or {}

            errs = validate_patch(patch, self.schema)
            if not errs:
                self._log(f"step {t}.{attempt}: reasoning={reasoning[:60]!r}")
                return patch, action
            last_err = "; ".join(errs)
            self._log(f"step {t}.{attempt}: rejected -> {last_err}")

        return None, {}

    # -- reporting ----------------------------------------------------------- #
    def report(self) -> Dict[str, Any]:
        sizes = self.prompt_sizes
        if not sizes:
            return {"steps": 0, "prompt_tokens_per_step": [],
                    "prompt_growth_ratio": 1.0, "prompt_footprint_constant": True,
                    "cumulative_tokens": 0, "max_prompt_tokens": 0}
        # O(1) proof: prompt size must NOT scale with step count t.
        # Compare max vs min prompt tokens; a conversational baseline would
        # grow ~linearly per step, so this ratio explodes; here it stays flat.
        ratio = max(sizes) / max(min(sizes), 1)
        return {
            "steps": self.steps_taken or len(sizes),
            "prompt_tokens_per_step": sizes,
            "prompt_growth_ratio_max_over_min": round(ratio, 3),
            "prompt_footprint_constant": ratio < 3.0,
            "cumulative_tokens": sum(sizes),
            "max_prompt_tokens": max(sizes),
        }

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SKILL.state] {msg}")
