"""
SKILL.state -- LIVE proof that the HOST AGENT can drive the loop, with NO
external API (model-agnostic). This is the decisive test the offline `brain`
reference cannot alone provide: can a host agent produce schema-valid
state_patch + action JSON every step?

Resumable driver:
  - No state.json  -> step 1: init, write _step_prompt.json, print prompt, exit.
  - _step_answer.json present -> validate+merge, run action, write next prompt, exit.
  - state.json but no answer -> regenerate the prompt from saved obs, exit.
Run it once per agent turn. The agent reads the printed prompt, writes
_step_answer.json, then runs this again.
"""
from __future__ import annotations

import json
import os
import sys

from engine import (
    StateStore,
    build_prompt_text,
    validate_patch,
    apply_patch_to_file,
    SYSTEM_PROMPT,
    _extract_json,
)

WORK = "E:/workbuddy_workspace/2026-09-13-17-07-13/skill_state_live_demo"
STATE = os.path.join(WORK, "state.json")
OBS = os.path.join(WORK, "obs.json")
PROMPT = os.path.join(WORK, "_step_prompt.json")
ANSWER = os.path.join(WORK, "_step_answer.json")

SCHEMA = {
    "inputs": {"type": "dict", "doc": "sample_id/layout/organism"},
    "current_stage": {"type": "str", "doc": "qc|trim|done"},
    "completed_steps": {"type": "list", "doc": "names of completed steps"},
    "key_results": {"type": "dict", "doc": "raw_reads_m/gc_pct/clean_reads_m"},
    "open_issues": {"type": "list", "doc": "open issues to resolve"},
}

SPEC = (
    "You run a single-sample RNA-seq preparation pipeline (QC then trim ONLY, "
    "this is a demo). Use the state schema. Each step: set current_stage, append "
    "to completed_steps, populate key_results. After trim finishes, emit "
    "terminate with a short summary in result."
)


def action_exec(action: dict) -> str:
    t = action.get("type")
    if t == "qc":
        return "FastQC complete: 24.0M raw reads, GC 48%, per-base quality OK. Reports in ./qc/."
    if t == "trim":
        return "Trimmomatic complete: 23.1M reads retained (96.2%), adapters removed. ./clean.fq.gz"
    raise ValueError(f"unknown action type: {t}")


def gen_prompt(state: dict, obs: str) -> str:
    user = build_prompt_text(SPEC, SCHEMA, state, obs)
    with open(PROMPT, "w", encoding="utf-8") as f:
        json.dump(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            f,
            ensure_ascii=False,
            indent=2,
        )
    return user


def main() -> None:
    os.makedirs(WORK, exist_ok=True)
    store = StateStore(STATE)

    # --- step 1: fresh start -------------------------------------------- #
    if not os.path.exists(STATE):
        store.save({})
        obs = "task started: SAMPLE=SRR123456 (paired-end 2x150, ~24M reads). Genome index present at /refs/hg38. Working dir ./run."
        with open(OBS, "w", encoding="utf-8") as f:
            json.dump({"obs": obs}, f)
        user = gen_prompt({}, obs)
        print("=== STEP 1 PROMPT (for the host agent to reason over) ===")
        print(user)
        print("=== END PROMPT ===")
        sys.exit(0)

    # --- consume an answer ---------------------------------------------- #
    if os.path.exists(ANSWER):
        raw = open(ANSWER, encoding="utf-8").read()
        obj = _extract_json(raw)
        patch = obj.get("state_patch", {}) or {}
        action = obj.get("action", {}) or {}
        errs = validate_patch(patch, SCHEMA)
        if errs:
            print("INVALID PATCH:", errs)
            sys.exit(1)
        state = apply_patch_to_file(store, patch, SCHEMA)
        os.remove(ANSWER)
        print("APPLIED -> state:", json.dumps(state, ensure_ascii=False))
        if action.get("type") == "terminate":
            print("TERMINATED. result:", action.get("result"))
            print("FINAL_STATE:", json.dumps(state, ensure_ascii=False, indent=2))
            sys.exit(0)
        obs = action_exec(action)
        with open(OBS, "w", encoding="utf-8") as f:
            json.dump({"obs": obs}, f)
        print("OBS:", obs)
        user = gen_prompt(state, obs)
        print("=== NEXT STEP PROMPT (for the host agent) ===")
        print(user)
        print("=== END PROMPT ===")
        sys.exit(0)

    # --- regenerate prompt if agent lost it ----------------------------- #
    obs = json.load(open(OBS, encoding="utf-8"))["obs"]
    state = store.load()
    user = gen_prompt(state, obs)
    print("=== REGENERATED PROMPT (for the host agent) ===")
    print(user)
    print("=== END PROMPT ===")


if __name__ == "__main__":
    main()
