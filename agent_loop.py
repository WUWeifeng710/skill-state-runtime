"""
SKILL.state -- agent-driven runtime (NO external API required by default).

The HOST AGENT (any model / host process executing the loop) IS the model.
The runtime is model-agnostic: it externalizes state to a file and drives a
stepwise handshake, and works with whatever reasoning engine the caller plugs
in -- a host LLM, the user in the loop, or any `llm_call` callable.

    loop:
      1. build step prompt  -> write <workdir>/_step_prompt.json
      2. the host agent reads it, reasons, writes <workdir>/_step_answer.json
         (for offline self-test, a bundled `brain` callable stands in for the agent)
      3. runtime validates + applies patch, executes action, writes observation
      4. repeat until terminate

No network is required. To run FOR REAL the host agent supplies
_step_answer.json each turn; to run AUTOMATED offline/self-test pass a
`brain` callable that mimics the agent.

Two ways to actually use it
---------------------------
A) Automated self-test (this file, no agent needed):
       python example_usage.py
   -> uses a reference `brain` so the loop runs end-to-end offline.

B) The host agent drives it, turn by turn, no API:
       from agent_loop import AgentDriver
       drv = AgentDriver(spec, schema, workdir, action_exec=my_tool)
       drv.run_interactive(initial_observation="start")
   At each step the runtime writes _step_prompt.json; the host agent reads it,
   reasons, and writes _step_answer.json; the runtime applies it and continues.
   State lives in <workdir>/state.json the whole time.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from engine import (
    SYSTEM_PROMPT,
    StateStore,
    DriftError,
    _extract_json,
    _estimate_tokens,
    validate_patch,
    merge_state,
    build_prompt_text,
)


class AgentDriver:
    def __init__(
        self,
        spec: str,
        schema: Dict[str, Any],
        workdir: str,
        action_exec: Callable[[Dict[str, Any]], str],
        brain: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        max_steps: int = 200,
        verbose: bool = True,
    ):
        self.spec = spec
        self.schema = schema
        self.workdir = workdir
        self.action_exec = action_exec
        self.brain = brain  # stand-in for the host agent in self-test
        self.max_steps = max_steps
        self.verbose = verbose
        self.store = StateStore(os.path.join(workdir, "state.json"))
        self.prompt_file = os.path.join(workdir, "_step_prompt.json")
        self.answer_file = os.path.join(workdir, "_step_answer.json")
        self.prompt_sizes: List[int] = []
        self.steps_taken: int = 0
        os.makedirs(workdir, exist_ok=True)

    # -- logging ----------------------------------------------------------- #
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SKILL.state] {msg}")

    # -- step prompt ------------------------------------------------------- #
    def _build_messages(self, state: dict, observation: str) -> List[Dict[str, str]]:
        user = build_prompt_text(self.spec, self.schema, state, observation)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    # -- get the agent's answer (host agent or stand-in brain) ------------- #
    def _get_answer(self, state: dict, observation: str) -> str:
        messages = self._build_messages(state, observation)
        self.prompt_sizes.append(_estimate_tokens(messages))
        # Hand the prompt to the model. In real use the model is the host
        # agent reading _step_prompt.json; in self-test it is `brain`.
        with open(self.prompt_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        if self.brain is not None:
            return self.brain(messages)
        return self._wait_for_agent_answer()

    def _wait_for_agent_answer(self) -> str:
        self._log(
            f"waiting for host agent answer -> write JSON to {self.answer_file}"
        )
        while not os.path.exists(self.answer_file):
            time.sleep(0.5)
        with open(self.answer_file, encoding="utf-8") as f:
            raw = f.read()
        os.remove(self.answer_file)
        return raw

    # -- main loop --------------------------------------------------------- #
    def run(self, initial_observation: str) -> Dict[str, Any]:
        state = self.store.load()
        obs = initial_observation
        for t in range(1, self.max_steps + 1):
            patch, action = self._attempt_step(state, obs)
            state = merge_state(state, patch)
            self.store.save(state)
            self._log(f"step {t}: state={state}")

            atype = action.get("type")
            if atype == "terminate":
                self.steps_taken = t
                self._log(f"terminate: {action.get('result')}")
                return state

            try:
                obs = self.action_exec(action)
            except DriftError as e:
                obs = f"STATE DRIFT DETECTED: {e}. Re-derive Sigma from this."
                self._log(f"step {t}: drift -> {e} (re-derive next step)")

        self.steps_taken = self.max_steps
        self._log("max_steps reached; returning current state")
        return state

    def _attempt_step(self, state: dict, obs: str):
        raw = self._get_answer(state, obs)
        obj = _extract_json(raw)
        patch = obj.get("state_patch", {}) or {}
        action = obj.get("action", {}) or {}
        errs = validate_patch(patch, self.schema)
        if not errs:
            return patch, action
        # rollback-retry: feed the validation error back for one more pass
        self._log(f"patch rejected -> {errs}; feeding back")
        raw2 = self._get_answer(state, obs + f"\n[PRIOR STATE PATCH REJECTED: {'; '.join(errs)}]")
        obj2 = _extract_json(raw2)
        patch2 = obj2.get("state_patch", {}) or {}
        action2 = obj2.get("action", {}) or {}
        errs2 = validate_patch(patch2, self.schema)
        if errs2:
            raise ValueError(f"invalid state_patch after retry: {errs2}")
        return patch2, action2

    # -- interactive entry point for the live host agent ------------------- #
    def run_interactive(self, initial_observation: str) -> Dict[str, Any]:
        """Drive the loop turn-by-turn. The host agent reads _step_prompt.json
        each step and writes _step_answer.json. State persists in state.json."""
        if self.brain is not None:
            raise RuntimeError("run_interactive needs brain=None (live agent)")
        return self.run(initial_observation)

    # -- reporting --------------------------------------------------------- #
    def report(self) -> Dict[str, Any]:
        sizes = self.prompt_sizes
        if not sizes:
            return {
                "steps": 0,
                "prompt_tokens_per_step": [],
                "prompt_growth_ratio_max_over_min": 1.0,
                "prompt_footprint_constant": True,
                "cumulative_tokens": 0,
                "max_prompt_tokens": 0,
            }
        ratio = max(sizes) / max(min(sizes), 1)
        return {
            "steps": self.steps_taken or len(sizes),
            "prompt_tokens_per_step": sizes,
            "prompt_growth_ratio_max_over_min": round(ratio, 3),
            "prompt_footprint_constant": ratio < 3.0,
            "cumulative_tokens": sum(sizes),
            "max_prompt_tokens": max(sizes),
        }
