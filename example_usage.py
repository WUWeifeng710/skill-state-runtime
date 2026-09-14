"""
SkillStateRuntime -- runnable demo (NO API key, NO network).

Uses a reference "brain" that stands in for the host agent (any model).
It drives a 4-stage pipeline so you can see:
  * the prompt footprint stays flat across steps (O(1)),
  * reasoning is never persisted,
  * a simulated external drift is recovered in one step (0 hallucination).

This is the SAME loop the live agent runs -- in real use you would
call drv.run_interactive(...) instead of passing a `brain`, and the host agent
itself supplies the per-step answer (see SKILL.md "Running the host agent").

Run:  python example_usage.py
"""

import json
import os
import re

from engine import DriftError
from agent_loop import AgentDriver

HERE = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.join(HERE, "_demo_run")

SPEC = (
    "Run the sample-processing pipeline: init -> trim -> align -> report. "
    "When report is produced, terminate with the summary."
)

SCHEMA = {
    "current_stage": {"type": "str"},
    "completed_steps": {"type": "list"},
    "key_results": {"type": "dict"},
    "open_issues": {"type": "list"},
}

STAGES = ["init", "trim", "align", "report"]
_drift_flag = {"done": False}


def brain(messages: list) -> str:
    """Reference stand-in for the host agent. Mirrors exactly what a host
    agent would output: read P + Σ + O + schema, return a JSON patch+action."""
    user = messages[1]["content"]
    m = re.search(r'"current_stage":\s*"([^"]*)"', user)
    stage = m.group(1) if m else "init"
    drift = "STATE DRIFT DETECTED" in user

    if drift:
        # external change: re-derive, record the issue, retry the same stage
        return json.dumps({
            "reasoning": "external drift; re-derive trim params and record issue",
            "state_patch": {"open_issues": ["drift handled: re-checked trim params"]},
            "action": {"type": "trim", "args": {"stage": "trim", "recheck": True}},
        }, ensure_ascii=False)

    idx = STAGES.index(stage) if stage in STAGES else 0
    if idx >= len(STAGES) - 1:
        return json.dumps({
            "reasoning": "all stages done, emit final report",
            "state_patch": {
                "current_stage": "done",
                "completed_steps": STAGES,
                "key_results": {"status": "ok", "reads_trimmed": 12_000_000},
            },
            "action": {"type": "terminate", "result": "pipeline complete"},
        }, ensure_ascii=False)

    nxt = STAGES[idx + 1]
    return json.dumps({
        "reasoning": f"advancing from {stage} to {nxt}",
        "state_patch": {
            "current_stage": nxt,
            "completed_steps": STAGES[: idx + 1],
            "key_results": {f"{nxt}_ok": True},
        },
        "action": {"type": nxt, "args": {"stage": nxt}},
    }, ensure_ascii=False)


def mock_tool(action: dict) -> str:
    t = action.get("type")
    if t == "trim" and not _drift_flag["done"]:
        _drift_flag["done"] = True
        # environment changed underneath us -> must be re-derived, not hallucinated
        raise DriftError("input file timestamp changed; re-check trim params")
    return f"observation: finished '{t}' successfully"


if __name__ == "__main__":
    drv = AgentDriver(
        spec=SPEC,
        schema=SCHEMA,
        workdir=WORKDIR,
        action_exec=mock_tool,
        brain=brain,  # <- replace with brain=None + run_interactive() for live agent
        verbose=True,
    )
    final = drv.run(initial_observation="pipeline start")
    print("\n=== FINAL STATE (state.json) ===")
    print(json.dumps(final, ensure_ascii=False, indent=2))
    rep = drv.report()
    print("\n=== TOKEN REPORT (proof of O(1) prompt) ===")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(
        "\nNOTE: for the LIVE agent, drop brain= and call "
        "drv.run_interactive('pipeline start'); the host agent reads "
        "and writes _step_answer.json each turn. No API is contacted."
    )
