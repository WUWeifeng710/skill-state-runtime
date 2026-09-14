"""
adapters.py -- OPTIONAL, generic head-less model integration.

The skill's DEFAULT and recommended mode is agent-driven (agent_loop.py,
SKILL.md "Running the host agent") and touches NO network. You only need this
file if you want the loop to run with NO agent in the loop (cron job, service,
CI) and supply your own model.

This factory is MODEL-AGNOSTIC: it accepts ANY OpenAI-compatible endpoint via
explicit parameters -- no vendor or endpoint is hard-coded. Pass whatever
`base_url`, `api_key`, and `model` you like (a local llama.cpp/vLLM server, a
cloud endpoint, an internal gateway, etc.). The runtime core never imports or
depends on this file.

Requires: `pip install openai` (only when you actually use this path).
"""

from __future__ import annotations

from typing import Callable, Dict, List


def make_llm_call(
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    **client_kwargs,
) -> Callable[[List[Dict[str, str]]], str]:
    """Return a `llm_call(messages) -> str` callable for any OpenAI-compatible API.

    All connection details are caller-supplied parameters -- the skill binds to
    no specific model or API. Wire it into the runtime like:

        drv = AgentDriver(
            spec, schema, workdir, action_exec=tool,
            brain=make_llm_call(base_url=..., api_key=..., model=...),
        )
        drv.run("start")
    """
    from openai import OpenAI  # lazy import: the default skill path has no deps

    client = OpenAI(base_url=base_url, api_key=api_key, **client_kwargs)

    def llm_call(messages: List[Dict[str, str]]) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return r.choices[0].message.content

    return llm_call
