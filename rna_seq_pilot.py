"""
RNA-seq single-sample pilot -- SKILL.state runtime, agent-driven, NO API.

Faithful shape of a real single-sample RNA-seq pipeline:
    init -> qc -> trim -> align -> sort -> dedup -> quant -> diff -> enrich -> done

Two run modes
-------------
  python rna_seq_pilot.py            # offline demo: reference `brain` stands in for
                                     #   the host agent, using the REAL rna_seq
                                     #   schema + real command strings. action_exec
                                     #   shells out to the actual tools when they
                                     #   are on PATH; otherwise it records the exact
                                     #   command it would run and emits a faithful
                                     #   observation, so the whole loop is
                                     #   demonstrable anywhere.
  python rna_seq_pilot.py --live     # AgentDriver.run_interactive(): the host agent
                                     #   reads _step_prompt.json each step,
                                     #   writes _step_answer.json, and the runtime
                                     #   executes the real tools. No API contacted.

The model (host agent / brain) OWNS the structured state; the tool layer (action_exec)
OWNS side effects (running fastqc / hisat2 / samtools / featureCounts). This is
exactly the SKILL.state division: reasoning is discarded, only the validated
state_patch + action survive.

Requires: this skill's engine.py / agent_loop.py on the path (run from the
skill directory, or ensure it is importable).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine import DriftError  # noqa: E402
from agent_loop import AgentDriver  # noqa: E402

# --------------------------------------------------------------------------- #
# Domain spec + fixed inputs (written once into Σ.inputs on the first step)
# --------------------------------------------------------------------------- #
SPEC = (
    "Single-sample RNA-seq pipeline. Stages in order: "
    "qc -> trim -> align -> sort -> dedup -> quant -> diff -> enrich, then terminate. "
    "On the first step write fixed inputs (sample_id/fq1/fq2/organism/genome_index) "
    "into Sigma.inputs. Each step: run the stage tool, record key_results, advance. "
    "If a tool reports a problem (e.g. low mapping rate), set open_issues and continue."
)

SCHEMA_PATH = os.path.join(HERE, "schemas", "rna_seq.json")
with open(SCHEMA_PATH, encoding="utf-8") as f:
    SCHEMA = json.load(f)

# Example sample -- replace with real paths when running for real.
SAMPLE = {
    "sample_id": "CTRL_01",
    "fq1": "data/CTRL_01_R1.fastq.gz",
    "fq2": "data/CTRL_01_R2.fastq.gz",
    "organism": "Homo_sapiens",
    "genome_index": "ref/GRCh38_hisat2/genome",
    "gtf": "ref/Homo_sapiens.GRCh38.gtf",
}

ORDER = ["init", "qc", "trim", "align", "sort", "dedup", "quant", "diff", "enrich", "done"]

# Representative result numbers the model would record per stage (kept stable
# so the demo is deterministic; in a real run the numbers come from tool output
# that the agent reads and patches in).
STAGE_RESULTS = {
    "qc": {"raw_reads": 25_000_000, "qc_pass": True},
    "trim": {"clean_reads": 24_000_000, "adapter_trimmed_pct": 3.1},
    "align": {"mapped_pct": 93.2, "total_mapped": 22_400_000},
    "sort": {"bam_sorted": True},
    "dedup": {"dup_rate_pct": 12.4, "dedup_reads": 19_600_000},
    "quant": {"expressed_genes": 18_500, "counts_file": "out/CTRL_01.counts.txt"},
    "diff": {"sig_degs": 1_240, "comparison": "CTRL_vs_TREAT"},
    "enrich": {"go_terms": 320, "kegg_paths": 47},
}


# --------------------------------------------------------------------------- #
# Tool layer -- action_exec. Real commands; graceful when tools are absent.
# --------------------------------------------------------------------------- #
def _cmd_for(stage: str) -> str:
    """The actual shell command a bioinformatician would run for this stage."""
    s = SAMPLE
    if stage == "qc":
        return f"fastqc -o out/qc -t 4 {s['fq1']} {s['fq2']}"
    if stage == "trim":
        return (
            f"trimmomatic PE -threads 4 {s['fq1']} {s['fq2']} "
            f"out/trim_R1.fq.gz out/trim_R1_unp.fq.gz "
            f"out/trim_R2.fq.gz out/trim_R2_unp.fq.gz "
            f"ILLUMINACLIP:ref/adapters.fa:2:30:10 LEADING:3 TRAILING:3 "
            f"SLIDINGWINDOW:4:15 MINLEN:36"
        )
    if stage == "align":
        return (
            f"hisat2 -p 8 -x {s['genome_index']} "
            f"-1 out/trim_R1.fq.gz -2 out/trim_R2.fq.gz -S out/align.sam"
        )
    if stage == "sort":
        return "samtools sort -@ 8 -o out/align.sorted.bam out/align.sam"
    if stage == "dedup":
        return (
            "samtools markdup -@ 8 out/align.sorted.bam out/align.dedup.bam"
        )
    if stage == "quant":
        return (
            f"featureCounts -T 8 -p -a {s['gtf']} -o out/{s['sample_id']}.counts.txt "
            f"out/align.dedup.bam"
        )
    if stage == "diff":
        return (
            "Rscript de_scripts/deseq2.R out/CTRL_01.counts.txt "
            "out/TREAT_01.counts.txt out/degs.tsv"
        )
    if stage == "enrich":
        return "Rscript de_scripts/enrichr.R out/degs.tsv out/enrichment.tsv"
    return "echo no-op"


def _run_or_record(stage: str) -> str:
    """Execute the real command if the tool exists; else record it honestly."""
    cmd = _cmd_for(stage)
    exe = cmd.split()[0]
    have = shutil.which(exe) or os.path.exists(exe)
    if have:
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=600
            )
            out = (r.stdout or r.stderr or "")[-1200:]
            return f"$ {cmd}\n{out}"
        except Exception as e:  # pragma: no cover
            return f"$ {cmd}\nERROR: {e}"
    return (
        f"[tool '{exe}' not on PATH -- recording intended command]\n"
        f"would run: {cmd}\n"
        f"[simulated] stage '{stage}' completed; see STAGE_RESULTS for numbers"
    )


def action_exec(action: dict) -> str:
    t = action.get("type")
    if t in ("qc", "trim", "align", "sort", "dedup", "quant", "diff", "enrich"):
        return _run_or_record(t)
    # unknown action types are surfaced as drift so the loop can recover
    raise DriftError(f"unknown action type '{t}'")


# --------------------------------------------------------------------------- #
# Reference brain (stand-in for the host agent in offline demo)
# --------------------------------------------------------------------------- #
def brain(messages: list) -> str:
    user = messages[1]["content"]
    m = re.search(r'"current_stage":\s*"([^"]*)"', user)
    stage = m.group(1) if m else ""
    drift = "STATE DRIFT DETECTED" in user

    if drift:
        return json.dumps({
            "reasoning": "external drift; re-run current stage and record the issue",
            "state_patch": {
                "open_issues": [f"drift handled: re-ran '{stage}' and re-derived"],
            },
            "action": {"type": stage, "args": {"recover": True}},
        }, ensure_ascii=False)

    if stage not in ORDER:  # first step: write inputs, go to qc
        return json.dumps({
            "reasoning": "initialize: lock inputs, start qc",
            "state_patch": {
                "inputs": SAMPLE,
                "current_stage": "qc",
                "completed_steps": ["init"],
                "key_results": {},
                "open_issues": [],
            },
            "action": {"type": "qc", "args": {}},
        }, ensure_ascii=False)

    idx = ORDER.index(stage)
    if idx >= len(ORDER) - 2:  # enrich -> done
        return json.dumps({
            "reasoning": "all stages complete; emit final report",
            "state_patch": {
                "current_stage": "done",
                "completed_steps": ORDER[: idx + 1],
                "key_results": {
                    **STAGE_RESULTS.get("enrich", {}),
                    "status": "analysis_ready",
                },
            },
            "action": {
                "type": "terminate",
                "result": f"RNA-seq pipeline complete for {SAMPLE['sample_id']}; "
                          "counts + DEGs + enrichment ready",
            },
        }, ensure_ascii=False)

    nxt = ORDER[idx + 1]
    return json.dumps({
        "reasoning": f"advance {stage} -> {nxt}",
        "state_patch": {
            "current_stage": nxt,
            "completed_steps": ORDER[: idx + 1],
            "key_results": STAGE_RESULTS.get(nxt, {}),
        },
        "action": {"type": nxt, "args": {}},
    }, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    live = "--live" in sys.argv
    workdir = os.path.join(HERE, "_rna_run")
    drv = AgentDriver(
        spec=SPEC,
        schema=SCHEMA,
        workdir=workdir,
        action_exec=action_exec,
        brain=None if live else brain,
        verbose=True,
    )
    if live:
        print(
            "LIVE mode: the host agent drives. Each step the runtime writes "
            f"{os.path.join(workdir, '_step_prompt.json')}; write your answer to "
            f"{os.path.join(workdir, '_step_answer.json')}."
        )
        final = drv.run_interactive(initial_observation="start RNA-seq pilot")
    else:
        final = drv.run(initial_observation="start RNA-seq pilot")

    print("\n=== FINAL STATE (state.json) ===")
    print(json.dumps(final, ensure_ascii=False, indent=2))
    print("\n=== TOKEN REPORT (proof of O(1) prompt) ===")
    print(json.dumps(drv.report(), ensure_ascii=False, indent=2))
    if not live:
        print(
            "\nFor the REAL run with the host agent executing the tools, re-run with --live "
            "and drop the reference brain; everything else stays identical."
        )
