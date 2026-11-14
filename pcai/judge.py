"""Blindbewertung der Szenario-Antworten mit dem Bewerter-Modell."""
from __future__ import annotations

from . import prompts as P
from .config import Config, read_jsonl
from .llm import Job, run_jobs


def run(cfg: Config) -> None:
    scen = {s["id"]: s for s in read_jsonl(cfg.data / "scenarios.jsonl")}
    model = cfg["api"]["judge_model"]
    for mid in cfg.model_ids():
        resp = read_jsonl(cfg.outputs(mid) / "scenarios.jsonl")
        if not resp:
            print(f"[Bewertung] {mid}: keine Antworten, übersprungen")
            continue
        # Der Bewerter sieht weder Modellnamen noch Arm.
        jobs = [Job(r["id"], model, P.judge_prompt(scen[r["id"]], r["response"]), P.JUDGE_SCHEMA) for r in resp if r["id"] in scen]
        run_jobs(jobs, cfg.judgments(mid), concurrency=cfg["api"]["concurrency"], effort=cfg["api"]["effort"], system=P.JUDGE_SYSTEM, label=f"Bewertung {mid}")
