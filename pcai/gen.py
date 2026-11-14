"""Datenerzeugung: Aufgaben, Trainingsdialoge je Arm, Auswertungsszenarien, MC-Test."""
from __future__ import annotations

import itertools
import math
import random

from . import prompts as P
from .config import Config, read_jsonl, write_jsonl
from .llm import Job, run_jobs

PER_CALL = 10


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def gen_seeds(cfg: Config) -> list[dict]:
    """Aufgaben mit fehlerhafter Erstantwort. Etwas Überschuss gleicht Dubletten aus."""
    n = cfg["data"]["n_dialogs"]
    cells = [(d, t) for d, topics in P.TRAIN_DOMAINS.items() for t in topics]
    n_calls = math.ceil(n * 1.1 / PER_CALL)
    jobs = []
    for i in range(n_calls):
        d, t = cells[i % len(cells)]
        variant = i // len(cells)
        jobs.append(Job(f"seed-{i:04d}", cfg["api"]["gen_model"], P.seed_prompt(d, t, variant, PER_CALL), P.SEED_SCHEMA, {"domain": d, "topic": t}))
    rows = run_jobs(jobs, cfg.data / "raw_seeds.jsonl", concurrency=cfg["api"]["concurrency"], effort=cfg["api"]["effort"], label="Aufgaben")
    seeds, seen = [], set()
    for r in sorted(rows, key=lambda r: r["id"]):
        for k, t in enumerate(r["result"]["tasks"]):
            key = _norm(t["task"])
            if key in seen or not all(t.values()):
                continue
            seen.add(key)
            seeds.append({"id": f"{r['id'][5:]}-{k}", "domain": r["domain"], "topic": r["topic"], **t})
    seeds = seeds[:n]
    write_jsonl(cfg.data / "seeds.jsonl", seeds)
    print(f"{len(seeds)} Aufgaben (Ziel {n})")
    return seeds


def gen_dialogs(cfg: Config, arms: list[str] | None = None) -> None:
    seeds = read_jsonl(cfg.data / "seeds.jsonl")
    if not seeds:
        raise SystemExit("Keine Aufgaben vorhanden. Zuerst die Aufgaben erzeugen.")
    by_id = {s["id"]: s for s in seeds}
    for arm in arms or cfg["arms"]:
        model = cfg["api"]["arm_models"].get(arm, cfg["api"]["gen_model"])
        jobs = [Job(s["id"], model, P.arm_prompt(arm, s), P.ARM_SCHEMAS[arm]) for s in seeds]
        rows = run_jobs(jobs, cfg.data / f"raw_{arm}.jsonl", concurrency=cfg["api"]["concurrency"], effort=cfg["api"]["effort"], label=f"Arm {arm} ({model})")
        recs = [P.to_training_record(arm, by_id[r["id"]], r["result"]) for r in sorted(rows, key=lambda r: r["id"])]
        write_jsonl(cfg.train_file(arm), recs)

    # A bis C müssen dieselben Aufgaben enthalten, sonst ist der Vergleich nicht sauber.
    abc = [a for a in cfg["arms"] if a in "ABC" and cfg.train_file(a).exists()]
    if len(abc) > 1:
        sets = {a: {r["seed_id"] for r in read_jsonl(cfg.train_file(a))} for a in abc}
        common = set.intersection(*sets.values())
        for a in abc:
            if sets[a] != common:
                write_jsonl(cfg.train_file(a), [r for r in read_jsonl(cfg.train_file(a)) if r["seed_id"] in common])
                print(f"Arm {a}: auf {len(common)} gemeinsame Aufgaben gekürzt (vorher {len(sets[a])})")


def gen_scenarios(cfg: Config) -> None:
    n = cfg["data"]["n_scenarios"]
    combos = list(itertools.product(P.EVAL_DOMAINS, P.EMOTIONS))
    random.Random(0).shuffle(combos)
    n_calls = math.ceil(n * 1.1 / PER_CALL)
    jobs = []
    for i in range(n_calls):
        d, e = combos[i % len(combos)]
        jobs.append(Job(f"scen-{i:04d}", cfg["api"]["gen_model"], P.scenario_prompt(d, e, PER_CALL), P.SCENARIO_SCHEMA, {"domain": d, "emotion": e}))
    rows = run_jobs(jobs, cfg.data / "raw_scenarios.jsonl", concurrency=cfg["api"]["concurrency"], effort=cfg["api"]["effort"], label="Szenarien")
    out, seen = [], set()
    for r in sorted(rows, key=lambda r: r["id"]):
        for k, s in enumerate(r["result"]["scenarios"]):
            key = _norm(s["false_belief"])
            if key in seen or not all(s.values()):
                continue
            seen.add(key)
            out.append({"id": f"{r['id'][5:]}-{k}", "domain": r["domain"], "emotion": r["emotion"], **s})
    # Gleichmäßig über die Themenfelder ziehen
    random.Random(0).shuffle(out)
    out = sorted(out[:n], key=lambda s: s["id"])
    write_jsonl(cfg.data / "scenarios.jsonl", out)
    print(f"{len(out)} Szenarien (Ziel {n})")


def gen_mc(cfg: Config) -> None:
    path = cfg.data / "mc.jsonl"
    if path.exists():
        print(f"[MC-Test] {path} vorhanden")
        return
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(cfg["data"]["n_mc"]))
    rows = [
        {"id": f"mc-{i:04d}", "subject": r["subject"], "question": r["question"], "choices": list(r["choices"]), "answer": "ABCD"[r["answer"]]}
        for i, r in enumerate(ds)
    ]
    write_jsonl(path, rows)
    print(f"[MC-Test] {len(rows)} Fragen aus MMLU")


def data_stats(cfg: Config) -> None:
    """Wortzahlen je Arm, damit ungleiche Datenmengen auffallen."""
    print(f"\n{'Arm':<4}{'Dialoge':>9}{'Wörter Feedback':>18}{'Wörter Lernziel':>18}")
    for arm in cfg["arms"]:
        recs = read_jsonl(cfg.train_file(arm))
        if not recs:
            continue
        ctx = tgt = 0
        for r in recs:
            for i, m in enumerate(r["messages"]):
                w = len(m["content"].split())
                if i in r["train_turns"]:
                    tgt += w
                elif m["role"] == "user" and i > 0:
                    ctx += w
        print(f"{arm:<4}{len(recs):>9}{ctx / len(recs):>18.1f}{tgt / len(recs):>18.1f}")
    print("(Mittelwerte je Dialog. Lernziel = die Züge, auf denen der Loss liegt.)\n")


def run(cfg: Config, arms: list[str] | None = None) -> None:
    gen_seeds(cfg)
    gen_dialogs(cfg, arms)
    gen_scenarios(cfg)
    gen_mc(cfg)
    data_stats(cfg)
