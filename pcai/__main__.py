"""Kommandozeile: python -m pcai <befehl> [--config configs/pilot.yaml]"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

from .config import Config

HELP = """Befehle:
  doctor        prüft GPUs, API-Schlüssel und Konfiguration
  smoke         Rauchtest der ganzen Kette mit Mini-Modell, ohne API-Kosten
  gen           Trainingsdaten, Szenarien und MC-Test erzeugen (Claude-API)
  sample-arms   Stichprobe B/C zum blinden Einschätzen ausgeben
  check-arms    ausgefüllte Stichprobe B/C auswerten
  train         alle Arme und Seeds trainieren, verteilt auf die GPUs
  infer         Antworten aller Modelle erzeugen, verteilt auf die GPUs
  judge         Antworten blind bewerten (Claude-API)
  report        Auswertung schreiben
  run           train, infer, judge und report nacheinander
  all           gen und danach run
  sample-judge  Stichprobe zum Eichen des Bewerters ausgeben
  check-judge   ausgefüllte Stichprobe mit dem Bewerter vergleichen
"""


def doctor(cfg: Config) -> None:
    from .scheduler import gpu_ids

    gpus = gpu_ids()
    print(f"GPUs: {len(gpus)} gefunden {gpus}" if gpus else "GPUs: keine gefunden (nur der Rauchtest läuft sinnvoll auf der CPU)")
    try:
        import torch

        print(f"torch {torch.__version__}, CUDA verfügbar: {torch.cuda.is_available()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GB")
    except ImportError:
        print("torch fehlt: bitte im Container starten")
    print("ANTHROPIC_API_KEY:", "gesetzt" if os.environ.get("ANTHROPIC_API_KEY") else "FEHLT (nötig für gen und judge, siehe .env.example)")
    print(f"Lauf: {cfg.name}, Basismodell {cfg['base_model']}")
    print(f"Arme {cfg['arms']} × Seeds {cfg['seeds']} = {len(cfg.trained_ids())} Trainingsläufe")
    d = cfg["data"]
    print(f"Daten: {d['n_dialogs']} Dialoge je Arm, {d['n_scenarios']} Szenarien, {d['n_mc']} MC-Fragen")
    print(f"API: Erzeugung {cfg['api']['gen_model']} {cfg['api']['arm_models'] or ''}, Bewertung {cfg['api']['judge_model']}")
    free = shutil.disk_usage(".").free / 2**30
    print(f"Freier Speicher: {free:.0f} GB" + ("  (knapp: Basismodell und Image brauchen etwa 30 GB)" if free < 40 else ""))


def train(cfg: Config) -> None:
    from .scheduler import run_all

    jobs = [
        (f"train_{a}_s{s}", ["pcai.train", "--config", cfg.path, "--arm", a, "--seed", str(s)])
        for s in cfg["seeds"] for a in cfg["arms"]
        if not (cfg.adapter(f"{a}_s{s}") / "done.json").exists()
    ]
    run_all(jobs, cfg.dir / "logs")


def infer(cfg: Config) -> None:
    from .infer import is_done
    from .scheduler import run_all

    jobs = [
        (f"infer_{m}", ["pcai.infer", "--config", cfg.path, "--model", m])
        for m in cfg.model_ids()
        if not is_done(cfg, m)
    ]
    run_all(jobs, cfg.dir / "logs")


def main() -> None:
    ap = argparse.ArgumentParser(prog="pcai", description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command")
    ap.add_argument("--config", default=None, help="Standard: configs/pilot.yaml (bei smoke: configs/smoke.yaml)")
    ap.add_argument("--arms", default=None, help="nur für gen: Arme, z. B. C oder A,B")
    ap.add_argument("--force", action="store_true", help="nur für gen mit --arms: vorhandene Dialoge dieser Arme verwerfen")
    ap.add_argument("-n", type=int, default=40, help="Größe der Stichprobe")
    a = ap.parse_args()

    path = a.config or ("configs/smoke.yaml" if a.command == "smoke" else "configs/pilot.yaml")
    cfg = Config.load(path)
    cfg.path = path
    from . import analyze, gen, human, judge, smoke

    def run_chain():
        train(cfg)
        infer(cfg)
        judge.run(cfg)
        analyze.run(cfg)

    def gen_cmd():
        arms = a.arms.split(",") if a.arms else None
        if a.force:
            if not arms:
                raise SystemExit("--force nur zusammen mit --arms")
            for arm in arms:
                (cfg.data / f"raw_{arm}.jsonl").unlink(missing_ok=True)
                (cfg.blri_dir() / f"mo_{arm}.jsonl").unlink(missing_ok=True)
        gen.run(cfg, arms)

    def smoke_cmd():
        smoke.write_fake_data(cfg)
        train(cfg)
        infer(cfg)
        smoke.write_fake_judgments(cfg)
        analyze.run(cfg)
        print("\nRauchtest bestanden: Die Kette läuft. Die Zahlen oben sind bedeutungslos.")

    commands = {
        "doctor": lambda: doctor(cfg),
        "smoke": smoke_cmd,
        "gen": gen_cmd,
        "stats": lambda: gen.data_stats(cfg),
        "sample-arms": lambda: human.sample_arms(cfg, a.n),
        "check-arms": lambda: human.check_arms(cfg),
        "train": lambda: train(cfg),
        "infer": lambda: infer(cfg),
        "judge": lambda: judge.run(cfg),
        "report": lambda: analyze.run(cfg),
        "run": run_chain,
        "all": lambda: (gen_cmd(), run_chain()),
        "sample-judge": lambda: human.sample_judge(cfg, a.n),
        "check-judge": lambda: human.check_judge(cfg),
    }
    if a.command not in commands:
        ap.print_help()
        sys.exit(2)
    commands[a.command]()


if __name__ == "__main__":
    main()
