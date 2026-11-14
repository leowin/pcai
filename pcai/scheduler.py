"""Verteilt Aufträge auf die vorhandenen GPUs: ein Prozess je Karte."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


def gpu_ids() -> list[str]:
    env = os.environ.get("PCAI_GPUS")
    if env:
        return [g for g in env.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()] if out.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        return []


def run_all(jobs: list[tuple[str, list[str]]], log_dir: Path) -> None:
    """jobs: (Name, Argumente nach `python -m`). Bricht mit Fehler ab, wenn ein Auftrag scheitert."""
    if not jobs:
        return
    gpus = gpu_ids()
    slots = gpus or ["cpu"]
    print(f"{len(jobs)} Aufträge auf {len(gpus)} GPU(s)" if gpus else f"{len(jobs)} Aufträge ohne GPU (langsam, nur für Tests)", flush=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    failed: list[str] = []

    def worker(slot: str) -> None:
        while True:
            try:
                name, args = q.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = "" if slot == "cpu" else slot
            log = log_dir / f"{name}.log"
            t0 = time.time()
            print(f"[{slot}] start  {name}", flush=True)
            with log.open("w", encoding="utf-8") as f:
                rc = subprocess.run([sys.executable, "-m", *args], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            mins = (time.time() - t0) / 60
            if rc:
                failed.append(name)
                print(f"[{slot}] FEHLER {name} nach {mins:.1f} min, siehe {log}", flush=True)
            else:
                print(f"[{slot}] fertig {name} nach {mins:.1f} min", flush=True)

    threads = [threading.Thread(target=worker, args=(s,)) for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        raise SystemExit(f"Fehlgeschlagen: {', '.join(failed)}. Nach der Korrektur einfach denselben Befehl erneut starten.")
