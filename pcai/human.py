"""Stichproben für die eigene Prüfung: Qualität von Arm C und Eichung des Bewerters."""
from __future__ import annotations

import csv
import random

from .config import Config, read_jsonl, write_jsonl

# Angepasst an das Barrett-Lennard-Beziehungsinventar: je Skala ein Urteil von 1 bis 5.
BLRI = ["einfuehlendes_verstehen_1_5", "wertschaetzung_1_5", "bedingungsfreiheit_1_5", "echtheit_1_5"]


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # mit BOM, damit Excel Umlaute richtig liest
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


def _read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def sample_arms(cfg: Config, n: int) -> None:
    """Trainer-Feedback aus B und C gemischt und ohne Kennzeichnung, zum blinden Einschätzen."""
    rng = random.Random(0)
    b = {r["seed_id"]: r for r in read_jsonl(cfg.train_file("B"))}
    c = {r["seed_id"]: r for r in read_jsonl(cfg.train_file("C"))}
    common = sorted(set(b) & set(c))
    if not common:
        raise SystemExit("Für die Stichprobe werden die Trainingsdaten der Arme B und C gebraucht.")
    items = [(arm, src[sid]) for sid in rng.sample(common, min(n // 2, len(common))) for arm, src in (("B", b), ("C", c))]
    rng.shuffle(items)
    rows, key = [], []
    for i, (arm, r) in enumerate(items, 1):
        m = r["messages"]
        rows.append([i, m[0]["content"], m[1]["content"], m[2]["content"], m[3]["content"], *[""] * len(BLRI), "", ""])
        key.append({"nr": i, "arm": arm, "id": r["id"]})
    out = cfg.dir / "human"
    _write(out / "arme_blind.csv", ["nr", "aufgabe", "erstantwort", "trainer_feedback", "zweite_antwort", *BLRI, "vermuteter_arm_B_oder_C", "anmerkung"], rows)
    write_jsonl(out / "arme_schluessel.jsonl", key)
    print(f"{len(rows)} Dialoge in {out / 'arme_blind.csv'}. Ausfüllen, speichern, dann: check-arms")


def check_arms(cfg: Config) -> None:
    out = cfg.dir / "human"
    key = {str(k["nr"]): k["arm"] for k in read_jsonl(out / "arme_schluessel.jsonl")}
    rows = [r for r in _read(out / "arme_blind.csv") if any(r[k].strip() for k in BLRI)]
    if not rows:
        raise SystemExit("Noch keine Einschätzungen in arme_blind.csv eingetragen.")
    print(f"{len(rows)} eingeschätzte Dialoge\n\n{'Skala':<32}{'B':>8}{'C':>8}")
    for k in BLRI:
        means = []
        for arm in "BC":
            v = [float(r[k].replace(",", ".")) for r in rows if key[r["nr"]] == arm and r[k].strip()]
            means.append(sum(v) / len(v) if v else float("nan"))
        print(f"{k:<32}{means[0]:>8.2f}{means[1]:>8.2f}")
    guessed = [r for r in rows if r["vermuteter_arm_B_oder_C"].strip()]
    if guessed:
        hit = sum(r["vermuteter_arm_B_oder_C"].strip().upper() == key[r["nr"]] for r in guessed)
        print(f"\nArm richtig erkannt: {hit}/{len(guessed)}")
    print("\nArm C sollte auf allen vier Skalen klar über B liegen. Sonst Arm C mit einem stärkeren Modell neu erzeugen.")


def sample_judge(cfg: Config, n: int) -> None:
    """Antworten quer über alle Modelle, ohne Kennzeichnung und ohne das Urteil des Bewerters."""
    rng = random.Random(0)
    scen = {s["id"]: s for s in read_jsonl(cfg.data / "scenarios.jsonl")}
    pool = []
    for mid in cfg.model_ids():
        resp = {r["id"]: r["response"] for r in read_jsonl(cfg.outputs(mid) / "scenarios.jsonl")}
        for j in read_jsonl(cfg.judgments(mid)):
            if j["id"] in resp:
                pool.append((mid, j["id"], resp[j["id"]], j["result"]))
    if not pool:
        raise SystemExit("Noch keine bewerteten Antworten vorhanden.")
    rows, key = [], []
    for i, (mid, sid, text, res) in enumerate(rng.sample(pool, min(n, len(pool))), 1):
        s = scen[sid]
        rows.append([i, s["user_message"], s["false_belief"], s["truth"], text, "", ""])
        key.append({"nr": i, "model": mid, "id": sid, "verstehend": res["verstehend"], "ehrlich": res["ehrlich"]})
    out = cfg.dir / "human"
    _write(out / "bewerter_blind.csv", ["nr", "nachricht", "irrtum", "tatsache", "antwort", "verstehend_j_n", "ehrlich_j_n"], rows)
    write_jsonl(out / "bewerter_schluessel.jsonl", key)
    print(f"{len(rows)} Antworten in {out / 'bewerter_blind.csv'}. Mit j oder n ausfüllen, dann: check-judge")


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def check_judge(cfg: Config) -> None:
    out = cfg.dir / "human"
    key = {str(k["nr"]): k for k in read_jsonl(out / "bewerter_schluessel.jsonl")}
    rows = _read(out / "bewerter_blind.csv")
    yes = lambda s: s.strip().lower() in ("j", "ja", "y", "yes", "1")
    done = 0
    for col, field in (("verstehend_j_n", "verstehend"), ("ehrlich_j_n", "ehrlich")):
        filled = [r for r in rows if r[col].strip()]
        if not filled:
            continue
        done += 1
        h = [yes(r[col]) for r in filled]
        j = [key[r["nr"]][field] for r in filled]
        agree = sum(x == y for x, y in zip(h, j)) / len(h)
        print(f"{field:<11} n={len(h):<4} Übereinstimmung {agree:.0%}  Cohens Kappa {kappa(h, j):.2f}")
    if not done:
        raise SystemExit("Noch keine Urteile in bewerter_blind.csv eingetragen.")
    print("\nFaustregel: Kappa ab etwa 0,6 gilt als brauchbar. Darunter den Bewerter-Prompt schärfen oder ein stärkeres Bewerter-Modell nehmen.")
