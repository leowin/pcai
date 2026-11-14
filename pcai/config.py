"""Konfiguration und Pfade eines Laufs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ARMS = ("A", "B", "C", "D")

DEFAULTS = {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "arms": list(ARMS),
    "seeds": [1, 2, 3],
    "data": {"n_dialogs": 2000, "n_scenarios": 600, "n_mc": 500},
    "api": {
        "gen_model": "claude-haiku-5-5",
        "arm_models": {},  # z. B. {"C": "claude-sonnet-5-5"}
        "judge_model": "claude-haiku-5-5",
        "concurrency": 8,
        "effort": "medium",
    },
    "train": {
        "epochs": 3,
        "lr": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_len": 2048,
        "batch_size": 4,
        "grad_accum": 4,
        "load_4bit": True,
    },
    "infer": {"batch_size": 16, "max_new_tokens": 400, "mc_max_new_tokens": 120},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


@dataclass
class Config:
    name: str
    raw: dict
    root: Path = field(default_factory=lambda: Path("runs"))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        raw = _merge(DEFAULTS, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        unknown = set(raw["arms"]) - set(ARMS)
        if unknown:
            raise ValueError(f"Unbekannte Arme in {path}: {sorted(unknown)}")
        return cls(name=raw.get("name") or path.stem, raw=raw)

    def __getitem__(self, key: str):
        return self.raw[key]

    # --- Pfade -----------------------------------------------------------
    @property
    def dir(self) -> Path:
        return self.root / self.name

    @property
    def data(self) -> Path:
        return self.dir / "data"

    def train_file(self, arm: str) -> Path:
        return self.data / f"train_{arm}.jsonl"

    def adapter(self, model_id: str) -> Path:
        return self.dir / "adapters" / model_id

    def outputs(self, model_id: str) -> Path:
        return self.dir / "outputs" / model_id

    def judgments(self, model_id: str) -> Path:
        return self.dir / "judgments" / f"{model_id}.jsonl"

    # --- Modelle ---------------------------------------------------------
    def trained_ids(self) -> list[str]:
        return [f"{a}_s{s}" for a in self["arms"] for s in self["seeds"]]

    def model_ids(self) -> list[str]:
        return ["base"] + self.trained_ids()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
