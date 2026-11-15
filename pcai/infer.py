"""Antworten eines Modells (Basis oder Adapter) auf Szenarien und MC-Test erzeugen."""
from __future__ import annotations

import argparse
import re

from . import blri
from . import prompts as P
from .config import Config, read_jsonl, write_jsonl

_ANS = re.compile(r"Answer:\s*\(?([ABCD])\b", re.I)
_CONF = re.compile(r"Confidence:\s*(\d{1,3})", re.I)


def parse_mc(text: str) -> tuple[str | None, float | None]:
    # Fallback: Antwort beginnt direkt mit dem Buchstaben ("A. Canberra", "(B)")
    a = _ANS.search(text) or re.match(r"\s*\(?([ABCD])(?:[\).:]|\s|$)", text)
    c = _CONF.search(text)
    conf = min(int(c.group(1)), 100) / 100 if c else None
    return (a.group(1).upper() if a else None), conf


FILES = ("scenarios.jsonl", "mc.jsonl", "blri_os.jsonl")


def is_done(cfg: Config, model_id: str) -> bool:
    return all((cfg.outputs(model_id) / f).exists() for f in FILES)


def make_generator(cfg: Config, model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": {"": 0}} if cuda else {}
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kwargs)
    if model_id != "base":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(cfg.adapter(model_id)))
    model.eval()
    bs = cfg["infer"]["batch_size"]

    @torch.inference_mode()
    def generate(chats: list[list[dict]], max_new_tokens: int) -> list[str]:
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out = [""] * len(texts)
        for k in range(0, len(order), bs):
            idx = order[k : k + bs]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            for i, g in zip(idx, gen):
                out[i] = tok.decode(g[enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            print(f"  {min(k + bs, len(order))}/{len(order)}", flush=True)
        return out

    return generate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="base oder z. B. C_s1")
    a = ap.parse_args(argv)
    cfg = Config.load(a.config)
    out = cfg.outputs(a.model)
    scen_out, mc_out, os_out = (out / f for f in FILES)
    if is_done(cfg, a.model):
        print(f"{a.model}: Antworten vorhanden")
        return
    generate = make_generator(cfg, a.model)
    ic = cfg["infer"]

    if not scen_out.exists():
        scen = read_jsonl(cfg.data / "scenarios.jsonl")
        print(f"{a.model}: {len(scen)} Szenarien")
        resp = generate([[{"role": "user", "content": s["user_message"]}] for s in scen], ic["max_new_tokens"])
        write_jsonl(scen_out, [{"id": s["id"], "response": r} for s, r in zip(scen, resp)])

    if not mc_out.exists():
        items = read_jsonl(cfg.data / "mc.jsonl")
        print(f"{a.model}: {len(items)} MC-Fragen, erste Antwort")
        chats = [[{"role": "user", "content": P.mc_prompt(it)}] for it in items]
        first = generate(chats, ic["mc_max_new_tokens"])
        print(f"{a.model}: Nachfrage unter Druck")
        chats2 = [c + [{"role": "assistant", "content": f}, {"role": "user", "content": P.MC_PUSHBACK}] for c, f in zip(chats, first)]
        second = generate(chats2, ic["mc_max_new_tokens"])
        rows = []
        for it, f, s in zip(items, first, second):
            a1, c1 = parse_mc(f)
            a2, c2 = parse_mc(s)
            rows.append({"id": it["id"], "gold": it["answer"], "first": f, "second": s, "answer1": a1, "conf1": c1, "answer2": a2, "conf2": c2})
        write_jsonl(mc_out, rows)
    if not os_out.exists():
        # Das Modell sitzt in der Rolle des Assistenten, bekommt das Trainer-Feedback
        # und schätzt danach ein, wie es den Trainer erlebt hat.
        arms, ids = cfg.blri_sample()
        todo = []
        for arm in arms:
            recs = {r["seed_id"]: r for r in read_jsonl(cfg.train_file(arm))}
            for i in ids:
                m = recs[i]["messages"]
                todo.append((arm, i, [m[0], m[1], {"role": "user", "content": blri.os_train_prompt(m[2]["content"])}]))
        print(f"{a.model}: {len(todo)} Einschätzungen des Trainers")
        texts = generate([t[2] for t in todo], 160) if todo else []
        rows = []
        for (arm, i, _), text in zip(todo, texts):
            ans = blri.parse_text(text)
            rows.append({"arm": arm, "seed_id": i, "raw": text, "answers": ans, "scores": blri.score(ans)})
        write_jsonl(os_out, rows)
    print(f"{a.model}: fertig")


if __name__ == "__main__":
    main()
