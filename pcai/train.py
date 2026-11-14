"""QLoRA-Feinjustierung eines Arms mit einem Seed. Läuft als eigener Prozess auf einer GPU."""
from __future__ import annotations

import argparse
import json
import time

from .config import Config, read_jsonl


def encode(tok, rec: dict, max_len: int) -> dict | None:
    """Tokenisiert einen Dialog; Labels nur auf den Zügen in `train_turns`."""
    msgs = rec["messages"]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    labels = [-100] * len(ids)
    for t in rec["train_turns"]:
        start = len(tok.apply_chat_template(msgs[:t], tokenize=True, add_generation_prompt=True))
        end_ids = tok.apply_chat_template(msgs[: t + 1], tokenize=True, add_generation_prompt=False)
        if end_ids != ids[: len(end_ids)]:
            raise ValueError("Das Chat-Template ist nicht präfixstabil; Loss-Maske nicht bestimmbar.")
        labels[start : len(end_ids)] = ids[start : len(end_ids)]
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": labels}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    cfg = Config.load(a.config)
    tc = cfg["train"]
    model_id = f"{a.arm}_s{a.seed}"
    out = cfg.adapter(model_id)
    if (out / "done.json").exists():
        print(f"{model_id}: bereits trainiert")
        return
    set_seed(a.seed)
    cuda = torch.cuda.is_available()

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    recs = read_jsonl(cfg.train_file(a.arm))
    if not recs:
        raise SystemExit(f"Keine Trainingsdaten für Arm {a.arm}: {cfg.train_file(a.arm)}")
    data = [e for e in (encode(tok, r, tc["max_len"]) for r in recs) if e]
    print(f"{model_id}: {len(data)} Dialoge ({len(recs) - len(data)} zu lang und übersprungen)")

    kwargs = {}
    if cuda and tc["load_4bit"]:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs["device_map"] = {"": 0}
    elif cuda:
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kwargs)
    if cuda and tc["load_4bit"]:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=tc["lora_r"],
            lora_alpha=tc["lora_alpha"],
            lora_dropout=tc["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }

    args = TrainingArguments(
        output_dir=str(out / "tmp"),
        num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc["grad_accum"],
        learning_rate=float(tc["lr"]),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=a.seed,
        data_seed=a.seed,
        bf16=cuda,
        optim="paged_adamw_8bit" if cuda and tc["load_4bit"] else "adamw_torch",
        group_by_length=True,
        use_cpu=not cuda,
    )
    t0 = time.time()
    trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=collate)
    result = trainer.train()
    model.save_pretrained(str(out))
    (out / "done.json").write_text(
        json.dumps({"model_id": model_id, "base_model": cfg["base_model"], "n_dialogs": len(data), "train_loss": result.training_loss, "minutes": round((time.time() - t0) / 60, 1)}, indent=2),
        encoding="utf-8",
    )
    print(f"{model_id}: fertig, Loss {result.training_loss:.4f}")


if __name__ == "__main__":
    main()
