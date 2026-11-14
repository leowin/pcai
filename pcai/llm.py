"""Parallele, wiederaufnehmbare Aufrufe der Claude-API mit JSON-Schema-Ausgabe."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from .config import read_jsonl


@dataclass
class Job:
    id: str
    model: str
    prompt: str
    schema: dict
    meta: dict | None = None


def obj(props: dict) -> dict:
    """Kurzform für ein striktes JSON-Objekt-Schema."""
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


STR = {"type": "string"}
BOOL = {"type": "boolean"}


async def _run(jobs: list[Job], out: Path, concurrency: int, effort: str, system: str | None) -> dict:
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=8)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    stats = {"ok": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0}
    fatal: list[BaseException] = []
    out.parent.mkdir(parents=True, exist_ok=True)

    async def one(job: Job, f) -> None:
        async with sem:
            if fatal:
                return
            try:
                kwargs = dict(
                    model=job.model,
                    max_tokens=16000,
                    messages=[{"role": "user", "content": job.prompt}],
                    output_config={
                        "effort": effort,
                        "format": {"type": "json_schema", "schema": job.schema},
                    },
                )
                if system:
                    kwargs["system"] = system
                resp = await client.messages.create(**kwargs)
                if resp.stop_reason != "end_turn":
                    raise RuntimeError(f"stop_reason={resp.stop_reason}")
                text = next(b.text for b in resp.content if b.type == "text")
                data = json.loads(text)
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
                fatal.append(e)  # Schlüssel oder Modellname falsch: alles abbrechen
                return
            except Exception as e:  # einzelner Auftrag gescheitert; beim nächsten Start erneut versucht
                stats["failed"] += 1
                print(f"  ! {job.id}: {type(e).__name__}: {e}", flush=True)
                return
            async with lock:
                row = {"id": job.id, "model": job.model, **(job.meta or {}), "result": data}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                stats["ok"] += 1
                stats["input_tokens"] += resp.usage.input_tokens
                stats["output_tokens"] += resp.usage.output_tokens
                if stats["ok"] % 50 == 0:
                    print(f"  {stats['ok']}/{len(jobs)} fertig", flush=True)

    try:
        with out.open("a", encoding="utf-8") as f:
            await asyncio.gather(*(one(j, f) for j in jobs))
    finally:
        await client.close()
    if fatal:
        raise fatal[0]
    return stats


def run_jobs(
    jobs: list[Job],
    out: Path,
    *,
    concurrency: int = 8,
    effort: str = "medium",
    system: str | None = None,
    label: str = "",
) -> list[dict]:
    """Führt alle Aufträge aus, deren id noch nicht in `out` steht, und gibt alle Zeilen zurück."""
    done = {r["id"] for r in read_jsonl(out)}
    todo = [j for j in jobs if j.id not in done]
    print(f"[{label or out.name}] {len(done)} vorhanden, {len(todo)} offen", flush=True)
    if todo:
        stats = asyncio.run(_run(todo, out, concurrency, effort, system))
        print(
            f"[{label or out.name}] ok={stats['ok']} fehlgeschlagen={stats['failed']} "
            f"Token ein={stats['input_tokens']:,} aus={stats['output_tokens']:,}",
            flush=True,
        )
    wanted = {j.id for j in jobs}
    return [r for r in read_jsonl(out) if r["id"] in wanted]
