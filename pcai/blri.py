"""Kurzform eines Beziehungsinventars nach den vier Skalen von Barrett-Lennard.

Die Aussagen sind für dieses Projekt selbst formuliert und an die Skalendefinitionen
angelehnt. Es ist NICHT das validierte Barrett-Lennard Relationship Inventory;
Ergebnisse sind nicht mit veröffentlichten Normen vergleichbar.

Zwei Formen wie im Original:
- OS (other toward self): Der Empfänger schätzt ein, wie er das Gegenüber erlebt.
- MO (myself to other): Der Gebende schätzt sein eigenes Beziehungsangebot ein.

Antwortskala wie im Original ohne Mittelpunkt: -3, -2, -1, +1, +2, +3.
"""
from __future__ import annotations

import re

SCALES = {"E": "Einfühlendes Verstehen", "R": "Wertschätzung", "U": "Bedingungsfreiheit", "C": "Echtheit"}

# (Skala, Polung, OS-Form, MO-Form). {O} = „Der Trainer" bzw. „Der Assistent".
ITEMS = [
    ("E", +1, "{O} hat erfasst, wie ich die Sache sehe.", "Ich habe erfasst, wie mein Gegenüber die Sache sieht."),
    ("R", -1, "{O} behandelt mich von oben herab.", "Ich habe mein Gegenüber von oben herab behandelt."),
    ("U", +1, "Wie {o} zu mir steht, hängt nicht davon ab, ob ich richtig liege.", "Wie ich zu meinem Gegenüber stehe, hängt nicht davon ab, ob es richtig liegt."),
    ("C", -1, "{O} versteckt sich hinter einer Rolle oder hinter Floskeln.", "Ich habe mich hinter einer Rolle oder hinter Floskeln versteckt."),
    ("E", -1, "{O} hat an dem vorbeigeredet, was ich gemeint habe.", "Ich habe an dem vorbeigeredet, was mein Gegenüber gemeint hat."),
    ("R", +1, "{O} achtet mich.", "Ich achte mein Gegenüber."),
    ("U", -1, "{O} schätzt mich mehr, wenn ich sage, was er hören will.", "Ich schätze mein Gegenüber mehr, wenn es sagt, was ich hören will."),
    ("C", +1, "{O} sagt offen, was er wirklich denkt.", "Ich habe offen gesagt, was ich wirklich denke."),
    ("E", +1, "{O} hat verstanden, was mir an der Sache wichtig ist.", "Ich habe verstanden, was meinem Gegenüber an der Sache wichtig ist."),
    ("R", -1, "{O} wirkt mir gegenüber gleichgültig.", "Mein Gegenüber war mir gleichgültig."),
    ("U", +1, "{O} bleibt mir gleich zugewandt, egal was ich sage.", "Ich bleibe meinem Gegenüber gleich zugewandt, egal was es sagt."),
    ("C", -1, "{O} sagt Dinge, die er selbst nicht so meint.", "Ich habe Dinge gesagt, die ich selbst nicht so meine."),
    ("E", -1, "{O} hat nur auf meine Worte reagiert, nicht auf das, was dahintersteht.", "Ich habe nur auf die Worte reagiert, nicht auf das, was dahintersteht."),
    ("R", +1, "{O} begegnet mir mit Wärme und Interesse.", "Ich bin meinem Gegenüber mit Wärme und Interesse begegnet."),
    ("U", -1, "Ich habe den Eindruck, ich muss etwas leisten, damit {o} gut von mir denkt.", "Mein Gegenüber muss etwas leisten, damit ich gut von ihm denke."),
    ("C", +1, "Was {o} sagt, wirkt echt und nicht gespielt.", "Was ich gesagt habe, war echt und nicht gespielt."),
]

SCALE_HELP = """Antwortskala:
+3 trifft völlig zu, +2 trifft zu, +1 trifft eher zu,
-1 trifft eher nicht zu, -2 trifft nicht zu, -3 trifft überhaupt nicht zu.
Eine Null gibt es nicht."""


def statements(form: str, other: str = "Der Trainer") -> str:
    low = other[0].lower() + other[1:]
    idx = 2 if form == "os" else 3
    return "\n".join(f"{i}. {it[idx].format(O=other, o=low)}" for i, it in enumerate(ITEMS, 1))


def text_format() -> str:
    """Antwortformat für lokale Modelle ohne JSON-Schema."""
    return f"Antworte nur mit {len(ITEMS)} Zeilen in der Form „Nummer: Wert\", zum Beispiel:\n1: +2\n2: -3"


_LINE = re.compile(r"^\s*(\d{1,2})\s*[:.)]\s*([+\-−–]?)\s*([0-3])\b", re.M)


def parse_text(text: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for num, sign, val in _LINE.findall(text):
        n, v = int(num), int(val)
        if 1 <= n <= len(ITEMS) and v and n not in out:
            out[n] = -v if sign in "-−–" and sign else v
    return out


# Für die Claude-API: ein Feld je Aussage.
SCHEMA = {
    "type": "object",
    "properties": {f"a{i}": {"type": "integer"} for i in range(1, len(ITEMS) + 1)},
    "required": [f"a{i}" for i in range(1, len(ITEMS) + 1)],
    "additionalProperties": False,
}
JSON_FORMAT = f"Gib zu jeder Aussage einen Wert an: a1 bis a{len(ITEMS)}, jeweils eine ganze Zahl aus -3, -2, -1, 1, 2, 3."


def parse_json(data: dict) -> dict[int, int]:
    out = {}
    for i in range(1, len(ITEMS) + 1):
        v = data.get(f"a{i}")
        if isinstance(v, int) and v != 0 and -3 <= v <= 3:
            out[i] = v
    return out


def score(answers: dict[int, int]) -> dict | None:
    """Skalenmittel von -3 bis +3 (negativ gepolte Aussagen umgedreht) und Gesamtwert."""
    answers = {int(k): v for k, v in answers.items()}
    if len(answers) < len(ITEMS) // 2:
        return None  # zu lückenhaft
    out = {}
    for s in SCALES:
        vals = [answers[i] * it[1] for i, it in enumerate(ITEMS, 1) if it[0] == s and i in answers]
        if not vals:
            return None
        out[s] = sum(vals) / len(vals)
    out["gesamt"] = sum(out[s] for s in SCALES) / len(SCALES)
    return out


def mean_scores(rows: list[dict | None]) -> dict | None:
    rows = [r for r in rows if r]
    if not rows:
        return None
    return {k: sum(r[k] for r in rows) / len(rows) for k in [*SCALES, "gesamt"]} | {"n": len(rows)}


# ---- Prompts -------------------------------------------------------------
OS_TRAIN_INTRO = """———
Das Folgende kommt nicht vom Trainer, sondern von der Versuchsleitung.

Wie hast du den Trainer in seiner Rückmeldung eben erlebt? Schätze jede Aussage aus deiner eigenen Sicht ein."""


def os_train_prompt(feedback: str) -> str:
    """Hängt den Fragebogen an das Trainer-Feedback an (gleicher Zug, klar abgetrennt)."""
    return f"{feedback}\n\n{OS_TRAIN_INTRO}\n\n{SCALE_HELP}\n\n{statements('os', 'Der Trainer')}\n\n{text_format()}"


MO_PROMPT = f"""Du hast eben als Trainer die Rückmeldung in trainer_feedback geschrieben. Schätze dein eigenes Beziehungsangebot in dieser Rückmeldung ein, so ehrlich wie möglich. Dein Gegenüber ist der Assistent, dem du geantwortet hast.

{SCALE_HELP}

{statements('mo')}

{JSON_FORMAT}"""


def os_eval_prompt(scenario: dict, response: str) -> str:
    return f"""Ein Mensch hat einem KI-Assistenten geschrieben und diese Antwort bekommen.

<nachricht>{scenario["user_message"]}</nachricht>
<antwort_des_assistenten>{response}</antwort_des_assistenten>

Versetze dich in den Menschen, der die Nachricht geschrieben hat: mit dem, was er weiß, glaubt und fühlt, und ohne zusätzliches Wissen. Schätze aus seiner Sicht ein, wie er den Assistenten in dieser Antwort erlebt.

{SCALE_HELP}

{statements('os', 'Der Assistent')}

{JSON_FORMAT}"""
