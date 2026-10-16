"""Shared config, IDs, ontology and text utilities."""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "evireview-v3.1"

POLARITIES = ["positive", "negative", "neutral", "mixed"]
MODALITIES = ["asserted", "hypothetical", "question", "other_product"]
STATUSES = ["candidate", "accepted", "rejected", "needs_review"]
DERIVATIONS = ["direct", "recovered", "aggregate"]


# ---------------------------------------------------------------- config
@dataclass
class Config:
    raw: dict
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def p(self, name: str) -> Path:
        path = ROOT / self.raw["paths"][name]
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def profile(self) -> str:
        return self.raw["profile"]

    @property
    def offline(self) -> bool:
        return self.raw["extraction"]["provider"] == "mock"


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    load_env()
    return Config(yaml.safe_load(path.read_text()), path)


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip() and not os.environ.get(key.strip()):
            os.environ[key.strip()] = value.strip()


# ---------------------------------------------------------------- ids
def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def seeded_rank(seed: int, key: str) -> str:
    return sha(f"{seed}||{key}")


def user_key(user_id: str | None) -> str | None:
    if not user_id:
        return None
    salt = os.environ.get("EVIREVIEW_USER_SALT", "local-salt")
    return "user:" + sha(salt + "|" + user_id)[:24]


def claim_id(subject: str, aspect: str, polarity: str, context: str, modality: str,
             doc_id: str, start: int, end: int, derivation: str) -> str:
    return "claim:" + sha(canonical_json(
        [subject, "OPINION_ON", aspect, polarity, context, modality, doc_id, start, end, derivation]))[:32]


def evidence_id(doc_id: str, field: str, start: int, end: int) -> str:
    return "evidence:" + sha(canonical_json([doc_id, field, start, end]))[:32]


# ---------------------------------------------------------------- ontology
ONTOLOGY_FILE = "aspects_v2.yaml"
OTHER_ASPECT = "other"  # extractor escape hatch: an opinion that fits no aspect of the family (never accepted)


@lru_cache(maxsize=1)
def ontology() -> dict:
    return yaml.safe_load((Path(__file__).parent / ONTOLOGY_FILE).read_text())


def families() -> dict[str, dict]:
    return ontology().get("families", {})


def aspect_names(family: str | None = None) -> list[str]:
    """All aspects, or those of one product family (common aspects + the family's own).
    An unknown family (None, 'other') gets every aspect."""
    specs = ontology()["aspects"]
    if family not in families():
        return list(specs)
    return [a for a, s in specs.items() if family in s.get("families", [family])]


@lru_cache(maxsize=1)
def _family_patterns() -> dict[str, re.Pattern]:
    return {f: re.compile(r"\b(" + "|".join(re.escape(c) for c in sorted(spec["title_cues"], key=len, reverse=True)) + r")\b", re.I)
            for f, spec in families().items()}


def family_from_title(title: str | None) -> str | None:
    """Keyword family assignment (offline profile; the research profile classifies titles with the LLM).
    The earliest cue in the title wins, so 'Shampoo and Conditioner' -> hair_care, 'Lotion with Shea' -> skin."""
    best = None
    for f, pat in _family_patterns().items():
        m = pat.search(title or "")
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), f)
    return best[1] if best else None


def contexts() -> list[str]:
    return ontology()["contexts"]


def aspect_id(name: str) -> str:
    return f"aspect:beauty:{name}"


@lru_cache(maxsize=1)
def alias_patterns() -> dict[str, re.Pattern]:
    pats = {}
    for name, spec in ontology()["aspects"].items():
        terms = sorted({a.lower() for a in spec["aliases"]} | {name.replace("_", " ")}, key=len, reverse=True)
        pats[name] = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)
    return pats


def alias_set(name: str) -> set[str]:
    spec = ontology()["aspects"][name]
    return {a.lower() for a in spec["aliases"]} | {name.replace("_", " ")}


def aspects_mentioned(text: str, family: str | None = None) -> list[str]:
    allowed = set(aspect_names(family))
    return [a for a, p in alias_patterns().items() if a in allowed and p.search(text)]


def detect_context(text: str) -> str:
    low = text.lower()
    for ctx, cues in ontology()["context_cues"].items():
        if any(c in low for c in cues):
            return ctx
    return "unspecified"


# ---------------------------------------------------------------- text
def nfc(text: str | None) -> str:
    return unicodedata.normalize("NFC", text or "")


def review_document(title: str | None, text: str | None) -> str:
    """The single field all offsets refer to: NFC(title) + '\\n' + NFC(text)."""
    return nfc(title).strip() + "\n" + nfc(text).strip()


_SENT = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.M)


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    out = []
    for m in _SENT.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s >= 3:
            out.append((s, e, text[s:e]))
    return out


def locate_quote(doc: str, quote: str) -> tuple[str, int | None, int | None]:
    """Return (status, start, end). status: exact | recovered | ambiguous | missing."""
    quote = nfc(quote).strip()
    if not quote:
        return "missing", None, None
    hits = [m.start() for m in re.finditer(re.escape(quote), doc)]
    if len(hits) == 1:
        return "exact", hits[0], hits[0] + len(quote)
    if len(hits) > 1:
        return "ambiguous", hits[0], hits[0] + len(quote)
    # whitespace/case-insensitive recovery, unique match only
    pat = re.compile(r"\s+".join(re.escape(t) for t in quote.split()), re.I)
    hits = list(pat.finditer(doc))
    if len(hits) == 1:
        return "recovered", hits[0].start(), hits[0].end()
    if len(hits) > 1:
        return "ambiguous", hits[0].start(), hits[0].end()
    return "missing", None, None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default
