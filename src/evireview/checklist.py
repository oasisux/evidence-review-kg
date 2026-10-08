"""The 110 check items (40 queries, 30 claims, 30 qrels, 10 answers), shared by the human tab and the LLM auditor."""
from __future__ import annotations

import random

import pandas as pd

from .common import Config, read_json

SEED = 7


def queries(cfg: Config) -> list[dict]:
    return read_json(cfg.p("annotations") / "queries.json", [])


def claim_items(cfg: Config) -> list[dict]:
    ann = cfg.p("annotations")
    jclaims = read_json(ann / "judge_claims.json", {})
    samples = read_json(ann / "judge_claim_samples.json", {})
    rnd = random.Random(SEED)
    picked = []
    for run in ("g1", "c1", "budget"):
        ids = [c for c in samples.get(run, []) if c in jclaims]
        rnd.shuffle(ids)
        picked += [(run, c) for c in ids[:10]]
    rnd.shuffle(picked)
    g = cfg.p("graphs")
    cand = pd.concat([pd.read_parquet(g / f"candidates_{r}.parquet") for r in ("g1", "c1", "budget")])
    cand = cand.drop_duplicates("cand_id").set_index("cand_id")
    out = []
    for run, cid in picked:
        c = cand.loc[cid]
        out.append({"id": cid, "run": run, "review_id": c.review_id, "aspect": c.aspect, "polarity": c.polarity,
                    "quote": c.quote, "start": c.start, "end": c.end, "judge": jclaims[cid]})
    return out


def qrel_items(cfg: Config) -> list[dict]:
    qmap = {q["qid"]: q for q in queries(cfg)}
    qrels = read_json(cfg.p("annotations") / "qrels_judge.json", {})
    pairs = [(qid, p, g) for qid, m in qrels.items() for p, g in m.items() if qid in qmap]
    rnd = random.Random(SEED)
    items = []
    for grade in (2, 1, 0):
        bucket = [x for x in pairs if x[2] == grade]
        rnd.shuffle(bucket)
        items += bucket[:10]
    rnd.shuffle(items)
    return [{"id": f"{qid}|{p}", "qid": qid, "product_id": p, "judge": g} for qid, p, g in items]


def answer_items(cfg: Config) -> list[dict]:
    answers = read_json(cfg.p("runs") / "answers.json", [])
    cands = [a for a in answers if a.get("citations") and not a.get("abstained")]
    random.Random(SEED).shuffle(cands)
    return [{"id": f"{a['qid']}|{a['system']}", **a} for a in cands[:10]]
