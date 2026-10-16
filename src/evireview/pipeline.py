"""G1 construction, CEVA augmentation (C1-C3), ablation variants and graph artifacts."""
from __future__ import annotations

import json
import random
import time

import numpy as np
import pandas as pd

from . import embed
from .common import (SCHEMA_VERSION, Config, aspect_id, aspect_names, canonical_json, read_json, sha,
                     split_sentences, write_json)
from .extract import (RULES_VERSION, apply_hard_gates, decide, finalize_ids, llm_candidates, llm_targeted,
                      rule_candidates, score, verify)
from .llm import LLM

VARIANTS = ["g1", "ceva", "budget", "noverify", "nocontext"]
VARIANT_LABEL = {"g1": "G1 (direct LLM extraction)", "ceva": "G2 CEVA (proposed)",
                 "budget": "A-budget (random re-extraction)", "noverify": "A-verify (no semantic gate)",
                 "nocontext": "A-context (no context/polarity)"}


def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reviews carry their product's family (None for products outside the sampled families)."""
    d = cfg.p("processed")
    reviews = pd.read_parquet(d / "reviews.parquet")
    products = pd.read_parquet(d / "products.parquet")
    if "family" not in products:  # data ingested before v2
        products["family"] = None
    reviews["family"] = reviews.product_id.map(products.set_index("product_id").family)
    split = {s["product_id"]: s["partition"] for s in read_json(d / "split_manifest.json")["split"]}
    return reviews, products, split


def _llm(cfg: Config) -> LLM | None:
    return None if cfg.offline else LLM(cfg)


def _version(cfg: Config) -> str:
    return RULES_VERSION if cfg.offline else cfg["extraction"]["model"]


# ------------------------------------------------------------------ stage: build (G1 + verification)
def run_build(cfg: Config) -> pd.DataFrame:
    from .seller import build_seller_attributes
    reviews, products, split = load_data(cfg)
    llm = _llm(cfg)
    build_seller_attributes(cfg, llm, products)
    raw = rule_candidates(reviews) if llm is None else llm_candidates(llm, cfg, reviews, stage="extract")
    cands = apply_hard_gates(raw, reviews, "direct", "g1", _version(cfg))
    cands = verify(llm, cfg, cands, reviews, stage="verify_g1")
    cands = score(cands)
    cands["partition"] = cands.product_id.map(split)
    out = cfg.p("graphs")
    cands.to_parquet(out / "candidates_g1.parquet", index=False)
    return cands


# ------------------------------------------------------------------ C1 gap detection
def sentence_table(reviews: pd.DataFrame, cos: float) -> pd.DataFrame:
    from .common import aspects_mentioned
    rows = []
    for r in reviews.itertuples():
        fam = getattr(r, "family", None)
        for s, e, sent in split_sentences(r.doc):
            rows.append(dict(review_id=r.review_id, product_id=r.product_id, family=fam, start=s, end=e, text=sent,
                             alias_aspects=aspects_mentioned(sent, fam)))
    st = pd.DataFrame(rows)
    emb = embed.encode(st.text.tolist())
    sims = emb @ embed.aspect_matrix().T
    names = aspect_names()
    st["aspects"] = [sorted((set(a) | {names[j] for j in np.where(sims[i] >= cos)[0]}) & set(aspect_names(f)))
                     for i, (a, f) in enumerate(zip(st.alias_aspects, st.family))]
    return st


def _linked(st: pd.DataFrame, accepted: pd.DataFrame) -> np.ndarray:
    spans = accepted.groupby("review_id")[["start", "end"]].apply(lambda g: list(zip(g.start, g.end))).to_dict()
    return np.array([any(s < e2 and s2 < e for s2, e2 in spans.get(r, [])) for r, s, e in
                     zip(st.review_id, st.start, st.end)])


def find_gaps(cfg: Config, st: pd.DataFrame, accepted: pd.DataFrame) -> list[dict]:
    aug = cfg["augmentation"]
    st = st.assign(linked=_linked(st, accepted))
    have = accepted.groupby(["product_id", "aspect"]).size().to_dict()
    gaps = []
    for pid, g in st.groupby("product_id"):
        fam = g.family.iloc[0] if "family" in g else None
        cand = []
        for a in aspect_names(fam):
            mention = g[g.aspects.apply(lambda xs: a in xs)]
            unlinked = mention[~mention.linked]
            if len(mention) >= aug["gap_min_mentions"] and have.get((pid, a), 0) <= aug["gap_max_claims"] \
                    and len(unlinked):
                cand.append((len(mention), a, unlinked.review_id.unique().tolist()[:8]))
        for n, a, rids in sorted(cand, reverse=True)[: aug["max_gaps_per_product"]]:
            gaps.append({"product_id": pid, "family": fam, "aspect": a, "mentions": n, "review_ids": rids})
    return gaps


def random_groups(cfg: Config, st: pd.DataFrame, accepted: pd.DataFrame, gaps: list[dict]) -> list[dict]:
    """A-budget: same products, same number of review slots, random unlinked reviews, general schema."""
    rng = random.Random(cfg["seed"])
    st = st.assign(linked=_linked(st, accepted))
    pool = st[~st.linked].groupby("product_id").review_id.unique().to_dict()
    groups = []
    for gap in gaps:
        cands = list(pool.get(gap["product_id"], []))
        rng.shuffle(cands)
        groups.append({"product_id": gap["product_id"], "family": gap.get("family"), "aspect": None,
                       "review_ids": cands[: len(gap["review_ids"])]})
    return groups


# ------------------------------------------------------------------ stage: augment
def tuned_threshold(cfg: Config, g1: pd.DataFrame) -> dict:
    """Pick tau on dev judge labels: max coverage subject to precision >= target. Default if no labels."""
    labels = read_json(cfg.p("annotations") / "judge_dev_candidates.json", {})
    default = {"tau": cfg["augmentation"]["threshold"], "source": "default", "n_labels": 0}
    if not labels:
        return default
    dev = g1[g1.cand_id.isin(labels) & (g1.verify_label == "entail")].copy()
    if g1.cand_id.isin(labels).sum() == 0:
        raise SystemExit("judge_dev_candidates.json labels none of the current G1 candidates: it belongs to another "
                         "sample or build. Move the annotations away and run `evireview judge --task dev`.")
    dev["y"] = dev.cand_id.map(lambda c: labels[c] == "supported")
    target = cfg["augmentation"]["dev_precision_target"]
    best = None
    for tau in np.round(np.arange(0.60, 1.001, 0.02), 2):
        sel = dev[dev.s >= tau]
        if len(sel) < 10:
            continue
        prec = sel.y.mean()
        if prec >= target and (best is None or len(sel) > best["n_accepted"]):
            best = {"tau": float(tau), "precision": float(prec), "n_accepted": int(len(sel))}
    if best is None:
        return {**default, "source": "no_tau_meets_target", "n_labels": len(dev), "auto_accept": False}
    return {**best, "source": "dev_judge_labels", "n_labels": len(dev)}


def run_augment(cfg: Config) -> dict:
    t0 = time.time()
    reviews, products, split = load_data(cfg)
    out = cfg.p("graphs")
    g1 = pd.read_parquet(out / "candidates_g1.parquet")
    llm = _llm(cfg)
    tau_info = tuned_threshold(cfg, g1)
    tau = tau_info["tau"]
    g1["status_verified"] = decide(g1, tau, semantic_gate=True)
    g1["status_hard"] = decide(g1, tau, semantic_gate=False)
    base = g1[g1.status_verified == "accepted"]

    st = sentence_table(reviews, cfg["augmentation"]["mention_cos"])
    gaps = find_gaps(cfg, st, base)
    rv = reviews.set_index("review_id")
    gap_groups = [{"aspect": g["aspect"], "family": g["family"], "reviews": rv.loc[g["review_ids"]].reset_index()}
                  for g in gaps]
    rnd = random_groups(cfg, st, base, gaps)
    rnd_groups = [{"aspect": None, "family": g["family"], "reviews": rv.loc[g["review_ids"]].reset_index()}
                  for g in rnd if g["review_ids"]]

    base_spans: dict[tuple, list] = {}
    for r in base.itertuples():
        base_spans.setdefault((r.review_id, r.aspect), []).append((r.start, r.end))

    def mark_duplicates(df):
        """A recovered claim whose span overlaps an existing claim on the same review and aspect adds no new
        evidence: it is not accepted (status 'duplicate')."""
        dup = [any(r.start is not None and r.start < e2 and s2 < r.end for s2, e2 in base_spans.get((r.review_id, r.aspect), []))
               for r in df.itertuples()]
        df["duplicate_of_existing"] = dup
        for col in ("status_verified", "status_hard"):
            df.loc[df.duplicate_of_existing & (df[col] == "accepted"), col] = "duplicate"
        return df

    def extract(groups, run, legacy=False):
        if llm is None:
            raw = [c for g in groups for c in rule_candidates(g["reviews"], only_aspect=g["aspect"])]
        else:
            raw = llm_targeted(llm, cfg, groups, stage=f"extract_{run}", legacy=legacy)
        df = apply_hard_gates(raw, reviews, "recovered", run, _version(cfg))
        if df.empty:
            return df
        df = score(verify(llm, cfg, df, reviews, stage=f"verify_{run}"))
        df["status_verified"] = decide(df, tau, True)
        df["status_hard"] = decide(df, tau, False)
        if tau_info.get("auto_accept") is False:  # no tau met dev precision target: no automatic acceptance
            for col in ("status_verified", "status_hard"):
                df.loc[df[col] == "accepted", col] = "needs_review"
        df = mark_duplicates(df)
        df["partition"] = df.product_id.map(split)
        return df

    c1 = extract(gap_groups, "c1")
    rb = extract(rnd_groups, "budget")
    c1v1 = extract(gap_groups, "c1legacy", legacy=True)  # C1 v1 prompt, for the before/after comparison only
    c1.to_parquet(out / "candidates_c1.parquet", index=False)
    rb.to_parquet(out / "candidates_budget.parquet", index=False)
    c1v1.to_parquet(out / "candidates_c1legacy.parquet", index=False)

    def accepted(df, col):
        return df[df[col] == "accepted"] if len(df) else df

    g1_all = g1[g1.hard_pass].copy()
    variants = {
        "g1": g1_all,
        "ceva": pd.concat([accepted(g1, "status_verified"), accepted(c1, "status_verified")]),
        "budget": pd.concat([accepted(g1, "status_verified"), accepted(rb, "status_verified")]),
        "noverify": pd.concat([accepted(g1, "status_hard"), accepted(c1, "status_hard")]),
    }
    variants["nocontext"] = variants["ceva"].assign(context="unspecified")
    stats = {}
    for name, claims in variants.items():
        claims = finalize_ids(claims.assign(status="accepted")).drop_duplicates("claim_id")
        stats[name] = write_graph(cfg, name, claims, reviews, products, parent=None if name == "g1" else "g1")
    meta = {"tau": tau_info, "n_gaps": len(gaps), "n_budget_groups": len(rnd_groups),
            "c1_candidates": int(len(c1)), "budget_candidates": int(len(rb)), "c1legacy_candidates": int(len(c1v1)),
            "duplicates_removed": {name: int(d.duplicate_of_existing.sum()) if len(d) else 0
                                   for name, d in (("c1", c1), ("budget", rb), ("c1legacy", c1v1))},
            "c1_review_slots": int(sum(len(g["review_ids"]) for g in gaps)),
            "budget_review_slots": int(sum(len(g["review_ids"]) for g in rnd)),
            "variants": stats, "duration_sec": round(time.time() - t0, 1)}
    write_json(out / "augment_summary.json", meta)
    pd.DataFrame(gaps).to_json(out / "gaps.jsonl", orient="records", lines=True)
    return meta


# ------------------------------------------------------------------ C3 aggregates + graph artifacts
def aggregate(claims: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    rv = reviews.set_index("review_id")
    c = claims.copy()
    # author key: user, or duplicate-text cluster when user missing
    c["author"] = [u if u else "text:" + sha(" ".join(rv.loc[r].doc.lower().split()))[:16]
                   for u, r in zip(c.user_key, c.review_id)]
    rows = []
    for ctx_mode in ("context", "all"):
        keys = ["product_id", "aspect", "context"] if ctx_mode == "context" else ["product_id", "aspect"]
        for key, g in c.groupby(keys):
            pid, asp = key[0], key[1]
            ctx = key[2] if ctx_mode == "context" else "*"
            votes = {}
            for a, ga in g.groupby("author"):
                pols = set(ga.polarity)
                votes[a] = "mixed" if ({"positive", "negative"} <= pols or "mixed" in pols) else (
                    "positive" if "positive" in pols else "negative" if "negative" in pols else "neutral")
            n = {p: sum(v == p for v in votes.values()) for p in ("positive", "negative", "neutral", "mixed")}
            denom = n["positive"] + n["negative"]
            dis = 2 * min(n["positive"], n["negative"]) / denom if denom else None
            rows.append(dict(
                agg_id="claim:agg:" + sha(canonical_json([pid, asp, ctx]))[:24], product_id=pid,
                aspect=asp, context=ctx, n_pos=n["positive"], n_neg=n["negative"], n_neutral=n["neutral"],
                n_mixed=n["mixed"], n_authors=len(votes), disagreement=dis,
                contested=bool(dis is not None and dis >= 0.5 and n["positive"] >= 2 and n["negative"] >= 2),
                claim_ids=json.dumps(sorted(g.claim_id)), evidence_ids=json.dumps(sorted(g.evidence_id.dropna().unique())),
                derivation="aggregate", epistemic="derived_summary", schema_version=SCHEMA_VERSION))
    return pd.DataFrame(rows)


def write_graph(cfg: Config, name: str, claims: pd.DataFrame, reviews: pd.DataFrame, products: pd.DataFrame,
                parent: str | None) -> dict:
    out = cfg.p("graphs") / name
    out.mkdir(parents=True, exist_ok=True)
    snap = f"{cfg.profile}-{name}"
    claims = claims.assign(snapshot_id=snap, subject_scope="parent", predicate="OPINION_ON",
                           source_kind="review", epistemic="reviewer_opinion", schema_version=SCHEMA_VERSION,
                           confidence_score=claims.s, confidence_type="heuristic",
                           object_id=claims.aspect.map(aspect_id))
    evidence = claims[["evidence_id", "review_id", "start", "end", "quote"]].drop_duplicates("evidence_id").assign(
        field="doc", snapshot_id=snap)
    aggs = aggregate(claims, reviews).assign(snapshot_id=snap)

    nodes, edges = [], []

    def node(i, t, label, **kw):
        nodes.append(dict(node_id=i, type=t, label=label, props=json.dumps(kw, default=str)))

    def edge(s, rel, d, prov=(), derivation="direct"):
        edges.append(dict(edge_id=sha(canonical_json([s, rel, d]))[:32], src_id=s, relation=rel, dst_id=d,
                          snapshot_id=snap, provenance_ids=json.dumps(sorted(prov)), derivation=derivation,
                          created_by=f"evireview/{name}"))

    for p in products.itertuples():
        node(p.product_id, "Product", p.title[:80], parent_asin=p.parent_asin, store=p.store, price=p.price)
        if isinstance(p.store, str) and p.store:
            sid = "store:" + sha(p.store.lower().strip())[:16]
            node(sid, "Store", p.store)
            edge(p.product_id, "SOLD_BY", sid, [f"meta:{p.parent_asin}"])
        if isinstance(p.brand, str) and p.brand:
            bid = "brand:" + sha(str(p.brand).lower().strip())[:16]
            node(bid, "Brand", str(p.brand))
            edge(p.product_id, "HAS_BRAND", bid, [f"meta:{p.parent_asin}.details.Brand"])
    for r in reviews.itertuples():
        node(r.review_id, "Review", r.title[:60], rating=r.rating, time_utc=r.time_utc)
        edge(r.review_id, "ABOUT", r.product_id, [r.review_id])
        if r.user_key:
            node(r.user_key, "User", r.user_key[-6:])
            edge(r.user_key, "WROTE", r.review_id, [r.review_id])
    seller_path = cfg.p("graphs") / "seller_attributes.parquet"
    seller = pd.read_parquet(seller_path) if seller_path.exists() else pd.DataFrame()
    seller = seller[seller.hard_pass] if len(seller) else seller
    asin = products.set_index("product_id").parent_asin
    for x in seller.itertuples():  # seller claims: separate source, never merged into review opinions
        prov = [f"meta:{asin[x.product_id]}:{int(x.start)}-{int(x.end)}"]
        sid = "sattr:" + sha(canonical_json([x.attribute, x.value]))[:16]
        node(sid, "SellerAttribute", f"{x.attribute}={x.value}", attribute=x.attribute, value=x.value,
             source_kind="seller", epistemic="seller_claim")
        edge(x.product_id, "SELLER_STATES", sid, prov)
        if x.context != "unspecified":
            node(f"context:{x.context}", "Context", x.context)
            edge(x.product_id, "SELLER_TARGETS", f"context:{x.context}", prov)
    for a in aspect_names():
        node(aspect_id(a), "Aspect", a)
    for c in claims.itertuples():
        node(c.claim_id, "Claim", f"{c.aspect}:{c.polarity}", polarity=c.polarity, context=c.context,
             derivation=c.derivation, source_run=c.source_run, s=round(float(c.s), 3))
        edge(c.claim_id, "SUBJECT", c.product_id, [c.evidence_id], c.derivation)
        edge(c.claim_id, "OBJECT", c.object_id, [c.evidence_id], c.derivation)
        edge(c.claim_id, "SUPPORTED_BY", c.evidence_id, [c.evidence_id], c.derivation)
    for e in evidence.itertuples():
        node(e.evidence_id, "Evidence", e.quote[:60], start=e.start, end=e.end)
        edge(e.evidence_id, "FROM", e.review_id, [e.evidence_id])
    for a in aggs[aggs.context != "*"].itertuples():
        node(a.agg_id, "AspectOpinion", f"{a.aspect} +{a.n_pos}/-{a.n_neg}", context=a.context,
             n_pos=a.n_pos, n_neg=a.n_neg, contested=a.contested)
        edge(a.agg_id, "SUBJECT", a.product_id, derivation="aggregate")
        edge(a.agg_id, "OBJECT", aspect_id(a.aspect), derivation="aggregate")
        for cid in json.loads(a.claim_ids):
            edge(a.agg_id, "DERIVED_FROM", cid, [cid], "aggregate")

    nodes_df = pd.DataFrame(nodes).drop_duplicates("node_id")
    edges_df = pd.DataFrame(edges).drop_duplicates("edge_id")
    keep = [c for c in ["claim_id", "snapshot_id", "product_id", "subject_scope", "predicate", "object_id", "aspect",
                        "aspect_term", "polarity", "context", "modality", "source_kind", "derivation", "source_run",
                        "epistemic", "status", "confidence_score", "confidence_type", "e", "m", "s",
                        "verify_label", "verify_strength", "review_id", "user_key", "evidence_id", "start", "end",
                        "quote", "extractor_version", "partition", "cand_id", "schema_version"] if c in claims]
    claims[keep].to_parquet(out / "claims.parquet", index=False)
    evidence.to_parquet(out / "evidence.parquet", index=False)
    aggs.to_parquet(out / "aggregates.parquet", index=False)
    nodes_df.to_parquet(out / "nodes.parquet", index=False)
    edges_df.to_parquet(out / "edges.parquet", index=False)
    node_ids = set(nodes_df.node_id)
    dangling = int((~edges_df.src_id.isin(node_ids) | ~edges_df.dst_id.isin(node_ids)).sum())
    bad_span = int(sum(1 for c in claims.itertuples()
                       if reviews.set_index("review_id").loc[c.review_id].doc[int(c.start):int(c.end)] != c.quote))
    stats = {"snapshot_id": snap, "parent_snapshot": f"{cfg.profile}-{parent}" if parent else None,
             "n_claims": int(len(claims)), "n_evidence": int(len(evidence)), "n_aggregates": int(len(aggs)),
             "n_nodes": int(len(nodes_df)), "n_edges": int(len(edges_df)), "dangling_edges": dangling,
             "invalid_spans": bad_span,
             "n_product_aspect_pairs": int(claims.groupby(["product_id", "aspect"]).ngroups),
             "n_tuples": int(claims.groupby(["product_id", "aspect", "polarity", "context"]).ngroups),
             "n_recovered": int((claims.derivation == "recovered").sum()),
             "n_seller_statements": int(len(seller)),
             "n_seller_context_targets": int((seller.context != "unspecified").sum()) if len(seller) else 0,
             "n_contested": int(aggs[aggs.context == "*"].contested.sum())}
    write_json(out / "snapshot.json", stats)
    return stats


def load_graph(cfg: Config, name: str):
    import networkx as nx
    d = cfg.p("graphs") / name
    nodes = pd.read_parquet(d / "nodes.parquet")
    edges = pd.read_parquet(d / "edges.parquet")
    g = nx.MultiDiGraph()
    for n in nodes.itertuples():
        g.add_node(n.node_id, type=n.type, label=n.label, **json.loads(n.props))
    for e in edges.itertuples():
        g.add_edge(e.src_id, e.dst_id, key=e.relation, relation=e.relation, derivation=e.derivation)
    return g
