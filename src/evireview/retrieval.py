"""Hybrid text retrieval (BM25 + dense, RRF) and polarity-aware graph scoring."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from . import embed
from .common import Config, aspect_names, aspects_mentioned, contexts, detect_context, ontology

SYSTEMS = {
    "B1": {"label": "B1 BM25", "text": ["bm25"], "graph": None},
    "B2": {"label": "B2 BM25+dense (hybrid)", "text": ["bm25", "dense"], "graph": None},
    "B3": {"label": "B3 G0 metadata (product type, seller targets) + hybrid", "text": ["bm25", "dense"], "graph": "metadata"},
    "B4": {"label": "B4 G1 + hybrid", "text": ["bm25", "dense"], "graph": "g1"},
    "B5": {"label": "B5 G2 CEVA + hybrid (proposed)", "text": ["bm25", "dense"], "graph": "ceva"},
    "A-budget": {"label": "A-budget", "text": ["bm25", "dense"], "graph": "budget"},
    "A-verify": {"label": "A-verify", "text": ["bm25", "dense"], "graph": "noverify"},
    "A-context": {"label": "A-context", "text": ["bm25", "dense"], "graph": "nocontext"},
}
NEG_CUES = re.compile(r"\b(not|no|without|few|less|avoid|never|free of|complaints?)\b", re.I)
_TOK = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


@dataclass
class ParsedQuery:
    text: str
    aspects: list[str]
    context: str
    negative_cue: bool
    product_type: str | None = None
    parser: str = "rules"


PARSE_SYSTEM = """Parse a beauty shopping query into structured fields.
aspects: the product aspects the shopper's condition is about (closed set below). Pick the most specific ones; usually one.
- "scent lasts all day" -> scent_longevity. "softer / smoother / glowing skin" -> skin_results. "wrinkles / puffiness / dark circles" -> anti_aging. "detangles" -> detangling. "less frizz" -> frizz_control. "defines curls" -> curl_definition. "dandruff / flaky scalp" -> scalp_condition. "shinier / fuller hair" -> hair_results.
- "container/bottle is small", "not enough product" -> size_quantity (not packaging).
- "not greasy/sticky", "no residue" -> residue. "doesn't irritate / no breakouts" -> irritation.
context: a skin/hair type, climate or time of use only if the query states it, else "unspecified".
product_type: the kind of product requested in 1-3 lowercase words (e.g. "body lotion", "makeup", "shampoo"), or "" if any product.
Aspects:
""" + "\n".join(f"- {a}: {s['description']}" for a, s in ontology()["aspects"].items())


def parse_query_llm(cfg: Config, q: str) -> ParsedQuery:
    """LLM query parsing (cached). The same parse is given to every graph system; text baselines use the raw query."""
    if cfg.offline:
        return parse_query(q)
    from .llm import LLM
    schema = {"type": "object", "additionalProperties": False, "required": ["aspects", "context", "product_type"],
              "properties": {"aspects": {"type": "array", "items": {"type": "string", "enum": aspect_names()}},
                             "context": {"type": "string", "enum": contexts()},
                             "product_type": {"type": "string"}}}
    try:
        res = LLM(cfg).json_call(stage="parse_query", model=cfg["extraction"]["model"], effort="low",
                                 system=PARSE_SYSTEM, user=q, schema=schema, max_tokens=4000)
    except Exception:  # no API key / network / budget: fall back to rule parsing instead of failing the page
        return parse_query(q)
    return ParsedQuery(q, res["aspects"] or parse_query(q).aspects, res["context"], bool(NEG_CUES.search(q)),
                       res["product_type"].strip().lower() or None, "llm")


def parse_query(q: str, cos: float = 0.40) -> ParsedQuery:
    aspects = aspects_mentioned(q)
    if not aspects:
        sims = embed.encode([q]) @ embed.aspect_matrix().T
        j = int(np.argmax(sims[0]))
        if sims[0, j] >= cos:
            aspects = [aspect_names()[j]]
    # "scent lasts" style queries: prefer the more specific aspect
    if "scent_longevity" in aspects:
        aspects = [a for a in aspects if a != "scent"] or ["scent_longevity"]
    return ParsedQuery(q, aspects, detect_context(q), bool(NEG_CUES.search(q)))


@dataclass
class Index:
    reviews: pd.DataFrame
    products: pd.DataFrame
    bm25: BM25Okapi
    dense: np.ndarray
    doc_pids: np.ndarray  # product of each indexed text (reviews, then seller listings when indexed)
    aggs: dict = field(default_factory=dict)
    type_emb: np.ndarray | None = None
    seller_ctx: dict = field(default_factory=dict)


@lru_cache(maxsize=4)
def _build(cfg_path: str, processed: str, graphs: str, index_metadata: bool) -> Index:
    """Text index over reviews; with index_metadata, every system (text baselines included) also indexes each
    product's seller listing, so the KG's seller layer is not an information advantage over the baselines."""
    from .seller import seller_context_targets
    reviews = pd.read_parquet(f"{processed}/reviews.parquet")
    products = pd.read_parquet(f"{processed}/products.parquet")
    docs, pids = reviews.doc.tolist(), reviews.product_id.tolist()
    if index_metadata and "meta_doc" in products:
        listed = products[products.meta_doc.fillna("").str.strip() != ""]
        docs += listed.meta_doc.tolist()
        pids += listed.product_id.tolist()
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    dense = embed.encode(docs)
    aggs = {}
    for v in ("g1", "ceva", "budget", "noverify", "nocontext"):
        try:
            aggs[v] = pd.read_parquet(f"{graphs}/{v}/aggregates.parquet")
        except FileNotFoundError:
            pass
    ptypes = products.product_type.fillna("").tolist() if "product_type" in products else [""] * len(products)
    type_emb = embed.encode([t or title[:60] for t, title in zip(ptypes, products.title)])
    try:
        seller = pd.read_parquet(f"{graphs}/seller_attributes.parquet")
    except FileNotFoundError:
        seller = None
    return Index(reviews, products, bm25, dense, np.array(pids), aggs, type_emb, seller_context_targets(seller))


def type_match(idx: Index, product_type: str | None, threshold: float = 0.55) -> dict[str, bool]:
    """G0 metadata: does the product's (LLM-classified) type match the requested type?"""
    if not product_type:
        return {}
    sims = idx.type_emb @ embed.encode([product_type])[0]
    return {p: bool(s >= threshold) for p, s in zip(idx.products.product_id, sims)}


def get_index(cfg: Config) -> Index:
    agg = cfg.p("graphs") / "ceva" / "aggregates.parquet"
    stamp = str(agg.stat().st_mtime) if agg.exists() else "none"  # rebuild when graphs change
    return _build(str(cfg.path) + "|" + stamp, str(cfg.p("processed")), str(cfg.p("graphs")),
                  bool(cfg["retrieval"].get("index_metadata", False)))


def _product_ranking(idx: Index, scores: np.ndarray, k: int) -> list[str]:
    order = np.argsort(-scores)[:k]
    seen, out = set(), []
    for i in order:
        p = idx.doc_pids[i]
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def graph_scores(idx: Index, variant: str, pq: ParsedQuery, cfg: Config | None = None) -> dict[str, dict]:
    """Opinion score per product. For a context query ("for oily skin") each (product, aspect) uses, in order:
    opinions of reviewers who stated that context; else all opinions, weighted by whether the seller says the
    product targets that context (seller_target_weight) or not (no_context_weight)."""
    aggs = idx.aggs.get(variant)
    if aggs is None or not pq.aspects:
        return {}
    r = (cfg["retrieval"] if cfg is not None else {})
    w_seller, w_none = r.get("seller_target_weight", 1.0), r.get("no_context_weight", 0.5)
    polarity_aware = variant != "nocontext"
    ctx = pq.context if polarity_aware else "unspecified"
    sel = aggs[aggs.aspect.isin(pq.aspects)]
    overall = sel[sel.context == "*"]
    in_ctx = sel[sel.context == ctx].set_index(["product_id", "aspect"]) if ctx != "unspecified" else None
    out = {}
    for pid, g in overall.groupby("product_id"):
        vals, n_pos, n_neg, n_auth, sources = [], 0, 0, 0, set()
        for row in g.itertuples():
            a, weight, src = row, 1.0, "all_reviewers"
            if in_ctx is not None:
                if (pid, row.aspect) in in_ctx.index:
                    a, src = in_ctx.loc[(pid, row.aspect)], "context_reviewers"
                elif ctx in idx.seller_ctx.get(pid, set()):
                    weight, src = w_seller, "seller_target"
                else:
                    weight, src = w_none, "no_context_evidence"
            support = min(a.n_authors / 3, 1.0)
            vals.append(weight * (((a.n_pos + 1) / (a.n_pos + a.n_neg + 2)) * support if polarity_aware else support))
            n_pos, n_neg, n_auth = n_pos + int(a.n_pos), n_neg + int(a.n_neg), n_auth + int(a.n_authors)
            sources.add(src)
        out[pid] = {"g": float(np.mean(vals)) * len(vals) / len(pq.aspects), "n_pos": n_pos, "n_neg": n_neg,
                    "n_authors": n_auth, "insufficient": bool(n_auth < 2), "context_source": sorted(sources)}
    return out


def search(cfg: Config, system: str, query: str, top_k: int = 10, graph_weight: float = 1.0,
           parsed: ParsedQuery | None = None) -> dict:
    """Text lists (BM25, dense) over reviews -> product rankings. Graph systems add a third list that re-ranks
    the text candidates by (product-type match, opinion score) and fuse all lists with RRF."""
    idx = get_index(cfg)
    spec = SYSTEMS[system]
    k = cfg["retrieval"]["candidate_k"]
    rrf_k = cfg["retrieval"]["rrf_k"]
    pq = parsed or (parse_query_llm(cfg, query) if spec["graph"] else parse_query(query))
    lists, weights = {}, {}
    if "bm25" in spec["text"]:
        lists["bm25"] = _product_ranking(idx, idx.bm25.get_scores(tokenize(query)), k)
        weights["bm25"] = 1.0
    if "dense" in spec["text"]:
        lists["dense"] = _product_ranking(idx, idx.dense @ embed.encode([query])[0], k)
        weights["dense"] = 1.0
    gs = {}
    if spec["graph"]:
        cands = list(dict.fromkeys(lists.get("bm25", []) + lists.get("dense", [])))
        tm = type_match(idx, pq.product_type)
        if spec["graph"] != "metadata":
            gs = graph_scores(idx, spec["graph"], pq, cfg)
        targets = pq.context != "unspecified"

        def seller_hit(p):
            return targets and pq.context in idx.seller_ctx.get(p, set())

        def key(p):
            return (tm.get(p, True), gs.get(p, {}).get("g", 0.0))
        ranked_c = sorted(cands, key=key, reverse=True)
        if spec["graph"] == "metadata":  # metadata only: products of the requested type the seller targets at the context
            ranked_c = [p for p in ranked_c if tm.get(p, True) and (not targets or seller_hit(p))] \
                if (tm or targets) else []
        lists["graph"] = ranked_c
        weights["graph"] = graph_weight
    fused: dict[str, float] = {}
    comps: dict[str, dict] = {}
    for name, ranking in lists.items():
        for r, p in enumerate(ranking, 1):
            fused[p] = fused.get(p, 0.0) + weights[name] / (rrf_k + r)
            comps.setdefault(p, {})[f"rank_{name}"] = r
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    titles = idx.products.set_index("product_id").title
    return {"query": query, "system": system, "parsed": pq.__dict__,
            "results": [{"product_id": p, "title": titles.get(p, ""), "score": round(s, 5),
                         "components": comps[p], "graph": gs.get(p)} for p, s in ranked]}


RERANK_SYSTEM = """You re-rank beauty products for a shopper's request using the review sentences shown for each product.
Rank products that reviewers clearly say satisfy the request first; for "few complaints / not X" requests, prefer products whose reviewers report no such problem. Products of a different kind than requested go last.
Return every product id exactly once, best first."""


def llm_rerank(cfg: Config, llm, query: str, top_n: int = 20, sentences: int = 6) -> dict:
    """B6: LLM reads review sentences of the hybrid (B2) top-N products and re-ranks them. No KG."""
    import time
    from .common import split_sentences
    t0 = time.time()
    idx = get_index(cfg)
    base = search(cfg, "B2", query, top_k=top_n)["results"]
    qv = embed.encode([query])[0]
    blocks = []
    for i, r in enumerate(base):
        rows = [sent for rv in idx.reviews[idx.reviews.product_id == r["product_id"]].itertuples()
                for _, _, sent in split_sentences(rv.doc)]
        sims = embed.encode(rows) @ qv if rows else np.zeros(0)
        top = [rows[j] for j in np.argsort(-sims)[:sentences]]
        blocks.append(f"<product id=\"P{i}\" title=\"{r['title'][:80]}\">\n" + "\n".join(f"- {x}" for x in top) + "\n</product>")
    schema = {"type": "object", "additionalProperties": False, "required": ["ranking"],
              "properties": {"ranking": {"type": "array", "items": {"type": "string"}}}}
    spent0 = llm.spent
    res = llm.json_call(stage="rerank_b6", model=cfg["answer"]["model"], effort="low", system=RERANK_SYSTEM,
                        user=f"Request: {query}\n\n" + "\n\n".join(blocks), schema=schema)
    order = [int(x.lstrip("P")) for x in res["ranking"] if x.lstrip("P").isdigit() and int(x.lstrip("P")) < len(base)]
    order = list(dict.fromkeys(order)) + [i for i in range(len(base)) if i not in order]
    return {"query": query, "system": "B6", "results": [base[i] for i in order][:10],
            "latency_ms": int((time.time() - t0) * 1000), "cost_usd": llm.spent - spent0}


def evidence_for(cfg: Config, variant: str | None, product_ids: list[str], aspects: list[str], query: str,
                 max_evidence: int = 8) -> list[dict]:
    """Evidence passed to the answer model. Text systems: best-matching review sentences.
    Graph systems: verified claims, balanced across polarities (disagreement preserved)."""
    from .common import split_sentences
    idx = get_index(cfg)
    per = max(1, max_evidence // max(len(product_ids), 1))
    out = []
    if variant is None:
        qv = embed.encode([query])[0]
        for pid in product_ids:
            rows = []
            for r in idx.reviews[idx.reviews.product_id == pid].itertuples():
                for s, e, sent in split_sentences(r.doc):
                    rows.append((r.review_id, s, e, sent))
            if not rows:
                continue
            sims = embed.encode([x[3] for x in rows]) @ qv
            for j in np.argsort(-sims)[:per]:
                rid, s, e, sent = rows[j]
                out.append({"product_id": pid, "review_id": rid, "quote": sent, "start": s, "end": e,
                            "polarity": None, "aspect": None})
        return out
    claims = pd.read_parquet(cfg.p("graphs") / variant / "claims.parquet")
    for pid in product_ids:
        c = claims[(claims.product_id == pid) & (claims.aspect.isin(aspects) if aspects else True)]
        c = c.sort_values("s", ascending=False)
        if variant == "nocontext":  # ablation: no disagreement handling, just top-scored claims
            picked = c.head(per).to_dict("records")
        else:  # round-robin across polarities so opposing opinions both reach the answer
            pools = [c[c.polarity == pol].to_dict("records") for pol in ("positive", "negative", "mixed", "neutral")]
            picked = []
            while len(picked) < per and any(pools):
                for pool in pools:
                    if pool and len(picked) < per:
                        picked.append(pool.pop(0))
        for x in picked:
            out.append({"product_id": pid, "review_id": x["review_id"], "quote": x["quote"], "start": x["start"],
                        "end": x["end"], "polarity": None if variant == "nocontext" else x["polarity"],
                        "aspect": x["aspect"], "evidence_id": x["evidence_id"], "claim_id": x["claim_id"]})
    return out
