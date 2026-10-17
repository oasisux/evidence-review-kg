"""Experiment 2: retrieval with new products mixed in.

The products in the new-product role (predict.role_products) keep only their first k reviews. For each k a view of
the data is built (reviews, CEVA claims, opinion aggregates and relation-based predictions recomputed from what
remains); systems search the whole catalog in that view. Relevance comes from the judge, who reads review sentences
of ALL reviews (the product's eventual reputation), on a pool of the top 20 of every system in every view.

Systems: B2 hybrid text, B6 LLM re-ranking of B2, B5 current KG (own opinions only), B7 relation-predicted KG."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Config, read_json, write_json
from .evaluate import mrr_at, ndcg_at, paired_bootstrap

SYSTEMS_X2 = ["B2", "B5", "B7", "B7L", "B8", "B8-ucb", "B8-slot"]
POOL_DEPTH = 20


def make_view(cfg: Config, k: int, roles: set[str], params: dict) -> Config:
    """data/<profile>/cold/k<k>/{processed,graphs}: role products keep their first k reviews."""
    from .pipeline import aggregate
    from .predict import predict_table, review_votes
    from .relations import load_relations
    root = cfg.p("processed").parent / "cold" / f"k{k}"
    raw = copy.deepcopy(cfg.raw)
    rel = Path(cfg.raw["paths"]["processed"]).parent / "cold" / f"k{k}"  # relative to the project root
    raw["paths"] = {**raw["paths"], "processed": str(rel / "processed"), "graphs": str(rel / "graphs")}
    view = Config(raw, cfg.path)
    src_p, src_g = cfg.p("processed"), cfg.p("graphs")
    dst_p, dst_g = view.p("processed"), view.p("graphs")

    reviews = pd.read_parquet(src_p / "reviews.parquet")
    order = reviews.sort_values(["product_id", "time_utc", "review_id"])
    rank = order.groupby("product_id").cumcount().reindex(reviews.index)
    keep = ~reviews.product_id.isin(roles) | (rank < k)
    rv = reviews[keep].reset_index(drop=True)
    rv.to_parquet(dst_p / "reviews.parquet", index=False)
    for f in ("products.parquet", "split_manifest.json", "manifest.json"):
        shutil.copy(src_p / f, dst_p / f)
    shutil.copy(src_g / "seller_attributes.parquet", dst_g / "seller_attributes.parquet")
    shutil.copytree(src_g / "relations", dst_g / "relations", dirs_exist_ok=True)

    (dst_g / "ceva").mkdir(parents=True, exist_ok=True)
    claims = pd.read_parquet(src_g / "ceva" / "claims.parquet")
    claims = claims[claims.review_id.isin(set(rv.review_id))]
    claims.to_parquet(dst_g / "ceva" / "claims.parquet", index=False)
    aggs = aggregate(claims, rv).assign(snapshot_id=f"{cfg.profile}-cold-k{k}")
    aggs.to_parquet(dst_g / "ceva" / "aggregates.parquet", index=False)
    products = pd.read_parquet(dst_p / "products.parquet")
    pred = predict_table(view, products, load_relations(view), review_votes(claims, rv), aggs, params)
    pred.to_parquet(dst_g / "ceva" / "predictions.parquet", index=False)
    write_json(root / "view.json", {"k": k, "n_reviews": int(len(rv)), "n_claims": int(len(claims)),
                                    "n_role_products": len(roles), "n_predictions": int(len(pred))})
    return view


def _judge_pool(cfg: Config, llm, queries: list[dict], pool: dict[str, set]) -> dict:
    """Pooled qrels: grade only (query, product) pairs some system ranked in its top 20 in some view."""
    from .judge import QREL_SCHEMA, QREL_SYSTEM, _judge, product_aspect_sentences
    from .pipeline import load_data
    path = cfg.p("annotations") / "qrels_pool.json"
    qrels = read_json(path, {})
    _, products, _ = load_data(cfg)
    titles = products.set_index("product_id").title
    jobs = []
    for q in queries:
        done = qrels.setdefault(q["qid"], {})
        todo = []
        for p in sorted(pool[q["qid"]] - set(done)):
            if product_aspect_sentences(cfg, p, q["target_aspects"], 1):
                todo.append(p)
            else:
                done[p] = 0  # no review sentence on the aspect: grade 0 without a judge call
        jobs += [(q, todo[i:i + 6]) for i in range(0, len(todo), 6)]

    def run(job):
        q, pids = job
        blocks = []
        for i, p in enumerate(pids):
            sents = product_aspect_sentences(cfg, p, q["target_aspects"], 8, q.get("target_context", "unspecified"))
            blocks.append(f"<product id=\"P{i}\" title=\"{titles[p][:80]}\">\n" + "\n".join(f"- {s}" for s in sents)
                          + "\n</product>")
        res = _judge(cfg, llm, "qrels", QREL_SYSTEM, f"Request: {q['text']}\n\n" + "\n\n".join(blocks), QREL_SCHEMA)
        return q["qid"], {pids[int(g["product"].lstrip("P"))]: g["grade"] for g in res["grades"]
                          if g["product"].lstrip("P").isdigit() and int(g["product"].lstrip("P")) < len(pids)}

    for r in llm.map(run, jobs, desc="judge_qrels_pool"):
        if r:
            qrels[r[0]].update(r[1])
    write_json(path, qrels)
    return qrels


def run_coldsearch(cfg: Config) -> dict:
    from .judge import RETRIEVAL_TYPES, check_sample, run_queries
    from .llm import LLM
    from .pipeline import load_data
    from .predict import role_products
    from .retrieval import llm_rerank, search
    if cfg.offline:
        res = {"status": "NOT_EVALUATED", "reason": "offline profile (needs judge queries and qrels)"}
        write_json(cfg.p("reports") / "coldsearch.json", res)
        return res
    llm = LLM(cfg)
    check_sample(cfg)
    queries = [q for q in run_queries(cfg, llm) if q["split"] == "test" and q["type"] in RETRIEVAL_TYPES]
    _, products, split = load_data(cfg)
    roles = set(role_products(cfg, products, split))
    params = read_json(cfg.p("graphs") / "predict_params.json")["proposed"]
    ks = cfg["coldstart"]["ks"]
    views = {f"k{k}": make_view(cfg, k, roles, params) for k in ks}
    views["full"] = cfg

    runs_path = cfg.p("runs") / "coldsearch_rankings.json"
    rankings = read_json(runs_path, {})
    for vname, vcfg in views.items():
        for sys_name in SYSTEMS_X2:
            key = f"{vname}|{sys_name}"
            if key in rankings:
                continue
            rankings[key] = {q["qid"]: [r["product_id"] for r in search(vcfg, sys_name, q["text"], top_k=POOL_DEPTH,
                                                                        graph_weight=1.0)["results"]] for q in queries}
        key = f"{vname}|B6"
        if key not in rankings:
            from .retrieval import get_index
            get_index(vcfg)  # build the view's index once, before the worker threads share it
            out = llm.map(lambda q, v=vcfg: (q["qid"], [r["product_id"] for r in llm_rerank(v, llm, q["text"])["results"]]),
                          queries, desc=f"rerank_b6_{vname}")
            rankings[key] = dict(x for x in out if x)
        write_json(runs_path, rankings)

    pool = {q["qid"]: set() for q in queries}
    for ranks in rankings.values():
        for qid, ps in ranks.items():
            if qid in pool:
                pool[qid] |= set(ps[:POOL_DEPTH])
    qrels = _judge_pool(cfg, llm, queries, pool)

    rows = []
    for key, ranks in rankings.items():
        vname, sys_name = key.split("|")
        for q in queries:
            g = qrels.get(q["qid"], {})
            top = ranks.get(q["qid"], [])[:10]
            ranked = [g.get(p, 0) for p in top]
            ideal = list(g.values())
            rel_roles = [p for p, x in g.items() if x >= 1 and p in roles]
            rows.append({"view": vname, "system": sys_name, "qid": q["qid"], "type": q["type"],
                         "context": q.get("target_context", "unspecified"),
                         "n_relevant": sum(1 for x in ideal if x >= 1), "n_relevant_new": len(rel_roles),
                         "ndcg@10": ndcg_at(ranked, ideal), "mrr@10": mrr_at(ranked),
                         "new_in_top10": sum(1 for p in top if p in roles),
                         "relevant_new_hit": (sum(1 for p in top if p in rel_roles) / len(rel_roles)) if rel_roles else np.nan,
                         "top10": json.dumps(top)})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.p("reports") / "coldsearch_per_query.csv", index=False, encoding="utf-8-sig")
    valid = df[df.n_relevant > 0]
    summary = []
    for (vname, sys_name), g in valid.groupby(["view", "system"]):
        summary.append({"view": vname, "system": sys_name, "n_queries": int(len(g)), "ndcg@10": float(g["ndcg@10"].mean()),
                        "mrr@10": float(g["mrr@10"].mean()), "new_in_top10": float(g.new_in_top10.mean()),
                        "relevant_new_hit": float(g.relevant_new_hit.mean()),
                        "n_queries_with_relevant_new": int(g.relevant_new_hit.notna().sum())})
    comparisons = []
    for vname in views:
        for a, b in (("B7", "B5"), ("B7", "B2"), ("B7", "B6"), ("B5", "B2"), ("B6", "B2"),
                     ("B7L", "B7"), ("B8", "B7"), ("B8", "B2"), ("B8", "B6"), ("B8-ucb", "B8"), ("B8-slot", "B8"),
                     ("B8-slot", "B7"), ("B8-slot", "B6")):
            for metric in ("ndcg@10", "relevant_new_hit"):
                x = valid[(valid.view == vname) & (valid.system == a)].set_index("qid")[metric]
                y = valid[(valid.view == vname) & (valid.system == b)].set_index("qid")[metric]
                both = x.dropna().index.intersection(y.dropna().index)
                if len(both) < 3:
                    continue
                d, lo, hi = paired_bootstrap(x[both].to_numpy(), y[both].to_numpy())
                comparisons.append({"view": vname, "variant": a, "baseline": b, "metric": metric, "n": int(len(both)),
                                    "variant_value": float(x[both].mean()), "baseline_value": float(y[both].mean()),
                                    "delta": d, "ci_low": lo, "ci_high": hi})
    res = {"n_test_queries": len(queries), "n_role_products": len(roles), "pool_depth": POOL_DEPTH,
           "n_judged_pairs": int(sum(len(v) for v in qrels.values())),
           "views": {v: read_json(cfg.p("processed").parent / "cold" / v / "view.json", {}) for v in views if v != "full"},
           "summary": summary, "comparisons": comparisons,
           "note": "Pooled qrels: products outside every system's top 20 in every view count as not relevant."}
    write_json(cfg.p("reports") / "coldsearch.json", res)
    return res
