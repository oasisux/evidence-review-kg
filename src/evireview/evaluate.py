"""Metrics for T1 (construction), T2 (augmentation), T3 (retrieval/QA), judge-human agreement; CSV export."""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .common import SCHEMA_VERSION, Config, read_json, sha, write_json
from .llm import ledger_summary
from .pipeline import VARIANTS, load_data
from .retrieval import SYSTEMS, search

NE = "NOT_EVALUATED"
RNG = np.random.default_rng(42)


# ------------------------------------------------------------------ stats helpers
def cluster_bootstrap(values: pd.Series, clusters: pd.Series, n: int = 2000) -> tuple[float | None, float | None]:
    if len(values) == 0:
        return None, None
    groups = [g.to_numpy(dtype=float) for _, g in values.groupby(clusters.values)]
    stats = []
    for _ in range(n):
        pick = RNG.integers(0, len(groups), len(groups))
        allv = np.concatenate([groups[i] for i in pick])
        stats.append(allv.mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000) -> tuple[float, float, float]:
    d = a - b
    idx = RNG.integers(0, len(d), (n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ndcg_at(grades: list[int], ideal: list[int], k: int = 10) -> float:
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(sorted(ideal, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else float("nan")


def mrr_at(grades: list[int], k: int = 10) -> float:
    for i, g in enumerate(grades[:k]):
        if g >= 1:
            return 1 / (i + 1)
    return 0.0


def _row(metric, value, unit="ratio", n=None, lo=None, hi=None, status="COMPLETED", **kw):
    return {"metric": metric, "value": None if value is None or (isinstance(value, float) and math.isnan(value)) else value,
            "unit": unit, "n_samples": n, "ci_low": lo, "ci_high": hi, "evaluation_status": status, **kw}


# ------------------------------------------------------------------ T1
def eval_construction(cfg: Config) -> list[dict]:
    g = cfg.p("graphs")
    ann = cfg.p("annotations")
    rows = []
    for v in VARIANTS:
        snap = read_json(g / v / "snapshot.json", {})
        if not snap:
            continue
        rows += [_row("n_claims", snap["n_claims"], "count", method=v),
                 _row("invalid_evidence_spans", snap["invalid_spans"], "count", method=v),
                 _row("dangling_edges", snap["dangling_edges"], "count", method=v),
                 _row("valid_provenance_rate", 1 - snap["invalid_spans"] / max(snap["n_claims"], 1), method=v),
                 _row("product_aspect_pairs", snap["n_product_aspect_pairs"], "count", method=v),
                 _row("contested_groups", snap["n_contested"], "count", method=v)]
    labels = read_json(ann / "judge_claims.json", {})
    samples = read_json(ann / "judge_claim_samples.json", {})
    g1 = pd.read_parquet(g / "candidates_g1.parquet") if (g / "candidates_g1.parquet").exists() else None
    if g1 is not None and samples.get("g1") and labels:
        tau = read_json(g / "augment_summary.json", {}).get("tau", {}).get("tau", cfg["augmentation"]["threshold"])
        s = g1[g1.cand_id.isin(samples["g1"]) & g1.cand_id.isin(labels)].copy()
        s["y"] = s.cand_id.map(lambda c: labels[c] == "supported").astype(float)
        for name, sub in (("g1", s), ("ceva", s[(s.verify_label == "entail") & (s.s >= tau)])):
            lo, hi = cluster_bootstrap(sub.y, sub.product_id)
            rows.append(_row("claim_precision_judge", float(sub.y.mean()) if len(sub) else None, n=len(sub),
                             lo=lo, hi=hi, method=name, split="test", label_source="judge"))
            for pol, sp in sub.groupby("polarity"):
                rows.append(_row(f"claim_precision_judge[{pol}]", float(sp.y.mean()), n=len(sp), method=name,
                                 split="test", label_source="judge"))
    else:
        rows.append(_row("claim_precision_judge", None, status=NE, method="g1"))
    ref = read_json(ann / "t1_reference.json")
    if ref:
        gold = {(c["review_id"], c["aspect"], c["polarity"]) for c in ref["claims"]
                if c["modality"] == "asserted"}
        test_reviews = set(ref["reviews"]["test"])
        gold = {t for t in gold if t[0] in test_reviews}
        for v in ("g1", "ceva"):
            c = pd.read_parquet(g / v / "claims.parquet")
            pred = {(r.review_id, r.aspect, r.polarity) for r in c.itertuples() if r.review_id in test_reviews}
            tp = len(pred & gold)
            p = tp / len(pred) if pred else None
            r = tp / len(gold) if gold else None
            f1 = 2 * p * r / (p + r) if p and r else None
            rows += [_row("t1_precision_vs_reference", p, n=len(pred), method=v, split="test", label_source="judge"),
                     _row("t1_recall_vs_reference", r, n=len(gold), method=v, split="test", label_source="judge"),
                     _row("t1_f1_vs_reference", f1, n=len(gold), method=v, split="test", label_source="judge")]
    else:
        rows.append(_row("t1_f1_vs_reference", None, status=NE))
    return rows


# ------------------------------------------------------------------ T2
def eval_augmentation(cfg: Config) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    g = cfg.p("graphs")
    labels = read_json(cfg.p("annotations") / "judge_claims.json", {})
    summary = read_json(g / "augment_summary.json", {})
    tau = summary.get("tau", {}).get("tau", cfg["augmentation"]["threshold"])
    rows, audit, curve = [], [], []
    if not summary:
        return [_row("augmentation", None, status=NE)], pd.DataFrame(), []
    base = pd.read_parquet(g / "ceva" / "claims.parquet")
    base = base[base.derivation == "direct"]
    base_pairs = set(zip(base.product_id, base.aspect))
    base_tuples = set(zip(base.product_id, base.aspect, base.polarity, base.context))
    for run, variant_name in (("c1", "ceva"), ("budget", "budget"), ("c1", "noverify"), ("c1legacy", "c1_v1")):
        path = g / f"candidates_{run}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        col = "status_hard" if variant_name == "noverify" else "status_verified"
        acc = df[df[col] == "accepted"]
        test_acc = acc[acc.partition == "test"]
        lab = test_acc[test_acc.cand_id.isin(labels)].copy()
        lab["y"] = lab.cand_id.map(lambda c: labels[c] == "supported").astype(float)
        new_pairs = set(zip(acc.product_id, acc.aspect)) - base_pairs
        new_tuples = set(zip(acc.product_id, acc.aspect, acc.polarity, acc.context)) - base_tuples
        lab["new_pair"] = [(p_, a_) not in base_pairs for p_, a_ in zip(lab.product_id, lab.aspect)]
        lab_np = lab[lab.new_pair]
        prec = float(lab.y.mean()) if len(lab) else None
        lo, hi = cluster_bootstrap(lab.y, lab.product_id) if len(lab) else (None, None)
        est_supported = prec * len(acc) if prec is not None else None
        status = "COMPLETED" if len(lab) else NE
        rows += [
            _row("accepted_recovered_claims", int(len(acc)), "count", method=variant_name),
            _row("new_product_aspect_pairs", len(new_pairs), "count", method=variant_name),
            _row("new_canonical_tuples", len(new_tuples), "count", method=variant_name),
            _row("support_enrichment_claims", int(len(acc)) - len(new_tuples), "count", method=variant_name),
            _row("augmentation_precision_judge", prec, n=len(lab), lo=lo, hi=hi, method=variant_name, split="test",
                 status=status, label_source="judge"),
            _row("estimated_supported_new_claims", est_supported, "count", method=variant_name, status=status),
            _row("augmentation_precision_new_pairs", float(lab_np.y.mean()) if len(lab_np) else None, n=len(lab_np),
                 lo=cluster_bootstrap(lab_np.y, lab_np.product_id)[0] if len(lab_np) else None,
                 hi=cluster_bootstrap(lab_np.y, lab_np.product_id)[1] if len(lab_np) else None,
                 method=variant_name, split="test", status="COMPLETED" if len(lab_np) else NE, label_source="judge"),
            _row("duplicates_removed", int(df.get("duplicate_of_existing", pd.Series(dtype=bool)).sum()), "count",
                 method=variant_name),
        ]
        if variant_name in ("ceva", "budget", "c1_v1"):
            for r in df[df.partition == "test"].itertuples():
                audit.append({"claim_id": r.cand_id, "method": variant_name, "derivation": "recovered",
                              "claim_status": getattr(r, col), "evidence_quote": r.quote, "aspect": r.aspect,
                              "polarity": r.polarity, "product_id": r.product_id, "s": round(r.s, 3),
                              "verify_label": r.verify_label, "hard_reason": r.hard_reason,
                              "human_label": None, "judge_label": labels.get(r.cand_id),
                              "is_new_tuple": (r.product_id, r.aspect, r.polarity, r.context) in new_tuples})
            # precision-coverage curve over tau (judge-labelled, entailed candidates)
            ent = df[(df.partition == "test") & df.hard_pass & (df.verify_label == "entail") & df.cand_id.isin(labels)].copy()
            ent["y"] = ent.cand_id.map(lambda c: labels[c] == "supported").astype(float)
            for t in np.round(np.arange(0.6, 1.001, 0.05), 2):
                sel = ent[ent.s >= t]
                curve.append({"method": variant_name, "tau": float(t), "accepted": int(len(sel)),
                              "precision": float(sel.y.mean()) if len(sel) else None,
                              "supported": int(sel.y.sum())})
    rows.append(_row("tau", tau, "score", status="COMPLETED", method="ceva",
                     note=json.dumps(summary.get("tau"))))
    rows.append(_row("llm_review_slots_c1", summary.get("c1_review_slots"), "count", method="ceva"))
    rows.append(_row("llm_review_slots_budget", summary.get("budget_review_slots"), "count", method="budget"))
    return rows, pd.DataFrame(audit), curve


# ------------------------------------------------------------------ T3 retrieval
def active_queries(cfg: Config) -> list[dict]:
    qs = read_json(cfg.p("annotations") / "queries.json", [])
    check = review_source(cfg)[0]
    dropped = {k for k, v in check.get("queries", {}).items() if v.get("decision") == "drop"}
    return [q for q in qs if q["qid"] not in dropped]


def eval_retrieval(cfg: Config) -> tuple[list[dict], list[dict], dict]:
    qrels = read_json(cfg.p("annotations") / "qrels_judge.json", {})
    queries = [q for q in active_queries(cfg) if q["type"] in ("pos_condition", "neg_condition")]
    if not qrels or not queries:
        return [_row("ndcg@10", None, status=NE)], [], {}
    _, products, _ = load_data(cfg)
    all_p = products.product_id.tolist()
    per_query, latencies = [], []

    def score_system(sys_name, w, qs):
        out = []
        for q in qs:
            t0 = time.time()
            res = search(cfg, sys_name, q["text"], top_k=10, graph_weight=w)
            latencies.append(time.time() - t0)
            grades_map = qrels.get(q["qid"], {})
            ranked = [grades_map.get(r["product_id"], 0) for r in res["results"]]
            ideal = [grades_map.get(p, 0) for p in all_p]
            rel = sum(1 for x in ideal if x >= 1)
            out.append({"qid": q["qid"], "type": q["type"], "split": q["split"], "system": sys_name,
                        "graph_weight": w, "ndcg@10": ndcg_at(ranked, ideal), "mrr@10": mrr_at(ranked),
                        "recall@10": (sum(1 for x in ranked if x >= 1) / rel) if rel else float("nan"),
                        "n_relevant": rel, "top10": [r["product_id"] for r in res["results"]],
                        "parsed_aspects": res["parsed"]["aspects"]})
        return out

    dev = [q for q in queries if q["split"] == "dev"]
    test = [q for q in queries if q["split"] == "test"]
    # graph weight fixed a priori (1.0 = same weight as each text list); tuning on few dev queries proved fragile
    weights = {s_: (0.0 if spec["graph"] is None else 1.0) for s_, spec in SYSTEMS.items()}
    for sys_name in SYSTEMS:
        per_query += score_system(sys_name, weights[sys_name], test)
    b6 = read_json(cfg.p("runs") / "b6_rankings.json", {})
    b6_lat, b6_cost = [], []
    for q in test:
        if q["qid"] not in b6:
            continue
        grades_map = qrels.get(q["qid"], {})
        ranked = [grades_map.get(p, 0) for p in b6[q["qid"]]["top10"]]
        ideal = [grades_map.get(p, 0) for p in all_p]
        rel = sum(1 for x in ideal if x >= 1)
        b6_lat.append(b6[q["qid"]]["latency_ms"])
        b6_cost.append(b6[q["qid"]]["cost_usd"])
        per_query.append({"qid": q["qid"], "type": q["type"], "split": q["split"], "system": "B6",
                          "graph_weight": None, "ndcg@10": ndcg_at(ranked, ideal), "mrr@10": mrr_at(ranked),
                          "recall@10": (sum(1 for x in ranked if x >= 1) / rel) if rel else float("nan"),
                          "n_relevant": rel, "top10": b6[q["qid"]]["top10"], "parsed_aspects": None})
    if b6:
        weights["B6"] = None
    df = pd.DataFrame(per_query)
    valid = df[df.n_relevant > 0]
    rows = []
    for sys_name in list(SYSTEMS) + (["B6"] if b6 else []):
        for qtype in ("all", "pos_condition", "neg_condition"):
            sub = valid[(valid.system == sys_name) & ((valid.type == qtype) if qtype != "all" else True)]
            for m in ("ndcg@10", "mrr@10", "recall@10"):
                rows.append(_row(m, float(sub[m].mean()) if len(sub) else None, n=len(sub), method=sys_name,
                                 split="test", query_type=qtype, label_source="judge",
                                 graph_weight=weights[sys_name]))
    comparisons = []
    for a_sys, b_sys in (("B5", "B2"), ("B4", "B2"), ("B3", "B2"), ("B5", "B3"), ("B5", "B4"), ("B5", "A-budget"),
                         ("B5", "A-verify"), ("B5", "A-context"), ("B5", "B6"), ("B6", "B2")):
        for qtype in ("all", "pos_condition", "neg_condition"):
            a = valid[(valid.system == a_sys) & ((valid.type == qtype) if qtype != "all" else True)].sort_values("qid")
            b = valid[(valid.system == b_sys) & ((valid.type == qtype) if qtype != "all" else True)].sort_values("qid")
            if len(a) and len(a) == len(b):
                d, lo, hi = paired_bootstrap(a["ndcg@10"].to_numpy(), b["ndcg@10"].to_numpy())
                comparisons.append({"baseline": b_sys, "variant": a_sys, "metric": "ndcg@10", "query_type": qtype,
                                    "baseline_value": float(b["ndcg@10"].mean()), "variant_value": float(a["ndcg@10"].mean()),
                                    "delta": d, "ci_low": lo, "ci_high": hi, "n_queries": len(a)})
    lat = np.array(latencies) * 1000
    extra = {"n_test_queries": len(test), "n_dev_queries": len(dev),
             "b6_latency_ms_p50": float(np.percentile(b6_lat, 50)) if b6_lat else None,
             "b6_latency_ms_p95": float(np.percentile(b6_lat, 95)) if b6_lat else None,
             "b6_cost_usd_per_query": float(np.mean(b6_cost)) if b6_cost else None,
             "n_test_queries_without_relevant": int(df[(df.system == "B2") & (df.n_relevant == 0)].shape[0]),
             "graph_weights": weights, "query_latency_ms_p50": float(np.percentile(lat, 50)) if len(lat) else None,
             "query_latency_ms_p95": float(np.percentile(lat, 95)) if len(lat) else None}
    return rows, comparisons, {"per_query": per_query, **extra}


# ------------------------------------------------------------------ T3 QA
def eval_qa(cfg: Config) -> tuple[list[dict], list[dict]]:
    judged = read_json(cfg.p("annotations") / "answer_judgments.json", [])
    answers = read_json(cfg.p("runs") / "answers.json", [])
    active = {q["qid"]: q for q in active_queries(cfg) if q["split"] == "test"}
    if not judged:
        return [_row("answer_correctness", None, status=NE)], []
    amap = {(a["qid"], a["system"]): a for a in answers}
    per = []
    for j in judged:
        if j["qid"] not in active:
            continue
        a = amap.get((j["qid"], j["system"]), {})
        q = active[j["qid"]]
        per.append({"qid": j["qid"], "system": j["system"], "type": q["type"], "correctness": j["correctness"],
                    "citation_precision": (j["supported_statements"] / j["statements"]) if j["statements"] else None,
                    "both_sides": j["both_sides"], "should_abstain": j["should_abstain"],
                    "abstained": a.get("abstained"), "answer": a.get("answer"),
                    "reference_available": not j["should_abstain"]})
    df = pd.DataFrame(per)
    rows = []
    for s, g in df.groupby("system"):
        rows += [_row("answer_correctness_0_2", float(g.correctness.mean()), "score_0_2", n=len(g), method=s,
                      split="test", label_source="judge"),
                 _row("citation_precision", float(g.citation_precision.dropna().mean()) if g.citation_precision.notna().any() else None,
                      n=int(g.citation_precision.notna().sum()), method=s, split="test", label_source="judge")]
        bs = g[g.both_sides != "na"]
        rows.append(_row("both_sides_rate", float((bs.both_sides == "true").mean()) if len(bs) else None, n=len(bs),
                         method=s, split="test", label_source="judge"))
        un = g[g.should_abstain]
        ans = g[~g.should_abstain]
        rows.append(_row("abstention_recall", float(un.abstained.mean()) if len(un) else None, n=len(un), method=s,
                         split="test", label_source="judge"))
        rows.append(_row("false_abstention_rate", float(ans.abstained.mean()) if len(ans) else None, n=len(ans),
                         method=s, split="test", label_source="judge"))
    return rows, per


# ------------------------------------------------------------------ KG opinion-distribution accuracy (#6)
def eval_kg_accuracy(cfg: Config) -> list[dict]:
    data = read_json(cfg.p("annotations") / "kg_accuracy_judge.json", {})
    if not data:
        return [_row("kg_positive_share_mae", None, status=NE)]
    rows = []
    for v in ("g1", "ceva", "budget", "noverify"):
        p = cfg.p("graphs") / v / "aggregates.parquet"
        if not p.exists():
            continue
        ag = pd.read_parquet(p)
        ag = ag[ag.context == "*"].set_index(["product_id", "aspect"])
        for stratum in ("all", "in_kg", "kg_sparse"):
            maes, maj, recall_num, recall_den, covered = [], [], 0, 0, 0
            items = [(k, d) for k, d in data.items() if stratum == "all" or d["stratum"] == stratum]
            for key, d in items:
                pid, asp = key.split("|")
                labs = list(d["labels"].values())
                jp, jn = labs.count("positive"), labs.count("negative")
                kp, kn = (int(ag.loc[(pid, asp)].n_pos), int(ag.loc[(pid, asp)].n_neg)) if (pid, asp) in ag.index else (0, 0)
                recall_num += min(kp, jp) + min(kn, jn)
                recall_den += jp + jn
                if kp + kn > 0:
                    covered += 1
                if jp + jn > 0 and kp + kn > 0:
                    maes.append(abs(jp / (jp + jn) - kp / (kp + kn)))
                    if jp != jn:
                        maj.append((jp > jn) == (kp > kn) and kp != kn)
            rows += [
                _row("kg_positive_share_mae", float(np.mean(maes)) if maes else None, n=len(maes), method=v,
                     split="test", scope=stratum, label_source="judge"),
                _row("kg_majority_polarity_agreement", float(np.mean(maj)) if maj else None, n=len(maj), method=v,
                     split="test", scope=stratum, label_source="judge"),
                _row("kg_opinion_recall", recall_num / recall_den if recall_den else None, n=recall_den, method=v,
                     split="test", scope=stratum, label_source="judge"),
                _row("kg_pair_coverage", covered / len(items) if items else None, n=len(items), method=v,
                     split="test", scope=stratum, label_source="judge")]
    return rows


# ------------------------------------------------------------------ judge vs human agreement
def review_source(cfg: Config) -> tuple[dict, str, str]:
    """Human check if present, else the LLM cross-audit. Returns (data, value_key, label_source)."""
    human = read_json(cfg.p("annotations") / "human_check.json", {})
    if human:
        return human, "human", "human"
    audit = read_json(cfg.p("annotations") / "llm_audit.json", {})
    if audit:
        return audit, "auditor", f"llm_auditor:{audit.get('model')}"
    return {}, "human", "human"


def eval_agreement(cfg: Config) -> list[dict]:
    data, key, source = review_source(cfg)
    if not data:
        return [_row("judge_check_claim_accuracy", None, status=NE), _row("judge_check_qrels_binary", None, status=NE)]
    from sklearn.metrics import cohen_kappa_score
    rows = []
    claims = data.get("claims", {})
    if claims:
        h = [v[key] for v in claims.values()]
        j = [v["judge"] for v in claims.values()]
        acc = float(np.mean([a == b for a, b in zip(h, j)]))
        kappa = float(cohen_kappa_score(h, j)) if len(set(h + j)) > 1 else None
        rows += [_row("judge_check_claim_accuracy", acc, n=len(h), label_source=source),
                 _row("judge_check_claim_kappa", kappa, "kappa", n=len(h), label_source=source)]
    qr = data.get("qrels", {})
    if qr:
        h = [v[key] for v in qr.values()]
        j = [v["judge"] for v in qr.values()]
        rows += [_row("judge_check_qrels_exact", float(np.mean([a == b for a, b in zip(h, j)])), n=len(h), label_source=source),
                 _row("judge_check_qrels_binary", float(np.mean([(a >= 1) == (b >= 1) for a, b in zip(h, j)])), n=len(h),
                      label_source=source),
                 _row("judge_check_qrels_weighted_kappa", float(cohen_kappa_score(h, j, weights="linear")), "kappa",
                      n=len(h), label_source=source)]
    an = data.get("answers", {})
    if an:
        rows.append(_row("check_citation_ok_rate", float(np.mean([v[key] == "ok" for v in an.values()])), n=len(an),
                         label_source=source))
    q = data.get("queries", {})
    if q:
        rows.append(_row("check_query_approval_rate", float(np.mean([v["decision"] == "approve" for v in q.values()])),
                         n=len(q), label_source=source))
    return rows


# ------------------------------------------------------------------ export
COMMON = ["schema_version", "run_id", "snapshot_id", "category", "split", "method", "scope", "filter_json",
          "label_source", "evaluation_status"]


def to_csv(df: pd.DataFrame, path, run_id: str, cfg: Config) -> None:
    df = df.copy()
    for c, v in (("schema_version", SCHEMA_VERSION), ("run_id", run_id), ("category", cfg["dataset"]["category"]),
                 ("scope", "run_all"), ("filter_json", "{}")):
        df[c] = v
    for c in COMMON:
        if c not in df:
            df[c] = None
    if "snapshot_id" in df and "method" in df:
        df["snapshot_id"] = df["snapshot_id"].fillna(df["method"].map(lambda m: f"{cfg.profile}-{m}" if m else None))
    for c in df.columns:
        if df[c].map(lambda x: isinstance(x, (list, dict))).any():
            df[c] = df[c].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run_evaluate(cfg: Config) -> dict:
    t0 = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + sha(str(time.time()))[:6]
    rep = cfg.p("reports")
    construction = eval_construction(cfg)
    aug_rows, audit, curve = eval_augmentation(cfg)
    ret_rows, comparisons, ret_extra = eval_retrieval(cfg)
    qa_rows, qa_per = eval_qa(cfg)
    agree = eval_agreement(cfg)
    kgacc = eval_kg_accuracy(cfg)
    cost = ledger_summary(cfg)
    result = {"run_id": run_id, "profile": cfg.profile, "created_utc": datetime.now(timezone.utc).isoformat(),
              "construction": construction, "augmentation": aug_rows, "tau_curve": curve,
              "retrieval": ret_rows, "comparisons": comparisons,
              "retrieval_extra": {k: v for k, v in ret_extra.items() if k != "per_query"},
              "retrieval_per_query": ret_extra.get("per_query", []), "qa": qa_rows, "qa_per_query": qa_per,
              "agreement": agree, "kg_accuracy": kgacc, "cost": cost, "duration_sec": round(time.time() - t0, 1)}
    write_json(rep / "evaluation.json", result)
    to_csv(pd.DataFrame(construction + kgacc), rep / "construction_quality.csv", run_id, cfg)
    to_csv(audit if len(audit) else pd.DataFrame(columns=["claim_id"]), rep / "augmentation_audit.csv", run_id, cfg)
    rq = pd.DataFrame(ret_rows + qa_rows + agree + aug_rows)
    to_csv(rq, rep / "retrieval_qa_metrics.csv", run_id, cfg)
    to_csv(pd.DataFrame(comparisons) if comparisons else pd.DataFrame(columns=["baseline"]), rep / "ablations.csv", run_id, cfg)
    to_csv(pd.DataFrame(cost) if cost else pd.DataFrame(columns=["stage"]), rep / "cost_latency.csv", run_id, cfg)
    to_csv(error_cases(cfg, audit, ret_extra.get("per_query", [])), rep / "error_cases.csv", run_id, cfg)
    from .report import write_results
    write_results(cfg)
    return {"run_id": run_id, "reports": str(rep), "comparisons": comparisons[:6],
            "retrieval_extra": result["retrieval_extra"]}


def error_cases(cfg: Config, audit: pd.DataFrame, per_query: list[dict]) -> pd.DataFrame:
    rows = []
    if len(audit):
        bad = audit[(audit.claim_status == "accepted") & (audit.judge_label == "unsupported")]
        for r in bad.head(200).itertuples():
            rows.append({"case_id": r.claim_id, "doc_id": r.product_id, "gold_claim": "unsupported (judge)",
                         "predicted_claim": f"{r.aspect}:{r.polarity}", "verdict": "false_accept",
                         "error_type": "augmentation_false_accept", "evidence_quote": r.evidence_quote, "method": r.method})
    pq = pd.DataFrame(per_query)
    if len(pq):
        b5 = pq[pq.system == "B5"].set_index("qid")
        b2 = pq[pq.system == "B2"].set_index("qid")
        for qid in b5.index.intersection(b2.index):
            d = b5.loc[qid, "ndcg@10"] - b2.loc[qid, "ndcg@10"]
            if not math.isnan(d) and d < -0.1:
                rows.append({"case_id": qid, "doc_id": None, "gold_claim": None, "predicted_claim": None,
                             "verdict": f"B5 worse than B2 by {d:.3f}", "error_type": "retrieval_regression",
                             "evidence_quote": None, "method": "B5"})
    return pd.DataFrame(rows, columns=["case_id", "doc_id", "gold_claim", "predicted_claim", "verdict", "error_type",
                                       "evidence_quote", "method"])
