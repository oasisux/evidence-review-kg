"""Render reports/RESULTS.md from reports/evaluation.json (measured values only)."""
from __future__ import annotations

import pandas as pd

from .common import Config, read_json


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(x) for x in r) + " |")
    return "\n".join(lines) + "\n"


def write_results(cfg: Config) -> str:
    ev = read_json(cfg.p("reports") / "evaluation.json", {})
    aug = read_json(cfg.p("graphs") / "augment_summary.json", {})
    man = read_json(cfg.p("processed") / "manifest.json", {})
    out = [f"# EviReview-KG results\n\nrun_id `{ev.get('run_id')}` · profile `{cfg.profile}` · "
           f"{man.get('n_selected_products')} products / {man.get('n_reviews_kept')} reviews (All_Beauty).\n",
           "All gold labels come from the LLM judge (claude-opus-5, raw text only). "
           "See the judge check section for how far to trust them. "
           "Values are measurements from this run, not guarantees.\n"]

    agree = pd.DataFrame(ev.get("agreement", []))
    src = agree.label_source.dropna().iloc[0] if len(agree) and "label_source" in agree and agree.label_source.notna().any() else "none"
    if src.startswith("llm_auditor"):
        out.append("## Judge check: LLM cross-audit (no human validation)\n\n"
                   f"The check items (all queries, 30 claims, 30 qrels, 10 answers) were answered by a third model (`{src.split(':', 1)[1]}`), different from the "
                   "extractor (claude-sonnet-5) and the judge (claude-opus-5), given the same information a human would see. "
                   "This measures agreement between two LLMs, not correctness against human judgement.\n\n")
    else:
        out.append("## Judge check: human (30-minute session)\n")
    out.append(_table(agree[["metric", "value", "n_samples", "evaluation_status"]]) if len(agree) else "_(none)_\n")

    con = pd.DataFrame(ev.get("construction", []))
    out.append("## T1 — KG construction\n")
    if len(con):
        keep = con[con.metric.isin(["n_claims", "valid_provenance_rate", "dangling_edges", "product_aspect_pairs",
                                    "contested_groups", "claim_precision_judge", "t1_precision_vs_reference",
                                    "t1_recall_vs_reference", "t1_f1_vs_reference"])]
        out.append(_table(keep[["metric", "method", "value", "n_samples", "ci_low", "ci_high"]]))
    kg = pd.DataFrame(ev.get("kg_accuracy", []))
    if len(kg) and "scope" in kg:
        out.append("\n### KG opinion-distribution accuracy (per product–aspect, vs judge labels of every mentioning review)\n\n")
        piv = kg.pivot_table(index=["method", "scope"], columns="metric", values="value").reset_index()
        out.append(_table(piv))
    out.append("## T2 — Augmentation (RQ1: targeting vs. same-budget random re-extraction)\n")
    a = pd.DataFrame(ev.get("augmentation", []))
    if len(a):
        out.append(_table(a[["metric", "method", "value", "n_samples", "ci_low", "ci_high", "evaluation_status"]]))
    if aug:
        out.append(f"\nτ = {aug['tau'].get('tau')} ({aug['tau'].get('source')}, dev labels n={aug['tau'].get('n_labels')}). "
                   f"Gaps: {aug['n_gaps']}. LLM review slots: C1 {aug['c1_review_slots']} vs A-budget {aug['budget_review_slots']}.\n")
    curve = pd.DataFrame(ev.get("tau_curve", []))
    if len(curve):
        out.append("\nPrecision–coverage by τ (test, judge-labelled entailed candidates):\n\n")
        out.append(_table(curve))
    out.append("## T3 — Usefulness (RQ2)\n\n### Retrieval (test queries, NDCG@10)\n")
    r = pd.DataFrame(ev.get("retrieval", []))
    if len(r) and "query_type" in r:
        piv = r[r.metric == "ndcg@10"].pivot_table(index="method", columns="query_type", values="value")
        out.append(_table(piv.reset_index()))
        mrr = r[r.metric == "mrr@10"].pivot_table(index="method", columns="query_type", values="value")
        out.append("\nMRR@10:\n\n" + _table(mrr.reset_index()))
    comp = pd.DataFrame(ev.get("comparisons", []))
    if len(comp):
        out.append("\n### Paired comparisons (ΔNDCG@10, bootstrap 95% CI)\n\n")
        out.append(_table(comp[["variant", "baseline", "query_type", "variant_value", "baseline_value", "delta",
                                "ci_low", "ci_high", "n_queries"]]))
    out.append(f"\nRetrieval details: {ev.get('retrieval_extra')}\n")
    qa = pd.DataFrame(ev.get("qa", []))
    out.append("\n### Answers (test QA queries)\n\n")
    if len(qa) and "method" in qa:
        out.append(_table(qa.pivot_table(index="method", columns="metric", values="value").reset_index()))
    cost = pd.DataFrame(ev.get("cost", []))
    out.append("\n## Cost and time\n\n")
    if len(cost):
        out.append(_table(cost[["stage", "model", "calls", "input_tokens", "output_tokens", "cost", "duration_sec"]]))
        out.append(f"\nTotal LLM cost: ${cost.cost.sum():.2f} USD.\n")
    text = "\n".join(out)
    (cfg.p("reports") / "RESULTS.md").write_text(text)
    return text
