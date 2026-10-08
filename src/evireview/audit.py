"""LLM cross-audit of the judge: a third model answers the same 110 check items a human would.
This measures agreement between two different LLMs. It is not human validation."""
from __future__ import annotations

from .checklist import answer_items, claim_items, qrel_items, queries
from .common import Config, write_json
from .judge import CLAIM_SYSTEM, QREL_SYSTEM, product_aspect_sentences
from .llm import LLM
from .pipeline import load_data

AUDIT_MODEL = "claude-fable-5-1"
AUDIT_EFFORT = "medium"


def _call(llm: LLM, stage: str, system: str, user: str, schema: dict) -> dict:
    return llm.json_call(stage=f"audit_{stage}", model=AUDIT_MODEL, effort=AUDIT_EFFORT, system=system,
                         user=user, schema=schema)


def _obj(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def _results(item_props: dict) -> dict:
    return _obj({"results": {"type": "array", "items": _obj(item_props)}})


QUERY_SYSTEM = """You review evaluation questions for a review-based beauty shopping assistant before they are used.
approve a question only if all hold:
- it reads like a natural shopper question;
- it fits its declared type (pos_condition: find products meeting a positive condition; neg_condition: find products with few complaints / absence of a problem; compare: compares the listed products on an aspect; disagreement: asks about disagreement or main complaints for the listed product; unanswerable: customer reviews could not answer it);
- for compare/disagreement it clearly refers to the listed products;
- it is not a near-duplicate of another question in the list.
Otherwise drop it. Give a short reason."""

ANSWER_SYSTEM = """You check a shopping assistant's answer against the review quotes it cites.
ok = every cited quote supports the sentence that cites it, and the answer does not state reviewer opinions as verified facts.
not_ok = otherwise. Give a short reason."""


def run_audit(cfg: Config) -> dict:
    llm = LLM(cfg)
    _, products, _ = load_data(cfg)
    titles = products.set_index("product_id").title
    reviews, _, _ = load_data(cfg)
    rv = reviews.set_index("review_id")
    out = {"model": AUDIT_MODEL, "effort": AUDIT_EFFORT, "label_source": "llm_auditor",
           "note": "LLM cross-audit, not human validation", "queries": {}, "claims": {}, "qrels": {}, "answers": {}}

    # 1. queries (one call, they must be seen together for duplicate checks)
    qs = queries(cfg)
    lines = [f"- id={q['qid']} type={q['type']} products=[{'; '.join(titles[p][:60] for p in q['product_ids'])}] "
             f"text=\"{q['text']}\"" for q in qs]
    res = _call(llm, "queries", QUERY_SYSTEM, "Questions:\n" + "\n".join(lines),
                _results({"id": {"type": "string"}, "decision": {"type": "string", "enum": ["approve", "drop"]},
                          "reason": {"type": "string"}}))
    for r in res["results"]:
        out["queries"][r["id"]] = {"decision": r["decision"], "reason": r["reason"]}

    # 2. claims (same review text + claim the human sees)
    items = claim_items(cfg)

    def claims_batch(batch):
        docs = {}
        for c in batch:
            docs.setdefault(c["review_id"], f"D{len(docs)}")
        parts = [f"<review id=\"{k}\">\n{rv.loc[rid].doc}\n</review>" for rid, k in docs.items()]
        lines = [f"- id={c['id']} review={docs[c['review_id']]} aspect={c['aspect']} polarity={c['polarity']} "
                 f"quote=\"{c['quote']}\"" for c in batch]
        return _call(llm, "claims", CLAIM_SYSTEM, "\n\n".join(parts) + "\n\nClaims:\n" + "\n".join(lines),
                     _results({"id": {"type": "string"}, "supported": {"type": "boolean"}}))["results"]

    judge_of = {c["id"]: c["judge"] for c in items}
    for res in llm.map(claims_batch, [items[i:i + 10] for i in range(0, len(items), 10)], desc="audit_claims"):
        for r in res or []:
            if r["id"] in judge_of:
                out["claims"][r["id"]] = {"auditor": "supported" if r["supported"] else "unsupported",
                                          "judge": judge_of[r["id"]]}

    # 3. qrels (request + product title + the same aspect sentences)
    qmap = {q["qid"]: q for q in qs}
    qitems = qrel_items(cfg)

    def qrels_batch(batch):
        blocks = []
        for k, x in enumerate(batch):
            q = qmap[x["qid"]]
            sents = product_aspect_sentences(cfg, x["product_id"], q["target_aspects"])
            blocks.append(f"<pair id=\"{k}\">\nRequest: {q['text']}\nProduct: {titles[x['product_id']][:80]}\n" +
                          "\n".join(f"- {s}" for s in sents) + "\n</pair>")
        res = _call(llm, "qrels", QREL_SYSTEM + "\nEach pair has its own request; judge each pair independently.",
                    "\n\n".join(blocks),
                    _results({"id": {"type": "string"}, "grade": {"type": "integer", "enum": [0, 1, 2]}}))
        return [(batch[int(r["id"])], r["grade"]) for r in res["results"] if r["id"].isdigit() and int(r["id"]) < len(batch)]

    for res in llm.map(qrels_batch, [qitems[i:i + 10] for i in range(0, len(qitems), 10)], desc="audit_qrels"):
        for x, g in res or []:
            out["qrels"][x["id"]] = {"auditor": g, "judge": x["judge"]}

    # 4. answers (question, answer, cited quotes)
    aitems = answer_items(cfg)
    blocks = []
    for k, a in enumerate(aitems):
        labels = {e["label"]: e for e in a["evidence"]}
        cited = "\n".join(f"[{c}] \"{labels[c]['quote']}\"" for c in a["citations"] if c in labels)
        blocks.append(f"<answer id=\"{k}\">\nQuestion: {a['query']}\nAnswer: {a['answer']}\nCited quotes:\n{cited}\n</answer>")
    res = _call(llm, "answers", ANSWER_SYSTEM, "\n\n".join(blocks),
                _results({"id": {"type": "string"}, "verdict": {"type": "string", "enum": ["ok", "not_ok"]},
                          "reason": {"type": "string"}}))
    for r in res["results"]:
        if r["id"].isdigit() and int(r["id"]) < len(aitems):
            out["answers"][aitems[int(r["id"])]["id"]] = {"auditor": r["verdict"], "reason": r["reason"]}

    write_json(cfg.p("annotations") / "llm_audit.json", out)
    print(f"audit done: {sum(len(out[k]) for k in ('queries', 'claims', 'qrels', 'answers'))} items, "
          f"spent total ${llm.spent:.2f}")
    return out
