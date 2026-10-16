"""LLM judge (different model from the extractor). Sees raw review text only - never KG output or system names."""
from __future__ import annotations

import random

import pandas as pd

from .common import Config, aspect_names, contexts, families, ontology, read_json, sha, write_json
from .extract import _claim_schema, extract_system
from .llm import LLM
from .pipeline import load_data, sentence_table

QUERY_TYPES = ["pos_condition", "neg_condition", "compare", "disagreement", "unanswerable"]
RETRIEVAL_TYPES = {"pos_condition", "neg_condition"}
QA_TYPES = {"compare", "disagreement", "unanswerable"}
QA_SYSTEMS = ["B2", "B4", "B5", "A-budget", "A-verify", "A-context"]
PLAN = {"dev": {"pos_condition": 8, "neg_condition": 8, "compare": 3, "disagreement": 2, "unanswerable": 2},
        "test": {"pos_condition": 30, "neg_condition": 30, "compare": 10, "disagreement": 6, "unanswerable": 6}}


def _judge(cfg: Config, llm: LLM, stage: str, system: str, user: str, schema: dict, max_tokens: int = 16000) -> dict:
    return llm.json_call(stage=f"judge_{stage}", model=cfg["judge"]["model"], effort=cfg["judge"]["effort"],
                         system=system, user=user, schema=schema, max_tokens=max_tokens)


def raw_sentences(cfg: Config) -> pd.DataFrame:
    path = cfg.p("processed") / "sentences.parquet"
    if path.exists():
        return pd.read_parquet(path)
    reviews, _, _ = load_data(cfg)
    st = sentence_table(reviews, cfg["augmentation"]["mention_cos"])
    st.to_parquet(path, index=False)
    return st


# ------------------------------------------------------------------ claims (dev tau labels, T1/T2 precision)
CLAIM_SYSTEM = """You are an independent annotator checking opinion claims against Amazon beauty reviews.
A claim says: in this review, the reviewer expresses <polarity> opinion about <aspect> of the reviewed product, and <quote> is the evidence.
supported=true only if the review text clearly supports that exact aspect AND polarity for the reviewed product (not another product, not hypothetical).
Aspect definitions:
""" + "\n".join(f"- {a}: {s['description']}" for a, s in ontology()["aspects"].items())

CLAIM_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["results"],
                "properties": {"results": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["id", "supported"],
                    "properties": {"id": {"type": "string"}, "supported": {"type": "boolean"}}}}}}


def judge_claims(cfg: Config, llm: LLM, cands: pd.DataFrame, stage: str) -> dict[str, str]:
    reviews, _, _ = load_data(cfg)
    rv = reviews.set_index("review_id")
    cands = cands.sort_values("review_id").reset_index(drop=True)
    batches = [cands.iloc[i:i + 15] for i in range(0, len(cands), 15)]

    def run(b):
        docs = {rid: f"D{i}" for i, rid in enumerate(b.review_id.unique())}
        parts = [f"<review id=\"{k}\">\n{rv.loc[rid].doc}\n</review>" for rid, k in docs.items()]
        lines = [f"- id={c.cand_id} review={docs[c.review_id]} aspect={c.aspect} polarity={c.polarity} quote=\"{c.quote}\""
                 for c in b.itertuples()]
        return _judge(cfg, llm, stage, CLAIM_SYSTEM, "\n\n".join(parts) + "\n\nClaims:\n" + "\n".join(lines),
                      CLAIM_SCHEMA)["results"]

    out = {}
    for res in llm.map(run, batches, desc=f"judge_{stage}"):
        for r in res or []:
            out[r["id"]] = "supported" if r["supported"] else "unsupported"
    return out


def run_dev(cfg: Config, llm: LLM) -> None:
    g1 = pd.read_parquet(cfg.p("graphs") / "candidates_g1.parquet")
    dev = g1[(g1.partition == "dev") & g1.hard_pass & (g1.verify_label == "entail")]
    dev = dev.sample(min(150, len(dev)), random_state=cfg["seed"])
    write_json(cfg.p("annotations") / "judge_dev_candidates.json", judge_claims(cfg, llm, dev, "dev"))


def run_claims(cfg: Config, llm: LLM) -> None:
    g = cfg.p("graphs")
    g1 = pd.read_parquet(g / "candidates_g1.parquet")
    seed = cfg["seed"]
    test_g1 = g1[(g1.partition == "test") & g1.hard_pass]
    sets = {"g1": test_g1.sample(min(250, len(test_g1)), random_state=seed)}
    for run in ("c1", "budget", "c1legacy"):
        df = pd.read_parquet(g / f"candidates_{run}.parquet")
        df = df[(df.partition == "test") & df.hard_pass]
        sample = df.sample(min(300, len(df)), random_state=seed)
        accepted = df[(df.status_verified == "accepted") | (df.status_hard == "accepted")]  # label all accepted
        sets[run] = pd.concat([sample, accepted]).drop_duplicates("cand_id")
    labels = read_json(cfg.p("annotations") / "judge_claims.json", {})
    for name, df in sets.items():
        todo = df[~df.cand_id.isin(labels)]
        if len(todo):
            labels.update(judge_claims(cfg, llm, todo, f"claims_{name}"))
    write_json(cfg.p("annotations") / "judge_claims.json", labels)
    write_json(cfg.p("annotations") / "judge_claim_samples.json", {k: v.cand_id.tolist() for k, v in sets.items()})


# ------------------------------------------------------------------ T1 reference claims
def run_reference(cfg: Config, llm: LLM) -> None:
    reviews, _, split = load_data(cfg)
    rng = random.Random(cfg["seed"])
    picks = {}
    for part, n in (("dev", 10), ("test", 30)):
        ids = sorted(reviews[reviews.product_id.map(split) == part].review_id)
        rng.shuffle(ids)
        picks[part] = ids[:n]
    rv = reviews.set_index("review_id")
    by_family: dict = {}
    for rid in (rid for ids in picks.values() for rid in ids):  # same family-specific aspect set as the extractor
        fam = rv.loc[rid].family if rv.loc[rid].family in families() else None
        by_family.setdefault(fam, []).append(rid)
    batches = [(fam, ids[i:i + 5]) for fam, ids in by_family.items() for i in range(0, len(ids), 5)]

    def run(job):
        fam, b = job
        user = ("Independently annotate ALL aspect opinions in these reviews (reference annotation).\n\n" +
                "\n\n".join(f"<review key=\"R{i}\">\n{rv.loc[rid].doc}\n</review>" for i, rid in enumerate(b)))
        res = _judge(cfg, llm, "reference", extract_system(fam), user, _claim_schema(aspect_names(fam), allow_other=True))
        return [{"review_id": b[int(x["key"].lstrip("R"))], **c} for x in res["reviews"]
                if x["key"].lstrip("R").isdigit() and int(x["key"].lstrip("R")) < len(b) for c in x["claims"]]

    claims = [c for r in llm.map(run, batches, desc="judge_reference") if r for c in r]
    write_json(cfg.p("annotations") / "t1_reference.json", {"reviews": picks, "claims": claims})


# ------------------------------------------------------------------ queries
QUERY_SYSTEM = """You write evaluation questions for a beauty-product search and QA system over Amazon reviews.
Write natural English questions a shopper would ask. Use only the product list and aspect statistics given.
Types:
- pos_condition: find products that satisfy a positive condition on an aspect ("lotion that absorbs quickly").
- neg_condition: find products with FEW complaints / absence of a problem ("moisturizer that is not greasy", "shampoo that doesn't irritate the scalp").
- compare: compare 2 given products on one aspect. product_ids must list both products; mention them by a short name.
- disagreement: ask what reviewers disagree about or the main complaints for 1 given product. product_ids lists it.
- unanswerable: plausible questions that reviews cannot answer (lab test results, ingredient certifications, prices in other stores, medical efficacy). product_ids may list 1 product.
For pos/neg conditions, product_ids is empty; pick aspects that many products mention so several products could qualify. About half of them should name a product type from the catalog (e.g. "a body lotion that...", "shampoo that...").
About a third of pos/neg conditions should be conditional on WHO uses the product: name a skin or hair type, climate or time of use from the context statistics (e.g. "a moisturizer that isn't greasy on oily skin", "conditioner that detangles curly hair"), and set target_context to it. Prefer contexts that reviewers of several products mention. All other questions use target_context "unspecified".
Questions must be clearly different from each other: vary aspect, product type, context and wording; never write two questions with the same aspect, product type AND context. target_aspects lists the aspects involved (empty for unanswerable if none fits)."""


def _query_schema() -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["queries"],
            "properties": {"queries": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["type", "text", "product_ids", "target_aspects", "target_context"],
                "properties": {"type": {"type": "string", "enum": QUERY_TYPES}, "text": {"type": "string"},
                               "product_ids": {"type": "array", "items": {"type": "string"}},
                               "target_aspects": {"type": "array", "items": {"type": "string", "enum": aspect_names()}},
                               "target_context": {"type": "string", "enum": contexts()}}}}}}


def run_queries(cfg: Config, llm: LLM) -> list[dict]:
    path = cfg.p("annotations") / "queries.json"
    if path.exists():
        return read_json(path)
    _, products, split = load_data(cfg)
    st = raw_sentences(cfg)
    ex = st.explode("aspects").dropna(subset=["aspects"])
    per_product = ex.groupby(["product_id", "aspects"]).size().unstack(fill_value=0)
    coverage = (per_product >= 2).sum().sort_values(ascending=False)
    titles = products.set_index("product_id").title
    out = []
    ptypes = products.set_index("product_id").product_type if "product_type" in products else None
    ctx_counts = context_mentions(st).groupby("context").product_id.nunique().sort_values(ascending=False)
    ctx_stats = "\n".join(f"- {c}: stated by reviewers of {n} products" for c, n in ctx_counts.items()) or "- (none)"
    for part, plan in PLAN.items():
        pids = [p for p in per_product.index if split.get(p) == part]
        listing = "\n".join(f"- {p}: {titles[p][:90]} | type: {ptypes[p] if ptypes is not None else '?'} | most discussed: "
                            f"{', '.join(per_product.loc[p].sort_values(ascending=False).head(5).index)}" for p in pids)
        type_counts = products.product_type.value_counts().head(25).to_dict() if ptypes is not None else {}
        stats = "\n".join(f"- {a}: discussed (>=2 sentences) in {n} of {len(per_product)} products" for a, n in coverage.items())
        for qtype, n in plan.items():
            user = (f"Aspect statistics over the whole catalog:\n{stats}\n\nContext statistics (review sentences stating "
                    f"a skin/hair type, climate or time of use):\n{ctx_stats}\n\n"
                    f"Product types in the catalog (count): {type_counts}\n\n"
                    f"Products you may reference in compare/disagreement/unanswerable questions:\n{listing}\n\n"
                    f"Write exactly {n} questions of type {qtype}.")
            res = _judge(cfg, llm, f"queries_{part}_{qtype}", QUERY_SYSTEM, user, _query_schema())
            for q in res["queries"][:n]:
                q["type"] = qtype
                q["product_ids"] = [p for p in q["product_ids"] if p in pids]
                if qtype not in RETRIEVAL_TYPES:
                    q["target_context"] = "unspecified"
                if q["type"] in {"compare", "disagreement"} and not q["product_ids"]:
                    continue
                out.append({"qid": "q" + sha(part + q["text"])[:8], "split": part, "status": "draft", **q})
    write_json(path, out)
    return out


# ------------------------------------------------------------------ qrels
QREL_SYSTEM = """You judge how well a product satisfies a shopper's request, using ONLY the review sentences shown for it.
Grades:
2 = reviewers clearly indicate the product satisfies the request (for "few complaints / not X" requests: reviewers report the absence of the problem and complaints are rare).
1 = partially or with mixed evidence (some reviewers satisfied, some complain).
0 = does not satisfy, mostly complaints, or no relevant evidence.
If the request names a kind of product (e.g. "body lotion", "shampoo") and this product is clearly a different kind, grade 0.
If the request names who uses the product (a skin or hair type, climate or time of use): grade 2 needs reviewers who state that condition and are satisfied; satisfied reviewers who do not state it give at most 1; reviewers with that condition who complain give 0.
Judge each product independently."""

QREL_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["grades"],
               "properties": {"grades": {"type": "array", "items": {
                   "type": "object", "additionalProperties": False, "required": ["product", "grade", "reason"],
                   "properties": {"product": {"type": "string"}, "grade": {"type": "integer", "enum": [0, 1, 2]},
                                  "reason": {"type": "string"}}}}}}


def context_mentions(st: pd.DataFrame) -> pd.DataFrame:
    """Sentences stating a context (same cue dictionary as extraction), one row per (sentence, context)."""
    from .common import detect_context
    ctx = st.text.map(detect_context)
    return st.assign(context=ctx)[ctx != "unspecified"]


def product_aspect_sentences(cfg: Config, pid: str, aspects: list[str], limit: int = 10,
                             context: str = "unspecified") -> list[str]:
    """Review sentences on the aspects; for a context request, sentences of the reviews that state the context
    come first (the judge needs them to give grade 2)."""
    st = raw_sentences(cfg)
    g = st[(st.product_id == pid) & st.aspects.apply(lambda xs: any(a in xs for a in aspects))]
    if context != "unspecified":
        cm = context_mentions(st[st.product_id == pid])
        ctx_reviews = set(cm[cm.context == context].review_id)
        stated = cm[cm.context == context].text.tolist()
        g = pd.concat([g[g.review_id.isin(ctx_reviews)], g[~g.review_id.isin(ctx_reviews)]])
        return list(dict.fromkeys(stated[:3] + g.text.tolist()))[:limit]
    return g.text.head(limit).tolist()


def run_qrels(cfg: Config, llm: LLM, queries: list[dict]) -> dict:
    path = cfg.p("annotations") / "qrels_judge.json"
    qrels = read_json(path, {})
    _, products, _ = load_data(cfg)
    titles = products.set_index("product_id").title
    jobs = []
    for q in queries:
        if q["type"] not in RETRIEVAL_TYPES or q["qid"] in qrels:
            continue
        cands = [p for p in products.product_id if product_aspect_sentences(cfg, p, q["target_aspects"], 1)]
        for i in range(0, len(cands), 6):
            jobs.append((q, cands[i:i + 6]))

    def run(job):
        q, pids = job
        blocks = []
        for k, p in enumerate(pids):
            sents = product_aspect_sentences(cfg, p, q["target_aspects"], 8, q.get("target_context", "unspecified"))
            blocks.append(f"<product id=\"P{k}\" title=\"{titles[p][:80]}\">\n" +
                          "\n".join(f"- {s}" for s in sents) + "\n</product>")
        res = _judge(cfg, llm, "qrels", QREL_SYSTEM, f"Request: {q['text']}\n\n" + "\n\n".join(blocks), QREL_SCHEMA)
        return q["qid"], {pids[int(g["product"].lstrip("P"))]: g["grade"] for g in res["grades"]
                          if g["product"].lstrip("P").isdigit() and int(g["product"].lstrip("P")) < len(pids)}

    for r in llm.map(run, jobs, desc="judge_qrels"):
        if r:
            qid, grades = r
            qrels.setdefault(qid, {}).update(grades)
    for q in queries:  # products with no aspect evidence get 0 (stored implicitly)
        if q["type"] in RETRIEVAL_TYPES:
            qrels.setdefault(q["qid"], {})
    write_json(path, qrels)
    return qrels


# ------------------------------------------------------------------ answers
ANSWER_JUDGE_SYSTEM = """You grade an answer from a review-based shopping assistant.
You get the question, reference review sentences for the products involved (ground truth material), the evidence the assistant was given (labeled E1..), and its answer.
- correctness: 2 = consistent with the reference material and answers the question; 1 = partially correct or misses an important side; 0 = wrong, unsupported, or answers when it should abstain.
- statements: number of factual sentences in the answer; supported_statements: how many are supported by the evidence they cite.
- both_sides: true if reviewers in the reference disagree AND the answer presents both sides; false if they disagree and it does not; null-like value "na" if no disagreement.
- should_abstain: true if the reference material cannot answer the question."""

ANSWER_JUDGE_SCHEMA = {"type": "object", "additionalProperties": False,
                       "required": ["correctness", "statements", "supported_statements", "both_sides", "should_abstain"],
                       "properties": {"correctness": {"type": "integer", "enum": [0, 1, 2]},
                                      "statements": {"type": "integer"}, "supported_statements": {"type": "integer"},
                                      "both_sides": {"type": "string", "enum": ["true", "false", "na"]},
                                      "should_abstain": {"type": "boolean"}}}


def run_answers(cfg: Config, llm: LLM, queries: list[dict]) -> None:
    from .answer import answer
    _, products, _ = load_data(cfg)
    titles = products.set_index("product_id").title
    jobs = [(q, s) for q in queries if q["type"] in QA_TYPES for s in QA_SYSTEMS]
    answers = llm.map(lambda j: answer(cfg, llm, j[1], j[0]["text"], j[0]["product_ids"] or None,
                                       stage="answer"), jobs, desc="answers")
    write_json(cfg.p("runs") / "answers.json", [{"qid": q["qid"], **a} for (q, _), a in zip(jobs, answers) if a])

    def grade(pair):
        (q, s), a = pair
        if a is None:
            return None
        ref = []
        for p in q["product_ids"]:
            ref.append(f"<product title=\"{titles[p][:80]}\">\n" + "\n".join(
                f"- {x}" for x in product_aspect_sentences(cfg, p, q["target_aspects"] or aspect_names(), 15)) + "\n</product>")
        ev = "\n".join(f"[{e['label']}] {e['quote']}" for e in a["evidence"])
        user = (f"Question: {q['text']}\n\nReference review sentences:\n" + ("\n".join(ref) or "(none)") +
                f"\n\nEvidence given to assistant:\n{ev or '(none)'}\n\nAnswer (abstained={a['abstained']}):\n{a['answer']}")
        return {"qid": q["qid"], "system": s, **_judge(cfg, llm, "answers", ANSWER_JUDGE_SYSTEM, user, ANSWER_JUDGE_SCHEMA)}

    graded = llm.map(grade, list(zip(jobs, answers)), desc="judge_answers")
    write_json(cfg.p("annotations") / "answer_judgments.json", [g for g in graded if g])


# ------------------------------------------------------------------ B6: LLM re-ranking baseline (no KG)
def run_b6(cfg: Config, llm: LLM, queries: list[dict]) -> None:
    from .retrieval import llm_rerank
    qs = [q for q in queries if q["type"] in RETRIEVAL_TYPES]
    res = llm.map(lambda q: llm_rerank(cfg, llm, q["text"]), qs, desc="b6_rerank")
    write_json(cfg.p("runs") / "b6_rankings.json",
               {q["qid"]: {"top10": [x["product_id"] for x in r["results"]], "latency_ms": r["latency_ms"],
                           "cost_usd": r["cost_usd"]} for q, r in zip(qs, res) if r})


# ------------------------------------------------------------------ KG opinion-distribution accuracy (#6)
KGACC_SYSTEM = """For each review, label the reviewer's opinion about ONE aspect of the reviewed product.
positive / negative / mixed (both) / neutral (mentions it without judging) / none (does not discuss this aspect of this product, or uses the word in another sense).
Aspect definitions:
""" + "\n".join(f"- {a}: {s['description']}" for a, s in ontology()["aspects"].items())


def run_kg_accuracy(cfg: Config, llm: LLM) -> None:
    reviews, _, split = load_data(cfg)
    st = raw_sentences(cfg)
    ex = st.explode("aspects").dropna(subset=["aspects"])
    ex = ex[ex.product_id.map(split) == "test"]
    mention = ex.groupby(["product_id", "aspects"]).review_id.unique()
    ag = pd.read_parquet(cfg.p("graphs") / "ceva" / "aggregates.parquet")
    ag = ag[ag.context == "*"].set_index(["product_id", "aspect"]).n_authors.to_dict()
    in_kg = [k for k in mention.index if ag.get(k, 0) >= 2]
    missing = [k for k in mention.index if ag.get(k, 0) < 2 and len(mention[k]) >= 3]
    rng = random.Random(cfg["seed"])
    rng.shuffle(in_kg)
    rng.shuffle(missing)
    pairs = in_kg[:40] + missing[:20]
    rv = reviews.set_index("review_id")

    def run(pair):
        pid, asp = pair
        rids = list(mention[pair])[:25]
        body = "\n\n".join(f"<review id=\"R{i}\">\n{rv.loc[r].doc}\n</review>" for i, r in enumerate(rids))
        schema = {"type": "object", "additionalProperties": False, "required": ["labels"],
                  "properties": {"labels": {"type": "array", "items": {
                      "type": "object", "additionalProperties": False, "required": ["id", "opinion"],
                      "properties": {"id": {"type": "string"}, "opinion": {"type": "string", "enum":
                                     ["positive", "negative", "mixed", "neutral", "none"]}}}}}}
        res = _judge(cfg, llm, "kgacc", KGACC_SYSTEM, f"Aspect: {asp}\n\n{body}", schema)
        return f"{pid}|{asp}", {rids[int(x["id"].lstrip("R"))]: x["opinion"] for x in res["labels"]
                                if x["id"].lstrip("R").isdigit() and int(x["id"].lstrip("R")) < len(rids)}

    out = {}
    for r in llm.map(run, pairs, desc="judge_kgacc"):
        if r:
            out[r[0]] = {"labels": r[1], "stratum": "in_kg" if tuple(r[0].split("|")) in set(in_kg) else "kg_sparse"}
    write_json(cfg.p("annotations") / "kg_accuracy_judge.json", out)


def check_sample(cfg: Config) -> None:
    """Labels are tied to one product sample. Refuse to mix them with a re-ingested sample: queries and qrels
    are cached by file and would otherwise be reused silently."""
    ann = cfg.p("annotations")
    current = read_json(cfg.p("processed") / "split_manifest.json", {}).get("hash")
    stamp = read_json(ann / "sample.json", {})
    if stamp.get("split_hash") not in (None, current) or (not stamp and any(ann.glob("*.json"))):
        raise SystemExit(f"{ann} holds labels for a different product sample (or labels made before sample stamping). "
                         "Move it away (e.g. to annotations_run2/) and re-run the judge.")
    write_json(ann / "sample.json", {"split_hash": current})


def run(cfg: Config, task: str = "all") -> None:
    if cfg.offline:
        print("offline profile: judge skipped (metrics will be NOT_EVALUATED)")
        return
    check_sample(cfg)
    llm = LLM(cfg)
    if task == "dev":
        run_dev(cfg, llm)
        return
    queries = run_queries(cfg, llm)
    if task in ("all", "reference"):
        run_reference(cfg, llm)
    if task in ("all", "claims"):
        run_claims(cfg, llm)
    if task in ("all", "qrels"):
        run_qrels(cfg, llm, queries)
    if task in ("all", "answers"):
        run_answers(cfg, llm, queries)
    if task in ("all", "b6"):
        run_b6(cfg, llm, queries)
    if task in ("all", "kgacc"):
        run_kg_accuracy(cfg, llm)
    print(f"judge done, total spent ${llm.spent:.2f}")
