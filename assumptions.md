# Assumptions and deviations

Decisions taken while implementing `spec/EviReview-KG_Spec_v3_TermProject_KO.md` (v3.1) under a same-day deadline with one person doing review only.

## Scope and data
- Research sample: 100 products × **≤20** reviews (spec allows reducing 30→20 first). Products need ≥15 reviews in the raw file and a metadata row. Actual: 1,880 reviews after de-duplication (some products have 15–19 unique reviews).
- Dataset pinned to Hugging Face commit `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`; raw JSONL files read directly (no loading script). SHA-256 of both files is in `data/processed/manifest.json`.
- Product split: 30 dev / 70 test products, shared by every track.
- The fixture has 6 products / 30 reviews (spec said 15/60). It is hand-written and covers each contract-test edge case.
- G0 metadata attributes are limited to Store and Brand (from `details.Brand`). Categories were empty in the sample, so no `IN_CATEGORY` edges were created.

## Models
- Extraction, verification and answers: `claude-sonnet-5` (effort `low`). Judge: `claude-opus-5` (effort `medium`). The judge is a different model and sees raw review text only.
- Calls are regular (not Batch API) to meet the deadline. The hard budget stop is `max_cost_usd: 60`.
- `temperature` is not set: Sonnet 5 / Opus 5 reject non-default sampling parameters. Responses are cached on disk (`.cache/llm/`), so re-runs are deterministic.

## Method details not fixed by the spec
- **Offsets.** The LLM returns a verbatim quote; the program computes offsets itself (an exact unique match, or a unique whitespace/case-insensitive match). Ambiguous matches become `needs_review` and are never accepted. All offsets refer to the field `doc = NFC(title) + "\n" + NFC(text)`.
- **Predicates.** Review claims use `OPINION_ON` only. Usage context is represented by `context_key`, not by separate `EXPERIENCED_IN` / `REPORTED_SUITABLE_FOR` claims.
- **G1** = direct LLM extraction that passes the hard gates (no semantic verification). This is the "plain LLM triple extraction" baseline.
- **CEVA (G2)** = G1 claims that pass the semantic gate and `s ≥ τ`, plus C1 gap-recovered claims that pass the same gates, plus C3 aggregates.
- **A-budget** shares CEVA's verification and aggregation. Only the C1 targeting is replaced: the same number of review slots per product is drawn at random from unlinked reviews and extracted with the general schema. So A-budget isolates the effect of targeting.
- **A-verify** = hard gates only (`e` treated as 1) for both G1 and C1 claims.
- **A-context** = CEVA claims with every context collapsed to `unspecified`. Graph scoring uses mention volume `min(n_authors/3,1)` instead of polarity, and answer evidence is not polarity-balanced.
- **τ** is tuned on judge labels of 150 dev G1 candidates. If no τ reaches dev precision ≥ 0.85, recovered claims are not auto-accepted (`needs_review`).
- **Graph weight in RRF** is chosen from {0.5, 1, 2, 3} on dev queries only.
- **qrels.** Products whose raw review sentences never mention a query's target aspect (alias or embedding match) get grade 0 without a judge call.
- **Recall@10** is over all judged products (no pooling, since the full catalog is 100 products).

## Labels
- Every gold label is produced by the LLM judge. One person checks the judge in a 30-minute blind session in the app's "검수" tab, and agreement (accuracy, Cohen's kappa) is reported. There is no independent double annotation, so inter-annotator agreement is not measured.

## Judge check by LLM cross-audit (decided 2026-10-07)
- At the user's request, the 30-minute human check was replaced by an LLM cross-audit. `claude-fable-5-1` (effort `medium`) answered the same 110 items the human tab shows (`src/evireview/checklist.py`), given the same information. It is a different model from both the extractor and the judge.
- Results are labelled `label_source = llm_auditor:claude-fable-5-1`. They measure agreement between two LLMs; **there is no human validation** in this run. If `annotations/human_check.json` exists (the human tab still works), it takes precedence.
- Queries the auditor dropped (5 of 40: four near-duplicates and one vague query) are excluded from evaluation, as human drops would be.
- Caveat found while re-evaluating: after the drops only 2 dev retrieval queries remain. The RRF graph weight is tuned on those, so it is fragile. With all 4 dev queries, A-budget was tuned to weight 1.0 and beat B5 (ΔNDCG −0.038, CI excluding 0). With 2 dev queries every graph system gets 0.5 and the difference disappears (+0.006, CI includes 0). Comparisons between graph variants should therefore be read at matched weights.

## Run 2 changes and the A-budget control (2026-10-07)
- Run 2: products need ≥30 reviews; up to 50 reviews per product (4,502 reviews); non-beauty products are excluded by an LLM title classifier (20 of 150 skipped). LLM query parsing; product-type match; the graph re-ranks only text candidates; graph weight fixed at 1.0; 105 queries; B3 (metadata only) and B6 (LLM re-ranking) baselines; KG opinion-distribution accuracy.
- **Duplicates.** A recovered claim whose span overlaps an existing claim on the same review and aspect gets status `duplicate` and is not accepted. Before this fix, 72% of A-budget's accepted claims were re-extractions of claims already in the KG, which inflated its precision (0.946 → 0.858 after the fix).
- **What A-budget controls.** A-budget is matched to C1 on the **number of reviews sent to the LLM** (839 review slots), not on dollar cost. Cost-based usefulness is out of scope by the user's decision, so per-dollar comparisons and B6's latency/cost are recorded but not used for conclusions.
- **Retrieval.** Recovered claims are under 2% of the KG, so retrieval cannot distinguish C1 from A-budget (ΔNDCG +0.002). A-budget is interpreted on T2 (claim quality and new coverage) only.
