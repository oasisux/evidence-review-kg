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

## Run 3 changes: product families, seller layer, aspects v2 (2026-10-16)
Motivation: in run 2, 94% of claims had context `unspecified` and only 5 of 105 queries were conditional; product metadata was stored but not used; the 18 aspects did not fit many products (25% of claims fell into `effectiveness`); 100 products spanned 84 product types.
- **Sample.** Two families, 50 products each: `skin_moisturizer` and `hair_care` (definitions in `aspects_v2.yaml`). Chosen because their reviewers state skin/hair types, which the context dictionary covers. The LLM title classifier labels `family`; products are taken in seeded order until each quota is full. Dev = 15 products per family. A keyword classifier replaces the LLM in offline profiles and is only approximate (e.g. travel bottles "for shampoo" count as hair care).
- **Aspects v2.** Common aspects + per-family aspects; coverage, shade_match, wear_time and hold are dropped; new: skin_results, cleansing, conditioning, scalp_condition, hair_results; `effectiveness` is narrowed to a fallback. The draft list is hand-written. `evireview induce-aspects` proposes changes from dev reviews only; nothing is applied automatically. `aspects_v1.yaml` is kept for the record of runs 1–2.
- **`other` aspect.** The extractor may label an opinion `other` instead of forcing it into the closest aspect. Such candidates are rejected (`no_matching_aspect`) and reported as `no_matching_aspect_rate`. An aspect outside the product's family is rejected (`aspect_not_in_family`).
- **Seller layer (EDC).** Extract: open (attribute, value, context, verbatim quote) from the listing. Define: one sentence per raw attribute name. Canonicalize: greedy merge of names whose `name: definition` embeddings have cosine ≥ 0.75, then values within an attribute at ≥ 0.88. Deviation from EDC: no LLM check of merges. The thresholds are guesses and should be checked against `seller_schema.json`. Seller statements are `source_kind=seller`, `epistemic=seller_claim`, and never enter the review opinion aggregates. A seller target is not stored as a review context.
- **Context queries.** For a query with a context, each (product, aspect) is scored from reviewers who state that context. When there are none, all opinions are used, with weight 1.0 if the seller targets that context and 0.5 if not. B3 (metadata only) keeps the products of the requested type that the seller targets at the context.
- **Fair baseline.** With `retrieval.index_metadata`, every system (B1/B2 included) also indexes each product's listing text, so a KG gain cannot come from the listing alone.
- **Judge.** About a third of pos/neg queries name a context (`target_context`). For those, qrels give grade 2 only when reviewers who state the context are satisfied. Satisfied reviewers who do not state it give at most 1. Seller text is never shown to the judge. Metrics are also reported for `context_condition` queries.
- **Label safety.** `annotations/sample.json` stamps the sample hash. The judge refuses to reuse labels from another sample. Augment stops when the dev labels match no current candidate.



## Run 4: relation-based rating prediction (decided 2026-10-16)
Plan: "상품별 리뷰 요약 KG -> 상품 간 관계로 평가를 예측하는 KG". Decisions where the plan left room or the data disagreed:
- **Families.** Two families (skin_moisturizer, hair_care) x 150 products, not about six types across skincare, hair care and makeup: the aspect dictionary is validated for these two families only, and makeup would need new aspects. Within a family, products get one of 5-7 canonical types (`aspects_v2.yaml` `types`), chosen by the title classifier from a closed list, so spelling variants cannot split a type.
- **Sample.** Products with >= 20 reviews and metadata. Every eligible title is classified; within a family, brands with >= 3 candidates are taken first (<= 8 products each, <= half the quota), then the quota is filled in seeded order. A pilot on the raw ratings (`evireview pilot`) found the brand relation the strongest (-11.7% MAE at k=0) but a plain random sample has almost no same-brand pairs.
- **Reviews.** The first 30 reviews of each product in time order (how reviews accumulate on a new product), not a random 50.
- **Seller text.** Only about 53% of products have feature bullets or a description; the rest have a title only. Attribute relations therefore cover fewer products than type and brand.
- **Attribute nodes.** Canonical (attribute, value) stated by >= 2 products. Related aspects come from one LLM mapping call per 60 attributes (empty when the statement says nothing about performance or feel). An attribute relation is used only for its related aspects.
- **Estimate.** Per relation, the mean positive share of neighbors with >= 2 opinions on the aspect, shrunk toward the family mean with 2 pseudo-neighbors; prior = weighted mean of the available relations (family mean when none); posterior = (n_pos + lambda * prior) / (n + lambda). Weights {0, 0.5, 1, 2} per relation and lambda {1, 2, 4, 8, 16} are tuned by MAE on dev products only, treated as new products (k = 0, 1, 3). No estimate when a product has no own opinions and fewer than 3 neighbors with opinions.
- **Experiment 1.** 30% of each family's test products play the new product. Truth = positive share of their opinions in reviews 6-30 (the same truth for every k), aspects with >= 5 such opinions. Neighbor statistics come from the other products only. LLM methods use claude-sonnet-5 and must give a number for every aspect. The 80% interval bounds the true share; since the truth is an observed share of a few opinions, coverage is also reported against the beta-binomial predictive interval.
- **Experiment 2.** 60 test queries (pos/neg) from the judge; in each view (role products keep their first k reviews) claims, aggregates and predictions are recomputed from what remains. Qrels are pooled over the top 20 of B2, B5, B7 and B6 in every view (1,770 judged pairs); unpooled products count as not relevant, so NDCG is comparable across systems but not with Run 3. Graph weight 1.0.
- **Experiment 3.** The judge (claude-opus-5) labels 60 type assignments, 60 product-attribute links (quote supports value, name fits, related aspects right) and 40 seller targets.
- **New-product exposure (B7L, B8, B8-ucb, B8-slot).** A diagnosis at k=0 found that most relevant new products never entered the text candidates. B7L adds a text list over seller listings only (one document per product). B8 also adds KG candidates: every product of the requested type (embedding match of the parsed product type, threshold 0.55) with an estimate on the query aspect. B8-slot reserves 2 of the top 10 for products with fewer than 5 reviews, the best of them by type match and estimate; the number of slots is a policy choice, not tuned. B8-ucb ranks by the upper end of the 80% interval.
- **Exposure policy B9** (`evireview exposure`). New products of the requested type are scored by listing relevance, seller-attribute match, seller-target match and predicted rating; up to max_slots enter the top 10. 160 settings, chosen on 16 dev queries (max exposure with dev NDCG >= B8). The dev choice (4 slots) lost NDCG on test, so the report gives the whole slot frontier and treats the slot count as a policy choice.
- **Adopted B9 setting (2026-10-17, user decision).** 7 slots with the LLM selector (`retrieval.exposure` in `configs/research_v4.yaml`): new-product exposure first, NDCG at the level of B7. One extra LLM call per search.
