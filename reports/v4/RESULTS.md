# EviReview-KG Run 4 results: relation-based rating prediction

Profile `research_v4` · 300 products (skin_moisturizer 150, hair_care 150) · 8,328 reviews (first 30 per product in time order) · All_Beauty, HF commit 2b6d039. Total LLM cost of the run: $71.84 (build and experiments 1, 3: $58.24; experiment 2 with the exposure work: $13.60).

All truth values come from the same extractor (claude-sonnet-5 claims, verified); judge labels come from claude-opus-5. There is no human validation.

## KG built

| item | value |
|---|---|
| G1 candidates / hard-pass | 17,951 / 16,534 |
| tau (dev precision 0.948 at 134 accepted, n=150) | 0.60 |
| canonical product types | 12 (296 of 300 products typed) |
| Attribute nodes (stated by >= 2 products) / with related aspects | 214 / 162 |
| SAME_TYPE / SAME_BRAND / SHARES_ATTRIBUTE edges | 4,655 / 162 / 3,701 |
| products with a SAME_TYPE / SAME_BRAND / SHARES_ATTRIBUTE neighbor | 296 / 101 / 267 |
| predicted (product, aspect) nodes / with no own opinion | 4,757 / 1,309 |

## Experiment 3: accuracy of the new nodes and edges (judge: claude-opus-5)

| entry | accuracy | n | 95% CI |
|---|---|---|---|
| product type | 0.967 | 60 | 0.89-0.99 |
| attribute: quote states the value | 0.883 | 60 | 0.78-0.94 |
| attribute: canonical name fits | 1.000 | 60 | 0.94-1.00 |
| attribute: related aspects right (right or partly) | 0.883 (0.967) | 60 | 0.78-0.94 |
| seller target (TARGETS) | 0.925 | 40 | 0.80-0.97 |

## Experiment 1: aspect ratings of new products

72 test products (30% per family) play the new product and keep only their first k reviews; 228 other products supply neighbors. Truth = positive share of their opinions in reviews 6-30, for aspects with >= 5 such opinions (172 product-aspect targets, the same for every k). Truth mean 0.837; 89.5% of targets have a positive majority; MAE from sampling noise alone ~0.082.

Tuning on dev products (411 cases) kept only the brand relation (weight 0.5, lambda 1); type and attribute relations got weight 0.

MAE (lower is better):

| method | k=0 | k=1 | k=3 | k=5 |
|---|---|---|---|---|
| own reviews only | 0.396 | 0.327 | 0.256 | 0.205 |
| always positive (1.0) | 0.164 | 0.164 | 0.164 | 0.164 |
| family mean + own reviews | 0.167 | 0.168 | 0.158 | 0.143 |
| title-similar products + own reviews | 0.168 | 0.163 | 0.150 | 0.141 |
| **proposed: relations + own reviews** | **0.166** | **0.167** | **0.157** | **0.143** |
| LLM alone (title, listing, k reviews) | 0.208 | 0.208 | 0.176 | 0.153 |
| LLM + reviews of 5 title-similar products | 0.186 | 0.192 | 0.171 | 0.155 |
| **LLM + proposed estimates** | **0.170** | **0.179** | **0.157** | **0.140** |

Paired differences (product-level bootstrap, 95% CI):

| comparison | k=0 | k=1 | k=3 | k=5 |
|---|---|---|---|---|
| proposed - family mean | -0.000 [-0.007, +0.007] | -0.000 [-0.004, +0.004] | -0.001 [-0.004, +0.002] | -0.000 [-0.003, +0.003] |
| LLM + proposed - LLM alone | **-0.042 [-0.058, -0.026]** | **-0.029 [-0.047, -0.013]** | **-0.022 [-0.038, -0.009]** | **-0.015 [-0.026, -0.005]** |
| LLM + proposed - LLM + similar-product reviews | **-0.018 [-0.031, -0.006]** | -0.011 [-0.023, +0.002] | -0.011 [-0.023, +0.001] | **-0.015 [-0.024, -0.005]** |

Rank correlation within (family, aspect), k=0: proposed 0.05, LLM alone 0.17, LLM + proposed 0.16. 80% interval of the proposed estimate: 54% contain the observed truth; against the beta-binomial predictive interval of the observed share, 97% (the interval is too wide, not too narrow).

## Reading

1. **KG augmentation (relations predict new products better): not supported.** The relation-based estimate is no better than the family mean (CI includes 0 at every k), and neither beats always predicting "positive". Tuning gave type and attribute relations weight 0; the brand relation helped on dev but only 23% of targets have a brand neighbor. Opinions are 84% positive, so the between-product signal is small relative to the noise in 5-25 opinions.
2. **Usefulness (the KG gives an LLM information it lacks): supported.** Giving the LLM the KG's estimates cuts its error by 0.042 at k=0 and stays significant at every k. It also beats handing the LLM the raw reviews of five similar products at k=0 and k=5. Most of the gain comes from calibration: alone, the LLM under-predicts satisfaction on high-share aspects (skin_results, conditioning, hair_results).
3. **Novelty (graph relations find better neighbors than text similarity): not supported.** Title-similar products are as good as or slightly better than graph relations (k=3: -0.010, CI includes 0).
4. **By aspect** (k=0): estimates are close to the noise floor for high-share aspects (skin_results 0.05, conditioning 0.06, hair_results 0.09) and poor for aspects where products differ (effectiveness 0.30, residue 0.29, texture 0.26, value_for_money 0.24). LLM alone does better than the KG on value_for_money and effectiveness.

## Experiment 2: retrieval with new products mixed in

The 72 role products keep only their first k reviews; reviews, CEVA claims, opinion aggregates and B7's predictions are recomputed in each view (`data/v4/cold/k*/`). 60 test queries (pos/neg condition) written by the judge. Qrels are pooled: the judge grades every product in the top 20 of any system in any view (2,100 pairs), reading sentences from all 30 reviews; unpooled products count as not relevant. Graph weight 1.0.

Systems added after the first run to fix new-product exposure (a diagnosis at k=0 found that 92% of relevant new products match the query's product type and 96% have a B7 estimate, but only 33% enter the text candidates):
- **B7L** = B7 + a text list over seller listings only (one document per product, so a product with no reviews is not outnumbered by 30 review documents).
- **B8** = B7L + KG candidates: every product of the requested type with an estimate on the query aspect joins the candidates, ranked by the estimate.
- **B8-ucb** = B8 ranked by the upper end of the 80% interval (optimistic exploration).
- **B8-slot** = B8 with 2 of the top 10 reserved for new products (fewer than 5 reviews), the best of them by type match and estimate.

NDCG@10 (60 queries):

| system | k=0 | k=1 | k=3 | k=5 | all 30 reviews |
|---|---|---|---|---|---|
| B2 hybrid text | 0.557 | 0.563 | 0.559 | 0.572 | 0.565 |
| B5 current KG | 0.603 | 0.607 | 0.598 | 0.614 | 0.628 |
| B7 relation-predicted KG | 0.615 | 0.616 | 0.608 | 0.624 | 0.644 |
| B7L + listing list | 0.633 | 0.630 | 0.625 | 0.639 | 0.643 |
| **B8 + KG candidates** | **0.643** | **0.638** | **0.637** | **0.650** | **0.647** |
| B8-ucb | 0.630 | 0.624 | 0.621 | 0.636 | 0.642 |
| **B8-slot** | **0.633** | **0.626** | **0.626** | **0.650** | **0.647** |
| B6 LLM re-ranking of B2 | 0.669 | 0.689 | 0.689 | 0.681 | 0.687 |

Share of relevant new products that reach the top 10 (58 queries with at least one):

| system | k=0 | k=1 | k=3 | k=5 | all 30 reviews |
|---|---|---|---|---|---|
| B2 | 0.063 | 0.097 | 0.102 | 0.175 | 0.434 |
| B5 | 0.051 | 0.085 | 0.087 | 0.141 | 0.462 |
| B7 | 0.053 | 0.088 | 0.090 | 0.167 | 0.466 |
| B7L | 0.106 | 0.146 | 0.153 | 0.224 | 0.470 |
| B8 | 0.106 | 0.146 | 0.157 | 0.227 | 0.456 |
| B8-ucb | 0.109 | 0.152 | 0.160 | 0.226 | 0.455 |
| **B8-slot** | **0.342** | **0.339** | **0.324** | 0.227 | 0.456 |
| B6 | 0.025 | 0.105 | 0.119 | 0.192 | 0.444 |

Paired differences (query-level bootstrap, 95% CI):

| comparison | metric | k=0 | k=3 | k=5 | all reviews |
|---|---|---|---|---|---|
| B7 - B2 | NDCG@10 | **+0.058 [+0.038, +0.078]** | **+0.049 [+0.027, +0.072]** | **+0.052 [+0.032, +0.072]** | **+0.079 [+0.058, +0.101]** |
| B8 - B7 | NDCG@10 | **+0.029 [+0.006, +0.053]** | **+0.030 [+0.007, +0.053]** | **+0.026 [+0.002, +0.050]** | +0.003 [-0.022, +0.029] |
| B8 - B7 | new-product hit | **+0.052 [+0.020, +0.092]** | **+0.067 [+0.025, +0.114]** | **+0.060 [+0.011, +0.112]** | -0.010 [-0.061, +0.047] |
| B8 - B6 | NDCG@10 | -0.026 [-0.060, +0.010] | **-0.052 [-0.085, -0.018]** | -0.031 [-0.067, +0.007] | **-0.039 [-0.072, -0.006]** |
| B8-slot - B8 | NDCG@10 | -0.010 [-0.022, +0.001] | -0.011 [-0.024, +0.002] | 0 (no product under 5 reviews) | 0 |
| B8-slot - B8 | new-product hit | **+0.236 [+0.177, +0.300]** | **+0.168 [+0.119, +0.219]** | 0 | 0 |
| B8-slot - B6 | new-product hit | **+0.317 [+0.250, +0.391]** | **+0.205 [+0.150, +0.265]** | +0.035 [-0.019, +0.096] | +0.012 [-0.063, +0.082] |
| B8-ucb - B8 | NDCG@10 | **-0.013 [-0.026, -0.002]** | **-0.016 [-0.029, -0.006]** | **-0.014 [-0.027, -0.002]** | -0.005 [-0.020, +0.006] |

Reading:
1. **The bottleneck was candidate generation, not ranking.** The listing list alone doubles the share of relevant new products in the top 10 (5% -> 11% at k=0); KG candidates add NDCG (+0.03 over B7, significant at k=0, 3, 5) and close most of the gap to the LLM re-ranker (B8 - B6 at k=0: -0.026, CI includes 0).
2. **Reserved slots solve exposure at a small cost.** B8-slot puts 34% of relevant new products into the top 10 at k=0 (B7: 5%, B6: 2.5%) while its NDCG stays above B7 and within 0.01 of B8 (CI includes 0). Once a product has 5 reviews it is no longer "new" and B8-slot equals B8.
3. **Optimistic ranking does not help.** B8-ucb gains no exposure and loses 0.013-0.016 NDCG.
4. B6 remains best on NDCG with all reviews but surfaces new products worst at k=0 (2.5%): it re-ranks review sentences, and a product with no reviews has none.
5. Numbers differ slightly from the first experiment-2 run because the pool grew from 1,770 to 2,100 judged pairs when B7L/B8 were added; comparisons within this table share one pool.

## Exposure policy B9: better choice of which new products fill the slots

B8-slot filled its 2 slots with new products picked by type match and predicted rating; only 58% were relevant (the predicted rating of a new product sits near its family mean). B9 scores each new product of the requested type by what is known before any review: listing relevance to the query (0-1) + w_attr if a seller attribute related to a query aspect is claimed + w_ctx if the seller targets the query's context + w_pred * (prediction - 0.5); products whose score passes a threshold fill up to max_slots places. Grid: 160 settings. Parameters were chosen on the 16 dev queries (views k0, k1, k3): maximize new-product exposure subject to dev NDCG >= B8's. Test queries were used once. The pool grew to 2,645 judged pairs (cost $1.90), so test numbers here differ slightly from the table above.

Chosen weights: w_ctx 0.3, w_attr 0, w_pred 0 (the threshold never bound). Test, 60 queries:

| system | NDCG@10 k=0 | k=3 | relevant new in top 10, k=0 | k=3 | NDCG vs B8, k=0 | NDCG vs B7, k=0 |
|---|---|---|---|---|---|---|
| B7 | 0.614 | 0.607 | 5.3% | 8.4% | **-0.029 [-0.053, -0.005]** | - |
| B8 (no slots) | 0.642 | 0.636 | 9.5% | 14.0% | - | **+0.029** |
| B8-slot (2 slots) | 0.632 | 0.625 | 30.0% | 28.9% | -0.010 [-0.022, +0.001] | +0.019 |
| B9, 1 slot | 0.649 | 0.642 | 19.1% | 20.4% | **+0.007 [+0.000, +0.014]** | **+0.036 [+0.011, +0.060]** |
| **B9, 2 slots** | **0.641** | **0.635** | **35.8%** | **36.3%** | -0.001 [-0.014, +0.010] | **+0.027 [+0.003, +0.052]** |
| B9, 3 slots | 0.630 | 0.627 | 41.9% | 43.2% | -0.012 [-0.027, +0.003] | +0.017 [-0.006, +0.041] |
| B9, 4 slots (dev choice) | 0.617 | 0.610 | 51.2% | 52.5% | **-0.025 [-0.044, -0.006]** | +0.004 [-0.022, +0.031] |

Reading:
1. **Choosing new products by listing relevance and seller targeting beats choosing by predicted rating.** With the same 2 slots, B9 surfaces more relevant new products than B8-slot (35.8% vs 30.0%) and keeps B8's NDCG (-0.001, CI includes 0), where B8-slot lost 0.010.
2. **The exposure-accuracy frontier:** each extra slot adds 6-10 points of exposure. Up to 2 slots NDCG equals B8; 3 slots cost 0.012 (not significant); 4 slots cost 0.025 against B8 (significant) but still match B7, the KG before the exposure work.
3. **The dev-chosen setting (4 slots) did not hold on test**: it met the NDCG constraint on 16 dev queries but lost NDCG on test. With so few dev queries the constraint is noisy; the slot count should be set as a product policy (2 for no accuracy loss, 3-4 to favor new products) rather than tuned.
4. Seller-attribute matching (w_attr) and the predicted rating (w_pred) did not help selection; seller targeting of the query's context did.

## Toward 70% exposure: more slots and an LLM selector

At k=0 a test query has 5.0 relevant new products on average, so 4 slots cap exposure at 79% even with perfect choice; the B9 scorer's picks were relevant 57% of the time. Three selectors were compared at 4-8 slots, with no tuning (test queries, views k0 and k3): **score** (B9's listing relevance + seller targeting), **llm** (claude-sonnet-5 orders the type-matched new candidates, about 17 per query, by reading their listings only; one call per query), and **rrf** (both orders fused). The pool grew to 2,855 judged pairs (cost $1.02), so values differ slightly from the B9 table above; all values here share one pool.

k=0 (NDCG difference against B8 without slots, query bootstrap 95% CI):

| slots | score: new in top 10 | score: NDCG vs B8 | **llm: new in top 10** | **llm: NDCG vs B8** | rrf: new in top 10 | rrf: NDCG vs B8 |
|---|---|---|---|---|---|---|
| 0 (B8) | 8.5% | - | 8.5% | - | 8.5% | - |
| 4 | 42.2% | **-0.024 [-0.043, -0.005]** | **54.7%** | +0.010 [-0.009, +0.029] | 49.8% | -0.003 [-0.023, +0.017] |
| 5 | 47.8% | **-0.045** | **62.6%** | -0.003 [-0.027, +0.020] | 56.0% | -0.021 |
| 6 | 53.0% | **-0.060** | **67.6%** | -0.014 [-0.041, +0.012] | 61.8% | **-0.033** |
| 7 | 57.8% | **-0.080** | **72.3%** | -0.029 [-0.062, +0.006] | 67.5% | **-0.048** |
| 8 | 63.1% | **-0.109** | 75.9% | **-0.062 [-0.097, -0.025]** | 72.9% | **-0.073** |

k=3 gives the same picture (llm: 55.0% / 62.9% / 68.5% / 72.4% / 77.1% for 4-8 slots; NDCG vs B8 +0.008, +0.003, -0.007, -0.025, -0.056).

Reading:
1. **The LLM selector is the better way to pick new products.** At every slot count it surfaces more relevant new products and loses less NDCG than the scorer. At 4 slots it reaches 55% with no NDCG loss (+0.010, CI includes 0), where the scorer reached 42% and lost 0.024.
2. **70% is reachable with 7 slots and the LLM selector** (72.3% at k=0, 72.4% at k=3). The NDCG cost against B8 is -0.029, not significant (CI -0.062 to +0.006), which puts it at about the level of B7, the KG before the exposure work. 6 slots give 68% at -0.014.
3. The LLM selector adds one LLM call per query at search time; B8 and the scorer need none.
4. B9 in `retrieval.py` reads `retrieval.exposure` (`max_slots`, `selector: score|llm`) from the config. **Adopted setting (2026-10-17): 7 slots with the LLM selector.** On one pool (2,855 pairs) at k=0: NDCG@10 0.606 vs B7 0.608 (-0.002 [-0.039, +0.033]); relevant products in the top 10 67.5% (B7 71.0%); top 1 relevant 91.7% (B7 88.3%); relevant new products in the top 10 72.3% (B7 4.7%).

Files: `reports/v4/exposure_curve.json`, `reports/v4/exposure.json`, `reports/v4/coldsearch.json`, `coldsearch_per_query.csv`, `reports/v4/coldstart.json`, `coldstart_cases.csv`, `relations_quality.json`; `data/v4/graphs/predict_params.json`, `relations/relations_summary.json`.
