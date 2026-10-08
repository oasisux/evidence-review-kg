# EviReview-KG results

run_id `20261007T090815Z-17a966` · profile `research` · 100 products / 4502 reviews (All_Beauty).

All gold labels come from the LLM judge (claude-opus-5, raw text only). See the judge check section for how far to trust them. Values are measurements from this run, not guarantees.

## Judge check: LLM cross-audit (no human validation)

The check items (all queries, 30 claims, 30 qrels, 10 answers) were answered by a third model (`claude-fable-5-1`), different from the extractor (claude-sonnet-5) and the judge (claude-opus-5), given the same information a human would see. This measures agreement between two LLMs, not correctness against human judgement.


| metric | value | n_samples | evaluation_status |
|---|---|---|---|
| judge_check_claim_accuracy | 0.833 | 30 | COMPLETED |
| judge_check_claim_kappa | 0.561 | 30 | COMPLETED |
| judge_check_qrels_exact | 0.667 | 30 | COMPLETED |
| judge_check_qrels_binary | 0.833 | 30 | COMPLETED |
| judge_check_qrels_weighted_kappa | 0.634 | 30 | COMPLETED |
| check_citation_ok_rate | 0.900 | 10 | COMPLETED |
| check_query_approval_rate | 0.886 | 105 | COMPLETED |

## T1 — KG construction

| metric | method | value | n_samples | ci_low | ci_high |
|---|---|---|---|---|---|
| n_claims | g1 | 7128.000 | – | – | – |
| dangling_edges | g1 | 0.000 | – | – | – |
| valid_provenance_rate | g1 | 1.000 | – | – | – |
| product_aspect_pairs | g1 | 928.000 | – | – | – |
| contested_groups | g1 | 175.000 | – | – | – |
| n_claims | ceva | 4693.000 | – | – | – |
| dangling_edges | ceva | 0.000 | – | – | – |
| valid_provenance_rate | ceva | 1.000 | – | – | – |
| product_aspect_pairs | ceva | 835.000 | – | – | – |
| contested_groups | ceva | 101.000 | – | – | – |
| n_claims | budget | 4852.000 | – | – | – |
| dangling_edges | budget | 0.000 | – | – | – |
| valid_provenance_rate | budget | 1.000 | – | – | – |
| product_aspect_pairs | budget | 820.000 | – | – | – |
| contested_groups | budget | 107.000 | – | – | – |
| n_claims | noverify | 7286.000 | – | – | – |
| dangling_edges | noverify | 0.000 | – | – | – |
| valid_provenance_rate | noverify | 1.000 | – | – | – |
| product_aspect_pairs | noverify | 950.000 | – | – | – |
| contested_groups | noverify | 178.000 | – | – | – |
| n_claims | nocontext | 4693.000 | – | – | – |
| dangling_edges | nocontext | 0.000 | – | – | – |
| valid_provenance_rate | nocontext | 1.000 | – | – | – |
| product_aspect_pairs | nocontext | 835.000 | – | – | – |
| contested_groups | nocontext | 101.000 | – | – | – |
| claim_precision_judge | g1 | 0.852 | 250.000 | 0.794 | 0.902 |
| claim_precision_judge | ceva | 0.930 | 171.000 | 0.887 | 0.966 |
| t1_precision_vs_reference | g1 | 0.854 | 48.000 | – | – |
| t1_recall_vs_reference | g1 | 0.732 | 56.000 | – | – |
| t1_f1_vs_reference | g1 | 0.788 | 56.000 | – | – |
| t1_precision_vs_reference | ceva | 0.833 | 42.000 | – | – |
| t1_recall_vs_reference | ceva | 0.625 | 56.000 | – | – |
| t1_f1_vs_reference | ceva | 0.714 | 56.000 | – | – |


### KG opinion-distribution accuracy (per product–aspect, vs judge labels of every mentioning review)


| method | scope | kg_majority_polarity_agreement | kg_opinion_recall | kg_pair_coverage | kg_positive_share_mae |
|---|---|---|---|---|---|
| budget | all | 0.892 | 0.836 | 0.767 | 0.119 |
| budget | in_kg | 0.882 | 0.891 | 1.000 | 0.118 |
| budget | kg_sparse | 1.000 | 0.200 | 0.300 | 0.125 |
| ceva | all | 0.892 | 0.828 | 0.767 | 0.124 |
| ceva | in_kg | 0.882 | 0.883 | 1.000 | 0.124 |
| ceva | kg_sparse | 1.000 | 0.200 | 0.300 | 0.125 |
| g1 | all | 0.842 | 0.888 | 0.800 | 0.145 |
| g1 | in_kg | 0.824 | 0.939 | 1.000 | 0.142 |
| g1 | kg_sparse | 1.000 | 0.300 | 0.400 | 0.167 |
| noverify | all | 0.846 | 0.900 | 0.817 | 0.138 |
| noverify | in_kg | 0.824 | 0.939 | 1.000 | 0.142 |
| noverify | kg_sparse | 1.000 | 0.450 | 0.450 | 0.111 |

## T2 — Augmentation (RQ1: targeting vs. same-budget random re-extraction)

| metric | method | value | n_samples | ci_low | ci_high | evaluation_status |
|---|---|---|---|---|---|---|
| accepted_recovered_claims | ceva | 64.000 | – | – | – | COMPLETED |
| new_product_aspect_pairs | ceva | 26.000 | – | – | – | COMPLETED |
| new_canonical_tuples | ceva | 35.000 | – | – | – | COMPLETED |
| support_enrichment_claims | ceva | 29.000 | – | – | – | COMPLETED |
| augmentation_precision_judge | ceva | 0.605 | 43.000 | 0.432 | 0.757 | COMPLETED |
| estimated_supported_new_claims | ceva | 38.698 | – | – | – | COMPLETED |
| augmentation_precision_new_pairs | ceva | 0.621 | 29.000 | 0.400 | 0.813 | COMPLETED |
| duplicates_removed | ceva | 14.000 | – | – | – | COMPLETED |
| accepted_recovered_claims | budget | 223.000 | – | – | – | COMPLETED |
| new_product_aspect_pairs | budget | 11.000 | – | – | – | COMPLETED |
| new_canonical_tuples | budget | 30.000 | – | – | – | COMPLETED |
| support_enrichment_claims | budget | 193.000 | – | – | – | COMPLETED |
| augmentation_precision_judge | budget | 0.858 | 148.000 | 0.806 | 0.905 | COMPLETED |
| estimated_supported_new_claims | budget | 191.358 | – | – | – | COMPLETED |
| augmentation_precision_new_pairs | budget | 0.909 | 11.000 | 0.727 | 1.000 | COMPLETED |
| duplicates_removed | budget | 649.000 | – | – | – | COMPLETED |
| accepted_recovered_claims | noverify | 158.000 | – | – | – | COMPLETED |
| new_product_aspect_pairs | noverify | 49.000 | – | – | – | COMPLETED |
| new_canonical_tuples | noverify | 83.000 | – | – | – | COMPLETED |
| support_enrichment_claims | noverify | 75.000 | – | – | – | COMPLETED |
| augmentation_precision_judge | noverify | 0.590 | 105.000 | 0.471 | 0.708 | COMPLETED |
| estimated_supported_new_claims | noverify | 93.295 | – | – | – | COMPLETED |
| augmentation_precision_new_pairs | noverify | 0.576 | 59.000 | 0.400 | 0.732 | COMPLETED |
| duplicates_removed | noverify | 14.000 | – | – | – | COMPLETED |
| accepted_recovered_claims | c1_v1 | 72.000 | – | – | – | COMPLETED |
| new_product_aspect_pairs | c1_v1 | 26.000 | – | – | – | COMPLETED |
| new_canonical_tuples | c1_v1 | 43.000 | – | – | – | COMPLETED |
| support_enrichment_claims | c1_v1 | 29.000 | – | – | – | COMPLETED |
| augmentation_precision_judge | c1_v1 | 0.596 | 47.000 | 0.440 | 0.755 | COMPLETED |
| estimated_supported_new_claims | c1_v1 | 42.894 | – | – | – | COMPLETED |
| augmentation_precision_new_pairs | c1_v1 | 0.519 | 27.000 | 0.300 | 0.714 | COMPLETED |
| duplicates_removed | c1_v1 | 16.000 | – | – | – | COMPLETED |
| tau | ceva | 0.600 | – | – | – | COMPLETED |
| llm_review_slots_c1 | ceva | 839.000 | – | – | – | COMPLETED |
| llm_review_slots_budget | budget | 839.000 | – | – | – | COMPLETED |


τ = 0.6 (dev_judge_labels, dev labels n=150). Gaps: 282. LLM review slots: C1 839 vs A-budget 839.


Precision–coverage by τ (test, judge-labelled entailed candidates):


| method | tau | accepted | precision | supported |
|---|---|---|---|---|
| ceva | 0.600 | 54 | 0.667 | 36 |
| ceva | 0.650 | 39 | 0.692 | 27 |
| ceva | 0.700 | 34 | 0.706 | 24 |
| ceva | 0.750 | 24 | 0.708 | 17 |
| ceva | 0.800 | 19 | 0.789 | 15 |
| ceva | 0.850 | 12 | 0.833 | 10 |
| ceva | 0.900 | 7 | 0.714 | 5 |
| ceva | 0.950 | 2 | 1.000 | 2 |
| ceva | 1.000 | 2 | 1.000 | 2 |
| budget | 0.600 | 288 | 0.913 | 263 |
| budget | 0.650 | 196 | 0.893 | 175 |
| budget | 0.700 | 158 | 0.905 | 143 |
| budget | 0.750 | 130 | 0.923 | 120 |
| budget | 0.800 | 91 | 0.934 | 85 |
| budget | 0.850 | 64 | 0.953 | 61 |
| budget | 0.900 | 51 | 0.961 | 49 |
| budget | 0.950 | 35 | 1.000 | 35 |
| budget | 1.000 | 24 | 1.000 | 24 |
| c1_v1 | 0.600 | 59 | 0.627 | 37 |
| c1_v1 | 0.650 | 45 | 0.667 | 30 |
| c1_v1 | 0.700 | 38 | 0.711 | 27 |
| c1_v1 | 0.750 | 29 | 0.690 | 20 |
| c1_v1 | 0.800 | 23 | 0.696 | 16 |
| c1_v1 | 0.850 | 13 | 0.692 | 9 |
| c1_v1 | 0.900 | 8 | 0.625 | 5 |
| c1_v1 | 0.950 | 2 | 0.500 | 1 |
| c1_v1 | 1.000 | 1 | 1.000 | 1 |

## T3 — Usefulness (RQ2)

### Retrieval (test queries, NDCG@10)

| method | all | neg_condition | pos_condition |
|---|---|---|---|
| A-budget | 0.839 | 0.810 | 0.861 |
| A-context | 0.813 | 0.771 | 0.847 |
| A-verify | 0.832 | 0.792 | 0.863 |
| B1 | 0.709 | 0.655 | 0.751 |
| B2 | 0.826 | 0.800 | 0.847 |
| B3 | 0.802 | 0.768 | 0.828 |
| B4 | 0.835 | 0.800 | 0.863 |
| B5 | 0.840 | 0.813 | 0.862 |
| B6 | 0.869 | 0.870 | 0.869 |


MRR@10:

| method | all | neg_condition | pos_condition |
|---|---|---|---|
| A-budget | 0.917 | 0.889 | 0.938 |
| A-context | 0.887 | 0.845 | 0.920 |
| A-verify | 0.913 | 0.873 | 0.944 |
| B1 | 0.784 | 0.775 | 0.791 |
| B2 | 0.927 | 0.929 | 0.926 |
| B3 | 0.851 | 0.843 | 0.858 |
| B4 | 0.913 | 0.873 | 0.944 |
| B5 | 0.917 | 0.889 | 0.938 |
| B6 | 0.938 | 0.976 | 0.907 |


### Paired comparisons (ΔNDCG@10, bootstrap 95% CI)


| variant | baseline | query_type | variant_value | baseline_value | delta | ci_low | ci_high | n_queries |
|---|---|---|---|---|---|---|---|---|
| B5 | B2 | all | 0.840 | 0.826 | 0.014 | -0.011 | 0.039 | 48 |
| B5 | B2 | pos_condition | 0.862 | 0.847 | 0.015 | -0.014 | 0.047 | 27 |
| B5 | B2 | neg_condition | 0.813 | 0.800 | 0.013 | -0.027 | 0.054 | 21 |
| B4 | B2 | all | 0.835 | 0.826 | 0.009 | -0.020 | 0.035 | 48 |
| B4 | B2 | pos_condition | 0.863 | 0.847 | 0.016 | -0.011 | 0.042 | 27 |
| B4 | B2 | neg_condition | 0.800 | 0.800 | 0.000 | -0.057 | 0.053 | 21 |
| B3 | B2 | all | 0.802 | 0.826 | -0.025 | -0.064 | 0.010 | 48 |
| B3 | B2 | pos_condition | 0.828 | 0.847 | -0.019 | -0.062 | 0.018 | 27 |
| B3 | B2 | neg_condition | 0.768 | 0.800 | -0.032 | -0.109 | 0.025 | 21 |
| B5 | B3 | all | 0.840 | 0.802 | 0.039 | 0.006 | 0.078 | 48 |
| B5 | B3 | pos_condition | 0.862 | 0.828 | 0.034 | 0.003 | 0.067 | 27 |
| B5 | B3 | neg_condition | 0.813 | 0.768 | 0.045 | -0.019 | 0.120 | 21 |
| B5 | B4 | all | 0.840 | 0.835 | 0.005 | -0.009 | 0.026 | 48 |
| B5 | B4 | pos_condition | 0.862 | 0.863 | -0.001 | -0.013 | 0.008 | 27 |
| B5 | B4 | neg_condition | 0.813 | 0.800 | 0.013 | -0.016 | 0.056 | 21 |
| B5 | A-budget | all | 0.840 | 0.839 | 0.002 | -0.001 | 0.005 | 48 |
| B5 | A-budget | pos_condition | 0.862 | 0.861 | 0.001 | -0.001 | 0.004 | 27 |
| B5 | A-budget | neg_condition | 0.813 | 0.810 | 0.003 | -0.000 | 0.009 | 21 |
| B5 | A-verify | all | 0.840 | 0.832 | 0.008 | -0.007 | 0.029 | 48 |
| B5 | A-verify | pos_condition | 0.862 | 0.863 | -0.001 | -0.013 | 0.009 | 27 |
| B5 | A-verify | neg_condition | 0.813 | 0.792 | 0.021 | -0.012 | 0.063 | 21 |
| B5 | A-context | all | 0.840 | 0.813 | 0.027 | 0.005 | 0.058 | 48 |
| B5 | A-context | pos_condition | 0.862 | 0.847 | 0.015 | 0.001 | 0.037 | 27 |
| B5 | A-context | neg_condition | 0.813 | 0.771 | 0.042 | -0.002 | 0.109 | 21 |
| B5 | B6 | all | 0.840 | 0.869 | -0.029 | -0.069 | 0.012 | 48 |
| B5 | B6 | pos_condition | 0.862 | 0.869 | -0.007 | -0.056 | 0.042 | 27 |
| B5 | B6 | neg_condition | 0.813 | 0.870 | -0.057 | -0.118 | 0.004 | 21 |
| B6 | B2 | all | 0.869 | 0.826 | 0.043 | 0.003 | 0.082 | 48 |
| B6 | B2 | pos_condition | 0.869 | 0.847 | 0.022 | -0.032 | 0.076 | 27 |
| B6 | B2 | neg_condition | 0.870 | 0.800 | 0.070 | 0.008 | 0.126 | 21 |


Retrieval details: {'n_test_queries': 50, 'n_dev_queries': 15, 'b6_latency_ms_p50': 23659.5, 'b6_latency_ms_p95': 27550.55, 'b6_cost_usd_per_query': 0.03124627999999916, 'n_test_queries_without_relevant': 2, 'graph_weights': {'B1': 0.0, 'B2': 0.0, 'B3': 1.0, 'B4': 1.0, 'B5': 1.0, 'A-budget': 1.0, 'A-verify': 1.0, 'A-context': 1.0, 'B6': None}, 'query_latency_ms_p50': 35.433053970336914, 'query_latency_ms_p95': 68.34059953689575}


### Answers (test QA queries)


| method | abstention_recall | answer_correctness_0_2 | both_sides_rate | citation_precision | false_abstention_rate |
|---|---|---|---|---|---|
| A-budget | 1.000 | 1.810 | 0.923 | 0.799 | 0.000 |
| A-context | 1.000 | 1.571 | 0.500 | 0.897 | 0.000 |
| A-verify | 1.000 | 1.857 | 1.000 | 0.756 | 0.000 |
| B2 | 1.000 | 1.381 | 0.583 | 0.888 | 0.067 |
| B4 | 1.000 | 1.857 | 1.000 | 0.764 | 0.000 |
| B5 | 1.000 | 1.762 | 1.000 | 0.758 | 0.000 |


## Cost and time


| stage | model | calls | input_tokens | output_tokens | cost | duration_sec |
|---|---|---|---|---|---|---|
| extract | claude-sonnet-5 | 789 | 2105315 | 873767 | 10.150 | 6983.760 |
| verify_g1 | claude-sonnet-5 | 681 | 1814703 | 421786 | 6.705 | 3653.320 |
| judge_dev | claude-opus-5 | 20 | 63141 | 15084 | 0.716 | 228.380 |
| extract_c1 | claude-sonnet-5 | 154 | 529273 | 80459 | 1.920 | 979.300 |
| verify_c1 | claude-sonnet-5 | 22 | 84089 | 15286 | 0.291 | 147.160 |
| extract_budget | claude-sonnet-5 | 154 | 432097 | 175214 | 2.114 | 1388.850 |
| verify_budget | claude-sonnet-5 | 119 | 327807 | 73053 | 1.183 | 643.820 |
| judge_queries_dev | claude-opus-5 | 1 | 4087 | 724 | 0.040 | 12.130 |
| judge_queries_test | claude-opus-5 | 1 | 7531 | 2578 | 0.098 | 34.380 |
| judge_reference | claude-opus-5 | 16 | 40377 | 18913 | 0.715 | 195.670 |
| judge_claims_g1 | claude-opus-5 | 34 | 119461 | 28651 | 1.234 | 377.720 |
| judge_claims_c1 | claude-opus-5 | 15 | 60655 | 18186 | 0.694 | 234.450 |
| judge_claims_budget | claude-opus-5 | 58 | 177175 | 52348 | 2.013 | 705.870 |
| judge_qrels | claude-opus-5 | 1216 | 1735467 | 349898 | 15.354 | 5733.770 |
| answer | claude-sonnet-5 | 291 | 345686 | 88255 | 1.574 | 1260.300 |
| judge_answers | claude-opus-5 | 291 | 452772 | 77883 | 3.516 | 1512.390 |
| audit_queries | claude-fable-5-1 | 2 | 11275 | 9911 | 0.611 | 106.410 |
| audit_claims | claude-fable-5-1 | 9 | 28464 | 7255 | 0.669 | 106.040 |
| audit_qrels | claude-fable-5-1 | 6 | 16000 | 774 | 0.203 | 30.420 |
| audit_answers | claude-fable-5-1 | 3 | 16180 | 6365 | 0.480 | 104.970 |
| parse_query | claude-sonnet-5 | 187 | 265181 | 7265 | 0.175 | 305.350 |
| ingest_types | claude-sonnet-5 | 3 | 12561 | 5015 | 0.075 | 38.060 |
| extract_c1legacy | claude-sonnet-5 | 94 | 319844 | 41742 | 1.068 | 502.790 |
| verify_c1legacy | claude-sonnet-5 | 17 | 70715 | 18407 | 0.283 | 175.040 |
| judge_queries_dev_pos_condition | claude-opus-5 | 1 | 4649 | 654 | 0.041 | 9.630 |
| judge_queries_dev_neg_condition | claude-opus-5 | 1 | 4649 | 853 | 0.040 | 12.200 |
| judge_queries_dev_compare | claude-opus-5 | 1 | 4646 | 471 | 0.030 | 7.940 |
| judge_queries_dev_disagreement | claude-opus-5 | 1 | 4648 | 215 | 0.024 | 4.210 |
| judge_queries_dev_unanswerable | claude-opus-5 | 1 | 4649 | 209 | 0.024 | 4.210 |
| judge_queries_test_pos_condition | claude-opus-5 | 1 | 8502 | 1907 | 0.086 | 23.740 |
| judge_queries_test_neg_condition | claude-opus-5 | 1 | 8502 | 2025 | 0.089 | 25.270 |
| judge_queries_test_compare | claude-opus-5 | 1 | 8499 | 1480 | 0.075 | 22.850 |
| judge_queries_test_disagreement | claude-opus-5 | 1 | 8501 | 560 | 0.052 | 9.270 |
| judge_queries_test_unanswerable | claude-opus-5 | 1 | 8502 | 539 | 0.051 | 10.670 |
| judge_claims_c1legacy | claude-opus-5 | 12 | 47328 | 15828 | 0.581 | 202.260 |
| rerank_b6 | claude-sonnet-5 | 76 | 315013 | 6045 | 0.690 | 157.600 |
| judge_kgacc | claude-opus-5 | 60 | 107266 | 10254 | 0.602 | 249.070 |


Total LLM cost: $54.27 USD.
