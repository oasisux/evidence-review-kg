# EviReview-KG results

run_id `20261016T073818Z-f4d35b` · profile `research` · 100 products / 4369 reviews (All_Beauty).

All gold labels come from the LLM judge (claude-opus-5, raw text only). See the judge check section for how far to trust them. Values are measurements from this run, not guarantees.

## Judge check: LLM cross-audit (no human validation)

The check items (all queries, 30 claims, 30 qrels, 10 answers) were answered by a third model (`claude-fable-5-1`), different from the extractor (claude-sonnet-5) and the judge (claude-opus-5), given the same information a human would see. This measures agreement between two LLMs, not correctness against human judgement.


| metric | value | n_samples | evaluation_status |
|---|---|---|---|
| judge_check_claim_accuracy | 0.967 | 30 | COMPLETED |
| judge_check_claim_kappa | 0.889 | 30 | COMPLETED |
| judge_check_qrels_exact | 0.700 | 30 | COMPLETED |
| judge_check_qrels_binary | 0.800 | 30 | COMPLETED |
| judge_check_qrels_weighted_kappa | 0.593 | 30 | COMPLETED |
| check_citation_ok_rate | 1.000 | 10 | COMPLETED |
| check_query_approval_rate | 0.857 | 105 | COMPLETED |

## T1 — KG construction

| metric | method | value | n_samples | ci_low | ci_high |
|---|---|---|---|---|---|
| n_claims | g1 | 8305.000 | – | – | – |
| dangling_edges | g1 | 0.000 | – | – | – |
| valid_provenance_rate | g1 | 1.000 | – | – | – |
| product_aspect_pairs | g1 | 1354.000 | – | – | – |
| contested_groups | g1 | 191.000 | – | – | – |
| n_claims | ceva | 6525.000 | – | – | – |
| dangling_edges | ceva | 0.000 | – | – | – |
| valid_provenance_rate | ceva | 1.000 | – | – | – |
| product_aspect_pairs | ceva | 1311.000 | – | – | – |
| contested_groups | ceva | 128.000 | – | – | – |
| n_claims | budget | 6662.000 | – | – | – |
| dangling_edges | budget | 0.000 | – | – | – |
| valid_provenance_rate | budget | 1.000 | – | – | – |
| product_aspect_pairs | budget | 1301.000 | – | – | – |
| contested_groups | budget | 134.000 | – | – | – |
| n_claims | noverify | 8508.000 | – | – | – |
| dangling_edges | noverify | 0.000 | – | – | – |
| valid_provenance_rate | noverify | 1.000 | – | – | – |
| product_aspect_pairs | noverify | 1384.000 | – | – | – |
| contested_groups | noverify | 192.000 | – | – | – |
| n_claims | nocontext | 6525.000 | – | – | – |
| dangling_edges | nocontext | 0.000 | – | – | – |
| valid_provenance_rate | nocontext | 1.000 | – | – | – |
| product_aspect_pairs | nocontext | 1311.000 | – | – | – |
| contested_groups | nocontext | 128.000 | – | – | – |
| claim_precision_judge | g1 | 0.948 | 250.000 | 0.921 | 0.973 |
| claim_precision_judge | ceva | 0.965 | 201.000 | 0.941 | 0.988 |
| t1_precision_vs_reference | g1 | 0.895 | 57.000 | – | – |
| t1_recall_vs_reference | g1 | 0.708 | 72.000 | – | – |
| t1_f1_vs_reference | g1 | 0.791 | 72.000 | – | – |
| t1_precision_vs_reference | ceva | 0.938 | 48.000 | – | – |
| t1_recall_vs_reference | ceva | 0.625 | 72.000 | – | – |
| t1_f1_vs_reference | ceva | 0.750 | 72.000 | – | – |


### KG opinion-distribution accuracy (per product–aspect, vs judge labels of every mentioning review)


| method | scope | kg_majority_polarity_agreement | kg_opinion_recall | kg_pair_coverage | kg_positive_share_mae |
|---|---|---|---|---|---|
| budget | all | 0.783 | 0.720 | 0.800 | 0.129 |
| budget | in_kg | 0.744 | 0.741 | 1.000 | 0.135 |
| budget | kg_sparse | 1.000 | 0.412 | 0.400 | 0.095 |
| ceva | all | 0.766 | 0.731 | 0.833 | 0.146 |
| ceva | in_kg | 0.744 | 0.749 | 1.000 | 0.134 |
| ceva | kg_sparse | 0.875 | 0.471 | 0.500 | 0.204 |
| g1 | all | 0.830 | 0.780 | 0.850 | 0.124 |
| g1 | in_kg | 0.821 | 0.797 | 1.000 | 0.118 |
| g1 | kg_sparse | 0.875 | 0.529 | 0.550 | 0.148 |
| noverify | all | 0.833 | 0.813 | 0.900 | 0.120 |
| noverify | in_kg | 0.821 | 0.829 | 1.000 | 0.117 |
| noverify | kg_sparse | 0.889 | 0.588 | 0.700 | 0.133 |

## T2 — Augmentation (RQ1: targeting vs. same-budget random re-extraction)

| metric | method | value | n_samples | ci_low | ci_high | evaluation_status |
|---|---|---|---|---|---|---|
| accepted_recovered_claims | ceva | 99.000 | – | – | – | COMPLETED |
| new_product_aspect_pairs | ceva | 25.000 | – | – | – | COMPLETED |
| new_canonical_tuples | ceva | 56.000 | – | – | – | COMPLETED |
| support_enrichment_claims | ceva | 43.000 | – | – | – | COMPLETED |
| augmentation_precision_judge | ceva | 0.827 | 75.000 | 0.733 | 0.914 | COMPLETED |
| estimated_supported_new_claims | ceva | 81.840 | – | – | – | COMPLETED |
| augmentation_precision_new_pairs | ceva | 0.829 | 35.000 | 0.619 | 0.943 | COMPLETED |
| duplicates_removed | ceva | 29.000 | – | – | – | COMPLETED |
| accepted_recovered_claims | budget | 236.000 | – | – | – | COMPLETED |
| new_product_aspect_pairs | budget | 15.000 | – | – | – | COMPLETED |
| new_canonical_tuples | budget | 52.000 | – | – | – | COMPLETED |
| support_enrichment_claims | budget | 184.000 | – | – | – | COMPLETED |
| augmentation_precision_judge | budget | 0.895 | 172.000 | 0.837 | 0.941 | COMPLETED |
| estimated_supported_new_claims | budget | 211.302 | – | – | – | COMPLETED |
| augmentation_precision_new_pairs | budget | 0.867 | 15.000 | 0.600 | 1.000 | COMPLETED |
| duplicates_removed | budget | 915.000 | – | – | – | COMPLETED |
| accepted_recovered_claims | noverify | 203.000 | – | – | – | COMPLETED |
| new_product_aspect_pairs | noverify | 60.000 | – | – | – | COMPLETED |
| new_canonical_tuples | noverify | 120.000 | – | – | – | COMPLETED |
| support_enrichment_claims | noverify | 83.000 | – | – | – | COMPLETED |
| augmentation_precision_judge | noverify | 0.613 | 150.000 | 0.512 | 0.707 | COMPLETED |
| estimated_supported_new_claims | noverify | 124.507 | – | – | – | COMPLETED |
| augmentation_precision_new_pairs | noverify | 0.581 | 74.000 | 0.373 | 0.735 | COMPLETED |
| duplicates_removed | noverify | 29.000 | – | – | – | COMPLETED |
| accepted_recovered_claims | c1_v1 | 115.000 | – | – | – | COMPLETED |
| new_product_aspect_pairs | c1_v1 | 32.000 | – | – | – | COMPLETED |
| new_canonical_tuples | c1_v1 | 64.000 | – | – | – | COMPLETED |
| support_enrichment_claims | c1_v1 | 51.000 | – | – | – | COMPLETED |
| augmentation_precision_judge | c1_v1 | 0.726 | 84.000 | 0.568 | 0.845 | COMPLETED |
| estimated_supported_new_claims | c1_v1 | 83.512 | – | – | – | COMPLETED |
| augmentation_precision_new_pairs | c1_v1 | 0.769 | 39.000 | 0.500 | 0.930 | COMPLETED |
| duplicates_removed | c1_v1 | 28.000 | – | – | – | COMPLETED |
| tau | ceva | 0.600 | – | – | – | COMPLETED |
| llm_review_slots_c1 | ceva | 927.000 | – | – | – | COMPLETED |
| llm_review_slots_budget | budget | 927.000 | – | – | – | COMPLETED |


τ = 0.6 (dev_judge_labels, dev labels n=150). Gaps: 332. LLM review slots: C1 927 vs A-budget 927.


Precision–coverage by τ (test, judge-labelled entailed candidates):


| method | tau | accepted | precision | supported |
|---|---|---|---|---|
| ceva | 0.600 | 93 | 0.849 | 79 |
| ceva | 0.650 | 79 | 0.848 | 67 |
| ceva | 0.700 | 61 | 0.869 | 53 |
| ceva | 0.750 | 45 | 0.867 | 39 |
| ceva | 0.800 | 40 | 0.850 | 34 |
| ceva | 0.850 | 27 | 0.926 | 25 |
| ceva | 0.900 | 20 | 0.950 | 19 |
| ceva | 0.950 | 15 | 0.933 | 14 |
| ceva | 1.000 | 4 | 1.000 | 4 |
| budget | 0.600 | 340 | 0.944 | 321 |
| budget | 0.650 | 292 | 0.945 | 276 |
| budget | 0.700 | 273 | 0.949 | 259 |
| budget | 0.750 | 232 | 0.957 | 222 |
| budget | 0.800 | 195 | 0.964 | 188 |
| budget | 0.850 | 162 | 0.963 | 156 |
| budget | 0.900 | 126 | 0.992 | 125 |
| budget | 0.950 | 73 | 1.000 | 73 |
| budget | 1.000 | 43 | 1.000 | 43 |
| c1_v1 | 0.600 | 99 | 0.758 | 75 |
| c1_v1 | 0.650 | 84 | 0.738 | 62 |
| c1_v1 | 0.700 | 69 | 0.754 | 52 |
| c1_v1 | 0.750 | 59 | 0.712 | 42 |
| c1_v1 | 0.800 | 49 | 0.714 | 35 |
| c1_v1 | 0.850 | 41 | 0.707 | 29 |
| c1_v1 | 0.900 | 31 | 0.645 | 20 |
| c1_v1 | 0.950 | 20 | 0.750 | 15 |
| c1_v1 | 1.000 | 7 | 0.714 | 5 |

## T3 — Usefulness (RQ2)

### Retrieval (test queries, NDCG@10)

| method | all | context_condition | neg_condition | pos_condition |
|---|---|---|---|---|
| A-budget | 0.616 | 0.641 | 0.560 | 0.666 |
| A-context | 0.569 | 0.593 | 0.516 | 0.615 |
| A-verify | 0.622 | 0.632 | 0.569 | 0.668 |
| B1 | 0.453 | 0.480 | 0.399 | 0.501 |
| B2 | 0.560 | 0.580 | 0.508 | 0.605 |
| B3 | 0.581 | 0.609 | 0.529 | 0.627 |
| B4 | 0.623 | 0.634 | 0.571 | 0.668 |
| B5 | 0.617 | 0.641 | 0.560 | 0.667 |
| B6 | 0.697 | 0.712 | 0.668 | 0.722 |


MRR@10:

| method | all | context_condition | neg_condition | pos_condition |
|---|---|---|---|---|
| A-budget | 0.928 | 0.976 | 0.881 | 0.969 |
| A-context | 0.865 | 0.897 | 0.794 | 0.927 |
| A-verify | 0.928 | 0.952 | 0.905 | 0.948 |
| B1 | 0.816 | 0.833 | 0.750 | 0.873 |
| B2 | 0.872 | 0.869 | 0.798 | 0.938 |
| B3 | 0.906 | 0.929 | 0.857 | 0.948 |
| B4 | 0.928 | 0.952 | 0.905 | 0.948 |
| B5 | 0.928 | 0.976 | 0.881 | 0.969 |
| B6 | 0.952 | 0.968 | 0.952 | 0.951 |


### Paired comparisons (ΔNDCG@10, bootstrap 95% CI)


| variant | baseline | query_type | variant_value | baseline_value | delta | ci_low | ci_high | n_queries |
|---|---|---|---|---|---|---|---|---|
| B5 | B2 | all | 0.617 | 0.560 | 0.057 | 0.028 | 0.087 | 45 |
| B5 | B2 | pos_condition | 0.667 | 0.605 | 0.062 | 0.022 | 0.106 | 24 |
| B5 | B2 | neg_condition | 0.560 | 0.508 | 0.052 | 0.011 | 0.095 | 21 |
| B5 | B2 | context_condition | 0.641 | 0.580 | 0.061 | 0.030 | 0.095 | 21 |
| B4 | B2 | all | 0.623 | 0.560 | 0.063 | 0.036 | 0.093 | 45 |
| B4 | B2 | pos_condition | 0.668 | 0.605 | 0.063 | 0.027 | 0.104 | 24 |
| B4 | B2 | neg_condition | 0.571 | 0.508 | 0.062 | 0.023 | 0.109 | 21 |
| B4 | B2 | context_condition | 0.634 | 0.580 | 0.054 | 0.024 | 0.088 | 21 |
| B3 | B2 | all | 0.581 | 0.560 | 0.021 | -0.000 | 0.042 | 45 |
| B3 | B2 | pos_condition | 0.627 | 0.605 | 0.022 | -0.015 | 0.055 | 24 |
| B3 | B2 | neg_condition | 0.529 | 0.508 | 0.020 | -0.002 | 0.041 | 21 |
| B3 | B2 | context_condition | 0.609 | 0.580 | 0.029 | 0.002 | 0.056 | 21 |
| B5 | B3 | all | 0.617 | 0.581 | 0.036 | 0.003 | 0.069 | 45 |
| B5 | B3 | pos_condition | 0.667 | 0.627 | 0.040 | -0.010 | 0.090 | 24 |
| B5 | B3 | neg_condition | 0.560 | 0.529 | 0.031 | -0.012 | 0.076 | 21 |
| B5 | B3 | context_condition | 0.641 | 0.609 | 0.032 | -0.008 | 0.073 | 21 |
| B5 | B4 | all | 0.617 | 0.623 | -0.005 | -0.015 | 0.004 | 45 |
| B5 | B4 | pos_condition | 0.667 | 0.668 | -0.001 | -0.015 | 0.012 | 24 |
| B5 | B4 | neg_condition | 0.560 | 0.571 | -0.010 | -0.024 | 0.002 | 21 |
| B5 | B4 | context_condition | 0.641 | 0.634 | 0.007 | 0.000 | 0.016 | 21 |
| B5 | A-budget | all | 0.617 | 0.616 | 0.001 | -0.002 | 0.003 | 45 |
| B5 | A-budget | pos_condition | 0.667 | 0.666 | 0.001 | -0.003 | 0.005 | 24 |
| B5 | A-budget | neg_condition | 0.560 | 0.560 | 0.000 | -0.002 | 0.002 | 21 |
| B5 | A-budget | context_condition | 0.641 | 0.641 | 0.000 | -0.003 | 0.003 | 21 |
| B5 | A-verify | all | 0.617 | 0.622 | -0.004 | -0.014 | 0.005 | 45 |
| B5 | A-verify | pos_condition | 0.667 | 0.668 | -0.000 | -0.014 | 0.013 | 24 |
| B5 | A-verify | neg_condition | 0.560 | 0.569 | -0.009 | -0.023 | 0.004 | 21 |
| B5 | A-verify | context_condition | 0.641 | 0.632 | 0.009 | 0.002 | 0.018 | 21 |
| B5 | A-context | all | 0.617 | 0.569 | 0.049 | 0.022 | 0.075 | 45 |
| B5 | A-context | pos_condition | 0.667 | 0.615 | 0.052 | 0.021 | 0.086 | 24 |
| B5 | A-context | neg_condition | 0.560 | 0.516 | 0.044 | 0.002 | 0.090 | 21 |
| B5 | A-context | context_condition | 0.641 | 0.593 | 0.049 | 0.017 | 0.081 | 21 |
| B5 | B6 | all | 0.617 | 0.697 | -0.080 | -0.120 | -0.035 | 45 |
| B5 | B6 | pos_condition | 0.667 | 0.722 | -0.055 | -0.116 | 0.010 | 24 |
| B5 | B6 | neg_condition | 0.560 | 0.668 | -0.108 | -0.169 | -0.056 | 21 |
| B5 | B6 | context_condition | 0.641 | 0.712 | -0.071 | -0.150 | 0.000 | 21 |
| B6 | B2 | all | 0.697 | 0.560 | 0.137 | 0.091 | 0.180 | 45 |
| B6 | B2 | pos_condition | 0.722 | 0.605 | 0.117 | 0.062 | 0.168 | 24 |
| B6 | B2 | neg_condition | 0.668 | 0.508 | 0.160 | 0.090 | 0.233 | 21 |
| B6 | B2 | context_condition | 0.712 | 0.580 | 0.131 | 0.057 | 0.208 | 21 |


Retrieval details: {'n_test_queries': 45, 'n_dev_queries': 16, 'n_test_context_queries': 21, 'b6_latency_ms_p50': 26070.0, 'b6_latency_ms_p95': 29663.8, 'b6_cost_usd_per_query': 0.026964044444443717, 'n_test_queries_without_relevant': 0, 'graph_weights': {'B1': 0.0, 'B2': 0.0, 'B3': 1.0, 'B4': 1.0, 'B5': 1.0, 'A-budget': 1.0, 'A-verify': 1.0, 'A-context': 1.0, 'B6': None}, 'query_latency_ms_p50': 37.5901460647583, 'query_latency_ms_p95': 59.24476385116577}


### Answers (test QA queries)


| method | abstention_recall | answer_correctness_0_2 | both_sides_rate | citation_precision | false_abstention_rate |
|---|---|---|---|---|---|
| A-budget | 0.833 | 1.545 | 0.786 | 0.775 | 0.062 |
| A-context | 0.833 | 1.545 | 0.308 | 0.901 | 0.062 |
| A-verify | 0.833 | 1.727 | 0.867 | 0.786 | 0.000 |
| B2 | 0.571 | 1.455 | 0.583 | 0.915 | 0.000 |
| B4 | 0.833 | 1.682 | 0.867 | 0.795 | 0.000 |
| B5 | 0.625 | 1.591 | 0.750 | 0.765 | 0.071 |


## Cost and time


| stage | model | calls | input_tokens | output_tokens | cost | duration_sec |
|---|---|---|---|---|---|---|
| parse_query | claude-sonnet-5 | 198 | 391824 | 7553 | 0.218 | 440.190 |
| ingest_types_v2 | claude-sonnet-5 | 68 | 301286 | 133533 | 1.938 | 1006.640 |
| induce_extract_skin_moisturizer | claude-sonnet-5 | 30 | 48364 | 22194 | 0.319 | 215.200 |
| induce_define_skin_moisturizer | claude-sonnet-5 | 5 | 8497 | 7133 | 0.088 | 66.240 |
| induce_extract_hair_care | claude-sonnet-5 | 30 | 48285 | 23337 | 0.330 | 203.820 |
| induce_define_hair_care | claude-sonnet-5 | 6 | 9134 | 8080 | 0.099 | 71.120 |
| seller_extract | claude-sonnet-5 | 20 | 55089 | 48082 | 0.581 | 368.640 |
| seller_define | claude-sonnet-5 | 2 | 2099 | 2049 | 0.025 | 19.390 |
| extract | claude-sonnet-5 | 547 | 1736195 | 744953 | 8.603 | 5824.950 |
| verify_g1 | claude-sonnet-5 | 554 | 1796349 | 394340 | 5.823 | 3364.880 |
| judge_dev | claude-opus-5 | 10 | 41083 | 7707 | 0.415 | 110.940 |
| extract_c1 | claude-sonnet-5 | 112 | 465695 | 71667 | 1.711 | 912.070 |
| verify_c1 | claude-sonnet-5 | 16 | 74396 | 15836 | 0.256 | 151.380 |
| extract_budget | claude-sonnet-5 | 112 | 364654 | 151380 | 1.826 | 1513.980 |
| verify_budget | claude-sonnet-5 | 96 | 307830 | 69932 | 1.054 | 619.290 |
| extract_c1legacy | claude-sonnet-5 | 112 | 437583 | 48281 | 1.415 | 630.820 |
| verify_c1legacy | claude-sonnet-5 | 20 | 94326 | 22213 | 0.347 | 210.150 |
| judge_queries_dev_pos_condition | claude-opus-5 | 1 | 5678 | 981 | 0.055 | 13.580 |
| judge_queries_dev_neg_condition | claude-opus-5 | 1 | 5678 | 949 | 0.046 | 12.670 |
| judge_queries_dev_compare | claude-opus-5 | 1 | 5675 | 407 | 0.032 | 5.800 |
| judge_queries_dev_disagreement | claude-opus-5 | 1 | 5677 | 253 | 0.028 | 4.770 |
| judge_queries_dev_unanswerable | claude-opus-5 | 1 | 5678 | 242 | 0.028 | 5.520 |
| judge_queries_test_pos_condition | claude-opus-5 | 1 | 9208 | 2258 | 0.096 | 25.370 |
| judge_queries_test_neg_condition | claude-opus-5 | 1 | 9208 | 3377 | 0.124 | 40.290 |
| judge_queries_test_compare | claude-opus-5 | 1 | 9205 | 1253 | 0.071 | 14.240 |
| judge_queries_test_disagreement | claude-opus-5 | 1 | 9207 | 623 | 0.055 | 8.480 |
| judge_queries_test_unanswerable | claude-opus-5 | 1 | 9208 | 651 | 0.056 | 8.800 |
| judge_reference | claude-opus-5 | 9 | 26334 | 8940 | 0.383 | 94.010 |
| judge_claims_g1 | claude-opus-5 | 17 | 65687 | 11958 | 0.617 | 160.520 |
| judge_claims_c1 | claude-opus-5 | 12 | 51194 | 13107 | 0.512 | 167.200 |
| judge_claims_budget | claude-opus-5 | 38 | 116678 | 36852 | 1.277 | 483.300 |
| judge_claims_c1legacy | claude-opus-5 | 15 | 65250 | 17533 | 0.675 | 224.720 |
| judge_qrels | claude-opus-5 | 1025 | 1964625 | 430947 | 17.789 | 6678.550 |
| answer | claude-sonnet-5 | 171 | 204370 | 52541 | 0.934 | 795.010 |
| judge_answers | claude-opus-5 | 174 | 282489 | 48282 | 2.192 | 926.950 |
| rerank_b6 | claude-sonnet-5 | 76 | 324070 | 6776 | 0.716 | 209.420 |
| judge_kgacc | claude-opus-5 | 60 | 145676 | 13314 | 0.793 | 225.810 |
| audit_queries | claude-fable-5-1 | 1 | 7729 | 7616 | 0.459 | 74.260 |
| audit_claims | claude-fable-5-1 | 3 | 10080 | 2445 | 0.233 | 34.280 |
| audit_qrels | claude-fable-5-1 | 3 | 10001 | 387 | 0.124 | 9.920 |
| audit_answers | claude-fable-5-1 | 1 | 4924 | 1904 | 0.144 | 25.610 |


Total LLM cost: $52.49 USD.
