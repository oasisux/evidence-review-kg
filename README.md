# EviReview-KG

An evidence-tracked knowledge graph built from Amazon Reviews 2023 (All_Beauty), with **CEVA** (Context-preserving Evidence-Verified Augmentation). The project measures whether the KG improves product search and question answering over a strong hybrid text-retrieval baseline.

- Spec: `spec/EviReview-KG_Spec_v3_TermProject_KO.md` (v3.1; takes precedence over the v1.2 PDF)
- Decisions and deviations from the spec: `assumptions.md`

## Setup (macOS arm64, Python ≥ 3.11)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
# put your key in .env  ->  ANTHROPIC_API_KEY=...
.venv/bin/evireview doctor
```

The raw data is pinned to the dataset's Hugging Face commit:

```bash
SHA=2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e
mkdir -p data/raw && cd data/raw
curl -LO https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/$SHA/raw/review_categories/All_Beauty.jsonl
curl -LO https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/$SHA/raw/meta_categories/meta_All_Beauty.jsonl
```

## Run

```bash
# Offline, no API key: hand-written fixture with the rule extractor
.venv/bin/evireview demo --profile fixture --offline
.venv/bin/python -m pytest tests -q

# Full research run (100 products): ingest -> build -> judge(dev) -> augment -> judge -> evaluate
.venv/bin/evireview all --config configs/research.yaml

# Optional, between ingest and build: propose aspect-dictionary changes from dev reviews only
.venv/bin/evireview ingest --config configs/research.yaml
.venv/bin/evireview induce-aspects --config configs/research.yaml   # -> reports/aspect_induction.json

# Dashboard (http://127.0.0.1:8501)
.venv/bin/evireview app --config configs/research.yaml
```

Every LLM response is cached in `.cache/llm/`. Re-running a stage reuses the cache and costs nothing. Each call is logged in `runs/cost_ledger.jsonl`, and the run stops at `budget.max_cost_usd`.

## Pipeline

| Stage | What it does |
|---|---|
| ingest | Streams the raw JSONL and keeps products with ≥30 reviews and metadata. An LLM title classifier assigns each product a family; products are taken in `SHA256(seed‖id)` order until each family quota is full (50 skin moisturizers + 50 hair-care products), with ≤50 reviews each. Splits products 15 dev / 35 test per family. Stores the seller listing (title, features, description) as `meta_doc`. |
| induce-aspects (optional) | EDC on dev reviews only: open aspect extraction → one-sentence definitions → map onto the dictionary or cluster into proposed new aspects. Writes a proposal; `aspects_v2.yaml` is edited by hand. |
| build (G1) | Seller layer: open extraction of seller statements from `meta_doc`, EDC canonicalization, `SELLER_STATES` / `SELLER_TARGETS` edges (source `seller`, never mixed into review opinions). Reviews: Sonnet 5 extracts aspect-opinion claims with verbatim quotes, using the closed aspect set of the product's family plus `other` for opinions no aspect fits. Hard gates check the schema, family, that the exact span exists, and the modality. The semantic verifier (separate call) labels each claim entail / neutral / contradict. |
| judge dev | Opus 5 labels 150 dev candidates. τ = maximum coverage subject to dev precision ≥ 0.85. |
| augment | **C1** finds (product, aspect) gaps that reviews mention but no claim covers, and re-extracts with a schema restricted to that aspect. **C2** applies the same two-stage gate. **C3** builds per-context opinion-distribution nodes (positive/negative counts, contested flag). It also builds the ablations A-budget, A-verify and A-context. |
| judge | Opus 5 (raw text only, blind to systems) drafts 40 queries, grades product relevance (qrels), labels claim support, writes T1 reference claims, and grades answers. |
| evaluate | T1/T2/T3 metrics, paired bootstrap 95% CI, CSV export to `reports/`. |

## Judge check (LLM cross-audit)

`evireview audit --config configs/research.yaml` has `claude-fable-5-1`, a model different from both the extractor and the judge, re-answer a fixed sample of check items: every query, 30 claims, 30 qrels and 10 answer citations (`src/evireview/checklist.py`). It then re-evaluates. Queries it drops are excluded from evaluation. Results are labelled as LLM-audited; there is no human validation.

## Outputs

- `data/graphs/<variant>/{nodes,edges,claims,evidence,aggregates}.parquet`, plus `snapshot.json`
- `data/graphs/seller_attributes.parquet` and `seller_schema.json` (induced seller attribute schema)
- `reports/evaluation.json`, `construction_quality.csv`, `augmentation_audit.csv`, `retrieval_qa_metrics.csv`, `ablations.csv`, `cost_latency.csv`, `error_cases.csv` (UTF-8 with BOM; open in Excel with ID columns as text)
- `annotations/` holds judge labels and `llm_audit.json`. These files are never read by ingest, extraction or indexing (test L01).

## Data use

Amazon Reviews 2023 is provided for research. This repository does not redistribute raw reviews: `data/` is git-ignored, and the download commands above rebuild it. Review opinions are shown as reviewer experiences, not as verified product facts.

## Results

See `reports/RESULTS.md` (written after the full run).

