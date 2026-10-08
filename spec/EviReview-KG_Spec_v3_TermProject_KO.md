# EviReview-KG 개발 명세서 v3.1 — 텀 프로젝트 조정판

- 기준 문서: `Amazon_Reviews_2023_KG_Development_Spec_KO_v2.pdf` (본문 표기 v1.2, 2026.10.06)
- 작성일: 2026.10.07 (v3.1: 사람 주석 30분·4주 일정으로 재조정)
- 적용 원칙: **이 문서가 v1.2보다 우선한다.** 이 문서에 없는 계약(Claim/Evidence 스키마, span 검증 규칙, 결정적 ID, provenance 원칙, 리뷰 해석 한계)은 v1.2를 그대로 따른다.

---

## 0. 조정 요약

| 항목 | v1.2 | v3.1 (텀 프로젝트) | 이유 |
|---|---|---|---|
| 기간 | 15–22 개발일 + 별도 주석 | **4주** | 학기 일정 |
| 연구 표본 | 5,000상품 × ≤10리뷰 (≤50k) | **100상품 × ≤30리뷰 (≤3,000)** | 비용·일정, 상품당 밀도 확보 |
| 정답 데이터 | 사람 주석 (수천 건) | **LLM 판정자(judge) 라벨 + 사람 30분 검수** (§7) | 주석 시간 제약 |
| 사람 작업 | 약 17,000+ 판정 | **1인 30분 이내** | |
| LLM 비용 | 예산·규모 모순 | 1회 전체 실행 약 $20–100 (모델별, §3.3) | |
| 그래프 저장 | DuckDB(P0) + Neo4j(P1) | **Parquet + DuckDB + NetworkX** | 100상품은 메모리 그래프로 충분 |
| 서비스 | FastAPI + Streamlit + Docker | **Streamlit 단일 앱** | 범위 축소 |
| 평가 트랙 | T1–T4 | **T1–T3** | |
| Baseline/ablation | 11개 | **7개 구성** (§8) | 신규성·유용성 검증에 필요한 것만 |
| 신규성 | 구성요소 불명확 | **CEVA 3요소** (§2) | 평가 가능한 형태로 |
| 점수식 | s=0.5e+0.3m+0.2d, τ=0.85 | **s=0.6e+0.4m**, d는 집계에만 사용 | v1.2에서 단일 리뷰 주장이 사실상 수용 불가 |

---

## 1. 목표와 연구 질문

**한 문장 목표:** All_Beauty 리뷰에서 근거 추적형 KG를 만들고, **문맥 보존·근거 검증형 확장(CEVA)** 이 텍스트 검색만으로는 어려운 질의(특히 부정·불만 조건, 상품 비교)에 도움이 되는지 보인다.

| 평가 관점 | 연구 질문 | 핵심 증거 |
|---|---|---|
| Construct a new KG | RQ0. 리뷰에서 근거가 붙은 aspect-opinion 주장을 얼마나 정확히 뽑는가? | T1 claim precision/F1, provenance 100% |
| **Technical novelty** | RQ1. 같은 LLM 예산에서, 갭 타겟 재추출 + 2단 검증이 단순 재추출보다 precision–coverage가 좋은가? | CEVA vs A-budget (T2) |
| **Usefulness** | RQ2. 극성·문맥이 붙은 KG가 강한 hybrid 검색(BM25+dense)보다 상품 탐색·비교·답변을 개선하는가? | 질의 유형별 NDCG@10, 답변 인용 정확도 (T3) |

**핵심 가설(Usefulness):** "끈적임 불만이 적은 상품" 같은 질의에서 텍스트 검색은 aspect를 **많이 언급한** 상품 — 종종 불만이 많은 상품 — 을 올린다. 극성별 집계 노드를 가진 KG는 이를 바로잡을 수 있다. 이 유형을 평가 질의에 명시적으로 포함하되, baseline에 유리한 일반 조건 질의도 함께 넣는다.

---

## 2. Technical novelty: CEVA (Context-preserving Evidence-Verified Augmentation)

"LLM으로 triple 추출" 자체는 신규성이 아니다(CAGK, ASKAT 등). 아래 **세 요소의 조합과 효과 측정**을 기여로 하며, 각 요소는 ablation 하나로 분리 검증한다.

### C1. 갭 타겟 재추출 — ablation: A-budget
1. G1 구축 후 각 (상품, aspect)의 **언급 신호** = 해당 상품 리뷰 문장 중 aspect 별칭 매칭 또는 aspect 설명 임베딩과 cos ≥ 0.45인 문장 수.
2. 언급 신호 ≥ 2인데 accepted claim이 0–1개인 (상품, aspect)를 **gap**으로 정의. 상품당 최대 5개.
3. gap의 미연결 문장만 **해당 aspect로 제한한 스키마**로 재추출.
4. A-budget: 같은 토큰 예산으로 **무작위 미연결 문장**을 일반 스키마로 재추출. → 개선이 "연산량"인지 "타겟팅"인지 분리.

### C2. 2단 검증 게이트 — ablation: A-verify
- Hard gate (결정적): 스키마/enum, ID 존재, `quote == text_nfc[start:end]`, 허용 predicate, subject가 해당 상품.
- Semantic gate (LLM verifier, 추출과 **별도 호출**): `entail / neutral / contradict` + `strong / weak`. entail이 아니면 자동 수용 금지.
- A-verify: semantic gate 제거, hard gate만 유지.

### C3. 문맥 분할 의견 분포 노드 — ablation: A-context
- `AspectOpinion` 집계 Claim: 그룹 키 `(subject_id, subject_scope, aspect_id, context_key)`; `n_pos, n_neg, n_neutral, n_mixed, n_authors, disagreement, contested` (정의는 v1.2 §10).
- 상반 의견을 지우지 않고 둘 다 근거로 유지.
- 검색 시 극성을 그래프 점수에 반영 (§9). A-context는 문맥을 `unspecified`로 합치고 극성 대신 언급량만 사용.

**발표용 기여문 (수치만 채움):**
> Amazon Reviews 2023 All_Beauty에서 근거 추적형 aspect-opinion KG와 평가셋을 구축했다. 같은 LLM 예산에서 CEVA는 단순 재추출 대비 precision ___ / 신규 커버리지 ___를 보였고, 부정·불만 조건 질의에서 hybrid 검색 대비 NDCG@10을 ___ (95% CI ___) 변화시켰다.

---

## 3. 데이터 규모

### 3.1 프로파일

| 프로파일 | 범위 | 용도 |
|---|---|---|
| fixture | 직접 작성 15상품 · 60리뷰 (실데이터 아님) | 오프라인 테스트, API 키 불필요 |
| smoke | All_Beauty 10상품 · ≤30리뷰 | end-to-end 확인, LLM 비용 실측 |
| **research** | **All_Beauty 100상품 · ≤30리뷰 (≤3,000리뷰)** | 본 평가 |

**왜 100상품인가 (하한):** 검색 평가가 "상위 10개 안에 좋은 상품이 있는가"이므로, 상품이 50개 이하면 top-10이 전체의 20%를 넘어 무작위로도 점수가 높게 나오고 시스템 간 차이가 드러나지 않는다. 또한 각 aspect 조건을 만족하는 상품이 질의당 몇 개는 있어야 한다. 비용·시간이 더 필요하면 **상품 수가 아니라 상품당 리뷰 수(30→20)** 를 먼저 줄인다.

### 3.2 표본 선택 (v1.2 §4 수정)
1. metadata 스트리밍 스캔 → 유효 `parent_asin` 집합.
2. review 파일 1회 스캔 → `parent_asin`별 리뷰 수.
3. **리뷰 수 ≥ 15인 상품만 적격** (C1·C3가 의미를 갖는 밀도).
4. 적격 상품 중 `SHA256(seed‖parent_asin)` 오름차순 100개 (seed=42).
5. 상품별 `SHA256(seed‖review_id)` 오름차순 최대 30개 리뷰.
6. 상품 split: dev 30 / test 70 (`split_manifest.json`, 모든 트랙 공유).

> 해석 한계: 리뷰 15건 이상 상품만 대상이므로 인기 상품 편향이 있다. 보고서에 명시한다.

### 3.3 LLM 사용량·비용 추정

가정: 리뷰 평균 ~100 tokens, 추출은 리뷰 8건/호출, 검증은 후보 15건/호출, 시스템 프롬프트는 prompt caching. **설계 추정치이며 smoke에서 실측해 갱신한다.**

| 단계 | 입력 tokens | 출력 tokens |
|---|---|---|
| 파이프라인: 추출·검증·C1·A-budget·답변(40질의 × 7구성) | ~4.7M | ~1.3M |
| 판정자: T1 참조 claim, claim 판정, qrels, 답변 판정 | ~6.8M | ~0.3M |
| **합계 (1회 전체 실행)** | **~11.5M** | **~1.6M** |

| 모델 | 단가 (입력/출력, $/MTok) | 1회 실행 | Batch API 사용 시 |
|---|---|---|---|
| Claude Haiku 4.5 | 1 / 5 | ~$20 | ~$10 |
| Claude Sonnet 5 | 2 / 10 | ~$40 | ~$20 |
| Claude Opus 5 | 5 / 25 | ~$100 | ~$50 |

- 추출·검증·판정은 오프라인 작업이므로 **Message Batches API**(50% 할인) 기본. 답변 데모만 일반 호출.
- **판정자는 추출기와 다른 모델을 쓴다** (§7.3). 두 모델 조합의 비용은 위 행을 단계별로 섞어 계산.
- 예산은 1회 실행 비용의 2–3배. `max_cost_usd` 미설정 시 external provider 실행 거부.

---

## 4. 범위: 유지 / 단순화 / 삭제

| 구분 | 항목 |
|---|---|
| **유지 (P0)** | 결정적 표본·ID, Pydantic 스키마(extra=forbid), Claim/Evidence 노드, exact span 검증, 규칙 baseline, fixture 오프라인 실행, 멱등 빌드 |
| **유지 (P1)** | LLM 추출(mock/external adapter), 2단 검증, 별칭 정규화, CEVA C1–C3, hybrid 검색, 그래프 검색, 근거 인용 답변, LLM 판정 평가, 30분 검수 탭, Streamlit 대시보드, CSV |
| **단순화** | Neo4j → NetworkX + DuckDB. FastAPI 생략. checkpoint → 단계별 산출물 캐시 + LLM 응답 캐시. HMAC → salted SHA-256. CSV 4종. 문장 기반 분할(1,500자 초과 리뷰만). calibration → dev threshold sweep |
| **삭제** | Docker, jobs/export API, T4 transfer, KG embedding, 한국어 질의 번역, 이미지, 증분 삭제·재계산, 사람 독립 이중 주석·kappa |

---

## 5. 미정의 항목 기본값

### 5.1 Aspect 사전 (beauty v1 seed)
`texture, scent, scent_longevity, moisturization, absorption, residue(greasy/sticky), irritation, coverage, shade_match, wear_time, hold(hair), lather, packaging, dispenser, size_quantity, value_for_money, ease_of_application, effectiveness`
- 각 aspect에 영문 설명 1문장과 별칭 목록을 `ontology/aspects_v1.yaml`로 관리. 보완은 dev 데이터로만.
- `effectiveness`는 사용자 경험으로만 표시.

### 5.2 context_key (닫힌 집합)
`skin_type={oily,dry,sensitive,combination,acne_prone}`, `hair_type={curly,fine,thick,color_treated}`, `climate={humid,dry,cold,hot}`, `time_of_use={day,night}`, 그 외 `unspecified`. 원문에 명시된 경우만 부여. 형식 `"skin_type=oily"`.

### 5.3 점수와 수용 규칙
- `e`: verifier strong entail=1.0, weak entail=0.7, 그 외=0.
- `m`: 원문 term이 별칭에 있으면 1.0, 아니면 cos(term, aspect 설명)을 [0,1]로 클립.
- **수용:** hard gate ∧ entail ∧ `s = 0.6e + 0.4m ≥ τ`.
- `τ` 기본 0.80. **dev 후보 150개의 판정자 라벨로만** 조정: dev precision ≥ 0.85인 τ 중 coverage 최대. 없으면 실패로 보고하고 자동 수용 끔.
- `d`(독립 작성자 수)는 `AspectOpinion`의 신뢰 표시에만 사용. s는 확률이 아니다.

### 5.4 기타
- 텍스트 중복: NFC·소문자·공백 정규화 후 SHA-256 동일 → 1표.
- 질의 파싱: aspect 별칭 매칭 + 극성 단서 사전(`few complaints, not, without, no, avoid, less`) 규칙 기반. 모든 그래프 구성에 동일 파싱 결과 사용.
- 임베딩: `sentence-transformers/all-MiniLM-L6-v2`, cpu 기본 / mps 선택.

---

## 6. 시스템 구성

| 계층 | 구성 |
|---|---|
| Ingest | Python 3.11+, 스트리밍 JSONL(.gz), HF 파일 직접 다운로드(revision 고정) |
| 저장 | PyArrow Parquet (canonical) + DuckDB |
| 그래프 | NetworkX MultiDiGraph (snapshot Parquet에서 로드) |
| 추출·검증·판정 | Pydantic, Anthropic SDK adapter + mock adapter, Message Batches, 응답 캐시 |
| 검색 | `rank_bm25` + MiniLM dense + RRF + 그래프 점수 (§9) |
| UI | Streamlit (그래프: pyvis 또는 streamlit-agraph) |
| 평가 | pytest, numpy bootstrap, scikit-learn(kappa) |

```
evireview doctor   --config configs/research.yaml
evireview ingest   --config configs/research.yaml
evireview build    --config configs/research.yaml            # G0 + G1
evireview augment  --snapshot <g1_id> --variant ceva|budget|no_verify|no_context
evireview judge    --snapshot <id> --task t1|t2|qrels|answers  # LLM 판정
evireview evaluate --suite all --snapshot <id>
evireview app      --snapshot <g2_id>                          # Streamlit (검수 탭 포함)
evireview demo     --profile fixture --offline
```

---

## 7. 평가 데이터: LLM 판정자 + 사람 30분 검수 (v1.2 §13 대체)

### 7.1 판정자가 만드는 라벨

| 트랙 | 판정자 작업 | 규모 |
|---|---|---|
| T1 construction | (a) 원문 리뷰와 온톨로지만 보고 **참조 claim 목록** 작성 → recall/F1 계산용<br>(b) 시스템 claim별 supported/unsupported 판정 → precision | (a) dev 10 / test 30 리뷰<br>(b) G1 accepted 표본 300 |
| τ 조정 | dev 후보 supported/unsupported | dev 150 |
| T2 augmentation | CEVA·A-budget·A-verify의 **신규 accepted 전수** + rejected 표본 100 판정 | 수백 건 |
| T3 질의 | aspect × 조건 템플릿으로 **질의 초안** 생성 (KG 출력 사용 금지) | 40개 초안 → 사람이 승인 |
| T3 qrels | 질의 × 상품 관련도 0/1/2, **원문 리뷰만 보고** 판정 + 근거 리뷰 ID | 질의당 aspect 언급 상품 전부(약 40), 미언급 상품은 0 |
| T3 답변 | 정답성 0–2, 인용이 문장을 뒷받침하는지, 상반 의견 포함 여부 | 질의 × 구성 |

**상품이 100개뿐이므로 qrels는 pooling 없이 aspect를 언급한 모든 상품을 판정한다.** 이로써 v1.2의 pooled recall 한계가 사라진다.

### 7.2 T3 질의 구성 (40개: dev 8 / test 32)

| 유형 | test | dev | 지표 |
|---|---|---|---|
| 긍정 조건 검색 ("long-lasting scent") | 10 | 2 | NDCG@10, MRR@10 |
| **부정·불만 조건 검색** ("not greasy", "few irritation complaints") | 10 | 2 | NDCG@10, MRR@10 |
| 상품 비교 (2–3개, 같은 aspect) | 6 | 2 | 정답성, 인용 정확도 |
| 의견 불일치·주요 불만 요약 | 3 | 1 | 정답성, 인용 정확도, 상반 의견 포함 |
| 답변 불가 | 3 | 1 | abstention |

### 7.3 판정자 타당성 장치
- **판정자 모델 ≠ 추출기 모델** (자기 선호 편향 방지). 예산상 같은 모델이면 한계로 명시.
- 판정자는 **원문만** 본다. KG, 시스템 이름, 순위, 점수를 입력하지 않는다 (blind).
- 판정 프롬프트·모델 ID·응답을 캐시와 함께 저장하고, 평가 중에 프롬프트를 바꾸지 않는다.
- 판정 라벨 경로는 ingest/index/extraction 입력에서 분리 (L01 유지).

### 7.4 사람 30분 검수 (1인)

Streamlit **"검수" 탭**이 무작위·blind로 문항을 제시하고 타이머를 표시한다. 판정자 라벨은 사람이 답한 뒤에만 비교한다.

| 순서 | 작업 | 문항 | 문항당 | 시간 |
|---|---|---|---|---|
| 1 | 질의 초안 승인/수정/삭제 | 40 | ~10초 | 7분 |
| 2 | claim supported/unsupported (G1 10 · CEVA 10 · A-budget 10) | 30 | ~12초 | 6분 |
| 3 | 질의–상품 관련도 0/1/2 (원문 하이라이트 표시) | 30 | ~20초 | 10분 |
| 4 | 답변 인용 검증 (인용 문장 ↔ 근거) | 10 | ~30초 | 5분 |
| | | | | **약 28분** |

산출:
- **판정자–사람 일치도**: claim accuracy·Cohen's kappa, qrels 정확 일치율과 이진(0 vs ≥1) 일치율.
- 기준: claim accuracy ≥ 0.80, qrels 이진 일치 ≥ 0.80이면 판정자 라벨을 주 지표로 사용. 미달이면 모든 지표에 "판정자 미검증" 표시하고 한계로 보고.
- 검수 문항 수가 적어 일치도 CI가 넓다는 점을 함께 보고한다.

---

## 8. 실험 구성

| ID | 구성 | 검증 대상 |
|---|---|---|
| B1 | BM25 | 어휘 검색 하한 |
| **B2** | BM25 + dense RRF, 동일 답변 LLM·context 예산 | **주 비교군** |
| B4 | G1 + B2 | 리뷰 추출 KG의 가치 |
| **B5** | **G2 = G1 + CEVA** + B2 | **제안 방법** |
| A-budget | G1 + 동일 예산 무작위 재추출 + B2 | C1 타겟팅 효과 (RQ1) |
| A-verify | B5에서 semantic gate 제거 | C2 의미 검증 효과 |
| A-context | B5에서 문맥·극성 제거 | C3 효과 (RQ2 핵심) |

모든 구성은 같은 100상품·리뷰, 같은 encoder·답변 모델, 같은 context 예산, 같은 질의를 사용한다.

---

## 9. 검색·그래프 점수

1. 텍스트: BM25 top-50, dense top-50 → 상품 단위 최대값으로 집계.
2. 그래프: 질의에서 파싱한 (aspect, context)로 `AspectOpinion` 노드를 조회하고 **조건을 충족하는 쪽의 의견 비율**로 점수.
   - "long-lasting scent", "not greasy", "few complaints about irritation"은 모두 해당 aspect의 **긍정 의견 비율이 높고 부정 의견이 적은** 상품을 원한다(residue에 대한 부정 의견 = 끈적임·기름짐 불만).
   - `g = (n_pos + 1) / (n_pos + n_neg + 2) × min(n_authors / 3, 1)`
   - A-context는 `g`를 언급량 `min(n_authors/3, 1)`로 대체.
   - n_authors < 2이면 "근거 부족" 표시, 답변에서 unknown.
   - context 지정 시 해당 그룹만, 없으면 전체 합산(작성자 중복 제거).
3. 융합: 텍스트 두 순위 + 그래프 순위를 RRF(k=60). 그래프 가중치는 dev 질의로만 고정.
4. 답변: 상품당 최대 3 evidence, 전체 최대 8 evidence / 2,000 tokens, 모든 사실 문장에 evidence ID 인용. 근거 부족이면 abstain 또는 조건별 unknown.

---

## 10. 평가 지표와 목표 (연구 목표이며 약속 아님)

| 관점 | 지표 | 목표 |
|---|---|---|
| KG 정확성 (T1) | claim precision(판정), 참조 대비 recall/F1, 극성별 분해 | precision ≥ 0.80 |
| 구조·근거 | accepted claim 유효 evidence 비율, dangling edge, type violation | 100%, 0, 0 |
| **확장 품질 (T2, RQ1)** | 신규 accepted precision, 신규 (상품, aspect) 커버리지, CEVA vs A-budget, τ별 precision–coverage 곡선 | precision ≥ 0.85, 같은 예산에서 A-budget보다 우세 |
| **검색 (T3, RQ2)** | NDCG@10, MRR@10 — 전체 및 **질의 유형별** | B5 − B2 ΔNDCG@10 > 0, 특히 부정·불만 조건 |
| 답변 | 정답성 0–2, citation precision, 상반 의견 포함률, abstention | citation precision ≥ 0.85 |
| 판정자 신뢰도 | 판정자–사람 일치도 (§7.4) | claim acc ≥ 0.80, qrels 이진 ≥ 0.80 |
| 효율 | 호출·tokens·비용, build 시간, 질의 p50/p95 | 기록 |

통계: 질의 단위 paired bootstrap 2,000회로 ΔNDCG 95% CI. test 검색 질의가 20개라 CI가 넓으므로 유의성을 단정하지 않고 효과 크기·CI·유형별 결과·사례 분석을 함께 보고한다. **모든 정답이 LLM 판정 기반임을 결과 표마다 명시한다.**

---

## 11. 대시보드·CSV

Streamlit 탭: **실행 개요 / 그래프 탐색 (G1·G2 색 구분, Claim 클릭 시 원문 하이라이트) / 검색·비교·답변 / 평가 (지표·유형별·오류 사례) / 검수 (30분, §7.4)**.

CSV 4종 (UTF-8 BOM, 표준 quoting, null=빈 값): `construction_quality.csv`, `augmentation_audit.csv`, `retrieval_qa_metrics.csv`, `error_cases.csv`. 공통 열 `run_id, snapshot_id, split, method, label_source(judge|human), evaluation_status`.

수용 테스트: UI01(필터 행 수 = CSV 행 수), UI03(한글·쉼표·줄바꿈 round-trip), UI04(라벨 없으면 NOT_EVALUATED, 값 null), UI07(조회·다운로드 시 LLM 호출 없음).

---

## 12. 일정 (4주)

| 주차 | 구현 | Gate |
|---|---|---|
| W1 | M0–M1: 골격, 스키마, aspect/context 사전, fixture, `demo --offline`, ingest·표본 선택, G0, 규칙 G1 | fixture 계약 테스트 통과, smoke 10상품 end-to-end, 멱등성 |
| W2 | M2–M3: LLM adapter(batch), 추출, 2단 검증, research 100상품 G1, CEVA C1–C3, A-budget/A-verify/A-context | smoke 비용 실측, 근거 없는 accepted 0건 |
| W3 | M4: 검색·그래프 점수·답변, 판정자 파이프라인(질의 초안·qrels·claim·답변), Streamlit 탭 | B2/B4/B5 검색·답변 동작 |
| W4 | M5: **사람 30분 검수**, 평가 러너, bootstrap, CSV, 보고서, 5분 데모 | test 1회 고정 평가, 제출 |

더 줄여야 하면: 상품당 리뷰 30→20, A-verify 생략, 답변 판정 생략. **줄이지 말 것:** 상품 100개, B2 vs B5, A-budget, A-context, 30분 검수.

---

## 13. 완료 판정

- **기능 완료:** 계약 테스트(D01–D03, E01–E03, A01–A03, L01, R01, S01) 통과, research 표본으로 G1/G2 생성, 7개 구성 실행, 판정·검수 완료, 대시보드 조회·CSV 다운로드.
- **연구 결과:** §10 지표를 측정값으로만 보고. 미달이면 미달로 보고하고 원인 분석.

## 14. 착수 전 결정 사항

1. 추출기 모델, 판정자 모델(추출기와 다르게), `max_cost_usd` (§3.3)
2. 마감일 → §12 일정 조정 여부
