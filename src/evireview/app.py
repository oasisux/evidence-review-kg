"""EviReview-KG Streamlit dashboard. Run: evireview app --config configs/research.yaml"""
from __future__ import annotations

import argparse
import html
import json
import sys

import pandas as pd
import streamlit as st

from evireview.common import load_config, read_json
from evireview.pipeline import VARIANT_LABEL, VARIANTS, load_data

st.set_page_config(page_title="EviReview-KG", layout="wide")

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/research.yaml")
args, _ = ap.parse_known_args(sys.argv[1:])
cfg = load_config(args.config)
ANN = cfg.p("annotations")
REP = cfg.p("reports")
G = cfg.p("graphs")


def _stamp(path) -> float:
    """File modification time: part of every cache key, so a new pipeline run invalidates the cache."""
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def data(stamp_key: tuple):
    reviews, products, split = load_data(cfg)
    return reviews, products, split


@st.cache_data
def _claims(variant: str, stamp_key: float) -> pd.DataFrame:
    p = G / variant / "claims.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def _aggs(variant: str, stamp_key: float) -> pd.DataFrame:
    p = G / variant / "aggregates.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def claims(variant: str) -> pd.DataFrame:
    return _claims(variant, _stamp(G / variant / "claims.parquet"))


def aggs(variant: str) -> pd.DataFrame:
    return _aggs(variant, _stamp(G / variant / "aggregates.parquet"))


def review_doc(review_id: str) -> str:
    """Review text for highlighting; never fails the page if a review is missing."""
    try:
        return rv.loc[review_id].doc
    except KeyError:
        return "(원문을 찾을 수 없습니다. 데이터와 그래프가 서로 다른 실행에서 생성되었을 수 있습니다.)"


def title_of(pid: str) -> str:
    return titles.get(pid, pid)


def evaluation() -> dict:
    return read_json(REP / "evaluation.json", {})


def highlight(doc: str, start, end) -> str:
    if start is None or pd.isna(start):
        return html.escape(doc)
    s, e = int(start), int(end)
    return (html.escape(doc[:s]) + "<mark style='background:#ffe08a;color:#000'>" + html.escape(doc[s:e]) +
            "</mark>" + html.escape(doc[e:])).replace("\n", "<br>")


def csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for c in out.columns:
        if out[c].map(lambda x: isinstance(x, (list, dict))).any():
            out[c] = out[c].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    return out.to_csv(index=False).encode("utf-8-sig")


def download_pair(df_all: pd.DataFrame, df_filtered: pd.DataFrame, kind: str, key: str):
    run_id = evaluation().get("run_id", "norun")
    c1, c2 = st.columns(2)
    c1.download_button(f"현재 필터 결과 CSV ({len(df_filtered)}행)", csv_bytes(df_filtered),
                       f"{kind}_{run_id}_filtered.csv", "text/csv", key=f"{key}_f")
    c2.download_button(f"실행 전체 결과 CSV ({len(df_all)}행)", csv_bytes(df_all),
                       f"{kind}_{run_id}_run_all.csv", "text/csv", key=f"{key}_a")


_proc = cfg.p("processed")
reviews, products, split = data((_stamp(_proc / "reviews.parquet"), _stamp(_proc / "products.parquet"),
                                 _stamp(_proc / "split_manifest.json")))
titles = products.set_index("product_id").title
rv = reviews.set_index("review_id")

st.title("EviReview-KG")
if cfg.profile == "fixture":
    st.warning("Fixture profile: 직접 작성한 예시 데이터입니다. 실제 Amazon 리뷰가 아닙니다.")
st.caption("리뷰 의견은 리뷰어의 경험이며 검증된 상품 사실이 아닙니다. 모든 평가 정답은 LLM 판정자(claude-opus-5) 라벨이며, 다른 모델(claude-fable-5-1)의 교차 검수로 신뢰도를 확인했습니다. 사람 검수는 하지 않았습니다.")

tabs = st.tabs(["실행 개요", "그래프 탐색", "검색·비교·답변", "평가", "Report"])

# ====================================================================== overview
with tabs[0]:
    man = read_json(cfg.p("processed") / "manifest.json", {})
    aug = read_json(G / "augment_summary.json", {})
    c = st.columns(3)
    c[0].metric("선택 상품", man.get("n_selected_products"))
    c[1].metric("리뷰", man.get("n_reviews_kept"))
    c[2].metric("적격 상품 (리뷰≥15)", man.get("n_eligible_products"))
    ledger = pd.DataFrame(evaluation().get("cost", []))
    st.subheader("데이터 manifest")
    st.json(man, expanded=False)
    st.subheader("스냅샷")
    snaps = [read_json(G / v / "snapshot.json", {}) | {"variant": VARIANT_LABEL[v]} for v in VARIANTS
             if (G / v / "snapshot.json").exists()]
    if snaps:
        st.dataframe(pd.DataFrame(snaps), hide_index=True, width="stretch")
    if aug:
        st.write(f"τ = **{aug['tau']['tau']}** (source: {aug['tau']['source']}), gaps = {aug['n_gaps']}, "
                 f"C1 review slots = {aug['c1_review_slots']}, A-budget review slots = {aug['budget_review_slots']}")
    if len(ledger):
        st.subheader("비용·시간 (단계별)")
        st.dataframe(ledger, hide_index=True, width="stretch")

# ====================================================================== graph explorer
with tabs[1]:
    c1, c2, c3 = st.columns([3, 1, 2])
    pid = c1.selectbox("상품", products.product_id, format_func=lambda p: f"{title_of(p)[:70]} ({p.split(':')[1]})")
    variant = c2.selectbox("스냅샷", ["g1", "ceva"], format_func=lambda v: VARIANT_LABEL[v], index=1)
    cl = claims(variant)
    if cl.empty:
        st.info("이 스냅샷은 아직 생성되지 않았습니다. 전체 실행의 augment 단계가 끝나면 표시됩니다.")
        cl = pd.DataFrame(columns=["product_id", "aspect", "polarity", "derivation", "claim_id", "quote", "s",
                                   "context", "review_id", "start", "end", "evidence_id", "verify_label",
                                   "verify_strength"])
    pc = cl[cl.product_id == pid]
    asp_sel = c3.multiselect("aspect 필터", sorted(pc.aspect.unique()) if len(pc) else [])
    if asp_sel:
        pc = pc[pc.aspect.isin(asp_sel)]
    ag = aggs(variant)
    ag = ag[(ag.product_id == pid) & (ag.context == "*")] if len(ag) else ag
    if len(ag):
        st.subheader("aspect별 의견 분포 (독립 작성자 기준)")
        show = ag[["aspect", "n_pos", "n_neg", "n_neutral", "n_mixed", "n_authors", "disagreement", "contested"]]
        st.dataframe(show.sort_values("n_authors", ascending=False), hide_index=True, width="stretch")
    try:
        from pyvis.network import Network
        net = Network(height="520px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222")
        net.add_node(pid, label=title_of(pid)[:40], color="#4c78a8", size=30, shape="box")
        for a in pc.aspect.unique()[:40]:
            net.add_node(f"aspect:{a}", label=a, color="#72b7b2", size=18)
        for r in pc.head(200).itertuples():
            color = {"positive": "#54a24b", "negative": "#e45756", "mixed": "#f58518"}.get(r.polarity, "#9d9d9d")
            dash = r.derivation == "recovered"
            net.add_node(r.claim_id, label=f"{r.polarity[:3]}", title=r.quote, color=color, size=8)
            net.add_edge(r.claim_id, pid, title="SUBJECT", dashes=dash)
            net.add_edge(r.claim_id, f"aspect:{r.aspect}", title="OBJECT", dashes=dash)
        st.components.v1.html(net.generate_html(), height=540)
        st.caption("초록=긍정, 빨강=부정, 주황=혼합, 회색=중립. 점선 = CEVA로 복구한 관계(recovered). 노드에 마우스를 올리면 근거 quote.")
    except Exception as e:  # visualization is optional
        st.info(f"그래프 시각화를 표시할 수 없습니다: {e}")
    st.subheader(f"Claim과 근거 ({len(pc)}건)")
    for r in pc.sort_values(["aspect", "polarity"]).head(60).itertuples():
        with st.expander(f"{r.aspect} · {r.polarity} · {r.derivation} · s={r.s:.2f} · context={r.context}"):
            st.markdown(highlight(review_doc(r.review_id), r.start, r.end), unsafe_allow_html=True)
            st.caption(f"{r.claim_id} · {r.evidence_id} · verifier={r.verify_label}/{r.verify_strength}")

# ====================================================================== search / compare / answer
with tabs[2]:
    from evireview.retrieval import SYSTEMS, search
    mode = st.radio("모드", ["검색", "상품 비교", "근거 기반 답변"], horizontal=True)
    if mode == "검색":
        q = st.text_input("질의 (영문)", "moisturizer that is not greasy")
        cols = st.columns(2)
        sa = cols[0].selectbox("시스템 A", list(SYSTEMS), index=1, format_func=lambda s: SYSTEMS[s]["label"])
        sb = cols[1].selectbox("시스템 B", list(SYSTEMS), index=3, format_func=lambda s: SYSTEMS[s]["label"])
        weights = evaluation().get("retrieval_extra", {}).get("graph_weights", {})
        for col, s in zip(cols, (sa, sb)):
            res = search(cfg, s, q, top_k=10, graph_weight=weights.get(s) or 1.0)
            col.caption(f"parsed aspects={res['parsed']['aspects']} context={res['parsed']['context']}")
            for i, r in enumerate(res["results"], 1):
                gtxt = ""
                if r["graph"]:
                    gtxt = f" · 👍{r['graph']['n_pos']} 👎{r['graph']['n_neg']}" + (" · 근거 부족" if r["graph"]["insufficient"] else "")
                col.markdown(f"**{i}.** {r['title'][:70]}{gtxt}")
    elif mode == "상품 비교":
        from evireview.common import aspect_names
        pids = st.multiselect("상품 2–4개", products.product_id, max_selections=4,
                              format_func=lambda p: title_of(p)[:70], default=list(products.product_id[:2]))
        asp = st.multiselect("aspect", aspect_names(), default=["scent", "texture"])
        variant = st.selectbox("스냅샷", ["g1", "ceva"], index=1, format_func=lambda v: VARIANT_LABEL[v], key="cmpv")
        ag = aggs(variant)
        cl = claims(variant)
        for a in asp:
            st.markdown(f"#### {a}")
            cols = st.columns(max(len(pids), 1))
            for col, p in zip(cols, pids):
                row = ag[(ag.product_id == p) & (ag.aspect == a) & (ag.context == "*")]
                col.markdown(f"**{title_of(p)[:50]}**")
                if row.empty:
                    col.write("근거 부족")
                    continue
                r = row.iloc[0]
                col.write(f"👍 {r.n_pos} · 👎 {r.n_neg} · 작성자 {r.n_authors}" + (" · ⚠️ 의견 갈림" if r.contested else ""))
                for x in cl[(cl.product_id == p) & (cl.aspect == a)].head(4).itertuples():
                    col.caption(f"[{x.polarity}] “{x.quote}”")
    else:
        from evireview.answer import answer, translate_ko
        from evireview.llm import LLM
        q = st.text_input("질문 (영문)", "Which of these products has a scent that reviewers like?")
        pids = st.multiselect("대상 상품 (선택)", products.product_id, format_func=lambda p: title_of(p)[:70])
        s = st.selectbox("시스템", ["B2", "B4", "B5", "A-context"], index=2, format_func=lambda s: SYSTEMS[s]["label"])
        st.caption("답변 버튼을 누를 때만 LLM을 호출합니다 (같은 입력은 캐시 재사용). 오프라인 프로필은 템플릿 답변.")
        if st.button("답변 생성"):
            llm = None if cfg.offline else LLM(cfg)
            a = answer(cfg, llm, s, q, pids or None, stage="answer_ui")
            with st.spinner("한국어 번역 중..."):
                ko = translate_ko(cfg, llm, a["answer"], a["reason"])
            st.markdown("#### 💬 답변 (LLM 생성)")
            held = a["abstained"]
            t_en, t_ko = st.tabs(["EN", "KR"])
            with t_en:
                with st.container(border=True):
                    st.markdown(("**[Abstained]** " if held else "") + a["answer"])
                st.caption(f"Reason: {a['reason']}")
            with t_ko:
                with st.container(border=True):
                    st.markdown(("**[답변 보류]** " if held else "") + ko["answer_ko"])
                st.caption(f"근거 요약: {ko['reason_ko']}")
            st.caption(f"latency {a['latency_ms']} ms · 리뷰어 의견이며 검증된 상품 사실이 아닙니다. "
                       "[E1] 등은 아래 근거 리뷰 번호입니다. 평가는 영어 답변 기준입니다.")
            st.markdown("#### 📄 근거 리뷰 (원문: 첫 줄 = 리뷰 제목, 이후 = 본문, 노란색 = 인용 구간)")
            for e in a["evidence"]:
                mark = "✅" if e["label"] in a["citations"] else "·"
                with st.expander(f"{mark} [{e['label']}] {titles.get(e['product_id'], '')[:50]} — {e['quote'][:80]}"):
                    st.markdown(highlight(review_doc(e["review_id"]), e["start"], e["end"]), unsafe_allow_html=True)
                    st.caption(e.get("evidence_id"))

# ====================================================================== evaluation
with tabs[3]:
    ev = evaluation()
    if not ev:
        st.info("평가 결과가 없습니다. `evireview evaluate`를 실행하세요.")
    else:
        st.caption(f"run_id {ev['run_id']} · {ev['created_utc']}")
        sections = {"KG 구축 (T1)": "construction", "증강 (T2)": "augmentation", "검색 (T3)": "retrieval",
                    "답변 (T3)": "qa", "판정자 교차 검수 일치도": "agreement"}
        sec = st.selectbox("결과 종류", list(sections))
        df = pd.DataFrame(ev.get(sections[sec], []))
        if len(df):
            fcols = st.columns(4)
            filters = {}
            for col, name in zip(fcols, ["method", "split", "query_type", "evaluation_status"]):
                if name in df and df[name].notna().any():
                    opts = sorted(df[name].dropna().astype(str).unique())
                    filters[name] = col.multiselect(name, opts)
            fdf = df
            for k, v in filters.items():
                if v:
                    fdf = fdf[fdf[k].astype(str).isin(v)]
            st.caption(f"일치 행 {len(fdf)} / {len(df)} · 필터는 행 선택만 하며 지표를 재계산하지 않습니다. 값 null = NOT_EVALUATED.")
            page = st.selectbox("페이지 크기", [50, 100], key=f"ps_{sec}")
            st.dataframe(fdf.head(page), hide_index=True, width="stretch")
            download_pair(df, fdf, sections[sec], sections[sec])
        if sec == "검색 (T3)" and ev.get("comparisons"):
            st.subheader("주 비교와 ablation (ΔNDCG@10, paired bootstrap 95% CI)")
            comp = pd.DataFrame(ev["comparisons"])
            st.dataframe(comp, hide_index=True, width="stretch")
            download_pair(comp, comp, "ablations", "abl")
            st.json(ev.get("retrieval_extra", {}), expanded=False)
            pq = pd.DataFrame(ev.get("retrieval_per_query", []))
            if len(pq):
                st.subheader("질의별 결과")
                qs = {q["qid"]: q for q in read_json(ANN / "queries.json", [])}
                qid = st.selectbox("질의", pq.qid.unique(), format_func=lambda x: f"[{qs.get(x, {}).get('type')}] {qs.get(x, {}).get('text', x)}")
                qrels = read_json(ANN / "qrels_judge.json", {}).get(qid, {})
                sub = pq[pq.qid == qid][["system", "ndcg@10", "mrr@10", "recall@10", "n_relevant", "top10"]]
                st.dataframe(sub, hide_index=True, width="stretch")
                st.caption("정답(판정자) grade=2 상품: " + ", ".join(title_of(p)[:40] for p, g in qrels.items() if g == 2))
        if sec == "증강 (T2)" and ev.get("tau_curve"):
            st.subheader("τ별 precision–coverage (test, 판정자 라벨)")
            curve = pd.DataFrame(ev["tau_curve"])
            st.line_chart(curve.pivot(index="tau", columns="method", values="precision"))
            st.line_chart(curve.pivot(index="tau", columns="method", values="supported"))
            audit = pd.read_csv(REP / "augmentation_audit.csv", encoding="utf-8-sig") if (REP / "augmentation_audit.csv").exists() else pd.DataFrame()
            if len(audit):
                st.subheader("증강 후보 감사")
                stt = st.multiselect("claim_status", sorted(audit.claim_status.dropna().unique()))
                fa = audit[audit.claim_status.isin(stt)] if stt else audit
                st.dataframe(fa.head(100), hide_index=True, width="stretch")
                download_pair(audit, fa, "augmentation_audit", "aud")
        if sec == "답변 (T3)" and ev.get("qa_per_query"):
            qa = pd.DataFrame(ev["qa_per_query"])
            st.dataframe(qa, hide_index=True, width="stretch")
        errs = REP / "error_cases.csv"
        if errs.exists():
            with st.expander("오류 사례"):
                edf = pd.read_csv(errs, encoding="utf-8-sig")
                st.dataframe(edf.head(100), hide_index=True, width="stretch")
                download_pair(edf, edf, "error_cases", "err")

# ====================================================================== report
with tabs[4]:
    from evireview.report_tab import render as render_report
    render_report(cfg)
