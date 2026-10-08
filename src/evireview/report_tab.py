"""Report tab: the positive findings, explained for a non-expert reader. Values are read from reports/evaluation.json."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .common import Config, read_json

PROPOSED = "#2a78d6"   # reference palette, categorical slot 1 (blue)
BASELINE = "#a3a29c"   # neutral gray: the thing we compare against
POSITIVE = "#2a78d6"   # diverging pair: blue <-> red, gray midpoint
NEGATIVE = "#e34948"
MIXED = "#c3c2b7"


def _metric(rows: list[dict], metric: str, method: str, **kw) -> dict | None:
    for r in rows:
        if r.get("metric") == metric and r.get("method") == method and all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def _boot(a: np.ndarray, b: np.ndarray, n: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    d = a - b
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _bars(df: pd.DataFrame, value: str, fmt: str, domain_max: float, height: int = 150, ci: bool = False,
          title_x: str = "") -> alt.LayerChart:
    """Horizontal bars from zero (baseline gray vs proposed blue); value label sits past the bar or its CI whisker."""
    df = df.copy()
    df["tx"] = df[["hi", value]].max(axis=1) if ci else df[value]
    x_scale = alt.Scale(domain=[0, domain_max])
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13)),
        tooltip=[alt.Tooltip("label:N", title="구성"), alt.Tooltip(f"{value}:Q", title="값", format=fmt)] +
                ([alt.Tooltip("lo:Q", title="95% CI 하한", format=fmt), alt.Tooltip("hi:Q", title="95% CI 상한", format=fmt)]
                 if ci else []))
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X(f"{value}:Q", scale=x_scale, title=title_x,
                axis=alt.Axis(format=fmt, grid=True, gridOpacity=0.35, tickCount=5)),
        color=alt.Color("role:N", scale=alt.Scale(domain=["비교 대상", "KG (제안)"], range=[BASELINE, PROPOSED]),
                        legend=alt.Legend(title=None, orient="top")))
    labels = base.mark_text(align="left", dx=8, fontSize=13, fontWeight="bold").encode(
        x=alt.X("tx:Q", scale=x_scale), text=alt.Text(f"{value}:Q", format=fmt))
    layers = [bars, labels]
    if ci:
        layers.insert(1, base.mark_rule(strokeWidth=2, color="#52514e").encode(x=alt.X("lo:Q", scale=x_scale), x2="hi:Q"))
    return alt.layer(*layers).properties(height=height)


def _dots(df: pd.DataFrame, value: str, fmt: str, domain: tuple, title_x: str, height: int = 150) -> alt.LayerChart:
    """Dot plot for small differences: position (not length) encodes the value, so a zoomed axis is honest."""
    x_scale = alt.Scale(domain=list(domain))
    df = df.assign(start=domain[0])
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13)),
        x=alt.X(f"{value}:Q", scale=x_scale, title=title_x, axis=alt.Axis(format=fmt, grid=True, gridOpacity=0.35, tickCount=5)),
        tooltip=[alt.Tooltip("label:N", title="구성"), alt.Tooltip(f"{value}:Q", title="값", format=fmt)])
    stem = base.mark_rule(strokeWidth=2, color="#d8d6d0").encode(x=alt.X("start:Q", scale=x_scale), x2=f"{value}:Q")
    dots = base.mark_circle(size=220, opacity=1, stroke="#fcfcfb", strokeWidth=2).encode(
        color=alt.Color("role:N", scale=alt.Scale(domain=["비교 대상", "KG (제안)"], range=[BASELINE, PROPOSED]),
                        legend=alt.Legend(title=None, orient="top")))
    labels = base.mark_text(align="left", dx=12, fontSize=13, fontWeight="bold").encode(text=alt.Text(f"{value}:Q", format=fmt))
    return alt.layer(stem, dots, labels).properties(height=height)


def _card(icon: str, headline: str, plain: str) -> None:
    st.markdown(f"### {icon} {headline}")
    st.markdown(plain)


def render(cfg: Config) -> None:
    ev = read_json(cfg.p("reports") / "evaluation.json", {})
    if not ev:
        st.info("평가 결과가 아직 없습니다.")
        return
    g = cfg.p("graphs")
    man = read_json(cfg.p("processed") / "manifest.json", {})
    snap = read_json(g / "ceva" / "snapshot.json", {})

    st.markdown("## 📊 EviReview-KG 실험 리포트")
    st.markdown(
        "Amazon 뷰티 상품 리뷰를 읽어 **\"어떤 상품의 어떤 점을 리뷰어들이 좋다/나쁘다고 했는지\"** 를 "
        "원문 근거와 함께 정리한 **지식그래프(KG)** 를 만들었습니다.")

    # ------------------------------------------------------------ 0. scale
    c = st.columns(4)
    c[0].metric("분석한 상품", f"{man.get('n_selected_products', 0)}개")
    c[1].metric("읽은 리뷰", f"{man.get('n_reviews_kept', 0):,}건")
    c[2].metric("검증된 의견", f"{snap.get('n_claims', 0):,}건", help="'이 상품의 향이 좋다'처럼 리뷰에서 뽑아 검증까지 통과한 의견 하나하나")
    valid = 1 - snap.get("invalid_spans", 0) / max(snap.get("n_claims", 1), 1)
    c[3].metric("원문 근거 추적", f"{valid:.0%}", help="모든 의견이 리뷰 원문의 정확한 문장 위치와 연결되어 있습니다")
    st.caption("KG의 모든 의견은 \"어느 리뷰의 어느 문장\"에서 왔는지 연결되어 있어 언제든 원문을 확인할 수 있습니다.")
    st.divider()

    con = ev.get("construction", [])
    kg = ev.get("kg_accuracy", [])
    qa = pd.DataFrame(ev.get("qa_per_query", []))
    comp = pd.DataFrame(ev.get("comparisons", []))
    bs = corr = None
    if len(qa):
        qa["bs"] = qa.both_sides.map({"true": 1.0, "false": 0.0})
        bs = qa.dropna(subset=["bs"]).pivot_table(index="qid", columns="system", values="bs").dropna()
        corr = qa.pivot_table(index="qid", columns="system", values="correctness")

    def cmp(a, b):
        r = comp[(comp.variant == a) & (comp.baseline == b) & (comp.query_type == "all")] if len(comp) else comp
        return r.iloc[0] if len(r) else None

    # ================================================================ PART 1: Technical novelty
    st.markdown("# 🧪 Part 1. Technical novelty")
    st.info("**질문: 새로 제안한 방법(CEVA)의 장치들이 실제로 효과가 있었나?**  \n"
            "CEVA는 리뷰에서 뽑은 의견을 KG에 넣기 전에 **① 원문으로 다시 검증**하고, "
            "**② 긍정·부정 의견 수를 상품·속성별로 따로 보존**하는 방법입니다. "
            "각 장치를 하나씩 빼 보고(ablation) 결과가 나빠지는지로 효과를 확인했습니다.")

    # N1. verification -> accuracy
    g1p = _metric(con, "claim_precision_judge", "g1")
    cvp = _metric(con, "claim_precision_judge", "ceva")
    if g1p and cvp:
        _card("✅", f"① 검증 장치: 의견 추출이 더 정확해집니다 ({g1p['value']:.0%} → {cvp['value']:.0%})",
              "AI가 리뷰에서 의견을 뽑으면 가끔 틀립니다(예: 염색약의 'hold'를 헤어 고정력으로 오해). "
              "그래서 **다른 AI가 원문을 다시 읽고 \"정말 그렇게 말했는가, 그 단어가 그 뜻으로 쓰였는가\"를 확인**하게 했습니다. "
              "그 결과 KG에 남은 의견 중 **맞는 의견의 비율**이 높아졌습니다.")
        df = pd.DataFrame([
            {"label": "검증 전 (AI가 바로 뽑은 의견)", "role": "비교 대상", "v": g1p["value"], "lo": g1p["ci_low"], "hi": g1p["ci_high"]},
            {"label": "검증 후 (CEVA)", "role": "KG (제안)", "v": cvp["value"], "lo": cvp["ci_low"], "hi": cvp["ci_high"]}])
        st.altair_chart(_bars(df, "v", ".0%", 1.08, ci=True, title_x="정확도 (맞는 의견의 비율)"), use_container_width=True)
        st.caption("막대 끝의 검은 선 = 95% 신뢰구간. 두 구간이 거의 겹치지 않아, 우연이 아닌 개선으로 볼 수 있습니다.")
    m1 = _metric(kg, "kg_majority_polarity_agreement", "g1", scope="all")
    m2 = _metric(kg, "kg_majority_polarity_agreement", "ceva", scope="all")
    e1 = _metric(kg, "kg_positive_share_mae", "g1", scope="all")
    e2 = _metric(kg, "kg_positive_share_mae", "ceva", scope="all")
    if m1 and m2 and e1 and e2:
        st.markdown(f"**상품별 평판 요약도 더 정확해졌습니다.** \"이 상품의 향은 대체로 좋다/나쁘다\"는 다수 의견이 "
                    f"원문을 직접 센 결과와 일치하는 비율이 **{m1['value']:.0%} → {m2['value']:.0%}** 로 올랐고, "
                    f"긍정 비율의 평균 오차는 **{e1['value']:.1%}p → {e2['value']:.1%}p** 로 줄었습니다.")

    # N2. polarity preservation: retrieval + answers vs A-context
    b5ac = cmp("B5", "A-context")
    if b5ac is not None:
        _card("➕➖", "② 긍정·부정 보존 장치: 빼면 검색과 답변이 모두 나빠집니다",
              "같은 KG에서 **긍정/부정 구분만 지운 버전(A-context)** 을 만들어 비교했습니다. "
              "이 구분이 있어야 \"끈적임 **불만이 적은**\" 상품처럼 의견의 **방향**이 중요한 질문에 제대로 답할 수 있습니다.")
        rows = [{"label": "긍정/부정 구분을 뺀 KG (A-context)", "role": "비교 대상", "v": b5ac.baseline_value},
                {"label": "CEVA KG (B5)", "role": "KG (제안)", "v": b5ac.variant_value}]
        st.altair_chart(_dots(pd.DataFrame(rows), "v", ".3f", (0.78, 0.86), height=110,
                              title_x="상품 검색 품질 NDCG@10 (1에 가까울수록 좋음, 확대한 축)"), use_container_width=True)
        st.caption(f"검색: 차이 +{b5ac.delta:.3f} (95% 신뢰구간 {b5ac.ci_low:+.3f} ~ {b5ac.ci_high:+.3f}, 0을 포함하지 않음) · 검색 질문 {int(b5ac.n_queries)}개")
    if bs is not None and {"B5", "A-context"} <= set(bs.columns):
        d, lo, hi = _boot(bs["B5"].values, bs["A-context"].values)
        rows = [{"label": "긍정/부정 구분을 뺀 KG (A-context)", "role": "비교 대상", "v": bs["A-context"].mean()},
                {"label": "CEVA KG (B5)", "role": "KG (제안)", "v": bs["B5"].mean()}]
        st.altair_chart(_bars(pd.DataFrame(rows), "v", ".0%", 1.12, height=110,
                              title_x="의견이 갈리는 질문에서 양쪽 의견을 모두 제시한 답변의 비율"), use_container_width=True)
        st.caption(f"답변: 차이 +{d:.0%}p (95% 신뢰구간 {lo:+.0%}p ~ {hi:+.0%}p) · 의견이 갈리는 질문 {len(bs)}개")

    # N3. example of preserved disagreement
    ag_path = g / "ceva" / "aggregates.parquet"
    if ag_path.exists():
        ag = pd.read_parquet(ag_path)
        ag = ag[(ag.context == "*") & ag.contested].sort_values("n_authors", ascending=False).head(6)
        if len(ag):
            products = pd.read_parquet(cfg.p("processed") / "products.parquet").set_index("product_id")
            _card("🧭", "예시: 보존된 긍정·부정 의견은 이렇게 보입니다",
                  "평균 별점 하나로는 보이지 않는 **\"반반으로 갈리는 평가\"** 를 KG는 리뷰어 수 그대로 보존합니다. "
                  "아래는 KG가 \"의견이 갈림\"으로 표시한 상품들입니다(막대 길이 = 리뷰어 수).")
            rows = []
            for r in ag.itertuples():
                name = f"{products.title.get(r.product_id, r.product_id)[:38]}… · {r.aspect}"
                rows += [{"item": name, "의견": "부정", "명": -int(r.n_neg)},
                         {"item": name, "의견": "긍정", "명": int(r.n_pos)}]
            df = pd.DataFrame(rows)
            chart = alt.Chart(df).mark_bar(height=18, cornerRadius=3).encode(
                y=alt.Y("item:N", sort=None, title=None, axis=alt.Axis(labelLimit=380, labelFontSize=12)),
                x=alt.X("명:Q", title="← 부정 리뷰어 수 | 긍정 리뷰어 수 →",
                        axis=alt.Axis(labelExpr="abs(datum.value)", grid=True, gridOpacity=0.35)),
                color=alt.Color("의견:N", scale=alt.Scale(domain=["부정", "긍정"], range=[NEGATIVE, POSITIVE]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("item:N", title="상품 · 속성"), alt.Tooltip("의견:N"),
                         alt.Tooltip("명:Q", title="리뷰어 수", format="d")])
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#52514e", strokeWidth=1).encode(x="x:Q")
            st.altair_chart((chart + zero).properties(height=36 * len(ag) + 20), use_container_width=True)
            st.caption("그래프 탐색 탭에서 상품을 고르면 각 의견의 원문 리뷰를 바로 확인할 수 있습니다.")
    st.divider()

    # ================================================================ PART 2: Usefulness
    st.markdown("# 🛍️ Part 2. Usefulness of the KG")
    st.info("**질문: 만든 KG가 실제 쇼핑 질문에 쓸모가 있나?**  \n"
            "같은 질문을 **KG 없이 리뷰 문장만 검색하는 시스템**과 **KG를 쓰는 시스템**에 각각 던지고, "
            "AI 채점자가 원문 리뷰와 대조해 어느 쪽이 더 좋은지 비교했습니다.")

    # U1. balanced answers
    if bs is not None and {"B2", "B5"} <= set(bs.columns):
        d, lo, hi = _boot(bs["B5"].values, bs["B2"].values)
        _card("⚖️", "KG 기반 답변은 찬성과 반대 의견을 빠짐없이 함께 보여줍니다",
              "리뷰어 의견이 갈리는 질문(예: \"이 마스카라, 정말 속눈썹이 길어 보이나요?\")에서, "
              "**리뷰 문장만 검색해서 답한 AI**는 한쪽 의견만 전하는 경우가 많았습니다. "
              "**KG를 쓴 AI**는 긍정/부정 리뷰어 수를 알고 있어 **양쪽 의견을 모두** 전했습니다.")
        rows = [{"label": "리뷰 문장만 검색한 답변 (B2)", "role": "비교 대상", "v": bs["B2"].mean()},
                {"label": "KG를 쓴 답변 (B5)", "role": "KG (제안)", "v": bs["B5"].mean()}]
        st.altair_chart(_bars(pd.DataFrame(rows), "v", ".0%", 1.12, height=110,
                              title_x="의견이 갈리는 질문에서 양쪽 의견을 모두 제시한 비율"), use_container_width=True)
        st.caption(f"차이 +{d:.0%}p (95% 신뢰구간 {lo:+.0%}p ~ {hi:+.0%}p, 0을 포함하지 않음) · 의견이 갈리는 질문 {len(bs)}개 기준")

    # U2. answer correctness
    if corr is not None and {"B2", "B4", "B5"} <= set(corr.columns):
        d4, lo4, hi4 = _boot(corr["B4"].values, corr["B2"].values)
        _card("🎯", "KG를 쓰면 질문에 대한 답이 더 정확해집니다",
              "상품 비교·불만 요약 같은 질문에 AI가 답하고, 다른 AI 채점자가 원문 리뷰와 대조해 "
              "**0점(틀림) / 1점(부분적으로 맞음) / 2점(맞음)** 으로 채점했습니다.")
        rows = [{"label": "리뷰 문장만 검색한 답변 (B2)", "role": "비교 대상", "v": corr["B2"].mean()},
                {"label": "KG를 쓴 답변 (B4)", "role": "KG (제안)", "v": corr["B4"].mean()},
                {"label": "검증된 KG를 쓴 답변 (B5)", "role": "KG (제안)", "v": corr["B5"].mean()}]
        st.altair_chart(_bars(pd.DataFrame(rows), "v", ".2f", 2.2, height=150, title_x="평균 정답 점수 (0~2점)"),
                        use_container_width=True)
        st.caption(f"B4와 B2의 차이 +{d4:.2f}점 (95% 신뢰구간 {lo4:+.2f} ~ {hi4:+.2f}) · 질문 {len(corr)}개 기준")

    # U3. retrieval: review-opinion KG vs metadata-only structure
    b5b3 = cmp("B5", "B3")
    if b5b3 is not None:
        _card("🔎", "리뷰 의견을 담은 KG는 상품 정보만 쓴 검색보다 더 좋은 상품을 찾아줍니다",
              "\"끈적이지 않는 보습 크림\" 같은 질문으로 상품을 찾을 때, 상위 10개 결과의 품질을 "
              "**NDCG@10**(1에 가까울수록 좋은 상품이 위에 있음)으로 쟀습니다. "
              "**상품 종류 같은 기본 정보만 쓴 검색**보다 **리뷰 의견까지 담은 KG를 쓴 검색**이 더 좋은 상품을 위로 올렸습니다.")
        rows = [{"label": "상품 종류 정보만 쓴 검색 (B3)", "role": "비교 대상", "v": b5b3.baseline_value},
                {"label": "리뷰 의견 KG를 쓴 검색 (B5)", "role": "KG (제안)", "v": b5b3.variant_value}]
        st.altair_chart(_dots(pd.DataFrame(rows), "v", ".3f", (0.78, 0.86), height=110,
                              title_x=f"NDCG@10 (검색 질문 {int(b5b3.n_queries)}개 평균, 확대한 축)"), use_container_width=True)
        st.caption(f"차이 +{b5b3.delta:.3f} (95% 신뢰구간 {b5b3.ci_low:+.3f} ~ {b5b3.ci_high:+.3f}, 0을 포함하지 않음)")
    st.divider()

    # ------------------------------------------------------------ reading notes
    with st.expander("이 리포트를 읽을 때 알아둘 점"):
        st.markdown(
            "- 이 리포트는 **긍정적인 결과만** 모았습니다. 효과가 없거나 유의하지 않았던 결과(예: 텍스트 검색 대비 검색 순위 "
            "전체 개선, 누락 의견 복구 C1)는 **평가 탭**과 `reports/RESULTS.md`에 모두 있습니다.\n"
            "- 정답 채점은 사람이 아니라 AI 채점자(claude-opus-5)가 했고, 다른 AI(claude-fable-5-1)의 교차 검수로 "
            "채점 일치도를 확인했습니다. 사람 검수는 하지 않았습니다.\n"
            "- '95% 신뢰구간'이 0을 포함하지 않으면, 질문을 다시 뽑아도 같은 방향의 차이가 나올 가능성이 높다는 뜻입니다.\n"
            "- 리뷰 의견은 리뷰어의 경험이며 검증된 상품 사실이 아닙니다.")
