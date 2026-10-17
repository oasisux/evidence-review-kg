"""Feasibility pilot for relation-based rating prediction (cold-start products). No LLM calls.

Question: does the mean of related products (same type / same brand / shared seller attribute) predict a product's
hidden ratings better than the global mean? Two views:
- raw: every product in the raw files with >= min_reviews reviews and metadata. Target = share of 4-5 star ratings
  among reviews after the first `hidden_after`, in time order (first 30 reviews). Type = title keywords, brand = store.
- kg: the current KG (ceva claims). Target = per (product, aspect) positive share of reviews after the first k,
  restricted to aspects mentioned >= min_mentions times in the hidden reviews.
Predictions combine the neighbor mean (prior, lambda pseudo-reviews) with the k visible reviews (Beta-Binomial)."""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .common import Config, write_json

TYPE_PATTERNS = [
    ("face_moisturizer", r"face cream|facial (moisturi|cream)|night cream|day cream"), ("eye_cream", r"eye cream"),
    ("serum", r"serum"), ("body_lotion", r"body lotion|hand (lotion|cream)|body butter|body cream"),
    ("shampoo", r"shampoo"), ("conditioner", r"conditioner"), ("hair_oil_mask", r"hair (oil|mask)"),
    ("nail_polish", r"nail polish|gel polish|nail lacquer"), ("lipstick", r"lipstick|lip gloss|lip stain"),
    ("mascara", r"mascara"), ("eyeliner", r"eyeliner|eye liner"), ("foundation", r"foundation|concealer"),
    ("perfume", r"perfume|eau de|cologne"), ("wig", r"\bwig"), ("hair_extension", r"extension"), ("brush", r"brush"),
    ("body_wash", r"body wash|soap|shower gel"), ("lip_balm", r"lip balm"), ("false_lashes", r"lashes"),
]
LAMBDA = 4.0
MIN_NEIGHBORS = 3


def _type_of(title: str) -> str:
    return next((n for n, pat in TYPE_PATTERNS if re.search(pat, title, re.I)), "")


def _loo_mean(keys: pd.Series, values: pd.Series) -> pd.Series:
    """Leave-one-out mean of `values` over rows sharing the key; NaN with fewer than MIN_NEIGHBORS others."""
    ok = keys != ""
    g = values[ok].groupby(keys[ok]).agg(["sum", "count"])
    s = keys.map(g["sum"]) - values
    c = keys.map(g["count"]) - 1
    return (s / c).where(ok & (c >= MIN_NEIGHBORS))


def _posterior(prior: pd.Series, fallback: float, n_pos: pd.Series, n: pd.Series) -> pd.Series:
    return (n_pos + LAMBDA * prior.fillna(fallback)) / (n + LAMBDA)


def _bootstrap_delta(d: np.ndarray, clusters: np.ndarray, seed: int, n: int = 2000) -> tuple[float, float]:
    per = pd.Series(d).groupby(clusters).mean().values
    rng = np.random.default_rng(seed)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _compare(truth: pd.Series, preds: dict[str, pd.Series], avail: dict[str, pd.Series], clusters: pd.Series,
             seed: int) -> list[dict]:
    base = (preds["global"] - truth).abs()
    rows = []
    for name, pred in preds.items():
        err = (pred - truth).abs()
        row = {"method": name, "mae": float(err.mean()), "majority_acc": float(((pred >= .5) == (truth >= .5)).mean()),
               "n": int(len(truth))}
        if name in avail:
            m = avail[name].values
            d = (err - base).values[m]
            lo, hi = _bootstrap_delta(d, clusters.values[m], seed) if m.sum() > 1 else (None, None)
            row.update(available=float(m.mean()), n_available=int(m.sum()),
                       delta_mae_vs_global_where_available=float(d.mean()) if m.sum() else None,
                       rel_delta=float(d.mean() / base.values[m].mean()) if m.sum() else None, ci_low=lo, ci_high=hi)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ raw ratings
def raw_pilot(cfg: Config, min_reviews: int = 20, first_n: int = 30, ks=(0, 1, 3, 5)) -> dict:
    ds = cfg["dataset"]
    seq = defaultdict(list)
    with open(_path(cfg, ds["review_file"]), encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                seq[r["parent_asin"]].append((r["timestamp"], r["rating"]))
            except (json.JSONDecodeError, KeyError):
                continue
    meta = {}
    with open(_path(cfg, ds["meta_file"]), encoding="utf-8") as fh:
        for line in fh:
            try:
                m = json.loads(line)
                meta[m["parent_asin"]] = (m.get("title") or "", (m.get("store") or "").strip().lower())
            except (json.JSONDecodeError, KeyError):
                continue
    rows = []
    for p, rs in seq.items():
        if len(rs) < min_reviews or p not in meta:
            continue
        y = (np.array([r for _, r in sorted(rs)[:first_n]]) >= 4).astype(float)
        rows.append({"p": p, "type": _type_of(meta[p][0]), "brand": meta[p][1], "y": y})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"status": "NOT_EVALUATED", "reason": f"no product with >= {min_reviews} reviews and metadata"}
    hidden_after = max(ks)
    df["truth"] = df.y.map(lambda y: y[hidden_after:].mean())
    df["full"] = df.y.map(np.mean)
    glob = float(df.truth.mean())
    priors = {"type": _loo_mean(df["type"], df.full), "brand": _loo_mean(df.brand, df.full)}
    priors["type+brand"] = pd.concat([priors["type"], priors["brand"]], axis=1).mean(axis=1)
    out = {"n_products": int(len(df)), "n_typed": int((df["type"] != "").sum()), "hidden_after": hidden_after,
           "truth_mean": glob, "type_counts": df["type"].value_counts().to_dict(), "by_k": {}}
    for k in ks:
        n_pos = df.y.map(lambda y: y[:k].sum())
        n = pd.Series(float(k), index=df.index)
        preds = {"self_only": (n_pos / k) if k else pd.Series(.5, index=df.index),
                 "global": _posterior(pd.Series(np.nan, index=df.index), glob, n_pos, n)}
        preds.update({name: _posterior(pr, glob, n_pos, n) for name, pr in priors.items()})
        avail = {name: pr.notna() for name, pr in priors.items()}
        out["by_k"][k] = _compare(df.truth, preds, avail, df.p, cfg["seed"])
    return out


# ------------------------------------------------------------------ current KG aspects
def kg_pilot(cfg: Config, variant: str = "ceva", min_mentions: int = 5, ks=(0, 1, 3)) -> dict:
    products = pd.read_parquet(cfg.p("processed") / "products.parquet").set_index("product_id")
    reviews = pd.read_parquet(cfg.p("processed") / "reviews.parquet")[["review_id", "time_utc"]]
    c = pd.read_parquet(cfg.p("graphs") / variant / "claims.parquet")
    c = c[c.polarity.isin(["positive", "negative"])].merge(reviews, on="review_id")
    # one vote per (review, aspect)
    v = (c.assign(y=(c.polarity == "positive").astype(float))
         .groupby(["product_id", "aspect", "review_id"]).agg(y=("y", "mean"), t=("time_utc", "first")).reset_index())
    v["y"] = (v.y >= .5).astype(float)
    order = v.drop_duplicates(["product_id", "review_id"]).sort_values("t")
    v["pos"] = v.review_id.map(order.groupby("product_id").cumcount().set_axis(order.review_id))
    pa = v.groupby(["product_id", "aspect"]).y.agg(["sum", "count"]).reset_index()
    pa = pa[pa["count"] >= 2]
    pa["share"] = pa["sum"] / pa["count"]

    seller_path = cfg.p("graphs") / "seller_attributes.parquet"
    sattr: dict[str, set] = {}
    if seller_path.exists():
        s = pd.read_parquet(seller_path)
        s = s[s.hard_pass.astype(bool)]
        sattr = {p: set(zip(g.attribute, g.value)) for p, g in s.groupby("product_id")}
    keys = {"family": products.get("family", pd.Series("", index=products.index)).fillna("").to_dict(),
            "type": products.product_type.fillna("").to_dict(),
            "brand": products.brand.fillna("").str.lower().to_dict()}

    out = {"variant": variant, "min_mentions": min_mentions, "by_k": {}}
    for k in ks:
        hid = v[v.pos >= k]
        truth = hid.groupby(["product_id", "aspect"]).y.agg(["mean", "count"])
        truth = truth[truth["count"] >= min_mentions].reset_index()
        vis = v[v.pos < k].groupby(["product_id", "aspect"]).y.agg(["sum", "count"])
        rows = []
        for t in truth.itertuples():
            others = pa[(pa.aspect == t.aspect) & (pa.product_id != t.product_id)]
            row = {"product_id": t.product_id, "aspect": t.aspect, "truth": t.mean, "global": others.share.mean()}
            for name, key in keys.items():
                me = key.get(t.product_id, "")
                m = others[others.product_id.map(lambda q: me != "" and key.get(q) == me)]
                row[name] = m.share.mean() if len(m) >= MIN_NEIGHBORS else np.nan
            mine = sattr.get(t.product_id, set())
            m = others[others.product_id.map(lambda q: bool(mine & sattr.get(q, set())))]
            row["seller_attr"] = m.share.mean() if len(m) >= MIN_NEIGHBORS else np.nan
            n_pos, n = vis.loc[(t.product_id, t.aspect)] if (t.product_id, t.aspect) in vis.index else (0.0, 0.0)
            row.update(n_pos=n_pos, n=n)
            rows.append(row)
        d = pd.DataFrame(rows)
        if d.empty:
            out["by_k"][k] = {"status": "NOT_EVALUATED", "reason": f"no (product, aspect) with >= {min_mentions} hidden mentions"}
            continue
        names =["family", "type", "brand", "seller_attr"]
        preds = {"self_only": (d.n_pos / d.n).where(d.n > 0, .5),
                 "always_positive": pd.Series(1.0, index=d.index),
                 "global": _posterior(d["global"], .5, d.n_pos, d.n)}
        preds.update({nm: _posterior(d[nm], 0, d.n_pos, d.n).where(d[nm].notna(), preds["global"]) for nm in names})
        avail = {nm: d[nm].notna() for nm in names}
        out["by_k"][k] = {"n_targets": int(len(d)), "n_products": int(d.product_id.nunique()),
                          "truth_mean": float(d.truth.mean()), "truth_majority_positive": float((d.truth >= .5).mean()),
                          "methods": _compare(d.truth, preds, avail, d.product_id, cfg["seed"])}
    out["signal"] = _signal(pa[pa["count"] >= min_mentions])
    return out


def _signal(pa: pd.DataFrame) -> dict:
    """Between-product variance per aspect after removing binomial sampling noise (method of moments)."""
    rows = []
    for a, g in pa.groupby("aspect"):
        if len(g) < 8:
            continue
        obs = float(g.share.var())
        noise = float((g.share * (1 - g.share) / g["count"]).mean())
        rows.append({"aspect": a, "n_products": int(len(g)), "mean_share": float(g.share.mean()),
                     "var_observed": obs, "var_noise": noise, "var_between_products": max(obs - noise, 0.0)})
    t = pd.DataFrame(rows)
    if t.empty:
        return {"status": "NOT_EVALUATED", "reason": "fewer than 8 products per aspect"}
    real =float((t.var_between_products * t.n_products).sum() / (t.var_observed * t.n_products).sum()) if len(t) else None
    return {"share_of_variance_between_products": real,
            "noise_mae_floor_at_min_mentions": float(np.sqrt(2 / np.pi) * np.sqrt(t.var_noise.mean())) if len(t) else None,
            "aspects": t.sort_values("n_products", ascending=False).to_dict("records")}


def _path(cfg: Config, rel: str):
    from .common import ROOT
    return ROOT / rel


def run_pilot(cfg: Config) -> dict:
    res = {"lambda": LAMBDA, "min_neighbors": MIN_NEIGHBORS, "kg": kg_pilot(cfg)}
    raw = _path(cfg, cfg["dataset"]["review_file"])
    res["raw"] = raw_pilot(cfg) if raw.exists() else {"status": "NOT_EVALUATED", "reason": f"{raw} missing"}
    write_json(cfg.p("reports") / "pilot_neighbors.json", res)
    return res
