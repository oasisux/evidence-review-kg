"""Stream raw JSONL, select a deterministic product/review sample, write normalized Parquet."""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .common import (ROOT, SCHEMA_VERSION, Config, brand_key, canonical_json, families, family_from_title, nfc,
                     product_types, review_document, seeded_rank, sha, type_family, type_from_title, user_key,
                     write_json)


def _stream(path: Path):
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                yield i, None


def _file_sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _price(v):
    if v is None or v in ("", "None", "—"):
        return None, None
    try:
        return float(str(v).replace("$", "").replace(",", "")), None
    except ValueError:
        return None, "parse_error"


def meta_document(m: dict) -> str:
    """The single seller-text field all seller-attribute offsets refer to: title, then features, then description,
    one entry per line (NFC)."""
    parts = [nfc(m.get("title")).strip()]
    for key in ("features", "description"):
        val = m.get(key) or []
        parts += [nfc(x).strip() for x in (val if isinstance(val, list) else [val]) if nfc(x).strip()]
    return "\n".join(parts)


def _details(m: dict) -> dict | None:
    details = m.get("details")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {"raw": details}
    return details


def review_uid(r: dict) -> str:
    return "review:" + sha(canonical_json([r.get("user_id"), r.get("parent_asin"), r.get("asin"),
                                           r.get("timestamp"), r.get("title"), r.get("text")]))[:32]


TYPE_SYSTEM = """Classify Amazon product titles from the All_Beauty category.
beauty=true only for beauty or personal-care products used on skin, hair, nails, face, lips, eyes or body (skincare, makeup, hair care, fragrance, nail care, lashes, shaving, bath and body, beauty tools such as brushes or curling irons).
beauty=false for anything else (coins, kitchenware, clothing, supplements, books, cleaning products for objects, pet items, etc.).
product_type: 1-3 lowercase words naming what it is (e.g. "body lotion", "shampoo", "false eyelashes", "lipstick").
family: one of the product families below, or "other" when none fits (kits and bundles mixing families are "other").
""" + "\n".join(f"- {f}: {spec['description']}" for f, spec in families().items())

TYPE_SYSTEM_V4 = TYPE_SYSTEM + """
type: the canonical type within the family, one of the names below, or "other" (also "other" when family is "other"):
""" + "\n".join(f"- {t} ({type_family(t)}): {spec['description']}" for t, spec in product_types().items())


def _quota_full(selected: dict[str, list], quotas: dict[str, int] | None, k: int) -> bool:
    if quotas:
        return all(len(selected.get(f, [])) >= n for f, n in quotas.items())
    return len(selected.get("*", [])) >= k


def _take(selected: dict[str, list], quotas: dict[str, int] | None, k: int, p: str, family: str | None) -> None:
    """Add p to its family's slot (quota sampling) or to the single pool (no quotas)."""
    if quotas:
        if family in quotas and len(selected.setdefault(family, [])) < quotas[family]:
            selected[family].append(p)
    elif len(selected.setdefault("*", [])) < k:
        selected["*"].append(p)


def _beauty_filter(cfg: Config, ranked: list[str], meta: dict, k: int,
                   quotas: dict[str, int] | None, classify_all: bool = False) -> tuple[dict[str, list], dict]:
    """Walk products in seeded order, classify titles in batches (beauty, product type, family) and fill the
    family quotas, or keep the first k beauty products when no quotas are configured. classify_all (brand-stratified
    sampling) classifies every eligible title, with a canonical type, and leaves the selection to the caller."""
    from .llm import LLM
    llm = LLM(cfg)
    props = {"id": {"type": "string"}, "beauty": {"type": "boolean"}, "product_type": {"type": "string"},
             "family": {"type": "string", "enum": list(families()) + ["other"]}}
    if classify_all:
        props["type"] = {"type": "string", "enum": list(product_types()) + ["other"]}
    schema = {"type": "object", "additionalProperties": False, "required": ["items"],
              "properties": {"items": {"type": "array", "items": {
                  "type": "object", "additionalProperties": False, "required": list(props), "properties": props}}}}
    batches = [ranked[i:i + 50] for i in range(0, len(ranked), 50)]

    def classify(batch):
        user = "\n".join(f"- id={p} title=\"{nfc(meta[p].get('title'))[:150]}\"" for p in batch)
        if classify_all:
            return llm.json_call(stage="ingest_types_v4", model=cfg["extraction"]["model"], effort="low",
                                 system=TYPE_SYSTEM_V4, user=user, schema=schema)
        return llm.json_call(stage="ingest_types_v2", model=cfg["extraction"]["model"], effort="low",
                             system=TYPE_SYSTEM, user=user, schema=schema)

    selected, types = {}, {}
    results = llm.map(classify, batches, desc="ingest_types") if classify_all else []
    for bi, batch in enumerate(batches):
        if not classify_all and _quota_full(selected, quotas, k):
            break
        res = results[bi] if classify_all else classify(batch)
        got = {x["id"]: x for x in (res or {}).get("items", [])}
        for p in batch:  # keep seeded order; unlabeled titles are skipped
            if p in got:
                fam = got[p]["family"] if got[p]["family"] in families() else None
                t = got[p].get("type")
                types[p] = {"beauty": got[p]["beauty"], "product_type": got[p]["product_type"].lower().strip(),
                            "family": fam, "type": t if fam and type_family(t) == fam else None}
                if got[p]["beauty"] and not classify_all:
                    _take(selected, quotas, k, p, fam)
    return selected, types


def _brand_select(ranked: list[str], types: dict, brands: dict, quotas: dict[str, int], ds: dict) -> dict[str, list]:
    """Brand-stratified quota sampling: within each family, brands with >= brand_min_products candidates are taken
    first (seeded brand order, at most brand_max_products each, until brand_share of the quota), then the quota is
    filled in seeded product order. Same-brand neighbors are rare in a plain random sample."""
    lo, hi, share = ds.get("brand_min_products", 3), ds.get("brand_max_products", 8), ds.get("brand_share", 0.5)
    selected = {}
    for fam, n in quotas.items():
        cands = [p for p in ranked if p in types and types[p]["beauty"] and types[p]["family"] == fam]
        by_brand: dict[str, list] = {}
        for p in cands:
            if brands.get(p):
                by_brand.setdefault(brands[p], []).append(p)
        take: list[str] = []
        for b in sorted((b for b, ps in by_brand.items() if len(ps) >= lo), key=lambda b: seeded_rank(0, b)):
            room = int(n * share) - len(take)
            if room < lo:
                break
            take += by_brand[b][:min(hi, room)]
        chosen = set(take)
        take += [p for p in cands if p not in chosen][: n - len(take)]
        selected[fam] = take[:n]
    return selected


def _keyword_select(ranked: list[str], meta: dict, k: int, quotas: dict[str, int] | None) -> tuple[dict[str, list], dict]:
    """Offline selection: families from title keywords, no beauty check."""
    selected, types = {}, {}
    for p in ranked:
        if _quota_full(selected, quotas, k):
            break
        fam = family_from_title(meta[p].get("title"))
        types[p] = {"beauty": True, "product_type": None, "family": fam}
        _take(selected, quotas, k, p, fam)
    return selected, types


def run_ingest(cfg: Config) -> dict:
    t0 = time.time()
    ds = cfg["dataset"]
    seed = cfg["seed"]
    review_path, meta_path = ROOT / ds["review_file"], ROOT / ds["meta_file"]
    out = cfg.p("processed")
    quarantine = []

    # pass 1: metadata
    meta = {}
    for line_no, m in _stream(meta_path):
        if not m or not m.get("parent_asin"):
            quarantine.append({"file": meta_path.name, "line": line_no, "reason": "bad_json_or_missing_parent_asin"})
            continue
        meta[m["parent_asin"]] = m

    # pass 2: review counts per parent_asin
    counts = Counter()
    for line_no, r in _stream(review_path):
        if not r or not r.get("parent_asin"):
            quarantine.append({"file": review_path.name, "line": line_no, "reason": "bad_json_or_missing_parent_asin"})
            continue
        counts[r["parent_asin"]] += 1

    eligible = [p for p, c in counts.items() if c >= ds["min_reviews_per_product"] and p in meta]
    ranked_eligible = sorted(eligible, key=lambda p: seeded_rank(seed, p))
    quotas = ds.get("family_quotas") or None
    if quotas and set(quotas) - set(families()):
        raise ValueError(f"family_quotas names unknown families: {sorted(set(quotas) - set(families()))}")
    brand_strata = ds.get("sampling") == "family_brand"
    brands = {p: brand_key(_details(meta[p]), meta[p].get("store")) for p in ranked_eligible}
    if brand_strata and not quotas:
        raise ValueError("sampling: family_brand needs family_quotas")
    if ds.get("beauty_filter") and not cfg.offline:
        by_family, types = _beauty_filter(cfg, ranked_eligible, meta, ds["max_products"], quotas,
                                          classify_all=brand_strata)
    else:
        by_family, types = _keyword_select(ranked_eligible, meta, 10**9 if brand_strata else ds["max_products"],
                                           None if brand_strata else quotas)
        for p, t in types.items():
            t["type"] = type_from_title(meta[p].get("title"), t["family"])
    if brand_strata:
        by_family = _brand_select(ranked_eligible, types, brands, quotas, ds)
    if quotas and not _quota_full(by_family, quotas, 0):
        short = {f: f"{len(by_family.get(f, []))}/{n}" for f, n in quotas.items()}
        raise ValueError(f"not enough eligible products to fill family quotas: {short}")
    rank_of = {p: i for i, p in enumerate(ranked_eligible)}
    selected = sorted((p for ps in by_family.values() for p in ps), key=rank_of.get)
    sel = set(selected)

    # pass 3: reviews of selected products
    by_product: dict[str, dict[str, dict]] = {p: {} for p in selected}
    for line_no, r in _stream(review_path):
        if not r or r.get("parent_asin") not in sel:
            continue
        rid = review_uid(r)
        entry = by_product[r["parent_asin"]].setdefault(rid, {"raw": r, "locators": []})
        entry["locators"].append(f"{review_path.name}:{line_no}")

    reviews = []
    for p in selected:
        if ds.get("review_order") == "time":  # earliest reviews first: how reviews accumulate on a new product
            ranked = sorted(by_product[p].items(), key=lambda kv: (kv[1]["raw"].get("timestamp") or 0, kv[0]))
        else:
            ranked = sorted(by_product[p].items(), key=lambda kv: seeded_rank(seed, kv[0]))
        ranked = ranked[: ds["reviews_per_product"]]
        for rid, entry in ranked:
            r = entry["raw"]
            ts = r.get("timestamp")
            unit = "ms" if ts and ts > 1e11 else "s"
            when = datetime.fromtimestamp(ts / (1000 if unit == "ms" else 1), tz=timezone.utc) if ts else None
            reviews.append({
                "review_id": rid,
                "product_id": f"product:{p}",
                "parent_asin": p,
                "asin": r.get("asin"),
                "user_key": user_key(r.get("user_id")),
                "rating": r.get("rating"),
                "title": nfc(r.get("title")),
                "text": nfc(r.get("text")),
                "doc": review_document(r.get("title"), r.get("text")),
                "timestamp_raw": ts,
                "timestamp_unit": unit,
                "time_utc": when.isoformat() if when else None,
                "helpful_vote": r.get("helpful_vote"),
                "verified_purchase": r.get("verified_purchase"),
                "source_locators": json.dumps(entry["locators"]),
                "schema_version": SCHEMA_VERSION,
            })

    products = []
    for p in selected:
        m = meta[p]
        price, perr = _price(m.get("price"))
        details = _details(m)
        products.append({
            "product_id": f"product:{p}",
            "parent_asin": p,
            "title": nfc(m.get("title")),
            "store": nfc(m.get("store")) or None,
            "categories": json.dumps(m.get("categories") or []),
            "features": json.dumps(m.get("features") or [], ensure_ascii=False),
            "description": json.dumps(m.get("description") or [], ensure_ascii=False),
            "details": json.dumps(details or {}, ensure_ascii=False),
            "brand": (details or {}).get("Brand") if isinstance(details, dict) else None,
            "price": price,
            "price_error": perr,
            "average_rating_at_collection": m.get("average_rating"),
            "rating_number_at_collection": m.get("rating_number"),
            "n_reviews_total": counts[p],
            "n_reviews_kept": sum(1 for r in reviews if r["parent_asin"] == p),
            "product_type": types.get(p, {}).get("product_type"),
            "family": types.get(p, {}).get("family"),
            "type": types.get(p, {}).get("type"),
            "brand_key": brands.get(p),
            "meta_doc": meta_document(m),
            "missing_metadata": False,
            "schema_version": SCHEMA_VERSION,
        })

    # product-level split shared by all tracks; with quotas, dev is drawn per family in proportion to its quota
    if quotas:
        total = sum(quotas.values())
        dev = set()
        for f, ps in by_family.items():
            n_dev = round(ds["dev_products"] * quotas[f] / total)
            dev |= set(sorted(ps, key=lambda p: seeded_rank(seed + 1, p))[:n_dev])
    else:
        ordered = sorted(selected, key=lambda p: seeded_rank(seed + 1, p))
        dev = set(ordered[: ds["dev_products"]])
    family_of = {p: types.get(p, {}).get("family") for p in selected}
    split = [{"product_id": f"product:{p}", "parent_asin": p, "partition": "dev" if p in dev else "test",
              "family": family_of[p]} for p in selected]

    (out / "sentences.parquet").unlink(missing_ok=True)  # judge sentence cache belongs to the previous sample
    pd.DataFrame(products).to_parquet(out / "products.parquet", index=False)
    pd.DataFrame(reviews).to_parquet(out / "reviews.parquet", index=False)
    write_json(out / "split_manifest.json", {"seed": seed, "split": split,
                                             "hash": sha(canonical_json(split))})
    with open(out / "quarantine.jsonl", "w") as fh:
        for q in quarantine:
            fh.write(json.dumps(q) + "\n")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile": cfg.profile,
        "dataset": ds["repo"], "revision": ds["revision"], "category": ds["category"],
        "files": {f.name: {"bytes": f.stat().st_size, "sha256": _file_sha(f)} for f in (review_path, meta_path)},
        "sampling_mode": ("family_brand_strata" if brand_strata else "seeded_hash_family_quota") if quotas
        else "seeded_hash_min_reviews",
        "review_order": ds.get("review_order", "seeded"),
        "n_selected_by_type": {str(k): int(v) for k, v in
                               pd.Series([str(types.get(p, {}).get("type")) for p in selected]).value_counts().items()},
        "n_brands_with_3plus": int((pd.Series([brands.get(p) for p in selected]).value_counts() >= 3).sum()),
        "family_quotas": quotas,
        "n_selected_by_family": {str(f): sum(1 for p in selected if str(family_of[p]) == f)
                                 for f in sorted({str(v) for v in family_of.values()})},
        "n_dev_by_family": {str(f): sum(1 for p in dev if str(family_of[p]) == f)
                            for f in sorted({str(v) for v in family_of.values()})},
        "seed": seed,
        "n_meta_products": len(meta),
        "n_reviews_scanned": sum(counts.values()),
        "n_products_with_reviews": len(counts),
        "n_eligible_products": len(eligible),
        "n_selected_products": len(selected),
        "beauty_filter": {"n_classified": len(types), "n_non_beauty_skipped": sum(not t["beauty"] for t in types.values()),
                          "n_classified_by_family": {str(f): sum(1 for t in types.values() if t["beauty"] and str(t["family"]) == f)
                                                     for f in sorted({str(t["family"]) for t in types.values()})}}
        if ds.get("beauty_filter") and not cfg.offline else None,
        "n_reviews_kept": len(reviews),
        "n_quarantined": len(quarantine),
        "missing_rate": {
            "price": sum(p["price"] is None for p in products) / max(len(products), 1),
            "store": sum(p["store"] is None for p in products) / max(len(products), 1),
            "user_id": sum(r["user_key"] is None for r in reviews) / max(len(reviews), 1),
        },
        "duration_sec": round(time.time() - t0, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "manifest.json", manifest)
    return manifest
