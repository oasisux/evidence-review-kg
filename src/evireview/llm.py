"""Anthropic adapter with response cache, cost ledger and a hard budget stop."""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .common import Config, canonical_json, sha

# $ per million tokens: (input, output). Cache write 1.25x input, cache read 0.1x input.
PRICES = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}
FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}


class BudgetExceeded(RuntimeError):
    pass


class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache_dir = cfg.p("cache")
        self.ledger_path = cfg.p("runs") / "cost_ledger.jsonl"
        self.max_cost = float(cfg["budget"]["max_cost_usd"])
        self.concurrency = int(cfg["budget"]["concurrency"])
        self._lock = threading.Lock()
        self._client = None
        self._fallbacks_ok = True
        self.spent = self._load_spent()

    # ------------------------------------------------------------ accounting
    def _load_spent(self) -> float:
        if not self.ledger_path.exists():
            return 0.0
        return sum(json.loads(l)["cost_usd"] for l in self.ledger_path.read_text().splitlines() if l.strip())

    def _cost(self, model: str, usage) -> float:
        pin, pout = PRICES[model]
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        return (usage.input_tokens * pin + cw * pin * 1.25 + cr * pin * 0.1 + usage.output_tokens * pout) / 1e6

    def _record(self, stage: str, model: str, usage, cost: float, latency: float) -> None:
        with self._lock:
            self.spent += cost
            with open(self.ledger_path, "a") as fh:
                fh.write(json.dumps({
                    "ts": time.time(), "stage": stage, "model": model,
                    "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                    "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
                    "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
                    "cost_usd": cost, "currency": "USD", "latency_s": round(latency, 2)}) + "\n")

    @property
    def client(self):
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is empty. Put your key in .env and re-run.")
            import anthropic
            self._client = anthropic.Anthropic(max_retries=6, timeout=300.0)
        return self._client

    # ------------------------------------------------------------ calls
    def json_call(self, *, stage: str, model: str, effort: str, system: str, user: str,
                  schema: dict, max_tokens: int = 16000) -> dict:
        key = sha(canonical_json([model, effort, system, user, schema, max_tokens]))
        path = self.cache_dir / stage / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["parsed"]
        if self.spent >= self.max_cost:
            raise BudgetExceeded(f"budget ${self.max_cost:.2f} reached (spent ${self.spent:.2f})")

        import anthropic
        params = dict(
            model=model, max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
        t0 = time.time()
        resp = None
        if model in FALLBACK_MODELS and self._fallbacks_ok:
            try:
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
            except anthropic.BadRequestError as e:
                if "fallback" not in str(e).lower():
                    raise
                self._fallbacks_ok = False
        if resp is None:
            resp = self.client.messages.create(**params)
        latency = time.time() - t0
        served = getattr(resp, "model", model) or model
        cost = self._cost(served if served in PRICES else model, resp.usage)
        self._record(stage, served, resp.usage, cost, latency)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refusal in stage {stage}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError(f"max_tokens hit in stage {stage}")
        text = next(b.text for b in resp.content if b.type == "text")
        parsed = json.loads(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "served_model": served, "effort": effort, "stage": stage,
                                    "request_id": getattr(resp, "_request_id", None),
                                    "usage": resp.usage.to_dict(), "parsed": parsed}, ensure_ascii=False))
        return parsed

    def map(self, fn: Callable[[Any], Any], items: list, desc: str = "") -> list:
        """Run fn over items concurrently. Stops scheduling on BudgetExceeded; returns None for failures."""
        results: list = [None] * len(items)
        errors = 0
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futs = {pool.submit(fn, it): i for i, it in enumerate(items)}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except BudgetExceeded as e:
                    print(f"[{desc}] {e}")
                    for f in futs:
                        f.cancel()
                    raise
                except Exception as e:  # keep going, log
                    errors += 1
                    print(f"[{desc}] item {i} failed: {type(e).__name__}: {str(e)[:200]}")
                done += 1
                if done % 25 == 0 or done == len(items):
                    print(f"[{desc}] {done}/{len(items)}  spent=${self.spent:.2f}", flush=True)
        if errors:
            print(f"[{desc}] {errors} failures")
        return results


def ledger_summary(cfg: Config) -> list[dict]:
    path = cfg.p("runs") / "cost_ledger.jsonl"
    if not path.exists():
        return []
    rows: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        k = (r["stage"], r["model"])
        agg = rows.setdefault(k, {"stage": r["stage"], "model": r["model"], "calls": 0, "input_tokens": 0,
                                  "output_tokens": 0, "cost": 0.0, "duration_sec": 0.0, "currency": "USD"})
        agg["calls"] += 1
        agg["input_tokens"] += r["input_tokens"] + r["cache_read"] + r["cache_write"]
        agg["output_tokens"] += r["output_tokens"]
        agg["cost"] += r["cost_usd"]
        agg["duration_sec"] += r["latency_s"]
    return list(rows.values())
