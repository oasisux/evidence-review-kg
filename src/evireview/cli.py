"""evireview command line."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .common import ROOT, load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="evireview")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("doctor", "ingest", "induce-aspects", "build", "augment", "judge", "audit", "evaluate", "app", "all",
                 "demo"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/research.yaml")
        if name == "judge":
            p.add_argument("--task", default="all",
                           choices=["all", "dev", "queries", "claims", "reference", "qrels", "answers", "b6", "kgacc"])
        if name == "demo":
            p.add_argument("--profile", default="fixture")
            p.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        a.config = f"configs/{a.profile}.yaml"
    cfg = load_config(a.config)
    from . import embed
    embed.configure(cfg["runtime"]["embedding_device"])

    if a.cmd == "doctor":
        import os
        import platform
        import torch
        print(json.dumps({"python": sys.version.split()[0], "platform": platform.platform(),
                          "mps_available": torch.backends.mps.is_available(),
                          "api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
                          "provider": cfg["extraction"]["provider"],
                          "max_cost_usd": cfg["budget"]["max_cost_usd"]}, indent=2))
        return
    if a.cmd in ("ingest", "all", "demo"):
        from .ingest import run_ingest
        print(json.dumps(run_ingest(cfg), indent=1))
    if a.cmd == "induce-aspects":
        from .induce import run_induce
        print(json.dumps(run_induce(cfg), indent=1, ensure_ascii=False))
        return
    if a.cmd in ("build", "all", "demo"):
        from .pipeline import run_build
        c = run_build(cfg)
        print(f"G1 candidates: {len(c)}  hard-pass: {int(c.hard_pass.sum())}")
    from . import judge
    if a.cmd == "judge":
        judge.run(cfg, a.task)
    if a.cmd == "all":
        judge.run(cfg, "dev")  # dev labels -> tau, before augmentation
    if a.cmd in ("augment", "all", "demo"):
        from .pipeline import run_augment
        print(json.dumps(run_augment(cfg), indent=1))
    if a.cmd == "all":
        judge.run(cfg, "all")
    if a.cmd == "audit":
        from .audit import run_audit
        run_audit(cfg)
    if a.cmd in ("evaluate", "all", "demo", "audit"):
        from .evaluate import run_evaluate
        print(json.dumps(run_evaluate(cfg), indent=1, default=str)[:3000])
    if a.cmd == "app":
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "src/evireview/app.py"),
                        "--server.address", "127.0.0.1", "--", "--config", a.config])


if __name__ == "__main__":
    main()
