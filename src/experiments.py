"""Experiment plans (what is run for each part of the paper's protocol).

All conditions that are compared with each other are generated from the SAME
(model, corpus, seed) tuple, hence share splits, initial weights and batch order.
"""

from __future__ import annotations
from itertools import product
from typing import Iterable, List

from src.pipeline import canonical_variant, run_experiment, VARIANTS

MAIN_VARIANTS = ["categorical", "full"]
ABLATION_VARIANTS = ["categorical", "dim", "int", "phase", "dim_int", "dim_phase",
                     "int_phase", "full", "full_shuffled_proto"]


def plan(stage: str, models: List[str], corpora: List[str], seeds: List[int],
         pairs: List[str], variants: List[str] | None = None):
    """Yield dicts of keyword arguments for ``run_experiment``."""
    if stage == "within":
        for m, c, s, v in product(models, corpora, seeds, variants or MAIN_VARIANTS):
            yield dict(model_name=m, corpus=c, seed=s, variant=v, group="main")
    elif stage == "combined":
        for m, s, v in product(models, seeds, variants or MAIN_VARIANTS):
            yield dict(model_name=m, corpus="combined", seed=s, variant=v, group="main")
    elif stage == "leave":
        for m, p, s, v in product(models, pairs, seeds, variants or MAIN_VARIANTS):
            src, tgt = p.split(":")
            yield dict(model_name=m, corpus=src, test_corpus=tgt, seed=s, variant=v,
                       group="leave_corpus_out")
    elif stage == "ablation":
        for m, c, s, v in product(models, corpora, seeds, variants or ABLATION_VARIANTS):
            yield dict(model_name=m, corpus=c, seed=s, variant=v, group="ablation")
    else:
        raise ValueError(stage)


def execute(cfg, jobs: Iterable[dict], device=None, force=False, dry_run=False):
    jobs = [dict(j, variant=canonical_variant(j["variant"])) for j in jobs]
    print(f"{len(jobs)} runs planned")
    for j in jobs:
        tgt = f"->{j['test_corpus']}" if j.get("test_corpus") else ""
        print(f"  {j['group']:<17} {j['model_name']:<15} {j['corpus']}{tgt:<8} "
              f"{j['variant']:<20} seed={j['seed']}")
        if not dry_run:
            run_experiment(cfg, device=device, force=force, **j)
