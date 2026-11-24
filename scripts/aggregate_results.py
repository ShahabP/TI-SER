#!/usr/bin/env python3
"""Aggregate run records into per-seed tables, summaries and paired statistics.

Reads every ``results/**/run_record.json`` and writes to ``results/summary/``:

  per_run.csv              one row per run (model, corpus, variant, seed, acc, F1, fingerprints)
  summary.csv              mean +/- SD over seeds per condition
  paired_differences.csv   descriptive paired differences vs ``categorical`` (Appendix B7 style):
                           mean Δ, 95% CI, wins/ties/losses (default export)
  paired_tests.csv         optional inferential tests (t-test, Wilcoxon, Holm, d_z) when
                           ``--inferential`` is passed
  report.md                readable tables with the evaluation protocol in each caption
  excluded_runs.txt        records NOT produced by this pipeline version (historical / unverified)

Pairs are only formed when split fingerprint, normalisation fingerprint and
training/model config are identical, i.e. a genuinely matched comparison.
"""

from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "2.0.0"
BASELINE = "categorical"
METRICS = ("accuracy", "macro_f1")


def load_records(results_dir: Path):
    keep, excluded = [], []
    for p in sorted(results_dir.glob("**/run_record.json")):
        with open(p) as f:
            r = json.load(f)
        (keep if r.get("pipeline_version") == PIPELINE_VERSION else excluded).append((p, r))
    return keep, excluded


def cond_key(r):
    tag = r["corpus"] if not r.get("test_corpus") else f"{r['corpus']}->{r['test_corpus']}"
    return (r["group"], r["protocol"], r["model"], tag)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired_stats(base, other):
    """base/other: arrays over matched seeds. Returns dict of paired statistics."""
    d = np.asarray(other) - np.asarray(base)
    n = len(d)
    out = {"n_pairs": n, "mean_diff": float(d.mean()), "wins": int((d > 0).sum()),
           "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}
    if n >= 2 and d.std(ddof=1) > 0:
        sd = d.std(ddof=1)
        se = sd / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        out.update(ci_low=float(d.mean() - tcrit * se), ci_high=float(d.mean() + tcrit * se),
                   t_p=float(stats.ttest_rel(other, base).pvalue), dz=float(d.mean() / sd))
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue) if (d != 0).sum() >= 1 else float("nan")
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    else:
        out.update(ci_low=float("nan"), ci_high=float("nan"), t_p=float("nan"),
                   dz=float("nan"), wilcoxon_p=float("nan"))
    return out


def same_setup(a, b):
    ca, cb = a["config"], b["config"]
    return (a["split"]["fingerprint"] == b["split"]["fingerprint"]
            and a["normalization"]["fingerprint"] == b["normalization"]["fingerprint"]
            and all(ca[k] == cb[k] for k in ("model", "train", "data")))


def caption(r):
    c, s = r["config"], r["split"]
    ev = {"within-corpus": f"{s['mode']} {c['data']['train_ratio']:.2f}/{c['data']['val_ratio']:.2f}/"
                           f"{1-c['data']['train_ratio']-c['data']['val_ratio']:.2f} train/val/test split",
          "combined": f"{s['mode']} split of RAVDESS+TESS",
          "leave-corpus-out": "train on source (stratified train/val), test on the entire target corpus"}
    return (f"{ev[r['protocol']]}; {c['train']['epochs']} epochs; best-validation checkpoint; "
            f"training-only normalisation; SpecAugment on training split only; "
            f"pipeline v{r['pipeline_version']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--inferential", action="store_true",
                    help="also compute t-test / Wilcoxon / Holm (optional; not part of the default report)")
    a = ap.parse_args()
    rdir = Path(a.results)
    out = rdir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    records, excluded = load_records(rdir)
    (out / "excluded_runs.txt").write_text(
        "Records not produced by pipeline v%s (historical / unverified; exclude from main evidence):\n%s\n"
        % (PIPELINE_VERSION, "\n".join(str(p) for p, _ in excluded)))
    if not records:
        print("No run records found. Run scripts/run_experiments.py first.")
        return

    # ---- per run
    with open(out / "per_run.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "protocol", "model", "corpus", "test_corpus", "variant", "seed",
                    "accuracy", "macro_f1", "best_epoch", "n_parameters",
                    "split_fingerprint", "norm_fingerprint", "config_hash", "run_dir"])
        for p, r in records:
            w.writerow([r["group"], r["protocol"], r["model"], r["corpus"], r.get("test_corpus") or "",
                        r["variant"], r["seed"], f"{r['test']['accuracy']:.6f}",
                        f"{r['test']['macro_f1']:.6f}", r["best_epoch"], r["n_parameters"],
                        r["split"]["fingerprint"], r["normalization"]["fingerprint"],
                        r["config_hash"], str(p.parent)])

    # ---- group
    groups = defaultdict(lambda: defaultdict(dict))      # cond -> variant -> seed -> record
    for _, r in records:
        groups[cond_key(r)][r["variant"]][r["seed"]] = r

    summary_rows, desc_rows, test_rows, md = [], [], [], ["# Results summary\n"]
    for cond, by_var in sorted(groups.items()):
        group, protocol, model, tag = cond
        any_rec = next(iter(next(iter(by_var.values())).values()))
        md.append(f"\n## {group} | {model} | {tag}\n\n*Protocol:* {caption(any_rec)}\n")
        md.append("| Variant | n seeds | Accuracy (%) | Macro-F1 (%) |\n|---|---|---|---|")
        for var, runs in sorted(by_var.items(), key=lambda kv: (kv[0] != BASELINE, kv[0])):
            acc = np.array([r["test"]["accuracy"] for r in runs.values()]) * 100
            f1 = np.array([r["test"]["macro_f1"] for r in runs.values()]) * 100
            sd = lambda x: x.std(ddof=1) if len(x) > 1 else float("nan")
            summary_rows.append([group, protocol, model, tag, var, len(runs), acc.mean(), sd(acc),
                                 f1.mean(), sd(f1)])
            md.append(f"| {var} | {len(runs)} | {acc.mean():.2f} ± {sd(acc):.2f} | "
                      f"{f1.mean():.2f} ± {sd(f1):.2f} |")

        if BASELINE not in by_var:
            md.append("\n*No categorical baseline for this condition; no paired differences.*")
            continue
        base_runs = by_var[BASELINE]
        fam = []
        for var, runs in by_var.items():
            if var == BASELINE:
                continue
            seeds = sorted(set(base_runs) & set(runs))
            seeds = [s for s in seeds if same_setup(base_runs[s], runs[s])]
            if not seeds:
                continue
            for metric in METRICS:
                b = np.array([base_runs[s]["test"][metric] for s in seeds]) * 100
                o = np.array([runs[s]["test"][metric] for s in seeds]) * 100
                st = paired_stats(b, o)
                fam.append((var, metric, st))
                # Descriptive paired differences (Appendix B7 style) always exported
                desc_rows.append([group, protocol, model, tag, var, metric, st["n_pairs"],
                                  st["mean_diff"], st["ci_low"], st["ci_high"],
                                  st["wins"], st["ties"], st["losses"]])
        if fam:
            md.append(f"\n**Descriptive paired difference vs `{BASELINE}`** "
                      "(same seeds, splits, init, batch order; percentage points):\n")
            md.append("| Variant | Metric | n | Δ mean | 95% CI | W/T/L |"
                      "\n|---|---|---|---|---|---|")
            for var, metric, st in fam:
                md.append(f"| {var} | {metric} | {st['n_pairs']} | {st['mean_diff']:+.2f} | "
                          f"[{st['ci_low']:+.2f}, {st['ci_high']:+.2f}] | "
                          f"{st['wins']}/{st['ties']}/{st['losses']} |")
            if a.inferential:
                for metric in METRICS:
                    items = [x for x in fam if x[1] == metric]
                    pv = np.array([x[2]["t_p"] for x in items])
                    ok = ~np.isnan(pv)
                    adj = np.full(len(pv), np.nan)
                    if ok.any():
                        adj[ok] = holm(pv[ok])
                    for (var, m, st), ap_ in zip(items, adj):
                        test_rows.append([group, protocol, model, tag, var, m, st["n_pairs"],
                                          st["mean_diff"], st["ci_low"], st["ci_high"],
                                          st["t_p"], ap_, st["wilcoxon_p"], st["dz"],
                                          st["wins"], st["ties"], st["losses"]])
                md.append("\n*Inferential tests (optional, enabled with --inferential):*")
                md.append("| Variant | Metric | n | Δ mean | 95% CI | p (t) | p Holm | p (Wilcoxon) | d_z | W/T/L |"
                          "\n|---|---|---|---|---|---|---|---|---|---|")
                for row in [r for r in test_rows if r[:4] == [group, protocol, model, tag]]:
                    md.append(f"| {row[4]} | {row[5]} | {row[6]} | {row[7]:+.2f} | "
                              f"[{row[8]:+.2f}, {row[9]:+.2f}] | {row[10]:.3f} | {row[11]:.3f} | "
                              f"{row[12]:.3f} | {row[13]:+.2f} | {row[14]}/{row[15]}/{row[16]} |")
            md.append("\n*With few seeds, treat small or inconsistent differences as descriptive only.*")

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "protocol", "model", "corpus", "variant", "n_seeds",
                    "acc_mean", "acc_sd", "f1_mean", "f1_sd"])
        w.writerows(summary_rows)
    # Default export: descriptive paired differences (Appendix B7)
    with open(out / "paired_differences.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "protocol", "model", "corpus", "variant", "metric", "n_pairs",
                    "mean_diff", "ci_low", "ci_high", "wins", "ties", "losses"])
        w.writerows(desc_rows)
    if a.inferential and test_rows:
        with open(out / "paired_tests.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "protocol", "model", "corpus", "variant", "metric", "n_pairs",
                        "mean_diff", "ci_low", "ci_high", "p_ttest", "p_holm", "p_wilcoxon",
                        "cohen_dz", "wins", "ties", "losses"])
            w.writerows(test_rows)
    (out / "report.md").write_text("\n".join(md) + "\n")
    extra = ", paired_tests.csv" if a.inferential and test_rows else ""
    print(f"Wrote {out}/per_run.csv, summary.csv, paired_differences.csv{extra}, report.md "
          f"({len(records)} runs, {len(excluded)} excluded)")


if __name__ == "__main__":
    main()
