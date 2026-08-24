"""The two figures the paper actually needs, and one it must not use.

The campaign kernel produced `fig_structure_vs_severity.png`, a scatter of measured
catastrophe rate against wide-bit share with one point per block type. **Do not use it.** Its
x-axis is a structural property and its y-axis is a sampling artifact: which strata a block
type received depends on how many blocks of it the file contained, so Q4_K drew zero exponent
sites and its zero percent means nothing. That figure would have put a spurious relationship in
front of reviewers.

What replaces it is a plot of the comparison the design does support: predicted exposure per
file, against the share of the file that would have to be hardened to remove it.

    python make_figures.py --results results/ --out results/
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPONENT = "fp16_scale_exponent"
STRATA = [EXPONENT, "fp16_scale_sign", "fp16_scale_mantissa",
          "packed_scale", "int_scale", "payload"]
LABEL = {EXPONENT: "fp16 scale, exponent", "fp16_scale_sign": "fp16 scale, sign",
         "fp16_scale_mantissa": "fp16 scale, mantissa",
         "packed_scale": "packed sub-block scale", "int_scale": "int8 sub-block scale",
         "payload": "packed weight value"}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def load(results_dir: str):
    files: Dict[str, List[Dict]] = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "injections-*.jsonl"))):
        name = os.path.basename(f)[len("injections-"):-len(".jsonl")]
        if name != "all":
            files[name] = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    census_path = os.path.join(results_dir, "faultscope_census.json")
    census = json.load(open(census_path, encoding="utf-8")) if os.path.exists(census_path) else {}
    return files, census


def fig_by_role(files, out):
    rows = [r for rs in files.values() for r in rs]
    g = defaultdict(list)
    for r in rows:
        g[r["site"]["stratum"]].append(r)
    names, ps, los, his, ns = [], [], [], [], []
    for s in STRATA:
        rs = g.get(s)
        if not rs:
            continue
        k = sum(1 for r in rs if r["catastrophic"])
        p, lo, hi = wilson(k, len(rs))
        names.append(LABEL[s]); ps.append(p); los.append(p - lo); his.append(hi - p)
        ns.append(len(rs))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = range(len(names))
    ax.barh(list(y), ps, xerr=[los, his], color="#3b6ea5", height=0.62,
            error_kw={"ecolor": "#333", "capsize": 3, "lw": 1})
    ax.set_yticks(list(y)); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("catastrophic injections (%), with 95% Wilson intervals")
    ax.set_xlim(0, max(ps) * 1.35 + 1)
    for i, (p, n) in enumerate(zip(ps, ns)):
        ax.text(p + max(ps) * 0.02 + 0.3, i, f"{p:.1f}%  (n={n})", va="center", fontsize=8.5)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out, "fig1_severity_by_role.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_exponent_bits(files, out):
    rows = [r for rs in files.values() for r in rs
            if r["site"]["stratum"] == EXPONENT]
    g = defaultdict(list)
    for r in rows:
        g[r["site"]["bit"] - 2].append(r)      # 0 is the least significant exponent bit
    bits = sorted(g, reverse=True)
    ps, los, his, labels, ns = [], [], [], [], []
    for b in bits:
        rs = g[b]
        k = sum(1 for r in rs if r["catastrophic"])
        p, lo, hi = wilson(k, len(rs))
        ps.append(p); los.append(p - lo); his.append(hi - p)
        labels.append(f"bit {b}\n(scale x 2^{2**b})")
        ns.append(len(rs))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = range(len(labels))
    ax.bar(list(x), ps, yerr=[los, his], color="#a53b3b", width=0.62,
           error_kw={"ecolor": "#333", "capsize": 3, "lw": 1})
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("catastrophic injections (%)")
    ax.set_ylim(0, 108)
    for i, (p, n) in enumerate(zip(ps, ns)):
        ax.text(i, p + 3, f"{p:.1f}%\nn={n}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out, "fig2_exponent_bits.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_per_file(files, census, out):
    pts = []
    for name, rows in files.items():
        exp = [r for r in rows if r["site"]["stratum"] == EXPONENT]
        if len(exp) < 30 or name not in census:
            continue
        k = sum(1 for r in exp if r["catastrophic"])
        p, lo, hi = wilson(k, len(exp))
        e = census[name]["exponent_bit_pct"]
        pts.append({"name": name, "predicted": e * p / 100,
                    "lo": e * lo / 100, "hi": e * hi / 100,
                    "protect": e * 2 / 5})
    pts.sort(key=lambda r: -r["predicted"])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = [p["protect"] for p in pts]
    ys = [p["predicted"] for p in pts]
    yerr = [[p["predicted"] - p["lo"] for p in pts], [p["hi"] - p["predicted"] for p in pts]]
    ax.errorbar(xs, ys, yerr=yerr, fmt="o", ms=8, color="#2f6f4f",
                ecolor="#666", capsize=4, lw=1)
    for p in pts:
        ax.annotate(p["name"], (p["protect"], p["predicted"]),
                    textcoords="offset points", xytext=(9, -3), fontsize=9)
    ax.set_xlabel("share of the file needing protection, top two bits of every scale (%)")
    ax.set_ylabel("predicted catastrophic rate\nper random bit flip (%)")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    lo_x, hi_x = min(xs), max(xs)
    ax.set_xlim(lo_x - 0.08, hi_x + 0.18)
    fig.tight_layout()
    path = os.path.join(out, "fig3_per_file_exposure.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files, census = load(a.results)
    if not files:
        print("no injection files found")
        return 1
    for f in (fig_by_role(files, a.out),
              fig_exponent_bits(files, a.out),
              fig_per_file(files, census, a.out)):
        print("wrote", f)

    bad = os.path.join(a.out, "fig_structure_vs_severity.png")
    if os.path.exists(bad):
        os.replace(bad, bad + ".CONFOUNDED-DO-NOT-USE")
        print("renamed the confounded scatter so it cannot be used by accident")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
