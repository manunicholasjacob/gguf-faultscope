"""Turn the campaign JSONL into the rows the manuscript needs.

**Read this before trusting any per-format number.** The first version of this file printed a
catastrophe rate per block type and the numbers were meaningless, in a way that would have
survived into the paper if nobody had checked the sampling.

The sample is stratified by structural role, and the strata a block type receives depend on
which file it came from and how many blocks of it that file holds. In the campaign, Q4_K
tensors drew 100 packed-scale sites and **zero** exponent sites, because in a Q4_K_M file the
Q4_K tensors are 12 of 169 while Q5_0 is 132. Its measured catastrophe rate of 0.0 percent was
therefore an artifact of never sampling the only stratum that produces catastrophes, and the
apparently significant Q4_0-against-Q4_K comparison that fell out of it was spurious.

So this file does the comparison the design actually supports. All catastrophes occur in one
stratum, so:

    P(catastrophic per random bit) = P(bit is an fp16 exponent bit) x P(catastrophic | that)

The left factor is structural and comes from the file's own census. The right factor is
measured, and it is estimated only where enough exponent sites were drawn. Multiplying them
gives a per-file exposure that is comparable across formats because it no longer depends on how
the sampler happened to distribute its draws.

    python make_tables.py --results results/ --census results/faultscope_census.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_faultscope import LAYOUTS, blast_profile

EXPONENT = "fp16_scale_exponent"
STRATUM_ORDER = [EXPONENT, "fp16_scale_sign", "fp16_scale_mantissa",
                 "packed_scale", "int_scale", "payload"]
MIN_N = 30          # below this, a rate is reported but flagged as not estimable


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def fmt_ratio(x: Optional[float]) -> str:
    """Perplexity ratios here span 1.0 to 1e86 and infinity, so print them readably."""
    if x is None:
        return "n/a"
    if not math.isfinite(x):
        return "inf"
    if x < 100:
        return f"{x:.3f}"
    return f"{x:.2e}"


def is_hard(r: Dict) -> bool:
    """Non-finite logits, or perplexity at least doubled. No top-1 clause.

    Every record carries three signals and the campaign's own `catastrophic` flag ORs them
    together, including a clause that fires when more than ten percent of token predictions
    move. That clause measures a different thing from the other two: a model that still works
    and answers differently is not a model that has failed. It is also threshold-sensitive, and
    switching it on moves the per-bit rates by several points, so a table that mixes the two
    criteria is not comparable with itself. Everything published from this repository uses the
    criterion here; `--criterion recorded` reproduces the looser one.
    """
    d = r.get("divergence") or {}
    if d.get("non_finite"):
        return True
    pr = d.get("ppl_ratio")
    return pr is not None and pr >= 2.0


def apply_criterion(files: Dict[str, List[Dict]], criterion: str) -> None:
    """Rewrite each record's verdict in memory so every table below agrees on one definition."""
    if criterion == "recorded":
        return
    for rows in files.values():
        for r in rows:
            r["catastrophic"] = is_hard(r)


def load_dir(path: str) -> Dict[str, List[Dict]]:
    """One list per source file, so per-file rates do not have to be reconstructed."""
    out: Dict[str, List[Dict]] = {}
    if os.path.isfile(path):
        out["all"] = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        return out
    for f in sorted(glob.glob(os.path.join(path, "injections-*.jsonl"))):
        name = os.path.basename(f)[len("injections-"):-len(".jsonl")]
        if name == "all":
            continue
        out[name] = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return out


# ------------------------------------------------------------------ tables

def table_by_stratum(rows: List[Dict]) -> List[Dict]:
    g: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        g[r["site"]["stratum"]].append(r)
    out = []
    for s in STRATUM_ORDER:
        rs = g.get(s)
        if not rs:
            continue
        k = sum(1 for r in rs if r["catastrophic"])
        p, lo, hi = wilson(k, len(rs))
        t1 = sorted(r["divergence"]["top1_diff_rate"] or 0.0 for r in rs)
        pr = sorted(x for x in (r["divergence"]["ppl_ratio"] for r in rs)
                    if x is not None and math.isfinite(x))
        out.append({
            "stratum": s, "n": len(rs),
            "mean_blast": statistics.fmean(r["site"]["blast"] for r in rs),
            "catastrophic": k, "cat_pct": p, "cat_lo": lo, "cat_hi": hi,
            "any_change_pct": 100 * sum(1 for x in t1 if x > 0) / len(t1),
            "ppl_median": statistics.median(pr) if pr else float("nan"),
            "ppl_p95": pr[min(len(pr) - 1, int(0.95 * len(pr)))] if pr else float("nan"),
            "ppl_max": pr[-1] if pr else float("nan"),
            "non_finite": sum(1 for r in rs if r["divergence"]["non_finite"]),
        })
    return out


def table_exponent_bits(rows: List[Dict]) -> List[Dict]:
    """Which of the five exponent bits carries the effect.

    An fp16 is 1 sign, 5 exponent, 10 mantissa, little endian in the file, so within the high
    byte bit 7 is the sign and bits 6 down to 2 are the exponent from most to least
    significant. Flipping exponent bit i changes the exponent by 2^i, so the scale is
    multiplied or divided by 2 raised to that.
    """
    g: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        if r["site"]["stratum"] == EXPONENT:
            g[r["site"]["bit"]].append(r)
    out = []
    for bit in sorted(g, reverse=True):
        rs = g[bit]
        k = sum(1 for r in rs if r["catastrophic"])
        p, lo, hi = wilson(k, len(rs))
        exp_index = bit - 2                  # 0 for the least significant exponent bit
        out.append({"byte_bit": bit, "exponent_bit": exp_index,
                    "scale_factor": 2 ** (2 ** exp_index),
                    "n": len(rs), "catastrophic": k,
                    "cat_pct": p, "cat_lo": lo, "cat_hi": hi})
    return out


def table_per_file(files: Dict[str, List[Dict]], census: Optional[Dict]) -> List[Dict]:
    out = []
    for name, rows in files.items():
        exp = [r for r in rows if r["site"]["stratum"] == EXPONENT]
        k = sum(1 for r in exp if r["catastrophic"])
        p, lo, hi = wilson(k, len(exp))
        e = (census or {}).get(name, {}).get("exponent_bit_pct")
        w = (census or {}).get(name, {}).get("wide_bit_pct")
        row = {"file": name, "n_total": len(rows), "n_exponent": len(exp),
               "catastrophic_in_exponent": k,
               "p_cat_given_exponent": p, "lo": lo, "hi": hi,
               "exponent_bit_pct": e, "wide_bit_pct": w}
        if e is not None and len(exp) >= MIN_N:
            row["predicted_rate_pct"] = e * p / 100
            row["predicted_lo"] = e * lo / 100
            row["predicted_hi"] = e * hi / 100
            # The top two exponent bits carry the effect, so this is the protection target.
            row["top2_bit_pct"] = e * 2 / 5
        out.append(row)
    out.sort(key=lambda r: -(r.get("predicted_rate_pct") or 0))
    return out


def sampling_audit(rows: List[Dict]) -> List[Dict]:
    """How many exponent sites each block type actually drew.

    This is the table that says which per-block-type numbers may be quoted at all.
    """
    g: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cat: Dict[str, int] = defaultdict(int)
    for r in rows:
        bt = r["site"]["tensor_type"]
        g[bt][r["site"]["stratum"]] += 1
        if r["site"]["stratum"] == EXPONENT and r["catastrophic"]:
            cat[bt] += 1
    out = []
    for bt, strata in g.items():
        n_exp = strata.get(EXPONENT, 0)
        p, lo, hi = wilson(cat[bt], n_exp)
        out.append({"block_type": bt, "n_total": sum(strata.values()),
                    "n_exponent": n_exp, "catastrophic": cat[bt],
                    "p_cat_given_exponent": p, "lo": lo, "hi": hi,
                    "estimable": n_exp >= MIN_N,
                    "strata": dict(strata)})
    out.sort(key=lambda r: -r["n_exponent"])
    return out


# ------------------------------------------------------------------ rendering

def render(files: Dict[str, List[Dict]], census: Optional[Dict], md: bool) -> str:
    rows = [r for rs in files.values() for r in rs]
    L: List[str] = []
    k = sum(1 for r in rows if r["catastrophic"])
    p, lo, hi = wilson(k, len(rows))
    nf = sum(1 for r in rows if r["divergence"]["non_finite"])
    L.append(f"{len(rows)} injections across {len(files)} files")
    L.append(f"catastrophic: {k}, {p:.2f}% [{lo:.2f}, {hi:.2f}]   non-finite: {nf}")

    strat = table_by_stratum(rows)
    in_exp = next((s for s in strat if s["stratum"] == EXPONENT), None)
    if in_exp and in_exp["catastrophic"] == k:
        L.append("")
        L.append(f"EVERY catastrophic injection landed in {EXPONENT}. "
                 f"{k} of {k}. The other {len(rows)-in_exp['n']} injections produced none.")
    L.append("")

    L.append("TABLE 4. Severity by structural role.")
    hdr = (f"{'role':<22}{'n':>5}{'blast':>7}{'catas%':>8}{'95% CI':>15}"
           f"{'changed':>9}{'ppl med':>9}{'ppl p95':>11}{'ppl max':>11}")
    L += [hdr, "-" * len(hdr)]
    for s in strat:
        L.append(f"{s['stratum']:<22}{s['n']:>5}{s['mean_blast']:>7.0f}"
                 f"{s['cat_pct']:>7.1f}%{'[%.1f, %.1f]' % (s['cat_lo'], s['cat_hi']):>15}"
                 f"{s['any_change_pct']:>8.0f}%{fmt_ratio(s['ppl_median']):>9}"
                 f"{fmt_ratio(s['ppl_p95']):>11}{fmt_ratio(s['ppl_max']):>11}")
    L.append("")
    L.append("'changed' is the share where any token prediction moved at all, catastrophic or")
    L.append("not. It is high everywhere, which is the point: perturbation is common and")
    L.append("consequence is rare.")
    L.append("")

    bits = table_exponent_bits(rows)
    if bits:
        L.append("TABLE 5. Within the exponent, which bit.")
        h = (f"{'exp bit':>8}{'scale x':>12}{'n':>6}{'catas':>7}{'catas%':>9}{'95% CI':>16}")
        L += [h, "-" * len(h)]
        for b in bits:
            L.append(f"{b['exponent_bit']:>8}{('2^%d' % (2**b['exponent_bit'])):>12}"
                     f"{b['n']:>6}{b['catastrophic']:>7}{b['cat_pct']:>8.1f}%"
                     f"{'[%.1f, %.1f]' % (b['cat_lo'], b['cat_hi']):>16}")
        top = [b for b in bits if b["cat_pct"] > 5]
        if top:
            nt = sum(b["n"] for b in top)
            kt = sum(b["catastrophic"] for b in top)
            pt, lt, ht = wilson(kt, nt)
            rest_n = sum(b["n"] for b in bits if b not in top)
            rest_k = sum(b["catastrophic"] for b in bits if b not in top)
            L.append("")
            L.append(f"The top {len(top)} exponent bits: {kt}/{nt} catastrophic, "
                     f"{pt:.1f}% [{lt:.1f}, {ht:.1f}].")
            L.append(f"The remaining {len(bits)-len(top)}: {rest_k}/{rest_n}.")
        L.append("")

    audit = sampling_audit(rows)
    L.append("TABLE 6. Sampling audit. Which block types drew enough exponent sites to be")
    L.append("estimable at all. THIS IS THE TABLE THAT SAYS WHAT MAY BE QUOTED.")
    h = f"{'block':<9}{'n':>7}{'n_exp':>7}{'catas':>7}{'P(cat|exp)':>12}{'95% CI':>16}  estimable"
    L += [h, "-" * len(h)]
    for a in audit:
        L.append(f"{a['block_type']:<9}{a['n_total']:>7}{a['n_exponent']:>7}"
                 f"{a['catastrophic']:>7}{a['p_cat_given_exponent']:>11.1f}%"
                 f"{'[%.1f, %.1f]' % (a['lo'], a['hi']):>16}  "
                 f"{'yes' if a['estimable'] else 'NO, too few exponent sites'}")
    L.append("")

    per = table_per_file(files, census)
    if census:
        L.append("TABLE 7. Per-file exposure, the comparison the design supports.")
        L.append("predicted rate = (share of bits that are fp16 exponent) x P(catastrophic | exponent)")
        h = (f"{'file':<9}{'expo%':>8}{'wide%':>8}{'n_exp':>7}{'P(cat|exp)':>12}"
             f"{'predicted':>11}{'95% CI':>18}{'protect top2':>14}")
        L += [h, "-" * len(h)]
        for r in per:
            if "predicted_rate_pct" not in r:
                L.append(f"{r['file']:<9}{'':>8}{'':>8}{r['n_exponent']:>7}"
                         f"{'':>12}{'not estimable':>11}")
                continue
            L.append(f"{r['file']:<9}{r['exponent_bit_pct']:>7.3f}%{r['wide_bit_pct']:>7.3f}%"
                     f"{r['n_exponent']:>7}{r['p_cat_given_exponent']:>11.1f}%"
                     f"{r['predicted_rate_pct']:>10.4f}%"
                     f"{'[%.4f, %.4f]' % (r['predicted_lo'], r['predicted_hi']):>18}"
                     f"{r['top2_bit_pct']:>13.3f}%")
        L.append("")
        best = per[0]
        worst = [r for r in per if "predicted_rate_pct" in r][-1]
        if best is not worst:
            L.append(f"Spread: {best['file']} is "
                     f"{best['predicted_rate_pct']/worst['predicted_rate_pct']:.2f}x more "
                     f"exposed than {worst['file']}, on files that differ in size by less "
                     f"than a factor of two.")
        L.append("'protect top2' is the share of the file that would have to be hardened to")
        L.append("remove essentially all catastrophic exposure, if the top two exponent bits")
        L.append("carry it. That is the number the mitigation argument rests on.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True,
                    help="a results directory, or a single combined jsonl")
    ap.add_argument("--census", help="faultscope_census.json")
    ap.add_argument("--json", help="write the computed rows here")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--criterion", choices=("hard", "recorded"), default="hard",
                    help="hard: non-finite logits or perplexity doubled, which is what every "
                         "published number here uses. recorded: the flag stored in the file, "
                         "which also fires on top-1 divergence above ten percent")
    a = ap.parse_args()

    files = load_dir(a.results)
    if not files:
        print("no injection files found")
        return 1
    apply_criterion(files, a.criterion)
    print(f"criterion: {a.criterion}"
          + ("  (non-finite logits or perplexity doubled)" if a.criterion == "hard"
             else "  (the recorded flag, which also fires on top-1 divergence over ten percent)"))
    census = json.load(open(a.census, encoding="utf-8")) if a.census else None
    print(render(files, census, a.markdown))

    if a.json:
        rows = [r for rs in files.values() for r in rs]
        payload = {
            "n": len(rows),
            "by_stratum": table_by_stratum(rows),
            "by_exponent_bit": table_exponent_bits(rows),
            "sampling_audit": sampling_audit(rows),
            "per_file": table_per_file(files, census),
        }
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
