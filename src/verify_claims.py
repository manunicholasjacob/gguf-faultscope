"""Re-derive every number this repository publishes, from the released records, and diff.

**Why this exists.** Over four rounds of work on this dataset, three published figures turned out
to be wrong, and every one of them failed the same way: a number computed under one criterion or
on one campaign, then quoted somewhere that scope no longer applied. Nothing caught them except
somebody recomputing by hand. This is that recomputation, written down and wired into CI, so the
next person does not have to trust a figure written in a document.

Nothing here reads a claim out of the README. Each expected value below is typed in by hand from
what the documents say, and the script recomputes it from `data/`. A mismatch means either the
data changed or a document is wrong, and either way somebody needs to look.

    python src/verify_claims.py            # exits non-zero on any mismatch
    python src/verify_claims.py --list     # print every claim and its re-derived value

**Two criteria, and mixing them is the mistake this file exists to prevent.** A *hard failure* is
a non-finite logit or perplexity at least doubled. The *recorded* verdict in each row also fires
on top-1 divergence above ten percent. Everything this repository publishes uses the strict one.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")

FIRST = "p100-qwen2.5-0.5b"          # role-only stratification, no per-format claims from it
V2 = "p100-qwen2.5-0.5b-v2"
BIG = "p100-qwen2.5-1.5b"
DUAL = {V2, BIG}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * p, 100 * max(0.0, (c - h) / d), 100 * min(1.0, (c + h) / d)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "*", "injections-*.jsonl"))):
        if f.endswith("-all.jsonl"):
            continue
        camp = os.path.basename(os.path.dirname(f))
        label = os.path.basename(f)[len("injections-"):-len(".jsonl")]
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["_camp"], r["_file"] = camp, label
                rows.append(r)
    return rows


def hard(r) -> bool:
    d = r.get("divergence") or {}
    if d.get("non_finite"):
        return True
    pr = d.get("ppl_ratio")
    return pr is not None and pr >= 2.0


def guard(rows, camps, ranges):
    """Score the range check. The range comes from the file the injection was in.

    That last part matters. Different quantized files built from one model collide on
    (offset, bit) often enough to notice, so a key without the file label can attribute one
    file's byte, or one file's allowed range, to another file's injection.
    """
    orig = {}
    for d in camps:
        for f in glob.glob(os.path.join(DATA, d, "*.repair")):
            lab = os.path.basename(f)[len("injections-"):-len(".jsonl.repair")]
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    x = json.loads(line)
                    if x.get("state") == "open":
                        orig[(d, lab, x["offset"], x["bit"])] = x["original"]
    tp = fn = fp = up = up_cat = down = down_cat = 0
    bit4_cat = bit4_caught = 0
    for r in rows:
        if r["_camp"] not in camps or r["site"]["stratum"] != "fp16_scale_exponent":
            continue
        key = (r["_camp"], r["_file"], r["site"]["abs_offset"], r["site"]["bit"])
        if key not in orig:
            continue
        o, bit = orig[key], r["site"]["bit"]
        detected = (((o ^ (1 << bit)) >> 2) & 31) > ranges[r["_file"]]
        cat = hard(r)
        if cat and detected:
            tp += 1
        elif cat:
            fn += 1
        elif detected:
            fp += 1
        if (o >> bit) & 1:
            down += 1; down_cat += cat
        else:
            up += 1; up_cat += cat
        if bit - 2 == 4 and cat:
            bit4_cat += 1; bit4_caught += detected
    return dict(tp=tp, fn=fn, fp=fp, up=up, up_cat=up_cat, down=down, down_cat=down_cat,
                bit4_cat=bit4_cat, bit4_caught=bit4_caught)


def per_file_rate(rows, camp, census_dir):
    """Exposure = share of bits that are fp16 exponent bits, times the rate measured in that file."""
    with open(os.path.join(DATA, census_dir, "faultscope_census.json"), encoding="utf-8") as fh:
        census = json.load(fh)
    out = {}
    for lab, v in census.items():
        sel = [r for r in rows if r["_camp"] == camp and r["_file"] == lab
               and r["site"]["stratum"] == "fp16_scale_exponent"]
        if not sel:
            continue
        out[lab] = v["exponent_bit_pct"] / 100 * (sum(hard(r) for r in sel) / len(sel)) * 100
    return out


def build(rows):
    """(claim, re-derived value, what the documents say). Expected values are typed by hand."""
    C = []
    add = C.append

    exp = [r for r in rows if r["site"]["stratum"] == "fp16_scale_exponent"]
    non = [r for r in rows if r["site"]["stratum"] != "fp16_scale_exponent"]
    sm = [r for r in rows if r["site"]["stratum"] in ("fp16_scale_sign", "fp16_scale_mantissa")]
    first = [r for r in rows if r["_camp"] == FIRST]
    dual = [r for r in rows if r["_camp"] in DUAL]

    add(("total injections", len(rows), 6725))
    add(("hard failures, all campaigns", sum(hard(r) for r in rows), 955))
    add(("hard failures outside the exponent field", sum(hard(r) for r in non), 0))
    add(("exponent injections", len(exp), 2800))
    add(("non-exponent injections", len(non), 3925))
    add(("sign and mantissa injections", len(sm), 2150))
    add(("hard failures in sign and mantissa", sum(hard(r) for r in sm), 0))
    for s in ("packed_scale", "int_scale", "payload"):
        add((f"hard failures in {s}",
             sum(hard(r) for r in rows if r["site"]["stratum"] == s), 0))

    # the two counts that look like a contradiction and are not
    add(("hard failures, dual-stratified campaigns", sum(hard(r) for r in dual), 817))
    add(("hard failures, first campaign, strict", sum(hard(r) for r in first), 138))
    add(("hard failures, first campaign, recorded",
         sum(bool(r["catastrophic"]) for r in first), 155))
    add(("817 + 138", sum(hard(r) for r in dual) + sum(hard(r) for r in first), 955))

    add(("P(hard | exponent flip), pooled",
         round(wilson(sum(hard(r) for r in exp), len(exp))[0], 1), 34.1))
    de = [r for r in dual if r["site"]["stratum"] == "fp16_scale_exponent"]
    add(("P(hard | exponent flip), dual-stratified",
         round(wilson(sum(hard(r) for r in de), len(de))[0], 1), 35.5))

    # the tolerance boundary moves with model size: the finding, bit by bit
    for camp, lbl, want in ((V2, "0.5B", {4: 99.0, 3: 62.8, 2: 2.1, 1: 0.0, 0: 0.0}),
                            (BIG, "1.5B", {4: 99.4, 3: 43.1, 2: 22.8, 1: 4.6, 0: 0.0})):
        g = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            if r["_camp"] != camp or r["site"]["stratum"] != "fp16_scale_exponent":
                continue
            eb = r["site"]["bit"] - 2
            g[eb][0] += 1; g[eb][1] += hard(r)
        for eb, w in sorted(want.items(), reverse=True):
            n, k = g[eb]
            add((f"{lbl}, exponent bit {eb}, hard-failure rate",
                 round(wilson(k, n)[0], 1), w))
    # and the non-overlap that makes it a finding rather than a trend
    a = [r for r in rows if r["_camp"] == V2 and r["site"]["stratum"] == "fp16_scale_exponent"
         and r["site"]["bit"] - 2 == 2]
    b = [r for r in rows if r["_camp"] == BIG and r["site"]["stratum"] == "fp16_scale_exponent"
         and r["site"]["bit"] - 2 == 2]
    add(("bit-2 intervals do not overlap between model sizes",
         round(wilson(sum(hard(r) for r in a), len(a))[2], 1)
         < round(wilson(sum(hard(r) for r in b), len(b))[1], 1), True))

    with open(os.path.join(DATA, FIRST, "scale_ranges.json"), encoding="utf-8") as fh:
        ranges = json.load(fh)
    add(("every file's maximum biased exponent is under 16, so the top bit is never set",
         max(ranges.values()) < 16, True))
    add(("scale exponent ranges run 8 to 13",
         (min(ranges.values()), max(ranges.values())), (8, 13)))

    g2 = guard(rows, {V2}, ranges)
    add(("range check on the dual-stratified 0.5B campaign, caught",
         (g2["tp"], g2["tp"] + g2["fn"]), (354, 515)))
    add(("range check detection rate",
         round(wilson(g2["tp"], g2["tp"] + g2["fn"])[0], 1), 68.7))
    add(("range check precision", round(wilson(g2["tp"], g2["tp"] + g2["fp"])[0], 1), 81.9))
    add(("flips of the most significant exponent bit caught",
         (g2["bit4_caught"], g2["bit4_cat"]), (310, 310)))

    g1 = guard(rows, {FIRST}, ranges)
    add(("the retired figure, restated under the strict criterion",
         round(wilson(g1["tp"], g1["tp"] + g1["fn"])[0], 1), 81.2))
    add(("the retired figure does not replicate: intervals do not overlap",
         round(wilson(g1["tp"], g1["tp"] + g1["fn"])[1], 1)
         > round(wilson(g2["tp"], g2["tp"] + g2["fn"])[2], 1), True))

    gb = guard(rows, {FIRST, V2}, ranges)
    add(("corruption is one-directional: upward flips that were hard failures",
         gb["up_cat"], 653))
    add(("corruption is one-directional: downward flips", gb["down"], 526))
    add(("corruption is one-directional: hard failures among them", gb["down_cat"], 0))

    big = per_file_rate(rows, BIG, BIG)
    small = per_file_rate(rows, V2, FIRST)
    add(("format spread at 1.5B", round(max(big.values()) / min(big.values()), 1), 13.5))
    add(("format spread at 0.5B", round(max(small.values()) / min(small.values()), 1), 2.6))
    add(("1.5B Q4_0 predicted rate per random bit flip", round(big["Q4_0"], 3), 1.085))
    add(("1.5B Q6_K predicted rate per random bit flip", round(big["Q6_K"], 3), 0.080))

    cost = []
    for d in (FIRST, BIG):
        with open(os.path.join(DATA, d, "faultscope_census.json"), encoding="utf-8") as fh:
            cost += [v["exponent_bit_pct"] * 2 / 5 for v in json.load(fh).values()]
    add(("cost of hardening the top two exponent bits, cheapest file",
         round(min(cost), 2), 0.12))
    add(("cost of hardening the top two exponent bits, dearest file",
         round(max(cost), 2), 1.13))

    edge = [r for r in rows if r.get("catastrophic") and not hard(r)
            and r["site"]["stratum"] != "fp16_scale_exponent"]
    add(("injections outside the exponent field crossing only the top-1 clause", len(edge), 1))
    add(("...and it is a sign flip in a Q5_1 tensor",
         (edge[0]["site"]["tensor_type"], edge[0]["site"]["stratum"]) if edge else None,
         ("Q5_1", "fp16_scale_sign")))
    add(("...moving this share of token predictions",
         round(edge[0]["divergence"]["top1_diff_rate"] * 100, 1) if edge else None, 11.3))
    return C


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print every claim, not only failures")
    a = ap.parse_args()

    rows = load()
    if not rows:
        print("no injection records found under data/")
        return 1
    checks = build(rows)

    bad = 0
    for claim, got, want in checks:
        ok = (abs(got - want) <= 0.05) if isinstance(want, float) else (got == want)
        if a.list:
            print(f"  {'ok ' if ok else 'BAD'}  {claim:<62} {got!r}")
        if not ok:
            bad += 1
            if not a.list:
                print(f"  MISMATCH  {claim}")
                print(f"            re-derived {got!r}, the documents say {want!r}")
    print(f"\n{len(checks)} published claims re-derived from data/, "
          f"{len(checks) - bad} confirmed, {bad} mismatched")
    if bad:
        print("\nEither the data changed or a document is wrong. Do not publish until this is zero.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
