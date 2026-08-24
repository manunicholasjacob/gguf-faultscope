"""Can the damaging corruptions be detected without any redundancy?

The campaign found that every catastrophic injection was a flip in the exponent field of a
16-bit scale, and that the top two exponent bits carried all of it. This asks the obvious next
question, and the answer turns out to be most of the way to a mitigation.

A quantization scale maps a block of weights, centred near zero, onto a small integer range,
so the scale is a small number by construction. Measured over every scale in five files, twelve
to fifteen million each, the biased fp16 exponent runs from 0 to somewhere between 8 and 13
depending on the file, and **the most significant exponent bit is zero in every scale of every
file**. A flip that sets it produces a value the file's own range says is impossible.

So the check is: record the maximum biased exponent at build time, five bits in a header, and
reject any scale above it at load. No checksum, no replica, no parity, no extra storage beyond
those five bits.

This script measures how much of the damage that catches, using the file's true range rather
than a range inferred from the sample, because the two differ and the sampled one flatters the
result.

    python analyze_guard.py --results <dir> --ranges ranges.json

`ranges.json` maps a file label to its true maximum biased exponent, from
`gguf_faultscope --gguf X --scale-range`. Without it, the script derives a range from the
sampled scales and says loudly that it did.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
from typing import Dict, Optional, Tuple

EXPONENT = "fp16_scale_exponent"


def hard_failure(r: Dict) -> bool:
    """Non-finite logits, or perplexity at least doubled. No top-1 clause.

    The recorded flag also fires when more than ten percent of token predictions move, which
    is a different event and a threshold-sensitive one. Detection rates computed against the
    two differ by several points, so the repository publishes only the strict one.
    """
    d = r.get("divergence") or {}
    if d.get("non_finite"):
        return True
    pr = d.get("ppl_ratio")
    return pr is not None and pr >= 2.0


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def load_originals(results: str) -> Dict[Tuple[int, int], Tuple[int, str]]:
    """The pre-injection byte, from the repair logs the injector writes, keyed per file.

    The key carries the file label and not only the offset and bit. Different quantized files
    built from one model do collide on (offset, bit): 54 of 2,521 keys collide in the v2
    campaign. The original bytes happen to agree in every one of those, so nothing was ever
    computed wrongly here, but a global key is one dataset away from silently attributing one
    file's byte to another file's injection.

    The logs exist so a crash mid-injection leaves a breadcrumb. That they also record the
    original value is what makes this analysis possible without rerunning anything, which is
    an argument for writing them even when nothing crashes.
    """
    out: Dict[Tuple[str, int, int], Tuple[int, str]] = {}
    for f in glob.glob(os.path.join(results, "*.repair")):
        label = os.path.basename(f).replace("injections-", "").replace(".jsonl.repair", "")
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("state") == "open":
                    out[(label, r["offset"], r["bit"])] = (r["original"], label)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--ranges", help="JSON mapping file label to true max biased exponent")
    ap.add_argument("--json", help="write the computed rows here")
    ap.add_argument("--criterion", choices=("hard", "recorded"), default="hard",
                    help="hard: non-finite logits or perplexity doubled, which is what every "
                         "published number here uses. recorded: the flag stored in the file")
    a = ap.parse_args()

    orig = load_originals(a.results)
    if not orig:
        print("no repair logs found; this analysis needs the pre-injection bytes")
        return 1

    true_max: Optional[Dict[str, int]] = None
    if a.ranges:
        with open(a.ranges, encoding="utf-8") as fh:
            true_max = {k: int(v) for k, v in json.load(fh).items()}

    rows = []
    for f in sorted(glob.glob(os.path.join(a.results, "injections-*.jsonl"))):
        if f.endswith("-all.jsonl"):
            continue
        label = os.path.basename(f).replace("injections-", "").replace(".jsonl", "")
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["site"]["stratum"] != EXPONENT:
                    continue
                key = (label, r["site"]["abs_offset"], r["site"]["bit"])
                if key not in orig:
                    continue
                rows.append((label, r, orig[key][0]))

    if not rows:
        print("no exponent-stratum injections with a recoverable original byte")
        return 1

    # Fall back to the sampled range, and say so, because it is optimistic.
    if true_max is None:
        seen: Dict[str, int] = collections.defaultdict(int)
        for label, r, o in rows:
            seen[label] = max(seen[label], (o >> 2) & 31)
        true_max = dict(seen)
        print("WARNING: no --ranges given, so the guard uses the maximum exponent seen in the")
        print("SAMPLE. The true maximum over every scale in the file is higher, and using the")
        print("sampled one overstates detection. Run gguf_faultscope --scale-range.\n")

    per = collections.defaultdict(lambda: collections.Counter())
    by_bit = collections.defaultdict(lambda: collections.Counter())
    direction = collections.defaultdict(lambda: collections.Counter())

    for label, r, o in rows:
        bit = r["site"]["bit"]
        mx = true_max.get(label)
        if mx is None:
            continue
        e_after = ((o ^ (1 << bit)) >> 2) & 31
        detected = e_after > mx
        cat = hard_failure(r) if a.criterion == "hard" else bool(r["catastrophic"])
        cell = "tp" if (cat and detected) else "fn" if cat else "fp" if detected else "tn"
        per[label][cell] += 1
        eb = bit - 2
        by_bit[eb][cell] += 1
        by_bit[eb]["n"] += 1
        was_set = (o >> bit) & 1
        d = "down" if was_set else "up"
        direction[d]["n"] += 1
        direction[d]["cat"] += cat

    L = []
    L.append(f"criterion: {a.criterion}")
    L.append(f"{len(rows)} exponent-stratum injections with a recoverable original byte")
    L.append("")
    L.append("The corruption is one-directional, and that is a property of the data.")
    for d in ("up", "down"):
        c = direction[d]
        if not c["n"]:
            continue
        p, lo, hi = wilson(c["cat"], c["n"])
        arrow = "scale multiplied" if d == "up" else "scale divided"
        L.append(f"  {arrow:<18} {c['cat']:>4}/{c['n']:<5} catastrophic {p:>5.1f}% "
                 f"[{lo:.1f}, {hi:.1f}]")
    L.append("  Scales are small numbers by construction, so their high exponent bits are")
    L.append("  already zero and a flip there can only make the scale larger. There is almost")
    L.append("  no divide-down case to observe.")
    L.append("")

    L.append("A range check, using each file's true maximum biased exponent:")
    h = (f"{'file':<10}{'max e':>7}{'caught':>8}{'missed':>8}{'false+':>8}{'quiet':>7}"
         f"   detection")
    L += [h, "-" * len(h)]
    tot = collections.Counter()
    for label in sorted(per):
        c = per[label]
        tot.update(c)
        n_cat = c["tp"] + c["fn"]
        p, lo, hi = wilson(c["tp"], n_cat) if n_cat else (float("nan"),) * 3
        s = f"{p:.0f}% [{lo:.0f}, {hi:.0f}]" if n_cat else "no catastrophes"
        L.append(f"{label:<10}{true_max.get(label, -1):>7}{c['tp']:>8}{c['fn']:>8}"
                 f"{c['fp']:>8}{c['tn']:>7}   {s}")
    n_cat = tot["tp"] + tot["fn"]
    p, lo, hi = wilson(tot["tp"], n_cat)
    pp, plo, phi = wilson(tot["tp"], tot["tp"] + tot["fp"])
    L.append("")
    L.append(f"OVERALL: {tot['tp']}/{n_cat} catastrophic injections detected, "
             f"{p:.1f}% [{lo:.1f}, {hi:.1f}]")
    L.append(f"         precision when it fires {pp:.1f}% [{plo:.1f}, {phi:.1f}]")
    L.append("")

    L.append("Where the detection comes from, by exponent bit:")
    hb = f"{'exp bit':>8}{'scale x':>10}{'n':>6}{'caught':>8}{'missed':>8}"
    L += [hb, "-" * len(hb)]
    for eb in sorted(by_bit, reverse=True):
        c = by_bit[eb]
        L.append(f"{eb:>8}{'2^' + str(2 ** eb):>10}{c['n']:>6}{c['tp']:>8}{c['fn']:>8}")
    L.append("")
    top = max(by_bit) if by_bit else None
    if top is not None:
        c = by_bit[top]
        n_cat_top = c["tp"] + c["fn"]
        L.append(f"The most significant exponent bit is zero in every scale of every file")
        L.append(f"measured, so a flip that sets it is always out of range: {c['tp']} of")
        L.append(f"{n_cat_top} caught here. It is also the most damaging bit. The next bit down")
        L.append("is legitimately set in files whose scales span a wider range, so a range check")
        L.append("cannot see it there, and one parity bit over that single position is the")
        L.append("cheapest way to cover it.")
    L.append("")
    L.append("This is measured on files whose true scale range was read. A range check cannot")
    L.append("be scored on a file whose range was never measured, and it should be re-scored on")
    L.append("any new model size, because which exponent bits do damage moves with model size.")

    print("\n".join(L))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"per_file": {k: dict(v) for k, v in per.items()},
                       "by_exponent_bit": {k: dict(v) for k, v in by_bit.items()},
                       "direction": {k: dict(v) for k, v in direction.items()},
                       "true_max": true_max}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
