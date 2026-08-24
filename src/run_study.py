"""The campaign driver: inject one bit, measure what it did to the model, put it back.

Design decisions worth knowing before reading the code.

**One forward pass, not a generation.** The obvious way to measure "did the output change"
is to generate tokens and compare. That costs a sequential decode per injection and a
thousand-injection campaign would take days. Instead this runs a single prefill over a fixed
token sequence with `logits_all=True`, which yields the model's prediction at every position
at once. From that one array you get three things: the negative log likelihood, so
perplexity; the argmax at every position, so top-1 divergence against the clean run; and
whether anything went non-finite. Roughly two orders of magnitude cheaper than generating.

**The model is reloaded every time, with mmap off.** llama.cpp memory-maps the weight file
by default, so flipping a byte underneath a loaded model does something undefined depending
on whether that page had been faulted in. Reloading with `use_mmap=False` forces a full
read and makes each measurement honest. Load time then dominates the campaign, which is
fine and is the price of not lying.

**The clean run is measured more than once.** Before any injection, the same file is
measured `--repeats` times. If those disagree, the backend is non-deterministic and every
downstream comparison needs a noise floor rather than an equality test. Reporting that floor
is part of the result, not a caveat on it.

**Nothing is thrown away.** Every injection writes a row whether it did something or not.
The distribution is the finding; a mean would hide it.

Run it:

    python run_study.py --model qwen.gguf --out results.jsonl
    python run_study.py --dry-run --n 5        # exercises the whole pipeline, no llama.cpp

**Stratification is by block type as well as by structural role, and that is not optional by
accident.** The first campaign stratified by role alone, and a block type that is a small share
of a file drew no exponent sites at all, so its zero percent catastrophe rate meant "never
asked" rather than "never failed". A per-format comparison built on that is spurious in a way
that survives a careless read. `run()` now prints the plan audit before flipping anything and
warns about any cell that cannot carry a rate. `--by-role-only` reproduces the old behaviour
and exists for that reason alone.

`--dry-run` swaps in a deterministic mock scorer that reads the actual file bytes, so the
plumbing, the stratification, the restore checking and the analysis are all exercised
locally without a model or a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from gguf_faultscope import LAYOUTS, blast_profile
from gguf_inject import GGUF, Site, Stratum, inject, plan, plan_audit, repair

# A fixed evaluation text. Short enough that a forward pass is quick, varied enough that a
# damaged model has somewhere to show it. Replace with a corpus slice for a real campaign
# and record which one in the manifest.
DEFAULT_PROBE = (
    "The memory bandwidth of a system determines how quickly weights can be streamed. "
    "When a model is quantized, several weights share one scale factor. "
    "A single bit flip in that scale changes every weight in the block at once. "
    "The question is whether the model notices. "
    "Large networks tolerate small perturbations because they are statistically redundant, "
    "but that tolerance is incidental rather than designed, and it does not extend to every "
    "part of the representation equally. "
    "Measurement is the only way to tell which parts matter."
)


# ------------------------------------------------------------------ scorers

@dataclass
class Score:
    """What one forward pass over the probe produced."""

    nll: float                # mean negative log likelihood, nats per token
    ppl: float
    argmax: List[int]         # predicted token at every position
    finite: bool
    n_positions: int
    load_s: float
    eval_s: float

    def compact(self) -> Dict:
        d = asdict(self)
        d.pop("argmax")
        d["argmax_digest"] = hashlib.sha1(
            ",".join(map(str, self.argmax)).encode()).hexdigest()[:16]
        return d


class LlamaScorer:
    """Real scoring through llama-cpp-python."""

    def __init__(self, model_path: str, n_gpu_layers: int, n_ctx: int, threads: int,
                 probe: str):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.threads = threads
        self.probe = probe
        self._tokens: Optional[List[int]] = None

    def _tokenize_once(self) -> List[int]:
        if self._tokens is None:
            from llama_cpp import Llama
            llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx, n_gpu_layers=0,
                        logits_all=False, verbose=False, use_mmap=False)
            self._tokens = llm.tokenize(self.probe.encode("utf-8"), add_bos=True)
            del llm
        return self._tokens

    def score(self) -> Score:
        import numpy as np
        from llama_cpp import Llama

        tokens = self._tokenize_once()
        t0 = time.perf_counter()
        llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers, n_threads=self.threads,
                    logits_all=True, verbose=False, use_mmap=False, seed=0)
        t1 = time.perf_counter()
        llm.reset()
        llm.eval(tokens)
        logits = np.asarray(llm.scores[: len(tokens)], dtype=np.float64)
        t2 = time.perf_counter()

        finite = bool(np.all(np.isfinite(logits)))
        argmax = logits.argmax(axis=-1).astype(int).tolist()

        # Predict position i+1 from the logits at position i.
        nlls = []
        if finite:
            for i in range(len(tokens) - 1):
                row = logits[i]
                m = row.max()
                lse = m + math.log(float(np.exp(row - m).sum()))
                nlls.append(float(lse - row[tokens[i + 1]]))
        mean_nll = statistics.fmean(nlls) if nlls else float("inf")

        del llm
        return Score(nll=mean_nll,
                     ppl=math.exp(mean_nll) if mean_nll < 700 else float("inf"),
                     argmax=argmax[: len(tokens) - 1], finite=finite,
                     n_positions=max(0, len(tokens) - 1),
                     load_s=round(t1 - t0, 3), eval_s=round(t2 - t1, 3))


class MockScorer:
    """A deterministic stand-in that actually reads the file, for testing the pipeline.

    It hashes a window of bytes around each of a fixed set of probe offsets and turns that
    into a pseudo-logit. Flipping a bit inside a window changes the result; flipping one
    outside it does not. That is enough to exercise every code path end to end, and it is
    obviously not a model, which is the point.
    """

    def __init__(self, model_path: str, n_positions: int = 64, window: int = 4096):
        self.model_path = model_path
        self.n_positions = n_positions
        self.window = window
        size = os.path.getsize(model_path)
        step = max(1, size // (n_positions + 1))
        self.offsets = [min(size - 1, (i + 1) * step) for i in range(n_positions)]

    def score(self) -> Score:
        t0 = time.perf_counter()
        vals = []
        with open(self.model_path, "rb") as fh:
            for off in self.offsets:
                start = max(0, off - self.window // 2)
                fh.seek(start)
                chunk = fh.read(self.window)
                h = hashlib.blake2b(chunk, digest_size=8).digest()
                vals.append(int.from_bytes(h, "big"))
        t1 = time.perf_counter()
        argmax = [v % 32000 for v in vals]
        nll = 2.0 + (vals[0] % 1000) / 10000.0
        return Score(nll=nll, ppl=math.exp(nll), argmax=argmax, finite=True,
                     n_positions=len(argmax), load_s=0.0, eval_s=round(t1 - t0, 4))


# ------------------------------------------------------------------ comparison

def divergence(clean: Score, dirty: Score) -> Dict:
    """How far the corrupted run moved, in the three ways that matter."""
    n = min(len(clean.argmax), len(dirty.argmax))
    diffs = [i for i in range(n) if clean.argmax[i] != dirty.argmax[i]]
    first = diffs[0] if diffs else None
    d_nll = dirty.nll - clean.nll if (dirty.finite and clean.finite) else None
    ppl_ratio = (dirty.ppl / clean.ppl) if (d_nll is not None and clean.ppl > 0) else None
    return {
        "top1_diff_count": len(diffs),
        "top1_diff_rate": round(len(diffs) / n, 6) if n else None,
        "first_divergence_position": first,
        "delta_nll": round(d_nll, 6) if d_nll is not None else None,
        "ppl_ratio": round(ppl_ratio, 6) if ppl_ratio is not None else None,
        "non_finite": not dirty.finite,
    }


def catastrophic(div: Dict, floor: Dict, ppl_threshold: float = 2.0) -> bool:
    """A deviation counts only if it clears the backend's own noise floor."""
    if div["non_finite"]:
        return True
    if div["ppl_ratio"] is not None and div["ppl_ratio"] >= ppl_threshold:
        return True
    rate = div["top1_diff_rate"]
    if rate is None:
        return False
    return rate > max(0.10, 3.0 * floor.get("top1_diff_rate", 0.0))


# ------------------------------------------------------------------ campaign

def measure_floor(scorer, repeats: int) -> Tuple[Score, Dict]:
    """Measure the clean file several times so later comparisons have a noise floor."""
    scores = [scorer.score() for _ in range(repeats)]
    base = scores[0]
    rates, dnlls = [], []
    for s in scores[1:]:
        d = divergence(base, s)
        rates.append(d["top1_diff_rate"] or 0.0)
        dnlls.append(abs(d["delta_nll"] or 0.0))
    floor = {
        "repeats": repeats,
        "top1_diff_rate": round(max(rates), 6) if rates else 0.0,
        "delta_nll": round(max(dnlls), 6) if dnlls else 0.0,
        "deterministic": bool(rates) and max(rates) == 0.0,
        "clean_ppl": round(base.ppl, 6),
        "clean_nll": round(base.nll, 6),
        "n_positions": base.n_positions,
        "mean_load_s": round(statistics.fmean(s.load_s for s in scores), 3),
        "mean_eval_s": round(statistics.fmean(s.eval_s for s in scores), 3),
    }
    return base, floor


# Round two established that five of the six strata are null: 1,900 injections outside the
# fp16 exponent produced zero catastrophes. Confirming a null needs fewer samples than
# estimating a rate, so the exponent gets the budget and the rest get a confirmation sample.
DEFAULT_QUOTA = {"fp16_scale_exponent": 100, "fp16_scale_sign": 25,
                 "fp16_scale_mantissa": 25, "packed_scale": 25,
                 "int_scale": 25, "payload": 25}


def run(model: str, out: str, n, seed: int, strata: Sequence[Stratum],
        scorer, repeats: int, resume: bool, exclude_output: bool,
        limit: Optional[int], by_block_type: bool = True,
        min_blocks: int = 64) -> Dict:
    g = GGUF(model)
    summ = g.summary()
    print(f"model      {os.path.basename(model)}")
    print(f"tensors    {summ['n_tensors']}, data {summ['data_start']} to {summ['data_end']}")
    print(f"types      {', '.join(sorted(summ['by_type']))}")

    done = set()
    if resume and os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["site"]["abs_offset"])
                except Exception:
                    continue
        print(f"resume     {len(done)} sites already recorded")

    print("\nmeasuring the clean baseline...")
    clean, floor = measure_floor(scorer, repeats)
    print(f"  clean ppl {floor['clean_ppl']}  over {floor['n_positions']} positions")
    print(f"  load {floor['mean_load_s']} s, eval {floor['mean_eval_s']} s per measurement")
    if floor["deterministic"]:
        print("  backend is deterministic across repeats")
    else:
        print(f"  NOT deterministic: top-1 differs at up to "
              f"{floor['top1_diff_rate']:.4f} of positions between identical runs.")
        print("  every comparison below is against that floor, not against equality.")

    audit = plan_audit(g, strata, n=n, seed=seed, by_block_type=by_block_type,
                       min_blocks=min_blocks, exclude_output=exclude_output)
    print()
    print("plan audit, before anything is flipped:")
    for c in audit["cells"]:
        flag = "" if c["estimable"] else "   NOT ESTIMABLE, a zero here means never sampled"
        print(f"  {c['block_type']:<9} blocks {c['blocks']:>9}  sites {c['sites']:>5}  "
              f"exponent {c['exponent_sites']:>4}{flag}")
    for s in audit["skipped_block_types"]:
        print(f"  {s['block_type']:<9} skipped: {s['reason']}")
    if audit["not_estimable"]:
        print(f"  WARNING: {audit['not_estimable']} will not support a rate. Raise the "
              f"exponent quota or lower --min-blocks, or do not report them.")
    with open(out.replace(".jsonl", "") + ".plan_audit.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    sites = plan(g, strata, n=n, seed=seed, exclude_output=exclude_output,
                 by_block_type=by_block_type, min_blocks=min_blocks)
    sites = [s for s in sites if s.abs_offset not in done]
    if limit:
        sites = sites[:limit]
    print(f"\n{len(sites)} injections to run")

    per_site = floor["mean_load_s"] + floor["mean_eval_s"]
    print(f"estimated {len(sites) * per_site / 60:.1f} minutes\n")

    log = out + ".repair"
    n_cat = 0
    t_start = time.perf_counter()
    with open(out, "a", encoding="utf-8") as fh:
        for i, site in enumerate(sites, 1):
            with inject(model, site, guard=g, repair_log=log):
                dirty = scorer.score()
            div = divergence(clean, dirty)
            cat = catastrophic(div, floor)
            n_cat += cat
            fh.write(json.dumps({
                "site": site.as_dict(),
                "divergence": div,
                "catastrophic": cat,
                "dirty": dirty.compact(),
            }) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(sites):
                rate = (time.perf_counter() - t_start) / i
                left = (len(sites) - i) * rate / 60
                print(f"  {i}/{len(sites)}  catastrophic {n_cat}  "
                      f"{rate:.1f} s/site  {left:.0f} min left")

    meta = {
        "model": os.path.basename(model),
        "model_summary": summ,
        "seed": seed,
        "quota": n,
        "by_block_type": by_block_type,
        "min_blocks": min_blocks,
        "plan_audit": audit,
        "strata": [s.value for s in strata],
        "floor": floor,
        "injections": len(sites),
        "catastrophic": n_cat,
        "scorer": type(scorer).__name__,
    }
    with open(out.replace(".jsonl", "") + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


# ------------------------------------------------------------------ analysis

def analyze(path: str) -> str:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return "no rows"

    by: Dict[str, List[Dict]] = {}
    for r in rows:
        by.setdefault(r["site"]["stratum"], []).append(r)

    order = [s.value for s in Stratum.all()]
    lines = [f"{len(rows)} injections", ""]
    head = (f"{'stratum':<24} {'n':>4} {'blast':>6} {'catas%':>7} "
            f"{'top1 med':>9} {'top1 p95':>9} {'dppl med':>9}")
    lines += [head, "-" * len(head)]
    for s in order:
        rs = by.get(s)
        if not rs:
            continue
        rates = sorted(r["divergence"]["top1_diff_rate"] or 0.0 for r in rs)
        ratios = sorted(r["divergence"]["ppl_ratio"] or float("inf") for r in rs)
        finite_ratios = [x for x in ratios if math.isfinite(x)]
        cat = sum(1 for r in rs if r["catastrophic"])
        blast = statistics.fmean(r["site"]["blast"] for r in rs)
        p95 = rates[min(len(rates) - 1, int(0.95 * len(rates)))]
        lines.append(f"{s:<24} {len(rs):>4} {blast:>6.0f} {100*cat/len(rs):>6.1f}% "
                     f"{statistics.median(rates):>9.4f} {p95:>9.4f} "
                     f"{statistics.median(finite_ratios) if finite_ratios else float('nan'):>9.4f}")

    by_type: Dict[str, List[Dict]] = {}
    for r in rows:
        by_type.setdefault(r["site"]["tensor_type"], []).append(r)
    if len(by_type) > 1:
        lines += ["", "by quantization format", ""]
        h2 = f"{'format':<10} {'n':>4} {'bpw':>7} {'wide%':>7} {'catas%':>7} {'top1 med':>9}"
        lines += [h2, "-" * len(h2)]
        for t, rs in sorted(by_type.items()):
            lay = LAYOUTS.get(t)
            prof = blast_profile(lay).as_dict() if lay else {}
            rates = [r["divergence"]["top1_diff_rate"] or 0.0 for r in rs]
            cat = sum(1 for r in rs if r["catastrophic"])
            lines.append(f"{t:<10} {len(rs):>4} {prof.get('bits_per_weight', 0):>7.3f} "
                         f"{prof.get('wide_bit_pct', 0):>6.2f}% {100*cat/len(rs):>6.1f}% "
                         f"{statistics.median(rates):>9.4f}")
    return "\n".join(lines)


# ------------------------------------------------------------------ cli

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    ap.add_argument("--out", default="injections.jsonl")
    ap.add_argument("--n", type=int, default=None,
                    help="sites per cell. Omit to use the asymmetric default, which spends "
                         "the budget on the exponent stratum where the effect lives")
    ap.add_argument("--by-role-only", action="store_true",
                    help="stratify by structural role alone. This is how the first campaign "
                         "was run and it produced a spurious per-format result, because a "
                         "block type that is a small share of a file draws no exponent sites. "
                         "Present so that result can be reproduced, not because it is right")
    ap.add_argument("--min-blocks", type=int, default=64,
                    help="skip a block type with fewer blocks than this in the file")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=3, help="clean measurements for the floor")
    ap.add_argument("--limit", type=int, help="stop after this many injections")
    ap.add_argument("--gpu-layers", type=int, default=99)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--probe-file", help="text file to score against, default is built in")
    ap.add_argument("--exclude-output", action="store_true",
                    help="skip the output head, which is often a different quantization")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="mock scorer, no llama.cpp needed")
    ap.add_argument("--analyze", help="summarize an existing results file and exit")
    ap.add_argument("--repair", help="undo abandoned flips from a repair log")
    a = ap.parse_args()

    if a.analyze:
        print(analyze(a.analyze))
        return 0
    if a.repair:
        print(f"{repair(a.repair)} repaired")
        return 0

    model = a.model
    tmp = None
    if a.dry_run and not model:
        import tempfile
        from gguf_inject import _synthetic_gguf_with_data
        tmp = tempfile.mkdtemp()
        model = os.path.join(tmp, "mock.gguf")
        _synthetic_gguf_with_data(model)
        print(f"dry run against a synthetic GGUF at {model}\n")
    if not model:
        ap.print_help()
        return 1

    probe = DEFAULT_PROBE
    if a.probe_file:
        with open(a.probe_file, encoding="utf-8") as fh:
            probe = fh.read()

    scorer = (MockScorer(model) if a.dry_run
              else LlamaScorer(model, a.gpu_layers, a.ctx, a.threads, probe))

    quota = a.n if a.n is not None else DEFAULT_QUOTA
    meta = run(model, a.out, quota, a.seed, Stratum.all(), scorer,
               a.repeats, a.resume, a.exclude_output, a.limit,
               by_block_type=not a.by_role_only, min_blocks=a.min_blocks)
    print(f"\n{meta['injections']} injections, {meta['catastrophic']} catastrophic")
    print(f"\n{analyze(a.out)}")
    print(f"\nrows in {a.out}, metadata beside it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
