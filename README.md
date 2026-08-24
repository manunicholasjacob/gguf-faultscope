# gguf-faultscope

What one flipped bit does to a quantized language model, and where it has to land to matter.

Silent data corruption research measures fault effects on models held in 32-bit or 16-bit
floating point, because that is how models are trained. Deployed inference does not use those
formats. It uses block quantization, where 32 or 256 weights share one scale stored as a
16-bit float, and that sharing changes what a single flipped bit can reach.

This repository holds the tool that computes the exposure from a file header, the harness that
measures it, and the data from the campaigns that have been run so far.

## The result, in one table

6,725 single-bit injections across three campaigns on a Tesla P100, into ten quantized files
built from two source models. A hard failure is non-finite logits or a doubling of perplexity
against a measured determinism floor of exactly zero.

| Structural role | n | Weights reached | Hard failures | 95% interval |
|---|---:|---:|---:|---|
| fp16 scale, exponent | 2,800 | 32 or 256 | **34.1%** | [32.4, 35.9] |
| fp16 scale, sign | 1,075 | same as exponent | 0.0% | [0.0, 0.36] |
| fp16 scale, mantissa | 1,075 | same as exponent | 0.0% | [0.0, 0.36] |
| packed sub-block scale | 350 | 32 | 0.0% | [0.0, 1.09] |
| int8 sub-block scale | 350 | 16 | 0.0% | [0.0, 1.09] |
| packed weight value | 1,075 | 1 | 0.0% | [0.0, 0.36] |

All 955 hard failures landed in an exponent field. The other 3,925 injections produced none,
including 2,150 into the sign and mantissa bits of the same scales, which reach exactly the
same weights. Reach is not what decides it.

This table pools all three campaigns because it is not a per-format claim. Every per-format and
per-bit number below is reported per campaign, for the reason given under "which campaign to
quote".

Inside the exponent it narrows further, and where it narrows to depends on the model. Flipping
exponent bit `i` multiplies or divides the block's scale by `2^(2^i)`:

| Exponent bit | Scale changes by | 0.5B, n | 0.5B hard failures | 1.5B, n | 1.5B hard failures |
|---:|---:|---:|---:|---:|---:|
| 4 | 2^16 | 313 | **99.0%** [97.2, 99.7] | 175 | **99.4%** [96.8, 99.9] |
| 3 | 2^8 | 317 | 62.8% [57.3, 67.9] | 188 | 43.1% [36.2, 50.2] |
| 2 | 2^4 | 280 | 2.1% [1.0, 4.6] | 171 | **22.8%** [17.2, 29.7] |
| 1 | 2^2 | 241 | 0.0% [0.0, 1.6] | 175 | 4.6% [2.3, 8.8] |
| 0 | 2^1 | 249 | 0.0% [0.0, 1.5] | 191 | 0.0% [0.0, 2.0] |

A 256-fold change in one block's scale breaks the model about half the time at either size. A
sixteenfold change is nearly harmless at 0.5B and breaks the 1.5B model roughly a quarter of
the time, and the intervals do not overlap. **The threshold between absorbed and not absorbed
moves down as the model grows.** That was the open question this repository was built to ask,
and two model sizes are enough to say the answer is not "it stays put". They are not enough to
say what the curve is, which is why the device table exists.

Two earlier statements of this table are superseded and worth naming, because the first
campaign's data is still shipped in `data/p100-qwen2.5-0.5b/`. It reported the lower three bits
as zero of 286, which held only at 0.5B and only at that sample size. And its headline pair,
96.9 and 52.1 percent, mixed the strict criterion with the looser one described at the bottom
of this file.

## The format you choose changes this thirteenfold

Because hard failures occur only in exponent flips, a file's exposure factors into the share of
its bits that are exponent bits, which the tool reads from the header, times the rate given such
a flip, which the campaign measures in that same file.

| Model | File | Exponent bits | Hard-failure rate per random bit flip | 95% interval |
|---|---|---:|---:|---|
| 1.5B | Q4_0 | 2.818% | **1.085%** | [0.903, 1.279] |
| 1.5B | Q8_0 | 1.838% | 0.386% | [0.260, 0.551] |
| 1.5B | Q4_K_M | 0.662% | 0.212% | [0.172, 0.257] |
| 1.5B | IQ4_XS | 0.435% | 0.164% | [0.141, 0.188] |
| 1.5B | Q6_K | 0.298% | **0.080%** | [0.057, 0.108] |

Thirteenfold, from a decision normally made on file size, across files that differ in size by
less than a factor of two. At 0.5B the same five formats spread only 2.6-fold, because the
K-quant fallback fires on the smaller model and flattens them. **The effect grows with model
size**, which is the opposite of the usual intuition.

```bash
python src/make_tables.py --results data/p100-qwen2.5-1.5b/ --census data/p100-qwen2.5-1.5b/faultscope_census.json
```

## And most of it is detectable for free

The corruption is one-directional. Over 1,900 exponent injections into the 0.5B files, all 653
hard failures were flips that *raised* a scale's exponent, and none of the 526 that lowered it
did any damage. That is structural rather than accidental: a quantization scale maps weights
centred near zero onto a small integer range, so it is a small number, and its high exponent
bits are already zero.

Reading every scale in all five 0.5B files, twelve to fifteen million each, the biased fp16
exponent runs from 0 to between 8 and 13, and **the most significant exponent bit is zero in
every scale of every file measured**. So a corrupted scale is often out of range on its face.

```bash
python src/gguf_faultscope.py --gguf your-model.gguf --scale-range
```

Store that maximum in a header, five bits, and compare at load. On the 0.5B campaign that
catches **68.7 percent of hard failures [64.6, 72.6]**, and **310 of 310** flips of the most
significant exponent bit, which is the 99.0 percent class. Precision when it fires is 81.9
percent [78.0, 85.3]: it also flags benign flips that leave a scale somewhere unusual but
harmless, which for a load-time validity check is the right way round. One parity bit over the
next bit down covers most of the remainder.

An earlier figure of 81.3 percent came from the first campaign alone and does not replicate:
that campaign drew its exponent sites overwhelmingly from the two files whose scale ranges are
narrowest, where a range check has the least work to do. The intervals do not overlap.

```bash
python src/analyze_guard.py --results data/p100-qwen2.5-0.5b-v2 --ranges data/p100-qwen2.5-0.5b/scale_ranges.json
```

No checksum, no replica, no parity for the main case. Five bits and a comparison.

The detection figure is measured on the 0.5B files only, because the true scale ranges of the
1.5B files were never read and a range check cannot be scored without them. That is the first
thing a rerun should fix, and the per-bit table above says which way the answer will move: at
1.5B a fourfold scale change does damage, and a fourfold change lands inside the file's own
range, so a range check will catch a smaller share there. Do not quote 68.7 percent for a
larger model.

To check a file against a stored range rather than only report one:

```bash
python src/gguf_faultscope.py --gguf your-model.gguf --check-range ranges.json
```

It exits non-zero and names the offending tensor and block if any scale is out of range, which
is what a deployment pipeline would actually run.

## What you can do with it in thirty seconds

```bash
pip install nvidia-ml-py     # optional, only for the power probe
python src/gguf_faultscope.py --table                  # the structural model
python src/gguf_faultscope.py --gguf your-model.gguf   # exposure of a real file
```

The second command tells you what share of your file is corruption-exposed, and it reads the
header rather than running the model. On the ten files measured here, hardening the top two
bits of every scale means protecting 0.12 to 1.13 percent of the file, which is the fallback
for the cases a range check cannot see.

It will also tell you something most people do not expect: **the format label on a GGUF does
not reliably describe its contents.** `llama-quantize` cannot apply a K-quant to a tensor whose
row length is not a multiple of 256 and falls back. A 0.5B file requested as `Q4_K_M` held 132
tensors in Q5_0 and only 12 in Q4_K. The same request on the 1.5B model delivered 168 tensors
in Q4_K and no Q5_0 at all, which is the whole reason the format spread widens with size.

## Layout

```
src/gguf_faultscope.py   block layouts, exposure model, GGUF census, range check. Self-tested.
src/gguf_inject.py       stratified single-bit injection with verified restore. Self-tested.
src/run_study.py         campaign driver: determinism floor, inject, score, analyse.
src/make_tables.py       the analysis. Read its docstring before quoting any per-format number.
src/make_figures.py      figures, and it renames one that must not be used.
src/analyze_guard.py     the range-check mitigation, scored against the repair logs.
notebooks/               the campaigns as they were run, on free Kaggle GPU time.
schema/                  the injection record, field by field.
data/<device>-<model>/   one directory per campaign. Add yours.
```

```bash
python src/gguf_faultscope.py --selftest
python src/gguf_inject.py --selftest
python src/run_study.py --dry-run --n 5 --out /tmp/t.jsonl
```

## Which campaign to quote

Three directories, and they are not interchangeable.

| Directory | n | Design | Use it for |
|---|---:|---|---|
| `p100-qwen2.5-0.5b` | 2,400 | stratified by structural role only | the determinism floor, the scale ranges, nothing per-format |
| `p100-qwen2.5-0.5b-v2` | 2,575 | by role and by block type | any 0.5B per-format number |
| `p100-qwen2.5-1.5b` | 1,750 | by role and by block type | any 1.5B per-format number |

The first campaign drew its exponent sites overwhelmingly from 32-weight formats, so a block
type that is a small share of a file drew none at all and its zero rate meant "never sampled"
rather than "never failed". Role-and-block-type stratification exists for that reason.
`run_study.py --by-role-only` reproduces the old behaviour and exists only to document it.

## Adding a device

This is the part the repository is built around. A single device is one data point; the
interesting question is whether the threshold between "absorbed" and "not absorbed" moves with
model size, layer depth or hardware, and that needs runs from machines nobody here owns.

See [CONTRIBUTING.md](CONTRIBUTING.md). It is a notebook, a change of two lines, and a pull
request with one JSONL file and one row in `data/devices.csv`.

## What this does not show

It models corruption of **stored weights**: memory, storage, or a checkpoint in transit. It
does not model a defective arithmetic unit quietly returning a wrong product, which is the
mechanism in the fleet studies.

Single bits only. One model family, two sizes. One 97-position probe. One GPU generation.

Two limits worth naming because they bound the numbers above. The 1.5B scale ranges were never
measured, so the range check is scored on 0.5B files alone. And the hard-failure criterion is
non-finite logits or a doubling of perplexity; a third clause on top-1 divergence above ten
percent is recorded in every row but excluded from these tables, because it is
threshold-sensitive and it moves the per-bit rates by several points. Exactly one injection
outside the exponent field ever crossed that looser clause: a sign flip in a Q5_1 tensor that
moved 11.3 percent of token predictions while perplexity rose 8 percent.

## Citing

See [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
