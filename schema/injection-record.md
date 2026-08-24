# The injection record

One JSON object per line, one line per injection. Everything needed to recompute any number in
the analysis, and enough provenance to interpret a row years later.

```json
{
  "site": {
    "abs_offset": 483241, "bit": 7,
    "tensor": "blk.3.ffn_down.weight", "tensor_type": "Q4_K",
    "block_index": 1042, "field": "d",
    "stratum": "fp16_scale_exponent", "blast": 256
  },
  "divergence": {
    "top1_diff_count": 44, "top1_diff_rate": 0.4536,
    "first_divergence_position": 3,
    "delta_nll": 6.2114, "ppl_ratio": 501.3, "non_finite": false
  },
  "catastrophic": true,
  "dirty": {
    "nll": 10.1055, "ppl": 24471.2, "finite": true,
    "n_positions": 97, "load_s": 0.58, "eval_s": 0.14,
    "argmax_digest": "9f2c1ab4de07c115"
  }
}
```

## site

Where the bit was, described so it can be found again without the tool.

| field | meaning |
|---|---|
| `abs_offset` | byte offset from the start of the file |
| `bit` | 0 to 7 within that byte, least significant first |
| `tensor` | GGUF tensor name |
| `tensor_type` | the ggml block type of that tensor, which is not always the file's label |
| `block_index` | which block within the tensor |
| `field` | the field within the block layout: `d`, `dmin`, `scales`, `qs` and so on |
| `stratum` | structural role, one of six. See below |
| `blast` | weights whose dequantized value changes, from the structural model |

**Strata.** `fp16_scale_sign`, `fp16_scale_exponent`, `fp16_scale_mantissa`, `packed_scale`,
`int_scale`, `payload`. The three fp16 strata split a 16-bit scale by IEEE binary16 field: bit
15 sign, bits 14 to 10 exponent, bits 9 to 0 mantissa. Within the exponent the bit index
matters and is recoverable: for a site in the high byte, exponent bit `i` is `bit - 2`, and
flipping it changes the scale by `2^(2^i)`.

## divergence

The corrupted run against the clean one, over the same fixed probe.

| field | meaning |
|---|---|
| `top1_diff_count` | positions where the predicted token changed |
| `top1_diff_rate` | that, over the number of scored positions |
| `first_divergence_position` | index of the first disagreement, `null` if none |
| `delta_nll` | mean negative log likelihood, corrupted minus clean, in nats |
| `ppl_ratio` | corrupted perplexity over clean. This is the severity measure |
| `non_finite` | whether any logit became NaN or infinite |

`ppl_ratio` spans 1.0 to about 1e86 and infinity in the data here. Anything reading it should
handle both.

## catastrophic

A derived boolean, and the threshold is stated so it can be changed. True when any logit is
non-finite, **or** perplexity at least doubles, **or** the top-1 divergence rate exceeds three
times the measured within-backend determinism floor. The threshold was fixed before any numbers
were seen. With a floor of zero the third clause never binds, so in practice it is a doubling
of perplexity.

## dirty

The corrupted run's own measurements, so a row is interpretable without the clean baseline.
`argmax_digest` is a short hash of the predicted token sequence, which makes two rows
comparable without storing 97 integers each.

## Files beside it

| file | what |
|---|---|
| `*.meta.json` | the determinism floor, the quota, the seed, and the campaign's own counts |
| `manifest.json` | model source, quantization provenance, probe text, backend, seed |
| `determinism_floor.json` | repeats per backend per format, and CPU against GPU |
| `faultscope_census.json` | per-file structural exposure, and any unmodelled tensor types |
| `plan_audit-*.json` | which cells the sampler was going to fill, recorded before the run |
| `nvml_preflight*.json` | which NVML queries the device answered, and the polling rate sweep |

## The one thing to check before quoting a per-format number

`plan_audit-*.json` says how many exponent sites each block type drew. A block type with fewer
than about thirty cannot carry a rate, and a zero from it means "never sampled" rather than
"never failed".

This is not hypothetical. The first campaign stratified by role alone and gave Q4_K a hundred
packed-scale sites and zero exponent sites, which produced a clean-looking and entirely
spurious per-format result. `make_tables.py` prints the audit for that reason and refuses to
report a rate for a cell that could not have one.
