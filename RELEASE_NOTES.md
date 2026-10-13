First release: the exposure model, the injection harness, the range check, and three campaigns.

**What it computes.** `gguf_faultscope` derives a GGUF file's corruption exposure from its
header, using the block layouts declared in `ggml-common.h`. It reports any tensor type it
cannot model rather than silently dropping it, because dropping one is how the first version
excluded 45 percent of a file without saying so. It also reads every scale in a file to report
the true exponent range, and checks a file against a range recorded at build time.

**What it measures.** `gguf_inject` plans a sample stratified by structural role and by block
type, flips one bit in place, and restores it with the byte re-read and compared. `run_study`
measures the determinism floor first, then scores one forward pass per injection. The floor came
out at exactly zero on this device, within backend and across backends, which is what makes
every post-injection divergence attributable.

**What the data shows.** In 6,725 injections across three campaigns on a Tesla P100, all 955
hard failures landed in the exponent field of a 16-bit scale. The other 3,925 produced none,
including 2,150 into the sign and mantissa bits of the same scales, which reach the same
weights. How far a bit reaches is not what decides whether it does damage.

**Which exponent bits did damage differed between the two sizes** (paragraph corrected 13 Oct
2026, see the correction below). A flip of exponent bit 3 breaks the model about half the time
at either size measured. A flip of bit 2 does so at 0.5B 2.1 percent [1.0, 4.6] of the time and
at 1.5B 22.8 percent [17.2, 29.7] of the time. The intervals do not overlap. Read these as what
happened, not as evidence that tolerance falls with model size.

**The format changes exposure thirteenfold at 1.5B**, against 2.6-fold at 0.5B, because a larger
model actually receives the format it asked for. The 0.5B file labelled Q4_K_M held 132 tensors
in Q5_0 and 12 in Q4_K; the same request on 1.5B delivered 168 in Q4_K and no Q5_0. The format
label on a GGUF does not reliably describe its contents.

**The mitigation.** The damaging corruption is one-directional: every hard failure among the
0.5B exponent injections raised a scale, and none of the flips that lowered one did anything.
The most significant exponent bit is zero in every scale of every file measured. So a five-bit
maximum recorded in the header, compared at load, catches 68.7 percent [64.6, 72.6] of hard
failures and all 310 flips of that bit, with no redundancy at all.
`gguf_faultscope --check-range` is that check as a command.

**Known limits.** Stored-weight corruption only, not compute-unit faults. Single bits. One model
family at two sizes, one 97-position probe, one device. The 1.5B scale ranges were never
measured, so the range check is scored on the 0.5B files alone and has not been scored on a
larger model. The first campaign
(`data/p100-qwen2.5-0.5b`) stratified by structural role only and must not be used for any
per-format number; it is kept because it carries the determinism floor and the scale ranges.

Contributions of runs from other hardware are the point of the repository. See CONTRIBUTING.md.

## Correction (13 Oct 2026, v0.1.2)

The v0.1.0 notes, README and schema said that flipping exponent bit `i` multiplies or divides
the block's scale by `2^(2^i)`, and read the per-bit table as "the threshold between absorbed
and not absorbed moves down as the model grows". Both statements are withdrawn as written.

- The `2^(2^i)` factor holds for a *normal* fp16 scale only. Many super-block scales in these
  files are fp16 subnormals, for which the factor is larger and depends on the stored value.
- The bit-2 and bit-1 failures at 1.5B are concentrated in one tensor whose block type differs
  between the two model sizes. Two sizes from one family cannot separate model size from the
  quantizer's per-tensor type choices, so the per-bit table is a record of what happened, not
  evidence that tolerance falls with model size.

No count, rate or interval in the per-bit table changes, and `src/verify_claims.py` checks the
same values as before. Only the interpretation is corrected. The v0.1.0 and v0.1.1 archives on
Zenodo keep the earlier wording because archived versions cannot be edited.
