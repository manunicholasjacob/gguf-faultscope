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

**Which exponent bits matter moves with model size.** A 256-fold change in one block's scale
breaks the model about half the time at either size measured. A sixteenfold change is nearly
harmless at 0.5B, at 2.1 percent [1.0, 4.6], and breaks the 1.5B model 22.8 percent of the time
[17.2, 29.7]. The intervals do not overlap. That was the open question the repository was built
to ask, and the answer is that the threshold moves down as the model grows.

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
measured, so the range check is scored on the 0.5B files alone, and the per-bit table says it
will catch less on a larger model rather than more. The first campaign
(`data/p100-qwen2.5-0.5b`) stratified by structural role only and must not be used for any
per-format number; it is kept because it carries the determinism floor and the scale ranges.

Contributions of runs from other hardware are the point of the repository. See CONTRIBUTING.md.
