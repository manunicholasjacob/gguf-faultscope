# Adding a device

The whole point of this repository is the second row in `data/devices.csv`. One device is a
data point. The question worth answering is whether the boundary between a scale change a
network absorbs and one it does not moves with model size, layer depth or hardware, and that
needs machines nobody here owns.

If you have a GPU and half an hour, you can add one.

## What you need

- A GPU, or four CPU cores and patience. The measurement is a single forward pass, so a 0.5B
  model takes about a second per injection on a Tesla P100 and about 1.3 on four CPU cores.
- `llama-cpp-python`, `nvidia-ml-py` if you want the power probe, and a GGUF file.

## The short version

```bash
git clone https://github.com/manunicholasjacob/gguf-faultscope
cd gguf-faultscope
pip install llama-cpp-python

# 1. Check the plan before running it. Every block type you intend to report must draw
#    enough exponent sites to carry a confidence interval rather than a zero.
python - <<'PY'
import sys; sys.path.insert(0, "src")
import gguf_inject as gi, json
g = gi.GGUF("your-model.gguf")
Q = {"fp16_scale_exponent": 100, "fp16_scale_sign": 25, "fp16_scale_mantissa": 25,
     "packed_scale": 25, "int_scale": 25, "payload": 25}
print(json.dumps(gi.plan_audit(g, gi.Stratum.all(), n=Q, seed=11,
                               by_block_type=True), indent=2))
PY

# 2. Run it.
python src/run_study.py --model your-model.gguf --out data/mydevice-mymodel/injections.jsonl \
    --n 100 --seed 11 --gpu-layers 99

# 3. Analyse it.
python src/make_tables.py --results data/mydevice-mymodel/
```

Then open a pull request with the JSONL, the `.meta.json` beside it, and one row in
`data/devices.csv`.

## Three things that will bite you if you skip them

**Quantize from one source, in one session.** Files carrying the same format label behave
differently depending on how they were made; same-label GGUFs of identical size have been
measured decoding up to 38 percent apart. If you download a pre-made quantization from one
repository and compare it against one from another, you are measuring provenance rather than
format. `notebooks/preflight.ipynb` converts to F16 once and quantizes everything from that
single file, which is why it builds `llama-quantize` rather than downloading.

**Turn memory mapping off.** The runtime maps the weight file by default, so flipping a byte
underneath a loaded model does something that depends on whether that page had already been
faulted in. `run_study.py` reloads with `use_mmap=False` for every measurement. Load time then
dominates, which is the correct trade and not a performance bug.

**Measure the determinism floor first.** If two runs of the *unmodified* model disagree, every
later comparison has to be distributional rather than an equality. On the hardware measured
here the floor was exactly zero, including across CPU and GPU, but that is a property of this
runtime and this model size and it should be re-measured rather than assumed. `run_study.py`
does it automatically and refuses to interpret a sweep without it.

## What a good contribution looks like

A row in `data/devices.csv`, a directory under `data/`, and honest metadata. Specifically:

- The **exact model file**, including where it came from and how it was quantized.
- The **determinism floor** you measured, even if it is not zero. Especially if it is not zero.
- The **plan audit**, so a reader can see which block types were estimable and which were not.
- Anything that **did not work**. A GPU where the NVML energy counter is missing, a runtime
  version that changed a kernel, a format that would not quantize. Those are the useful parts.

You do not need to reproduce our numbers, and it is more interesting if you do not.

## Reporting a problem with the tool

The two modules carry self-tests that check the block layouts against the compile-time size
assertions in `ggml-common.h`. If `python src/gguf_faultscope.py --selftest` fails after a
`llama.cpp` update, a block layout has changed upstream, and that is worth an issue on its own.

If `gguf_faultscope --gguf` reports anything in `unmodelled_types` other than `F32`, the census
does not describe your whole file and the percentages are computed over a subset. That has
happened once already, silently, and it is why the field exists.
