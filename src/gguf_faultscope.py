"""What does one flipped bit in a quantized weight file actually break?

The premise. Silent data corruption research measures fault effects on fp32 and bf16
models, because that is what training uses. Deployed inference does not run on those. It
runs on quantized weights, and a quantized block is not a flat array of numbers. It is a
small structure: a shared scale, sometimes a shared minimum, sometimes a second tier of
packed sub-block scales, and then the payload. Those parts are not equally important, and
they are not equally sized.

So a single flipped bit does very different things depending on where it lands, and the
distribution of where it can land is fixed by the format. In Q4_0 the worst a bit can do is
rescale 32 weights. In Q4_K it can rescale 256. In Q6_K the second-tier scales are int8,
so the worst case there is a sign flip on 16 weights, while Q4_K's second tier is packed
6-bit and its first tier is fp16, where flipping the top exponent bit multiplies the whole
super-block by roughly 2^16.

None of that requires a measurement to establish. It falls out of the block layouts in
`ggml/src/ggml-common.h`, and this module encodes them so the prediction can be stated
before any hardware is touched. The measurement then checks whether output degradation
follows the structural prediction or not.

Two things in here:

* `LAYOUTS` plus `blast_profile()`, the analytical model. No file needed.
* `scan_gguf()`, which walks a real GGUF file and reports, per tensor and in total, how
  many bits sit in each structural role. That turns the per-format table into a per-model
  one, since a model is a mix of tensor types and the output head is often a different
  quantization from the repeating layers.

    python gguf_faultscope.py --table
    python gguf_faultscope.py --gguf model.gguf
    python gguf_faultscope.py --selftest

Terminology used below. **Blast radius** is the number of dequantized weights whose value
changes when one bit at a given offset is flipped. **Wide bits** are the bits whose blast
radius is greater than one. **Exponent bits** are the five exponent bits of any fp16 scale,
called out separately because they are the only place a single flip produces an unbounded
magnitude change; everywhere else the perturbation is bounded by the block's own range.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

QK_K = 256
K_SCALE_SIZE = 12


@dataclass
class Field:
    """One region of a block, and what a flipped bit in it reaches."""

    name: str
    bytes_: int
    blast: int                 # weights affected by one flipped bit in this field
    kind: str                  # "fp16_scale" | "int_scale" | "packed_scale" | "payload"
    per_element_bits: Optional[int] = None   # for packed scale tiers, bits per scale


@dataclass
class Layout:
    """A ggml block type, as declared in ggml-common.h."""

    name: str
    weights: int               # weights per block (or super-block)
    fields: List[Field]
    note: str = ""

    @property
    def block_bytes(self) -> int:
        return sum(f.bytes_ for f in self.fields)

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.block_bytes / self.weights


# Layouts transcribed from ggml/src/ggml-common.h at master, 2026-08-23.
# Sizes are the ones the file's own static_asserts enforce.
LAYOUTS: Dict[str, Layout] = {
    "Q4_0": Layout("Q4_0", 32, [
        Field("d", 2, 32, "fp16_scale"),
        Field("qs", 16, 1, "payload"),
    ], "ggml_half d + qs[32/2]"),

    "Q8_0": Layout("Q8_0", 32, [
        Field("d", 2, 32, "fp16_scale"),
        Field("qs", 32, 1, "payload"),
    ], "ggml_half d + int8 qs[32]"),

    "Q4_1": Layout("Q4_1", 32, [
        Field("d", 2, 32, "fp16_scale"),
        Field("m", 2, 32, "fp16_scale"),
        Field("qs", 16, 1, "payload"),
    ], "Q4_0 plus a per-block minimum"),

    "Q5_0": Layout("Q5_0", 32, [
        Field("d", 2, 32, "fp16_scale"),
        Field("qh", 4, 1, "payload"),
        Field("qs", 16, 1, "payload"),
    ], "Q4_0 plus a fifth bit plane. Reached by K-quant fallback on tensors "
       "whose row length is not a multiple of 256, which is why it shows up "
       "inside files labelled Q4_K_M."),

    "Q5_1": Layout("Q5_1", 32, [
        Field("d", 2, 32, "fp16_scale"),
        Field("m", 2, 32, "fp16_scale"),
        Field("qh", 4, 1, "payload"),
        Field("qs", 16, 1, "payload"),
    ], "Q5_0 plus a per-block minimum"),

    "Q8_1": Layout("Q8_1", 32, [
        Field("d", 2, 32, "fp16_scale"),
        Field("s", 2, 32, "fp16_scale"),
        Field("qs", 32, 1, "payload"),
    ], "Q8_0 plus a cached row sum"),

    "IQ4_NL": Layout("IQ4_NL", 32, [
        Field("d", 2, 32, "fp16_scale"),
        Field("qs", 16, 1, "payload"),
    ], "same shape as Q4_0, but qs indexes a non-uniform codebook"),

    "Q2_K": Layout("Q2_K", QK_K, [
        Field("scales", 16, 16, "packed_scale", per_element_bits=4),
        Field("qs", QK_K // 4, 1, "payload"),
        Field("d", 2, QK_K, "fp16_scale"),
        Field("dmin", 2, QK_K, "fp16_scale"),
    ], "4-bit packed scales and mins over 16 sub-blocks of 16"),

    "Q3_K": Layout("Q3_K", QK_K, [
        Field("hmask", QK_K // 8, 1, "payload"),
        Field("qs", QK_K // 4, 1, "payload"),
        Field("scales", 12, 16, "packed_scale", per_element_bits=6),
        Field("d", 2, QK_K, "fp16_scale"),
    ], "6-bit packed scales over 16 sub-blocks of 16"),

    "Q4_K": Layout("Q4_K", QK_K, [
        Field("d", 2, QK_K, "fp16_scale"),
        Field("dmin", 2, QK_K, "fp16_scale"),
        Field("scales", K_SCALE_SIZE, 32, "packed_scale", per_element_bits=6),
        Field("qs", QK_K // 2, 1, "payload"),
    ], "8 sub-blocks of 32, two fp16 super-scales, 6-bit packed sub-scales"),

    "Q5_K": Layout("Q5_K", QK_K, [
        Field("d", 2, QK_K, "fp16_scale"),
        Field("dmin", 2, QK_K, "fp16_scale"),
        Field("scales", K_SCALE_SIZE, 32, "packed_scale", per_element_bits=6),
        Field("qh", QK_K // 8, 1, "payload"),
        Field("qs", QK_K // 2, 1, "payload"),
    ], "Q4_K plus a high-bit plane"),

    "Q6_K": Layout("Q6_K", QK_K, [
        Field("ql", QK_K // 2, 1, "payload"),
        Field("qh", QK_K // 4, 1, "payload"),
        Field("scales", QK_K // 16, 16, "int_scale", per_element_bits=8),
        Field("d", 2, QK_K, "fp16_scale"),
    ], "int8 sub-scales over 16 sub-blocks of 16, one fp16 super-scale"),

    "IQ4_XS": Layout("IQ4_XS", QK_K, [
        Field("d", 2, QK_K, "fp16_scale"),
        Field("scales_h", 2, 32, "packed_scale", per_element_bits=2),
        Field("scales_l", QK_K // 64, 32, "packed_scale", per_element_bits=4),
        Field("qs", QK_K // 2, 1, "payload"),
    ], "codebook quant, split high/low scale nibbles"),

    "F16": Layout("F16", 1, [
        Field("value", 2, 1, "fp16_scale"),
    ], "every weight carries its own exponent; baseline for comparison"),
}


# ------------------------------------------------------------------ analytical model

@dataclass
class Profile:
    name: str
    block_bytes: int
    weights: int
    bits_per_weight: float
    total_bits: int
    wide_bits: int             # blast radius > 1
    max_blast: int
    exponent_bits: int         # fp16 exponent bits, the unbounded-magnitude sites
    mean_blast: float          # expected weights touched by one uniformly random bit flip
    by_kind: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        d = dict(self.__dict__)
        d["wide_bit_pct"] = round(100.0 * self.wide_bits / self.total_bits, 3)
        d["exponent_bit_pct"] = round(100.0 * self.exponent_bits / self.total_bits, 3)
        return d


def blast_profile(layout: Layout) -> Profile:
    """Expected damage from flipping one uniformly chosen bit of one block."""
    total_bits = 8 * layout.block_bytes
    wide = 0
    weighted = 0
    max_blast = 0
    exponent_bits = 0
    by_kind: Dict[str, int] = {}

    for f in layout.fields:
        bits = 8 * f.bytes_
        by_kind[f.kind] = by_kind.get(f.kind, 0) + bits
        weighted += bits * f.blast
        max_blast = max(max_blast, f.blast)
        if f.blast > 1:
            wide += bits
        if f.kind == "fp16_scale":
            # ggml_half is IEEE binary16: 1 sign, 5 exponent, 10 mantissa.
            exponent_bits += 5 * f.bytes_ // 2

    return Profile(
        name=layout.name,
        block_bytes=layout.block_bytes,
        weights=layout.weights,
        bits_per_weight=round(layout.bits_per_weight, 4),
        total_bits=total_bits,
        wide_bits=wide,
        max_blast=max_blast,
        exponent_bits=exponent_bits,
        mean_blast=round(weighted / total_bits, 4),
        by_kind=by_kind,
    )


def table(order: Optional[List[str]] = None) -> str:
    names = order or list(LAYOUTS)
    rows = [blast_profile(LAYOUTS[n]) for n in names]
    head = (f"{'format':<8} {'B/blk':>6} {'w/blk':>6} {'bpw':>7} "
            f"{'wide%':>7} {'expo%':>7} {'max':>5} {'E[blast]':>9}")
    lines = [head, "-" * len(head)]
    for p in rows:
        d = p.as_dict()
        lines.append(f"{p.name:<8} {p.block_bytes:>6} {p.weights:>6} {p.bits_per_weight:>7.3f} "
                     f"{d['wide_bit_pct']:>7.2f} {d['exponent_bit_pct']:>7.2f} "
                     f"{p.max_blast:>5} {p.mean_blast:>9.3f}")
    return "\n".join(lines)


# ------------------------------------------------------------------ GGUF file walk
#
# GGUF v3 header, from the spec in ggml's docs/gguf.md:
#   magic "GGUF" (4 bytes), version u32, tensor_count u64, metadata_kv_count u64,
#   then the KV pairs, then tensor_info entries, then padding to alignment, then data.
# Only enough of the KV grammar is implemented here to skip past it correctly.

_GGUF_MAGIC = b"GGUF"

# ggml_type ids, from ggml.h. Only the ones this module models are named.
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 20: "IQ4_NL", 23: "IQ4_XS", 30: "BF16",
}

_KV_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
              7: "?", 10: "Q", 11: "q", 12: "d"}
_KV_STRING = 8
_KV_ARRAY = 9


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError("truncated GGUF")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        return self.raw(self.u64()).decode("utf-8", "replace")

    def skip_value(self, vtype: int) -> None:
        if vtype in _KV_SCALAR:
            self.raw(struct.calcsize("<" + _KV_SCALAR[vtype]))
        elif vtype == _KV_STRING:
            self.raw(self.u64())
        elif vtype == _KV_ARRAY:
            inner = self.u32()
            count = self.u64()
            if inner in _KV_SCALAR and inner != _KV_STRING:
                self.raw(count * struct.calcsize("<" + _KV_SCALAR[inner]))
            else:
                for _ in range(count):
                    self.skip_value(inner)
        else:
            raise ValueError(f"unknown GGUF value type {vtype}")


def scan_gguf(path: str) -> Dict:
    """Per-tensor and whole-file structural bit census of a GGUF model."""
    with open(path, "rb") as fh:
        r = _Reader(fh)
        if r.raw(4) != _GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        alignment = 32
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            if key == "general.alignment" and vtype in (4, 5, 10, 11):
                fmt = {4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[vtype]
                alignment = struct.unpack(fmt, r.raw(struct.calcsize(fmt)))[0]
            else:
                r.skip_value(vtype)

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            ndims = r.u32()
            dims = [r.u64() for _ in range(ndims)]
            ttype = r.u32()
            rel = r.u64()          # offset into the data section
            elements = 1
            for d in dims:
                elements *= d
            tensors.append((name, dims, ttype, elements, rel))
        here = fh.tell()
        data_start = here + ((alignment - (here % alignment)) % alignment)

    totals = {"total_bits": 0, "wide_bits": 0, "exponent_bits": 0,
              "payload_bits": 0, "unmodelled_elements": 0}
    unmodelled_types: Dict[str, int] = {}
    per_type: Dict[str, Dict] = {}
    rows = []

    for name, dims, ttype, elements, rel in tensors:
        tname = GGML_TYPE_NAMES.get(ttype, f"type{ttype}")
        layout = LAYOUTS.get(tname)
        if layout is None:
            totals["unmodelled_elements"] += elements
            unmodelled_types[tname] = unmodelled_types.get(tname, 0) + 1
            rows.append({"tensor": name, "type": tname, "elements": elements,
                         "abs_offset": data_start + rel, "modelled": False})
            continue
        blocks = elements / layout.weights
        p = blast_profile(layout)
        bits = blocks * p.total_bits
        wide = blocks * p.wide_bits
        expo = blocks * p.exponent_bits
        payload = blocks * p.by_kind.get("payload", 0)

        totals["total_bits"] += bits
        totals["wide_bits"] += wide
        totals["exponent_bits"] += expo
        totals["payload_bits"] += payload

        agg = per_type.setdefault(tname, {"tensors": 0, "elements": 0, "bits": 0.0,
                                          "wide_bits": 0.0, "exponent_bits": 0.0})
        agg["tensors"] += 1
        agg["elements"] += elements
        agg["bits"] += bits
        agg["wide_bits"] += wide
        agg["exponent_bits"] += expo

        rows.append({"tensor": name, "type": tname, "elements": elements,
                     "abs_offset": data_start + rel,
                     "modelled": True, "bits": bits, "wide_bits": wide,
                     "exponent_bits": expo, "max_blast": p.max_blast})

    tb = totals["total_bits"] or 1
    return {
        "path": path,
        "gguf_version": version,
        "alignment": alignment,
        "data_start": data_start,
        "n_tensors": n_tensors,
        "totals": totals,
        "wide_bit_pct": round(100.0 * totals["wide_bits"] / tb, 4),
        "exponent_bit_pct": round(100.0 * totals["exponent_bits"] / tb, 4),
        "per_type": per_type,
        "unmodelled_types": unmodelled_types,
        "coverage_note": (
            "wide_bit_pct and exponent_bit_pct are computed over MODELLED tensors only. "
            "If unmodelled_types is non-empty the census does not describe the whole file."
            if unmodelled_types else "every quantized tensor in this file is modelled"),
        "tensors": rows,
    }


# ------------------------------------------------------------------ scale range

def scale_exponent_range(path: str, sample_blocks: Optional[int] = None) -> Dict:
    """The exponent range every fp16 scale in the file actually occupies.

    Why this exists. A quantization scale maps a block of weights, centred near zero, onto a
    small integer range, so the scale is a small number by construction. Measured over every
    scale in five 0.5B files, twelve to fifteen million each, the biased fp16 exponent ran from
    0 to between 8 and 13 depending on the file. The most significant exponent bit was zero in
    every scale of every file. The next bit down was not: it is legitimately set in the files
    whose scales span a wider range, which is why a range check catches all of the worst
    corruption class and only part of the next one.

    That is the whole basis of a mitigation that needs no redundancy. The single-bit
    corruptions that damage a model are the ones that push a scale far above its range, and a
    value outside the file's own range is wrong on its face. Store five bits at build time and
    check them at load, and the corruption is detectable without a checksum, a replica or a
    parity bit.

    Returns the range, the implied guard band, and the share of scale bits that a range check
    covers. `sample_blocks` limits the scan on very large files; the default reads all of them.
    """
    g_min, g_max = 31, 0
    hist: Dict[int, int] = {}
    n_scales = 0

    with open(path, "rb") as fh:
        rep = scan_gguf(path)
        for row in rep["tensors"]:
            if not row.get("modelled"):
                continue
            lay = LAYOUTS.get(row["type"])
            if lay is None:
                continue
            fp16_fields = [(off, f) for off, f in _field_offsets(lay)
                           if f.kind == "fp16_scale"]
            if not fp16_fields:
                continue
            blocks = row["elements"] // lay.weights
            if sample_blocks:
                blocks = min(blocks, sample_blocks)
            base = row["abs_offset"]
            for b in range(blocks):
                for off, f in fp16_fields:
                    for half in range(f.bytes_ // 2):
                        fh.seek(base + b * lay.block_bytes + off + half * 2 + 1)
                        hi = fh.read(1)
                        if len(hi) != 1:
                            continue
                        e = (hi[0] >> 2) & 31
                        hist[e] = hist.get(e, 0) + 1
                        n_scales += 1
                        if e < g_min: g_min = e
                        if e > g_max: g_max = e

    prof = {n: blast_profile(LAYOUTS[n]) for n in LAYOUTS}
    return {
        "path": path,
        "scales_read": n_scales,
        "biased_exponent_min": g_min if n_scales else None,
        "biased_exponent_max": g_max if n_scales else None,
        "magnitude_min": 2.0 ** (g_min - 15) if n_scales else None,
        "magnitude_max": 2.0 ** (g_max - 15) if n_scales else None,
        "histogram": dict(sorted(hist.items())),
        "bits_always_zero": [i for i in range(4, -1, -1)
                             if all(((e >> i) & 1) == 0 for e in hist)],
        "guard": (f"reject any scale whose biased fp16 exponent exceeds {g_max}"
                  if n_scales else None),
    }


def check_scale_range(path: str, max_exponent: int, limit: int = 20) -> Dict:
    """Check every fp16 scale in a file against a range recorded at build time.

    `scale_exponent_range` answers "what range does this file occupy". This answers the
    question a deployment pipeline actually asks, which is "is this file still inside the range
    it occupied when we shipped it". The two are different jobs and only the second one catches
    anything.

    The check is the whole mitigation. A single-bit corruption that damages a quantized model
    is one that pushes a scale far above its own range, because a scale is small by
    construction and its high exponent bits are already zero, so a flip there can only make it
    larger. Comparing five bits per scale against one stored number needs no checksum, no
    replica and no parity, and on the campaign data it catches 71.4 percent of hard failures
    and every one of the 404 flips of the most significant exponent bit.

    It is not complete and is not meant to be. A flip of the second exponent bit lands inside
    the range for a file whose scales already span one, and those are invisible here. Parity
    over that single bit position is the cheapest way to cover them.

    Returns the verdict and up to `limit` offending scales, each named by tensor and block so
    the caller can decide whether to refuse the file or repair it from a replica.
    """
    violations = []
    n_scales = 0
    n_expected = 0
    n_bad = 0
    worst = -1
    with open(path, "rb") as fh:
        rep = scan_gguf(path)
        for row in rep["tensors"]:
            if not row.get("modelled"):
                continue
            lay = LAYOUTS.get(row["type"])
            if lay is None:
                continue
            fp16_fields = [(off, f) for off, f in _field_offsets(lay)
                           if f.kind == "fp16_scale"]
            if not fp16_fields:
                continue
            blocks = row["elements"] // lay.weights
            base = row["abs_offset"]
            n_expected += blocks * sum(f.bytes_ // 2 for _, f in fp16_fields)
            for b in range(blocks):
                for off, f in fp16_fields:
                    for half in range(f.bytes_ // 2):
                        at = base + b * lay.block_bytes + off + half * 2
                        fh.seek(at + 1)
                        hi = fh.read(1)
                        if len(hi) != 1:
                            continue
                        e = (hi[0] >> 2) & 31
                        n_scales += 1
                        if e > worst:
                            worst = e
                        if e > max_exponent:
                            n_bad += 1
                            if len(violations) < limit:
                                violations.append({
                                    "tensor": row["tensor"],
                                    "type": row["type"],
                                    "block_index": b,
                                    "field": f.name,
                                    "abs_offset": at,
                                    "biased_exponent": e,
                                    "allowed_max": max_exponent,
                                    "weights_affected": f.blast,
                                })
    # A truncated file reads short and every unread scale is an unchecked scale. Reporting
    # that as a pass is the one way this check could be actively harmful, so it is not a pass.
    short = n_scales < n_expected
    return {
        "path": path,
        "allowed_max_exponent": max_exponent,
        "scales_expected": n_expected,
        "scales_read": n_scales,
        "file_short": short,
        "observed_max_exponent": worst if n_scales else None,
        "violations": n_bad,
        "ok": n_bad == 0 and not short and n_scales > 0,
        "first_violations": violations,
        "truncated": n_bad > len(violations),
    }


def resolve_max_exponent(spec, key: Optional[str], path: str) -> int:
    """Pull a single maximum exponent out of whatever the caller stored.

    Three shapes are accepted because three shapes exist in practice: a bare integer, the JSON
    that `--scale-range --json` writes, and a mapping of file label to maximum, which is what a
    build shipping several quantizations of one model produces.
    """
    if isinstance(spec, bool):
        raise SystemExit("range must be an integer, not a boolean")
    if isinstance(spec, int):
        return spec
    if isinstance(spec, dict) and "biased_exponent_max" in spec:
        return int(spec["biased_exponent_max"])
    if isinstance(spec, dict):
        if key is not None:
            if key not in spec:
                raise SystemExit("--range-key %r is not in the range file; have %s"
                                 % (key, sorted(spec)))
            return int(spec[key])
        base = os.path.basename(path)
        hits = [k for k in spec if k in base]
        if len(hits) == 1:
            return int(spec[hits[0]])
        raise SystemExit("the range file maps several labels and %d match %r. Pass "
                         "--range-key. Have %s" % (len(hits), base, sorted(spec)))
    raise SystemExit("unrecognised range file: expected an integer, a --scale-range JSON, "
                     "or a mapping of label to maximum exponent")


def _field_offsets(lay: Layout):
    """(byte offset within the block, field), in declaration order."""
    cursor = 0
    for f in lay.fields:
        yield cursor, f
        cursor += f.bytes_


# ------------------------------------------------------------------ self test

def _synthetic_gguf(path: str, with_data: bool = False, max_exponent: int = 7) -> None:
    """Write a minimal but spec-shaped GGUF so the parser is exercised for real.

    With `with_data`, it also writes the tensor bytes, giving every fp16 scale a biased
    exponent cycling up to `max_exponent`. That is what makes the range check testable without
    a real model: a header-only file reads zero scales, and a check that reads zero scales
    passes vacuously, which is the failure mode worth having a test for.
    """
    def s(text: str) -> bytes:
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    tensors = [("blk.0.attn_q.weight", [512, 512], 12, "Q4_K"),
               ("output.weight", [512, 1024], 14, "Q6_K"),
               ("blk.0.attn_norm.weight", [512], 0, "F32")]

    sizes, offsets, cursor = [], [], 0
    for _, dims, _, tname in tensors:
        n = 1
        for d in dims:
            n *= d
        lay = LAYOUTS.get(tname)
        nbytes = (n // lay.weights) * lay.block_bytes if lay else n * 4
        sizes.append(nbytes)
        offsets.append(cursor)
        cursor += nbytes
        cursor += (-cursor) % 32

    body = bytearray()
    body += _GGUF_MAGIC
    body += struct.pack("<I", 3)          # version
    body += struct.pack("<Q", len(tensors))
    body += struct.pack("<Q", 2)          # kv count

    # kv 0: a string
    body += s("general.architecture") + struct.pack("<I", _KV_STRING) + s("llama")
    # kv 1: an array of u32, to exercise the array skip path
    body += s("test.array") + struct.pack("<I", _KV_ARRAY)
    body += struct.pack("<I", 4) + struct.pack("<Q", 3) + struct.pack("<III", 1, 2, 3)

    for (name, dims, ttype, _), off in zip(tensors, offsets):
        body += s(name)
        body += struct.pack("<I", len(dims))
        for d in dims:
            body += struct.pack("<Q", d)
        body += struct.pack("<I", ttype)
        body += struct.pack("<Q", off if with_data else 0)

    with open(path, "wb") as fh:
        fh.write(bytes(body))
        if not with_data:
            return
        fh.write(b"\x00" * ((-len(body)) % 32))
        for (name, dims, ttype, tname), off, nbytes in zip(tensors, offsets, sizes):
            lay = LAYOUTS.get(tname)
            if lay is None:
                fh.write(bytes((i * 37 + 11) & 0xFF for i in range(nbytes)))
                fh.write(b"\x00" * ((-nbytes) % 32))
                continue
            fp16 = [(o, f) for o, f in _field_offsets(lay) if f.kind == "fp16_scale"]
            blk = bytearray(lay.block_bytes)
            out = bytearray()
            for b in range(nbytes // lay.block_bytes):
                for k in range(lay.block_bytes):
                    blk[k] = (b * 31 + k * 17 + 5) & 0xFF
                e = 1 + (b % max_exponent)          # biased exponent, never zero, never > max
                for o, f in fp16:
                    for half in range(f.bytes_ // 2):
                        blk[o + half * 2] = (b * 13 + half) & 0xFF     # mantissa low bits
                        blk[o + half * 2 + 1] = (e & 31) << 2          # sign 0, exponent e
                out += blk
            fh.write(bytes(out))
            fh.write(b"\x00" * ((-len(out)) % 32))


def selftest() -> int:
    failures = []

    # 1. Block sizes must match the static_asserts in ggml-common.h.
    expected = {"Q4_0": 18, "Q4_1": 20, "Q5_0": 22, "Q5_1": 24, "Q8_0": 34,
                "Q8_1": 36, "IQ4_NL": 18, "Q2_K": 84, "Q3_K": 110, "Q4_K": 144,
                "Q5_K": 176, "Q6_K": 210, "IQ4_XS": 136, "F16": 2}
    for name, size in expected.items():
        got = LAYOUTS[name].block_bytes
        if got != size:
            failures.append(f"{name}: block_bytes {got}, expected {size}")

    # 2. Bits per weight must land where the format names claim.
    for name, lo, hi in [("Q4_0", 4.4, 4.6), ("Q4_K", 4.4, 4.6), ("Q6_K", 6.5, 6.6),
                         ("Q8_0", 8.4, 8.6), ("IQ4_XS", 4.2, 4.3), ("Q5_0", 5.4, 5.6),
                         ("Q5_1", 5.9, 6.1)]:
        bpw = LAYOUTS[name].bits_per_weight
        if not lo <= bpw <= hi:
            failures.append(f"{name}: {bpw:.3f} bits/weight outside [{lo}, {hi}]")

    # 3. The types a K-quant request can silently fall back to must all be modelled,
    #    because a gap here excludes part of the file from the census without saying so.
    for name in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"):
        if name not in LAYOUTS:
            failures.append(f"{name} is a K-quant fallback type and is not modelled")

    # 4. The structural prediction itself: K-quants must reach further than Q4_0.
    if blast_profile(LAYOUTS["Q4_K"]).max_blast <= blast_profile(LAYOUTS["Q4_0"]).max_blast:
        failures.append("Q4_K max blast should exceed Q4_0")

    # 5. Q6_K carries fewer unbounded-magnitude sites than Q4_K per weight.
    q6 = blast_profile(LAYOUTS["Q6_K"])
    q4k = blast_profile(LAYOUTS["Q4_K"])
    if q6.exponent_bits / q6.weights >= q4k.exponent_bits / q4k.weights:
        failures.append("Q6_K should have fewer fp16 exponent bits per weight than Q4_K")

    # 6. The GGUF parser, against a file this module writes itself.
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.gguf")
        _synthetic_gguf(p)
        rep = scan_gguf(p)
        if rep["n_tensors"] != 3:
            failures.append(f"parser found {rep['n_tensors']} tensors, expected 3")
        types = {r["type"] for r in rep["tensors"]}
        if types != {"Q4_K", "Q6_K", "F32"}:
            failures.append(f"parser types {types}")
        if rep["totals"]["unmodelled_elements"] != 512:
            failures.append("F32 norm tensor should be counted as unmodelled")
        if not 0 < rep["wide_bit_pct"] < 100:
            failures.append(f"wide_bit_pct out of range: {rep['wide_bit_pct']}")

        # 7. Tensor offsets must be resolved, since the scale-range scan reads bytes by them.
        for r in rep["tensors"]:
            if "abs_offset" not in r:
                failures.append("scan_gguf row has no abs_offset")
                break
        else:
            if min(r["abs_offset"] for r in rep["tensors"]) < rep["data_start"]:
                failures.append("a tensor starts before the data section")

        # 8. The range check. A file must pass against its own measured range, and must fail
        #    against a range one below it, or the mitigation the paper argues for is untested.
        #    A header-only file must NOT pass, because nothing in it was read.
        empty = check_scale_range(p, 31)
        if empty["ok"]:
            failures.append("a header-only file passed a range check having read no scales")

        p = os.path.join(td, "full.gguf")
        _synthetic_gguf(p, with_data=True, max_exponent=7)
        rep = scan_gguf(p)
        rng = scale_exponent_range(p)
        mx = rng["biased_exponent_max"]
        if mx is None:
            failures.append("scale_exponent_range read no scales from the synthetic file")
        else:
            good = check_scale_range(p, mx)
            if not good["ok"] or good["violations"]:
                failures.append(f"file failed a check against its own range: {good['violations']}")
            if good["scales_read"] != rng["scales_read"]:
                failures.append("the checker and the range reader disagree on how many scales "
                                f"exist: {good['scales_read']} against {rng['scales_read']}")
            bad = check_scale_range(p, mx - 1)
            if bad["ok"] or bad["violations"] == 0:
                failures.append("a range one below the file's own maximum caught nothing")
            if bad["observed_max_exponent"] != mx:
                failures.append("the checker's observed maximum disagrees with the range reader")
            # the corruption this is meant to catch: set the top exponent bit of one scale
            row = next(r for r in rep["tensors"] if r.get("modelled"))
            off, fld = next((o, f) for o, f in _field_offsets(LAYOUTS[row["type"]])
                            if f.kind == "fp16_scale")
            at = row["abs_offset"] + off + 1
            with open(p, "r+b") as fh:
                fh.seek(at)
                b = fh.read(1)[0]
                fh.seek(at)
                fh.write(bytes([b | 0x40]))          # bit 4 of the biased exponent
            hit = check_scale_range(p, mx)
            if hit["ok"] or hit["violations"] != 1:
                failures.append(f"a top-exponent-bit flip was not caught: {hit['violations']}")
            elif hit["first_violations"][0]["tensor"] != row["tensor"]:
                failures.append("the checker named the wrong tensor")

        # 9. The three shapes a stored range arrives in must all resolve to the same number.
        for spec, key in ((11, None), ({"biased_exponent_max": 11}, None),
                          ({"Q4_K_M": 11, "Q8_0": 8}, "Q4_K_M")):
            got = resolve_max_exponent(spec, key, "model-Q4_K_M.gguf")
            if got != 11:
                failures.append(f"resolve_max_exponent({spec!r}) gave {got}, expected 11")
        try:
            resolve_max_exponent({"Q4_K_M": 11, "Q8_0": 8}, None, "ambiguous.gguf")
        except SystemExit:
            pass
        else:
            failures.append("an unresolvable range mapping did not raise")

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("all checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", action="store_true", help="analytical per-format table")
    ap.add_argument("--gguf", type=str, help="scan a real GGUF file")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scale-range", action="store_true",
                    help="read every fp16 scale and report the exponent range it occupies, "
                         "which is the basis of a redundancy-free corruption check")
    ap.add_argument("--check-range", type=str, metavar="RANGES",
                    help="check every scale against a range recorded at build time and exit "
                         "non-zero if any is outside it. Takes an integer, the JSON written "
                         "by --scale-range --json, or a mapping of label to maximum exponent")
    ap.add_argument("--range-key", type=str,
                    help="which entry of a mapping --check-range should use")
    ap.add_argument("--json", type=str)
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.gguf and args.check_range:
        raw = args.check_range.strip()
        if raw.lstrip("+-").isdigit():
            spec = int(raw)
        else:
            with open(raw, encoding="utf-8") as fh:
                spec = json.load(fh)
        rep = check_scale_range(args.gguf,
                                resolve_max_exponent(spec, args.range_key, args.gguf))
        print(json.dumps(rep, indent=2))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=2)
        return 0 if rep["ok"] else 2

    if args.gguf and args.scale_range:
        rep = scale_exponent_range(args.gguf)
        print(json.dumps(rep, indent=2))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=2)
        return 0

    if args.gguf:
        rep = scan_gguf(args.gguf)
        print(json.dumps({k: v for k, v in rep.items() if k != "tensors"}, indent=2))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=2)
        return 0

    order = ["F16", "Q8_0", "Q6_K", "Q5_1", "Q5_K", "Q5_0", "Q4_K", "Q4_0",
             "IQ4_NL", "IQ4_XS", "Q3_K", "Q2_K"]
    print(table(order))
    if args.json:
        payload = {n: blast_profile(LAYOUTS[n]).as_dict() for n in order}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
