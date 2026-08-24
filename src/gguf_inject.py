"""Single-bit fault injection into GGUF weight files, stratified by block structure.

`gguf_faultscope` answers what one flipped bit *reaches*, from the block layouts alone.
This module is the other half: it finds where those bits actually live in a real file, picks
them by structural role rather than uniformly, flips one, and puts it back.

Why stratified. In a Q4_K model roughly 89 percent of the bits are payload nibbles, so
uniform sampling spends nine tenths of the budget in the one place where a flip moves a
single weight by at most a fifteenth of its block range. The interesting bits are the fp16
super-block scales, which are under one percent of the file and reach 256 weights each.
Sampling by stratum is the difference between a study that finds the tail and one that
does not.

Why flip in place. A 0.5B model is several hundred megabytes and a campaign runs a thousand
injections. Copying the file each time costs more disk traffic than the whole experiment.
So a flip is a seek, a one-byte read, an XOR, a one-byte write, and the inverse afterwards.
`inject()` is a context manager that restores on the way out, including on exception, and
verifies the restore before returning.

    from gguf_inject import GGUF, Stratum, plan, inject

    g = GGUF("model.gguf")
    sites = plan(g, strata=[Stratum.FP16_SCALE_EXPONENT], n=30, seed=7)
    for site in sites:
        with inject("model.gguf", site):
            run_the_model()          # file is corrupt only inside this block

Safety. `inject` refuses to touch a file whose size or header digest does not match what
`GGUF` parsed, it re-reads the byte after restoring and raises if it does not match, and it
never writes outside the tensor data region. A crash mid-block leaves one flipped bit in the
file, so `verify()` and `repair_log` exist to find and undo that.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import struct
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from gguf_faultscope import (
    LAYOUTS,
    GGML_TYPE_NAMES,
    Layout,
    Field,
    _GGUF_MAGIC,
    _Reader,
    _synthetic_gguf,
)

DEFAULT_ALIGNMENT = 32

# Bytes per element for the non-block types, so tensor sizes come out right even when the
# tensor itself is never an injection target.
SCALAR_TYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I8": 1, "I16": 2,
                     "I32": 4, "I64": 8}


class Stratum(str, Enum):
    """Structural roles a bit can occupy. The experiment's independent variable."""

    FP16_SCALE_SIGN = "fp16_scale_sign"
    FP16_SCALE_EXPONENT = "fp16_scale_exponent"
    FP16_SCALE_MANTISSA = "fp16_scale_mantissa"
    PACKED_SCALE = "packed_scale"        # 6-bit or 4-bit sub-block scales, K-quants
    INT_SCALE = "int_scale"              # int8 sub-block scales, Q6_K
    PAYLOAD = "payload"                  # the quantized weights themselves

    @classmethod
    def all(cls) -> List["Stratum"]:
        return list(cls)


# fp16 is IEEE binary16: bit 15 sign, bits 14 down to 10 exponent, bits 9 down to 0 mantissa.
# Byte order in the file is little endian, so byte 0 holds bits 7..0 and byte 1 holds 15..8.
def _fp16_stratum(byte_in_field: int, bit_in_byte: int) -> Stratum:
    bit = byte_in_field * 8 + bit_in_byte      # 0..15, little endian
    if bit == 15:
        return Stratum.FP16_SCALE_SIGN
    if bit >= 10:
        return Stratum.FP16_SCALE_EXPONENT
    return Stratum.FP16_SCALE_MANTISSA


# ------------------------------------------------------------------ parsing

@dataclass
class TensorInfo:
    name: str
    dims: List[int]
    ggml_type: int
    type_name: str
    elements: int
    rel_offset: int             # offset within the data section, from the file
    abs_offset: int = 0         # absolute byte offset, filled in by GGUF
    nbytes: int = 0

    @property
    def layout(self) -> Optional[Layout]:
        return LAYOUTS.get(self.type_name)

    @property
    def injectable(self) -> bool:
        return self.layout is not None and self.type_name != "F16"


@dataclass
class Site:
    """One bit, fully described so a result row can be interpreted years later."""

    abs_offset: int             # byte offset in the file
    bit: int                    # 0..7 within that byte
    tensor: str
    tensor_type: str
    block_index: int
    field: str
    stratum: str
    blast: int                  # weights reached, from the structural model

    def key(self) -> str:
        return f"{self.abs_offset}:{self.bit}"

    def as_dict(self) -> Dict:
        return dict(self.__dict__)


class GGUF:
    """A parsed GGUF file, with every tensor's absolute byte range resolved."""

    def __init__(self, path: str):
        self.path = path
        self.size = os.path.getsize(path)
        self.tensors: List[TensorInfo] = []
        self.alignment = DEFAULT_ALIGNMENT
        self.version = 0
        self.data_start = 0
        self._parse()
        self.header_digest = self._digest_header()

    # -------------------------------------------------------------- internals

    def _parse(self) -> None:
        with open(self.path, "rb") as fh:
            r = _Reader(fh)
            if r.raw(4) != _GGUF_MAGIC:
                raise ValueError("not a GGUF file")
            self.version = r.u32()
            n_tensors = r.u64()
            n_kv = r.u64()

            for _ in range(n_kv):
                key = r.string()
                vtype = r.u32()
                if key == "general.alignment" and vtype in (4, 5, 10, 11):
                    # u32/i32/u64/i64. Read it rather than skipping it.
                    fmt = {4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[vtype]
                    self.alignment = struct.unpack(fmt, r.raw(struct.calcsize(fmt)))[0]
                else:
                    r.skip_value(vtype)

            for _ in range(n_tensors):
                name = r.string()
                ndims = r.u32()
                dims = [r.u64() for _ in range(ndims)]
                ttype = r.u32()
                rel = r.u64()
                elements = 1
                for d in dims:
                    elements *= d
                self.tensors.append(TensorInfo(
                    name=name, dims=dims, ggml_type=ttype,
                    type_name=GGML_TYPE_NAMES.get(ttype, f"type{ttype}"),
                    elements=elements, rel_offset=rel))

            here = fh.tell()

        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)):
            raise ValueError(f"implausible alignment {self.alignment}")
        pad = (self.alignment - (here % self.alignment)) % self.alignment
        self.data_start = here + pad

        for t in self.tensors:
            t.abs_offset = self.data_start + t.rel_offset
            t.nbytes = self._tensor_bytes(t)

    def _tensor_bytes(self, t: TensorInfo) -> int:
        lay = t.layout
        if lay is not None and lay.weights > 1:
            if t.elements % lay.weights:
                raise ValueError(f"{t.name}: {t.elements} elements is not a whole number "
                                 f"of {lay.name} blocks")
            return (t.elements // lay.weights) * lay.block_bytes
        if t.type_name in SCALAR_TYPE_BYTES:
            return t.elements * SCALAR_TYPE_BYTES[t.type_name]
        if lay is not None:                       # F16 goes through here
            return t.elements * lay.block_bytes
        return 0                                  # unmodelled; never an injection target

    def _digest_header(self) -> str:
        with open(self.path, "rb") as fh:
            return hashlib.sha256(fh.read(min(self.data_start, self.size))).hexdigest()

    # -------------------------------------------------------------- public

    def injectable_tensors(self, include: Optional[str] = None,
                           exclude_output: bool = False) -> List[TensorInfo]:
        out = [t for t in self.tensors if t.injectable and t.nbytes > 0]
        if include:
            out = [t for t in out if include in t.name]
        if exclude_output:
            out = [t for t in out if not t.name.startswith("output")]
        return out

    def classify(self, tensor: TensorInfo, abs_offset: int, bit: int) -> Site:
        """Which structural role does this exact bit occupy."""
        lay = tensor.layout
        if lay is None:
            raise ValueError(f"{tensor.name} has no modelled layout")
        rel = abs_offset - tensor.abs_offset
        if not 0 <= rel < tensor.nbytes:
            raise ValueError("offset outside tensor")
        block_index, byte_in_block = divmod(rel, lay.block_bytes)

        cursor = 0
        for f in lay.fields:
            if byte_in_block < cursor + f.bytes_:
                byte_in_field = byte_in_block - cursor
                if f.kind == "fp16_scale":
                    # A field may hold several fp16s; locate the one this byte is in.
                    within = byte_in_field % 2
                    stratum = _fp16_stratum(within, bit)
                else:
                    stratum = {"packed_scale": Stratum.PACKED_SCALE,
                               "int_scale": Stratum.INT_SCALE,
                               "payload": Stratum.PAYLOAD}[f.kind]
                return Site(abs_offset=abs_offset, bit=bit, tensor=tensor.name,
                            tensor_type=tensor.type_name, block_index=block_index,
                            field=f.name, stratum=stratum.value, blast=f.blast)
            cursor += f.bytes_
        raise AssertionError("byte fell off the end of the block layout")

    def field_ranges(self, lay: Layout) -> List[Tuple[Field, int, int]]:
        """(field, first byte, last byte inclusive) within one block."""
        out, cursor = [], 0
        for f in lay.fields:
            out.append((f, cursor, cursor + f.bytes_ - 1))
            cursor += f.bytes_
        return out

    def summary(self) -> Dict:
        by_type: Dict[str, Dict] = {}
        for t in self.tensors:
            d = by_type.setdefault(t.type_name, {"tensors": 0, "elements": 0, "bytes": 0})
            d["tensors"] += 1
            d["elements"] += t.elements
            d["bytes"] += t.nbytes
        end = max((t.abs_offset + t.nbytes) for t in self.tensors) if self.tensors else 0
        return {
            "path": self.path, "size": self.size, "gguf_version": self.version,
            "alignment": self.alignment, "data_start": self.data_start,
            "n_tensors": len(self.tensors), "data_end": end,
            "trailing_bytes": self.size - end, "by_type": by_type,
        }


# ------------------------------------------------------------------ planning

def _stratum_bit_positions(lay: Layout, stratum: Stratum) -> List[Tuple[int, int]]:
    """Every (byte_in_block, bit) inside one block that belongs to this stratum."""
    out: List[Tuple[int, int]] = []
    cursor = 0
    for f in lay.fields:
        for b in range(f.bytes_):
            for bit in range(8):
                if f.kind == "fp16_scale":
                    s = _fp16_stratum(b % 2, bit)
                else:
                    s = {"packed_scale": Stratum.PACKED_SCALE,
                         "int_scale": Stratum.INT_SCALE,
                         "payload": Stratum.PAYLOAD}[f.kind]
                if s == stratum:
                    out.append((cursor + b, bit))
        cursor += f.bytes_
    return out


def plan(g: GGUF, strata: Sequence[Stratum], n, seed: int,
         include: Optional[str] = None, exclude_output: bool = False,
         types: Optional[Sequence[str]] = None,
         by_block_type: bool = False, min_blocks: int = 64) -> List[Site]:
    """n sites per cell, drawn without replacement, reproducible from the seed.

    A cell is a stratum when `by_block_type` is false, and a (block type, stratum) pair when
    it is true. **Prefer true.** The first campaign stratified by role alone and the result
    was that Q4_K tensors, which are 12 of 169 in a Q4_K_M file, drew 100 packed-scale sites
    and zero exponent sites. Their measured catastrophe rate of 0.0 percent was therefore an
    artifact of never sampling the only stratum that produces catastrophes, and a per-format
    comparison built on it would have been spurious in a way that survives a careless read.

    Stratifying by block type as well fixes it: every block type with at least `min_blocks`
    blocks gets its own quota in every stratum its layout contains, so a zero means zero
    rather than "never asked".

    Within a cell, tensors are still drawn in proportion to their block count, so the sample
    reflects where that block type's weight actually sits in the model.

    `n` may be an int, or a dict from stratum value to int. The dict form exists because the
    first campaign established that five of the six strata are null: 1,900 injections outside
    the exponent produced zero catastrophes. Spending the same quota on a stratum with a
    measured ceiling of 0.8 percent and on the one carrying the entire effect wastes most of
    the budget. Confirming a null needs fewer samples than estimating a rate.
    """
    def quota(s: Stratum) -> int:
        return n[s.value] if isinstance(n, dict) else n
    rng = random.Random(seed)
    tensors = g.injectable_tensors(include=include, exclude_output=exclude_output)
    if types:
        tensors = [t for t in tensors if t.type_name in types]
    if not tensors:
        raise ValueError("no injectable tensors matched the filter")

    pool = []
    for t in tensors:
        lay = t.layout
        blocks = t.nbytes // lay.block_bytes
        if blocks:
            pool.append((t, lay, blocks))

    def draw(candidates, stratum, want) -> List[Site]:
        if not candidates:
            return []
        w = [b for (_, _, b) in candidates]
        seen, out, tries = set(), [], 0
        # A cell can be smaller than the quota. Cap the attempts rather than spinning.
        while len(seen) < want and tries < want * 400:
            tries += 1
            tt, lay, blocks = rng.choices(candidates, weights=w, k=1)[0]
            positions = _stratum_bit_positions(lay, stratum)
            block_index = rng.randrange(blocks)
            byte_in_block, bit = positions[rng.randrange(len(positions))]
            off = tt.abs_offset + block_index * lay.block_bytes + byte_in_block
            if (off, bit) in seen:
                continue
            seen.add((off, bit))
            out.append(g.classify(tt, off, bit))
        return out

    sites: List[Site] = []
    for stratum in strata:
        eligible = [(t, lay, b) for (t, lay, b) in pool if _stratum_bit_positions(lay, stratum)]
        if not eligible:
            continue
        want = quota(stratum)
        if not want:
            continue
        if not by_block_type:
            sites.extend(draw(eligible, stratum, want))
            continue
        groups: Dict[str, List] = {}
        for item in eligible:
            groups.setdefault(item[0].type_name, []).append(item)
        for bt in sorted(groups):
            cand = groups[bt]
            if sum(b for (_, _, b) in cand) < min_blocks:
                continue          # too little of this type in this file to be worth a quota
            sites.extend(draw(cand, stratum, want))
    return sites


def plan_audit(g: GGUF, strata: Sequence[Stratum], n, seed: int,
               by_block_type: bool = True, min_blocks: int = 64,
               **kw) -> Dict:
    """What the plan will actually produce, before a single bit is flipped.

    The point of running this first is that a cell with too few sites cannot carry a
    confidence interval, and discovering that after the campaign is how the first one went
    wrong. `estimable` marks the cells that will support a rate rather than a zero.
    """
    sites = plan(g, strata, n=n, seed=seed, by_block_type=by_block_type,
                 min_blocks=min_blocks, **kw)
    cells: Dict[str, Dict[str, int]] = {}
    for s in sites:
        cells.setdefault(s.tensor_type, {}).setdefault(s.stratum, 0)
        cells[s.tensor_type][s.stratum] += 1

    blocks: Dict[str, int] = {}
    for t in g.injectable_tensors():
        blocks[t.type_name] = blocks.get(t.type_name, 0) + t.nbytes // t.layout.block_bytes

    MIN_ESTIMABLE = 30
    report = {"total_sites": len(sites), "n_per_cell": n,
              "by_block_type": by_block_type, "min_blocks": min_blocks,
              "cells": [], "skipped_block_types": []}
    for bt in sorted(blocks):
        if bt not in cells:
            report["skipped_block_types"].append(
                {"block_type": bt, "blocks": blocks[bt],
                 "reason": "fewer than min_blocks" if blocks[bt] < min_blocks
                           else "no stratum matched"})
            continue
        exp = cells[bt].get(Stratum.FP16_SCALE_EXPONENT.value, 0)
        report["cells"].append({
            "block_type": bt, "blocks": blocks[bt],
            "sites": sum(cells[bt].values()), "strata": dict(cells[bt]),
            "exponent_sites": exp,
            "estimable": exp >= MIN_ESTIMABLE,
        })
    report["estimable_block_types"] = [c["block_type"] for c in report["cells"]
                                       if c["estimable"]]
    report["not_estimable"] = [c["block_type"] for c in report["cells"]
                               if not c["estimable"]]
    return report


# ------------------------------------------------------------------ injection

class InjectionError(RuntimeError):
    pass


def _read_byte(path: str, offset: int) -> int:
    with open(path, "rb") as fh:
        fh.seek(offset)
        b = fh.read(1)
    if len(b) != 1:
        raise InjectionError(f"cannot read byte at {offset}")
    return b[0]


def _write_byte(path: str, offset: int, value: int) -> None:
    with open(path, "r+b") as fh:
        fh.seek(offset)
        fh.write(bytes([value]))
        fh.flush()
        os.fsync(fh.fileno())


@contextlib.contextmanager
def inject(path: str, site: Site, guard: Optional[GGUF] = None,
           repair_log: Optional[str] = None) -> Iterator[Site]:
    """Flip one bit for the duration of the block, then put it back and check.

    `guard` is the parsed GGUF this site came from. When given, the file's size and header
    digest are re-checked before anything is written, so a site can never be applied to the
    wrong file. `repair_log` records the pending flip on disk, so a crash inside the block
    leaves a breadcrumb rather than a silently corrupt model.
    """
    if guard is not None:
        if os.path.getsize(path) != guard.size:
            raise InjectionError("file size changed since it was parsed")
        if guard._digest_header() != guard.header_digest:
            raise InjectionError("file header changed since it was parsed")
        end = max(t.abs_offset + t.nbytes for t in guard.tensors)
        if not guard.data_start <= site.abs_offset < end:
            raise InjectionError("site is outside the tensor data region")

    original = _read_byte(path, site.abs_offset)
    flipped = original ^ (1 << site.bit)

    if repair_log:
        with open(repair_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"state": "open", "path": path,
                                 "offset": site.abs_offset, "bit": site.bit,
                                 "original": original}) + "\n")
    try:
        _write_byte(path, site.abs_offset, flipped)
        if _read_byte(path, site.abs_offset) != flipped:
            raise InjectionError("flip did not take")
        yield site
    finally:
        _write_byte(path, site.abs_offset, original)
        back = _read_byte(path, site.abs_offset)
        if repair_log:
            with open(repair_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"state": "closed", "path": path,
                                     "offset": site.abs_offset, "bit": site.bit,
                                     "restored_ok": back == original}) + "\n")
        if back != original:
            raise InjectionError(
                f"RESTORE FAILED at {site.abs_offset}: expected {original}, got {back}. "
                f"The model file is now corrupt. Re-download it.")


def repair(repair_log: str) -> int:
    """Undo any flip whose block never closed. Returns how many were repaired."""
    opens: Dict[str, Dict] = {}
    with open(repair_log, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            k = f"{r['path']}:{r['offset']}:{r['bit']}"
            if r["state"] == "open":
                opens[k] = r
            else:
                opens.pop(k, None)
    for k, r in opens.items():
        _write_byte(r["path"], r["offset"], r["original"])
        print(f"repaired {k}")
    return len(opens)


# ------------------------------------------------------------------ self test

def _selftest() -> int:
    import tempfile
    failures: List[str] = []

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.gguf")
        _synthetic_gguf_with_data(path)
        g = GGUF(path)

        # 1. Structure parsed, sizes computed, nothing runs past the end of the file.
        if g.n_tensors_expected != len(g.tensors):
            failures.append("tensor count mismatch")
        end = max(t.abs_offset + t.nbytes for t in g.tensors)
        if end > g.size:
            failures.append(f"tensor data ends at {end}, past file size {g.size}")
        if g.data_start % g.alignment:
            failures.append("data_start is not aligned")

        q4k = next(t for t in g.tensors if t.type_name == "Q4_K")
        if q4k.nbytes != (q4k.elements // 256) * 144:
            failures.append("Q4_K size wrong")
        q6k = next(t for t in g.tensors if t.type_name == "Q6_K")
        if q6k.nbytes != (q6k.elements // 256) * 210:
            failures.append("Q6_K size wrong")

        # 2. Classification agrees with the block layout at known offsets.
        s = g.classify(q4k, q4k.abs_offset + 0, 0)          # d, low byte, bit 0 = mantissa
        if s.field != "d" or s.stratum != Stratum.FP16_SCALE_MANTISSA.value:
            failures.append(f"Q4_K byte 0 bit 0 classified as {s.field}/{s.stratum}")
        if s.blast != 256:
            failures.append(f"Q4_K d blast {s.blast}, expected 256")
        s = g.classify(q4k, q4k.abs_offset + 1, 7)          # d, high byte, bit 7 = sign
        if s.stratum != Stratum.FP16_SCALE_SIGN.value:
            failures.append(f"Q4_K sign bit classified as {s.stratum}")
        s = g.classify(q4k, q4k.abs_offset + 1, 6)          # top exponent bit
        if s.stratum != Stratum.FP16_SCALE_EXPONENT.value:
            failures.append(f"Q4_K top exponent bit classified as {s.stratum}")
        s = g.classify(q4k, q4k.abs_offset + 4, 0)          # first packed scale byte
        if s.field != "scales" or s.stratum != Stratum.PACKED_SCALE.value:
            failures.append(f"Q4_K byte 4 classified as {s.field}/{s.stratum}")
        s = g.classify(q4k, q4k.abs_offset + 16, 3)         # into qs
        if s.field != "qs" or s.blast != 1:
            failures.append(f"Q4_K byte 16 classified as {s.field}/blast {s.blast}")
        s = g.classify(q4k, q4k.abs_offset + 144, 0)        # first byte of block 1
        if s.block_index != 1 or s.field != "d":
            failures.append(f"block boundary wrong: {s.block_index}/{s.field}")

        # Q6_K puts d last, not first. If the field walk were order-blind this would fail.
        s = g.classify(q6k, q6k.abs_offset + 208, 0)
        if s.field != "d" or s.blast != 256:
            failures.append(f"Q6_K byte 208 classified as {s.field}/{s.blast}")
        s = g.classify(q6k, q6k.abs_offset + 192, 0)
        if s.field != "scales" or s.stratum != Stratum.INT_SCALE.value:
            failures.append(f"Q6_K byte 192 classified as {s.field}/{s.stratum}")

        # 3. Planning: right count, right strata, no duplicates, inside the data region.
        sites = plan(g, Stratum.all(), n=12, seed=1)
        by_s: Dict[str, int] = {}
        for st in sites:
            by_s[st.stratum] = by_s.get(st.stratum, 0) + 1
        for st in [Stratum.FP16_SCALE_EXPONENT, Stratum.FP16_SCALE_MANTISSA,
                   Stratum.FP16_SCALE_SIGN, Stratum.PACKED_SCALE,
                   Stratum.INT_SCALE, Stratum.PAYLOAD]:
            if by_s.get(st.value, 0) != 12:
                failures.append(f"stratum {st.value}: {by_s.get(st.value, 0)} sites, wanted 12")
        if len({s.key() for s in sites}) != len(sites):
            failures.append("planner produced duplicate sites")
        for st in sites:
            if not g.data_start <= st.abs_offset < end:
                failures.append(f"site {st.abs_offset} outside data region")
        # sign stratum is one bit in 16 of a scale field, so it must still be findable
        if by_s.get(Stratum.FP16_SCALE_SIGN.value, 0) == 0:
            failures.append("no sign-bit sites found")

        # 4. Same seed, same plan. Different seed, different plan.
        again = plan(g, Stratum.all(), n=12, seed=1)
        if [s.key() for s in again] != [s.key() for s in sites]:
            failures.append("planning is not reproducible from the seed")
        other = plan(g, Stratum.all(), n=12, seed=2)
        if [s.key() for s in other] == [s.key() for s in sites]:
            failures.append("different seeds produced identical plans")

        # 5. Injection flips exactly one bit and restores it exactly.
        before = open(path, "rb").read()
        site = sites[0]
        log = os.path.join(td, "repair.jsonl")
        with inject(path, site, guard=g, repair_log=log):
            during = open(path, "rb").read()
            diff = [i for i in range(len(before)) if before[i] != during[i]]
            if diff != [site.abs_offset]:
                failures.append(f"injection changed bytes {diff[:5]}, wanted [{site.abs_offset}]")
            if before[site.abs_offset] ^ during[site.abs_offset] != (1 << site.bit):
                failures.append("injection flipped the wrong bit")
        after = open(path, "rb").read()
        if after != before:
            failures.append("file not byte-identical after restore")

        # 6. Restores even when the body raises.
        try:
            with inject(path, sites[1], guard=g):
                raise KeyboardInterrupt("simulated")
        except KeyboardInterrupt:
            pass
        if open(path, "rb").read() != before:
            failures.append("file not restored after an exception in the block")

        # 7. The guard refuses a site from a different file.
        other_path = os.path.join(td, "u.gguf")
        _synthetic_gguf_with_data(other_path, seed=99)
        g2 = GGUF(other_path)
        g2.size += 1                     # pretend the file changed under us
        try:
            with inject(other_path, sites[2], guard=g2):
                failures.append("guard did not reject a mismatched file")
        except InjectionError:
            pass

        # 8. repair() undoes an abandoned flip.
        stray = sites[3]
        orig = _read_byte(path, stray.abs_offset)
        log2 = os.path.join(td, "repair2.jsonl")
        with open(log2, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"state": "open", "path": path,
                                 "offset": stray.abs_offset, "bit": stray.bit,
                                 "original": orig}) + "\n")
        _write_byte(path, stray.abs_offset, orig ^ (1 << stray.bit))
        n = repair(log2)
        if n != 1 or open(path, "rb").read() != before:
            failures.append("repair() did not restore the abandoned flip")

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("all checks passed")
    return 0


def _synthetic_gguf_with_data(path: str, seed: int = 3) -> None:
    """The header writer from gguf_faultscope, plus a real aligned data section."""
    import io

    def s(text: str) -> bytes:
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    rng = random.Random(seed)
    # (name, dims, ggml type id)
    specs = [("blk.0.attn_q.weight", [512, 512], 12),      # Q4_K
             ("blk.0.ffn_down.weight", [512, 1024], 14),   # Q6_K
             ("blk.1.attn_k.weight", [256, 256], 2),       # Q4_0
             ("blk.1.ffn_up.weight", [256, 512], 23),      # IQ4_XS
             ("output_norm.weight", [512], 0)]             # F32

    sizes = []
    for _, dims, ttype in specs:
        elements = 1
        for d in dims:
            elements *= d
        tname = GGML_TYPE_NAMES[ttype]
        lay = LAYOUTS.get(tname)
        if lay is not None and lay.weights > 1:
            sizes.append((elements // lay.weights) * lay.block_bytes)
        else:
            sizes.append(elements * SCALAR_TYPE_BYTES.get(tname, 4))

    head = bytearray()
    head += _GGUF_MAGIC
    head += struct.pack("<I", 3)
    head += struct.pack("<Q", len(specs))
    head += struct.pack("<Q", 2)
    head += s("general.architecture") + struct.pack("<I", 8) + s("llama")
    head += s("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", 32)

    offset = 0
    for (name, dims, ttype), nbytes in zip(specs, sizes):
        head += s(name)
        head += struct.pack("<I", len(dims))
        for d in dims:
            head += struct.pack("<Q", d)
        head += struct.pack("<I", ttype)
        head += struct.pack("<Q", offset)
        offset += nbytes + (-nbytes % 32)

    pad = (-len(head)) % 32
    body = bytearray(head) + b"\x00" * pad
    for nbytes in sizes:
        body += bytes(rng.randrange(256) for _ in range(nbytes))
        body += b"\x00" * (-nbytes % 32)

    with open(path, "wb") as fh:
        fh.write(bytes(body))


# `GGUF` learns how many tensors the header claimed, for the self test.
_orig_parse = GGUF._parse


def _parse_with_count(self) -> None:  # noqa: D401
    _orig_parse(self)
    self.n_tensors_expected = len(self.tensors)


GGUF._parse = _parse_with_count


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf")
    ap.add_argument("--plan", type=int, default=0, help="sites per stratum to print")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--repair", help="undo abandoned flips from a repair log")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if a.repair:
        print(f"{repair(a.repair)} repaired")
        return 0
    if not a.gguf:
        ap.print_help()
        return 1

    g = GGUF(a.gguf)
    print(json.dumps(g.summary(), indent=2))
    if a.plan:
        sites = plan(g, Stratum.all(), n=a.plan, seed=a.seed)
        print(f"\n{len(sites)} sites")
        for s in sites[:40]:
            print(f"  {s.abs_offset:>12} bit {s.bit}  {s.stratum:<22} "
                  f"{s.tensor_type:<7} blast {s.blast:>4}  {s.tensor}")
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump([s.as_dict() for s in sites], fh, indent=2)
            print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
