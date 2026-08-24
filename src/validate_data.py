"""Check that every shipped record matches the published schema.

A schema nobody validates against is documentation, not a contract. This runs in CI so that a
change to the record format either updates `schema/injection-record.schema.json` or fails,
which is the only way the schema stays true.

Deliberately dependency-free. Pulling `jsonschema` in would mean a contributor cannot run the
check without installing something, and the subset of the spec used here is small enough to
implement honestly in fifty lines.

    python src/validate_data.py
    python src/validate_data.py --data data/some-other-campaign/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCHEMA = os.path.join(ROOT, "schema", "injection-record.schema.json")

_PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
       "object": dict, "array": list, "null": type(None)}


def check(value: Any, spec: Dict, path: str, errors: List[str]) -> None:
    types = spec.get("type")
    if types is not None:
        allowed = types if isinstance(types, list) else [types]
        # bool is a subclass of int in Python, so integer must not accept True.
        ok = False
        for t in allowed:
            py = _PY.get(t)
            if py is None:
                continue
            if t in ("integer", "number") and isinstance(value, bool):
                continue
            if isinstance(value, py):
                ok = True
                break
        if not ok:
            errors.append(f"{path}: expected {allowed}, got {type(value).__name__}")
            return

    if "enum" in spec and value not in spec["enum"]:
        errors.append(f"{path}: {value!r} not in enum")
        return

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"{path}: {value} below minimum {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            errors.append(f"{path}: {value} above maximum {spec['maximum']}")

    if isinstance(value, dict):
        for req in spec.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required field {req!r}")
        for k, sub in (spec.get("properties") or {}).items():
            if k in value:
                check(value[k], sub, f"{path}.{k}", errors)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"),
                    help="a campaign directory, or the data root")
    ap.add_argument("--max-report", type=int, default=20)
    a = ap.parse_args()

    with open(SCHEMA, encoding="utf-8") as fh:
        schema = json.load(fh)

    files = sorted(glob.glob(os.path.join(a.data, "**", "injections-*.jsonl"),
                             recursive=True))
    files = [f for f in files if not f.endswith("-all.jsonl")] or files
    if not files:
        print(f"no injection files under {a.data}")
        return 1

    total, bad = 0, []
    for f in files:
        n = 0
        with open(f, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                n += 1
                total += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    bad.append(f"{os.path.basename(f)}:{i}: not JSON, {e}")
                    continue
                errs: List[str] = []
                check(rec, schema, "record", errs)
                bad.extend(f"{os.path.basename(f)}:{i}: {e}" for e in errs)
        print(f"  {os.path.basename(f):<40} {n:>6} records")

    print(f"\n{total} records across {len(files)} files")
    if bad:
        print(f"{len(bad)} violations, first {a.max_report}:")
        for b in bad[:a.max_report]:
            print("  ", b)
        return 1
    print("all records match schema/injection-record.schema.json")

    # The device table has to list every campaign directory, or the data is unfindable.
    devices = os.path.join(ROOT, "data", "devices.csv")
    if os.path.exists(devices):
        import csv
        with open(devices, encoding="utf-8") as fh:
            listed = {r["directory"] for r in csv.DictReader(fh)}
        present = {os.path.basename(os.path.dirname(f)) for f in files}
        missing = present - listed
        if missing:
            print(f"campaign directories not listed in devices.csv: {sorted(missing)}")
            return 1
        print(f"devices.csv lists all {len(present)} campaign directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
