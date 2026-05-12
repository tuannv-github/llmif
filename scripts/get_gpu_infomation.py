#!/usr/bin/env python3
"""GPU report from NVIDIA tooling only: ``nvidia-smi`` CSV + optional NVML details.

Per-GPU SM / CUDA core / tensor core totals use NVML (``pip install nvidia-ml-py``), the
same management API ``nvidia-smi`` uses — no PyTorch or CUDA toolkit required for this script.

Default: two-column key | value with units, plus CSV file (field,value,unit rows; default gpu_report.csv, see -o).
CSV on stdout: --csv (after the table) or --csv-only. Use --no-csv-file to skip writing a file.
Use --wide for one row per GPU. Optional --topology / --full append `nvidia-smi` extras.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def _run(cmd: Sequence[str], *, timeout: int = 120) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", "command timed out"


def _nvidia_smi_csv(fields: str) -> tuple[int, str, str]:
    return _run(
        (
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ),
    )


def _query_gpu_table(fields: str) -> list[dict[str, str]]:
    code, out, err = _nvidia_smi_csv(fields)
    if code != 0:
        raise RuntimeError(err.strip() or f"nvidia-smi exited {code}")
    reader = csv.reader(io.StringIO(out))
    names = [n.strip() for n in fields.split(",")]
    rows: list[dict[str, str]] = []
    for parts in reader:
        parts = [p.strip() for p in parts]
        if len(parts) != len(names):
            continue
        if not any(parts):
            continue
        rows.append({names[i]: parts[i] for i in range(len(names))})
    return rows


_QUERY_TRIES: tuple[str, ...] = (
    "index,name,uuid,serial,driver_version,vbios_version,"
    "pci.bus_id,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,"
    "compute_cap,mig.mode.current,"
    "memory.total,memory.used,memory.free,memory.reserved,"
    "utilization.gpu,utilization.memory,"
    "temperature.gpu,temperature.memory,"
    "power.draw,power.limit,power.default_limit,"
    "clocks.current.graphics,clocks.max.graphics,"
    "clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory,"
    "ecc.mode.current,persistence_mode,pstate,fan.speed",
    "index,name,uuid,driver_version,pci.bus_id,"
    "pcie.link.gen.current,pcie.link.width.current,"
    "memory.total,memory.used,memory.free,"
    "utilization.gpu,utilization.memory,temperature.gpu,"
    "power.draw,power.limit,clocks.current.sm,clocks.max.sm,pstate",
)

_NVML_DETAIL_KEYS: tuple[str, ...] = (
    "nvml.sm_count",
    "nvml.architecture",
    "nvml.cuda_cores_per_sm",
    "nvml.tensor_cores_per_sm",
    "nvml.tensor_core_note",
    "nvml.cuda_cores_total",
    "nvml.tensor_cores_total",
    "nvml.total_memory_gib",
)

# Suffixes for human-readable text (nvidia-smi CSV uses MiB for memory fields; see NVIDIA docs).
_FIELD_UNITS: dict[str, str] = {
    "memory.total": " MiB",
    "memory.used": " MiB",
    "memory.free": " MiB",
    "memory.reserved": " MiB",
    "power.draw": " W",
    "power.limit": " W",
    "power.default_limit": " W",
    "clocks.current.graphics": " MHz",
    "clocks.max.graphics": " MHz",
    "clocks.current.sm": " MHz",
    "clocks.max.sm": " MHz",
    "clocks.current.memory": " MHz",
    "clocks.max.memory": " MHz",
    "utilization.gpu": " %",
    "utilization.memory": " %",
    "temperature.gpu": " °C",
    "temperature.memory": " °C",
    "fan.speed": " %",
    "pcie.link.gen.max": " (PCIe)",
    "pcie.link.gen.current": " (PCIe)",
    "pcie.link.width.max": " lanes",
    "pcie.link.width.current": " lanes",
    "nvml.total_memory_gib": " GiB",
    "nvml.sm_count": " SMs",
    "nvml.cuda_cores_per_sm": " (per SM)",
    "nvml.tensor_cores_per_sm": " (per SM)",
    "nvml.cuda_cores_total": " CUDA cores",
    "nvml.tensor_cores_total": " Tensor cores",
}


@dataclass(frozen=True)
class _ArchSpec:
    cuda_cores_per_sm: int
    tensor_cores_per_sm: int
    arch_label: str
    tensor_note: str


_CC_ARCH: dict[tuple[int, int], _ArchSpec] = {
    (3, 0): _ArchSpec(192, 0, "Kepler", "—"),
    (3, 2): _ArchSpec(192, 0, "Kepler", "—"),
    (3, 5): _ArchSpec(192, 0, "Kepler", "—"),
    (3, 7): _ArchSpec(192, 0, "Kepler", "—"),
    (5, 0): _ArchSpec(128, 0, "Maxwell", "—"),
    (5, 2): _ArchSpec(128, 0, "Maxwell", "—"),
    (5, 3): _ArchSpec(128, 0, "Maxwell", "—"),
    (6, 0): _ArchSpec(64, 0, "Pascal (GP100)", "—"),
    (6, 1): _ArchSpec(128, 0, "Pascal", "—"),
    (6, 2): _ArchSpec(128, 0, "Pascal", "—"),
    (7, 0): _ArchSpec(64, 8, "Volta", "1st gen"),
    (7, 2): _ArchSpec(64, 8, "Volta", "1st gen"),
    (7, 5): _ArchSpec(64, 8, "Turing", "2nd gen"),
    (8, 0): _ArchSpec(64, 4, "Ampere (A100-class)", "3rd gen"),
    (8, 6): _ArchSpec(128, 4, "Ampere", "3rd gen"),
    (8, 7): _ArchSpec(128, 4, "Ampere (Jetson)", "3rd gen"),
    (8, 9): _ArchSpec(128, 4, "Ada Lovelace", "4th gen"),
    (9, 0): _ArchSpec(128, 4, "Hopper", "4th gen"),
    (10, 0): _ArchSpec(128, 4, "Blackwell (approx.)", "5th gen"),
    (10, 1): _ArchSpec(128, 4, "Blackwell (approx.)", "5th gen"),
    (12, 0): _ArchSpec(128, 4, "Blackwell (GB20x, approx.)", "5th gen"),
}


def _arch_for_cc(major: int, minor: int) -> _ArchSpec | None:
    if (major, minor) in _CC_ARCH:
        return _CC_ARCH[major, minor]
    for m in range(minor, -1, -1):
        if (major, m) in _CC_ARCH:
            return _CC_ARCH[major, m]
    for m in range(minor + 1, 10):
        if (major, m) in _CC_ARCH:
            return _CC_ARCH[major, m]
    return None


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _parse_driver_cuda_versions(smi_summary: str) -> tuple[str | None, str | None]:
    driver = re.search(r"Driver Version:\s*(\S+)", smi_summary)
    cuda = re.search(r"CUDA Version:\s*(\S+)", smi_summary)
    return (
        driver.group(1).strip() if driver else None,
        cuda.group(1).strip() if cuda else None,
    )


def _row_for_gpu_index(rows: list[dict[str, str]], index: int) -> dict[str, str] | None:
    s = str(index)
    for r in rows:
        if r.get("index", "").strip() == s:
            return r
    return None


def _nvml_gpu_detail_rows(nvidia_csv_gpu_count: int) -> tuple[dict[int, dict[str, str]], list[str]]:
    """SM / core / memory details from NVML (``nvidia-ml-py``). If ``nvidia_csv_gpu_count`` is 0, query all NVML GPUs."""
    warnings: list[str] = []
    out: dict[int, dict[str, str]] = {}
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        warnings.append(
            "nvidia-ml-py not installed — run `pip install nvidia-ml-py` for SM and CUDA core columns.",
        )
        return out, warnings
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 — best-effort diagnostics
        warnings.append(f"NVML init failed ({exc}); SM/core columns omitted.")
        return out, warnings
    try:
        try:
            n_nvml = int(pynvml.nvmlDeviceGetCount())
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"NVML device count failed ({exc}).")
            return out, warnings
        limit = nvidia_csv_gpu_count if nvidia_csv_gpu_count > 0 else n_nvml
        n_loop = min(limit, n_nvml)
        if nvidia_csv_gpu_count > n_nvml:
            warnings.append(
                f"nvidia-smi reports {nvidia_csv_gpu_count} GPUs but NVML sees {n_nvml} (capped).",
            )
        for i in range(n_loop):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    cores = int(pynvml.nvmlDeviceGetNumGpuCores(handle))
                except Exception:
                    cores = -1
                arch = _arch_for_cc(int(major), int(minor))
                g: dict[str, str] = {
                    "nvml.total_memory_gib": f"{mem.total / (1024**3):.6f}",
                }
                if arch is None:
                    g.update(
                        {
                            "nvml.architecture": "",
                            "nvml.sm_count": "",
                            "nvml.cuda_cores_per_sm": "",
                            "nvml.tensor_cores_per_sm": "",
                            "nvml.tensor_core_note": "",
                            "nvml.cuda_cores_total": "",
                            "nvml.tensor_cores_total": "",
                        },
                    )
                    warnings.append(
                        f"GPU {i}: compute capability {int(major)}.{int(minor)} not in built-in table — "
                        "NVML CUDA core totals omitted.",
                    )
                elif cores < 0:
                    g.update(
                        {
                            "nvml.architecture": arch.arch_label,
                            "nvml.sm_count": "",
                            "nvml.cuda_cores_per_sm": str(arch.cuda_cores_per_sm),
                            "nvml.tensor_cores_per_sm": str(arch.tensor_cores_per_sm),
                            "nvml.tensor_core_note": arch.tensor_note,
                            "nvml.cuda_cores_total": "",
                            "nvml.tensor_cores_total": "",
                        },
                    )
                    warnings.append(f"GPU {i}: NVML did not report CUDA core count.")
                elif arch.cuda_cores_per_sm <= 0:
                    g["nvml.architecture"] = arch.arch_label
                    g["nvml.sm_count"] = ""
                    warnings.append(f"GPU {i}: zero cuda_cores_per_sm in arch table.")
                elif cores % arch.cuda_cores_per_sm != 0:
                    sm = cores // arch.cuda_cores_per_sm
                    g.update(
                        {
                            "nvml.architecture": arch.arch_label,
                            "nvml.sm_count": str(sm),
                            "nvml.cuda_cores_per_sm": str(arch.cuda_cores_per_sm),
                            "nvml.tensor_cores_per_sm": str(arch.tensor_cores_per_sm),
                            "nvml.tensor_core_note": arch.tensor_note,
                            "nvml.cuda_cores_total": str(cores),
                            "nvml.tensor_cores_total": str(sm * arch.tensor_cores_per_sm),
                        },
                    )
                    warnings.append(
                        f"GPU {i}: NVML CUDA core count {cores} is not divisible by "
                        f"{arch.cuda_cores_per_sm}/SM — SM count may be approximate.",
                    )
                else:
                    sm = cores // arch.cuda_cores_per_sm
                    g.update(
                        {
                            "nvml.architecture": arch.arch_label,
                            "nvml.sm_count": str(sm),
                            "nvml.cuda_cores_per_sm": str(arch.cuda_cores_per_sm),
                            "nvml.tensor_cores_per_sm": str(arch.tensor_cores_per_sm),
                            "nvml.tensor_core_note": arch.tensor_note,
                            "nvml.cuda_cores_total": str(sm * arch.cuda_cores_per_sm),
                            "nvml.tensor_cores_total": str(sm * arch.tensor_cores_per_sm),
                        },
                    )
                out[i] = g
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"GPU {i}: NVML query failed ({exc}).")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out, warnings


def _fetch_nvidia_smi_brief_and_rows() -> tuple[int, str, str, list[dict[str, str]] | None, str, str]:
    code, brief_out, brief_err = _run(("nvidia-smi",))
    last_err = ""
    for fields in _QUERY_TRIES:
        try:
            rows = _query_gpu_table(fields)
            return code, brief_out, brief_err, rows, "", fields
        except RuntimeError as e:
            last_err = str(e)
    return code, brief_out, brief_err, None, last_err, ""


def _merge_gpu_snapshot(
    nvidia_rows: list[dict[str, str]] | None,
    smi_summary_stdout: str,
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    warnings: list[str] = []
    rn = list(nvidia_rows or [])
    drv, cuda = _parse_driver_cuda_versions(smi_summary_stdout)

    meta: dict[str, str] = {
        "driver_version_banner": drv or "",
        "cuda_version_banner": cuda or "",
    }

    n_nvidia = len(rn)
    nvml_per_gpu, nvml_warnings = _nvml_gpu_detail_rows(n_nvidia)
    warnings.extend(nvml_warnings)

    n_nvml_detail = len(nvml_per_gpu)
    count = max(n_nvidia, n_nvml_detail)
    if count == 0:
        return [], [], warnings + ["No GPU devices found."]

    if n_nvidia == 0 and n_nvml_detail > 0:
        warnings.append("No nvidia-smi CSV rows — report uses NVML device indices only.")

    empty_nvml = {k: "" for k in _NVML_DETAIL_KEYS}
    rows_out: list[dict[str, str]] = []
    for i in range(count):
        row: dict[str, str] = dict(meta)
        if rn:
            r = _row_for_gpu_index(rn, i)
            if r is None and i < len(rn):
                r = rn[i]
            if r:
                row.update(r)
            else:
                row["index"] = str(i)
                warnings.append(f"No nvidia-smi CSV row for GPU index {i}.")
        else:
            row["index"] = str(i)
        row.update(nvml_per_gpu.get(i, dict(empty_nvml)))
        rows_out.append(row)

    nvidia_key_order = list((_row_for_gpu_index(rn, 0) or rn[0]).keys()) if rn else []

    global_keys = [
        "driver_version_banner",
        "cuda_version_banner",
    ]
    order: list[str] = []
    seen: set[str] = set()
    for k in global_keys + nvidia_key_order + list(_NVML_DETAIL_KEYS):
        if k not in seen:
            order.append(k)
            seen.add(k)
    for r in rows_out:
        for k in r:
            if k not in seen:
                order.append(k)
                seen.add(k)

    return order, rows_out, warnings


def _looks_like_number(s: str) -> bool:
    t = s.strip().replace(",", "")
    if not t or t in ("N/A", "[N/A]"):
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


def _with_unit(key: str, raw: str) -> str:
    """Append a unit label for known fields (text / wide table only)."""
    s = (raw or "").strip()
    if not s or s in ("N/A", "[N/A]"):
        return s
    unit = _FIELD_UNITS.get(key)
    if not unit:
        return s
    if unit in (" MiB", " W", " MHz", " %", " °C") and not _looks_like_number(s):
        return s
    if unit in (" lanes",) and not _looks_like_number(s):
        return s
    if unit.endswith("SMs") or "cores" in unit or "per SM" in unit:
        if not _looks_like_number(s):
            return s
    if key == "nvml.total_memory_gib" and not _looks_like_number(s):
        return s
    if unit == " (PCIe)" and _looks_like_number(s):
        return f"PCIe Gen {s}"
    return f"{s}{unit}"


def _row_display(columns: Sequence[str], row: dict[str, str]) -> list[str]:
    return [_with_unit(c, (row.get(c, "") or "").replace("\n", " ")) for c in columns]


def _print_kv_two_column(columns: Sequence[str], row: dict[str, str]) -> None:
    """Print one GPU as key | value rows (column order preserved)."""
    keys = list(columns)
    if not keys:
        return
    vals = _row_display(keys, row)
    kw = max(len("key"), max(len(k) for k in keys))
    vw = max(len("value"), max(len(v) for v in vals))
    print(f"{'key'.ljust(kw)} | value (MiB = mebibytes; GiB = gibibytes)")
    print(f"{'-' * kw}-+-{'-' * vw}")
    for k, v in zip(keys, vals):
        print(f"{k.ljust(kw)} | {v}")


def _print_aligned_columns(columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    disp_rows = [_row_display(columns, r) for r in rows]
    widths = {
        c: max(len(c), *(len(disp_rows[i][j]) for i in range(len(rows))))
        for j, c in enumerate(columns)
    }
    print(" | ".join(c.ljust(widths[c]) for c in columns))
    print("-+-".join("-" * widths[c] for c in columns))
    for disp in disp_rows:
        print(" | ".join(disp[j].ljust(widths[c]) for j, c in enumerate(columns)))


def _csv_unit_for_field(key: str) -> str:
    """Third CSV column: unit label for `field` (empty if none)."""
    raw = _FIELD_UNITS.get(key)
    if not raw:
        return ""
    t = raw.strip()
    if t == "(PCIe)":
        return "PCIe gen"
    if t == "(per SM)":
        return "per SM"
    return t


def _write_long_csv(fp, columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    """Write field, value, unit rows (one row per field per GPU)."""
    w = csv.writer(fp, lineterminator="\n")
    w.writerow(["field", "value", "unit"])
    multi = len(rows) > 1
    for ri, row in enumerate(rows):
        prefix = f"gpu[{ri}]." if multi else ""
        for c in columns:
            val = (row.get(c, "") or "").replace("\n", " ")
            w.writerow([f"{prefix}{c}", val, _csv_unit_for_field(c)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        default="gpu_report.csv",
        metavar="PATH",
        help="Write long-format CSV (field,value,unit) to this path (default: gpu_report.csv). Ignored with --csv-only.",
    )
    parser.add_argument(
        "--no-csv-file",
        action="store_true",
        help="Do not write a CSV file (only affects default / -o behavior).",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Print one wide pipe-separated row per GPU instead of key | value.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="After the text table, print long CSV (field,value,unit) on stdout.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Print only long CSV (field,value,unit) on stdout (warnings on stderr).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="After the report, print `nvidia-smi -q` (very verbose).",
    )
    parser.add_argument(
        "--topology",
        action="store_true",
        help="After the report, print `nvidia-smi topo -m`.",
    )
    args = parser.parse_args()

    brief_out = ""
    rows_merge: list[dict[str, str]] | None = None
    if shutil.which("nvidia-smi"):
        _code, brief_out, _berr, rows_merge, last_err, _fields = _fetch_nvidia_smi_brief_and_rows()
        if rows_merge is None and last_err:
            print(last_err, file=sys.stderr)
    else:
        print(
            "nvidia-smi not in PATH; SM/query columns need nvidia-smi or nvidia-ml-py (NVML).",
            file=sys.stderr,
        )

    cols, merged, warns = _merge_gpu_snapshot(rows_merge, brief_out)
    for w in warns:
        print(w, file=sys.stderr)
    if not merged:
        print("No GPU data to output.", file=sys.stderr)
        return 1

    if args.csv_only:
        _write_long_csv(sys.stdout, cols, merged)
    else:
        if args.wide:
            _print_aligned_columns(cols, merged)
        else:
            for gi, row in enumerate(merged):
                if len(merged) > 1:
                    gpu_id = row.get("index", str(gi)).strip() or str(gi)
                    print(f"GPU {gpu_id}")
                    print()
                _print_kv_two_column(cols, row)
                if gi < len(merged) - 1:
                    print()
        if not args.no_csv_file:
            out_path = Path(args.output).expanduser()
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", newline="", encoding="utf-8") as fp:
                    _write_long_csv(fp, cols, merged)
            except OSError as exc:
                print(f"Could not write CSV file {out_path}: {exc}", file=sys.stderr)
                return 1
            print(f"Wrote CSV: {out_path.resolve()}", file=sys.stderr)
        if args.csv:
            print()
            _print_section("CSV")
            _write_long_csv(sys.stdout, cols, merged)

    if args.topology:
        _print_section("Topology (nvidia-smi topo -m)")
        t_code, t_out, t_err = _run(("nvidia-smi", "topo", "-m"))
        print((t_out if t_code == 0 else t_err or t_out).rstrip())

    if args.full:
        _print_section("Full query (nvidia-smi -q)")
        q_code, q_out, q_err = _run(("nvidia-smi", "-q"), timeout=300)
        print((q_out if q_code == 0 else q_err or q_out).rstrip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
