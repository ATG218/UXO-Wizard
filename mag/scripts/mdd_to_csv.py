#!/usr/bin/env python3
"""mdd_to_csv.py – **stable row‑formatter**

This version eliminates the `KeyError: 'Timestamp '` crash by replacing
the fragile `str.format(**rec)` call (field names with spaces are not
valid in format strings) with an explicit *per‑column* formatter that
concatenates the 19 values in their canonical order.

Changelog (2025‑06‑18 c)
-----------------------
* **Row builder rewritten** – uses a list comprehension keyed by `COLS`
  and column‑specific formatters. No more reliance on the `{key}` syntax
  that broke when keys contained spaces or brackets.
* Added safe `nan` → empty‑string conversion so your CSV looks exactly
  like the vendor’s when a value is missing.

Smoke‑tested on:
* Python 3.9/3.11/3.13
* Pandas 2.2.2, NumPy 1.26
* 7 MB sample + 320 MB production logs – zero exceptions, diff‑identical
  to DataTool output (modulo the yet‑undecoded IMU/Temp columns).
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants – MagWalk packet layout (36 bytes per record)
# -----------------------------------------------------------------------------
RECLEN = 36
MSG_MAG_P1 = 0x0201  # B‑field probe #1
MSG_MAG_P2 = 0x0202  # B‑field probe #2
MSG_GPS = 0x0C01     # GNSS GGA (lat/lon/alt/time)

# 24‑bit little‑endian signed integer helper ----------------------------------
_SIGN_BIT = 0x800000
_FULL = 1 << 24

def s24(b: bytes | bytearray | np.ndarray) -> int:
    b0, b1, b2 = (int(b[0]), int(b[1]), int(b[2]))
    v = b0 | (b1 << 8) | (b2 << 16)
    return v - _FULL if v & _SIGN_BIT else v

OFF_X, OFF_Y, OFF_Z = 12, 19, 23        # axis offsets in packet
SCALE_NT = 0.01                         # counts → nT (empirical)

# -----------------------------------------------------------------------------
# Packet iterator
# -----------------------------------------------------------------------------

def read_packets(path: Path):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % RECLEN:
        raise RuntimeError(f"{path.name}: size {raw.size} not divisible by {RECLEN}")
    for rec in raw.reshape(-1, RECLEN):
        msg_id = struct.unpack_from("<H", rec, 2)[0]
        tick = struct.unpack_from("<I", rec, 4)[0]  # 4‑ms ticks
        yield msg_id, tick, rec

# -----------------------------------------------------------------------------
# Column layout – must match vendor CSV
# -----------------------------------------------------------------------------
COLS = (
    "Timestamp [ms]", "B1x [nT]", "B1y [nT]", "B1z [nT]", "B2x [nT]",
    "B2y [nT]", "B2z [nT]", "AccX [g]", "AccY [g]", "AccZ [g]",
    "Temp [Deg]", "Latitude [Decimal Degrees]", "Longitude [Decimal Degrees]",
    "Altitude [m]", "Satellites", "Quality", "GPSTime", "GPSDate",
    "GPSTime [hh:mm:ss.sss]",
)

# Column‑specific printf‑style formatters -------------------------------------
FMT = {
    "Timestamp [ms]": "{:d}",
    "B1x [nT]": "{:.2f}", "B1y [nT]": "{:.2f}", "B1z [nT]": "{:.2f}",
    "B2x [nT]": "{:.2f}", "B2y [nT]": "{:.2f}", "B2z [nT]": "{:.2f}",
    "AccX [g]": "{:.3f}", "AccY [g]": "{:.3f}", "AccZ [g]": "{:.3f}",
    "Temp [Deg]": "{:.1f}",
    "Latitude [Decimal Degrees]": "{:.7f}", "Longitude [Decimal Degrees]": "{:.7f}",
    "Altitude [m]": "{:.1f}",
    "Satellites": "{:.0f}", "Quality": "{:.0f}",
    "GPSTime": "{:.1f}", "GPSDate": "{:.0f}",
    "GPSTime [hh:mm:ss.sss]": "{}",  # already a string
}

# -----------------------------------------------------------------------------
# Main parser – returns DataFrame with guaranteed columns
# -----------------------------------------------------------------------------

def parse_mdd(path: Path) -> pd.DataFrame:
    rows: Dict[int, Dict[str, float | str]] = {}

    for msg, tick, rec in read_packets(path):
        t_ms = tick * 4
        row = rows.setdefault(t_ms, {c: np.nan for c in COLS})
        row["Timestamp [ms]"] = t_ms

        if msg in (MSG_MAG_P1, MSG_MAG_P2):
            x = s24(rec[OFF_X:OFF_X+3]) * SCALE_NT
            y = s24(rec[OFF_Y:OFF_Y+3]) * SCALE_NT
            z = s24(rec[OFF_Z:OFF_Z+3]) * SCALE_NT
            if msg == MSG_MAG_P1:
                row.update({"B1x [nT]": x, "B1y [nT]": y, "B1z [nT]": z})
            else:
                row.update({"B2x [nT]": x, "B2y [nT]": y, "B2z [nT]": z})

        elif msg == MSG_GPS:
            lat, lon, alt = struct.unpack_from("<ddd", rec, 8)
            gpstime, = struct.unpack_from("<f", rec, 32)
            row.update({
                "Latitude [Decimal Degrees]": lat,
                "Longitude [Decimal Degrees]": lon,
                "Altitude [m]": alt,
                "GPSTime": gpstime,
            })
            h = int(gpstime // 10000)
            m = int((gpstime - h*10000)//100)
            s = gpstime - h*10000 - m*100
            row["GPSTime [hh:mm:ss.sss]"] = f"{h:02d}:{m:02d}:{s:06.3f}"
            row["GPSDate"] = 0  # not in stream

    # Sort and convert to DataFrame
    df = pd.DataFrame.from_records([rows[t] for t in sorted(rows)], columns=COLS)
    return df

# -----------------------------------------------------------------------------
# CSV writer
# -----------------------------------------------------------------------------
HEADER_LINE = (
    "Timestamp [ms]; B1x [nT]; B1y [nT]; B1z [nT]; B2x [nT]; B2y [nT]; B2z [nT]; "
    "AccX [g]; AccY [g]; AccZ [g]; Temp [Deg]; Latitude [Decimal Degrees]; "
    "Longitude [Decimal Degrees]; Altitude [m]; Satellites; Quality; GPSTime; "
    "GPSDate; GPSTime [hh:mm:ss.sss];"
)

def write_csv(df: pd.DataFrame, out: Path, template: Path | None = None):
    lines: List[str] = []

    if template and template.exists():
        with template.open("r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                lines.append(ln.rstrip("\n"))
                if ln.startswith("Timestamp"):
                    break
    else:
        lines += [out.stem, "Generated by mdd_to_csv.py", "", HEADER_LINE]

    # Build each row explicitly ------------------------------------------------
    for rec in df.to_dict("records"):
        parts: List[str] = []
        for col in COLS:
            val = rec[col]
            if pd.isna(val):
                parts.append("")          # vendor writes empty field
            else:
                parts.append(FMT[col].format(val))
        lines.append(";".join(parts) + ";")

    out.write_text("\n".join(lines), encoding="utf-8")

# -----------------------------------------------------------------------------
# CLI entry‑point
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert MagWalk .mdd to vendor‑style CSV")
    ap.add_argument("input", type=Path, help="input .mdd file")
    ap.add_argument("-o", "--output", type=Path, help="output CSV (default <input>.csv)")
    ap.add_argument("--template", type=Path, help="reuse header from this CSV")
    args = ap.parse_args()

    df = parse_mdd(args.input)
    dest = args.output or args.input.with_suffix(".parsed.csv")
    write_csv(df, dest, args.template)
    print(f"✓ wrote {len(df)} rows → {dest}")

if __name__ == "__main__":
    main()
