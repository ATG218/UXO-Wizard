import json
import pandas as pd
from pathlib import Path
import argparse


def extract_concentrations(json_path: Path, output_csv: Path, spectro_eid: str = "STABSPECTRO_4112", gps_eid: str = "GPS_external", env_eid: str = "PTH_0007", altitude_eid: str = "ALT_external") -> pd.DataFrame:
    """Extract concentration style rows from *Tarva* raw JSON logs.

    Parameters
    ----------
    json_path : Path
        Path to the raw JSON file (each line is a JSON array).
    output_csv : Path
        Where to write the resulting CSV. Parent folders are created automatically.
    spectro_eid : str, optional
        eID to look for when parsing stabilized spectrometer packets, by default ``"STABSPECTRO_4112"``.
    gps_eid : str, optional
        eID for the GPS packets that contain Lat/Lon/Height, by default ``"GPS_external"``.
    env_eid : str, optional
        eID for environmental sensor packets with Press/Temp/Hum, by default ``"PTH_0007"``.
    altitude_eid : str, optional
        eID for barometric altitude packets, by default ``"ALT_external"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the extracted rows.
    """

    rows = []

    latest_gps = None  # cache of last GPS packet (dict)
    latest_env = None  # last environmental packet
    latest_alt = None  # last altitude packet

    with json_path.open("r") as fp:
        for line_num, raw in enumerate(fp, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                packets = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines but keep going.
                continue

            # Each line is an array with (usually) a single object
            if not isinstance(packets, list):
                packets = [packets]

            for pkt in packets:
                eid = pkt.get("eID", "")
                v = pkt.get("v") or {}

                # Update caches first – they might be needed for this spectro packet too
                if eid == gps_eid and v:
                    latest_gps = v
                elif eid == env_eid and v:
                    latest_env = v
                elif eid == altitude_eid and v:
                    latest_alt = v

                if eid != spectro_eid:
                    continue

                # Only keep packets that actually contain isotope information
                needed_keys = {"Total", "Countrate", "U238", "K40", "Th232", "Cs137"}
                if not needed_keys.issubset(v.keys()):
                    # Skip early STABSPECTRO packets before isotope fit is ready
                    continue

                ts = pkt.get("vT") or pkt.get("processed_timestamp")
                if ts is None:
                    # Can't align without timestamp
                    continue

                row = {
                    "timestamp": ts,
                    "Total": v.get("Total"),
                    "Countrate": v.get("Countrate"),
                    "U238": v.get("U238"),
                    "K40": v.get("K40"),
                    "Th232": v.get("Th232"),
                    "Cs137": v.get("Cs137"),
                }

                # Attach latest GPS data
                if latest_gps:
                    row["lat"] = latest_gps.get("Lat") or latest_gps.get("lat")
                    row["lon"] = latest_gps.get("Lon") or latest_gps.get("lon")
                    row["Height"] = latest_gps.get("Height") or latest_gps.get("HeightMSL")
                else:
                    row["lat"] = None
                    row["lon"] = None
                    row["Height"] = None

                # Environmental data
                if latest_env:
                    row["Press"] = latest_env.get("Press")
                    row["Temp"] = latest_env.get("Temp")
                    row["Hum"] = latest_env.get("Hum")
                else:
                    row["Press"] = row["Temp"] = row["Hum"] = None

                # If altitude packet present but GPS Height missing, use it
                if row["Height"] is None and latest_alt:
                    row["Height"] = latest_alt.get("altHeight")

                # Source file for traceability
                row["source_file"] = str(json_path)

                rows.append(row)

    if not rows:
        print("⚠️  No spectrometer packets with isotope data found – check eID or file integrity.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Reorder columns to match existing clean CSVs
    ordered_cols = [
        "timestamp", "lat", "lon", "Total", "Countrate", "U238", "K40", "Th232", "Cs137",
        "Height", "Press", "Temp", "Hum", "source_file",
    ]
    df = df.reindex(columns=ordered_cols)

    # Sort by timestamp for consistency
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Extracted {len(df)} concentration records to {output_csv}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract concentration-like data from Tarva JSON logs.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input-json", type=str, help="Path to a single raw JSON file")
    grp.add_argument("--input-dir", type=str, help="Directory containing multiple raw JSON logs to ingest")

    parser.add_argument("--output-csv", type=str, default=None, help="Destination CSV file (default: <input>.csv or combined_concentrations_clean.csv)")

    # Sensor IDs can be overridden from the CLI
    parser.add_argument("--spectro-eid", default="STABSPECTRO_4112", help="eID for stabilised spectrometer packets")
    parser.add_argument("--gps-eid", default="GPS_0007", help="eID for GPS packets to take Lat/Lon from")
    parser.add_argument("--env-eid", default="PTH_0007", help="eID for environmental sensor packets")
    parser.add_argument("--altitude-eid", default="ALT_external", help="eID for barometric altitude packets")
    args = parser.parse_args()

    # Collect DataFrames from one or many files
    dfs = []

    if args.input_json:
        input_path = Path(args.input_json)
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        # Derive default output name if none supplied
        if args.output_csv:
            out_path = Path(args.output_csv)
        else:
            out_path = input_path.with_suffix(".csv")

        df = extract_concentrations(
            input_path,
            out_path,
            spectro_eid=args.spectro_eid,
            gps_eid=args.gps_eid,
            env_eid=args.env_eid,
            altitude_eid=args.altitude_eid,
        )
        if not df.empty:
            dfs.append(df)
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(input_dir)

        json_files = sorted(input_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in {input_dir}")

        for jp in json_files:
            print(f"➡️  Processing {jp.name}…")
            # Use temporary csv path (ignored afterwards)
            tmp_csv = jp.with_suffix(".extracted.csv")
            df = extract_concentrations(
                jp,
                tmp_csv,
                spectro_eid=args.spectro_eid,
                gps_eid=args.gps_eid,
                env_eid=args.env_eid,
                altitude_eid=args.altitude_eid,
            )
            if not df.empty:
                dfs.append(df)

        # Cleanup temporary per-file outputs (optional)
        for jp in json_files:
            tmp_csv = jp.with_suffix(".extracted.csv")
            if tmp_csv.exists():
                tmp_csv.unlink()

        if args.output_csv:
            out_path = Path(args.output_csv)
        else:
            out_path = Path("concentrations_combined_clean.csv")

    # Combine & write final CSV if multiple DataFrames accumulated
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        print(f"📊 Combined {len(combined)} rows from {len(dfs)} file(s) → {out_path}")
    else:
        print("⚠️  No data extracted from any file.") 