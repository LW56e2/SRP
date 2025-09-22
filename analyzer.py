import argparse
import csv
import glob
import os
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_rms_csv(path: str) -> Tuple[List[float], List[float], List[float], List[int]]:
    t, rms, peak, clip = [], [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is not None and header and header[0] not in ("t_seconds",):
            try:
                t.append(float(header[0]))
                rms.append(float(header[1]))
                peak.append(float(header[2]))
                clip.append(int(header[3]) if len(header) > 3 else 0)
            except Exception:
                pass

        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                t.append(float(row[0]))
                rms.append(float(row[1]))
                peak.append(float(row[2]))
                clip.append(int(row[3]) if len(row) > 3 else 0)
            except ValueError:
                continue
    return t, rms, peak, clip


def main():
    parser = argparse.ArgumentParser(description="Plots the RMS logs of audio files")
    parser.add_argument("folder", help="folder containing the csv files")
    parser.add_argument("--pattern", default="*_rms.csv",
                        help="ending pattern for CSVs within the folder (default: *_rms.csv)")
    parser.add_argument("--out-dir", default="plots", help="output folder for plots")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")

    paths = sorted(glob.glob(os.path.join(folder, args.pattern)))
    if not paths:
        raise SystemExit(f"No matching csv files in {folder!r}")


    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Found {len(paths)} files\n")

    for path in paths:
        fname = os.path.basename(path)
        stem = os.path.splitext(fname)[0]

        t, rms, peak, clip = read_rms_csv(path)
        if not t or not rms:
            print(f"skipping {fname}: no data!!!")
            continue

        max_rms_val = max(rms)
        idx_rms = rms.index(max_rms_val)
        t_rms = t[idx_rms]

        max_peak_val = max(peak) if peak else float("nan")
        idx_peak = peak.index(max_peak_val) if peak else None
        t_peak = t[idx_peak] if idx_peak is not None else float("nan")

        clip_events = sum(1 for c in clip if c) if clip else 0

        print(f"{fname}")
        print(f"  Max RMS : {max_rms_val:.6f} at t={t_rms:.3f}s")
        print(f"  Max Peak: {max_peak_val:.1f} at t={t_peak:.3f}s")
        print(f"  Clip/Near-clip events: {clip_events}")
        print()

        plt.figure()
        plt.plot(t, rms)
        plt.xlabel("Time (s)")
        plt.ylabel("Loudness (RMS, linear)")
        plt.title(f"{stem} — Sound Intensity Over Time")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()


        out_path = os.path.join(args.out_dir, f"{stem}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")


    plt.show()

if __name__ == "__main__":
    main()
