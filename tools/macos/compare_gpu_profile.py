#!/usr/bin/env python3
"""Compare BB_GPU_PROFILE output between runs (in-process vs native GPU process, A/B switches).

    python3 tools/macos/compare_gpu_profile.py out/run1.log out/run2.log [out/run3.log ...]

Every "GPU profile:" block printed while playing (GPU time at least --min-ms per frame, so menus
and loading screens drop out) is averaged per label, weighted by its frames. The table lists the
labels with the most GPU time, each run's ms per frame, and the change against the first run.
"""
import argparse
import re
import statistics
import sys
from collections import defaultdict

HEADER = re.compile(r"^GPU profile: ([0-9.]+) ms/frame over (\d+) frames")
ROW = re.compile(r"^\s+([0-9.]+) ms/frame\s+([0-9.]+)/frame\s+(.*)$")


def read(path, min_ms):
    blocks = []
    current = None
    with open(path, errors="replace") as f:
        for line in f:
            header = HEADER.match(line)
            if header:
                current = {"total": float(header.group(1)), "frames": int(header.group(2)),
                           "rows": {}}
                blocks.append(current)
                continue
            row = ROW.match(line) if current else None
            if row:
                current["rows"][row.group(3).strip()] = float(row.group(1))
            elif current and not line.startswith("  "):
                current = None
    playing = [b for b in blocks if b["total"] >= min_ms]
    frames = sum(b["frames"] for b in playing)
    labels = defaultdict(float)
    for block in playing:
        for label, ms in block["rows"].items():
            labels[label] += ms * block["frames"]
    if frames:
        for label in labels:
            labels[label] /= frames
    totals = [b["total"] for b in playing]
    return {
        "blocks": len(playing),
        "frames": frames,
        "median": statistics.median(totals) if totals else 0.0,
        "mean": sum(b["total"] * b["frames"] for b in playing) / frames if frames else 0.0,
        "labels": labels,
    }


def short(label, width):
    return label if len(label) <= width else label[: width - 1] + "…"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--min-ms", type=float, default=5.0,
                        help="skip profile blocks below this GPU time per frame (menus)")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--width", type=int, default=70)
    args = parser.parse_args()

    runs = [read(path, args.min_ms) for path in args.logs]
    for path, run in zip(args.logs, runs):
        print(f"{path}: {run['blocks']} blocks, {run['frames']} frames, GPU "
              f"{run['mean']:.2f} ms/frame (median block {run['median']:.2f})")
    if not any(run["frames"] for run in runs):
        print("no GPU profile blocks found (BB_GPU_PROFILE=1, or the GPU profile switch)")
        return 1

    names = [f"run{i + 1}" for i in range(len(runs))]
    every = set().union(*(run["labels"] for run in runs))
    weight = {label: max(run["labels"].get(label, 0.0) for run in runs) for label in every}
    rows = sorted(every, key=lambda label: weight[label], reverse=True)[: args.top]
    print()
    print(f"{'label':<{args.width}} " + " ".join(f"{n:>8}" for n in names)
          + ("   vs run1" if len(runs) > 1 else ""))
    for label in rows:
        values = [run["labels"].get(label) for run in runs]
        cells = " ".join(f"{v:8.3f}" if v is not None else f"{'-':>8}" for v in values)
        change = ""
        if len(runs) > 1 and values[0]:
            changes = [f"{(v / values[0] - 1) * 100:+.0f}%" if v is not None else "-"
                       for v in values[1:]]
            change = "   " + " ".join(f"{c:>6}" for c in changes)
        print(f"{short(label, args.width):<{args.width}} {cells}{change}")
    totals = " ".join(f"{run['mean']:8.2f}" for run in runs)
    print(f"{'all GPU work':<{args.width}} {totals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
