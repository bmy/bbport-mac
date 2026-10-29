#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Hitches around area loads, from the frames.csv of a session (BB_FRAME_LOG / the launcher's log).

  load_hitches.py FRAMES.csv [FRAMES.csv ...] [--gap S] [--before S] [--after S] [--target MS]

A load is a burst of image registrations or device memory allocations (a lamp travel, a gate
into a new area): the first frame of a burst starts it, bursts closer than --gap seconds (8) are
one. For each load the window [start - before (1), start + after (12)] is judged by the frame
times: the worst frame, frames over 60 / 100 ms, and the time lost over --target (33.3 ms, the
30 FPS frame) summed. A summary of all loads follows; compare builds by it when the route is
walked the same way (the loads differ a little by the player's path).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys


def load_frames(path: str):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in row.items() if k and v not in (None, "")})
            except ValueError:
                continue
    return rows


def find_loads(rows, gap: float, min_images: float, min_alloc_mb: float):
    starts = []
    last_burst = -1e9
    for row in rows:
        burst = row.get("images", 0) >= min_images or row.get("alloc_mb", 0) >= min_alloc_mb
        if not burst:
            continue
        t = row["t_s"]
        if t - last_burst >= gap:
            starts.append(t)
        last_burst = t
    return starts


def judge(rows, t0: float, before: float, after: float, target: float):
    window = [r for r in rows if t0 - before <= r["t_s"] <= t0 + after]
    if not window:
        return None
    frames = [r["frame_ms"] for r in window]
    return {
        "t": t0,
        "frames": len(window),
        "worst": max(frames),
        "over60": sum(1 for f in frames if f > 60),
        "over100": sum(1 for f in frames if f > 100),
        "lost": sum(max(0.0, f - target) for f in frames),
        "images": sum(r.get("images", 0) for r in window),
        "alloc": sum(r.get("alloc_mb", 0) for r in window),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--before", type=float, default=1.0)
    ap.add_argument("--after", type=float, default=12.0)
    ap.add_argument("--target", type=float, default=33.3)
    ap.add_argument("--min-images", type=float, default=150)
    ap.add_argument("--min-alloc-mb", type=float, default=400)
    args = ap.parse_args()
    for path in args.csv:
        rows = [r for r in load_frames(path) if r.get("draws", 0) > 0 or r.get("frame_ms", 0) > 0]
        # the title and the first world frames are not a load of a played area
        starts = find_loads(rows, args.gap, args.min_images, args.min_alloc_mb)
        results = [j for t in starts if (j := judge(rows, t, args.before, args.after, args.target))]
        print(f"== {path}: {len(rows)} frames, {len(results)} loads")
        print("  at s   worst ms  >60  >100  lost ms   images  alloc MB")
        for j in results:
            print(f"  {j['t']:6.0f}  {j['worst']:8.0f}  {j['over60']:3d}  {j['over100']:4d}  "
                  f"{j['lost']:7.0f}  {j['images']:7.0f}  {j['alloc']:8.0f}")
        if results:
            lost = [j["lost"] for j in results]
            worst = [j["worst"] for j in results]
            print(f"  per load: lost {statistics.mean(lost):.0f} ms on average "
                  f"(median {statistics.median(lost):.0f}, max {max(lost):.0f}); "
                  f"worst frame {statistics.mean(worst):.0f} ms on average, {max(worst):.0f} at most; "
                  f">60 ms frames {statistics.mean(j['over60'] for j in results):.1f}, "
                  f">100 ms {statistics.mean(j['over100'] for j in results):.1f}")
        calm = [r["frame_ms"] for r in rows
                if not any(t - args.before <= r["t_s"] <= t + args.after for t in starts)]
        if calm:
            calm.sort()
            print(f"  outside the loads: median {calm[len(calm) // 2]:.1f} ms, "
                  f"p99 {calm[int(len(calm) * 0.99)]:.1f} ms, "
                  f"over 40 ms {100.0 * sum(1 for f in calm if f > 40) / len(calm):.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
