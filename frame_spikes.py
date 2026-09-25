#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
"""Explains frame-time spikes from a shadPS4 telemetry dump.

Usage: python frame_spikes.py [telemetry.csv|telemetry.csv.zst] [--threshold-ms 17.5]

Reads the per-frame counters of a DetailedTelemetry dump, splits frames into normal and slow
(interval above the threshold) and ranks every counter by how much more it costs in slow frames.
Writes frame_spikes.tsv with one row per frame for plotting.
"""
import argparse
import glob
import io
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:
    zstd = None

# Counters grouped by where the time goes. Time counters end in _ns and are shown in ms.
GROUPS = [
    ("GCP totals", ["gcp_active_ns", "gcp_blocked_ns", "draw_cpu_ns", "dispatch_cpu_ns",
                    "gcp_other_ns", "draws", "dispatches", "pm4_packets"]),
    ("Draw phases", ["draw_phase_pending_ops_ns", "draw_phase_filter_ns", "draw_phase_pipeline_ns",
                     "draw_phase_render_state_ns", "draw_phase_bind_ns", "bind_buffers_ns",
                     "bind_textures_ns", "draw_phase_vertex_index_ns",
                     "draw_phase_begin_rendering_ns", "draw_phase_stream_copy_ns",
                     "draw_phase_finalize_ns", "draw_phase_descriptors_ns",
                     "draw_phase_dynamic_state_ns", "draw_phase_cmd_ns",
                     "draw_phase_mark_writes_ns"]),
    ("Dispatch phases", ["dispatch_phase_pending_ops_ns", "dispatch_phase_pipeline_ns",
                         "dispatch_phase_hle_ns", "dispatch_phase_bind_ns",
                         "dispatch_phase_record_ns"]),
    ("GCP stalls", ["guest_copy_producer_wait_ns", "gcp_sync_packet_ns", "stream_buffer_wait_ns",
                    "wait_ns", "wait_calls", "wait_reg_mem_spin_ns", "wait_reg_mem_calls",
                    "submit_prepare_ns"]),
    ("Guest copies", ["guest_copy_bytes", "guest_copy_jobs", "guest_copy_worker_ns",
                      "guest_copy_help_ns", "guest_copy_wait_ns", "guest_copy_overlap_waits",
                      "guest_copy_inline_bytes", "guest_copy_protected_inline_ops",
                      "guest_copy_queue_depth_max", "guest_copy_gpu_served_ops",
                      "guest_copy_gpu_served_bytes", "guest_copy_backing_bytes"]),
    ("GPU authority", ["authority_materializations", "authority_materialize_ns",
                       "authority_retirements", "authority_retired_bytes", "authority_live_max",
                       "pending_op_poll_skips"]),
    ("Recording thread", ["vk_record_commands", "vk_record_chunks", "vk_record_worker_ns",
                          "vk_record_producer_wait_ns", "vk_record_queue_depth_max"]),
    ("Draw caches", ["image_token_hits", "image_token_misses", "image_lookup_hits",
                     "stage_cache_current_hits", "stage_shape_hits", "stage_cache_search_hits",
                     "stage_cache_misses"]),
    ("Resource churn", ["texture_uploads", "texture_upload_bytes", "texture_upload_ns",
                        "texture_hash_bytes", "texture_hash_ns", "buffer_creates",
                        "buffer_create_ns", "staging_bytes", "pipeline_misses",
                        "shader_module_compile_jobs", "shader_module_pending_draws",
                        "writeback_calls", "writeback_bytes", "priority_ops_execute_ns"]),
    ("Submission / GPU", ["driver_submit_calls", "driver_submit_ns", "gpu_idle_gap_ns",
                          "present_cpu_ns"]),
]


def open_telemetry(path: Path):
    if path.suffix == ".zst":
        if zstd is None:
            sys.exit("zstandard module required for .zst files: pip install zstandard")
        reader = zstd.ZstdDecompressor().stream_reader(path.open("rb"))
        return io.TextIOWrapper(reader, encoding="utf-8")
    return path.open("r", encoding="utf-8")


def find_latest() -> Path | None:
    candidates = []
    for pattern in ["*.csv", "*.csv.zst", "user/log/*telemetry*.csv*",
                    "../../GAMES/EMULATION/EMULADORES/shadps4/user/log/*telemetry*.csv*"]:
        candidates += [Path(p) for p in glob.glob(pattern)]
    candidates = [c for c in candidates if "telemetry" in c.name]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def load_frames(path: Path):
    frames = {}
    counters = defaultdict(dict)
    with open_telemetry(path) as f:
        for line in f:
            if line.startswith("frame_counter,"):
                parts = line.rstrip("\n").split(",")
                if len(parts) >= 7:
                    counters[parts[1]][parts[3]] = int(parts[6])
            elif line.startswith("frame,"):
                parts = line.rstrip("\n").split(",")
                if len(parts) >= 6 and parts[3] == "timing":
                    frames[parts[1]] = int(parts[5])
    rows = []
    for seq, interval in frames.items():
        row = dict(counters.get(seq, {}))
        row["frame_seq"] = int(seq)
        row["interval_ns"] = interval
        row["gcp_other_ns"] = max(0, row.get("gcp_active_ns", 0) - row.get("draw_cpu_ns", 0) -
                                  row.get("dispatch_cpu_ns", 0))
        rows.append(row)
    rows.sort(key=lambda r: r["frame_seq"])
    # The first frame accumulates startup.
    return rows[1:]


def fmt(name: str, value: float) -> str:
    if name.endswith("_ns"):
        return f"{value / 1e6:9.3f} ms"
    if name.endswith("_bytes"):
        return f"{value / (1024 * 1024):9.2f} MiB"
    return f"{value:12.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?")
    parser.add_argument("--threshold-ms", type=float, default=17.5)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    path = Path(args.file) if args.file else find_latest()
    if path is None or not path.exists():
        sys.exit("telemetry file not found; pass it as the first argument")
    rows = load_frames(path)
    if not rows:
        sys.exit("no frame records (was this a DetailedTelemetry build with logging enabled?)")

    threshold = args.threshold_ms * 1e6
    slow = [r for r in rows if r["interval_ns"] > threshold]
    normal = [r for r in rows if r["interval_ns"] <= threshold]
    intervals = sorted(r["interval_ns"] for r in rows)

    def pct(p):
        return intervals[min(len(intervals) - 1, int(p * len(intervals)))] / 1e6

    print(f"File: {path}")
    print(f"Frames: {len(rows)}  slow (> {args.threshold_ms} ms): {len(slow)} "
          f"({100 * len(slow) / len(rows):.1f}%)")
    print(f"Frame time ms: median {statistics.median(intervals) / 1e6:.2f}  p90 {pct(0.90):.2f}  "
          f"p99 {pct(0.99):.2f}  max {intervals[-1] / 1e6:.2f}")

    def mean(frames, key):
        return sum(f.get(key, 0) for f in frames) / len(frames) if frames else 0.0

    for title, keys in GROUPS:
        print(f"\n== {title}: mean per frame (normal | slow | delta)")
        for key in keys:
            if not any(key in r for r in rows) and key != "gcp_other_ns":
                continue
            n, s = mean(normal, key), mean(slow, key)
            print(f"  {key:34s} {fmt(key, n)} | {fmt(key, s)} | {fmt(key, s - n)}")

    # Rank every time counter by its growth in slow frames.
    time_keys = sorted({k for r in rows for k in r if k.endswith("_ns") and k != "interval_ns"})
    if slow and normal:
        growth = sorted(((mean(slow, k) - mean(normal, k), k) for k in time_keys), reverse=True)
        print(f"\n== Time counters that grow most in slow frames")
        for delta, key in growth[:args.top]:
            print(f"  {key:34s} +{delta / 1e6:8.3f} ms")

    # Per-frame breakdown of the worst frames against the median frame.
    median_row = {k: statistics.median([r.get(k, 0) for r in rows]) for k in time_keys}
    print(f"\n== Worst frames: biggest excess over the median frame")
    for row in sorted(rows, key=lambda r: r["interval_ns"], reverse=True)[:10]:
        excess = sorted(((row.get(k, 0) - median_row[k], k) for k in time_keys), reverse=True)
        top = ", ".join(f"{k}+{d / 1e6:.2f}" for d, k in excess[:5] if d > 0)
        print(f"  frame {row['frame_seq']:6d} {row['interval_ns'] / 1e6:7.2f} ms  "
              f"draws {row.get('draws', 0):5d}  {top}")

    columns = ["frame_seq", "interval_ns"] + [k for _, keys in GROUPS for k in keys]
    out = Path("frame_spikes.tsv")
    with out.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, 0)) for c in columns) + "\n")
    print(f"\nPer-frame table written to {out.resolve()}")


if __name__ == "__main__":
    main()
