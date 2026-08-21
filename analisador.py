#!/usr/bin/env python3
import csv
import glob
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:
    zstd = None


def open_telemetry_file(path: Path):
    if path.suffix == ".zst":
        if zstd is None:
            raise RuntimeError("zstandard module required to read .zst files. Run `pip install zstandard`")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(path.open("rb"))
        import io
        return io.TextIOWrapper(stream_reader, encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def find_latest_telemetry_file() -> Path:
    candidates = []
    for pattern in ["user/log/telemetry_*.csv", "user/log/telemetry_*.csv.zst", "*.csv", "*.csv.zst"]:
        for f in glob.glob(pattern):
            candidates.append(Path(f))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    else:
        latest = find_latest_telemetry_file()
        if latest is None:
            print("Usage: python analisador.py <telemetry.csv[.zst]> [output_dir]")
            sys.exit(1)
        input_path = latest
        print(f"Auto-selected latest telemetry file: {input_path}")

    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.parent

    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    print(f"Loading telemetry from {input_path}...")

    counters = {}
    candidates = []
    authorities = []
    vfences = []
    vwaits = []
    label_sigs = []
    gpu_consumes = []
    barriers = []
    ram_demands = []
    mat_begins = []
    mat_ends = []
    ram_consumes = []
    cpu_reads = []
    supersedes = []
    fallbacks = []
    conservative_decisions = []
    readback_suppressed = []
    materialize_required = []
    wait_decisions = []
    forced_completions = []
    cpu_to_gpu_waits = []
    frames = []

    candidate_headers = ["kind", "candidate_seq", "timestamp_ns", "eligibility", "reject_reason", "producer_seq", "producer_packet_seq", "producer_kind", "resource_id", "resource_version", "image_id", "image_uid", "guest_addr", "size", "fence_seq", "eos_packet_seq", "label_addr", "label_value", "label_num_bytes", "wait_packet_seq", "wait_compare", "wait_ref", "wait_mask", "acquire_packet_seq", "acquire_raw_cntl"]
    authority_headers = ["kind", "authority_seq", "timestamp_ns", "candidate_seq", "resource_id", "resource_version", "image_id", "image_uid", "guest_begin", "guest_end", "size", "producer_seq", "producer_packet_seq", "producer_tick", "cmd_buffer_seq", "submit_seq", "fence_seq", "virtual_fence_seq", "label_addr", "label_generation"]
    vfence_headers = ["kind", "virtual_fence_seq", "timestamp_ns", "authority_seq", "fence_seq", "label_addr", "label_generation", "expected_value", "producer_tick", "producer_packet_seq", "eos_packet_seq", "wait_packet_seq", "acquire_packet_seq"]
    vwait_headers = ["kind", "virtual_fence_seq", "timestamp_ns", "result", "authority_seq", "wait_seq", "wait_packet_seq", "label_addr", "label_generation", "producer_tick", "producer_tick_complete_at_consume"]
    label_sig_headers = ["kind", "virtual_fence_seq", "timestamp_ns", "action", "authority_seq", "label_addr", "scheduled_generation", "current_generation", "producer_tick", "current_completed_tick"]
    gpu_consume_headers = ["kind", "authority_seq", "timestamp_ns", "resource_id", "resource_version", "image_id", "image_uid", "consumer_seq", "consumer_packet_seq", "consumer_kind", "requested_access", "requested_layout", "producer_tick", "consumer_cmd_buffer_seq", "consumer_submit_seq"]
    barrier_headers = ["kind", "authority_seq", "timestamp_ns", "consumer_seq", "resource_id", "resource_version", "old_layout", "new_layout", "src_stage", "src_access", "dst_stage", "dst_access", "subresource_range", "valid_write_dependency"]
    ram_demand_headers = ["kind", "ram_demand_seq", "timestamp_ns", "path", "ram_demand_group_seq", "authority_seq", "resource_id", "resource_version", "authority_begin", "authority_end", "request_addr", "request_size", "overlap_begin", "overlap_size", "consumer_seq", "consumer_producer_seq", "consumer_packet_seq", "destination_kind", "destination_resource_id", "authority_state_before"]
    mat_begin_headers = ["kind", "materialize_seq", "timestamp_ns", "ram_demand_seq", "authority_seq", "resource_id", "resource_version", "image_id", "image_uid", "producer_tick", "current_completed_tick", "guest_begin", "guest_end", "reason"]
    mat_end_headers = ["kind", "materialize_seq", "timestamp_ns", "result", "ram_demand_seq", "authority_seq", "producer_tick", "completed_tick", "bytes_materialized", "guest_begin", "guest_end", "host_current_after", "validation_bytes_equal"]
    ram_consume_headers = ["kind", "ram_demand_seq", "timestamp_ns", "path", "authority_seq", "materialize_seq", "consumer_seq", "request_addr", "request_size", "overlap_begin", "overlap_size", "host_current", "producer_tick_complete", "materialize_success"]
    cpu_read_headers = ["kind", "authority_seq", "timestamp_ns", "origin", "fault_addr", "guest_read_begin", "guest_read_size", "materialize_seq", "resumed_after_materialize"]
    supersede_headers = ["kind", "old_authority_seq", "new_authority_seq", "timestamp_ns", "overlap_begin", "overlap_size", "old_resource_id", "old_resource_version", "new_resource_id", "new_resource_version", "old_host_current"]
    fallback_headers = ["kind", "candidate_seq", "timestamp_ns", "phase", "reason", "authority_seq"]
    conservative_headers = ["kind", "timestamp_ns", "trigger", "fence_seq", "image_id", "image_uid", "resource_id", "resource_version", "guest_addr", "size", "authority_seq", "authority_state", "decision", "reason", "readback_schedule_seen", "readback_seq"]
    suppressed_headers = ["kind", "authority_seq", "resource_id", "resource_version", "trigger", "fence_seq", "guest_addr", "size", "image_id", "image_uid"]
    mat_req_headers = ["kind", "authority_seq", "reason", "request_addr", "request_size", "overlap_addr", "overlap_size"]
    wait_dec_headers = ["kind", "candidate_seq", "virtual_fence_seq", "authority_seq", "wait_seq", "wait_packet_seq", "label_addr", "label_generation", "ref", "mask", "compare", "producer_tick", "current_tick", "decision", "reason"]
    forced_comp_headers = ["kind", "virtual_fence_seq", "authority_seq", "reason", "producer_tick", "was_submitted", "waited", "duration_ns"]
    cpu_label_wait_headers = ["kind", "wait_seq", "frame_id", "label_addr", "ref", "mask", "wait_begin", "wait_end", "duration_ns", "last_guest_write_ts", "last_guest_write_value", "writer_thread_id", "delta_write_to_wait_complete_ns", "yield_count", "wait_progress_submit_count", "gpu_idle_overlap_ns"]

    def make_dict(headers, parts):
        return {headers[i]: parts[i] if i < len(parts) else "" for i in range(min(len(headers), len(parts)))}

    with open_telemetry_file(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            kind = parts[0]
            if kind == "counter":
                if len(parts) >= 4:
                    counters[parts[2]] = int(parts[3])
            elif kind == "fastpath_candidate":
                candidates.append(make_dict(candidate_headers, parts))
            elif kind == "gpu_authority_create":
                authorities.append(make_dict(authority_headers, parts))
            elif kind == "virtual_fence_create":
                vfences.append(make_dict(vfence_headers, parts))
            elif kind == "virtual_wait_consume":
                vwaits.append(make_dict(vwait_headers, parts))
            elif kind == "async_label_signal":
                label_sigs.append(make_dict(label_sig_headers, parts))
            elif kind == "authority_gpu_consume":
                gpu_consumes.append(make_dict(gpu_consume_headers, parts))
            elif kind == "authority_barrier_validation":
                barriers.append(make_dict(barrier_headers, parts))
            elif kind == "authority_ram_demand":
                ram_demands.append(make_dict(ram_demand_headers, parts))
            elif kind == "lazy_materialize_begin":
                mat_begins.append(make_dict(mat_begin_headers, parts))
            elif kind == "lazy_materialize_end":
                mat_ends.append(make_dict(mat_end_headers, parts))
            elif kind == "authority_ram_consume":
                ram_consumes.append(make_dict(ram_consume_headers, parts))
            elif kind == "authority_cpu_read":
                cpu_reads.append(make_dict(cpu_read_headers, parts))
            elif kind == "authority_supersede":
                supersedes.append(make_dict(supersede_headers, parts))
            elif kind == "fastpath_fallback":
                fallbacks.append(make_dict(fallback_headers, parts))
            elif kind == "conservative_download_decision":
                conservative_decisions.append(make_dict(conservative_headers, parts))
            elif kind == "authority_conservative_readback_suppressed":
                readback_suppressed.append(make_dict(suppressed_headers, parts))
            elif kind == "authority_host_materialize_required":
                materialize_required.append(make_dict(mat_req_headers, parts))
            elif kind == "fastpath_wait_decision":
                wait_decisions.append(make_dict(wait_dec_headers, parts))
            elif kind == "virtual_fence_forced_completion":
                forced_completions.append(make_dict(forced_comp_headers, parts))
            elif kind == "cpu_to_gpu_label_wait":
                cpu_to_gpu_waits.append(make_dict(cpu_label_wait_headers, parts))
            elif kind == "frame":
                frames.append({"frame_seq": parts[1] if len(parts) > 1 else "", "raw": line})

    print(f"Parsed records:")
    print(f"  Candidates:                {len(candidates)}")
    print(f"  Authorities:               {len(authorities)}")
    print(f"  Virtual Fences:            {len(vfences)}")
    print(f"  Virtual Waits Consumed:    {len(vwaits)}")
    print(f"  Conservative Decisions:    {len(conservative_decisions)}")
    print(f"  Readbacks Suppressed:      {len(readback_suppressed)}")
    print(f"  Wait Decisions:            {len(wait_decisions)}")
    print(f"  Forced Completions:        {len(forced_completions)}")
    print(f"  CPU->GPU Label Waits:      {len(cpu_to_gpu_waits)}")
    print(f"  GPU Consumes:              {len(gpu_consumes)}")
    print(f"  RAM Demands:               {len(ram_demands)}")
    print(f"  Materializations:          {len(mat_ends)}")
    print(f"  Fallbacks:                 {len(fallbacks)}")
    print(f"  Frames:                    {len(frames)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    def write_tsv(filename: str, rows: list):
        if not rows:
            return
        out_file = output_dir / filename
        fieldnames = list(rows[0].keys())
        with out_file.open("w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_file}")

    write_tsv("fastpath_candidates.tsv", candidates)
    write_tsv("fastpath_authorities.tsv", authorities)
    write_tsv("fastpath_conservative_decisions.tsv", conservative_decisions)
    write_tsv("fastpath_readbacks_suppressed.tsv", readback_suppressed)
    write_tsv("fastpath_wait_decisions.tsv", wait_decisions)
    write_tsv("fastpath_forced_completions.tsv", forced_completions)
    write_tsv("fastpath_cpu_to_gpu_waits.tsv", cpu_to_gpu_waits)
    write_tsv("fastpath_barriers.tsv", barriers)
    write_tsv("fastpath_ram_demands.tsv", ram_demands)
    write_tsv("fastpath_materializations.tsv", mat_ends)
    write_tsv("fastpath_fallbacks.tsv", fallbacks)
    write_tsv("fastpath_frames.tsv", frames)

    # Fastpath Summary
    print("\n" + "=" * 76)
    print("        GOW3 FASTPATH CONSERVATIVE DRAIN & VIRTUAL WAIT REPORT")
    print("=" * 76)

    total_candidates = counters.get("fastpath_candidates", len(candidates))
    fastpath_taken = counters.get("fastpath_taken", len(authorities))
    vwaits_consumed = counters.get("virtual_wait_consumed", len(vwaits))
    gpu_consumed = counters.get("authority_gpu_first_consumer", len(gpu_consumes))
    ram_demanded = counters.get("ram_demand_events", len(ram_demands))
    suppressed_cnt = counters.get("conservative_download_suppressed", len(readback_suppressed))
    considered_cnt = counters.get("conservative_download_considered", len(conservative_decisions))
    forced_cnt = counters.get("virtual_fence_forced_completions", len(forced_completions))
    progress_submits = counters.get("fastpath_wait_progress_submits", 0)

    print(f"Candidates Examined:                 {total_candidates}")
    print(f"Fastpath Authorities Created:        {fastpath_taken}")
    print(f"Conservative Downloads Considered:   {considered_cnt}")
    print(f"Conservative Downloads Suppressed:   {suppressed_cnt}")
    print(f"Virtual Waits Consumed:              {vwaits_consumed}")
    print(f"Fastpath Wait Progress Submits:      {progress_submits}")
    print(f"Virtual Fence Forced Completions:    {forced_cnt}")
    print(f"GPU Consumes (In-VRAM):              {gpu_consumed}")
    print(f"RAM Demands Triggered:               {ram_demanded}")

    if suppressed_cnt > 0:
        pct = (suppressed_cnt / max(1, considered_cnt)) * 100.0
        print(f"\n[DRAIN SUPPRESSION]: SUCCESS -> {suppressed_cnt}/{considered_cnt} ({pct:.1f}%) readbacks suppressed!")
    
    virtualized_waits = sum(1 for w in wait_decisions if w.get("decision") == "virtualized")
    legacy_waits = sum(1 for w in wait_decisions if w.get("decision") == "legacy")
    print(f"\n[WAIT VIRTUALIZATION]: Virtualized={virtualized_waits}, Legacy={legacy_waits}")

    if cpu_to_gpu_waits:
        durations = [float(w["duration_ns"]) / 1e6 for w in cpu_to_gpu_waits if w.get("duration_ns")]
        if durations:
            avg_dur = sum(durations) / len(durations)
            max_dur = max(durations)
            print(f"\n[CPU->GPU LABEL WAITS (e.g. 0x2B0000028)]: Count={len(cpu_to_gpu_waits)}, Avg={avg_dur:.2f}ms, Max={max_dur:.2f}ms")

    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
