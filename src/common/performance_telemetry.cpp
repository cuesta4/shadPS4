// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "common/path_util.h"
#include "common/performance_telemetry.h"
#include "common/thread.h"

namespace Common::PerformanceTelemetry {
namespace {

constexpr u64 RingCapacity = 128_KB;
constexpr u64 RingMask = RingCapacity - 1;
static_assert(std::has_single_bit(RingCapacity));

constexpr size_t HistogramSubdivisions = 8;
constexpr size_t HistogramExactValues = 8;
constexpr size_t HistogramBucketCount =
    HistogramExactValues + (std::numeric_limits<u64>::digits - 3) * HistogramSubdivisions;

constexpr std::array CounterNames{
    "pm4_packets",          "pm4_type2_packets",  "dcb_bytes",
    "ccb_bytes",            "acb_bytes",          "gfx_submits",
    "asc_submits",          "gcp_wakes",          "queue_scans",
    "queue_resumes",        "gcp_active_ns",      "gcp_blocked_ns",
    "queue_ready_ns",       "queue_resume_ns",    "ib_depth_max",
    "draws",
    "dispatches",           "draw_cpu_ns",        "dispatch_cpu_ns",
    "pipeline_hits",        "pipeline_misses",    "pipeline_compile_ns",
    "render_target_hits",   "render_target_misses", "image_token_hits",
    "image_token_misses",   "descriptor_hits",    "descriptor_misses",
    "buffer_token_hits",    "buffer_token_misses", "stream_slice_hits",
    "stream_slice_misses",  "staging_bytes",      "barrier_calls",
    "copy_calls",           "copy_bytes",         "timeline_polls",
    "timeline_poll_ns",     "driver_submit_calls", "driver_submit_ns",
    "driver_present_calls", "driver_present_ns",  "submit_queue_depth_max",
    "wait_calls",           "wait_ns",            "writeback_calls",
    "writeback_bytes",      "writeback_ns",       "present_prepare_ns",
    "present_cpu_ns",       "gpu_idle_gaps",       "gpu_idle_gap_ns",
};
static_assert(CounterNames.size() == static_cast<size_t>(Counter::Count));

constexpr std::array EventNames{
    "none",              "gcp_active",      "gcp_blocked",    "submit_done",
    "timeline_complete", "driver_submit",   "wait",           "writeback",
    "present_prepare",   "present_cpu",     "driver_present", "frame_presented",
    "gpu_idle_gap",      "pm4_packet",
};
static_assert(EventNames.size() == static_cast<size_t>(EventType::Pm4Packet) + 1);

constexpr std::array HistogramCounters{
    Counter::GcpActiveNs,      Counter::GcpBlockedNs,    Counter::QueueReadyNs,
    Counter::QueueResumeNs,    Counter::DrawCpuNs,       Counter::DispatchCpuNs,
    Counter::PipelineCompileNs, Counter::TimelinePollNs, Counter::DriverSubmitNs,
    Counter::DriverPresentNs,  Counter::WaitNs,          Counter::WritebackNs,
    Counter::PresentPrepareNs, Counter::PresentCpuNs,    Counter::GpuIdleGapNs,
    Counter::SubmitQueueDepthMax,
};

constexpr std::array HistogramNames{
    "gcp_active_ns",       "gcp_blocked_ns",    "queue_ready_ns",
    "queue_resume_ns",     "draw_cpu_ns",       "dispatch_cpu_ns",
    "pipeline_compile_ns", "timeline_poll_ns",  "driver_submit_ns",
    "driver_present_ns",   "wait_ns",           "writeback_ns",
    "present_prepare_ns",  "present_cpu_ns",    "gpu_idle_gap_ns",
    "submit_queue_depth",
};
static_assert(HistogramCounters.size() == HistogramNames.size());

constexpr u8 NoHistogram = std::numeric_limits<u8>::max();
constexpr auto HistogramIndices = [] {
    std::array<u8, static_cast<size_t>(Counter::Count)> indices{};
    indices.fill(NoHistogram);
    for (u8 i = 0; i < HistogramCounters.size(); ++i) {
        indices[static_cast<size_t>(HistogramCounters[i])] = i;
    }
    return indices;
}();

struct EventSlot {
    std::atomic<u64> timestamp_ns{};
    std::atomic<u64> arg0{};
    std::atomic<u64> arg1{};
    std::atomic<u64> metadata{};
    std::atomic<u64> committed_sequence{};
};

struct ThreadRing {
    explicit ThreadRing(u32 id_, std::string name_) : id{id_}, name{std::move(name_)} {}

    alignas(64) std::atomic<u64> next_sequence{};
    std::array<EventSlot, RingCapacity> events{};
    alignas(64) std::array<std::atomic<u64>, static_cast<size_t>(Counter::Count)> counters{};
    std::array<std::atomic<u64>, 256> opcodes{};
    std::array<std::array<std::atomic<u64>, HistogramBucketCount>, HistogramCounters.size()>
        histograms{};
    u32 id;
    std::string name;
};

struct EventSnapshot {
    u64 timestamp_ns;
    u64 arg0;
    u64 arg1;
    u64 sequence;
    u32 thread_id;
    EventType type;
};

std::mutex g_rings_mutex;
std::vector<std::unique_ptr<ThreadRing>> g_rings;
std::atomic_bool g_dumping{};
thread_local ThreadRing* g_thread_ring{};
const u64 g_session_start_ns = Timestamp();

[[nodiscard]] ThreadRing* GetThreadRing() {
    if (g_thread_ring != nullptr) {
        return g_thread_ring;
    }
    if (g_dumping.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    std::scoped_lock lock{g_rings_mutex};
    if (g_dumping.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    const u32 id = static_cast<u32>(g_rings.size());
    std::string name{Common::GetCurrentThreadNameView()};
    if (name.empty()) {
        name = "unnamed";
    }
    g_thread_ring =
        g_rings.emplace_back(std::make_unique<ThreadRing>(id, std::move(name))).get();
    return g_thread_ring;
}

void AddSingleWriter(std::atomic<u64>& counter, u64 value) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + value, std::memory_order_relaxed);
}

void ObserveSingleWriterMax(std::atomic<u64>& counter, u64 value) noexcept {
    if (const u64 current = counter.load(std::memory_order_relaxed); current < value) {
        counter.store(value, std::memory_order_relaxed);
    }
}

[[nodiscard]] size_t HistogramBucket(u64 value) noexcept {
    if (value < HistogramExactValues) {
        return static_cast<size_t>(value);
    }
    const u32 exponent = std::bit_width(value) - 1;
    const u64 base = u64{1} << exponent;
    const u32 shift = exponent - 3;
    const u64 subdivision = (value - base) >> shift;
    return HistogramExactValues + (exponent - 3) * HistogramSubdivisions + subdivision;
}

[[nodiscard]] std::pair<u64, u64> HistogramBounds(size_t bucket) noexcept {
    if (bucket < HistogramExactValues) {
        return {bucket, bucket};
    }
    const size_t scaled = bucket - HistogramExactValues;
    const u32 exponent = 3 + static_cast<u32>(scaled / HistogramSubdivisions);
    const u64 subdivision = scaled % HistogramSubdivisions;
    const u64 width = u64{1} << (exponent - 3);
    const u64 lower = (u64{1} << exponent) + subdivision * width;
    return {lower, lower + width - 1};
}

void ObserveHistogramSingleWriter(ThreadRing& ring, Counter counter, u64 value) noexcept {
    const u8 histogram = HistogramIndices[static_cast<size_t>(counter)];
    if (histogram != NoHistogram) {
        AddSingleWriter(ring.histograms[histogram][HistogramBucket(value)], 1);
    }
}

void AddDurationSingleWriter(ThreadRing& ring, Counter counter, u64 duration) noexcept {
    AddSingleWriter(ring.counters[static_cast<size_t>(counter)], duration);
    ObserveHistogramSingleWriter(ring, counter, duration);
}

void WriteEvent(ThreadRing& ring, EventType type, u64 arg0, u64 arg1) noexcept {
    const u64 sequence = ring.next_sequence.load(std::memory_order_relaxed);
    ring.next_sequence.store(sequence + 1, std::memory_order_release);
    auto& slot = ring.events[sequence & RingMask];
    slot.timestamp_ns.store(Timestamp(), std::memory_order_relaxed);
    slot.arg0.store(arg0, std::memory_order_relaxed);
    slot.arg1.store(arg1, std::memory_order_relaxed);
    slot.metadata.store(static_cast<u64>(type) | (static_cast<u64>(ring.id) << 16),
                        std::memory_order_relaxed);
    slot.committed_sequence.store(sequence + 1, std::memory_order_release);
}

[[nodiscard]] bool IsMaxCounter(Counter counter) noexcept {
    return counter == Counter::IbDepthMax || counter == Counter::SubmitQueueDepthMax;
}

[[nodiscard]] std::string CsvSafe(std::string value) {
    std::ranges::replace(value, ',', '_');
    std::ranges::replace(value, '\n', '_');
    std::ranges::replace(value, '\r', '_');
    return value;
}

} // namespace

void AddEnabled(Counter counter, u64 value) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(ring->counters[static_cast<size_t>(counter)], value);
    }
}

void ObserveMaxEnabled(Counter counter, u64 value) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        ObserveSingleWriterMax(ring->counters[static_cast<size_t>(counter)], value);
        ObserveHistogramSingleWriter(*ring, counter, value);
    }
}

void RecordEnabled(EventType type, u64 arg0, u64 arg1) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        WriteEvent(*ring, type, arg0, arg1);
    }
}

void RecordDurationEnabled(Counter counter, EventType type, u64 start_ns, u64 arg0) noexcept {
    if (!Enabled() || g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        const u64 duration = Timestamp() - start_ns;
        AddDurationSingleWriter(*ring, counter, duration);
        if (type != EventType::None) {
            WriteEvent(*ring, type, arg0, duration);
        }
    }
}

void RecordDurationValueEnabled(Counter counter, EventType type, u64 duration, u64 arg0) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddDurationSingleWriter(*ring, counter, duration);
        if (type != EventType::None) {
            WriteEvent(*ring, type, arg0, duration);
        }
    }
}

void CountPm4PacketEnabled(Pm4Engine engine, u32 queue_id, u32 opcode, u32 depth,
                           uintptr_t address, u32 words) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(ring->counters[static_cast<size_t>(Counter::Pm4Packets)], 1);
        AddSingleWriter(ring->opcodes[opcode & 0xff], 1);
        ObserveSingleWriterMax(ring->counters[static_cast<size_t>(Counter::IbDepthMax)], depth + 1);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
        const u64 packed = static_cast<u64>(opcode & 0xff) |
                           (static_cast<u64>(engine) << 8) |
                           (static_cast<u64>(queue_id & 0xff) << 16) |
                           (static_cast<u64>(depth & 0xff) << 24) |
                           (static_cast<u64>(words) << 32);
        WriteEvent(*ring, EventType::Pm4Packet, packed, address);
#else
        static_cast<void>(engine);
        static_cast<void>(queue_id);
        static_cast<void>(address);
        static_cast<void>(words);
#endif
    }
}

std::filesystem::path Dump() {
    if (!Enabled() || g_dumping.exchange(true, std::memory_order_acq_rel)) {
        return {};
    }

    std::vector<ThreadRing*> snapshot_rings;
    {
        std::scoped_lock lock{g_rings_mutex};
        snapshot_rings.reserve(g_rings.size());
        for (const auto& ring : g_rings) {
            snapshot_rings.push_back(ring.get());
        }
    }
    if (snapshot_rings.empty()) {
        return {};
    }

    std::array<u64, static_cast<size_t>(Counter::Count)> totals{};
    std::array<u64, 256> opcode_totals{};
    std::array<std::array<u64, HistogramBucketCount>, HistogramCounters.size()>
        histogram_totals{};
    std::vector<EventSnapshot> events;
    events.reserve(snapshot_rings.size() * RingCapacity);
    u64 total_written{};
    u64 total_overwritten{};

    for (const auto* ring : snapshot_rings) {
        for (size_t i = 0; i < totals.size(); ++i) {
            const auto counter = static_cast<Counter>(i);
            const u64 value = ring->counters[i].load(std::memory_order_relaxed);
            if (IsMaxCounter(counter)) {
                totals[i] = std::max(totals[i], value);
            } else {
                totals[i] += value;
            }
        }
        for (size_t i = 0; i < opcode_totals.size(); ++i) {
            opcode_totals[i] += ring->opcodes[i].load(std::memory_order_relaxed);
        }
        for (size_t histogram = 0; histogram < histogram_totals.size(); ++histogram) {
            for (size_t bucket = 0; bucket < HistogramBucketCount; ++bucket) {
                histogram_totals[histogram][bucket] +=
                    ring->histograms[histogram][bucket].load(std::memory_order_relaxed);
            }
        }

        const u64 end = ring->next_sequence.load(std::memory_order_acquire);
        const u64 begin = end > RingCapacity ? end - RingCapacity : 0;
        total_written += end;
        total_overwritten += begin;
        for (u64 sequence = begin; sequence < end; ++sequence) {
            const auto& slot = ring->events[sequence & RingMask];
            const u64 expected = sequence + 1;
            if (slot.committed_sequence.load(std::memory_order_acquire) != expected) {
                continue;
            }
            EventSnapshot event{
                .timestamp_ns = slot.timestamp_ns.load(std::memory_order_relaxed),
                .arg0 = slot.arg0.load(std::memory_order_relaxed),
                .arg1 = slot.arg1.load(std::memory_order_relaxed),
                .sequence = sequence,
                .thread_id = ring->id,
                .type = static_cast<EventType>(slot.metadata.load(std::memory_order_relaxed) &
                                               0xffff),
            };
            if (slot.committed_sequence.load(std::memory_order_acquire) == expected) {
                events.push_back(event);
            }
        }
    }

    std::ranges::sort(events, {}, &EventSnapshot::timestamp_ns);

    const auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) /
                      ("shadps4-telemetry-" + std::to_string(timestamp_ms) + ".csv");
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    if (!file) {
        return {};
    }

    file << "kind,thread,timestamp_ns,name,arg0,arg1,value\n";
    file << "metadata,,0,schema_version,0,0,2\n";
    file << "metadata,,0,session_duration_ns,0,0," << Timestamp() - g_session_start_ns << '\n';
    file << "metadata,,0,ring_capacity,0,0," << RingCapacity << '\n';
    file << "metadata,,0,thread_count,0,0," << snapshot_rings.size() << '\n';
    file << "metadata,,0,event_count,0,0," << events.size() << '\n';
    file << "metadata,,0,event_written,0,0," << total_written << '\n';
    file << "metadata,,0,event_overwritten,0,0," << total_overwritten << '\n';
    file << "metadata,,0,histogram_subdivisions,0,0," << HistogramSubdivisions << '\n';
    for (const auto* ring : snapshot_rings) {
        const u64 written = ring->next_sequence.load(std::memory_order_relaxed);
        const u64 overwritten = written > RingCapacity ? written - RingCapacity : 0;
        file << "thread," << ring->id << ",0," << CsvSafe(ring->name) << "," << written << ','
             << overwritten << ",0\n";
        for (size_t i = 0; i < totals.size(); ++i) {
            const u64 value = ring->counters[i].load(std::memory_order_relaxed);
            if (value != 0) {
                file << "thread_counter," << ring->id << ",0," << CounterNames[i]
                     << ",0,0," << value << '\n';
            }
        }
        for (size_t histogram = 0; histogram < HistogramCounters.size(); ++histogram) {
            for (size_t bucket = 0; bucket < HistogramBucketCount; ++bucket) {
                const u64 count =
                    ring->histograms[histogram][bucket].load(std::memory_order_relaxed);
                if (count != 0) {
                    const auto [lower, upper] = HistogramBounds(bucket);
                    file << "thread_histogram," << ring->id << ",0,"
                         << HistogramNames[histogram] << ',' << lower << ',' << upper << ','
                         << count << '\n';
                }
            }
        }
    }
    for (size_t i = 0; i < totals.size(); ++i) {
        file << "counter,,0," << CounterNames[i] << ",0,0," << totals[i] << '\n';
    }
    for (size_t opcode = 0; opcode < opcode_totals.size(); ++opcode) {
        if (opcode_totals[opcode] != 0) {
            file << "opcode,,0,pm4_opcode," << opcode << ",0," << opcode_totals[opcode]
                 << '\n';
        }
    }
    for (size_t histogram = 0; histogram < HistogramCounters.size(); ++histogram) {
        for (size_t bucket = 0; bucket < HistogramBucketCount; ++bucket) {
            const u64 count = histogram_totals[histogram][bucket];
            if (count != 0) {
                const auto [lower, upper] = HistogramBounds(bucket);
                file << "histogram,,0," << HistogramNames[histogram] << ',' << lower << ','
                     << upper << ',' << count << '\n';
            }
        }
    }
    for (const auto& event : events) {
        const size_t type = static_cast<size_t>(event.type);
        const auto name = type < EventNames.size() ? EventNames[type] : "unknown";
        file << "event," << event.thread_id << ',' << event.timestamp_ns << ',' << name << ','
             << event.arg0 << ',' << event.arg1 << ',' << event.sequence << '\n';
    }
    return path;
}

} // namespace Common::PerformanceTelemetry
