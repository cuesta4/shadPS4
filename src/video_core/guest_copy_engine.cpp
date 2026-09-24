// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include <boost/container/static_vector.hpp>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/performance_telemetry.h"
#include "common/thread.h"
#include "core/memory.h"
#include "video_core/guest_copy_engine.h"

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#define GUEST_COPY_PAUSE() _mm_pause()
#else
#define GUEST_COPY_PAUSE() std::this_thread::yield()
#endif

namespace VideoCore {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] u64 NowNs() noexcept {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

/// Time a worker keeps polling for new jobs before parking.
constexpr u64 WorkerSpinNs = 40'000;
/// Time a waiter keeps polling for completion before parking.
constexpr u64 WaiterSpinNs = 20'000;
/// Interval between summary log lines.
constexpr u64 StatsLogIntervalNs = 10'000'000'000ULL;

} // Anonymous namespace

thread_local bool GuestCopyEngine::is_producer_thread = false;

GuestCopyEngine& GuestCopyEngine::Instance() {
    static GuestCopyEngine instance;
    return instance;
}

GuestCopyEngine::GuestCopyEngine()
    : slots{std::make_unique<std::array<Slot, SlotCount>>()},
      pending_reads{std::make_unique<std::array<std::atomic<u32>, PendingTableSize>>()},
      read_protect_intents{std::make_unique<std::array<std::atomic<u32>, PendingTableSize>>()} {
    for (auto& counter : *pending_reads) {
        counter.store(0, std::memory_order_relaxed);
    }
    for (auto& counter : *read_protect_intents) {
        counter.store(0, std::memory_order_relaxed);
    }
}

GuestCopyEngine::~GuestCopyEngine() {
    Stop();
}

void GuestCopyEngine::Start(u32 num_workers) {
    if (active.load(std::memory_order_acquire) || num_workers == 0) {
        return;
    }
    workers.reserve(num_workers);
    for (u32 i = 0; i < num_workers; ++i) {
        workers.emplace_back([this, i](std::stop_token stoken) { WorkerLoop(stoken, i); });
    }
    active.store(true, std::memory_order_release);
    LOG_INFO(Render_Vulkan, "Guest copy engine started with {} workers", num_workers);
}

void GuestCopyEngine::Stop() {
    if (!active.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    // Finish everything that was queued; new copies run inline from now on.
    WaitCompleted(SubmittedSeq());
    for (auto& worker : workers) {
        worker.request_stop();
    }
    wake_signal.fetch_add(1, std::memory_order_seq_cst);
    wake_signal.notify_all();
    workers.clear();
}

void GuestCopyEngine::SetProducerThread() noexcept {
    is_producer_thread = true;
}

u64 GuestCopyEngine::Enqueue(std::span<const Op> ops) {
    if (!CanDefer()) {
        ExecuteInline(ops);
        return SubmittedSeq();
    }
    for (const Op& op : ops) {
        // Copies spanning several pieces are resolved whole so the resolver records one GPU
        // copy per shadow instead of one per piece.
        if (op.size > SplitBytes && op.kind != OpKind::Zero && op.dst_buffer != 0 &&
            IsReadProtected(op.source, op.size) && TryResolveProtected(op)) {
            continue;
        }
        EnqueueOp(op, true);
    }
    if (building_ops != 0) {
        PublishJob(building_ops, building_bytes);
    }
    return submitted.load(std::memory_order_relaxed);
}

bool GuestCopyEngine::TryResolveProtected(const Op& op) {
    const auto resolver = protected_resolver.load(std::memory_order_acquire);
    if (resolver == nullptr) {
        return false;
    }
    std::array<Op, MaxResolverRemainder> remainder{};
    u32 remainder_count = 0;
    const u64 served = resolver(protected_resolver_context, op, remainder, remainder_count);
    if (served == 0) {
        return false;
    }
    stats.gpu_served_ops.fetch_add(1, std::memory_order_relaxed);
    stats.gpu_served_bytes.fetch_add(served, std::memory_order_relaxed);
    if (Common::PerformanceTelemetry::Enabled()) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::GuestCopyGpuServedOps, 1);
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::GuestCopyGpuServedBytes, served);
    }
    for (u32 i = 0; i < remainder_count; ++i) {
        EnqueueOp(remainder[i], false);
    }
    return true;
}

void GuestCopyEngine::EnqueueOp(const Op& op, bool allow_resolve) {
    if (op.size == 0) {
        return;
    }
    u64 offset = 0;
    while (offset < op.size) {
        const u64 piece = std::min<u64>(op.size - offset, SplitBytes);
        Op piece_op = op;
        if (op.kind != OpKind::Zero) {
            piece_op.source += offset;
        }
        piece_op.destination += offset;
        piece_op.dst_offset += offset;
        piece_op.size = piece;
        offset += piece;

        // Publish the pending read before checking for read protection; BeginReadProtect
        // publishes its intent before checking for pending reads. One side always sees the
        // other, so a worker never reads a page after its read access is revoked.
        MarkPending(piece_op, true);
        if (piece_op.kind != OpKind::Zero) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (IsReadProtected(piece_op.source, piece_op.size)) {
                MarkPending(piece_op, false);
                // Reading the range here would run the fault handlers, which may wait for the
                // GPU. The resolver serves what it can without touching the protected pages and
                // hands back the rest, which goes through this protocol again.
                if (allow_resolve && piece_op.dst_buffer != 0 && TryResolveProtected(piece_op)) {
                    continue;
                }
                // Faults on this range must be handled on the command processor thread.
                stats.protected_inline_ops.fetch_add(1, std::memory_order_relaxed);
                if (Common::PerformanceTelemetry::Enabled()) {
                    Common::PerformanceTelemetry::AddEnabled(
                        Common::PerformanceTelemetry::Counter::GuestCopyProtectedInlineOps, 1);
                }
                ExecuteInline(std::span{&piece_op, 1});
                continue;
            }
        }

        if (building_ops == MaxOpsPerJob ||
            (building_ops != 0 && building_bytes + piece > SplitBytes)) {
            PublishJob(building_ops, building_bytes);
        }
        if (building_ops == 0) {
            WaitForSlot();
        }
        Slot& slot = (*slots)[(submitted.load(std::memory_order_relaxed) + 1) & SlotMask];
        slot.ops[building_ops++] = piece_op;
        building_bytes += piece;
    }
}

void GuestCopyEngine::PublishJob(u32 num_ops, u64 bytes) {
    const u64 seq = submitted.load(std::memory_order_relaxed) + 1;
    Slot& slot = (*slots)[seq & SlotMask];
    slot.num_ops = num_ops;
    slot.bytes = bytes;
    building_ops = 0;
    building_bytes = 0;

    stats.jobs.fetch_add(1, std::memory_order_relaxed);
    stats.ops.fetch_add(num_ops, std::memory_order_relaxed);
    stats.bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (Common::PerformanceTelemetry::Enabled()) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::GuestCopyJobs, 1);
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::GuestCopyBytes, bytes);
        Common::PerformanceTelemetry::ObserveMaxEnabled(
            Common::PerformanceTelemetry::Counter::GuestCopyQueueDepthMax,
            seq - completed.load(std::memory_order_relaxed));
    }

    submitted.store(seq, std::memory_order_seq_cst);
    wake_signal.fetch_add(1, std::memory_order_seq_cst);
    if (parked_workers.load(std::memory_order_seq_cst) != 0) {
        wake_signal.notify_one();
    }

    // Periodic summary so the effect is visible without a telemetry build.
    static thread_local u64 next_log_ns = 0;
    static thread_local Stats last_logged{};
    if ((seq & 0x3FF) == 0) {
        const u64 now = NowNs();
        if (next_log_ns == 0) {
            next_log_ns = now + StatsLogIntervalNs;
        } else if (now >= next_log_ns) {
            next_log_ns = now + StatsLogIntervalNs;
            const Stats current = GetStats();
            constexpr double MiB = 1024.0 * 1024.0;
            LOG_INFO(Render_Vulkan,
                     "Guest copies (last 10s): {} jobs, {:.1f} MiB deferred, {:.1f} MiB inline, "
                     "workers {:.1f} ms, waiters helped {:.1f} ms, producer blocked {} times "
                     "({:.1f} ms), overlap waits {}, ring-full waits {}, protected inline ops {}, "
                     "GPU-served ops {} ({:.1f} MiB)",
                     current.jobs - last_logged.jobs,
                     static_cast<double>(current.bytes - last_logged.bytes) / MiB,
                     static_cast<double>(current.inline_bytes - last_logged.inline_bytes) / MiB,
                     static_cast<double>(current.worker_ns - last_logged.worker_ns) / 1e6,
                     static_cast<double>(current.help_ns - last_logged.help_ns) / 1e6,
                     current.wait_calls - last_logged.wait_calls,
                     static_cast<double>(current.wait_ns - last_logged.wait_ns) / 1e6,
                     current.overlap_waits - last_logged.overlap_waits,
                     current.slot_full_waits - last_logged.slot_full_waits,
                     current.protected_inline_ops - last_logged.protected_inline_ops,
                     current.gpu_served_ops - last_logged.gpu_served_ops,
                     static_cast<double>(current.gpu_served_bytes - last_logged.gpu_served_bytes) /
                         MiB);
            last_logged = current;
        }
    }
}

void GuestCopyEngine::WaitForSlot() {
    const u64 seq = submitted.load(std::memory_order_relaxed) + 1;
    if (seq <= SlotCount) {
        return;
    }
    const u64 previous = seq - SlotCount;
    if (completed.load(std::memory_order_acquire) >= previous) {
        return;
    }
    stats.slot_full_waits.fetch_add(1, std::memory_order_relaxed);
    WaitCompleted(previous);
}

bool GuestCopyEngine::TryRunOne(bool from_worker) {
    u64 current = claimed.load(std::memory_order_acquire);
    for (;;) {
        if (current >= submitted.load(std::memory_order_acquire)) {
            return false;
        }
        if (claimed.compare_exchange_weak(current, current + 1, std::memory_order_acq_rel,
                                          std::memory_order_acquire)) {
            break;
        }
    }
    RunJob(current + 1, from_worker);
    return true;
}

void GuestCopyEngine::RunJob(u64 seq, bool from_worker) {
    Slot& slot = (*slots)[seq & SlotMask];
    const u64 start = NowNs();
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    const std::span<const Op> ops{slot.ops.data(), slot.num_ops};
    if (const u64 forbidden = self_test_forbidden.load(std::memory_order_acquire);
        forbidden != 0) [[unlikely]] {
        const VAddr begin = self_test_base + (forbidden >> 32);
        const VAddr end = begin + (forbidden & 0xFFFFFFFFULL);
        for (const Op& op : ops) {
            if (op.kind != OpKind::Zero && op.source < end && begin < op.source + op.size) {
                self_test_violations.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    ExecuteOps(ops, telemetry_enabled);
    for (const Op& op : ops) {
        MarkPending(op, false);
    }
    const u64 elapsed = NowNs() - start;
    if (from_worker) {
        stats.worker_ns.fetch_add(elapsed, std::memory_order_relaxed);
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::GuestCopyWorkerNs, elapsed);
        }
    } else {
        stats.help_ns.fetch_add(elapsed, std::memory_order_relaxed);
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::GuestCopyHelpNs, elapsed);
        }
    }
    slot.done_seq.store(seq, std::memory_order_seq_cst);
    AdvanceCompleted();
}

void GuestCopyEngine::AdvanceCompleted() {
    u64 current = completed.load(std::memory_order_seq_cst);
    bool advanced = false;
    for (;;) {
        const u64 next = current + 1;
        if ((*slots)[next & SlotMask].done_seq.load(std::memory_order_seq_cst) != next) {
            break;
        }
        if (completed.compare_exchange_weak(current, next, std::memory_order_seq_cst,
                                            std::memory_order_seq_cst)) {
            current = next;
            advanced = true;
        }
    }
    if (advanced && completion_waiters.load(std::memory_order_seq_cst) != 0) {
        completed.notify_all();
    }
}

void GuestCopyEngine::WaitCompleted(u64 seq) {
    if (completed.load(std::memory_order_acquire) >= seq) {
        return;
    }
    const u64 start = NowNs();
    stats.wait_calls.fetch_add(1, std::memory_order_relaxed);
    u64 spin_start = start;
    while (completed.load(std::memory_order_acquire) < seq) {
        if (TryRunOne(false)) {
            spin_start = NowNs();
            continue;
        }
        if (NowNs() - spin_start < WaiterSpinNs) {
            for (u32 i = 0; i < 32; ++i) {
                GUEST_COPY_PAUSE();
            }
            continue;
        }
        completion_waiters.fetch_add(1, std::memory_order_seq_cst);
        const u64 observed = completed.load(std::memory_order_seq_cst);
        if (observed < seq && claimed.load(std::memory_order_seq_cst) >=
                                  submitted.load(std::memory_order_seq_cst)) {
            completed.wait(observed, std::memory_order_seq_cst);
        }
        completion_waiters.fetch_sub(1, std::memory_order_seq_cst);
        spin_start = NowNs();
    }
    const u64 elapsed = NowNs() - start;
    stats.wait_ns.fetch_add(elapsed, std::memory_order_relaxed);
    if (Common::PerformanceTelemetry::Enabled()) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::GuestCopyWaitCalls, 1);
        Common::PerformanceTelemetry::AddEnabled(
            is_producer_thread ? Common::PerformanceTelemetry::Counter::GuestCopyProducerWaitNs
                               : Common::PerformanceTelemetry::Counter::GuestCopyWaitNs,
            elapsed);
    }
}

void GuestCopyEngine::AddReadIntent(VAddr addr, u64 size, bool add) noexcept {
    const u64 first = addr >> GranuleBits;
    const u64 last = (addr + size - 1) >> GranuleBits;
    for (u64 granule = first; granule <= last; ++granule) {
        auto& counter = (*read_protect_intents)[PendingIndex(granule)];
        if (add) {
            counter.fetch_add(1, std::memory_order_seq_cst);
        } else {
            counter.fetch_sub(1, std::memory_order_release);
        }
    }
    if (add) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
}

void GuestCopyEngine::DrainRange(VAddr addr, u64 size) {
    // Pending marks cover jobs the producer is still building, which a sequence number does not.
    // With the read intent published, new reads of the range run inline on the producer, so the
    // marks can only drain.
    const u64 start = NowNs();
    stats.wait_calls.fetch_add(1, std::memory_order_relaxed);
    u64 spin_start = start;
    while (OverlapsPending(addr, size)) {
        if (TryRunOne(false)) {
            spin_start = NowNs();
            continue;
        }
        if (NowNs() - spin_start < WaiterSpinNs) {
            for (u32 i = 0; i < 32; ++i) {
                GUEST_COPY_PAUSE();
            }
            continue;
        }
        std::this_thread::yield();
    }
    const u64 elapsed = NowNs() - start;
    stats.wait_ns.fetch_add(elapsed, std::memory_order_relaxed);
    if (Common::PerformanceTelemetry::Enabled()) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::GuestCopyWaitCalls, 1);
        Common::PerformanceTelemetry::AddEnabled(
            is_producer_thread ? Common::PerformanceTelemetry::Counter::GuestCopyProducerWaitNs
                               : Common::PerformanceTelemetry::Counter::GuestCopyWaitNs,
            elapsed);
    }
}

void GuestCopyEngine::BeginReadProtect(VAddr addr, u64 size) {
    if (size == 0) {
        return;
    }
    AddReadIntent(addr, size, true);
    if (OverlapsPending(addr, size)) {
        stats.overlap_waits.fetch_add(1, std::memory_order_relaxed);
        DrainRange(addr, size);
    }
}

void GuestCopyEngine::EndReadProtect(VAddr addr, u64 size) noexcept {
    if (size == 0) {
        return;
    }
    AddReadIntent(addr, size, false);
}

bool GuestCopyEngine::IsReadProtected(VAddr addr, u64 size) const noexcept {
    const u64 first = addr >> GranuleBits;
    const u64 last = (addr + size - 1) >> GranuleBits;
    for (u64 granule = first; granule <= last; ++granule) {
        if ((*read_protect_intents)[PendingIndex(granule)].load(std::memory_order_acquire) != 0) {
            return true;
        }
    }
    const ReadProtectionProbe probe = read_probe.load(std::memory_order_acquire);
    return probe != nullptr && probe(read_probe_context, addr, size);
}

void GuestCopyEngine::WaitForGuestWriteSlow(VAddr addr, u64 size) {
    if (size == 0 || !OverlapsPending(addr, size)) {
        return;
    }
    stats.overlap_waits.fetch_add(1, std::memory_order_relaxed);
    if (Common::PerformanceTelemetry::Enabled()) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::GuestCopyOverlapWaits, 1);
    }
    if (is_producer_thread) {
        // The producer never holds an unpublished job outside Enqueue.
        WaitCompleted(SubmittedSeq());
        return;
    }
    AddReadIntent(addr, size, true);
    DrainRange(addr, size);
    AddReadIntent(addr, size, false);
}

void GuestCopyEngine::WorkerLoop(std::stop_token stoken, u32 index) {
    const std::string name = "shadPS4:GuestCopy" + std::to_string(index);
    Common::SetCurrentThreadName(name.c_str());
    Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
    while (!stoken.stop_requested()) {
        if (TryRunOne(true)) {
            continue;
        }
        const u64 spin_start = NowNs();
        bool found = false;
        while (NowNs() - spin_start < WorkerSpinNs) {
            for (u32 i = 0; i < 64; ++i) {
                GUEST_COPY_PAUSE();
            }
            if (claimed.load(std::memory_order_relaxed) <
                submitted.load(std::memory_order_acquire)) {
                found = true;
                break;
            }
            if (stoken.stop_requested()) {
                return;
            }
        }
        if (found) {
            continue;
        }
        parked_workers.fetch_add(1, std::memory_order_seq_cst);
        const u64 observed = wake_signal.load(std::memory_order_seq_cst);
        if (claimed.load(std::memory_order_seq_cst) >= submitted.load(std::memory_order_seq_cst) &&
            !stoken.stop_requested()) {
            wake_signal.wait(observed, std::memory_order_seq_cst);
        }
        parked_workers.fetch_sub(1, std::memory_order_seq_cst);
    }
}

void GuestCopyEngine::ExecuteInline(std::span<const Op> ops) {
    u64 bytes = 0;
    for (const Op& op : ops) {
        bytes += op.size;
    }
    stats.inline_bytes.fetch_add(bytes, std::memory_order_relaxed);
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::GuestCopyInlineBytes, bytes);
    }
    ExecuteOps(ops, telemetry_enabled);
}

void GuestCopyEngine::ExecuteOps(std::span<const Op> ops, bool telemetry_enabled) {
    boost::container::static_vector<Core::MemoryManager::SparseCopyRequest, MaxOpsPerJob> batch;
    u64 batch_bytes = 0;
    const auto flush = [&] {
        if (batch.empty()) {
            return;
        }
        Common::PerformanceTelemetry::ScopedSemanticReadOrigin upload_origin{
            Common::PerformanceTelemetry::SemanticReadOrigin::GpuUploadFromGuestRam};
        Core::Memory::Instance()->CopySparseMemoryBatch(
            std::span<const Core::MemoryManager::SparseCopyRequest>{batch.data(), batch.size()},
            batch_bytes, telemetry_enabled, false);
        batch.clear();
        batch_bytes = 0;
    };
    for (const Op& op : ops) {
        if (op.size == 0) {
            continue;
        }
        if (op.kind == OpKind::Zero) {
            std::memset(op.destination, 0, op.size);
            continue;
        }
        if (op.kind == OpKind::Host) {
            std::memcpy(op.destination, reinterpret_cast<const void*>(op.source), op.size);
            continue;
        }
        if (batch.size() == batch.capacity()) {
            flush();
        }
        batch.push_back({.source = op.source, .destination = op.destination, .size = op.size});
        batch_bytes += op.size;
    }
    flush();
}

void GuestCopyEngine::MarkPending(const Op& op, bool add) noexcept {
    if (op.kind == OpKind::Zero || op.size == 0) {
        return;
    }
    const u64 first = op.source >> GranuleBits;
    const u64 last = (op.source + op.size - 1) >> GranuleBits;
    for (u64 granule = first; granule <= last; ++granule) {
        auto& counter = (*pending_reads)[PendingIndex(granule)];
        if (add) {
            counter.fetch_add(1, std::memory_order_relaxed);
        } else {
            counter.fetch_sub(1, std::memory_order_release);
        }
    }
}

bool GuestCopyEngine::OverlapsPending(VAddr addr, u64 size) const noexcept {
    const u64 first = addr >> GranuleBits;
    const u64 last = (addr + size - 1) >> GranuleBits;
    for (u64 granule = first; granule <= last; ++granule) {
        if ((*pending_reads)[PendingIndex(granule)].load(std::memory_order_acquire) != 0) {
            return true;
        }
    }
    return false;
}

GuestCopyEngine::Stats GuestCopyEngine::GetStats() const noexcept {
    return Stats{
        .jobs = stats.jobs.load(std::memory_order_relaxed),
        .ops = stats.ops.load(std::memory_order_relaxed),
        .bytes = stats.bytes.load(std::memory_order_relaxed),
        .inline_bytes = stats.inline_bytes.load(std::memory_order_relaxed),
        .worker_ns = stats.worker_ns.load(std::memory_order_relaxed),
        .help_ns = stats.help_ns.load(std::memory_order_relaxed),
        .wait_calls = stats.wait_calls.load(std::memory_order_relaxed),
        .wait_ns = stats.wait_ns.load(std::memory_order_relaxed),
        .overlap_waits = stats.overlap_waits.load(std::memory_order_relaxed),
        .slot_full_waits = stats.slot_full_waits.load(std::memory_order_relaxed),
        .protected_inline_ops = stats.protected_inline_ops.load(std::memory_order_relaxed),
        .gpu_served_ops = stats.gpu_served_ops.load(std::memory_order_relaxed),
        .gpu_served_bytes = stats.gpu_served_bytes.load(std::memory_order_relaxed),
    };
}


namespace {

struct XorShift {
    u64 state;

    u64 Next() noexcept {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    u64 Range(u64 low, u64 high) noexcept {
        return low + Next() % (high - low + 1);
    }
};

} // Anonymous namespace

bool GuestCopyEngine::RunSelfTest() {
    constexpr u64 SourceSize = 32_MB;
    constexpr u64 DestinationSize = 32_MB;
    constexpr u32 Rounds = 24;
    constexpr u32 NumWaiters = 2;

    struct Expectation {
        u64 seq;
        u64 source_offset;
        u64 destination_offset;
        u64 size;
        bool zero;
    };

    std::vector<u8> source(SourceSize);
    for (u64 i = 0; i < SourceSize; ++i) {
        source[i] = static_cast<u8>(((i * 2654435761ULL) >> 13) ^ (i >> 20));
    }
    std::vector<u8> destination(DestinationSize);

    std::string failure;
    u64 total_ops = 0;
    u64 total_bytes = 0;
    u64 overlap_checks = 0;
    std::atomic<bool> stop_waiters{false};
    std::atomic<bool> waiter_failed{false};
    std::atomic<u64> waiter_checks{0};
    std::atomic<u64> protect_cycles{0};
    self_test_base = reinterpret_cast<VAddr>(source.data());
    self_test_violations.store(0, std::memory_order_relaxed);
    const u64 inline_before = stats.protected_inline_ops.load(std::memory_order_relaxed);
    const u64 served_before = stats.gpu_served_ops.load(std::memory_order_relaxed);

    // Stands in for the GPU: serves the first half of a protected copy directly and leaves the
    // second half to the regular protocol, which must still keep workers off protected ranges.
    const auto saved_resolver = protected_resolver.load(std::memory_order_acquire);
    void* const saved_resolver_context = protected_resolver_context;
    SetProtectedCopyResolver(
        [](void*, const Op& op, std::span<Op, MaxResolverRemainder> remainder,
           u32& remainder_count) -> u64 {
            const u64 served = std::max<u64>(op.size / 2, 1);
            std::memcpy(op.destination, reinterpret_cast<const u8*>(op.source), served);
            remainder_count = 0;
            if (served < op.size) {
                remainder[remainder_count++] = Op{
                    .source = op.source + served,
                    .destination = op.destination + served,
                    .size = op.size - served,
                    .kind = op.kind,
                    .dst_buffer = op.dst_buffer,
                    .dst_offset = op.dst_offset + served,
                };
            }
            return served;
        },
        nullptr);

    std::thread producer([&] {
        is_producer_thread = true;
        std::vector<std::jthread> waiters;
        for (u32 w = 0; w < NumWaiters; ++w) {
            waiters.emplace_back([&, w] {
                XorShift rng{0x1234567ULL + w};
                while (!stop_waiters.load(std::memory_order_acquire)) {
                    const u64 submitted_now = SubmittedSeq();
                    if (submitted_now == 0) {
                        std::this_thread::yield();
                        continue;
                    }
                    const u64 target =
                        submitted_now - rng.Next() % std::min<u64>(submitted_now, 64);
                    WaitCompleted(target);
                    if (CompletedSeq() < target) {
                        waiter_failed.store(true, std::memory_order_relaxed);
                    }
                    waiter_checks.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }

        // Revokes "read access" to source ranges the way read watchers do; no queued job may
        // read a range between BeginReadProtect returning and EndReadProtect.
        waiters.emplace_back([&] {
            XorShift rng{0xC0FFEEULL};
            while (!stop_waiters.load(std::memory_order_acquire)) {
                const u64 size = rng.Range(4_KB, 64_KB);
                const u64 offset = rng.Range(0, SourceSize - size);
                const VAddr begin = self_test_base + offset;
                BeginReadProtect(begin, size);
                self_test_forbidden.store((offset << 32) | size, std::memory_order_release);
                const u64 hold_until = NowNs() + rng.Range(1'000, 50'000);
                while (NowNs() < hold_until) {
                    GUEST_COPY_PAUSE();
                }
                self_test_forbidden.store(0, std::memory_order_release);
                EndReadProtect(begin, size);
                protect_cycles.fetch_add(1, std::memory_order_relaxed);
            }
        });

        XorShift rng{0x9E3779B97F4A7C15ULL};
        std::vector<Expectation> round_ops;
        const auto verify = [&](const Expectation& expected) {
            const u8* bytes = destination.data() + expected.destination_offset;
            if (expected.zero) {
                return std::all_of(bytes, bytes + expected.size, [](u8 b) { return b == 0; });
            }
            return std::memcmp(bytes, source.data() + expected.source_offset, expected.size) == 0;
        };

        for (u32 round = 0; round < Rounds && failure.empty(); ++round) {
            std::memset(destination.data(), 0xCD, DestinationSize);
            round_ops.clear();
            size_t verified_index = 0;
            u64 destination_cursor = 0;
            const auto verify_through = [&](u64 seq) {
                while (verified_index < round_ops.size() && round_ops[verified_index].seq <= seq) {
                    const auto& expected = round_ops[verified_index];
                    if (!verify(expected)) {
                        failure = "job " + std::to_string(expected.seq) +
                                  " incomplete after WaitCompleted(" + std::to_string(seq) + ")";
                        return;
                    }
                    ++verified_index;
                }
            };
            bool full = false;
            while (!full && failure.empty()) {
                std::array<Op, 6> ops{};
                std::array<Expectation, 6> expected{};
                const u32 wanted = static_cast<u32>(rng.Range(1, ops.size()));
                u32 count = 0;
                for (u32 i = 0; i < wanted; ++i) {
                    const u64 roll = rng.Next() % 100;
                    const u64 size = roll < 80   ? rng.Range(1, 16_KB)
                                     : roll < 97 ? rng.Range(16_KB, 256_KB)
                                                 : rng.Range(256_KB, 3_MB);
                    if (destination_cursor + size > DestinationSize) {
                        full = true;
                        break;
                    }
                    const bool zero = rng.Next() % 16 == 0;
                    const u64 source_offset = rng.Range(0, SourceSize - size);
                    const bool resolvable = !zero && rng.Next() % 2 == 0;
                    ops[count] = Op{
                        .source = zero ? 0 : reinterpret_cast<VAddr>(source.data() + source_offset),
                        .destination = destination.data() + destination_cursor,
                        .size = size,
                        .kind = zero ? OpKind::Zero : OpKind::Host,
                        .dst_buffer = resolvable ? 1ULL : 0ULL,
                        .dst_offset = destination_cursor,
                    };
                    expected[count] = Expectation{
                        .source_offset = source_offset,
                        .destination_offset = destination_cursor,
                        .size = size,
                        .zero = zero,
                    };
                    destination_cursor += size + rng.Range(0, 64);
                    total_bytes += size;
                    ++count;
                }
                if (count == 0) {
                    break;
                }
                const u64 seq = Enqueue(std::span<const Op>{ops.data(), count});
                for (u32 i = 0; i < count; ++i) {
                    expected[i].seq = seq;
                    round_ops.push_back(expected[i]);
                }
                total_ops += count;

                // A write to a range a pending job reads must wait for that job.
                const auto& last = expected[count - 1];
                if (!last.zero && rng.Next() % 32 == 0) {
                    WaitForGuestWrite(reinterpret_cast<VAddr>(source.data() + last.source_offset),
                                      last.size);
                    ++overlap_checks;
                    if (!verify(last)) {
                        failure = "overlap wait returned before job " + std::to_string(seq) +
                                  " finished reading its source";
                    }
                }
                if (rng.Next() % 8 == 0) {
                    const u64 target = round_ops[rng.Next() % round_ops.size()].seq;
                    WaitCompleted(target);
                    verify_through(target);
                }
            }
            Drain();
            verify_through(SubmittedSeq());
            if (failure.empty() && CompletedSeq() != SubmittedSeq()) {
                failure = "drain left completed behind submitted";
            }
        }
        if (failure.empty()) {
            for (const auto& counter : *pending_reads) {
                if (counter.load(std::memory_order_acquire) != 0) {
                    failure = "pending read table not empty after drain";
                    break;
                }
            }
        }
        stop_waiters.store(true, std::memory_order_release);
        waiters.clear();
        is_producer_thread = false;
    });
    producer.join();
    SetProtectedCopyResolver(saved_resolver, saved_resolver_context);

    if (failure.empty() && waiter_failed.load(std::memory_order_relaxed)) {
        failure = "a waiter returned before its target sequence completed";
    }
    if (const u64 violations = self_test_violations.load(std::memory_order_relaxed);
        failure.empty() && violations != 0) {
        failure = std::to_string(violations) + " queued job(s) read a read-protected range";
    }
    self_test_base = 0;
    if (!failure.empty()) {
        LOG_ERROR(Render_Vulkan, "Guest copy self-test FAILED: {}", failure);
        return false;
    }
    LOG_INFO(Render_Vulkan,
             "Guest copy self-test passed: {} workers, {} ops, {:.1f} MiB, {} overlap checks, "
             "{} concurrent waits, {} protect cycles, {} ops rerouted inline, {} ops resolved",
             workers.size(), total_ops, static_cast<double>(total_bytes) / (1024.0 * 1024.0),
             overlap_checks, waiter_checks.load(std::memory_order_relaxed),
             protect_cycles.load(std::memory_order_relaxed),
             stats.protected_inline_ops.load(std::memory_order_relaxed) - inline_before,
             stats.gpu_served_ops.load(std::memory_order_relaxed) - served_before);
    return true;
}

} // namespace VideoCore
