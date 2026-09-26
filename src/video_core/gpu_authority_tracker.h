// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <tsl/robin_map.h>

#include "common/performance_telemetry.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/buffer_cache/stream_buffer_pin.h"

namespace Vulkan {
class Rasterizer;
}

namespace VideoCore {

enum class GpuAuthorityState : u8 {
    GpuAuthoritative,
    Materializing,
    HostCurrent,
    Superseded,
    Failed,
};

struct GpuAuthorityShadow final : StreamBufferPin {
    GpuAuthorityShadow(u8* data_, VAddr guest_addr_, u64 buffer_offset_, u32 size_, u64 tick_)
        : data{data_}, guest_addr{guest_addr_}, buffer_offset{buffer_offset_}, size{size_},
          ready_tick{tick_}, tick{tick_} {}

    ~GpuAuthorityShadow() override {
        Release();
    }

    void Reclaim() noexcept override {
        if (IsReleased()) {
            return;
        }
        std::scoped_lock lock{data_mutex};
        if (IsReleased() || owned_data) {
            return;
        }
        auto storage = std::unique_ptr<u8[]>{new (std::nothrow) u8[size]};
        if (!storage) {
            return;
        }
        std::memcpy(storage.get(), data, size);
        data = storage.get();
        owned_data = std::move(storage);
        Release();
    }

    void ExtendLifetime(u64 required_tick) noexcept {
        u64 current_tick = tick.load(std::memory_order_relaxed);
        while (current_tick < required_tick &&
               !tick.compare_exchange_weak(current_tick, required_tick,
                                           std::memory_order_release,
                                           std::memory_order_relaxed)) {
        }
    }

    /// Tick until which the download buffer region must stay reserved. GPU consumers of the
    /// shadow extend it.
    [[nodiscard]] u64 Tick() const noexcept {
        return tick.load(std::memory_order_acquire);
    }

    /// Tick after which the shadow bytes are valid.
    [[nodiscard]] u64 ReadyTick() const noexcept {
        return ready_tick;
    }

    [[nodiscard]] u64 RequiredTick(u64 allocation_tick) const noexcept override {
        return std::max(allocation_tick, Tick());
    }

    std::mutex data_mutex;
    u8* data{};
    VAddr guest_addr{};
    u64 buffer_offset{};
    u32 size{};
    const u64 ready_tick{};
    std::atomic<u64> tick{};
    std::unique_ptr<u8[]> owned_data;
};

/// Byte range [first, second) of guest memory.
using GuestRange = std::pair<VAddr, VAddr>;

struct GpuAuthorityEntry {
    u64 authority_seq{0};
    u64 candidate_seq{0};
    Common::PerformanceTelemetry::ScopeSeq scope_seq{0};
    Common::PerformanceTelemetry::CauseSeq cause_seq{0};
    Common::PerformanceTelemetry::SignalSeq signal_seq{0};
    u32 image_id{0};
    u64 image_uid{0};
    u64 resource_id{0};
    u64 resource_version{0};
    VAddr guest_begin{0};
    VAddr guest_end{0};
    u32 download_size{0};
    u64 producer_seq{0};
    Common::PerformanceTelemetry::PacketSeq producer_packet_seq{0};
    u64 producer_tick{0};
    GpuAuthorityState state{GpuAuthorityState::GpuAuthoritative};
    /// The bytes still live in the producing image; no copy was recorded yet. The command
    /// processor records one before anything overwrites or frees the image, or when a consumer
    /// needs the bytes outside the image (see TextureCache::PreserveDirectAuthority).
    bool direct{false};
    bool host_current{false};
    bool gpu_consumed{false};
    /// Command buffers that read the shadow instead of materialized guest RAM.
    u32 gpu_serves{0};
    u64 last_gpu_serve_tick{0};
    /// Parts of the range written after the authority was produced, by a newer authority or
    /// through the backing view. Materializing never writes them.
    boost::container::small_vector<GuestRange, 2> masked;
    std::shared_ptr<GpuAuthorityShadow> shadow;
    std::shared_ptr<std::mutex> entry_mutex{std::make_shared<std::mutex>()};
    std::shared_ptr<std::condition_variable> cv{std::make_shared<std::condition_variable>()};
};

/// Part of a guest range whose current bytes live in an authority shadow on the GPU.
struct GpuShadowPiece {
    VAddr addr{};
    u64 size{};
    /// Offset of the bytes at addr inside the download buffer.
    u64 buffer_offset{};
    std::shared_ptr<GpuAuthorityEntry> entry;
    std::shared_ptr<GpuAuthorityShadow> shadow;
};

using GpuShadowPieces = boost::container::small_vector<GpuShadowPiece, 4>;

/// A completion signal (label write and/or interrupt) whose publication follows program order
/// relative to eager readback commits.
struct SignalPublication {
    /// Label bytes the publication writes, empty when it only raises an interrupt.
    VAddr label_addr{};
    u64 label_size{};
    /// Publishes the signal. The argument is false when the label lies in memory that was
    /// unmapped in the meantime; the interrupt is still raised.
    Common::UniqueFunction<void, bool> publish;
};

/// Backing writes on this thread carry bytes produced when bound was taken from
/// GpuAuthorityTracker::AuthoritySeqBound, such as an eager readback committed when the GPU
/// completes. They mask only older authorities: a newer one keeps its bytes and writes them over
/// the committed ones when it materializes.
class ScopedBackingWriteBound {
public:
    explicit ScopedBackingWriteBound(u64 bound) noexcept;
    ~ScopedBackingWriteBound() noexcept;
    ScopedBackingWriteBound(const ScopedBackingWriteBound&) = delete;
    ScopedBackingWriteBound& operator=(const ScopedBackingWriteBound&) = delete;

private:
    u64 prev_bound{};
};

/// Keeps guest RAM coherent for readbacks whose bytes stay on the GPU.
///
/// A readback of a linear single-mip image produces bytes that are identical, byte for byte, to
/// what guest RAM should hold. Instead of writing them to RAM when the GPU completes, which makes
/// every completion signal of the scope wait for the GPU, the range becomes an authority: its
/// pages are read-protected, GPU consumers read the image or the shadow directly, and guest RAM
/// is written only when something on the host touches the bytes. Every signal whose scope only
/// produced authorities is then published when the command processor reaches it, like a signal
/// without readbacks.
class GpuAuthorityTracker {
public:
    static GpuAuthorityTracker& Instance() noexcept;

    void SetRasterizer(Vulkan::Rasterizer* rasterizer_) noexcept;

    /// True when an authority may own guest bytes. Lock-free; everything below returns right
    /// away while it is false.
    [[nodiscard]] bool HasAuthorities() const noexcept {
        return live_count.load(std::memory_order_acquire) != 0;
    }

    /// Registers an authority, superseding older authorities it fully covers and masking the
    /// bytes it shares with the others. Command processor thread, no cache lock held.
    void RegisterAuthority(GpuAuthorityEntry&& entry);

    [[nodiscard]] u64 NextAuthoritySeq() noexcept {
        return next_authority_seq.fetch_add(1, std::memory_order_relaxed);
    }

    /// Sequence the next authority gets. Bytes produced before it are older than every
    /// authority from it on (see ScopedBackingWriteBound).
    [[nodiscard]] u64 AuthoritySeqBound() const noexcept {
        return next_authority_seq.load(std::memory_order_relaxed);
    }

    /// Materializes authorities whose shadow completed a while ago and that nothing superseded
    /// or read since. Nothing waits for the GPU. Keeps the tracker, its read watches and the
    /// pinned shadows small once GPU consumers stop materializing authorities by accident.
    /// Command processor thread, no cache lock held.
    void RetireStaleAuthorities();

    [[nodiscard]] std::vector<std::shared_ptr<GpuAuthorityEntry>> FindOverlaps(VAddr addr,
                                                                                size_t size) const;
    [[nodiscard]] std::shared_ptr<GpuAuthorityEntry> FindBySeq(u64 authority_seq) const;
    [[nodiscard]] std::shared_ptr<GpuAuthorityEntry> GetAuthorityForImage(u64 image_uid,
                                                                          u64 version) const;
    [[nodiscard]] std::shared_ptr<GpuAuthorityEntry> GetAuthorityForRange(VAddr addr,
                                                                         size_t size) const;

    /// Returns a shadow holding the current bytes of [addr, addr + size), for a GPU copy that
    /// starts offset_alignment-aligned in the download buffer. Command processor thread.
    [[nodiscard]] std::shared_ptr<GpuAuthorityShadow> AcquireGpuShadowForImage(
        VAddr addr, size_t size, u32 offset_alignment);

    /// Records the copy of a direct authority into a shadow when it has none yet. Command
    /// processor thread only; other threads go through the command processor.
    void EnsureShadow(const std::shared_ptr<GpuAuthorityEntry>& entry);

    /// Attaches the shadow recorded for a direct authority.
    void AttachShadow(const std::shared_ptr<GpuAuthorityEntry>& entry,
                      std::shared_ptr<GpuAuthorityShadow> shadow);

    /// Gives up an authority whose bytes can no longer be recovered.
    void DropAuthority(const std::shared_ptr<GpuAuthorityEntry>& entry);

    /// Makes guest RAM current for a read of [addr, addr + size), materializing GPU
    /// authoritative ranges. With keep_gpu_servable, authorities whose shadow a GPU consumer can
    /// read directly (see CollectGpuShadowPieces) are left alone: the caller must then serve
    /// those bytes from the shadow instead of reading guest RAM.
    bool ResolveForRamRead(VAddr addr, size_t size,
                           Common::PerformanceTelemetry::GuestSourceConsumePath path,
                           Common::PerformanceTelemetry::ResourceType dest_kind =
                               Common::PerformanceTelemetry::ResourceType::Buffer,
                           u64 dest_res_id = 0, bool keep_gpu_servable = false);

    /// Collects the parts of [addr, addr + size) covered by GPU authoritative ranges whose shadow
    /// the GPU can copy from, so a GPU consumer gets the bytes without the command processor
    /// waiting for the producer. Returns false when an overlapping authority cannot be served
    /// that way, or when the range touches no authority page at all (the caller then reads the
    /// range like any other). The pieces are sorted and disjoint. Nothing changes until
    /// CommitGpuShadowPieces. Command processor thread.
    [[nodiscard]] bool CollectGpuShadowPieces(VAddr addr, size_t size, GpuShadowPieces& pieces);

    /// Keeps the shadows of the pieces alive until consumer_tick completes.
    void CommitGpuShadowPieces(const GpuShadowPieces& pieces, u64 consumer_tick);

    /// True when [addr, addr + size) touches a page protected for authorities.
    [[nodiscard]] bool TouchesAuthorityPages(VAddr addr, size_t size) const;

    /// True when every read-protected page of [addr, addr + size) is protected only for
    /// authorities, so bytes that no authority owns can be read through the backing view.
    [[nodiscard]] bool IsBackingReadable(VAddr addr, size_t size) const;

    /// Reads [addr, addr + size) for the emulator without ever faulting: authority bytes are
    /// materialized first, and pages protected for authorities are read through the backing
    /// view. Returns false when the range could not be read that way; the caller then reads it
    /// directly.
    bool ReadGuestMemory(VAddr addr, u8* destination, size_t size);

    /// Page fault on a page protected for authorities. Returns true when the fault was handled
    /// and the faulting instruction must not run again (it was emulated), or false when the
    /// instruction can simply run again. Sets handled when the tracker owned the fault.
    bool HandleCpuAccess(VAddr fault_addr, void* context, bool is_write, bool& handled);
    void HandleUnmap(VAddr addr, size_t size);

    /// Guest memory was written through the backing view, bypassing the page protection.
    void HandleBackingWrite(VAddr addr, size_t size);

    /// Consumer classification: ranges the guest CPU reads keep the eager readback, which
    /// commits them to RAM when the GPU completes.
    [[nodiscard]] bool PrefersCpuCommit(VAddr begin, u64 size);
    void NoteCpuConsumer(VAddr begin, u64 size);
    void NoteFalseSharing(VAddr page);

    /// Publication timeline. Eager readbacks commit guest RAM when the GPU completes; a signal
    /// that follows one of them in program order must not become visible before that commit.
    /// With orders_all_signals, every signal follows the write until it lands, also the ones
    /// of scopes without readbacks.
    [[nodiscard]] u64 BeginEagerWriteback(bool orders_all_signals = false);
    void CompleteEagerWriteback(u64 seq);

    /// True while a write that every signal has to follow is pending.
    [[nodiscard]] bool MustOrderSignals() const noexcept {
        return ordering_items.load(std::memory_order_acquire) != 0;
    }

    /// Publishes the signal now when nothing it has to follow is pending, or queues it behind
    /// the pending eager readbacks. Returns true when the signal was queued; the caller then
    /// makes sure PublishReady runs once the GPU reaches the current tick. Command processor.
    bool PublishSignal(SignalPublication&& signal);

    /// Publishes queued signals whose preceding readbacks committed. Blocking.
    void PublishReady();

    /// Same from the command processor, which never blocks on a publication in progress.
    /// Returns true when it published something.
    bool TryPublishReady(Common::PerformanceTelemetry::Counter counter);

    [[nodiscard]] bool HasQueuedWork() const noexcept {
        return timeline_items.load(std::memory_order_acquire) != 0;
    }

    void ValidateGpuConsumerBarrier(u64 image_uid, u64 version, u32 image_id,
                                    u32 old_layout, u32 new_layout,
                                    u64 src_stage, u64 src_access,
                                    u64 dst_stage, u64 dst_access,
                                    u64 subresource_range);

private:
    struct TimelineItem {
        u64 seq{};
        bool is_writeback{};
        bool done{};
        bool orders_all{};
        bool write_label{true};
        VAddr label_addr{};
        u64 label_size{};
        Common::UniqueFunction<void, bool> publish;
    };

    struct HotPage {
        u32 faults{};
        u64 window_tick{};
        u64 hot_tick{};
    };

    /// Entry mutex held.
    [[nodiscard]] bool IsGpuServableLocked(const GpuAuthorityEntry& entry) const;
    void RefreshAuthorityReadWatches(VAddr addr, size_t size);
    /// Tracker mutex held.
    void PruneLocked();
    void DrainTimelineLocked(Common::PerformanceTelemetry::Counter counter);
    [[nodiscard]] u64 CurrentTick() const noexcept;
    bool MaterializeEntry(const std::shared_ptr<GpuAuthorityEntry>& entry,
                          std::unique_lock<std::mutex>& entry_lk);

    mutable std::recursive_mutex tracker_mutex;
    Vulkan::Rasterizer* rasterizer{nullptr};
    std::vector<std::shared_ptr<GpuAuthorityEntry>> authorities;
    RangeSet authority_read_watch_ranges;
    std::atomic<size_t> live_count{0};
    std::atomic<u64> next_authority_seq{1};

    std::mutex class_mutex;
    tsl::robin_map<u64, u64> cpu_consumers;
    tsl::robin_map<VAddr, HotPage> hot_pages;
    std::atomic<u32> classified_count{0};

    std::mutex timeline_mutex;
    std::mutex publish_mutex;
    std::deque<TimelineItem> timeline;
    std::atomic<u32> timeline_items{0};
    std::atomic<u32> ordering_items{0};
    u64 next_timeline_seq{1};
};

} // namespace VideoCore
