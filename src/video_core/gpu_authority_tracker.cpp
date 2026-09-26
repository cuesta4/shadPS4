// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <span>

#include <boost/container/small_vector.hpp>

#include "core/memory.h"
#include "video_core/gpu_authority_tracker.h"
#include "video_core/guest_access.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {

namespace {

constexpr VAddr ReadWatchPageMask = 4095;
constexpr u64 ReadWatchPageSize = 4096;

std::pair<VAddr, size_t> GetReadWatchRange(VAddr addr, size_t size) {
    const VAddr begin = addr & ~ReadWatchPageMask;
    const VAddr end = (addr + size + ReadWatchPageMask) & ~ReadWatchPageMask;
    return {begin, end - begin};
}

constexpr bool HasGpuAuthority(GpuAuthorityState state) noexcept {
    return state == GpuAuthorityState::GpuAuthoritative ||
           state == GpuAuthorityState::Materializing;
}

/// After the producer completed, materializing guest RAM no longer waits for the GPU. A shadow
/// that more command buffers than this keep reading is materialized once instead of being
/// copied again for every consumer.
constexpr u32 MaxGpuServesAfterReady = 16;

/// Ticks after its shadow completed before an authority nothing superseded counts as stale.
/// A frame submits a dozen command buffers, and a render target produced again next frame
/// supersedes its old authority well before this.
constexpr u64 RetireAfterTicks = 48;
/// Above this many live authorities the oldest completed ones are retired regardless of age.
constexpr size_t MaxLiveAuthorities = 64;
/// Bytes materialized per retirement pass beyond the first authority.
constexpr u64 RetireBudgetBytes = 4ULL << 20;

/// A range the guest CPU read keeps the eager readback for this many ticks after the read,
/// several seconds of submits. The next read after that costs one wait again.
constexpr u64 CpuConsumerDecayTicks = 8192;
/// Faults on bytes next to authorities within HotPageWindowTicks after which the ranges of
/// the page keep the eager readback, which leaves the page unprotected.
constexpr u32 HotPageFaults = 64;
constexpr u64 HotPageWindowTicks = 256;
constexpr u64 HotPageDecayTicks = 8192;
constexpr size_t MaxTrackedPages = 4096;

[[nodiscard]] constexpr u64 ClassKey(VAddr begin, u64 size) noexcept {
    return begin ^ (size * 0x9E3779B97F4A7C15ULL);
}

[[nodiscard]] inline VAddr DownloadEnd(const GpuAuthorityEntry& entry) noexcept {
    return entry.guest_begin + entry.download_size;
}

/// Entry mutex held. True when a byte of [begin, end) is downloaded by the entry and was not
/// written since by something newer.
[[nodiscard]] bool OwnsBytesLocked(const GpuAuthorityEntry& entry, VAddr begin, VAddr end) {
    VAddr cursor = std::max(begin, entry.guest_begin);
    const VAddr limit = std::min(end, DownloadEnd(entry));
    if (cursor >= limit) {
        return false;
    }
    bool advanced = true;
    while (cursor < limit && advanced) {
        advanced = false;
        for (const auto& [mask_begin, mask_end] : entry.masked) {
            if (mask_begin <= cursor && cursor < mask_end) {
                cursor = mask_end;
                advanced = true;
            }
        }
    }
    return cursor < limit;
}

/// Entry mutex held. True when no byte of [begin, end) is masked.
[[nodiscard]] bool UnmaskedLocked(const GpuAuthorityEntry& entry, VAddr begin, VAddr end) {
    return std::ranges::none_of(entry.masked, [&](const GuestRange& range) {
        return std::max(range.first, begin) < std::min(range.second, end);
    });
}

/// Entry mutex held.
void AddMaskLocked(GpuAuthorityEntry& entry, VAddr begin, VAddr end) {
    begin = std::max(begin, entry.guest_begin);
    end = std::min(end, DownloadEnd(entry));
    if (begin >= end) {
        return;
    }
    for (auto it = entry.masked.begin(); it != entry.masked.end();) {
        if (it->first <= end && begin <= it->second) {
            begin = std::min(begin, it->first);
            end = std::max(end, it->second);
            it = entry.masked.erase(it);
        } else {
            ++it;
        }
    }
    entry.masked.emplace_back(begin, end);
}

/// Entry mutex held.
void DropLocked(GpuAuthorityEntry& entry, GpuAuthorityState state) {
    entry.state = state;
    entry.direct = false;
    if (entry.shadow) {
        entry.shadow->Release();
        entry.shadow.reset();
    }
    entry.cv->notify_all();
}

thread_local u64 tls_active_materializing_authority_seq{0};
thread_local u64 tls_backing_write_bound{0};

class ScopedAuthorityMaterialization {
public:
    explicit ScopedAuthorityMaterialization(u64 authority_seq) noexcept
        : prev_seq(tls_active_materializing_authority_seq) {
        tls_active_materializing_authority_seq = authority_seq;
    }
    ~ScopedAuthorityMaterialization() noexcept {
        tls_active_materializing_authority_seq = prev_seq;
    }
    ScopedAuthorityMaterialization(const ScopedAuthorityMaterialization&) = delete;
    ScopedAuthorityMaterialization& operator=(const ScopedAuthorityMaterialization&) = delete;

private:
    u64 prev_seq{0};
};

[[nodiscard]] bool IsCurrentThreadMaterializing(u64 authority_seq) noexcept {
    return tls_active_materializing_authority_seq != 0 &&
           tls_active_materializing_authority_seq == authority_seq;
}

} // namespace

ScopedBackingWriteBound::ScopedBackingWriteBound(u64 bound) noexcept
    : prev_bound{tls_backing_write_bound} {
    tls_backing_write_bound = bound;
}

ScopedBackingWriteBound::~ScopedBackingWriteBound() noexcept {
    tls_backing_write_bound = prev_bound;
}

GpuAuthorityTracker& GpuAuthorityTracker::Instance() noexcept {
    static GpuAuthorityTracker instance;
    return instance;
}

void GpuAuthorityTracker::SetRasterizer(Vulkan::Rasterizer* rasterizer_) noexcept {
    std::scoped_lock lock{tracker_mutex};
    rasterizer = rasterizer_;
}

u64 GpuAuthorityTracker::CurrentTick() const noexcept {
    return rasterizer ? rasterizer->CurrentTick() : 0;
}

void GpuAuthorityTracker::PruneLocked() {
    std::erase_if(authorities, [](const auto& entry) {
        std::scoped_lock entry_lock{*entry->entry_mutex};
        return !HasGpuAuthority(entry->state);
    });
    live_count.store(authorities.size(), std::memory_order_release);
}

void GpuAuthorityTracker::RegisterAuthority(GpuAuthorityEntry&& entry) {
    const VAddr begin = entry.guest_begin;
    const VAddr end = DownloadEnd(entry);
    entry.guest_end = end;
    boost::container::small_vector<std::pair<VAddr, size_t>, 4> dropped_ranges;
    std::scoped_lock lock{tracker_mutex};
    for (auto& old_entry : authorities) {
        if (std::max(old_entry->guest_begin, begin) >= std::min(DownloadEnd(*old_entry), end)) {
            continue;
        }
        std::scoped_lock entry_lock{*old_entry->entry_mutex};
        if (!HasGpuAuthority(old_entry->state)) {
            continue;
        }
        // The new bytes are newer than the old ones wherever they overlap. A materialization
        // already running reads the mask when it writes, so it never lands on top of them.
        AddMaskLocked(*old_entry, begin, end);
        if (old_entry->state != GpuAuthorityState::GpuAuthoritative ||
            OwnsBytesLocked(*old_entry, old_entry->guest_begin, DownloadEnd(*old_entry))) {
            continue;
        }
        Common::PerformanceTelemetry::RecordAuthoritySupersede(
            Common::PerformanceTelemetry::AuthoritySupersedeSample{
                .old_authority_seq = old_entry->authority_seq,
                .new_authority_seq = entry.authority_seq,
                .overlap_begin = old_entry->guest_begin,
                .overlap_size = old_entry->download_size,
                .old_resource_id = old_entry->resource_id,
                .old_resource_version = old_entry->resource_version,
                .new_resource_id = entry.resource_id,
                .new_resource_version = entry.resource_version,
                .old_host_current = static_cast<u8>(old_entry->host_current),
            });
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::AuthoritySupersededWithoutHostUse);
        if (old_entry->direct) {
            Common::PerformanceTelemetry::Add(
                Common::PerformanceTelemetry::Counter::DirectAuthorityUnconsumed);
        }
        Common::PerformanceTelemetry::RecordCandidateTerminal(
            Common::PerformanceTelemetry::CandidateTerminalSample{
                .candidate_id = old_entry->candidate_seq,
                .resource_uid = old_entry->resource_id,
                .resource_epoch = old_entry->resource_version,
                .terminal_timestamp_ns = Common::PerformanceTelemetry::Timestamp(),
                .bytes_preserved = old_entry->download_size,
                .reason = Common::PerformanceTelemetry::CandidateTerminalReason::Superseded,
                .had_cpu_consumer = static_cast<u8>(old_entry->host_current),
                .had_gpu_consumer = static_cast<u8>(old_entry->gpu_consumed),
            });
        DropLocked(*old_entry, GpuAuthorityState::Superseded);
        dropped_ranges.emplace_back(old_entry->guest_begin, old_entry->download_size);
    }
    PruneLocked();
    authorities.push_back(std::make_shared<GpuAuthorityEntry>(std::move(entry)));
    live_count.store(authorities.size(), std::memory_order_release);

    const auto [watch_addr, watch_size] = GetReadWatchRange(begin, end - begin);
    boost::container::small_vector<std::pair<VAddr, size_t>, 2> ranges_to_arm;
    authority_read_watch_ranges.ForEachNotInRange(
        watch_addr, watch_size,
        [&](VAddr addr, size_t size) { ranges_to_arm.emplace_back(addr, size); });
    authority_read_watch_ranges.Add(watch_addr, watch_size);
    if (rasterizer) {
        for (const auto [addr, size] : ranges_to_arm) {
            rasterizer->ArmSemanticReadWatch(addr, size);
        }
    }
    for (const auto [addr, size] : dropped_ranges) {
        RefreshAuthorityReadWatches(addr, size);
    }
}

void GpuAuthorityTracker::RetireStaleAuthorities() {
    if (!HasAuthorities() || rasterizer == nullptr) {
        return;
    }
    const u64 known_tick = rasterizer->KnownGpuTick();
    const u64 current_tick = rasterizer->CurrentTick();
    boost::container::small_vector<std::pair<VAddr, u32>, 8> stale;
    boost::container::small_vector<std::shared_ptr<GpuAuthorityEntry>, 4> to_preserve;
    {
        std::scoped_lock lock{tracker_mutex};
        const bool crowded = authorities.size() > MaxLiveAuthorities;
        if (Common::PerformanceTelemetry::Enabled()) {
            Common::PerformanceTelemetry::ObserveMaxEnabled(
                Common::PerformanceTelemetry::Counter::AuthorityLiveMax, authorities.size());
        }
        u64 bytes = 0;
        size_t excess = crowded ? authorities.size() - MaxLiveAuthorities : 0;
        // Oldest first.
        for (const auto& entry : authorities) {
            std::scoped_lock entry_lock{*entry->entry_mutex};
            if (entry->state != GpuAuthorityState::GpuAuthoritative) {
                continue;
            }
            if (entry->direct) {
                // A direct authority pins nothing; it only gets a copy when the tracker holds
                // too many of them. The copy completes and retires on a later pass.
                if (excess != 0) {
                    --excess;
                    to_preserve.push_back(entry);
                }
                continue;
            }
            if (!entry->shadow) {
                continue;
            }
            const u64 ready_tick = entry->shadow->ReadyTick();
            if (known_tick < ready_tick ||
                (!crowded && current_tick < ready_tick + RetireAfterTicks)) {
                continue;
            }
            if (!stale.empty() && bytes + entry->download_size > RetireBudgetBytes) {
                break;
            }
            bytes += entry->download_size;
            stale.emplace_back(entry->guest_begin, entry->download_size);
        }
    }
    for (const auto& entry : to_preserve) {
        EnsureShadow(entry);
    }
    for (const auto [addr, size] : stale) {
        // Materializing a range also materializes authorities overlapping it; skip ranges where
        // one of them still waits for the GPU.
        const auto overlaps = FindOverlaps(addr, size);
        const bool all_ready = std::ranges::all_of(overlaps, [&](const auto& entry) {
            std::scoped_lock entry_lock{*entry->entry_mutex};
            return entry->state == GpuAuthorityState::GpuAuthoritative && entry->shadow &&
                   !entry->direct && known_tick >= entry->shadow->ReadyTick();
        });
        if (overlaps.empty() || !all_ready) {
            continue;
        }
        ResolveForRamRead(addr, size,
                          Common::PerformanceTelemetry::GuestSourceConsumePath::BufferUpload,
                          Common::PerformanceTelemetry::ResourceType::Buffer, 0);
        if (Common::PerformanceTelemetry::Enabled()) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::AuthorityRetirements, 1);
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::AuthorityRetiredBytes, size);
        }
    }
}

std::vector<std::shared_ptr<GpuAuthorityEntry>> GpuAuthorityTracker::FindOverlaps(
    VAddr addr, size_t size) const {
    std::vector<std::shared_ptr<GpuAuthorityEntry>> result;
    if (!HasAuthorities() || size == 0) {
        return result;
    }
    const VAddr query_end = addr + size;
    std::scoped_lock lock{tracker_mutex};
    for (const auto& entry : authorities) {
        // The range of an entry never changes after registration.
        if (std::max(entry->guest_begin, addr) >= std::min(DownloadEnd(*entry), query_end)) {
            continue;
        }
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (HasGpuAuthority(entry->state)) {
            result.push_back(entry);
        }
    }
    return result;
}

std::shared_ptr<GpuAuthorityEntry> GpuAuthorityTracker::FindBySeq(u64 authority_seq) const {
    if (!HasAuthorities() || authority_seq == 0) {
        return nullptr;
    }
    std::scoped_lock lock{tracker_mutex};
    for (const auto& entry : authorities) {
        if (entry->authority_seq == authority_seq) {
            return entry;
        }
    }
    return nullptr;
}

std::shared_ptr<GpuAuthorityEntry> GpuAuthorityTracker::GetAuthorityForImage(
    u64 image_uid, u64 version) const {
    if (!HasAuthorities()) {
        return nullptr;
    }
    std::scoped_lock lock{tracker_mutex};
    for (auto it = authorities.rbegin(); it != authorities.rend(); ++it) {
        const auto& entry = *it;
        if (entry->image_uid != image_uid || entry->resource_version != version) {
            continue;
        }
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (HasGpuAuthority(entry->state)) {
            return entry;
        }
    }
    return nullptr;
}

std::shared_ptr<GpuAuthorityEntry> GpuAuthorityTracker::GetAuthorityForRange(
    VAddr addr, size_t size) const {
    if (!HasAuthorities() || size == 0) {
        return nullptr;
    }
    std::scoped_lock lock{tracker_mutex};
    for (auto it = authorities.rbegin(); it != authorities.rend(); ++it) {
        const auto& entry = *it;
        if (addr < entry->guest_begin || size > entry->download_size ||
            addr - entry->guest_begin > entry->download_size - size) {
            continue;
        }
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (entry->state != GpuAuthorityState::GpuAuthoritative ||
            !UnmaskedLocked(*entry, addr, addr + size)) {
            continue;
        }
        return entry;
    }
    return nullptr;
}

bool GpuAuthorityTracker::IsGpuServableLocked(const GpuAuthorityEntry& entry) const {
    if (entry.state != GpuAuthorityState::GpuAuthoritative || entry.direct || !entry.shadow) {
        return false;
    }
    auto& shadow = *entry.shadow;
    {
        std::scoped_lock shadow_lock{shadow.data_mutex};
        if (shadow.IsReleased() || shadow.owned_data || shadow.data == nullptr ||
            shadow.guest_addr != entry.guest_begin) {
            return false;
        }
    }
    if (entry.gpu_serves >= MaxGpuServesAfterReady && rasterizer &&
        rasterizer->KnownGpuTick() >= shadow.ReadyTick()) {
        return false;
    }
    return true;
}

std::shared_ptr<GpuAuthorityShadow> GpuAuthorityTracker::AcquireGpuShadowForImage(
    VAddr addr, size_t size, u32 offset_alignment) {
    if (!HasAuthorities() || size == 0) {
        return nullptr;
    }
    const VAddr end = addr + size;
    for (u32 pass = 0; pass < 2; ++pass) {
        std::shared_ptr<GpuAuthorityEntry> owner;
        {
            std::scoped_lock lock{tracker_mutex};
            for (auto it = authorities.rbegin(); it != authorities.rend(); ++it) {
                const auto& entry = *it;
                if (std::max(entry->guest_begin, addr) >= std::min(DownloadEnd(*entry), end)) {
                    continue;
                }
                std::scoped_lock entry_lock{*entry->entry_mutex};
                if (!HasGpuAuthority(entry->state) || !OwnsBytesLocked(*entry, addr, end)) {
                    continue;
                }
                // The newest authority touching the range must hold all of it; anything else
                // mixes sources and goes through guest RAM.
                if (entry->state == GpuAuthorityState::GpuAuthoritative &&
                    addr >= entry->guest_begin && end <= DownloadEnd(*entry) &&
                    UnmaskedLocked(*entry, addr, end)) {
                    owner = entry;
                }
                break;
            }
        }
        if (!owner) {
            return nullptr;
        }
        {
            std::scoped_lock entry_lock{*owner->entry_mutex};
            if (owner->direct) {
                if (pass != 0) {
                    return nullptr;
                }
            } else {
                if (owner->state != GpuAuthorityState::GpuAuthoritative || !owner->shadow) {
                    return nullptr;
                }
                auto& shadow = owner->shadow;
                std::scoped_lock shadow_lock{shadow->data_mutex};
                if (shadow->IsReleased() || shadow->owned_data) {
                    return nullptr;
                }
                const u64 offset = shadow->buffer_offset + (addr - shadow->guest_addr);
                if (offset_alignment > 1 && offset % offset_alignment != 0) {
                    return nullptr;
                }
                shadow->ExtendLifetime(CurrentTick());
                owner->gpu_consumed = true;
                return shadow;
            }
        }
        // The bytes are still in the image; record the copy now. Command processor thread.
        EnsureShadow(owner);
    }
    return nullptr;
}

void GpuAuthorityTracker::EnsureShadow(const std::shared_ptr<GpuAuthorityEntry>& entry) {
    {
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (!entry->direct || !HasGpuAuthority(entry->state)) {
            return;
        }
    }
    if (rasterizer == nullptr) {
        return;
    }
    rasterizer->GetTextureCache().PreserveDirectAuthority(entry, true);
}

void GpuAuthorityTracker::AttachShadow(const std::shared_ptr<GpuAuthorityEntry>& entry,
                                       std::shared_ptr<GpuAuthorityShadow> shadow) {
    std::scoped_lock entry_lock{*entry->entry_mutex};
    if (!entry->direct || !HasGpuAuthority(entry->state)) {
        if (shadow) {
            shadow->Release();
        }
        return;
    }
    entry->shadow = std::move(shadow);
    entry->direct = false;
    Common::PerformanceTelemetry::Add(
        Common::PerformanceTelemetry::Counter::DirectAuthorityCopies);
}

void GpuAuthorityTracker::DropAuthority(const std::shared_ptr<GpuAuthorityEntry>& entry) {
    {
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (!HasGpuAuthority(entry->state)) {
            return;
        }
        if (entry->state == GpuAuthorityState::GpuAuthoritative) {
            DropLocked(*entry, GpuAuthorityState::Failed);
        } else {
            // A materialization in progress finishes on its own.
            return;
        }
    }
    RefreshAuthorityReadWatches(entry->guest_begin, entry->download_size);
    std::scoped_lock lock{tracker_mutex};
    PruneLocked();
}

bool GpuAuthorityTracker::CollectGpuShadowPieces(VAddr addr, size_t size, GpuShadowPieces& pieces) {
    pieces.clear();
    if (!HasAuthorities() || size == 0) {
        return false;
    }
    const VAddr end = addr + size;
    boost::container::small_vector<std::shared_ptr<GpuAuthorityEntry>, 4> direct_entries;
    {
        std::scoped_lock lock{tracker_mutex};
        const auto [watch_addr, watch_size] = GetReadWatchRange(addr, size);
        if (!authority_read_watch_ranges.Intersects(watch_addr, watch_size)) {
            // Nothing here belongs to an authority; the range reads like any other.
            return false;
        }
        for (const auto& entry : authorities) {
            if (std::max(entry->guest_begin, addr) >= std::min(DownloadEnd(*entry), end)) {
                continue;
            }
            std::scoped_lock entry_lock{*entry->entry_mutex};
            if (entry->direct && entry->state == GpuAuthorityState::GpuAuthoritative &&
                OwnsBytesLocked(*entry, addr, end)) {
                direct_entries.push_back(entry);
            }
        }
    }
    // The resolver runs on the command processor, which can record the copies right away.
    for (const auto& entry : direct_entries) {
        EnsureShadow(entry);
    }

    std::scoped_lock lock{tracker_mutex};
    for (const auto& entry : authorities) {
        if (std::max(entry->guest_begin, addr) >= std::min(DownloadEnd(*entry), end)) {
            continue;
        }
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (!HasGpuAuthority(entry->state) || !OwnsBytesLocked(*entry, addr, end)) {
            continue;
        }
        if (!entry->masked.empty() || !IsGpuServableLocked(*entry)) {
            pieces.clear();
            return false;
        }
        // Materializing writes only the shadow bytes; the rest of the entry range reads the same
        // from guest RAM before and after.
        const auto& shadow = entry->shadow;
        const VAddr piece_begin = std::max(shadow->guest_addr, addr);
        const VAddr piece_end = std::min<VAddr>(shadow->guest_addr + shadow->size, end);
        if (piece_begin >= piece_end) {
            continue;
        }
        pieces.push_back(GpuShadowPiece{
            .addr = piece_begin,
            .size = piece_end - piece_begin,
            .buffer_offset = shadow->buffer_offset + (piece_begin - shadow->guest_addr),
            .entry = entry,
            .shadow = shadow,
        });
    }
    std::ranges::sort(pieces, {}, &GpuShadowPiece::addr);
    for (size_t i = 1; i < pieces.size(); ++i) {
        if (pieces[i].addr < pieces[i - 1].addr + pieces[i - 1].size) {
            pieces.clear();
            return false;
        }
    }
    return true;
}

void GpuAuthorityTracker::CommitGpuShadowPieces(const GpuShadowPieces& pieces, u64 consumer_tick) {
    for (const auto& piece : pieces) {
        piece.shadow->ExtendLifetime(consumer_tick);
        std::scoped_lock entry_lock{*piece.entry->entry_mutex};
        piece.entry->gpu_consumed = true;
        if (piece.entry->last_gpu_serve_tick != consumer_tick) {
            piece.entry->last_gpu_serve_tick = consumer_tick;
            ++piece.entry->gpu_serves;
        }
    }
}

bool GpuAuthorityTracker::IsBackingReadable(VAddr addr, size_t size) const {
    if (rasterizer == nullptr || size == 0) {
        return false;
    }
    const auto& page_manager = rasterizer->GetPageManager();
    const VAddr first = addr & ~ReadWatchPageMask;
    const VAddr last = (addr + size - 1) & ~ReadWatchPageMask;
    std::scoped_lock lock{tracker_mutex};
    for (VAddr page = first;; page += ReadWatchPageSize) {
        const u32 watchers = page_manager.ReadWatchCount(page);
        if (watchers != 0) {
            // A read watch of anything else guards bytes RAM does not hold yet.
            const bool authority_page = authority_read_watch_ranges.Contains(page, 1);
            if (!authority_page || watchers != 1) {
                return false;
            }
        }
        if (page == last) {
            break;
        }
    }
    return true;
}

bool GpuAuthorityTracker::TouchesAuthorityPages(VAddr addr, size_t size) const {
    if (!HasAuthorities() || size == 0) {
        return false;
    }
    const auto [watch_addr, watch_size] = GetReadWatchRange(addr, size);
    std::scoped_lock lock{tracker_mutex};
    return authority_read_watch_ranges.Intersects(watch_addr, watch_size);
}

bool GpuAuthorityTracker::ReadGuestMemory(VAddr addr, u8* destination, size_t size) {
    if (!HasAuthorities() || size == 0 || rasterizer == nullptr) {
        return false;
    }
    {
        const auto [watch_addr, watch_size] = GetReadWatchRange(addr, size);
        std::scoped_lock lock{tracker_mutex};
        if (!authority_read_watch_ranges.Intersects(watch_addr, watch_size)) {
            return false;
        }
    }
    auto* memory = Core::Memory::Instance();
    if (!IsBackingReadable(addr, size) || !memory->IsBackedRange(addr, size)) {
        return false;
    }
    return memory->ReadBacking(addr, destination, size);
}

bool GpuAuthorityTracker::MaterializeEntry(const std::shared_ptr<GpuAuthorityEntry>& entry,
                                           std::unique_lock<std::mutex>& entry_lk) {
    if (entry->direct) {
        // The bytes are still in the image; only the command processor records the copy.
        entry_lk.unlock();
        if (rasterizer) {
            if (rasterizer->IsGpuThread()) {
                EnsureShadow(entry);
            } else {
                rasterizer->PreserveAuthorityForHost(entry);
            }
        }
        entry_lk.lock();
        if (entry->state == GpuAuthorityState::Materializing) {
            entry->cv->wait(entry_lk,
                            [&] { return entry->state != GpuAuthorityState::Materializing; });
        }
        if (entry->state != GpuAuthorityState::GpuAuthoritative) {
            return entry->state == GpuAuthorityState::HostCurrent;
        }
        if (entry->direct || !entry->shadow) {
            // The image went away without its bytes; nothing can make RAM current anymore.
            DropLocked(*entry, GpuAuthorityState::Failed);
            return false;
        }
    }
    if (!entry->shadow) {
        DropLocked(*entry, GpuAuthorityState::Failed);
        return false;
    }
    entry->state = GpuAuthorityState::Materializing;
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    const u64 materialize_start = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    const u64 auth_seq = entry->authority_seq;
    const VAddr begin = entry->guest_begin;
    const u32 size = entry->download_size;
    auto shadow = entry->shadow;

    // The copy into the shadow is recorded; waiting for its tick never needs a synchronous
    // command on the command processor.
    entry_lk.unlock();
    if (rasterizer) {
        rasterizer->GetTextureCache().WaitGpuAuthorityShadow(shadow);
    }
    entry_lk.lock();

    bool success = false;
    if (entry->state == GpuAuthorityState::Materializing) {
        ScopedAuthorityMaterialization scoped_mat{auth_seq};
        s8 val_equal = -1;
        if (rasterizer) {
            success = rasterizer->GetTextureCache().MaterializeGpuAuthority(
                shadow, begin, size,
                std::span<const GuestRange>{entry->masked.data(), entry->masked.size()},
                &val_equal);
        }
        entry->state = success ? GpuAuthorityState::HostCurrent : GpuAuthorityState::Failed;
        entry->host_current = success;
    } else {
        success = entry->state == GpuAuthorityState::HostCurrent;
    }
    if (entry->shadow) {
        entry->shadow->Release();
        entry->shadow.reset();
    }
    entry->cv->notify_all();
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::AuthorityMaterializations, 1);
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::AuthorityMaterializeNs,
            Common::PerformanceTelemetry::Timestamp() - materialize_start);
        Common::PerformanceTelemetry::AddEnabled(
            success ? Common::PerformanceTelemetry::Counter::MaterializeSuccess
                    : Common::PerformanceTelemetry::Counter::MaterializeFailure,
            1);
    }
    return success;
}

bool GpuAuthorityTracker::ResolveForRamRead(
    VAddr addr, size_t size, Common::PerformanceTelemetry::GuestSourceConsumePath path,
    Common::PerformanceTelemetry::ResourceType dest_kind, u64 dest_res_id, bool keep_gpu_servable) {
    if (!HasAuthorities() || size == 0) {
        return true;
    }
    const VAddr end = addr + size;
    bool all_succeeded = true;
    boost::container::small_vector<std::pair<VAddr, size_t>, 4> touched_ranges;
    // An authority registered or dropped meanwhile changes who owns the bytes; look again.
    for (u32 pass = 0; pass < 4; ++pass) {
        const auto overlaps = FindOverlaps(addr, size);
        if (overlaps.empty()) {
            break;
        }
        bool retry = false;
        all_succeeded = true;
        for (const auto& entry : overlaps) {
            std::unique_lock entry_lk{*entry->entry_mutex};
            if (!OwnsBytesLocked(*entry, addr, end) ||
                IsCurrentThreadMaterializing(entry->authority_seq)) {
                continue;
            }
            touched_ranges.emplace_back(entry->guest_begin, entry->download_size);
            if (Common::PerformanceTelemetry::HeavyEnabled()) {
                Common::PerformanceTelemetry::RecordAuthorityRamDemand(
                    Common::PerformanceTelemetry::AuthorityRamDemandSample{
                        .ram_demand_seq = Common::PerformanceTelemetry::NextRamDemandSeq(),
                        .timestamp_ns = Common::PerformanceTelemetry::Timestamp(),
                        .authority_seq = entry->authority_seq,
                        .resource_id = entry->resource_id,
                        .resource_version = entry->resource_version,
                        .authority_begin = entry->guest_begin,
                        .authority_end = entry->guest_end,
                        .request_addr = addr,
                        .request_size = size,
                        .path = path,
                        .destination_kind = dest_kind,
                        .destination_resource_id = dest_res_id,
                        .authority_state_before = static_cast<u8>(entry->state),
                    });
                Common::PerformanceTelemetry::Add(
                    Common::PerformanceTelemetry::Counter::RamDemandEvents);
            }
            switch (entry->state) {
            case GpuAuthorityState::HostCurrent:
                break;
            case GpuAuthorityState::Superseded:
            case GpuAuthorityState::Failed:
                retry = true;
                break;
            case GpuAuthorityState::Materializing:
                entry->cv->wait(entry_lk, [&] {
                    return entry->state != GpuAuthorityState::Materializing;
                });
                if (entry->state != GpuAuthorityState::HostCurrent) {
                    retry = true;
                }
                break;
            case GpuAuthorityState::GpuAuthoritative:
                if (keep_gpu_servable) {
                    if (entry->direct && rasterizer && rasterizer->IsGpuThread()) {
                        // Served from a copy recorded now, without waiting for it.
                        entry_lk.unlock();
                        EnsureShadow(entry);
                        entry_lk.lock();
                        if (entry->state != GpuAuthorityState::GpuAuthoritative) {
                            retry = true;
                            break;
                        }
                    }
                    if (IsGpuServableLocked(*entry)) {
                        break;
                    }
                }
                if (!MaterializeEntry(entry, entry_lk)) {
                    all_succeeded = false;
                }
                break;
            }
        }
        if (!retry) {
            break;
        }
    }
    for (const auto [range_addr, range_size] : touched_ranges) {
        RefreshAuthorityReadWatches(range_addr, range_size);
    }
    {
        std::scoped_lock lock{tracker_mutex};
        PruneLocked();
    }
    return all_succeeded;
}

void GpuAuthorityTracker::RefreshAuthorityReadWatches(VAddr addr, size_t size) {
    const auto [watch_addr, watch_size] = GetReadWatchRange(addr, size);
    boost::container::small_vector<std::pair<VAddr, size_t>, 4> ranges_to_disarm;
    std::scoped_lock lock{tracker_mutex};
    RangeSet required_ranges;
    for (const auto& entry : authorities) {
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (!HasGpuAuthority(entry->state)) {
            continue;
        }
        const auto [required_addr, required_size] =
            GetReadWatchRange(entry->guest_begin, entry->download_size);
        required_ranges.Add(required_addr, required_size);
    }
    authority_read_watch_ranges.ForEachInRange(
        watch_addr, watch_size, [&](VAddr range_addr, VAddr range_end) {
            required_ranges.ForEachNotInRange(
                range_addr, range_end - range_addr, [&](VAddr gap_addr, size_t gap_size) {
                    ranges_to_disarm.emplace_back(gap_addr, gap_size);
                });
        });
    for (const auto [range_addr, range_size] : ranges_to_disarm) {
        authority_read_watch_ranges.Subtract(range_addr, range_size);
        if (rasterizer) {
            rasterizer->DisarmSemanticReadWatch(range_addr, range_size);
        }
    }
}

bool GpuAuthorityTracker::HandleCpuAccess(VAddr fault_addr, void* context, bool is_write,
                                          bool& handled) {
    handled = false;
    if (!HasAuthorities() || rasterizer == nullptr) {
        return false;
    }
    const VAddr page = fault_addr & ~ReadWatchPageMask;
    {
        std::scoped_lock lock{tracker_mutex};
        if (!authority_read_watch_ranges.Contains(page, 1)) {
            return false;
        }
    }
    handled = true;
    if (FindOverlaps(page, ReadWatchPageSize).empty()) {
        // Nothing on the page belongs to an authority anymore.
        RefreshAuthorityReadWatches(page, ReadWatchPageSize);
        return false;
    }

    GuestAccess access{};
    const bool decoded = DecodeGuestAccess(context, fault_addr, access) && access.is_write == is_write;
    const VAddr begin = decoded ? access.address : page;
    const VAddr end = decoded ? access.address + access.size : page + ReadWatchPageSize;

    boost::container::small_vector<std::pair<VAddr, u64>, 2> touched;
    for (const auto& entry : FindOverlaps(begin, end - begin)) {
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (HasGpuAuthority(entry->state) && OwnsBytesLocked(*entry, begin, end)) {
            touched.emplace_back(entry->guest_begin, entry->download_size);
        }
    }

    const auto emulate = [&] {
        if (!decoded || !access.Emulatable() || !IsBackingReadable(access.address, access.size) ||
            !Core::Memory::Instance()->IsBackedRange(access.address, access.size)) {
            return false;
        }
        if (access.is_write) {
            // Same bookkeeping as a write that faults on a write-watched page.
            rasterizer->InvalidateMemory(access.address, access.size);
        }
        return EmulateGuestAccess(context, access);
    };

    // Accesses of the command processor are the emulator's own, not a guest consumer.
    const bool guest_access = !rasterizer->IsGpuThread();
    if (touched.empty()) {
        // The instruction only touches bytes next to an authority, which RAM holds already.
        if (guest_access) {
            NoteFalseSharing(page);
        }
        if (emulate()) {
            Common::PerformanceTelemetry::Add(
                Common::PerformanceTelemetry::Counter::FaultFalseSharingEmulated);
            return true;
        }
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::FaultFalseSharingUnemulated);
        ResolveForRamRead(page, ReadWatchPageSize,
                          Common::PerformanceTelemetry::GuestSourceConsumePath::BufferUpload);
        return false;
    }

    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::FaultAuthorityBytes);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::AuthorityCpuRead);
    for (const auto [range_begin, range_size] : touched) {
        if (!guest_access) {
            break;
        }
        if (is_write) {
            NoteFalseSharing(page);
        } else {
            NoteCpuConsumer(range_begin, range_size);
        }
    }
    // Only the authorities the instruction touches wait for their producer.
    ResolveForRamRead(begin, end - begin,
                      Common::PerformanceTelemetry::GuestSourceConsumePath::BufferUpload);
    if (rasterizer->GetPageManager().ReadWatchCount(page) != 0 && emulate()) {
        return true;
    }
    ResolveForRamRead(page, ReadWatchPageSize,
                      Common::PerformanceTelemetry::GuestSourceConsumePath::BufferUpload);
    return false;
}

void GpuAuthorityTracker::HandleUnmap(VAddr addr, size_t size) {
    if (size == 0) {
        return;
    }
    if (HasQueuedWork()) {
        // Labels in memory that goes away are not written anymore; their interrupts still are.
        std::scoped_lock lock{timeline_mutex};
        for (auto& item : timeline) {
            if (!item.is_writeback && item.label_size != 0 && item.label_addr >= addr &&
                item.label_addr - addr < size) {
                item.write_label = false;
            }
        }
    }
    const auto overlaps = FindOverlaps(addr, size);
    for (const auto& entry : overlaps) {
        std::unique_lock entry_lk{*entry->entry_mutex};
        if (!HasGpuAuthority(entry->state)) {
            continue;
        }
        AddMaskLocked(*entry, addr, addr + size);
        if (OwnsBytesLocked(*entry, entry->guest_begin, DownloadEnd(*entry))) {
            continue;
        }
        if (entry->state == GpuAuthorityState::GpuAuthoritative) {
            DropLocked(*entry, GpuAuthorityState::Superseded);
        }
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::AuthorityDestroyedWithoutHostUse);
        Common::PerformanceTelemetry::RecordCandidateTerminal(
            Common::PerformanceTelemetry::CandidateTerminalSample{
                .candidate_id = entry->candidate_seq,
                .resource_uid = entry->resource_id,
                .resource_epoch = entry->resource_version,
                .terminal_timestamp_ns = Common::PerformanceTelemetry::Timestamp(),
                .bytes_preserved = entry->download_size,
                .reason = Common::PerformanceTelemetry::CandidateTerminalReason::Unmapped,
                .had_cpu_consumer = static_cast<u8>(entry->host_current),
                .had_gpu_consumer = static_cast<u8>(entry->gpu_consumed),
            });
    }
    for (const auto& entry : overlaps) {
        RefreshAuthorityReadWatches(entry->guest_begin, entry->download_size);
    }
    std::scoped_lock lock{tracker_mutex};
    PruneLocked();
}

void GpuAuthorityTracker::HandleBackingWrite(VAddr addr, size_t size) {
    if (!HasAuthorities() || size == 0 || tls_active_materializing_authority_seq != 0) {
        return;
    }
    const auto overlaps = FindOverlaps(addr, size);
    if (overlaps.empty()) {
        return;
    }
    const u64 bound = tls_backing_write_bound;
    bool dropped = false;
    for (const auto& entry : overlaps) {
        if (bound != 0 && entry->authority_seq >= bound) {
            // Produced after the written bytes.
            continue;
        }
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (!HasGpuAuthority(entry->state) || !OwnsBytesLocked(*entry, addr, addr + size)) {
            continue;
        }
        // The written bytes are newer than the ones the authority downloaded.
        AddMaskLocked(*entry, addr, addr + size);
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::AuthorityMaskedWrites);
        if (entry->state == GpuAuthorityState::GpuAuthoritative &&
            !OwnsBytesLocked(*entry, entry->guest_begin, DownloadEnd(*entry))) {
            DropLocked(*entry, GpuAuthorityState::Superseded);
            dropped = true;
        }
    }
    if (dropped) {
        for (const auto& entry : overlaps) {
            RefreshAuthorityReadWatches(entry->guest_begin, entry->download_size);
        }
        std::scoped_lock lock{tracker_mutex};
        PruneLocked();
    }
}

bool GpuAuthorityTracker::PrefersCpuCommit(VAddr begin, u64 size) {
    if (classified_count.load(std::memory_order_acquire) == 0) {
        return false;
    }
    const u64 now = CurrentTick();
    std::scoped_lock lock{class_mutex};
    if (const auto it = cpu_consumers.find(ClassKey(begin, size)); it != cpu_consumers.end()) {
        if (now - it->second <= CpuConsumerDecayTicks) {
            return true;
        }
        cpu_consumers.erase(it);
        classified_count.fetch_sub(1, std::memory_order_release);
    }
    if (hot_pages.empty() || size == 0) {
        return false;
    }
    const VAddr first = begin & ~ReadWatchPageMask;
    const VAddr last = (begin + size - 1) & ~ReadWatchPageMask;
    const auto is_hot = [&](VAddr page) {
        const auto it = hot_pages.find(page);
        return it != hot_pages.end() && it->second.hot_tick != 0 &&
               now - it->second.hot_tick <= HotPageDecayTicks;
    };
    if ((last - first) / ReadWatchPageSize + 1 > hot_pages.size()) {
        for (const auto& [page, state] : hot_pages) {
            if (page >= first && page <= last && is_hot(page)) {
                return true;
            }
        }
        return false;
    }
    for (VAddr page = first;; page += ReadWatchPageSize) {
        if (is_hot(page)) {
            return true;
        }
        if (page == last) {
            break;
        }
    }
    return false;
}

void GpuAuthorityTracker::NoteCpuConsumer(VAddr begin, u64 size) {
    const u64 now = CurrentTick();
    std::scoped_lock lock{class_mutex};
    const auto [it, inserted] = cpu_consumers.try_emplace(ClassKey(begin, size), now);
    if (inserted) {
        classified_count.fetch_add(1, std::memory_order_release);
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::ClassifiedCpuConsumer);
    } else {
        it.value() = now;
    }
}

void GpuAuthorityTracker::NoteFalseSharing(VAddr page) {
    const u64 now = CurrentTick();
    std::scoped_lock lock{class_mutex};
    if (hot_pages.size() >= MaxTrackedPages) {
        for (auto it = hot_pages.begin(); it != hot_pages.end();) {
            if (it->second.hot_tick == 0 || now - it->second.hot_tick > HotPageDecayTicks) {
                it = hot_pages.erase(it);
                classified_count.fetch_sub(1, std::memory_order_release);
            } else {
                ++it;
            }
        }
    }
    const auto [it, inserted] = hot_pages.try_emplace(page);
    if (inserted) {
        classified_count.fetch_add(1, std::memory_order_release);
    }
    HotPage& state = it.value();
    if (now - state.window_tick > HotPageWindowTicks) {
        state.window_tick = now;
        state.faults = 0;
    }
    if (++state.faults >= HotPageFaults) {
        if (state.hot_tick == 0) {
            Common::PerformanceTelemetry::Add(
                Common::PerformanceTelemetry::Counter::ClassifiedHotPage);
        }
        state.hot_tick = now;
    }
}

u64 GpuAuthorityTracker::BeginEagerWriteback(bool orders_all_signals) {
    std::scoped_lock lock{timeline_mutex};
    const u64 seq = next_timeline_seq++;
    timeline.push_back(
        TimelineItem{.seq = seq, .is_writeback = true, .orders_all = orders_all_signals});
    timeline_items.fetch_add(1, std::memory_order_release);
    if (orders_all_signals) {
        ordering_items.fetch_add(1, std::memory_order_release);
    }
    return seq;
}

void GpuAuthorityTracker::CompleteEagerWriteback(u64 seq) {
    {
        std::scoped_lock lock{timeline_mutex};
        for (auto& item : timeline) {
            if (item.seq == seq) {
                if (!item.done && item.orders_all) {
                    ordering_items.fetch_sub(1, std::memory_order_release);
                }
                item.done = true;
                break;
            }
        }
    }
    PublishReady();
}

void GpuAuthorityTracker::DrainTimelineLocked(Common::PerformanceTelemetry::Counter counter) {
    for (;;) {
        TimelineItem item;
        {
            std::scoped_lock lock{timeline_mutex};
            if (timeline.empty()) {
                return;
            }
            auto& front = timeline.front();
            if (front.is_writeback) {
                if (!front.done) {
                    return;
                }
                timeline.pop_front();
                timeline_items.fetch_sub(1, std::memory_order_release);
                continue;
            }
            item = std::move(front);
            timeline.pop_front();
            timeline_items.fetch_sub(1, std::memory_order_release);
        }
        item.publish(static_cast<bool>(item.write_label));
        Common::PerformanceTelemetry::Add(counter);
    }
}

bool GpuAuthorityTracker::PublishSignal(SignalPublication&& signal) {
    std::unique_lock publish_lock{publish_mutex, std::try_to_lock};
    if (publish_lock.owns_lock()) {
        DrainTimelineLocked(Common::PerformanceTelemetry::Counter::SignalsPublishedAtCp);
        std::unique_lock lock{timeline_mutex};
        if (timeline.empty()) {
            lock.unlock();
            signal.publish(true);
            Common::PerformanceTelemetry::Add(
                Common::PerformanceTelemetry::Counter::SignalsPublishedAtCp);
            return false;
        }
    }
    std::scoped_lock lock{timeline_mutex};
    timeline.push_back(TimelineItem{
        .seq = next_timeline_seq++,
        .label_addr = signal.label_addr,
        .label_size = signal.label_size,
        .publish = std::move(signal.publish),
    });
    timeline_items.fetch_add(1, std::memory_order_release);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::SignalsQueued);
    return true;
}

void GpuAuthorityTracker::PublishReady() {
    if (!HasQueuedWork()) {
        return;
    }
    std::scoped_lock publish_lock{publish_mutex};
    DrainTimelineLocked(Common::PerformanceTelemetry::Counter::SignalsPublishedAtCompletion);
}

bool GpuAuthorityTracker::TryPublishReady(Common::PerformanceTelemetry::Counter counter) {
    if (!HasQueuedWork()) {
        return false;
    }
    std::unique_lock publish_lock{publish_mutex, std::try_to_lock};
    if (!publish_lock.owns_lock()) {
        return false;
    }
    const u32 before = timeline_items.load(std::memory_order_acquire);
    DrainTimelineLocked(counter);
    return timeline_items.load(std::memory_order_acquire) != before;
}

void GpuAuthorityTracker::ValidateGpuConsumerBarrier(
    u64 image_uid, u64 version, u32 image_id,
    u32 old_layout, u32 new_layout,
    u64 src_stage, u64 src_access,
    u64 dst_stage, u64 dst_access,
    u64 subresource_range) {
    if (!Common::PerformanceTelemetry::HeavyEnabled()) {
        return;
    }
    auto entry = GetAuthorityForImage(image_uid, version);
    if (!entry) {
        return;
    }

    const u64 consumer_seq = Common::PerformanceTelemetry::NextConsumerSeq();
    constexpr u64 write_mask = static_cast<u64>(vk::AccessFlagBits2::eShaderWrite) |
                               static_cast<u64>(vk::AccessFlagBits2::eTransferWrite) |
                               static_cast<u64>(vk::AccessFlagBits2::eMemoryWrite);
    const u8 valid_dep = (src_access & write_mask) != 0 ? 1 : 0;

    Common::PerformanceTelemetry::RecordAuthorityBarrierValidation(
        Common::PerformanceTelemetry::AuthorityBarrierValidationSample{
            .authority_seq = entry->authority_seq,
            .consumer_seq = consumer_seq,
            .resource_id = image_uid,
            .resource_version = version,
            .old_layout = old_layout,
            .new_layout = new_layout,
            .src_stage = src_stage,
            .src_access = src_access,
            .dst_stage = dst_stage,
            .dst_access = dst_access,
            .subresource_range = subresource_range,
            .valid_write_dependency = valid_dep,
        });
    Common::PerformanceTelemetry::Add(
        valid_dep ? Common::PerformanceTelemetry::Counter::BarrierValidationSuccess
                  : Common::PerformanceTelemetry::Counter::BarrierValidationFailure);

    std::scoped_lock entry_lock{*entry->entry_mutex};
    if (!entry->gpu_consumed) {
        entry->gpu_consumed = true;
        Common::PerformanceTelemetry::RecordAuthorityGpuConsume(
            Common::PerformanceTelemetry::AuthorityGpuConsumeSample{
                .authority_seq = entry->authority_seq,
                .resource_id = image_uid,
                .resource_version = version,
                .image_id = image_id,
                .image_uid = image_uid,
                .consumer_seq = consumer_seq,
                .consumer_packet_seq = Common::PerformanceTelemetry::CurrentPacketSeq(),
                .consumer_kind = 0,
                .requested_access = static_cast<u32>(dst_access),
                .requested_layout = new_layout,
                .producer_tick = entry->producer_tick,
                .consumer_cmd_buffer_seq = Common::PerformanceTelemetry::CurrentCmdBufferSeq(),
                .consumer_submit_seq = 0,
            });
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::AuthorityGpuFirstConsumer);
    }
}

} // namespace VideoCore
