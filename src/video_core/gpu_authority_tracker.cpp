// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>

#include <boost/container/small_vector.hpp>

#include "common/elf_info.h"
#include "core/memory.h"
#include "video_core/gpu_authority_tracker.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {

namespace {

constexpr VAddr ReadWatchPageMask = 4095;

std::pair<VAddr, size_t> GetReadWatchRange(VAddr addr, size_t size) {
    const VAddr begin = addr & ~ReadWatchPageMask;
    const VAddr end = (addr + size + ReadWatchPageMask) & ~ReadWatchPageMask;
    return {begin, end - begin};
}

constexpr bool HasGpuAuthority(GpuAuthorityState state) noexcept {
    return state == GpuAuthorityState::GpuAuthoritative ||
           state == GpuAuthorityState::Materializing;
}

} // namespace

GpuAuthorityTracker& GpuAuthorityTracker::Instance() noexcept {
    static GpuAuthorityTracker instance;
    return instance;
}

void GpuAuthorityTracker::SetRasterizer(Vulkan::Rasterizer* rasterizer_) noexcept {
    std::scoped_lock lock{tracker_mutex};
    rasterizer = rasterizer_;
}

bool GpuAuthorityTracker::IsGow3FastpathActive() const noexcept {
    return Common::ElfInfo::Instance().GameSerial() == "CUSA01715";
}

GpuAuthorityIds GpuAuthorityTracker::AllocateIds(VAddr label_addr) {
    std::scoped_lock lock{tracker_mutex};
    return {
        .authority_seq = next_authority_seq++,
        .virtual_fence_seq = next_virtual_fence_seq++,
        .label_generation = ++label_generations[label_addr],
    };
}

u64 GpuAuthorityTracker::GetCurrentLabelGeneration(VAddr label_addr) const {
    std::scoped_lock lock{tracker_mutex};
    const auto it = label_generations.find(label_addr);
    return it != label_generations.end() ? it->second : 0;
}

void GpuAuthorityTracker::RegisterAuthority(const GpuAuthorityEntry& entry) {
    if (!IsGow3FastpathActive()) {
        return;
    }
    std::scoped_lock lock{tracker_mutex};
    for (auto& old_entry : authorities) {
        std::scoped_lock entry_lock{*old_entry->entry_mutex};
        if (!HasGpuAuthority(old_entry->state)) {
            continue;
        }
        if (entry.guest_begin <= old_entry->guest_begin && entry.guest_end >= old_entry->guest_end) {
            const u8 old_host_curr = old_entry->host_current ? 1 : 0;
            old_entry->state = GpuAuthorityState::Superseded;
            if (old_entry->shadow) {
                old_entry->shadow->Release();
                old_entry->shadow.reset();
            }
            Common::PerformanceTelemetry::RecordAuthoritySupersede(Common::PerformanceTelemetry::AuthoritySupersedeSample{
                .old_authority_seq = old_entry->authority_seq,
                .new_authority_seq = entry.authority_seq,
                .overlap_begin = old_entry->guest_begin,
                .overlap_size = old_entry->download_size,
                .old_resource_id = old_entry->resource_id,
                .old_resource_version = old_entry->resource_version,
                .new_resource_id = entry.resource_id,
                .new_resource_version = entry.resource_version,
                .old_host_current = old_host_curr,
            });
            Common::PerformanceTelemetry::Add(
                Common::PerformanceTelemetry::Counter::AuthoritySupersededWithoutHostUse);
        }
    }
    std::erase_if(authorities, [](const auto& old_entry) {
        std::scoped_lock entry_lock{*old_entry->entry_mutex};
        return !HasGpuAuthority(old_entry->state);
    });
    authorities.push_back(std::make_shared<GpuAuthorityEntry>(entry));

    const auto [watch_addr, watch_size] =
        GetReadWatchRange(entry.guest_begin, entry.download_size);
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
}

void GpuAuthorityTracker::RegisterVirtualFence(const VirtualGpuFence& fence) {
    if (!IsGow3FastpathActive()) {
        return;
    }
    std::scoped_lock lock{tracker_mutex};
    auto fence_ptr = std::make_shared<VirtualGpuFence>(fence);
    active_virtual_fences[fence.label_addr] = fence_ptr;
    virtual_fences_by_seq[fence.virtual_fence_seq] = fence_ptr;
}

std::vector<std::shared_ptr<GpuAuthorityEntry>> GpuAuthorityTracker::FindOverlaps(
    VAddr addr, size_t size) const {
    std::vector<std::shared_ptr<GpuAuthorityEntry>> result;
    if (!IsGow3FastpathActive()) {
        return result;
    }
    const VAddr query_end = addr + size;
    std::scoped_lock lock{tracker_mutex};
    for (const auto& entry : authorities) {
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (!HasGpuAuthority(entry->state)) {
            continue;
        }
        if (std::max(entry->guest_begin, addr) < std::min(entry->guest_end, query_end)) {
            result.push_back(entry);
        }
    }
    return result;
}

std::shared_ptr<GpuAuthorityEntry> GpuAuthorityTracker::GetAuthorityForImage(
    u64 image_uid, u64 version) const {
    if (!IsGow3FastpathActive()) {
        return nullptr;
    }
    std::scoped_lock lock{tracker_mutex};
    for (auto it = authorities.rbegin(); it != authorities.rend(); ++it) {
        const auto& entry = *it;
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (entry->image_uid == image_uid && entry->resource_version == version &&
            HasGpuAuthority(entry->state)) {
            return entry;
        }
    }
    return nullptr;
}

std::shared_ptr<GpuAuthorityEntry> GpuAuthorityTracker::GetAuthorityForRange(
    VAddr addr, size_t size) const {
    if (!IsGow3FastpathActive() || size == 0) {
        return nullptr;
    }
    std::scoped_lock lock{tracker_mutex};
    for (auto it = authorities.rbegin(); it != authorities.rend(); ++it) {
        const auto& entry = *it;
        std::scoped_lock entry_lock{*entry->entry_mutex};
        if (entry->state != GpuAuthorityState::GpuAuthoritative || addr < entry->guest_begin ||
            size > entry->download_size || addr - entry->guest_begin > entry->download_size - size) {
            continue;
        }
        return entry;
    }
    return nullptr;
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

std::shared_ptr<VirtualGpuFence> GpuAuthorityTracker::MatchVirtualWait(
    VAddr label_addr, u32 ref, u32 mask, u32 function,
    Common::PerformanceTelemetry::PacketSeq wait_pkt,
    Common::PerformanceTelemetry::WaitSeq wait_seq) {
    if (!IsGow3FastpathActive()) {
        return nullptr;
    }
    // GOW3 wait signature: memory wait, Equal (function == 3) or GreaterThanEqual (function == 5), ref == 1, mask == 0xFFFFFFFF
    if ((function != 3 && function != 5) || ref != 1 || mask != 0xFFFFFFFF) {
        return nullptr;
    }
    std::scoped_lock lock{tracker_mutex};
    const auto it = active_virtual_fences.find(label_addr);
    if (it == active_virtual_fences.end()) {
        return nullptr;
    }
    auto fence = it->second;
    const auto gen_it = label_generations.find(label_addr);
    const u64 current_gen = gen_it != label_generations.end() ? gen_it->second : 0;
    if (fence->label_generation != current_gen || fence->wait_consumed) {
        return nullptr;
    }
    fence->wait_consumed = true;
    fence->wait_packet_seq = wait_pkt;
    return fence;
}

void GpuAuthorityTracker::SignalAsyncLabel(u64 virtual_fence_seq, u64 producer_tick) {
    std::shared_ptr<VirtualGpuFence> fence;
    u64 current_gen = 0;
    {
        std::scoped_lock lock{tracker_mutex};
        const auto it = virtual_fences_by_seq.find(virtual_fence_seq);
        if (it == virtual_fences_by_seq.end()) {
            return;
        }
        fence = it->second;
        const auto gen_it = label_generations.find(fence->label_addr);
        current_gen = gen_it != label_generations.end() ? gen_it->second : 0;
    }

    const u64 completed_tick = rasterizer ? rasterizer->KnownGpuTick() : 0;
    fence->gpu_complete = true;
    if (fence->label_generation == current_gen) {
        *reinterpret_cast<u32*>(fence->label_addr) = fence->expected_value;
        fence->host_label_written = true;
        if (rasterizer) {
            rasterizer->NotifyMemoryWrite(fence->label_addr, sizeof(u32),
                                          VideoCore::MemoryWriteSource::CommandProcessor);
        }
        Common::PerformanceTelemetry::RecordAsyncLabelSignal(Common::PerformanceTelemetry::AsyncLabelSignalSample{
            .virtual_fence_seq = fence->virtual_fence_seq,
            .authority_seq = fence->authority_seq,
            .label_addr = fence->label_addr,
            .scheduled_generation = fence->label_generation,
            .current_generation = current_gen,
            .producer_tick = fence->producer_tick,
            .current_completed_tick = completed_tick,
            .action = Common::PerformanceTelemetry::AsyncLabelAction::Wrote,
        });
        Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::AsyncLabelWrites);
    } else {
        Common::PerformanceTelemetry::RecordAsyncLabelSignal(Common::PerformanceTelemetry::AsyncLabelSignalSample{
            .virtual_fence_seq = fence->virtual_fence_seq,
            .authority_seq = fence->authority_seq,
            .label_addr = fence->label_addr,
            .scheduled_generation = fence->label_generation,
            .current_generation = current_gen,
            .producer_tick = fence->producer_tick,
            .current_completed_tick = completed_tick,
            .action = Common::PerformanceTelemetry::AsyncLabelAction::StaleGenerationSkipped,
        });
        Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::StaleLabelCallbacks);
    }
}

void GpuAuthorityTracker::EnsureVirtualFenceComplete(
    u64 virtual_fence_seq,
    Common::PerformanceTelemetry::VirtualFenceForcedCompletionReason reason) {
    if (!IsGow3FastpathActive()) {
        return;
    }
    std::shared_ptr<VirtualGpuFence> fence;
    {
        std::scoped_lock lock{tracker_mutex};
        const auto it = virtual_fences_by_seq.find(virtual_fence_seq);
        if (it == virtual_fences_by_seq.end()) {
            return;
        }
        fence = it->second;
    }
    if (fence->gpu_complete && fence->host_label_written) {
        return;
    }
    const u64 start_ts = Common::PerformanceTelemetry::Timestamp();
    u8 was_submitted = 0;
    u8 waited = 0;
    if (rasterizer) {
        const u64 current_tick = rasterizer->CurrentTick();
        if (fence->producer_tick >= current_tick) {
            rasterizer->Flush(Common::PerformanceTelemetry::SubmitReason::WaitProgress);
            was_submitted = 1;
        }
        if (rasterizer->KnownGpuTick() < fence->producer_tick) {
            rasterizer->WaitTick(
                fence->producer_tick,
                Common::PerformanceTelemetry::HostWaitReason::FenceCpuVisibility);
            waited = 1;
        }
    }
    fence->gpu_complete = true;
    SignalAsyncLabel(fence->virtual_fence_seq, fence->producer_tick);
    const u64 end_ts = Common::PerformanceTelemetry::Timestamp();
    const u64 duration = end_ts >= start_ts ? end_ts - start_ts : 0;

    Common::PerformanceTelemetry::RecordVirtualFenceForcedCompletion(
        Common::PerformanceTelemetry::VirtualFenceForcedCompletionSample{
            .virtual_fence_seq = fence->virtual_fence_seq,
            .authority_seq = fence->authority_seq,
            .reason = reason,
            .producer_tick = fence->producer_tick,
            .was_submitted = was_submitted,
            .waited = waited,
            .duration_ns = duration,
        });
    Common::PerformanceTelemetry::Add(
        Common::PerformanceTelemetry::Counter::VirtualFenceForcedCompletions);
}

void GpuAuthorityTracker::EnsureAllVirtualFencesComplete(
    Common::PerformanceTelemetry::VirtualFenceForcedCompletionReason reason) {
    if (!IsGow3FastpathActive()) {
        return;
    }
    std::vector<u64> pending_seqs;
    {
        std::scoped_lock lock{tracker_mutex};
        for (const auto& [seq, fence] : virtual_fences_by_seq) {
            if (!fence->gpu_complete || !fence->host_label_written) {
                pending_seqs.push_back(seq);
            }
        }
    }
    for (u64 seq : pending_seqs) {
        EnsureVirtualFenceComplete(seq, reason);
    }
}

namespace {
thread_local u64 tls_active_materializing_authority_seq{0};

class ScopedAuthorityMaterialization {
public:
    explicit ScopedAuthorityMaterialization(u64 authority_seq) noexcept
        : prev_seq(tls_active_materializing_authority_seq), active_seq(authority_seq) {
        tls_active_materializing_authority_seq = authority_seq;
    }
    ~ScopedAuthorityMaterialization() noexcept {
        tls_active_materializing_authority_seq = prev_seq;
    }
    ScopedAuthorityMaterialization(const ScopedAuthorityMaterialization&) = delete;
    ScopedAuthorityMaterialization& operator=(const ScopedAuthorityMaterialization&) = delete;

private:
    u64 prev_seq{0};
    u64 active_seq{0};
};

[[nodiscard]] bool IsCurrentThreadMaterializing(u64 authority_seq) noexcept {
    return tls_active_materializing_authority_seq != 0 &&
           tls_active_materializing_authority_seq == authority_seq;
}
} // anonymous namespace

bool GpuAuthorityTracker::ResolveForRamRead(
    VAddr addr, size_t size,
    Common::PerformanceTelemetry::GuestSourceConsumePath path,
    Common::PerformanceTelemetry::ResourceType dest_kind,
    u64 dest_res_id) {
    if (!IsGow3FastpathActive()) {
        return true;
    }
    const auto overlaps = FindOverlaps(addr, size);
    if (overlaps.empty()) {
        return true;
    }
    const u64 consumer_seq = Common::PerformanceTelemetry::NextConsumerSeq();
    const u64 group_seq = Common::PerformanceTelemetry::NextRamDemandGroupSeq();
    const u64 cur_prod_seq = Common::PerformanceTelemetry::CurrentProducerSeq();
    const u64 cur_pkt_seq = Common::PerformanceTelemetry::CurrentPacketSeq();

    bool all_succeeded = true;

    for (const auto& entry : overlaps) {
        const VAddr overlap_begin = std::max(entry->guest_begin, addr);
        const VAddr overlap_end = std::min(entry->guest_end, addr + size);
        const u64 overlap_size = overlap_end > overlap_begin ? overlap_end - overlap_begin : 0;
        if (overlap_size == 0) {
            continue;
        }

        std::unique_lock entry_lk{*entry->entry_mutex};

        if (IsCurrentThreadMaterializing(entry->authority_seq)) {
            // Internal access from this authority's own materializer: bypass self-wait
            continue;
        }

        const u64 ram_demand_seq = Common::PerformanceTelemetry::NextRamDemandSeq();
        const u8 state_before = static_cast<u8>(entry->state);

        Common::PerformanceTelemetry::RecordAuthorityRamDemand(Common::PerformanceTelemetry::AuthorityRamDemandSample{
            .ram_demand_seq = ram_demand_seq,
            .ram_demand_group_seq = group_seq,
            .timestamp_ns = Common::PerformanceTelemetry::Timestamp(),
            .authority_seq = entry->authority_seq,
            .resource_id = entry->resource_id,
            .resource_version = entry->resource_version,
            .authority_begin = entry->guest_begin,
            .authority_end = entry->guest_end,
            .request_addr = addr,
            .request_size = size,
            .overlap_begin = overlap_begin,
            .overlap_size = overlap_size,
            .path = path,
            .consumer_seq = consumer_seq,
            .consumer_producer_seq = cur_prod_seq,
            .consumer_packet_seq = cur_pkt_seq,
            .destination_kind = dest_kind,
            .destination_resource_id = dest_res_id,
            .authority_state_before = state_before,
        });
        Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::RamDemandEvents);

        if (entry->state == GpuAuthorityState::HostCurrent) {
            Common::PerformanceTelemetry::RecordAuthorityRamConsume(Common::PerformanceTelemetry::AuthorityRamConsumeSample{
                .ram_demand_seq = ram_demand_seq,
                .authority_seq = entry->authority_seq,
                .materialize_seq = 0,
                .consumer_seq = consumer_seq,
                .path = path,
                .request_addr = addr,
                .request_size = size,
                .overlap_begin = overlap_begin,
                .overlap_size = overlap_size,
                .host_current = 1,
                .producer_tick_complete = 1,
                .materialize_success = 1,
            });
            continue;
        }

        if (entry->state == GpuAuthorityState::Superseded || entry->state == GpuAuthorityState::Failed) {
            all_succeeded = false;
            continue;
        }

        if (entry->state == GpuAuthorityState::Materializing) {
            entry->cv->wait(entry_lk, [&] { return entry->state != GpuAuthorityState::Materializing; });
            const bool is_current = (entry->state == GpuAuthorityState::HostCurrent);
            if (is_current) {
                Common::PerformanceTelemetry::RecordAuthorityRamConsume(Common::PerformanceTelemetry::AuthorityRamConsumeSample{
                    .ram_demand_seq = ram_demand_seq,
                    .authority_seq = entry->authority_seq,
                    .materialize_seq = 0,
                    .consumer_seq = consumer_seq,
                    .path = path,
                    .request_addr = addr,
                    .request_size = size,
                    .overlap_begin = overlap_begin,
                    .overlap_size = overlap_size,
                    .host_current = 1,
                    .producer_tick_complete = 1,
                    .materialize_success = 1,
                });
            } else {
                all_succeeded = false;
            }
            continue;
        }

        if (entry->state == GpuAuthorityState::GpuAuthoritative) {
            entry->state = GpuAuthorityState::Materializing;
            const u64 mat_seq = Common::PerformanceTelemetry::NextMaterializeSeq();
            const u64 cur_tick = rasterizer ? rasterizer->KnownGpuTick() : 0;
            const u32 img_id = entry->image_id;
            const u64 img_uid = entry->image_uid;
            const VAddr g_begin = entry->guest_begin;
            const VAddr g_end = entry->guest_end;
            const u32 dl_size = entry->download_size;
            const u64 prod_tick = entry->producer_tick;
            const u64 auth_seq = entry->authority_seq;
            const u64 res_id = entry->resource_id;
            const u64 res_ver = entry->resource_version;
            auto authority_shadow = entry->shadow;

            Common::PerformanceTelemetry::RecordLazyMaterializeBegin(Common::PerformanceTelemetry::LazyMaterializeBeginSample{
                .materialize_seq = mat_seq,
                .ram_demand_seq = ram_demand_seq,
                .authority_seq = auth_seq,
                .resource_id = res_id,
                .resource_version = res_ver,
                .image_id = img_id,
                .image_uid = img_uid,
                .producer_tick = prod_tick,
                .current_completed_tick = cur_tick,
                .guest_begin = g_begin,
                .guest_end = g_end,
                .reason = 0,
            });

            // The shadow copy was recorded when the authority was promoted. Waiting for its
            // timeline never requires a synchronous command on the GCP thread.
            entry_lk.unlock();
            if (rasterizer && authority_shadow) {
                rasterizer->GetTextureCache().WaitGpuAuthorityShadow(authority_shadow);
            }
            entry_lk.lock();

            s8 val_equal = -1;
            bool success = false;
            if (entry->state == GpuAuthorityState::Materializing) {
                ScopedAuthorityMaterialization scoped_mat{auth_seq};
                if (rasterizer && authority_shadow) {
                    success = rasterizer->GetTextureCache().MaterializeGpuAuthority(
                        authority_shadow, g_begin, dl_size, &val_equal);
                }
                entry->state = success ? GpuAuthorityState::HostCurrent
                                       : GpuAuthorityState::Failed;
                entry->host_current = success;
            }
            if (entry->shadow) {
                entry->shadow->Release();
                entry->shadow.reset();
            }
            entry->cv->notify_all();

            const u64 completed_tick = rasterizer ? rasterizer->KnownGpuTick() : 0;
            Common::PerformanceTelemetry::RecordLazyMaterializeEnd(Common::PerformanceTelemetry::LazyMaterializeEndSample{
                .materialize_seq = mat_seq,
                .ram_demand_seq = ram_demand_seq,
                .authority_seq = auth_seq,
                .producer_tick = prod_tick,
                .completed_tick = completed_tick,
                .bytes_materialized = dl_size,
                .guest_begin = g_begin,
                .guest_end = g_end,
                .result = success ? Common::PerformanceTelemetry::LazyMaterializeResult::Success
                                  : Common::PerformanceTelemetry::LazyMaterializeResult::DownloadFailed,
                .host_current_after = static_cast<u8>(success ? 1 : 0),
                .validation_bytes_equal = val_equal,
            });

            if (success) {
                if (dl_size == 3072) {
                    Common::PerformanceTelemetry::Add(
                        Common::PerformanceTelemetry::Counter::Lazy3kMaterializations);
                }
                Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::MaterializeSuccess);
                if (val_equal != 1 && val_equal != -1) {
                    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::MaterializeByteMismatch);
                }

                Common::PerformanceTelemetry::RecordAuthorityRamConsume(Common::PerformanceTelemetry::AuthorityRamConsumeSample{
                    .ram_demand_seq = ram_demand_seq,
                    .authority_seq = auth_seq,
                    .materialize_seq = mat_seq,
                    .consumer_seq = consumer_seq,
                    .path = path,
                    .request_addr = addr,
                    .request_size = size,
                    .overlap_begin = overlap_begin,
                    .overlap_size = overlap_size,
                    .host_current = 1,
                    .producer_tick_complete = 1,
                    .materialize_success = 1,
                });
            } else {
                Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::MaterializeFailure);
                all_succeeded = false;
            }
        }
    }
    for (const auto& entry : overlaps) {
        RefreshAuthorityReadWatches(entry->guest_begin, entry->download_size);
    }
    return all_succeeded;
}

bool GpuAuthorityTracker::HandleCpuRead(VAddr fault_addr, size_t size) {
    if (!IsGow3FastpathActive()) {
        return false;
    }
    const auto [watch_addr, watch_size] =
        GetReadWatchRange(fault_addr, size > 0 ? size : 4);
    const auto overlaps = FindOverlaps(watch_addr, watch_size);
    if (overlaps.empty()) {
        RefreshAuthorityReadWatches(watch_addr, watch_size);
        return false;
    }
    for (const auto& entry : overlaps) {
        if (IsCurrentThreadMaterializing(entry->authority_seq)) {
            // Internal read during materialization of this authority: unprotect and continue
            if (rasterizer) {
                rasterizer->DisarmSemanticReadWatch(fault_addr, size > 0 ? size : 4);
            }
            return true;
        }
        Common::PerformanceTelemetry::RecordAuthorityCpuRead(Common::PerformanceTelemetry::AuthorityCpuReadSample{
            .authority_seq = entry->authority_seq,
            .fault_addr = fault_addr,
            .guest_read_begin = entry->guest_begin,
            .guest_read_size = entry->download_size,
            .origin = Common::PerformanceTelemetry::SemanticReadOrigin::GuestDirect,
            .materialize_seq = 0,
            .resumed_after_materialize = 1,
        });
        Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::AuthorityCpuRead);
    }
    const bool ok = ResolveForRamRead(watch_addr, watch_size,
                                      Common::PerformanceTelemetry::GuestSourceConsumePath::BufferUpload,
                                      Common::PerformanceTelemetry::ResourceType::Buffer, 0);
    return ok;
}

void GpuAuthorityTracker::HandleCpuWrite(VAddr addr, size_t size) {
    if (!IsGow3FastpathActive()) {
        return;
    }
    const size_t access_size = size > 0 ? size : 4;
    const auto [watch_addr, watch_size] = GetReadWatchRange(addr, access_size);
    const auto overlaps = FindOverlaps(watch_addr, watch_size);
    const bool external_overlap = std::ranges::any_of(overlaps, [](const auto& entry) {
        return !IsCurrentThreadMaterializing(entry->authority_seq);
    });
    if (external_overlap) {
        ResolveForRamRead(watch_addr, watch_size,
                          Common::PerformanceTelemetry::GuestSourceConsumePath::BufferUpload,
                          Common::PerformanceTelemetry::ResourceType::Buffer, 0);
    }
    if (size >= 4) {
        u32 val = 0;
        std::memcpy(&val, reinterpret_cast<const void*>(addr), sizeof(u32));
        Common::PerformanceTelemetry::RecordGuestCpuLabelWrite(
            addr, val, Common::PerformanceTelemetry::Timestamp(),
#ifdef _WIN32
            static_cast<u64>(::GetCurrentThreadId())
#else
            0
#endif
        );
    }
}

void GpuAuthorityTracker::HandleUnmap(VAddr addr, size_t size) {
    if (!IsGow3FastpathActive()) {
        return;
    }
    EnsureAllVirtualFencesComplete(Common::PerformanceTelemetry::VirtualFenceForcedCompletionReason::Unmap);
    const auto overlaps = FindOverlaps(addr, size > 0 ? size : 4);
    for (const auto& entry : overlaps) {
        std::unique_lock entry_lk{*entry->entry_mutex};
        entry->state = GpuAuthorityState::Superseded;
        if (entry->shadow) {
            entry->shadow->Release();
            entry->shadow.reset();
        }
        entry->cv->notify_all();
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::AuthorityDestroyedWithoutHostUse);
    }
    for (const auto& entry : overlaps) {
        RefreshAuthorityReadWatches(entry->guest_begin, entry->download_size);
    }
}

void GpuAuthorityTracker::ValidateGpuConsumerBarrier(
    u64 image_uid, u64 version, u32 image_id,
    u32 old_layout, u32 new_layout,
    u64 src_stage, u64 src_access,
    u64 dst_stage, u64 dst_access,
    u64 subresource_range) {
    if (!IsGow3FastpathActive()) {
        return;
    }
    auto entry = GetAuthorityForImage(image_uid, version);
    if (!entry) {
        return;
    }

    const u64 consumer_seq = Common::PerformanceTelemetry::NextConsumerSeq();
    const u64 prod_tick = entry->producer_tick;
    const auto cur_cmdbuf = Common::PerformanceTelemetry::CurrentCmdBufferSeq();
    const auto cur_pkt = Common::PerformanceTelemetry::CurrentPacketSeq();

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

    if (valid_dep) {
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::BarrierValidationSuccess);
    } else {
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::BarrierValidationFailure);
    }

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
                .consumer_packet_seq = cur_pkt,
                .consumer_kind = 0,
                .requested_access = static_cast<u32>(dst_access),
                .requested_layout = new_layout,
                .producer_tick = prod_tick,
                .consumer_cmd_buffer_seq = cur_cmdbuf,
                .consumer_submit_seq = 0,
            });
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::AuthorityGpuFirstConsumer);
    }
}

} // namespace VideoCore
