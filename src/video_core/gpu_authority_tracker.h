// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/performance_telemetry.h"
#include "common/types.h"
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
          tick{tick_} {}

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

    std::mutex data_mutex;
    u8* data{};
    VAddr guest_addr{};
    u64 buffer_offset{};
    u32 size{};
    u64 tick{};
    std::unique_ptr<u8[]> owned_data;
};

struct GpuAuthorityEntry {
    u64 authority_seq{0};
    u64 candidate_seq{0};
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
    Common::PerformanceTelemetry::FenceSeq fence_seq{0};
    u64 virtual_fence_seq{0};
    VAddr label_addr{0};
    u32 label_value{0};
    u64 label_generation{0};
    GpuAuthorityState state{GpuAuthorityState::GpuAuthoritative};
    bool gpu_complete{false};
    bool host_current{false};
    bool gpu_consumed{false};
    std::shared_ptr<GpuAuthorityShadow> shadow;
    std::shared_ptr<std::mutex> entry_mutex{std::make_shared<std::mutex>()};
    std::shared_ptr<std::condition_variable> cv{std::make_shared<std::condition_variable>()};
};

struct VirtualGpuFence {
    u64 virtual_fence_seq{0};
    u64 authority_seq{0};
    Common::PerformanceTelemetry::FenceSeq fence_seq{0};
    VAddr label_addr{0};
    u64 label_generation{0};
    u32 expected_value{0};
    u64 producer_tick{0};
    Common::PerformanceTelemetry::PacketSeq producer_packet_seq{0};
    Common::PerformanceTelemetry::PacketSeq eos_packet_seq{0};
    Common::PerformanceTelemetry::PacketSeq wait_packet_seq{0};
    Common::PerformanceTelemetry::PacketSeq acquire_packet_seq{0};
    bool gpu_complete{false};
    bool host_label_written{false};
    bool wait_consumed{false};
};

struct GpuAuthorityIds {
    u64 authority_seq{};
    u64 virtual_fence_seq{};
    u64 label_generation{};
};

class GpuAuthorityTracker {
public:
    static GpuAuthorityTracker& Instance() noexcept;

    void SetRasterizer(Vulkan::Rasterizer* rasterizer_) noexcept;

    [[nodiscard]] bool IsGow3FastpathActive() const noexcept;

    [[nodiscard]] GpuAuthorityIds AllocateIds(VAddr label_addr);
    [[nodiscard]] u64 GetCurrentLabelGeneration(VAddr label_addr) const;

    void RegisterAuthority(const GpuAuthorityEntry& entry);
    void RegisterVirtualFence(const VirtualGpuFence& fence);

    [[nodiscard]] std::vector<std::shared_ptr<GpuAuthorityEntry>> FindOverlaps(VAddr addr, size_t size) const;
    [[nodiscard]] std::shared_ptr<GpuAuthorityEntry> GetAuthorityForImage(u64 image_uid, u64 version) const;
    [[nodiscard]] std::shared_ptr<GpuAuthorityEntry> GetAuthorityForRange(VAddr addr,
                                                                         size_t size) const;

    [[nodiscard]] std::shared_ptr<VirtualGpuFence> MatchVirtualWait(
        VAddr label_addr, u32 ref, u32 mask, u32 function,
        Common::PerformanceTelemetry::PacketSeq wait_pkt,
        Common::PerformanceTelemetry::WaitSeq wait_seq);

    void SignalAsyncLabel(u64 virtual_fence_seq, u64 producer_tick);

    void EnsureVirtualFenceComplete(
        u64 virtual_fence_seq,
        Common::PerformanceTelemetry::VirtualFenceForcedCompletionReason reason);
    void EnsureAllVirtualFencesComplete(
        Common::PerformanceTelemetry::VirtualFenceForcedCompletionReason reason);

    bool ResolveForRamRead(VAddr addr, size_t size,
                          Common::PerformanceTelemetry::GuestSourceConsumePath path,
                          Common::PerformanceTelemetry::ResourceType dest_kind =
                              Common::PerformanceTelemetry::ResourceType::Buffer,
                          u64 dest_res_id = 0);

    bool HandleCpuRead(VAddr fault_addr, size_t size);
    void HandleCpuWrite(VAddr addr, size_t size);
    void HandleUnmap(VAddr addr, size_t size);

    void ValidateGpuConsumerBarrier(u64 image_uid, u64 version, u32 image_id,
                                    u32 old_layout, u32 new_layout,
                                    u64 src_stage, u64 src_access,
                                    u64 dst_stage, u64 dst_access,
                                    u64 subresource_range);

private:
    void RetireVirtualFenceLocked(const std::shared_ptr<VirtualGpuFence>& fence);
    void RefreshAuthorityReadWatches(VAddr addr, size_t size);

    mutable std::recursive_mutex tracker_mutex;
    Vulkan::Rasterizer* rasterizer{nullptr};
    std::vector<std::shared_ptr<GpuAuthorityEntry>> authorities;
    RangeSet authority_read_watch_ranges;
    std::unordered_map<VAddr, u64> label_generations;
    std::unordered_map<VAddr, std::shared_ptr<VirtualGpuFence>> active_virtual_fences;
    std::unordered_map<u64, std::shared_ptr<VirtualGpuFence>> virtual_fences_by_seq;
    u64 next_authority_seq{1};
    u64 next_virtual_fence_seq{1};
};

} // namespace VideoCore
