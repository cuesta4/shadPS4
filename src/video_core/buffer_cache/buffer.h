// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>
#include "common/incremental_id.h"
#include "common/performance_telemetry.h"
#include "common/types.h"
#include "core/memory.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/buffer_cache/stream_buffer_pin.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

struct VmaAllocationInfo;

namespace VideoCore {

/// Hints and requirements for the backing memory type of a commit
enum class MemoryUsage {
    DeviceLocal, ///< Requests device local buffer.
    Upload,      ///< Requires a host visible memory type optimized for CPU to GPU uploads
    Download,    ///< Requires a host visible memory type optimized for GPU to CPU readbacks
    Stream,      ///< Requests device local host visible buffer, falling back host memory.
};

constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

struct UniqueBuffer {
    explicit UniqueBuffer(vk::Device device, VmaAllocator allocator);
    ~UniqueBuffer();

    UniqueBuffer(const UniqueBuffer&) = delete;
    UniqueBuffer& operator=(const UniqueBuffer&) = delete;

    UniqueBuffer(UniqueBuffer&& other)
        : allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          buffer{std::exchange(other.buffer, VK_NULL_HANDLE)} {}
    UniqueBuffer& operator=(UniqueBuffer&& other) {
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        return *this;
    }

    void Create(const vk::BufferCreateInfo& image_ci, MemoryUsage usage,
                VmaAllocationInfo* out_alloc_info);

    operator vk::Buffer() const {
        return buffer;
    }

    vk::Device device;
    VmaAllocator allocator;
    VmaAllocation allocation;
    vk::Buffer buffer{};
    vk::DeviceAddress bda_addr = 0;
};

class Buffer {
public:
    explicit Buffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                    MemoryUsage usage, VAddr cpu_addr_, vk::BufferUsageFlags flags,
                    u64 size_bytes_);

    Buffer& operator=(const Buffer&) = delete;
    Buffer(const Buffer&) = delete;

    Buffer& operator=(Buffer&&) = default;
    Buffer(Buffer&&) = default;

    void IncreaseStreamScore(int score) noexcept {
        stream_score += score;
    }

    [[nodiscard]] int StreamScore() const noexcept {
        return stream_score;
    }

    [[nodiscard]] bool IsInBounds(VAddr addr, u64 size) const noexcept {
        return addr >= cpu_addr && addr + size <= cpu_addr + SizeBytes();
    }

    [[nodiscard]] VAddr CpuAddr() const noexcept {
        return cpu_addr;
    }

    [[nodiscard]] u64 Offset(VAddr other_cpu_addr) const noexcept {
        return other_cpu_addr - cpu_addr;
    }

    size_t SizeBytes() const {
        return size_bytes;
    }

    void SetLRUId(u64 id) noexcept {
        lru_id = id;
    }

    u64 LRUId() const noexcept {
        return lru_id;
    }

    u64 Uid() const noexcept {
        return uid;
    }

    vk::Buffer Handle() const noexcept {
        return buffer;
    }

    vk::DeviceAddress BufferDeviceAddress() const noexcept {
        ASSERT_MSG(buffer.bda_addr != 0, "Can't get BDA from a non BDA buffer");
        return buffer.bda_addr;
    }

    [[nodiscard]] u32 MemoryTypeIndex() const noexcept {
        return memory_type_index;
    }

    [[nodiscard]] u32 MemoryHeapIndex() const noexcept {
        return memory_heap_index;
    }

    [[nodiscard]] u32 MemoryPropertyFlags() const noexcept {
        return memory_property_flags;
    }

    std::optional<vk::BufferMemoryBarrier2> GetBarrier(vk::AccessFlags2 dst_acess_mask,
                                                       vk::PipelineStageFlagBits2 dst_stage,
                                                       u32 offset = 0) {
        if (dst_acess_mask == access_mask && stage == dst_stage) {
            return {};
        }

        DEBUG_ASSERT(offset < size_bytes);

        const auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = stage,
            .srcAccessMask = access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_acess_mask,
            .buffer = buffer.buffer,
            .offset = offset,
            .size = size_bytes - offset,
        };
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
        if (Common::PerformanceTelemetry::HasActiveReadbackSourceWatch(uid, 0)) {
            Common::PerformanceTelemetry::RecordResourceBarrierLink(Common::PerformanceTelemetry::ResourceBarrierLinkSample{
                .resource_id = uid,
                .resource_version = 0,
                .fence_seq = 0,
                .readback_seq = 0,
                .cmd_buffer_seq = Common::PerformanceTelemetry::CurrentCmdBufferSeq(),
                .submit_seq = 0,
                .old_layout = 0,
                .new_layout = 0,
                .src_stage = static_cast<u64>(stage),
                .src_access = static_cast<u64>(access_mask),
                .dst_stage = static_cast<u64>(dst_stage),
                .dst_access = static_cast<u64>(dst_acess_mask),
                .subresource_or_range = offset,
                .reason_path = "buffer_barrier",
            });
        }
#endif
        access_mask = dst_acess_mask;
        stage = dst_stage;
        return barrier;
    }

    void Fill(u64 offset, u32 num_bytes, u32 value);

    /// Makes a completed download range visible to the CPU.
    void Invalidate(u64 offset, u64 size);

public:
    VAddr cpu_addr = 0;
    bool is_picked{};
    bool is_coherent{};
    bool is_deleted{};
    bool has_image_alias{};
    int stream_score = 0;
    size_t size_bytes = 0;
    u64 lru_id = 0;
    u64 uid = 0;
    std::span<u8> mapped_data;
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    MemoryUsage usage;
    UniqueBuffer buffer;
    vk::Flags<vk::AccessFlagBits2> access_mask{
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
    vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};

private:
    u32 memory_type_index{};
    u32 memory_heap_index{};
    u32 memory_property_flags{};
    static Common::IncrementalIdProvider<u64> global_uid;
};

class StreamBuffer : public Buffer {
public:
    explicit StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                          MemoryUsage usage, u64 size_bytes_);

    /// Reserves a region of memory from the stream buffer.
    std::pair<u8*, u64> Map(u64 size, u64 alignment = 0, bool allow_wait = true);

    /// Ensures that reserved bytes of memory are available to the GPU.
    void Commit(StreamBufferPinHandle pin = {});

    /// Returns the ring-buffer generation. It changes whenever allocations wrap to offset zero.
    [[nodiscard]] u64 Generation() const noexcept {
        return generation;
    }

    /// Maps and commits a memory region with user provided data
    u64 Copy(auto src, size_t size, size_t alignment = 0) {
        const auto [data, offset] = Map(size, alignment);
        auto* memory = Core::Memory::Instance();
        const VAddr src_vaddr = reinterpret_cast<const VAddr>(src);
        if (memory->IsValidMapping(src_vaddr)) {
            memory->CopySparseMemory(src_vaddr, data, size);
        } else {
            std::memcpy(data, reinterpret_cast<const void*>(src), size);
        }
        Commit();
        return offset;
    }

private:
    struct Watch {
        u64 tick{};
        u64 upper_bound{};
    };

    /// Increases the amount of watches available.
    void ReserveWatches(std::vector<Watch>& watches, std::vector<StreamBufferPinHandle>& pins,
                        std::size_t grow_size);

    /// Waits pending watches until requested upper bound.
    bool WaitPendingOperations(u64 requested_upper_bound, bool allow_wait);

private:
    u64 offset{};
    u64 mapped_size{};
    u64 generation{1};
    std::vector<Watch> current_watches;
    std::vector<StreamBufferPinHandle> current_watch_pins;
    std::size_t current_watch_cursor{};
    std::optional<size_t> invalidation_mark;
    std::vector<Watch> previous_watches;
    std::vector<StreamBufferPinHandle> previous_watch_pins;
    std::size_t wait_cursor{};
    u64 wait_bound{};
};

} // namespace VideoCore
