// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include <boost/container/static_vector.hpp>
#include "common/alignment.h"
#include "common/debug.h"
#include "common/performance_telemetry.h"
#include "common/scope_exit.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {

static constexpr size_t DataShareBufferSize = 64_KB;
static constexpr size_t StagingBufferSize = 512_MB;
static constexpr size_t DownloadBufferSize = 32_MB;
static constexpr size_t UboStreamBufferSize = 64_MB;
static constexpr size_t DeviceBufferSize = 128_MB;

struct BufferCache::StreamCopyScratch {
    static constexpr size_t HashTableSize = 256;
    static constexpr u16 NoCanonicalCopy = std::numeric_limits<u16>::max();
    static_assert(std::has_single_bit(HashTableSize));

    struct CanonicalCopy {
        StreamCopyRequest request{};
        u64 relative_offset{};
    };

    struct RequestMap {
        u16 canonical{NoCanonicalCopy};
        u32 source_offset{};
    };

    struct HashSlot {
        u16 generation{};
        u16 canonical{};
    };

    std::array<CanonicalCopy, MaxStreamCopyRequests> canonical_copies{};
    std::array<RequestMap, MaxStreamCopyRequests> request_map{};
    std::array<HashSlot, HashTableSize> hash_table{};
    std::array<Core::MemoryManager::SparseCopyRequest, MaxStreamCopyRequests> guest_copies{};
    u16 hash_generation{1};
};

struct BufferCache::StreamSliceReuseState {
    static constexpr size_t WayCount = 2;
    static constexpr size_t SetCount = 64;
    static constexpr size_t EntryCount = WayCount * SetCount;
    static_assert(std::has_single_bit(SetCount));

    struct Entry {
        VAddr address{};
        u64 generation{};
        u64 tick{};
        u32 size{};
        u32 offset{};
        bool valid{};
    };

    struct Set {
        std::array<Entry, WayCount> ways{};
        u8 next_replacement{};
    };

    std::array<Set, SetCount> sets{};
    std::unique_ptr<u8[]> shadow =
        std::make_unique_for_overwrite<u8[]>(EntryCount * CACHING_PAGESIZE);
    std::array<u8, CACHING_PAGESIZE> scratch{};

    [[nodiscard]] u8* Shadow(size_t set_index, size_t way) noexcept {
        return shadow.get() + (set_index * WayCount + way) * CACHING_PAGESIZE;
    }
};

struct BufferCache::VertexIndexState {
    static constexpr u16 NoStreamCopy = std::numeric_limits<u16>::max();

    struct BufferRange {
        VAddr base_address{};
        VAddr end_address{};
        vk::Buffer vk_buffer{};
        Buffer* buffer{};
        u64 offset{};
        u32 binding_mask{};
        u32 size{};
        u16 stream_index{NoStreamCopy};
        bool was_gpu_modified{};

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    struct IndexBinding {
        Buffer* buffer{};
        VAddr address{};
        u64 offset{};
        u32 size{};
        u16 stream_index{NoStreamCopy};
        vk::IndexType type{vk::IndexType::eUint32};
        bool was_gpu_modified{};
    };

    Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    Vulkan::VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    Vulkan::VertexInputs<AmdGpu::Buffer> guest_buffers;
    Vulkan::VertexInputs<BufferRange> ranges;
    Vulkan::VertexInputs<BufferRange> ranges_merged;
    IndexBinding index{};
    bool bind_index_buffer{};
    bool prepared{};

    u64 emitted_tick{std::numeric_limits<u64>::max()};
    bool vertex_input_valid{};
    bool vertex_buffers_valid{};
    bool index_buffer_valid{};
    u32 emitted_attribute_count{};
    u32 emitted_binding_count{};
    u32 emitted_buffer_count{};
    std::array<vk::VertexInputAttributeDescription2EXT, Vulkan::MaxVertexBufferCount>
        emitted_attributes{};
    std::array<vk::VertexInputBindingDescription2EXT, Vulkan::MaxVertexBufferCount>
        emitted_bindings{};
    std::array<vk::Buffer, Vulkan::MaxVertexBufferCount> emitted_buffers{};
    std::array<vk::DeviceSize, Vulkan::MaxVertexBufferCount> emitted_offsets{};
    std::array<vk::DeviceSize, Vulkan::MaxVertexBufferCount> emitted_sizes{};
    std::array<vk::DeviceSize, Vulkan::MaxVertexBufferCount> emitted_strides{};
    std::array<vk::Buffer, Vulkan::MaxVertexBufferCount> host_buffers{};
    std::array<vk::DeviceSize, Vulkan::MaxVertexBufferCount> host_offsets{};
    std::array<vk::DeviceSize, Vulkan::MaxVertexBufferCount> host_sizes{};
    std::array<vk::DeviceSize, Vulkan::MaxVertexBufferCount> host_strides{};
    vk::Buffer emitted_index_buffer{};
    vk::DeviceSize emitted_index_offset{};
    vk::IndexType emitted_index_type{vk::IndexType::eUint32};
};

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         AmdGpu::Liverpool* liverpool_, TextureCache& texture_cache_,
                         PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      fault_manager{instance, scheduler, *this, CACHING_PAGEBITS, CACHING_NUMPAGES},
      staging_buffer{instance, scheduler, MemoryUsage::Upload, StagingBufferSize},
      stream_buffer{instance, scheduler, MemoryUsage::Stream, UboStreamBufferSize},
      download_buffer{instance, scheduler, MemoryUsage::Download, DownloadBufferSize},
      device_buffer{instance, scheduler, MemoryUsage::DeviceLocal, DeviceBufferSize},
      gds_buffer{instance, scheduler, MemoryUsage::Stream, 0, AllFlags, DataShareBufferSize},
      bda_pagetable_buffer{instance, scheduler, MemoryUsage::DeviceLocal,
                           0,        AllFlags,  BDA_PAGETABLE_SIZE} {
    Vulkan::SetObjectName(instance.GetDevice(), gds_buffer.Handle(), "GDS Buffer");
    Vulkan::SetObjectName(instance.GetDevice(), bda_pagetable_buffer.Handle(),
                          "BDA Page Table Buffer");

    memory_tracker = std::make_unique<MemoryTracker>(tracker);
    stream_copy_scratch = std::make_unique<StreamCopyScratch>();
    stream_slice_reuse = std::make_unique<StreamSliceReuseState>();
    vertex_index_state = std::make_unique<VertexIndexState>();

    std::memset(gds_buffer.mapped_data.data(), 0, DataShareBufferSize);

    // Set up garbage collection parameters
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = DEFAULT_TRIGGER_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    trigger_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_TRIGGER_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
}

BufferCache::~BufferCache() = default;

void BufferCache::BeginStreamCopyBatch() noexcept {
    stream_copy_request_count = 0;
    stream_copy_finalized = false;
    vertex_index_state->prepared = false;
}

u16 BufferCache::QueueStreamCopy(const StreamCopyRequest& request) {
    ASSERT(!stream_copy_finalized);
    ASSERT(stream_copy_request_count < MaxStreamCopyRequests);
    ASSERT(request.size != 0);
    ASSERT(request.alignment != 0 && std::has_single_bit(request.alignment));
    ASSERT(request.source_type == StreamCopySource::Guest ||
           request.source_type == StreamCopySource::Host ||
           request.source_type == StreamCopySource::Zero);
    ASSERT(request.source_type != StreamCopySource::Host || request.host_address != nullptr);
    const u16 index = stream_copy_request_count++;
    stream_copy_requests[index] = request;
    return index;
}

void BufferCache::FinalizeStreamCopyBatch() {
    ASSERT(!stream_copy_finalized);
    if (stream_copy_request_count != 0) {
        ExecuteStreamCopyBatch(
            std::span<const StreamCopyRequest>{stream_copy_requests.data(),
                                               stream_copy_request_count},
            std::span<StreamCopyResult>{stream_copy_results.data(), stream_copy_request_count});
    }
    stream_copy_finalized = true;
}

const BufferCache::StreamCopyResult& BufferCache::GetStreamCopyResult(u16 index) const {
    ASSERT(stream_copy_finalized);
    ASSERT(index < stream_copy_request_count);
    return stream_copy_results[index];
}

void BufferCache::ExecuteStreamCopyBatch(std::span<const StreamCopyRequest> requests,
                                         std::span<StreamCopyResult> results) {
    ASSERT(requests.size() == results.size());
    ASSERT(requests.size() <= MaxStreamCopyRequests);
    if (requests.empty()) {
        return;
    }

    auto& scratch = *stream_copy_scratch;
    if (++scratch.hash_generation == 0) {
        for (auto& slot : scratch.hash_table) {
            slot.generation = 0;
        }
        scratch.hash_generation = 1;
    }

    const auto source_key = [](const StreamCopyRequest& request) noexcept -> u64 {
        switch (request.source_type) {
        case StreamCopySource::Guest:
            return request.guest_address;
        case StreamCopySource::Host:
            return reinterpret_cast<uintptr_t>(request.host_address);
        case StreamCopySource::Zero:
            return 0;
        }
        std::unreachable();
    };
    const auto equivalent = [&](const StreamCopyRequest& lhs, const StreamCopyRequest& rhs) {
        return lhs.source_type == rhs.source_type && lhs.size == rhs.size &&
               source_key(lhs) == source_key(rhs);
    };
    const auto hash_request = [&](const StreamCopyRequest& request) {
        u64 value = source_key(request) ^ (static_cast<u64>(request.size) << 17) ^
                    (static_cast<u64>(request.source_type) << 61);
        value ^= value >> 29;
        value *= 0x9E3779B185EBCA87ULL;
        value ^= value >> 32;
        return static_cast<size_t>(value) & (StreamCopyScratch::HashTableSize - 1);
    };

    if (requests.size() == 1) {
        const auto& request = requests.front();
        const auto [destination, offset] = stream_buffer.Map(request.size, request.alignment);
        ASSERT(destination != nullptr);
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::StagingBytes, request.size);
        switch (request.source_type) {
        case StreamCopySource::Guest:
            memory->CopySparseMemory(request.guest_address, destination, request.size);
            break;
        case StreamCopySource::Host:
            std::memcpy(destination, request.host_address, request.size);
            break;
        case StreamCopySource::Zero:
            std::memset(destination, 0, request.size);
            break;
        }
        stream_buffer.Commit();
        results.front() = {.buffer = &stream_buffer, .offset = offset};
        return;
    }

    u16 canonical_count = 0;
    const bool use_hash = requests.size() > 4;
    for (u16 request_index = 0; request_index < requests.size(); ++request_index) {
        const auto& request = requests[request_index];
        auto& mapping = scratch.request_map[request_index];
        mapping = {};

        s32 canonical_index = -1;
        if (request.deduplicate && request.source_type != StreamCopySource::Zero) {
            if (use_hash) {
                size_t slot_index = hash_request(request);
                for (size_t probe = 0; probe < StreamCopyScratch::HashTableSize; ++probe) {
                    const auto& slot = scratch.hash_table[slot_index];
                    if (slot.generation != scratch.hash_generation) {
                        break;
                    }
                    if (equivalent(scratch.canonical_copies[slot.canonical].request, request)) {
                        canonical_index = slot.canonical;
                        break;
                    }
                    slot_index = (slot_index + 1) & (StreamCopyScratch::HashTableSize - 1);
                }
            } else {
                for (u16 candidate = 0; candidate < canonical_count; ++candidate) {
                    if (equivalent(scratch.canonical_copies[candidate].request, request)) {
                        canonical_index = candidate;
                        break;
                    }
                }
            }

            if (canonical_index < 0) {
                const u64 request_start = source_key(request);
                for (u16 candidate = 0; candidate < canonical_count; ++candidate) {
                    auto& canonical = scratch.canonical_copies[candidate].request;
                    if (!canonical.deduplicate || canonical.source_type != request.source_type ||
                        canonical.source_type == StreamCopySource::Zero) {
                        continue;
                    }
                    const u64 canonical_start = source_key(canonical);
                    if (request_start < canonical_start) {
                        continue;
                    }
                    const u64 source_offset = request_start - canonical_start;
                    if (source_offset > canonical.size ||
                        request.size > canonical.size - source_offset ||
                        source_offset % request.alignment != 0) {
                        continue;
                    }
                    canonical.alignment = std::max(canonical.alignment, request.alignment);
                    canonical_index = candidate;
                    mapping.source_offset = static_cast<u32>(source_offset);
                    break;
                }
            }
        }

        if (canonical_index < 0) {
            canonical_index = canonical_count++;
            auto& canonical = scratch.canonical_copies[canonical_index];
            canonical = {.request = request, .relative_offset = 0};
            if (use_hash && request.deduplicate && request.source_type != StreamCopySource::Zero) {
                size_t slot_index = hash_request(request);
                while (scratch.hash_table[slot_index].generation == scratch.hash_generation) {
                    slot_index = (slot_index + 1) & (StreamCopyScratch::HashTableSize - 1);
                }
                scratch.hash_table[slot_index] = {
                    .generation = scratch.hash_generation,
                    .canonical = static_cast<u16>(canonical_index),
                };
            }
        } else {
            auto& canonical = scratch.canonical_copies[canonical_index].request;
            canonical.alignment = std::max(canonical.alignment, request.alignment);
        }
        mapping.canonical = static_cast<u16>(canonical_index);
    }

    u64 total_size = 0;
    u64 max_alignment = 1;
    for (u16 canonical_index = 0; canonical_index < canonical_count; ++canonical_index) {
        auto& canonical = scratch.canonical_copies[canonical_index];
        max_alignment = std::max(max_alignment, canonical.request.alignment);
        total_size = Common::AlignUp(total_size, canonical.request.alignment);
        canonical.relative_offset = total_size;
        total_size += canonical.request.size;
    }

    const auto [destination, base_offset] = stream_buffer.Map(total_size, max_alignment);
    ASSERT(destination != nullptr);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::StagingBytes,
                                      total_size);

    u16 guest_copy_count = 0;
    u64 guest_copy_size = 0;
    for (u16 canonical_index = 0; canonical_index < canonical_count; ++canonical_index) {
        const auto& canonical = scratch.canonical_copies[canonical_index];
        u8* const copy_destination = destination + canonical.relative_offset;
        switch (canonical.request.source_type) {
        case StreamCopySource::Guest:
            scratch.guest_copies[guest_copy_count++] = Core::MemoryManager::SparseCopyRequest{
                .source = canonical.request.guest_address,
                .destination = copy_destination,
                .size = canonical.request.size,
            };
            guest_copy_size += canonical.request.size;
            break;
        case StreamCopySource::Host:
            std::memcpy(copy_destination, canonical.request.host_address, canonical.request.size);
            break;
        case StreamCopySource::Zero:
            std::memset(copy_destination, 0, canonical.request.size);
            break;
        }
    }
    memory->CopySparseMemoryBatch(
        std::span<const Core::MemoryManager::SparseCopyRequest>{scratch.guest_copies.data(),
                                                                guest_copy_count},
        guest_copy_size);
    stream_buffer.Commit();

    for (u16 request_index = 0; request_index < requests.size(); ++request_index) {
        const auto& mapping = scratch.request_map[request_index];
        ASSERT(mapping.canonical != StreamCopyScratch::NoCanonicalCopy);
        const auto& canonical = scratch.canonical_copies[mapping.canonical];
        results[request_index] = {
            .buffer = &stream_buffer,
            .offset = base_offset + canonical.relative_offset + mapping.source_offset,
        };
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size) {
    if (!IsRegionRegistered(device_addr, size)) {
        return;
    }
    memory_tracker->InvalidateRegion(
        device_addr, size, [this, device_addr, size] { ReadMemory(device_addr, size, true); });
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write) {
    liverpool->SendCommand<true>([this, device_addr, size, is_write] {
        Buffer& buffer = slot_buffers[FindBuffer(device_addr, size)];
        DownloadBufferMemory<false>(buffer, device_addr, size, is_write);
    });
}

template <bool async>
void BufferCache::DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size, bool is_write) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const VAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](VAddr start, VAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(vk::BufferCopy{
                    .srcOffset = new_offset,
                    .dstOffset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
            };
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    const auto [download, offset] = download_buffer.Map(total_size_bytes);
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += offset;
    }
    download_buffer.Commit();
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyCalls);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyBytes,
                                      total_size_bytes);
    cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(), copies);
    const auto write_data = [&]() {
        auto* memory = Core::Memory::Instance();
        for (const auto& copy : copies) {
            const VAddr copy_device_addr = buffer.CpuAddr() + copy.srcOffset;
            const u64 dst_offset = copy.dstOffset - offset;
            memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr), download + dst_offset,
                                    copy.size, Core::MemoryWriteOrigin::GpuCompletion);
        }
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if constexpr (async) {
        scheduler.DeferOperation(write_data);
    } else {
        scheduler.Finish();
        write_data();
    }
}

void BufferCache::PrepareVertexIndexBuffers(const Vulkan::GraphicsPipeline& pipeline,
                                            bool bind_index_buffer, u32 index_offset) {
    auto& state = *vertex_index_state;
    const auto& regs = liverpool->regs;

    state.attributes.clear();
    state.bindings.clear();
    state.divisors.clear();
    state.guest_buffers.clear();
    state.ranges.clear();
    state.ranges_merged.clear();
    state.index = {};
    state.bind_index_buffer = bind_index_buffer;
    state.prepared = true;

    pipeline.GetVertexInputs(state.attributes, state.bindings, state.divisors, state.guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    for (u32 binding_index = 0; binding_index < state.guest_buffers.size(); ++binding_index) {
        const auto& buffer = state.guest_buffers[binding_index];
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            state.ranges.emplace_back(VertexIndexState::BufferRange{
                .base_address = buffer.base_address,
                .end_address = buffer.base_address + buffer.GetSize(),
                .binding_mask = 1U << binding_index,
            });
        }
    }

    if (!state.ranges.empty()) {
        const auto less_by_address = [](const auto& lhs, const auto& rhs) {
            return lhs.base_address < rhs.base_address;
        };
        if (state.ranges.size() <= 8) {
            for (u32 range_index = 1; range_index < state.ranges.size(); ++range_index) {
                auto range = state.ranges[range_index];
                u32 insert_index = range_index;
                while (insert_index != 0 &&
                       less_by_address(range, state.ranges[insert_index - 1])) {
                    state.ranges[insert_index] = state.ranges[insert_index - 1];
                    --insert_index;
                }
                state.ranges[insert_index] = range;
            }
        } else {
            std::ranges::sort(state.ranges, less_by_address);
        }
        state.ranges_merged.emplace_back(state.ranges.front());
        for (u32 range_index = 1; range_index < state.ranges.size(); ++range_index) {
            const auto& range = state.ranges[range_index];
            auto& previous = state.ranges_merged.back();
            if (previous.end_address < range.base_address) {
                state.ranges_merged.emplace_back(range);
            } else {
                previous.end_address = std::max(previous.end_address, range.end_address);
                previous.binding_mask |= range.binding_mask;
            }
        }
    }

    for (auto& range : state.ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        ASSERT(size <= std::numeric_limits<u32>::max());
        range.size = static_cast<u32>(size);
        range.was_gpu_modified = IsRegionGpuModified(range.base_address, size);
        if (!range.was_gpu_modified && size <= CACHING_PAGESIZE) {
            range.stream_index = QueueStreamCopy(StreamCopyRequest{
                .source_type = StreamCopySource::Guest,
                .guest_address = range.base_address,
                .size = range.size,
                .alignment = instance.UniformMinAlignment(),
            });
        }
    }

    if (!bind_index_buffer) {
        return;
    }

    auto& index = state.index;
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    index.type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    index.address = regs.index_base_address.Address<VAddr>() + index_offset * index_size;
    index.size = regs.num_indices * index_size;
    index.was_gpu_modified = IsRegionGpuModified(index.address, index.size);
    if (index.size != 0 && !index.was_gpu_modified && index.size <= CACHING_PAGESIZE) {
        index.stream_index = QueueStreamCopy(StreamCopyRequest{
            .source_type = StreamCopySource::Guest,
            .guest_address = index.address,
            .size = index.size,
            .alignment = std::max<u64>(index_size, instance.UniformMinAlignment()),
        });
    }
}

void BufferCache::FinalizeVertexIndexBuffers(
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    auto& state = *vertex_index_state;
    ASSERT(state.prepared);
    ASSERT(stream_copy_finalized);

    for (auto& range : state.ranges_merged) {
        if (range.stream_index != VertexIndexState::NoStreamCopy) {
            const auto& result = GetStreamCopyResult(range.stream_index);
            range.buffer = result.buffer;
            range.vk_buffer = result.buffer->Handle();
            range.offset = result.offset;
        } else {
            const BufferId buffer_id = FindBuffer(range.base_address, range.size);
            range.buffer = &slot_buffers[buffer_id];
            SynchronizeBuffer(*range.buffer, range.base_address, range.size, false, false);
            range.vk_buffer = range.buffer->Handle();
            range.offset = range.buffer->Offset(range.base_address);
        }
        ASSERT(range.buffer != nullptr);
        if (range.was_gpu_modified) {
            if (auto barrier =
                    range.buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                             vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                barriers.emplace_back(*barrier);
            }
        }
    }

    auto& index = state.index;
    if (state.bind_index_buffer) {
        if (index.size == 0) {
            const auto [buffer, offset] = ObtainBuffer(index.address, 0, false);
            index.buffer = buffer;
            index.offset = offset;
        } else if (index.stream_index != VertexIndexState::NoStreamCopy) {
            const auto& result = GetStreamCopyResult(index.stream_index);
            index.buffer = result.buffer;
            index.offset = result.offset;
        } else {
            const BufferId buffer_id = FindBuffer(index.address, index.size);
            index.buffer = &slot_buffers[buffer_id];
            SynchronizeBuffer(*index.buffer, index.address, index.size, false, false);
            index.offset = index.buffer->Offset(index.address);
        }
        ASSERT(index.buffer != nullptr);
        if (index.was_gpu_modified) {
            if (auto barrier = index.buffer->GetBarrier(vk::AccessFlagBits2::eIndexRead,
                                                        vk::PipelineStageFlagBits2::eIndexInput)) {
                barriers.emplace_back(*barrier);
            }
        }
    }

    const u64 current_tick = scheduler.CurrentTick();
    if (state.emitted_tick != current_tick) {
        state.emitted_tick = current_tick;
        state.vertex_input_valid = false;
        state.vertex_buffers_valid = false;
        state.index_buffer_valid = false;
    }

    const auto attributes_equal = [](const auto& lhs, const auto& rhs, u32 rhs_count) {
        if (lhs.size() != rhs_count) {
            return false;
        }
        for (u32 i = 0; i < rhs_count; ++i) {
            const auto& left = lhs[i];
            const auto& right = rhs[i];
            if (left.location != right.location || left.binding != right.binding ||
                left.format != right.format || left.offset != right.offset) {
                return false;
            }
        }
        return true;
    };
    const auto bindings_equal = [](const auto& lhs, const auto& rhs, u32 rhs_count) {
        if (lhs.size() != rhs_count) {
            return false;
        }
        for (u32 i = 0; i < rhs_count; ++i) {
            const auto& left = lhs[i];
            const auto& right = rhs[i];
            if (left.binding != right.binding || left.stride != right.stride ||
                left.inputRate != right.inputRate || left.divisor != right.divisor) {
                return false;
            }
        }
        return true;
    };

    const auto cmdbuf = scheduler.CommandBuffer();
    if (instance.IsVertexInputDynamicState() &&
        (!state.vertex_input_valid ||
         !attributes_equal(state.attributes, state.emitted_attributes,
                           state.emitted_attribute_count) ||
         !bindings_equal(state.bindings, state.emitted_bindings, state.emitted_binding_count))) {
        cmdbuf.setVertexInputEXT(state.bindings, state.attributes);
        state.emitted_attribute_count = static_cast<u32>(state.attributes.size());
        state.emitted_binding_count = static_cast<u32>(state.bindings.size());
        std::ranges::copy(state.attributes, state.emitted_attributes.begin());
        std::ranges::copy(state.bindings, state.emitted_bindings.begin());
        state.vertex_input_valid = true;
    }

    if (!state.bindings.empty()) {
        const u32 num_buffers = static_cast<u32>(state.guest_buffers.size());
        auto& host_buffers = state.host_buffers;
        auto& host_offsets = state.host_offsets;
        auto& host_sizes = state.host_sizes;
        auto& host_strides = state.host_strides;
        for (u32 i = 0; i < num_buffers; ++i) {
            const auto& buffer = state.guest_buffers[i];
            host_buffers[i] = VK_NULL_HANDLE;
            host_offsets[i] = 0;
            host_sizes[i] = buffer.GetSize();
            host_strides[i] = buffer.GetStride();
        }

        for (const auto& range : state.ranges_merged) {
            u32 binding_mask = range.binding_mask;
            while (binding_mask != 0) {
                const u32 binding_index = std::countr_zero(binding_mask);
                binding_mask &= binding_mask - 1;
                const auto& buffer = state.guest_buffers[binding_index];
                host_buffers[binding_index] = range.vk_buffer;
                host_offsets[binding_index] =
                    range.offset + buffer.base_address - range.base_address;
            }
        }

        const auto values_equal = [num_buffers](const auto& lhs, const auto& rhs) {
            return std::equal(lhs.begin(), lhs.begin() + num_buffers, rhs.begin());
        };
        const bool same_buffers = state.vertex_buffers_valid &&
                                  state.emitted_buffer_count == num_buffers &&
                                  values_equal(host_buffers, state.emitted_buffers) &&
                                  values_equal(host_offsets, state.emitted_offsets) &&
                                  (instance.IsVertexInputDynamicState() ||
                                   (values_equal(host_sizes, state.emitted_sizes) &&
                                    values_equal(host_strides, state.emitted_strides)));
        if (!same_buffers) {
            if (instance.IsVertexInputDynamicState()) {
                cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
            } else {
                cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                          host_sizes.data(), host_strides.data());
            }
            state.emitted_buffer_count = num_buffers;
            std::copy_n(host_buffers.begin(), num_buffers, state.emitted_buffers.begin());
            std::copy_n(host_offsets.begin(), num_buffers, state.emitted_offsets.begin());
            std::copy_n(host_sizes.begin(), num_buffers, state.emitted_sizes.begin());
            std::copy_n(host_strides.begin(), num_buffers, state.emitted_strides.begin());
            state.vertex_buffers_valid = true;
        }
    }

    if (state.bind_index_buffer) {
        const vk::Buffer handle = index.buffer->Handle();
        if (!state.index_buffer_valid || state.emitted_index_buffer != handle ||
            state.emitted_index_offset != index.offset || state.emitted_index_type != index.type) {
            cmdbuf.bindIndexBuffer(handle, index.offset, index.type);
            state.emitted_index_buffer = handle;
            state.emitted_index_offset = index.offset;
            state.emitted_index_type = index.type;
            state.index_buffer_valid = true;
        }
    }

    state.prepared = false;
}

void BufferCache::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    ASSERT_MSG(address % 4 == 0, "GDS offset must be dword aligned");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + num_bytes / sizeof(u32), value);
            return;
        }
    }
    Buffer* buffer = [&] {
        if (is_gds) {
            return &gds_buffer;
        }
        const auto [buffer, offset] = ObtainBuffer(address, num_bytes, true);
        return buffer;
    }();
    buffer->Fill(buffer->Offset(address), num_bytes, value);
}

void BufferCache::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds && !IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
        // Without a readback there's nothing we can do with this
        // Fallback to creating dst buffer on GPU to at least have this data there
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    auto& src_buffer = [&] -> const Buffer& {
        if (src_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(src, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, src, num_bytes, false, true);
        return buffer;
    }();
    auto& dst_buffer = [&] -> const Buffer& {
        if (dst_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(dst, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, dst, num_bytes, true, true);
        gpu_modified_ranges.Add(dst, num_bytes);
        return buffer;
    }();
    const vk::BufferCopy region = {
        .srcOffset = src_buffer.Offset(src),
        .dstOffset = dst_buffer.Offset(dst),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 buf_barriers_before[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::BarrierCalls, 2);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyCalls);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyBytes,
                                      num_bytes);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_before,
    });
    cmdbuf.copyBuffer(src_buffer.Handle(), dst_buffer.Handle(), region);
    const vk::BufferMemoryBarrier2 buf_barriers_after[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_after,
    });
}

std::pair<Buffer*, u32> BufferCache::ObtainBuffer(VAddr device_addr, u32 size, bool is_written,
                                                  bool is_texel_buffer, BufferId buffer_id) {
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= CACHING_PAGESIZE && !IsRegionGpuModified(device_addr, size)) {
        if (size == 0) {
            const u64 offset =
                stream_buffer.Copy(device_addr, size, instance.UniformMinAlignment());
            return {&stream_buffer, static_cast<u32>(offset)};
        }

        auto& reuse = *stream_slice_reuse;
        u64 hash = device_addr ^ (static_cast<u64>(size) << 32);
        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccdULL;
        hash ^= hash >> 33;
        const size_t set_index = static_cast<size_t>(hash) & (StreamSliceReuseState::SetCount - 1);
        auto& set = reuse.sets[set_index];

        memory->CopySparseMemory(device_addr, reuse.scratch.data(), size);
        StreamSliceReuseState::Entry* replacement{};
        size_t replacement_way{};
        for (size_t way = 0; way < StreamSliceReuseState::WayCount; ++way) {
            auto& entry = set.ways[way];
            if (entry.valid && entry.address == device_addr && entry.size == size) {
                if (entry.generation == stream_buffer.Generation() &&
                    entry.tick == scheduler.CurrentTick() &&
                    std::memcmp(reuse.Shadow(set_index, way), reuse.scratch.data(), size) == 0) {
                    Common::PerformanceTelemetry::Add(
                        Common::PerformanceTelemetry::Counter::StreamSliceHits);
                    return {&stream_buffer, entry.offset};
                }
                replacement = &entry;
                replacement_way = way;
                break;
            }
            if (!entry.valid && replacement == nullptr) {
                replacement = &entry;
                replacement_way = way;
            }
        }
        if (replacement == nullptr) {
            replacement_way = set.next_replacement;
            set.next_replacement = (set.next_replacement + 1) % StreamSliceReuseState::WayCount;
            replacement = &set.ways[replacement_way];
        }

        const auto [destination, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        ASSERT(destination != nullptr);
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::StreamSliceMisses);
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::StagingBytes, size);
        std::memcpy(destination, reuse.scratch.data(), size);
        stream_buffer.Commit();
        std::memcpy(reuse.Shadow(set_index, replacement_way), reuse.scratch.data(), size);
        *replacement = {
            .address = device_addr,
            .generation = stream_buffer.Generation(),
            .tick = scheduler.CurrentTick(),
            .size = size,
            .offset = static_cast<u32>(offset),
            .valid = true,
        };
        return {&stream_buffer, static_cast<u32>(offset)};
    }
    if (IsBufferInvalid(buffer_id)) {
        buffer_id = FindBuffer(device_addr, size);
    }
    Buffer& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
    }
    return {&buffer, buffer.Offset(device_addr)};
}

std::pair<Buffer*, u32> BufferCache::ObtainBufferForImage(VAddr gpu_addr, u32 size) {
    // Check if any buffer contains the full requested range.
    const BufferId buffer_id = page_table[gpu_addr >> CACHING_PAGEBITS].buffer_id;
    if (buffer_id) {
        if (Buffer& buffer = slot_buffers[buffer_id]; buffer.IsInBounds(gpu_addr, size)) {
            SynchronizeBuffer(buffer, gpu_addr, size, false, false);
            return {&buffer, buffer.Offset(gpu_addr)};
        }
    }
    // If some buffer within was GPU modified create a full buffer to avoid losing GPU data.
    if (IsRegionGpuModified(gpu_addr, size)) {
        return ObtainBuffer(gpu_addr, size, false, false);
    }
    // In all other cases, just do a CPU copy to the staging buffer.
    const auto [data, offset] = staging_buffer.Map(size, 16);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::StagingBytes, size);
    memory->CopySparseMemory(gpu_addr, data, size);
    staging_buffer.Commit();
    return {&staging_buffer, offset};
}

bool BufferCache::IsRegionRegistered(VAddr addr, size_t size) {
    // Check if we are missing some edge case here
    return buffer_ranges.Intersects(addr, size);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

bool BufferCache::IsBufferCacheEntryValid(BufferId id, u64 uid, VAddr address, u64 size) const {
    if (!id || !slot_buffers.is_allocated(id)) {
        return false;
    }
    const auto& buffer = slot_buffers[id];
    return !buffer.is_deleted && buffer.Uid() == uid && buffer.IsInBounds(address, size);
}

u64 BufferCache::GetBufferUid(BufferId id) const {
    ASSERT(id && slot_buffers.is_allocated(id));
    return slot_buffers[id].Uid();
}

BufferId BufferCache::FindBuffer(VAddr device_addr, u32 size) {
    ASSERT(device_addr != 0);
    const u64 page = device_addr >> CACHING_PAGEBITS;
    const BufferId buffer_id = page_table[page].buffer_id;
    if (!buffer_id) {
        return CreateBuffer(device_addr, size);
    }
    const Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.IsInBounds(device_addr, size)) {
        return buffer_id;
    }
    return CreateBuffer(device_addr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(VAddr device_addr, u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    VAddr begin = device_addr;
    VAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    const auto expand_begin = [&](VAddr add_value) {
        static constexpr VAddr min_page = CACHING_PAGESIZE + DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    const auto expand_end = [&](VAddr add_value) {
        static constexpr VAddr max_page = 1ULL << MemoryTracker::MAX_CPU_PAGE_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.is_picked) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.is_picked = true;
        const VAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const VAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_end(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

void BufferCache::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                              bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    const vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = dst_base_offset,
        .size = overlap.SizeBytes(),
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> pre_barriers{};
    if (auto src_barrier = overlap.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                              vk::PipelineStageFlagBits2::eTransfer)) {
        pre_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier =
            new_buffer.GetBarrier(vk::AccessFlagBits2::eTransferWrite,
                                  vk::PipelineStageFlagBits2::eTransfer, dst_base_offset)) {
        pre_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });

    cmdbuf.copyBuffer(overlap.Handle(), new_buffer.Handle(), copy);

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> post_barriers{};
    if (auto src_barrier =
            overlap.GetBarrier(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                               vk::PipelineStageFlagBits2::eAllCommands)) {
        post_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier = new_buffer.GetBarrier(
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands, dst_base_offset)) {
        post_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(post_barriers.size()),
        .pBufferMemoryBarriers = post_barriers.data(),
    });
    DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(VAddr device_addr, u32 wanted_size) {
    const VAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(instance, scheduler, MemoryUsage::DeviceLocal, overlap.begin,
                            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);
    auto& new_buffer = slot_buffers[new_buffer_id];
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    Register(new_buffer_id);
    return new_buffer_id;
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager.ProcessFaultBuffer();
}

void BufferCache::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
    ++topology_epoch;
}

void BufferCache::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
    ++topology_epoch;
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    const VAddr device_addr_begin = buffer.CpuAddr();
    const VAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    const u64 size_pages = page_end - page_begin;
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page].buffer_id = buffer_id;
        } else {
            page_table[page].buffer_id = BufferId{};
        }
    }
    if constexpr (insert) {
        total_used_memory += Common::AlignUp(size, CACHING_PAGESIZE);
        buffer.SetLRUId(lru_cache.Insert(buffer_id, gc_tick));
        boost::container::small_vector<vk::DeviceAddress, 128> bda_addrs;
        bda_addrs.reserve(size_pages);
        for (u64 i = 0; i < size_pages; ++i) {
            vk::DeviceAddress addr = buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS);
            bda_addrs.push_back(addr);
        }
        WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                        bda_addrs.data(), bda_addrs.size() * sizeof(vk::DeviceAddress));
        buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
    } else {
        total_used_memory -= Common::AlignUp(size, CACHING_PAGESIZE);
        lru_cache.Free(buffer.LRUId());
        const u64 offset = bda_pagetable_buffer.Offset(page_begin * sizeof(vk::DeviceAddress));
        bda_pagetable_buffer.Fill(offset, size_pages * sizeof(vk::DeviceAddress), 0);
        buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
    }
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                                    bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes = 0;
    VAddr buffer_start = buffer.CpuAddr();
    vk::Buffer src_buffer = VK_NULL_HANDLE;
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 device_addr_out, u64 range_size) {
            copies.emplace_back(total_size_bytes, device_addr_out - buffer_start, range_size);
            total_size_bytes += range_size;
        },
        [&] { src_buffer = UploadCopies(buffer, copies, total_size_bytes); });

    if (src_buffer) {
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::BarrierCalls, 2);
        Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyCalls);
        Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyBytes,
                                          total_size_bytes);
        const vk::BufferMemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eTransferRead |
                             vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        const vk::BufferMemoryBarrier2 post_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });
        cmdbuf.copyBuffer(src_buffer, buffer.buffer, copies);
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &post_barrier,
        });
        TouchBuffer(buffer);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeBufferFromImage(buffer, device_addr, size);
    }
    return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     size_t total_size_bytes) {
    if (copies.empty()) {
        return VK_NULL_HANDLE;
    }
    const auto [staging, offset] = staging_buffer.Map(total_size_bytes);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::StagingBytes,
                                      total_size_bytes);
    if (staging) {
        for (auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
            // Apply the staging offset
            copy.srcOffset += offset;
        }
        staging_buffer.Commit();
        return staging_buffer.Handle();
    } else {
        // For large one time transfers use a temporary host buffer.
        auto temp_buffer =
            std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                     vk::BufferUsageFlagBits::eTransferSrc, total_size_bytes);
        const vk::Buffer src_buffer = temp_buffer->Handle();
        u8* const staging = temp_buffer->mapped_data.data();
        for (const auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
        }
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable { buffer.reset(); });
        return src_buffer;
    }
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size) {
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u32 buf_offset = buffer.Offset(image.info.guest_address);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    u32 copy_size = 0;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (buf_offset + mip_info.offset + mip_info.size > buffer.SizeBytes()) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
        copy_size += mip_info.size;
    }
    if (copy_size == 0) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, buffer.Handle(), buf_offset, copy_size);
    return true;
}

void BufferCache::SynchronizeBuffersInRange(VAddr device_addr, u64 size) {
    const VAddr device_addr_end = device_addr + size;
    ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
        RENDERER_TRACE;
        VAddr start = std::max(buffer.CpuAddr(), device_addr);
        VAddr end = std::min(buffer.CpuAddr() + buffer.SizeBytes(), device_addr_end);
        u32 size = static_cast<u32>(end - start);
        SynchronizeBuffer(buffer, start, size, false, false);
    });
}

void BufferCache::WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes) {
    vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = buffer.Offset(address),
        .size = num_bytes,
    };
    vk::Buffer src_buffer = staging_buffer.Handle();
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::StagingBytes,
                                      num_bytes);
    if (num_bytes < StagingBufferSize) {
        const auto [staging, offset] = staging_buffer.Map(num_bytes);
        std::memcpy(staging, value, num_bytes);
        copy.srcOffset = offset;
        staging_buffer.Commit();
    } else {
        // For large one time transfers use a temporary host buffer.
        // RenderDoc can lag quite a bit if the stream buffer is too large.
        Buffer temp_buffer{
            instance, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc,
            num_bytes};
        src_buffer = temp_buffer.Handle();
        u8* const staging = temp_buffer.mapped_data.data();
        std::memcpy(staging, value, num_bytes);
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable {});
    }
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::BarrierCalls, 2);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyCalls);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyBytes,
                                      num_bytes);
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(src_buffer, buffer.Handle(), copy);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

void BufferCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    const bool aggressive = total_used_memory >= critical_gc_memory;
    const u64 ticks_to_destroy = std::min<u64>(aggressive ? 80 : 160, gc_tick);
    int max_deletions = aggressive ? 64 : 32;
    const auto clean_up = [&](BufferId buffer_id) {
        if (max_deletions == 0) {
            return;
        }
        --max_deletions;
        Buffer& buffer = slot_buffers[buffer_id];
        // InvalidateMemory(buffer.CpuAddr(), buffer.SizeBytes());
        DownloadBufferMemory<true>(buffer, buffer.CpuAddr(), buffer.SizeBytes(), true);
        DeleteBuffer(buffer_id);
    };
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
    lru_cache.Touch(buffer.LRUId(), gc_tick);
}

void BufferCache::DeleteBuffer(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    Unregister(buffer_id);
    scheduler.DeferOperation([this, buffer_id] { slot_buffers.erase(buffer_id); });
    buffer.is_deleted = true;
}

} // namespace VideoCore
