// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/gpu_memory_observer.h"
#include "video_core/memory/mapped_range_registry.h"

namespace VideoCore {

class BufferCache;
class PageManager;
class TextureCache;

/**
 * Narrow memory/fault orchestration boundary.
 *
 * PageManager reports faults through this observer. The coordinator validates GPU mapping state,
 * then forwards ordered cache operations. It never records Vulkan commands directly.
 */
class GpuMemoryCoordinator final : public GpuMemoryObserver {
public:
    explicit GpuMemoryCoordinator(MappedRangeRegistry& mapped_ranges_)
        : mapped_ranges{mapped_ranges_} {}

    void Bind(BufferCache& buffers_, TextureCache& images_, PageManager& pages_) noexcept;
    void Unbind() noexcept;

    bool InvalidateMemory(VAddr address, u64 size) override;
    bool ReadMemory(VAddr address, u64 size) override;
    bool IsMapped(VAddr address, u64 size) override;
    void MapMemory(VAddr address, u64 size) override;
    void UnmapMemory(VAddr address, u64 size) override;

private:
    MappedRangeRegistry& mapped_ranges;
    BufferCache* buffers{};
    TextureCache* images{};
    PageManager* pages{};
};

} // namespace VideoCore
