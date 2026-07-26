// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/memory/gpu_memory_coordinator.h"
#include "video_core/page_manager.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {

void GpuMemoryCoordinator::Bind(BufferCache& buffers_, TextureCache& images_,
                                PageManager& pages_) noexcept {
    ASSERT(buffers == nullptr && images == nullptr && pages == nullptr);
    buffers = &buffers_;
    images = &images_;
    pages = &pages_;
}

void GpuMemoryCoordinator::Unbind() noexcept {
    buffers = nullptr;
    images = nullptr;
    pages = nullptr;
}

bool GpuMemoryCoordinator::InvalidateMemory(VAddr address, u64 size) {
    if (!IsMapped(address, size)) {
        return false;
    }
    ASSERT(buffers != nullptr && images != nullptr);
    buffers->InvalidateMemory(address, size);
    images->InvalidateMemory(address, size);
    return true;
}

bool GpuMemoryCoordinator::ReadMemory(VAddr address, u64 size) {
    if (!IsMapped(address, size)) {
        return false;
    }
    ASSERT(buffers != nullptr);
    buffers->ReadMemory(address, size);
    return true;
}

bool GpuMemoryCoordinator::IsMapped(VAddr address, u64 size) {
    return mapped_ranges.Contains(address, size);
}

void GpuMemoryCoordinator::MapMemory(VAddr address, u64 size) {
    ASSERT(pages != nullptr);
    mapped_ranges.Map(address, size);
    pages->OnGpuMap(address, size);
}

void GpuMemoryCoordinator::UnmapMemory(VAddr address, u64 size) {
    ASSERT(buffers != nullptr && images != nullptr && pages != nullptr);
    buffers->InvalidateMemory(address, size);
    images->UnmapMemory(address, size);
    pages->OnGpuUnmap(address, size);
    mapped_ranges.Unmap(address, size);
}

} // namespace VideoCore
