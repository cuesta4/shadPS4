// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/resources/transfer_buffer_pool.h"

namespace VideoCore {

static constexpr size_t StagingBufferSize = 512_MB;
static constexpr size_t DownloadBufferSize = 32_MB;
static constexpr size_t UboStreamBufferSize = 64_MB;
static constexpr size_t DeviceBufferSize = 128_MB;

TransferBufferPool::TransferBufferPool(const Vulkan::Instance& instance,
                                       Vulkan::Scheduler& scheduler)
    : staging{instance, scheduler, MemoryUsage::Upload, StagingBufferSize},
      stream{instance, scheduler, MemoryUsage::Stream, UboStreamBufferSize},
      download{instance, scheduler, MemoryUsage::Download, DownloadBufferSize},
      device_local{instance, scheduler, MemoryUsage::DeviceLocal, DeviceBufferSize} {}

StreamBuffer& TransferBufferPool::Get(MemoryUsage usage) noexcept {
    switch (usage) {
    case MemoryUsage::Stream:
        return stream;
    case MemoryUsage::Download:
        return download;
    case MemoryUsage::DeviceLocal:
        return device_local;
    default:
        ASSERT(usage == MemoryUsage::Upload);
        return staging;
    }
}

const StreamBuffer& TransferBufferPool::Get(MemoryUsage usage) const noexcept {
    return const_cast<TransferBufferPool*>(this)->Get(usage);
}

} // namespace VideoCore
