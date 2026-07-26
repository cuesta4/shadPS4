// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/buffer_cache/buffer.h"

namespace Vulkan {
class Instance;
class Scheduler;
}

namespace VideoCore {

class TransferBufferPool {
public:
    TransferBufferPool(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler);

    StreamBuffer& Get(MemoryUsage usage) noexcept;
    const StreamBuffer& Get(MemoryUsage usage) const noexcept;

private:
    StreamBuffer staging;
    StreamBuffer stream;
    StreamBuffer download;
    StreamBuffer device_local;
};

} // namespace VideoCore
