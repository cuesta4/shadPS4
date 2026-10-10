// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>
#include <memory>

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

enum class GpuWork : u32 {
    Draw,
    Dispatch,
    Detile,
    Tile,
    PostFx,
    PostProcess,
    Composition,
    BufferCopy,
    ImageUpload,
    ImageReadback,
    ImageCopy,
    Resolve,
    Fill,
    Clear,
    Barrier,
    RenderBegin,
    RenderEnd,
    TransferUpload,
    ImageBlit,
    BufferFault,
    Count,
};

template <typename Handle>
u64 GpuHandle(Handle handle) {
    return std::bit_cast<u64>(handle);
}

struct GpuTimingLabel {
    GpuWork kind{GpuWork::Count};
    u64 pipeline{};
    u64 shader0{};
    u64 shader1{};
    u64 resource0{};
    u64 resource1{};
};

class GpuTiming {
public:
    static bool Requested();
    GpuTiming(const Instance& instance, vk::Semaphore timeline, bool presentation);
    ~GpuTiming();
    bool IsEnabled() const;

    void BeginCommandBuffer(vk::CommandBuffer cmdbuf);
    void Begin(vk::CommandBuffer cmdbuf, GpuTimingLabel label, u64 work);
    void End(vk::CommandBuffer cmdbuf);
    void EndFrame(vk::CommandBuffer cmdbuf);
    void OnSubmit(u64 tick);
    void BeginTransfer(vk::CommandBuffer cmdbuf, GpuTimingLabel label, u64 bytes);
    void EndTransfer(vk::CommandBuffer cmdbuf);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

class GpuTimingScope {
public:
    GpuTimingScope(Scheduler& scheduler, GpuWork kind, u64 resource0 = 0, u64 resource1 = 0,
                   u64 work = 0);
    ~GpuTimingScope();
    GpuTimingScope(const GpuTimingScope&) = delete;
    GpuTimingScope& operator=(const GpuTimingScope&) = delete;

private:
    Scheduler* scheduler{};
};

class GpuTimingContext {
public:
    GpuTimingContext(Scheduler& scheduler, GpuTimingLabel label);
    ~GpuTimingContext();
    GpuTimingContext(const GpuTimingContext&) = delete;
    GpuTimingContext& operator=(const GpuTimingContext&) = delete;

private:
    Scheduler* scheduler{};
    GpuTimingLabel previous{};
};

} // namespace Vulkan
