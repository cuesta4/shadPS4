// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <span>
#include <utility>

#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/gpu_queue.h"
#include "video_core/amdgpu/gpu_thread_dispatcher.h"
#include "video_core/gpu_commands/command_sink.h"

namespace VideoCore {
class GpuCommandSink;
}

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

class GpuCommandProcessor;

/** Public GPU device facade and composition root. */
class Liverpool final : public GpuThreadDispatcher, public VideoCore::GpuCommandSinkBinder {
    std::unique_ptr<GpuCommandProcessor> processor;

public:
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;
    static constexpr u32 NumComputePipes = 7u;
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;

    Liverpool();
    ~Liverpool();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);
    void SubmitDone() noexcept;
    void WaitGpuIdle() noexcept;
    [[nodiscard]] bool IsGpuIdle() const;
    void SetVoPort(Libraries::VideoOut::VideoOutPort* port);
    void BindCommandSink(VideoCore::GpuCommandSink* command_sink) override;
    void ReserveCopyBufferSpace();

    void ExecuteOnGpuThread(Common::UniqueFunction<void> command) override;

    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        Enqueue(Common::UniqueFunction<void>{std::forward<decltype(func)>(func)}, wait_done);
    }

    Common::SlotVector<AscQueueInfo>& asc_queues;

private:
    void Enqueue(Common::UniqueFunction<void> command, bool wait);
};

} // namespace AmdGpu
