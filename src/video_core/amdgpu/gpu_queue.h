// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <queue>
#include <vector>

#include "common/types.h"
#include "common/slot_vector.h"
#include "video_core/amdgpu/gpu_task.h"
#include "video_core/amdgpu/regs.h"

namespace AmdGpu {

struct AscQueueInfo {
    static constexpr size_t Pm4BufferSize = 1024;
    VAddr map_addr{};
    u32* read_addr{};
    u32 ring_size_dw{};
    u32 pipe_id{};
    std::array<u32, Pm4BufferSize> tmp_packet{};
    u32 tmp_dwords{};
};

struct GpuSubmission {
    GpuTask::Handle task{};
    u64 sequence{};
};

struct GpuQueue {
    std::mutex access_mutex{};
    std::atomic<u32> dcb_buffer_offset{};
    std::atomic<u32> ccb_buffer_offset{};
    std::vector<u32> dcb_buffer;
    std::vector<u32> ccb_buffer;
    std::queue<GpuSubmission> submissions{};
    ComputeProgram compute_state{};
};

template <size_t Count>
class GpuQueueManager {
public:
    GpuQueue& operator[](size_t index) noexcept {
        return queues[index];
    }

    const GpuQueue& operator[](size_t index) const noexcept {
        return queues[index];
    }

    static constexpr size_t Size() noexcept {
        return Count;
    }

private:
    std::array<GpuQueue, Count> queues{};
};

} // namespace AmdGpu
