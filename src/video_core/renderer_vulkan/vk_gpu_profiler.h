// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

#include "common/performance_telemetry.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class MasterSemaphore;

[[nodiscard]] bool PipelineExecutableCaptureEnabled() noexcept;
void RecordPipelineExecutableStatistics(const Instance& instance, vk::Pipeline pipeline,
                                        u64 pipeline_hash, bool is_compute);

// A sampled, non-blocking GPU profiler. Query pools are retired through the scheduler timeline;
// result collection never requests VK_QUERY_RESULT_WAIT_BIT.
class GpuProfiler {
public:
    GpuProfiler(const Instance& instance, MasterSemaphore& master_semaphore);
    ~GpuProfiler();

    void BeginCommandBuffer(vk::CommandBuffer cmdbuf,
                            Common::PerformanceTelemetry::CmdBufferSeq command_buffer_seq,
                            Common::PerformanceTelemetry::FrameSeq frame_seq);
    void EndCommandBuffer(Common::PerformanceTelemetry::SubmitSeq submit_seq, u64 retire_tick);
    void Collect();

    void BeginRendering(u64 attachment_hash);
    void EndRendering();
    void GraphicsDraw(u64 pipeline_hash, u32 command_count = 1);
    void ComputeDispatch(u64 pipeline_hash, u32 command_count = 1);

    [[nodiscard]] u64 BeginInterval(Common::PerformanceTelemetry::GpuIntervalKind kind,
                                    u64 object_hash = 0, u64 bytes = 0);
    void EndInterval(u64 token);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
