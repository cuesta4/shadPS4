// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/debug.h"
#include "common/performance_telemetry.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, master_semaphore{instance}, command_pool{instance, &master_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    AllocateWorkerCommandBuffers();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    Common::PerformanceTelemetry::SampledDuration<
        Common::PerformanceTelemetry::TimerSite::SchedulerBeginRendering>
        duration;
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;

    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    current_cmdbuf.beginRendering(rendering_info);
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    current_cmdbuf.endRendering();
}

void Scheduler::Flush(SubmitInfo& info, Common::PerformanceTelemetry::SubmitReason reason) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info, reason);
}

void Scheduler::Flush(Common::PerformanceTelemetry::SubmitReason reason) {
    SubmitInfo info{};
    Flush(info, reason);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info, Common::PerformanceTelemetry::SubmitReason::Finish);
    Wait(presubmit_tick, Common::PerformanceTelemetry::HostWaitReason::SchedulerFinish);
}

void Scheduler::Wait(u64 tick, Common::PerformanceTelemetry::HostWaitReason reason) {
    if (tick >= master_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info, Common::PerformanceTelemetry::SubmitReason::WaitProgress);
    }
    master_semaphore.Wait(tick, reason);
}

void Scheduler::PopPendingOperations() {
    std::unique_lock lk(pending_ops_mutex);
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    if (pending_ops.empty()) [[likely]] {
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::PendingOpEmptyHits, 1);
        }
        return;
    }

    bool refreshed = false;
    if (master_semaphore.IsFree(pending_ops.front().gpu_tick)) [[likely]] {
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::PendingOpKnownTickHits, 1);
        }
    } else {
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::PendingOpRefreshes, 1);
        }
        master_semaphore.Refresh();
        refreshed = true;
    }
    while (!pending_ops.empty() && master_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
    if (!pending_ops.empty() && !refreshed) {
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::PendingOpRefreshes, 1);
        }
        master_semaphore.Refresh();
        while (!pending_ops.empty() && master_semaphore.IsFree(pending_ops.front().gpu_tick)) {
            pending_ops.front().callback();
            pending_ops.pop();
        }
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    Common::PerformanceTelemetry::NextCmdBufferSeq();
    current_cmdbuf = command_pool.Commit();
    Check(current_cmdbuf.begin(begin_info));

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();
    Common::PerformanceTelemetry::Add(
        Common::PerformanceTelemetry::Counter::DynamicStateInvalidations);

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::SubmitExecution(SubmitInfo& info,
                                Common::PerformanceTelemetry::SubmitReason reason) {
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    const u64 wait_start = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    std::unique_lock lk{instance.GetGraphicsQueueMutex()};
    const u64 lock_acquired = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    const u64 signal_value = master_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    EndRendering();
    Check(current_cmdbuf.end());

    const vk::Semaphore timeline = master_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = info.wait_stages.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &current_cmdbuf,
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    master_semaphore.TelemetrySubmit(signal_value);
    const u64 driver_start = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    const auto submit_result = [&] {
        Common::PerformanceTelemetry::ScopedDuration submit_duration{
            Common::PerformanceTelemetry::Counter::DriverSubmitNs};
        return instance.GetGraphicsQueue().submit(submit_info, info.fence);
    }();
    const u64 driver_end = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    master_semaphore.Refresh();
    AllocateWorkerCommandBuffers();

    // Apply pending operations
    PopPendingOperations();
    if (telemetry_enabled) {
        const u64 post_end = Common::PerformanceTelemetry::Timestamp();
        lk.unlock();
        Common::PerformanceTelemetry::RecordSubmitTimingEnabled(
            reason, lock_acquired - wait_start, driver_start - lock_acquired,
            driver_end - driver_start, post_end - driver_end, post_end - lock_acquired);
        Common::PerformanceTelemetry::RecordEnabled(
            Common::PerformanceTelemetry::EventType::VulkanSubmit, static_cast<u64>(reason),
            signal_value);

        const auto cur_cmdbuf = Common::PerformanceTelemetry::CurrentCmdBufferSeq();
        const auto submit_seq = Common::PerformanceTelemetry::NextSubmitSeq();
        Common::PerformanceTelemetry::RegisterSubmitTick(signal_value, submit_seq);
        Common::PerformanceTelemetry::RegisterCmdBufferSubmit(cur_cmdbuf, submit_seq);
        Common::PerformanceTelemetry::PromotePendingReadbacksOnSubmit(cur_cmdbuf, submit_seq, signal_value);
        const u64 gpu_tick_val = master_semaphore.KnownGpuTick();
        const u64 ahead_ticks = signal_value > gpu_tick_val ? signal_value - gpu_tick_val : 0;
        Common::PerformanceTelemetry::RecordSubmitRecord(Common::PerformanceTelemetry::SubmitRecordSample{
            .submit_seq = submit_seq,
            .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
            .reason = reason,
            .signal_tick = signal_value,
            .cpu_ahead_ticks = ahead_ticks,
            .gpu_completed_tick = gpu_tick_val,
            .scheduler_id = 0,
            .queue_role = 0,
            .cmd_buffer_seq = cur_cmdbuf,
        });
    }
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    std::vector<PendingOp> ready_ops;
    while (!stoken.stop_requested()) {
        u64 wait_tick = 0;
        Common::PerformanceTelemetry::PendingOpTraceToken trace{};
        ready_ops.clear();
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            wait_tick = priority_pending_ops.front().gpu_tick;
            trace = priority_pending_ops.front().trace;
        }

        const u64 wait_start = Common::PerformanceTelemetry::Timestamp();
        master_semaphore.Wait(wait_tick, Common::PerformanceTelemetry::HostWaitReason::FenceCpuVisibility, trace);
        Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::PriorityOpsWaitNs,
                                          Common::PerformanceTelemetry::Timestamp() - wait_start);
        if (stoken.stop_requested()) {
            break;
        }

        const u64 completed_tick = master_semaphore.KnownGpuTick();
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            while (!priority_pending_ops.empty() &&
                   priority_pending_ops.front().gpu_tick <= completed_tick) {
                ready_ops.emplace_back(std::move(priority_pending_ops.front()));
                priority_pending_ops.pop();
            }
        }

        Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::PriorityOpsDrainCount,
                                          ready_ops.size());
        const u64 exec_start = Common::PerformanceTelemetry::Timestamp();
        for (auto& op : ready_ops) {
            op.callback();
        }
        Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::PriorityOpsExecuteNs,
                                          Common::PerformanceTelemetry::Timestamp() - exec_start);
    }
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    if (dirty_bits == 0) [[likely]] {
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::RecordDynamicCommitEnabled(0, 0);
        }
        return;
    }
    const u32 dirty_before = dirty_bits;

    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                      ? vk::ImageAspectFlagBits::eColor
                                                      : vk::ImageAspectFlagBits::eNone);
    }
    if (telemetry_enabled) {
        constexpr auto GroupMask = [](u32 bits) {
            u32 groups{};
            groups |= static_cast<u32>((bits & 0x00000003u) != 0) << 0;
            groups |= static_cast<u32>((bits & 0x0003FFFCu) != 0) << 1;
            groups |= static_cast<u32>((bits & 0x00040000u) != 0) << 2;
            groups |= static_cast<u32>((bits & 0x01380000u) != 0) << 3;
            groups |= static_cast<u32>((bits & 0x02C00000u) != 0) << 4;
            return groups;
        };
        Common::PerformanceTelemetry::RecordDynamicCommitEnabled(
            GroupMask(dirty_before), GroupMask(dirty_before & ~dirty_bits));
    }
}

} // namespace Vulkan
