// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>

#include "common/assert.h"
#include "common/debug.h"
#include "common/hash.h"
#include "common/performance_telemetry.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/guest_copy_engine.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_record_audit.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

namespace {

u64 RenderStateHash(const RenderState& state) {
    u64 hash = HashCombine(static_cast<u64>(state.width), static_cast<u64>(state.height));
    hash = HashCombine(hash, HashCombine(static_cast<u64>(state.num_layers),
                                         static_cast<u64>(state.num_color_attachments)));
    const auto hash_attachment = [&hash](const RenderAttachment& attachment) {
        hash = HashCombine(hash, static_cast<u64>(std::hash<VkImageView>{}(
                                     static_cast<VkImageView>(attachment.image_view))));
        hash = HashCombine(hash, static_cast<u64>(attachment.image_layout));
    };
    for (u32 index = 0; index < state.num_color_attachments; ++index) {
        hash_attachment(state.color_attachments[index]);
    }
    if (state.depth_stencil_attachment.has_depth ||
        state.depth_stencil_attachment.has_stencil) {
        hash_attachment(state.depth_stencil_attachment);
    }
    return hash;
}

/// Minimum interval between driver queries for the GPU tick made on behalf of deferred
/// operations. Draws come every few microseconds; a query per draw costs more than the draw.
constexpr u64 PendingOpsPollIntervalNs = 50'000;
/// Draw-path calls between two checks of the GPU progress while an operation waits for it.
constexpr u32 PendingOpsCheckPeriod = 8;

[[nodiscard]] u64 SteadyNowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Chunks the command processor may queue ahead of the recording thread before it waits.
constexpr size_t MaxQueuedWork = 64;
/// Interval between two audit reports of the recording thread.
constexpr u64 AuditReportIntervalNs = 10'000'000'000ULL;

void BeginRenderingCommand(vk::CommandBuffer cmdbuf, const RenderState& state) {
    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < state.num_color_attachments; ++i) {
        const auto& cb = state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = state.depth_stencil_attachment;
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
                .extent = {state.width, state.height},
            },
        .layerCount = state.num_layers,
        .colorAttachmentCount = state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };
    cmdbuf.beginRendering(rendering_info);
}

/// Dynamic state commands decided by DynamicState::Commit, with the values they set. The
/// decision happens where the state is tracked, the driver calls wherever commands are replayed.
struct DynamicStateEmit {
    enum Bits : u32 {
        DepthTestEnable = 1U << 0,
        DepthWriteEnable = 1U << 1,
        DepthCompareOp = 1U << 2,
        DepthBoundsTestEnable = 1U << 3,
        DepthBounds = 1U << 4,
        DepthBiasEnable = 1U << 5,
        DepthBias = 1U << 6,
        StencilTestEnable = 1U << 7,
        StencilOpBoth = 1U << 8,
        StencilOpFront = 1U << 9,
        StencilOpBack = 1U << 10,
        StencilReferenceBoth = 1U << 11,
        StencilReferenceFront = 1U << 12,
        StencilReferenceBack = 1U << 13,
        StencilWriteMaskBoth = 1U << 14,
        StencilWriteMaskFront = 1U << 15,
        StencilWriteMaskBack = 1U << 16,
        StencilCompareMaskBoth = 1U << 17,
        StencilCompareMaskFront = 1U << 18,
        StencilCompareMaskBack = 1U << 19,
        PrimitiveRestartEnable = 1U << 20,
        RasterizerDiscardEnable = 1U << 21,
        CullMode = 1U << 22,
        FrontFace = 1U << 23,
        BlendConstants = 1U << 24,
        ColorWriteMask = 1U << 25,
        LineWidth = 1U << 26,
        FeedbackLoop = 1U << 27,
    };

    void Apply(vk::CommandBuffer cmdbuf) const {
        if (bits & DepthTestEnable) {
            cmdbuf.setDepthTestEnable(depth_test_enabled);
        }
        if (bits & DepthWriteEnable) {
            cmdbuf.setDepthWriteEnable(depth_write_enabled);
        }
        if (bits & DepthCompareOp) {
            cmdbuf.setDepthCompareOp(depth_compare_op);
        }
        if (bits & DepthBoundsTestEnable) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
        if (bits & DepthBounds) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
        if (bits & DepthBiasEnable) {
            cmdbuf.setDepthBiasEnable(depth_bias_enabled);
        }
        if (bits & DepthBias) {
            cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
        }
        if (bits & StencilTestEnable) {
            cmdbuf.setStencilTestEnable(stencil_test_enabled);
        }
        const auto set_stencil_op = [&](vk::StencilFaceFlags faces, const StencilOps& ops) {
            cmdbuf.setStencilOp(faces, ops.fail_op, ops.pass_op, ops.depth_fail_op, ops.compare_op);
        };
        if (bits & StencilOpBoth) {
            set_stencil_op(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops);
        }
        if (bits & StencilOpFront) {
            set_stencil_op(vk::StencilFaceFlagBits::eFront, stencil_front_ops);
        }
        if (bits & StencilOpBack) {
            set_stencil_op(vk::StencilFaceFlagBits::eBack, stencil_back_ops);
        }
        if (bits & StencilReferenceBoth) {
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        }
        if (bits & StencilReferenceFront) {
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront, stencil_front_reference);
        }
        if (bits & StencilReferenceBack) {
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
        }
        if (bits & StencilWriteMaskBoth) {
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        }
        if (bits & StencilWriteMaskFront) {
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront, stencil_front_write_mask);
        }
        if (bits & StencilWriteMaskBack) {
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
        }
        if (bits & StencilCompareMaskBoth) {
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        }
        if (bits & StencilCompareMaskFront) {
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                         stencil_front_compare_mask);
        }
        if (bits & StencilCompareMaskBack) {
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack, stencil_back_compare_mask);
        }
        if (bits & PrimitiveRestartEnable) {
            cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
        }
        if (bits & RasterizerDiscardEnable) {
            cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
        }
        if (bits & CullMode) {
            cmdbuf.setCullMode(cull_mode);
        }
        if (bits & FrontFace) {
            cmdbuf.setFrontFace(front_face);
        }
        if (bits & BlendConstants) {
            cmdbuf.setBlendConstants(blend_constants.data());
        }
        if (bits & ColorWriteMask) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
        if (bits & LineWidth) {
            cmdbuf.setLineWidth(line_width);
        }
        if (bits & FeedbackLoop) {
            cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                          ? vk::ImageAspectFlagBits::eColor
                                                          : vk::ImageAspectFlagBits::eNone);
        }
    }

    u32 bits{};
    bool depth_test_enabled{};
    bool depth_write_enabled{};
    bool depth_bounds_test_enabled{};
    bool depth_bias_enabled{};
    bool stencil_test_enabled{};
    bool primitive_restart_enable{};
    bool rasterizer_discard_enable{};
    bool feedback_loop_enabled{};
    vk::CompareOp depth_compare_op{};
    float depth_bounds_min{};
    float depth_bounds_max{};
    float depth_bias_constant{};
    float depth_bias_clamp{};
    float depth_bias_slope{};
    StencilOps stencil_front_ops{};
    StencilOps stencil_back_ops{};
    u32 stencil_front_reference{};
    u32 stencil_back_reference{};
    u32 stencil_front_write_mask{};
    u32 stencil_back_write_mask{};
    u32 stencil_front_compare_mask{};
    u32 stencil_back_compare_mask{};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};
    std::array<float, 4> blend_constants{};
    ColorWriteMasks color_write_masks{};
    float line_width{};
};

} // namespace

Scheduler::Scheduler(const Instance& instance, bool async_submit, bool threaded_recording_)
    : instance{instance}, async_submit{async_submit}, master_semaphore{instance},
      command_pool{instance, &master_semaphore} {
    bool gpu_profiling = false;
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    // GPU timestamps and pipeline statistics serialize GPU work; heavy telemetry only.
    if (Common::PerformanceTelemetry::HeavyEnabled()) {
        gpu_profiler = std::make_unique<GpuProfiler>(instance, master_semaphore);
        gpu_profiling = true;
    }
#endif
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
    // Tracy zones wrap the command buffer on the recording side.
    gpu_profiling = true;
#endif
    if (threaded_recording_ && gpu_profiling) {
        // The GPU profiler writes timestamps straight into the command buffer.
        LOG_INFO(Render_Vulkan, "GPU profiling keeps command recording on the calling thread");
    }
    threaded_recording = threaded_recording_ && !gpu_profiling;
    if (threaded_recording) {
        if (RecordAudit::RequestedByEnvironment()) {
            RecordAudit::Install();
        }
        LOG_INFO(Render_Vulkan, "Vulkan commands are recorded on a dedicated thread");
        chunk = TakeChunk();
        record_tick = master_semaphore.CurrentTick();
        current_command_buffer_seq = Common::PerformanceTelemetry::NextCmdBufferSeq();
        dynamic_state.Invalidate();
        record_thread = std::jthread(std::bind_front(&Scheduler::RecordThread, this));
    } else {
        AllocateWorkerCommandBuffers();
    }
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
    if (async_submit) {
        submit_thread = std::jthread(std::bind_front(&Scheduler::SubmitThread, this));
    }
}

Scheduler::~Scheduler() {
    if (threaded_recording) {
        // Everything dispatched gets recorded and handed off; the unfinished command buffer is
        // dropped with the pool.
        WaitRecordIdle();
        record_thread.request_stop();
        record_thread.join();
        if (RecordAudit::IsInstalled()) {
            RecordAudit::LogStats("shutdown");
        }
    }
    if (async_submit) {
        WaitSubmitted(CurrentTick() - 1);
        submit_thread.request_stop();
        submit_thread.join();
    }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    gpu_profiler.reset();
#endif
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
    EndRendering(Common::PerformanceTelemetry::ScopeBreakReason::AttachmentSetChange,
                 Common::PerformanceTelemetry::Avoidability::ProvenRequired);
    is_rendering = true;
    render_state = new_state;
    Record(
        [state = render_state](vk::CommandBuffer cmdbuf) { BeginRenderingCommand(cmdbuf, state); });
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    attachment_hash = RenderStateHash(new_state);
    rendering_scope_id = Common::PerformanceTelemetry::NextScopeSeq();
    if (gpu_profiler) {
        gpu_profiler->BeginRendering(attachment_hash);
    }
#endif
}

void Scheduler::EndRendering(Common::PerformanceTelemetry::ScopeBreakReason reason,
                             Common::PerformanceTelemetry::Avoidability avoidability) {
    if (!is_rendering) {
        return;
    }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    const auto context = Common::PerformanceTelemetry::CurrentCausalContext();
    const auto scope_break_id = Common::PerformanceTelemetry::NextScopeBreakSeq();
    Common::PerformanceTelemetry::RecordScopeBreak(
        Common::PerformanceTelemetry::ScopeBreakSample{
            .scope_break_id = scope_break_id,
            .cause_id = context.cause_id,
            .candidate_id = context.candidate_id,
            .completion_scope_id = context.scope_id,
            .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
            .command_buffer_seq = current_command_buffer_seq,
            .rendering_scope_id = rendering_scope_id,
            .attachment_hash = attachment_hash,
            .pipeline_hash = current_pipeline_hash,
            .reason = reason,
            .avoidability = avoidability,
        });
    Common::PerformanceTelemetry::RecordCausalEffect(
        Common::PerformanceTelemetry::CausalEffectSample{
            .effect_id = Common::PerformanceTelemetry::NextEffectSeq(),
            .cause_id = context.cause_id,
            .candidate_id = context.candidate_id,
            .scope_id = context.scope_id,
            .object_id = scope_break_id,
            .command_buffer_seq = current_command_buffer_seq,
            .kind = Common::PerformanceTelemetry::CausalEffectKind::ScopeBreak,
            .attribution = Common::PerformanceTelemetry::EffectAttribution::Exclusive,
            .avoidability = avoidability,
            .confidence = 255,
        });
    if (gpu_profiler) {
        gpu_profiler->EndRendering();
    }
#endif
    is_rendering = false;
    Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRendering(); });
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    rendering_scope_id = 0;
    attachment_hash = 0;
    current_pipeline_hash = 0;
#endif
}

void Scheduler::ProfileGraphicsDraw(u64 pipeline_hash, u32 command_count) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    current_pipeline_hash = pipeline_hash;
    if (gpu_profiler) {
        gpu_profiler->GraphicsDraw(pipeline_hash, command_count);
    }
#else
    static_cast<void>(pipeline_hash);
    static_cast<void>(command_count);
#endif
}

void Scheduler::ProfileComputeDispatch(u64 pipeline_hash, u32 command_count) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    current_pipeline_hash = pipeline_hash;
    if (gpu_profiler) {
        gpu_profiler->ComputeDispatch(pipeline_hash, command_count);
    }
#else
    static_cast<void>(pipeline_hash);
    static_cast<void>(command_count);
#endif
}

u64 Scheduler::BeginGpuInterval(Common::PerformanceTelemetry::GpuIntervalKind kind,
                                u64 object_hash, u64 bytes) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return gpu_profiler ? gpu_profiler->BeginInterval(kind, object_hash, bytes) : 0;
#else
    static_cast<void>(kind);
    static_cast<void>(object_hash);
    static_cast<void>(bytes);
    return 0;
#endif
}

void Scheduler::EndGpuInterval(u64 token) {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (gpu_profiler) {
        gpu_profiler->EndInterval(token);
    }
#else
    static_cast<void>(token);
#endif
}

void Scheduler::Flush(SubmitInfo& info, Common::PerformanceTelemetry::SubmitReason reason) {
    SubmitExecution(info, reason);
    // The presentation thread can enqueue a wait for this frame as soon as it is published.
    if (reason == Common::PerformanceTelemetry::SubmitReason::PresentFrameBuild) {
        WaitSubmitted(CurrentTick() - 1);
    }
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
    WaitSubmitted(tick);
    master_semaphore.Wait(tick, reason);
}

void Scheduler::WaitSubmitted(u64 tick) const {
    if (!async_submit && !threaded_recording) {
        return;
    }
    u64 submitted = submitted_tick.load(std::memory_order_acquire);
    if (submitted >= tick) {
        return;
    }
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    const u64 wait_start = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    while (submitted < tick) {
        submitted_tick.wait(submitted, std::memory_order_relaxed);
        submitted = submitted_tick.load(std::memory_order_acquire);
    }
    if (telemetry_enabled && threaded_recording) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::VkRecordProducerWaitNs,
            Common::PerformanceTelemetry::Timestamp() - wait_start);
    }
}

void Scheduler::SubmitThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:VkQueueSubmit");
    while (!stoken.stop_requested()) {
        SubmitJob job{};
        submit_queue.PopWait(job, stoken);
        if (stoken.stop_requested()) {
            break;
        }
        SubmitJobNow(job);
    }
}

void Scheduler::SubmitJobNow(SubmitJob& job) {
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = job.info.num_wait_semas,
        .pWaitSemaphoreValues = job.info.wait_ticks.data(),
        .signalSemaphoreValueCount = job.info.num_signal_semas,
        .pSignalSemaphoreValues = job.info.signal_ticks.data(),
    };
    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = job.info.num_wait_semas,
        .pWaitSemaphores = job.info.wait_semas.data(),
        .pWaitDstStageMask = job.info.wait_stages.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &job.cmdbuf,
        .signalSemaphoreCount = job.info.num_signal_semas,
        .pSignalSemaphores = job.info.signal_semas.data(),
    };
    if (job.guest_copy_seq != 0) {
        // The command buffer reads staging bytes that copy workers may still be writing.
        VideoCore::GuestCopyEngine::Instance().WaitCompleted(job.guest_copy_seq);
    }
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    const u64 wait_start = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    std::unique_lock lk{instance.GetGraphicsQueueMutex()};
    if (job.texture_uploads) {
        // Overlay texture uploads use the same queue; keep them ahead of this job like the
        // command processor does when it submits itself.
        ImGui::Core::TextureManager::Submit();
    }
    const u64 driver_start = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    master_semaphore.TelemetrySubmit(job.signal_tick);
    const auto result = [&] {
        Common::PerformanceTelemetry::ScopedDuration submit_duration{
            Common::PerformanceTelemetry::Counter::DriverSubmitNs};
        return instance.GetGraphicsQueue().submit(submit_info, job.info.fence);
    }();
    const u64 driver_end = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    ASSERT_MSG(result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    lk.unlock();
    submitted_tick.store(job.signal_tick, std::memory_order_release);
    submitted_tick.notify_all();
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::RecordSubmitTimingEnabled(
            job.reason, driver_start - wait_start, 0, driver_end - driver_start, 0,
            driver_end - wait_start);
    }
}

void Scheduler::PopPendingOperations(bool force) {
    const u64 front_tick = pending_ops_front_tick.load(std::memory_order_acquire);
    if (front_tick == NoPendingOps) [[likely]] {
        return;
    }
    // A deferred operation waits for the GPU far longer than the few microseconds between draws.
    static thread_local u32 calls_since_check = 0;
    if (!force && ++calls_since_check < PendingOpsCheckPeriod) {
        return;
    }
    calls_since_check = 0;

    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    if (!master_semaphore.IsFree(front_tick)) {
        // Waits, submits and the priority operation thread keep the known tick fresh; ask the
        // driver only at a bounded rate.
        const u64 now = SteadyNowNs();
        if (!force && now < next_pending_ops_poll_ns.load(std::memory_order_relaxed)) {
            if (telemetry_enabled) {
                Common::PerformanceTelemetry::AddEnabled(
                    Common::PerformanceTelemetry::Counter::PendingOpPollSkips, 1);
            }
            return;
        }
        next_pending_ops_poll_ns.store(now + PendingOpsPollIntervalNs, std::memory_order_relaxed);
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::PendingOpRefreshes, 1);
        }
        master_semaphore.Refresh();
        if (!master_semaphore.IsFree(front_tick)) {
            return;
        }
    } else if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::PendingOpKnownTickHits, 1);
    }

    std::unique_lock lk(pending_ops_mutex);
    while (!pending_ops.empty() && master_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
    pending_ops_front_tick.store(pending_ops.empty() ? NoPendingOps : pending_ops.front().gpu_tick,
                                 std::memory_order_release);
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    current_command_buffer_seq = Common::PerformanceTelemetry::NextCmdBufferSeq();
    current_cmdbuf = command_pool.Commit(master_semaphore.CurrentTick());
    Check(current_cmdbuf.begin(begin_info));
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (gpu_profiler) {
        gpu_profiler->BeginCommandBuffer(current_cmdbuf, current_command_buffer_seq,
                                         Common::PerformanceTelemetry::CurrentFrameSeq());
    }
#endif

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
    if (threaded_recording) {
        SubmitRecordedExecution(info, reason);
        return;
    }
    if (async_submit) {
        // TextureManager::Submit uses the same queue directly; submit the previous job first.
        WaitSubmitted(CurrentTick() - 1);
    }
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    const u64 wait_start = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    std::unique_lock lk{instance.GetGraphicsQueueMutex(), std::defer_lock};
    if (!async_submit) {
        lk.lock();
    }
    const u64 lock_acquired = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    const u64 signal_value = master_semaphore.NextTick();
    const auto submitted_cmdbuf = current_command_buffer_seq;
    const auto submit_seq = Common::PerformanceTelemetry::NextSubmitSeq();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    const bool present_submit = reason == Common::PerformanceTelemetry::SubmitReason::PresentFrameBuild ||
                                reason == Common::PerformanceTelemetry::SubmitReason::PresentSubmit ||
                                reason == Common::PerformanceTelemetry::SubmitReason::QueuePresent;
    EndRendering(present_submit ? Common::PerformanceTelemetry::ScopeBreakReason::Present
                                : Common::PerformanceTelemetry::ScopeBreakReason::RequiredNonGraphicsCommand,
                 Common::PerformanceTelemetry::Avoidability::ProvenRequired);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (gpu_profiler) {
        gpu_profiler->EndCommandBuffer(submit_seq, signal_value);
    }
#endif
    Check(current_cmdbuf.end());

    const vk::Semaphore timeline = master_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    // Every staging write recorded into this command buffer has been enqueued by now.
    const u64 guest_copy_seq =
        gate_guest_copies ? VideoCore::GuestCopyEngine::Instance().SubmittedSeq() : 0;

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

    if (async_submit) {
        lk.lock();
    }
    ImGui::Core::TextureManager::Submit();
    if (async_submit) {
        lk.unlock();
    }
    const u64 driver_start = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    u64 driver_end = driver_start;
    if (async_submit) {
        submit_queue.EmplaceWait(SubmitJob{.info = info,
                                           .cmdbuf = current_cmdbuf,
                                           .guest_copy_seq = guest_copy_seq,
                                           .signal_tick = signal_value,
                                           .reason = reason});
    } else {
        if (guest_copy_seq != 0) {
            VideoCore::GuestCopyEngine::Instance().WaitCompleted(guest_copy_seq);
        }
        master_semaphore.TelemetrySubmit(signal_value);
        const auto submit_result = [&] {
            Common::PerformanceTelemetry::ScopedDuration submit_duration{
                Common::PerformanceTelemetry::Counter::DriverSubmitNs};
            return instance.GetGraphicsQueue().submit(submit_info, info.fence);
        }();
        driver_end = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
        ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    }

    if (!async_submit) {
        master_semaphore.Refresh();
    }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (gpu_profiler) {
        gpu_profiler->Collect();
    }
#endif
    AllocateWorkerCommandBuffers();

    // Apply pending operations
    PopPendingOperations(true);
    if (telemetry_enabled) {
        const u64 post_end = Common::PerformanceTelemetry::Timestamp();
        if (!async_submit) {
            lk.unlock();
            Common::PerformanceTelemetry::RecordSubmitTimingEnabled(
                reason, lock_acquired - wait_start, driver_start - lock_acquired,
                driver_end - driver_start, post_end - driver_end, post_end - lock_acquired);
        }
        Common::PerformanceTelemetry::RecordEnabled(
            Common::PerformanceTelemetry::EventType::VulkanSubmit, static_cast<u64>(reason),
            signal_value);

        Common::PerformanceTelemetry::RegisterSubmitTick(signal_value, submit_seq);
        Common::PerformanceTelemetry::RegisterCmdBufferSubmit(submitted_cmdbuf, submit_seq);
        Common::PerformanceTelemetry::PromotePendingReadbacksOnSubmit(submitted_cmdbuf, submit_seq, signal_value);
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
            .cmd_buffer_seq = submitted_cmdbuf,
        });
        const auto context = Common::PerformanceTelemetry::CurrentCausalContext();
        Common::PerformanceTelemetry::RecordCausalEffect(
            Common::PerformanceTelemetry::CausalEffectSample{
                .effect_id = Common::PerformanceTelemetry::NextEffectSeq(),
                .cause_id = context.cause_id,
                .candidate_id = context.candidate_id,
                .scope_id = context.scope_id,
                .command_buffer_seq = submitted_cmdbuf,
                .submit_seq = submit_seq,
                .timeline_tick = signal_value,
                .duration_ns = driver_end - driver_start,
                .kind = Common::PerformanceTelemetry::CausalEffectKind::Submit,
                .attribution = Common::PerformanceTelemetry::EffectAttribution::Shared,
                .avoidability = Common::PerformanceTelemetry::Avoidability::ConservativeFallback,
                .confidence = 255,
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

void Scheduler::SubmitRecordedExecution(SubmitInfo& info,
                                        Common::PerformanceTelemetry::SubmitReason reason) {
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    const bool present_submit =
        reason == Common::PerformanceTelemetry::SubmitReason::PresentFrameBuild ||
        reason == Common::PerformanceTelemetry::SubmitReason::PresentSubmit ||
        reason == Common::PerformanceTelemetry::SubmitReason::QueuePresent;
    EndRendering(present_submit
                     ? Common::PerformanceTelemetry::ScopeBreakReason::Present
                     : Common::PerformanceTelemetry::ScopeBreakReason::RequiredNonGraphicsCommand,
                 Common::PerformanceTelemetry::Avoidability::ProvenRequired);

    const u64 signal_value = master_semaphore.NextTick();
    const auto submitted_cmdbuf = current_command_buffer_seq;
    const auto submit_seq = Common::PerformanceTelemetry::NextSubmitSeq();
    info.AddSignal(master_semaphore.Handle(), signal_value);

    // Every staging write recorded into this command buffer has been enqueued by now.
    const u64 guest_copy_seq =
        gate_guest_copies ? VideoCore::GuestCopyEngine::Instance().SubmittedSeq() : 0;

    PushWork(RecordWork{
        .chunk = std::exchange(chunk, TakeChunk()),
        .submit = true,
        .request =
            {
                .info = info,
                .signal_tick = signal_value,
                .guest_copy_seq = guest_copy_seq,
                .reason = reason,
            },
    });

    // The next command buffer starts here for everything tracked on this thread.
    current_command_buffer_seq = Common::PerformanceTelemetry::NextCmdBufferSeq();
    dynamic_state.Invalidate();
    Common::PerformanceTelemetry::Add(
        Common::PerformanceTelemetry::Counter::DynamicStateInvalidations);

    // Apply pending operations
    PopPendingOperations(true);
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::RecordEnabled(
            Common::PerformanceTelemetry::EventType::VulkanSubmit, static_cast<u64>(reason),
            signal_value);
        Common::PerformanceTelemetry::RegisterSubmitTick(signal_value, submit_seq);
        Common::PerformanceTelemetry::RegisterCmdBufferSubmit(submitted_cmdbuf, submit_seq);
        Common::PerformanceTelemetry::PromotePendingReadbacksOnSubmit(submitted_cmdbuf, submit_seq,
                                                                      signal_value);
        const u64 gpu_tick_val = master_semaphore.KnownGpuTick();
        const u64 ahead_ticks = signal_value > gpu_tick_val ? signal_value - gpu_tick_val : 0;
        Common::PerformanceTelemetry::RecordSubmitRecord(
            Common::PerformanceTelemetry::SubmitRecordSample{
                .submit_seq = submit_seq,
                .frame_seq = Common::PerformanceTelemetry::CurrentFrameSeq(),
                .reason = reason,
                .signal_tick = signal_value,
                .cpu_ahead_ticks = ahead_ticks,
                .gpu_completed_tick = gpu_tick_val,
                .scheduler_id = 0,
                .queue_role = 0,
                .cmd_buffer_seq = submitted_cmdbuf,
            });
        const auto context = Common::PerformanceTelemetry::CurrentCausalContext();
        Common::PerformanceTelemetry::RecordCausalEffect(
            Common::PerformanceTelemetry::CausalEffectSample{
                .effect_id = Common::PerformanceTelemetry::NextEffectSeq(),
                .cause_id = context.cause_id,
                .candidate_id = context.candidate_id,
                .scope_id = context.scope_id,
                .command_buffer_seq = submitted_cmdbuf,
                .submit_seq = submit_seq,
                .timeline_tick = signal_value,
                .kind = Common::PerformanceTelemetry::CausalEffectKind::Submit,
                .attribution = Common::PerformanceTelemetry::EffectAttribution::Shared,
                .avoidability = Common::PerformanceTelemetry::Avoidability::ConservativeFallback,
                .confidence = 255,
            });
    }
}

std::unique_ptr<CommandChunk> Scheduler::TakeChunk() {
    {
        std::scoped_lock lock{reserve_mutex};
        if (!chunk_reserve.empty()) {
            auto reused = std::move(chunk_reserve.back());
            chunk_reserve.pop_back();
            return reused;
        }
    }
    return std::make_unique<CommandChunk>();
}

void Scheduler::DispatchWork() {
    if (chunk->Empty()) {
        return;
    }
    PushWork(RecordWork{.chunk = std::exchange(chunk, TakeChunk())});
}

void Scheduler::PushWork(RecordWork&& work) {
    size_t depth;
    {
        std::unique_lock lock{work_mutex};
        if (work_queue.size() >= MaxQueuedWork) [[unlikely]] {
            // The recording thread fell behind; do not run ahead of it without bound.
            const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
            const u64 wait_start =
                telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
            work_done_cv.wait(lock, [this] { return work_queue.size() < MaxQueuedWork; });
            if (telemetry_enabled) {
                Common::PerformanceTelemetry::AddEnabled(
                    Common::PerformanceTelemetry::Counter::VkRecordProducerWaitNs,
                    Common::PerformanceTelemetry::Timestamp() - wait_start);
            }
        }
        work_queue.push_back(std::move(work));
        ++dispatched_work;
        depth = work_queue.size();
    }
    work_cv.notify_one();
    if (Common::PerformanceTelemetry::Enabled()) {
        Common::PerformanceTelemetry::ObserveMaxEnabled(
            Common::PerformanceTelemetry::Counter::VkRecordQueueDepthMax, depth);
    }
}

void Scheduler::WaitRecordIdle() {
    std::unique_lock lock{work_mutex};
    work_done_cv.wait(lock, [this] { return executed_work == dispatched_work; });
}

void Scheduler::ResetCommandBuffer() {
    if (!threaded_recording) {
        Check(current_cmdbuf.reset());
        return;
    }
    chunk->Reset();
    WaitRecordIdle();
    // The recording thread only touches its command buffer when it is given work.
    if (record_cmdbuf) {
        Check(record_cmdbuf.reset());
        record_cmdbuf = vk::CommandBuffer{};
    }
}

void Scheduler::RecordThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:VkRecord");
    // Every submission waits for this thread; like the copy workers, it must not queue behind
    // guest threads.
    Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
    u64 next_audit_report = SteadyNowNs() + AuditReportIntervalNs;
    while (true) {
        RecordWork work;
        {
            std::unique_lock lock{work_mutex};
            if (!work_cv.wait(lock, stoken, [this] { return !work_queue.empty(); })) {
                break;
            }
            work = std::move(work_queue.front());
            work_queue.pop_front();
        }
        ExecuteWork(work);
        work.chunk->Reset();
        {
            std::scoped_lock lock{reserve_mutex};
            chunk_reserve.push_back(std::move(work.chunk));
        }
        {
            std::scoped_lock lock{work_mutex};
            ++executed_work;
        }
        work_done_cv.notify_all();
        if (RecordAudit::IsInstalled()) [[unlikely]] {
            const u64 now = SteadyNowNs();
            if (now >= next_audit_report) {
                next_audit_report = now + AuditReportIntervalNs;
                RecordAudit::LogStats("recording thread");
            }
        }
    }
}

void Scheduler::ExecuteWork(RecordWork& work) {
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    const u64 start = telemetry_enabled ? Common::PerformanceTelemetry::Timestamp() : 0;
    if (!record_cmdbuf) {
        record_cmdbuf = command_pool.Commit(record_tick);
        if (RecordAudit::IsInstalled()) [[unlikely]] {
            RecordAudit::ClaimCommandBuffer(record_cmdbuf);
        }
        Check(record_cmdbuf.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        }));
    }
    work.chunk->ExecuteAll(record_cmdbuf);
    if (work.submit) {
        SubmitRecorded(work.request);
    }
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::VkRecordChunks, 1);
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::VkRecordCommands, work.chunk->NumCommands());
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::VkRecordWorkerNs,
            Common::PerformanceTelemetry::Timestamp() - start);
    }
}

void Scheduler::SubmitRecorded(const SubmitRequest& request) {
    ASSERT_MSG(request.signal_tick == record_tick,
               "Recorded command buffer for tick {} ended as tick {}", record_tick,
               request.signal_tick);
    Check(record_cmdbuf.end());
    SubmitJob job{
        .info = request.info,
        .cmdbuf = record_cmdbuf,
        .guest_copy_seq = request.guest_copy_seq,
        .signal_tick = request.signal_tick,
        .reason = request.reason,
        .texture_uploads = true,
    };
    record_cmdbuf = vk::CommandBuffer{};
    ++record_tick;
    if (async_submit) {
        submit_queue.EmplaceWait(std::move(job));
    } else {
        SubmitJobNow(job);
    }
}

void DynamicState::Commit(const Instance& instance, Scheduler& scheduler) {
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    if (dirty_bits == 0) [[likely]] {
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::RecordDynamicCommitEnabled(0, 0);
        }
        return;
    }
    const u32 dirty_before = dirty_bits;

    const CommandRecorder cmdbuf = scheduler.CommandBuffer();
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    DynamicStateEmit emit{};
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        emit.bits |= DynamicStateEmit::DepthTestEnable;
        emit.depth_test_enabled = depth_test_enabled;
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        emit.bits |= DynamicStateEmit::DepthWriteEnable;
        emit.depth_write_enabled = depth_write_enabled;
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        emit.bits |= DynamicStateEmit::DepthCompareOp;
        emit.depth_compare_op = depth_compare_op;
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            emit.bits |= DynamicStateEmit::DepthBoundsTestEnable;
            emit.depth_bounds_test_enabled = depth_bounds_test_enabled;
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            emit.bits |= DynamicStateEmit::DepthBounds;
            emit.depth_bounds_min = depth_bounds_min;
            emit.depth_bounds_max = depth_bounds_max;
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        emit.bits |= DynamicStateEmit::DepthBiasEnable;
        emit.depth_bias_enabled = depth_bias_enabled;
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        emit.bits |= DynamicStateEmit::DepthBias;
        emit.depth_bias_constant = depth_bias_constant;
        emit.depth_bias_clamp = depth_bias_clamp;
        emit.depth_bias_slope = depth_bias_slope;
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        emit.bits |= DynamicStateEmit::StencilTestEnable;
        emit.stencil_test_enabled = stencil_test_enabled;
    }
    if (stencil_test_enabled) {
        emit.stencil_front_ops = stencil_front_ops;
        emit.stencil_back_ops = stencil_back_ops;
        emit.stencil_front_reference = stencil_front_reference;
        emit.stencil_back_reference = stencil_back_reference;
        emit.stencil_front_write_mask = stencil_front_write_mask;
        emit.stencil_back_write_mask = stencil_back_write_mask;
        emit.stencil_front_compare_mask = stencil_front_compare_mask;
        emit.stencil_back_compare_mask = stencil_back_compare_mask;
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            emit.bits |= DynamicStateEmit::StencilOpBoth;
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                emit.bits |= DynamicStateEmit::StencilOpFront;
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                emit.bits |= DynamicStateEmit::StencilOpBack;
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            emit.bits |= DynamicStateEmit::StencilReferenceBoth;
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                emit.bits |= DynamicStateEmit::StencilReferenceFront;
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                emit.bits |= DynamicStateEmit::StencilReferenceBack;
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            emit.bits |= DynamicStateEmit::StencilWriteMaskBoth;
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                emit.bits |= DynamicStateEmit::StencilWriteMaskFront;
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                emit.bits |= DynamicStateEmit::StencilWriteMaskBack;
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            emit.bits |= DynamicStateEmit::StencilCompareMaskBoth;
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                emit.bits |= DynamicStateEmit::StencilCompareMaskFront;
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                emit.bits |= DynamicStateEmit::StencilCompareMaskBack;
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        emit.bits |= DynamicStateEmit::PrimitiveRestartEnable;
        emit.primitive_restart_enable = primitive_restart_enable;
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        emit.bits |= DynamicStateEmit::RasterizerDiscardEnable;
        emit.rasterizer_discard_enable = rasterizer_discard_enable;
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        emit.bits |= DynamicStateEmit::CullMode;
        emit.cull_mode = cull_mode;
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        emit.bits |= DynamicStateEmit::FrontFace;
        emit.front_face = front_face;
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        emit.bits |= DynamicStateEmit::BlendConstants;
        emit.blend_constants = blend_constants;
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            emit.bits |= DynamicStateEmit::ColorWriteMask;
            emit.color_write_masks = color_write_masks;
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        emit.bits |= DynamicStateEmit::LineWidth;
        emit.line_width = line_width;
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        emit.bits |= DynamicStateEmit::FeedbackLoop;
        emit.feedback_loop_enabled = feedback_loop_enabled;
    }
    if (emit.bits != 0) {
        scheduler.Record([emit](vk::CommandBuffer cmdbuf) { emit.Apply(cmdbuf); });
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
