// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <cstring>
#include <limits>
#include <span>

#include "common/assert.h"
#include "common/debug.h"
#include "common/performance_telemetry.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace Vulkan {

[[nodiscard]] AmdGpu::Buffer GetResolvedBuffer(const Shader::Info& info, u32 index) {
    ASSERT(index < info.buffers.size());
    ASSERT(info.resolved_buffers.size() == info.buffers.size());
    return info.resolved_buffers[index];
}

[[nodiscard]] AmdGpu::Image GetResolvedImage(const Shader::Info& info, u32 index) {
    ASSERT(index < info.images.size());
    ASSERT(info.resolved_images.size() == info.images.size());
    return info.resolved_images[index];
}

[[nodiscard]] AmdGpu::Sampler GetResolvedSampler(const Shader::Info& info, u32 index) {
    ASSERT(index < info.samplers.size());
    ASSERT(info.resolved_samplers.size() == info.samplers.size());
    return info.resolved_samplers[index];
}

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

static SHAD_NO_INLINE void ReportUnsupportedWindowOffset() {
    LOG_ERROR(Render_Vulkan,
              "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
}

[[nodiscard]] static vk::Viewport MakeViewport(const Instance& instance, const AmdGpu::Regs& regs,
                                               const u32 index) {
    const auto& vp = regs.viewports[index];
    const auto& vp_ctl = regs.viewport_control;
    const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
    const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

    vk::Viewport viewport{};
    if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
        viewport.minDepth = zoffset - zscale;
        viewport.maxDepth = zoffset + zscale;
    } else {
        viewport.minDepth = zoffset;
        viewport.maxDepth = zoffset + zscale;
    }

    if (!instance.IsDepthRangeUnrestrictedSupported()) {
        viewport.minDepth = std::max(viewport.minDepth, 0.f);
        viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
    }

    if (regs.IsClipDisabled()) {
        viewport.x = 0.f;
        viewport.y = 0.f;
        viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
        viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
    } else {
        const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
        const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
        const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
        const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

        viewport.x = xoffset - xscale;
        viewport.y = yoffset - yscale;
        viewport.width = xscale * 2.0f;
        viewport.height = yscale * 2.0f;
    }
    return viewport;
}

[[nodiscard]] static vk::Rect2D MakeViewportScissor(const AmdGpu::Regs& regs,
                                                    const AmdGpu::Scissor& combined_scissor,
                                                    const u32 index) {
    auto scissor = combined_scissor;
    if (regs.mode_control.vport_scissor_enable) {
        scissor.top_left_x =
            std::max(scissor.top_left_x, s16(regs.viewport_scissors[index].top_left_x));
        scissor.top_left_y =
            std::max(scissor.top_left_y, s16(regs.viewport_scissors[index].top_left_y));
        scissor.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(scissor.bottom_right_x),
                                          regs.viewport_scissors[index].bottom_right_x);
        scissor.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(scissor.bottom_right_y),
                                          regs.viewport_scissors[index].bottom_right_y);
    }
    return {
        .offset = {scissor.top_left_x, scissor.top_left_y},
        .extent = {scissor.GetWidth(), scissor.GetHeight()},
    };
}

static SHAD_NO_INLINE void SetMultipleViewportScissorState(
    const Instance& instance, const AmdGpu::Regs& regs, const AmdGpu::Scissor& combined_scissor,
    DynamicState& dynamic_state) {
    Viewports viewports;
    Scissors scissors;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; ++i) {
        if (regs.viewports[i].xscale == 0.f) {
            continue;
        }
        viewports.push_back(MakeViewport(instance, regs, i));
        scissors.push_back(MakeViewportScissor(regs, combined_scissor, i));
    }
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, page_manager{this},
      buffer_cache{instance, scheduler, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
}

Rasterizer::~Rasterizer() = default;

void Rasterizer::CpSync() {
    scheduler.EndRendering();
    auto cmdbuf = scheduler.CommandBuffer();

    const vk::MemoryBarrier ib_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eDrawIndirect,
                           vk::DependencyFlagBits::eByRegion, ib_barrier, {}, {});
}

bool Rasterizer::FilterDraw() {
    const auto& regs = liverpool->regs;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->regs;
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->last_cb_extent[cb];
        std::construct_at(&desc, col_buf, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->last_db_extent;
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::Info& info,
    const std::optional<const Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->last_cb_extent[0]);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    image.Clear(clear_value, desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
    telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(Common::PerformanceTelemetry::Counter::Draws, 1);
    }
    Common::PerformanceTelemetry::ScopedDuration draw_duration{
        telemetry_enabled, Common::PerformanceTelemetry::Counter::DrawCpuNs};

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const auto& regs = liverpool->regs;
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    buffer_cache.PrepareVertexIndexBuffers(*pipeline, is_indexed, index_offset);
    const auto state = BeginRendering(pipeline);
    buffer_cache.FinalizeStreamCopyBatch();
    FinalizeBuffers(push_data, true);
    buffer_cache.FinalizeVertexIndexBuffers(buffer_barriers);

    BindPipelineResources(pipeline);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindGraphicsPipeline(pipeline->Handle());

    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    DebugState.IncDrawCall();

    ResetBindings();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    RENDERER_TRACE;
    telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(Common::PerformanceTelemetry::Counter::Draws, 1);
    }
    Common::PerformanceTelemetry::ScopedDuration draw_duration{
        telemetry_enabled, Common::PerformanceTelemetry::Counter::DrawCpuNs};

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    buffer_cache.PrepareVertexIndexBuffers(*pipeline, is_indexed, 0);
    const auto state = BeginRendering(pipeline);
    buffer_cache.FinalizeStreamCopyBatch();
    FinalizeBuffers(push_data, true);
    buffer_cache.FinalizeVertexIndexBuffers(buffer_barriers);

    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);

    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) = buffer_cache.ObtainBuffer(count_address, 4, false);
    }

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }
    if (count_buffer) {
        if (auto barrier = count_buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                                    vk::PipelineStageFlagBits2::eDrawIndirect)) {
            buffer_barriers.emplace_back(*barrier);
        }
    }

    BindPipelineResources(pipeline);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
    // instance offsets will be automatically applied by Vulkan from indirect args buffer.

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindGraphicsPipeline(pipeline->Handle());

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_base, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_base,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    }

    ResetBindings();
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;
    telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::Dispatches, 1);
    }
    Common::PerformanceTelemetry::ScopedDuration dispatch_duration{
        telemetry_enabled, Common::PerformanceTelemetry::Counter::DispatchCpuNs};

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    scheduler.EndRendering();
    BindPipelineResources(pipeline);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
    DebugState.IncDispatch();

    ResetBindings();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;
    telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    if (telemetry_enabled) {
        Common::PerformanceTelemetry::AddEnabled(
            Common::PerformanceTelemetry::Counter::Dispatches, 1);
    }
    Common::PerformanceTelemetry::ScopedDuration dispatch_duration{
        telemetry_enabled, Common::PerformanceTelemetry::Counter::DispatchCpuNs};

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }

    scheduler.EndRendering();
    BindPipelineResources(pipeline);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatchIndirect(buffer->Handle(), base);
    DebugState.IncDispatch();

    ResetBindings();
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    buffer_cache.RunGarbageCollector();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (pipeline->IsCompute() &&
        (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
         IsComputeImageClear(pipeline))) [[unlikely]] {
        return false;
    }

    buffer_cache.BeginStreamCopyBatch();
    set_write_index = 0;
    set_writes.clear();
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();
    pending_buffer_bindings.clear();

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        stage->PushUd(binding, push_data);
        PrepareBuffers(*stage, binding);
        FinalizeBuffers(push_data, false);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
    }

    if (pipeline->IsCompute()) {
        buffer_cache.FinalizeStreamCopyBatch();
        FinalizeBuffers(push_data, true);
    }

    if (uses_dma) {
        SynchronizeDmaBuffers();
    }

    return true;
}

SHAD_NO_INLINE void Rasterizer::SynchronizeDmaBuffers() {
    // We only use fault buffer for DMA right now.
    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    for (auto& range : mapped_ranges) {
        buffer_cache.SynchronizeBuffersInRange(range.lower(), range.upper() - range.lower());
    }
    fault_process_pending = true;
}

void Rasterizer::CaptureDescriptorState(const Pipeline* pipeline) {
    auto& state = descriptor_state;
    state.pipeline = pipeline;
    state.command_buffer = scheduler.CommandBuffer();
    state.push_descriptor_epoch = scheduler.GraphicsPushDescriptorEpoch();
    state.writes.clear();
    state.image_infos.clear();
    state.buffer_infos.clear();

    for (const auto& write : set_writes) {
        const bool is_buffer = write.pBufferInfo != nullptr;
        if (write.pImageInfo == nullptr && !is_buffer) {
            continue;
        }
        ASSERT(state.writes.size() < state.writes.capacity());
        const u32 first_info = is_buffer ? static_cast<u32>(state.buffer_infos.size())
                                         : static_cast<u32>(state.image_infos.size());
        state.writes.push_back({
            .binding = write.dstBinding,
            .array_element = write.dstArrayElement,
            .count = write.descriptorCount,
            .first_info = first_info,
            .type = write.descriptorType,
            .is_buffer = is_buffer,
        });
        if (is_buffer) {
            ASSERT(state.buffer_infos.size() + write.descriptorCount <=
                   state.buffer_infos.capacity());
            state.buffer_infos.insert(state.buffer_infos.end(), write.pBufferInfo,
                                      write.pBufferInfo + write.descriptorCount);
        } else {
            ASSERT(state.image_infos.size() + write.descriptorCount <=
                   state.image_infos.capacity());
            state.image_infos.insert(state.image_infos.end(), write.pImageInfo,
                                     write.pImageInfo + write.descriptorCount);
        }
    }
    state.valid = true;
}

void Rasterizer::BindPipelineResources(const Pipeline* pipeline) {
    if (pipeline->IsCompute() || !pipeline->UsesPushDescriptors()) {
        pipeline->BindResources(set_writes, buffer_barriers, push_data);
        return;
    }

    const auto& cached = descriptor_state;
    const bool can_reuse = cached.valid && cached.pipeline == pipeline &&
                           cached.command_buffer == scheduler.CommandBuffer() &&
                           cached.push_descriptor_epoch == scheduler.GraphicsPushDescriptorEpoch();
    partial_set_writes.clear();
    u32 cached_write_index = 0;
    for (const auto& write : set_writes) {
        bool unchanged = false;
        const bool is_buffer = write.pBufferInfo != nullptr;
        const bool is_cacheable = write.pImageInfo != nullptr || is_buffer;
        if (is_cacheable && can_reuse && cached_write_index < cached.writes.size()) {
            const auto& old_write = cached.writes[cached_write_index];
            unchanged = old_write.binding == write.dstBinding &&
                        old_write.array_element == write.dstArrayElement &&
                        old_write.count == write.descriptorCount &&
                        old_write.type == write.descriptorType && old_write.is_buffer == is_buffer;
            for (u32 info_index = 0; unchanged && info_index < write.descriptorCount;
                 ++info_index) {
                if (is_buffer) {
                    const auto& lhs = cached.buffer_infos[old_write.first_info + info_index];
                    const auto& rhs = write.pBufferInfo[info_index];
                    unchanged = lhs.buffer == rhs.buffer && lhs.offset == rhs.offset &&
                                lhs.range == rhs.range;
                } else {
                    const auto& lhs = cached.image_infos[old_write.first_info + info_index];
                    const auto& rhs = write.pImageInfo[info_index];
                    unchanged = lhs.sampler == rhs.sampler && lhs.imageView == rhs.imageView &&
                                lhs.imageLayout == rhs.imageLayout;
                }
            }
            ++cached_write_index;
        } else if (is_cacheable) {
            ++cached_write_index;
        }
        if (!unchanged) {
            partial_set_writes.push_back(write);
        }
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                unchanged ? Common::PerformanceTelemetry::Counter::DescriptorHits
                          : Common::PerformanceTelemetry::Counter::DescriptorMisses,
                1);
        }
    }

    auto& writes = can_reuse ? partial_set_writes : set_writes;
    pipeline->BindResources(writes, buffer_barriers, push_data);
    CaptureDescriptorState(pipeline);
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (u32 index = 0; index < info.buffers.size(); ++index) {
        const auto& desc = info.buffers[index];
        const VAddr address = GetResolvedBuffer(info, index).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (u32 index = 0; index < info.buffers.size(); ++index) {
            const auto& desc = info.buffers[index];
            const VAddr address = GetResolvedBuffer(info, index).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = GetResolvedBuffer(info, 0);
    const AmdGpu::Buffer buf1 = GetResolvedBuffer(info, 1);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    if (instance.IsMaintenance8Supported() ||
        src_image.info.props.is_depth == dst_image.info.props.is_depth) {
        dst_image.CopyImage(src_image);
    } else {
        const auto& copy_buffer =
            buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::DeviceLocal);
        dst_image.CopyImageWithBuffer(src_image, copy_buffer.Handle(), 0);
    }
    dst_image.flags |= VideoCore::ImageFlagBits::GpuModified;
    dst_image.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = GetResolvedBuffer(info, 0);
    const AmdGpu::Buffer buf1 = GetResolvedBuffer(info, 1);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    image1.Clear(clear, range);
    image1.flags |= VideoCore::ImageFlagBits::GpuModified;
    image1.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

void Rasterizer::PrepareBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    const u32 stage_index = static_cast<u32>(stage.l_stage);
    ASSERT(stage_index < MaxShaderStages);
    ASSERT(pending_buffer_bindings.size() + stage.buffers.size() <= Shader::NUM_BUFFERS);

    for (u32 i = 0; i < stage.buffers.size(); ++i) {
        const auto& desc = stage.buffers[i];
        const auto vsharp = GetResolvedBuffer(stage, i);
        const bool is_storage = desc.IsStorage(vsharp);
        const u64 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();

        auto& pending = pending_buffer_bindings.emplace_back(PendingBufferBinding{
            .desc = &desc,
            .sharp = vsharp,
            .alignment = alignment,
            .unified_binding = binding.unified++,
            .buffer_binding = binding.buffer++,
            .set_write_index = set_write_index++,
            .is_storage = is_storage,
        });

        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            pending.size = size;
            if (!desc.is_written && size <= VideoCore::BufferCache::CACHING_PAGESIZE &&
                !buffer_cache.IsRegionGpuModified(vsharp.base_address, size)) {
                pending.stream_source = VideoCore::BufferCache::StreamCopySource::Guest;
                pending.stream_index =
                    buffer_cache.QueueStreamCopy(VideoCore::BufferCache::StreamCopyRequest{
                        .source_type = VideoCore::BufferCache::StreamCopySource::Guest,
                        .guest_address = vsharp.base_address,
                        .size = static_cast<u32>(size),
                        .alignment = alignment,
                    });
                continue;
            }

            auto& cached = cached_buffer_bindings[stage_index][i];
            if (cached.owner != &stage) {
                cached = {};
                cached.owner = &stage;
            }

            const bool cache_hit =
                cached.valid && cached.sharp == vsharp && cached.size == size &&
                cached.topology_epoch == buffer_cache.TopologyEpoch() &&
                buffer_cache.IsBufferCacheEntryValid(cached.buffer_id, cached.buffer_uid,
                                                     vsharp.base_address, size);
            if (telemetry_enabled) {
                Common::PerformanceTelemetry::AddEnabled(
                    cache_hit ? Common::PerformanceTelemetry::Counter::BufferTokenHits
                              : Common::PerformanceTelemetry::Counter::BufferTokenMisses,
                    1);
            }
            if (cache_hit) {
                pending.buffer_id = cached.buffer_id;
            } else {
                pending.buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
            }

            cached.sharp = vsharp;
            cached.buffer_id = pending.buffer_id;
            cached.buffer_uid = buffer_cache.GetBufferUid(pending.buffer_id);
            cached.topology_epoch = buffer_cache.TopologyEpoch();
            cached.size = size;
            cached.valid = true;
            continue;
        }

        if (desc.buffer_type == Shader::BufferType::Flatbuf) {
            pending.size = stage.flattened_ud_buf.size() * sizeof(u32);
            if (pending.size != 0) {
                pending.stream_source = VideoCore::BufferCache::StreamCopySource::Host;
                pending.stream_index =
                    buffer_cache.QueueStreamCopy(VideoCore::BufferCache::StreamCopyRequest{
                        .source_type = VideoCore::BufferCache::StreamCopySource::Host,
                        .host_address = reinterpret_cast<const u8*>(stage.flattened_ud_buf.data()),
                        .size = static_cast<u32>(pending.size),
                        .alignment = alignment,
                    });
            }
            continue;
        }

        if (desc.buffer_type == Shader::BufferType::SharedMemory) {
            const auto& cs_program = liverpool->GetCsRegs();
            pending.size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
            if (pending.size != 0) {
                ASSERT(pending.size <= std::numeric_limits<u32>::max());
                pending.stream_source = VideoCore::BufferCache::StreamCopySource::Zero;
                pending.stream_index =
                    buffer_cache.QueueStreamCopy(VideoCore::BufferCache::StreamCopyRequest{
                        .source_type = VideoCore::BufferCache::StreamCopySource::Zero,
                        .size = static_cast<u32>(pending.size),
                        .alignment = alignment,
                        .deduplicate = false,
                    });
            }
        }
    }
}

void Rasterizer::FinalizeBuffers(Shader::PushData& push_data, bool stream_only) {
    static constexpr u16 NoStreamCopy = std::numeric_limits<u16>::max();

    for (auto& pending : pending_buffer_bindings) {
        if (pending.finalized) {
            continue;
        }
        const bool uses_stream = pending.stream_index != NoStreamCopy;
        if (uses_stream != stream_only) {
            continue;
        }

        const auto& desc = *pending.desc;
        const auto& vsharp = pending.sharp;

        if (uses_stream) {
            const auto& result = buffer_cache.GetStreamCopyResult(pending.stream_index);
            const u64 offset_aligned = Common::AlignDown(result.offset, u64{pending.alignment});
            const u32 adjust = static_cast<u32>(result.offset - offset_aligned);
            if (pending.stream_source == VideoCore::BufferCache::StreamCopySource::Guest) {
                ASSERT(adjust % 4 == 0);
                push_data.AddOffset(pending.buffer_binding, adjust);
                buffer_infos.emplace_back(result.buffer->Handle(), offset_aligned,
                                          pending.size + adjust);
                if (auto barrier =
                        result.buffer->GetBarrier(vk::AccessFlagBits2::eShaderRead,
                                                  vk::PipelineStageFlagBits2::eAllCommands)) {
                    buffer_barriers.emplace_back(*barrier);
                }
            } else {
                buffer_infos.emplace_back(result.buffer->Handle(), offset_aligned,
                                          pending.size + adjust);
            }
        } else if (!pending.buffer_id) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->regs.clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset =
                        vk_buffer.Copy(planes.data(), ubo_size, pending.alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto [vk_buffer, offset] =
                buffer_cache.ObtainBuffer(vsharp.base_address, pending.size, desc.is_written,
                                          desc.is_formatted, pending.buffer_id);
            const u64 offset_aligned = Common::AlignDown(offset, u64{pending.alignment});
            const u32 adjust = static_cast<u32>(offset - offset_aligned);
            ASSERT(adjust % 4 == 0);
            push_data.AddOffset(pending.buffer_binding, adjust);
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, pending.size + adjust);
            if (auto barrier =
                    vk_buffer->GetBarrier(desc.is_written ? vk::AccessFlagBits2::eShaderWrite
                                                          : vk::AccessFlagBits2::eShaderRead,
                                          vk::PipelineStageFlagBits2::eAllCommands)) {
                buffer_barriers.emplace_back(*barrier);
            }
            if (desc.is_written && desc.is_formatted) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, pending.size);
            }
        }

        auto& set_write = set_writes[pending.set_write_index];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = pending.unified_binding;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = pending.is_storage ? vk::DescriptorType::eStorageBuffer
                                                      : vk::DescriptorType::eUniformBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        pending.finalized = true;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    const u32 stage_index = static_cast<u32>(stage.l_stage);
    ASSERT(stage_index < MaxShaderStages);
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    for (u32 image_index = 0; image_index < stage.images.size(); ++image_index) {
        const auto& image_desc = stage.images[image_index];
        const auto tsharp = GetResolvedImage(stage, image_index);
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
            const u32 cache_index = image_bindings.size();
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            if (cache_index < Shader::NUM_IMAGES) {
                auto& cached = cached_image_bindings[stage_index][cache_index];
                cached.valid = false;
                cached.owner = &stage;
            }
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(tsharp);

        for (u32 i = 0; i < num_bindings; ++i) {
            const u32 cache_index = image_bindings.size();
            ASSERT(cache_index < Shader::NUM_IMAGES);

            auto& cached = cached_image_bindings[stage_index][cache_index];
            if (cached.owner != &stage) {
                cached = {};
                cached.owner = &stage;
            }

            const bool sharp_matches = std::memcmp(&cached.sharp, &tsharp, sizeof(tsharp)) == 0;
            const bool cache_hit = cached.valid && sharp_matches &&
                                   cached.topology_epoch == texture_cache.TopologyEpoch() &&
                                   texture_cache.TryReuseImage(cached.image_id, cached.image_uid,
                                                               cached.topology_epoch);
            if (telemetry_enabled) {
                Common::PerformanceTelemetry::AddEnabled(
                    cache_hit ? Common::PerformanceTelemetry::Counter::ImageTokenHits
                              : Common::PerformanceTelemetry::Counter::ImageTokenMisses,
                    1);
            }

            if (cache_hit) {
                image_bindings.emplace_back(cached.image_id, cached.resolved_desc);
            } else {
                auto& [image_id, desc] = image_bindings.emplace_back(
                    std::piecewise_construct, std::tuple{}, std::tuple{tsharp, image_desc});
                if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                    ASSERT(num_bindings == 1);
                    desc.view_info.range.base.level += image_desc.constant_mip_index;
                    desc.view_info.range.extent.levels = 1;
                } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                    desc.view_info.range.base.level += i;
                    desc.view_info.range.extent.levels = 1;
                }
                image_id = texture_cache.FindImage(desc);
            }

            auto& [image_id, desc] = image_bindings.back();
            const bool cache_needs_update = !cache_hit;
            auto* image = &texture_cache.GetImage(image_id);
            if (cache_needs_update) {
                cached.sharp = tsharp;
                cached.image_id = image_id;
                cached.image_uid = image->image_uid;
                cached.topology_epoch = texture_cache.TopologyEpoch();
                cached.resolved_desc = desc;
                cached.valid = true;
            }

            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (image->binding.is_bound) {
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    u32 texture_binding_index = 0;
    for (auto& [image_id, desc] : image_bindings) {
        auto& cached_view = cached_texture_views[stage_index][texture_binding_index++];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            cached_view.valid = false;
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
                cached_view.valid = false;
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            texture_cache.PrepareTexture(image_id, desc);
            const u64 topology_epoch = texture_cache.TopologyEpoch();
            const bool view_cache_hit = cached_view.valid && cached_view.image_id == image_id &&
                                        cached_view.image_uid == image.image_uid &&
                                        cached_view.topology_epoch == topology_epoch &&
                                        cached_view.backing_image == image.GetImage() &&
                                        cached_view.info == desc.view_info;
            vk::ImageView image_view_handle{};
            if (view_cache_hit) {
                image_view_handle = cached_view.image_view;
            } else {
                auto& image_view = image.FindView(desc.view_info);
                image_view_handle = *image_view.image_view;
                cached_view = {
                    .image_id = image_id,
                    .image_uid = image.image_uid,
                    .topology_epoch = topology_epoch,
                    .backing_image = image.GetImage(),
                    .image_view = image_view_handle,
                    .info = image_view.info,
                    .valid = true,
                };
            }

            if ((image.binding.force_general || image.binding.is_target) &&
                !image.info.props.is_depth) {
                image.Transit(instance.IsAttachmentFeedbackLoopLayoutSupported() &&
                                      image.binding.is_target
                                  ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                  : vk::ImageLayout::eGeneral,
                              vk::AccessFlagBits2::eShaderRead |
                                  (image.info.props.is_depth
                                       ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                       : vk::AccessFlagBits2::eColorAttachmentWrite |
                                             vk::AccessFlagBits2::eColorAttachmentRead),
                              {});
            } else if (is_storage) {
                image.Transit(vk::ImageLayout::eGeneral,
                              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                              desc.view_info.range);
            } else {
                const auto new_layout = image.info.props.is_depth
                                            ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                            : vk::ImageLayout::eShaderReadOnlyOptimal;
                image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, image_view_handle,
                                     image.backing->state.layout);
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (u32 sampler_index = 0; sampler_index < stage.samplers.size(); ++sampler_index) {
        const auto& sampler = stage.samplers[sampler_index];
        auto ssharp = GetResolvedSampler(stage, sampler_index);
        if (sampler.disable_aniso) {
            const auto tsharp = GetResolvedImage(stage, sampler.associated_image);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        const auto vk_sampler = texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->regs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state{};
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
            cached_color_target_views[cb].valid = false;
        }
        texture_cache.UpdateImage(image_id);
        image->SetBackingSamples(key.color_samples[cb]);
        texture_cache.PrepareRenderTarget(image_id, desc);
        auto& cached_view = cached_color_target_views[cb];
        const u64 topology_epoch = texture_cache.TopologyEpoch();
        const bool view_cache_hit = cached_view.valid && cached_view.image_id == image_id &&
                                    cached_view.image_uid == image->image_uid &&
                                    cached_view.topology_epoch == topology_epoch &&
                                    cached_view.backing_image == image->GetImage() &&
                                    cached_view.info == desc.view_info;
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                view_cache_hit ? Common::PerformanceTelemetry::Counter::RenderTargetHits
                               : Common::PerformanceTelemetry::Counter::RenderTargetMisses,
                1);
        }
        if (!view_cache_hit) {
            auto& image_view = image->FindView(desc.view_info, false);
            cached_view = {
                .image_id = image_id,
                .image_uid = image->image_uid,
                .topology_epoch = topology_epoch,
                .backing_image = image->GetImage(),
                .image_view = *image_view.image_view,
                .info = image_view.info,
                .valid = true,
            };
        }
        const auto slice = cached_view.info.range.base.layer;
        const auto mip = cached_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            image->Transit(instance.IsAttachmentFeedbackLoopLayoutSupported()
                               ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                               : vk::ImageLayout::eGeneral,
                           vk::AccessFlagBits2::eColorAttachmentWrite, {});
            attachment_feedback_loop = true;
        } else {
            image->Transit(vk::ImageLayout::eColorAttachmentOptimal,
                           vk::AccessFlagBits2::eColorAttachmentWrite |
                               vk::AccessFlagBits2::eColorAttachmentRead,
                           desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, cached_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = cached_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        texture_cache.PrepareDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);
        const u64 topology_epoch = texture_cache.TopologyEpoch();
        const bool view_cache_hit = cached_depth_target_view.valid &&
                                    cached_depth_target_view.image_id == image_id &&
                                    cached_depth_target_view.image_uid == image.image_uid &&
                                    cached_depth_target_view.topology_epoch == topology_epoch &&
                                    cached_depth_target_view.backing_image == image.GetImage() &&
                                    cached_depth_target_view.info == desc.view_info;
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                view_cache_hit ? Common::PerformanceTelemetry::Counter::RenderTargetHits
                               : Common::PerformanceTelemetry::Counter::RenderTargetMisses,
                1);
        }
        if (!view_cache_hit) {
            auto& image_view = image.FindView(desc.view_info, false);
            cached_depth_target_view = {
                .image_id = image_id,
                .image_uid = image.image_uid,
                .topology_epoch = topology_epoch,
                .backing_image = image.GetImage(),
                .image_view = *image_view.image_view,
                .info = image_view.info,
                .valid = true,
            };
        }

        const auto slice = cached_depth_target_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        image.Transit(new_layout,
                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                          vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                      desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers =
            std::min<u32>(state.num_layers, cached_depth_target_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = cached_depth_target_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->regs.color_buffers[0].Address(),
                                 liverpool->regs.color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->regs;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    read_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                       sub_range);
    write_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        sub_range);

    auto aspect_mask = vk::ImageAspectFlags(0);
    if (is_depth) {
        aspect_mask |= vk::ImageAspectFlagBits::eDepth;
    }
    if (is_stencil) {
        aspect_mask |= vk::ImageAspectFlagBits::eStencil;
    }

    vk::ImageCopy region = {
        .srcSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .srcOffset = {0, 0, 0},
        .dstSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .dstOffset = {0, 0, 0},
        .extent = {write_image.info.size.width, write_image.info.size.height, 1},
    };
    scheduler.CommandBuffer().copyImage(read_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                        write_image.GetImage(),
                                        vk::ImageLayout::eTransferDstOptimal, region);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    buffer_cache.CopyBuffer(dst, src, num_bytes, dst_gds, src_gds);
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.InvalidateMemory(addr, size);
    page_manager.NotifyWrite(addr, size, VideoCore::MemoryWriteSource::Cpu);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size);
    return true;
}

void Rasterizer::NotifyMemoryWrite(VAddr addr, u64 size, VideoCore::MemoryWriteSource source) {
    page_manager.NotifyWrite(addr, size, source);
}

bool Rasterizer::ProcessDownloadImages() {
    return texture_cache.ProcessDownloadImages();
}

void Rasterizer::DeferGpuCompletion(Common::UniqueFunction<void>&& callback) {
    scheduler.DeferPriorityOperation(std::move(callback));
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    const u64 generation = liverpool->GraphicsStateGeneration();
    const bool state_unchanged = dynamic_state_generation == generation &&
                                 dynamic_state_pipeline == pipeline &&
                                 dynamic_state_indexed == is_indexed &&
                                 dynamic_state_feedback_loop == attachment_feedback_loop;
    if (state_unchanged) [[likely]] {
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::DynamicStateHits, 1);
        }
    } else {
        UpdateViewportScissorState();
        UpdateDepthStencilState();
        UpdatePrimitiveState(is_indexed);
        UpdateRasterizationState();
        UpdateColorBlendingState(pipeline);
        dynamic_state_generation = generation;
        dynamic_state_pipeline = pipeline;
        dynamic_state_indexed = is_indexed;
        dynamic_state_feedback_loop = attachment_feedback_loop;
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::DynamicStateMisses, 1);
        }
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.Commit(instance, scheduler.CommandBuffer());
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->regs;

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max(scr, std::max(s16(win + win_offset), s16(gen + win_offset)));
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min(scr, std::min(s16(win + win_offset), s16(gen + win_offset)));
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        ReportUnsupportedWindowOffset();
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    u32 first = AmdGpu::NUM_VIEWPORTS;
#pragma clang loop unroll(disable)
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; ++i) {
        const u32 xscale = std::bit_cast<u32>(regs.viewports[i].xscale);
        if ((xscale << 1) == 0) {
            continue;
        }
        if (first != AmdGpu::NUM_VIEWPORTS) [[unlikely]] {
            SetMultipleViewportScissorState(instance, regs, scsr, dynamic_state);
            return;
        }
        first = i;
    }

    if (first == AmdGpu::NUM_VIEWPORTS) {
        constexpr vk::Viewport empty_viewport{
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor{
            .offset = {0, 0},
            .extent = {1, 1},
        };
        dynamic_state.SetSingleViewportScissor(empty_viewport, empty_scissor);
        return;
    }

    dynamic_state.SetSingleViewportScissor(MakeViewport(instance, regs, first),
                                           MakeViewportScissor(regs, scsr, first));
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        dynamic_state.SetStencilReferences(front.stencil_test_val, back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

} // namespace Vulkan
