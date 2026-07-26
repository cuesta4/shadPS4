// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "common/debug.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/gpu_thread_dispatcher.h"
#include "video_core/renderer_vulkan/execution/graphics_operation_classifier.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/preparation/graphics_request_builder.h"
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

static Shader::PushData MakeUserData(const VideoCore::GraphicsDynamicState& state) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = state.viewport_control.xoffset_enable ? state.viewports[0].xoffset : 0.f;
    push_data.xscale = state.viewport_control.xscale_enable ? state.viewports[0].xscale : 1.f;
    push_data.yoffset = state.viewport_control.yoffset_enable ? state.viewports[0].yoffset : 0.f;
    push_data.yscale = state.viewport_control.yscale_enable ? state.viewports[0].yscale : 1.f;
    return push_data;
}

static void TrackResolvedBuffer(CommandPreparationContext& context,
                                const VideoCore::BufferRegistry& registry,
                                VideoCore::BufferId id, vk::Buffer buffer, u64 offset,
                                u64 size) {
    if (!id || !buffer) {
        return;
    }
    const u32 generation = registry.Generation(id);
    const auto duplicate = std::ranges::find_if(context.resolved_buffers, [&](const auto& ref) {
        return ref.id == id && ref.buffer == buffer && ref.offset == offset && ref.size == size;
    });
    if (duplicate == context.resolved_buffers.end()) {
        context.resolved_buffers.push_back({
            .id = id,
            .generation = generation,
            .buffer = buffer,
            .offset = offset,
            .size = size,
        });
    }
}

static void PlanBufferAccess(CommandPreparationContext& context,
                             const VideoCore::BufferRegistry& registry,
                             VideoCore::BufferId id, VideoCore::Buffer& buffer,
                             vk::AccessFlags2 access, vk::PipelineStageFlagBits2 stage,
                             u32 offset = 0) {
    context.buffer_accesses.push_back({
        .resource = &buffer,
        .id = id,
        .generation = id ? registry.Generation(id) : 0,
        .resource_uid = buffer.Uid(),
        .access = access,
        .stage = stage,
        .offset = offset,
    });
}

static vk::ImageLayout PlanImageTransition(
    CommandPreparationContext& context, const VideoCore::ImageRegistry& registry,
    VideoCore::ImageId id, VideoCore::Image& image, vk::ImageLayout layout,
    vk::AccessFlags2 access, std::optional<VideoCore::SubresourceRange> subresource) {
    const auto* entry = registry.Find(id);
    ASSERT(entry != nullptr);
    const vk::PipelineStageFlags2 stage =
        (access == vk::AccessFlagBits2::eTransferRead ||
         access == vk::AccessFlagBits2::eTransferWrite)
            ? vk::PipelineStageFlagBits2::eTransfer
            : vk::PipelineStageFlagBits2::eAllGraphics |
                  vk::PipelineStageFlagBits2::eComputeShader;
    context.image_transitions.push_back({
        .resource = &image,
        .id = id,
        .generation = entry->backing_generation,
        .resource_uid = image.image_uid,
        .layout = layout,
        .access = access,
        .stage = stage,
        .subresource = subresource,
    });
    return layout;
}

static void TrackResolvedImage(CommandPreparationContext& context,
                               const VideoCore::ImageRegistry& registry,
                               VideoCore::ImageId id, const VideoCore::Image& image,
                               VideoCore::ImageViewId view_id, vk::ImageView view,
                               VideoCore::SubresourceRange subresource) {
    if (!id || !view_id || !view) {
        return;
    }
    const auto* entry = registry.Find(id);
    ASSERT(entry != nullptr);
    const auto duplicate = std::ranges::find_if(context.resolved_images, [&](const auto& ref) {
        return ref.id == id && ref.view == view && ref.subresource == subresource;
    });
    if (duplicate == context.resolved_images.end()) {
        context.resolved_images.push_back({
            .id = id,
            .generation = entry->backing_generation,
            .view_id = view_id,
            .view_generation = entry->view_generation,
            .image = image.GetImage(),
            .view = view,
            .aspects = image.aspect_mask,
            .subresource = subresource,
        });
    }
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::GpuThreadDispatcher* gpu_thread_dispatcher)
    : instance{instance_}, scheduler{scheduler_},
      memory_coordinator{mapped_ranges}, page_manager{&memory_coordinator},
      transfer_buffers{instance, scheduler},
      tile_manager{instance, scheduler, transfer_buffers.Get(VideoCore::MemoryUsage::Stream)},
      alias_coordinator{
          buffer_registry,
          image_registry,
          [this](VAddr address, u32 size) {
              return buffer_cache.ObtainBufferForImage(address, size);
          },
          [this](VAddr address) { return texture_cache.ClearMetadata(address); },
          [this](VAddr address, u32 size) { return texture_cache.HasImage(address, size); },
          [this](VAddr address, u32 size) {
              texture_cache.InvalidateImagesFromGpu(address, size);
          },
          [this](VideoCore::Buffer& buffer, VAddr address, u32 size) {
              return texture_cache.SynchronizeBufferFromImage(buffer, address, size);
          }},
      buffer_cache{instance, scheduler, gpu_thread_dispatcher, transfer_buffers, alias_coordinator,
                   page_manager, buffer_access_states},
       texture_cache{instance, scheduler, transfer_buffers, alias_coordinator, tile_manager,
                     page_manager, image_access_states},
       memory{Core::Memory::Instance()}, pipeline_cache{instance, scheduler},
       descriptor_binder{instance, scheduler},
       resource_committer{buffer_registry, image_registry, buffer_access_states,
                          image_access_states},
       command_recorder{instance, scheduler, descriptor_binder} {
    memory_coordinator.Bind(buffer_cache, texture_cache, page_manager);
    memory->SetGpuMemoryObserver(&memory_coordinator);
}

Rasterizer::~Rasterizer() {
    memory->SetGpuMemoryObserver(nullptr);
    page_manager.StopFaultHandling();
    memory_coordinator.Unbind();
}

void Rasterizer::Execute(const VideoCore::DrawCommand& command) {
    Draw(command);
}

void Rasterizer::Execute(const VideoCore::DrawIndirectCommand& command) {
    DrawIndirect(command);
}

void Rasterizer::Execute(const VideoCore::DispatchDirectCommand& command) {
    DispatchDirect(command);
}

void Rasterizer::Execute(const VideoCore::DispatchIndirectCommand& command) {
    DispatchIndirect(command);
}

void Rasterizer::Execute(const VideoCore::FillBufferCommand& command) {
    FillBuffer(command.address, command.size, command.value, command.is_gds);
}

void Rasterizer::Execute(const VideoCore::CopyBufferCommand& command) {
    CopyBuffer(command.destination, command.source, command.size, command.destination_is_gds,
               command.source_is_gds);
}

void Rasterizer::CpSync() {
    command_recorder.EndRendering();
    command_recorder.RecordComputeToIndirectBarrier();
}

bool Rasterizer::FilterDraw(const VideoCore::CapturedGraphicsState& state) {
    const auto classification = ClassifyGraphicsOperation(state);
    switch (classification.operation) {
    case GraphicsOperation::EliminateFastClear:
        EliminateFastClear(state);
        return false;
    case GraphicsOperation::FmaskDecompress:
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    case GraphicsOperation::Resolve:
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve(state);
        return false;
    case GraphicsOperation::SkipPrimitiveNone:
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    case GraphicsOperation::DepthStencilCopy:
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(state, classification.copy_depth, classification.copy_stencil);
        return false;
    case GraphicsOperation::Draw:
        return true;
    }
    return true;
}

void Rasterizer::PrepareRenderState(CommandPreparationContext& context,
                                    const VideoCore::CapturedGraphicsState& state,
                                    const u32 mrt_mask) {
    const auto& attachments = state.attachments;
    const bool color_disabled =
        state.pipeline.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    u32 active_color_count{};
    for (u32 slot = 0; slot < attachments.color_buffers.size(); ++slot) {
        const auto& descriptor = attachments.color_buffers[slot];
        active_color_count += !color_disabled && descriptor &&
                              state.pipeline.color_target_mask.GetMask(slot) != 0 &&
                              (mrt_mask & (1U << slot)) != 0;
    }
    const bool depth_enabled =
        (attachments.depth_control.depth_enable && attachments.depth_buffer.DepthValid()) ||
        (attachments.depth_control.stencil_enable && attachments.depth_buffer.StencilValid());
    context.resolution->render_targets.reserve(
        active_color_count + static_cast<u32>(depth_enabled));
    for (u32 slot = 0; slot < attachments.color_buffers.size(); ++slot) {
        const auto& descriptor = attachments.color_buffers[slot];
        if (color_disabled || !descriptor ||
            state.pipeline.color_target_mask.GetMask(slot) == 0 ||
            (mrt_mask & (1U << slot)) == 0) {
            continue;
        }
        auto& [image_id, desc] = context.resolution->AddColorTarget(
            slot, VideoCore::TextureCache::ImageDesc(
                      descriptor, attachments.color_extent_hints[slot]));
        image_id =
            texture_cache.FindImage(desc, false, &context.resolution->image_uses);
        context.resolution->bound_images.emplace_back(image_id);
        context.resolution->image_uses.Get(image_id).is_target = true;
    }

    if (depth_enabled) {
        auto& [image_id, desc] = context.resolution->AddDepthTarget(
            VideoCore::TextureCache::ImageDesc(
                attachments.depth_buffer, attachments.depth_view, attachments.depth_control,
                attachments.depth_htile_data_base.GetAddress(),
                attachments.depth_extent_hint));
        image_id =
            texture_cache.FindImage(desc, false, &context.resolution->image_uses);
        context.resolution->bound_images.emplace_back(image_id);
        context.resolution->image_uses.Get(image_id).is_target = true;
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const VideoCore::GraphicsDrawState& draw, const Shader::ShaderInvocationData& invocation,
    const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = draw.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = invocation.user_data[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = invocation.user_data[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear(const VideoCore::CapturedGraphicsState& state) {
    const auto& col_buf = state.attachments.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, state.attachments.color_extent_hints[0]);
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

void Rasterizer::Draw(const VideoCore::DrawCommand& command) {
    CommandResolutionScratch scratch;
    for (u32 retry = 0; !PrepareDraw(command, retry, scratch); ++retry) {
    }
}

bool Rasterizer::PrepareDraw(const VideoCore::DrawCommand& command, u32 retry_count,
                             CommandResolutionScratch& scratch) {
    RENDERER_TRACE;

    PreparedGraphicsCommand prepared;
    CommandPreparationContext context{scratch};

    scheduler.PopPendingOperations();

    if (!FilterDraw(command.state)) {
        return true;
    }

    auto lookup = pipeline_cache.GetGraphicsPipeline(command.state);
    if (!lookup) {
        return true;
    }
    const GraphicsPipeline* pipeline = lookup.pipeline;
    const auto vertex_request = InputRequestBuilder::BuildVertexRequest(
        *pipeline, lookup.invocations[u32(Shader::LogicalStage::Vertex)], command.state.draw,
        instance);
    const auto index_request =
        InputRequestBuilder::BuildIndexRequest(command.state.draw, command.index_offset);

    BuildShaderResourceRequests(context.resolution->resource_requests, pipeline->GetStages(),
                                lookup.invocations, false);
    PrepareRenderState(context, command.state, pipeline->GetGraphicsKey().mrt_mask);
    if (!BindResources(context, pipeline, context.resolution->resource_requests,
                       command.state.resources, &command.state.dynamic, nullptr)) {
        return true;
    }
    prepared.render_state = BeginRendering(context, pipeline, command.state);

    prepared.vertex_input =
        buffer_cache.PrepareVertexBuffers(vertex_request, context.buffer_accesses);
    if (command.is_indexed) {
        prepared.index_input =
            buffer_cache.PrepareIndexBuffer(index_request, context.buffer_accesses);
    }

    prepared.provenance = command.provenance;
    prepared.pipeline = pipeline;
    prepared.dynamic_state = GraphicsRequestBuilder::BuildDynamicStatePlan(
        command.state, *pipeline, instance, command.is_indexed,
        context.attachment_feedback_loop);

    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(
        command.state.draw, lookup.invocations[u32(Shader::LogicalStage::Vertex)], fetch_shader);
    prepared.direct = {
        .indexed = command.is_indexed,
        .index_count = command.state.draw.num_indices,
        .instance_count = command.state.draw.num_instances.NumInstances(),
        .vertex_offset = vertex_offset,
        .instance_offset = instance_offset,
    };
    context.buffer_registry_epoch = buffer_registry.Epoch();
    context.image_registry_epoch = image_registry.Epoch();
    const bool committed = resource_committer.Commit(context);
    if (!committed) {
        if (retry_count > 0) {
            LOG_WARNING(Render_Vulkan,
                        "Replanning graphics command {} after resource generation changed "
                        "again (attempt {})",
                        command.provenance.sequence, retry_count + 2);
        }
        return false;
    }
    prepared.commit = {
        .sequence = command.provenance.sequence,
        .validated = true,
    };
    prepared.resources = std::move(context).Finish();
    command_recorder.Record(prepared);

    return true;
}

void Rasterizer::DrawIndirect(const VideoCore::DrawIndirectCommand& command) {
    CommandResolutionScratch scratch;
    for (u32 retry = 0; !PrepareDrawIndirect(command, retry, scratch); ++retry) {
    }
}

bool Rasterizer::PrepareDrawIndirect(const VideoCore::DrawIndirectCommand& command,
                                     u32 retry_count, CommandResolutionScratch& scratch) {
    RENDERER_TRACE;

    PreparedGraphicsCommand prepared;
    CommandPreparationContext context{scratch};

    scheduler.PopPendingOperations();

    if (!FilterDraw(command.state)) {
        return true;
    }

    auto lookup = pipeline_cache.GetGraphicsPipeline(command.state);
    if (!lookup) {
        return true;
    }
    const GraphicsPipeline* pipeline = lookup.pipeline;
    const auto vertex_request = InputRequestBuilder::BuildVertexRequest(
        *pipeline, lookup.invocations[u32(Shader::LogicalStage::Vertex)], command.state.draw,
        instance);
    const auto index_request =
        InputRequestBuilder::BuildIndexRequest(command.state.draw, 0);

    BuildShaderResourceRequests(context.resolution->resource_requests, pipeline->GetStages(),
                                lookup.invocations, false);
    PrepareRenderState(context, command.state, pipeline->GetGraphicsKey().mrt_mask);
    if (!BindResources(context, pipeline, context.resolution->resource_requests,
                       command.state.resources, &command.state.dynamic, nullptr)) {
        return true;
    }
    prepared.render_state = BeginRendering(context, pipeline, command.state);

    prepared.vertex_input =
        buffer_cache.PrepareVertexBuffers(vertex_request, context.buffer_accesses);
    if (command.is_indexed) {
        prepared.index_input =
            buffer_cache.PrepareIndexBuffer(index_request, context.buffer_accesses);
    }
    const VAddr argument_address = command.argument_address + command.offset;
    const u32 argument_size = command.stride * command.max_count;
    const auto arguments =
        buffer_cache.ResolveBuffer(argument_address, argument_size, false, false);
    auto* buffer = arguments.buffer;
    const u32 base = arguments.offset;

    VideoCore::Buffer* count_buffer{};
    VideoCore::BufferId count_id{};
    u32 count_base{};
    if (command.count_address != 0) {
        const auto count = buffer_cache.ResolveBuffer(command.count_address, 4, false, false);
        count_buffer = count.buffer;
        count_id = count.id;
        count_base = count.offset;
    }

    PlanBufferAccess(context, buffer_registry, arguments.id, *buffer,
                     vk::AccessFlagBits2::eIndirectCommandRead,
                     vk::PipelineStageFlagBits2::eDrawIndirect);
    if (count_buffer) {
        PlanBufferAccess(context, buffer_registry, count_id, *count_buffer,
                         vk::AccessFlagBits2::eIndirectCommandRead,
                         vk::PipelineStageFlagBits2::eDrawIndirect);
    }
    TrackResolvedBuffer(context, buffer_registry, arguments.id,
                        buffer->Handle(), base, command.stride * command.max_count);
    if (count_buffer) {
        TrackResolvedBuffer(context, buffer_registry, count_id, count_buffer->Handle(),
                            count_base, 4);
    }

    prepared.provenance = command.provenance;
    prepared.pipeline = pipeline;
    prepared.dynamic_state = GraphicsRequestBuilder::BuildDynamicStatePlan(
        command.state, *pipeline, instance, command.is_indexed,
        context.attachment_feedback_loop);
    prepared.is_indirect = true;
    prepared.indirect = {
        .indexed = command.is_indexed,
        .arguments = buffer->Handle(),
        .argument_offset = base,
        .count_buffer = count_buffer ? count_buffer->Handle() : vk::Buffer{},
        .count_offset = count_base,
        .max_count = command.max_count,
        .stride = command.stride,
    };
    context.buffer_registry_epoch = buffer_registry.Epoch();
    context.image_registry_epoch = image_registry.Epoch();
    const bool committed = resource_committer.Commit(context);
    if (!committed) {
        if (retry_count > 0) {
            LOG_WARNING(Render_Vulkan,
                        "Replanning indirect graphics command {} after resource generation "
                        "changed again (attempt {})",
                        command.provenance.sequence, retry_count + 2);
        }
        return false;
    }
    prepared.commit = {
        .sequence = command.provenance.sequence,
        .validated = true,
    };
    prepared.resources = std::move(context).Finish();
    command_recorder.Record(prepared);

    return true;
}

void Rasterizer::DispatchDirect(const VideoCore::DispatchDirectCommand& command) {
    CommandResolutionScratch scratch;
    for (u32 retry = 0; !PrepareDispatchDirect(command, retry, scratch); ++retry) {
    }
}

bool Rasterizer::PrepareDispatchDirect(const VideoCore::DispatchDirectCommand& command,
                                       u32 retry_count, CommandResolutionScratch& scratch) {
    RENDERER_TRACE;

    PreparedComputeCommand prepared;
    CommandPreparationContext context{scratch};

    scheduler.PopPendingOperations();

    const auto& cs_program = command.state.program;
    auto lookup = pipeline_cache.GetComputePipeline(command.state);
    if (!lookup) {
        return true;
    }
    const ComputePipeline* pipeline = lookup.pipeline;

    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (ExecuteShaderHLE(cs, lookup.invocation, cs_program,
                         ShaderHleServices{buffer_cache, command_recorder})) {
        return true;
    }

    const std::span invocations{&lookup.invocation, 1};
    BuildShaderResourceRequests(context.resolution->resource_requests, pipeline->GetStages(),
                                invocations, true);
    if (!BindResources(context, pipeline, context.resolution->resource_requests,
                       command.state.resources, nullptr, &command.state)) {
        return true;
    }

    prepared.provenance = command.provenance;
    prepared.pipeline = pipeline;
    prepared.dispatch = ComputeRequestBuilder::BuildDispatchPlan(command.state);
    context.buffer_registry_epoch = buffer_registry.Epoch();
    context.image_registry_epoch = image_registry.Epoch();
    const bool committed = resource_committer.Commit(context);
    if (!committed) {
        if (retry_count > 0) {
            LOG_WARNING(Render_Vulkan,
                        "Replanning compute command {} after resource generation changed "
                        "again (attempt {})",
                        command.provenance.sequence, retry_count + 2);
        }
        return false;
    }
    prepared.commit = {
        .sequence = command.provenance.sequence,
        .validated = true,
    };
    prepared.resources = std::move(context).Finish();
    command_recorder.Record(prepared);

    return true;
}

void Rasterizer::DispatchIndirect(const VideoCore::DispatchIndirectCommand& command) {
    CommandResolutionScratch scratch;
    for (u32 retry = 0; !PrepareDispatchIndirect(command, retry, scratch); ++retry) {
    }
}

bool Rasterizer::PrepareDispatchIndirect(const VideoCore::DispatchIndirectCommand& command,
                                         u32 retry_count, CommandResolutionScratch& scratch) {
    RENDERER_TRACE;

    PreparedComputeCommand prepared;
    CommandPreparationContext context{scratch};

    scheduler.PopPendingOperations();

    auto lookup = pipeline_cache.GetComputePipeline(command.state);
    if (!lookup) {
        return true;
    }
    const ComputePipeline* pipeline = lookup.pipeline;

    const std::span invocations{&lookup.invocation, 1};
    BuildShaderResourceRequests(context.resolution->resource_requests, pipeline->GetStages(),
                                invocations, true);
    if (!BindResources(context, pipeline, context.resolution->resource_requests,
                       command.state.resources, nullptr, &command.state)) {
        return true;
    }
    const VAddr argument_address = command.argument_address + command.offset;
    const auto arguments =
        buffer_cache.ResolveBuffer(argument_address, command.size, false, false);
    auto* buffer = arguments.buffer;
    const u32 base = arguments.offset;
    PlanBufferAccess(context, buffer_registry, arguments.id, *buffer,
                     vk::AccessFlagBits2::eIndirectCommandRead,
                     vk::PipelineStageFlagBits2::eDrawIndirect);
    TrackResolvedBuffer(context, buffer_registry, arguments.id,
                        buffer->Handle(), base, command.size);

    prepared.provenance = command.provenance;
    prepared.pipeline = pipeline;
    prepared.is_indirect = true;
    prepared.indirect_arguments = buffer->Handle();
    prepared.indirect_offset = base;
    context.buffer_registry_epoch = buffer_registry.Epoch();
    context.image_registry_epoch = image_registry.Epoch();
    const bool committed = resource_committer.Commit(context);
    if (!committed) {
        if (retry_count > 0) {
            LOG_WARNING(Render_Vulkan,
                        "Replanning indirect compute command {} after resource generation "
                        "changed again (attempt {})",
                        command.provenance.sequence, retry_count + 2);
        }
        return false;
    }
    prepared.commit = {
        .sequence = command.provenance.sequence,
        .validated = true,
    };
    prepared.resources = std::move(context).Finish();
    command_recorder.Record(prepared);

    return true;
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

bool Rasterizer::BindResources(
    CommandPreparationContext& context, const Pipeline* pipeline,
    const ShaderResourceRequestSet& requests,
    const VideoCore::GraphicsResourceState& resource_state,
    const VideoCore::GraphicsDynamicState* graphics_state,
    const VideoCore::CapturedComputeState* compute_state) {
    if (pipeline->IsCompute()) {
        ASSERT(requests.active_stage_count == 1 && compute_state != nullptr);
        const auto& invocation =
            *requests.stages[requests.active_stage_indices[0]].invocation;
        if (IsComputeImageCopy(pipeline, invocation, *compute_state) ||
            IsComputeMetaClear(pipeline, invocation) ||
            IsComputeImageClear(pipeline, invocation, *compute_state)) {
            return false;
        }
    }

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    context.push_data = graphics_state ? MakeUserData(*graphics_state) : Shader::PushData{};
    context.buffer_infos.reserve(requests.buffers.size());
    context.image_infos.reserve(requests.image_descriptor_count);
    context.descriptor_writes.reserve(requests.descriptor_write_count);
    context.resolved_buffers.reserve(requests.buffers.size());
    context.resolved_images.reserve(requests.image_descriptor_count);
    context.resolution->bound_images.reserve(requests.image_descriptor_count + 9);

    for (const u8 stage_index : requests.ActiveStageIndices()) {
        const auto& stage_requests = requests.stages[stage_index];
        const auto& analysis = *stage_requests.analysis;
        const auto& invocation = *stage_requests.invocation;
        invocation.PushUd(analysis, binding, context.push_data);
        BindBuffers(context, requests.Buffers(stage_requests), invocation, binding,
                    context.push_data, compute_state);
        BindTextures(context, requests.Images(stage_requests),
                     requests.Samplers(stage_requests), binding, resource_state);
        uses_dma |= analysis.uses_dma;
    }

    if (uses_dma) {
        // We only use fault buffer for DMA right now.
        mapped_ranges.ForEach([&](VAddr address, u64 size) {
            buffer_cache.SynchronizeBuffersInRange(address, size);
        });
        fault_process_pending = true;
    }

    return true;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline,
                                    const Shader::ShaderInvocationData& invocation) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(invocation).base_address;
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
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(invocation).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline,
                                    const Shader::ShaderInvocationData& invocation,
                                    const VideoCore::CapturedComputeState& state) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = state.program;
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
    const AmdGpu::Buffer buf0 = desc0.GetSharp(invocation);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(invocation);
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

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline,
                                     const Shader::ShaderInvocationData& invocation,
                                     const VideoCore::CapturedComputeState& state) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = state.program;
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
    const AmdGpu::Buffer buf0 = desc0.GetSharp(invocation);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(invocation);
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

void Rasterizer::BindBuffers(CommandPreparationContext& context,
                             std::span<const BufferResourceRequest> requests,
                             const Shader::ShaderInvocationData& invocation,
                             Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data,
                             const VideoCore::CapturedComputeState* compute_state) {
    for (const auto& request : requests) {
        const auto& vsharp = request.sharp;
        u64 size{};
        if (!request.is_special && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
        }

        const bool is_storage = request.is_storage;
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        const bool is_regular_buffer =
            !request.is_special && vsharp.base_address != 0 && size > 0;
        if (!is_regular_buffer) {
            if (request.type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                context.buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
            } else if (request.type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = invocation.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(invocation.flattened_ud_buf.data(), ubo_size, alignment);
                context.buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (request.type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                context.buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (request.type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                context.buffer_infos.emplace_back(fault_buffer->Handle(), 0,
                                                  fault_buffer->SizeBytes());
            } else if (request.type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                ASSERT(compute_state != nullptr);
                const auto& cs_program = compute_state->program;
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                context.buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                context.buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto resolved = buffer_cache.ResolveBuffer(
                vsharp.base_address, static_cast<u32>(size), request.is_written,
                request.is_formatted);
            auto* vk_buffer = resolved.buffer;
            const u32 offset = resolved.offset;
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = offset - offset_aligned;
            ASSERT(adjust % 4 == 0);
            push_data.AddOffset(binding.buffer, adjust);
            context.buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            TrackResolvedBuffer(context, buffer_registry, resolved.id, vk_buffer->Handle(),
                                offset_aligned, size + adjust);
            PlanBufferAccess(context, buffer_registry, resolved.id, *vk_buffer,
                             request.is_written ? vk::AccessFlagBits2::eShaderWrite
                                                : vk::AccessFlagBits2::eShaderRead,
                             vk::PipelineStageFlagBits2::eAllCommands);
            if (request.is_written && request.is_formatted) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
        }

        context.descriptor_writes.push_back({
            .binding = binding.unified++,
            .array_element = 0,
            .descriptor_count = 1,
            .info_index = static_cast<u32>(context.buffer_infos.size() - 1),
            .type = is_storage ? vk::DescriptorType::eStorageBuffer
                               : vk::DescriptorType::eUniformBuffer,
            .info_kind = DescriptorInfoKind::Buffer,
        });
        ++binding.buffer;
    }
}

void Rasterizer::BindTextures(CommandPreparationContext& context,
                              std::span<const ImageResourceRequest> image_requests,
                              std::span<const SamplerResourceRequest> sampler_requests,
                              Shader::Backend::Bindings& binding,
                              const VideoCore::GraphicsResourceState& resource_state) {
    context.resolution->image_bindings.clear();
    const u32 first_image_idx = context.image_infos.size();
    // For loading/storing to explicit mip levels, when no native instruction support, bind an array
    // of descriptors consecutively, 1 for each mip level. The shader can index this with LOD
    // operand.
    // This array holds the size of each consecutive array with the number of bindings consumed.
    // This is currently always 1 for anything other than mip fallback arrays.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    size_t image_binding_count{};
    for (const auto& request : image_requests) {
        image_binding_count += request.binding_count;
    }
    context.resolution->image_bindings.reserve(image_binding_count);
    image_descriptor_array_sizes.reserve(image_requests.size());

    for (const auto& request : image_requests) {
        const auto& image_desc = request.descriptor;
        const auto& tsharp = request.sharp;
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
            context.resolution->image_bindings.emplace_back(
                std::piecewise_construct, std::tuple{}, std::tuple{});
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = request.binding_count;

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = context.resolution->image_bindings.emplace_back(
                std::piecewise_construct, std::tuple{}, std::tuple{tsharp, image_desc});

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            image_id =
                texture_cache.FindImage(desc, false, &context.resolution->image_uses);
            auto* image = &texture_cache.GetImage(image_id);
            if (const auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                image_id = depth_image_id;
            }
            auto& use = context.resolution->image_uses.Get(image_id);
            if (use.is_bound) {
                use.force_general |=
                    desc.type == VideoCore::TextureCache::BindingType::Storage;
            }
            use.is_bound = true;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    for (auto& [image_id, desc] : context.resolution->image_bindings) {
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            context.image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE,
                                             vk::ImageLayout::eGeneral);
        } else {
            if (context.resolution->image_uses.NeedsRebind(image_id)) {
                context.resolution->image_uses.Erase(image_id);
                image_id = texture_cache.FindImage(desc, false, &context.resolution->image_uses);
            }

            context.resolution->bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(image_id, desc);

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            const auto& use = context.resolution->image_uses.Get(image_id);
            vk::ImageLayout descriptor_layout{};
            if ((use.force_general || use.is_target) && !image.info.props.is_depth) {
                descriptor_layout = PlanImageTransition(
                    context, image_registry, image_id, image,
                    instance.IsAttachmentFeedbackLoopLayoutSupported() && use.is_target
                        ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                        : vk::ImageLayout::eGeneral,
                    vk::AccessFlagBits2::eShaderRead |
                        vk::AccessFlagBits2::eColorAttachmentWrite |
                        vk::AccessFlagBits2::eColorAttachmentRead,
                    {});
            } else {
                if (is_storage) {
                    descriptor_layout = PlanImageTransition(
                        context, image_registry, image_id, image, vk::ImageLayout::eGeneral,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    descriptor_layout =
                        PlanImageTransition(context, image_registry, image_id, image, new_layout,
                                            vk::AccessFlagBits2::eShaderRead,
                                            desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            context.image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                             descriptor_layout);
            TrackResolvedImage(context, image_registry, image_id, image,
                               image.FindViewId(desc.view_info), *image_view.image_view,
                               desc.view_info.range);
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = context.resolution->image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        context.descriptor_writes.push_back({
            .binding = binding.unified,
            .array_element = 0,
            .descriptor_count = array_size,
            .info_index = image_info_idx,
            .type = is_storage ? vk::DescriptorType::eStorageImage
                               : vk::DescriptorType::eSampledImage,
            .info_kind = DescriptorInfoKind::Image,
        });

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (const auto& sampler : sampler_requests) {
        const auto vk_sampler =
            texture_cache.GetSampler(sampler.sharp, resource_state.border_color_base);
        context.image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        context.descriptor_writes.push_back({
            .binding = binding.unified++,
            .array_element = 0,
            .descriptor_count = 1,
            .info_index = static_cast<u32>(context.image_infos.size() - 1),
            .type = vk::DescriptorType::eSampler,
            .info_kind = DescriptorInfoKind::Image,
        });
    }
}

RenderState Rasterizer::BeginRendering(CommandPreparationContext& context,
                                       const GraphicsPipeline* pipeline,
                                       const VideoCore::CapturedGraphicsState& captured) {
    context.attachment_feedback_loop = false;
    const auto& regs = captured.attachments;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto* target = context.resolution->ColorTarget(cb);
        if (target == nullptr) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto& [image_id, desc] = *target;
        auto* image = &texture_cache.GetImage(image_id);
        if (context.resolution->image_uses.NeedsRebind(image_id)) {
            context.resolution->image_uses.Erase(image_id);
            image_id = context.resolution->bound_images.emplace_back(
                texture_cache.FindImage(desc, false, &context.resolution->image_uses));
            context.resolution->image_uses.Get(image_id).is_target = true;
            image = &texture_cache.GetImage(image_id);
        }
        const auto& image_view =
            texture_cache.FindRenderTarget(image_id, desc, key.color_samples[cb]);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        const auto& use = context.resolution->image_uses.Get(image_id);
        vk::ImageLayout attachment_layout{};
        if (use.is_bound) {
            ASSERT_MSG(!use.force_general,
                       "Having image both as storage and render target is unsupported");
            attachment_layout = PlanImageTransition(
                context, image_registry, image_id, *image,
                instance.IsAttachmentFeedbackLoopLayoutSupported()
                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                    : vk::ImageLayout::eGeneral,
                vk::AccessFlagBits2::eColorAttachmentWrite, {});
            context.attachment_feedback_loop = true;
        } else {
            attachment_layout = PlanImageTransition(
                context, image_registry, image_id, *image,
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::AccessFlagBits2::eColorAttachmentWrite |
                    vk::AccessFlagBits2::eColorAttachmentRead,
                desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = attachment_layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;
        TrackResolvedImage(context, image_registry, image_id, *image,
                           image->FindViewId(desc.view_info), *image_view.image_view,
                           desc.view_info.range);

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto* target = context.resolution->DepthTarget(); target != nullptr) {
        auto& [image_id, desc] = *target;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear = regs.depth_render_control.depth_clear_enable ||
                                    texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 &&
               !context.resolution->image_uses.NeedsRebind(image_id));

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
        PlanImageTransition(context, image_registry, image_id, image, new_layout,
                            vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                            desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = new_layout;
        attachment.clear_value = {};
        TrackResolvedImage(context, image_registry, image_id, image,
                           image.FindViewId(desc.view_info), *image_view.image_view,
                           desc.view_info.range);

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

void Rasterizer::Resolve(const VideoCore::CapturedGraphicsState& state) {
    const auto& mrt0_hint = state.attachments.color_extent_hints[0];
    const auto& mrt1_hint = state.attachments.color_extent_hints[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{state.attachments.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{state.attachments.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 state.attachments.color_buffers[0].Address(),
                                 state.attachments.color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(const VideoCore::CapturedGraphicsState& state, bool is_depth,
                                  bool is_stencil) {
    const auto& regs = state.attachments;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), regs.depth_extent_hint, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), regs.depth_extent_hint, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = regs.depth_view.slice_start;
    sub_range.extent.layers = regs.depth_view.NumSlices() - sub_range.base.layer;

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
    command_recorder.RecordImageCopy(
        read_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
        write_image.GetImage(), vk::ImageLayout::eTransferDstOptimal, region);

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

void Rasterizer::ProcessDownloadImages() {
    texture_cache.ProcessDownloadImages();
}

void Rasterizer::ScopeMarkerBegin(std::string_view str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    command_recorder.BeginDebugLabel(str);
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    command_recorder.EndDebugLabel();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    command_recorder.InsertDebugLabel(str);
}

void Rasterizer::ScopedMarkerInsertColor(std::string_view str, u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    command_recorder.InsertDebugLabel(
        str, std::array<f32, 4>{
                 static_cast<f32>((color >> 16) & 0xff) / 255.0f,
                 static_cast<f32>((color >> 8) & 0xff) / 255.0f,
                 static_cast<f32>(color & 0xff) / 255.0f,
                 static_cast<f32>((color >> 24) & 0xff) / 255.0f,
             });
}

} // namespace Vulkan
