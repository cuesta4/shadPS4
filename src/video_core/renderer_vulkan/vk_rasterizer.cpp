// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <optional>

#include "common/debug.h"
#include "common/vector_bytes.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_depth_stencil_state.h"
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

namespace {

[[nodiscard]] bool IsBufferDescriptorType(const vk::DescriptorType type) {
    return type == vk::DescriptorType::eUniformBuffer ||
           type == vk::DescriptorType::eStorageBuffer ||
           type == vk::DescriptorType::eUniformBufferDynamic ||
           type == vk::DescriptorType::eStorageBufferDynamic;
}

[[nodiscard]] bool EqualDescriptorImageInfos(
    const boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES>& lhs,
    const boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t index = 0; index < lhs.size(); ++index) {
        if (lhs[index].sampler != rhs[index].sampler ||
            lhs[index].imageView != rhs[index].imageView ||
            lhs[index].imageLayout != rhs[index].imageLayout) {
            return false;
        }
    }
    return true;
}

} // namespace

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

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_},
      high_draw_call_optimization{EmulatorSettings.IsHighDrawCallOptimization()},
      page_manager{this},
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
    liverpool->InvalidateGraphicsPipelineRevision();
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
    const bool enabled = high_draw_call_optimization;
    const auto& regs = liverpool->regs;
    std::optional<RenderTargetPlanKey> plan_key;
    if (enabled) {
        plan_key.emplace();
        plan_key->pipeline = pipeline;
        std::memcpy(plan_key->color_buffers.data(), regs.color_buffers, sizeof(regs.color_buffers));
        plan_key->color_extents = liverpool->last_cb_extent;
        plan_key->color_control = regs.color_control;
        plan_key->color_target_mask = regs.color_target_mask;
        plan_key->depth_buffer = regs.depth_buffer;
        plan_key->depth_view = regs.depth_view;
        plan_key->depth_control = regs.depth_control;
        plan_key->depth_htile_data_base = regs.depth_htile_data_base;
        plan_key->depth_extent = liverpool->last_db_extent;
    }

    render_target_plan_hit =
        enabled && render_target_plan.valid &&
        render_target_plan.texture_generation == texture_cache.BindingGeneration() &&
        render_target_plan.key == *plan_key;
    if (render_target_plan_hit) {
        const auto num_color_attachments = std::bit_width(pipeline->GetGraphicsKey().mrt_mask);
        for (u32 cb = 0; cb < num_color_attachments; ++cb) {
            const auto image_id = cb_descs[cb].first;
            if (!image_id) {
                continue;
            }
            bound_images.emplace_back(image_id);
            texture_cache.GetImage(image_id).binding.is_target = 1u;
        }
        if (const auto image_id = db_desc.first; image_id) {
            bound_images.emplace_back(image_id);
            texture_cache.GetImage(image_id).binding.is_target = 1u;
        }
        return;
    }

    if (enabled) {
        render_target_plan.color_views = {};
        render_target_plan.depth_view = {};
    }

    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen).
    const auto& key = pipeline->GetGraphicsKey();
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
        texture_cache.GetImage(image_id).binding.is_target = 1u;
    }

    const auto depth_stencil = GetEffectiveDepthStencilState(regs);
    if (depth_stencil.needs_attachment) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->last_db_extent;
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        texture_cache.GetImage(image_id).binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }

    if (enabled) {
        render_target_plan.valid = true;
        render_target_plan.key = *plan_key;
        render_target_plan.texture_generation = texture_cache.BindingGeneration();
    } else {
        render_target_plan = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::Info& info,
    const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader) {
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

void Rasterizer::BeginDescriptorWritePlan(const Pipeline* pipeline) {
    if (!high_draw_call_optimization) {
        set_writes.clear();
        return;
    }
    descriptor_write_plan_candidate =
        descriptor_write_plan.valid && descriptor_write_plan.pipeline == pipeline;
    descriptor_write_plan_hit = descriptor_write_plan_candidate;
}

void Rasterizer::EnsureDescriptorWriteCapacity(const u32 required_size) {
    if (set_writes.size() < required_size) {
        set_writes.resize(required_size);
    }
}

void Rasterizer::SetBufferDescriptorWrite(const u32 binding, const vk::DescriptorType type,
                                          const vk::DescriptorBufferInfo* info) {
    EnsureDescriptorWriteCapacity(set_write_index + 1);
    auto& write = set_writes[set_write_index++];
    const bool metadata_matches = write.dstBinding == binding && write.dstArrayElement == 0 &&
                                  write.descriptorCount == 1 && write.descriptorType == type;
    if (descriptor_write_plan_hit && !metadata_matches) {
        descriptor_write_plan_hit = false;
    }
    if (!descriptor_write_plan_hit) {
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = type;
    }
    write.dstSet = VK_NULL_HANDLE;
    write.pBufferInfo = info;
    write.pImageInfo = nullptr;
    write.pTexelBufferView = nullptr;
}

void Rasterizer::SetImageDescriptorWrite(const u32 binding, const u32 count,
                                         const vk::DescriptorType type,
                                         const vk::DescriptorImageInfo* info) {
    EnsureDescriptorWriteCapacity(set_write_index + 1);
    auto& write = set_writes[set_write_index++];
    const bool metadata_matches = write.dstBinding == binding && write.dstArrayElement == 0 &&
                                  write.descriptorCount == count && write.descriptorType == type;
    if (descriptor_write_plan_hit && !metadata_matches) {
        descriptor_write_plan_hit = false;
    }
    if (!descriptor_write_plan_hit) {
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorCount = count;
        write.descriptorType = type;
    }
    write.dstSet = VK_NULL_HANDLE;
    write.pBufferInfo = nullptr;
    write.pImageInfo = info;
    write.pTexelBufferView = nullptr;
}

void Rasterizer::ReserveOrSetImageDescriptorWrite(const u32 binding, const u32 count,
                                                  const vk::DescriptorType type,
                                                  const vk::DescriptorImageInfo* info) {
    if (!high_draw_call_optimization || !descriptor_partial_materialization_candidate) {
        SetImageDescriptorWrite(binding, count, type, info);
        return;
    }

    EnsureDescriptorWriteCapacity(set_write_index + 1);
    const u32 write_index = set_write_index++;
    auto& write = set_writes[write_index];
    const bool metadata_matches = write.dstBinding == binding && write.dstArrayElement == 0 &&
                                  write.descriptorCount == count && write.descriptorType == type;
    if (descriptor_write_plan_hit && !metadata_matches) {
        descriptor_write_plan_hit = false;
    }
    if (!descriptor_write_plan_hit) {
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorCount = count;
        write.descriptorType = type;
    }
    write.dstSet = VK_NULL_HANDLE;
    write.pBufferInfo = nullptr;
    write.pImageInfo = nullptr;
    write.pTexelBufferView = nullptr;
    deferred_image_writes.push_back({
        .write_index = write_index,
        .image_info_index = static_cast<u32>(info - image_infos.data()),
    });
}

void Rasterizer::MaterializeDeferredImageDescriptorWrites() {
    for (const auto& deferred : deferred_image_writes) {
        auto& write = set_writes[deferred.write_index];
        write.dstSet = VK_NULL_HANDLE;
        write.pBufferInfo = nullptr;
        write.pImageInfo = &image_infos[deferred.image_info_index];
        write.pTexelBufferView = nullptr;
    }
}

bool Rasterizer::PartialPushStateMatches(const Pipeline* pipeline) const {
    const auto& state = descriptor_partial_push_state;
    return state.valid && !pipeline->IsCompute() && pipeline->UsesPushDescriptors() &&
           state.pipeline == pipeline && state.cmdbuf == scheduler.CommandBuffer() &&
           state.texture_generation == texture_cache.BindingGeneration() &&
           state.graphics_push_descriptor_epoch == scheduler.GraphicsPushDescriptorEpoch();
}

void Rasterizer::FinalizeDeferredImageWrites(const Pipeline* pipeline) {
    descriptor_partial_image_writes_omitted = false;
    if (deferred_image_writes.empty()) {
        return;
    }

    const auto& state = descriptor_partial_push_state;
    bool has_buffer_write = false;
    bool has_storage_image = false;
    for (u32 index = 0; index < set_write_index; ++index) {
        const auto type = set_writes[index].descriptorType;
        has_buffer_write |= IsBufferDescriptorType(type);
        has_storage_image |= type == vk::DescriptorType::eStorageImage;
    }

    const bool stable_state = descriptor_partial_materialization_candidate &&
                              PartialPushStateMatches(pipeline) &&
                              state.write_count == set_write_index && descriptor_write_plan_hit;
    const bool images_equal =
        stable_state && EqualDescriptorImageInfos(state.image_infos, image_infos);
    const bool eligible =
        images_equal && buffer_barriers.empty() && has_buffer_write && !has_storage_image;
    if (!eligible) {
        MaterializeDeferredImageDescriptorWrites();
    } else {
        descriptor_partial_image_writes_omitted = true;
    }
}

void Rasterizer::FinalizeDescriptorWritePlan(const Pipeline* pipeline) {
    if (set_writes.size() != set_write_index) {
        set_writes.resize(set_write_index);
    }
    if (!high_draw_call_optimization) {
        return;
    }
    if (!descriptor_write_plan_candidate || !descriptor_write_plan_hit) {
        descriptor_write_plan.valid = true;
        descriptor_write_plan.pipeline = pipeline;
    }
}

void Rasterizer::ResetDescriptorPartialPushState() {
    descriptor_partial_push_state = {};
    descriptor_partial_writes.clear();
    deferred_image_writes.clear();
    descriptor_partial_materialization_candidate = false;
    descriptor_partial_image_writes_omitted = false;
}

void Rasterizer::BindPipelineResources(const Pipeline* pipeline) {
    const bool enabled = high_draw_call_optimization;
    const auto full_push = [&](const bool keep_graphics_state) {
        if (descriptor_partial_image_writes_omitted) {
            MaterializeDeferredImageDescriptorWrites();
            descriptor_partial_image_writes_omitted = false;
        }
        pipeline->BindResources(set_writes, buffer_barriers, push_data);
        if (!enabled || !keep_graphics_state || set_writes.empty()) {
            ResetDescriptorPartialPushState();
            return;
        }
        descriptor_partial_push_state.valid = true;
        descriptor_partial_push_state.pipeline = pipeline;
        descriptor_partial_push_state.cmdbuf = scheduler.CommandBuffer();
        descriptor_partial_push_state.texture_generation = texture_cache.BindingGeneration();
        descriptor_partial_push_state.graphics_push_descriptor_epoch =
            scheduler.GraphicsPushDescriptorEpoch();
        descriptor_partial_push_state.write_count = set_write_index;
        descriptor_partial_push_state.image_infos = image_infos;
        deferred_image_writes.clear();
        descriptor_partial_materialization_candidate = false;
        descriptor_partial_image_writes_omitted = false;
    };

    if (!enabled || pipeline->IsCompute() || !pipeline->UsesPushDescriptors()) {
        full_push(enabled && !pipeline->IsCompute() && pipeline->UsesPushDescriptors());
        return;
    }

    bool has_buffer_write = false;
    bool has_image_write = false;
    bool has_storage_image = false;
    u32 current_buffer_writes{};
    for (const auto& write : set_writes) {
        if (IsBufferDescriptorType(write.descriptorType)) {
            has_buffer_write = true;
            ++current_buffer_writes;
        } else {
            has_image_write = true;
            has_storage_image |= write.descriptorType == vk::DescriptorType::eStorageImage;
        }
    }

    const bool eligible =
        PartialPushStateMatches(pipeline) && descriptor_write_plan_hit &&
        descriptor_partial_push_state.write_count == set_write_index &&
        (descriptor_partial_image_writes_omitted ||
         EqualDescriptorImageInfos(descriptor_partial_push_state.image_infos, image_infos)) &&
        !has_storage_image && has_buffer_write && has_image_write && buffer_barriers.empty();
    if (!eligible) {
        full_push(true);
        return;
    }

    descriptor_partial_writes.clear();
    if (descriptor_partial_writes.capacity() < current_buffer_writes) {
        descriptor_partial_writes.reserve(current_buffer_writes);
    }
    for (const auto& write : set_writes) {
        if (IsBufferDescriptorType(write.descriptorType)) {
            descriptor_partial_writes.push_back(write);
        }
    }
    pipeline->BindResources(descriptor_partial_writes, buffer_barriers, push_data);
    descriptor_partial_push_state.graphics_push_descriptor_epoch =
        scheduler.GraphicsPushDescriptorEpoch();
    deferred_image_writes.clear();
    descriptor_partial_materialization_candidate = false;
    descriptor_partial_image_writes_omitted = false;
}

void Rasterizer::BindGraphicsPipelineIfNeeded(const vk::CommandBuffer cmdbuf,
                                              const vk::Pipeline pipeline) {
    const bool enabled = high_draw_call_optimization;
    if (enabled && last_graphics_cmdbuf == cmdbuf && last_bound_graphics_pipeline == pipeline) {
        return;
    }
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    if (enabled) {
        last_graphics_cmdbuf = cmdbuf;
        last_bound_graphics_pipeline = pipeline;
    } else {
        last_graphics_cmdbuf = nullptr;
        last_bound_graphics_pipeline = nullptr;
    }
}

void Rasterizer::ResetCachedCommandBufferState() {
    last_graphics_cmdbuf = nullptr;
    last_bound_graphics_pipeline = nullptr;
    ResetDescriptorPartialPushState();
    buffer_cache.ResetCachedBindings();
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
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
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(index_offset, buffer_barriers);
    }
    BindPipelineResources(pipeline);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);
    const auto cmdbuf = scheduler.CommandBuffer();
    BindGraphicsPipelineIfNeeded(cmdbuf, pipeline->Handle());
    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    ResetBindings();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    RENDERER_TRACE;
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
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(0, buffer_barriers);
    }

    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false, false, {});
    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) =
            buffer_cache.ObtainBuffer(count_address, 4, false, false, {});
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
    const auto cmdbuf = scheduler.CommandBuffer();
    BindGraphicsPipelineIfNeeded(cmdbuf, pipeline->Handle());
    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);
        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_base, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);
        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_base,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
    }
    ResetBindings();
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;
    liverpool->InvalidateGraphicsPipelineRevision();
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
    ResetBindings();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;
    liverpool->InvalidateGraphicsPipelineRevision();
    scheduler.PopPendingOperations();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline || !BindResources(pipeline)) {
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
    ResetBindings();
}

u64 Rasterizer::Flush() {
    liverpool->InvalidateGraphicsPipelineRevision();
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    ResetCachedCommandBufferState();
    return current_tick;
}

void Rasterizer::Finish() {
    liverpool->InvalidateGraphicsPipelineRevision();
    scheduler.Finish();
    ResetCachedCommandBufferState();
}

void Rasterizer::OnSubmit() {
    liverpool->InvalidateGraphicsPipelineRevision();
    ResetCachedCommandBufferState();
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    buffer_cache.RunGarbageCollector();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (pipeline->IsCompute()) {
        if (IsComputeMetaClear(pipeline) || IsComputeImageCopy(pipeline) ||
            IsComputeImageClear(pipeline)) {
            return false;
        }
    }

    set_write_index = 0;
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();
    deferred_image_writes.clear();
    descriptor_partial_image_writes_omitted = false;
    BeginDescriptorWritePlan(pipeline);
    descriptor_partial_materialization_candidate =
        high_draw_call_optimization && PartialPushStateMatches(pipeline);

    bool uses_dma = false;
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        EnsureDescriptorWriteCapacity(set_write_index + stage->buffers.size() +
                                      stage->images.size() + stage->samplers.size());
        stage->PushUd(binding, push_data);
        BindBuffers(*stage, binding, push_data);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
    }

    if (uses_dma) {
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (auto& range : mapped_ranges) {
            buffer_cache.SynchronizeBuffersInRange(range.lower(), range.upper() - range.lower());
        }
        fault_process_pending = true;
    }

    FinalizeDeferredImageWrites(pipeline);
    FinalizeDescriptorWritePlan(pipeline);
    return true;
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
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
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
            const VAddr address = desc.GetSharp(info).base_address;
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
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
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
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
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

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    buffer_bindings.clear();
    for (const auto& desc : stage.buffers) {
        const auto vsharp = desc.GetSharp(stage);
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            buffer_bindings.emplace_back(buffer_cache.FindBuffer(vsharp.base_address, size), vsharp,
                                         size);
        } else {
            buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0);
        }
    }

    // Second pass to re-bind buffers that were updated after binding.
    for (u32 i = 0; i < buffer_bindings.size(); ++i) {
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& desc = stage.buffers[i];
        const bool is_storage = desc.IsStorage(vsharp);
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        if (!buffer_id) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto& cs_program = liverpool->GetCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id);
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = offset - offset_aligned;
            ASSERT(adjust % 4 == 0);
            push_data.AddOffset(binding.buffer, adjust);
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            if (auto barrier =
                    vk_buffer->GetBarrier(desc.is_written ? vk::AccessFlagBits2::eShaderWrite
                                                          : vk::AccessFlagBits2::eShaderRead,
                                          vk::PipelineStageFlagBits2::eAllCommands)) {
                buffer_barriers.emplace_back(*barrier);
            }
            if (desc.is_written && desc.is_formatted) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
        }

        SetBufferDescriptorWrite(binding.unified++,
                                 is_storage ? vk::DescriptorType::eStorageBuffer
                                            : vk::DescriptorType::eUniformBuffer,
                                 &buffer_infos.back());
        ++binding.buffer;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;
    boost::container::static_vector<AmdGpu::Image, Shader::NUM_IMAGES> image_sharps;
    image_sharps.reserve(stage.images.size());
    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        image_sharps.emplace_back(tsharp);
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }
    }

    const bool enabled = high_draw_call_optimization;
    auto& texture_plan = texture_binding_plans[static_cast<u32>(stage.l_stage)];
    bool texture_plan_hit = enabled && texture_plan.valid && texture_plan.stage == &stage &&
                            texture_plan.texture_generation == texture_cache.BindingGeneration() &&
                            Common::EqualVectorBytes(texture_plan.image_sharps, image_sharps);
    if (texture_plan_hit) {
        for (const auto& [image_id, _] : texture_plan.image_bindings) {
            if (image_id && texture_cache.GetImage(image_id).binding.needs_rebind) {
                texture_plan_hit = false;
                texture_plan.valid = false;
                break;
            }
        }
    }

    if (texture_plan_hit) {
        image_bindings = texture_plan.image_bindings;
        image_descriptor_array_sizes = texture_plan.image_descriptor_array_sizes;
        for (auto& [image_id, desc] : image_bindings) {
            if (!image_id) {
                continue;
            }
            auto& image = texture_cache.GetImage(image_id);
            if (image.binding.is_bound) {
                image.binding.force_general |=
                    desc.type == VideoCore::TextureCache::BindingType::Storage;
            }
            image.binding.is_bound = 1u;
        }
    } else {
        u32 sharp_index = 0;
        for (const auto& image_desc : stage.images) {
            const auto tsharp = image_sharps[sharp_index++];
            if (tsharp.Address() == 0 ||
                tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
                image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
                image_descriptor_array_sizes.push_back(1);
                continue;
            }
            const auto mip_fallback_mode = image_desc.mip_fallback_mode;
            const u32 num_bindings = image_desc.NumBindings(stage);
            for (u32 i = 0; i < num_bindings; ++i) {
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
                auto* image = &texture_cache.GetImage(image_id);
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
    }

    boost::container::static_vector<CachedTextureView, Shader::NUM_IMAGES> resolved_views;
    resolved_views.reserve(image_bindings.size());
    u32 image_binding_index{};
    for (auto& [image_id, desc] : image_bindings) {
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            resolved_views.emplace_back();
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }
            bound_images.emplace_back(image_id);
            auto& image = texture_cache.GetImage(image_id);
            texture_cache.PrepareTexture(image_id, desc);
            image.SetBackingSamples(image.info.num_samples);

            vk::ImageView image_view_handle{};
            const bool cached_view_valid =
                texture_plan_hit && image_binding_index < texture_plan.texture_views.size() &&
                texture_plan.texture_views[image_binding_index].valid &&
                texture_plan.texture_views[image_binding_index].image_id == image_id &&
                texture_plan.texture_views[image_binding_index].backing_image == image.GetImage();
            if (cached_view_valid) {
                image_view_handle = texture_plan.texture_views[image_binding_index].view;
            } else {
                auto& image_view = image.FindView(desc.view_info);
                image_view_handle = *image_view.image_view;
            }
            resolved_views.push_back({
                .image_id = image_id,
                .backing_image = image.GetImage(),
                .view = image_view_handle,
                .valid = true,
            });

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
        ++image_binding_index;
    }

    boost::container::small_vector<TextureDescriptorWritePlan, 8> descriptor_writes;
    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    u32 descriptor_index{};
    for (u32 array_size : image_descriptor_array_sizes) {
        vk::DescriptorType descriptor_type;
        if (texture_plan_hit && descriptor_index < texture_plan.descriptor_writes.size() &&
            texture_plan.descriptor_writes[descriptor_index].count == array_size) {
            descriptor_type = texture_plan.descriptor_writes[descriptor_index].type;
        } else {
            const auto& [_, desc] = image_bindings[image_binding_idx];
            descriptor_type = desc.type == VideoCore::TextureCache::BindingType::Storage
                                  ? vk::DescriptorType::eStorageImage
                                  : vk::DescriptorType::eSampledImage;
        }
        descriptor_writes.push_back({.type = descriptor_type, .count = array_size});
        ReserveOrSetImageDescriptorWrite(binding.unified, array_size, descriptor_type,
                                         &image_infos[image_info_idx]);
        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
        ++descriptor_index;
    }

    boost::container::static_vector<AmdGpu::Sampler, Shader::NUM_SAMPLERS> sampler_sharps;
    boost::container::static_vector<vk::Sampler, Shader::NUM_SAMPLERS> resolved_samplers;
    sampler_sharps.reserve(stage.samplers.size());
    resolved_samplers.reserve(stage.samplers.size());
    const VAddr border_color_address = liverpool->regs.ta_bc_base.Address<VAddr>();
    u32 sampler_index{};
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(stage);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        sampler_sharps.push_back(ssharp);
        vk::Sampler vk_sampler;
        const bool sampler_hit = texture_plan_hit &&
                                 texture_plan.border_color_address == border_color_address &&
                                 sampler_index < texture_plan.sampler_sharps.size() &&
                                 sampler_index < texture_plan.samplers.size() &&
                                 texture_plan.sampler_sharps[sampler_index] == ssharp;
        if (sampler_hit) {
            vk_sampler = texture_plan.samplers[sampler_index];
        } else {
            vk_sampler = texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base);
        }
        resolved_samplers.push_back(vk_sampler);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        ReserveOrSetImageDescriptorWrite(binding.unified++, 1, vk::DescriptorType::eSampler,
                                         &image_infos.back());
        ++sampler_index;
    }

    if (enabled) {
        texture_plan.valid = true;
        texture_plan.stage = &stage;
        texture_plan.texture_generation = texture_cache.BindingGeneration();
        texture_plan.border_color_address = border_color_address;
        texture_plan.image_sharps = image_sharps;
        texture_plan.image_bindings = image_bindings;
        texture_plan.image_descriptor_array_sizes = image_descriptor_array_sizes;
        texture_plan.descriptor_writes = descriptor_writes;
        texture_plan.texture_views = resolved_views;
        texture_plan.sampler_sharps = sampler_sharps;
        texture_plan.samplers = resolved_samplers;
    }
}

std::pair<vk::ImageView, VideoCore::SubresourceRange> Rasterizer::ResolveRenderTargetView(
    CachedRenderTargetView& cached_view, VideoCore::ImageId image_id, VideoCore::Image& image,
    const VideoCore::TextureCache::ImageDesc& desc) {
    const bool cached_view_valid =
        high_draw_call_optimization && render_target_plan.valid &&
        render_target_plan.texture_generation == texture_cache.BindingGeneration() &&
        cached_view.valid && cached_view.image_id == image_id &&
        cached_view.backing_image == image.GetImage();
    if (cached_view_valid) {
        return {cached_view.view, cached_view.range};
    }
    const auto& image_view = image.FindView(desc.view_info, false);
    const auto handle = *image_view.image_view;
    if (high_draw_call_optimization) {
        cached_view = {
            .image_id = image_id,
            .backing_image = image.GetImage(),
            .view = handle,
            .range = image_view.info.range,
            .valid = true,
        };
    }
    return {handle, image_view.info.range};
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->regs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
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
            render_target_plan.color_views[cb] = {};
        }
        texture_cache.UpdateImage(image_id);
        image->SetBackingSamples(key.color_samples[cb]);
        texture_cache.PrepareRenderTarget(image_id, desc);

        const auto [image_view_handle, view_range] =
            ResolveRenderTargetView(render_target_plan.color_views[cb], image_id, *image, desc);

        const auto slice = view_range.base.layer;
        const auto mip = view_range.base.level;
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
        state.num_layers = std::min<u32>(state.num_layers, view_range.extent.layers);
        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = image_view_handle;
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
        const auto [image_view_handle, view_range] =
            ResolveRenderTargetView(render_target_plan.depth_view, image_id, image, desc);
        const auto slice = view_range.base.layer;
        const bool is_depth_clear = regs.depth_render_control.depth_clear_enable ||
                                    texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);
        const bool has_stencil = image.info.props.has_stencil;
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
        state.num_layers = std::min<u32>(state.num_layers, view_range.extent.layers);
        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = image_view_handle;
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
    if (high_draw_call_optimization &&
        render_target_plan.texture_generation != texture_cache.BindingGeneration()) {
        render_target_plan = {};
        render_target_plan_hit = false;
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
    liverpool->InvalidateGraphicsPipelineRevision();
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    liverpool->InvalidateGraphicsPipelineRevision();
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
    liverpool->InvalidateGraphicsPipelineRevision();
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.ReadMemory(addr, size);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size);
    texture_cache.ReadMemory(addr, size);
    return true;
}

bool Rasterizer::ProcessDownloadImages() {
    return texture_cache.ProcessDownloadImages();
}

void Rasterizer::InsertGuestSyncBarrier(const GuestSyncDomain domain) {
    scheduler.EndRendering();

    vk::PipelineStageFlags2 src_stage;
    vk::AccessFlags2 src_access;
    switch (domain) {
    case GuestSyncDomain::ComputeShader:
        src_stage = vk::PipelineStageFlagBits2::eComputeShader;
        src_access = vk::AccessFlagBits2::eShaderWrite;
        break;
    case GuestSyncDomain::PixelShader:
        src_stage = vk::PipelineStageFlagBits2::eFragmentShader;
        src_access = vk::AccessFlagBits2::eShaderWrite;
        break;
    case GuestSyncDomain::EndOfPipe:
        src_stage = vk::PipelineStageFlagBits2::eAllCommands;
        src_access = vk::AccessFlagBits2::eMemoryWrite;
        break;
    }

    const vk::MemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
}

u64 Rasterizer::FlushGuestCompletionPoint() {
    if (scheduler.HasUnsubmittedGpuWork()) {
        return Flush();
    }
    return scheduler.LastSubmittedTick();
}

void Rasterizer::DeferGuestCompletion(const u64 tick,
                                      Common::UniqueFunction<void>&& callback) {
    scheduler.DeferPriorityOperationAt(tick, std::move(callback));
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
    liverpool->InvalidateGraphicsPipelineRevision();
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    liverpool->InvalidateGraphicsPipelineRevision();
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.Commit(instance, scheduler.CommandBuffer());
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->regs;

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
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

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
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

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_stencil = GetEffectiveDepthStencilState(regs);
    const bool depth_test_enabled = depth_stencil.depth_test_enable;
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    dynamic_state.SetDepthWriteEnabled(depth_stencil.depth_write_enable &&
                                       !regs.depth_render_control.depth_clear_enable);
    if (depth_test_enabled) {
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const bool depth_bounds_test_enabled = depth_stencil.depth_bounds_enable;
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

    const bool stencil_test_enabled = depth_stencil.stencil_test_enable;
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
