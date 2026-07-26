// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_command_recorder.h"
#include "video_core/renderer_vulkan/vk_descriptor_binder.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"

namespace Vulkan {

void VulkanCommandRecorder::RecordComputeToIndirectBarrier() const {
    const vk::MemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    scheduler.CommandBuffer().pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eDrawIndirect,
        vk::DependencyFlagBits::eByRegion, barrier, {}, {});
}

void VulkanCommandRecorder::RecordMemoryBarrier(
    vk::PipelineStageFlags source_stage, vk::PipelineStageFlags destination_stage,
    vk::AccessFlags source_access, vk::AccessFlags destination_access) const {
    const vk::MemoryBarrier barrier{
        .srcAccessMask = source_access,
        .dstAccessMask = destination_access,
    };
    scheduler.CommandBuffer().pipelineBarrier(
        source_stage, destination_stage, vk::DependencyFlagBits::eByRegion, barrier, {}, {});
}

void VulkanCommandRecorder::RecordBufferCopies(
    vk::Buffer source, vk::Buffer destination,
    std::span<const vk::BufferCopy> copies) const {
    scheduler.CommandBuffer().copyBuffer(source, destination, copies);
}

void VulkanCommandRecorder::RecordImageCopy(
    vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
    vk::ImageLayout destination_layout, const vk::ImageCopy& region) const {
    scheduler.CommandBuffer().copyImage(source, source_layout, destination,
                                        destination_layout, region);
}

void VulkanCommandRecorder::BeginDebugLabel(std::string_view label) const {
    scheduler.CommandBuffer().beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = label.data(),
    });
}

void VulkanCommandRecorder::EndDebugLabel() const {
    scheduler.CommandBuffer().endDebugUtilsLabelEXT();
}

void VulkanCommandRecorder::InsertDebugLabel(
    std::string_view label, const std::array<f32, 4>& color) const {
    scheduler.CommandBuffer().insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = label.data(),
        .color = color,
    });
}

void VulkanCommandRecorder::BindVertexInput(const VertexInputPlan& plan) const {
    const auto cmdbuf = scheduler.CommandBuffer();
    if (plan.use_dynamic_vertex_input) {
        cmdbuf.setVertexInputEXT(plan.bindings, plan.attributes);
    }
    if (plan.Empty()) {
        return;
    }
    const u32 count = static_cast<u32>(plan.buffers.size());
    if (plan.use_dynamic_vertex_input) {
        cmdbuf.bindVertexBuffers(0, count, plan.buffers.data(), plan.offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, count, plan.buffers.data(), plan.offsets.data(),
                                  plan.sizes.data(), plan.strides.data());
    }
}

void VulkanCommandRecorder::BindDescriptors(
    const Pipeline& pipeline, const PreparedResourcePayload& resources,
    vk::CommandBuffer command_buffer) const {
    descriptor_binder.Bind(pipeline, resources.descriptor_writes, resources.buffer_infos,
                           resources.image_infos, resources.buffer_barriers,
                           resources.push_data, command_buffer);
}

void VulkanCommandRecorder::RecordImageBarriers(PreparedResourcePayload& resources) const {
    if (resources.image_barriers.empty()) {
        return;
    }
    scope_controller.End();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(resources.image_barriers.size()),
        .pImageMemoryBarriers = resources.image_barriers.data(),
    });
    resources.image_barriers.clear();
}

void VulkanCommandRecorder::Record(PreparedGraphicsCommand& command) const {
    ASSERT(command.pipeline != nullptr);
    ASSERT(command.commit.validated);
    const Pipeline& pipeline = *command.pipeline;
    const auto cmdbuf = scheduler.CommandBuffer();

    RecordImageBarriers(command.resources);
    BindVertexInput(command.vertex_input);
    if (command.index_input.IsValid()) {
        cmdbuf.bindIndexBuffer(command.index_input.buffer, command.index_input.offset,
                               command.index_input.type);
    }
    BindDescriptors(pipeline, command.resources, cmdbuf);
    dynamic_emitter.Emit(command.dynamic_state, scheduler.GetDynamicState(), cmdbuf);
    scope_controller.Begin(command.render_state);
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.Handle());

    if (command.is_indirect) {
        const auto& draw = command.indirect;
        if (draw.indexed) {
            ASSERT(sizeof(VkDrawIndexedIndirectCommand) == draw.stride);
            if (draw.count_buffer) {
                cmdbuf.drawIndexedIndirectCount(draw.arguments, draw.argument_offset,
                                                draw.count_buffer, draw.count_offset,
                                                draw.max_count, draw.stride);
            } else {
                cmdbuf.drawIndexedIndirect(draw.arguments, draw.argument_offset,
                                           draw.max_count, draw.stride);
            }
        } else {
            ASSERT(sizeof(VkDrawIndirectCommand) == draw.stride);
            if (draw.count_buffer) {
                cmdbuf.drawIndirectCount(draw.arguments, draw.argument_offset,
                                         draw.count_buffer, draw.count_offset,
                                         draw.max_count, draw.stride);
            } else {
                cmdbuf.drawIndirect(draw.arguments, draw.argument_offset,
                                    draw.max_count, draw.stride);
            }
        }
        return;
    }

    const auto& draw = command.direct;
    if (draw.indexed) {
        cmdbuf.drawIndexed(draw.index_count, draw.instance_count, 0, s32(draw.vertex_offset),
                           draw.instance_offset);
    } else {
        cmdbuf.draw(draw.index_count, draw.instance_count, draw.vertex_offset,
                    draw.instance_offset);
    }
}

void VulkanCommandRecorder::Record(PreparedComputeCommand& command) const {
    ASSERT(command.pipeline != nullptr);
    ASSERT(command.commit.validated);
    const Pipeline& pipeline = *command.pipeline;
    scope_controller.End();
    RecordImageBarriers(command.resources);
    const auto cmdbuf = scheduler.CommandBuffer();
    BindDescriptors(pipeline, command.resources, cmdbuf);
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.Handle());
    if (command.is_indirect) {
        cmdbuf.dispatchIndirect(command.indirect_arguments, command.indirect_offset);
    } else {
        cmdbuf.dispatch(command.dispatch.group_count_x, command.dispatch.group_count_y,
                        command.dispatch.group_count_z);
    }
}

} // namespace Vulkan
