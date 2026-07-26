// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <span>
#include <string_view>

#include "common/types.h"
#include "video_core/gpu_commands/captured_state.h"
#include "video_core/renderer_vulkan/execution/dynamic_state_emitter.h"
#include "video_core/renderer_vulkan/execution/rendering_scope_controller.h"
#include "video_core/renderer_vulkan/preparation/command_preparation_context.h"
#include "video_core/renderer_vulkan/preparation/compute_request_builder.h"
#include "video_core/renderer_vulkan/preparation/input_binding_plan.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

class DescriptorBinder;
class Instance;
class Pipeline;
class Scheduler;

using PipelineArtifactRef = const Pipeline*;

struct ResourceCommitToken {
    u64 sequence{};
    bool validated{};
};

struct DirectDrawPlan {
    bool indexed{};
    u32 index_count{};
    u32 instance_count{};
    u32 vertex_offset{};
    u32 instance_offset{};
};

struct IndirectDrawPlan {
    bool indexed{};
    vk::Buffer arguments{};
    u64 argument_offset{};
    vk::Buffer count_buffer{};
    u64 count_offset{};
    u32 max_count{};
    u32 stride{};
};

struct PreparedGraphicsCommand {
    VideoCore::CommandProvenance provenance{};
    PipelineArtifactRef pipeline{};
    PreparedResourcePayload resources{};
    RenderState render_state{};
    DynamicStatePlan dynamic_state{};
    VertexInputPlan vertex_input{};
    IndexInputPlan index_input{};
    DirectDrawPlan direct{};
    IndirectDrawPlan indirect{};
    ResourceCommitToken commit{};
    bool is_indirect{};

    void Reset() {
        pipeline = nullptr;
        resources.Reset();
        render_state = {};
        dynamic_state = {};
        vertex_input = {};
        index_input = {};
        direct = {};
        indirect = {};
        commit = {};
        is_indirect = false;
    }
};

struct PreparedComputeCommand {
    VideoCore::CommandProvenance provenance{};
    PipelineArtifactRef pipeline{};
    PreparedResourcePayload resources{};
    ComputeDispatchPlan dispatch{};
    vk::Buffer indirect_arguments{};
    u64 indirect_offset{};
    ResourceCommitToken commit{};
    bool is_indirect{};

    void Reset() {
        pipeline = nullptr;
        resources.Reset();
        dispatch = {};
        indirect_arguments = vk::Buffer{};
        indirect_offset = 0;
        commit = {};
        is_indirect = false;
    }
};

/**
 * Sole encoder for prepared draw/dispatch commands.
 *
 * Resource lookup and synchronization happen before this boundary. This class only consumes
 * materialized Vulkan handles, descriptor payloads, state plans, and draw/dispatch arguments.
 */
class VulkanCommandRecorder {
public:
    VulkanCommandRecorder(const Instance& instance_, Scheduler& scheduler_,
                          DescriptorBinder& descriptor_binder_)
        : scheduler{scheduler_}, descriptor_binder{descriptor_binder_},
          scope_controller{scheduler_}, dynamic_emitter{instance_} {}

    void Record(PreparedGraphicsCommand& command) const;
    void Record(PreparedComputeCommand& command) const;
    void RecordImageBarriers(PreparedResourcePayload& resources) const;
    void RecordComputeToIndirectBarrier() const;
    void RecordMemoryBarrier(vk::PipelineStageFlags source_stage,
                             vk::PipelineStageFlags destination_stage,
                             vk::AccessFlags source_access,
                             vk::AccessFlags destination_access) const;
    void RecordBufferCopies(vk::Buffer source, vk::Buffer destination,
                            std::span<const vk::BufferCopy> copies) const;
    void RecordImageCopy(vk::Image source, vk::ImageLayout source_layout,
                         vk::Image destination, vk::ImageLayout destination_layout,
                         const vk::ImageCopy& region) const;
    void BeginDebugLabel(std::string_view label) const;
    void EndDebugLabel() const;
    void InsertDebugLabel(std::string_view label,
                          const std::array<f32, 4>& color = {}) const;

    void EndRendering() const {
        scope_controller.End();
    }

private:
    void BindVertexInput(const VertexInputPlan& plan) const;
    void BindDescriptors(const Pipeline& pipeline,
                         const PreparedResourcePayload& resources,
                         vk::CommandBuffer command_buffer) const;

    Scheduler& scheduler;
    DescriptorBinder& descriptor_binder;
    RenderingScopeController scope_controller;
    DynamicStateEmitter dynamic_emitter;
};

} // namespace Vulkan
