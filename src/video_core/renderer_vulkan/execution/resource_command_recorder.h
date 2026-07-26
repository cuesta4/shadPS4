// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

/**
 * Serial recording boundary for cache/content-synchronization operations.
 *
 * Resource subsystems choose the operation and payload, but only this execution service obtains
 * and writes the current primary command buffer.
 */
class ResourceCommandRecorder {
public:
    static void PipelineBarrier2(Scheduler& scheduler, const vk::DependencyInfo& dependency) {
        scheduler.CommandBuffer().pipelineBarrier2(dependency);
    }

    static void PipelineBarrier2(vk::CommandBuffer command_buffer,
                                 const vk::DependencyInfo& dependency) {
        command_buffer.pipelineBarrier2(dependency);
    }

    template <typename Regions>
    static void CopyBuffer(Scheduler& scheduler, vk::Buffer source, vk::Buffer destination,
                           Regions&& regions) {
        scheduler.CommandBuffer().copyBuffer(source, destination,
                                             std::forward<Regions>(regions));
    }

    static void FillBuffer(Scheduler& scheduler, vk::Buffer buffer, vk::DeviceSize offset,
                           vk::DeviceSize size, u32 value) {
        scheduler.CommandBuffer().fillBuffer(buffer, offset, size, value);
    }

    template <typename Regions>
    static void CopyBufferToImage(Scheduler& scheduler, vk::Buffer source,
                                  vk::Image destination, vk::ImageLayout layout,
                                  Regions&& regions) {
        scheduler.CommandBuffer().copyBufferToImage(
            source, destination, layout, std::forward<Regions>(regions));
    }

    template <typename Regions>
    static void CopyImageToBuffer(Scheduler& scheduler, vk::Image source,
                                  vk::ImageLayout layout, vk::Buffer destination,
                                  Regions&& regions) {
        scheduler.CommandBuffer().copyImageToBuffer(
            source, layout, destination, std::forward<Regions>(regions));
    }

    template <typename Regions>
    static void CopyImage(Scheduler& scheduler, vk::Image source,
                          vk::ImageLayout source_layout, vk::Image destination,
                          vk::ImageLayout destination_layout, Regions&& regions) {
        scheduler.CommandBuffer().copyImage(
            source, source_layout, destination, destination_layout,
            std::forward<Regions>(regions));
    }

    template <typename Regions>
    static void ResolveImage(Scheduler& scheduler, vk::Image source,
                             vk::ImageLayout source_layout, vk::Image destination,
                             vk::ImageLayout destination_layout, Regions&& regions) {
        scheduler.CommandBuffer().resolveImage(
            source, source_layout, destination, destination_layout,
            std::forward<Regions>(regions));
    }

    template <typename Ranges>
    static void ClearColorImage(Scheduler& scheduler, vk::Image image,
                                vk::ImageLayout layout, const vk::ClearColorValue& value,
                                Ranges&& ranges) {
        scheduler.CommandBuffer().clearColorImage(
            image, layout, value, std::forward<Ranges>(ranges));
    }

    static void BindPipeline(Scheduler& scheduler, vk::PipelineBindPoint bind_point,
                             vk::Pipeline pipeline) {
        scheduler.CommandBuffer().bindPipeline(bind_point, pipeline);
    }

    template <typename Writes>
    static void PushDescriptorSet(Scheduler& scheduler, vk::PipelineBindPoint bind_point,
                                  vk::PipelineLayout layout, u32 set, Writes&& writes) {
        scheduler.CommandBuffer().pushDescriptorSetKHR(
            bind_point, layout, set, std::forward<Writes>(writes));
    }

    static void Dispatch(Scheduler& scheduler, u32 x, u32 y, u32 z) {
        scheduler.CommandBuffer().dispatch(x, y, z);
    }

    template <typename Viewports>
    static void SetViewports(Scheduler& scheduler, Viewports&& viewports) {
        scheduler.CommandBuffer().setViewportWithCount(
            std::forward<Viewports>(viewports));
    }

    template <typename Scissors>
    static void SetScissors(Scheduler& scheduler, Scissors&& scissors) {
        scheduler.CommandBuffer().setScissorWithCount(std::forward<Scissors>(scissors));
    }

    static void Draw(Scheduler& scheduler, u32 vertex_count, u32 instance_count,
                     u32 first_vertex, u32 first_instance) {
        scheduler.CommandBuffer().draw(vertex_count, instance_count, first_vertex,
                                       first_instance);
    }
};

} // namespace Vulkan
