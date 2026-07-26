// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_descriptor_binder.h"

#include <array>
#include <boost/container/small_vector.hpp>
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "shader_recompiler/resource.h"

namespace Vulkan {

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

DescriptorBinder::DescriptorBinder(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_},
      descriptor_heap{instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes} {}

void DescriptorBinder::Bind(const Pipeline& pipeline,
                            const Pipeline::DescriptorWrites& write_plan,
                            std::span<const vk::DescriptorBufferInfo> buffer_infos,
                            std::span<const vk::DescriptorImageInfo> image_infos,
                            const Pipeline::BufferBarriers& barriers,
                            const Shader::PushData& push_data,
                            vk::CommandBuffer command_buffer) const {
    const auto bind_point = pipeline.IsCompute() ? vk::PipelineBindPoint::eCompute
                                                 : vk::PipelineBindPoint::eGraphics;

    if (!barriers.empty()) {
        const vk::DependencyInfo dependencies{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = u32(barriers.size()),
            .pBufferMemoryBarriers = barriers.data(),
        };
        scheduler.EndRendering();
        command_buffer.pipelineBarrier2(dependencies);
    }

    const auto stage_flags =
        pipeline.IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    command_buffer.pushConstants(pipeline.GetLayout(), stage_flags, 0u, sizeof(push_data),
                                 &push_data);

    if (write_plan.empty()) {
        return;
    }

    boost::container::small_vector<vk::WriteDescriptorSet, 32> writes;
    writes.reserve(write_plan.size());
    for (const auto& record : write_plan) {
        auto& write = writes.emplace_back();
        write.dstSet = VK_NULL_HANDLE;
        write.dstBinding = record.binding;
        write.dstArrayElement = record.array_element;
        write.descriptorCount = record.descriptor_count;
        write.descriptorType = record.type;
        if (record.info_kind == DescriptorInfoKind::Buffer) {
            ASSERT(record.info_index + record.descriptor_count <= buffer_infos.size());
            write.pBufferInfo = &buffer_infos[record.info_index];
        } else {
            ASSERT(record.info_index + record.descriptor_count <= image_infos.size());
            write.pImageInfo = &image_infos[record.info_index];
        }
    }

    if (pipeline.UsesPushDescriptors()) {
        command_buffer.pushDescriptorSetKHR(bind_point, pipeline.GetLayout(), 0, writes);
        return;
    }

    const auto descriptor_set = descriptor_heap.Commit(pipeline.GetDescriptorSetLayout());
    for (auto& write : writes) {
        write.dstSet = descriptor_set;
    }
    instance.GetDevice().updateDescriptorSets(writes, {});
    command_buffer.bindDescriptorSets(bind_point, pipeline.GetLayout(), 0, descriptor_set, {});
}

} // namespace Vulkan
