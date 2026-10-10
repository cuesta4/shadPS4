// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include "common/assert.h"
#include "video_core/host_shaders/startup_probe_comp.h"
#include "video_core/renderer_vulkan/host_passes/startup_probe.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include <vk_mem_alloc.h>

namespace Vulkan::HostPasses {

StartupProbe::StartupProbe(const Instance& instance)
    : allocator{instance.GetAllocator()}, result{instance.GetDevice(), allocator} {
    const auto device = instance.GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
        {0, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    }};
    descriptors = Check<"create startup probe descriptors">(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange range{vk::ShaderStageFlagBits::eCompute, 0, sizeof(float)};
    layout = Check<"create startup probe layout">(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &descriptors.get(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    const auto module =
        Compile(HostShaders::STARTUP_PROBE_COMP, vk::ShaderStageFlagBits::eCompute, device);
    ASSERT(module);
    pipeline = Check<"create startup probe pipeline">(device.createComputePipelineUnique({}, {
        .stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
        .layout = layout.get(),
    }));
    device.destroyShaderModule(module);

    VmaAllocationInfo info{};
    result.Create({.size = sizeof(u32),
                   .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                            vk::BufferUsageFlagBits::eTransferDst},
                  VideoCore::MemoryUsage::Download, &info);
    mapped = static_cast<const u32*>(info.pMappedData);
}

void StartupProbe::Record(const CommandRecorder& cmd, vk::ImageView frame, vk::Extent2D size,
                          bool srgb) {
    cmd.fillBuffer(result, 0, sizeof(u32), 0);
    const vk::MemoryBarrier cleared{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                                    .dstAccessMask = vk::AccessFlagBits::eShaderWrite};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader, {}, cleared, {}, {});

    const vk::DescriptorImageInfo image{{}, frame, vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorBufferInfo buffer{result, 0, sizeof(u32)};
    const std::array<vk::WriteDescriptorSet, 2> writes{{
        {.dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &image},
        {.dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &buffer},
    }};
    // 4/255 of the encoded value; an sRGB view hands the shader linear values.
    const float threshold = srgb ? 4.0f / 255.0f / 12.92f : 4.0f / 255.0f;
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.get());
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout.get(), 0, writes);
    cmd.pushConstants(layout.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(threshold),
                      &threshold);
    cmd.dispatch((size.width + 15) / 16, (size.height + 15) / 16, 1);

    const vk::MemoryBarrier written{.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                                    .dstAccessMask = vk::AccessFlagBits::eHostRead};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eHost, {}, written, {}, {});
}

bool StartupProbe::Visible() const {
    vmaInvalidateAllocation(allocator, result.allocation, 0, sizeof(u32));
    return *mapped != 0;
}

} // namespace Vulkan::HostPasses
