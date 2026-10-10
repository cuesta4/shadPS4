// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include "common/assert.h"
#include "core/debug_state.h"
#include "video_core/host_shaders/cmaa2_0_comp.h"
#include "video_core/host_shaders/cmaa2_1_comp.h"
#include "video_core/host_shaders/cmaa2_2_comp.h"
#include "video_core/host_shaders/cmaa2_3_comp.h"
#include "video_core/host_shaders/fsr_comp.h"
#include "video_core/host_shaders/postfx_copy_comp.h"
#include "video_core/host_shaders/psmaa0_comp.h"
#include "video_core/host_shaders/psmaa1_comp.h"
#include "video_core/host_shaders/psmaa2_comp.h"
#include "video_core/host_shaders/psmaa3_comp.h"
#include "video_core/host_shaders/psmaa4_comp.h"
#include "video_core/host_shaders/psmaa5_comp.h"
#include "video_core/host_shaders/psmaa6_comp.h"
#include "video_core/host_shaders/postfx/areatex.h"
#include "video_core/host_shaders/postfx/searchtex.h"
#include "video_core/renderer_vulkan/host_passes/postfx_pass.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include <vk_mem_alloc.h>

#define A_CPU
#include "video_core/host_shaders/fsr/ffx_a.h"
#include "video_core/host_shaders/fsr/ffx_fsr1.h"

namespace Vulkan::HostPasses {

namespace {
constexpr vk::ImageSubresourceRange ColorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
constexpr std::array Sources{
    HostShaders::POSTFX_COPY_COMP, HostShaders::FSR_COMP, HostShaders::FSR_COMP,
    HostShaders::PSMAA0_COMP, HostShaders::PSMAA1_COMP,
    HostShaders::PSMAA2_COMP, HostShaders::PSMAA3_COMP, HostShaders::PSMAA4_COMP,
    HostShaders::PSMAA5_COMP, HostShaders::PSMAA6_COMP,
    HostShaders::CMAA2_0_COMP,
    HostShaders::CMAA2_1_COMP, HostShaders::CMAA2_2_COMP, HostShaders::CMAA2_3_COMP,
};

std::array<float, 4> Metrics(vk::Extent2D size) {
    return {1.f / size.width, 1.f / size.height, static_cast<float>(size.width),
            static_cast<float>(size.height)};
}

void ComputeBarrier(const CommandRecorder& cmd) {
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
                         vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eTransferWrite,
    };
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}
} // namespace

void PostFxPass::Create(const Instance& instance) {
    device = instance.GetDevice();
    use_raw_access_chains = instance.UsesRawAccessChains();
    allocator = instance.GetAllocator();
    const bool extended = instance.GetPhysicalDevice().getFeatures().shaderStorageImageExtendedFormats;
    const auto filtered = vk::FormatFeatureFlagBits2::eSampledImage |
                          vk::FormatFeatureFlagBits2::eSampledImageFilterLinear |
                          vk::FormatFeatureFlagBits2::eStorageImage;
    compact_images = extended && instance.IsFormatSupported(vk::Format::eR16Sfloat, filtered) &&
                     instance.IsFormatSupported(vk::Format::eR16G16Sfloat, filtered);
    packed_edges = extended && instance.IsFormatSupported(vk::Format::eR8Uint,
                                                          vk::FormatFeatureFlagBits2::eStorageImage);
    vk::SamplerCreateInfo info{
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    };
    point_sampler = Check<"create postfx point sampler">(device.createSamplerUnique(info));
    info.magFilter = info.minFilter = vk::Filter::eLinear;
    linear_sampler = Check<"create postfx linear sampler">(device.createSamplerUnique(info));
}

PostFxPass::Kernel& PostFxPass::GetKernel(Program program) {
    static_assert(Sources.size() == static_cast<u32>(Program::Count));
    auto& kernel = kernels[static_cast<u32>(program)];
    if (kernel.pipeline) {
        return kernel;
    }
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    const bool cmaa = program >= Program::Edges;
    const u32 count = cmaa ? 8 : 13;
    for (u32 i = 0; i < count; ++i) {
        vk::DescriptorType type;
        if (cmaa) {
            type = i == 0 ? (program == Program::Apply ? vk::DescriptorType::eStorageImage
                                                      : vk::DescriptorType::eSampledImage)
                          : (i == 1 || i == 5 ? vk::DescriptorType::eStorageImage
                                              : vk::DescriptorType::eStorageBuffer);
        } else {
            type = i < 8 ? vk::DescriptorType::eSampledImage
                         : (i < 10 ? vk::DescriptorType::eSampler : vk::DescriptorType::eStorageImage);
        }
        bindings.push_back({i, type, 1, vk::ShaderStageFlagBits::eCompute});
    }
    kernel.descriptors = Check<"create postfx descriptors">(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange range{vk::ShaderStageFlagBits::eCompute, 0, 80};
    kernel.layout = Check<"create postfx layout">(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &kernel.descriptors.get(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    std::vector<std::string> defines;
    defines.push_back(compact_images ? "PSMAA_LUMA_FORMAT=r16f" : "PSMAA_LUMA_FORMAT=rgba16f");
    defines.push_back(compact_images ? "PSMAA_VECTOR_FORMAT=rg16f" : "PSMAA_VECTOR_FORMAT=rgba16f");
    defines.push_back(packed_edges ? "CMAA_EDGE_FORMAT=r8ui" : "CMAA_EDGE_FORMAT=r32ui");
    if (program == Program::Easu || program == Program::Rcas) {
        defines.push_back(program == Program::Easu ? "SAMPLE_EASU=1" : "SAMPLE_RCAS=1");
    }
    const auto module = Compile(Sources[static_cast<u32>(program)], vk::ShaderStageFlagBits::eCompute,
                                device, defines, use_raw_access_chains);
    ASSERT(module);
    kernel.pipeline = Check<"create postfx pipeline">(device.createComputePipelineUnique({}, {
        .stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
        .layout = kernel.layout.get(),
    }));
    device.destroyShaderModule(module);
    return kernel;
}

std::unique_ptr<PostFxPass::Surface> PostFxPass::CreateSurface(vk::Extent2D size, vk::Format format,
                                                            bool lookup) const {
    auto surface = std::make_unique<Surface>();
    surface->size = size;
    surface->format = format;
    surface->image = VideoCore::UniqueImage(device, allocator, nullptr, true);
    surface->image.Create({
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent{size.width, size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                 (lookup ? vk::ImageUsageFlags{} : vk::ImageUsageFlagBits::eStorage),
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    surface->view = Check<"create postfx view">(device.createImageViewUnique({
        .image = surface->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = ColorRange,
    }));
    return surface;
}

PostFxPass::Surface& PostFxPass::GetSurface(Scheduler& scheduler, vk::Extent2D size, vk::Format format) {
    if (next_image == images.size()) {
        images.emplace_back();
    }
    auto& image = images[next_image++];
    if (!image || image->size != size || image->format != format) {
        if (image) {
            scheduler.DeferOperation([old = std::move(image)] {});
        }
        image = CreateSurface(size, format);
    }
    return *image;
}

void PostFxPass::Transition(const CommandRecorder& cmd, Surface& image, vk::ImageLayout layout) const {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = image.layout == vk::ImageLayout::eUndefined
                            ? vk::PipelineStageFlagBits2::eNone : vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = image.layout == vk::ImageLayout::eUndefined
                             ? vk::AccessFlags2{} : vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .oldLayout = image.layout,
        .newLayout = layout,
        .image = image.image,
        .subresourceRange = ColorRange,
    };
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    image.layout = layout;
}

void PostFxPass::Dispatch(const CommandRecorder& cmd, Program program,
                          std::span<const vk::ImageView> inputs, std::span<Surface* const> outputs,
                          vk::Extent2D input_size, const void* constants, u32 constants_size, u32 flags) {
    auto& kernel = GetKernel(program);
    std::array<vk::DescriptorImageInfo, 13> info{};
    std::array<vk::WriteDescriptorSet, 13> writes{};
    for (u32 i = 0; i < 13; ++i) {
        if (i < 8) {
            info[i].imageView = i < inputs.size() ? inputs[i] : inputs[0];
            info[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        } else if (i < 10) {
            info[i].sampler = i == 8 ? point_sampler.get() : linear_sampler.get();
        } else {
            auto& output = *outputs[std::min<size_t>(i - 10, outputs.size() - 1)];
            info[i].imageView = output.view.get();
            info[i].imageLayout = vk::ImageLayout::eGeneral;
        }
        writes[i] = {.dstBinding = i, .descriptorCount = 1,
                     .descriptorType = i < 8 ? vk::DescriptorType::eSampledImage
                                            : (i < 10 ? vk::DescriptorType::eSampler : vk::DescriptorType::eStorageImage),
                     .pImageInfo = &info[i]};
    }
    for (auto* output : outputs) {
        Transition(cmd, *output, vk::ImageLayout::eGeneral);
    }
    const auto size = outputs[0]->size;
    const Constants params{Metrics(input_size), Metrics(size), flags};
    if (!constants) {
        constants = &params;
        constants_size = sizeof(params);
    }
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, kernel.pipeline.get());
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, kernel.layout.get(), 0, writes);
    cmd.pushConstants(kernel.layout.get(), vk::ShaderStageFlagBits::eCompute, 0,
                      constants_size, constants);
    const u32 group = program == Program::Easu || program == Program::Rcas ? 16 : 8;
    cmd.dispatch((size.width + group - 1) / group, (size.height + group - 1) / group, 1);
    for (auto* output : outputs) {
        Transition(cmd, *output, vk::ImageLayout::eShaderReadOnlyOptimal);
    }
}

PostFxPass::Surface& PostFxPass::Copy(Scheduler& scheduler, vk::ImageView input,
                                    vk::Extent2D size, u32 conversion) {
    auto& output = GetSurface(scheduler, size);
    Dispatch(scheduler.CommandBuffer(), Program::Copy, std::array{input}, std::array{&output},
             size, nullptr, 0, conversion);
    return output;
}

PostFxPass::Surface& PostFxPass::Spatial(Scheduler& scheduler, Program program, vk::ImageView input,
                                       vk::Extent2D size, vk::Extent2D output_size, int attenuation) {
    auto& output = GetSurface(scheduler, output_size);
    std::array<std::array<u32, 4>, 5> constants{};
    if (program == Program::Easu) {
        FsrEasuCon(constants[0].data(), constants[1].data(), constants[2].data(), constants[3].data(),
                   size.width, size.height, size.width, size.height, output_size.width, output_size.height);
    } else if (program == Program::Rcas) {
        FsrRcasCon(constants[0].data(), static_cast<float>(attenuation) / 1000.f);
    }
    Dispatch(scheduler.CommandBuffer(), program, std::array{input}, std::array{&output}, size,
             program == Program::Copy ? nullptr : constants.data(),
             program == Program::Copy ? 0 : sizeof(constants));
    return output;
}

void PostFxPass::UploadLookup(Scheduler& scheduler) {
    if (lookup[0]) {
        return;
    }
    const std::array sizes{vk::Extent2D{AREATEX_WIDTH, AREATEX_HEIGHT},
                           vk::Extent2D{SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT}};
    const std::array formats{vk::Format::eR8G8Unorm, vk::Format::eR8Unorm};
    const std::array<std::span<const unsigned char>, 2> data{areaTexBytes, searchTexBytes};
    for (u32 i = 0; i < 2; ++i) {
        lookup[i] = CreateSurface(sizes[i], formats[i], true);
        auto upload = std::make_unique<VideoCore::UniqueBuffer>(device, allocator);
        VmaAllocationInfo allocation{};
        upload->Create({.size = data[i].size(), .usage = vk::BufferUsageFlagBits::eTransferSrc},
                       VideoCore::MemoryUsage::Upload, &allocation);
        std::memcpy(allocation.pMappedData, data[i].data(), data[i].size());
        Check<"flush postfx lookup upload">(
            vk::Result{vmaFlushAllocation(allocator, upload->allocation, 0, data[i].size())});
        const auto cmd = scheduler.CommandBuffer();
        Transition(cmd, *lookup[i], vk::ImageLayout::eTransferDstOptimal);
        const vk::BufferImageCopy region{
            .imageSubresource{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .imageExtent{sizes[i].width, sizes[i].height, 1},
        };
        cmd.copyBufferToImage(upload->buffer, lookup[i]->image, vk::ImageLayout::eTransferDstOptimal, region);
        Transition(cmd, *lookup[i], vk::ImageLayout::eShaderReadOnlyOptimal);
        scheduler.DeferOperation([upload = std::move(upload)] {});
    }
}

PostFxPass::Surface& PostFxPass::Psmaa(Scheduler& scheduler, vk::ImageView input, vk::Extent2D size) {
    UploadLookup(scheduler);
    const auto luma_format = compact_images ? vk::Format::eR16Sfloat : vk::Format::eR16G16B16A16Sfloat;
    const auto vector_format = compact_images ? vk::Format::eR16G16Sfloat : vk::Format::eR16G16B16A16Sfloat;
    auto& luma = GetSurface(scheduler, size, luma_format);
    auto& strength = GetSurface(scheduler, size, vector_format);
    const auto cmd = scheduler.CommandBuffer();
    Dispatch(cmd, Program::Psmaa0, std::array{input}, std::array{&luma, &strength}, size);
    auto& filtered = GetSurface(scheduler, size);
    Dispatch(cmd, Program::Psmaa1, std::array{input, strength.view.get()}, std::array{&filtered}, size);
    auto& deltas = GetSurface(scheduler, size, vector_format);
    Dispatch(cmd, Program::Psmaa2, std::array{filtered.view.get()}, std::array{&deltas}, size);
    auto& edges = GetSurface(scheduler, size, vector_format);
    Dispatch(cmd, Program::Psmaa3, std::array{filtered.view.get(), deltas.view.get(), luma.view.get()},
             std::array{&edges}, size);
    auto& weights = GetSurface(scheduler, size, vk::Format::eR8G8B8A8Unorm);
    const std::array weight_inputs{filtered.view.get(), edges.view.get(), filtered.view.get(),
                                   filtered.view.get(), filtered.view.get(), filtered.view.get(),
                                   lookup[0]->view.get(), lookup[1]->view.get()};
    Dispatch(cmd, Program::Psmaa4, weight_inputs, std::array{&weights}, size);
    auto& blended = GetSurface(scheduler, size);
    Dispatch(cmd, Program::Psmaa5, std::array{filtered.view.get(), weights.view.get(), strength.view.get()},
             std::array{&blended}, size);
    auto& smoothed = GetSurface(scheduler, size);
    Dispatch(cmd, Program::Psmaa6, std::array{blended.view.get(), deltas.view.get(), weights.view.get(), luma.view.get()},
             std::array{&smoothed}, size);
    return smoothed;
}

PostFxPass::Surface& PostFxPass::Cmaa(Scheduler& scheduler, vk::ImageView input, vk::Extent2D size) {
    const u64 pixels = static_cast<u64>(size.width) * size.height;
    if (cmaa_size != size) {
        const std::array<u64, 5> sizes{std::max<u64>(4, pixels), std::max<u64>(4, (pixels + 3) / 6 * 4),
                                      std::max<u64>(8, pixels / 2 * 8), 64, 16};
        for (u32 i = 0; i < cmaa_buffers.size(); ++i) {
            if (cmaa_buffers[i]) {
                scheduler.DeferOperation([old = std::move(cmaa_buffers[i])] {});
            }
            cmaa_buffers[i] = std::make_unique<VideoCore::UniqueBuffer>(device, allocator);
            cmaa_buffers[i]->Create({.size = sizes[i], .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer},
                VideoCore::MemoryUsage::DeviceLocal, nullptr);
        }
        cmaa_size = size;
    }
    auto& output = Copy(scheduler, input, size, 2);
    auto& edges = GetSurface(scheduler, {(size.width + 1) / 2, size.height},
                            packed_edges ? vk::Format::eR8Uint : vk::Format::eR32Uint);
    auto& heads = GetSurface(scheduler, {(size.width + 1) / 2, (size.height + 1) / 2}, vk::Format::eR32Uint);
    const auto cmd = scheduler.CommandBuffer();
    ComputeBarrier(cmd);
    cmd.fillBuffer(cmaa_buffers[3]->buffer, 0, 64, 0);
    Transition(cmd, edges, vk::ImageLayout::eGeneral);
    Transition(cmd, heads, vk::ImageLayout::eGeneral);
    ComputeBarrier(cmd);
    const auto dispatch = [&](Program program, u32 x, u32 y, bool indirect = false) {
        auto& kernel = GetKernel(program);
        std::array<vk::DescriptorImageInfo, 8> image_info{};
        image_info[0] = {.imageView = output.view.get(), .imageLayout = program == Program::Apply
                         ? vk::ImageLayout::eGeneral : vk::ImageLayout::eShaderReadOnlyOptimal};
        image_info[1] = {.imageView = edges.view.get(), .imageLayout = vk::ImageLayout::eGeneral};
        image_info[5] = {.imageView = heads.view.get(), .imageLayout = vk::ImageLayout::eGeneral};
        std::array<vk::DescriptorBufferInfo, 8> buffer_info{};
        std::array<vk::WriteDescriptorSet, 8> writes{};
        u32 buffer = 0;
        for (u32 i = 0; i < writes.size(); ++i) {
            const bool image = i == 0 || i == 1 || i == 5;
            if (!image) {
                buffer_info[i] = {cmaa_buffers[buffer++]->buffer, 0, VK_WHOLE_SIZE};
            }
            writes[i] = {.dstBinding = i, .descriptorCount = 1,
                         .descriptorType = image ? (i == 0 && program != Program::Apply
                             ? vk::DescriptorType::eSampledImage : vk::DescriptorType::eStorageImage)
                             : vk::DescriptorType::eStorageBuffer,
                         .pImageInfo = image ? &image_info[i] : nullptr,
                         .pBufferInfo = image ? nullptr : &buffer_info[i]};
        }
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, kernel.pipeline.get());
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, kernel.layout.get(), 0, writes);
        if (indirect) {
            cmd.dispatchIndirect(cmaa_buffers[4]->buffer, 0);
        } else {
            cmd.dispatch(x, y, 1);
        }
        ComputeBarrier(cmd);
    };
    dispatch(Program::Edges, (size.width + 27) / 28, (size.height + 27) / 28);
    dispatch(Program::DispatchArgs, 2, 1);
    dispatch(Program::Candidates, 0, 0, true);
    dispatch(Program::DispatchArgs, 1, 2);
    Transition(cmd, output, vk::ImageLayout::eGeneral);
    dispatch(Program::Apply, 0, 0, true);
    Transition(cmd, output, vk::ImageLayout::eShaderReadOnlyOptimal);
    return Copy(scheduler, output.view.get(), size, 1);
}

PostFxPass::Output PostFxPass::Render(Scheduler& scheduler, vk::ImageView input,
                                    vk::Extent2D input_size, vk::Extent2D output_size,
                                    Settings settings, bool input_linear) {
    const GpuTimingContext timing{scheduler, {.kind = GpuWork::PostFx, .resource0 = GpuHandle(input)}};
    next_image = 0;
    const auto trim = [&] {
        while (images.size() > next_image) {
            scheduler.DeferOperation([old = std::move(images.back())] {});
            images.pop_back();
        }
        if (settings.anti_aliasing != 4 && cmaa_buffers[0]) {
            for (auto& buffer : cmaa_buffers) {
                scheduler.DeferOperation([old = std::move(buffer)] {});
            }
            cmaa_size = {};
        }
    };
    const bool needs_scaling = settings.upscaler == 1 && input_size != output_size;
    DebugState.is_using_fsr = settings.anti_aliasing == 1 || (settings.upscaler == 1 && needs_scaling);
    if (!needs_scaling && settings.anti_aliasing == 0 && settings.sharpening == 0) {
        trim();
        return {input, input_size};
    }
    auto view = input;
    auto size = input_size;
    if (input_linear) {
        view = Copy(scheduler, view, size, 1).view.get();
    }
    if (settings.anti_aliasing == 1) {
        view = Spatial(scheduler, Program::Easu, view, size, size).view.get();
    } else if (settings.anti_aliasing == 3) {
        view = Psmaa(scheduler, view, size).view.get();
    } else if (settings.anti_aliasing == 4) {
        view = Cmaa(scheduler, view, size).view.get();
    }
    if (needs_scaling) {
        view = Spatial(scheduler, Program::Easu, view, size, output_size).view.get();
        size = output_size;
    } else if (settings.sharpening == 1 && size != output_size) {
        view = Spatial(scheduler, Program::Copy, view, size, output_size).view.get();
        size = output_size;
    }
    if (settings.sharpening == 1) {
        view = Spatial(scheduler, Program::Rcas, view, size, size, settings.attenuation).view.get();
    }
    if (input_linear) {
        view = Copy(scheduler, view, size, 2).view.get();
    }
    trim();
    return {view, size};
}

} // namespace Vulkan::HostPasses
