// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/alignment.h"
#include "common/div_ceil.h"
#include "common/logging/log.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/tile_manager.h"

#include "video_core/host_shaders/tiling_comp.h"

#include <magic_enum/magic_enum.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_format_traits.hpp>

namespace VideoCore {

struct TilingInfo {
    u32 bank_swizzle;
    u32 num_slices;
    u32 num_mips;
    std::array<ImageInfo::MipInfo, 16> mips;
    u32 image_width;
    u32 image_height;
    u32 range_begin;
    u32 range_end{std::numeric_limits<u32>::max()};
    u32 block_scale;
    u32 is_volume;
};

struct TilingRegion {
    u32 mip;
    u32 first_tile;
    u32 num_tiles;
    u32 tile_width;
    u32 tile_height;
};

static TilingInfo MakeTilingInfo(const ImageInfo& info, u32 range_begin = 0,
                               u32 range_end = std::numeric_limits<u32>::max()) {
    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.num_slices = info.props.is_volume ? info.size.depth : info.resources.layers;
    params.num_mips = info.resources.levels;
    params.mips = info.mips_layout;
    if (info.props.is_block) {
        for (u32 mip = 0; mip < params.num_mips; ++mip) {
            params.mips[mip].pitch = std::max((params.mips[mip].pitch + 3) / 4, 1U);
            params.mips[mip].height = std::max((params.mips[mip].height + 3) / 4, 1U);
        }
    }
    params.image_width = info.size.width;
    params.image_height = info.size.height;
    params.range_begin = range_begin;
    params.range_end = range_end;
    params.block_scale = info.props.is_block ? 4 : 1;
    params.is_volume = info.props.is_volume;
    return params;
}

using TilingRegions = boost::container::small_vector<TilingRegion, 16>;

static TilingRegions MakeTilingRegions(const ImageInfo& info, const TilingInfo& params,
                                     std::span<const vk::BufferImageCopy> copies) {
    TilingRegions regions;
    const u32 bytes_per_pixel = info.num_bits / 8;
    const u32 thickness = AmdGpu::GetMicroTileThickness(info.array_mode);
    for (const auto& copy : copies) {
        const u32 mip = copy.imageSubresource.mipLevel;
        const auto& layout = params.mips[mip];
        if (params.range_begin >= layout.offset + layout.size ||
            params.range_end <= layout.offset) {
            continue;
        }
        const u32 begin = std::max(params.range_begin, layout.offset) - layout.offset;
        const u32 end = std::min(params.range_end, layout.offset + layout.size) - layout.offset;
        if (info.num_samples != 1 || layout.pitch % 8 || layout.height % 8) {
            regions.push_back({mip, 0, Common::DivCeil(layout.size / bytes_per_pixel, 64U), 0, 0});
            continue;
        }
        u32 tile_width = 8;
        u32 tile_height = 8;
        u32 num_pipes = 1;
        u32 num_banks = 1;
        u32 tile_bytes = 64 * thickness * bytes_per_pixel;
        u32 split_bytes = tile_bytes;
        if (AmdGpu::IsMacroTiled(info.array_mode)) {
            const auto mode = AmdGpu::CalculateMacrotileMode(info.tile_mode, info.num_bits, 1);
            num_pipes = AmdGpu::GetPipeCount(AmdGpu::GetPipeConfig(info.tile_mode));
            num_banks = AmdGpu::GetNumBanks(mode);
            const u32 aspect = AmdGpu::GetMacrotileAspect(mode);
            const u32 width = 8 * AmdGpu::GetBankWidth(mode) * num_pipes * aspect;
            const u32 height = 8 * AmdGpu::GetBankHeight(mode) * num_banks / aspect;
            if (layout.pitch >= width && layout.height >= height) {
                tile_width = width;
                tile_height = height;
                if (thickness == 1) {
                    split_bytes = std::min(tile_bytes, AmdGpu::CalculateTileSplit(
                        info.tile_mode, info.array_mode, AmdGpu::GetMicroTileMode(info.tile_mode),
                        info.num_bits));
                }
            } else {
                num_pipes = num_banks = 1;
            }
        }
        const u32 num_tiles = layout.size / (tile_width * tile_height * thickness * bytes_per_pixel);
        const auto append = [&](u32 first, u32 count) {
            if (count != 0) {
                regions.push_back({mip, first, count, tile_width, tile_height});
            }
        };
        if (begin == 0 && end == layout.size) {
            append(0, num_tiles);
            continue;
        }
        if (num_pipes == 1 && num_banks == 1) {
            const u32 first = begin / tile_bytes;
            append(first, Common::DivCeil(end, tile_bytes) - first);
            continue;
        }
        const u32 stripe_bytes = 256 * num_pipes * num_banks;
        const u32 macro_bytes = split_bytes * (tile_width / 8) * (tile_height / 8) /
                                (num_pipes * num_banks);
        const u32 first = (begin / stripe_bytes * 256) / macro_bytes;
        const u32 last = Common::DivCeil(Common::DivCeil(end, stripe_bytes) * 256, macro_bytes);
        const u32 num_splits = tile_bytes / split_bytes;
        if (num_splits == 1) {
            append(first, std::min(last, num_tiles) - first);
            continue;
        }
        const u32 tiles_per_slice = (layout.pitch / tile_width) * (layout.height / tile_height);
        const u32 indices_per_slice = tiles_per_slice * num_splits;
        const u32 first_slice = first / indices_per_slice;
        const u32 last_slice = (std::min(last, num_tiles * num_splits) - 1) / indices_per_slice;
        for (u32 slice = first_slice; slice <= last_slice; ++slice) {
            const u32 a = std::max(first, slice * indices_per_slice) - slice * indices_per_slice;
            const u32 b = std::min(last, (slice + 1) * indices_per_slice) - slice * indices_per_slice;
            const u32 base = slice * tiles_per_slice;
            if (b - a >= tiles_per_slice) {
                append(base, tiles_per_slice);
            } else if (a / tiles_per_slice == (b - 1) / tiles_per_slice) {
                append(base + a % tiles_per_slice, b - a);
            } else {
                append(base, b % tiles_per_slice);
                append(base + a % tiles_per_slice, tiles_per_slice - a % tiles_per_slice);
            }
        }
    }
    return regions;
}

static void DispatchTiling(Vulkan::CommandRecorder cmdbuf, vk::PipelineLayout pl_layout,
                           const ImageInfo& info, const TilingRegions& regions) {
    const u32 thickness = AmdGpu::GetMicroTileThickness(info.array_mode);
    for (auto region : regions) {
        const u32 groups_per_tile = region.tile_width == 0
                                        ? 1
                                        : region.tile_width * region.tile_height * thickness / 64;
        const u32 max_tiles = 65535 / groups_per_tile;
        while (region.num_tiles != 0) {
            const u32 count = std::min(region.num_tiles, max_tiles);
            cmdbuf.pushConstants(pl_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(region), &region);
            cmdbuf.dispatch(count * groups_per_tile, 1, 1);
            region.first_tile += count;
            region.num_tiles -= count;
        }
    }
}

static auto MakeLinearCopies(const ImageInfo& info, const TilingInfo& params,
                             std::span<const vk::BufferImageCopy> copies,
                             const TilingRegions& regions) {
    boost::container::small_vector<vk::BufferImageCopy, 16> result;
    const u32 thickness = AmdGpu::GetMicroTileThickness(info.array_mode);
    const u32 bytes_per_pixel = info.num_bits / 8;
    const u32 block_scale = info.props.is_block ? 4 : 1;
    for (const auto& region : regions) {
        const auto& mip = params.mips[region.mip];
        const auto it = std::ranges::find_if(copies, [&](const auto& copy) {
            return copy.imageSubresource.mipLevel == region.mip;
        });
        if (region.tile_width == 0 ||
            (region.first_tile == 0 && region.num_tiles * region.tile_width *
                                          region.tile_height * thickness * bytes_per_pixel == mip.size)) {
            result.push_back(*it);
            continue;
        }
        const u32 tiles_x = mip.pitch / region.tile_width;
        const u32 tiles_y = mip.height / region.tile_height;
        u32 tile = region.first_tile;
        u32 remaining = region.num_tiles;
        while (remaining != 0) {
            const u32 tile_x = tile % tiles_x;
            const u32 tile_y = (tile / tiles_x) % tiles_y;
            const u32 slice = tile / (tiles_x * tiles_y) * thickness;
            const u32 rows = tile_x == 0 && remaining >= tiles_x
                                 ? std::min(remaining / tiles_x, tiles_y - tile_y)
                                 : 1;
            const u32 columns = rows > 1 ? tiles_x : std::min(remaining, tiles_x - tile_x);
            const u32 x = tile_x * region.tile_width * block_scale;
            const u32 y = tile_y * region.tile_height * block_scale;
            if (x < it->imageExtent.width && y < it->imageExtent.height) {
                const u32 depth = info.props.is_volume ? std::max(info.size.depth >> region.mip, 1U)
                                                      : info.resources.layers;
                for (u32 z = slice; z < std::min(slice + thickness, depth); ++z) {
                    auto copy = *it;
                    copy.bufferOffset = mip.offset +
                        ((z * mip.height + tile_y * region.tile_height) * mip.pitch +
                         tile_x * region.tile_width) * bytes_per_pixel;
                    copy.imageSubresource.baseArrayLayer = info.props.is_volume ? 0 : z;
                    copy.imageSubresource.layerCount = 1;
                    copy.imageOffset = {static_cast<s32>(x), static_cast<s32>(y),
                                        info.props.is_volume ? static_cast<s32>(z) : 0};
                    copy.imageExtent = {
                        std::min(columns * region.tile_width * block_scale, it->imageExtent.width - x),
                        std::min(rows * region.tile_height * block_scale, it->imageExtent.height - y),
                        1,
                    };
                    result.push_back(copy);
                }
            }
            const u32 count = columns * rows;
            tile += count;
            remaining -= count;
        }
    }
    return result;
}

TileManager::TileManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         StreamBuffer& stream_buffer_)
    : instance{instance}, scheduler{scheduler}, stream_buffer{stream_buffer_} {
    const auto device = instance.GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }};

    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto desc_layout_result = device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(desc_layout_result.result == vk::Result::eSuccess,
               "Failed to create descriptor set layout: {}",
               vk::to_string(desc_layout_result.result));
    desc_layout = std::move(desc_layout_result.value);

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PushConstantRange push_range{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .size = sizeof(TilingRegion),
    };
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_range,
    };
    auto [layout_result, layout] = device.createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess, "Failed to create pipeline layout: {}",
               vk::to_string(layout_result));
    pl_layout = std::move(layout);

    // The tiler that reads the image takes it at binding 1.
    auto image_bindings = bindings;
    image_bindings[1].descriptorType = vk::DescriptorType::eSampledImage;
    const vk::DescriptorSetLayoutCreateInfo image_desc_layout_ci = {
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(image_bindings.size()),
        .pBindings = image_bindings.data(),
    };
    auto image_desc_layout_result = device.createDescriptorSetLayoutUnique(image_desc_layout_ci);
    ASSERT_MSG(image_desc_layout_result.result == vk::Result::eSuccess,
               "Failed to create descriptor set layout: {}",
               vk::to_string(image_desc_layout_result.result));
    image_desc_layout = std::move(image_desc_layout_result.value);

    const vk::DescriptorSetLayout image_set_layout = *image_desc_layout;
    const vk::PipelineLayoutCreateInfo image_layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &image_set_layout,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_range,
    };
    auto [image_layout_result, image_layout] =
        device.createPipelineLayoutUnique(image_layout_info);
    ASSERT_MSG(image_layout_result == vk::Result::eSuccess,
               "Failed to create pipeline layout: {}", vk::to_string(image_layout_result));
    image_pl_layout = std::move(image_layout);
}

vk::Format TileManager::TilingViewFormat(const Image& image) noexcept {
    const auto& info = image.info;
    // A uint view reinterprets the texels of a color image of the same texel size; the tiler
    // reads one sample of 2D texels.
    if (info.props.is_depth || info.props.is_block || info.props.is_volume ||
        info.num_samples != 1 || image.aspect_mask != vk::ImageAspectFlagBits::eColor ||
        (info.type != AmdGpu::ImageType::Color2D && info.type != AmdGpu::ImageType::Color2DArray) ||
        u32{vk::blockSize(info.pixel_format)} * 8 != info.num_bits) {
        return vk::Format::eUndefined;
    }
    switch (info.num_bits) {
    case 8:
        return vk::Format::eR8Uint;
    case 16:
        return vk::Format::eR16Uint;
    case 32:
        return vk::Format::eR32Uint;
    case 64:
        return vk::Format::eR32G32Uint;
    case 128:
        return vk::Format::eR32G32B32A32Uint;
    default:
        return vk::Format::eUndefined;
    }
}

TileManager::~TileManager() = default;

TileManager::ScratchBuffer TileManager::GetScratchBuffer(u32 size) {
    constexpr auto usage =
        vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;

    const vk::BufferCreateInfo buffer_ci = {
        .size = size,
        .usage = usage,
    };

    const VmaAllocationCreateInfo alloc_info{
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };

    VkBuffer buffer;
    VmaAllocation allocation;
    const auto buffer_ci_unsafe = static_cast<VkBufferCreateInfo>(buffer_ci);
    const auto result = vmaCreateBuffer(instance.GetAllocator(), &buffer_ci_unsafe, &alloc_info,
                                        &buffer, &allocation, nullptr);
    ASSERT(result == VK_SUCCESS);
    return {buffer, allocation};
}

vk::Pipeline TileManager::GetTilingPipeline(const ImageInfo& info, bool is_tiler,
                                            bool from_image) {
    const u32 pl_id = u32(info.tile_mode) * NUM_BPPS + std::bit_width(info.num_bits) - 4;
    auto& tiling_pipelines = from_image ? image_tilers : is_tiler ? tilers : detilers;
    if (auto pipeline = *tiling_pipelines[pl_id]; pipeline != VK_NULL_HANDLE) {
        return pipeline;
    }

    const auto device = instance.GetDevice();
    const auto micro_tile_mode = AmdGpu::GetMicroTileMode(info.tile_mode);
    std::vector<std::string> defines = {
        fmt::format("BITS_PER_PIXEL={}", info.num_bits),
        fmt::format("NUM_SAMPLES={}", info.num_samples),
        fmt::format("ARRAY_MODE={}", u32(info.array_mode)),
        fmt::format("MICRO_TILE_MODE={}", u32(micro_tile_mode)),
        fmt::format("MICRO_TILE_THICKNESS={}", AmdGpu::GetMicroTileThickness(info.array_mode)),
    };
    if (AmdGpu::IsMacroTiled(info.array_mode)) {
        const auto macro_tile_mode =
            AmdGpu::CalculateMacrotileMode(info.tile_mode, info.num_bits, info.num_samples);
        const u32 num_banks = AmdGpu::GetNumBanks(macro_tile_mode);
        defines.emplace_back(
            fmt::format("PIPE_CONFIG={}", u32(AmdGpu::GetPipeConfig(info.tile_mode))));
        defines.emplace_back(fmt::format("BANK_WIDTH={}", AmdGpu::GetBankWidth(macro_tile_mode)));
        defines.emplace_back(fmt::format("BANK_HEIGHT={}", AmdGpu::GetBankHeight(macro_tile_mode)));
        defines.emplace_back(fmt::format("NUM_BANKS={}", num_banks));
        defines.emplace_back(fmt::format("NUM_BANK_BITS={}", std::bit_width(num_banks) - 1));
        defines.emplace_back(fmt::format(
            "TILE_SPLIT_BYTES={}", AmdGpu::CalculateTileSplit(info.tile_mode, info.array_mode,
                                                              micro_tile_mode, info.num_bits)));
        defines.emplace_back(
            fmt::format("MACRO_TILE_ASPECT={}", AmdGpu::GetMacrotileAspect(macro_tile_mode)));
    }
    if (is_tiler) {
        defines.emplace_back(fmt::format("IS_TILER=1"));
    }
    if (from_image) {
        defines.emplace_back("TILE_FROM_IMAGE=1");
    }

    const auto& module = Vulkan::Compile(HostShaders::TILING_COMP,
                                         vk::ShaderStageFlagBits::eCompute, device, defines,
                                         instance.UsesRawAccessChains());
    const auto module_name =
        fmt::format("{}_{} {}", magic_enum::enum_name(info.tile_mode), info.num_bits,
                    from_image ? "image tiler" : is_tiler ? "tiler" : "detiler");
    LOG_INFO(Render_Vulkan, "Creating pipeline {}", module_name);
    Vulkan::SetObjectName(device, module, module_name);
    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };
    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .stage = shader_ci,
        .layout = from_image ? *image_pl_layout : *pl_layout,
    };
    auto [result, pipeline] =
        device.createComputePipelineUnique(VK_NULL_HANDLE, compute_pipeline_ci);
    ASSERT_MSG(result == vk::Result::eSuccess, "Detiler pipeline creation failed {}",
               vk::to_string(result));
    tiling_pipelines[pl_id] = std::move(pipeline);
    device.destroyShaderModule(module);
    return *tiling_pipelines[pl_id];
}

TileManager::Result TileManager::DetileImage(vk::Buffer in_buffer, u32 in_offset,
                                             const ImageInfo& info,
                                             std::span<const vk::BufferImageCopy> copies,
                                             bool in_host_memory) {
    if (!info.props.is_tiled) {
        return {in_buffer, in_offset};
    }

    const Vulkan::GpuTimingContext timing{scheduler,
                                         {.kind = Vulkan::GpuWork::Detile,
                                          .resource0 = Vulkan::GpuHandle(in_buffer),
                                          .resource1 = info.guest_address}};

    const auto params = MakeTilingInfo(info);
    const auto regions = MakeTilingRegions(info, params, copies);
    u32 buffer_size = 0;
    boost::container::small_vector<vk::BufferCopy, 16> input_copies;
    for (const auto& copy : copies) {
        const auto& mip = info.mips_layout[copy.imageSubresource.mipLevel];
        buffer_size = std::max(buffer_size, mip.offset + mip.size);
        input_copies.push_back({in_offset + mip.offset, mip.offset, mip.size});
    }

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    const bool stage_input = in_host_memory;
    const u32 out_offset =
        stage_input ? static_cast<u32>(Common::AlignUp(u64{buffer_size},
                                                       instance.StorageMinAlignment()))
                    : 0;
    const auto [out_buffer, out_allocation] = GetScratchBuffer(out_offset + buffer_size);
    scheduler.DeferOperation([this, out_buffer, out_allocation]() {
        vmaDestroyBuffer(instance.GetAllocator(), out_buffer, out_allocation);
    });

    scheduler.EndRendering();

    if (stage_input) {
        const auto cmdbuf = scheduler.CommandBuffer();
        cmdbuf.copyBuffer(in_buffer, out_buffer, input_copies);
        const vk::BufferMemoryBarrier2 staged_barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
            .buffer = out_buffer,
            .offset = 0,
            .size = buffer_size,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &staged_barrier,
        });
        in_buffer = out_buffer;
        in_offset = 0;
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, GetTilingPipeline(info, false));

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = in_buffer,
        .offset = in_offset,
        .range = buffer_size,
    };

    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = out_buffer,
        .offset = out_offset,
        .range = buffer_size,
    };

    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tiled_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &linear_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &params_buffer_info,
        },
    }};
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pl_layout, 0, set_writes);

    DispatchTiling(cmdbuf, *pl_layout, info, regions);
    const vk::BufferMemoryBarrier2 output_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = out_buffer,
        .offset = out_offset,
        .size = buffer_size,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &output_barrier,
    });
    return {out_buffer, out_offset};
}

void TileManager::TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                            vk::Buffer out_buffer, u32 out_offset, u32 copy_size,
                            u32 range_begin, u32 range_end) {
    const auto& info = in_image.info;
    if (!info.props.is_tiled) {
        for (auto& copy : buffer_copies) {
            copy.bufferOffset += out_offset;
        }
        in_image.Download(buffer_copies, out_buffer, out_offset, copy_size);
        return;
    }

    const auto params = MakeTilingInfo(info, range_begin, range_end);
    const Vulkan::GpuTimingContext timing{scheduler,
                                         {.kind = Vulkan::GpuWork::Tile,
                                          .resource0 = Vulkan::GpuHandle(out_buffer),
                                          .resource1 = Vulkan::GpuHandle(in_image.GetImage())}};
    const auto regions = MakeTilingRegions(info, params, buffer_copies);
    if (regions.empty()) {
        return;
    }

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    if (const vk::Format view_format = TilingViewFormat(in_image);
        view_format != vk::Format::eUndefined) {
        // Reading the image directly skips the linear copy the tiler would read back, and the
        // scratch buffer that holds it.
        in_image.SetBackingSamples(info.num_samples);
        scheduler.EndRendering();
        const auto device = instance.GetDevice();
        const vk::ImageViewUsageCreateInfo view_usage_ci{
            .usage = vk::ImageUsageFlagBits::eSampled,
        };
        const vk::ImageViewCreateInfo view_ci{
            .pNext = &view_usage_ci,
            .image = in_image.GetImage(),
            .viewType = vk::ImageViewType::e2DArray,
            .format = view_format,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = params.num_mips,
                .baseArrayLayer = 0,
                .layerCount = info.resources.layers,
            },
        };
        auto [view_result, view] = device.createImageView(view_ci);
        ASSERT_MSG(view_result == vk::Result::eSuccess, "Failed to create tiling view: {}",
                   vk::to_string(view_result));
        scheduler.DeferOperation([device, view] { device.destroyImageView(view); });

        const auto cmdbuf = scheduler.CommandBuffer();
        Image::Barriers image_barriers;
        in_image.AppendBarriers(image_barriers, vk::ImageLayout::eShaderReadOnlyOptimal,
                                vk::AccessFlagBits2::eShaderRead,
                                vk::PipelineStageFlagBits2::eComputeShader, {});
        if (!image_barriers.empty()) {
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .imageMemoryBarrierCount = static_cast<u32>(image_barriers.size()),
                .pImageMemoryBarriers = image_barriers.data(),
            });
        }

        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute,
                            GetTilingPipeline(info, true, true));

        const vk::DescriptorBufferInfo tiled_buffer_info{
            .buffer = out_buffer,
            .offset = out_offset,
            .range = info.guest_size,
        };
        const vk::DescriptorImageInfo source_image_info{
            .imageView = view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };
        const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
            {
                .dstSet = VK_NULL_HANDLE,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &tiled_buffer_info,
            },
            {
                .dstSet = VK_NULL_HANDLE,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eSampledImage,
                .pImageInfo = &source_image_info,
            },
            {
                .dstSet = VK_NULL_HANDLE,
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &params_buffer_info,
            },
        }};
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *image_pl_layout, 0,
                                    set_writes);

        DispatchTiling(cmdbuf, *image_pl_layout, info, regions);
        return;
    }

    const auto [temp_buffer, temp_allocation] = GetScratchBuffer(info.guest_size);
    scheduler.DeferOperation([this, temp_buffer, temp_allocation]() {
        vmaDestroyBuffer(instance.GetAllocator(), temp_buffer, temp_allocation);
    });

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto linear_copies = MakeLinearCopies(info, params, buffer_copies, regions);
    if (linear_copies.empty()) {
        return;
    }
    in_image.Download(linear_copies, temp_buffer, 0, info.guest_size, true);

    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, GetTilingPipeline(info, true));

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = out_buffer,
        .offset = out_offset,
        .range = info.guest_size,
    };

    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = temp_buffer,
        .offset = 0,
        .range = info.guest_size,
    };

    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tiled_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &linear_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &params_buffer_info,
        },
    }};
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pl_layout, 0, set_writes);

    DispatchTiling(cmdbuf, *pl_layout, info, regions);
}

} // namespace VideoCore
