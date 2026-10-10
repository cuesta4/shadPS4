// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <span>
#include <vector>
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {
class Instance;
class Scheduler;
class CommandRecorder;
}

namespace Vulkan::HostPasses {

class PostFxPass {
public:
    struct Settings {
        int upscaler{};
        int anti_aliasing{};
        int sharpening{};
        int attenuation{250};
    };
    struct Output {
        vk::ImageView view;
        vk::Extent2D size;
    };

    void Create(const Instance& instance);
    Output Render(Scheduler& scheduler, vk::ImageView input, vk::Extent2D input_size,
                  vk::Extent2D output_size, Settings settings, bool input_linear);

private:
    enum class Program : u32 {
        Copy, Easu, Rcas, Psmaa0, Psmaa1, Psmaa2, Psmaa3, Psmaa4, Psmaa5, Psmaa6,
        Edges, DispatchArgs, Candidates, Apply, Count
    };
    struct Surface {
        VideoCore::UniqueImage image;
        vk::UniqueImageView view;
        vk::Extent2D size{};
        vk::Format format{};
        vk::ImageLayout layout{vk::ImageLayout::eUndefined};
    };
    struct Kernel {
        vk::UniqueDescriptorSetLayout descriptors;
        vk::UniquePipelineLayout layout;
        vk::UniquePipeline pipeline;
    };
    struct Constants {
        std::array<float, 4> metrics;
        std::array<float, 4> target;
        u32 conversion{};
    };

    Kernel& GetKernel(Program program);
    std::unique_ptr<Surface> CreateSurface(vk::Extent2D size, vk::Format format,
                                         bool lookup = false) const;
    Surface& GetSurface(Scheduler& scheduler, vk::Extent2D size,
                        vk::Format format = vk::Format::eR16G16B16A16Sfloat);
    void Transition(const CommandRecorder& cmd, Surface& image, vk::ImageLayout layout) const;
    void Dispatch(const CommandRecorder& cmd, Program program, std::span<const vk::ImageView> inputs,
                  std::span<Surface* const> outputs, vk::Extent2D input_size,
                  const void* constants = nullptr, u32 constants_size = 0, u32 flags = 0);
    Surface& Copy(Scheduler& scheduler, vk::ImageView input, vk::Extent2D size, u32 conversion = 0);
    Surface& Spatial(Scheduler& scheduler, Program program, vk::ImageView input,
                     vk::Extent2D size, vk::Extent2D output_size, int attenuation = 0);
    Surface& Psmaa(Scheduler& scheduler, vk::ImageView input, vk::Extent2D size);
    Surface& Cmaa(Scheduler& scheduler, vk::ImageView input, vk::Extent2D size);
    void UploadLookup(Scheduler& scheduler);

    vk::Device device{};
    VmaAllocator allocator{};
    vk::UniqueSampler point_sampler;
    vk::UniqueSampler linear_sampler;
    bool compact_images{};
    bool packed_edges{};
    bool use_raw_access_chains{};
    std::array<Kernel, static_cast<u32>(Program::Count)> kernels;
    std::vector<std::unique_ptr<Surface>> images;
    u32 next_image{};
    std::array<std::unique_ptr<Surface>, 2> lookup;
    std::array<std::unique_ptr<VideoCore::UniqueBuffer>, 5> cmaa_buffers;
    vk::Extent2D cmaa_size{};
};

} // namespace Vulkan::HostPasses
