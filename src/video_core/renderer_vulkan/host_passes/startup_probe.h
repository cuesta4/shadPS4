// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class CommandRecorder;
} // namespace Vulkan

namespace Vulkan::HostPasses {

/// Tells whether a frame has visible content with a GPU reduction to a 4 byte flag, so the
/// startup screen can wait for the game's first non-black frame without reading the frame back.
class StartupProbe {
public:
    explicit StartupProbe(const Instance& instance);

    /// The frame must be shader readable in the compute stage. The result is valid once the
    /// recorded commands completed.
    void Record(const CommandRecorder& cmd, vk::ImageView frame, vk::Extent2D size, bool srgb);
    bool Visible() const;

private:
    VmaAllocator allocator;
    vk::UniqueDescriptorSetLayout descriptors;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    VideoCore::UniqueBuffer result;
    const u32* mapped{};
};

} // namespace Vulkan::HostPasses
