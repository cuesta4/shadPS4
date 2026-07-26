// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/resources/resource_ids.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {
class Buffer;
struct Image;
}

namespace Vulkan {

struct PlannedBufferAccess {
    VideoCore::Buffer* resource{};
    VideoCore::BufferId id{};
    u32 generation{};
    u64 resource_uid{};
    vk::AccessFlags2 access{};
    vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};
    u32 offset{};
};

using BufferAccessPlan = std::vector<PlannedBufferAccess>;

struct PlannedImageTransition {
    VideoCore::Image* resource{};
    VideoCore::ImageId id{};
    u32 generation{};
    u64 resource_uid{};
    vk::ImageLayout layout{vk::ImageLayout::eUndefined};
    vk::AccessFlags2 access{};
    vk::PipelineStageFlags2 stage{};
    std::optional<VideoCore::SubresourceRange> subresource;
};

using ImageTransitionPlan = std::vector<PlannedImageTransition>;
using ImageBarrierPlan = std::vector<vk::ImageMemoryBarrier2>;

} // namespace Vulkan
