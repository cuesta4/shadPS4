// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>

#include "video_core/buffer_cache/buffer_access_state_store.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/preparation/resource_access_plan.h"
#include "video_core/resources/resolved_resource_refs.h"
#include "video_core/texture_cache/image_access_state_store.h"

namespace Vulkan {

struct BufferHazardPlan {
    std::vector<vk::BufferMemoryBarrier2> barriers;
    struct Update {
        VideoCore::Buffer* resource{};
        VideoCore::BufferId id{};
        u64 resource_uid{};
        u32 generation{};
        u32 observed_state_version{};
        vk::PipelineStageFlags2 stage{};
        vk::AccessFlags2 access{};
    };
    std::vector<Update> updates;

    [[nodiscard]] bool Empty() const noexcept {
        return barriers.empty();
    }
};

struct ImageHazardPlan {
    std::vector<vk::ImageMemoryBarrier2> barriers;
    struct Update {
        VideoCore::Image* resource{};
        VideoCore::ImageId id{};
        u32 generation{};
        u32 observed_state_version{};
        VideoCore::ImageAccessState state;
    };
    std::vector<Update> updates;

    [[nodiscard]] bool Empty() const noexcept {
        return barriers.empty();
    }
};

class ResourceHazardPlanner {
public:
    [[nodiscard]] static BufferHazardPlan PlanBufferHazards(
        const VideoCore::BufferAccessStateStore& store,
        const BufferAccessPlan& accesses);

    [[nodiscard]] static BufferHazardPlan PlanBufferHazards(
        const VideoCore::BufferAccessStateStore& store,
        std::span<const VideoCore::ResolvedBufferRef> buffers,
        vk::PipelineStageFlags2 target_stage, vk::AccessFlags2 target_access);

    [[nodiscard]] static ImageHazardPlan PlanImageHazards(
        const VideoCore::ImageAccessStateStore& store,
        const ImageTransitionPlan& transitions);

    [[nodiscard]] static bool CommitBufferHazards(
        VideoCore::BufferAccessStateStore& store,
        const BufferHazardPlan& plan) noexcept;

    [[nodiscard]] static bool CommitImageHazards(
        VideoCore::ImageAccessStateStore& store,
        const ImageHazardPlan& plan) noexcept;
};

} // namespace Vulkan
