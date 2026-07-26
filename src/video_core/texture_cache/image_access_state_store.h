// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/resources/resource_ids.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {

struct ImageSubresourceAccessState {
    vk::ImageLayout layout{vk::ImageLayout::eUndefined};
    vk::PipelineStageFlags2 stage_mask{vk::PipelineStageFlagBits2::eAllCommands};
    vk::AccessFlags2 access_mask{vk::AccessFlagBits2::eNone};
};

struct ImageAccessState {
    vk::ImageLayout layout{vk::ImageLayout::eUndefined};
    vk::PipelineStageFlags2 stage_mask{vk::PipelineStageFlagBits2::eAllCommands};
    vk::AccessFlags2 access_mask{vk::AccessFlagBits2::eNone};
    u32 state_version{1};
    u32 backing_generation{};
    std::vector<ImageSubresourceAccessState> subresources;
};

struct ImageAccessTransition {
    vk::Image image{};
    vk::ImageAspectFlags aspects{};
    SubresourceExtent resources{};
    vk::ImageLayout layout{vk::ImageLayout::eUndefined};
    vk::PipelineStageFlags2 stage{};
    vk::AccessFlags2 access{};
    std::optional<SubresourceRange> subresource;
};

using ImageAccessBarriers = std::vector<vk::ImageMemoryBarrier2>;

void PlanImageAccessTransition(ImageAccessState& state,
                               const ImageAccessTransition& transition,
                               ImageAccessBarriers& barriers);

class ImageAccessStateStore {
public:
    [[nodiscard]] bool Contains(ImageId id) const noexcept {
        return id && id.index < headers.size() && headers[id.index].present;
    }

    [[nodiscard]] ImageAccessState GetAccessState(ImageId id) const noexcept {
        if (Contains(id)) {
            const auto& header = headers[id.index];
            return {
                .layout = header.layout,
                .stage_mask = header.stage_mask,
                .access_mask = header.access_mask,
                .state_version = header.state_version,
                .backing_generation = header.backing_generation,
                .subresources = cold_subresources[id.index],
            };
        }
        return {};
    }

    [[nodiscard]] bool CanUpdateAccessState(ImageId id, u32 expected_state_version,
                                            u32 backing_generation) const noexcept {
        if (!Contains(id)) {
            return expected_state_version == 1;
        }
        const auto& state = headers[id.index];
        return state.state_version == expected_state_version &&
               (state.backing_generation == 0 ||
                state.backing_generation == backing_generation);
    }

    void ResetResource(ImageId id, u32 backing_generation,
                       vk::ImageLayout initial_layout = vk::ImageLayout::eUndefined) noexcept {
        EnsureCapacity(id);
        headers[id.index] = AccessHeader{
            .layout = initial_layout,
            .backing_generation = backing_generation,
            .present = true,
        };
        cold_subresources[id.index].clear();
    }

    void Clear() noexcept {
        headers.clear();
        cold_subresources.clear();
    }

    void RemoveResource(ImageId id) noexcept {
        if (id && id.index < headers.size()) {
            headers[id.index].present = false;
            cold_subresources[id.index].clear();
        }
    }

    void ReplaceAccessState(ImageId id, u32 backing_generation,
                            ImageAccessState state) noexcept {
        state.state_version = Contains(id) ? headers[id.index].state_version + 1 : 2;
        state.backing_generation = backing_generation;
        EnsureCapacity(id);
        headers[id.index] = AccessHeader{
            .stage_mask = state.stage_mask,
            .access_mask = state.access_mask,
            .layout = state.layout,
            .state_version = state.state_version,
            .backing_generation = state.backing_generation,
            .present = true,
        };
        cold_subresources[id.index] = std::move(state.subresources);
    }

private:
    struct AccessHeader {
        vk::PipelineStageFlags2 stage_mask{vk::PipelineStageFlagBits2::eAllCommands};
        vk::AccessFlags2 access_mask{vk::AccessFlagBits2::eNone};
        vk::ImageLayout layout{vk::ImageLayout::eUndefined};
        u32 state_version{1};
        u32 backing_generation{};
        bool present{};
    };

    void EnsureCapacity(ImageId id) {
        if (id.index >= headers.size()) {
            const auto size = static_cast<size_t>(id.index) + 1;
            headers.resize(size);
            cold_subresources.resize(size);
        }
    }

    // The sequential hazard path only touches this compact header array. Per-subresource
    // vectors live in a separate cold array and are fetched only for non-uniform images.
    std::vector<AccessHeader> headers;
    std::vector<std::vector<ImageSubresourceAccessState>> cold_subresources;
};

} // namespace VideoCore
