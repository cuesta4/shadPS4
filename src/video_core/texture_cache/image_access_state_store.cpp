// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/texture_cache/image_access_state_store.h"

#include "common/assert.h"

namespace VideoCore {
namespace {

constexpr vk::AccessFlags2 WriteAccess =
    vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderWrite |
    vk::AccessFlagBits2::eMemoryWrite;

template <typename State>
bool RequiresBarrier(const State& state, const ImageAccessTransition& transition) {
    return state.layout != transition.layout || state.access_mask != transition.access ||
           static_cast<bool>(state.access_mask & WriteAccess);
}

template <typename State>
void SetState(State& state, const ImageAccessTransition& transition) {
    state.layout = transition.layout;
    state.access_mask = transition.access;
    state.stage_mask = transition.stage;
}

template <typename State>
vk::ImageMemoryBarrier2 MakeBarrier(const State& state,
                                    const ImageAccessTransition& transition, u32 mip,
                                    u32 levels, u32 layer, u32 layers) {
    return {
        .srcStageMask = state.stage_mask,
        .srcAccessMask = state.access_mask,
        .dstStageMask = transition.stage,
        .dstAccessMask = transition.access,
        .oldLayout = state.layout,
        .newLayout = transition.layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = transition.image,
        .subresourceRange{
            .aspectMask = transition.aspects,
            .baseMipLevel = mip,
            .levelCount = levels,
            .baseArrayLayer = layer,
            .layerCount = layers,
        },
    };
}

} // Anonymous namespace

void PlanImageAccessTransition(ImageAccessState& state,
                               const ImageAccessTransition& transition,
                               ImageAccessBarriers& barriers) {
    const bool partial =
        transition.subresource &&
        (transition.subresource->base != SubresourceBase{} ||
         transition.subresource->extent != transition.resources);
    const bool was_partial = !state.subresources.empty();

    if (partial || was_partial) {
        if (!was_partial) {
            state.subresources.resize(
                transition.resources.levels * transition.resources.layers,
                ImageSubresourceAccessState{
                    .layout = state.layout,
                    .stage_mask = state.stage_mask,
                    .access_mask = state.access_mask,
                });
        }

        const u32 base_mip = partial ? transition.subresource->base.level : 0;
        const u32 mip_count =
            partial ? transition.subresource->extent.levels : transition.resources.levels;
        const u32 base_layer = partial ? transition.subresource->base.layer : 0;
        const u32 layer_count =
            partial ? transition.subresource->extent.layers : transition.resources.layers;
        for (u32 mip = base_mip; mip < base_mip + mip_count; ++mip) {
            for (u32 layer = base_layer; layer < base_layer + layer_count; ++layer) {
                const u32 index = mip * transition.resources.layers + layer;
                ASSERT(index < state.subresources.size());
                auto& subresource = state.subresources[index];
                if (RequiresBarrier(subresource, transition)) {
                    barriers.push_back(
                        MakeBarrier(subresource, transition, mip, 1, layer, 1));
                    SetState(subresource, transition);
                }
            }
        }
        if (!partial) {
            state.subresources.clear();
        }
    } else if (RequiresBarrier(state, transition)) {
        barriers.push_back(MakeBarrier(state, transition, 0, VK_REMAINING_MIP_LEVELS, 0,
                                       VK_REMAINING_ARRAY_LAYERS));
    }
    SetState(state, transition);
}

} // namespace VideoCore
