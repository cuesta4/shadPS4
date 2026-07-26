// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/execution/resource_hazard_planner.h"

#include <algorithm>
#include <ranges>
#include <utility>

#include "common/assert.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {

BufferHazardPlan ResourceHazardPlanner::PlanBufferHazards(
    const VideoCore::BufferAccessStateStore& store,
    const BufferAccessPlan& accesses) {
    BufferHazardPlan plan;
    plan.barriers.reserve(accesses.size());
    plan.updates.reserve(accesses.size());
    struct BufferShadow {
        VideoCore::BufferId id{};
        u64 resource_uid{};
        VideoCore::BufferAccessState state;
        size_t update_index{};
    };
    boost::container::small_vector<BufferShadow, 4> shadow_states;

    for (const auto& access : accesses) {
        if (!access.resource) {
            continue;
        }
        const auto initial_state =
            access.id
                ? store.GetAccessState(access.id)
                : store.GetSpecialAccessState(access.resource_uid,
                                              access.resource->LocalStageMask(),
                                              access.resource->LocalAccessMask());
        auto it = std::ranges::find_if(shadow_states, [&](const auto& shadow) {
            return access.id ? shadow.id == access.id
                             : !shadow.id && shadow.resource_uid == access.resource_uid;
        });
        const bool inserted = it == shadow_states.end();
        if (inserted) {
            const size_t update_index = plan.updates.size();
            it = shadow_states.emplace(it, BufferShadow{
                                               .id = access.id,
                                               .resource_uid = access.resource_uid,
                                               .state = initial_state,
                                               .update_index = update_index,
                                           });
            plan.updates.push_back({
                .resource = access.resource,
                .id = access.id,
                .resource_uid = access.resource_uid,
                .generation = access.generation,
                .observed_state_version = initial_state.state_version,
            });
        }
        auto& state = it->state;
        if (state.stage_mask != access.stage || state.access_mask != access.access) {
            ASSERT(access.offset < access.resource->SizeBytes());
            plan.barriers.push_back({
                .srcStageMask = state.stage_mask,
                .srcAccessMask = state.access_mask,
                .dstStageMask = access.stage,
                .dstAccessMask = access.access,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = access.resource->Handle(),
                .offset = access.offset,
                .size = access.resource->SizeBytes() - access.offset,
            });
        }
        state.stage_mask = access.stage;
        state.access_mask = access.access;
    }

    for (const auto& shadow : shadow_states) {
        auto& update = plan.updates[shadow.update_index];
        const auto& state = shadow.state;
        update.stage = state.stage_mask;
        update.access = state.access_mask;
    }
    return plan;
}

BufferHazardPlan ResourceHazardPlanner::PlanBufferHazards(
    const VideoCore::BufferAccessStateStore& store,
    std::span<const VideoCore::ResolvedBufferRef> buffers,
    vk::PipelineStageFlags2 target_stage, vk::AccessFlags2 target_access) {
    BufferHazardPlan plan;
    plan.barriers.reserve(buffers.size());
    plan.updates.reserve(buffers.size());
    for (const auto& ref : buffers) {
        if (!ref.IsValid()) {
            continue;
        }
        const auto state = store.GetAccessState(ref.id);
        plan.updates.push_back({
            .id = ref.id,
            .generation = ref.generation,
            .observed_state_version = state.state_version,
            .stage = target_stage,
            .access = target_access,
        });
        if (state.stage_mask != target_stage || state.access_mask != target_access) {
            plan.barriers.push_back({
                .srcStageMask = state.stage_mask,
                .srcAccessMask = state.access_mask,
                .dstStageMask = target_stage,
                .dstAccessMask = target_access,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = ref.buffer,
                .offset = ref.offset,
                .size = ref.size,
            });
        }
    }
    return plan;
}

ImageHazardPlan ResourceHazardPlanner::PlanImageHazards(
    const VideoCore::ImageAccessStateStore& store,
    const ImageTransitionPlan& transitions) {
    ImageHazardPlan plan;
    plan.barriers.reserve(transitions.size());
    plan.updates.reserve(transitions.size());
    struct ImageShadow {
        VideoCore::ImageId id{};
        VideoCore::ImageAccessState state;
        size_t update_index{};
    };
    boost::container::small_vector<ImageShadow, 4> shadow_states;

    for (const auto& transition : transitions) {
        if (!transition.id || !transition.resource) {
            continue;
        }
        auto it = std::ranges::find(shadow_states, transition.id, &ImageShadow::id);
        const bool inserted = it == shadow_states.end();
        if (inserted) {
            const size_t update_index = plan.updates.size();
            it = shadow_states.emplace(
                it, ImageShadow{
                        .id = transition.id,
                        .state = store.GetAccessState(transition.id),
                        .update_index = update_index,
                    });
            plan.updates.push_back({
                .resource = transition.resource,
                .id = transition.id,
                .generation = transition.generation,
                .observed_state_version = it->state.state_version,
            });
        }
        VideoCore::PlanImageAccessTransition(
            it->state,
            {
                .image = transition.resource->GetImage(),
                .aspects = transition.resource->aspect_mask,
                .resources = transition.resource->info.resources,
                .layout = transition.layout,
                .stage = transition.stage,
                .access = transition.access,
                .subresource = transition.subresource,
            },
            plan.barriers);
    }

    for (auto& shadow : shadow_states) {
        plan.updates[shadow.update_index].state = std::move(shadow.state);
    }
    return plan;
}

bool ResourceHazardPlanner::CommitBufferHazards(
    VideoCore::BufferAccessStateStore& store,
    const BufferHazardPlan& plan) noexcept {
    if (!std::ranges::all_of(plan.updates, [&](const auto& update) {
            return update.id
                       ? store.CanUpdateAccessState(update.id, update.observed_state_version,
                                                    update.generation)
                       : store.CanUpdateSpecialAccessState(update.resource_uid,
                                                           update.observed_state_version);
        })) {
        return false;
    }
    for (const auto& update : plan.updates) {
        if (update.id) {
            store.UpdateAccessState(update.id, update.generation, update.stage, update.access);
        } else {
            store.UpdateSpecialAccessState(update.resource_uid, update.stage, update.access);
        }
    }
    return true;
}

bool ResourceHazardPlanner::CommitImageHazards(
    VideoCore::ImageAccessStateStore& store,
    const ImageHazardPlan& plan) noexcept {
    if (!std::ranges::all_of(plan.updates, [&](const auto& update) {
            return store.CanUpdateAccessState(update.id, update.observed_state_version,
                                              update.generation);
        })) {
        return false;
    }
    for (const auto& update : plan.updates) {
        store.ReplaceAccessState(update.id, update.generation, update.state);
    }
    return true;
}

} // namespace Vulkan
