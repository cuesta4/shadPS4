// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/execution/ordered_resource_committer.h"

#include <algorithm>

#include "video_core/renderer_vulkan/execution/resource_hazard_planner.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {

bool OrderedResourceCommitter::Validate(const PreparedResourcePayload& context) const {
    if (context.buffer_registry_epoch != buffers.Epoch()) {
        for (const auto& access : context.buffer_accesses) {
            if (!access.resource || access.resource->Uid() != access.resource_uid) {
                return false;
            }
            if (access.id && !buffers.Validate(access.id, access.generation)) {
                return false;
            }
        }
        for (const auto& ref : context.resolved_buffers) {
            if (!buffers.Validate(ref.id, ref.generation)) {
                return false;
            }
        }
    }
    if (context.image_registry_epoch != images.Epoch()) {
        for (const auto& transition : context.image_transitions) {
            if (!transition.resource ||
                transition.resource->image_uid != transition.resource_uid ||
                !images.Validate(transition.id, transition.generation)) {
                return false;
            }
        }
        for (const auto& ref : context.resolved_images) {
            if (!ref.view_id ||
                !images.Validate(ref.id, ref.generation, ref.view_generation)) {
                return false;
            }
        }
    }
    return true;
}

bool OrderedResourceCommitter::Commit(PreparedResourcePayload& context) const {
    if (!Validate(context)) {
        return false;
    }

    auto buffer_plan =
        ResourceHazardPlanner::PlanBufferHazards(buffer_states, context.buffer_accesses);
    auto image_plan =
        ResourceHazardPlanner::PlanImageHazards(image_states, context.image_transitions);

    if (!ResourceHazardPlanner::CommitBufferHazards(buffer_states, buffer_plan) ||
        !ResourceHazardPlanner::CommitImageHazards(image_states, image_plan)) {
        return false;
    }
    context.buffer_barriers = std::move(buffer_plan.barriers);
    context.image_barriers = std::move(image_plan.barriers);
    context.buffer_accesses.clear();
    context.image_transitions.clear();
    return true;
}

} // namespace Vulkan
