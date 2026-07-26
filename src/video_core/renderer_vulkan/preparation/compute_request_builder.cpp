// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/preparation/compute_request_builder.h"

namespace Vulkan {

ComputeDispatchPlan ComputeRequestBuilder::BuildDispatchPlan(
    const VideoCore::CapturedComputeState& state) noexcept {
    ComputeDispatchPlan plan{};
    plan.group_count_x = state.dispatch.dim_x;
    plan.group_count_y = state.dispatch.dim_y;
    plan.group_count_z = state.dispatch.dim_z;
    plan.shared_memory_bytes = state.dispatch.shared_memory_size;
    return plan;
}
} // namespace Vulkan
