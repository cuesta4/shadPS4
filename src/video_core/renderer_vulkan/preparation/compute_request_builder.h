// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/gpu_commands/captured_state.h"

namespace Vulkan {

struct ComputeDispatchPlan {
    u32 group_count_x{1};
    u32 group_count_y{1};
    u32 group_count_z{1};
    u32 shared_memory_bytes{};

    [[nodiscard]] constexpr bool operator==(const ComputeDispatchPlan&) const noexcept = default;
};

class ComputeRequestBuilder {
public:
    [[nodiscard]] static ComputeDispatchPlan BuildDispatchPlan(
        const VideoCore::CapturedComputeState& state) noexcept;
};

} // namespace Vulkan
