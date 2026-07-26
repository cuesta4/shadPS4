// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

class RenderingScopeController {
public:
    explicit RenderingScopeController(Scheduler& scheduler_) : scheduler{scheduler_} {}

    void Begin(const RenderState& state) const {
        scheduler.BeginRendering(state);
    }

    void End() const {
        scheduler.EndRendering();
    }

private:
    Scheduler& scheduler;
};

} // namespace Vulkan
