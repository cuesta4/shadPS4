// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/preparation/graphics_request_builder.h"

namespace Vulkan {

class Instance;
class DynamicState;

class DynamicStateEmitter {
public:
    explicit DynamicStateEmitter(const Instance& instance_) : instance{instance_} {}

    void Emit(const DynamicStatePlan& plan, DynamicState& state,
              vk::CommandBuffer command_buffer) const;

private:
    const Instance& instance;
};

} // namespace Vulkan
