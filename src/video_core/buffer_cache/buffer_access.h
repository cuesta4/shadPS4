// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>

#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

struct BufferAccessTransition {
    vk::PipelineStageFlags2 source_stages;
    vk::AccessFlags2 source_access;
    vk::PipelineStageFlags2 destination_stages;
    vk::AccessFlags2 destination_access;
};

class BufferAccessTracker {
public:
    explicit BufferAccessTracker(
        vk::AccessFlags2 access_ = vk::AccessFlagBits2::eMemoryRead |
                                   vk::AccessFlagBits2::eMemoryWrite |
                                   vk::AccessFlagBits2::eTransferRead |
                                   vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlags2 stages_ = vk::PipelineStageFlagBits2::eAllCommands)
        : access{access_}, stages{stages_} {}

    [[nodiscard]] std::optional<BufferAccessTransition> TransitionTo(
        const vk::AccessFlags2 destination_access,
        const vk::PipelineStageFlags2 destination_stages) {
        if (destination_access == access && destination_stages == stages) {
            return {};
        }

        if (!HasWriteAccess(access) && !HasWriteAccess(destination_access)) {
            // Read-only consumers do not depend on one another. Keep all reader stages and access
            // types so a later writer still waits for every outstanding read.
            access |= destination_access;
            stages |= destination_stages;
            return {};
        }

        const BufferAccessTransition transition{
            .source_stages = stages,
            .source_access = access,
            .destination_stages = destination_stages,
            .destination_access = destination_access,
        };
        access = destination_access;
        stages = destination_stages;
        return transition;
    }

    [[nodiscard]] vk::AccessFlags2 Access() const noexcept {
        return access;
    }

    [[nodiscard]] vk::PipelineStageFlags2 Stages() const noexcept {
        return stages;
    }

private:
    static bool HasWriteAccess(const vk::AccessFlags2 value) noexcept {
        constexpr vk::AccessFlags2 WriteAccess =
            vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite |
            vk::AccessFlagBits2::eHostWrite | vk::AccessFlagBits2::eMemoryWrite;
        return static_cast<bool>(value & WriteAccess);
    }

    vk::AccessFlags2 access;
    vk::PipelineStageFlags2 stages;
};

} // namespace VideoCore
