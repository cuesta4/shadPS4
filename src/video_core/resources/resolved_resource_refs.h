// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/resources/resource_ids.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {

/**
 * Generation-versioned reference to a resolved GPU image resource.
 * Invalidated automatically when merge/expand or backing allocation changes.
 */
struct ResolvedImageRef {
    ImageId id{};
    u32 generation{};
    ImageViewId view_id{};
    u32 view_generation{};
    vk::Image image{};
    vk::ImageView view{};
    vk::ImageAspectFlags aspects{vk::ImageAspectFlagBits::eColor};
    SubresourceRange subresource{};
    u32 observed_state_version{};

    [[nodiscard]] constexpr bool IsValid() const noexcept {
        return static_cast<bool>(id);
    }
};

/**
 * Generation-versioned reference to a resolved GPU buffer resource.
 */
struct ResolvedBufferRef {
    BufferId id{};
    u32 generation{};
    vk::Buffer buffer{};
    u64 offset{};
    u64 size{};
    u32 observed_state_version{};

    [[nodiscard]] constexpr bool IsValid() const noexcept {
        return static_cast<bool>(id);
    }
};

} // namespace VideoCore
