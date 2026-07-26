// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/buffer_cache/buffer_registry.h"
#include "video_core/buffer_cache/buffer_access_state_store.h"
#include "video_core/renderer_vulkan/preparation/command_preparation_context.h"
#include "video_core/texture_cache/image_access_state_store.h"
#include "video_core/texture_cache/image_registry.h"

namespace Vulkan {

/**
 * Serial commit boundary for resource state owned by prepared commands.
 *
 * Validation and barrier calculation complete before any global resource state is changed.
 */
class OrderedResourceCommitter {
public:
    OrderedResourceCommitter(const VideoCore::BufferRegistry& buffers_,
                             const VideoCore::ImageRegistry& images_,
                             VideoCore::BufferAccessStateStore& buffer_states_,
                             VideoCore::ImageAccessStateStore& image_states_)
        : buffers{buffers_}, images{images_}, buffer_states{buffer_states_},
          image_states{image_states_} {}

    [[nodiscard]] bool Commit(PreparedResourcePayload& context) const;

private:
    [[nodiscard]] bool Validate(const PreparedResourcePayload& context) const;
    const VideoCore::BufferRegistry& buffers;
    const VideoCore::ImageRegistry& images;
    VideoCore::BufferAccessStateStore& buffer_states;
    VideoCore::ImageAccessStateStore& image_states;
};

} // namespace Vulkan
