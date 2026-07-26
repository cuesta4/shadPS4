// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <utility>

#include "common/types.h"
#include "video_core/buffer_cache/buffer_registry.h"
#include "video_core/texture_cache/image_registry.h"

namespace VideoCore {

class Buffer;

class BufferSynchronizer {
public:
    virtual ~BufferSynchronizer() = default;
    virtual std::pair<Buffer*, u32> ObtainBufferForImage(VAddr address, u32 size) = 0;
};

class ImageSynchronizer {
public:
    virtual ~ImageSynchronizer() = default;
    virtual bool ClearMetadata(VAddr address) = 0;
    virtual bool HasImage(VAddr address, u32 size) = 0;
    virtual void InvalidateImagesFromGpu(VAddr address, u32 size) = 0;
    virtual bool SynchronizeBufferFromImage(Buffer& buffer, VAddr address, u32 size) = 0;
};

class ResourceAliasCoordinator {
public:
    /**
     * Serial GPU-thread coordinator for cross-domain buffer/image alias effects.
     *
     * Callbacks are deliberately operation-shaped rather than cache-shaped. They may enter the
     * texture-cache mutex, so callers must not hold a buffer-cache lock while invoking them.
     */
    ResourceAliasCoordinator(BufferRegistry& buffer_registry_,
                             ImageRegistry& image_registry_,
                             std::function<std::pair<Buffer*, u32>(VAddr, u32)>
                                 obtain_buffer_for_image_,
                             std::function<bool(VAddr)> clear_metadata_,
                             std::function<bool(VAddr, u32)> has_image_,
                             std::function<void(VAddr, u32)> invalidate_images_,
                             std::function<bool(Buffer&, VAddr, u32)>
                                 synchronize_buffer_from_image_)
        : obtain_buffer_for_image{std::move(obtain_buffer_for_image_)},
          clear_metadata{std::move(clear_metadata_)}, has_image{std::move(has_image_)},
          invalidate_images{std::move(invalidate_images_)},
          synchronize_buffer_from_image{std::move(synchronize_buffer_from_image_)},
          buffer_registry{buffer_registry_}, image_registry{image_registry_} {}

    [[nodiscard]] BufferRegistry& Buffers() noexcept { return buffer_registry; }
    [[nodiscard]] const BufferRegistry& Buffers() const noexcept { return buffer_registry; }
    [[nodiscard]] ImageRegistry& Images() noexcept { return image_registry; }
    [[nodiscard]] const ImageRegistry& Images() const noexcept { return image_registry; }

    bool ClearMetadata(VAddr address);
    bool HasImage(VAddr address, u32 size);
    void InvalidateImagesFromGpu(VAddr address, u32 size);
    bool SynchronizeBufferFromImage(Buffer& buffer, VAddr address, u32 size);
    std::pair<Buffer*, u32> ObtainBufferForImage(VAddr address, u32 size);

private:
    std::function<std::pair<Buffer*, u32>(VAddr, u32)> obtain_buffer_for_image;
    std::function<bool(VAddr)> clear_metadata;
    std::function<bool(VAddr, u32)> has_image;
    std::function<void(VAddr, u32)> invalidate_images;
    std::function<bool(Buffer&, VAddr, u32)> synchronize_buffer_from_image;
    BufferRegistry& buffer_registry;
    ImageRegistry& image_registry;
};

} // namespace VideoCore
